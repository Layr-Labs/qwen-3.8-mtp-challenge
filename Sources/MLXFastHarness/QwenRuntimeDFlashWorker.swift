import Foundation
import MLX
import MLXFastCore
import MLXFastModel
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXSpeculative
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

/// Validated `dflash_decode_block` request.
struct ExperimentalDFlashBlockRequest: Equatable {
    let previousToken: Int
    let maxBlockSize: Int
}

struct ExperimentalDFlashWorkerState {
    var began = false
    /// Set by `dflash_decode_warm`. The warm is input-independent, so the
    /// trusted parent runs it BEFORE starting its clock; `began` then skips it.
    var warmed = false
    var poisoned = false
    var seedTokenCount = 0
    var decodedTokenCount = 0
    /// Reference-side only. Holds ONE continuously-advanced width-1 cache for
    /// the whole reference pass, so the frame is built exactly the way the
    /// candidate builds its own (seed bulk forward, then one single-token
    /// forward per position) and the pass costs O(n) rather than the O(n^2) a
    /// per-round bulk re-prefill costs. Created lazily: the candidate worker
    /// never receives a reference request, so it never allocates this.
    var referenceSession: LagunaDFlashReference?
}

/// Strict validation for a DFlash block request.
///
/// Deliberately does NOT bound `maxBlockSize` by any remaining-token count: the
/// worker is never told how much of the decode window is left, so it cannot
/// special-case the tail. The trusted parent always asks for a full block and
/// truncates the scored prefix itself.
func validateExperimentalDFlashBlockRequest(
    _ request: RuntimeWorkerRequest,
    decodedTokenCount: Int
) throws -> ExperimentalDFlashBlockRequest {
    guard request.id > 0, request.kind == "dflash_decode_block" else {
        throw MLXFastError.invalidInput(
            "DFlash block request has an invalid id or kind"
        )
    }
    guard request.promptTokens == nil,
          request.seedTokens == nil,
          request.steps == nil,
          request.topK == nil,
          request.expectedToken == nil,
          request.prefixTokens == nil,
          request.startOffset == nil,
          request.rowCount == nil,
          request.declaredBlockWidth == nil,
          request.seedTokenCount == nil,
          request.verifyBlockTokens == nil,
          let previousToken = request.token,
          previousToken >= 0,
          previousToken < MLXFastConstants.vocabSize,
          let maxBlockSize = request.maxBlockSize,
          // 1 is legal: it is the serial control the paired score divides by,
          // served by the same worker and the same protocol.
          maxBlockSize >= 1,
          maxBlockSize <= MLXFastConstants.experimentalDFlashMaxBlockSize
    else {
        throw MLXFastError.invalidInput(
            "DFlash block request has invalid or cross-kind fields"
        )
    }
    guard decodedTokenCount >= 0 else {
        throw MLXFastError.invalidInput(
            "DFlash worker has a negative committed token count"
        )
    }
    let (requestedTotal, overflow) =
        decodedTokenCount.addingReportingOverflow(maxBlockSize)
    guard !overflow,
          requestedTotal
              <= MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens
    else {
        throw MLXFastError.invalidInput(
            "DFlash block request exceeds the configured decode ceiling"
        )
    }
    return ExperimentalDFlashBlockRequest(
        previousToken: previousToken,
        maxBlockSize: maxBlockSize
    )
}

extension QwenRuntime {
    /// Runtime worker for the DFlash block-decode track.
    ///
    /// Loads the organizer-pinned target and drafter, warms the block graph, and
    /// then serves exactly three request kinds over the isolated protocol pipe.
    /// Everything expensive happens before the protocol hello, i.e. outside
    /// every scored window.
    public static func runExperimentalDFlashWorker(
        targetWeightsPath: String,
        drafterPath: String
    ) throws {
        startRuntimeWorkerOrphanReaper()
        let protocolIO = try RuntimeWorkerProtocolIO.isolatingStandardIO()

        // Select the startup memory profile BEFORE the target load and the
        // separate drafter load, mirroring the policy's only other call site
        // (Qwen35RuntimeWeightCache.init on the serial path). This adds no
        // policy and no threshold: the same 64 GiB boundary, the same
        // no-overwrite precedence for explicitly exported values, and the same
        // DARKBLOOM_STARTUP_MEMORY_PROFILE=full|low|auto opt-out -- which does
        // reach here, because sanitizedRuntimeWorkerEnvironment forwards the
        // DARKBLOOM_ and MLX_ prefixes into this worker (MLXFAST_ is NOT
        // forwarded, so no MLXFAST_ spelling would ever arrive). It exists
        // because the DFlash worker never constructs Qwen35RuntimeWeightCache,
        // so it was the one startup path the policy never reached -- and it is
        // the path that needs it most: this worker holds the ~21.6 GB target
        // AND the separate drafter, and warms every legal block width at a
        // past-the-ring seed, so a <64 GiB machine ran the LARGER resident set
        // with no allocator-cache cap and stock command buffers. The full
        // profile stays a deliberate no-op, so the ranked box keeps the stock
        // allocator behavior the pinned baseline was measured with.
        let startupMemoryPolicy: RuntimeStartupMemoryPolicy?
        let resolvedStartupMemoryPolicy = RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            requestedProfile: ProcessInfo.processInfo.environment[
                RuntimeStartupMemoryPolicy.profileOverrideEnvironmentName
            ]
        )
        if resolvedStartupMemoryPolicy.isLowMemory {
            // Applied HERE, in trusted non-editable code, rather than by calling
            // the policy's own `apply()`. That method is `internal`, and it must
            // stay internal: its file sits under `Sources/MLXFastModel`, a
            // DIRECTORY entry in both tracks' editablePaths, so `submit` packages
            // it with every submission. Widening it would mean this trusted worker
            // failed to compile whenever a submission overlaid an older copy --
            // including every serial submission in flight today. Every member read
            // below has been `public` since before the DFlash track existed, so an
            // older overlay still builds.
            setenv(
                "MLX_MAX_MB_PER_BUFFER",
                String(resolvedStartupMemoryPolicy.maxMegabytesPerCommandBuffer),
                1
            )
            setenv(
                "MLX_MAX_OPS_PER_BUFFER",
                String(resolvedStartupMemoryPolicy.maxOperationsPerCommandBuffer),
                1
            )
            // No-overwrite (0): an explicitly exported value always wins, matching
            // the policy's own semantics for these opt-in flags.
            for (name, value) in resolvedStartupMemoryPolicy.environmentOverrides {
                setenv(name, value, 0)
            }
            Memory.cacheLimit = resolvedStartupMemoryPolicy.cacheLimitBytes
            fputs(
                "mlxfast-worker: low-memory startup profile engaged ("
                    + resolvedStartupMemoryPolicy.selectionReason + ")\n",
                stderr
            )
            startupMemoryPolicy = resolvedStartupMemoryPolicy
        } else {
            startupMemoryPolicy = nil
        }

        // Load the target through the vendored factory: `LagunaModel` is the
        // type that conforms to `DFlashTargetModel` (the scored serial model,
        // Qwen35TextModel, deliberately does not), and this is the same load
        // path the validated `mlx-bench dflash` run used.
        let targetURL = URL(fileURLWithPath: targetWeightsPath)
        let context = try waitForExperimentalDFlashAsync {
            try await LLMModelFactory.shared.load(
                from: targetURL,
                using: #huggingFaceTokenizerLoader()
            )
        }
        guard let target = context.model as? any DFlashTargetModel else {
            throw MLXFastError.invalidInput(
                "DFlash target model does not conform to DFlashTargetModel: "
                    + "\(type(of: context.model))"
            )
        }
        let drafter = try waitForExperimentalDFlashAsync {
            try await DFlashDraftModel.load(
                from: URL(fileURLWithPath: drafterPath),
                bindTo: target
            )
        }
        eval(context.model, drafter)

        // Warm the block-decode shapes on throwaway cache state before the
        // hello. The real begin request performs the trusted allocator clear and
        // re-warms the working set it frees.
        // Same warm the post-allocator-reset re-warm runs, via the same helper:
        // a seed past the sliding-window ring and EVERY legal block width. These
        // two warm points used to warm different things.
        let warmup = try LagunaDFlashBlockSession(target: target, drafter: drafter)
        try warmup.warmAllBlockWidths()

        let session = try LagunaDFlashBlockSession(target: target, drafter: drafter)

        if startupMemoryPolicy?.clearAllocatorCacheAfterWarmup == true {
            // The same post-warmup clear the serial path performs, at the point
            // the policy documents: before the worker protocol hello. Metal
            // pipeline state is process-lifetime state, while the free buffers
            // the warm and the session build leave behind are exactly the
            // pressure a low-memory machine cannot afford to hold across the
            // hello. The later `dflash_decode_warm` phase-start reset clears
            // again, but that is the parent's untimed phase boundary and long
            // after the pre-hello peak this avoids.
            Memory.clearCache()
        }

        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let sessionNonce = generateRuntimeWorkerNonce()
        try protocolIO.writeLine(try encoder.encode(RuntimeWorkerResponse(
            id: 0,
            nonce: sessionNonce,
            ok: true
        )))

        var state = ExperimentalDFlashWorkerState()
        var expectedRequestID = 1
        while let line = try protocolIO.readLine() {
            guard !line.isEmpty else {
                continue
            }
            let response: RuntimeWorkerResponse
            do {
                let request = try decoder.decode(
                    RuntimeWorkerRequest.self,
                    from: Data(line.utf8)
                )
                guard request.id == expectedRequestID else {
                    throw MLXFastError.invalidInput(
                        "DFlash request id must be monotonic; expected "
                            + "\(expectedRequestID), got \(request.id)"
                    )
                }
                expectedRequestID += 1
                do {
                    response = try handleExperimentalDFlashWorkerRequest(
                        request,
                        sessionNonce: sessionNonce,
                        session: session,
                        state: &state
                    )
                } catch {
                    response = RuntimeWorkerResponse(
                        id: request.id,
                        nonce: sessionNonce,
                        ok: false,
                        error: "\(error)"
                    )
                }
            } catch {
                response = RuntimeWorkerResponse(
                    id: -1,
                    nonce: sessionNonce,
                    ok: false,
                    error: "\(error)"
                )
            }
            try protocolIO.writeLine(try encoder.encode(response))
        }
    }

    static func handleExperimentalDFlashWorkerRequest(
        _ request: RuntimeWorkerRequest,
        sessionNonce: String,
        session: LagunaDFlashBlockSession,
        state: inout ExperimentalDFlashWorkerState
    ) throws -> RuntimeWorkerResponse {
        guard !state.poisoned else {
            throw MLXFastError.invalidInput(
                "DFlash decode session is poisoned after an earlier failure"
            )
        }

        switch request.kind {
        case "dflash_decode_warm":
            // Untimed phase start: clear the allocator, then re-touch the working
            // set it freed. Carries NO seed and applies none, so there is nothing
            // input-dependent a submission could hide in here -- which is exactly
            // why it is safe to run outside the scored window.
            guard !state.began,
                  request.id > 0,
                  request.seedTokens == nil,
                  request.promptTokens == nil,
                  request.token == nil,
                  request.steps == nil,
                  request.maxBlockSize == nil,
                  request.topK == nil,
                  request.expectedToken == nil
            else {
                throw MLXFastError.invalidInput(
                    "DFlash warm request is malformed or arrived after begin"
                )
            }
            do {
                try resetRuntimeWorkerAllocatorForPhaseStart()
                try session.warmWorkingSetAfterAllocatorReset()
                state.warmed = true
                return RuntimeWorkerResponse(
                    id: request.id,
                    nonce: sessionNonce,
                    ok: true
                )
            } catch {
                state.poisoned = true
                throw error
            }

        case "dflash_decode_begin":
            guard !state.began,
                  request.id > 0,
                  let seedTokens = request.seedTokens,
                  !seedTokens.isEmpty,
                  request.promptTokens == nil,
                  request.token == nil,
                  request.steps == nil,
                  request.maxBlockSize == nil,
                  request.topK == nil,
                  request.expectedToken == nil
            else {
                throw MLXFastError.invalidInput(
                    "DFlash begin request is repeated or malformed"
                )
            }
            // The allocator clear and the re-touch that follows it are
            // input-independent -- the seed is not applied by either -- so the
            // trusted parent issues them as `dflash_decode_warm` BEFORE starting
            // its clock. Doing them here instead charged the whole warm to the
            // measurement, which is why the warm had to stay small: widening it
            // to cover every block width and a past-the-ring seed cost 24-27% of
            // absolute decode time. Kept as a fallback so a parent that does not
            // warm still gets correct (if slower) behaviour.
            if !state.warmed {
                try resetRuntimeWorkerAllocatorForPhaseStart()
                try session.warmWorkingSetAfterAllocatorReset()
                state.warmed = true
            }
            do {
                let seedToken = try session.begin(seedTokens: seedTokens)
                state.began = true
                state.seedTokenCount = seedTokens.count
                state.decodedTokenCount = 0
                return RuntimeWorkerResponse(
                    id: request.id,
                    nonce: sessionNonce,
                    ok: true,
                    seedToken: seedToken
                )
            } catch {
                state.poisoned = true
                throw error
            }

        case "dflash_decode_block":
            guard state.began else {
                throw MLXFastError.invalidInput(
                    "DFlash block requested before begin"
                )
            }
            let block = try validateExperimentalDFlashBlockRequest(
                request,
                decodedTokenCount: state.decodedTokenCount
            )
            do {
                let result = try session.generateBlock(
                    previousCommittedToken: block.previousToken,
                    maxBlockSize: block.maxBlockSize
                )
                let (nextCount, overflow) =
                    state.decodedTokenCount.addingReportingOverflow(
                        result.tokens.count
                    )
                let (expectedCacheOffset, offsetOverflow) =
                    state.seedTokenCount.addingReportingOverflow(nextCount)
                guard !overflow,
                      !offsetOverflow,
                      nextCount
                          <= MLXFastConstants
                              .experimentalDFlashMaxConfiguredTotalTokens,
                      result.targetCacheOffset == expectedCacheOffset
                else {
                    throw MLXFastError.invalidInput(
                        "DFlash logical token count and target cache offset "
                            + "diverged"
                    )
                }
                state.decodedTokenCount = nextCount
                return RuntimeWorkerResponse(
                    id: request.id,
                    nonce: sessionNonce,
                    ok: true,
                    tokens: result.tokens,
                    declaredRows: result.declaredRows,
                    perRowHiddenDigest: result.perRowHiddenDigest,
                    perRowTop2Tokens: result.perRowTop2Tokens,
                    perRowTop2Logits: result.perRowTop2Logits,
                    draftTokens: result.draftTokens,
                    acceptedDraftCount: result.acceptedDraftCount,
                    rejectedDraftCount: result.rejectedDraftCount,
                    targetCacheOffset: result.targetCacheOffset
                )
            } catch {
                state.poisoned = true
                throw error
            }

        case "dflash_phase_diagnostics":
            guard state.began,
                  request.promptTokens == nil,
                  request.seedTokens == nil,
                  request.token == nil,
                  request.steps == nil,
                  request.maxBlockSize == nil,
                  request.topK == nil,
                  request.expectedToken == nil
            else {
                throw MLXFastError.invalidInput(
                    "DFlash diagnostics request is malformed or before begin"
                )
            }
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                peakRamGB: peakResidentMemoryGB(),
                mlxActiveMemoryBytes: Memory.activeMemory,
                mlxCacheMemoryBytes: Memory.cacheMemory,
                mlxPeakMemoryBytes: Memory.peakMemory,
                acceptedDraftCount: session.acceptedDraftTotal,
                rejectedDraftCount: session.rejectedDraftTotal,
                rollbackRoundCount: session.rollbackRoundCount,
                targetCacheOffset: session.seedTokenCount
                    + session.decodedTokenCount
            )

        case "dflash_reference_prefill":
            // REFERENCE SIDE ONLY (contract layer L1). Establishes the run's
            // seed token in the CANDIDATE's frame -- one bulk forward over the
            // whole seed, argmax of the last row, which is exactly what
            // `LagunaDFlashBlockSession.begin` computes -- and primes the
            // continuous width-1 frame at the end of the seed.
            guard request.token == nil,
                  request.promptTokens == nil,
                  request.steps == nil,
                  request.topK == nil,
                  request.expectedToken == nil,
                  request.maxBlockSize == nil,
                  request.prefixTokens == nil,
                  request.startOffset == nil,
                  request.rowCount == nil,
                  request.declaredBlockWidth == nil,
                  request.seedTokenCount == nil,
                  request.verifyBlockTokens == nil,
                  let seedTokens = request.seedTokens,
                  !seedTokens.isEmpty
            else {
                throw MLXFastError.invalidInput(
                    "DFlash reference-prefill request is malformed or has "
                        + "cross-kind fields"
                )
            }
            let prefillSession = LagunaDFlashReference(
                target: session.referenceTarget,
                targetLayerIds: session.referenceTargetLayerIds
            )
            let seedToken = try prefillSession.prefillSeed(seedTokens)
            state.referenceSession = prefillSession
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                seedToken: seedToken
            )

        case "dflash_reference_rows":
            // REFERENCE SIDE ONLY (contract layer L1). Served by a worker built
            // from the pinned baseline tree over organizer-transformed weights,
            // strictly after the timed window with the candidate torn down.
            //
            // Deliberately STATEFUL, against the earlier stateless design. A
            // stateless request has to reconstruct the prefix somehow, and the
            // only cheap reconstruction -- one bulk forward over
            // `prefixTokens[0 ..< startOffset]` -- is not the computation the
            // candidate performs once `startOffset` passes the seed. The
            // accumulation order differs, and the sliding-window cache takes a
            // different code path (`updateConcat` versus `updateInPlace`), so at
            // a near tie the "sequential" argmax stopped being what a sequential
            // decoder produces. Keeping one continuously-advanced cache and
            // walking it forward is both faithful and O(n); determinism is still
            // provable because an out-of-order replay rebuilds the frame from
            // scratch with the SAME construction and must agree.
            guard request.token == nil,
                  request.seedTokens == nil,
                  request.promptTokens == nil,
                  request.steps == nil,
                  request.topK == nil,
                  request.expectedToken == nil,
                  request.maxBlockSize == nil,
                  let prefixTokens = request.prefixTokens,
                  !prefixTokens.isEmpty,
                  let seedTokenCount = request.seedTokenCount,
                  seedTokenCount > 0,
                  seedTokenCount <= prefixTokens.count,
                  let startOffset = request.startOffset,
                  startOffset >= seedTokenCount,
                  let rowCount = request.rowCount,
                  rowCount > 0,
                  let widestFrame = request.declaredBlockWidth,
                  widestFrame >= rowCount,
                  widestFrame
                      <= MLXFastConstants.experimentalDFlashMaxBlockSize,
                  // The verify block is the parent's reconstruction of the
                  // candidate's own verify input for this round. Row 0 is the
                  // parent's committed token; the rest are the journalled drafts.
                  // Width is bounded exactly like any other frame.
                  request.verifyBlockTokens.map({
                      !$0.isEmpty
                          && $0.count
                              <= MLXFastConstants.experimentalDFlashMaxBlockSize
                  }) ?? true
            else {
                throw MLXFastError.invalidInput(
                    "DFlash reference-rows request is malformed or has "
                        + "cross-kind fields"
                )
            }
            let referenceSession: LagunaDFlashReference
            if let existing = state.referenceSession {
                referenceSession = existing
            } else {
                referenceSession = LagunaDFlashReference(
                    target: session.referenceTarget,
                    targetLayerIds: session.referenceTargetLayerIds
                )
                state.referenceSession = referenceSession
            }
            let answer = try referenceSession.rows(
                tokens: prefixTokens,
                seedTokenCount: seedTokenCount,
                startOffset: startOffset,
                count: rowCount,
                widestFrame: widestFrame,
                verifyBlockTokens: request.verifyBlockTokens
            )
            // The emitted-token readouts exist only for rows whose context
            // carries a next token, which is a PREFIX of the batch, so they are
            // reported as a prefix rather than padded with an invented value a
            // consumer could mistake for a real logit.
            let emittedPrefix = answer.rows.prefix {
                $0.emittedToken != nil && $0.emittedTokenLogit != nil
            }
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                referenceK1Argmax: answer.rows.map(\.sequentialArgmax),
                referenceBlockArgmax: answer.rows.map(\.blockArgmax),
                referenceTop2Tokens: answer.rows.map(\.top2Tokens),
                referenceTop2Logits: answer.rows.map(\.top2Logits),
                referenceTop1Logits: answer.rows.map(\.top1Logit),
                referenceEmittedTokens: emittedPrefix.compactMap(\.emittedToken),
                referenceEmittedTokenLogits: emittedPrefix
                    .compactMap(\.emittedTokenLogit),
                referenceFrameWidths: answer.frameWidths,
                referenceFrameArgmax: answer.frameArgmax,
                referenceVerifyTop2Tokens: answer.verifyBlockTop2Tokens,
                referenceVerifyTop2Logits: answer.verifyBlockTop2Logits
            )

        default:
            throw MLXFastError.invalidInput(
                "DFlash worker rejects request kind \(request.kind)"
            )
        }
    }
}

private final class ExperimentalDFlashAsyncResultBox<T>: @unchecked Sendable {
    var result: Result<T, Error>?
}

/// Bridge the vendored async loaders into the synchronous worker startup path.
///
/// T is deliberately NOT constrained to `Sendable` and the operation is not a
/// `@Sendable` closure: the values crossing this boundary are the loaded
/// `ModelContext` and `any DFlashTargetModel`, neither of which is Sendable and
/// neither of which can be made so from here (both are vendored types). The
/// hop is safe because this bridge is only ever used during single-threaded
/// worker startup -- once, before the protocol loop begins -- so the loaded
/// model is never touched concurrently. `nonisolated(unsafe)` states that
/// reasoning explicitly instead of hiding it behind a false conformance.
private func waitForExperimentalDFlashAsync<T>(
    _ operation: @escaping () async throws -> T
) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    let box = ExperimentalDFlashAsyncResultBox<T>()
    nonisolated(unsafe) let unsafeOperation = operation
    nonisolated(unsafe) let unsafeBox = box
    Task {
        do {
            unsafeBox.result = .success(try await unsafeOperation())
        } catch {
            unsafeBox.result = .failure(error)
        }
        semaphore.signal()
    }
    semaphore.wait()
    guard let result = box.result else {
        throw MLXFastError.invalidInput(
            "DFlash async model load completed without a result"
        )
    }
    return try result.get()
}
