#if !MLXFAST_TRUSTED_HARNESS
import Foundation
import MLX
import MLXFastCore
import MLXFastModel
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

/// Validated `mtp_decode_round` request.
struct QwenMTPRoundRequest: Equatable {
    let depth: Int
}

struct QwenMTPWorkerState {
    var began = false
    /// Set by `mtp_decode_warm`. The warm is input-independent, so the trusted
    /// parent runs it BEFORE starting its clock; `began` then skips it.
    var warmed = false
    var poisoned = false
    var seedTokenCount = 0
    var decodedTokenCount = 0
    /// Reference-side only, created lazily: the candidate worker never receives a
    /// reference request, so it never allocates the second cache stack.
    var referenceSession: Qwen36MTPReferenceSession?
}

/// Strict validation for an MTP round request.
///
/// Like the DFlash equivalent this deliberately does NOT bound the depth by any
/// remaining-token count: the worker is never told how much of the decode window
/// is left, so it cannot special-case the tail. The trusted parent always asks
/// for a full round and truncates the scored prefix itself.
///
/// The depth rides on the existing `max_block_size` wire field rather than a new
/// one. That field is documented as "the parent-chosen block width for this
/// round" and is already the ONLY block-shaped field on the request; adding a
/// second spelling for the same idea would give a submission two places to look
/// for the window shape and give the protocol two fields that must agree.
func validateQwenMTPRoundRequest(
    _ request: RuntimeWorkerRequest,
    decodedTokenCount: Int
) throws -> QwenMTPRoundRequest {
    guard request.id > 0, request.kind == "mtp_decode_round" else {
        throw MLXFastError.invalidInput(
            "MTP round request has an invalid id or kind")
    }
    guard request.promptTokens == nil,
          request.seedTokens == nil,
          request.token == nil,
          request.steps == nil,
          request.topK == nil,
          request.expectedToken == nil,
          request.prefixTokens == nil,
          request.startOffset == nil,
          request.rowCount == nil,
          request.declaredBlockWidth == nil,
          request.seedTokenCount == nil,
          request.verifyBlockTokens == nil,
          let depth = request.maxBlockSize,
          // 1 is legal: it is the serial control the paired score divides by,
          // served by the same worker, the same protocol and the same forward.
          depth >= 1,
          depth <= Qwen36MTPLimits.maxDepth
    else {
        throw MLXFastError.invalidInput(
            "MTP round request has invalid or cross-kind fields")
    }
    guard decodedTokenCount >= 0 else {
        throw MLXFastError.invalidInput(
            "MTP worker has a negative committed token count")
    }
    let (requestedTotal, overflow) =
        decodedTokenCount.addingReportingOverflow(depth + 1)
    guard !overflow,
          requestedTotal
              <= MLXFastConstants.experimentalDFlashMaxConfiguredTotalTokens
    else {
        throw MLXFastError.invalidInput(
            "MTP round request exceeds the configured decode ceiling")
    }
    return QwenMTPRoundRequest(depth: depth)
}

/// Resolve the stop set the way MTPLX's `_default_stop_tokens` does: the union of
/// `eos_token_id` / `pad_token_id` over `config.json` and `generation_config.json`
/// (each accepting a scalar or a list) plus the tokenizer's own EOS id.
///
/// Applied identically to the speculative loop and to the serial reference, which
/// is what keeps an EOS branch comparable between them.
func resolveQwenMTPStopTokens(
    directory: URL,
    tokenizer: (any MLXLMCommon.Tokenizer)?
) -> Set<Int> {
    var ids = Set<Int>()
    for name in ["config.json", "generation_config.json"] {
        guard let data = try? Data(
            contentsOf: directory.appendingPathComponent(name)),
            let root = (try? JSONSerialization.jsonObject(with: data))
                as? [String: Any]
        else { continue }
        for key in ["eos_token_id", "pad_token_id"] {
            switch root[key] {
            case let value as Int:
                ids.insert(value)
            case let values as [Any]:
                ids.formUnion(values.compactMap { $0 as? Int })
            default:
                continue
            }
        }
    }
    if let eos = tokenizer?.eosTokenId {
        ids.insert(eos)
    }
    return ids
}

extension QwenRuntime {
    /// Runtime worker for the Qwen 3.6 native-MTP track.
    ///
    /// Loads the organizer-pinned backbone with the separately pinned MTP head
    /// merged at load, warms the round shapes, and then serves the `mtp_*`
    /// request kinds. Everything expensive happens before the protocol hello,
    /// i.e. outside every scored window.
    public static func runQwenMTPWorker(
        targetWeightsPath: String,
        mtpHeadPath: String
    ) throws {
        startRuntimeWorkerOrphanReaper()
        let protocolIO = try RuntimeWorkerProtocolIO.isolatingStandardIO()
        applyQwenMTPStartupMemoryProfile()

        let targetURL = URL(fileURLWithPath: targetWeightsPath)
        let headURL = URL(fileURLWithPath: mtpHeadPath)
        // The layout is read from the backbone's OWN config and decides both
        // which class the factory builds and whether the transform's
        // `language_model.` text-tower prefix has to be stripped from the
        // primary tree. Both Qwen 3.6 classes are accepted, because
        // `Qwen35Model` is a pure pass-through wrapper around the same
        // `Qwen35TextModel` for every call this session makes -- see
        // `Qwen36MTPTarget`. Accepting only one of them is what made this worker
        // unloadable in both directions: the transformed tree builds the bare
        // text model, and the raw pinned reference builds the wrapper.
        var backboneLayout = Qwen36MTPHeadAttachment.BackboneLayout.textModel
        let context = try Qwen36MTPHeadAttachment.withHeadAttached(
            backboneDirectory: targetURL,
            headDirectory: headURL
        ) { layout in
            backboneLayout = layout
            return try waitForQwenMTPAsync {
                try await LLMModelFactory.shared.load(
                    from: targetURL,
                    using: #huggingFaceTokenizerLoader()
                )
            }
        }
        guard let model = context.model as? any Qwen36MTPTarget else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP backbone loaded as \(type(of: context.model)), "
                    + "which is not an MTP-capable Qwen 3.6 model. This track "
                    + "serves Qwen35TextModel (the transformed weights/ tree, "
                    + "model_type qwen3_5_text) and Qwen35Model (the raw pinned "
                    + "reference, model_type qwen3_5) and nothing else.")
        }
        fputs(
            "mlxfast-worker: qwen-mtp backbone layout=\(backboneLayout.rawValue) "
                + "class=\(type(of: context.model)) "
                + "key_prefix_strip="
                + "\(backboneLayout.primaryKeyPrefixStrip ?? "<none>")\n",
            stderr
        )
        // Fail here rather than at the first draft. `_qwen35MTPEnabled` is set
        // and cleared around the load, and the head only attaches when the
        // configuration also declares `mtp_num_hidden_layers > 0`; a tree whose
        // config lost that field would otherwise load, never draft, and report a
        // perfectly exact run at zero acceptance.
        guard model.hasMTPHead else {
            throw MLXFastError.invalidInput(
                "the Qwen MTP head did not attach to the loaded backbone: the "
                    + "runtime config.json must declare mtp_num_hidden_layers > 0 "
                    + "and the head tree must carry its "
                    + "\(Qwen36MTPHeadAttachment.expectedHeadTensorCount) tensors")
        }
        eval(context.model)

        let stopTokens = resolveQwenMTPStopTokens(
            directory: targetURL, tokenizer: context.tokenizer)

        // Warm every legal round shape on throwaway cache state, before the
        // hello. The real begin request performs the trusted allocator clear and
        // re-warms the working set it frees.
        let warmup = try Qwen36MTPBlockSession(
            model: model, stopTokens: stopTokens)
        try warmup.warmAllDepths(maxDepth: Qwen36MTPLimits.maxDepth)
        let session = try Qwen36MTPBlockSession(
            model: model, stopTokens: stopTokens)

        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let sessionNonce = generateRuntimeWorkerNonce()
        try protocolIO.writeLine(try encoder.encode(RuntimeWorkerResponse(
            id: 0,
            nonce: sessionNonce,
            ok: true
        )))

        var state = QwenMTPWorkerState()
        var expectedRequestID = 1
        while let line = try protocolIO.readLine() {
            guard !line.isEmpty else { continue }
            let response: RuntimeWorkerResponse
            do {
                let request = try decoder.decode(
                    RuntimeWorkerRequest.self,
                    from: Data(line.utf8)
                )
                guard request.id == expectedRequestID else {
                    throw MLXFastError.invalidInput(
                        "MTP request id must be monotonic; expected "
                            + "\(expectedRequestID), got \(request.id)")
                }
                expectedRequestID += 1
                do {
                    response = try handleQwenMTPWorkerRequest(
                        request,
                        sessionNonce: sessionNonce,
                        model: model,
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

    static func handleQwenMTPWorkerRequest(
        _ request: RuntimeWorkerRequest,
        sessionNonce: String,
        model: any Qwen36MTPTarget,
        session: Qwen36MTPBlockSession,
        state: inout QwenMTPWorkerState
    ) throws -> RuntimeWorkerResponse {
        guard !state.poisoned else {
            throw MLXFastError.invalidInput(
                "the MTP decode session is poisoned after an earlier failure")
        }

        switch request.kind {
        case "mtp_decode_warm":
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
                    "MTP warm request is malformed or arrived after begin")
            }
            do {
                try resetRuntimeWorkerAllocatorForPhaseStart()
                try session.warmAllDepths(maxDepth: Qwen36MTPLimits.maxDepth)
                state.warmed = true
                return RuntimeWorkerResponse(
                    id: request.id, nonce: sessionNonce, ok: true)
            } catch {
                state.poisoned = true
                throw error
            }

        case "mtp_decode_begin":
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
                    "MTP begin request is repeated or malformed")
            }
            if !state.warmed {
                try resetRuntimeWorkerAllocatorForPhaseStart()
                try session.warmAllDepths(maxDepth: Qwen36MTPLimits.maxDepth)
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

        case "mtp_decode_round":
            guard state.began else {
                throw MLXFastError.invalidInput(
                    "MTP round requested before begin")
            }
            let round = try validateQwenMTPRoundRequest(
                request, decodedTokenCount: state.decodedTokenCount)
            do {
                let result = try session.generateRound(depth: round.depth)
                let (nextCount, overflow) =
                    state.decodedTokenCount.addingReportingOverflow(
                        result.tokens.count)
                let (expectedOffset, offsetOverflow) =
                    state.seedTokenCount.addingReportingOverflow(nextCount)
                guard !overflow,
                      !offsetOverflow,
                      nextCount
                          <= MLXFastConstants
                              .experimentalDFlashMaxConfiguredTotalTokens,
                      result.targetCacheOffset == expectedOffset,
                      // The ledger the parent audits has to close inside the
                      // worker too, so a broken round is caught at the boundary
                      // it was produced at rather than a round later.
                      result.acceptedDraftCount + result.rejectedDraftCount
                          + 1 == result.declaredRows,
                      result.perRowTop2Tokens.count == result.declaredRows,
                      result.perRowTop2Logits.count == result.declaredRows
                else {
                    throw MLXFastError.invalidInput(
                        "MTP round ledger or target cache offset diverged")
                }
                state.decodedTokenCount = nextCount
                return RuntimeWorkerResponse(
                    id: request.id,
                    nonce: sessionNonce,
                    ok: true,
                    tokens: result.tokens,
                    declaredRows: result.declaredRows,
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

        case "mtp_reference_prefill":
            // REFERENCE SIDE ONLY. Establishes the run's seed token in the
            // candidate's own frame -- one bulk forward over the whole seed --
            // and primes the continuous width-1 frame at the end of the seed so
            // the first row request is a continuation rather than a rebuild.
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
                    "MTP reference-prefill request is malformed or has "
                        + "cross-kind fields")
            }
            let reference = Qwen36MTPReferenceSession(model: model)
            let seedToken = try reference.prefillSeed(seedTokens)
            state.referenceSession = reference
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                seedToken: seedToken
            )

        case "mtp_reference_rows":
            guard request.token == nil,
                  request.seedTokens == nil,
                  request.promptTokens == nil,
                  request.steps == nil,
                  request.topK == nil,
                  request.expectedToken == nil,
                  request.maxBlockSize == nil,
                  request.declaredBlockWidth == nil,
                  let prefixTokens = request.prefixTokens,
                  !prefixTokens.isEmpty,
                  let seedTokenCount = request.seedTokenCount,
                  seedTokenCount > 0,
                  seedTokenCount <= prefixTokens.count,
                  let startOffset = request.startOffset,
                  startOffset >= seedTokenCount,
                  let rowCount = request.rowCount,
                  rowCount > 0,
                  // The verify block is the parent's reconstruction of the
                  // candidate's own verify input for a round: row 0 is the
                  // parent's committed primary, the rest are the journalled
                  // drafts. Width is bounded exactly like a round.
                  request.verifyBlockTokens.map({
                      !$0.isEmpty && $0.count <= Qwen36MTPLimits.maxDepth + 1
                  }) ?? true
            else {
                throw MLXFastError.invalidInput(
                    "MTP reference-rows request is malformed or has cross-kind "
                        + "fields")
            }
            let reference: Qwen36MTPReferenceSession
            if let existing = state.referenceSession {
                reference = existing
            } else {
                reference = Qwen36MTPReferenceSession(model: model)
                state.referenceSession = reference
            }
            let answer = try reference.rows(
                tokens: prefixTokens,
                seedTokenCount: seedTokenCount,
                startOffset: startOffset,
                count: rowCount,
                verifyBlockTokens: request.verifyBlockTokens
            )
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                referenceK1Argmax: answer.rows.map(\.sequentialArgmax),
                referenceTop2Tokens: answer.rows.map(\.top2Tokens),
                referenceTop2Logits: answer.rows.map(\.top2Logits),
                referenceTop1Logits: answer.rows.map(\.top1Logit),
                referenceVerifyTop2Tokens: answer.verifyBlockTop2Tokens,
                referenceVerifyTop2Logits: answer.verifyBlockTop2Logits
            )

        default:
            throw MLXFastError.invalidInput(
                "MTP worker rejects request kind \(request.kind)")
        }
    }
}

/// Select the startup memory profile before the backbone and head loads, for the
/// same reason the DFlash worker does: this worker never constructs
/// `Qwen35RuntimeWeightCache`, so it is a startup path the policy would otherwise
/// never reach, and it is the path that needs it — it holds the whole text tower
/// plus the head and warms every legal round width before the protocol hello.
private func applyQwenMTPStartupMemoryProfile() {
    let policy = RuntimeStartupMemoryPolicy.resolve(
        physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
        requestedProfile: ProcessInfo.processInfo.environment[
            RuntimeStartupMemoryPolicy.profileOverrideEnvironmentName
        ]
    )
    guard policy.isLowMemory else { return }
    setenv("MLX_MAX_MB_PER_BUFFER", String(policy.maxMegabytesPerCommandBuffer), 1)
    setenv("MLX_MAX_OPS_PER_BUFFER", String(policy.maxOperationsPerCommandBuffer), 1)
    for (name, value) in policy.environmentOverrides {
        setenv(name, value, 0)
    }
    Memory.cacheLimit = policy.cacheLimitBytes
    fputs(
        "mlxfast-worker: low-memory startup profile engaged ("
            + policy.selectionReason + ")\n",
        stderr
    )
}

private final class QwenMTPAsyncResultBox<T>: @unchecked Sendable {
    var result: Result<T, Error>?
}

/// Bridge the vendored async loader into the synchronous worker startup path.
/// Same reasoning as the DFlash bridge: the values crossing this boundary are
/// non-`Sendable` vendored types, and the hop is safe because it happens once,
/// on a single thread, before the protocol loop begins.
private func waitForQwenMTPAsync<T>(
    _ operation: @escaping () async throws -> T
) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    let box = QwenMTPAsyncResultBox<T>()
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
            "the MTP async model load completed without a result")
    }
    return try result.get()
}
#endif
