import Foundation
import MLXFastCore

// Trusted-parent contract types for the Qwen 3.6 native-MTP track
// (`qwen3.8-27b-mtp-v1`). Links no MLX: every type here is arithmetic over what
// the sandboxed worker reported plus what the pinned reference replayed.
//
// THE FIDELITY CONTRACT, and how it differs from DFlash's. Native MTP on this
// checkpoint is measured exact-greedy against the serial trajectory (12/12 runs
// to 512 tokens, all EOS branches, MTPLX-corroborated), which is a STRONGER
// property than the DFlash track can assert: DFlash needed a bounded near-tie
// budget because its target's own block-shaped forward diverges from its
// sequential forward with no drafter involved. Here the emitted stream is
// required to equal the serial trajectory OUTRIGHT -- `allTokensMatched` is not
// a budget, it is an equality.
//
// The one place a tolerance survives is the REJECTED TAIL, and it is not about
// emitted tokens at all. Those rows are replayed by the pinned reference in the
// candidate's own verify frame; the argmax should agree, but the two frames were
// built differently (the candidate's cache carries a verify + rollback + repair
// history, the reference's a width-1 walk), so a near-tie row can legitimately
// flip. Such a row is admitted and COUNTED, never hidden.
//
// A claim about this track should be worded "decision-level exact up to bf16
// shape-dependent reduction geometry", never "bitwise": the logit VALUES do move
// between a batched and a sequential frame; what does not move is which token
// wins.

/// Wire shape of the track's reference rows.
///
/// Deliberately its own type rather than a reuse of `DFlashReferenceGolden`,
/// despite carrying the same keys: the Laguna/DFlash surface is scheduled for
/// excision when the dedicated Qwen repository is created, and this track's
/// golden format must not go with it. The keys are identical on purpose -- the
/// local runner and the provisioning workflow already validate goldens in that
/// shape.
public struct QwenMTPReferenceGolden: Codable {
    public struct Row: Codable {
        public let sequentialArgmax: Int
        public let top2Tokens: [Int]?
        public let top2Logits: [Double]?
        public let top1Logit: Double?

        enum CodingKeys: String, CodingKey {
            case sequentialArgmax = "sequential_argmax"
            case top2Tokens = "top2_tokens"
            case top2Logits = "top2_logits"
            case top1Logit = "top1_logit"
        }

        public init(
            sequentialArgmax: Int,
            top2Tokens: [Int]?,
            top2Logits: [Double]?,
            top1Logit: Double?
        ) {
            self.sequentialArgmax = sequentialArgmax
            self.top2Tokens = top2Tokens
            self.top2Logits = top2Logits
            self.top1Logit = top1Logit
        }
    }

    public let seedTokens: [Int]
    /// The serial argmax of the seed's last row -- the run's FIRST emitted token.
    /// It has no `rows` entry because no row precedes it; `rows[i]` describes the
    /// token emitted at index `i + 1`.
    public let referenceSeedToken: Int
    public let rows: [Row]
    public let referenceSelfConsistent: Bool?
    public let emittedTokens: [Int]?

    enum CodingKeys: String, CodingKey {
        case seedTokens = "seed_tokens"
        case referenceSeedToken = "reference_seed_token"
        case rows
        case referenceSelfConsistent = "reference_self_consistent"
        case emittedTokens = "emitted_tokens"
    }

    public init(
        seedTokens: [Int],
        referenceSeedToken: Int,
        rows: [Row],
        referenceSelfConsistent: Bool?,
        emittedTokens: [Int]? = nil
    ) {
        self.seedTokens = seedTokens
        self.referenceSeedToken = referenceSeedToken
        self.rows = rows
        self.referenceSelfConsistent = referenceSelfConsistent
        self.emittedTokens = emittedTokens
    }
}

public struct QwenMTPContractViolation: Error, CustomStringConvertible {
    public enum Kind: String, Sendable, CaseIterable {
        case seedTokenMismatch = "seed_token_mismatch"
        case tokenNotExact = "token_not_exact"
        case rowLedgerNotClosed = "row_ledger_not_closed"
        case rowNotReferenceChecked = "row_not_reference_checked"
        case rejectedTailDiverged = "rejected_tail_diverged"
        case referenceNotSelfConsistent = "reference_not_self_consistent"
        /// RETIRED as a refusal on 2026-08-14 (non-drafting rounds are legal
        /// under serial-anchored scoring) and RETAINED as a case so the
        /// redactor's kind coverage, the wrapper's rejection-class table and
        /// any archived report that names it still resolve. Nothing throws it.
        case stopTokenInsideWindow = "stop_token_inside_window"
        case headNotUsed = "head_not_used"
    }

    public let kind: Kind
    public let step: Int?
    public let detail: String

    public init(kind: Kind, step: Int? = nil, detail: String) {
        self.kind = kind
        self.step = step
        self.detail = detail
    }

    public var description: String {
        "qwen-mtp contract violation [\(kind.rawValue)]"
            + (step.map { " at step \($0)" } ?? "") + ": " + detail
    }
}

/// One journalled round, as the parent recorded it.
public struct QwenMTPObservedRound: Sendable {
    /// Index into the emitted stream of this round's primary.
    public let emittedBaseIndex: Int
    public let requestedDepth: Int
    public let tokens: [Int]
    public let declaredRows: Int
    public let draftTokens: [Int]
    public let acceptedDraftCount: Int
    public let rejectedDraftCount: Int
    public let perRowTop2Tokens: [[Int]]
    public let perRowTop2Logits: [[Double]]
    public let targetCacheOffset: Int
    public let latencySeconds: Double

    public init(
        emittedBaseIndex: Int,
        requestedDepth: Int,
        tokens: [Int],
        declaredRows: Int,
        draftTokens: [Int],
        acceptedDraftCount: Int,
        rejectedDraftCount: Int,
        perRowTop2Tokens: [[Int]],
        perRowTop2Logits: [[Double]],
        targetCacheOffset: Int,
        latencySeconds: Double
    ) {
        self.emittedBaseIndex = emittedBaseIndex
        self.requestedDepth = requestedDepth
        self.tokens = tokens
        self.declaredRows = declaredRows
        self.draftTokens = draftTokens
        self.acceptedDraftCount = acceptedDraftCount
        self.rejectedDraftCount = rejectedDraftCount
        self.perRowTop2Tokens = perRowTop2Tokens
        self.perRowTop2Logits = perRowTop2Logits
        self.targetCacheOffset = targetCacheOffset
        self.latencySeconds = latencySeconds
    }

    /// The candidate's own verify input for this round, reconstructed by the
    /// PARENT from its committed primary plus the journalled drafts, so the
    /// worker never chooses the row-0 token of its own audit.
    public var verifyBlockTokens: [Int] {
        guard let primary = tokens.first, !draftTokens.isEmpty else { return [] }
        return [primary] + draftTokens
    }
}

/// One audited row of the ledger, and how it was reference-checked.
public struct QwenMTPLedgerRow: Sendable {
    public enum Kind: String, Sendable {
        /// A row of the batched verify forward that scored a head proposal.
        case draft
        /// The single per-round row whose argmax becomes the next primary.
        case targetTail
    }

    public enum ReferenceSource: String, Sendable {
        /// Checked against the serial golden's row at this emitted index.
        case serialGolden = "serial_golden"
        /// Checked against the pinned reference's replay of the candidate's own
        /// verify block -- the only way to reach a row conditioned on a rejected
        /// prefix, which no serial golden describes.
        case verifyBlockReplay = "verify_block_replay"
    }

    public let rowIndex: Int
    public let round: Int
    public let kind: Kind
    /// Position within the round's draft window, or nil for the tail row.
    public let draftIndex: Int?
    public let accepted: Bool
    /// The head's proposal for a draft row; the committed token for a tail row.
    public let token: Int
    public let top2Tokens: [Int]
    public let top2Logits: [Double]
    public let referenceToken: Int
    public let referenceCheckedBy: ReferenceSource
    /// Reference top1-top2 gap. A near-zero gap is where a batched-versus-
    /// sequential argmax may legitimately flip; a large gap means the two paths
    /// genuinely disagree, which is a logic bug and not numerics.
    public let referenceMargin: Double

    public var top2Margin: Double {
        guard top2Logits.count >= 2 else { return .infinity }
        return top2Logits[0] - top2Logits[1]
    }
}

/// How close two logit readouts of the same row must be before a token
/// disagreement is read as numerics rather than as a logic bug.
///
/// PROVISIONAL AND UNCALIBRATED FOR THIS CHECKPOINT. It applies ONLY to
/// rejected-tail rows replayed in a different cache history; no emitted token is
/// ever admitted by it. `1e-2` is the margin the validated driver used to
/// classify a mismatch as a tie-flip rather than a logic bug, carried over
/// unchanged. Box-3 calibration required before the first ranked session.
public struct QwenMTPNearTieTolerance: Sendable {
    public let referenceMargin: Double

    public init(referenceMargin: Double = 1e-2) {
        self.referenceMargin = referenceMargin
    }
}

/// Where the MTP head this run drafted from actually came from.
///
/// RECORDED, NEVER GATED (operator-ratified 2026-08-14). Head weights are part
/// of the competitive surface, so "which head" stopped being a pass-condition
/// and became provenance: the sealed report carries the digest and the source
/// of whatever head was loaded, and the leaderboard entry is readable back to
/// it. The BASELINE leg always resolves to `pinned`, which is what keeps the
/// paired ratio anchored to one fixed reference.
public struct QwenMTPHeadProvenance: Equatable, Sendable {
    /// `pinned` | `remote` | `in_branch`, as declared by
    /// `mtp-head.manifest.json` (absent manifest = `pinned`).
    public let source: String
    /// SHA-256 over the head tree: the digest of the sorted
    /// `"<sha256>  <relative path>\n"` manifest of every regular file, so one
    /// number describes a multi-file head and the participant can recompute it
    /// with `find | sort | shasum | shasum`.
    public let sha256: String
    public let bytes: Int
    public let fileCount: Int
    /// Where it was declared from, verbatim (`hf:repo@rev`, `r2:key`, a
    /// repo-relative path, or the pinned cache path). Diagnostic only.
    public let origin: String

    public init(
        source: String,
        sha256: String,
        bytes: Int,
        fileCount: Int,
        origin: String
    ) {
        self.source = source
        self.sha256 = sha256
        self.bytes = bytes
        self.fileCount = fileCount
        self.origin = origin
    }
}

/// Options shared by both verbs. `depth == 0` is the serial control.
public struct QwenMTPOptions: Equatable {
    public let targetWeightsPath: String
    public let mtpHeadPath: String
    public let goldenPath: String
    public let depth: Int
    public let totalTokenCount: Int
    public let referenceWeightsPath: String?
    public let referenceHeadPath: String?

    public init(
        targetWeightsPath: String,
        mtpHeadPath: String,
        goldenPath: String,
        depth: Int,
        totalTokenCount: Int,
        referenceWeightsPath: String? = nil,
        referenceHeadPath: String? = nil
    ) {
        self.targetWeightsPath = targetWeightsPath
        self.mtpHeadPath = mtpHeadPath
        self.goldenPath = goldenPath
        self.depth = depth
        self.totalTokenCount = totalTokenCount
        self.referenceWeightsPath = referenceWeightsPath
        self.referenceHeadPath = referenceHeadPath
    }
}

/// The evidence payload both verbs emit. Field names are the consumer contract:
/// `deploy/qwen36-mtp/measure-qwen-mtp-job.sh`, the ranked workflow's gate jq,
/// and `benchmark-qwen-mtp.sh`.
public struct QwenMTPReport: Equatable {
    public let verb: String
    public let depth: Int
    public let seedTokenCount: Int
    public let decodeTokenCount: Int
    public let emittedTokenTotal: Int
    public let allTokensMatched: Bool
    public let firstDivergenceIndex: Int?
    public let firstDivergenceReferenceMargin: Double?
    public let roundCount: Int
    public let acceptedDraftTotal: Int
    public let rejectedDraftTotal: Int
    public let targetTailTotal: Int
    public let declaredRowTotal: Int
    public let referenceCheckedRowTotal: Int
    public let rejectedRowsReferenceChecked: Int
    public let verifyBlockReplayedRoundCount: Int
    public let residualDivergenceCount: Int
    public let maxRejectedTailLogitDelta: Double
    public let targetCacheOffsetFinal: Int
    public let decodeSeconds: Double
    /// Per-round wall time, one entry per round, in emission order.
    ///
    /// THE STALL GUARDRAIL'S PRIMARY INPUT. The box wrapper's
    /// `check_stall_guardrail` fails CLOSED unless a timed report carries either
    /// this array or the after-first trio, and it will not fall back to a
    /// whole-window max/p50 -- because the first block is a measured, one-time
    /// post-prefill warmup (flat across a 64x window sweep) and folding it back
    /// into the ratio is exactly the false rejection the exclusion exists to
    /// stop. Emitting the raw array is preferred over emitting summaries: the
    /// wrapper then does its own slice, max and median, so the guard's arithmetic
    /// is not something the measured side gets to assert.
    public let roundRequestSeconds: [Double]
    /// Whole-window max/p50, RETAINED FOR AUDIT ONLY. No guard reads these any
    /// more; they exist so a human comparing an old report to a new one is not
    /// looking at two different quantities.
    public let maxRoundRequestSeconds: Double
    public let p50RoundRequestSeconds: Double

    /// The seed prefill's own wall time, measured by the parent as the
    /// sub-interval of the decode window that `beginMTPDecode` occupied.
    ///
    /// OBSERVABILITY ONLY, and deliberately NOT subtracted from
    /// `decodeSeconds`: the paired contract charges the seed prefill to the
    /// decode measurement on both sides of the pair (see the clock note in
    /// QwenRuntimeMTPDriver), and this field does not move that boundary — it
    /// only names how much of the charged window the prefill was, so the
    /// board can state a prefill rate. Zero means "not measured" (an older
    /// caller or a verify-verb report), never "instant"; the payload writer
    /// omits the key at zero rather than publishing a rate no clock produced.
    public let seedPrefillSeconds: Double

    /// EFFECTIVE per-round draft counts, in emission order — what the
    /// candidate actually proposed, not what the parent offered.
    ///
    /// THE PROVENANCE FIX (k-matrix finding 5). Before 2026-08-14 the report
    /// carried only `depth`, the parent's requested value, and the driver
    /// happened to force the two to agree; once the candidate owns its own
    /// schedule those are different quantities and a report that named only
    /// the request would describe a run that never happened. Every summary
    /// below is derived from this array by the PARENT, from the journal it
    /// wrote itself, so none of it is a worker-asserted counter.
    public let effectiveDraftLengths: [Int]

    /// Mean drafts per round over the whole window, 0 for the serial control.
    public var effectiveMeanDraftLength: Double {
        guard !effectiveDraftLengths.isEmpty else { return 0 }
        return Double(effectiveDraftLengths.reduce(0, +))
            / Double(effectiveDraftLengths.count)
    }

    public var effectiveMaxDraftLength: Int {
        effectiveDraftLengths.max() ?? 0
    }

    /// Rounds that proposed nothing and committed only their primary. Legal
    /// since 2026-08-14; counted so an adaptive policy is legible in the
    /// sealed output rather than invisible.
    public var nonDraftingRoundCount: Int {
        effectiveDraftLengths.filter { $0 == 0 }.count
    }

    /// The trusted bound the ledger was closed against, restated in the report
    /// so a reader does not have to know the constant.
    public var maxDraftDepthBound: Int {
        MLXFastConstants.qwenMTPMaxDraftDepth
    }

    /// The excluded first block, reported rather than hidden.
    public var firstBlockSeconds: Double { roundRequestSeconds.first ?? 0 }

    /// Max over rounds AFTER the first -- the wrapper's array route, computed
    /// here so the two accepted routes cannot disagree.
    public var maxRoundRequestSecondsAfterFirst: Double {
        roundRequestSeconds.dropFirst().max() ?? 0
    }

    /// p50 over rounds AFTER the first, using the wrapper's EXACT rule:
    /// `sorted[floor((n - 1) / 2)]`, i.e. the LOWER median. Matching the jq
    /// bit-for-bit is the point -- a report that offered both routes with two
    /// different median conventions would hand the guard two different answers
    /// depending on which branch it happened to take.
    public var p50RoundRequestSecondsAfterFirst: Double {
        Self.lowerMedian(Array(roundRequestSeconds.dropFirst()))
    }

    public static func lowerMedian(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        return sorted[(sorted.count - 1) / 2]
    }
    /// Retained ledger rows. `mtp-verify` carries the whole ledger as the
    /// fidelity evidence; `mtp-timed` carries none (the timed report is a timing
    /// artifact, and a 512-token ledger of top-2 readouts would dominate it).
    public let ledger: [QwenMTPLedgerRow]

    /// True when the MTP head ACTUALLY DRAFTED during this run.
    ///
    /// REDEFINED 2026-08-14, from "the parent asked for a speculative depth"
    /// to "at least one draft was proposed". The old predicate was `depth > 0`,
    /// which was only ever a proxy: with the candidate owning its schedule a
    /// run can be offered depth 8 and legally draft nothing, and reporting
    /// `true` for it would be a false statement about what happened. Reading
    /// the ledger instead makes the field honest at the cost of nothing — the
    /// serial control still reports `false`, because depth 0 cannot draft.
    ///
    /// It is now PROVENANCE, not a gate. The ranked workflow, the local runner
    /// and the box wrapper all stopped asserting it on 2026-08-14; a
    /// non-drafting candidate is a legal submission that scores 1.0.
    public var usesNativeMTPHead: Bool {
        acceptedDraftTotal + rejectedDraftTotal > 0
    }

    /// True only for the depth this track divides by.
    public var isSerialControl: Bool {
        depth == MLXFastConstants.qwenMTPSerialControlDepth
    }
    public var decodeSecondsPerToken: Double {
        decodeSeconds / Double(Swift.max(decodeTokenCount, 1))
    }
    public var acceptedDraftRate: Double {
        let drafted = acceptedDraftTotal + rejectedDraftTotal
        return drafted > 0 ? Double(acceptedDraftTotal) / Double(drafted) : 0
    }
    /// `parity_all_ok`: the bit-exact token gate as a single top-level boolean.
    /// It is an AND of the exactness verdict and ledger closure, not a copy of
    /// either -- a run whose ledger did not close has not proven its tokens were
    /// produced by the work it declared, whatever the tokens happened to be.
    public var parityAllOK: Bool {
        allTokensMatched
            && referenceCheckedRowTotal == declaredRowTotal
            && acceptedDraftTotal + rejectedDraftTotal + targetTailTotal
                == declaredRowTotal
    }

    public init(
        verb: String,
        depth: Int,
        seedTokenCount: Int,
        decodeTokenCount: Int,
        emittedTokenTotal: Int,
        allTokensMatched: Bool,
        firstDivergenceIndex: Int?,
        firstDivergenceReferenceMargin: Double?,
        roundCount: Int,
        acceptedDraftTotal: Int,
        rejectedDraftTotal: Int,
        targetTailTotal: Int,
        declaredRowTotal: Int,
        referenceCheckedRowTotal: Int,
        rejectedRowsReferenceChecked: Int,
        verifyBlockReplayedRoundCount: Int,
        residualDivergenceCount: Int,
        maxRejectedTailLogitDelta: Double,
        targetCacheOffsetFinal: Int,
        decodeSeconds: Double,
        roundRequestSeconds: [Double] = [],
        maxRoundRequestSeconds: Double,
        p50RoundRequestSeconds: Double,
        seedPrefillSeconds: Double = 0,
        effectiveDraftLengths: [Int] = [],
        ledger: [QwenMTPLedgerRow] = []
    ) {
        self.verb = verb
        self.depth = depth
        self.seedTokenCount = seedTokenCount
        self.decodeTokenCount = decodeTokenCount
        self.emittedTokenTotal = emittedTokenTotal
        self.allTokensMatched = allTokensMatched
        self.firstDivergenceIndex = firstDivergenceIndex
        self.firstDivergenceReferenceMargin = firstDivergenceReferenceMargin
        self.roundCount = roundCount
        self.acceptedDraftTotal = acceptedDraftTotal
        self.rejectedDraftTotal = rejectedDraftTotal
        self.targetTailTotal = targetTailTotal
        self.declaredRowTotal = declaredRowTotal
        self.referenceCheckedRowTotal = referenceCheckedRowTotal
        self.rejectedRowsReferenceChecked = rejectedRowsReferenceChecked
        self.verifyBlockReplayedRoundCount = verifyBlockReplayedRoundCount
        self.residualDivergenceCount = residualDivergenceCount
        self.maxRejectedTailLogitDelta = maxRejectedTailLogitDelta
        self.targetCacheOffsetFinal = targetCacheOffsetFinal
        self.decodeSeconds = decodeSeconds
        self.roundRequestSeconds = roundRequestSeconds
        self.maxRoundRequestSeconds = maxRoundRequestSeconds
        self.p50RoundRequestSeconds = p50RoundRequestSeconds
        self.seedPrefillSeconds = seedPrefillSeconds
        self.effectiveDraftLengths = effectiveDraftLengths
        self.ledger = ledger
    }
}

extension QwenMTPLedgerRow: Equatable {}

/// Parent-side ledger arithmetic, shared by both verbs.
///
/// Every equation the box wrapper's `check_row_accounting` enforces is closed
/// HERE first, from rows the parent reference-checked itself, so the wrapper is
/// auditing a ledger rather than a set of worker-reported counters.
public enum QwenMTPRowAccounting {
    /// Rows a round declares for a given ACTUAL draft count: that many draft
    /// rows plus one target tail row. This is the constant the wrapper's
    /// `rows_per_round` mirrors; if one moves, both move.
    ///
    /// Since 2026-08-14 the argument is always the count the round actually
    /// proposed, never a configured or requested depth — a round that drafted
    /// 1 of an offered 8 declares 2 rows, and a round that drafted 0 declares
    /// the single tail row.
    public static func rowsPerRound(depth: Int) -> Int { depth + 1 }

    /// The wrapper's L3 equations, as a single predicate.
    public static func closes(
        emittedTokenTotal: Int,
        configuredTokenTotal: Int,
        declaredRowTotal: Int,
        referenceCheckedRowTotal: Int,
        acceptedDraftTotal: Int,
        rejectedDraftTotal: Int,
        targetTailTotal: Int,
        roundCount: Int,
        depth: Int,
        seedTokenCount: Int,
        targetCacheOffsetFinal: Int
    ) -> Bool {
        emittedTokenTotal == configuredTokenTotal
            // THE SERIAL CONTROL MAY NOT DRAFT. The one place a REQUESTED depth
            // still binds after 2026-08-14: depth 0 is the denominator this
            // track divides by, and a "serial" leg that drafted anything is an
            // already-accelerated baseline -- the mislabel that once produced a
            // 0.875x "speedup". Everything else below is bounded by the trusted
            // maximum instead, because the candidate owns its own schedule.
            && (depth != MLXFastConstants.qwenMTPSerialControlDepth
                || (acceptedDraftTotal == 0 && rejectedDraftTotal == 0))
            && declaredRowTotal >= emittedTokenTotal
            // BOUNDED BY THE TRUSTED MAXIMUM, not by the requested depth.
            // `depth` is what the parent offered; a candidate that owns its own
            // schedule may draft up to qwenMTPMaxDraftDepth in a round, so the
            // row ceiling has to be the constant the driver actually enforces.
            // Using the offer here would reject a legal adaptive run whose
            // rounds went wider than the offer near the tail.
            && declaredRowTotal
                <= rowsPerRound(
                    depth: Swift.max(
                        depth, MLXFastConstants.qwenMTPMaxDraftDepth))
                    * roundCount
            && acceptedDraftTotal + rejectedDraftTotal + targetTailTotal
                == declaredRowTotal
            && referenceCheckedRowTotal == declaredRowTotal
            && targetCacheOffsetFinal == seedTokenCount + emittedTokenTotal
    }
}
