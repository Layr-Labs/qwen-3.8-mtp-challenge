import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon

// The SERIAL reference for the native-MTP track.
//
// Two jobs, one session type:
//
//   1. It generates the track's reference rows (`mtp-verify --generate`): the
//      plain autoregressive greedy trajectory, one token at a time, which is the
//      trajectory native-MTP decode is required to reproduce EXACTLY. This is the
//      `serialGreedy` half of the validated driver, moved into the worker.
//   2. It prices the rows a round's golden cannot reach. After a rejection at
//      draft index `a`, verify rows `a+1 ...` are conditioned on a prefix that
//      contains a rejected token, so no serial golden row describes them. The
//      parent hands this session the candidate's OWN verify block
//      (`[primary] + drafts`) and it replays it as a branch off the same
//      continuously-advanced frame, which is what lets every declared row be
//      reference-checked instead of leaving the rejected tail unpriced.
//
// FRAME DISCIPLINE, which is the whole reason this is stateful. The frame a
// token was produced in has to be the frame it is checked in: one bulk forward
// over the seed, then one single-token forward per position after it. A
// stateless implementation would have to reconstruct the prefix with a bulk
// forward over `tokens[0 ..< startOffset]`, and bulk-forwarding 517 tokens is not
// bulk-forwarding 512 and then single-stepping 5 — the accumulation order differs
// and the recurrent layers take a different chunk path, so at a near tie the
// "sequential" argmax stops being what a sequential decoder produces. Keeping one
// cache and walking it forward is both faithful and O(n). Determinism stays
// provable because an out-of-order request rebuilds the frame from scratch with
// the same construction and must agree — that is the self-consistency replay.

/// One reference row: the serial argmax at a position plus its top-2 readout.
public struct Qwen36MTPReferenceRow {
    public let sequentialArgmax: Int
    public let top2Tokens: [Int]
    public let top2Logits: [Double]
    /// Top-1 logit value, carried separately so a consumer does not have to
    /// re-derive it from the ordered pair.
    public let top1Logit: Double
}

/// A batch of reference rows plus, when one was requested, the replay of the
/// candidate's own verify block.
public struct Qwen36MTPReferenceAnswer {
    public let rows: [Qwen36MTPReferenceRow]
    public let verifyBlockTop2Tokens: [[Int]]?
    public let verifyBlockTop2Logits: [[Double]]?
}

public final class Qwen36MTPReferenceSession {
    private let model: any Qwen36MTPTarget
    private var cache: [any KVCache] = []
    /// Absolute position of the token whose logits the frame currently holds,
    /// i.e. how many tokens have been fed. -1 before the seed prefill.
    private var fed = -1
    private var seedTokens: [Int] = []

    public init(model: any Qwen36MTPTarget) {
        self.model = model
    }

    /// Bulk-forward the seed and return the argmax of its last row.
    ///
    /// This is exactly what `Qwen36MTPBlockSession.begin` computes, so a correct
    /// candidate reproduces it; bulk-forwarding all but the last seed token and
    /// then single-stepping it is a DIFFERENT computation that can disagree at a
    /// near tie and fail an honest run at its very first check.
    @discardableResult
    public func prefillSeed(_ tokens: [Int]) throws -> Int {
        guard !tokens.isEmpty else {
            throw MLXFastError.invalidInput(
                "the MTP reference seed prefill requires a non-empty seed")
        }
        cache = model.newCache(parameters: nil)
        let (logits, _) = model.callWithHidden(
            input: LMInput.Text(tokens: MLXArray(tokens).reshaped([1, tokens.count])),
            cache: cache, nConfirmed: 0)
        eval(cache.flatMap { $0.state })
        eval(logits)
        seedTokens = tokens
        fed = tokens.count
        return argmaxLast(logits)
    }

    /// Walk the width-1 frame over `count` positions starting at `startOffset`.
    ///
    /// Row `j` is fed `tokens[startOffset + j]` and predicts `tokens[startOffset +
    /// j + 1]`. Requests arriving in order are a pure continuation; an
    /// out-of-order request rebuilds the frame from the seed with the SAME
    /// construction, which is what makes the replay a determinism proof rather
    /// than a cache read.
    public func rows(
        tokens: [Int],
        seedTokenCount: Int,
        startOffset: Int,
        count: Int,
        verifyBlockTokens: [Int]? = nil
    ) throws -> Qwen36MTPReferenceAnswer {
        guard seedTokenCount > 0, seedTokenCount <= tokens.count else {
            throw MLXFastError.invalidInput(
                "the MTP reference row request has an out-of-range seed length")
        }
        guard startOffset >= seedTokenCount, count > 0,
              startOffset + count <= tokens.count
        else {
            throw MLXFastError.invalidInput(
                "the MTP reference row request reaches outside the supplied "
                    + "context: offset \(startOffset) + \(count) > \(tokens.count)")
        }
        if fed < 0 || fed > startOffset
            || !seedTokens.elementsEqual(tokens[0 ..< seedTokenCount])
        {
            try rebuildFrame(tokens: tokens, seedTokenCount: seedTokenCount)
        }
        // Catch the frame up to the first requested position without recording.
        while fed < startOffset {
            _ = try step(tokens[fed])
        }

        // The verify-block replay runs FIRST, at exactly `startOffset` — the
        // frame the candidate's round stood in when it ran the same block. It
        // branches off the walk, runs the block in ONE batched forward exactly as
        // the candidate ran it, then undoes the branch with the same snapshot +
        // trim + restore the candidate uses, so the walk below is still a pure
        // continuation. Replaying it AFTER the row walk would price the tail at
        // the wrong position.
        var verifyTokens: [[Int]]?
        var verifyLogits: [[Double]]?
        if let block = verifyBlockTokens, !block.isEmpty {
            let base = Qwen36MTPBlockSession.trimmableOffset(cache)
            let snapshot = Qwen36MTPBlockSession.snapshotRecurrent(cache)
            let (blockLogits, _) = model.callWithHidden(
                input: LMInput.Text(
                    tokens: MLXArray(block).reshaped([1, block.count])),
                cache: cache, nConfirmed: 0)
            var ids: [[Int]] = []
            var values: [[Double]] = []
            for index in 0 ..< block.count {
                let (rowIds, rowValues) =
                    Qwen36MTPBlockSession.topTwo(of: blockLogits[0, index])
                ids.append(rowIds)
                values.append(rowValues)
            }
            Qwen36MTPBlockSession.rollbackAfterVerify(
                cache, snapshot, verifiedTokens: block.count, to: base)
            eval(cache.flatMap { $0.state })
            verifyTokens = ids
            verifyLogits = values
        }

        var rows: [Qwen36MTPReferenceRow] = []
        rows.reserveCapacity(count)
        for index in 0 ..< count {
            let logits = try step(tokens[startOffset + index])
            let row = logits[0, logits.dim(1) - 1]
            let (ids, values) = Qwen36MTPBlockSession.topTwo(of: row)
            rows.append(
                Qwen36MTPReferenceRow(
                    sequentialArgmax: ids.first ?? -1,
                    top2Tokens: ids,
                    top2Logits: values,
                    top1Logit: values.first ?? 0
                ))
        }

        return Qwen36MTPReferenceAnswer(
            rows: rows,
            verifyBlockTop2Tokens: verifyTokens,
            verifyBlockTop2Logits: verifyLogits
        )
    }

    private func rebuildFrame(tokens: [Int], seedTokenCount: Int) throws {
        _ = try prefillSeed(Array(tokens[0 ..< seedTokenCount]))
    }

    /// One single-token forward. `fed` counts tokens handed to the model, so the
    /// returned logits predict `tokens[fed]` after the call.
    private func step(_ token: Int) throws -> MLXArray {
        let (logits, _) = model.callWithHidden(
            input: LMInput.Text(tokens: MLXArray([token]).reshaped([1, 1])),
            cache: cache, nConfirmed: 0)
        eval(cache.flatMap { $0.state })
        eval(logits)
        fed += 1
        return logits
    }

    private func argmaxLast(_ logits: MLXArray) -> Int {
        let row = logits[0..., (logits.dim(1) - 1) ..< logits.dim(1), 0...]
        return argMax(row, axis: -1).item(Int.self)
    }
}
