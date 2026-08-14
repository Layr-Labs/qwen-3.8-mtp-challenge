// QuantizedSequenceKV.swift
//
// ContinuousBatchingV2 quantized per-sequence KV storage for FULL attention
// layers. Same growth/rollback semantics as `CBv2FullSequenceKV`, but the
// stored state is mlx-quantized triples (weights, scales, biases) following
// the existing `QuantizedKVCache` scheme (group size 64, 4/8-bit, affine).
//
// `update` returns DEQUANTIZED views for the v1 attention backend. Because
// storage is per-request, the dequantization cost is O(this sequence's
// retained KV), not O(whole batch * max length) as in the left-padded engine.

import Foundation
import MLX

/// `CBv2SequenceKV` storing quantized K/V triples.
public final class CBv2QuantizedSequenceKV: CBv2SequenceKV, CBv2InnerStateProviding {

    public private(set) var absoluteOffset: Int = 0
    public var retainedCount: Int { absoluteOffset }

    public let maxLength: Int
    public private(set) var groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    let kvHeads: Int
    let headDim: Int

    private var qKeys: (MLXArray, MLXArray, MLXArray?)?
    private var qValues: (MLXArray, MLXArray, MLXArray?)?
    private var capacity: Int
    /// dtype of the unquantized inputs; dequantized views are returned in it.
    private var sourceDType: DType = .float16

    public init(
        promptLength: Int, maxLength: Int, kvHeads: Int, headDim: Int,
        groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        precondition(maxLength > 0, "CBv2QuantizedSequenceKV: maxLength must be > 0")
        precondition(
            promptLength <= maxLength,
            "CBv2QuantizedSequenceKV: promptLength \(promptLength) exceeds maxLength \(maxLength)")
        precondition(
            bits == 4 || bits == 8,
            "CBv2QuantizedSequenceKV: only 4/8-bit supported (got \(bits))")
        self.maxLength = maxLength
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.groupSize = Self.resolvedGroupSize(requested: groupSize, headDim: headDim)
        self.bits = bits
        self.mode = mode
        self.capacity = min(maxLength, max(1, promptLength + CBv2FullSequenceKV.initialSlack))
    }

    /// Pick the supported group size ({32, 64, 128}) closest to the requested
    /// one that divides the head dim (same policy as `QuantizedKVCache`).
    static func resolvedGroupSize(requested: Int, headDim: Int) -> Int {
        let compatible = [32, 64, 128].filter { headDim.isMultiple(of: $0) }
        guard !compatible.isEmpty else {
            fatalError(
                "CBv2QuantizedSequenceKV: head dim \(headDim) is not divisible by any supported group size (32, 64, 128)"
            )
        }
        return compatible.min {
            let l = abs($0 - requested)
            let r = abs($1 - requested)
            return l == r ? $0 < $1 : l < r
        }!
    }

    public var byteCount: Int {
        var total = 0
        for triple in [qKeys, qValues].compactMap({ $0 }) {
            total += triple.0.nbytes + triple.1.nbytes + (triple.2?.nbytes ?? 0)
        }
        return total
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let n = newKeys.dim(2)
        precondition(newKeys.dim(0) == 1 && newValues.dim(0) == 1,
            "CBv2QuantizedSequenceKV holds ONE sequence; got batch \(newKeys.dim(0))")
        precondition(newKeys.dim(1) == kvHeads,
            "CBv2QuantizedSequenceKV: kvHeads mismatch (\(newKeys.dim(1)) != \(kvHeads))")
        precondition(newKeys.dim(3) == headDim && newValues.dim(3) == headDim,
            "CBv2QuantizedSequenceKV: head dim mismatch")
        precondition(newValues.dim(2) == n,
            "CBv2QuantizedSequenceKV: keys/values token count mismatch")
        precondition(
            absoluteOffset + n <= maxLength,
            "CBv2QuantizedSequenceKV: append past maxLength (\(absoluteOffset) + \(n) > \(maxLength)) — admission bug"
        )

        sourceDType = newKeys.dtype
        ensureCapacity(absoluteOffset + n, template: newKeys)

        let quantizedNewKeys = quantized(newKeys, groupSize: groupSize, bits: bits, mode: mode)
        let quantizedNewValues = quantized(newValues, groupSize: groupSize, bits: bits, mode: mode)

        assign(
            (quantizedNewKeys.wq, quantizedNewKeys.scales, quantizedNewKeys.biases),
            into: qKeys!, at: absoluteOffset, count: n)
        assign(
            (quantizedNewValues.wq, quantizedNewValues.scales, quantizedNewValues.biases),
            into: qValues!, at: absoluteOffset, count: n)
        absoluteOffset += n

        return (dequantizedKeys(upTo: absoluteOffset), dequantizedValues(upTo: absoluteOffset))
    }

    /// Snapshot returns DEQUANTIZED arrays (a bounded one-time copy, "zero-
    /// copy-ish") in the backend-agnostic contract shape.
    ///
    /// LOSSY (`snapshotIsLossless == false`): MLX affine quantization snaps
    /// the group scale to its dominant edge (`scale = edge / round(edge /
    /// scale)`), so re-quantizing these dequantized values on adoption lands
    /// on a DIFFERENT grid — up to ~1 quantization step of drift per
    /// donate→adopt generation, compounding. The engine therefore never
    /// donates quantized rows to the prefix cache (see
    /// `CBv2SequenceKV.snapshotIsLossless`); this snapshot remains valid for
    /// checkpointing/inspection and for adoption INTO fresh state where the
    /// single re-quantization is explicitly accepted.
    public func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        guard qKeys != nil, qValues != nil, absoluteOffset > 0 else {
            return (
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: sourceDType),
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: sourceDType),
                absoluteOffset
            )
        }
        return (
            dequantizedKeys(upTo: absoluteOffset),
            dequantizedValues(upTo: absoluteOffset),
            absoluteOffset
        )
    }

    /// Quantized snapshots dequantize — re-quantization is not value-exact
    /// (see `snapshot()`), so the engine must not donate this row's state.
    public var snapshotIsLossless: Bool { false }

    /// See `CBv2FullSequenceKV.rollback` — the stale tail is structurally
    /// unreachable (views sliced to `..<absoluteOffset`, overwritten before
    /// re-exposure).
    public func rollback(_ n: Int) {
        precondition(n >= 0, "CBv2QuantizedSequenceKV.rollback: negative n")
        precondition(
            n <= absoluteOffset,
            "CBv2QuantizedSequenceKV.rollback(\(n)) exceeds retained \(absoluteOffset)")
        absoluteOffset -= n
    }

    func cbv2InnerState() -> [MLXArray] {
        var arrays: [MLXArray] = []
        for triple in [qKeys, qValues].compactMap({ $0 }) {
            arrays.append(triple.0)
            arrays.append(triple.1)
            if let biases = triple.2 { arrays.append(biases) }
        }
        return arrays
    }

    // MARK: - Private

    private func dequantizedKeys(upTo offset: Int) -> MLXArray {
        dequantizedView(qKeys!, upTo: offset)
    }

    private func dequantizedValues(upTo offset: Int) -> MLXArray {
        dequantizedView(qValues!, upTo: offset)
    }

    private func dequantizedView(_ triple: (MLXArray, MLXArray, MLXArray?), upTo offset: Int)
        -> MLXArray
    {
        dequantized(
            triple.0[.ellipsis, ..<offset, 0...],
            scales: triple.1[.ellipsis, ..<offset, 0...],
            biases: triple.2.map { $0[.ellipsis, ..<offset, 0...] },
            groupSize: groupSize, bits: bits, mode: mode,
            dtype: sourceDType)
    }

    private func assign(
        _ source: (MLXArray, MLXArray, MLXArray?),
        into destination: (MLXArray, MLXArray, MLXArray?),
        at offset: Int, count: Int
    ) {
        destination.0[.ellipsis, offset ..< (offset + count), 0...] = source.0
        destination.1[.ellipsis, offset ..< (offset + count), 0...] = source.1
        if let sourceBiases = source.2 {
            destination.2![.ellipsis, offset ..< (offset + count), 0...] = sourceBiases
        }
    }

    private func ensureCapacity(_ needed: Int, template: MLXArray) {
        if qKeys == nil {
            capacity = min(maxLength, max(capacity, needed))
            qKeys = quantizedZeros(count: capacity, template: template)
            qValues = quantizedZeros(count: capacity, template: template)
            return
        }
        guard needed > capacity else { return }

        let newCapacity = min(maxLength, max(capacity * 2, needed))
        let growth = newCapacity - capacity
        let zeros = quantizedZeros(count: growth, template: template)
        qKeys = concatTriples(qKeys!, zeros)
        let zerosV = quantizedZeros(count: growth, template: template)
        qValues = concatTriples(qValues!, zerosV)
        capacity = newCapacity
    }

    /// Quantized triple representing `count` zero tokens (the `initQuant`
    /// idiom from `QuantizedKVCache`, kept for wire-format compatibility).
    private func quantizedZeros(count: Int, template: MLXArray)
        -> (MLXArray, MLXArray, MLXArray?)
    {
        let zeros = MLXArray.zeros([1, kvHeads, count, headDim], dtype: template.dtype)
        let q = quantized(zeros, groupSize: groupSize, bits: bits, mode: mode)
        return (q.wq, q.scales, q.biases)
    }

    private func concatTriples(
        _ lhs: (MLXArray, MLXArray, MLXArray?), _ rhs: (MLXArray, MLXArray, MLXArray?)
    ) -> (MLXArray, MLXArray, MLXArray?) {
        (
            concatenated([lhs.0, rhs.0], axis: -2),
            concatenated([lhs.1, rhs.1], axis: -2),
            lhs.2.flatMap { l in rhs.2.map { r in concatenated([l, r], axis: -2) } }
        )
    }
}
