// ContiguousKVBackend.swift
//
// The v1 `CBv2KVBackend`: per-sequence contiguous MLX buffers
// (`CBv2FullSequenceKV` / `CBv2WindowedSequenceKV` / `CBv2QuantizedSequenceKV`).
// The paged backend (workstream C) implements the same protocol behind a
// Metal kernel; the scheduler and models never see the difference.

import Foundation
import MLX

/// Configuration for `CBv2ContiguousKVBackend`.
public struct CBv2ContiguousBackendConfig: Sendable {
    /// Byte budget for all live sequence KV (admission ceiling).
    public var bytesCapacity: Int
    /// Optional KV quantization for FULL-attention layers (group size, bits).
    /// Windowed layers stay unquantized (small ring, quantization gains
    /// nothing and the modular writes would fight group boundaries).
    public var quantization: (groupSize: Int, bits: Int)?
    /// dtype assumed for admission estimates (actual allocation adopts the
    /// dtype of the first appended K/V).
    public var kvDType: DType

    public init(
        bytesCapacity: Int,
        quantization: (groupSize: Int, bits: Int)? = nil,
        kvDType: DType = .float16
    ) {
        self.bytesCapacity = bytesCapacity
        self.quantization = quantization
        self.kvDType = kvDType
    }
}

/// Factory + accounting for per-sequence contiguous KV state.
///
/// Thread-safe: the live-row registry is lock-protected (`makeSequenceState`
/// runs on the admission path while `release` runs on the engine loop).
/// `bytesInUse` is truthful — it sums the ACTUAL allocated bytes of live
/// rows (which grow by doubling), not a worst-case estimate.
///
/// Admission RESERVES: rows allocate lazily (`byteCount == 0` until their
/// first update), so judging capacity against `bytesInUse` alone would let
/// several same-step admissions collectively exceed `bytesCapacity`. Each
/// admitted row therefore holds a reservation equal to its estimated
/// initial bytes until its actual allocation exceeds it
/// (`max(byteCount, reservation)` per row — see `bytesReserved`), and the
/// capacity check + registration are a single atomic section.
public final class CBv2ContiguousKVBackend: CBv2KVBackend {

    public let config: CBv2ContiguousBackendConfig

    /// Compiled [B, 1] decode can bind the fp16 rows this backend mints
    /// (`CBv2FullSequenceKV` / `CBv2WindowedSequenceKV`) — but NOT the
    /// `CBv2QuantizedSequenceKV` rows a quantized config produces for
    /// full-attention layers, so a quantized backend must veto the compiled
    /// build (no warmup, no admission padding reserve — PR#62 review).
    public var producesCompiledDecodeEligibleRows: Bool { config.quantization == nil }

    private let lock = NSLock()
    private var live: [ObjectIdentifier: CBv2SequenceKV] = [:]
    /// Admission reservation per live row (estimated initial bytes),
    /// released with the row. NOTE: estimates assume `config.kvDType`; a
    /// model that caches wider elements (e.g. fp32) under-reserves until
    /// the first update trues the row up to its actual `byteCount` —
    /// `AdmissionV2` (which can carry per-layer element sizes) remains the
    /// primary admission gate.
    private var reservations: [ObjectIdentifier: Int] = [:]

    public init(config: CBv2ContiguousBackendConfig) {
        self.config = config
    }

    public var bytesCapacity: Int { config.bytesCapacity }

    public var bytesInUse: Int {
        lock.lock()
        defer { lock.unlock() }
        return live.values.reduce(0) { $0 + $1.byteCount }
    }

    /// Actual bytes plus outstanding admission reservations — what the
    /// capacity check judges against (`CBv2KVBackend.bytesReserved`).
    public var bytesReserved: Int {
        lock.lock()
        defer { lock.unlock() }
        return accountedBytesLocked()
    }

    public func makeSequenceState(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        try validate(layerKinds: layerKinds)
        guard promptLength <= maxLength else {
            throw CBv2KVError.backendIneligible(
                reason: "promptLength \(promptLength) exceeds maxLength \(maxLength)")
        }

        let state = layerKinds.map { kind -> CBv2SequenceKV? in
            makeRow(kind: kind, promptLength: promptLength, maxLength: maxLength)
        }
        try registerReserving(
            state,
            estimates: rowEstimates(
                layerKinds: layerKinds, promptLength: promptLength, maxLength: maxLength))
        return state
    }

    public func makeSequenceState(
        adopting prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        layerKinds: [CBv2LayerKind], maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        try validate(layerKinds: layerKinds)
        guard prefix.count == layerKinds.count else {
            throw CBv2KVError.backendIneligible(
                reason: "prefix count \(prefix.count) != layer count \(layerKinds.count)")
        }

        // The matched prefix length: uniform across all donated full-attention
        // layers by construction of the prefix cache.
        var matched = 0
        for (index, entry) in prefix.enumerated() {
            guard let entry else { continue }
            let kind = layerKinds[index]
            guard kind.sharesKVWithLayer == nil else {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index) is KV-shared but received a prefix snapshot")
            }
            guard case .full = kind.attention else {
                throw CBv2KVError.backendIneligible(
                    reason:
                        "layer \(index) is windowed but received a prefix snapshot (windowed layers are recomputed)"
                )
            }
            if matched == 0 { matched = entry.offset }
            guard entry.offset == matched else {
                throw CBv2KVError.backendIneligible(
                    reason:
                        "non-uniform prefix offsets (\(entry.offset) vs \(matched)) at layer \(index)"
                )
            }
        }
        guard matched <= maxLength else {
            throw CBv2KVError.backendIneligible(
                reason: "prefix offset \(matched) exceeds maxLength \(maxLength)")
        }

        let state = layerKinds.enumerated().map { index, kind -> CBv2SequenceKV? in
            guard kind.sharesKVWithLayer == nil else { return nil }
            switch kind.attention {
            case .slidingWindow(let window):
                // Windowed layers never receive a snapshot. Reconciled
                // adoption semantics (contract `makeSequenceState(adopting:)`):
                // the engine already sliced the prefix down by
                // `cbv2RequiredRecompute`, so EVERY row starts at the uniform
                // adopted offset and the engine replays [matched, prompt)
                // through all layers.
                return CBv2WindowedSequenceKV(
                    window: window, kvHeads: kind.kvHeads, headDim: kind.headDim,
                    initialOffset: matched)
            case .full:
                let row = makeRow(kind: kind, promptLength: matched, maxLength: maxLength)!
                if let entry = prefix[index], entry.offset > 0 {
                    // Bounded one-time copy of the donated arrays into fresh
                    // sequence state (the paged backend makes this free).
                    _ = row.update(keys: entry.keys, values: entry.values)
                }
                return row
            }
        }
        try registerReserving(
            state,
            estimates: rowEstimates(
                layerKinds: layerKinds, promptLength: matched, maxLength: maxLength))
        return state
    }

    public func release(_ state: [CBv2SequenceKV?]) {
        lock.lock()
        defer { lock.unlock() }
        for row in state {
            guard let row else { continue }
            let key = ObjectIdentifier(row)
            live.removeValue(forKey: key)
            reservations.removeValue(forKey: key)
        }
    }

    // MARK: - Private

    /// Bytes currently charged against capacity: each live row counts for
    /// the LARGER of its actual allocation and its outstanding admission
    /// reservation, so a not-yet-allocated row still occupies its estimate
    /// and a grown row is charged its true size (`bytesInUse`-truthful).
    /// Caller holds `lock`.
    private func accountedBytesLocked() -> Int {
        live.reduce(0) { total, entry in
            total + max(entry.value.byteCount, reservations[entry.key] ?? 0)
        }
    }

    /// Atomically admit + register: the capacity check and the reservation
    /// write share one critical section so N same-step admissions cannot
    /// collectively overshoot `bytesCapacity` (the pre-fix bug — rows report
    /// `byteCount == 0` until first update).
    private func registerReserving(_ state: [CBv2SequenceKV?], estimates: [Int?]) throws {
        precondition(state.count == estimates.count, "estimate/state count mismatch")
        lock.lock()
        defer { lock.unlock() }
        let needed = estimates.reduce(0) { $0 + ($1 ?? 0) }
        let available = bytesCapacity - accountedBytesLocked()
        guard needed <= available else {
            throw CBv2KVError.capacityExhausted(needed: needed, available: max(0, available))
        }
        for (row, estimate) in zip(state, estimates) {
            guard let row else { continue }
            let key = ObjectIdentifier(row)
            live[key] = row
            reservations[key] = estimate ?? 0
        }
    }

    private func makeRow(kind: CBv2LayerKind, promptLength: Int, maxLength: Int)
        -> CBv2SequenceKV?
    {
        guard kind.sharesKVWithLayer == nil else { return nil }
        switch kind.attention {
        case .slidingWindow(let window):
            return CBv2WindowedSequenceKV(
                window: window, kvHeads: kind.kvHeads, headDim: kind.headDim)
        case .full:
            if let quantization = config.quantization {
                return CBv2QuantizedSequenceKV(
                    promptLength: promptLength, maxLength: maxLength,
                    kvHeads: kind.kvHeads, headDim: kind.headDim,
                    groupSize: quantization.groupSize, bits: quantization.bits)
            }
            return CBv2FullSequenceKV(
                promptLength: promptLength, maxLength: maxLength,
                kvHeads: kind.kvHeads, headDim: kind.headDim)
        }
    }

    private func validate(layerKinds: [CBv2LayerKind]) throws {
        for (index, kind) in layerKinds.enumerated() {
            if case .slidingWindow(let window) = kind.attention, window <= 0 {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index): non-positive window \(window)")
            }
            if let source = kind.sharesKVWithLayer {
                guard source >= 0, source < layerKinds.count, source != index else {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index): invalid KV-share source \(source)")
                }
                guard layerKinds[source].sharesKVWithLayer == nil else {
                    throw CBv2KVError.backendIneligible(
                        reason:
                            "layer \(index): KV-share source \(source) is itself a shared layer")
                }
            }
            // The v1 contiguous backend attends through MLXFast SDPA, which
            // supports attention sinks natively — sink models are eligible.
        }
    }

    /// Per-layer estimated initial allocation bytes, aligned to `layerKinds`
    /// (nil for KV-shared layers, which own no storage). Full layers allocate
    /// `promptLength + 256` slots capped at maxLength; windowed layers
    /// allocate their full ring up front. The sum is the reservation charged
    /// against capacity at admission.
    private func rowEstimates(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int
    ) -> [Int?] {
        let itemSize = config.kvDType.size
        return layerKinds.map { kind -> Int? in
            guard kind.sharesKVWithLayer == nil else { return nil }
            switch kind.attention {
            case .slidingWindow(let window):
                return window * kind.kvHeads * kind.headDim * itemSize * 2
            case .full:
                let slots = min(maxLength, max(1, promptLength + CBv2FullSequenceKV.initialSlack))
                if let quantization = config.quantization {
                    let groupSize = CBv2QuantizedSequenceKV.resolvedGroupSize(
                        requested: quantization.groupSize, headDim: kind.headDim)
                    // Packed weights + fp scales/biases per group.
                    let perToken =
                        kind.headDim * quantization.bits / 8
                        + 2 * (kind.headDim / groupSize) * itemSize
                    return slots * kind.kvHeads * perToken * 2
                }
                return slots * kind.kvHeads * kind.headDim * itemSize * 2
            }
        }
    }
}
