// CompiledLaneStateV2.swift
//
// Per-bucket lane state for compiled [B, 1] decode.
//
// MLX `compile(inputs:outputs:)` re-reads `Updatable.innerState()` at EVERY
// call: the current arrays are passed as hidden graph inputs and the graph's
// hidden outputs are written back into those same array objects via
// `_updateInternal`. That mechanism is what makes batch-membership churn
// cheap here — rebinding a lane to a different request's arrays swaps which
// buffers the next replay reads/writes, with no copies and no retrace (the
// shapes are pinned by the bucket).
//
// Two containers, split by write-back behavior:
//  - `CBv2CompiledLaneCounters` — per-lane absolute offsets. Registered as
//    compile INPUT + OUTPUT state: the traced graph advances each lane's
//    offset by 1 (`advanceAllInGraph`, called inside the trace closure after
//    the forward, so every in-forward read observes PRE-update offsets —
//    the RoPE-snapshot invariant). Host mirrors track the device values so
//    binds are rebuilt from host truth and out-of-band eager mutations are
//    detected (self-healing rebind, never a device readback).
//  - `CBv2CompiledLaneStatics` — per-lane, per-window oldest-valid-entry
//    bounds. INPUT-only state: static for the duration of a lane binding
//    (fastForward'ed prefix adoptions and post-wrap rollbacks are encoded
//    here at bind time), so the graph never writes them back.

import Foundation
import MLX

/// Per-lane absolute-offset counters for one compiled bucket.
/// Engine-thread confined.
final class CBv2CompiledLaneCounters: Updatable {

    let lanes: Int

    /// Device-side `[1] int32` per lane. Fresh objects on bind; advanced
    /// in-graph between binds. `innerState()` order: lane 0, 1, ….
    private(set) var offsets: [MLXArray]

    /// Host mirror of each lane's device offset. Updated on bind and after
    /// every compiled step (`noteAdvance`) — comparing against a row's
    /// `absoluteOffset` detects any out-of-band eager mutation.
    private(set) var hostOffsets: [Int]

    init(lanes: Int) {
        precondition(lanes > 0)
        self.lanes = lanes
        self.offsets = (0 ..< lanes).map { _ in MLXArray([Int32(0)]) }
        self.hostOffsets = Array(repeating: 0, count: lanes)
    }

    func innerState() -> [MLXArray] { offsets }

    /// Pre-update per-lane offsets `[lanes] int32` — the batch's absolute
    /// RoPE positions for this step. Traced (reads the tracer state inside
    /// the compile closure).
    var positionOffsets: MLXArray {
        offsets.count == 1 ? offsets[0] : concatenated(offsets, axis: 0)
    }

    /// The single point where lane offsets advance, called INSIDE the trace
    /// closure AFTER the model forward: every read during the forward
    /// (RoPE, masks, write positions) observes pre-update values.
    /// `_updateInternal` preserves object identity so the compile wrapper
    /// collects the advanced values as hidden outputs.
    func advanceAllInGraph() {
        for offset in offsets {
            offset._updateInternal(offset + Int32(1))
        }
    }

    /// Rebind one lane's device offset from host truth (membership change /
    /// self-heal). Host→device upload of one scalar; never a sync.
    func bind(lane: Int, offset: Int) {
        offsets[lane] = MLXArray([Int32(offset)])
        hostOffsets[lane] = offset
    }

    /// Mirror one compiled step's in-graph advance on the host.
    func noteAdvance() {
        for lane in 0 ..< lanes { hostOffsets[lane] += 1 }
    }
}

/// Per-lane, per-window static mask bounds for one compiled bucket.
/// INPUT-only compile state (never written by the graph).
final class CBv2CompiledLaneStatics: Updatable {

    let lanes: Int
    /// Distinct sliding windows of the model, ascending. Empty for
    /// all-full-attention models (then this container carries no state).
    let windows: [Int]

    /// `entries[lane][windowIndex]` — `[1] int32` oldest-valid absolute
    /// position for that lane's windowed layers with that window size.
    private(set) var entries: [[MLXArray]]

    init(lanes: Int, windows: [Int]) {
        precondition(lanes > 0)
        self.lanes = lanes
        self.windows = windows.sorted()
        self.entries = (0 ..< lanes).map { _ in
            windows.map { _ in MLXArray([Int32(0)]) }
        }
    }

    func innerState() -> [MLXArray] { entries.flatMap { $0 } }

    func entry(lane: Int, window: Int) -> MLXArray {
        guard let index = windows.firstIndex(of: window) else {
            preconditionFailure("CBv2CompiledLaneStatics: unknown window \(window)")
        }
        return entries[lane][index]
    }

    /// Rebind one lane's oldest-valid bounds from host truth, in `windows`
    /// order (membership change only).
    func bind(lane: Int, oldestByWindow: [Int]) {
        precondition(oldestByWindow.count == windows.count)
        entries[lane] = oldestByWindow.map { MLXArray([Int32($0)]) }
    }
}
