// Copyright © 2026 Apple Inc.
//
// Parity tests for the BatchRotatingKVCache in-place decode ring
// (`DARKBLOOM_FAST_BATCH_ROTATING_KV`, default ON).
//
// The fast path replaces the legacy per-step `concatenated` + front-`trim`
// (two full-window copies every decode step) with a single in-place slot write
// into an over-allocated buffer plus a host-side base pointer, compacting only
// every `allocationStep` steps. It MUST be byte-for-byte equivalent to the
// legacy path it falls back to. These tests run with no model — pure MLXArray
// ops — so they execute in CI and pin parity directly:
//
//   * returned (keys, values) window contents (exact, no fp tolerance),
//   * `state` = [keys, values, batchOffset, leftPadding] at every step,
//   * `size()`, `offset`, `isTrimmable`, and `makeMask` mode,
//   * `extract`, `filter`, `extend`, `copy`, and `trim` after the run,
//
// across B=1/2/3, mixed left-padding, prompts shorter and longer than the
// window, window crossings, and a run long enough to force a ring compaction.
// `useFastDecodePath` is flipped per instance so the comparison never depends
// on the process-wide env default.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

@Suite("BatchRotatingKVCache fast-decode parity (fast vs legacy)", .serialized)
struct BatchRotatingKVCacheFastPathTests {

    // MARK: - comparison helpers

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        (a.asType(.float32) - b.asType(.float32)).abs().max().item(Float.self)
    }

    private func expectArrayEqual(_ a: MLXArray, _ b: MLXArray, _ label: String) {
        #expect(a.shape == b.shape, Comment(rawValue: "\(label): shape \(a.shape) != \(b.shape)"))
        guard a.shape == b.shape else { return }
        let d = maxAbsDiff(a, b)
        #expect(d == 0, Comment(rawValue: "\(label): values differ (maxAbsDiff=\(d))"))
    }

    private func expectStateEqual(
        _ fast: BatchRotatingKVCache, _ legacy: BatchRotatingKVCache, _ label: String
    ) {
        let fs = fast.state
        let ls = legacy.state
        #expect(fs.count == ls.count, Comment(rawValue: "\(label): state count \(fs.count) != \(ls.count)"))
        guard fs.count == ls.count else { return }
        let names = ["keys", "values", "batchOffset", "leftPadding"]
        for (i, (f, l)) in zip(fs, ls).enumerated() {
            expectArrayEqual(f, l, "\(label).state.\(i < names.count ? names[i] : "\(i)")")
        }
    }

    private func maskMatches(
        _ a: MLXFast.ScaledDotProductAttentionMaskMode,
        _ b: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> Bool {
        switch (a, b) {
        case (.none, .none): return true
        case (.causal, .causal): return true
        case (.array(let x), .array(let y)):
            return x.shape == y.shape
                && (x.asType(.int32) - y.asType(.int32)).abs().max().item(Int32.self) == 0
        default: return false
        }
    }

    // MARK: - input script (identical data for both caches)

    private struct Script {
        let maxSize: Int
        let leftPad: [Int]
        let prefillK: MLXArray?
        let prefillV: MLXArray?
        let dk: [MLXArray]
        let dv: [MLXArray]
    }

    private func makeScript(
        maxSize: Int, leftPad: [Int], prefillLen: Int, decodeSteps: Int,
        H: Int = 2, Dk: Int = 3, Dv: Int = 4, seed: UInt64 = 0x9E37_79B9
    ) -> Script {
        let B = leftPad.count
        MLXRandom.seed(seed)
        var prefillK: MLXArray?
        var prefillV: MLXArray?
        if prefillLen > 0 {
            let pk = MLXRandom.normal([B, H, prefillLen, Dk])
            let pv = MLXRandom.normal([B, H, prefillLen, Dv])
            eval(pk, pv)
            prefillK = pk
            prefillV = pv
        }
        var dk: [MLXArray] = []
        var dv: [MLXArray] = []
        dk.reserveCapacity(decodeSteps)
        dv.reserveCapacity(decodeSteps)
        for _ in 0 ..< decodeSteps {
            let k = MLXRandom.normal([B, H, 1, Dk])
            let v = MLXRandom.normal([B, H, 1, Dv])
            eval(k, v)
            dk.append(k)
            dv.append(v)
        }
        return Script(
            maxSize: maxSize, leftPad: leftPad, prefillK: prefillK, prefillV: prefillV,
            dk: dk, dv: dv)
    }

    private func build(_ s: Script, fast: Bool) -> BatchRotatingKVCache {
        let c = BatchRotatingKVCache(maxSize: s.maxSize, leftPadding: s.leftPad)
        c.useFastDecodePath = fast
        if let pk = s.prefillK, let pv = s.prefillV {
            _ = c.update(keys: pk, values: pv)
        }
        for i in 0 ..< s.dk.count {
            _ = c.update(keys: s.dk[i], values: s.dv[i])
        }
        eval(c.state)
        return c
    }

    /// Drive the fast and legacy caches through the same script and compare the
    /// full observable surface after every update.
    private func runParity(_ label: String, _ s: Script) {
        let fast = BatchRotatingKVCache(maxSize: s.maxSize, leftPadding: s.leftPad)
        fast.useFastDecodePath = true
        let legacy = BatchRotatingKVCache(maxSize: s.maxSize, leftPadding: s.leftPad)
        legacy.useFastDecodePath = false

        if let pk = s.prefillK, let pv = s.prefillV {
            let (fk, fv) = fast.update(keys: pk, values: pv)
            let (lk, lv) = legacy.update(keys: pk, values: pv)
            eval(fk, fv, lk, lv)
            expectArrayEqual(fk, lk, "\(label) prefill K")
            expectArrayEqual(fv, lv, "\(label) prefill V")
            expectStateEqual(fast, legacy, "\(label) prefill")
        }

        for step in 0 ..< s.dk.count {
            let (fk, fv) = fast.update(keys: s.dk[step], values: s.dv[step])
            let (lk, lv) = legacy.update(keys: s.dk[step], values: s.dv[step])
            eval(fk, fv, lk, lv)
            expectArrayEqual(fk, lk, "\(label) step \(step) K")
            expectArrayEqual(fv, lv, "\(label) step \(step) V")
            expectStateEqual(fast, legacy, "\(label) step \(step)")
            #expect(
                fast.size() == legacy.size(),
                Comment(rawValue: "\(label) step \(step): size \(fast.size()) != \(legacy.size())"))
            #expect(
                fast.offset == legacy.offset,
                Comment(rawValue: "\(label) step \(step): offset mismatch"))
            #expect(
                fast.isTrimmable == legacy.isTrimmable,
                Comment(rawValue: "\(label) step \(step): isTrimmable mismatch"))
            // Gemma sliding layers call makeMask with windowSize == maxSize.
            let fm = fast.makeMask(n: 1, windowSize: s.maxSize, returnArray: false)
            let lm = legacy.makeMask(n: 1, windowSize: s.maxSize, returnArray: false)
            #expect(maskMatches(fm, lm), Comment(rawValue: "\(label) step \(step): mask mode mismatch"))
        }
    }

    // MARK: - decode parity

    @Test("B=1, prompt < window: fast engages immediately, crosses window")
    func b1ShortPrompt() {
        runParity("B1 short", makeScript(maxSize: 16, leftPad: [0], prefillLen: 8, decodeSteps: 40))
    }

    @Test("B=1, prompt > window: first decode legacy-trims, then fast")
    func b1LongPrompt() {
        runParity("B1 long", makeScript(maxSize: 16, leftPad: [0], prefillLen: 23, decodeSteps: 40))
    }

    @Test("B=2 mixed left-padding crosses the window")
    func b2Mixed() {
        runParity("B2 mixed", makeScript(maxSize: 12, leftPad: [0, 5], prefillLen: 10, decodeSteps: 50))
    }

    @Test("B=3 mixed left-padding crosses the window")
    func b3Mixed() {
        runParity("B3 mixed", makeScript(maxSize: 10, leftPad: [0, 3, 7], prefillLen: 9, decodeSteps: 60))
    }

    @Test("decode from empty (no prefill)")
    func b2FromEmpty() {
        runParity("B2 empty", makeScript(maxSize: 8, leftPad: [0, 0], prefillLen: 0, decodeSteps: 30))
    }

    @Test("long run forces a ring compaction (decodeSteps > window + allocationStep)")
    func compactionParity() {
        let steps = BatchRotatingKVCache.allocationStep + 40  // ≥ one compaction
        runParity(
            "compaction", makeScript(maxSize: 4, leftPad: [0, 1], prefillLen: 3, decodeSteps: steps))
    }

    // MARK: - "fast path actually engaged" guard
    //
    // Parity is only meaningful if the fast path ran. After at least one fast
    // decode the physical buffer is over-allocated to `maxSize + allocationStep`;
    // the legacy buffer never exceeds `maxSize`.

    @Test("fast path over-allocates the physical buffer; legacy does not")
    func fastPathEngages() {
        let s = makeScript(maxSize: 16, leftPad: [0], prefillLen: 8, decodeSteps: 20)
        let fast = build(s, fast: true)
        let legacy = build(s, fast: false)
        #expect(
            fast.keys?.dim(2) == s.maxSize + BatchRotatingKVCache.allocationStep,
            Comment(rawValue: "fast physical buffer = \(String(describing: fast.keys?.dim(2)))"))
        #expect(
            (legacy.keys?.dim(2) ?? 0) <= s.maxSize,
            Comment(rawValue: "legacy physical buffer = \(String(describing: legacy.keys?.dim(2)))"))
        // ...and the logical view still agrees.
        expectStateEqual(fast, legacy, "engage")
    }

    // MARK: - op parity after a fast/legacy run

    @Test("extract(row) parity for every row (B=3 mixed)")
    func extractParity() {
        let s = makeScript(maxSize: 10, leftPad: [0, 3, 6], prefillLen: 9, decodeSteps: 25)
        let fast = build(s, fast: true)
        let legacy = build(s, fast: false)
        for row in 0 ..< s.leftPad.count {
            let fe = fast.extract(row)
            let le = legacy.extract(row)
            let fState = fe.state
            let lState = le.state
            #expect(fState.count == lState.count, Comment(rawValue: "extract[\(row)] state count"))
            for (i, (f, l)) in zip(fState, lState).enumerated() {
                expectArrayEqual(f, l, "extract[\(row)].state[\(i)]")
            }
            #expect(fe.metaState == le.metaState, Comment(rawValue: "extract[\(row)] metaState"))
        }
    }

    @Test("filter(rows) parity (B=3 → keep rows 0,2)")
    func filterParity() {
        let s = makeScript(maxSize: 10, leftPad: [0, 3, 6], prefillLen: 9, decodeSteps: 25)
        let fast = build(s, fast: true)
        let legacy = build(s, fast: false)
        let keep = MLXArray([Int32(0), Int32(2)])
        fast.filter(batchIndices: keep)
        legacy.filter(batchIndices: keep)
        expectStateEqual(fast, legacy, "after filter")
        // continue decoding the filtered (B=2) caches and stay in parity
        let cont = makeScript(maxSize: 10, leftPad: [0, 0], prefillLen: 0, decodeSteps: 15, seed: 0x1234)
        for i in 0 ..< cont.dk.count {
            let (fk, fv) = fast.update(keys: cont.dk[i], values: cont.dv[i])
            let (lk, lv) = legacy.update(keys: cont.dk[i], values: cont.dv[i])
            eval(fk, fv, lk, lv)
            expectArrayEqual(fk, lk, "post-filter step \(i) K")
            expectStateEqual(fast, legacy, "post-filter step \(i)")
        }
    }

    @Test("extend(other) parity")
    func extendParity() {
        let sA = makeScript(maxSize: 12, leftPad: [0, 2], prefillLen: 8, decodeSteps: 20, seed: 0xAAAA)
        let sB = makeScript(maxSize: 12, leftPad: [1], prefillLen: 6, decodeSteps: 14, seed: 0xBBBB)
        let fastA = build(sA, fast: true)
        let fastB = build(sB, fast: true)
        let legacyA = build(sA, fast: false)
        let legacyB = build(sB, fast: false)
        fastA.extend(fastB)
        legacyA.extend(legacyB)
        expectStateEqual(fastA, legacyA, "after extend")
    }

    @Test("copy() parity (deep, independent of source mutation)")
    func copyParity() {
        let s = makeScript(maxSize: 8, leftPad: [0, 2], prefillLen: 6, decodeSteps: 18)
        let fast = build(s, fast: true)
        let legacy = build(s, fast: false)
        guard let fastCopy = fast.copy() as? BatchRotatingKVCache,
            let legacyCopy = legacy.copy() as? BatchRotatingKVCache
        else {
            Issue.record("copy() did not return BatchRotatingKVCache")
            return
        }
        expectStateEqual(fastCopy, legacyCopy, "copy")
        // mutating the source must not disturb the copy
        let extra = makeScript(maxSize: 8, leftPad: [0, 0], prefillLen: 0, decodeSteps: 5, seed: 0xCAFE)
        for i in 0 ..< extra.dk.count {
            _ = fast.update(keys: extra.dk[i], values: extra.dv[i])
        }
        expectStateEqual(fastCopy, legacyCopy, "copy after source mutation")
    }

    @Test("trim(n) parity then continued decode")
    func trimParity() {
        let s = makeScript(maxSize: 10, leftPad: [0, 0], prefillLen: 7, decodeSteps: 12)
        let fast = build(s, fast: true)
        let legacy = build(s, fast: false)
        _ = fast.trim(3)
        _ = legacy.trim(3)
        expectStateEqual(fast, legacy, "after trim")
        let cont = makeScript(maxSize: 10, leftPad: [0, 0], prefillLen: 0, decodeSteps: 20, seed: 0xFEED)
        for i in 0 ..< cont.dk.count {
            let (fk, fv) = fast.update(keys: cont.dk[i], values: cont.dv[i])
            let (lk, lv) = legacy.update(keys: cont.dk[i], values: cont.dv[i])
            eval(fk, fv, lk, lv)
            expectArrayEqual(fk, lk, "post-trim step \(i) K")
            expectStateEqual(fast, legacy, "post-trim step \(i)")
        }
    }
}
