// CBv2PagedBackendTests.swift
//
// WS-C unit tests: paged pool bookkeeping, per-sequence page tables,
// windowed rings, reservation-based admission, and backend lifecycle.
// No model weights required.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

@Suite("CBv2PagedBackend")
struct CBv2PagedBackendTests {

    // MARK: - Helpers

    private func fullKind(
        headDim: Int = 64, kvHeads: Int = 2, queryHeads: Int = 4
    ) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .full, headDim: headDim, kvHeads: kvHeads, queryHeads: queryHeads)
    }

    private func windowedKind(
        _ window: Int, headDim: Int = 64, kvHeads: Int = 2, queryHeads: Int = 4
    ) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .slidingWindow(window), headDim: headDim, kvHeads: kvHeads,
            queryHeads: queryHeads)
    }

    private func config(
        capacityBytes: Int = 8 << 20, maxPrefillChunk: Int = 64,
        nominalMaxLen: Int = 1024
    ) -> PagedKVPoolConfig {
        PagedKVPoolConfig(
            capacityBytes: capacityBytes, maxPrefillChunk: maxPrefillChunk,
            nominalMaxSequenceLength: nominalMaxLen)
    }

    private func randomKV(heads: Int, tokens: Int, dim: Int) -> MLXArray {
        MLXRandom.normal([heads, tokens, dim], dtype: .float16)
    }

    private func assertEqualArrays(
        _ a: MLXArray, _ b: MLXArray,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(a.shape == b.shape, sourceLocation: sourceLocation)
        #expect(
            arrayEqual(a, b).item(Bool.self),
            "arrays differ",
            sourceLocation: sourceLocation)
    }

    // MARK: - Pool

    @Test func poolGroupsAndAccounting() throws {
        let kinds = [fullKind(), fullKind(), windowedKind(32)]
        let pool = try PagedKVPool(layerKinds: kinds, config: config())
        // One group: all layers share (kvHeads=2, headDim=64).
        #expect(pool.groupKeys == [PagedKVGroupKey(kvHeads: 2, headDim: 64)])
        #expect(pool.bytesInUse == 0)
        #expect(pool.bytesReserved == 0)
        #expect(pool.bytesCapacity > 0)
        #expect(pool.bytesCapacity <= 8 << 20)
    }

    @Test func poolRejectsQuantSchemes() {
        #expect(throws: CBv2KVError.self) {
            _ = try PagedKVPool(
                layerKinds: [fullKind()],
                config: PagedKVPoolConfig(
                    capacityBytes: 1 << 20,
                    quantScheme: .affine(groupSize: 64, bits: 4)))
        }
    }

    @Test func reservationExhaustionThrowsAtomically() throws {
        let kinds = [fullKind()]
        // Tiny pool: room for a handful of pages only.
        let pageBytes = 2 * 2 * 16 * 64 * 2
        let pool = try PagedKVPool(
            layerKinds: kinds,
            config: config(capacityBytes: pageBytes * 8, nominalMaxLen: 128))
        let key = PagedKVGroupKey(kvHeads: 2, headDim: 64)
        try pool.reserve([key: 6])
        #expect(throws: CBv2KVError.self) {
            try pool.reserve([key: 3])
        }
        // Failed reserve must leave no partial state.
        try pool.reserve([key: 2])
        pool.unreserve([key: 8])
        #expect(pool.bytesReserved == 0)
    }

    // MARK: - Sequence state: full attention

    @Test func fullSequenceRoundtrip() throws {
        let kinds = [fullKind()]
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 256)
        let row = try #require(state[0] as? PagedSequenceKV)

        var mirrorK: [MLXArray] = []
        var mirrorV: [MLXArray] = []
        for n in [5, 16, 12, 1] {
            let k = randomKV(heads: 2, tokens: n, dim: 64)
            let v = randomKV(heads: 2, tokens: n, dim: 64)
            mirrorK.append(k)
            mirrorV.append(v)
            let (viewK, viewV) = row.update(
                keys: k.expandedDimensions(axis: 0), values: v.expandedDimensions(axis: 0))
            #expect(viewK.shape == [1, 2, row.retainedCount, 64])
            #expect(viewV.shape == viewK.shape)
        }
        #expect(row.absoluteOffset == 34)
        #expect(row.retainedCount == 34)
        // 34 tokens => 3 pages of 16.
        #expect(row.table.count == 3)
        #expect(backend.bytesInUse == 3 * backend.pool.group(row.groupKey).pageBytes)

        let expectedK = concatenated(mirrorK, axis: 1).expandedDimensions(axis: 0)
        let expectedV = concatenated(mirrorV, axis: 1).expandedDimensions(axis: 0)
        let snap = row.snapshot()
        #expect(snap.offset == 34)
        assertEqualArrays(snap.keys, expectedK)
        assertEqualArrays(snap.values, expectedV)

        backend.release(state)
        #expect(backend.bytesInUse == 0)
        #expect(backend.pool.bytesReserved == 0)
    }

    @Test func rollbackFreesTailPagesAndRewrites() throws {
        let kinds = [fullKind()]
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 128)
        let row = try #require(state[0] as? PagedSequenceKV)

        let k0 = randomKV(heads: 2, tokens: 20, dim: 64)
        let v0 = randomKV(heads: 2, tokens: 20, dim: 64)
        row.write(keys: k0, values: v0)
        #expect(row.table.count == 2)

        row.rollback(6)
        #expect(row.absoluteOffset == 14)
        #expect(row.table.count == 1)

        // Rewrite different tail tokens; gather must reflect the new tail.
        let k1 = randomKV(heads: 2, tokens: 4, dim: 64)
        let v1 = randomKV(heads: 2, tokens: 4, dim: 64)
        row.write(keys: k1, values: v1)
        #expect(row.absoluteOffset == 18)

        let expectedK = concatenated(
            [k0[0..., 0 ..< 14, 0...], k1], axis: 1
        ).expandedDimensions(axis: 0)
        let (gotK, _) = row.attendableViews()
        assertEqualArrays(gotK, expectedK)
        backend.release(state)
    }

    // MARK: - Sequence state: sliding window ring

    @Test func windowedRingWrapKeepsRecentEnd() throws {
        let window = 32
        let kinds = [windowedKind(window)]
        let cfg = config(maxPrefillChunk: 16)
        let backend = try PagedKVBackend(layerKinds: kinds, config: cfg)
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 512)
        let row = try #require(state[0] as? PagedSequenceKV)
        let ring = PagedKVPool.ringPageCount(window: window, config: cfg)
        // ceil((32 + 16) / 16) + 1 = 4
        #expect(ring == 4)

        var mirrorK: [MLXArray] = []
        var mirrorV: [MLXArray] = []
        var written = 0
        for n in [16, 7, 16, 16, 3, 16, 16, 10] {
            let k = randomKV(heads: 2, tokens: n, dim: 64)
            let v = randomKV(heads: 2, tokens: n, dim: 64)
            mirrorK.append(k)
            mirrorV.append(v)
            row.write(keys: k, values: v)
            written += n
            #expect(row.absoluteOffset == written)
            #expect(row.table.count <= ring)

            let retained = row.retainedCount
            #expect(retained == min(written, window - 1 + n))
            let allK = concatenated(mirrorK, axis: 1)
            let allV = concatenated(mirrorV, axis: 1)
            let (gotK, gotV) = row.attendableViews()
            assertEqualArrays(
                gotK, allK[0..., (written - retained) ..< written, 0...]
                    .expandedDimensions(axis: 0))
            assertEqualArrays(
                gotV, allV[0..., (written - retained) ..< written, 0...]
                    .expandedDimensions(axis: 0))
        }
        // Ring must have wrapped (100 tokens through a 4-page/64-token ring).
        #expect(row.table.count == ring)
        backend.release(state)
        #expect(backend.bytesInUse == 0)
    }

    @Test func decodeAttendRangeClampsToWindow() throws {
        let kinds = [windowedKind(32)]
        let backend = try PagedKVBackend(
            layerKinds: kinds, config: config(maxPrefillChunk: 16))
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 512)
        let row = try #require(state[0] as? PagedSequenceKV)

        row.write(keys: randomKV(heads: 2, tokens: 10, dim: 64),
                  values: randomKV(heads: 2, tokens: 10, dim: 64))
        var range = row.decodeAttendRange
        #expect(range.start == 0 && range.length == 10)

        for _ in 0 ..< 6 {
            row.write(keys: randomKV(heads: 2, tokens: 16, dim: 64),
                      values: randomKV(heads: 2, tokens: 16, dim: 64))
        }
        // 106 tokens written; query position 105; window 32 => start 74.
        range = row.decodeAttendRange
        #expect(range.start == 74 && range.length == 32)
        backend.release(state)
    }

    // MARK: - Backend lifecycle

    @Test func sharedLayersGetNilStateAndValidation() throws {
        var shared = fullKind()
        shared.sharesKVWithLayer = 0
        let kinds = [fullKind(), shared]
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 64)
        #expect(state.count == 2)
        #expect(state[0] != nil)
        #expect(state[1] == nil)
        backend.release(state)

        // Invalid head dim -> ineligible at build.
        #expect(throws: CBv2KVError.self) {
            _ = try PagedKVBackend(
                layerKinds: [fullKind(headDim: 96)], config: config())
        }
        // Bad GQA ratio -> ineligible at build.
        #expect(throws: CBv2KVError.self) {
            _ = try PagedKVBackend(
                layerKinds: [fullKind(kvHeads: 3, queryHeads: 4)], config: config())
        }
        // Shared layer pointing at another shared layer -> ineligible.
        var badShared = fullKind()
        badShared.sharesKVWithLayer = 1
        #expect(throws: CBv2KVError.self) {
            _ = try PagedKVBackend(
                layerKinds: [fullKind(), shared, badShared].map { $0 },
                config: config())
        }
    }

    @Test func admissionThrowsCapacityExhaustedAndRecovers() throws {
        let kinds = [fullKind()]
        let pageBytes = 2 * 2 * 16 * 64 * 2
        // Room for ~8 pages == 128 tokens.
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: config(capacityBytes: pageBytes * 8, nominalMaxLen: 128))

        let a = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 96)  // 6 pages
        #expect(throws: CBv2KVError.self) {
            _ = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: 0, maxLength: 96)
        }
        backend.release(a)
        let b = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 96)
        backend.release(b)
    }

    @Test func adoptPrefixRoundtrip() throws {
        let kinds = [fullKind()]
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())

        // Donor sequence.
        let donor = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 128)
        let donorRow = try #require(donor[0] as? PagedSequenceKV)
        let k = randomKV(heads: 2, tokens: 40, dim: 64)
        let v = randomKV(heads: 2, tokens: 40, dim: 64)
        donorRow.write(keys: k, values: v)
        let snap = donorRow.snapshot()
        // Materialize the snapshot before the donor pages are recycled —
        // gathered views reference the live slabs.
        eval(snap.keys, snap.values)
        backend.release(donor)

        let adopted = try backend.makeSequenceState(
            adopting: [(keys: snap.keys, values: snap.values, offset: snap.offset)],
            layerKinds: kinds, maxLength: 128)
        let row = try #require(adopted[0] as? PagedSequenceKV)
        #expect(row.absoluteOffset == 40)
        let (gotK, gotV) = row.attendableViews()
        assertEqualArrays(gotK, k.expandedDimensions(axis: 0))
        assertEqualArrays(gotV, v.expandedDimensions(axis: 0))
        backend.release(adopted)
    }

    @Test func fastForwardPlacesWindowedRecompute() throws {
        let kinds = [windowedKind(32)]
        let backend = try PagedKVBackend(
            layerKinds: kinds, config: config(maxPrefillChunk: 16))
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 512)
        let row = try #require(state[0] as? PagedSequenceKV)

        row.fastForward(to: 100)
        #expect(row.absoluteOffset == 100)
        #expect(row.retainedCount == 0)

        row.write(keys: randomKV(heads: 2, tokens: 16, dim: 64),
                  values: randomKV(heads: 2, tokens: 16, dim: 64))
        #expect(row.absoluteOffset == 116)
        // Only replayed tokens are attendable.
        #expect(row.retainedCount == 16)
        let range = row.decodeAttendRange
        #expect(range.start == 100 && range.length == 16)
        backend.release(state)
    }

    @Test func updateHonorsMaxLengthReservation() throws {
        let kinds = [fullKind()]
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 32)
        let row = try #require(state[0] as? PagedSequenceKV)
        row.write(keys: randomKV(heads: 2, tokens: 32, dim: 64),
                  values: randomKV(heads: 2, tokens: 32, dim: 64))
        #expect(row.table.count == 2)
        backend.release(state)
    }

    // MARK: - Sink gating

    /// A layer whose kind declares `hasSinks == false` must IGNORE a
    /// (mistakenly) passed sinks array — the same gate CBv2AttentionV1
    /// applies — on both the prefill and decode dispatch paths.
    /// (Deterministic position-coded inputs, NOT the global MLXRandom
    /// state: Swift Testing runs suites in parallel, and a concurrent test
    /// advancing the global RNG between runs would break the comparison.)
    @Test func sinksGatedByLayerKind() throws {
        let kinds = [fullKind()]
        #expect(!kinds[0].hasSinks)
        let garbageSinks = MLXArray(converting: [50.0, 50.0, 50.0, 50.0])

        func coded(_ shape: [Int], seed: Float) -> MLXArray {
            let count = shape.reduce(1, *)
            return MLXArray(0 ..< Int32(count)).asType(.float16)
                .reshaped(shape) * 0.001 + seed
        }

        func run(sinks: MLXArray?) throws -> (prefill: MLXArray, decode: MLXArray) {
            let backend = try PagedKVBackend(layerKinds: kinds, config: config())
            let cache = backend.makeLayerCaches()[0]
            let state = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: 8, maxLength: 32)
            defer { backend.release(state) }
            cache.setRows([state[0]!])
            let prefill = cache.updateAndAttend(
                queries: coded([1, 4, 8, 64], seed: 0.3),
                keys: coded([1, 2, 8, 64], seed: 0.1),
                values: coded([1, 2, 8, 64], seed: 0.2),
                scale: 0.125, sinks: sinks)
            let decode = cache.updateAndAttend(
                queries: coded([1, 4, 1, 64], seed: 0.6),
                keys: coded([1, 2, 1, 64], seed: 0.4),
                values: coded([1, 2, 1, 64], seed: 0.5),
                scale: 0.125, sinks: sinks)
            eval(prefill, decode)
            return (prefill, decode)
        }

        let with = try run(sinks: garbageSinks)
        let without = try run(sinks: nil)
        assertEqualArrays(with.prefill, without.prefill)
        assertEqualArrays(with.decode, without.decode)
    }

    // MARK: - Device block-table cache identity

    /// Regression (cross-request page-table reuse): the device-table cache
    /// used to fingerprint rows by (ObjectIdentifier, tableVersion). A heap
    /// address can be recycled after a finished row deallocates, and — via
    /// the LIFO free list — the replacement row gets the SAME page ids and
    /// the SAME tableVersion trajectory, so a stale fingerprint hit would
    /// make the kernel attend the finished request's cached tables. The
    /// fingerprint is now a pool-issued monotonic serial: never reused, so
    /// two identical-geometry rows can NEVER collide, and any row-identity
    /// change forces a rebuild.
    @Test func deviceTablesRebuildOnRowIdentityReuse() throws {
        let kinds = [fullKind()]
        let backend = try PagedKVBackend(layerKinds: kinds, config: config())
        let cache = backend.makeLayerCaches()[0]

        func makeWrittenRow() throws -> [CBv2SequenceKV?] {
            let state = try backend.makeSequenceState(
                layerKinds: kinds, promptLength: 8, maxLength: 32)
            let row = try #require(state[0] as? PagedSequenceKV)
            row.write(
                keys: randomKV(heads: 2, tokens: 8, dim: 64),
                values: randomKV(heads: 2, tokens: 8, dim: 64))
            return state
        }

        let stateA = try makeWrittenRow()
        let rowA = try #require(stateA[0] as? PagedSequenceKV)
        cache.setRows([rowA])
        _ = cache.deviceTables(rows: [rowA])
        #expect(cache.tablesRebuildCount == 1)
        _ = cache.deviceTables(rows: [rowA])
        #expect(cache.tablesRebuildCount == 1, "unchanged table must hit the cache")
        let tableA = rowA.table
        let versionA = rowA.tableVersion

        // Release A, then allocate B with identical geometry: the LIFO free
        // list hands B the same physical page ids and B's tableVersion
        // trajectory matches A's — only the pool serial tells them apart.
        backend.release(stateA)
        let stateB = try makeWrittenRow()
        let rowB = try #require(stateB[0] as? PagedSequenceKV)
        #expect(rowB.table == tableA, "LIFO free list reuses the same page ids")
        #expect(rowB.tableVersion == versionA, "same allocation trajectory")
        #expect(rowB.serial != rowA.serial, "pool serials are never reused")

        cache.setRows([rowB])
        _ = cache.deviceTables(rows: [rowB])
        #expect(
            cache.tablesRebuildCount == 2,
            "a row-identity change must rebuild the device tables even when page ids and tableVersion match"
        )
        backend.release(stateB)
    }

    // MARK: - Admission-time page reservation (Codex P2)

    /// Reserving worst-case pages UP FRONT means several same-step admissions
    /// cannot over-commit the pool: the excess reservation throws before any
    /// request is accepted, and `bytesReserved` never exceeds capacity.
    @Test func admissionReservationCannotOvercommitPool() throws {
        let kinds = [fullKind()]  // one group, kv2 x d64
        // Pool sized for exactly 2 requests of maxLength 128 (8 pages each).
        let perRequestPages = PagedKVPool.pageDemand(
            kind: kinds[0], maxLength: 128,
            config: config(nominalMaxLen: 128))
        let pageBytes = 2 * 2 * 16 * 64 * 2
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: config(capacityBytes: pageBytes * (perRequestPages * 2), nominalMaxLen: 128))
        #expect(backend.bytesReserved == 0)

        try backend.reserve(layerKinds: kinds, maxLength: 128)
        try backend.reserve(layerKinds: kinds, maxLength: 128)
        #expect(backend.bytesReserved <= backend.bytesCapacity)
        let reservedAtCap = backend.bytesReserved

        // The third same-step admission cannot fit — it must throw, not
        // over-commit the pool.
        #expect(throws: CBv2KVError.self) {
            try backend.reserve(layerKinds: kinds, maxLength: 128)
        }
        #expect(backend.bytesReserved == reservedAtCap, "failed reserve leaves no partial state")

        // A pre-reserved request materializes WITHOUT double-charging, and
        // release returns exactly what admission held.
        let state = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: 0, maxLength: 128, reserved: true)
        #expect(backend.bytesReserved == reservedAtCap, "materialization must not re-charge")
        backend.release(state)
        backend.unreserve(layerKinds: kinds, maxLength: 128)  // release the 2nd hold
        #expect(backend.bytesReserved == 0, "every hold balanced")
    }

    // MARK: - position offsets: device-cached, membership-gated (Codex P2)

    private func decodeQKV(queryHeads: Int, kvHeads: Int, dim: Int, tokens: Int)
        -> (q: MLXArray, k: MLXArray, v: MLXArray)
    {
        (
            MLXRandom.normal([1, queryHeads, tokens, dim], dtype: .float16),
            MLXRandom.normal([1, kvHeads, tokens, dim], dtype: .float16),
            MLXRandom.normal([1, kvHeads, tokens, dim], dtype: .float16)
        )
    }

    @Test func positionOffsetsAdvanceOnDeviceWithoutHostRebuild() throws {
        let kind = fullKind()
        let backend = try PagedKVBackend(layerKinds: [kind], config: config())
        let cache = backend.makeLayerCaches()[0]
        let scale = 1.0 / Float(kind.headDim).squareRoot()

        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 512)
        defer { backend.release(state) }
        cache.setRows([state[0]!])
        #expect(cache.positionOffsetsHostRebuilds == 1, "setRows rebuilds once")

        // Prefill chunk of 4 tokens: offset 0 -> 4 on device, no host rebuild.
        let pf = decodeQKV(queryHeads: kind.queryHeads, kvHeads: kind.kvHeads, dim: kind.headDim, tokens: 4)
        _ = cache.updateAndAttend(queries: pf.q, keys: pf.k, values: pf.v, scale: scale, sinks: nil)
        #expect(cache.positionOffsetsHostRebuilds == 1)
        #expect(cache.positionOffsets.item(Int32.self) == 4)

        // 20 decode steps: offsets advance on device, counter stays 1.
        for _ in 0 ..< 20 {
            let d = decodeQKV(
                queryHeads: kind.queryHeads, kvHeads: kind.kvHeads, dim: kind.headDim, tokens: 1)
            _ = cache.updateAndAttend(queries: d.q, keys: d.k, values: d.v, scale: scale, sinks: nil)
        }
        #expect(cache.positionOffsetsHostRebuilds == 1, "decode loop must not host-rebuild offsets")
        #expect(cache.positionOffsets.item(Int32.self) == 24, "4 prefill + 20 decode tokens")

        // A batch membership change rebuilds from host truth.
        let state2 = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 512)
        defer { backend.release(state2) }
        cache.setRows([state[0]!, state2[0]!])
        #expect(cache.positionOffsetsHostRebuilds == 2, "membership change rebuilds")
    }

    // MARK: - windowed adoption decode-table length (Codex P2)

    private func assertClose(
        _ got: MLXArray, _ want: MLXArray, rtol: Float = 1e-2, atol: Float = 2e-3,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(got.shape == want.shape, sourceLocation: sourceLocation)
        let ok = allClose(
            got.asType(.float32), want.asType(.float32), rtol: Double(rtol), atol: Double(atol)
        ).item(Bool.self)
        if !ok {
            let diff = abs(got.asType(.float32) - want.asType(.float32)).max().item(Float.self)
            Issue.record("arrays differ, max abs err \(diff)", sourceLocation: sourceLocation)
        }
    }

    /// A prefix-adopted windowed row whose trailing replay covers LESS than
    /// the ring leaves `table.count < ringPages`. The decode kernel must
    /// divide logical pages by the RING length, not `table.count`, or it
    /// wraps at the wrong divisor and reads the wrong physical pages — decode
    /// then diverges from a fresh-prefill reference. Token-exact regression.
    @Test func windowedAdoptionUsesRingLengthForDecodeTable() throws {
        let window = 32
        let kind = windowedKind(window, headDim: 64, kvHeads: 2, queryHeads: 4)
        let dim = kind.headDim
        let heads = kind.kvHeads
        let scale = 1.0 / Float(dim).squareRoot()
        // pageSize 16, maxPrefillChunk 16 ⇒ ring = (32+16)/16 + 1 = 4 pages.
        let backend = try PagedKVBackend(
            layerKinds: [kind],
            config: config(capacityBytes: 32 << 20, maxPrefillChunk: 16, nominalMaxLen: 4096))
        let ring = PagedKVPool.ringPageCount(window: window, config: backend.pool.config)
        #expect(ring == 4)

        // Deterministic position-coded K/V so a mis-resolved page is visible:
        // every absolute position gets a distinct vector.
        func coded(_ positions: Range<Int>) -> (MLXArray, MLXArray) {
            let n = positions.count
            var kflat = [Float](repeating: 0, count: heads * n * dim)
            var vflat = [Float](repeating: 0, count: heads * n * dim)
            var i = 0
            for h in 0 ..< heads {
                for p in positions {
                    for d in 0 ..< dim {
                        kflat[i] = Float(p % 97) * 0.01 + Float(d) * 0.001 + Float(h) * 0.1
                        vflat[i] = Float(p % 97) * 0.02 - Float(d) * 0.0005 + Float(h) * 0.05
                        i += 1
                    }
                }
            }
            return (
                MLXArray(kflat, [heads, n, dim]).asType(.float16),
                MLXArray(vflat, [heads, n, dim]).asType(.float16)
            )
        }

        // Adopt at position 64 (logical page 4), replay [64, 96): logical
        // pages 4,5 → ring slots 0,1 ⇒ table.count = 2 (slot 3 never touched).
        let cache = backend.makeLayerCaches()[0]
        let state = try backend.makeSequenceState(
            layerKinds: [kind], promptLength: 0, maxLength: 4096)
        defer { backend.release(state) }
        let row = state[0] as! PagedSequenceKV
        row.fastForward(to: 64)
        for chunkStart in stride(from: 64, to: 96, by: 16) {
            let (ck, cv) = coded(chunkStart ..< chunkStart + 16)
            row.write(keys: ck, values: cv)
        }
        #expect(row.table.count < ring, "replay must leave the ring partially allocated")
        #expect(row.decodeTableLength == ring, "windowed decode must divide by the ring length")
        cache.setRows([row])

        // Decode at position 96: window [65, 96].
        let q = MLXRandom.normal([1, kind.queryHeads, 1, dim], dtype: .float16)
        let (nk, nv) = coded(96 ..< 97)
        let out = cache.updateAndAttend(
            queries: q,
            keys: nk.expandedDimensions(axis: 0),
            values: nv.expandedDimensions(axis: 0),
            scale: scale, sinks: nil)

        // Reference: attention over the true window [65, 96].
        let (wk, wv) = coded(65 ..< 97)
        let reference = PagedAttentionReference.composedAttention(
            queries: q, keys: wk.expandedDimensions(axis: 0),
            values: wv.expandedDimensions(axis: 0), scale: scale)
        assertClose(out, reference)
    }
}
