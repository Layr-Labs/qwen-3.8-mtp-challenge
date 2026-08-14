// PagedSequenceKV.swift
//
// Per-sequence, per-layer KV state for the paged backend (WS-C).
//
// A `PagedSequenceKV` owns a page TABLE (`[Int32]` physical page ids into
// its group's slabs) plus its absolute position counter. There is no shared
// frontier: joining a batch adds an object, leaving drops it, and nothing a
// batchmate does can move this sequence's positions or evict its pages.
//
// Windowed layers use a RING of pages: token at absolute position `p` lives
// in table slot `(p / pageSize) % ringPages`, slot `p % pageSize`. Eviction
// is therefore implicit overwrite keyed to absolute positions — the recent
// end always survives, no window masks over shared buffers exist, and the
// ring is sized to hold a full prefill chunk plus the trailing window so
// every query in a chunk can still see its complete window after the chunk
// is written.

import Foundation
import MLX

public final class PagedSequenceKV: CBv2SequenceKV {
    let pool: PagedKVPool
    let groupKey: PagedKVGroupKey
    /// Pool-issued monotonic identity (never reused, unlike a heap address).
    /// Device block-table caches fingerprint rows by (serial, tableVersion)
    /// so a new row can never alias a released one.
    let serial: UInt64
    /// Sliding window in tokens (nil == full attention).
    public let windowSize: Int?
    /// Ring capacity in pages for windowed layers (nil for full).
    let ringPages: Int?
    /// Worst-case pages this sequence may allocate (reserved at admission;
    /// the pool guarantees allocations up to this count cannot fail).
    let reservedPages: Int
    /// Hard bound on `absoluteOffset` (from the request's maxLength).
    public let maxLength: Int

    /// Physical page ids. Full: index == logical page. Windowed: index ==
    /// logical page % ringPages.
    private(set) var table: [Int32] = []
    /// Bumped whenever `table` changes — lets the layer cache reuse device
    /// block tables across steps that only append tokens within a page.
    private(set) var tableVersion: Int = 0

    public private(set) var absoluteOffset: Int = 0
    /// Absolute position before which nothing was ever written here
    /// (nonzero only after `fastForward(to:)` for prefix-adopted rows).
    private(set) var baseOffset: Int = 0
    /// Size of the most recent update — governs how much trailing context
    /// `retainedCount` exposes for windowed layers (a chunk's earliest query
    /// must see its full window).
    private var lastUpdateTokens: Int = 1

    private var released = false

    init(
        pool: PagedKVPool, kind: CBv2LayerKind, maxLength: Int, reservedPages: Int
    ) {
        self.pool = pool
        self.groupKey = PagedKVGroupKey(kind)
        self.serial = pool.nextRowSerial()
        switch kind.attention {
        case .full:
            self.windowSize = nil
            self.ringPages = nil
        case .slidingWindow(let window):
            self.windowSize = window
            self.ringPages = PagedKVPool.ringPageCount(window: window, config: pool.config)
        }
        self.maxLength = maxLength
        self.reservedPages = reservedPages
    }

    deinit {
        assert(released, "[PagedSequenceKV] leaked without release — pages still allocated")
    }

    // MARK: - CBv2SequenceKV

    public var retainedCount: Int {
        let written = absoluteOffset - baseOffset
        guard let window = windowSize else { return written }
        return min(written, window - 1 + lastUpdateTokens)
    }

    public var byteCount: Int {
        table.count * pool.group(groupKey).pageBytes
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        var k = keys
        var v = values
        if k.ndim == 4 {
            precondition(k.dim(0) == 1, "per-sequence update requires batch dim 1")
            k = k.squeezed(axis: 0)
            v = v.squeezed(axis: 0)
        }
        write(keys: k, values: v)
        return attendableViews()
    }

    public func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        let (k, v) = attendableViews()
        return (k, v, absoluteOffset)
    }

    public func rollback(_ n: Int) {
        precondition(n >= 0 && n <= absoluteOffset - baseOffset, "rollback past written tokens")
        guard n > 0 else { return }
        absoluteOffset -= n
        if ringPages == nil {
            // Full layer: free pages past the new frontier. Freed pages may
            // hold stale bytes; they are only ever re-read by their NEXT
            // owner after that owner writes them (all reads are bounded by
            // the owner's absoluteOffset), so no scrubbing pass is needed.
            let keepPages = (absoluteOffset + pool.config.pageSize - 1) / pool.config.pageSize
            if keepPages < table.count {
                pool.freePages(group: groupKey, pages: table[keepPages...])
                table.removeSubrange(keepPages...)
                tableVersion += 1
            }
        }
        // Windowed rings keep their pages; the rolled-back slots are
        // overwritten before they can re-enter any attendable range.
        lastUpdateTokens = 1
    }

    // MARK: - Writes

    /// Append `keys`/`values` (`[kvHeads, n, headDim]`) at the current
    /// frontier via ONE in-place write-kernel dispatch. Never throws: the
    /// pool guarantees pages up to the admission-time reservation.
    func write(keys: MLXArray, values: MLXArray) {
        let n = keys.dim(1)
        guard n > 0 else { return }
        precondition(
            absoluteOffset + n <= maxLength,
            "write past maxLength (\(absoluteOffset) + \(n) > \(maxLength))")
        if windowSize != nil {
            precondition(
                n <= pool.config.maxPrefillChunk,
                "windowed update of \(n) tokens exceeds maxPrefillChunk "
                    + "\(pool.config.maxPrefillChunk) — the ring cannot hold it")
        }
        let s = pool.config.pageSize

        // Physical destination (page * pageSize + slot) per token.
        var slots = [Int32]()
        slots.reserveCapacity(n)
        for i in 0 ..< n {
            let pos = absoluteOffset + i
            let physical = ensurePage(logicalPage: pos / s)
            slots.append(physical * Int32(s) + Int32(pos % s))
        }
        pool.writeTokens(group: groupKey, slots: slots, keys: keys, values: values)

        absoluteOffset += n
        lastUpdateTokens = n
    }

    /// Reserve the destination for ONE decode token and advance the
    /// frontier WITHOUT touching storage — the paged decode kernel writes
    /// the tile in place (fused write, see PagedAttentionKernel.decode).
    /// Host Int math only; never a device sync.
    func prepareDecodeWrite() -> (page: Int32, slot: Int) {
        precondition(
            absoluteOffset + 1 <= maxLength,
            "write past maxLength (\(absoluteOffset) + 1 > \(maxLength))")
        let s = pool.config.pageSize
        let page = ensurePage(logicalPage: absoluteOffset / s)
        let slot = absoluteOffset % s
        absoluteOffset += 1
        lastUpdateTokens = 1
        return (page, slot)
    }

    private func ensurePage(logicalPage lp: Int) -> Int32 {
        let slotIndex: Int
        if let ring = ringPages {
            slotIndex = lp % ring
        } else {
            slotIndex = lp
        }
        while table.count <= slotIndex {
            precondition(
                table.count < reservedPages,
                "[PagedSequenceKV] allocation past reservation (\(reservedPages) pages) — "
                    + "admission accounting bug")
            table.append(pool.allocatePage(group: groupKey))
            tableVersion += 1
        }
        return table[slotIndex]
    }

    // MARK: - Reads

    /// Gathered `[1, kvHeads, retainedCount, headDim]` views over the
    /// attendable range, oldest to newest, in the pool dtype. The views
    /// are lazy reads of the live slabs — consume them within the current
    /// engine step and drop them: the slabs are mutated in place, so a
    /// stale unevaluated gather could observe recycled pages.
    func attendableViews() -> (keys: MLXArray, values: MLXArray) {
        let retained = retainedCount
        let start = absoluteOffset - retained
        return gatherRange(start: start, count: retained)
    }

    /// Gather an arbitrary written range (used by prefill fallback and
    /// snapshots). `start` is an absolute position.
    func gatherRange(start: Int, count: Int) -> (keys: MLXArray, values: MLXArray) {
        guard count > 0 else {
            return pool.gather(group: groupKey, pages: [], firstSlot: 0, count: 0)
        }
        precondition(start >= baseOffset && start + count <= absoluteOffset)
        if let window = windowSize {
            // The ring only holds the recent end; older positions are gone.
            let ring = ringPages! * pool.config.pageSize
            precondition(
                start >= absoluteOffset - ring,
                "gather of evicted window range (window \(window))")
        }
        let s = pool.config.pageSize
        let lpFirst = start / s
        let lpLast = (start + count - 1) / s
        var pages: [Int32] = []
        pages.reserveCapacity(lpLast - lpFirst + 1)
        for lp in lpFirst ... lpLast {
            let idx = ringPages.map { lp % $0 } ?? lp
            pages.append(table[idx])
        }
        return pool.gather(
            group: groupKey, pages: pages, firstSlot: start % s, count: count)
    }

    /// Modular table length the decode kernel divides by to resolve a
    /// logical page to a physical one (`phys = table[logicalPage % len]`).
    ///
    /// WINDOWED rows MUST report the FULL ring length, never `table.count`.
    /// Pages are placed at `logicalPage % ringPages` (`ensurePage`), but the
    /// physical table is grown LAZILY only up to the slots the writes/replay
    /// touched. A prefix-adopted row (`fastForward` then a trailing replay
    /// that covers less than the ring) can leave `table.count < ringPages`;
    /// feeding that as the divisor makes the kernel wrap at the wrong length
    /// and alias the WRONG physical pages during decode. The ring length is
    /// the divisor the writes used, so it is the only correct divisor.
    /// Every position the decode actually attends was written, so its ring
    /// slot (`< table.count`) is allocated — the larger divisor never indexes
    /// an unallocated slot.
    ///
    /// FULL rows keep `table.count` (identity modulo: `table.count` already
    /// exceeds every logical page a full row can reach).
    var decodeTableLength: Int { ringPages ?? table.count }

    /// Kernel-facing row descriptor for decode: the absolute range the
    /// current query may attend to, plus the (modular) table length.
    /// All plain Swift Int math — never a device sync.
    var decodeAttendRange: (start: Int, length: Int) {
        let qPos = absoluteOffset - 1
        precondition(qPos >= baseOffset, "decode before any token was written")
        var start = baseOffset
        if let window = windowSize {
            start = max(start, qPos - window + 1)
        }
        return (start, qPos - start + 1)
    }

    // MARK: - Lifecycle

    /// Advance the position counter without writing storage. Only valid on
    /// a fresh WINDOWED state during prefix-cache adoption: the engine
    /// recomputes the trailing window tokens for windowed layers, and those
    /// recomputed tokens must land at their true absolute positions.
    public func fastForward(to offset: Int) {
        precondition(windowSize != nil, "fastForward is only for windowed layers")
        precondition(
            table.isEmpty && absoluteOffset == 0 && baseOffset == 0,
            "fastForward requires a fresh state")
        precondition(offset >= 0 && offset <= maxLength)
        absoluteOffset = offset
        baseOffset = offset
    }

    /// Return every page to the pool and release the reservation.
    /// O(pages) metadata, no device work. Idempotent.
    ///
    /// RECYCLE INVARIANT: the freed pages may be handed to a new row whose
    /// bulk writes have no graph edge to THIS row's in-flight reads (the
    /// no-write kernel variants and gathers consume fences but never
    /// advance them). Callers must therefore only release a row after its
    /// last consuming step has been host-synced (the engine's finalize
    /// discipline — releases happen on the engine thread between steps).
    /// A release-without-host-sync fast path would race silently.
    func releaseStorage() {
        guard !released else { return }
        released = true
        pool.freePages(group: groupKey, pages: table)
        table.removeAll()
        tableVersion += 1
        pool.unreserve([groupKey: reservedPages])
    }
}
