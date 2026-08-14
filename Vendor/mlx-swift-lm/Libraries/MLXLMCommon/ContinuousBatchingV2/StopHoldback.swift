// StopHoldback.swift
//
// Streaming stop-string matcher for ContinuousBatchingV2 (workstream E).
// One instance per in-flight request, fed the incremental text produced by
// `DetokenizerV2`.
//
// Guarantee: `ingest` only ever returns text that provably cannot be part
// of a stop string — the longest suffix of the pending text that is a
// prefix of ANY stop string is held back until it is either completed
// (stop: truncate at the match start, signal stop, emit nothing further)
// or disambiguated (released with the next chunk). This makes workstream
// B's one-step-late stop detection invisible: a user can never observe
// text at or past a stop match, even when the stop string spans several
// tokens or overlaps another candidate ("</s>" vs "</section>").
//
// `flush()` releases the held tail at natural end of generation (EOS /
// max-tokens), where no stop can complete anymore.
//
// Matching is over unicode scalars (exact substring semantics, no
// normalization), mirroring how OpenAI-compatible servers compare stop
// strings against the decoded stream.

import Foundation

public struct CBv2StopScan: Sendable, Equatable {
    /// Text that is safe to emit now.
    public var text: String
    /// True when a stop string fully matched. All text at and after the
    /// match start has been discarded; every further `ingest`/`flush`
    /// yields nothing.
    public var stopped: Bool

    public init(text: String, stopped: Bool) {
        self.text = text
        self.stopped = stopped
    }
}

public final class StopHoldback {

    private let stops: [[Unicode.Scalar]]
    /// Longest stop length — an upper bound on how much text can be held.
    private let maxHold: Int
    /// Held-back tail: the longest suffix of everything ingested so far
    /// that is a prefix of at least one stop string.
    private var pending: [Unicode.Scalar] = []
    private var stopped = false

    public init(stopStrings: [String]) {
        self.stops = stopStrings.filter { !$0.isEmpty }.map { Array($0.unicodeScalars) }
        self.maxHold = stops.map(\.count).max() ?? 0
    }

    /// True when no stop strings are configured (pass-through mode).
    public var isPassthrough: Bool { stops.isEmpty }

    /// Feed newly detokenized text; returns what may be emitted.
    public func ingest(_ text: String) -> CBv2StopScan {
        if stopped {
            return CBv2StopScan(text: "", stopped: true)
        }
        if stops.isEmpty {
            return CBv2StopScan(text: text, stopped: false)
        }

        pending.append(contentsOf: text.unicodeScalars)

        // 1) Earliest full match across all stop strings wins; everything
        //    from the match start on is discarded.
        if let matchStart = earliestMatchStart() {
            stopped = true
            let emit = String(String.UnicodeScalarView(pending[..<matchStart]))
            pending.removeAll()
            return CBv2StopScan(text: emit, stopped: true)
        }

        // 2) No full match: hold the longest suffix that could still grow
        //    into a stop string; emit the rest.
        let hold = longestAmbiguousSuffix()
        let emitCount = pending.count - hold
        guard emitCount > 0 else {
            return CBv2StopScan(text: "", stopped: false)
        }
        let emit = String(String.UnicodeScalarView(pending[..<emitCount]))
        pending.removeFirst(emitCount)
        return CBv2StopScan(text: emit, stopped: false)
    }

    /// Release held text at natural end of generation (no stop matched).
    public func flush() -> String {
        guard !stopped, !pending.isEmpty else { return "" }
        let emit = String(String.UnicodeScalarView(pending))
        pending.removeAll()
        return emit
    }

    // MARK: - Internals

    /// Index in `pending` of the earliest occurrence of any stop string,
    /// or nil. Scanning is bounded: text before the held tail was already
    /// proven stop-free, so only positions where a stop could overlap the
    /// newly appended scalars need checking — but `pending` only ever
    /// contains the held tail plus this chunk, so a full scan is O(|pending|·Σ|stop|).
    private func earliestMatchStart() -> Int? {
        var earliest: Int? = nil
        for stop in stops {
            guard stop.count <= pending.count else { continue }
            let lastStart = pending.count - stop.count
            var start = 0
            while start <= lastStart {
                if earliest != nil, start >= earliest! { break }
                var i = 0
                while i < stop.count, pending[start + i] == stop[i] { i += 1 }
                if i == stop.count {
                    if earliest == nil || start < earliest! { earliest = start }
                    break
                }
                start += 1
            }
        }
        return earliest
    }

    /// Length of the longest suffix of `pending` that is a proper prefix
    /// of at least one stop string (the part that must be held back).
    private func longestAmbiguousSuffix() -> Int {
        var longest = 0
        for stop in stops {
            let maxLen = min(stop.count - 1, pending.count)
            guard maxLen > longest else { continue }
            var k = maxLen
            while k > longest {
                var matches = true
                let offset = pending.count - k
                for i in 0 ..< k where pending[offset + i] != stop[i] {
                    matches = false
                    break
                }
                if matches {
                    longest = k
                    break
                }
                k -= 1
            }
        }
        return longest
    }
}
