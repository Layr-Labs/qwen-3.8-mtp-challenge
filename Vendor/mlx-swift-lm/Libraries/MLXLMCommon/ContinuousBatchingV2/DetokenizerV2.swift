// DetokenizerV2.swift
//
// Per-request incremental detokenization for ContinuousBatchingV2
// (workstream E). One instance per in-flight request; runs OFF the engine
// step thread (detokenization is host work — architecture invariant 6).
//
// Guarantee: emitted chunks never contain a replacement character caused by
// a multi-byte UTF-8 sequence split across tokens (byte-fallback BPE,
// emoji, CJK). Bytes are held back until the tokenizer produces a valid
// UTF-8 boundary, and the concatenation of all emitted chunks plus `flush`
// is byte-exact with `tokenizer.decode(allTokens)` segment by segment.
//
// Mechanism (mlx-lm-style): decode the current token segment, strip any
// trailing U+FFFD run (the tokenizer's marker for incomplete byte
// sequences), and emit only the bytes past the previously emitted prefix.
// When the pending token later completes the character, the re-decode
// yields the real bytes and the suffix is released. A genuine U+FFFD in
// model output is only ever delayed by one token, never dropped.
//
// The segment is periodically restarted (on newline boundaries, or after
// `maxSegmentTokens` when at a clean boundary) so per-token decode cost
// stays bounded. The last token is carried into the new segment because
// SentencePiece-style decoders strip leading whitespace markers from the
// first token of a decode call; keeping one token of context pins the
// boundary.

import Foundation

public final class DetokenizerV2 {

    private let tokenizer: any Tokenizer
    private let skipSpecialTokens: Bool
    private let maxSegmentTokens: Int

    /// Tokens of the current decode segment.
    private var segmentTokens: [Int] = []
    /// UTF-8 byte count of the current segment's decode output that has
    /// already been emitted.
    private var emittedBytes = 0

    /// - Parameters:
    ///   - tokenizer: tokenizer used for decoding.
    ///   - skipSpecialTokens: forwarded to `Tokenizer.decode`.
    ///   - maxSegmentTokens: soft cap on segment length before a restart
    ///     is attempted at the next clean boundary (keeps incremental
    ///     decode cost O(segment), not O(total generation)).
    public init(
        tokenizer: any Tokenizer, skipSpecialTokens: Bool = false, maxSegmentTokens: Int = 256
    ) {
        self.tokenizer = tokenizer
        self.skipSpecialTokens = skipSpecialTokens
        self.maxSegmentTokens = max(2, maxSegmentTokens)
    }

    /// Append one generated token; returns the text that became safe to
    /// emit (possibly empty while bytes are held back at an incomplete
    /// UTF-8 boundary).
    public func append(_ token: Int) -> String {
        segmentTokens.append(token)
        let decoded = decodeSegment()
        let stable = Self.strippingTrailingReplacementRun(decoded)
        let emitted = emit(upTo: stable)

        // Restart the segment at clean boundaries so decode cost stays
        // bounded. BOTH triggers (newline, maxSegmentTokens) sit inside the
        // clean-boundary guard: the first two conditions require that no
        // bytes are held back (`stable == decoded`) and everything stable
        // was emitted — a restart while a multi-byte sequence straddles the
        // boundary would record its U+FFFD decode as "already emitted" and
        // the completing token's real bytes would never be released. The
        // `maxSegmentTokens` cap is therefore SOFT: it defers past the limit
        // until the holdback clears (PR#62 review).
        if stable.count == decoded.count,
            stable.count == emittedBytes,
            (decoded.last == UInt8(ascii: "\n") || segmentTokens.count >= maxSegmentTokens)
        {
            startNewSegment()
        }
        return emitted
    }

    /// Release any held-back text at end of generation. A trailing
    /// incomplete character (the model stopped mid-sequence) decodes to
    /// U+FFFD here — that is end-of-stream, not mid-stream.
    public func flush() -> String {
        let decoded = decodeSegment()
        let emitted = String(decoding: decoded[emittedBytes...], as: UTF8.self)
        emittedBytes = decoded.count
        return emitted
    }

    // MARK: - Internals

    private func decodeSegment() -> [UInt8] {
        guard !segmentTokens.isEmpty else { return [] }
        return Array(
            tokenizer.decode(tokenIds: segmentTokens, skipSpecialTokens: skipSpecialTokens).utf8)
    }

    /// Emit the bytes of `stable` past the already-emitted prefix.
    private func emit(upTo stable: [UInt8]) -> String {
        guard stable.count > emittedBytes else { return "" }
        let chunk = String(decoding: stable[emittedBytes...], as: UTF8.self)
        emittedBytes = stable.count
        return chunk
    }

    /// Keep the last already-emitted token as context for the next segment
    /// (guards SentencePiece leading-whitespace stripping at the boundary).
    private func startNewSegment() {
        guard let last = segmentTokens.last else { return }
        segmentTokens = [last]
        emittedBytes = decodeSegment().count
    }

    /// Drop the trailing run of U+FFFD (EF BF BD) from `bytes`. Trailing
    /// replacement characters mark a potentially incomplete multi-byte
    /// sequence and must be held back; earlier ones are already stable.
    static func strippingTrailingReplacementRun(_ bytes: [UInt8]) -> [UInt8] {
        var end = bytes.count
        while end >= 3, bytes[end - 3] == 0xEF, bytes[end - 2] == 0xBF, bytes[end - 1] == 0xBD {
            end -= 3
        }
        return end == bytes.count ? bytes : Array(bytes[..<end])
    }
}
