// DetokenizerBridgeV2.swift
//
// Bridges WS-E's incremental text machinery — `DetokenizerV2` (UTF-8
// holdback) + `StopHoldback` (stop-string holdback) — into the engine
// loop's `CBv2IncrementalDetokenizer` seam. One instance per in-flight
// request. `push` runs on the engine thread but performs pure host string
// work on already-confirmed tokens (no MLX, no syncs); guarantees:
//
//  - emitted chunks never split a multi-byte UTF-8 sequence;
//  - no text at or past a stop-string match is ever emitted, even when the
//    stop spans several tokens or is completed by the end-of-generation
//    flush;
//  - concatenated chunks + flush are byte-exact with a whole-sequence
//    decode, minus any matched stop suffix.

import Foundation

/// `CBv2IncrementalDetokenizer` over DetokenizerV2 + StopHoldback.
public final class CBv2TextDetokenizer: CBv2IncrementalDetokenizer {

    private let detokenizer: DetokenizerV2
    private let holdback: StopHoldback

    /// True once a stop string fully matched; the engine finishes the
    /// request with `.stop` and no further text is ever emitted.
    public private(set) var matchedStopString = false

    public init(detokenizer: DetokenizerV2, holdback: StopHoldback) {
        self.detokenizer = detokenizer
        self.holdback = holdback
    }

    public func push(_ tokens: [Int]) -> String {
        guard !matchedStopString else { return "" }
        var text = ""
        for token in tokens {
            text += detokenizer.append(token)
        }
        guard !holdback.isPassthrough else { return text }
        let scan = holdback.ingest(text)
        if scan.stopped { matchedStopString = true }
        return scan.text
    }

    public func flush() -> String {
        guard !matchedStopString else { return "" }
        let tail = detokenizer.flush()
        guard !holdback.isPassthrough else { return tail }
        // The released UTF-8 tail can still complete a stop string.
        let scan = holdback.ingest(tail)
        if scan.stopped {
            matchedStopString = true
            return scan.text
        }
        return scan.text + holdback.flush()
    }
}

/// Engine-level factory: one tokenizer, one detokenizer per request with
/// that request's stop strings.
public final class CBv2TextDetokenizerFactory: CBv2DetokenizerFactory {

    private let tokenizer: any Tokenizer
    private let skipSpecialTokens: Bool

    public init(tokenizer: any Tokenizer, skipSpecialTokens: Bool = false) {
        self.tokenizer = tokenizer
        self.skipSpecialTokens = skipSpecialTokens
    }

    public func makeDetokenizer(stopStrings: [String]) -> CBv2IncrementalDetokenizer {
        CBv2TextDetokenizer(
            detokenizer: DetokenizerV2(
                tokenizer: tokenizer, skipSpecialTokens: skipSpecialTokens),
            holdback: StopHoldback(stopStrings: stopStrings))
    }
}
