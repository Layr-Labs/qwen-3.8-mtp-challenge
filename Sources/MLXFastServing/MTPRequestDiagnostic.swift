import Foundation

/// Native session work, including tokens committed beyond an API stop.
/// These counters are distinct from the API's delivered completion usage.
public struct MTPServingStatistics: Sendable, Equatable {
    public let cycles: Int
    public let acceptedDrafts: Int
    public let rejectedDrafts: Int
    public let committedTokens: Int

    public init(cycles: Int, acceptedDrafts: Int, rejectedDrafts: Int, committedTokens: Int) {
        self.cycles = cycles
        self.acceptedDrafts = acceptedDrafts
        self.rejectedDrafts = rejectedDrafts
        self.committedTokens = committedTokens
    }
}

public struct MTPRequestDiagnostic: Sendable {
    public enum Outcome: String, Sendable { case completed, cancelled, error }
    public let requestID: String
    public let outcome: Outcome
    public let maxDepth: Int
    public let statistics: MTPServingStatistics?
    public let completionTokens: Int?
    public let generationElapsedMilliseconds: Int

    /// Fixed keys and scalar values only. No model, prompt, token IDs, or error text.
    public var stderrLine: String {
        let counts: String
        if let statistics {
            counts = "cycles=\(statistics.cycles) accepted_drafts=\(statistics.acceptedDrafts) "
                + "rejected_drafts=\(statistics.rejectedDrafts) committed_tokens=\(statistics.committedTokens)"
        } else { counts = "statistics=unavailable" }
        return "native_mtp request=\(requestID) outcome=\(outcome.rawValue) max_depth=\(maxDepth) "
            + counts + " completion_tokens=\(completionTokens.map(String.init) ?? "unknown") "
            + "generation_elapsed_ms=\(generationElapsedMilliseconds)\n"
    }

    public static func writeStandardError(_ diagnostic: Self) {
        FileHandle.standardError.write(Data(diagnostic.stderrLine.utf8))
    }
}

/// Only completed session-boundary snapshots cross the generation task.
/// One scalar copy per begin/round; no per-token lock or log operation.
final class MTPRequestStatisticsStore: @unchecked Sendable {
    private let lock = NSLock()
    private var value: MTPServingStatistics?
    func record(_ statistics: MTPServingStatistics?) { lock.withLock { value = statistics } }
    var snapshot: MTPServingStatistics? { lock.withLock { value } }
}

final class ObservedMTPServingSession: MTPServingSession {
    private let session: any MTPServingSession
    private let store: MTPRequestStatisticsStore
    init(session: any MTPServingSession, store: MTPRequestStatisticsStore) {
        self.session = session
        self.store = store
    }
    func begin(seedTokens: [Int]) throws {
        defer { store.record(session.statistics) }
        try session.begin(seedTokens: seedTokens)
    }
    func round(depth: Int) throws -> [Int] {
        defer { store.record(session.statistics) }
        return try session.round(depth: depth)
    }
    var statistics: MTPServingStatistics? { session.statistics }
}
