import Foundation

public struct NGramSelfSimilarityOrderResult: Codable, Equatable {
    public let order: Int
    public let recurrentContextCount: Int
    public let recurrentContextRate: Double
    public let mostRecentFollowerHitCount: Int
    public let mostRecentFollowerHitRate: Double
    public let anyMatchingFollowerHitCount: Int
    public let anyMatchingFollowerHitRate: Double

    enum CodingKeys: String, CodingKey {
        case order
        case recurrentContextCount = "recurrent_context_count"
        case recurrentContextRate = "recurrent_context_rate"
        case mostRecentFollowerHitCount = "most_recent_follower_hit_count"
        case mostRecentFollowerHitRate = "most_recent_follower_hit_rate"
        case anyMatchingFollowerHitCount = "any_matching_follower_hit_count"
        case anyMatchingFollowerHitRate = "any_matching_follower_hit_rate"
    }
}

public struct NGramSelfSimilarityReport: Codable, Equatable {
    public let contextTokenCount: Int
    public let continuationTokenCount: Int
    public let orders: [NGramSelfSimilarityOrderResult]
    public let longestMatchOpportunityCount: Int
    public let longestMatchOpportunityRate: Double
    public let longestMatchMostRecentHitCount: Int
    public let longestMatchMostRecentHitRate: Double
    public let optimisticAnyOrderHitCount: Int
    public let optimisticAnyOrderHitRate: Double

    enum CodingKeys: String, CodingKey {
        case contextTokenCount = "context_token_count"
        case continuationTokenCount = "continuation_token_count"
        case orders
        case longestMatchOpportunityCount = "longest_match_opportunity_count"
        case longestMatchOpportunityRate = "longest_match_opportunity_rate"
        case longestMatchMostRecentHitCount = "longest_match_most_recent_hit_count"
        case longestMatchMostRecentHitRate = "longest_match_most_recent_hit_rate"
        case optimisticAnyOrderHitCount = "optimistic_any_order_hit_count"
        case optimisticAnyOrderHitRate = "optimistic_any_order_hit_rate"
    }

    public func passes(maximumHitRate: Double) -> Bool {
        maximumHitRate.isFinite
            && maximumHitRate >= 0
            && longestMatchMostRecentHitRate <= maximumHitRate
    }
}

public enum NGramSelfSimilarity {
    /// Scores how often a suffix-matching draft predictor can recover the next
    /// continuation token from earlier request context.
    ///
    /// For each continuation position and n-gram order, the analyzer looks up
    /// the immediately preceding n-gram among earlier occurrences whose
    /// following token is already known. It reports both:
    ///
    /// - a deterministic predictor that uses the most recent earlier follower;
    /// - an optimistic upper bound that counts a hit when any earlier
    ///   occurrence had the actual following token.
    ///
    /// The aggregate predictor chooses the longest recurrent order, then uses
    /// that order's most recent follower. This mirrors a practical
    /// longest-suffix prompt-lookup draft without using future tokens.
    public static func analyze(
        contextTokens: [Int],
        continuationTokens: [Int],
        orders requestedOrders: [Int] = [1, 2, 3]
    ) throws -> NGramSelfSimilarityReport {
        guard !continuationTokens.isEmpty else {
            throw MLXFastError.invalidInput("n-gram continuation must not be empty")
        }
        let orders = Array(Set(requestedOrders)).sorted()
        guard !orders.isEmpty, orders.allSatisfy({ $0 > 0 }) else {
            throw MLXFastError.invalidInput("n-gram orders must contain positive integers")
        }

        var histories: [Int: [[Int]: FollowerHistory]] = [:]
        var counters = Dictionary(
            uniqueKeysWithValues: orders.map { ($0, OrderCounter()) }
        )
        for order in orders {
            var history: [[Int]: FollowerHistory] = [:]
            if contextTokens.count > order {
                for followerIndex in order..<contextTokens.count {
                    let key = Array(contextTokens[(followerIndex - order)..<followerIndex])
                    history[key, default: FollowerHistory()].record(contextTokens[followerIndex])
                }
            }
            histories[order] = history
        }

        var knownTokens = contextTokens
        var longestMatchOpportunityCount = 0
        var longestMatchMostRecentHitCount = 0
        var optimisticAnyOrderHitCount = 0

        for targetToken in continuationTokens {
            var longestHistory: FollowerHistory?
            var anyOrderOracleHit = false

            for order in orders.reversed() where knownTokens.count >= order {
                let key = Array(knownTokens.suffix(order))
                guard let followerHistory = histories[order]?[key] else {
                    continue
                }

                if longestHistory == nil {
                    longestHistory = followerHistory
                }
                var counter = counters[order, default: OrderCounter()]
                counter.recurrentContextCount += 1
                if followerHistory.mostRecentFollower == targetToken {
                    counter.mostRecentFollowerHitCount += 1
                }
                if followerHistory.followers.contains(targetToken) {
                    counter.anyMatchingFollowerHitCount += 1
                    anyOrderOracleHit = true
                }
                counters[order] = counter
            }

            if let longestHistory {
                longestMatchOpportunityCount += 1
                if longestHistory.mostRecentFollower == targetToken {
                    longestMatchMostRecentHitCount += 1
                }
            }
            if anyOrderOracleHit {
                optimisticAnyOrderHitCount += 1
            }

            for order in orders where knownTokens.count >= order {
                let key = Array(knownTokens.suffix(order))
                var orderHistory = histories[order, default: [:]]
                orderHistory[key, default: FollowerHistory()].record(targetToken)
                histories[order] = orderHistory
            }
            knownTokens.append(targetToken)
        }

        let total = continuationTokens.count
        let orderResults = orders.map { order in
            let counter = counters[order, default: OrderCounter()]
            return NGramSelfSimilarityOrderResult(
                order: order,
                recurrentContextCount: counter.recurrentContextCount,
                recurrentContextRate: rate(counter.recurrentContextCount, total: total),
                mostRecentFollowerHitCount: counter.mostRecentFollowerHitCount,
                mostRecentFollowerHitRate: rate(counter.mostRecentFollowerHitCount, total: total),
                anyMatchingFollowerHitCount: counter.anyMatchingFollowerHitCount,
                anyMatchingFollowerHitRate: rate(counter.anyMatchingFollowerHitCount, total: total)
            )
        }

        return NGramSelfSimilarityReport(
            contextTokenCount: contextTokens.count,
            continuationTokenCount: continuationTokens.count,
            orders: orderResults,
            longestMatchOpportunityCount: longestMatchOpportunityCount,
            longestMatchOpportunityRate: rate(longestMatchOpportunityCount, total: total),
            longestMatchMostRecentHitCount: longestMatchMostRecentHitCount,
            longestMatchMostRecentHitRate: rate(longestMatchMostRecentHitCount, total: total),
            optimisticAnyOrderHitCount: optimisticAnyOrderHitCount,
            optimisticAnyOrderHitRate: rate(optimisticAnyOrderHitCount, total: total)
        )
    }

    private struct FollowerHistory {
        var followers: Set<Int> = []
        var mostRecentFollower: Int?

        mutating func record(_ token: Int) {
            followers.insert(token)
            mostRecentFollower = token
        }
    }

    private struct OrderCounter {
        var recurrentContextCount = 0
        var mostRecentFollowerHitCount = 0
        var anyMatchingFollowerHitCount = 0
    }

    private static func rate(_ count: Int, total: Int) -> Double {
        Double(count) / Double(total)
    }
}
