import Foundation

public struct ExpertStreamingStats: Codable, Equatable, Sendable {
    public let cacheHits: UInt64
    public let cacheMisses: UInt64
    public let cacheEvictions: UInt64
    public let bytesRead: UInt64
    public let readSeconds: Double
    public let peakCachedTensors: UInt64

    enum CodingKeys: String, CodingKey {
        case cacheHits = "expert_cache_hits"
        case cacheMisses = "expert_cache_misses"
        case cacheEvictions = "expert_cache_evictions"
        case bytesRead = "expert_bytes_read"
        case readSeconds = "expert_read_seconds"
        case peakCachedTensors = "expert_peak_cached_tensors"
    }

    public static let zero = ExpertStreamingStats()

    public init(
        cacheHits: UInt64 = 0,
        cacheMisses: UInt64 = 0,
        cacheEvictions: UInt64 = 0,
        bytesRead: UInt64 = 0,
        readSeconds: Double = 0,
        peakCachedTensors: UInt64 = 0
    ) {
        self.cacheHits = cacheHits
        self.cacheMisses = cacheMisses
        self.cacheEvictions = cacheEvictions
        self.bytesRead = bytesRead
        self.readSeconds = readSeconds
        self.peakCachedTensors = peakCachedTensors
    }

    public var totalLookups: UInt64 {
        cacheHits + cacheMisses
    }

    public var hitRate: Double {
        totalLookups == 0 ? 0 : Double(cacheHits) / Double(totalLookups)
    }
}
