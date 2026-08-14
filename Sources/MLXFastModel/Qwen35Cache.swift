import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon

struct Qwen35CacheTopology: Equatable {
    let mambaCount: Int
    let fullAttentionCount: Int

    init(layerTypes: [Qwen35LayerType]) {
        self.mambaCount = layerTypes.count(where: { $0 == .linear })
        self.fullAttentionCount = layerTypes.count(where: { $0 == .full })
    }
}

/// Owns the selected backend's hybrid Qwen35 cache for one prompt.
///
/// The pinned model has 48 `MambaCache` entries and 16 `KVCacheSimple`
/// entries. Mamba state caches intentionally do not advance `KVCache.offset`
/// like full-attention caches do, so host position tracking is the
/// authoritative caller contract.
public final class Qwen35ModelCache {
    private let layerTypes: [Qwen35LayerType]
    public private(set) var expectedPositionOffset = 0
    public private(set) var caches: [any KVCache]?
    private var fastPathCache: Qwen35FastCache?

    public init(config: Qwen35Config) {
        self.layerTypes = config.layerTypes
    }

    init(layerTypes: [Qwen35LayerType]) {
        self.layerTypes = layerTypes
    }

    func nextExpectedPositionOffset(
        positionOffset: Int,
        inputLength: Int
    ) throws -> Int {
        try advancedQwen35CachePosition(
            positionOffset: positionOffset,
            expectedPositionOffset: expectedPositionOffset,
            inputLength: inputLength
        )
    }

    func commitExpectedPositionOffset(_ positionOffset: Int) {
        expectedPositionOffset = positionOffset
    }

    /// Create the cache through Qwen35TextModel's public API and retain it for
    /// all subsequent forwards in this prompt.
    public func cache(for model: Qwen35TextModel) throws -> [any KVCache] {
        if let caches {
            return caches
        }
        let created = model.newCache(parameters: nil)
        try validateQwen35CacheTopology(
            caches: created,
            layerTypes: layerTypes
        )
        caches = created
        return created
    }

    func cache(for engine: Qwen35FastEngine) -> Qwen35FastCache {
        if let fastPathCache {
            return fastPathCache
        }
        let created = engine.newCache()
        fastPathCache = created
        return created
    }

    func validatePosition(
        _ positionOffset: Int,
        caches: [any KVCache]
    ) throws {
        try validateQwen35CachePosition(
            positionOffset: positionOffset,
            caches: caches,
            layerTypes: layerTypes
        )
    }

    func validatePosition(
        _ positionOffset: Int,
        cache: Qwen35FastCache
    ) throws {
        guard positionOffset == expectedPositionOffset,
              positionOffset == cache.expectedPositionOffset
        else {
            throw MLXFastError.invalidInput(
                "Qwen35 custom cache position \(positionOffset) does not "
                    + "match expected offsets"
            )
        }
    }

    public func materializeCachedState() {
        if let caches {
            eval(caches)
        }
        fastPathCache?.materialize()
    }
}

func validateQwen35CacheTopology(
    caches: [any KVCache],
    layerTypes: [Qwen35LayerType]
) throws {
    guard caches.count == layerTypes.count else {
        throw MLXFastError.invalidInput(
            "Qwen35 cache count \(caches.count) does not match "
                + "layer count \(layerTypes.count)"
        )
    }

    for (index, pair) in zip(layerTypes, caches).enumerated() {
        switch pair.0 {
        case .linear:
            guard pair.1 is MambaCache else {
                throw MLXFastError.invalidInput(
                    "Qwen35 layer \(index) requires MambaCache"
                )
            }
        case .full:
            guard pair.1 is KVCacheSimple else {
                throw MLXFastError.invalidInput(
                    "Qwen35 layer \(index) requires KVCacheSimple"
                )
            }
        }
    }
}

/// Validate only full-attention offsets. Mamba caches hold recurrent state and
/// keep their inherited offset at zero in the pinned public implementation.
func validateQwen35CachePosition(
    positionOffset: Int,
    caches: [any KVCache],
    layerTypes: [Qwen35LayerType]
) throws {
    guard positionOffset >= 0 else {
        throw MLXFastError.invalidInput(
            "Qwen35 position offset must be non-negative"
        )
    }
    try validateQwen35CacheTopology(caches: caches, layerTypes: layerTypes)

    for (index, pair) in zip(layerTypes, caches).enumerated()
    where pair.0 == .full {
        guard pair.1.offset == positionOffset else {
            throw MLXFastError.invalidInput(
                "Qwen35 full-attention cache \(index) offset "
                    + "\(pair.1.offset) does not match position "
                    + "\(positionOffset)"
            )
        }
    }
}

func advancedQwen35CachePosition(
    positionOffset: Int,
    expectedPositionOffset: Int,
    inputLength: Int
) throws -> Int {
    guard positionOffset >= 0, expectedPositionOffset >= 0 else {
        throw MLXFastError.invalidInput(
            "Qwen35 position offset must be non-negative"
        )
    }
    guard positionOffset == expectedPositionOffset else {
        throw MLXFastError.invalidInput(
            "Qwen35 position offset \(positionOffset) does not match "
                + "expected cache offset \(expectedPositionOffset)"
        )
    }
    guard inputLength > 0 else {
        throw MLXFastError.invalidInput(
            "Qwen35 cached input length must be positive"
        )
    }
    let next = expectedPositionOffset.addingReportingOverflow(inputLength)
    guard !next.overflow else {
        throw MLXFastError.invalidInput(
            "Qwen35 cache position offset overflows Int"
        )
    }
    return next.partialValue
}
