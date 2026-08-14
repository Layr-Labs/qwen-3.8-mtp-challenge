import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon

/// Qwen35 text forward used by the scored runtime.
///
/// The internal readiness selector remains `.libraryOracle`, so the pinned
/// Qwen35TextModel is still the active production implementation. The custom
/// engine branch is compile-checked but cannot be selected by environment.
public enum Qwen35Model {
    /// Returns all-position logits from Qwen35TextModel.
    ///
    /// The pinned public API does not expose the inner final norm and explicit
    /// LM head as a supported last-token seam. Slicing before the vocabulary
    /// projection is therefore deferred until the library provides one.
    public static func logits(
        inputIDs: MLXArray,
        weightCache: Qwen35RuntimeWeightCache,
        cache: Qwen35ModelCache? = nil,
        positionOffset: Int = 0
    ) throws -> MLXArray {
        try validateQwen35Input(inputIDs, positionOffset: positionOffset)
        switch weightCache.executionBackend {
        case .libraryOracle:
            return try libraryLogits(
                inputIDs: inputIDs,
                weightCache: weightCache,
                cache: cache,
                positionOffset: positionOffset
            )
        case .customFastPath:
            return try fastPathLogits(
                inputIDs: inputIDs,
                weightCache: weightCache,
                cache: cache,
                positionOffset: positionOffset
            )
        }
    }

    private static func libraryLogits(
        inputIDs: MLXArray,
        weightCache: Qwen35RuntimeWeightCache,
        cache: Qwen35ModelCache?,
        positionOffset: Int
    ) throws -> MLXArray {
        let model = try weightCache.requireLibraryModel()

        guard let cache else {
            let modelCaches = model.newCache(parameters: nil)
            try validateQwen35CachePosition(
                positionOffset: positionOffset,
                caches: modelCaches,
                layerTypes: weightCache.config.layerTypes
            )
            return qwen35EagerLogits(
                model: model,
                inputIDs: inputIDs,
                cache: modelCaches
            )
        }

        let modelCaches = try cache.cache(for: model)
        return try executeQwen35CachedForward(
            cache: cache,
            positionOffset: positionOffset,
            inputLength: inputIDs.dim(1),
            validateLibraryCachePosition: {
                try cache.validatePosition(
                    positionOffset,
                    caches: modelCaches
                )
            },
            forward: {
                qwen35EagerLogits(
                    model: model,
                    inputIDs: inputIDs,
                    cache: modelCaches
                )
            }
        )
    }

    private static func fastPathLogits(
        inputIDs: MLXArray,
        weightCache: Qwen35RuntimeWeightCache,
        cache: Qwen35ModelCache?,
        positionOffset: Int
    ) throws -> MLXArray {
        let engine = try weightCache.requireFastEngine()
        guard let cache else {
            return try engine.allPositionLogits(
                inputIDs,
                positionOffset: positionOffset
            )
        }

        let fastCache = cache.cache(for: engine)
        return try executeQwen35CachedForward(
            cache: cache,
            positionOffset: positionOffset,
            inputLength: inputIDs.dim(1),
            validateLibraryCachePosition: {
                try cache.validatePosition(
                    positionOffset,
                    cache: fastCache
                )
            },
            forward: {
                try engine.allPositionLogits(
                    inputIDs,
                    cache: fastCache,
                    positionOffset: positionOffset
                )
            }
        )
    }
}

@inline(__always)
func qwen35EagerLogits(
    model: Qwen35TextModel,
    inputIDs: MLXArray,
    cache: [any KVCache]
) -> MLXArray {
    model(inputIDs, cache: cache)
}

@inline(__always)
func executeQwen35CachedForward<Result>(
    cache: Qwen35ModelCache,
    positionOffset: Int,
    inputLength: Int,
    validateLibraryCachePosition: () throws -> Void,
    forward: () throws -> Result
) throws -> Result {
    let nextExpectedPositionOffset = try cache.nextExpectedPositionOffset(
        positionOffset: positionOffset,
        inputLength: inputLength
    )
    try validateLibraryCachePosition()
    let result = try forward()
    cache.commitExpectedPositionOffset(nextExpectedPositionOffset)
    return result
}

private func validateQwen35Input(
    _ inputIDs: MLXArray,
    positionOffset: Int
) throws {
    guard positionOffset >= 0 else {
        throw MLXFastError.invalidInput(
            "Qwen35 position offset must be non-negative"
        )
    }
    guard inputIDs.ndim == 2 else {
        throw MLXFastError.invalidInput(
            "Qwen35 input IDs must have shape [batch, length]"
        )
    }
    guard inputIDs.dim(0) > 0, inputIDs.dim(1) > 0 else {
        throw MLXFastError.invalidInput(
            "Qwen35 input IDs must have non-empty batch and length"
        )
    }
}
