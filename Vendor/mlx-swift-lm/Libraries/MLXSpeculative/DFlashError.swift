// Copyright © 2026 Apple Inc.

import Foundation

public enum DFlashError: LocalizedError, Sendable, Equatable {
    case unsupportedTarget(String)
    case rebindForbidden
    case incompatibleDrafter(field: String, drafter: String, target: String)
    case drafterNotBound
    case invalidCacheCount(expected: Int, actual: Int)
    case invalidLogitsStart(Int)
    case targetHiddenSizeMismatch(expected: Int, actual: Int)
    case missingSlidingWindow
    case invalidBlockSize(Int)
    case unsupportedSamplingTemperature(Float)
    case untrimmableCache
    case missingConfig(String)
    case unreadableDirectory(String)
    case noSafetensorsFound(String)
    case duplicateWeightKey(String)
    case invalidBatchArguments(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedTarget(let target):
            return "DFlash requires a target model conforming to DFlashTargetModel; got \(target)."
        case .rebindForbidden:
            return "DFlashDraftModel cannot be rebound to a different target."
        case .incompatibleDrafter(let field, let drafter, let target):
            return
                "DFlash drafter/target mismatch on \(field): drafter=\(drafter), target=\(target)."
        case .drafterNotBound:
            return "DFlashDraftModel.bind(target:) must be called before any forward pass."
        case .invalidCacheCount(let expected, let actual):
            return "DFlash draft cache count mismatch: expected \(expected), got \(actual)."
        case .invalidLogitsStart(let value):
            return "DFlash logitsStart must be non-negative; got \(value)."
        case .targetHiddenSizeMismatch(let expected, let actual):
            return "DFlash target hidden width mismatch: expected \(expected), got \(actual)."
        case .missingSlidingWindow:
            return "DFlash sliding_attention layers require sliding_window."
        case .invalidBlockSize(let value):
            return "DFlash block size must be at least 2; got \(value)."
        case .unsupportedSamplingTemperature(let value):
            return "DFlashTokenIterator currently supports greedy temperature=0 only; got \(value)."
        case .untrimmableCache:
            return "DFlash speculative decoding requires trimmable target and draft caches."
        case .missingConfig(let directory):
            return "DFlash draft directory is missing config.json: \(directory)."
        case .unreadableDirectory(let directory):
            return "Could not enumerate DFlash draft directory: \(directory)."
        case .noSafetensorsFound(let directory):
            return "DFlash draft directory contains no .safetensors weights: \(directory)."
        case .duplicateWeightKey(let key):
            return "DFlash draft weights contain duplicate tensor key: \(key)."
        case .invalidBatchArguments(let message):
            return "Invalid DFlash batch arguments: \(message)."
        }
    }
}
