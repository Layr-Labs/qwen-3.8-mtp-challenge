import Foundation

/// Parse a transformed-weights byte cap from CLI flags or `MLXFAST_MAX_WEIGHTS_BYTES`.
/// Empty input uses `defaultByteCount`; `0`, `none`, and `unlimited` disable the cap.
public func parseTransformedWeightsByteLimit(
    raw: String,
    defaultByteCount: Int? = MLXFastConstants.defaultMaxTransformedWeightsBytes,
    optionLabel: String = "MLXFAST_MAX_WEIGHTS_BYTES"
) throws -> Int? {
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else {
        return defaultByteCount
    }

    let lowercased = trimmed.lowercased()
    if lowercased == "0" || lowercased == "none" || lowercased == "unlimited" {
        return nil
    }
    guard let value = Int(trimmed), value > 0 else {
        throw MLXFastError.invalidInput(
            "\(optionLabel) must be a positive byte count, 0, none, or unlimited"
        )
    }
    return value
}

public func transformedWeightsByteLimit(
    from environment: [String: String] = ProcessInfo.processInfo.environment
) throws -> Int? {
    try parseTransformedWeightsByteLimit(raw: environment["MLXFAST_MAX_WEIGHTS_BYTES"] ?? "")
}
