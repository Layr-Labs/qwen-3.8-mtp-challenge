enum Qwen35ExecutionBackend: Equatable {
    case libraryOracle
    case customFastPath
}

/// Internal, compile-time activation boundary. Submission-visible environment
/// variables cannot select the custom implementation.
enum Qwen35FastPathReadiness {
    // These stay false until real-checkpoint all-layer, prefill/decode, and
    // token-parity gates have passed against Qwen35TextModel. MTP rollback is
    // intentionally outside this gate: MTP is disabled and the frozen
    // checkpoint contains no mtp.* tensors.
    static let realCheckpointParityPassed = false
    static let productionActivationApproved = false

    static let productionBackend = selectQwen35ExecutionBackend(
        realCheckpointParityPassed: realCheckpointParityPassed,
        productionActivationApproved: productionActivationApproved
    )
}

/// Injectable only through internal/testable Swift calls; production uses the
/// hardcoded values above, never process environment.
func selectQwen35ExecutionBackend(
    realCheckpointParityPassed: Bool,
    productionActivationApproved: Bool
) -> Qwen35ExecutionBackend {
    realCheckpointParityPassed && productionActivationApproved
        ? .customFastPath
        : .libraryOracle
}
