import Foundation
@testable import MLXFastModel
import Testing

@Test
func startupMemoryPolicyProtectsDocumented36GiBLocalMachine() {
    let policy = RuntimeStartupMemoryPolicy.resolve(
        physicalMemoryBytes: UInt64(36) << 30
    )

    #expect(policy.isLowMemory)
    #expect(policy.cacheLimitBytes == 6 << 30)
    #expect(policy.maxMegabytesPerCommandBuffer == 128)
    #expect(policy.maxOperationsPerCommandBuffer == 64)
    #expect(policy.clearAllocatorCacheAfterWarmup)
    // Consistency contract: the low-memory profile is pure memory
    // management and must not default any feature flag off. A memory-gated
    // feature-disable default makes small machines silently skip code paths
    // the ranked box runs (the old profile defaulted the compiled-decode
    // flags off below 64 GiB exactly this way, so <64 GiB machines never
    // executed the compile(shapeless:) decode closures the ranked M5
    // exercises). If this expectation fails, a feature divergence has been
    // reintroduced; read the policy's doc comment before shipping it.
    #expect(policy.environmentOverrides.isEmpty)
}

// The low-memory profile no longer disables the compiled-decode flags, so
// both must be unread by the policy and remain default-on where the model
// code consults them (MLXLMCommon CompiledDecode/MLXHardwareInfo). This
// pins the flag spellings so a vendored rename cannot silently orphan the
// documented opt-out names.
@Test
func compiledDecodeFlagsStayReadableByModelSources() throws {
    let vendoredRuntimeDirectory = "Vendor/mlx-swift-lm/Libraries/MLXLMCommon"
    let sourceFiles = try FileManager.default
        .contentsOfDirectory(atPath: vendoredRuntimeDirectory)
        .filter { $0.hasSuffix(".swift") }
    #expect(!sourceFiles.isEmpty)
    let combinedSources = try sourceFiles
        .map {
            try String(
                contentsOfFile: "\(vendoredRuntimeDirectory)/\($0)",
                encoding: .utf8
            )
        }
        .joined(separator: "\n")
    for name in ["MLX_COMPILED_DECODE", "DARKBLOOM_COMPILED_DECODE"] {
        #expect(
            combinedSources.contains("\"\(name)\""),
            "\(name) is not read anywhere in \(vendoredRuntimeDirectory)"
        )
    }
}

// The documented low-memory startup profile (<64 GiB machines: 6 GiB MLX
// allocator cap, shortened command buffers, warmup-buffer clear before the
// protocol hello; no env feature defaults) applies to the LAGUNA runtime
// worker. The Laguna
// weight cache must resolve the policy
// before its model load and honor clearAllocatorCacheAfterWarmup; the full
// profile stays a no-op there (the ranked box keeps stock allocator
// behavior, matching how the pinned baseline was measured).
@Test
func lagunaWeightCacheConsultsStartupMemoryPolicy() throws {
    let source = try String(
        contentsOfFile: "Sources/MLXFastModel/LagunaRuntimeWeights.swift",
        encoding: .utf8
    )
    #expect(source.contains("RuntimeStartupMemoryPolicy.resolve("))
    #expect(source.contains("RuntimeStartupMemoryPolicy.profileOverrideEnvironmentName"))
    #expect(source.contains("if policy.isLowMemory {"))
    #expect(source.contains("startupMemoryPolicy?.clearAllocatorCacheAfterWarmup == true"))
}

@Test
func startupMemoryPolicyKeepsRanked128GiBProfile() {
    let policy = RuntimeStartupMemoryPolicy.resolve(
        physicalMemoryBytes: UInt64(128) << 30
    )

    #expect(!policy.isLowMemory)
    #expect(policy.cacheLimitBytes == 32 << 30)
    #expect(policy.maxMegabytesPerCommandBuffer == 320)
    #expect(policy.maxOperationsPerCommandBuffer == 128)
    #expect(!policy.clearAllocatorCacheAfterWarmup)
    #expect(policy.environmentOverrides.isEmpty)

    // The ranked path must stay silent and write no feature defaults.
    let plan = policy.environmentPlan { _ in nil }
    #expect(plan.defaultsToApply.isEmpty)
    #expect(plan.preservedUserValues.isEmpty)
    #expect(plan.noticeLines.isEmpty)
}

@Test
func startupMemoryPolicyUses64GiBAsFullProfileBoundary() {
    #expect(
        RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: (UInt64(64) << 30) - 1
        ).isLowMemory
    )
    #expect(
        !RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(64) << 30
        ).isLowMemory
    )
}

@Test
func startupMemoryPolicyHonorsExplicitProfileRequest() {
    #expect(
        !RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(36) << 30,
            requestedProfile: "full"
        ).isLowMemory
    )
    #expect(
        RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(128) << 30,
            requestedProfile: "low"
        ).isLowMemory
    )
    #expect(
        !RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(36) << 30,
            requestedProfile: "FULL"
        ).isLowMemory
    )
    #expect(
        RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(36) << 30,
            requestedProfile: "auto"
        ).isLowMemory
    )
    #expect(
        RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(36) << 30,
            requestedProfile: ""
        ).isLowMemory
    )
    // The override must keep its DARKBLOOM_ prefix: the trusted worker
    // environment allowlist forwards that family, so renaming it (e.g. to
    // MLXFAST_*) would silently stop it from ever reaching the worker.
    #expect(
        RuntimeStartupMemoryPolicy.profileOverrideEnvironmentName
            == "DARKBLOOM_STARTUP_MEMORY_PROFILE"
    )
}

// The low-memory plan writes no environment defaults -- compiled decode and
// every other ranked code path stay enabled -- and its stderr notice states
// exactly what it does (cap + warmup clear), that ranked code paths stay
// enabled, and that a genuinely-too-small machine fails loudly with an
// out-of-memory error instead of silently skipping ranked code paths (the
// silent divergence the old feature-disable defaults caused).
@Test
func lowMemoryPlanWritesNoFeatureDefaultsAndAnnouncesRankedParity() throws {
    let policy = RuntimeStartupMemoryPolicy.resolve(
        physicalMemoryBytes: UInt64(36) << 30
    )
    // Even with the historical compiled-decode flag names exported, the plan
    // neither writes nor tracks them: the profile no longer touches feature
    // flags at all.
    let plan = policy.environmentPlan { name in
        ["MLX_COMPILED_DECODE", "DARKBLOOM_COMPILED_DECODE"].contains(name)
            ? "1" : nil
    }

    #expect(plan.defaultsToApply.isEmpty)
    #expect(plan.preservedUserValues.isEmpty)
    try #require(plan.noticeLines.count == 2)
    #expect(plan.noticeLines[0].contains("low-memory startup profile active"))
    #expect(plan.noticeLines[0].contains("capping the MLX allocator cache at 6 GiB"))
    #expect(plan.noticeLines[0].contains(
        "compiled decode and every other ranked code path stay enabled"
    ))
    #expect(plan.noticeLines[0].contains("DARKBLOOM_STARTUP_MEMORY_PROFILE=full"))
    #expect(plan.noticeLines[1].contains("out-of-memory"))
    #expect(plan.noticeLines[1].contains("rely on the ranked run"))
    // Deliberately NOT asserting the machine-size phrasing in this line.
    // RuntimeStartupMemoryPolicy.swift is an editable path, so a submission
    // ships its own copy of this notice string. Pinning cosmetic guidance
    // wording here fails every submission whose snapshot predates a wording
    // change in trusted main, for no contract reason -- observed 2026-07-28,
    // when rewording "64 GiB+ machine" to "machine with more unified memory"
    // reddened in-flight submissions cb418c4c and others. The assertions that
    // remain (out-of-memory guidance, ranked-run fallback) are stable across
    // both wordings, and the noticeLines[0] assertions above are kept because
    // they encode the actual contract (no feature-disable defaults; compiled
    // decode stays enabled), not phrasing.
}

@Test
func startupMemoryPolicyIsAppliedBeforeModelLoadAndCleansWarmupCache() throws {
    let source = try String(
        contentsOfFile: "Sources/MLXFastModel/LagunaRuntimeWeights.swift",
        encoding: .utf8
    )
    // Scope every assertion to the initializer body so this pins statement
    // order inside init, not merely textual order anywhere in the file.
    let initBody = try initializerBody(of: source)

    let resolve = try #require(
        initBody.range(of: "RuntimeStartupMemoryPolicy.resolve(")
    )
    let apply = try #require(initBody.range(of: "policy.apply()"))
    let load = try #require(initBody.range(of: "loadLibraryModel("))
    #expect(resolve.lowerBound < apply.lowerBound)
    #expect(apply.lowerBound < load.lowerBound)
    // The explicit profile request must come from the worker-reachable
    // DARKBLOOM_ override name, not a hardcoded or harness-only variable.
    #expect(
        initBody.contains("RuntimeStartupMemoryPolicy.profileOverrideEnvironmentName")
    )

    let warmup = try #require(initBody.range(of: "Self.warmLibraryModel(model)"))
    let cleanup = try #require(
        initBody.range(
            of: "if startupMemoryPolicy?.clearAllocatorCacheAfterWarmup == true"
        )
    )
    let clear = try #require(
        initBody.range(of: "Memory.clearCache()", range: cleanup.lowerBound..<initBody.endIndex)
    )
    #expect(warmup.lowerBound < cleanup.lowerBound)
    #expect(cleanup.lowerBound < clear.lowerBound)
}

private func initializerBody(of source: String) throws -> String {
    let start = try #require(
        source.range(of: "public init(loader: LagunaWeightLoader, config: LagunaConfig)")
    )
    let end = try #require(
        source.range(
            of: "private static func warmLibraryModel",
            range: start.upperBound..<source.endIndex
        )
    )
    return String(source[start.lowerBound..<end.lowerBound])
}
