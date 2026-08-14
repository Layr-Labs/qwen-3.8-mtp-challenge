import Foundation
import MLXFastModel
import Testing

// The DFlash block-decode worker holds a STRICTLY LARGER resident set than the
// serial worker: the ~21.6 GB pinned target plus the separately-loaded drafter,
// warmed across every legal block width at a seed past the sliding-window ring.
// It reaches that residency through LLMModelFactory, so it never constructs
// Qwen35RuntimeWeightCache -- which was the documented low-memory startup
// policy's only call site. These tests pin the wiring that closed that gap.
//
// Scope note: no MLX device work, no weights, no hidden material. Two of the
// three tests are source-level, and the third drives the pure `resolve()`
// function directly.

private let dflashWorkerTrustedPath =
    "Sources/MLXFastTrustedHarness/QwenRuntimeDFlashWorker.swift"
private let dflashWorkerParticipantPath =
    "Sources/MLXFastHarness/QwenRuntimeDFlashWorker.swift"
private let dflashDriverTrustedPath =
    "Sources/MLXFastTrustedHarness/QwenRuntimeDFlashDriver.swift"
private let dflashDriverParticipantPath =
    "Sources/MLXFastHarness/QwenRuntimeDFlashDriver.swift"

private let trustedHarnessGuardOpen = "#if !MLXFAST_TRUSTED_HARNESS\n"
private let trustedHarnessGuardClose = "#endif\n"

private func sourceText(_ path: String) throws -> String {
    try String(contentsOfFile: path, encoding: .utf8)
}

// Sources/MLXFastTrustedHarness/X.swift and Sources/MLXFastHarness/X.swift are
// TWINS: the same file compiled into the trusted harness target and into the
// participant runtime-worker target. Any edit that lands in one and not the
// other silently changes what the worker runs relative to what the trusted
// build audits. The DFlash worker twin's ONLY legitimate difference is the
// `#if !MLXFAST_TRUSTED_HARNESS` wrapper that compiles the whole body out of
// the trusted target (which links no MLX and no MLXFastModel).
@Test
func dflashWorkerTwinsDifferOnlyByTheTrustedHarnessGuard() throws {
    let trusted = try sourceText(dflashWorkerTrustedPath)
    let participant = try sourceText(dflashWorkerParticipantPath)

    // The exact byte relationship, which is stronger than a line-by-line
    // comparison: the trusted copy is the participant copy wrapped in the
    // guard, with nothing else added, removed, or reordered.
    #expect(
        trusted
            == trustedHarnessGuardOpen + participant + trustedHarnessGuardClose,
        """
        DFlash worker twins diverged beyond the #if !MLXFAST_TRUSTED_HARNESS \
        guard; edits must land in BOTH files with identical content
        """
    )
    // Stated as a line count too, because the guard delta is exactly two lines
    // and a size drift is the cheapest signal that an edit landed in one twin.
    let trustedLines = trusted.split(separator: "\n", omittingEmptySubsequences: false)
    let participantLines = participant.split(
        separator: "\n",
        omittingEmptySubsequences: false
    )
    #expect(trustedLines.count == participantLines.count + 2)
    #expect(trustedLines.first == "#if !MLXFAST_TRUSTED_HARNESS")
}

// The DFlash driver twin carries NO guard at all: it is byte-identical across
// the two targets. Pinned here because the startup-policy wiring sits next to
// it, and a change that spills from the worker into the driver must land
// identically in both copies.
@Test
func dflashDriverTwinsAreByteIdentical() throws {
    let trusted = try sourceText(dflashDriverTrustedPath)
    let participant = try sourceText(dflashDriverParticipantPath)
    #expect(
        trusted == participant,
        "DFlash driver twins must stay byte-identical"
    )
}

// The wiring contract: the DFlash worker resolves the startup memory policy
// from the worker-reachable DARKBLOOM_ override, applies it when the low
// profile engages, and does BOTH before either model load -- an allocator cap
// installed after the ~21.6 GB target is already resident protects nothing.
// The post-warmup clear must follow the block-width warm and precede the
// protocol hello, matching the policy's documented behavior on the serial path.
@Test
func dflashWorkerAppliesStartupMemoryPolicyBeforeItsModelLoads() throws {
    for path in [dflashWorkerTrustedPath, dflashWorkerParticipantPath] {
        let body = try dflashWorkerStartupBody(of: try sourceText(path))

        let resolve = try #require(
            body.range(of: "RuntimeStartupMemoryPolicy.resolve("),
            "\(path) never resolves the startup memory policy"
        )
        // The application is INLINE in the trusted worker rather than a call to the
        // policy's own `apply()`: that method is internal and must stay internal,
        // because its file ships inside every submission (see
        // DFlashTrustedEditableCouplingTests). Anchor on the allocator cap, which
        // is the load-bearing effect -- an installed cap is what protects the box.
        let apply = try #require(
            body.range(of: "Memory.cacheLimit = resolvedStartupMemoryPolicy.cacheLimitBytes"),
            "\(path) resolves the startup memory policy but never installs its cap"
        )
        let targetLoad = try #require(
            body.range(of: "LLMModelFactory.shared.load("),
            "\(path) no longer loads the target through the vendored factory"
        )
        let drafterLoad = try #require(
            body.range(of: "DFlashDraftModel.load("),
            "\(path) no longer loads the DFlash drafter"
        )
        #expect(resolve.lowerBound < apply.lowerBound, "\(path)")
        // Before BOTH residencies: the target AND the separate drafter.
        #expect(apply.lowerBound < targetLoad.lowerBound, "\(path)")
        #expect(apply.lowerBound < drafterLoad.lowerBound, "\(path)")

        // The profile request must come from the DARKBLOOM_ override name the
        // trusted worker environment allowlist actually forwards, not a
        // hardcoded value or a harness-only variable the worker never sees.
        #expect(
            body.contains(
                "RuntimeStartupMemoryPolicy.profileOverrideEnvironmentName"
            ),
            "\(path)"
        )

        let warm = try #require(
            body.range(of: "warmAllBlockWidths()"),
            "\(path) no longer warms the block widths before the hello"
        )
        let cleanup = try #require(
            body.range(
                of: "startupMemoryPolicy?.clearAllocatorCacheAfterWarmup == true"
            ),
            "\(path) ignores clearAllocatorCacheAfterWarmup"
        )
        let clear = try #require(
            body.range(
                of: "Memory.clearCache()",
                range: cleanup.lowerBound..<body.endIndex
            ),
            "\(path) guards the warmup clear but never performs it"
        )
        let hello = try #require(
            body.range(of: "try protocolIO.writeLine("),
            "\(path) no longer writes the protocol hello"
        )
        #expect(warm.lowerBound < cleanup.lowerBound, "\(path)")
        #expect(cleanup.lowerBound < clear.lowerBound, "\(path)")
        // Before the hello, so the freed bytes are never held across it.
        #expect(clear.lowerBound < hello.lowerBound, "\(path)")
    }
}

// Scoped to runExperimentalDFlashWorker so the assertions above pin statement
// ORDER inside the startup path, not merely textual order anywhere in the file.
private func dflashWorkerStartupBody(of source: String) throws -> String {
    let start = try #require(
        source.range(of: "public static func runExperimentalDFlashWorker(")
    )
    let end = try #require(
        source.range(
            of: "static func handleExperimentalDFlashWorkerRequest(",
            range: start.upperBound..<source.endIndex
        )
    )
    return String(source[start.lowerBound..<end.lowerBound])
}

// The policy the DFlash worker now resolves is the SAME pure function the
// serial weight cache resolves -- no DFlash-specific threshold and no second
// policy. RuntimeStartupMemoryPolicyTests covers the serial call site and the
// notice text; this covers the inputs the DFlash worker actually passes
// (`ProcessInfo.physicalMemory` and the DARKBLOOM_ override), including the
// override precedence that lets the ranked 128 GiB box and an explicit local
// opt-out both win.
@Test
func dflashResidentSetGetsTheDocumentedLowMemoryProfileBelow64GiB() {
    // Below the 64 GiB full-profile minimum: the DFlash residency (target plus
    // drafter) is exactly the case the cap exists for.
    let small = RuntimeStartupMemoryPolicy.resolve(
        physicalMemoryBytes: UInt64(36) << 30
    )
    #expect(small.isLowMemory)
    #expect(small.cacheLimitBytes == 6 << 30)
    #expect(small.clearAllocatorCacheAfterWarmup)

    // At and above the boundary, and on the ranked box, the profile is a
    // deliberate no-op so the DFlash worker keeps the stock allocator behavior
    // the pinned on-box baseline was measured with.
    #expect(
        !RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(64) << 30
        ).isLowMemory
    )
    let ranked = RuntimeStartupMemoryPolicy.resolve(
        physicalMemoryBytes: UInt64(128) << 30
    )
    #expect(!ranked.isLowMemory)
    #expect(!ranked.clearAllocatorCacheAfterWarmup)

    // DARKBLOOM_STARTUP_MEMORY_PROFILE wins over the measured memory in both
    // directions; auto (and unset) falls back to the measurement.
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
        RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(36) << 30,
            requestedProfile: "auto"
        ).isLowMemory
    )
    #expect(
        !RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: UInt64(128) << 30,
            requestedProfile: nil
        ).isLowMemory
    )

    // The profile is pure memory management: it must never default a feature
    // flag, on the DFlash path any more than on the serial one, or a <64 GiB
    // machine silently stops executing code the ranked box runs.
    #expect(small.environmentOverrides.isEmpty)
    #expect(ranked.environmentOverrides.isEmpty)
}

// MARK: - Trusted code must not depend on an editable access level

/// `Sources/MLXFastModel` is a DIRECTORY entry in both tracks' `editablePaths`, so
/// `submit` packages every file under it with every submission — including
/// `RuntimeStartupMemoryPolicy.swift`, unmodified. If trusted, non-editable code
/// depends on a declaration's access level there, then any submission overlaying an
/// older copy fails to build the package through no fault of the submitter. That
/// would have broken every serial submission in flight when this change promoted.
///
/// This pins the shape of the fix so it cannot be undone by a future "just make it
/// public" edit, which is exactly what the original implementation did.
@Suite
struct DFlashTrustedEditableCouplingTests {
    private static let policyPath = "Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift"
    private static let workerPaths = [
        "Sources/MLXFastTrustedHarness/QwenRuntimeDFlashWorker.swift",
        "Sources/MLXFastHarness/QwenRuntimeDFlashWorker.swift",
    ]

    @Test
    func startupPolicyApplyStaysInternal() throws {
        let src = try String(contentsOfFile: Self.policyPath, encoding: .utf8)
        #expect(src.contains("    func apply() {"))
        #expect(
            !src.contains("public func apply()"),
            "apply() must stay internal: this file ships inside every submission"
        )
    }

    @Test
    func trustedWorkerDoesNotCallTheEditableApply() throws {
        for path in Self.workerPaths {
            let src = try String(contentsOfFile: path, encoding: .utf8)
            #expect(
                !src.contains("resolvedStartupMemoryPolicy.apply()"),
                "trusted worker must apply the policy from long-public members"
            )
            // It must still actually apply the policy — the fix is not "stop doing it".
            #expect(src.contains("Memory.cacheLimit = resolvedStartupMemoryPolicy.cacheLimitBytes"))
            #expect(src.contains("resolvedStartupMemoryPolicy.environmentOverrides"))
            #expect(src.contains("MLX_MAX_MB_PER_BUFFER"))
            #expect(src.contains("MLX_MAX_OPS_PER_BUFFER"))
        }
    }
}
