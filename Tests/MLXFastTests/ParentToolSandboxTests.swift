import Foundation
import Testing

// The `transform` and `attach-gpqa-gates` command paths run the
// submission-built binary directly on the ranked box (they do not spawn the
// separately sandboxed runtime worker), so they must self-confine behind a
// Seatbelt profile before reading the reference checkpoint or the raw hidden
// golden / GPQA answer key. These tests pin that wiring in the CLI and the
// workflow so a refactor cannot silently drop the parent-tool sandbox.
@Suite
struct ParentToolSandboxTests {
    private func cliSource() throws -> String {
        try String(contentsOfFile: "Sources/MLXFastCLI/main.swift", encoding: .utf8)
    }

    private func workflowSource() throws -> String {
        try String(contentsOfFile: ".github/workflows/dflash-benchmark.yml", encoding: .utf8)
    }

    @Test
    func transformAndAttachRouteThroughTheParentToolSandbox() throws {
        let cli = try cliSource()

        // Both untrusted-parent subcommands invoke the re-exec guard as their
        // first action, before any option validation or file access.
        let transformRange = try #require(cli.range(of: "private static func runTransform("))
        let transformBody = String(cli[transformRange.lowerBound...].prefix(400))
        #expect(transformBody.contains(
            "try reexecUnderParentToolSandboxIfRequested(subcommand: \"transform\")"
        ))

        let attachRange = try #require(cli.range(of: "private static func runAttachGPQAGates("))
        let attachBody = String(cli[attachRange.lowerBound...].prefix(400))
        #expect(attachBody.contains(
            "try reexecUnderParentToolSandboxIfRequested(subcommand: \"attach-gpqa-gates\")"
        ))
    }

    @Test
    func parentToolSandboxIsArmedFailClosedAndReentryGuarded() throws {
        let cli = try cliSource()

        // Armed by the trusted workflow trigger or the official-run marker.
        #expect(cli.contains(
            "environmentValue(\"MLXFAST_SANDBOX_PARENT_TOOLS\", fallback: \"0\") == \"1\""
        ))
        #expect(cli.contains(
            "let officialRun = environmentValue(\"MLXFAST_OFFICIAL_BENCHMARK_RUN\", fallback: \"0\") == \"1\""
        ))
        // Re-exec sets a sentinel so the sandboxed child does not recurse.
        #expect(cli.contains(
            "environmentValue(\"MLXFAST_PARENT_SANDBOX_ACTIVE\", fallback: \"0\") == \"1\""
        ))
        #expect(cli.contains("setenv(\"MLXFAST_PARENT_SANDBOX_ACTIVE\", \"1\", 1)"))
        // Fail closed: an armed context with the sandbox disabled or missing
        // aborts instead of running the submission-built binary unconfined.
        #expect(cli.contains("requires the parent-tool sandbox; unset MLXFAST_NO_SANDBOX"))
        #expect(cli.contains("requires sandbox-exec for the parent-tool sandbox"))
        #expect(cli.contains("execv"))
    }

    @Test
    func parentToolSandboxProfileDeniesNetworkForkExecAndResolver() throws {
        let cli = try cliSource()
        let profileRange = try #require(
            cli.range(of: "private static func writeParentToolSandboxProfile(")
        )
        let profileBody = String(cli[profileRange.lowerBound...].prefix(1200))

        #expect(profileBody.contains("(deny network*)"))
        #expect(profileBody.contains("(deny process-fork)"))
        #expect(profileBody.contains("(deny process-exec*)"))
        #expect(profileBody.contains("(allow process-exec (literal"))
        // Coordinated with the operator worker profile's mDNSResponder deny.
        // All three resolver names are pinned: the exact service name, the
        // legacy system alias, and the prefix form that covers relaunched
        // (".reloaded"-style) resolver registrations.
        #expect(profileBody.contains(
            "(deny mach-lookup (global-name \"com.apple.mDNSResponder\"))"
        ))
        #expect(profileBody.contains(
            "(deny mach-lookup (global-name \"com.apple.system.mDNSResponder\"))"
        ))
        #expect(profileBody.contains(
            "(deny mach-lookup (global-name-prefix \"com.apple.mDNSResponder\"))"
        ))
    }

    @Test
    func runtimeWorkerFallbackProfileDeniesResolverMachLookup() throws {
        let cli = try cliSource()
        // Scope the assertions to the runtime-worker profile builder's own body
        // (up to its `try profile.write`), so a match against the parent-tool
        // profile cannot satisfy this test -- it must be the FALLBACK worker
        // profile that carries the resolver denies.
        let profileRange = try #require(
            cli.range(of: "private static func writeRuntimeWorkerSandboxProfile(")
        )
        let profileTail = String(cli[profileRange.lowerBound...])
        let writeRange = try #require(profileTail.range(of: "try profile.write("))
        let profileBody = String(profileTail[..<writeRange.lowerBound])

        // Pre-existing worker-profile guarantees remain, proving the denies were
        // added to this profile without disturbing its structure.
        #expect(profileBody.contains("(deny network*)"))
        #expect(profileBody.contains("(deny process-fork)"))
        #expect(profileBody.contains("(deny process-exec*)"))
        #expect(profileBody.contains("(deny file-write*)"))
        // All three mDNSResponder mach-lookup deny variants, matching the
        // parent-tool profile (#494) and the operator-layer worker profile:
        // getaddrinfo(3) egresses through mDNSResponder from ITS own uid, so a
        // uid/socket-scoped block never sees the DNS query -- the mach-lookup
        // deny is what actually closes the DNS side channel for this profile.
        #expect(profileBody.contains(
            "(deny mach-lookup (global-name \"com.apple.mDNSResponder\"))"
        ))
        #expect(profileBody.contains(
            "(deny mach-lookup (global-name \"com.apple.system.mDNSResponder\"))"
        ))
        #expect(profileBody.contains(
            "(deny mach-lookup (global-name-prefix \"com.apple.mDNSResponder\"))"
        ))
    }


    @Test
    func runtimeWorkerAllowlistDropsSSHAgentSocket() throws {
        let worker = try String(
            contentsOfFile: "Sources/MLXFastHarness/QwenRuntimeWorker.swift",
            encoding: .utf8
        )
        let allowlistRange = try #require(
            worker.range(of: "let allowedExactKeys: Set<String> = [")
        )
        let allowlistEnd = try #require(
            worker.range(of: "]", range: allowlistRange.upperBound..<worker.endIndex)
        )
        let allowlist = String(worker[allowlistRange.lowerBound..<allowlistEnd.upperBound])
        #expect(!allowlist.contains("SSH_AUTH_SOCK"))
    }
}
