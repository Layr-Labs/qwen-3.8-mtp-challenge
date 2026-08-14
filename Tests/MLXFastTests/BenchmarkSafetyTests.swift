import Darwin
import CryptoKit
import Foundation
@testable import MLXFastHarness
import Testing

@Suite(.serialized)
struct BenchmarkSafetyTests {
    @Test
    func directoryDigestPreservesLegacyDigestWithBoundedUncachedReads() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let nested = root.appendingPathComponent("nested", isDirectory: true)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try Data([0, 1, 2, 3, 255]).write(to: nested.appendingPathComponent("b.bin"))
        try Data("alpha\n".utf8).write(to: root.appendingPathComponent("a.txt"))
        try Data("ignored".utf8).write(to: root.appendingPathComponent(".gitkeep"))

        let digest = try QwenRuntime.directoryDigest(
            rootPath: root.path,
            ignoredRelativePaths: [".gitkeep"]
        )

        // Golden from the pre-change recipe: sorted relative path + NUL +
        // raw per-file SHA-256 + NUL. The read strategy must not alter it.
        #expect(
            digest == QwenRuntime.DirectoryDigest(
                fileCount: 2,
                byteCount: 11,
                sha256: "bc1b1f56a33b786645d24771234b988228e7b97266ba74337fc2329ea7101134"
            )
        )

        let source = try String(
            contentsOfFile: "Sources/MLXFastHarness/QwenRuntimePreflight.swift",
            encoding: .utf8
        )
        let noCache = try #require(
            source.range(of: "Darwin.fcntl(handle.fileDescriptor, F_NOCACHE, 1)")
        )
        let read = try #require(source.range(of: "handle.readData(ofLength: chunkSize)"))
        #expect(noCache.lowerBound < read.lowerBound)
        #expect(source.contains("let reachedEOF = autoreleasepool {"))
    }

    @Test
    func localThermalGateRunsAtEveryPhaseBoundary() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let helper = root.appendingPathComponent("cool-gate-helper.sh")
        let log = root.appendingPathComponent("phases.log")
        try """
        #!/bin/sh
        printf '%s\\n' "$MLXFAST_LOCAL_COOL_GATE_PHASE" >> "$MLXFAST_COOL_GATE_TEST_LOG"
        """.write(to: helper, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: helper.path
        )

        let previousHelper = ProcessInfo.processInfo.environment["MLXFAST_LOCAL_COOL_GATE_HELPER"]
        let previousLog = ProcessInfo.processInfo.environment["MLXFAST_COOL_GATE_TEST_LOG"]
        setenv("MLXFAST_LOCAL_COOL_GATE_HELPER", helper.path, 1)
        setenv("MLXFAST_COOL_GATE_TEST_LOG", log.path, 1)
        defer {
            restoreEnvironment("MLXFAST_LOCAL_COOL_GATE_HELPER", value: previousHelper)
            restoreEnvironment("MLXFAST_COOL_GATE_TEST_LOG", value: previousLog)
        }

        try QwenRuntime.runLocalPhaseCoolGate(phase: "prefill")
        try QwenRuntime.runLocalPhaseCoolGate(phase: "decode")

        #expect(try String(contentsOf: log, encoding: .utf8) == "prefill\ndecode\n")
    }

    @Test
    func benchmarkScriptPreservesCompactJSONIntegrityMetrics() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        try """
        #!/bin/sh
        printf '%s\\n' '{"score":null,"passed":true,"metrics":{"weights_hash":"compact-hash","weights_file_count":7,"weights_byte_count":11}}'
        """.write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: ["MLXFAST_SKIP_TRANSFORM": "1"]
        )

        #expect(result.status == 0)
        let object = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: fixture.integrity))
                as? [String: Any]
        )
        #expect(object["weights_sha256"] as? String == "compact-hash")
        #expect(object["weights_file_count"] as? Int == 7)
        #expect(object["weights_byte_count"] as? Int == 11)
    }

    @Test(arguments: [".", ".."])
    func benchmarkScriptRejectsWeightsPathsThatCanEraseWorkspace(weightsPath: String) throws {
        let fixture = try makeBenchmarkScriptFixture(nestedWorkingDirectory: true)
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let sentinel = fixture.root.appendingPathComponent("do-not-delete.txt")
        try "sentinel".write(to: sentinel, atomically: true, encoding: .utf8)
        let invocation = fixture.root.appendingPathComponent("swift-invoked")
        try """
        #!/bin/sh
        touch "\(invocation.path)"
        exit 99
        """.write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_WEIGHTS_PATH": weightsPath,
                "MLXFAST_FORCE_TRANSFORM": "1",
            ]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("refusing"))
        #expect(FileManager.default.fileExists(atPath: sentinel.path))
        #expect(!FileManager.default.fileExists(atPath: invocation.path))
    }

    @Test
    func benchmarkScriptRejectsTrackedSourcesAsWeightsPath() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        try initializeGitIndex(at: fixture.workingDirectory)
        let trackedSource = fixture.workingDirectory
            .appendingPathComponent("Sources/MLXFastCore/Fixture.swift")
        let originalSource = try Data(contentsOf: trackedSource)
        let invocation = fixture.root.appendingPathComponent("swift-invoked")
        try "#!/bin/sh\ntouch \"\(invocation.path)\"\nexit 99\n".write(
            to: fixture.swift,
            atomically: true,
            encoding: .utf8
        )
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_WEIGHTS_PATH": trackedSource.deletingLastPathComponent().path,
                "MLXFAST_FORCE_TRANSFORM": "1",
            ]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("contains tracked file Sources/MLXFastCore/Fixture.swift"))
        #expect(try Data(contentsOf: trackedSource) == originalSource)
        #expect(!FileManager.default.fileExists(atPath: invocation.path))
    }

    @Test
    func benchmarkScriptPreservesUnmanagedExternalWeightsDirectory() throws {
        let fixture = try makeBenchmarkScriptFixture(nestedWorkingDirectory: true)
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let sentinel = fixture.weights.appendingPathComponent("do-not-delete.txt")
        try "sentinel".write(to: sentinel, atomically: true, encoding: .utf8)
        let invocation = fixture.root.appendingPathComponent("swift-invoked")
        try "#!/bin/sh\ntouch \"\(invocation.path)\"\nexit 99\n".write(
            to: fixture.swift,
            atomically: true,
            encoding: .utf8
        )
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: ["MLXFAST_FORCE_TRANSFORM": "1"]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("existing directories outside the workspace must be MLXFast-managed"))
        #expect(try String(contentsOf: sentinel, encoding: .utf8) == "sentinel")
        #expect(!FileManager.default.fileExists(atPath: invocation.path))
    }

    @Test
    func benchmarkScriptCanInstallIntoEmptyExternalWeightsDirectory() throws {
        let fixture = try makeBenchmarkScriptFixture(nestedWorkingDirectory: true)
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        try FileManager.default.removeItem(
            at: fixture.weights.appendingPathComponent("config.json")
        )
        try """
        #!/bin/sh
        set -eu
        if [ "${1:-}" = transform ]; then
          shift
          output=""
          while [ "$#" -gt 0 ]; do
            if [ "$1" = --output ]; then
              output="$2"
              break
            fi
            shift
          done
          test -n "$output"
          mkdir -p "$output"
          printf '%s\n' '{"transformed":true}' > "$output/config.json"
          printf '%s\n' '{"weight_map":{"tensor":"model.safetensors"}}' > "$output/model.safetensors.index.json"
          printf '\\100\\000\\000\\000\\000\\000\\000\\000' > "$output/model.safetensors"
          printf '%s' '{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}      ' >> "$output/model.safetensors"
          printf '\\001' >> "$output/model.safetensors"
          exit 0
        fi
        printf '%s\n' '{"score":null,"passed":true,"metrics":{"weights_hash":"hash","weights_file_count":1,"weights_byte_count":12}}'
        """.write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: ["MLXFAST_FORCE_TRANSFORM": "1"]
        )

        #expect(result.status == 0)
        #expect(
            try String(
                contentsOf: fixture.weights.appendingPathComponent("config.json"),
                encoding: .utf8
            ) == "{\"transformed\":true}\n"
        )
        #expect(
            FileManager.default.fileExists(
                atPath: fixture.weights.appendingPathComponent(".benchmark-source.sha256").path
            )
        )
    }

    @Test
    func benchmarkScriptRejectsReservedTransformMetadataSymlink() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let sentinel = fixture.root.appendingPathComponent("do-not-touch.txt")
        try "sentinel".write(to: sentinel, atomically: true, encoding: .utf8)
        try """
        #!/bin/sh
        set -eu
        if [ "${1:-}" = transform ]; then
          shift
          output=""
          while [ "$#" -gt 0 ]; do
            if [ "$1" = --output ]; then
              output="$2"
              break
            fi
            shift
          done
          test -n "$output"
          mkdir -p "$output"
          printf '%s\n' transformed > "$output/config.json"
          ln -s "\(sentinel.path)" "$output/.benchmark-source.sha256"
          exit 0
        fi
        exit 99
        """.write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: ["MLXFAST_FORCE_TRANSFORM": "1"]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("created reserved .benchmark-source.sha256 path"))
        #expect(try String(contentsOf: sentinel, encoding: .utf8) == "sentinel")
    }

    @Test
    func benchmarkScriptPreservesWeightsWhenTransformOutputIsIncomplete() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let originalConfig = try Data(
            contentsOf: fixture.weights.appendingPathComponent("config.json")
        )
        try """
        #!/bin/sh
        set -eu
        shift
        output=""
        while [ "$#" -gt 0 ]; do
          if [ "$1" = --output ]; then
            output="$2"
            break
          fi
          shift
        done
        test -n "$output"
        mkdir -p "$output"
        printf '%s\n' incomplete > "$output/config.json"
        """.write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: ["MLXFAST_FORCE_TRANSFORM": "1"]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("missing regular config/index files"))
        #expect(
            try Data(contentsOf: fixture.weights.appendingPathComponent("config.json"))
                == originalConfig
        )
    }

    @Test
    func benchmarkScriptValidatesStagedJSONIndexAndSafetensorsBeforeReplacement() throws {
        let cases = [
            ("malformed-config", "config.json is not a JSON object"),
            ("missing-shard", "index references missing shard"),
            ("invalid-shard", "safetensors shard is too small"),
            ("unsupported-dtype", "safetensors shard has an invalid header"),
            ("byte-span-mismatch", "safetensors shard has an invalid header"),
            ("overlapping-ranges", "safetensors shard has an invalid header"),
        ]
        for (caseName, expectedError) in cases {
            let fixture = try makeBenchmarkScriptFixture()
            defer { try? FileManager.default.removeItem(at: fixture.root) }
            let originalConfig = try Data(
                contentsOf: fixture.weights.appendingPathComponent("config.json")
            )
            try """
            #!/bin/sh
            set -eu
            shift
            output=""
            while [ "$#" -gt 0 ]; do
              if [ "$1" = --output ]; then
                output="$2"
                break
              fi
              shift
            done
            test -n "$output"
            mkdir -p "$output"
            printf '%s\n' '{}' > "$output/config.json"
            printf '%s\n' '{"weight_map":{"tensor":"model.safetensors"}}' > "$output/model.safetensors.index.json"
            printf '\\100\\000\\000\\000\\000\\000\\000\\000' > "$output/model.safetensors"
            printf '%s' '{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}      ' >> "$output/model.safetensors"
            printf '\\001' >> "$output/model.safetensors"
            case "$MLXFAST_TEST_BAD_TRANSFORM" in
              malformed-config)
                printf '%s\n' 'not-json' > "$output/config.json"
                ;;
              missing-shard)
                printf '%s\n' '{"weight_map":{"tensor":"missing.safetensors"}}' > "$output/model.safetensors.index.json"
                ;;
              invalid-shard)
                printf '%s\n' invalid > "$output/model.safetensors"
                ;;
              unsupported-dtype)
                printf '\\100\\000\\000\\000\\000\\000\\000\\000' > "$output/model.safetensors"
                printf '%s' '{"tensor":{"dtype":"XX","shape":[1],"data_offsets":[0,1]}}      ' >> "$output/model.safetensors"
                printf '\\001' >> "$output/model.safetensors"
                ;;
              byte-span-mismatch)
                printf '\\100\\000\\000\\000\\000\\000\\000\\000' > "$output/model.safetensors"
                printf '%s' '{"tensor":{"dtype":"U8","shape":[2],"data_offsets":[0,1]}}      ' >> "$output/model.safetensors"
                printf '\\001' >> "$output/model.safetensors"
                ;;
              overlapping-ranges)
                printf '%s\n' '{"weight_map":{"a":"model.safetensors","b":"model.safetensors"}}' > "$output/model.safetensors.index.json"
                printf '\\160\\000\\000\\000\\000\\000\\000\\000' > "$output/model.safetensors"
                printf '%s' '{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}       ' >> "$output/model.safetensors"
                printf '\\001' >> "$output/model.safetensors"
                ;;
            esac
            """.write(to: fixture.swift, atomically: true, encoding: .utf8)
            try makeExecutable(fixture.swift)

            let result = try runBenchmarkScript(
                fixture: fixture,
                arguments: ["--local-iterate"],
                environment: [
                    "MLXFAST_FORCE_TRANSFORM": "1",
                    "MLXFAST_TEST_BAD_TRANSFORM": caseName,
                ]
            )

            #expect(result.status != 0, "case \(caseName)")
            #expect(result.stderr.contains(expectedError), "case \(caseName)")
            #expect(
                try Data(contentsOf: fixture.weights.appendingPathComponent("config.json"))
                    == originalConfig,
                "case \(caseName)"
            )
        }
    }

    @Test
    func benchmarkScriptRejectsReservedTransformRollbackSymlink() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let sentinelDirectory = fixture.root.appendingPathComponent("rollback-target")
        try FileManager.default.createDirectory(
            at: sentinelDirectory,
            withIntermediateDirectories: true
        )
        let sentinel = sentinelDirectory.appendingPathComponent("do-not-touch.txt")
        try "sentinel".write(to: sentinel, atomically: true, encoding: .utf8)
        let originalConfig = try Data(
            contentsOf: fixture.weights.appendingPathComponent("config.json")
        )
        try """
        #!/bin/sh
        set -eu
        shift
        output=""
        while [ "$#" -gt 0 ]; do
          if [ "$1" = --output ]; then
            output="$2"
            break
          fi
          shift
        done
        test -n "$output"
        mkdir -p "$output"
        printf '%s\n' '{"transformed":true}' > "$output/config.json"
        printf '%s\n' '{"weight_map":{"tensor":"model.safetensors"}}' > "$output/model.safetensors.index.json"
        printf '\\100\\000\\000\\000\\000\\000\\000\\000' > "$output/model.safetensors"
        printf '%s' '{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}      ' >> "$output/model.safetensors"
        printf '\\001' >> "$output/model.safetensors"
        ln -s "\(sentinelDirectory.path)" "$output/../previous-weights"
        """.write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: ["MLXFAST_FORCE_TRANSFORM": "1"]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("created reserved rollback path"))
        #expect(try String(contentsOf: sentinel, encoding: .utf8) == "sentinel")
        #expect(
            try Data(contentsOf: fixture.weights.appendingPathComponent("config.json"))
                == originalConfig
        )
    }

    @Test
    func benchmarkScriptRejectsTrackedScorePath() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        try initializeGitIndex(at: fixture.workingDirectory)
        let trackedSource = fixture.workingDirectory
            .appendingPathComponent("Sources/MLXFastCore/Fixture.swift")
        let originalSource = try Data(contentsOf: trackedSource)
        let invocation = fixture.root.appendingPathComponent("swift-invoked")
        try "#!/bin/sh\ntouch \"\(invocation.path)\"\nexit 99\n".write(
            to: fixture.swift,
            atomically: true,
            encoding: .utf8
        )
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_SCORE_PATH": trackedSource.path,
                "MLXFAST_SKIP_TRANSFORM": "1",
            ]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("would overwrite tracked file Sources/MLXFastCore/Fixture.swift"))
        #expect(try Data(contentsOf: trackedSource) == originalSource)
        #expect(!FileManager.default.fileExists(atPath: invocation.path))
    }

    @Test
    func benchmarkScriptRejectsWorkspaceAsVerificationTemporaryDirectory() throws {
        let fixture = try makeBenchmarkScriptFixture(nestedWorkingDirectory: true)
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let sentinel = fixture.workingDirectory.appendingPathComponent("do-not-delete.txt")
        try "sentinel".write(to: sentinel, atomically: true, encoding: .utf8)
        try "#!/bin/sh\nexit 99\n".write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_SKIP_TRANSFORM": "1",
                "MLXFAST_VERIFY_TRANSFORM": "1",
                "MLXFAST_VERIFY_TRANSFORM_TMP_PARENT": ".",
            ]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("refusing to clear unsafe transform verification temporary path"))
        #expect(FileManager.default.fileExists(atPath: sentinel.path))
    }

    @Test
    func benchmarkScriptPreservesVerificationTemporaryRootContents() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let scratchRoot = fixture.root.appendingPathComponent("verification-root")
        try FileManager.default.createDirectory(at: scratchRoot, withIntermediateDirectories: true)
        let sentinel = scratchRoot.appendingPathComponent("unrelated.txt")
        try "sentinel".write(to: sentinel, atomically: true, encoding: .utf8)
        try """
        #!/bin/sh
        if [ "${1:-}" = verify-transform ]; then
          test -d "${7:-}"
          exit 0
        fi
        printf '%s\n' '{"score":null,"passed":true,"metrics":{"weights_hash":"hash","weights_file_count":1,"weights_byte_count":12}}'
        """.write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_SKIP_TRANSFORM": "1",
                "MLXFAST_VERIFY_TRANSFORM": "1",
                "MLXFAST_VERIFY_TRANSFORM_TMP_PARENT": scratchRoot.path,
            ]
        )

        #expect(result.status == 0)
        #expect(try String(contentsOf: sentinel, encoding: .utf8) == "sentinel")
        #expect(try FileManager.default.contentsOfDirectory(atPath: scratchRoot.path) == ["unrelated.txt"])
    }

    @Test
    func artifactStagingRejectsTraversalWithoutDeletingDestination() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let artifactRoot = root.appendingPathComponent("mlxfast-artifacts-run")
        let victim = root.appendingPathComponent("victim")
        try FileManager.default.createDirectory(at: artifactRoot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: victim, withIntermediateDirectories: true)
        let sentinel = victim.appendingPathComponent("do-not-delete.txt")
        try "sentinel".write(to: sentinel, atomically: true, encoding: .utf8)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".github/scripts/stage-benchmark-artifacts.sh").path,
            artifactRoot.appendingPathComponent("../victim").path,
            sentinel.path,
        ]
        process.environment = ProcessInfo.processInfo.environment.merging([
            "RUNNER_TEMP": root.path,
        ]) { _, new in new }
        let stderr = Pipe()
        process.standardError = stderr
        try process.run()
        let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()

        #expect(process.terminationStatus != 0)
        #expect((String(data: stderrData, encoding: .utf8) ?? "").contains("artifact destination"))
        #expect(try String(contentsOf: sentinel, encoding: .utf8) == "sentinel")
    }

    @Test
    func artifactStagingRejectsSymlinkedRunRoot() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let outside = root.appendingPathComponent("outside")
        let upload = outside.appendingPathComponent("benchmark-results")
        try FileManager.default.createDirectory(at: upload, withIntermediateDirectories: true)
        let sentinel = upload.appendingPathComponent("do-not-delete.txt")
        try "sentinel".write(to: sentinel, atomically: true, encoding: .utf8)
        let runRoot = root.appendingPathComponent("mlxfast-artifacts-run")
        try FileManager.default.createSymbolicLink(at: runRoot, withDestinationURL: outside)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".github/scripts/stage-benchmark-artifacts.sh").path,
            runRoot.appendingPathComponent("benchmark-results").path,
            "sentinel.txt=\(sentinel.path)",
        ]
        process.environment = ProcessInfo.processInfo.environment.merging([
            "RUNNER_TEMP": root.path,
        ]) { _, new in new }
        let stderr = Pipe()
        process.standardError = stderr
        try process.run()
        let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()

        #expect(process.terminationStatus != 0)
        #expect((String(data: stderrData, encoding: .utf8) ?? "").contains("must not be a symlink"))
        #expect(try String(contentsOf: sentinel, encoding: .utf8) == "sentinel")
    }

    @Test
    func artifactStagingCreatesOwnedRunRootAndCopiesFiles() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("score-source.json")
        try "{}".write(to: source, atomically: true, encoding: .utf8)
        let destination = root
            .appendingPathComponent("mlxfast-artifacts-run")
            .appendingPathComponent("benchmark-results")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".github/scripts/stage-benchmark-artifacts.sh").path,
            destination.path,
            "score.json=\(source.path)",
        ]
        process.environment = ProcessInfo.processInfo.environment.merging([
            "RUNNER_TEMP": root.path,
        ]) { _, new in new }
        try process.run()
        process.waitUntilExit()

        #expect(process.terminationStatus == 0)
        #expect(
            try String(
                contentsOf: destination.appendingPathComponent("score.json"),
                encoding: .utf8
            ) == "{}"
        )
    }

    // A worker crash at the public behavior gate happens before any
    // score.json exists. It must surface as an explicit public-gate
    // category (not the opaque "no_score" that invites consumers to guess
    // the failing gate from the surrounding skipped step names, e.g. the
    // GPQA steps), and the worker-controlled stderr embedded in the gate
    // report's error string must never reach the uploaded artifact.
    @Test
    func redactedBenchmarkFailureCategorizesPublicGateWorkerCrashWithoutQuotingWorkerStderr() throws {
        let workspace = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: workspace) }
        try #"""
        {"actual_token":null,"case_count":1,"checked_steps":0,"error":"runtime worker closed stdout before returning a response: exit_status=5 stderr=WORKER-CONTROLLED-SENTINEL Fatal error: [reshape] Cannot reshape array","expected_token":null,"first_failing_case":null,"first_failing_step":null,"golden_hash":"b9509697","passed":false}
        """#.write(
            to: workspace.appendingPathComponent("public-gate-report.json"),
            atomically: true,
            encoding: .utf8
        )

        let result = try runRedactBenchmarkFailureScript(
            workspace: workspace,
            environment: ["MLXFAST_BENCHMARK_MODE": "single-machine"]
        )

        #expect(result.status == 0)
        #expect(result.artifact.contains("\"failure_category\": \"public_gate_worker_crash\""))
        #expect(result.artifact.contains("\"public_gate_passed\": false"))
        #expect(result.artifact.contains("\"mode\": \"single-machine\""))
        #expect(!result.artifact.contains("WORKER-CONTROLLED-SENTINEL"))
        #expect(!result.artifact.contains("reshape"))
    }

    @Test
    func redactedBenchmarkFailureCategorizesPublicGateTokenMismatchWithoutTokenValues() throws {
        let workspace = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: workspace) }
        try #"""
        {"passed":false,"error":"teacher-forced token mismatch","checked_steps":12,"first_failing_step":12,"expected_token":314159,"actual_token":271828}
        """#.write(
            to: workspace.appendingPathComponent("public-gate-report.json"),
            atomically: true,
            encoding: .utf8
        )

        let result = try runRedactBenchmarkFailureScript(workspace: workspace)

        #expect(result.status == 0)
        #expect(result.artifact.contains("\"failure_category\": \"public_gate_token_mismatch\""))
        #expect(result.artifact.contains("\"public_gate_passed\": false"))
        #expect(!result.artifact.contains("314159"))
        #expect(!result.artifact.contains("271828"))
    }

    @Test
    func redactedBenchmarkFailureCollapsesUnknownPublicGateErrorToOpaqueCategory() throws {
        let workspace = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: workspace) }
        try #"""
        {"passed":false,"error":"UNRECOGNIZED-SENTINEL hidden golden token 99999"}
        """#.write(
            to: workspace.appendingPathComponent("public-gate-report.json"),
            atomically: true,
            encoding: .utf8
        )

        let result = try runRedactBenchmarkFailureScript(workspace: workspace)

        #expect(result.status == 0)
        #expect(result.artifact.contains("\"failure_category\": \"public_gate_failed\""))
        #expect(!result.artifact.contains("UNRECOGNIZED-SENTINEL"))
        #expect(!result.artifact.contains("99999"))
    }

    // Without a failing gate report the pre-change behavior is preserved:
    // a passing report keeps "no_score" while recording the gate verdict, a
    // non-boolean (attacker-shaped) verdict is coerced to null instead of
    // riding --argjson into the artifact, and a missing report emits null.
    @Test
    func redactedBenchmarkFailureKeepsNoScoreUnlessPublicGateReportShowsFailure() throws {
        let workspace = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: workspace) }
        let report = workspace.appendingPathComponent("public-gate-report.json")

        try #"{"passed":true,"error":""}"#.write(to: report, atomically: true, encoding: .utf8)
        let gatePassed = try runRedactBenchmarkFailureScript(workspace: workspace)
        #expect(gatePassed.status == 0)
        #expect(gatePassed.artifact.contains("\"failure_category\": \"no_score\""))
        #expect(gatePassed.artifact.contains("\"public_gate_passed\": true"))

        try #"{"passed":"EXFIL-STRING","error":"x"}"#.write(
            to: report,
            atomically: true,
            encoding: .utf8
        )
        let coerced = try runRedactBenchmarkFailureScript(workspace: workspace)
        #expect(coerced.status == 0)
        #expect(coerced.artifact.contains("\"failure_category\": \"no_score\""))
        #expect(coerced.artifact.contains("\"public_gate_passed\": null"))
        #expect(!coerced.artifact.contains("EXFIL-STRING"))

        try FileManager.default.removeItem(at: report)
        let missing = try runRedactBenchmarkFailureScript(workspace: workspace)
        #expect(missing.status == 0)
        #expect(missing.artifact.contains("\"failure_category\": \"no_score\""))
        #expect(missing.artifact.contains("\"public_gate_passed\": null"))
    }

    @Test
    func redactedBenchmarkFailurePrefersScoreDerivedCategoryOverPublicGateReport() throws {
        let workspace = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: workspace) }
        try #"""
        {"passed":false,"metrics":{"error":"performance floor failed: decode","passed_correctness":true,"passed_decode_speedup_floor":false,"passed_prefill_speedup_floor":true,"partial_result":false,"benchmark_wall_seconds":100.5,"timed_benchmark_seconds":60.1}}
        """#.write(
            to: workspace.appendingPathComponent("score.json"),
            atomically: true,
            encoding: .utf8
        )
        try #"{"passed":true,"error":""}"#.write(
            to: workspace.appendingPathComponent("public-gate-report.json"),
            atomically: true,
            encoding: .utf8
        )

        let result = try runRedactBenchmarkFailureScript(workspace: workspace)

        #expect(result.status == 0)
        #expect(result.artifact.contains("\"failure_category\": \"floor_failed\""))
        #expect(result.artifact.contains("\"public_gate_passed\": true"))
        #expect(result.artifact.contains("\"passed_decode_speedup_floor\": false"))
    }

    /// The acceptance band is evaluated before correctness runs, so the band
    /// path writes passed_correctness=false with correctness never executed.
    /// Reporting that to a participant as "correctness_failed" sent them
    /// hunting a nonexistent numerics bug when the real (and actionable)
    /// verdict was "your win is too large for one submission -- chunk it".
    @Test
    func redactedBenchmarkFailureReportsAcceptanceBandSeparatelyFromCorrectness() throws {
        let workspace = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: workspace) }
        try #"""
        {"passed":false,"metrics":{"error":"acceptance band failed: decode 0.0111 below -5.0% of reference 0.0124 (< 0.0118): improvement too large for one submission (chunk it) or a suspiciously lucky reading","passed_correctness":false,"passed_decode_speedup_floor":true,"passed_prefill_speedup_floor":true,"partial_result":false,"benchmark_wall_seconds":100.5,"timed_benchmark_seconds":60.1}}
        """#.write(
            to: workspace.appendingPathComponent("score.json"),
            atomically: true,
            encoding: .utf8
        )

        let result = try runRedactBenchmarkFailureScript(workspace: workspace)

        #expect(result.status == 0)
        #expect(result.artifact.contains("\"failure_category\": \"acceptance_band_failed\""))
        #expect(!result.artifact.contains("correctness_failed"))
        // The raw reason still never leaves the runner.
        #expect(!result.artifact.contains("chunk it"))
    }

    @Test
    func offlineRunnerRemovesProfilesAndPreservesCommandFailure() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let bin = root.appendingPathComponent("bin")
        let temporaryDirectory = root.appendingPathComponent("tmp")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: true
        )
        let curl = bin.appendingPathComponent("curl")
        try "#!/bin/sh\nexit 99\n".write(to: curl, atomically: true, encoding: .utf8)
        try makeExecutable(curl)
        let sandboxExec = bin.appendingPathComponent("sandbox-exec")
        try """
        #!/bin/sh
        set -eu
        test "$1" = -f
        shift 2
        case "$1" in
          */curl) exit 1 ;;
        esac
        exec "$@"
        """.write(to: sandboxExec, atomically: true, encoding: .utf8)
        try makeExecutable(sandboxExec)
        let command = root.appendingPathComponent("command.sh")
        try "#!/bin/sh\nexit 37\n".write(to: command, atomically: true, encoding: .utf8)
        try makeExecutable(command)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".github/scripts/run-offline.sh").path,
            command.path,
        ]
        process.environment = ProcessInfo.processInfo.environment.merging([
            "PATH": "\(bin.path):/usr/bin:/bin",
            "TMPDIR": temporaryDirectory.path,
        ]) { _, new in new }
        try process.run()
        process.waitUntilExit()

        #expect(process.terminationStatus == 37)
        let leftovers = try FileManager.default.contentsOfDirectory(
            atPath: temporaryDirectory.path
        )
        #expect(!leftovers.contains { $0.hasPrefix("mlxfast-offline.") })
    }

    @Test
    func offlineRunnerForwardsTerminationAndRemovesProfiles() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let bin = root.appendingPathComponent("bin")
        let temporaryDirectory = root.appendingPathComponent("tmp")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: true
        )
        let curl = bin.appendingPathComponent("curl")
        try "#!/bin/sh\nexit 99\n".write(to: curl, atomically: true, encoding: .utf8)
        try makeExecutable(curl)
        let sandboxExec = bin.appendingPathComponent("sandbox-exec")
        try """
        #!/bin/sh
        set -eu
        test "$1" = -f
        shift 2
        case "$1" in
          */curl) exit 1 ;;
        esac
        exec "$@"
        """.write(to: sandboxExec, atomically: true, encoding: .utf8)
        try makeExecutable(sandboxExec)
        let childPIDPath = root.appendingPathComponent("child.pid")
        let terminatedPath = root.appendingPathComponent("child-terminated")
        let command = root.appendingPathComponent("command.sh")
        try """
        #!/bin/sh
        trap 'printf terminated > "\(terminatedPath.path)"; exit 0' TERM
        printf '%s\n' "$$" > "\(childPIDPath.path)"
        while :; do :; done
        """.write(to: command, atomically: true, encoding: .utf8)
        try makeExecutable(command)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".github/scripts/run-offline.sh").path,
            command.path,
        ]
        process.environment = ProcessInfo.processInfo.environment.merging([
            "PATH": "\(bin.path):/usr/bin:/bin",
            "TMPDIR": temporaryDirectory.path,
        ]) { _, new in new }
        try process.run()
        for _ in 0..<200 where !FileManager.default.fileExists(atPath: childPIDPath.path) {
            Thread.sleep(forTimeInterval: 0.01)
        }
        let childPID = try #require(
            Int32(String(contentsOf: childPIDPath, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines))
        )

        process.terminate()
        process.waitUntilExit()

        #expect(FileManager.default.fileExists(atPath: terminatedPath.path))
        #expect(kill(childPID, 0) != 0)
        let leftovers = try FileManager.default.contentsOfDirectory(
            atPath: temporaryDirectory.path
        )
        #expect(!leftovers.contains { $0.hasPrefix("mlxfast-offline.") })
    }

    @Test
    func pairedTimingOverlayRejectsWeightsChangedAfterGates() throws {
        let result = try runPairedTimingOverlay(
            gatesHarnessHash: String(repeating: "c", count: 64),
            candidateHarnessHash: String(repeating: "c", count: 64),
            gatesWeightsHash: String(repeating: "a", count: 64),
            candidateWeightsHash: String(repeating: "b", count: 64)
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("candidate timing score failed the pre-merge checks"))
    }

    @Test
    func pairedTimingOverlayRejectsHarnessChangedAfterGates() throws {
        let weightsHash = String(repeating: "c", count: 64)
        let result = try runPairedTimingOverlay(
            gatesHarnessHash: String(repeating: "a", count: 64),
            candidateHarnessHash: String(repeating: "b", count: 64),
            gatesWeightsHash: weightsHash,
            candidateWeightsHash: weightsHash
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("candidate timing score failed the pre-merge checks"))
    }

    @Test
    func pairedTimingOverlayAcceptsMatchingHarnessAndWeightsIdentity() throws {
        let harnessHash = String(repeating: "c", count: 64)
        let weightsHash = String(repeating: "d", count: 64)
        let result = try runPairedTimingOverlay(
            gatesHarnessHash: harnessHash,
            candidateHarnessHash: harnessHash,
            gatesWeightsHash: weightsHash,
            candidateWeightsHash: weightsHash
        )

        #expect(result.status == 0)
        #expect(result.score?["passed"] as? Bool == true)
        #expect(result.score?["score"] as? Double == 1.0)
    }

    @Test
    func finalArtifactValidatorEnforcesTimingAndIntegrityContracts() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let score = root.appendingPathComponent("score.json")
        let integrity = root.appendingPathComponent("benchmark-integrity.json")
        let golden = root.appendingPathComponent("correctness_golden.json")
        let goldenHash = String(repeating: "a", count: 64)
        let weightsHash = String(repeating: "b", count: 64)
        let harnessHash = String(repeating: "c", count: 64)
        let commit = String(repeating: "d", count: 40)

        let metrics: [String: Any] = [
            "actual_token": NSNull(), "bandwidth_gb_per_token": 0,
            "bandwidth_source": "ram_resident_model",
            "baseline_decode_seconds_per_token": 0.00936,
            "baseline_prefill_seconds_per_token": 0.000361, "benchmark_wall_seconds": 3,
            "case_count": 1, "checked_steps": 64, "commit": commit,
            "correctness_seconds": 1, "decode_seconds_per_token": 0.00936,
            "decode_speedup": 1,
            "decode_speedup_floor": NSDecimalNumber(string: "0.95"), "error": "",
            "expected_token": NSNull(), "expert_bytes_read": 0,
            "expert_cache_evictions": 0, "expert_cache_hits": 0,
            "expert_cache_misses": 0, "expert_hit_rate": 0,
            "expert_peak_cached_tensors": 0, "expert_read_seconds": 0,
            "first_failing_case": NSNull(), "first_failing_layer": NSNull(),
            "first_failing_step": NSNull(), "golden_hash": goldenHash,
            "gpqa_ttft_case_count": 1, "gpqa_ttft_max_seconds": 1,
            "gpqa_ttft_p50_seconds": 1, "gpqa_ttft_pass_count": 1,
            "gpqa_ttft_passed": true, "gpqa_ttft_seconds": 1,
            "gpqa_ttft_source": "hidden_gpqa_first_token", "harness_hash": harnessHash,
            "max_abs_diff": 0, "num_layers": 40, "partial_result": false,
            "passed_correctness": true, "peak_ram_gb": 1,
            "passed_decode_speedup_floor": true, "passed_prefill_speedup_floor": true,
            "prefill_seconds_per_token": 0.000361, "prefill_speedup": 1,
            "prefill_speedup_floor": NSDecimalNumber(string: "0.95"), "preflight_seconds": 1,
            "process_resident_memory_gb": 1, "runtime": "swift",
            "semantic_gpqa_case_count": 1, "semantic_gpqa_model": "fixture",
            "semantic_gpqa_pass_count": 1, "semantic_gpqa_passed": true,
            "timed_benchmark_seconds": 1, "timestamp": "2026-01-01T00:00:00Z",
            "weights_byte_count": 1, "weights_file_count": 1,
            "weights_hash": weightsHash,
        ]
        try "{}".write(to: golden, atomically: true, encoding: .utf8)
        try "\(goldenHash)  correctness_golden.json\n".write(
            to: URL(fileURLWithPath: golden.path + ".sha256"), atomically: true, encoding: .utf8
        )
        try "2\n".write(
            to: URL(fileURLWithPath: golden.path + ".bytes"), atomically: true, encoding: .utf8
        )

        func writeScoreArtifacts(
            metrics inputMetrics: [String: Any],
            scientificTiming: (field: String, literal: String)? = nil
        ) throws -> [String: Any] {
            var serializedMetrics = inputMetrics
            let scientificMarker = "__MLXFAST_SCIENTIFIC_TIMING__"
            if let scientificTiming {
                serializedMetrics[scientificTiming.field] = scientificMarker
            }

            var scoreData = try JSONSerialization.data(withJSONObject: [
                "metrics": serializedMetrics, "passed": true, "score": 1,
            ])
            if let scientificTiming {
                let quotedMarker = "\"\(scientificMarker)\""
                guard var scoreJSON = String(data: scoreData, encoding: .utf8),
                      scoreJSON.contains(quotedMarker)
                else {
                    throw NSError(
                        domain: "BenchmarkSafetyTests.scientificTiming",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "scientific timing marker missing"]
                    )
                }
                scoreJSON = scoreJSON.replacingOccurrences(
                    of: quotedMarker,
                    with: scientificTiming.literal
                )
                scoreData = Data(scoreJSON.utf8)
            }

            try scoreData.write(to: score)
            let scoreHash = SHA256.hash(data: scoreData)
                .map { String(format: "%02x", $0) }.joined()
            try "\(scoreHash)  score.json\n".write(
                to: URL(fileURLWithPath: score.path + ".sha256"),
                atomically: true,
                encoding: .utf8
            )
            let generatedIntegrity: [String: Any] = [
                "golden_path": "[private]", "golden_sha256": goldenHash,
                "score_path": "score.json", "score_sha256": scoreHash,
                "transform_source_sha256": String(repeating: "e", count: 64),
                "weights_byte_count": 1, "weights_file_count": 1,
                "weights_path": "weights", "weights_sha256": weightsHash,
            ]
            try JSONSerialization.data(withJSONObject: generatedIntegrity).write(to: integrity)
            return generatedIntegrity
        }

        _ = try writeScoreArtifacts(metrics: metrics)
        let accepted = try runFinalArtifactValidator(root: root, golden: golden, commit: commit)
        #expect(accepted.status == 0, Comment(rawValue: accepted.stderr))

        let validatorSource = try String(
            contentsOfFile: ".github/scripts/validate-benchmark-artifacts.sh",
            encoding: .utf8
        )
        #expect(validatorSource.contains(
            "MAX_PLAUSIBLE_SPEEDUP=\"${MLXFAST_MAX_PLAUSIBLE_SPEEDUP:-5.0}\""
        ))
        #expect(validatorSource.contains(
            "MIN_DECODE_SECONDS_PER_TOKEN=\"${MLXFAST_MIN_DECODE_SECONDS_PER_TOKEN:-0.001}\""
        ))
        #expect(validatorSource.contains(
            "MIN_PREFILL_SECONDS_PER_TOKEN=\"${MLXFAST_MIN_PREFILL_SECONDS_PER_TOKEN:-0.00004}\""
        ))
        #expect(validatorSource.contains(
            "mirror /opt/bench-runner/measure-job.sh exactly"
        ))
        // The measured Laguna baseline values remain comfortably above the
        // recalibrated defaults, including when JSON spells a timing in
        // scientific notation.
        _ = try writeScoreArtifacts(
            metrics: metrics,
            scientificTiming: ("decode_seconds_per_token", "9.36e-3")
        )
        let scientificLegitimate = try runFinalArtifactValidator(
            root: root,
            golden: golden,
            commit: commit
        )
        #expect(
            scientificLegitimate.status == 0,
            Comment(rawValue: scientificLegitimate.stderr)
        )

        // Both candidate and paired-baseline timings fail below either
        // default floor.
        let belowFloorCases: [(field: String, value: Double)] = [
            ("decode_seconds_per_token", 0.000999),
            ("baseline_decode_seconds_per_token", 0.000999),
            ("prefill_seconds_per_token", 0.000039),
            ("baseline_prefill_seconds_per_token", 0.000039),
        ]
        for testCase in belowFloorCases {
            var tooFast = metrics
            tooFast[testCase.field] = testCase.value
            _ = try writeScoreArtifacts(metrics: tooFast)
            let rejected = try runFinalArtifactValidator(
                root: root,
                golden: golden,
                commit: commit
            )
            #expect(rejected.status != 0, "validator accepted \(testCase.field) below its floor")
            #expect(rejected.stderr.contains("scored timing failed the plausibility ceiling"))
        }

        for testCase in [
            (field: "decode_seconds_per_token", literal: "9e-4"),
            (field: "prefill_seconds_per_token", literal: "3e-5"),
        ] {
            _ = try writeScoreArtifacts(
                metrics: metrics,
                scientificTiming: (testCase.field, testCase.literal)
            )
            let rejected = try runFinalArtifactValidator(
                root: root,
                golden: golden,
                commit: commit
            )
            #expect(
                rejected.status != 0,
                "validator accepted scientific-notation \(testCase.field) below its floor"
            )
        }

        // Overrides remain operator-tunable plain decimals only. Values that
        // jq could otherwise parse (scientific notation or a negative number)
        // must fail closed before comparison.
        for invalidOverride in [
            (name: "MLXFAST_MAX_PLAUSIBLE_SPEEDUP", value: "5e0"),
            (name: "MLXFAST_MIN_DECODE_SECONDS_PER_TOKEN", value: "1e-3"),
            (name: "MLXFAST_MIN_PREFILL_SECONDS_PER_TOKEN", value: "-0.00004"),
        ] {
            _ = try writeScoreArtifacts(metrics: metrics)
            let rejected = try runFinalArtifactValidator(
                root: root,
                golden: golden,
                commit: commit,
                plausibilityBounds: [invalidOverride.name: invalidOverride.value]
            )
            #expect(rejected.status != 0)
            #expect(rejected.stderr.contains(
                "plausibility bound override must be a plain non-negative decimal"
            ))
        }

        let baseIntegrity = try writeScoreArtifacts(metrics: metrics)
        let wrongLayerCount = try runFinalArtifactValidator(
            root: root,
            golden: golden,
            commit: commit,
            expectedNumLayers: "60"
        )
        #expect(wrongLayerCount.status != 0)

        let missingLayerCount = try runFinalArtifactValidator(
            root: root,
            golden: golden,
            commit: commit,
            expectedNumLayers: nil
        )
        #expect(missingLayerCount.status != 0)
        #expect(missingLayerCount.stderr.contains("MLXFAST_EXPECTED_NUM_LAYERS is required"))

        for invalidLayerCount in ["forty", "0"] {
            let invalid = try runFinalArtifactValidator(
                root: root,
                golden: golden,
                commit: commit,
                expectedNumLayers: invalidLayerCount
            )
            #expect(invalid.status != 0)
            #expect(invalid.stderr.contains("MLXFAST_EXPECTED_NUM_LAYERS must be a positive integer"))
        }

        let mismatches: [(String, Any)] = [
            ("weights_sha256", String(repeating: "f", count: 64)),
            ("weights_file_count", 2),
            ("weights_byte_count", 2),
            ("golden_sha256", String(repeating: "f", count: 64)),
        ]
        for (field, value) in mismatches {
            var changed = baseIntegrity
            changed[field] = value
            try JSONSerialization.data(withJSONObject: changed).write(to: integrity)
            #expect(
                try runFinalArtifactValidator(root: root, golden: golden, commit: commit).status != 0,
                "validator accepted mismatched \(field)"
            )
        }
    }

    @Test
    func benchmarkScriptStagesTransformInItsOnlySandboxWritableDirectory() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let scripts = fixture.workingDirectory.appendingPathComponent(".github/scripts")
        try FileManager.default.createDirectory(at: scripts, withIntermediateDirectories: true)
        let sandboxLog = fixture.root.appendingPathComponent("sandbox-writable-path.txt")
        let transformOutputLog = fixture.root.appendingPathComponent("transform-output-path.txt")
        let runtimeProfile = fixture.root.appendingPathComponent("runtime-worker.sb")
        try "(version 1)\n(allow default)\n".write(
            to: runtimeProfile,
            atomically: true,
            encoding: .utf8
        )

        let runOffline = scripts.appendingPathComponent("run-offline.sh")
        try """
        #!/bin/sh
        set -eu
        printf '%s\n' "${MLXFAST_OFFLINE_WRITABLE_PATHS:-}" > "${MLXFAST_SANDBOX_LOG:?}"
        exec "$@"
        """.write(to: runOffline, atomically: true, encoding: .utf8)
        try makeExecutable(runOffline)

        try """
        #!/bin/sh
        set -eu
        if [ "${1:-}" = transform ]; then
          shift
          output=""
          while [ "$#" -gt 0 ]; do
            case "$1" in
              --output)
                output="$2"
                shift 2
                ;;
              *)
                shift
                ;;
            esac
          done
          test -n "$output"
          printf '%s\n' "$output" > "$MLXFAST_TRANSFORM_OUTPUT_LOG"
          mkdir -p "$output"
          printf '%s\n' '{"transformed":true}' > "$output/config.json"
          printf '%s\n' '{"weight_map":{"tensor":"model.safetensors"}}' > "$output/model.safetensors.index.json"
          printf '\\100\\000\\000\\000\\000\\000\\000\\000' > "$output/model.safetensors"
          printf '%s' '{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}      ' >> "$output/model.safetensors"
          printf '\\001' >> "$output/model.safetensors"
          exit 0
        fi
        printf '%s\n' '{"score":null,"passed":true,"metrics":{"weights_hash":"hash","weights_file_count":1,"weights_byte_count":12}}'
        """.write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_FORCE_TRANSFORM": "1",
                "MLXFAST_NO_SANDBOX": "0",
                "MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE": runtimeProfile.path,
                "MLXFAST_SANDBOX_LOG": sandboxLog.path,
                "MLXFAST_TRANSFORM_OUTPUT_LOG": transformOutputLog.path,
            ]
        )

        #expect(result.status == 0)
        let writablePath = try String(contentsOf: sandboxLog, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let transformOutputPath = try String(contentsOf: transformOutputLog, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let writableURL = URL(fileURLWithPath: writablePath)
        let writableParent = canonicalExistingPath(
            writableURL.deletingLastPathComponent().path
        )
        let expectedParent = canonicalExistingPath(
            fixture.weights.deletingLastPathComponent().path
        )
        #expect(writableParent == expectedParent)
        #expect(writableURL.lastPathComponent.hasPrefix(".weights.mlxfast-transform."))
        #expect(transformOutputPath == writableURL.appendingPathComponent("weights").path)
        #expect(!FileManager.default.fileExists(atPath: writablePath))
        #expect(
            try String(
                contentsOf: fixture.weights.appendingPathComponent("config.json"),
                encoding: .utf8
            ) == "{\"transformed\":true}\n"
        )
    }

    @Test
    func benchmarkScriptPropagatesUnsandboxedTransformVerificationFailure() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let benchmarkInvocation = fixture.root.appendingPathComponent("benchmark-invoked")
        try """
        #!/bin/sh
        if [ "${1:-}" = verify-transform ]; then
          exit 37
        fi
        touch "\(benchmarkInvocation.path)"
        exit 99
        """.write(to: fixture.swift, atomically: true, encoding: .utf8)
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_SKIP_TRANSFORM": "1",
                "MLXFAST_VERIFY_TRANSFORM": "1",
            ]
        )

        #expect(result.status == 37)
        #expect(!FileManager.default.fileExists(atPath: benchmarkInvocation.path))
    }

    @Test
    func benchmarkScriptRejectsScoreInsideSymlinkedReferenceDirectory() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let referenceAlias = fixture.root.appendingPathComponent("reference-alias")
        try FileManager.default.createSymbolicLink(
            at: referenceAlias,
            withDestinationURL: fixture.reference
        )
        let config = fixture.reference.appendingPathComponent("config.json")
        let original = try Data(contentsOf: config)
        let invocation = fixture.root.appendingPathComponent("swift-invoked")
        try "#!/bin/sh\ntouch \"\(invocation.path)\"\nexit 99\n".write(
            to: fixture.swift,
            atomically: true,
            encoding: .utf8
        )
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_REFERENCE_DIR": referenceAlias.path,
                "MLXFAST_SCORE_PATH": config.path,
                "MLXFAST_SKIP_TRANSFORM": "1",
            ]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("overlaps protected path"))
        #expect(try Data(contentsOf: config) == original)
        #expect(!FileManager.default.fileExists(atPath: invocation.path))
    }

    @Test
    func benchmarkScriptFailsWhenTransformSourceGitInspectionFails() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let fakeBin = fixture.root.appendingPathComponent("bin")
        try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)
        let fakeGit = fakeBin.appendingPathComponent("git")
        try """
        #!/bin/sh
        case " $* " in
          *" rev-parse --is-inside-work-tree "*) exit 0 ;;
          *" --cached --others --exclude-standard "*) exit 19 ;;
          *) exit 0 ;;
        esac
        """.write(to: fakeGit, atomically: true, encoding: .utf8)
        try makeExecutable(fakeGit)
        let invocation = fixture.root.appendingPathComponent("swift-invoked")
        try "#!/bin/sh\ntouch \"\(invocation.path)\"\nexit 99\n".write(
            to: fixture.swift,
            atomically: true,
            encoding: .utf8
        )
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_SKIP_TRANSFORM": "1",
                "PATH": "\(fakeBin.path):/usr/bin:/bin",
            ]
        )

        #expect(result.status == 19, "stderr: \(result.stderr)")
        #expect(!FileManager.default.fileExists(atPath: invocation.path))
    }

    @Test
    func benchmarkScriptRejectsConflictingLocalModesBeforeInvokingSwift() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let invocation = fixture.root.appendingPathComponent("swift-invoked")
        try "#!/bin/sh\ntouch \"\(invocation.path)\"\n".write(
            to: fixture.swift,
            atomically: true,
            encoding: .utf8
        )
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate", "--local-submit"]
        )

        #expect(result.status != 0)
        #expect(result.stderr.contains("--local-iterate and --local-submit cannot be used together"))
        #expect(!FileManager.default.fileExists(atPath: invocation.path))
    }

    @Test
    func benchmarkScriptCleansOwnedTemporaryFilesWithoutMaskingFailureStatus() throws {
        let fixture = try makeBenchmarkScriptFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let temporaryDirectory = fixture.root.appendingPathComponent("tmp")
        let fakeBin = fixture.root.appendingPathComponent("bin")
        try FileManager.default.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)
        let sandboxExec = fakeBin.appendingPathComponent("sandbox-exec")
        try "#!/bin/sh\nexit 99\n".write(
            to: sandboxExec,
            atomically: true,
            encoding: .utf8
        )
        try makeExecutable(sandboxExec)
        try "#!/bin/sh\nexit 37\n".write(
            to: fixture.swift,
            atomically: true,
            encoding: .utf8
        )
        try makeExecutable(fixture.swift)

        let result = try runBenchmarkScript(
            fixture: fixture,
            arguments: ["--local-iterate"],
            environment: [
                "MLXFAST_SKIP_TRANSFORM": "1",
                "MLXFAST_NO_SANDBOX": "0",
                "PATH": "\(fakeBin.path):/usr/bin:/bin",
                "TMPDIR": temporaryDirectory.path,
            ]
        )

        #expect(result.status == 37)
        let leftovers = try FileManager.default.contentsOfDirectory(atPath: temporaryDirectory.path)
        #expect(!leftovers.contains { $0.hasPrefix("mlxfast-runtime-worker.") })
        #expect(!leftovers.contains { $0.hasPrefix("mlxfast-score.") })
    }

    @Test
    func scoredWorkerResponsesExcludeDiagnostics() throws {
        let worker = try String(
            contentsOfFile: "Sources/MLXFastHarness/QwenRuntimeWorker.swift",
            encoding: .utf8
        )
        let benchmark = try String(
            contentsOfFile: "Sources/MLXFastHarness/QwenRuntimeBenchmark.swift",
            encoding: .utf8
        )
        let local = try String(
            contentsOfFile: "Sources/MLXFastHarness/QwenRuntimeLocalIterate.swift",
            encoding: .utf8
        )
        let support = try String(
            contentsOfFile: "Sources/MLXFastHarness/QwenRuntimeSupport.swift",
            encoding: .utf8
        )
        let decodeStep = try sourceSlice(worker, from: "case \"decode_step\":", to: "case \"phase_diagnostics\":")
        #expect(!decodeStep.contains("currentResidentMemoryGB"))
        #expect(!decodeStep.contains("expertStats"))
        #expect(!worker.contains("let seconds: Double?"))
        #expect(worker.contains("case \"phase_diagnostics\":"))
        #expect(!worker.contains("peakRamGB: currentResidentMemoryGB()"))

        let measuredDecode = try sourceSlice(
            benchmark,
            from: "static func measureWorkerDecode(",
            to: "static let bandwidthSource"
        )
        let timerEnd = try #require(measuredDecode.range(of: "let measuredSeconds = secondsSince(decodePhaseStart)"))
        let diagnostics = try #require(measuredDecode.range(of: "worker.phaseDiagnostics()"))
        #expect(timerEnd.lowerBound < diagnostics.lowerBound)
        let diagnosticResponse = try sourceSlice(
            worker,
            from: "case \"phase_diagnostics\":",
            to: "default:"
        )
        #expect(diagnosticResponse.contains("peakResidentMemoryGB()"))
        #expect(!diagnosticResponse.contains("currentResidentMemoryGB()"))
        #expect(support.contains("Double(info.resident_size_max)"))

        let localMeasuredTime = try sourceSlice(
            local,
            from: "let timingWallSeconds = secondsSince(timingWallStart)",
            to: "correctnessReport = timing.correctness"
        )
        #expect(localMeasuredTime.contains("timedSeconds = timing.prefillSecondsPerToken"))
        #expect(localMeasuredTime.contains("localCase.promptTokens.count * options.timingRepeats"))
        #expect(localMeasuredTime.contains("timing.decode.secondsPerToken"))
        #expect(localMeasuredTime.contains("options.benchmarkDecodeSteps * options.timingRepeats"))
        #expect(localMeasuredTime.contains("correctnessSeconds = timedSeconds"))
        #expect(!localMeasuredTime.contains("timedSeconds = secondsSince"))
        #expect(!local.contains("let warmupCache = Gemma4ModelCache"))
        #expect(local.components(separatedBy: "runLocalPhaseCoolGate(phase:").count == 5)
    }

    // The hidden-GPQA capture embeds hidden prompt text, answer keys, and
    // reference answers. It must never exist on disk while the correctness
    // worker -- the only process that runs editable model code in this phase --
    // is alive. runLayeredCorrectnessWithWorker therefore only COLLECTS and
    // validates the answers and returns them in its result; the trusted
    // benchmark caller writes them only AFTER closing/reaping the worker. This
    // is a source-ordering guard because the exposure requires a live worker
    // plus real weights to reproduce behaviorally (see the sibling comment on
    // benchmarkSplitsGatesAndTimingOntoSeparateMachinesWithoutSpuriousSemantic-
    // CaptureFailure).
    @Test
    func semanticGPQACaptureIsWrittenOnlyAfterCorrectnessWorkerCloses() throws {
        let correctness = try String(
            contentsOfFile: "Sources/MLXFastHarness/QwenRuntimeCorrectness.swift",
            encoding: .utf8
        )
        let benchmark = try String(
            contentsOfFile: "Sources/MLXFastHarness/QwenRuntimeBenchmark.swift",
            encoding: .utf8
        )

        // The result struct carries the collected answers back to the caller.
        #expect(correctness.contains("let semanticGPQAAnswers: [SemanticGPQAAnswerCase]?"))

        // runLayeredCorrectnessWithWorker collects + returns the answers and
        // never writes them from inside the live-worker window.
        let layered = try sourceSlice(
            correctness,
            from: "static func runLayeredCorrectnessWithWorker(",
            to: "static func failedCorrectnessReport("
        )
        #expect(!layered.contains("writeSemanticGPQAAnswers"))
        #expect(layered.contains("capturedSemanticAnswers = semanticAnswers"))
        #expect(layered.contains("semanticGPQAAnswers: capturedSemanticAnswers"))
        // The count guard that fails a bad capture still runs inside the worker
        // window; only the disk write is deferred.
        #expect(layered.contains("guard semanticAnswers.count == semanticCapture.caseCount else {"))

        // In benchmarkWithWorker the single capture write happens only AFTER the
        // worker-closing defer, guarded on the returned answers.
        let benchmarkTail = try sourceSlice(
            benchmark,
            from: "let correctnessResult: WorkerLayeredCorrectnessResult",
            to: "correctnessSeconds = secondsSince(correctnessStart)"
        )
        let close = try #require(benchmarkTail.range(of: "correctnessWorker.close()"))
        let write = try #require(benchmarkTail.range(of: "writeSemanticGPQAAnswers("))
        #expect(close.lowerBound < write.lowerBound)
        #expect(benchmarkTail.contains(
            "if let semanticCapture, let semanticAnswers = correctnessResult.semanticGPQAAnswers"
        ))
        // Exactly one worker is created in this window, and its close() is a
        // defer, so the write provably runs after the worker is reaped.
        #expect(benchmarkTail.components(separatedBy: "RuntimeWorkerClient(").count - 1 == 1)
        #expect(benchmarkTail.components(separatedBy: "writeSemanticGPQAAnswers(").count - 1 == 1)
        let deferClose = try #require(benchmarkTail.range(of: "defer {"))
        #expect(deferClose.lowerBound < write.lowerBound)
    }
}

private struct BenchmarkScriptFixture {
    let root: URL
    let workingDirectory: URL
    let weights: URL
    let reference: URL
    let golden: URL
    let swift: URL
    let metallib: URL
    let score: URL
    let integrity: URL
}

private func makeBenchmarkScriptFixture(
    nestedWorkingDirectory: Bool = false
) throws -> BenchmarkScriptFixture {
    let root = try makeTemporaryDirectory()
    let workingDirectory = nestedWorkingDirectory ? root.appendingPathComponent("run") : root
    let weights = root.appendingPathComponent("weights")
    let reference = root.appendingPathComponent("reference")
    let golden = root.appendingPathComponent("golden.json")
    let swift = root.appendingPathComponent("mlxfast-swift")
    let metallib = root.appendingPathComponent("mlx.metallib")
    let score = root.appendingPathComponent("score.json")
    let integrity = root.appendingPathComponent("integrity.json")
    try FileManager.default.createDirectory(at: workingDirectory, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    let coreSources = workingDirectory.appendingPathComponent("Sources/MLXFastCore")
    let transformSources = workingDirectory.appendingPathComponent("Sources/MLXFastTransform")
    try FileManager.default.createDirectory(at: coreSources, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: transformSources, withIntermediateDirectories: true)
    try "// fixture".write(
        to: workingDirectory.appendingPathComponent("Package.swift"),
        atomically: true,
        encoding: .utf8
    )
    try "{}".write(
        to: workingDirectory.appendingPathComponent("Package.resolved"),
        atomically: true,
        encoding: .utf8
    )
    try "// fixture".write(
        to: coreSources.appendingPathComponent("Fixture.swift"),
        atomically: true,
        encoding: .utf8
    )
    try "// fixture".write(
        to: transformSources.appendingPathComponent("Fixture.swift"),
        atomically: true,
        encoding: .utf8
    )
    try "{}".write(to: weights.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
    try "{}".write(to: reference.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
    try "{}".write(to: golden, atomically: true, encoding: .utf8)
    try "fixture metallib".write(to: metallib, atomically: true, encoding: .utf8)
    return BenchmarkScriptFixture(
        root: root,
        workingDirectory: workingDirectory,
        weights: weights,
        reference: reference,
        golden: golden,
        swift: swift,
        metallib: metallib,
        score: score,
        integrity: integrity
    )
}

private func runBenchmarkScript(
    fixture: BenchmarkScriptFixture,
    arguments: [String],
    environment: [String: String] = [:]
) throws -> (status: Int32, stderr: String) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = [
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("benchmark.sh").path,
    ] + arguments
    process.currentDirectoryURL = fixture.workingDirectory
    process.environment = ProcessInfo.processInfo.environment.merging([
        "MLXFAST_NO_SANDBOX": "1",
        "MLXFAST_LOCAL_COOL_GATE": "0",
        // Parallel tests must not contend on the per-user local run lock or
        // abort on a genuinely resident model on the host machine.
        "MLXFAST_LOCAL_RUN_GUARD": "0",
        "MLXFAST_SWIFT_BIN": fixture.swift.path,
        "MLXFAST_RUNTIME_WORKER_EXECUTABLE": fixture.swift.path,
        "MLXFAST_MLX_METALLIB": fixture.metallib.path,
        "MLXFAST_WEIGHTS_PATH": fixture.weights.path,
        "MLXFAST_REFERENCE_DIR": fixture.reference.path,
        "MLXFAST_CORRECTNESS_GOLDEN_PATH": fixture.golden.path,
        "MLXFAST_SCORE_PATH": fixture.score.path,
        "MLXFAST_INTEGRITY_PATH": fixture.integrity.path,
    ].merging(environment) { _, new in new }) { _, new in new }
    let stderr = Pipe()
    process.standardError = stderr
    try process.run()
    let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return (
        process.terminationStatus,
        String(data: stderrData, encoding: .utf8) ?? ""
    )
}

private func runPairedTimingOverlay(
    gatesHarnessHash: String,
    candidateHarnessHash: String,
    gatesWeightsHash: String,
    candidateWeightsHash: String
) throws -> (status: Int32, stderr: String, score: [String: Any]?) {
    let root = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let gates = root.appendingPathComponent("score.json")
    let candidate = root.appendingPathComponent("candidate.json")
    let results = root.appendingPathComponent("results.json")
    let integrity = root.appendingPathComponent("integrity.json")
    let commit = String(repeating: "c", count: 40)

    let commonExpertMetrics: [String: Any] = [
        "expert_bytes_read": 0,
        "expert_cache_hits": 0,
        "expert_cache_misses": 0,
        "expert_cache_evictions": 0,
        "expert_read_seconds": 0,
        "expert_peak_cached_tensors": 0,
    ]
    var gatesMetrics = commonExpertMetrics
    gatesMetrics.merge([
        "harness_hash": gatesHarnessHash,
        "weights_hash": gatesWeightsHash,
        "weights_file_count": 1,
        "weights_byte_count": 1,
        "benchmark_wall_seconds": 1,
        "peak_ram_gb": 1,
        "process_resident_memory_gb": 1,
        "partial_result": true,
    ]) { _, new in new }
    var candidateMetrics = commonExpertMetrics
    candidateMetrics.merge([
        "commit": commit,
        "first_failing_case": NSNull(),
        "first_failing_step": NSNull(),
        "expected_token": NSNull(),
        "actual_token": NSNull(),
        "bandwidth_source": "ram_resident_model",
        "bandwidth_gb_per_token": 0,
        "timed_benchmark_seconds": 1,
        "benchmark_wall_seconds": 1,
        "peak_ram_gb": 1,
        "process_resident_memory_gb": 1,
        "harness_hash": candidateHarnessHash,
        "weights_hash": candidateWeightsHash,
        "weights_file_count": 1,
        "weights_byte_count": 1,
    ]) { _, new in new }

    try JSONSerialization.data(withJSONObject: [
        "passed": true,
        "score": NSNull(),
        "metrics": gatesMetrics,
    ]).write(to: gates)
    try JSONSerialization.data(withJSONObject: [
        "passed": true,
        "score": NSNull(),
        "metrics": candidateMetrics,
    ]).write(to: candidate)
    try JSONSerialization.data(withJSONObject: [
        "mode": "paired",
        "paired": ["decode_speedup": 1, "prefill_speedup": 1],
        "candidate": [
            "decode_seconds_per_token": 1,
            "prefill_seconds_per_token": 1,
            "verdict": "ACCEPT",
        ],
        "baseline": [
            "decode_seconds_per_token": 1,
            "prefill_seconds_per_token": 1,
            "verdict": "ACCEPT",
        ],
    ]).write(to: results)
    try Data("{}".utf8).write(to: integrity)

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = [
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".github/scripts/overlay-paired-timing.sh").path,
    ]
    process.environment = ProcessInfo.processInfo.environment.merging([
        "MLXFAST_SCORE_PATH": gates.path,
        "MLXFAST_CANDIDATE_SCORE_PATH": candidate.path,
        "MLXFAST_MEASURE_RESULTS_PATH": results.path,
        "MLXFAST_INTEGRITY_PATH": integrity.path,
        "MLXFAST_EXPECTED_COMMIT": commit,
        "MLXFAST_DECODE_SPEEDUP_FLOOR": "0.95",
        "MLXFAST_PREFILL_SPEEDUP_FLOOR": "0.95",
    ]) { _, new in new }
    let stderr = Pipe()
    process.standardError = stderr
    try process.run()
    let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()

    let scoreObject = try? JSONSerialization.jsonObject(with: Data(contentsOf: gates))
        as? [String: Any]
    return (
        process.terminationStatus,
        String(data: stderrData, encoding: .utf8) ?? "",
        scoreObject
    )
}

private func runFinalArtifactValidator(
    root: URL,
    golden: URL,
    commit: String,
    expectedNumLayers: String? = "40",
    plausibilityBounds: [String: String] = [:]
) throws -> (status: Int32, stderr: String) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = [
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".github/scripts/validate-benchmark-artifacts.sh").path,
    ]
    process.currentDirectoryURL = root
    var environment = ProcessInfo.processInfo.environment.merging([
        "MLXFAST_SCORE_PATH": "score.json",
        "MLXFAST_INTEGRITY_PATH": "benchmark-integrity.json",
        "MLXFAST_CORRECTNESS_GOLDEN_PATH": golden.path,
        "MLXFAST_EXPECTED_CORRECTNESS_GOLDEN_SHA256": String(repeating: "a", count: 64),
        "MLXFAST_EXPECTED_CORRECTNESS_STEPS": "64",
        "MLXFAST_EXPECTED_CORRECTNESS_CASES": "1",
        "MLXFAST_EXPECTED_CORRECTNESS_CHECKED_STEPS": "64",
        "MLXFAST_GPQA_TTFT_CASE_COUNT": "1",
        "MLXFAST_SEMANTIC_GPQA_CASE_COUNT": "1",
        "MLXFAST_SEMANTIC_GPQA_MIN_PASS": "1",
        "MLXFAST_SEMANTIC_GPQA_REQUIRED": "1",
        "MLXFAST_EXPECTED_COMMIT": commit,
    ]) { _, new in new }
    for name in [
        "MLXFAST_MAX_PLAUSIBLE_SPEEDUP",
        "MLXFAST_MIN_DECODE_SECONDS_PER_TOKEN",
        "MLXFAST_MIN_PREFILL_SECONDS_PER_TOKEN",
    ] {
        environment.removeValue(forKey: name)
    }
    environment.merge(plausibilityBounds) { _, new in new }
    if let expectedNumLayers {
        environment["MLXFAST_EXPECTED_NUM_LAYERS"] = expectedNumLayers
    } else {
        environment.removeValue(forKey: "MLXFAST_EXPECTED_NUM_LAYERS")
    }
    process.environment = environment
    process.standardOutput = FileHandle.nullDevice
    let stderr = Pipe()
    process.standardError = stderr
    try process.run()
    let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    return (
        process.terminationStatus,
        String(data: stderrData, encoding: .utf8) ?? ""
    )
}

private func makeTemporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

/// Runs `.github/scripts/redact-benchmark-failure.sh` with the workflow's
/// default relative paths resolved against `workspace`, mirroring the
/// "Surface redacted benchmark failure" step, and returns the exit status
/// plus the produced redacted artifact.
private func runRedactBenchmarkFailureScript(
    workspace: URL,
    environment: [String: String] = [:]
) throws -> (status: Int32, artifact: String) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = [
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".github/scripts/redact-benchmark-failure.sh").path,
        "benchmark-failure.json",
    ]
    process.currentDirectoryURL = workspace
    // Blank out GITHUB_STEP_SUMMARY so test runs inside CI do not append
    // fixture artifacts to the real job summary.
    process.environment = ProcessInfo.processInfo.environment
        .merging(["GITHUB_STEP_SUMMARY": ""]) { _, new in new }
        .merging(environment) { _, new in new }
    let stdout = Pipe()
    process.standardOutput = stdout
    try process.run()
    _ = stdout.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    let artifact =
        (try? String(
            contentsOf: workspace.appendingPathComponent("benchmark-failure.json"),
            encoding: .utf8
        )) ?? ""
    return (process.terminationStatus, artifact)
}

private func makeExecutable(_ url: URL) throws {
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
}

private func canonicalExistingPath(_ path: String) -> String {
    path.withCString { source in
        var buffer = [CChar](repeating: 0, count: Int(PATH_MAX))
        guard realpath(source, &buffer) != nil else {
            return path
        }
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }
}

private func initializeGitIndex(at directory: URL) throws {
    try runGit(["init", "-q"], at: directory)
    try runGit(["add", "--", "Package.swift", "Package.resolved", "Sources"], at: directory)
}

private func runGit(_ arguments: [String], at directory: URL) throws {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
    process.arguments = arguments
    process.currentDirectoryURL = directory
    let errorPipe = Pipe()
    process.standardError = errorPipe
    try process.run()
    let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        throw NSError(
            domain: "BenchmarkSafetyTests.git",
            code: Int(process.terminationStatus),
            userInfo: [
                NSLocalizedDescriptionKey: String(data: errorData, encoding: .utf8) ?? "git failed",
            ]
        )
    }
}

private func restoreEnvironment(_ name: String, value: String?) {
    if let value {
        setenv(name, value, 1)
    } else {
        unsetenv(name)
    }
}

private func sourceSlice(_ source: String, from start: String, to end: String) throws -> String {
    let startRange = try #require(source.range(of: start))
    let endRange = try #require(source.range(of: end, range: startRange.upperBound..<source.endIndex))
    return String(source[startRange.lowerBound..<endRange.lowerBound])
}
