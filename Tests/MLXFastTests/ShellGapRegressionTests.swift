import Darwin
import Foundation
import Testing

@Suite(.serialized)
struct ShellGapRegressionTests {
    @Test
    func offlineRunnerForwardsSignalReceivedBeforeCommandLaunch() throws {
        let root = try shellGapTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let bin = root.appendingPathComponent("bin")
        let temporaryDirectory = root.appendingPathComponent("tmp")
        try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: true
        )

        let probeStarted = root.appendingPathComponent("probe-started")
        let releaseProbe = root.appendingPathComponent("release-probe")
        let commandCompleted = root.appendingPathComponent("command-completed")
        let curl = bin.appendingPathComponent("curl")
        try writeShellGapExecutable(at: curl, contents: "#!/bin/sh\nexit 99\n")
        let sandboxExec = bin.appendingPathComponent("sandbox-exec")
        try writeShellGapExecutable(
            at: sandboxExec,
            contents: """
            #!/usr/bin/env bash
            set -euo pipefail
            [[ "$1" == -f ]]
            shift 2
            case "$1" in
              */curl)
                : > "$MLXFAST_TEST_PROBE_STARTED"
                while [[ ! -f "$MLXFAST_TEST_RELEASE_PROBE" ]]; do
                  sleep 0.01
                done
                exit 1
                ;;
            esac
            exec "$@"
            """
        )
        let command = root.appendingPathComponent("command.sh")
        try writeShellGapExecutable(
            at: command,
            contents: """
            #!/bin/sh
            sleep 1
            printf complete > "\(commandCompleted.path)"
            """
        )

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [shellGapRepositoryURL
            .appendingPathComponent(".github/scripts/run-offline.sh").path, command.path]
        process.currentDirectoryURL = shellGapRepositoryURL
        process.environment = shellGapEnvironment(overrides: [
            "PATH": "\(bin.path):/usr/bin:/bin",
            "TMPDIR": temporaryDirectory.path,
            "MLXFAST_TEST_PROBE_STARTED": probeStarted.path,
            "MLXFAST_TEST_RELEASE_PROBE": releaseProbe.path,
        ])
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        #expect(waitForShellGapPath(probeStarted, timeout: 3))

        #expect(kill(process.processIdentifier, SIGTERM) == 0)
        try Data().write(to: releaseProbe)
        process.waitUntilExit()
        let stdoutText = String(
            data: stdout.fileHandleForReading.readDataToEndOfFile(),
            encoding: .utf8
        ) ?? ""
        _ = stderr.fileHandleForReading.readDataToEndOfFile()

        // Reaching this line proves TERM was trapped while the network probe
        // was still the foreground process. A successful command would prove
        // that the pending signal was dropped instead of forwarded at launch.
        #expect(stdoutText.contains("running:"))
        #expect(process.terminationStatus == 128 + SIGTERM)
        #expect(!FileManager.default.fileExists(atPath: commandCompleted.path))
        let leftovers = try FileManager.default.contentsOfDirectory(
            atPath: temporaryDirectory.path
        )
        #expect(!leftovers.contains { $0.hasPrefix("mlxfast-offline.") })
    }

    @Test
    func metallibLockReleaseRefusesOwnerAndPublishedLinkReplacement() throws {
        let root = try shellGapTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let result = try runShellGapBash(
            """
            eval "$(sed \
              -e 's|^ROOT_DIR=.*|ROOT_DIR="${REPO_ROOT}"|' \
              -e '/^METAL_TOOLCHAIN_IDENTIFIER=/,$d' \
              "${REPO_ROOT}/tools/build-mlx-metallib.sh")"

            acquire_build_lock
            first_generation="${BUILD_LOCK_GENERATION_DIR}"
            original_owner="$(cat "${first_generation}/owner")"
            printf 'token=someone-else pid=1 host=test started_at=now\n' \
              > "${first_generation}/owner"
            owner_status=0
            release_build_lock || owner_status=$?
            [[ "${owner_status}" != 0 ]]
            [[ "${BUILD_LOCK_HELD}" == 1 ]]
            [[ -L "${BUILD_LOCK_DIR}" ]]
            [[ "$(readlink "${BUILD_LOCK_DIR}")" == "${first_generation}" ]]
            [[ -d "${first_generation}" ]]

            printf '%s\n' "${original_owner}" > "${first_generation}/owner"
            release_build_lock
            [[ ! -e "${BUILD_LOCK_DIR}" && ! -L "${BUILD_LOCK_DIR}" ]]
            [[ ! -e "${first_generation}" ]]

            acquire_build_lock
            owned_generation="${BUILD_LOCK_GENERATION_DIR}"
            replacement_generation="${BUILD_LOCK_DIR}.generation.replacement"
            mkdir "${replacement_generation}"
            cp "${owned_generation}/owner" "${replacement_generation}/owner"
            rm -f "${BUILD_LOCK_DIR}"
            ln -s -h -- "${replacement_generation}" "${BUILD_LOCK_DIR}"

            link_status=0
            release_build_lock || link_status=$?
            [[ "${link_status}" != 0 ]]
            [[ "${BUILD_LOCK_HELD}" == 1 ]]
            [[ -L "${BUILD_LOCK_DIR}" ]]
            [[ "$(readlink "${BUILD_LOCK_DIR}")" == "${replacement_generation}" ]]
            [[ -d "${replacement_generation}" ]]
            [[ -d "${owned_generation}" ]]

            rm -f "${BUILD_LOCK_DIR}"
            ln -s -h -- "${owned_generation}" "${BUILD_LOCK_DIR}"
            release_build_lock
            [[ ! -e "${owned_generation}" ]]
            [[ -d "${replacement_generation}" ]]
            rm -rf "${replacement_generation}"
            """,
            environment: [
                "REPO_ROOT": shellGapRepositoryURL.path,
                "MLXFAST_MLX_METAL_BUILD_DIR": root.appendingPathComponent("metal-build").path,
                "MLXFAST_MLX_METAL_BUILD_LOCK_DIR": root.appendingPathComponent("metal-build.lock").path,
                "MLXFAST_MLX_METALLIB": root.appendingPathComponent("mlx.metallib").path,
                "MLXFAST_CLANG_MODULE_CACHE": root.appendingPathComponent("clang-cache").path,
                "MLXFAST_METAL_COMPILER_HOME": root.appendingPathComponent("metal-home").path,
            ]
        )

        #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
        #expect(result.stderr.contains("owned by another process"))
        #expect(result.stderr.contains("refusing to release a replaced build lock"))
    }

    @Test(arguments: [
        ("tensor-inventory", "tensor inventory disagrees with the index"),
        ("unindexed-shard", "contains unindexed shard"),
        ("range-hole", "staged safetensors shard has an invalid header: model.safetensors"),
        ("trailing-byte", "staged safetensors shard has an invalid header: model.safetensors"),
    ])
    func benchmarkRejectsMalformedStagedShardWithoutReplacingWeights(
        caseName: String,
        expectedError: String
    ) throws {
        let fixture = try makeShellGapBenchmarkFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let oldConfig = try Data(contentsOf: fixture.weights.appendingPathComponent("config.json"))
        let sentinel = fixture.weights.appendingPathComponent("old-weights-sentinel")
        try "keep".write(to: sentinel, atomically: true, encoding: .utf8)

        try writeShellGapExecutable(
            at: fixture.swift,
            contents: """
            #!/usr/bin/env bash
            set -euo pipefail
            [[ "${1:-}" == transform ]]
            shift
            output=""
            while [[ "$#" -gt 0 ]]; do
              if [[ "$1" == --output ]]; then
                output="$2"
                break
              fi
              shift
            done
            [[ -n "${output}" ]]
            mkdir -p "${output}"

            write_shard() {
              local path="$1"
              local header="$2"
              local data_bytes="$3"
              local index
              (( ${#header} < 256 ))
              printf "\\\\$(printf '%03o' "${#header}")\\000\\000\\000\\000\\000\\000\\000" > "${path}"
              printf '%s' "${header}" >> "${path}"
              for ((index = 0; index < data_bytes; index++)); do
                printf '\\001' >> "${path}"
              done
            }

            printf '%s\n' '{}' > "${output}/config.json"
            printf '%s\n' '{"weight_map":{"tensor":"model.safetensors"}}' \
              > "${output}/model.safetensors.index.json"
            write_shard "${output}/model.safetensors" \
              '{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}' 1

            case "${MLXFAST_TEST_BAD_SHARD}" in
              tensor-inventory)
                write_shard "${output}/model.safetensors" \
                  '{"other":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}' 1
                ;;
              unindexed-shard)
                write_shard "${output}/extra.safetensors" \
                  '{"extra":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}' 1
                ;;
              range-hole)
                printf '%s\n' '{"weight_map":{"a":"model.safetensors","b":"model.safetensors"}}' \
                  > "${output}/model.safetensors.index.json"
                write_shard "${output}/model.safetensors" \
                  '{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}}' 3
                ;;
              trailing-byte)
                write_shard "${output}/model.safetensors" \
                  '{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}' 2
                ;;
              *) exit 91 ;;
            esac
            """
        )

        let result = try runShellGapBenchmark(
            fixture,
            environment: [
                "MLXFAST_FORCE_TRANSFORM": "1",
                "MLXFAST_TEST_BAD_SHARD": caseName,
            ]
        )

        #expect(result.status != 0, "case \(caseName)")
        #expect(result.stderr.contains(expectedError), "case \(caseName): \(result.stderr)")
        #expect(
            try Data(contentsOf: fixture.weights.appendingPathComponent("config.json")) == oldConfig,
            "case \(caseName)"
        )
        #expect(try String(contentsOf: sentinel, encoding: .utf8) == "keep", "case \(caseName)")
        let installedFiles = try FileManager.default
            .contentsOfDirectory(atPath: fixture.weights.path)
            .sorted()
        #expect(installedFiles == ["config.json", "old-weights-sentinel"], "case \(caseName)")
    }

    @Test
    func hashWeightsRejectsSymlinkRootAndNonregularEntryWithoutWritingOutput() throws {
        let root = try shellGapTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let actualWeights = root.appendingPathComponent("actual-weights")
        try FileManager.default.createDirectory(
            at: actualWeights,
            withIntermediateDirectories: true
        )
        try "{}".write(
            to: actualWeights.appendingPathComponent("config.json"),
            atomically: true,
            encoding: .utf8
        )
        let symlinkRoot = root.appendingPathComponent("weights-link")
        try FileManager.default.createSymbolicLink(
            at: symlinkRoot,
            withDestinationURL: actualWeights
        )
        let output = root.appendingPathComponent("weights.sha256")
        try "preserve\n".write(to: output, atomically: true, encoding: .utf8)

        let symlinkResult = try runShellGapHash(weights: symlinkRoot, output: output)
        #expect(symlinkResult.status != 0)
        #expect(symlinkResult.stderr.contains("is not a non-symlink directory"))
        #expect(try String(contentsOf: output, encoding: .utf8) == "preserve\n")

        let fifo = actualWeights.appendingPathComponent("unexpected.fifo")
        #expect(mkfifo(fifo.path, 0o600) == 0)
        let fifoResult = try runShellGapHash(weights: actualWeights, output: output)
        #expect(fifoResult.status != 0)
        #expect(fifoResult.stderr.contains("symlink or non-regular entry"))
        #expect(fifoResult.stderr.contains("unexpected.fifo"))
        #expect(try String(contentsOf: output, encoding: .utf8) == "preserve\n")
    }
}

private let shellGapRepositoryURL = URL(
    fileURLWithPath: FileManager.default.currentDirectoryPath,
    isDirectory: true
)

private struct ShellGapProcessResult {
    let status: Int32
    let stdout: String
    let stderr: String
}

private func runShellGapBash(
    _ script: String,
    environment: [String: String] = [:]
) throws -> ShellGapProcessResult {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["-c", script]
    process.currentDirectoryURL = shellGapRepositoryURL
    process.environment = shellGapEnvironment(overrides: environment)
    let stdout = Pipe()
    let stderr = Pipe()
    process.standardOutput = stdout
    process.standardError = stderr
    try process.run()
    process.waitUntilExit()
    return ShellGapProcessResult(
        status: process.terminationStatus,
        stdout: String(
            data: stdout.fileHandleForReading.readDataToEndOfFile(),
            encoding: .utf8
        ) ?? "",
        stderr: String(
            data: stderr.fileHandleForReading.readDataToEndOfFile(),
            encoding: .utf8
        ) ?? ""
    )
}

private struct ShellGapBenchmarkFixture {
    let root: URL
    let weights: URL
    let reference: URL
    let golden: URL
    let swift: URL
    let score: URL
    let integrity: URL
    let metallib: URL
}

private func makeShellGapBenchmarkFixture() throws -> ShellGapBenchmarkFixture {
    let root = try shellGapTemporaryDirectory()
    let weights = root.appendingPathComponent("weights")
    let reference = root.appendingPathComponent("reference")
    let golden = root.appendingPathComponent("golden.json")
    let swift = root.appendingPathComponent("mlxfast-swift")
    let coreSources = root.appendingPathComponent("Sources/MLXFastCore")
    let transformSources = root.appendingPathComponent("Sources/MLXFastTransform")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: coreSources, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: transformSources, withIntermediateDirectories: true)
    try "// fixture".write(
        to: root.appendingPathComponent("Package.swift"),
        atomically: true,
        encoding: .utf8
    )
    try "{}".write(
        to: root.appendingPathComponent("Package.resolved"),
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
    try "{\"old\":true}\n".write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try "{}".write(
        to: reference.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    try "{}".write(to: golden, atomically: true, encoding: .utf8)
    // benchmark.sh fails fast when the MLX metallib is missing (incomplete
    // setup); these fixtures exercise later gates, so satisfy that guard.
    let metallib = root.appendingPathComponent("mlx.metallib")
    try "fixture metallib".write(to: metallib, atomically: true, encoding: .utf8)
    return ShellGapBenchmarkFixture(
        root: root,
        weights: weights,
        reference: reference,
        golden: golden,
        swift: swift,
        score: root.appendingPathComponent("score.json"),
        integrity: root.appendingPathComponent("integrity.json"),
        metallib: metallib
    )
}

private func runShellGapBenchmark(
    _ fixture: ShellGapBenchmarkFixture,
    environment overrides: [String: String]
) throws -> ShellGapProcessResult {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = [
        shellGapRepositoryURL.appendingPathComponent("benchmark.sh").path,
        "--local-iterate",
    ]
    process.currentDirectoryURL = fixture.root
    var environment = shellGapEnvironment(overrides: [
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
    ])
    environment.merge(overrides) { _, new in new }
    process.environment = environment
    let stdout = Pipe()
    let stderr = Pipe()
    process.standardOutput = stdout
    process.standardError = stderr
    try process.run()
    process.waitUntilExit()
    return ShellGapProcessResult(
        status: process.terminationStatus,
        stdout: String(
            data: stdout.fileHandleForReading.readDataToEndOfFile(),
            encoding: .utf8
        ) ?? "",
        stderr: String(
            data: stderr.fileHandleForReading.readDataToEndOfFile(),
            encoding: .utf8
        ) ?? ""
    )
}

private func runShellGapHash(weights: URL, output: URL) throws -> ShellGapProcessResult {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = [
        shellGapRepositoryURL
            .appendingPathComponent(".github/scripts/hash-weights-directory.sh").path,
        weights.path,
        output.path,
    ]
    process.currentDirectoryURL = shellGapRepositoryURL
    process.environment = shellGapEnvironment()
    let stdout = Pipe()
    let stderr = Pipe()
    process.standardOutput = stdout
    process.standardError = stderr
    try process.run()
    process.waitUntilExit()
    return ShellGapProcessResult(
        status: process.terminationStatus,
        stdout: String(
            data: stdout.fileHandleForReading.readDataToEndOfFile(),
            encoding: .utf8
        ) ?? "",
        stderr: String(
            data: stderr.fileHandleForReading.readDataToEndOfFile(),
            encoding: .utf8
        ) ?? ""
    )
}

private func shellGapEnvironment(
    overrides: [String: String] = [:]
) -> [String: String] {
    var environment = ProcessInfo.processInfo.environment
    for key in environment.keys.filter({ $0.hasPrefix("MLXFAST_") }) {
        environment.removeValue(forKey: key)
    }
    environment.merge(overrides) { _, new in new }
    return environment
}

private func shellGapTemporaryDirectory() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
        "mlxfast-shell-gap-tests-\(UUID().uuidString)",
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}

private func writeShellGapExecutable(at url: URL, contents: String) throws {
    try contents.write(to: url, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: url.path
    )
}

private func waitForShellGapPath(_ url: URL, timeout: TimeInterval) -> Bool {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
        if FileManager.default.fileExists(atPath: url.path) {
            return true
        }
        Thread.sleep(forTimeInterval: 0.01)
    }
    return FileManager.default.fileExists(atPath: url.path)
}
