import Foundation
@testable import MLXFastHarness
import Testing

@Test
func setupScriptCoordinatesCacheAndMetallibState() throws {
    let setup = try String(contentsOfFile: "setup.sh", encoding: .utf8)
    let metallibBuilder = try String(
        contentsOfFile: "tools/build-mlx-metallib.sh",
        encoding: .utf8
    )

    #expect(setup.contains("CANONICAL_REFERENCE_LOCK_BASE="))
    #expect(setup.contains("REFERENCE_CACHE_MUTATION_LOCK_DIR=\"${MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR:-${CANONICAL_REFERENCE_LOCK_BASE}.mlxfast-setup.lock}\""))
    #expect(setup.contains("METALLIB_BUILD_STATE=\"not_started\""))
    #expect(!setup.contains("existing-stamp-only"))
    #expect(setup.contains("acquire_reference_cache_mutation_lock"))
    #expect(setup.contains("recover_stale_reference_cache_mutation_lock"))
    #expect(setup.contains("mktemp \"${lock_path}.tmp.XXXXXX\""))
    #expect(setup.contains("metal_toolchain_identifier"))
    #expect(setup.contains("export TOOLCHAINS=\"${identifier}\""))
    // Independent SwiftPM build/cache roots: the trusted CLI builds in .build
    // and the participant worker in its own .build-worker scratch root with
    // its own clang module cache, so a participant-code build never writes
    // into the trusted product tree. mlx.metallib (a participant artifact)
    // colocates with the worker binary.
    #expect(setup.contains("RUNTIME_WORKER_BIN=\"${MLXFAST_RUNTIME_WORKER_EXECUTABLE:-.build-worker/release/mlxfast-runtime-worker}\""))
    #expect(setup.contains("MLX_METALLIB=\"${MLXFAST_MLX_METALLIB:-$(dirname \"${RUNTIME_WORKER_BIN}\")/mlx.metallib}\""))
    #expect(setup.contains("swift build -c release --force-resolved-versions --product mlxfast-swift"))
    #expect(setup.contains("swift build -c release --force-resolved-versions --scratch-path .build-worker --product mlxfast-runtime-worker"))
    #expect(setup.contains("mkdir -p .build/clang-module-cache .build-worker/clang-module-cache"))
    #expect(setup.contains("CLANG_MODULE_CACHE_PATH=\"${CLANG_MODULE_CACHE_PATH:-${PWD}/.build/clang-module-cache}\" \\"))
    #expect(setup.contains("CLANG_MODULE_CACHE_PATH=\"${CLANG_MODULE_CACHE_PATH:-${PWD}/.build-worker/clang-module-cache}\" \\"))
    #expect(setup.contains("participant runtime worker missing at ${RUNTIME_WORKER_BIN}"))
    #expect(!setup.contains("TODO(security): Give the trusted CLI and participant worker independent"))
    #expect(metallibBuilder.contains("-DMLX_BUILD_GGUF=OFF"))
    #expect(metallibBuilder.contains("export CLANG_MODULE_CACHE_PATH"))
    #expect(metallibBuilder.contains("CLANG_MODULE_CACHE_PATH=\"$(repository_path"))
    // The metallib compiles participant-editable vendored Metal sources, so
    // its build tree, module cache, and output live under the worker root.
    #expect(metallibBuilder.contains("${CLANG_MODULE_CACHE_PATH:-.build-worker/clang-module-cache}"))
    #expect(metallibBuilder.contains("${MLXFAST_MLX_METAL_BUILD_DIR:-.build-worker/mlx-metal}"))
    #expect(metallibBuilder.contains("${MLXFAST_RUNTIME_WORKER_EXECUTABLE:-.build-worker/${BUILD_CONFIGURATION}/mlxfast-runtime-worker}"))
    #expect(metallibBuilder.contains("HOME=\"${METAL_COMPILER_HOME}\" \"${CMAKE_BIN}\""))
    // Every published metallib carries a fingerprint sidecar over the
    // vendored kernel sources (the AOT tree plus the runtime-effective JIT
    // strings); --print-fingerprint exposes the same recipe to the ranked
    // cache key and verification steps, and the marker TODO is resolved.
    #expect(!metallibBuilder.contains("TODO(security)"))
    #expect(metallibBuilder.contains("FINGERPRINT_RECORD_PREFIX=\"mlxfast-metallib-fingerprint-v1\""))
    #expect(metallibBuilder.contains("compute_vendored_metal_fingerprint()"))
    #expect(metallibBuilder.contains("find mlx mlx-generated -type f -print0"))
    #expect(metallibBuilder.contains("LC_ALL=C sort -z"))
    #expect(metallibBuilder.contains("xargs -0 shasum -a 256"))
    #expect(metallibBuilder.contains("if [[ \"${1:-}\" == \"--print-fingerprint\" ]]; then"))
    #expect(metallibBuilder.contains("publish_fingerprint_record \"${OUTPUT_PATH}.fingerprint\""))
    #expect(metallibBuilder.contains("VENDORED_METAL_FINGERPRINT=\"$(compute_vendored_metal_fingerprint)\""))
}

@Test
func metallibBuilderRegeneratesStaleCMakeCache() throws {
    let metallibBuilder = try String(
        contentsOfFile: "tools/build-mlx-metallib.sh",
        encoding: .utf8
    )

    // A build tree restored into a relocated workspace (for example the
    // ranked job workspace moving between absolute paths) carries a
    // CMakeCache.txt whose recorded build/source directories no longer
    // match; cmake configure hard-fails on that mismatch. The builder must
    // detect it while holding the build lock and regenerate the build tree
    // instead of failing, and must keep a matching cache untouched so
    // incremental rebuilds stay cheap.
    #expect(metallibBuilder.contains("CMAKE_CACHE_FILE=\"${CMAKE_BUILD_DIR}/CMakeCache.txt\""))
    #expect(metallibBuilder.contains("CMAKE_CACHEFILE_DIR:INTERNAL="))
    #expect(metallibBuilder.contains("CMAKE_HOME_DIRECTORY:INTERNAL="))
    #expect(metallibBuilder.contains("|| \"${recorded_source_dir}\" != \"${MLX_SOURCE}\""))
    #expect(metallibBuilder.contains("rm -rf \"${CMAKE_BUILD_DIR}\""))

    // A build tree owned by another uid is equally unusable, but fails
    // LATER: cmake's configure_file always chmod()s its output, and chmod
    // by a non-owner fails with EPERM even where writing file data is
    // allowed. On the ranked box the worker build-products cache restores
    // as the runner uid while the bench uid builds through an ACL grant
    // that withholds writesecurity, so the first kernel-touching
    // submission (metallib cache miss) died reconfiguring the restored
    // tree with "Operation not permitted" (run 30048514684). The builder
    // must regenerate a foreign-owned tree, and both regeneration paths
    // must recreate the Metal compiler HOME they may have just deleted
    // (its default lives inside the build tree).
    #expect(metallibBuilder.contains("! -O \"${CMAKE_BUILD_DIR}\""))
    #expect(metallibBuilder.contains("! -O \"${CMAKE_CACHE_FILE}\""))
    #expect(metallibBuilder.contains("owned by another user"))
    #expect(metallibBuilder.contains(
        "rm -rf \"${CMAKE_BUILD_DIR}\"\n  mkdir -p \"${METAL_COMPILER_HOME}\""
    ))
    #expect(metallibBuilder.contains(
        "rm -rf \"${CMAKE_BUILD_DIR}\"\n    mkdir -p \"${METAL_COMPILER_HOME}\""
    ))
}

@Test
func setupAndMetallibLocksPublishWithBSDAndGNUCoreutils() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let fakeBin = root.appendingPathComponent("bin")
    try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("ln"),
        contents: """
        #!/usr/bin/env bash
        set -euo pipefail
        printf '%s\n' "$*" >> "${MLXFAST_TEST_LN_LOG:?}"
        mode="${MLXFAST_TEST_COREUTILS_MODE:?}"
        if [[ "${mode}" == "gnu" && "${1:-}" == "-s" && "${2:-}" == "-h" ]]; then
          if [[ "${MLXFAST_TEST_INJECT_COMPETING_LOCK:-0}" == "1" ]]; then
            mkdir "${5:?}"
          fi
          exit 1
        fi
        if [[ "${1:-}" == "-s" && "${2:-}" == "-T" && "${3:-}" == "--" ]]; then
          [[ "${mode}" == "gnu" ]]
          [[ ! -e "${5:?}" && ! -L "${5}" ]]
          exec /bin/ln -s -h -- "${4:?}" "${5}"
        fi
        exec /bin/ln "$@"
        """
    )

    for mode in ["bsd", "gnu"] {
        let modeRoot = root.appendingPathComponent(mode)
        try FileManager.default.createDirectory(at: modeRoot, withIntermediateDirectories: true)
        let linkLog = modeRoot.appendingPathComponent("ln.log")
        let raceLog = modeRoot.appendingPathComponent("race-ln.log")
        let result = try runSetupBash(
            """
            set -euo pipefail
            eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
            acquire_reference_cache_mutation_lock
            setup_generation="${REFERENCE_CACHE_MUTATION_LOCK_GENERATION_DIR}"
            [[ -L "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]
            [[ "$(readlink "${REFERENCE_CACHE_MUTATION_LOCK_DIR}")" == "${setup_generation}" ]]
            release_reference_cache_mutation_lock
            [[ ! -e "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" \
                && ! -L "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]
            [[ ! -e "${setup_generation}" ]]

            eval "$(sed \
              -e 's|^ROOT_DIR=.*|ROOT_DIR="${REPO_ROOT}"|' \
              -e '/^METAL_TOOLCHAIN_IDENTIFIER=/,$d' \
              "${REPO_ROOT}/tools/build-mlx-metallib.sh")"
            acquire_build_lock
            build_generation="${BUILD_LOCK_GENERATION_DIR}"
            [[ -L "${BUILD_LOCK_DIR}" ]]
            [[ "$(readlink "${BUILD_LOCK_DIR}")" == "${build_generation}" ]]
            release_build_lock
            [[ ! -e "${BUILD_LOCK_DIR}" && ! -L "${BUILD_LOCK_DIR}" ]]
            [[ ! -e "${build_generation}" ]]

            if [[ "${MLXFAST_TEST_COREUTILS_MODE}" == "gnu" ]]; then
              export MLXFAST_TEST_LN_LOG="${RACE_LOG}"
              export MLXFAST_TEST_INJECT_COMPETING_LOCK=1
              mkdir "${MODE_ROOT}/race-target"
              publish_status=0
              publish_lock_symlink \
                "${MODE_ROOT}/race-target" "${MODE_ROOT}/race.lock" || publish_status=$?
              [[ "${publish_status}" != "0" ]]
              [[ -d "${MODE_ROOT}/race.lock" && ! -L "${MODE_ROOT}/race.lock" ]]
            fi
            """,
            environment: [
                "REPO_ROOT": FileManager.default.currentDirectoryPath,
                "PATH": "\(fakeBin.path):/usr/bin:/bin",
                "MODE_ROOT": modeRoot.path,
                "RACE_LOG": raceLog.path,
                "MLXFAST_TEST_LN_LOG": linkLog.path,
                "MLXFAST_TEST_COREUTILS_MODE": mode,
                "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": modeRoot
                    .appendingPathComponent("reference.lock").path,
                "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS": "0",
                "MLXFAST_MLX_METAL_BUILD_DIR": modeRoot
                    .appendingPathComponent("metal-build").path,
                "MLXFAST_MLX_METAL_BUILD_LOCK_DIR": modeRoot
                    .appendingPathComponent("metal-build.lock").path,
                "MLXFAST_MLX_METAL_BUILD_LOCK_TIMEOUT_SECONDS": "0",
                "MLXFAST_MLX_METALLIB": modeRoot.appendingPathComponent("mlx.metallib").path,
                "MLXFAST_CLANG_MODULE_CACHE": modeRoot.appendingPathComponent("clang-cache").path,
                "MLXFAST_METAL_COMPILER_HOME": modeRoot.appendingPathComponent("metal-home").path,
            ]
        )

        #expect(
            result.status == 0,
            "mode=\(mode) stdout: \(result.stdout) stderr: \(result.stderr)"
        )
        let linkAttempts = try String(contentsOf: linkLog, encoding: .utf8)
            .split(separator: "\n")
        if mode == "bsd" {
            #expect(linkAttempts.count == 2)
            #expect(linkAttempts.allSatisfy { $0.contains("-s -h --") })
        } else {
            #expect(linkAttempts.count == 4)
            #expect(linkAttempts.filter { $0.contains("-s -h --") }.count == 2)
            #expect(linkAttempts.filter { $0.contains("-s -T --") }.count == 2)
            let raceAttempts = try String(contentsOf: raceLog, encoding: .utf8)
                .split(separator: "\n")
            #expect(raceAttempts.count == 1)
            #expect(raceAttempts.first?.contains("-s -h --") == true)
        }
    }
}

@Test
func setupAndMetallibStatFallbackDiscardsFailedProbeOutput() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let fakeBin = root.appendingPathComponent("bin")
    try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("stat"),
        contents: """
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ "${1:-}" == "-f" ]]; then
          printf 'failed-bsd-probe-output\n'
          exit 1
        fi
        [[ "${1:-}" == "-c" && "$#" == "3" ]]
        case "${2}" in
          '%Y') printf '1700000000\n' ;;
          '%d:%i') printf '11:22\n' ;;
          '%d:%i:%s:%Y:%Z:0') printf '11:22:33:44:55:0\n' ;;
          '%d:%i:%s:%Y:0') printf '11:22:33:44:0\n' ;;
          '%u') /usr/bin/id -u ;;
          *) exit 2 ;;
        esac
        """
    )

    let result = try runSetupBash(
        """
        set -euo pipefail
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        [[ "$(file_mtime_seconds "${TEST_ROOT}")" == "1700000000" ]]
        [[ "$(file_metadata_signature "${TEST_ROOT}")" == "11:22:33:44:55:0" ]]
        [[ "$(file_content_identity_signature "${TEST_ROOT}")" == "11:22:33:44:0" ]]
        [[ "$(directory_identity "${TEST_ROOT}")" == "11:22" ]]

        eval "$(sed \
          -e 's|^ROOT_DIR=.*|ROOT_DIR="${REPO_ROOT}"|' \
          -e '/^METAL_TOOLCHAIN_IDENTIFIER=/,$d' \
          "${REPO_ROOT}/tools/build-mlx-metallib.sh")"
        [[ "$(lock_mtime_seconds "${TEST_ROOT}")" == "1700000000" ]]
        [[ "$(lock_directory_identity "${TEST_ROOT}")" == "11:22" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "PATH": "\(fakeBin.path):/usr/bin:/bin",
            "TEST_ROOT": root.path,
            "MLXFAST_MLX_METAL_BUILD_DIR": root.appendingPathComponent("metal-build").path,
            "MLXFAST_MLX_METAL_BUILD_LOCK_DIR": root
                .appendingPathComponent("metal-build.lock").path,
            "MLXFAST_MLX_METALLIB": root.appendingPathComponent("mlx.metallib").path,
            "MLXFAST_CLANG_MODULE_CACHE": root.appendingPathComponent("clang-cache").path,
            "MLXFAST_METAL_COMPILER_HOME": root.appendingPathComponent("metal-home").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
    let setup = try String(contentsOfFile: "setup.sh", encoding: .utf8)
    let metallibBuilder = try String(
        contentsOfFile: "tools/build-mlx-metallib.sh",
        encoding: .utf8
    )
    let setupStatHelperCount = setup.components(separatedBy: "stat_value() {").count - 1
    let builderStatHelperCount =
        metallibBuilder.components(separatedBy: "stat_value() {").count - 1
    #expect(setupStatHelperCount == 2)
    #expect(builderStatHelperCount == 1)
    #expect(!setup.contains("|| stat -c"))
    #expect(!metallibBuilder.contains("|| stat -c"))
}

@Test
func setupStartsMetallibBuildOnlyOnceInSynchronousAndParallelModes() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    for parallel in ["0", "1"] {
        let modeRoot = root.appendingPathComponent("mode-\(parallel)")
        try FileManager.default.createDirectory(at: modeRoot, withIntermediateDirectories: true)
        let result = try runSetupBash(
            """
            eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
            build_mlx_metallib() {
              printf 'build\n' >> "${BUILD_LOG}"
              sleep 0.1
              mkdir -p "$(dirname "${MLX_METALLIB}")"
              : > "${MLX_METALLIB}"
            }

            # These repeated calls model the warm-cache verification, fresh
            # download, and repair call sites that can all request the build.
            start_mlx_metallib_build
            start_mlx_metallib_build
            start_mlx_metallib_build
            wait_for_mlx_metallib_build
            start_mlx_metallib_build
            wait_for_mlx_metallib_build

            [[ "${METALLIB_BUILD_STATE}" == "completed" ]]
            [[ "$(wc -l < "${BUILD_LOG}" | tr -d ' ')" == "1" ]]
            """,
            environment: [
                "REPO_ROOT": FileManager.default.currentDirectoryPath,
                "BUILD_LOG": modeRoot.appendingPathComponent("build.log").path,
                "MLXFAST_MLX_METALLIB": modeRoot.appendingPathComponent("mlx.metallib").path,
                "MLXFAST_SETUP_PARALLEL_METALLIB": parallel,
            ]
        )
        #expect(result.status == 0, "parallel=\(parallel): \(result.stderr)")
    }
}

@Test
func setupRetainsFailedMetallibStateInSynchronousAndParallelModes() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    for parallel in ["0", "1"] {
        let result = try runSetupBash(
            """
            eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
            build_mlx_metallib() { return 19; }

            start_status=0
            start_mlx_metallib_build || start_status=$?
            wait_status=0
            wait_for_mlx_metallib_build || wait_status=$?
            retry_status=0
            start_mlx_metallib_build || retry_status=$?

            if [[ "${SETUP_PARALLEL_METALLIB}" == "1" ]]; then
              [[ "${start_status}" == "0" ]]
            else
              [[ "${start_status}" != "0" ]]
            fi
            [[ "${wait_status}" != "0" ]]
            [[ "${retry_status}" != "0" ]]
            [[ "${METALLIB_BUILD_STATE}" == "failed" ]]
            [[ -z "${METALLIB_BUILD_PID}" ]]
            """,
            environment: [
                "REPO_ROOT": FileManager.default.currentDirectoryPath,
                "MLXFAST_MLX_METALLIB": root.appendingPathComponent("missing-\(parallel).metallib").path,
                "MLXFAST_SETUP_PARALLEL_METALLIB": parallel,
            ]
        )
        #expect(result.status == 0, "parallel=\(parallel): \(result.stderr)")
    }
}

@Test
func setupFailureCleanupTerminatesMetallibProcessGroup() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        build_mlx_metallib() {
          sleep 30 &
          child_pid=$!
          printf '%s\n' "${child_pid}" > "${CHILD_PID_PATH}"
          wait "${child_pid}"
        }

        start_mlx_metallib_build
        for attempt in 1 2 3 4 5 6 7 8 9 10; do
          [[ -s "${CHILD_PID_PATH}" ]] && break
          sleep 0.05
        done
        [[ -s "${CHILD_PID_PATH}" ]]

        set +e
        false
        cleanup_background_builds
        cleanup_status=$?
        set -e
        [[ "${cleanup_status}" != "0" ]]
        child_pid="$(cat "${CHILD_PID_PATH}")"
        ! kill -0 "${child_pid}" 2>/dev/null
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "CHILD_PID_PATH": root.appendingPathComponent("child.pid").path,
            "MLXFAST_MLX_METALLIB": root.appendingPathComponent("mlx.metallib").path,
            "MLXFAST_SETUP_PARALLEL_METALLIB": "1",
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupSerializesSharedReferenceCacheMutationAndRechecksAfterWaiting() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try "{}".write(
        to: referenceDir.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        verify_reference_weights() {
          if [[ ! -d "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]; then
            printf 'unlocked verification\n' >> "${UNLOCKED_VERIFY_LOG}"
          fi
          [[ -f "$1/ready" ]]
        }
        ensure_reference_compat_link() { :; }
        download_reference_weights_locked() {
          printf 'mutation\n' >> "${MUTATION_LOG}"
          sleep 1.2
          : > "$1/ready"
        }

        download_reference_weights "${REFERENCE_DIR}" &
        first_pid=$!
        sleep 0.1
        download_reference_weights "${REFERENCE_DIR}" &
        second_pid=$!
        wait "${first_pid}"
        wait "${second_pid}"

        [[ "$(wc -l < "${MUTATION_LOG}" | tr -d ' ')" == "1" ]]
        [[ ! -e "${UNLOCKED_VERIFY_LOG}" ]]
        [[ ! -e "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MUTATION_LOG": root.appendingPathComponent("mutations.log").path,
            "UNLOCKED_VERIFY_LOG": root.appendingPathComponent("unlocked-verification.log").path,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache.lock").path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS": "5",
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
    #expect(result.stdout.contains("waiting for reference cache mutation lock"))
    #expect(result.stdout.contains("reference weights became ready while waiting"))
}

@Test
func setupCompatibilityLinkFailurePropagatesThroughConditionalCaller() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    let blockedParent = root.appendingPathComponent("not-a-directory")
    let compatibilityLink = blockedParent.appendingPathComponent("reference-link")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try "blocker".write(to: blockedParent, atomically: true, encoding: .utf8)

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        link_status=0
        ensure_reference_compat_link "${REFERENCE_DIR}" || link_status=$?
        [[ "${link_status}" != "0" ]]
        [[ ! -e "${REFERENCE_COMPAT_LINK}" && ! -L "${REFERENCE_COMPAT_LINK}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_COMPAT_LINK": compatibilityLink.path,
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
    #expect(!result.stdout.contains("setup.sh: linked"))
}

@Test
func setupAcceptsLegacyReferenceDirectoryAsItsOwnCompatibilityPath() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        cd "${TEST_ROOT}"
        mkdir -p "${REFERENCE_DIR}"
        ensure_reference_compat_link "${REFERENCE_DIR}"
        [[ -d "${REFERENCE_DIR}" ]]
        [[ ! -L "${REFERENCE_DIR}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "TEST_ROOT": root.path,
            // setup.sh's DEFAULT_REFERENCE_DIR (and therefore
            // REFERENCE_COMPAT_LINK) is the adopted Qwen 3.8 path since
            // 2026-08-14; pointing MLXFAST_REFERENCE_DIR at it is what makes
            // the reference directory its own compatibility path.
            "MLXFAST_REFERENCE_DIR": "reference_weights/Qwen3.8-27B-4bit",
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
    #expect(!result.stdout.contains("setup.sh: linked"))
}

@Test
func setupRecoversMutationLockOwnedByDeadLocalProcess() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let lockDir = root.appendingPathComponent("cache.lock")

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        mkdir -p "${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
        printf 'token=dead-token pid=99999999 host=%s started_at=2000-01-01T00:00:00Z\n' "$(hostname)" \
          > "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"

        acquire_reference_cache_mutation_lock
        [[ "${REFERENCE_CACHE_MUTATION_LOCK_HELD}" == "1" ]]
        grep -q "token=${REFERENCE_CACHE_MUTATION_LOCK_TOKEN} pid=$$ " \
          "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"
        release_reference_cache_mutation_lock
        [[ ! -e "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": lockDir.path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS": "2",
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS": "60",
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
    #expect(result.stdout.contains("recovered stale reference cache mutation lock"))
}

@Test
func setupAllowsOnlyOneCompetingStaleLockRecoverer() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        mkdir -p "${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
        printf 'token=dead-token pid=99999999 host=%s started_at=2000-01-01T00:00:00Z\n' \
          "$(hostname)" > "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"
        : > "${RECOVERY_LOG}"

        pids=()
        for ordinal in 1 2 3 4 5 6 7 8; do
          (
            if recover_stale_reference_cache_mutation_lock; then
              printf '%s\n' "${ordinal}" >> "${RECOVERY_LOG}"
            fi
          ) &
          pids+=("$!")
        done
        for pid in "${pids[@]}"; do
          wait "${pid}" || true
        done

        [[ "$(wc -l < "${RECOVERY_LOG}" | tr -d ' ')" == "1" ]]
        [[ ! -e "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]
        acquire_reference_cache_mutation_lock
        grep -q "token=${REFERENCE_CACHE_MUTATION_LOCK_TOKEN}" \
          "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"
        release_reference_cache_mutation_lock
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "RECOVERY_LOG": root.appendingPathComponent("recoveries.log").path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache.lock").path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS": "2",
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupStalledLockCreatorCannotPublishIntoReplacementLock() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        ln() {
          local slot=""
          local status=0
          if mkdir "${FIRST_LINK_SLOT}" 2>/dev/null; then
            slot=first
            : > "${FIRST_STALLED}"
            while [[ ! -f "${RESUME_FIRST}" ]]; do sleep 0.05; done
          elif mkdir "${SECOND_LINK_SLOT}" 2>/dev/null; then
            slot=second
            : > "${SECOND_STALLED}"
            while [[ ! -f "${RESUME_SECOND}" ]]; do sleep 0.05; done
          fi
          command ln "$@" || status=$?
          if [[ "${slot}" == "first" ]]; then
            : > "${FIRST_ATTEMPTED}"
          elif [[ "${slot}" == "second" ]]; then
            : > "${SECOND_ATTEMPTED}"
          fi
          return "${status}"
        }

        (status=0
         if acquire_reference_cache_mutation_lock; then
           cat "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner" > "${FIRST_OWNER}"
           : > "${FIRST_ACQUIRED}"
           while [[ ! -f "${RELEASE_FIRST}" ]]; do sleep 0.05; done
           release_reference_cache_mutation_lock || status=$?
         else
           status=$?
         fi
         printf '%s\n' "${status}" > "${FIRST_STATUS}") &
        first_pid=$!
        while [[ ! -f "${FIRST_STALLED}" ]]; do sleep 0.05; done

        (status=0
         if acquire_reference_cache_mutation_lock; then
           cat "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner" > "${SECOND_OWNER}"
           : > "${SECOND_ACQUIRED}"
           while [[ ! -f "${RELEASE_SECOND}" ]]; do sleep 0.05; done
           release_reference_cache_mutation_lock || status=$?
         else
           status=$?
         fi
         printf '%s\n' "${status}" > "${SECOND_STATUS}") &
        second_pid=$!
        while [[ ! -f "${SECOND_STALLED}" ]]; do sleep 0.05; done

        : > "${RESUME_FIRST}"
        while [[ ! -f "${FIRST_ACQUIRED}" ]] && kill -0 "${first_pid}" 2>/dev/null; do
          sleep 0.05
        done
        if [[ ! -f "${FIRST_ACQUIRED}" ]]; then
          : > "${RESUME_SECOND}"
          : > "${RELEASE_FIRST}"
          : > "${RELEASE_SECOND}"
          wait "${first_pid}" || true
          wait "${second_pid}" || true
          exit 1
        fi
        : > "${RESUME_SECOND}"
        while [[ ! -f "${SECOND_ATTEMPTED}" ]]; do sleep 0.05; done

        : > "${RELEASE_FIRST}"
        wait "${first_pid}"
        [[ "$(cat "${FIRST_STATUS}")" == "0" ]]
        while [[ ! -f "${SECOND_ACQUIRED}" ]] && kill -0 "${second_pid}" 2>/dev/null; do
          sleep 0.05
        done
        [[ -f "${SECOND_ACQUIRED}" ]]
        ! cmp -s "${FIRST_OWNER}" "${SECOND_OWNER}"
        : > "${RELEASE_SECOND}"
        wait "${second_pid}"
        [[ "$(cat "${SECOND_STATUS}")" == "0" ]]
        [[ ! -e "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]
        [[ -z "$(find "$(dirname "${REFERENCE_CACHE_MUTATION_LOCK_DIR}")" \
          -maxdepth 1 -name "$(basename "${REFERENCE_CACHE_MUTATION_LOCK_DIR}").generation.*" -print -quit)" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache.lock").path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS": "1",
            "FIRST_LINK_SLOT": root.appendingPathComponent("first-link-slot").path,
            "SECOND_LINK_SLOT": root.appendingPathComponent("second-link-slot").path,
            "FIRST_STALLED": root.appendingPathComponent("first-stalled").path,
            "SECOND_STALLED": root.appendingPathComponent("second-stalled").path,
            "RESUME_FIRST": root.appendingPathComponent("resume-first").path,
            "RESUME_SECOND": root.appendingPathComponent("resume-second").path,
            "FIRST_ATTEMPTED": root.appendingPathComponent("first-attempted").path,
            "SECOND_ATTEMPTED": root.appendingPathComponent("second-attempted").path,
            "FIRST_STATUS": root.appendingPathComponent("first-status").path,
            "SECOND_STATUS": root.appendingPathComponent("second-status").path,
            "FIRST_OWNER": root.appendingPathComponent("first-owner").path,
            "SECOND_OWNER": root.appendingPathComponent("second-owner").path,
            "FIRST_ACQUIRED": root.appendingPathComponent("first-acquired").path,
            "SECOND_ACQUIRED": root.appendingPathComponent("second-acquired").path,
            "RELEASE_FIRST": root.appendingPathComponent("release-first").path,
            "RELEASE_SECOND": root.appendingPathComponent("release-second").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupReclaimsOnlyDeadOrBrokenAtomicLockGenerations() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        host_hash="$(printf '%s' "$(hostname)" | shasum -a 256 | awk '{print substr($1, 1, 16)}')"
        dead_pid=999999
        while kill -0 "${dead_pid}" 2>/dev/null; do dead_pid=$((dead_pid + 1)); done
        dead_generation="${REFERENCE_CACHE_MUTATION_LOCK_DIR}.generation.dead"
        live_generation="${REFERENCE_CACHE_MUTATION_LOCK_DIR}.generation.live"
        dead_initializing="${REFERENCE_CACHE_MUTATION_LOCK_DIR}.initializing.${host_hash}.${dead_pid}.dead"
        live_initializing="${REFERENCE_CACHE_MUTATION_LOCK_DIR}.initializing.${host_hash}.$$.live"
        mkdir "${dead_generation}" "${live_generation}" \
          "${dead_initializing}" "${live_initializing}"
        printf 'token=dead pid=%s host=%s started_at=now\n' "${dead_pid}" "$(hostname)" \
          > "${dead_generation}/owner"
        printf 'token=live pid=%s host=%s started_at=now\n' "$$" "$(hostname)" \
          > "${live_generation}/owner"
        broken_target="${REFERENCE_CACHE_MUTATION_LOCK_DIR}.generation.broken"
        ln -s -h -- "${broken_target}" "${REFERENCE_CACHE_MUTATION_LOCK_DIR}"

        acquire_reference_cache_mutation_lock
        [[ ! -e "${dead_generation}" ]]
        [[ ! -e "${dead_initializing}" ]]
        [[ -d "${live_generation}" ]]
        [[ -d "${live_initializing}" ]]
        release_reference_cache_mutation_lock
        [[ ! -e "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" && ! -L "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]
        rm -rf "${live_generation}" "${live_initializing}"
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache.lock").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupRecoversAgedMalformedMutationLockOwner() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        mkdir -p "${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
        : > "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"
        touch -t 200001010000.00 "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"

        acquire_reference_cache_mutation_lock
        [[ "${REFERENCE_CACHE_MUTATION_LOCK_HELD}" == "1" ]]
        grep -q "token=${REFERENCE_CACHE_MUTATION_LOCK_TOKEN}" \
          "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"
        release_reference_cache_mutation_lock
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache.lock").path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS": "2",
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS": "1",
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupRefusesToReleaseMutationLockWithDifferentOwnerToken() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        acquire_reference_cache_mutation_lock
        original_token="${REFERENCE_CACHE_MUTATION_LOCK_TOKEN}"
        printf 'token=other-token pid=%s host=%s started_at=now\n' "$$" "$(hostname)" \
          > "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"

        release_status=0
        release_reference_cache_mutation_lock || release_status=$?
        [[ "${release_status}" != "0" ]]
        [[ -d "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]

        printf 'token=%s pid=%s host=%s started_at=now\n' \
          "${original_token}" "$$" "$(hostname)" > "${REFERENCE_CACHE_MUTATION_LOCK_DIR}/owner"
        release_reference_cache_mutation_lock
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache.lock").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
    #expect(result.stderr.contains("owned by another process"))
}

@Test
func setupMutationLockWaiterRetriesWhenHolderReleasesDuringPublishAttempt() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        holder_generation="${REFERENCE_CACHE_MUTATION_LOCK_DIR}.generation.holder"
        mkdir "${holder_generation}"
        printf 'token=holder pid=%s host=%s started_at=now\n' "$$" "$(hostname)" \
          > "${holder_generation}/owner"
        ln -s -h -- "${holder_generation}" "${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
        lock_name="$(basename "${REFERENCE_CACHE_MUTATION_LOCK_DIR}")"

        # Deterministically land a concurrent holder's release inside the
        # waiter's race window: the waiter's first removal of its own
        # unpublished generation directory happens between its failed publish
        # attempt and the obstruction check, so dropping the published lock
        # there reproduces the transient absent-path state that used to be
        # misclassified as a fatal non-directory obstruction.
        rm() {
          if [[ ! -e "${RELEASE_SENTINEL}" && "${1:-}" == "-rf" \
              && "$(basename "${2:-none}")" == "${lock_name}.generation."* ]]; then
            : > "${RELEASE_SENTINEL}"
            command rm -f "${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
            command rm -rf "${holder_generation}"
          fi
          command rm "$@"
        }

        acquire_reference_cache_mutation_lock
        [[ "${REFERENCE_CACHE_MUTATION_LOCK_HELD}" == "1" ]]
        [[ -e "${RELEASE_SENTINEL}" ]]
        release_reference_cache_mutation_lock
        [[ ! -e "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" && ! -L "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "RELEASE_SENTINEL": root.appendingPathComponent("released").path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache-mutation.lock").path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_TIMEOUT_SECONDS": "10",
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
    #expect(result.stdout.contains("waiting for reference cache mutation lock"))
    #expect(result.stdout.contains("acquired reference cache mutation lock"))
    #expect(!result.stderr.contains("is not a lock directory"))
}

@Test
func setupRejectsZeroMutationLockStaleThreshold() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        lock_status=0
        acquire_reference_cache_mutation_lock || lock_status=$?
        [[ "${lock_status}" != "0" ]]
        [[ ! -e "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache.lock").path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_STALE_SECONDS": "0",
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
    #expect(result.stderr.contains("must be a positive integer"))
}

@Test
func setupUsesOneCanonicalMutationLockForReferenceSymlinkAliases() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference", isDirectory: true)
    let referenceAlias = root.appendingPathComponent("reference-alias", isDirectory: true)
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try FileManager.default.createSymbolicLink(
        at: referenceAlias,
        withDestinationURL: referenceDir
    )

    func mutationLockPath(reference: URL) throws -> SetupBashResult {
        try runSetupBash(
            """
            unset MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR
            eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
            printf '%s\n' "${REFERENCE_CACHE_MUTATION_LOCK_DIR}"
            """,
            environment: [
                "REPO_ROOT": FileManager.default.currentDirectoryPath,
                "MLXFAST_REFERENCE_DIR": reference.path,
            ]
        )
    }

    let direct = try mutationLockPath(reference: referenceDir)
    let aliased = try mutationLockPath(reference: referenceAlias)
    #expect(direct.status == 0, "stderr: \(direct.stderr)")
    #expect(aliased.status == 0, "stderr: \(aliased.stderr)")
    let directPath = direct.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
    let aliasedPath = aliased.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
    #expect(directPath == aliasedPath)
    #expect(directPath.hasSuffix("/reference.mlxfast-setup.lock"))
}

@Test
func setupReleasesReferenceCacheMutationLockWithoutOverwritingOperationFailure() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try "{}".write(
        to: referenceDir.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        verify_reference_weights() { return 1; }
        download_reference_weights_locked() { return 37; }

        operation_status=0
        download_reference_weights "${REFERENCE_DIR}" || operation_status=$?
        [[ "${operation_status}" == "37" ]]
        [[ ! -e "${REFERENCE_CACHE_MUTATION_LOCK_DIR}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache.lock").path,
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
}

@Test
func setupReferenceCacheStampUsesUniqueAtomicTemporaryFiles() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    let fixture = referenceDir.appendingPathComponent("config.json")
    try "fixture".write(to: fixture, atomically: true, encoding: .utf8)

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        fixture_hash="$(shasum -a 256 "${REFERENCE_DIR}/config.json" | awk '{print $1}')"
        fixture_size="$(wc -c < "${REFERENCE_DIR}/config.json" | tr -d ' ')"
        printf '%s %s config.json\n' "${fixture_hash}" "${fixture_size}" > "${REFERENCE_MANIFEST_PATH}"

        (acquire_reference_cache_mutation_lock
         verify_reference_manifest "${REFERENCE_DIR}"
         write_reference_cache_lock "${REFERENCE_DIR}"
         release_reference_cache_mutation_lock) &
        first_pid=$!
        (acquire_reference_cache_mutation_lock
         verify_reference_manifest "${REFERENCE_DIR}"
         write_reference_cache_lock "${REFERENCE_DIR}"
         release_reference_cache_mutation_lock) &
        second_pid=$!
        wait "${first_pid}"
        wait "${second_pid}"

        reference_cache_lock_is_current "${REFERENCE_DIR}"
        stamp_dir="$(dirname "${REFERENCE_CACHE_LOCK_PATH}")"
        ! compgen -G "${stamp_dir}/.mlxfast-reference-cache.lock.tmp.*" >/dev/null
        ! compgen -G "${stamp_dir}/.mlxfast-reference-cache.lock.files.*" >/dev/null
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_CACHE_LOCK_PATH": referenceDir.appendingPathComponent(".mlxfast-reference-cache.lock").path,
            "MLXFAST_REFERENCE_CACHE_MUTATION_LOCK_DIR": root.appendingPathComponent("cache-mutation.lock").path,
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
}

@Test
func setupFallsBackWhenPrimaryReferenceDownloadFails() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let fakeBin = root.appendingPathComponent("bin")
    try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("curl"),
        contents: """
        #!/usr/bin/env bash
        set -euo pipefail
        output=""
        url=""
        while [[ "$#" -gt 0 ]]; do
          case "$1" in
            --output) output="$2"; shift 2 ;;
            http*) url="$1"; shift ;;
            *) shift ;;
          esac
        done
        printf '%s\n' "${url}" >> "${CURL_LOG}"
        if [[ "${url}" == https://primary.invalid/* ]]; then
          exit 28
        fi
        printf 'fallback fixture' > "${output}"
        """
    )

    let output = root.appendingPathComponent("model.bin")
    let curlLog = root.appendingPathComponent("curl.log")
    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        download_reference_file "model.bin" "${OUTPUT_PATH}"
        [[ "$(cat "${OUTPUT_PATH}")" == "fallback fixture" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "PATH": "\(fakeBin.path):/usr/bin:/bin:/usr/sbin:/sbin",
            "CURL_LOG": curlLog.path,
            "OUTPUT_PATH": output.path,
            "MLXFAST_REFERENCE_BASE_URL": "https://primary.invalid",
            "MLXFAST_REFERENCE_FALLBACK_BASE_URL": "https://fallback.invalid",
            "MLXFAST_REFERENCE_HASH_VERIFY": "0",
            "MLXFAST_REFERENCE_DOWNLOAD_PROGRESS_SECONDS": "1",
            "MLXFAST_REFERENCE_DOWNLOAD_STALL_SECONDS": "1",
            "MLXFAST_REFERENCE_DOWNLOAD_MIN_BYTES_PER_SECOND": "1",
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
    let requests = try String(contentsOf: curlLog, encoding: .utf8)
    #expect(requests.contains("https://primary.invalid/model.bin"))
    #expect(requests.contains("https://fallback.invalid/model.bin"))
    #expect(result.stdout.contains("trying fallback source"))
}

@Test
func poolsideReferenceDownloadersUseSourceSpecificQueryHandling() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let fakeBin = root.appendingPathComponent("bin")
    try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("curl"),
        contents: """
        #!/usr/bin/env bash
        set -euo pipefail
        output=""
        url=""
        while [[ "$#" -gt 0 ]]; do
          case "$1" in
            --output) output="$2"; shift 2 ;;
            http*) url="$1"; shift ;;
            *) shift ;;
          esac
        done
        printf '%s\n' "${url}" >> "${CURL_LOG}"
        if [[ "${url}" == https://ds4.darkbloom.ai/* ]]; then
          exit 28
        fi
        printf 'fallback fixture' > "${output}"
        """
    )

    let curlLog = root.appendingPathComponent("curl.log")
    let output = root.appendingPathComponent("setup/config.json")
    let cache = root.appendingPathComponent("cache")
    let manifest = root.appendingPathComponent("manifest.sha256")
    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        download_reference_file "config.json" "${OUTPUT_PATH}"
        [[ "$(cat "${OUTPUT_PATH}")" == "fallback fixture" ]]

        fixture_hash="$(printf 'fallback fixture' | shasum -a 256 | awk '{print $1}')"
        printf '%s 16 config.json\n' "${fixture_hash}" > "${REFERENCE_MANIFEST_PATH}"
        "${REPO_ROOT}/.github/scripts/download-reference-cache-scope.sh" metadata
        [[ "$(cat "${REFERENCE_DIR}/config.json")" == "fallback fixture" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "PATH": "\(fakeBin.path):/usr/bin:/bin:/usr/sbin:/sbin",
            "CURL_LOG": curlLog.path,
            "OUTPUT_PATH": output.path,
            "MLXFAST_REFERENCE_DIR": cache.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": manifest.path,
            "MLXFAST_REFERENCE_HASH_VERIFY": "0",
            "MLXFAST_REFERENCE_DOWNLOAD_PROGRESS_SECONDS": "1",
            "MLXFAST_REFERENCE_DOWNLOAD_STALL_SECONDS": "1",
            "MLXFAST_REFERENCE_DOWNLOAD_MIN_BYTES_PER_SECOND": "1",
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
    let requests = try String(contentsOf: curlLog, encoding: .utf8)
        .split(separator: "\n")
        .map(String.init)
    // The behaviour under test is source-specific query handling: only
    // huggingface.co URLs get "?download=true" appended. Two downloaders are
    // exercised and they no longer share an identity on this branch --
    // setup.sh resolves the adopted Qwen 3.8 revision on Hugging Face directly (no
    // mirror exists for it yet, so there is no R2 leg and no fallback), while
    // .github/scripts/download-reference-cache-scope.sh still carries the
    // Laguna R2-then-Hugging-Face pair. Re-aiming that script at Qwen is
    // Phase 5 workflow work; the query-handling contract asserted here holds
    // for both sources exactly as before.
    let qwenHF =
        "https://huggingface.co/EigenLabs/Qwen3.8-27B-4bit/resolve/"
        + "eda45ab47f465d08d6558f0353a2346e2eb9d5b3/config.json?download=true"
    let r2 = "https://ds4.darkbloom.ai/laguna-xs-2.1-nvfp4-mlx/config.json"
    let hf =
        "https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4-mlx/resolve/"
        + "841778bda563a36104dd521e37d99218e46f4f25/config.json?download=true"
    #expect(requests == [qwenHF, r2, hf])
    #expect(!requests.contains("\(r2)?download=true"))
}

@Test
func setupDownloadContractReportsProgressAndDetectsStalls() throws {
    let setup = try String(contentsOfFile: "setup.sh", encoding: .utf8)

    #expect(setup.contains("MLXFAST_REFERENCE_DOWNLOAD_PROGRESS_SECONDS"))
    #expect(setup.contains("MLXFAST_REFERENCE_DOWNLOAD_STALL_SECONDS"))
    #expect(setup.contains("MLXFAST_REFERENCE_DOWNLOAD_MIN_BYTES_PER_SECOND"))
    #expect(setup.contains("--speed-limit"))
    #expect(setup.contains("--speed-time"))
    // The parallel shard phase has exactly one progress reporter: the
    // aggregate heartbeat, which also carries the transfer rate and ETA.
    #expect(setup.contains("still downloading safetensors shard(s)"))
    #expect(setup.contains("rate=%.1f MiB/s eta=%ds"))
    // The heartbeat's byte counter includes in-flight .partial transfers, so
    // progress moves from the first transferred byte instead of printing 0%
    // until a whole shard completes and is renamed.
    #expect(setup.contains("elif [[ -f \"${file_path}.partial\" ]]; then"))
}

@Test
func setupDownloadHeartbeatCountsInProgressPartialBytes() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let output = root.appendingPathComponent("output")
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)

    // Shard A is mid-transfer (curl still writing the .partial); shard B has
    // completed (renamed to its final name) and also has a stale leftover
    // .partial, which must NOT be double counted. Expected: 7 + 5 = 12.
    try Data(repeating: 0x61, count: 7).write(
        to: output.appendingPathComponent("file-a.safetensors.partial")
    )
    try Data(repeating: 0x62, count: 5).write(
        to: output.appendingPathComponent("file-b.safetensors")
    )
    try Data(repeating: 0x63, count: 9).write(
        to: output.appendingPathComponent("file-b.safetensors.partial")
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        reference_download_on_disk_bytes "${OUTPUT_DIR}" file-a.safetensors file-b.safetensors
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "OUTPUT_DIR": output.path,
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
    let lastLine = result.stdout
        .split(separator: "\n", omittingEmptySubsequences: true)
        .last
        .map(String.init)
    #expect(lastLine == "12", "stdout: \(result.stdout)")
}

@Test
func setupParallelShardDownloadsEmitProgress() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let fakeBin = root.appendingPathComponent("bin")
    let output = root.appendingPathComponent("output")
    try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("curl"),
        contents: """
        #!/usr/bin/env bash
        set -euo pipefail
        output=""
        while [[ "$#" -gt 0 ]]; do
          case "$1" in
            --output) output="$2"; shift 2 ;;
            *) shift ;;
          esac
        done
        for chunk in 1 2 3 4; do
          printf 'chunk-%s\n' "${chunk}" >> "${output}"
          sleep 0.45
        done
        """
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        download_reference_shards "${OUTPUT_DIR}" file-a.safetensors file-b.safetensors
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "PATH": "\(fakeBin.path):/usr/bin:/bin:/usr/sbin:/sbin",
            "OUTPUT_DIR": output.path,
            "MLXFAST_REFERENCE_BASE_URL": "https://primary.invalid",
            "MLXFAST_REFERENCE_FALLBACK_BASE_URL": "",
            "MLXFAST_REFERENCE_HASH_VERIFY": "0",
            "MLXFAST_REFERENCE_DOWNLOAD_JOBS": "2",
            "MLXFAST_REFERENCE_DOWNLOAD_PROGRESS_SECONDS": "1",
            "MLXFAST_REFERENCE_DOWNLOAD_STALL_SECONDS": "5",
            "MLXFAST_REFERENCE_DOWNLOAD_MIN_BYTES_PER_SECOND": "1",
        ]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
    #expect(result.stdout.contains("still downloading safetensors shard(s)"))
    #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent("file-a.safetensors").path))
    #expect(FileManager.default.fileExists(atPath: output.appendingPathComponent("file-b.safetensors").path))
}

@Test
func setupRejectsCommandLineToolsWithFullXcodeInstructions() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let fakeBin = root.appendingPathComponent("bin")
    try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)

    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("swift"),
        contents: """
        #!/usr/bin/env bash
        exit 0
        """
    )
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("xcodebuild"),
        contents: """
        #!/usr/bin/env bash
        echo "xcode-select: error: tool 'xcodebuild' requires Xcode, but active developer directory '/Library/Developer/CommandLineTools' is a command line tools instance" >&2
        exit 1
        """
    )
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("xcode-select"),
        contents: """
        #!/usr/bin/env bash
        printf '/Library/Developer/CommandLineTools\n'
        """
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        ensure_swift_toolchain
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "PATH": "\(fakeBin.path):/usr/bin:/bin:/usr/sbin:/sbin",
        ]
    )

    #expect(result.status != 0)
    #expect(result.stderr.contains("full Xcode is required"))
    #expect(result.stderr.contains("Command Line Tools alone"))
    #expect(result.stderr.contains("sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"))
}

@Test
func setupReferenceCacheStampRejectsSameSizeImmediateMutation() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try "fixture".write(
        to: referenceDir.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        fixture_hash="$(shasum -a 256 "${REFERENCE_DIR}/config.json" | awk '{print $1}')"
        printf '%s 7 config.json\n' "${fixture_hash}" > "${REFERENCE_MANIFEST_PATH}"
        verify_reference_manifest "${REFERENCE_DIR}"
        write_reference_cache_lock "${REFERENCE_DIR}"
        reference_cache_lock_is_current "${REFERENCE_DIR}"

        printf 'changed' > "${REFERENCE_DIR}/config.json"
        if reference_cache_lock_is_current "${REFERENCE_DIR}"; then
          echo "cache stamp accepted changed content" >&2
          exit 1
        fi
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_CACHE_LOCK_PATH": referenceDir.appendingPathComponent(".mlxfast-reference-cache.lock").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupReferenceCacheStampRejectsMutationAfterHashBeforePublication() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try "fixture".write(
        to: referenceDir.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        fixture_hash="$(shasum -a 256 "${REFERENCE_DIR}/config.json" | awk '{print $1}')"
        printf '%s 7 config.json\n' "${fixture_hash}" > "${REFERENCE_MANIFEST_PATH}"
        verify_reference_manifest "${REFERENCE_DIR}"

        printf 'changed' > "${REFERENCE_DIR}/config.json"
        if write_reference_cache_lock "${REFERENCE_DIR}"; then
          echo "cache stamp accepted bytes changed after hash verification" >&2
          exit 1
        fi
        [[ ! -e "${REFERENCE_CACHE_LOCK_PATH}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_CACHE_LOCK_PATH": root.appendingPathComponent("cache.stamp").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupReferenceCacheStampRejectsManifestSwapAfterVerification() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try "fixture".write(
        to: referenceDir.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        fixture_hash="$(shasum -a 256 "${REFERENCE_DIR}/config.json" | awk '{print $1}')"
        printf '%s 7 config.json\n' "${fixture_hash}" > "${REFERENCE_MANIFEST_PATH}"
        verify_reference_manifest "${REFERENCE_DIR}"

        printf '%s 7 config.json\n' \
          '0000000000000000000000000000000000000000000000000000000000000000' \
          > "${REFERENCE_MANIFEST_PATH}"
        if write_reference_cache_lock "${REFERENCE_DIR}"; then
          echo "cache stamp accepted a different manifest" >&2
          exit 1
        fi
        [[ ! -e "${REFERENCE_CACHE_LOCK_PATH}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_CACHE_LOCK_PATH": root.appendingPathComponent("cache.stamp").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupReferenceCacheStampRejectsManifestSwapDuringFastValidation() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try "fixture".write(
        to: referenceDir.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        fixture_hash="$(printf fixture | shasum -a 256 | awk '{print $1}')"
        printf '%s 7 config.json\n' "${fixture_hash}" > "${REFERENCE_MANIFEST_PATH}"
        printf '%s 7 config.json\n' \
          '0000000000000000000000000000000000000000000000000000000000000000' \
          > "${MANIFEST_B}"
        verify_reference_manifest "${REFERENCE_DIR}"
        write_reference_cache_lock "${REFERENCE_DIR}"

        file_metadata_signature() {
          if mkdir "${SWAP_ONCE}" 2>/dev/null; then
            cp "${MANIFEST_B}" "${REFERENCE_MANIFEST_PATH}"
          fi
          stat -f '%d:%i:%z:%Fm:%Fc:%v' "$1" 2>/dev/null \
            || stat -c '%d:%i:%s:%Y:%Z:0' "$1"
        }
        if reference_cache_lock_is_current "${REFERENCE_DIR}"; then
          echo "cache stamp accepted a manifest swapped during validation" >&2
          exit 1
        fi
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_CACHE_LOCK_PATH": root.appendingPathComponent("cache.stamp").path,
            "MANIFEST_B": root.appendingPathComponent("manifest-b.sha256").path,
            "SWAP_ONCE": root.appendingPathComponent("swap-once").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
    #expect(result.stderr.contains("manifest changed while the cache lock was being validated"))
}

@Test
func setupReferenceVerificationRejectsManifestABA() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try "fixture".write(
        to: referenceDir.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        fixture_hash="$(printf 'fixture' | shasum -a 256 | awk '{print $1}')"
        changed_hash="$(printf 'changed' | shasum -a 256 | awk '{print $1}')"
        printf '%s 7 config.json\n' "${fixture_hash}" > "${REFERENCE_MANIFEST_PATH}"
        cp "${REFERENCE_MANIFEST_PATH}" "${MANIFEST_A}"

        eval "$(declare -f create_reference_verified_signatures_path \
          | sed '1s/create_reference_verified_signatures_path/create_reference_verified_signatures_path_original/')"
        create_reference_verified_signatures_path() {
          create_reference_verified_signatures_path_original
          printf '%s 7 config.json\n' "${changed_hash}" > "${REFERENCE_MANIFEST_PATH}"
          printf 'changed' > "${REFERENCE_DIR}/config.json"
        }
        file_metadata_signature() {
          count=0
          [[ -f "${SIGNATURE_COUNT}" ]] && count="$(cat "${SIGNATURE_COUNT}")"
          count=$((count + 1))
          printf '%s\n' "${count}" > "${SIGNATURE_COUNT}"
          if [[ "${count}" == "2" ]]; then
            cp "${MANIFEST_A}" "${REFERENCE_MANIFEST_PATH}"
          fi
          stat -f '%d:%i:%z:%Fm:%Fc:%v' "$1"
        }

        if verify_reference_manifest "${REFERENCE_DIR}" \
            && write_reference_cache_lock "${REFERENCE_DIR}"; then
          echo "cache stamp accepted A/B/A manifest verification" >&2
          exit 1
        fi
        [[ ! -e "${REFERENCE_CACHE_LOCK_PATH}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_CACHE_LOCK_PATH": root.appendingPathComponent("cache.stamp").path,
            "MANIFEST_A": root.appendingPathComponent("manifest-a.sha256").path,
            "SIGNATURE_COUNT": root.appendingPathComponent("signature-count").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupVerifiedDownloadMarkerRejectsManifestABA() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    try "fixture".write(
        to: referenceDir.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        fixture_hash="$(printf 'fixture' | shasum -a 256 | awk '{print $1}')"
        changed_hash="$(printf 'changed' | shasum -a 256 | awk '{print $1}')"
        printf '%s 7 config.json\n' "${fixture_hash}" > "${REFERENCE_MANIFEST_PATH}"
        cp "${REFERENCE_MANIFEST_PATH}" "${MANIFEST_A}"
        printf '%s 7 config.json\n' "${changed_hash}" > "${REFERENCE_MANIFEST_PATH}"
        printf 'changed' > "${REFERENCE_DIR}/config.json"

        file_metadata_signature() {
          count=0
          [[ -f "${SIGNATURE_COUNT}" ]] && count="$(cat "${SIGNATURE_COUNT}")"
          count=$((count + 1))
          printf '%s\n' "${count}" > "${SIGNATURE_COUNT}"
          if [[ "${count}" == "2" ]]; then
            cp "${MANIFEST_A}" "${REFERENCE_MANIFEST_PATH}"
          fi
          stat -f '%d:%i:%z:%Fm:%Fc:%v' "$1"
        }

        marker="${REFERENCE_DIR}/config.json.complete"
        if reference_file_is_current config.json "${REFERENCE_DIR}/config.json" \
            && mark_reference_file_complete \
              config.json "${REFERENCE_DIR}/config.json" "${marker}" \
              "${REFERENCE_VERIFIED_CONTENT_IDENTITY}" \
              "${REFERENCE_VERIFIED_EXPECTED_HASH}" \
              "${REFERENCE_VERIFIED_EXPECTED_SIZE}" \
              "${REFERENCE_VERIFIED_FILE_MANIFEST_HASH}"; then
          echo "verified marker accepted A/B/A manifest verification" >&2
          exit 1
        fi
        [[ ! -e "${marker}" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_CACHE_LOCK_PATH": root.appendingPathComponent("cache.stamp").path,
            "MANIFEST_A": root.appendingPathComponent("manifest-a.sha256").path,
            "SIGNATURE_COUNT": root.appendingPathComponent("signature-count").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupReferenceCacheStampDoesNotModifyReadOnlyCheckpoint() throws {
    let root = try setupTemporaryDirectory()
    let referenceDir = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: referenceDir, withIntermediateDirectories: true)
    let fixture = referenceDir.appendingPathComponent("config.json")
    try "fixture".write(to: fixture, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o444],
        ofItemAtPath: fixture.path
    )
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o555],
        ofItemAtPath: referenceDir.path
    )
    defer {
        try? FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: referenceDir.path
        )
        try? FileManager.default.setAttributes(
            [.posixPermissions: 0o644],
            ofItemAtPath: fixture.path
        )
        try? FileManager.default.removeItem(at: root)
    }

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        fixture_hash="$(shasum -a 256 "${REFERENCE_DIR}/config.json" | awk '{print $1}')"
        printf '%s 7 config.json\n' "${fixture_hash}" > "${REFERENCE_MANIFEST_PATH}"

        before_signature="$(file_metadata_signature "${REFERENCE_DIR}/config.json")"
        verify_reference_manifest "${REFERENCE_DIR}"
        write_reference_cache_lock "${REFERENCE_DIR}"
        after_signature="$(file_metadata_signature "${REFERENCE_DIR}/config.json")"

        [[ "${before_signature}" == "${after_signature}" ]]
        reference_cache_lock_is_current "${REFERENCE_DIR}"
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_REFERENCE_DIR": referenceDir.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_CACHE_LOCK_PATH": root.appendingPathComponent("cache.stamp").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupDownloadPublishesVerifiedPartialAtomically() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let published = root.appendingPathComponent("config.json")
    try "old-data".write(to: published, atomically: true, encoding: .utf8)

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        new_hash="$(printf 'new-data' | shasum -a 256 | awk '{print $1}')"
        printf '%s 8 config.json\n' "${new_hash}" > "${REFERENCE_MANIFEST_PATH}"
        curl() {
          output=""
          while [[ "$#" -gt 0 ]]; do
            if [[ "$1" == "--output" ]]; then
              output="$2"
              shift 2
            else
              shift
            fi
          done
          [[ "${output}" == "${PUBLISHED_PATH}.partial" ]]
          [[ "$(cat "${PUBLISHED_PATH}")" == "old-data" ]]
          printf 'new-data' > "${output}"
        }

        download_reference_file config.json "${PUBLISHED_PATH}"
        [[ "$(cat "${PUBLISHED_PATH}")" == "new-data" ]]
        [[ ! -e "${PUBLISHED_PATH}.partial" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "PUBLISHED_PATH": published.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_BASE_URL": "https://invalid.example",
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupFailedDownloadPreservesPreviouslyPublishedFile() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let published = root.appendingPathComponent("config.json")
    try "old-data".write(to: published, atomically: true, encoding: .utf8)

    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        new_hash="$(printf 'new-data' | shasum -a 256 | awk '{print $1}')"
        printf '%s 8 config.json\n' "${new_hash}" > "${REFERENCE_MANIFEST_PATH}"
        curl() {
          output=""
          while [[ "$#" -gt 0 ]]; do
            if [[ "$1" == "--output" ]]; then
              output="$2"
              shift 2
            else
              shift
            fi
          done
          printf 'partial' > "${output}"
          return 22
        }

        download_status=0
        download_reference_file config.json "${PUBLISHED_PATH}" || download_status=$?
        [[ "${download_status}" != "0" ]]
        [[ "$(cat "${PUBLISHED_PATH}")" == "old-data" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "PUBLISHED_PATH": published.path,
            "MLXFAST_REFERENCE_MANIFEST_PATH": root.appendingPathComponent("manifest.sha256").path,
            "MLXFAST_REFERENCE_BASE_URL": "https://invalid.example",
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func setupUsesDownloadedMetalToolchainIdentifierForCompilerProbe() throws {
    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        xcodebuild() {
          printf 'Toolchain Identifier: com.apple.dt.toolchain.Metal.1\n'
        }
        xcrun() {
          [[ "${TOOLCHAINS:-}" == "com.apple.dt.toolchain.Metal.1" ]]
        }
        unset TOOLCHAINS
        metal_compiler_is_available
        [[ "${TOOLCHAINS}" == "com.apple.dt.toolchain.Metal.1" ]]
        """,
        environment: ["REPO_ROOT": FileManager.default.currentDirectoryPath]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
}

@Test
func setupDetectedMetalToolchainOverridesUnrelatedToolchainsEnvironment() throws {
    let result = try runSetupBash(
        """
        eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
        xcodebuild() {
          printf 'Toolchain Identifier: com.apple.dt.toolchain.Metal.1\n'
        }
        xcrun() {
          [[ "${TOOLCHAINS:-}" == "com.apple.dt.toolchain.Metal.1" ]]
        }
        export TOOLCHAINS="org.swift.unrelated"
        metal_compiler_is_available
        """,
        environment: ["REPO_ROOT": FileManager.default.currentDirectoryPath]
    )

    #expect(result.status == 0, "stderr: \(result.stderr)")
}

@Test
func metallibBuilderRejectsAmbiguousCMakeOutputs() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let fakeBin = root.appendingPathComponent("bin")
    let checkout = root.appendingPathComponent("mlx-swift")
    try writeSetupVendoredFixture(at: checkout)
    try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)

    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("cmake"),
        contents: """
        #!/usr/bin/env bash
        set -euo pipefail
        printf '%s\n' "$*" >> "${CMAKE_ARGUMENT_LOG}"
        if [[ " $* " == *" --build "* ]]; then
          mkdir -p "${MLXFAST_MLX_METAL_BUILD_DIR}/first" "${MLXFAST_MLX_METAL_BUILD_DIR}/second"
          printf 'first' > "${MLXFAST_MLX_METAL_BUILD_DIR}/first/mlx.metallib"
          printf 'second' > "${MLXFAST_MLX_METAL_BUILD_DIR}/second/mlx.metallib"
        fi
        """
    )
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("xcodebuild"),
        contents: """
        #!/usr/bin/env bash
        printf 'Toolchain Identifier: com.apple.dt.toolchain.Metal.1\n'
        """
    )
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("xcrun"),
        contents: """
        #!/usr/bin/env bash
        [[ "${TOOLCHAINS:-}" == "com.apple.dt.toolchain.Metal.1" ]]
        """
    )

    let output = root.appendingPathComponent("output/mlx.metallib")
    let result = try runSetupBash(
        """
        "${REPO_ROOT}/tools/build-mlx-metallib.sh"
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "PATH": "\(fakeBin.path):/usr/bin:/bin",
            "MLXFAST_CMAKE_BIN": fakeBin.appendingPathComponent("cmake").path,
            "MLXFAST_MLX_SWIFT_VENDOR": checkout.path,
            "MLXFAST_MLX_METAL_BUILD_DIR": root.appendingPathComponent("metal-build").path,
            "MLXFAST_MLX_METALLIB": output.path,
            "CMAKE_ARGUMENT_LOG": root.appendingPathComponent("cmake-arguments.log").path,
            "TOOLCHAINS": "org.swift.unrelated",
        ]
    )

    #expect(result.status != 0)
    #expect(result.stderr.contains("produced multiple mlx.metallib files"))
    #expect(!FileManager.default.fileExists(atPath: output.path))
    let arguments = try String(
        contentsOf: root.appendingPathComponent("cmake-arguments.log"),
        encoding: .utf8
    )
    #expect(
        arguments.contains(
            "/mlx-swift/Source/Cmlx/metal-cpp"
        )
    )
    #expect(!arguments.contains("\(FileManager.default.currentDirectoryPath)/\(checkout.path)"))
}

@Test
func metallibBuilderSerializesConcurrentProcesses() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let fakeBin = root.appendingPathComponent("bin")
    let checkout = root.appendingPathComponent("mlx-swift")
    try writeSetupVendoredFixture(at: checkout)
    try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)

    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("cmake"),
        contents: """
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ " $* " == *" --build "* ]]; then
          if ! mkdir "${ACTIVE_BUILD_DIR}" 2>/dev/null; then
            : > "${OVERLAP_LOG}"
          else
            sleep 0.3
            rmdir "${ACTIVE_BUILD_DIR}"
          fi
          mkdir -p "${MLXFAST_MLX_METAL_BUILD_DIR}/result"
          printf 'metallib' > "${MLXFAST_MLX_METAL_BUILD_DIR}/result/mlx.metallib"
        fi
        """
    )
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("xcodebuild"),
        contents: "#!/usr/bin/env bash\nprintf 'Toolchain Identifier: metal.test\\n'\n"
    )
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("xcrun"),
        contents: "#!/usr/bin/env bash\nexit 0\n"
    )

    let output = root.appendingPathComponent("output/mlx.metallib")
    let result = try runSetupBash(
        """
        "${REPO_ROOT}/tools/build-mlx-metallib.sh" &
        first_pid=$!
        "${REPO_ROOT}/tools/build-mlx-metallib.sh" &
        second_pid=$!
        wait "${first_pid}"
        wait "${second_pid}"
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "PATH": "\(fakeBin.path):/usr/bin:/bin",
            "MLXFAST_CMAKE_BIN": fakeBin.appendingPathComponent("cmake").path,
            "MLXFAST_MLX_SWIFT_VENDOR": checkout.path,
            "MLXFAST_MLX_METAL_BUILD_DIR": root.appendingPathComponent("metal-build").path,
            "MLXFAST_MLX_METALLIB": output.path,
            "ACTIVE_BUILD_DIR": root.appendingPathComponent("active-build").path,
            "OVERLAP_LOG": root.appendingPathComponent("overlap").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
    #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("overlap").path))
    #expect(try String(contentsOf: output, encoding: .utf8) == "metallib")
    // The publish also records the vendored-source fingerprint sidecar the
    // ranked cache and the trusted CLI verify against.
    let record = try String(
        contentsOf: root.appendingPathComponent("output/mlx.metallib.fingerprint"),
        encoding: .utf8
    )
    #expect(record.hasPrefix("mlxfast-metallib-fingerprint-v1 "))
    #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("metal-build.mlxfast-build.lock").path))
}

@Test
func metallibBuilderStalledLockCreatorCannotPublishIntoReplacementLock() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed \
          -e 's|^ROOT_DIR=.*|ROOT_DIR="${REPO_ROOT}"|' \
          -e '/^METAL_TOOLCHAIN_IDENTIFIER=/,$d' \
          "${REPO_ROOT}/tools/build-mlx-metallib.sh")"
        ln() {
          local slot=""
          local status=0
          if mkdir "${FIRST_LINK_SLOT}" 2>/dev/null; then
            slot=first
            : > "${FIRST_STALLED}"
            while [[ ! -f "${RESUME_FIRST}" ]]; do sleep 0.05; done
          elif mkdir "${SECOND_LINK_SLOT}" 2>/dev/null; then
            slot=second
            : > "${SECOND_STALLED}"
            while [[ ! -f "${RESUME_SECOND}" ]]; do sleep 0.05; done
          fi
          command ln "$@" || status=$?
          if [[ "${slot}" == "first" ]]; then
            : > "${FIRST_ATTEMPTED}"
          elif [[ "${slot}" == "second" ]]; then
            : > "${SECOND_ATTEMPTED}"
          fi
          return "${status}"
        }

        (status=0
         if acquire_build_lock; then
           cat "${BUILD_LOCK_DIR}/owner" > "${FIRST_OWNER}"
           : > "${FIRST_ACQUIRED}"
           while [[ ! -f "${RELEASE_FIRST}" ]]; do sleep 0.05; done
           release_build_lock || status=$?
         else
           status=$?
         fi
         printf '%s\n' "${status}" > "${FIRST_STATUS}") &
        first_pid=$!
        while [[ ! -f "${FIRST_STALLED}" ]]; do sleep 0.05; done

        (status=0
         if acquire_build_lock; then
           cat "${BUILD_LOCK_DIR}/owner" > "${SECOND_OWNER}"
           : > "${SECOND_ACQUIRED}"
           while [[ ! -f "${RELEASE_SECOND}" ]]; do sleep 0.05; done
           release_build_lock || status=$?
         else
           status=$?
         fi
         printf '%s\n' "${status}" > "${SECOND_STATUS}") &
        second_pid=$!
        while [[ ! -f "${SECOND_STALLED}" ]]; do sleep 0.05; done

        : > "${RESUME_FIRST}"
        while [[ ! -f "${FIRST_ACQUIRED}" ]] && kill -0 "${first_pid}" 2>/dev/null; do
          sleep 0.05
        done
        if [[ ! -f "${FIRST_ACQUIRED}" ]]; then
          : > "${RESUME_SECOND}"
          : > "${RELEASE_FIRST}"
          : > "${RELEASE_SECOND}"
          wait "${first_pid}" || true
          wait "${second_pid}" || true
          exit 1
        fi
        : > "${RESUME_SECOND}"
        while [[ ! -f "${SECOND_ATTEMPTED}" ]]; do sleep 0.05; done

        : > "${RELEASE_FIRST}"
        wait "${first_pid}"
        [[ "$(cat "${FIRST_STATUS}")" == "0" ]]
        while [[ ! -f "${SECOND_ACQUIRED}" ]] && kill -0 "${second_pid}" 2>/dev/null; do
          sleep 0.05
        done
        [[ -f "${SECOND_ACQUIRED}" ]]
        ! cmp -s "${FIRST_OWNER}" "${SECOND_OWNER}"
        : > "${RELEASE_SECOND}"
        wait "${second_pid}"
        [[ "$(cat "${SECOND_STATUS}")" == "0" ]]
        [[ ! -e "${BUILD_LOCK_DIR}" ]]
        [[ -z "$(find "$(dirname "${BUILD_LOCK_DIR}")" \
          -maxdepth 1 -name "$(basename "${BUILD_LOCK_DIR}").generation.*" -print -quit)" ]]
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_MLX_METAL_BUILD_DIR": root.appendingPathComponent("metal-build").path,
            "MLXFAST_MLX_METAL_BUILD_LOCK_DIR": root.appendingPathComponent("metal-build.lock").path,
            "MLXFAST_MLX_METAL_BUILD_LOCK_STALE_SECONDS": "1",
            "MLXFAST_MLX_METALLIB": root.appendingPathComponent("mlx.metallib").path,
            "MLXFAST_CLANG_MODULE_CACHE": root.appendingPathComponent("clang-cache").path,
            "MLXFAST_METAL_COMPILER_HOME": root.appendingPathComponent("metal-home").path,
            "FIRST_LINK_SLOT": root.appendingPathComponent("first-link-slot").path,
            "SECOND_LINK_SLOT": root.appendingPathComponent("second-link-slot").path,
            "FIRST_STALLED": root.appendingPathComponent("first-stalled").path,
            "SECOND_STALLED": root.appendingPathComponent("second-stalled").path,
            "RESUME_FIRST": root.appendingPathComponent("resume-first").path,
            "RESUME_SECOND": root.appendingPathComponent("resume-second").path,
            "FIRST_ATTEMPTED": root.appendingPathComponent("first-attempted").path,
            "SECOND_ATTEMPTED": root.appendingPathComponent("second-attempted").path,
            "FIRST_STATUS": root.appendingPathComponent("first-status").path,
            "SECOND_STATUS": root.appendingPathComponent("second-status").path,
            "FIRST_OWNER": root.appendingPathComponent("first-owner").path,
            "SECOND_OWNER": root.appendingPathComponent("second-owner").path,
            "FIRST_ACQUIRED": root.appendingPathComponent("first-acquired").path,
            "SECOND_ACQUIRED": root.appendingPathComponent("second-acquired").path,
            "RELEASE_FIRST": root.appendingPathComponent("release-first").path,
            "RELEASE_SECOND": root.appendingPathComponent("release-second").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func metallibBuilderReclaimsOnlyDeadOrBrokenAtomicLockGenerations() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let result = try runSetupBash(
        """
        eval "$(sed \
          -e 's|^ROOT_DIR=.*|ROOT_DIR="${REPO_ROOT}"|' \
          -e '/^METAL_TOOLCHAIN_IDENTIFIER=/,$d' \
          "${REPO_ROOT}/tools/build-mlx-metallib.sh")"
        host_hash="$(printf '%s' "$(hostname)" | shasum -a 256 | awk '{print substr($1, 1, 16)}')"
        dead_pid=999999
        while kill -0 "${dead_pid}" 2>/dev/null; do dead_pid=$((dead_pid + 1)); done
        dead_generation="${BUILD_LOCK_DIR}.generation.dead"
        live_generation="${BUILD_LOCK_DIR}.generation.live"
        dead_initializing="${BUILD_LOCK_DIR}.initializing.${host_hash}.${dead_pid}.dead"
        live_initializing="${BUILD_LOCK_DIR}.initializing.${host_hash}.$$.live"
        mkdir "${dead_generation}" "${live_generation}" \
          "${dead_initializing}" "${live_initializing}"
        printf 'token=dead pid=%s host=%s started_at=now\n' "${dead_pid}" "$(hostname)" \
          > "${dead_generation}/owner"
        printf 'token=live pid=%s host=%s started_at=now\n' "$$" "$(hostname)" \
          > "${live_generation}/owner"
        broken_target="${BUILD_LOCK_DIR}.generation.broken"
        ln -s -h -- "${broken_target}" "${BUILD_LOCK_DIR}"

        acquire_build_lock
        [[ ! -e "${dead_generation}" ]]
        [[ ! -e "${dead_initializing}" ]]
        [[ -d "${live_generation}" ]]
        [[ -d "${live_initializing}" ]]
        release_build_lock
        [[ ! -e "${BUILD_LOCK_DIR}" && ! -L "${BUILD_LOCK_DIR}" ]]
        rm -rf "${live_generation}" "${live_initializing}"
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "MLXFAST_MLX_METAL_BUILD_DIR": root.appendingPathComponent("metal-build").path,
            "MLXFAST_MLX_METAL_BUILD_LOCK_DIR": root.appendingPathComponent("metal-build.lock").path,
            "MLXFAST_MLX_METALLIB": root.appendingPathComponent("mlx.metallib").path,
            "MLXFAST_CLANG_MODULE_CACHE": root.appendingPathComponent("clang-cache").path,
            "MLXFAST_METAL_COMPILER_HOME": root.appendingPathComponent("metal-home").path,
        ]
    )

    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
}

@Test
func metallibBuilderPreservesPublishedOutputWhenCopyFails() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let fakeBin = root.appendingPathComponent("bin")
    let checkout = root.appendingPathComponent("mlx-swift")
    try writeSetupVendoredFixture(at: checkout)
    try FileManager.default.createDirectory(at: fakeBin, withIntermediateDirectories: true)

    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("cmake"),
        contents: """
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ " $* " == *" --build "* ]]; then
          mkdir -p "${MLXFAST_MLX_METAL_BUILD_DIR}/result"
          printf 'replacement' > "${MLXFAST_MLX_METAL_BUILD_DIR}/result/mlx.metallib"
        fi
        """
    )
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("xcodebuild"),
        contents: "#!/usr/bin/env bash\nprintf 'Toolchain Identifier: metal.test\\n'\n"
    )
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("xcrun"),
        contents: "#!/usr/bin/env bash\nexit 0\n"
    )
    try writeSetupExecutable(
        at: fakeBin.appendingPathComponent("cp"),
        contents: "#!/usr/bin/env bash\nprintf 'partial' > \"$2\"\nexit 23\n"
    )

    let output = root.appendingPathComponent("output/mlx.metallib")
    try FileManager.default.createDirectory(
        at: output.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try "original".write(to: output, atomically: true, encoding: .utf8)
    let result = try runSetupBash(
        """
        "${REPO_ROOT}/tools/build-mlx-metallib.sh"
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "PATH": "\(fakeBin.path):/usr/bin:/bin",
            "MLXFAST_CMAKE_BIN": fakeBin.appendingPathComponent("cmake").path,
            "MLXFAST_MLX_SWIFT_VENDOR": checkout.path,
            "MLXFAST_MLX_METAL_BUILD_DIR": root.appendingPathComponent("metal-build").path,
            "MLXFAST_MLX_METALLIB": output.path,
        ]
    )

    #expect(result.status != 0)
    #expect(try String(contentsOf: output, encoding: .utf8) == "original")
    let outputFiles = try FileManager.default.contentsOfDirectory(
        atPath: output.deletingLastPathComponent().path
    )
    #expect(outputFiles == ["mlx.metallib"])

    let directoryOutput = root.appendingPathComponent("directory-output/mlx.metallib")
    try FileManager.default.createDirectory(
        at: directoryOutput,
        withIntermediateDirectories: true
    )
    let sentinel = directoryOutput.appendingPathComponent("sentinel")
    try "preserve".write(to: sentinel, atomically: true, encoding: .utf8)
    let directoryResult = try runSetupBash(
        """
        "${REPO_ROOT}/tools/build-mlx-metallib.sh"
        """,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "PATH": "\(fakeBin.path):/usr/bin:/bin",
            "MLXFAST_CMAKE_BIN": fakeBin.appendingPathComponent("cmake").path,
            "MLXFAST_MLX_SWIFT_VENDOR": checkout.path,
            "MLXFAST_MLX_METAL_BUILD_DIR": root.appendingPathComponent("metal-build").path,
            "MLXFAST_MLX_METALLIB": directoryOutput.path,
        ]
    )

    #expect(directoryResult.status != 0)
    #expect(try String(contentsOf: sentinel, encoding: .utf8) == "preserve")
    #expect(
        try FileManager.default.contentsOfDirectory(atPath: directoryOutput.path)
            == ["sentinel"]
    )
}

private struct SetupBashResult {
    let status: Int32
    let stdout: String
    let stderr: String
}

private func runSetupBash(
    _ script: String,
    environment overrides: [String: String] = [:]
) throws -> SetupBashResult {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["-c", script]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    var environment = ProcessInfo.processInfo.environment
    for key in environment.keys.filter({ $0.hasPrefix("MLXFAST_") }) {
        environment.removeValue(forKey: key)
    }
    for (key, value) in overrides {
        environment[key] = value
    }
    process.environment = environment
    let stdout = Pipe()
    let stderr = Pipe()
    process.standardOutput = stdout
    process.standardError = stderr
    try process.run()
    process.waitUntilExit()
    let stdoutData = stdout.fileHandleForReading.readDataToEndOfFile()
    let stderrData = stderr.fileHandleForReading.readDataToEndOfFile()
    return SetupBashResult(
        status: process.terminationStatus,
        stdout: String(data: stdoutData, encoding: .utf8) ?? "",
        stderr: String(data: stderrData, encoding: .utf8) ?? ""
    )
}

private func setupTemporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        "mlxfast-setup-script-tests-\(UUID().uuidString)"
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func writeSetupExecutable(at url: URL, contents: String) throws {
    try contents.write(to: url, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: url.path
    )
}

/// Minimal vendored mlx-swift layout for build-mlx-metallib.sh fixtures: the
/// CMake source dirs the builder requires plus non-empty kernel/JIT trees the
/// vendored-source fingerprint hashes.
private func writeSetupVendoredFixture(at checkout: URL) throws {
    for path in [
        "Source/Cmlx/mlx",
        "Source/Cmlx/mlx-generated",
        "Source/Cmlx/metal-cpp",
        "Source/Cmlx/json",
        "Source/Cmlx/fmt",
    ] {
        try FileManager.default.createDirectory(
            at: checkout.appendingPathComponent(path),
            withIntermediateDirectories: true
        )
    }
    try "kernel void fixture() {}\n".write(
        to: checkout.appendingPathComponent("Source/Cmlx/mlx/rope.metal"),
        atomically: true,
        encoding: .utf8
    )
    try "const char* fixture = \"jit\";\n".write(
        to: checkout.appendingPathComponent("Source/Cmlx/mlx-generated/quantized.cpp"),
        atomically: true,
        encoding: .utf8
    )
}

// The dependency graph is frozen by challenge policy: both build entrypoints
// assert Package.swift/Package.resolved match the committed state before
// building and pass --force-resolved-versions so SwiftPM fails closed instead
// of silently re-resolving an out-of-date graph.
@Test
func buildEntrypointsAssertFrozenDependencyGraph() throws {
    let setup = try String(contentsOfFile: "setup.sh", encoding: .utf8)
    let benchmark = try String(contentsOfFile: "benchmark.sh", encoding: .utf8)
    for script in [setup, benchmark] {
        #expect(script.contains("assert_frozen_dependency_graph"))
        #expect(script.contains("for manifest in Package.swift Package.resolved; do"))
        #expect(script.contains("git diff --quiet HEAD -- \"${manifest}\""))
        #expect(script.contains("the dependency graph is frozen by challenge policy"))
    }
    // benchmark.sh asserts before its fallback build; setup.sh inside
    // build_swift_harness, before any swift build runs.
    #expect(benchmark.contains("  assert_frozen_dependency_graph\n"))
    #expect(setup.contains("  assert_frozen_dependency_graph || return 1\n"))

    let manifest = try String(contentsOfFile: "Package.swift", encoding: .utf8)
    #expect(!manifest.contains("TODO(security): Assert the resolved dependency graph"))
    #expect(manifest.contains("--force-resolved-versions"))

    // Every ranked resolve/build passes --force-resolved-versions too.
    for workflowPath in [
        ".github/workflows/dflash-benchmark.yml",
        ".github/workflows/ci.yml",
    ] {
        let workflow = try String(contentsOfFile: workflowPath, encoding: .utf8)
        #expect(
            !workflow.contains("swift package resolve)"),
            "\(workflowPath) must resolve with --force-resolved-versions"
        )
        #expect(
            !workflow.contains("swift build -c release --product"),
            "\(workflowPath) must build with --force-resolved-versions"
        )
        #expect(
            !workflow.contains("swift test -c release\n"),
            "\(workflowPath) must test with --force-resolved-versions"
        )
    }
}

/// The contestant docs' own command blocks must not instruct a bare
/// `swift build` / `swift test` / `swift package resolve`: a bare invocation
/// can silently rewrite the frozen Package.resolved (SwiftPM re-resolves the
/// ranged transitive pins on toolchain drift), after which setup.sh and
/// benchmark.sh refuse to run via assert_frozen_dependency_graph — blaming
/// the contestant for following the docs.
@Test
func contestantDocsCommandBlocksKeepTheDependencyGraphFrozen() throws {
    for docPath in ["AGENTS.md", "TASK.md", "README.md"] {
        let doc = try String(contentsOfFile: docPath, encoding: .utf8)
        for bare in ["\nswift test\n", "\nswift build -c release\n", "\nswift package resolve\n"] {
            #expect(
                !doc.contains(bare),
                "\(docPath) documents a bare\(bare)which can rewrite the frozen Package.resolved; add --force-resolved-versions"
            )
        }
        // Also catches env-prefixed bare invocations like
        // `MLXFAST_RUN_MLX_RUNTIME_TESTS=1 swift test`.
        #expect(
            !doc.contains(" swift test\n"),
            "\(docPath) documents a bare env-prefixed swift test; add --force-resolved-versions"
        )
    }
    for docPath in ["AGENTS.md", "TASK.md"] {
        let doc = try String(contentsOfFile: docPath, encoding: .utf8)
        #expect(doc.contains("swift test --force-resolved-versions"))
        #expect(doc.contains("swift build -c release --force-resolved-versions"))
        #expect(doc.contains("git checkout -- Package.resolved"))
    }
}

@Test
func frozenDependencyGraphAssertionDetectsManifestDrift() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let repo = root.appendingPathComponent("repo")
    try FileManager.default.createDirectory(at: repo, withIntermediateDirectories: true)
    try "// manifest\n".write(
        to: repo.appendingPathComponent("Package.swift"), atomically: true, encoding: .utf8)
    try "{\"pins\":[]}\n".write(
        to: repo.appendingPathComponent("Package.resolved"), atomically: true, encoding: .utf8)

    let script = """
    eval "$(sed '/^ensure_swift_toolchain$/,$d' "${REPO_ROOT}/setup.sh")"
    cd "${FIXTURE_REPO}"
    git init -q
    git -c user.email=t@t -c user.name=t -c commit.gpgsign=false add .
    git -c user.email=t@t -c user.name=t -c commit.gpgsign=false commit -q -m base
    # Clean tree passes.
    assert_frozen_dependency_graph
    # A drifted resolved file (e.g. SwiftPM silently re-resolving) fails.
    printf '{"pins":["drift"]}\\n' > Package.resolved
    drift_status=0
    assert_frozen_dependency_graph || drift_status=$?
    [[ "${drift_status}" != "0" ]]
    git checkout -q -- Package.resolved
    assert_frozen_dependency_graph
    """
    let result = try runSetupBash(
        script,
        environment: [
            "REPO_ROOT": FileManager.default.currentDirectoryPath,
            "FIXTURE_REPO": repo.path,
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
        ]
    )
    #expect(result.status == 0, "stdout: \(result.stdout) stderr: \(result.stderr)")
    #expect(result.stderr.contains("dependency graph is frozen by challenge policy"))
}

// The bash fingerprint recipe in tools/build-mlx-metallib.sh and its Swift
// twin (VendoredMetalFingerprint, used by the trusted CLI before every worker
// spawn) must agree bit for bit, stay sensitive to kernel edits, and ignore
// symlinks.
@Test
func metallibFingerprintScriptAndSwiftTwinAgree() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let vendor = root.appendingPathComponent("mlx-swift")
    let cmlx = vendor.appendingPathComponent("Source/Cmlx")
    try writeSetupVendoredFixture(at: vendor)
    try FileManager.default.createDirectory(
        at: cmlx.appendingPathComponent("mlx/backend/metal/kernels"),
        withIntermediateDirectories: true
    )
    try "sdpa kernel v1\n".write(
        to: cmlx.appendingPathComponent("mlx/backend/metal/kernels/sdpa_vector.h"),
        atomically: true,
        encoding: .utf8
    )
    try "dotfile\n".write(
        to: cmlx.appendingPathComponent("mlx/.hidden"),
        atomically: true,
        encoding: .utf8
    )
    // A symlink must not contribute to the fingerprint in either recipe.
    try FileManager.default.createSymbolicLink(
        atPath: cmlx.appendingPathComponent("mlx/alias.metal").path,
        withDestinationPath: "rope.metal"
    )

    func scriptFingerprint() throws -> String {
        let result = try runSetupBash(
            """
            "${REPO_ROOT}/tools/build-mlx-metallib.sh" --print-fingerprint
            """,
            environment: [
                "REPO_ROOT": FileManager.default.currentDirectoryPath,
                "MLXFAST_MLX_SWIFT_VENDOR": vendor.path,
            ]
        )
        #expect(result.status == 0, "stderr: \(result.stderr)")
        return result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    let initialScript = try scriptFingerprint()
    let initialSwift = try VendoredMetalFingerprint.compute(cmlxRoot: cmlx.path)
    #expect(initialScript == initialSwift)
    #expect(initialScript.count == 64)

    // A kernel edit changes the fingerprint identically in both recipes.
    try "sdpa kernel v2 (edited)\n".write(
        to: cmlx.appendingPathComponent("mlx/backend/metal/kernels/sdpa_vector.h"),
        atomically: true,
        encoding: .utf8
    )
    let editedScript = try scriptFingerprint()
    let editedSwift = try VendoredMetalFingerprint.compute(cmlxRoot: cmlx.path)
    #expect(editedScript == editedSwift)
    #expect(editedScript != initialScript)

    // A JIT-twin edit is covered too.
    try "const char* fixture = \"jit v2\";\n".write(
        to: cmlx.appendingPathComponent("mlx-generated/quantized.cpp"),
        atomically: true,
        encoding: .utf8
    )
    let jitEditedScript = try scriptFingerprint()
    let jitEditedSwift = try VendoredMetalFingerprint.compute(cmlxRoot: cmlx.path)
    #expect(jitEditedScript == jitEditedSwift)
    #expect(jitEditedScript != editedScript)
}

// The pre-spawn record check the trusted CLI runs: verified on a matching
// sidecar, mismatch on stale/missing/malformed records, skipped only when the
// check has nothing to verify (no vendored tree / no metallib).
@Test
func metallibFingerprintRecordVerificationFailsClosedOnStaleRecord() throws {
    let root = try setupTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let vendor = root.appendingPathComponent("mlx-swift")
    let cmlx = vendor.appendingPathComponent("Source/Cmlx")
    try writeSetupVendoredFixture(at: vendor)
    let metallib = root.appendingPathComponent("mlx.metallib")
    try Data("metallib".utf8).write(to: metallib)
    let record = metallib.path + ".fingerprint"

    func outcome() -> MetallibFingerprintVerification {
        verifyMetallibFingerprintRecord(metallibPath: metallib.path, cmlxRoot: cmlx.path)
    }

    // No record at all: fails (never trust an unattributed metallib).
    guard case .mismatch(let missingReason) = outcome() else {
        Issue.record("expected mismatch without a fingerprint record")
        return
    }
    #expect(missingReason.contains("no fingerprint record"))

    // A record matching the sources verifies.
    let fingerprint = try VendoredMetalFingerprint.compute(cmlxRoot: cmlx.path)
    try VendoredMetalFingerprint.recordLine(fingerprint: fingerprint)
        .write(toFile: record, atomically: true, encoding: .utf8)
    #expect(outcome() == .verified)

    // Editing a kernel after the build makes the record stale.
    try "kernel void fixture() { /* edited */ }\n".write(
        to: cmlx.appendingPathComponent("mlx/rope.metal"),
        atomically: true,
        encoding: .utf8
    )
    guard case .mismatch(let staleReason) = outcome() else {
        Issue.record("expected mismatch after a kernel edit")
        return
    }
    #expect(staleReason.contains("built from different"))

    // A malformed or wrong-version record is a mismatch, not a pass.
    try "not-a-record\n".write(toFile: record, atomically: true, encoding: .utf8)
    guard case .mismatch(let malformedReason) = outcome() else {
        Issue.record("expected mismatch for a malformed record")
        return
    }
    #expect(malformedReason.contains("malformed"))

    // Nothing to verify: missing metallib or missing vendored tree skip, and
    // the caller (official runs) decides that skipping is fatal.
    try FileManager.default.removeItem(at: metallib)
    guard case .skipped = outcome() else {
        Issue.record("expected skipped without a metallib")
        return
    }
    try Data("metallib".utf8).write(to: metallib)
    try FileManager.default.removeItem(at: cmlx.appendingPathComponent("mlx-generated"))
    guard case .skipped = outcome() else {
        Issue.record("expected skipped without the vendored tree")
        return
    }
}

// MARK: - setup-dflash.sh must provision the target the track actually measures

/// `setup-dflash.sh` was restored from the retired MTP scaffolding and still
/// described that experiment: it pinned `mlx-community/Laguna-XS-2.1-4bit` as the
/// target, read the deleted `fixtures/mtp_laguna_xs_2_1_4bit.sha256` manifest, and
/// claimed the drafter is "converted to MLX affine 4-bit (group size 64) at
/// setup". The DFlash contract fixture says otherwise: the target is the SAME
/// NVFP4 reference checkpoint the serial track measures (which is the entire
/// reason a DFlash speedup is comparable with a serial one), and the drafter is
/// pinned by its own manifest rather than converted.
///
/// A setup script that provisions a different checkpoint than the workflow
/// measures makes every local number meaningless, so the target identity is
/// pinned against the contract fixture rather than restated here.
@Suite
struct SetupDFlashTargetIdentityTests {
    @Test
    func setupDFlashProvisionsTheSharedNVFP4ReferenceAndClaimsNoConversion() throws {
        let script = try String(contentsOfFile: "setup-dflash.sh", encoding: .utf8)

        for retired in [
            "fixtures/mtp_laguna_xs_2_1_4bit.sha256",
            "mtp_laguna_xs_2_1_4bit.sha256",
            "mlx-community/Laguna-XS-2.1-4bit",
            "laguna-xs-2.1-mtp-v1",
        ] {
            #expect(
                !script.contains(retired),
                """
                setup-dflash.sh still names '\(retired)', which belongs to the \
                retired MTP experiment. The manifest was deleted, so the script \
                cannot run; the model id is a different checkpoint from the one \
                the DFlash workflow measures.
                """
            )
        }

        // No conversion claim: the drafter is hash-pinned by its own manifest.
        for claim in [
            "converted to MLX affine 4-bit",
            "converted to MLX 4-bit",
            "affine 4-bit (group size 64)",
        ] {
            #expect(
                !script.contains(claim),
                """
                setup-dflash.sh claims the drafter is \(claim) at setup. Nothing \
                in the DFlash path converts it: the drafter is provisioned and \
                verified against fixtures/dflash_laguna_xs_2_1_drafter.sha256.
                """
            )
        }

        // Whatever it provisions must agree with the contract fixture. The
        // target is deliberately NOT downloaded here — it is the serial track's
        // NVFP4 reference, which ./setup.sh already fetches and verifies — so
        // the invariant is that the script names that shared manifest and
        // provisions no second target of its own.
        let fixtureData = try Data(
            contentsOf: URL(fileURLWithPath: "fixtures/laguna_xs_2_1_dflash_track.json")
        )
        let fixture = try #require(
            try JSONSerialization.jsonObject(with: fixtureData) as? [String: Any]
        )
        let target = try #require(fixture["target"] as? [String: Any])
        let targetManifest = try #require(target["manifest_path"] as? String)
        let assistant = try #require(fixture["assistant"] as? [String: Any])
        let assistantManifest = try #require(assistant["manifest_path"] as? String)

        #expect(
            script.contains(targetManifest),
            """
            setup-dflash.sh does not name \(targetManifest). The DFlash target IS \
            the serial track's NVFP4 reference — that is what makes a DFlash \
            speedup comparable with a serial one — so the script has to say so \
            rather than leave a reader to assume a second checkpoint.
            """
        )
        #expect(
            script.contains(assistantManifest),
            """
            setup-dflash.sh does not verify the drafter against \
            \(assistantManifest), the manifest the contract fixture pins
            """
        )
        // No second target download of any kind.
        #expect(
            !script.contains("TARGET_MODEL_ID="),
            """
            setup-dflash.sh declares its own TARGET_MODEL_ID. Provisioning a \
            second target is how the local path ends up measuring a different \
            checkpoint from the ranked one.
            """
        )
        #expect(
            !script.contains("mlx-community/"),
            "setup-dflash.sh must not provision an mlx-community checkpoint"
        )
    }
}
