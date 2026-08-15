import CryptoKit
import Foundation
@testable import MLXFastCore
@testable import MLXFastHarness
import Testing

@Test
func setupScriptDefaultsToFastReferenceMirror() throws {
    let setup = try String(
        contentsOfFile: "setup.sh",
        encoding: .utf8
    )

    // Qwen 3.6 track identity (QWEN36-MTP-CHALLENGE-PLAN.md phase 1). Unlike
    // the retired Laguna target there is NO organizer-hosted fast mirror for
    // this checkpoint yet -- the Darkbloom R2 bucket serves Laguna only -- so
    // setup.sh resolves the immutable Hugging Face revision directly and the
    // fallback is deliberately empty. The pins below mirror setup.sh exactly;
    // authoring the Qwen mirror is Phase 5 work.
    // REPINNED 2026-08-14 onto the Qwen 3.8 backbone. setup.sh, the ranked
    // workflow env and MLXFastConstants name one artifact; the cutover commit
    // had left setup.sh and the constants at 3.6 while the workflow read a
    // placeholder, and closing that split is what these three assertions hold
    // open. The REVISION is the published sha (2026-08-14): our own
    // mlx-0.32.0 conversion, replacing a third-party one terminated by the
    // validation kill-switch, published to EigenLabs.
    #expect(setup.contains("REFERENCE_MODEL_REPO=\"${MLXFAST_REFERENCE_MODEL_REPO:-EigenLabs/Qwen3.8-27B-4bit}\""))
    #expect(setup.contains("REFERENCE_REVISION=\"${MLXFAST_REFERENCE_REVISION:-eda45ab47f465d08d6558f0353a2346e2eb9d5b3}\""))
    #expect(setup.contains("DEFAULT_REFERENCE_BASE_URL=\"https://huggingface.co/EigenLabs/Qwen3.8-27B-4bit/resolve/${REFERENCE_REVISION}\""))
    #expect(setup.contains("DEFAULT_REFERENCE_FALLBACK_BASE_URL=\"\""))
    #expect(setup.contains("REFERENCE_BASE_URL=\"${MLXFAST_REFERENCE_BASE_URL:-${DEFAULT_REFERENCE_BASE_URL}}\""))
    #expect(setup.contains("REFERENCE_MANIFEST_PATH=\"${MLXFAST_REFERENCE_MANIFEST_PATH:-fixtures/reference_qwen3_8_27b_4bit.sha256}\""))
    // The metadata download list is DERIVED FROM THE SELECTED MANIFEST, never
    // hard-coded, because ONE downloader provisions TWO artifacts whose
    // metadata trees legitimately disagree: the backbone pins
    // generation_config.json and NO .gitattributes/LICENSE.md, while the MTP
    // head (setup-qwen-mtp.sh drives this same downloader at
    // fixtures/qwen3_8_27b_mtp_head.sha256) pins .gitattributes and carries no
    // README, no tokenizer and no chat template at all. The hard-coded array
    // that used to live here was wrong for BOTH of them in both directions: a
    // fresh ./setup.sh died on "reference manifest has no entry for
    // .gitattributes", head provisioning demanded files that will never exist,
    // and generation_config.json -- which the backbone manifest DOES pin -- was
    // never fetched, which the ranked workflow's both-directions verify_cache
    // inventory then rejects the cache for.
    #expect(!setup.contains("REFERENCE_REQUIRED_METADATA_FILES"))
    #expect(!setup.contains("REFERENCE_OPTIONAL_METADATA_FILES"))
    #expect(setup.contains("reference_manifest_metadata_files() {"))
    #expect(setup.contains("if ! metadata_list=\"$(reference_manifest_metadata_files)\"; then"))
    // *.safetensors is excluded from the derived list: shards are downloaded
    // separately and in parallel, driven by model.safetensors.index.json
    // through list_reference_shards.
    #expect(setup.contains("if [[ \"${relative_path}\" == *.safetensors ]]; then"))
    // Fail closed on a derivation that yields nothing -- a header-only manifest
    // stub, or one that is all shards -- instead of provisioning an empty
    // directory quietly.
    #expect(setup.contains("names no non-shard metadata files"))
    // ...and config.json plus the index stay hard post-download requirements
    // whatever the manifest happens to name.
    #expect(setup.contains("downloaded checkpoint is missing config.json"))
    #expect(setup.contains("downloaded checkpoint is missing model.safetensors.index.json"))
    // MLXFAST_REFERENCE_HASH_VERIFY=0 is the one mode where a missing manifest
    // is legitimate (reference_file_is_current short-circuits before reading
    // it), so the derivation falls back to that minimal required pair rather
    // than failing or inventing names.
    #expect(setup.contains("printf '%s\\n' \"config.json\" \"model.safetensors.index.json\""))
    // 3 parallel shard downloads by default; env-overridable.
    #expect(setup.contains("REFERENCE_DOWNLOAD_JOBS=\"${MLXFAST_REFERENCE_DOWNLOAD_JOBS:-3}\""))
    #expect(setup.contains("DEFAULT_HF_HOME=\"${MLXFAST_HF_HOME:-${HF_HOME:-${HOME:-${PWD}}/.cache/huggingface}}\""))
    #expect(setup.contains("REFERENCE_CACHE_DIR=\"${MLXFAST_REFERENCE_CACHE_DIR:-${DEFAULT_HF_HUB_CACHE}/${REFERENCE_CACHE_REPO_DIR}/snapshots/${REFERENCE_CACHE_REVISION_DIR}}\""))
    #expect(setup.contains("REFERENCE_CACHE_LOCK_PATH=\"${MLXFAST_REFERENCE_CACHE_LOCK_PATH:-${REFERENCE_DIR}/.mlxfast-reference-cache.lock}\""))
    #expect(setup.contains("REFERENCE_POST_DOWNLOAD_FULL_VERIFY=\"${MLXFAST_REFERENCE_POST_DOWNLOAD_FULL_VERIFY:-1}\""))
    #expect(setup.contains("SETUP_PARALLEL_METALLIB=\"${MLXFAST_SETUP_PARALLEL_METALLIB:-${MLXFAST_SETUP_PARALLEL_BUILD:-1}}\""))
    #expect(setup.contains("Usage: ./setup.sh"))
    #expect(setup.contains("reference_file_is_current"))
    #expect(setup.contains("reference_cache_lock_is_current"))
    #expect(setup.contains("reference_post_download_full_verify_enabled"))
    #expect(setup.contains("verify_reference_weights_after_verified_download"))
    #expect(setup.contains("cannot skip post-download full verification unless MLXFAST_REFERENCE_HASH_VERIFY=1"))
    #expect(setup.contains("write_reference_cache_lock"))
    #expect(setup.contains("redownloading ${label} from scratch after hash verification failed"))
    #expect(setup.contains("If you only installed the Command Line Tools and this still fails, install full"))
    #expect(setup.contains("reference cache path ${reference_dir}"))
    #expect(setup.contains("compatibility reference path exists and is not a symlink"))
    #expect(setup.contains("if ! verify_reference_manifest \"${reference_dir}\"; then"))
    #expect(setup.contains("downloaded ${total}/${total} safetensors shard(s)"))
    #expect(setup.contains("ensure_swift_harness_ready || return 1"))
    #expect(setup.contains("start_mlx_metallib_build"))
    #expect(setup.contains("MLXFAST_SETUP_PARALLEL_METALLIB must be 0 or 1"))
    #expect(setup.contains("setup.sh: mlx.metallib build running in background"))
    let mainRange = try #require(setup.range(of: "ensure_swift_toolchain\nensure_macmon\ntrap cleanup_background_builds EXIT"))
    let main = String(setup[mainRange.lowerBound...])
    #expect(main.contains("build_swift_harness\nstart_mlx_metallib_build\ndownload_reference_weights \"${REFERENCE_DIR}\""))
    #expect(setup.contains("setup.sh: setup complete elapsed="))
    #expect(setup.contains("MLXFAST_OFFLINE_WRITABLE_PATHS=\"${PWD}/weights\" .github/scripts/run-offline.sh ${SWIFT_BIN} transform --reference \"${REFERENCE_DIR}\" --output weights"))
    #expect(setup.contains("${SWIFT_BIN} correctness --weights weights"))
    // The summary's benchmark hint must reflect reality: a bare ./benchmark.sh
    // defaults to --local-iterate against the checked-in public fixtures; only
    // --official needs the organizer-provisioned oracle.
    #expect(setup.contains("./benchmark.sh  # defaults to --local-iterate against the public fixtures"))
    #expect(!setup.contains("requires organizer-supplied correctness_golden.json"))

    // macmon is optional (local thermal cool-gate only): installed as a
    // pinned, hash-verified release binary dropped in ~/bin -- a location
    // benchmark.sh's gate lookup already searches -- and never through
    // Homebrew, so a normal setup run cannot mutate global Homebrew state.
    #expect(setup.contains("MACMON_RELEASE_URL=\"https://github.com/vladkens/macmon/releases/download/v${MACMON_VERSION}/macmon-v${MACMON_VERSION}.tar.gz\""))
    #expect(setup.contains("MACMON_RELEASE_SHA256="))
    #expect(setup.contains("MACMON_INSTALL_DIR=\"${HOME}/bin\""))
    #expect(setup.contains("install_macmon_release"))
    #expect(setup.contains("macmon release sha256 mismatch"))
    #expect(setup.contains("MLXFAST_SKIP_MACMON_INSTALL"))
    #expect(!setup.contains("brew install macmon"))

    // A CLT-only machine (xcodebuild present but "requires Xcode") must be
    // diagnosed as missing full Xcode, distinct from an unaccepted license.
    #expect(setup.contains("command line tools instance"))
    #expect(setup.contains("sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"))
    #expect(setup.contains("sudo xcodebuild -license accept"))

    // The cold ~20 GiB parallel shard download prints an aggregate progress
    // heartbeat instead of going silent for the whole transfer.
    #expect(setup.contains("start_reference_download_heartbeat \"${output_dir}\" \"${expected_total_bytes}\" \"$@\""))
    #expect(setup.contains("stop_reference_download_heartbeat"))
    #expect(setup.contains("still downloading safetensors shard(s)"))

    // The Yukon CLI (`yukon`) is installed by the external Yukon
    // installer; setup only surfaces install/PATH-activation guidance and
    // must never fail because of it.
    #expect(main.contains("wait_for_mlx_metallib_build\ncheck_yukon_cli\nprint_setup_summary \"ready\""))
    #expect(setup.contains("is not on PATH, so"))
    #expect(setup.contains("external Yukon installer"))
    #expect(setup.contains("use the Yukon CLI ('yukon')"))

    // The pinned backbone repository is PUBLIC since the 2026-08-14 publish, so
    // the documented default is an anonymous fetch and
    // MLXFAST_REFERENCE_AUTH_HEADER is an optional fallback (private mirror,
    // rate-limited fetch), not a prerequisite. The old text told every
    // participant they needed a token to run ./setup.sh at all.
    #expect(setup.contains("# The repository is PUBLIC, so the default fetch is ANONYMOUS"))
    #expect(!setup.contains("The repository is PRIVATE, so a fetch needs credentials"))
    #expect(setup.contains("REFERENCE_AUTH_HEADER=\"${MLXFAST_REFERENCE_AUTH_HEADER:-}\""))

    // ...and the help text names the checkpoint this script actually downloads.
    // It described the retired Poolside Laguna XS 2.1 NVFP4 target long after
    // the repin.
    #expect(setup.contains("builds mlx.metallib, and downloads the Qwen 3.8 27B MLX 4-bit reference"))
    #expect(!setup.contains("Poolside Laguna XS 2.1 NVFP4 reference"))

    // MLXFAST_SKIP_SWIFT_BUILD lets setup-qwen-mtp.sh's delegated download reuse
    // the products ./setup.sh just built instead of relinking both of them
    // (~25s of pure waste on the ordinary two-command path). It FAILS OPEN: a
    // missing product builds anyway, so a standalone ./setup-qwen-mtp.sh on a
    // fresh clone still works. The knob never decides whether the harness
    // exists, only whether it is rebuilt.
    #expect(setup.contains("if [[ \"${MLXFAST_SKIP_SWIFT_BUILD:-0}\" == \"1\" ]]; then"))
    #expect(setup.contains("MLXFAST_SKIP_SWIFT_BUILD=1 and both products are present; reusing"))
    #expect(setup.contains("MLXFAST_SKIP_SWIFT_BUILD=1 but a product is missing; building anyway"))
    #expect(setup.contains("  MLXFAST_SKIP_SWIFT_BUILD=1         Reuse the Swift products from a previous"))

    // MLXFAST_SETUP_SUMMARY_ROLE keeps the closing summary honest during
    // delegated head provisioning: REFERENCE_DIR is the MTP head there, so the
    // generic "transform --reference ... --output weights" next step would tell
    // the reader to overwrite the target weights with a 15-tensor head. Prose
    // only; an unrecognised value is refused rather than defaulted.
    #expect(setup.contains("SETUP_SUMMARY_ROLE=\"${MLXFAST_SETUP_SUMMARY_ROLE:-target}\""))
    #expect(setup.contains("MLXFAST_SETUP_SUMMARY_ROLE must be 'target' or 'mtp-head'"))
    #expect(setup.contains("  ${checkpoint_label}: ${reference_line}"))
    #expect(setup.contains("./benchmark-qwen-mtp.sh --local-iterate"))
    #expect(setup.contains("head provisioning is complete; the target weights/ tree belongs to ./setup.sh"))
}

/// `setup-qwen-mtp.sh` provisions the MTP head by DELEGATING to `setup.sh`'s
/// downloader, so its contract is the environment it hands over. Two entries
/// were added after a QA pass measured the delegated run rebuilding the Swift
/// harness for nothing and printing target-shaped advice over a head cache.
@Test
func headProvisioningDelegatesWithoutRebuildingOrMisadvising() throws {
    let runner = try String(contentsOfFile: "setup-qwen-mtp.sh", encoding: .utf8)

    // The head is provisioned through setup.sh's downloader, driven at the head
    // manifest -- which is also why the metadata list must come FROM that
    // manifest (see setupScriptDefaultsToFastReferenceMirror): the head tree is
    // four files and shares almost nothing with the backbone's.
    #expect(runner.contains("MLXFAST_REFERENCE_MANIFEST_PATH=\"${MTP_HEAD_MANIFEST}\""))
    #expect(runner.contains("fixtures/qwen3_8_27b_mtp_head.sha256"))

    // No second Swift build: ./setup.sh already built both products, and
    // nothing about them can have changed between the two commands.
    #expect(runner.contains("MLXFAST_SKIP_SWIFT_BUILD=1"))
    #expect(!runner.contains("rebuilds the Swift binaries if they are stale"))

    // The summary must not tell a head-provisioning reader to transform
    // REFERENCE_DIR into weights/.
    #expect(runner.contains("MLXFAST_SETUP_SUMMARY_ROLE=mtp-head"))

    // Both pinned repositories are public as of the 2026-08-14 publish; the
    // header claimed an unauthenticated fetch 401s and demanded a token.
    #expect(runner.contains("# THE HEAD REPOSITORY IS PUBLIC."))
    #expect(!runner.contains("THE HEAD REPOSITORY IS PRIVATE"))
    #expect(!runner.contains("401"))

    // Prose names the artifacts actually pinned (3.8), while the historical
    // note about the 3.6 defaults this file used to carry stays intact.
    #expect(runner.contains("Provision the organizer-pinned Qwen 3.8 27B MTP HEAD"))
    #expect(runner.contains("Provision the organizer-pinned Qwen 3.8 27B MTP head."))
    #expect(runner.contains("the Qwen 3.8 27B reference checkpoint"))
    #expect(runner.contains("EigenLabs/Qwen3.8-27B-MTP-bf16"))

    // The delegated run still must not mutate global tool state or repoint the
    // shared reference_weights/ compatibility symlink at the head.
    #expect(runner.contains("MLXFAST_REFERENCE_COMPAT_LINK="))
    #expect(runner.contains("MLXFAST_SKIP_HOMEBREW_INSTALL=1"))
    #expect(runner.contains("MLXFAST_SKIP_MLX_METALLIB=1"))
}


@Test
func poolsideNVFP4DistributionIdentityIsPinned() throws {
    // Repointed to the Qwen target this branch declares, and REPINNED
    // 2026-08-14 from the 3.6 backbone onto the 3.8 one. The retired
    // Poolside Laguna identity is still pinned, but by the Laguna-specific
    // fixtures that describe that checkpoint --
    // Tests/Fixtures/PoolsideLagunaXS21NVFP4 and
    // fixtures/reference_laguna_xs_2_1_nvfp4_mlx.sha256 -- not by
    // MLXFastConstants, which carries the Qwen track identity.
    //
    // The pinned artifact is OUR OWN MLX 4-bit affine / group_size 64
    // conversion (mlx 0.32.0) of the official bf16 base Qwen/Qwen3.8-27B @
    // 1d4bf0f2, which replaced a third-party conversion adopted and then
    // terminated the same day by the validation kill-switch. It is the same
    // string the ranked workflow's MLXFAST_QWEN_MTP_CHECKPOINT_REPO /
    // _REVISION name, pinned equal to these constants by
    // QwenMTPTrackNamingTests.theQwenMTPTrackShipsInertPendingTheQwen38BringUp.
    //
    // The REVISION is the published sha (2026-08-14). Asserting it exactly
    // keeps the compiled constant, setup.sh, the workflow pins and the fixture
    // header from ever disagreeing about which bytes are the reference.
    let repository = "EigenLabs/Qwen3.8-27B-4bit"
    let revision = "eda45ab47f465d08d6558f0353a2346e2eb9d5b3"
    #expect(MLXFastConstants.referenceModelRepository == repository)
    #expect(MLXFastConstants.referenceModelRevision == revision)
    #expect(MLXFastConstants.referenceModelName == "Qwen3.8-27B-4bit")
    #expect(
        MLXFastConstants.defaultReferencePath
            == "reference_weights/Qwen3.8-27B-4bit"
    )
    #expect(
        MLXFastConstants.defaultReferenceCachePath
            == ".cache/huggingface/hub/models--EigenLabs--Qwen3.8-27B-4bit/snapshots/\(revision)"
    )
    let manifest = try String(
        contentsOfFile: "fixtures/reference_qwen3_8_27b_4bit.sha256",
        encoding: .utf8
    )
    // The three PINNED HEADER LINES keep the shape the 3.6 manifest had, so a
    // reader (and setup.sh's own header parse) sees the same file. Asserting
    // the Revision line through the same `revision` binding as the constants
    // above is what keeps the compiled pin and the fixture from drifting apart.
    #expect(manifest.contains("# SHA256 manifest for \(repository)."))
    #expect(manifest.contains("# Revision: \(revision)"))
    #expect(manifest.contains("# Format: <sha256> <byte_count> <relative_path>"))

    // THE BODY LANDED with the 2026-08-14 publish, and this assertion inverted
    // with it: it previously required an EMPTY record body plus the
    // QWEN38-PENDING-RELEASE / "BODY PLACEHOLDER" stub markers, which is what
    // the file looked like while the upload was outstanding. Those three
    // assertions were left behind by the publish commit and were failing
    // against the shipped fixture; the record-level checks below are what they
    // were always meant to become.
    let records = manifest
        .split(separator: "\n")
        .filter { !$0.hasPrefix("#") && !$0.trimmingCharacters(in: .whitespaces).isEmpty }
    let recordPaths = records.compactMap { $0.split(separator: " ").last.map(String.init) }
    // Field-safe on purpose: a malformed record must fail the byte assertion,
    // not trap on an index.
    let recordBytes = records.reduce(0) { total, line in
        total + (line.split(separator: " ").dropFirst().first.flatMap { Int($0) } ?? 0)
    }
    // The shape pins are summed from the digests they have to agree with, and
    // they are the same numbers the ranked workflow carries as
    // MLXFAST_QWEN_MTP_TARGET_MANIFEST_RECORDS / _BYTES. A manifest that LOSES
    // a record cannot be caught by per-file hashing -- it would verify fewer
    // files and report success -- which is why the count is pinned here too.
    #expect(records.count == 10, "manifest carries \(records.count) records, not 10")
    #expect(recordBytes == 15_153_237_117, "manifest sums to \(recordBytes) bytes")
    #expect(
        recordPaths == [
            "README.md",
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
        "manifest names \(recordPaths)"
    )
    // Every non-shard record here is a file setup.sh downloads, because the
    // metadata download list is DERIVED from this body rather than hard-coded
    // (see setupScriptDefaultsToFastReferenceMirror). generation_config.json is
    // the reason that matters: the old hard-coded list never fetched it even
    // though it is pinned, and the ranked workflow's `verify_cache` runs a
    // strict inventory in BOTH directions.
    //
    // KNOWN OPEN ITEM, recorded rather than asserted: the published repository
    // also carries the `.gitattributes` Hugging Face generates for every repo,
    // and this manifest deliberately does NOT pin it -- the pin was published
    // as a ten-record shape and moving it to eleven is the operator's call, so
    // the fixture's own header carries the item. Do not "fix" that here; a
    // stock snapshot_download of this revision stages a file the manifest does
    // not name, which is what run 31665285024 died on.
    #expect(!recordPaths.contains(".gitattributes"))
    #expect(!manifest.contains("QWEN38-PENDING-RELEASE"))
    #expect(!manifest.contains("BODY PLACEHOLDER"))
}

@Test
func benchmarkFailsFastWhenSetupArtifactsAreMissing() throws {
    let benchmark = try String(contentsOfFile: "benchmark.sh", encoding: .utf8)
    let prerequisite = try #require(
        benchmark.range(of: "if [[ ! -s \"${MLX_METALLIB}\" ]]")
    )
    let automaticBuild = try #require(
        benchmark.range(
            of:
                "benchmark.sh: trusted CLI or participant runtime worker missing or stale; building"
        )
    )
    #expect(prerequisite.lowerBound < automaticBuild.lowerBound)

    let guardBody = String(benchmark[prerequisite.lowerBound..<automaticBuild.lowerBound])
    #expect(guardBody.contains("YUKON_CLI_COMMAND"))
    #expect(guardBody.contains("exit 1"))
}

/// The scored binary is the .build-worker worker, and an existence-only build
/// gate timed a worker built before the participant's edit -- a silently stale
/// measurement with no warning anywhere in the output. The gate must also
/// compare mtimes against the build inputs that produce that worker.
@Test
func benchmarkRebuildsWhenParticipantSourcesAreNewerThanTheBuiltWorker() throws {
    let benchmark = try String(contentsOfFile: "benchmark.sh", encoding: .utf8)
    let gate = try #require(
        benchmark.range(of: "swift_build_required() {")
    )
    let gateEnd = try #require(
        benchmark.range(of: "\n}", range: gate.upperBound..<benchmark.endIndex)
    )
    let body = String(benchmark[gate.upperBound..<gateEnd.lowerBound])

    // Still builds when either product is absent.
    #expect(body.contains("[[ ! -x \"${SWIFT_BIN}\" ]]"))
    #expect(body.contains("! -x \"${RUNTIME_WORKER_BIN}\""))
    // ...and when any build input is newer than the product we would run.
    #expect(body.contains("-newer"))
    #expect(body.contains("Sources"))
    #expect(body.contains("Vendor"))
    #expect(body.contains("Package.resolved"))
    // The older of the two products is the comparison reference, so a stale
    // worker is caught even when the trusted CLI was rebuilt more recently.
    #expect(body.contains("-ot"))
}

/// mlx.metallib is a CMake/Metal artifact outside SwiftPM's build graph, so
/// the swift-build freshness gate can never refresh it; without its own gate
/// an AOT kernel edit (RoPE, RMSNorm, SDPA vector, arg_reduce) is silently
/// benchmarked against the stale library in local modes, where the trusted
/// CLI's fingerprint verification only prints one stderr warning line.
@Test
func benchmarkRebuildsTheMetallibWhenVendoredKernelSourcesAreNewer() throws {
    let benchmark = try String(contentsOfFile: "benchmark.sh", encoding: .utf8)
    let gate = try #require(
        benchmark.range(of: "metallib_rebuild_required() {")
    )
    let gateEnd = try #require(
        benchmark.range(of: "\n}", range: gate.upperBound..<benchmark.endIndex)
    )
    let body = String(benchmark[gate.upperBound..<gateEnd.lowerBound])

    // Never rebuilds over an explicit MLXFAST_MLX_METALLIB override -- the
    // caller (test fixtures, operator layouts) owns that artifact's
    // lifecycle, and the fixture-driven benchmark.sh tests rely on it.
    #expect(body.contains("-n \"${MLXFAST_MLX_METALLIB:-}\""))
    // Rebuilds when anything under the two fingerprinted vendored subtrees
    // -- the exact input set of mlx.metallib.fingerprint -- is newer than
    // the published metallib.
    #expect(body.contains("-newer \"${MLX_METALLIB}\""))
    #expect(body.contains("Vendor/mlx-swift/Source/Cmlx/mlx \\"))
    #expect(body.contains("Vendor/mlx-swift/Source/Cmlx/mlx-generated \\"))

    // The gate triggers the real builder (which republishes the fingerprint
    // sidecar), fails the run when that rebuild fails, and never runs on the
    // official path, where the fingerprint check fails closed instead.
    let trigger = try #require(
        benchmark.range(of: "&& metallib_rebuild_required; then")
    )
    let triggerLineStart = benchmark[..<trigger.lowerBound].lastIndex(of: "\n")
        ?? benchmark.startIndex
    let triggerBlockEnd = try #require(
        benchmark.range(of: "\nfi\n", range: trigger.upperBound..<benchmark.endIndex)
    )
    let triggerBlock = String(benchmark[triggerLineStart..<triggerBlockEnd.upperBound])
    #expect(triggerBlock.contains("[[ \"${OFFICIAL}\" != \"1\" ]]"))
    #expect(triggerBlock.contains("tools/build-mlx-metallib.sh"))
    #expect(triggerBlock.contains("exit 1"))
}

@Test
func benchmarkRejectsPartialSetupBeforeInvokingExistingBinary() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let invocation = root.appendingPathComponent("swift-invoked")
    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    try """
    #!/bin/sh
    touch "\(invocation.path)"
    exit 99
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: fakeSwift.path
    )

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["benchmark.sh", "--local-iterate"]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    process.environment = benchmarkTestEnvironment([
        "MLXFAST_NO_SANDBOX": "1",
        "MLXFAST_SWIFT_BIN": fakeSwift.path,
        "MLXFAST_MLX_METALLIB": root.appendingPathComponent("missing.metallib").path,
        "YUKON_CLI_COMMAND": "yukon-dev",
        "MLXFAST_SCORE_PATH": root.appendingPathComponent("score.json").path,
        "MLXFAST_INTEGRITY_PATH": root.appendingPathComponent("integrity.json").path,
    ])
    let stderr = Pipe()
    process.standardError = stderr

    try process.run()
    process.waitUntilExit()
    let stderrText = String(
        data: stderr.fileHandleForReading.readDataToEndOfFile(),
        encoding: .utf8
    ) ?? ""

    #expect(process.terminationStatus != 0)
    #expect(stderrText.contains("yukon-dev setup"))
    #expect(!FileManager.default.fileExists(atPath: invocation.path))
}

@Test
func participantDocsExposeDefaultCLIInstallDirectory() throws {
    let pathExport = #"export PATH="${HOME}/.local/bin:${PATH}""#
    let retiredParticipantPrefix = ["mlx", "fast"].joined()
    for path in ["README.md", "TASK.md", "AGENTS.md", "CLAUDE.md"] {
        let document = try String(contentsOfFile: path, encoding: .utf8)
        #expect(document.contains(pathExport), "\(path) must expose the default CLI install directory")
        #expect(document.contains("Yukon CLI (`yukon`)"), "\(path) must name the Yukon CLI explicitly")
        for verb in ["login", "clone", "submit", "submissions", "run", "sync"] {
            let retiredCommand = "\(retiredParticipantPrefix) \(verb)"
            #expect(
                !document.contains(retiredCommand),
                "\(path) must direct participants to `yukon`, not `\(retiredCommand)`"
            )
        }
    }
}

@Test
func participantDocsDescribeDefaultDFlashScoreAndFloor() throws {
    let readme = try String(contentsOfFile: "README.md", encoding: .utf8)
    let challenge = try String(contentsOfFile: "TASK.md", encoding: .utf8)

    for document in [readme, challenge] {
        // The default (and only) ranked track is DFlash: a DECODE-ONLY paired
        // speedup, per-prompt NORMALISED by each prompt's pinned no-op reference,
        // with a single 0.95 normalised floor over a 512-token window (Amendment
        // 32). The retired serial weighted formula must be gone; so must the raw
        // 0.83 floor as the stated ranked floor.
        #expect(document.contains("dflash_decode_speedup"))
        #expect(document.lowercased().contains("decode-only"))
        #expect(document.contains("0.95"))
        #expect(document.lowercased().contains("normalis"))
        #expect(document.lowercased().contains("no-op reference"))
        #expect(document.contains("512"))
        #expect(!document.contains("dflash_decode_speedup >= 0.83"))
        #expect(!document.contains("decode_speedup^0.75 * prefill_speedup^0.25"))
        // The retired MTP track's score description must be gone.
        #expect(!document.contains("MTP_seconds_per_token"))
        #expect(!document.contains("serial_K1_seconds_per_token"))
        #expect(!document.contains("3.177180971604"))
        #expect(!document.contains("0.149183255724"))
        #expect(!document.contains("256-step greedy decode latency"))
        #expect(!document.contains("256 expected continuation token IDs"))
        #expect(!document.contains("all 256 tokens produced"))
        #expect(!document.contains("256-token greedy continuation"))
    }
    #expect(!challenge.contains("bandwidth_source=expert_streaming_reads"))
    #expect(!challenge.contains("bandwidth_GB_per_token"))
}


@Test
func benchmarkWorkflowProbesAndEnforcesRuntimeWorkerSandbox() throws {
    let workflow = try String(
        contentsOfFile: ".github/workflows/dflash-benchmark.yml",
        encoding: .utf8
    )
    let benchmark = try String(
        contentsOfFile: "benchmark.sh",
        encoding: .utf8
    )
    let probe = try String(
        contentsOfFile: ".github/scripts/probe-runtime-worker-sandbox.sh",
        encoding: .utf8
    )
    // The probe proves this host's sandbox-exec semantics before any build,
    // transform, or hidden material reaches the bench workspace.
    let probeRange = try #require(workflow.range(of: "- name: Probe runtime worker sandbox"))
    let prepareWorkspaceRange = try #require(workflow.range(of: "- name: Prepare bench workspace"))
    let hiddenGoldenRange = try #require(workflow.range(of: "- name: Prepare hidden correctness golden"))
    #expect(probeRange.lowerBound < prepareWorkspaceRange.lowerBound)
    #expect(probeRange.lowerBound < hiddenGoldenRange.lowerBound)
    #expect(workflow.contains("run: .github/scripts/probe-runtime-worker-sandbox.sh"))
    #expect(workflow.contains("MLXFAST_OFFICIAL_BENCHMARK_RUN: \"1\""))

    #expect(benchmark.contains("enforce_official_sandbox"))
    #expect(benchmark.contains("MLXFAST_OFFICIAL_BENCHMARK_RUN"))
    #expect(benchmark.contains("official GitHub benchmark runs must not set MLXFAST_NO_SANDBOX=1"))
    #expect(benchmark.contains("official GitHub benchmark runs must use the runtime worker sandbox"))
    let setupGuard = try #require(
        benchmark.range(
            of: "enforce_official_sandbox\n\nif [[ ! -s \"${MLX_METALLIB}\" ]]"
        )
    )
    let automaticBuild = try #require(
        benchmark.range(
            of: "if [[ \"${MLXFAST_IN_SANDBOX:-0}\" != \"1\" ]] && swift_build_required; then"
        )
    )
    #expect(setupGuard.lowerBound < automaticBuild.lowerBound)

    #expect(probe.contains("(deny network*)"))
    #expect(probe.contains("(deny process-fork)"))
    #expect(probe.contains("(deny process-exec*)"))
    #expect(probe.contains("(allow process-exec (literal"))
    #expect(probe.contains("(deny file-write*)"))
    #expect(probe.contains("(allow file-write* (literal \"/dev/null\"))"))
    #expect(probe.contains("(deny file-read* (literal"))
    #expect(probe.contains("(deny file-read* (subpath"))
    #expect(probe.contains("expect_inet_network_denied()"))
    #expect(probe.contains("expect_unix_network_denied(argv[5])"))
    #expect(probe.contains("expect_fork_denied()"))
    #expect(probe.contains("expect_spawn_denied()"))
}

@Test
func referenceCacheProbeWorkflowIsManualAndExperimental() throws {
    let workflow = try String(
        contentsOfFile: ".github/workflows/reference-cache-probe.yml",
        encoding: .utf8
    )
    let downloader = try String(
        contentsOfFile: ".github/scripts/download-reference-cache-scope.sh",
        encoding: .utf8
    )
    #expect(workflow.contains("name: reference-cache-probe"))
    #expect(workflow.contains("workflow_dispatch:"))
    #expect(!workflow.contains("pull_request:"))
    #expect(!workflow.contains("push:"))
    #expect(workflow.contains("cache_scope:"))
    #expect(workflow.contains("actions/cache/restore@55cc8345863c7cc4c66a329aec7e433d2d1c52a9"))
    #expect(workflow.contains("actions/cache/save@55cc8345863c7cc4c66a329aec7e433d2d1c52a9"))
    #expect(workflow.contains(".github/scripts/download-reference-cache-scope.sh \"${CACHE_SCOPE}\""))
    #expect(workflow.contains("MLXFAST_REFERENCE_POST_DOWNLOAD_FULL_VERIFY: \"0\""))

    // Secret-free by design (a blanket `secrets.` ban): this workflow runs
    // setup.sh and the download script FROM THE DISPATCHED REF with no
    // `environment:` approval gate, so a repo credential injected here (as a
    // prior version did with secrets.MLXFAST_REFERENCE_BASE_URL/
    // MLXFAST_REFERENCE_AUTH_HEADER) would be exfiltratable by any ref with a
    // modified setup.sh. The public R2 mirror and immutable Hugging Face
    // fallback need no credential, and every downloaded byte is manifest-pinned.
    #expect(!workflow.contains("environment:"))
    #expect(!workflow.contains("secrets."))
    #expect(!workflow.contains("secrets: inherit"))
    #expect(!workflow.contains("MLXFAST_REFERENCE_AUTH_HEADER"))
    #expect(workflow.contains("MLXFAST_REFERENCE_BASE_URL: ${{ inputs.reference_base_url || 'https://ds4.darkbloom.ai/laguna-xs-2.1-nvfp4-mlx' }}"))
    // The probe must target the LAGUNA reference checkpoint: the Gemma-era
    // manifest and cache directory were deleted with the migration, so stale
    // defaults would fail every dispatch at the manifest hash step.
    #expect(workflow.contains("MLXFAST_REFERENCE_MANIFEST_PATH: fixtures/reference_laguna_xs_2_1_nvfp4_mlx.sha256"))
    #expect(workflow.contains("cache_key=\"laguna-xs-2.1-nvfp4-mlx-${CACHE_SCOPE}-"))
    #expect(downloader.contains("DEFAULT_REFERENCE_BASE_URL=\"https://ds4.darkbloom.ai/laguna-xs-2.1-nvfp4-mlx\""))
    #expect(downloader.contains("DEFAULT_REFERENCE_FALLBACK_BASE_URL=\"https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4-mlx/resolve/841778bda563a36104dd521e37d99218e46f4f25\""))
    #expect(downloader.contains("trying fallback source for ${relative_path}"))
    #expect(!workflow.contains("gemma-4-31b-4bit"))
}


@Test
func trustedBenchmarkWorkflowGuardAllowsOnlyPermittedBranches() throws {
    func run(
        script: String,
        workflow: String,
        ref: String
    ) throws -> (status: Int32, stderr: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [script]
        process.currentDirectoryURL = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath
        )
        process.environment = ProcessInfo.processInfo.environment.merging([
            "GITHUB_REPOSITORY": "Layr-Labs/mlxfast-challenge-dev",
            "GITHUB_REF": ref,
            "GITHUB_WORKFLOW_REF":
                "Layr-Labs/mlxfast-challenge-dev/.github/workflows/\(workflow)@\(ref)",
            "GITHUB_EVENT_NAME": "workflow_dispatch",
        ]) { _, new in new }
        let stderr = Pipe()
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        return (
            process.terminationStatus,
            String(
                data: stderr.fileHandleForReading.readDataToEndOfFile(),
                encoding: .utf8
            ) ?? ""
        )
    }

    for (script, workflow) in [
        (".github/scripts/enforce-trusted-benchmark-workflow.sh", "benchmark.yml")
    ] {
        for ref in [
            "refs/heads/main",
            "refs/heads/submissions/example",
            "refs/heads/baseline/reference",
            "refs/heads/yukon/baseline/718528521cd7a7df341b750bc3ccb28478ff045b",
        ] {
            #expect(
                try run(script: script, workflow: workflow, ref: ref).status == 0,
                "expected \(workflow) on \(ref) to be allowed"
            )
        }

        let rejected = try run(
            script: script,
            workflow: workflow,
            ref: "refs/heads/feature/not-allowed"
        )
        #expect(rejected.status != 0)
        #expect(rejected.stderr.contains("allowed branches are main, submissions/*, baseline/*, and yukon/baseline/*"))
    }
}

@Test
func benchmarkWorkspaceACLLocksTrustedSurfacesAgainstBench() throws {
    let workflow = try String(
        contentsOfFile: ".github/workflows/dflash-benchmark.yml",
        encoding: .utf8
    )
    let prepareRange = try #require(workflow.range(of: "- name: Prepare bench workspace"))
    let buildRange = try #require(workflow.range(of: "- name: Build trusted CLI in bench sandbox"))
    let prepare = String(workflow[prepareRange.lowerBound..<buildRange.lowerBound])

    // The functional allow ACE still lets bench read/execute the tree and
    // create its own build/transform/score outputs.
    #expect(prepare.contains(
        "/bin/chmod -R +a \"user:bench allow list,search,readattr,readextattr,read,execute,"
            + "add_file,add_subdirectory,delete_child,write,append,writeattr,writeextattr,"
            + "file_inherit,directory_inherit\" \"${MLXFAST_JOB_WS}\""
    ))
    // Defense-in-depth deny: bench cannot write/delete the trusted, runner-
    // owned harness surfaces (scripts, tooling, public fixtures, the hidden-
    // input staging dir, the driver, the contract). Deny ACEs on runner-owned
    // paths cannot be stripped by the bench uid.
    #expect(prepare.contains(
        "/bin/chmod -R +a \"user:bench deny write,append,writeattr,writeextattr,delete,"
            + "delete_child,add_file,add_subdirectory,chown,file_inherit,directory_inherit\""
    ))
    #expect(prepare.contains("\"${MLXFAST_JOB_WS}/.github\""))
    #expect(prepare.contains("\"${MLXFAST_JOB_WS}/tools\""))
    #expect(prepare.contains("\"${MLXFAST_JOB_WS}/Sources\""))
    #expect(prepare.contains("\"${MLXFAST_JOB_WS}/correctness_prompts\""))
    #expect(prepare.contains("\"${MLXFAST_JOB_WS}/.ranked-src\""))
    #expect(prepare.contains(
        "/bin/chmod +a \"user:bench deny write,append,writeattr,writeextattr,delete,chown\""
    ))
    #expect(prepare.contains("\"${MLXFAST_JOB_WS}/benchmark.sh\""))
    #expect(prepare.contains("\"${MLXFAST_JOB_WS}/benchmark.json\""))
    // The runner-owned, bench-non-writable staging dir for predictable-name
    // hidden inputs is created before the ACL is applied.
    #expect(prepare.contains("mkdir -p \"${MLXFAST_JOB_WS}/.ranked-src\""))
}

@Test
func benchmarkPlacesHiddenGoldenInputsSymlinkSafely() throws {
    let workflow = try String(
        contentsOfFile: ".github/workflows/dflash-benchmark.yml",
        encoding: .utf8
    )
    let attachRange = try #require(workflow.range(of: "- name: Attach GPQA gates and verify augmented golden"))
    let gatesGuardRange = try #require(workflow.range(of: "- name: Verify trusted harness before gates"))
    let attach = String(workflow[attachRange.lowerBound..<gatesGuardRange.lowerBound])

    // Hidden inputs land in the runner-owned, bench-non-writable staging dir,
    // never at a predictable workspace-root name the transform could pre-plant
    // as a symlink.
    #expect(attach.contains("golden_src=\"${MLXFAST_JOB_WS}/.ranked-src/golden.json\""))
    #expect(attach.contains("gpqa_src=\"${MLXFAST_JOB_WS}/.ranked-src/gpqa.json\""))
    // Reject-if-exists/symlink guard plus install(1) (fresh regular file, never
    // writes through a symlink).
    #expect(attach.contains("if [[ -e \"${hidden_target}\" || -L \"${hidden_target}\" ]]; then"))
    #expect(attach.contains("refusing to place hidden input over pre-existing path"))
    #expect(attach.contains("install -m 0444 \"${MLXFAST_PRIVATE_DIR}/raw_golden.json\" \"${golden_src}\""))
    #expect(attach.contains("install -m 0444 \"${MLXFAST_PRIVATE_DIR}/gpqa_reference.json\" \"${gpqa_src}\""))
    #expect(attach.contains("--golden .ranked-src/golden.json"))
    #expect(attach.contains("--gpqa .ranked-src/gpqa.json"))
    // Cleanup unlinks the runner-owned staging dir (never an attacker symlink).
    #expect(attach.contains("rm -rf \"${MLXFAST_JOB_WS}/.ranked-src\""))
    // The old predictable workspace-root names are gone entirely.
    #expect(!workflow.contains(".ranked-golden-src.json"))
    #expect(!workflow.contains(".ranked-gpqa-src.json"))
}


@Test
func pinTrustedHarnessScriptDetectsTamper() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let ws = root.appendingPathComponent("ws")
    let releaseDir = ws.appendingPathComponent(".build/release")
    let workerReleaseDir = ws.appendingPathComponent(".build-worker/release")
    try FileManager.default.createDirectory(at: releaseDir, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: workerReleaseDir, withIntermediateDirectories: true)
    try "driver\n".write(to: ws.appendingPathComponent("benchmark.sh"), atomically: true, encoding: .utf8)
    try "BINARY".write(to: releaseDir.appendingPathComponent("mlxfast-swift"), atomically: true, encoding: .utf8)
    try "WORKER".write(to: workerReleaseDir.appendingPathComponent("mlxfast-runtime-worker"), atomically: true, encoding: .utf8)
    try "METALLIB".write(to: workerReleaseDir.appendingPathComponent("mlx.metallib"), atomically: true, encoding: .utf8)
    let trustedPin = root.appendingPathComponent("trusted-harness.sha256")
    let workerPin = root.appendingPathComponent("participant-worker.sha256")

    func run(_ mode: String, _ artifactSet: String) throws -> Int32 {
        let pin = artifactSet == "trusted" ? trustedPin : workerPin
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            ".github/scripts/pin-trusted-harness.sh", mode, ws.path, pin.path, artifactSet,
        ]
        process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        return process.terminationStatus
    }

    #expect(try run("write", "trusted") == 0)
    #expect(try run("write", "worker") == 0)
    #expect(FileManager.default.fileExists(atPath: trustedPin.path))
    #expect(FileManager.default.fileExists(atPath: workerPin.path))
    #expect(try run("verify", "trusted") == 0)
    #expect(try run("verify", "worker") == 0)

    // A swapped trusted binary fails the trusted verify closed and leaves the
    // worker attestation untouched (the sets are independent).
    try "EVIL".write(to: releaseDir.appendingPathComponent("mlxfast-swift"), atomically: true, encoding: .utf8)
    #expect(try run("verify", "trusted") != 0)
    #expect(try run("verify", "worker") == 0)

    // A swapped driver fails too.
    try "BINARY".write(to: releaseDir.appendingPathComponent("mlxfast-swift"), atomically: true, encoding: .utf8)
    try "hacked\n".write(to: ws.appendingPathComponent("benchmark.sh"), atomically: true, encoding: .utf8)
    #expect(try run("verify", "trusted") != 0)

    // A swapped worker binary fails the worker attestation, not the trusted pin.
    try "driver\n".write(to: ws.appendingPathComponent("benchmark.sh"), atomically: true, encoding: .utf8)
    #expect(try run("verify", "trusted") == 0)
    try "EVILWORKER".write(
        to: workerReleaseDir.appendingPathComponent("mlxfast-runtime-worker"),
        atomically: true,
        encoding: .utf8
    )
    #expect(try run("verify", "worker") != 0)
    #expect(try run("verify", "trusted") == 0)

    // A symlinked artifact is rejected (never dereferenced).
    try "WORKER".write(
        to: workerReleaseDir.appendingPathComponent("mlxfast-runtime-worker"),
        atomically: true,
        encoding: .utf8
    )
    #expect(try run("verify", "worker") == 0)
    try FileManager.default.removeItem(at: workerReleaseDir.appendingPathComponent("mlx.metallib"))
    try FileManager.default.createSymbolicLink(
        atPath: workerReleaseDir.appendingPathComponent("mlx.metallib").path,
        withDestinationPath: "/etc/hosts"
    )
    #expect(try run("verify", "worker") != 0)

    // An unknown artifact set is rejected.
    #expect(try run("verify", "everything") == 2)
}

// The trusted-harness source scope (manifests + timer/gates/score sources) is
// byte-verified against trusted git content in the trusted shell before every
// ranked build, independently of the overlay and surface-enforcement scripts,
// and the manifests are ACL-locked against bench writes.
@Test
func benchmarkWorkflowsPinTrustedSourceScopeBeforeBuilding() throws {
    let script = try String(
        contentsOfFile: ".github/scripts/verify-trusted-source-scope.sh",
        encoding: .utf8
    )
    #expect(script.contains("\"Package.swift\""))
    #expect(script.contains("\"Package.resolved\""))
    #expect(script.contains("\"Sources/MLXFastCLI\""))
    #expect(script.contains("\"Sources/MLXFastTrustedHarness\""))
    #expect(script.contains("\"Sources/MLXFastCore\""))
    // Sources/MLXFastTransform is participant-editable by contract and is
    // deliberately outside the trusted scope pin.
    #expect(script.contains("Sources/MLXFastTransform is deliberately OUT of scope"))
    #expect(!script.contains("\"Sources/MLXFastTransform\""))
    // Byte comparison against trusted git content, through the hardened git
    // wrapper, plus inventory and non-regular-entry checks.
    #expect(script.contains("hardened-git.sh"))
    #expect(script.contains("cat-file blob \"HEAD:${rel}\" | cmp -s - \"${ws_path}\""))
    #expect(script.contains("ls-tree -r --name-only \"HEAD\" --"))
    #expect(script.contains("find \"${BENCH_WORKSPACE}/${dir}\" -mindepth 1 ! -type d ! -type f"))
    // The editable contract comes from the trusted ref, never the work tree,
    // and must not overlap the trusted scope.
    #expect(script.contains("cat-file blob \"HEAD:benchmark.json\""))
    #expect(script.contains("overlaps trusted scope path"))

    // The manifest replaces the old TODO with a pointer at the enforcement.
    let manifest = try String(contentsOfFile: "Package.swift", encoding: .utf8)
    #expect(!manifest.contains("TODO(security): Pin the trusted-harness source scope"))
    #expect(manifest.contains("verify-trusted-source-scope.sh"))

    for workflowPath in [
        ".github/workflows/dflash-benchmark.yml",
        ".github/workflows/dflash-benchmark.yml",
    ] {
        let workflow = try String(contentsOfFile: workflowPath, encoding: .utf8)
        let prepareRange = try #require(
            workflow.range(of: "- name: Prepare bench workspace"),
            "expected prepare step in \(workflowPath)"
        )
        let scopeRange = try #require(
            workflow.range(of: "- name: Verify trusted source scope"),
            "expected source-scope step in \(workflowPath)"
        )
        let buildRange = try #require(
            workflow.range(of: "- name: Build trusted CLI in bench sandbox"),
            "expected trusted build step in \(workflowPath)"
        )
        // After the workspace copy exists, before anything is compiled.
        #expect(prepareRange.lowerBound < scopeRange.lowerBound)
        #expect(scopeRange.lowerBound < buildRange.lowerBound)
        #expect(workflow.contains(
            ".github/scripts/verify-trusted-source-scope.sh \"${GITHUB_WORKSPACE}\" \"${MLXFAST_JOB_WS}\""
        ))
        // The frozen manifests are also bench-deny ACL-locked like the driver
        // and the contract.
        #expect(workflow.contains("\"${MLXFAST_JOB_WS}/Package.swift\""))
        #expect(workflow.contains("\"${MLXFAST_JOB_WS}/Package.resolved\""))
    }
}

@Test
func verifyTrustedSourceScopeScriptDetectsScopeTamper() throws {
    let fm = FileManager.default
    let scriptPath = URL(fileURLWithPath: fm.currentDirectoryPath)
        .appendingPathComponent(".github/scripts/verify-trusted-source-scope.sh").path

    let root = try temporaryDirectory()
    defer { try? fm.removeItem(at: root) }
    let trusted = root.appendingPathComponent("trusted")
    let ws = root.appendingPathComponent("ws")

    func run(
        _ argv: [String], cwd: String
    ) throws -> (status: Int32, output: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = argv
        process.currentDirectoryURL = URL(fileURLWithPath: cwd)
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(data: data, encoding: .utf8) ?? "")
    }
    for tool in ["git", "jq"] {
        guard try run(["sh", "-c", "command -v \(tool)"], cwd: fm.currentDirectoryPath).status == 0
        else { return }
    }

    @discardableResult
    func git(_ args: [String]) throws -> String {
        let result = try run(
            ["git", "-c", "user.email=test@test", "-c", "user.name=test",
             "-c", "commit.gpgsign=false"] + args,
            cwd: trusted.path
        )
        #expect(result.status == 0, "git \(args.joined(separator: " ")): \(result.output)")
        return result.output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func write(_ relative: String, _ contents: String, under base: URL) throws {
        let url = base.appendingPathComponent(relative)
        try fm.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try contents.write(to: url, atomically: true, encoding: .utf8)
    }

    try fm.createDirectory(at: trusted, withIntermediateDirectories: true)
    try write("Package.swift", "// manifest v1\n", under: trusted)
    try write("Package.resolved", "{\"pins\":[]}\n", under: trusted)
    try write(
        "benchmark.json",
        #"{"editablePaths":["Sources/MLXFastModel","Sources/MLXFastTransform"]}"#,
        under: trusted)
    try write("Sources/MLXFastCLI/main.swift", "// cli\n", under: trusted)
    try write("Sources/MLXFastTrustedHarness/Runtime.swift", "// harness\n", under: trusted)
    try write("Sources/MLXFastCore/Core.swift", "// core\n", under: trusted)
    try write("Sources/MLXFastModel/Editable.swift", "// editable v1\n", under: trusted)
    try git(["init", "-q"])
    try git(["add", "."])
    try git(["commit", "-q", "-m", "trusted base"])

    func resetWorkspace() throws {
        try? fm.removeItem(at: ws)
        try fm.copyItem(at: trusted, to: ws)
        try? fm.removeItem(at: ws.appendingPathComponent(".git"))
    }
    func verify() throws -> (status: Int32, output: String) {
        try run(["bash", scriptPath, trusted.path, ws.path], cwd: fm.currentDirectoryPath)
    }

    // Clean copy passes; an overlaid EDITABLE path does not trip the check.
    try resetWorkspace()
    #expect(try verify().status == 0)
    try write("Sources/MLXFastModel/Editable.swift", "// editable v2 overlay\n", under: ws)
    #expect(try verify().status == 0)

    // A mutated manifest fails.
    try write("Package.swift", "// manifest TAMPERED\n", under: ws)
    let manifestTamper = try verify()
    #expect(manifestTamper.status != 0)
    #expect(manifestTamper.output.contains("Package.swift"))

    // A new source added under a trusted dir fails (scope expansion).
    try resetWorkspace()
    try write("Sources/MLXFastCore/Injected.swift", "// injected\n", under: ws)
    let injected = try verify()
    #expect(injected.status != 0)
    #expect(injected.output.contains("file inventory under Sources/MLXFastCore differs"))

    // A mutated trusted source fails.
    try resetWorkspace()
    try write("Sources/MLXFastCLI/main.swift", "// cli TAMPERED\n", under: ws)
    #expect(try verify().status != 0)

    // A symlink inside the trusted scope fails (redirection).
    try resetWorkspace()
    try fm.removeItem(at: ws.appendingPathComponent("Sources/MLXFastTrustedHarness/Runtime.swift"))
    try fm.createSymbolicLink(
        atPath: ws.appendingPathComponent("Sources/MLXFastTrustedHarness/Runtime.swift").path,
        withDestinationPath: "../MLXFastModel/Editable.swift"
    )
    let symlinked = try verify()
    #expect(symlinked.status != 0)
    #expect(symlinked.output.contains("non-regular entry"))

    // A trusted contract that exposes the trusted scope as editable fails,
    // even when the workspace bytes all match.
    try resetWorkspace()
    try write(
        "benchmark.json",
        #"{"editablePaths":["Sources/MLXFastModel","Sources/MLXFastCore"]}"#,
        under: trusted)
    try git(["commit", "-q", "-am", "expose trusted scope"])
    try write(
        "benchmark.json",
        #"{"editablePaths":["Sources/MLXFastModel","Sources/MLXFastCore"]}"#,
        under: ws)
    let exposed = try verify()
    #expect(exposed.status != 0)
    #expect(exposed.output.contains("overlaps trusted scope path"))
}

@Test
func hashWeightsDirectoryRejectsHardlinks() throws {
    let hashScript = try String(
        contentsOfFile: ".github/scripts/hash-weights-directory.sh",
        encoding: .utf8
    )
    // Mirrors overlay-editable-paths.sh and QwenRuntime.directoryDigest.
    #expect(hashScript.contains("-links +1"))
    #expect(hashScript.contains("weights tree contains a hardlinked file"))
    // Reproducible digest recipe (path-order-stable, content-addressed).
    #expect(hashScript.contains("shasum -a 256"))
    #expect(hashScript.contains("LC_ALL=C sort -z"))

    let preflight = try String(
        contentsOfFile: "Sources/MLXFastHarness/QwenRuntimePreflight.swift",
        encoding: .utf8
    )
    #expect(preflight.contains("func requireSingleHardLink("))
    #expect(preflight.contains("info.st_nlink != 1"))
    #expect(preflight.contains("try requireSingleHardLink("))

    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let weights = root.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try "content".write(to: weights.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

    func hashStatus() throws -> Int32 {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [".github/scripts/hash-weights-directory.sh", weights.path]
        process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        return process.terminationStatus
    }

    // A clean single-link tree hashes fine.
    #expect(try hashStatus() == 0)

    // A hardlinked file (link count 2) is rejected.
    try FileManager.default.linkItem(
        atPath: weights.appendingPathComponent("config.json").path,
        toPath: weights.appendingPathComponent("aliased.json").path
    )
    #expect(try hashStatus() != 0)
}


@Test
func benchmarkWorkflowPinsTrustedLayerCountForFinalValidation() throws {
    let workflow = try String(
        contentsOfFile: ".github/workflows/dflash-benchmark.yml",
        encoding: .utf8
    )
    let validator = try String(
        contentsOfFile: ".github/scripts/validate-benchmark-artifacts.sh",
        encoding: .utf8
    )
    let steps = try #require(workflow.range(of: "\n    steps:"))
    let jobHeader = String(workflow[..<steps.lowerBound])
    let dispatch = try #require(workflow.range(of: "  workflow_dispatch:"))
    let permissions = try #require(
        workflow.range(of: "\npermissions:", range: dispatch.upperBound..<workflow.endIndex)
    )
    let dispatchInputs = String(workflow[dispatch.lowerBound..<permissions.lowerBound])

    // The model shape is a trusted, literal workflow contract. A dispatch or
    // participant-controlled expression must never choose the accepted count.
    //
    // TWO WORKFLOWS, TWO MODELS, ONE RULE. Constants now describe Qwen 3.6
    // (64 layers), so the workflow that must mirror them is the Qwen-MTP one;
    // dflash-benchmark.yml pins Laguna's 40 and keeps doing so while that track
    // is the live ranked one. This was a withKnownIssue only because no Qwen
    // workflow existed to carry the repointed pin. Both sides are now asserted
    // HARD, and both keep the negative assertions -- the "never an expression,
    // never a dispatch input" property is what stops a submitter choosing the
    // accepted shape, and it has to hold on every ranked workflow, not just the
    // one whose number currently matches Constants.
    let qwenWorkflow = try String(
        contentsOfFile: ".github/workflows/qwen-mtp-ranked-benchmark.yml",
        encoding: .utf8
    )
    let qwenSteps = try #require(qwenWorkflow.range(of: "\n    steps:"))
    let qwenJobHeader = String(qwenWorkflow[..<qwenSteps.lowerBound])
    let qwenDispatch = try #require(qwenWorkflow.range(of: "  workflow_dispatch:"))
    let qwenPermissions = try #require(
        qwenWorkflow.range(of: "\npermissions:", range: qwenDispatch.upperBound..<qwenWorkflow.endIndex)
    )
    let qwenDispatchInputs = String(qwenWorkflow[qwenDispatch.lowerBound..<qwenPermissions.lowerBound])

    let assignment = "MLXFAST_EXPECTED_NUM_LAYERS: \"\(MLXFastConstants.numHiddenLayers)\""
    #expect(
        qwenJobHeader.components(separatedBy: assignment).count - 1 == 1,
        """
        the Qwen-MTP ranked workflow must pin MLXFAST_EXPECTED_NUM_LAYERS to \
        MLXFastConstants.numHiddenLayers (\(MLXFastConstants.numHiddenLayers)) \
        exactly once
        """
    )
    #expect(!qwenDispatchInputs.contains("MLXFAST_EXPECTED_NUM_LAYERS"))
    #expect(!qwenJobHeader.contains("MLXFAST_EXPECTED_NUM_LAYERS: ${{"))

    // DFLASH-SIDE COVERAGE, DELIBERATELY KEPT: Laguna's literal, because after
    // the Qwen repoint it can no longer be derived from Constants. Dropping it
    // would leave the live ranked track's shape contract unguarded.
    let lagunaAssignment = "MLXFAST_EXPECTED_NUM_LAYERS: \"40\""
    #expect(
        jobHeader.components(separatedBy: lagunaAssignment).count - 1 == 1,
        """
        the DFlash workflow must still pin Laguna's 40 layers exactly once; if \
        that track has been repointed, update this literal in the same commit
        """
    )
    #expect(!dispatchInputs.contains("MLXFAST_EXPECTED_NUM_LAYERS"))
    #expect(!jobHeader.contains("MLXFAST_EXPECTED_NUM_LAYERS: ${{"))

    #expect(validator.contains(
        "MLXFAST_EXPECTED_NUM_LAYERS:?MLXFAST_EXPECTED_NUM_LAYERS is required"
    ))
    #expect(validator.contains(
        "MLXFAST_EXPECTED_NUM_LAYERS must be a positive integer"
    ))
    #expect(validator.contains(
        "--argjson expected_num_layers \"${MLXFAST_EXPECTED_NUM_LAYERS}\""
    ))
    #expect(validator.contains(".metrics.num_layers == $expected_num_layers"))
    #expect(!validator.contains(".metrics.num_layers == 60"))
}


// Behavioral pin for run-submission-static-review.sh's diff-only mode, run
// against a throwaway git repo. Every guard case exits before any network
// call (a real curl attempt with the dummy key would fail the script); the
// happy path shims curl on PATH and asserts the judge request contains only
// the changed editable file, never the unchanged suspicious baseline file.
@Test
func submissionStaticReviewPromptCoversMeasurementStructureExploitation() throws {
    // The seed-forward memo (submissions/0ddfcd37) passed review as legitimate
    // memoization because the rubric had no category for gains that come from
    // the benchmark's own call pattern rather than real inference work, and
    // the judge had no harness call-pattern context to decide "this cache can
    // only hit when the harness repeats itself". Pin both additions.
    let staticReview = try String(
        contentsOfFile: ".github/scripts/run-submission-static-review.sh",
        encoding: .utf8
    )
    #expect(staticReview.contains("measurement-structure exploitation"))
    #expect(staticReview.contains("Bit-identical outputs do not make that legitimate"))
    #expect(staticReview.contains("harness_protocol"))
    #expect(staticReview.contains("decision_test"))
    #expect(staticReview.contains("input-independent caching"))
    let speculativeCategories = [
        "prompt-lookup decoding",
        "n-gram, suffix, or token-history drafting",
        "same-target lookahead",
        "two-, three-, or more-row target-model paths",
        "cross-request future-token, future-logit, or future-KV buffering",
        "deferred KV rows and commit, rollback, recommit, or discard markers",
        "pre-hello or initialization warmup",
    ]
    for category in speculativeCategories {
        #expect(
            staticReview.contains(category),
            "static-review policy is missing explicit category: \(category)"
        )
    }
    #expect(staticReview.contains("Controlling serial-track rule: ${serial_decode_rule}"))
    #expect(staticReview.contains("controlling_serial_track_rule: $serial_decode_rule"))
    #expect(staticReview.contains("generic, bit-exact, or production-useful"))
    #expect(staticReview.contains("current-token-only serial decode"))
    #expect(staticReview.contains("ordinary within-request KV reuse"))
    #expect(staticReview.contains("every row corresponds to a token supplied in that same invocation"))
    #expect(staticReview.contains("separate_mtp_track"))
    // Attribution: the judge receives the base..head diff and is told to judge
    // the CHANGES, so code inherited from trusted main inside a touched file
    // cannot by itself fail an innocent submission.
    #expect(staticReview.contains("submission_diff: $submission_diff"))
    #expect(staticReview.contains("judge WHAT THIS SUBMISSION CHANGED"))
    // The published participant rule the review cites must exist.
    let contract = try String(contentsOfFile: "CLAUDE.md", encoding: .utf8)
    #expect(contract.contains("whose only\npossible hit is the benchmark harness repeating an identical computation"))

    for path in ["TASK.md", "AGENTS.md", "CLAUDE.md"] {
        let publishedRule = try String(contentsOfFile: path, encoding: .utf8)
        let normalizedRule = publishedRule.lowercased()
        #expect(
            normalizedRule.contains("serial non-speculative"),
            "\(path) is missing the serial-track rule"
        )
        #expect(normalizedRule.contains("prompt-lookup decoding"))
        #expect(normalizedRule.contains("same-target lookahead"))
        #expect(normalizedRule.contains("deferred cache rows"))
        #expect(normalizedRule.contains("one position and leaves no pending future token"))
        #expect(normalizedRule.contains("default"))
        #expect(normalizedRule.contains("laguna xs 2.1"))
        // The MTP track is retired: participant docs must no longer describe
        // its block protocol as a live default.
        #expect(!normalizedRule.contains("mtp_decode_block"))
    }
}

@Test
func submissionStaticReviewDiffModeFailsClosedAndSendsOnlyChangedFiles() throws {
    let fm = FileManager.default
    let scriptPath = URL(fileURLWithPath: fm.currentDirectoryPath)
        .appendingPathComponent(".github/scripts/run-submission-static-review.sh").path

    let root = fm.temporaryDirectory
        .appendingPathComponent("static-review-\(UUID().uuidString)")
    try fm.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? fm.removeItem(at: root) }
    let repo = root.appendingPathComponent("repo").path
    let privatePath = root.appendingPathComponent("private").path
    try fm.createDirectory(atPath: repo, withIntermediateDirectories: true)

    func run(
        _ argv: [String], env extra: [String: String] = [:]
    ) throws -> (status: Int32, output: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = argv
        process.currentDirectoryURL = URL(fileURLWithPath: repo)
        var env = ProcessInfo.processInfo.environment
        let strayKeys = env.keys.filter {
            $0.hasPrefix("MLXFAST_") || $0.hasPrefix("ANTHROPIC_")
        }
        for key in strayKeys { env.removeValue(forKey: key) }
        env.merge(extra) { _, override in override }
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(data: data, encoding: .utf8) ?? "")
    }

    @discardableResult
    func git(_ args: [String]) throws -> String {
        let result = try run(
            ["git", "-c", "user.email=test@test", "-c", "user.name=test",
             "-c", "commit.gpgsign=false"] + args)
        #expect(result.status == 0, "git \(args.joined(separator: " ")): \(result.output)")
        return result.output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func review(
        base: String?, head: String, env extra: [String: String] = [:]
    ) throws -> (status: Int32, output: String) {
        var env: [String: String] = [
            "ANTHROPIC_API_KEY": "test-key-never-sent",
            "MLXFAST_PRIVATE_DIR": privatePath,
            "HEAD_SHA": head,
        ]
        if let base { env["MLXFAST_SUBMISSION_REVIEW_BASE_SHA"] = base }
        env.merge(extra) { _, override in override }
        return try run(["bash", scriptPath], env: env)
    }

    // The script's own runtime dependencies; skip quietly where they are absent.
    for tool in ["git", "jq", "bash"] {
        guard try run(["sh", "-c", "command -v \(tool)"]).status == 0 else { return }
    }

    try git(["init", "-q"])
    try #"{"editablePaths":["Sources/MLXFastModel"]}"#
        .write(toFile: repo + "/benchmark.json", atomically: true, encoding: .utf8)
    try fm.createDirectory(
        atPath: repo + "/Sources/MLXFastModel", withIntermediateDirectories: true)
    try "let changed = 1\n".write(
        toFile: repo + "/Sources/MLXFastModel/Changed.swift", atomically: true, encoding: .utf8)
    // Unchanged baseline file that LOOKS suspicious -- must never reach the judge.
    try "// prove the benchmark detects slower measured decode\nlet hook = 0\n".write(
        toFile: repo + "/Sources/MLXFastModel/Baseline.swift", atomically: true, encoding: .utf8)
    try git(["add", "."])
    try git(["commit", "-q", "-m", "base"])
    let baseSha = try git(["rev-parse", "HEAD"])
    try "let changed = 2\n".write(
        toFile: repo + "/Sources/MLXFastModel/Changed.swift", atomically: true, encoding: .utf8)
    try git(["commit", "-q", "-am", "change"])
    let headSha = try git(["rev-parse", "HEAD"])

    // Set-but-empty base (the masked merge-base failure) fails closed.
    let emptyBase = try review(base: "", head: headSha)
    #expect(emptyBase.status != 0)
    #expect(emptyBase.output.contains("is set but empty"))

    // An unresolvable base fails closed.
    let badBase = try review(base: String(repeating: "0", count: 40), head: headSha)
    #expect(badBase.status != 0)
    #expect(badBase.output.contains("is not a resolvable commit"))

    // A head that is not the checked-out HEAD fails closed (the diff would not
    // describe the work-tree content the judge reads).
    let staleHead = try review(base: baseSha, head: baseSha)
    #expect(staleHead.status != 0)
    #expect(staleHead.output.contains("not the checked-out HEAD"))

    // Nothing changed (base == head) is a clean pass without any API call.
    let emptyDiff = try review(base: headSha, head: headSha)
    #expect(emptyDiff.status == 0, emptyDiff.output.isEmpty ? "no output" : "\(emptyDiff.output)")
    #expect(emptyDiff.output.contains("no editable files changed versus"))
    let emptyDiffResults = try String(
        contentsOfFile: privatePath + "/submission_static_review.json", encoding: .utf8)
    #expect(emptyDiffResults.contains("\"passed\":true"))
    #expect(emptyDiffResults.contains(headSha))

    // A changed path that is not a regular file in the work tree fails closed.
    try fm.removeItem(atPath: repo + "/Sources/MLXFastModel/Changed.swift")
    try fm.createSymbolicLink(
        atPath: repo + "/Sources/MLXFastModel/Changed.swift",
        withDestinationPath: "../../benchmark.json")
    let symlinked = try review(base: baseSha, head: headSha)
    #expect(symlinked.status != 0)
    #expect(symlinked.output.contains("missing or not a regular file"))
    try git(["checkout", "--", "Sources/MLXFastModel/Changed.swift"])

    // A base contract with no editablePaths is an error, not a clean pass
    // (jq failures inside process substitutions are invisible to set -e).
    try #"{"schemaVersion":1}"#
        .write(toFile: repo + "/benchmark.json", atomically: true, encoding: .utf8)
    try "let changed = 3\n".write(
        toFile: repo + "/Sources/MLXFastModel/Changed.swift", atomically: true, encoding: .utf8)
    try git(["commit", "-q", "-am", "contract without editablePaths"])
    let contractlessHead = try git(["rev-parse", "HEAD"])
    let noPaths = try review(base: contractlessHead, head: contractlessHead)
    #expect(noPaths.status != 0)
    #expect(noPaths.output.contains("lists no editablePaths"))

    // Happy path via a curl shim: the request must carry only the changed
    // editable file (the good contract comes from the BASE commit even though
    // the work-tree copy now lacks editablePaths).
    let shimDir = root.appendingPathComponent("bin").path
    try fm.createDirectory(atPath: shimDir, withIntermediateDirectories: true)
    let capturePath = root.appendingPathComponent("request-capture.json").path
    let shim = """
    #!/usr/bin/env bash
    set -euo pipefail
    out=""
    data=""
    prev=""
    for arg in "$@"; do
      case "${prev}" in
        --output) out="${arg}" ;;
        --data) data="${arg}" ;;
      esac
      prev="${arg}"
    done
    cp "${data#@}" "${CURL_SHIM_CAPTURE}"
    printf '%s' '{"content":[{"type":"text","text":"{\\"passed\\":true,\\"severity\\":\\"none\\",\\"summary\\":\\"ok\\",\\"findings\\":[]}"}]}' > "${out}"
    """
    try shim.write(toFile: shimDir + "/curl", atomically: true, encoding: .utf8)
    try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: shimDir + "/curl")
    let inheritedPath = ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin"
    let happy = try review(
        base: baseSha, head: contractlessHead,
        env: ["PATH": shimDir + ":" + inheritedPath, "CURL_SHIM_CAPTURE": capturePath])
    #expect(happy.status == 0, happy.output.isEmpty ? "no output" : "\(happy.output)")
    let request = try String(contentsOfFile: capturePath, encoding: .utf8)
    #expect(request.contains("Sources/MLXFastModel/Changed.swift"))
    #expect(!request.contains("Baseline.swift"))
    #expect(!request.contains("prove the benchmark detects slower measured decode"))

    // Assert the actual API payload, not merely inert strings in the script,
    // carries the controlling rule and every serial-track category in both the
    // system instruction and structured user policy.
    let requestObject = try #require(
        try JSONSerialization.jsonObject(with: Data(request.utf8)) as? [String: Any]
    )
    let systemPrompt = try #require(requestObject["system"] as? String)
    let messages = try #require(requestObject["messages"] as? [[String: Any]])
    let content = try #require(messages.first?["content"] as? [[String: Any]])
    let userText = try #require(content.first?["text"] as? String)
    let userPolicy = try #require(
        try JSONSerialization.jsonObject(with: Data(userText.utf8)) as? [String: Any]
    )
    let controllingRule = try #require(
        userPolicy["controlling_serial_track_rule"] as? String
    )
    #expect(systemPrompt.contains(controllingRule))
    #expect(controllingRule.contains("only for tokens supplied in that invocation"))
    #expect(controllingRule.contains("exactly the supplied input length"))
    #expect(controllingRule.contains("leaves no pending future token, logits, or KV state"))

    let policy = try #require(userPolicy["policy"] as? [String: Any])
    let failOn = try #require(policy["fail_on"] as? [String])
    let allowed = try #require(policy["allow"] as? [String])
    for category in [
        "prompt-lookup decoding",
        "same-target lookahead",
        "two-, three-, or more-row target-model paths",
        "cross-request future-token, future-logit, or future-KV buffering",
        "pre-hello or initialization warmup",
    ] {
        #expect(
            failOn.contains { $0.contains(category) },
            "outbound static-review policy is missing category: \(category)"
        )
        #expect(systemPrompt.contains(category))
    }
    for category in [
        "ordinary within-request KV reuse",
        "current-token-only serial decode",
        "multi-row kernels or batching",
    ] {
        #expect(
            allowed.contains { $0.contains(category) },
            "outbound static-review allowlist is missing category: \(category)"
        )
    }

    // A deletion-only submission still has executable meaning and must reach
    // the judge through submission_diff instead of taking the no-files pass.
    try git(["checkout", "-q", "-b", "deletion-only", baseSha])
    try fm.removeItem(atPath: repo + "/Sources/MLXFastModel/Baseline.swift")
    try git(["add", "-A"])
    try git(["commit", "-q", "-m", "delete editable source"])
    let deletionHead = try git(["rev-parse", "HEAD"])
    let deletionCapture = root.appendingPathComponent("deletion-request.json").path
    let deletion = try review(
        base: baseSha,
        head: deletionHead,
        env: ["PATH": shimDir + ":" + inheritedPath, "CURL_SHIM_CAPTURE": deletionCapture]
    )
    #expect(deletion.status == 0, deletion.output.isEmpty ? "no output" : "\(deletion.output)")
    let deletionRequest = try String(contentsOfFile: deletionCapture, encoding: .utf8)
    #expect(deletionRequest.contains("Sources/MLXFastModel/Baseline.swift"))
    #expect(deletionRequest.contains("prove the benchmark detects slower measured decode"))
    // Any deletion of trusted-base editable files also prints the stale-clone
    // note, even when the review itself passes.
    #expect(deletion.output.contains("1 editable file(s) deleted versus base"))
}

// Behavioral pin for the Qwen-MTP arm of run-submission-static-review.sh's
// track policy allowlist -- the RANKED track of this repository.
//
// Before this arm existed the script accepted only `serial` and
// `dflash|laguna-xs-2.1-dflash-v1` and fell through to the fail-closed default,
// so qwen-mtp-ranked-benchmark.yml's review step refused every submission. The
// fail-closed default was CORRECT interim behaviour and must survive: the two
// live policies are opposite, and reviewing a native-MTP submission under the
// serial policy would make every legal submission a critical finding (the MTP
// head is literally an "auxiliary prediction head" the serial arm fails on),
// while reviewing a serial submission under the MTP policy would permit exactly
// the lookahead the serial track exists to prohibit.
//
// This asserts the OUTBOUND PAYLOAD, not merely inert strings in the script:
// which controlling rule the judge is handed, which fail_on/allow categories,
// and which harness_protocol facts -- because the serial categories are what
// would otherwise fail an honest k-test submission.
@Test
func submissionStaticReviewQwenMTPArmSwapsThePolicyAndStillFailsClosed() throws {
    let fm = FileManager.default
    let scriptPath = URL(fileURLWithPath: fm.currentDirectoryPath)
        .appendingPathComponent(".github/scripts/run-submission-static-review.sh").path

    let root = fm.temporaryDirectory
        .appendingPathComponent("static-review-qwen-mtp-\(UUID().uuidString)")
    try fm.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? fm.removeItem(at: root) }
    let repo = root.appendingPathComponent("repo").path
    let privatePath = root.appendingPathComponent("private").path
    try fm.createDirectory(atPath: repo, withIntermediateDirectories: true)

    func run(
        _ argv: [String], env extra: [String: String] = [:]
    ) throws -> (status: Int32, output: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = argv
        process.currentDirectoryURL = URL(fileURLWithPath: repo)
        var env = ProcessInfo.processInfo.environment
        let strayKeys = env.keys.filter {
            $0.hasPrefix("MLXFAST_") || $0.hasPrefix("ANTHROPIC_")
        }
        for key in strayKeys { env.removeValue(forKey: key) }
        env.merge(extra) { _, override in override }
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(data: data, encoding: .utf8) ?? "")
    }

    @discardableResult
    func git(_ args: [String]) throws -> String {
        let result = try run(
            ["git", "-c", "user.email=test@test", "-c", "user.name=test",
             "-c", "commit.gpgsign=false"] + args)
        #expect(result.status == 0, "git \(args.joined(separator: " ")): \(result.output)")
        return result.output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    for tool in ["git", "jq", "bash"] {
        guard try run(["sh", "-c", "command -v \(tool)"]).status == 0 else { return }
    }

    // A minimal stand-in for this track's editable surface: the real manifest
    // lists Sources/MLXFastModel as a directory, and the k-test seam
    // Qwen36MTPBlockSession.swift lives inside it.
    try git(["init", "-q"])
    try #"{"editablePaths":["Sources/MLXFastModel"]}"#
        .write(toFile: repo + "/benchmark.json", atomically: true, encoding: .utf8)
    try fm.createDirectory(
        atPath: repo + "/Sources/MLXFastModel", withIntermediateDirectories: true)
    try "let depth = 2\n".write(
        toFile: repo + "/Sources/MLXFastModel/Qwen36MTPBlockSession.swift",
        atomically: true, encoding: .utf8)
    try git(["add", "."])
    try git(["commit", "-q", "-m", "base"])
    let baseSha = try git(["rev-parse", "HEAD"])
    try "let effectiveDepth = Swift.min(depth, 1)\n".write(
        toFile: repo + "/Sources/MLXFastModel/Qwen36MTPBlockSession.swift",
        atomically: true, encoding: .utf8)
    try git(["commit", "-q", "-am", "k-test k=1"])
    let headSha = try git(["rev-parse", "HEAD"])

    let shimDir = root.appendingPathComponent("bin").path
    try fm.createDirectory(atPath: shimDir, withIntermediateDirectories: true)
    let shim = """
    #!/usr/bin/env bash
    set -euo pipefail
    out=""
    data=""
    prev=""
    for arg in "$@"; do
      case "${prev}" in
        --output) out="${arg}" ;;
        --data) data="${arg}" ;;
      esac
      prev="${arg}"
    done
    cp "${data#@}" "${CURL_SHIM_CAPTURE}"
    printf '%s' '{"content":[{"type":"text","text":"{\\"passed\\":true,\\"severity\\":\\"none\\",\\"summary\\":\\"ok\\",\\"findings\\":[]}"}]}' > "${out}"
    """
    try shim.write(toFile: shimDir + "/curl", atomically: true, encoding: .utf8)
    try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: shimDir + "/curl")
    let inheritedPath = ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin"

    func review(trackID: String?, capture: String) throws -> (status: Int32, output: String) {
        var env: [String: String] = [
            "ANTHROPIC_API_KEY": "test-key-never-sent",
            "MLXFAST_PRIVATE_DIR": privatePath,
            "HEAD_SHA": headSha,
            "MLXFAST_SUBMISSION_REVIEW_BASE_SHA": baseSha,
            "PATH": shimDir + ":" + inheritedPath,
            "CURL_SHIM_CAPTURE": capture,
        ]
        if let trackID { env["MLXFAST_SUBMISSION_TRACK_ID"] = trackID }
        return try run(["bash", scriptPath], env: env)
    }

    func payload(_ capture: String) throws -> (system: String, user: [String: Any]) {
        let request = try String(contentsOfFile: capture, encoding: .utf8)
        let requestObject = try #require(
            try JSONSerialization.jsonObject(with: Data(request.utf8)) as? [String: Any]
        )
        let system = try #require(requestObject["system"] as? String)
        let messages = try #require(requestObject["messages"] as? [[String: Any]])
        let content = try #require(messages.first?["content"] as? [[String: Any]])
        let userText = try #require(content.first?["text"] as? String)
        let user = try #require(
            try JSONSerialization.jsonObject(with: Data(userText.utf8)) as? [String: Any]
        )
        return (system, user)
    }

    // 1. THE GAP IS CLOSED. The manifest's staticReviewTrackId is accepted and
    //    normalises to the short id.
    let qwenCapture = root.appendingPathComponent("qwen-request.json").path
    let qwen = try review(trackID: "qwen3.8-27b-mtp-v1", capture: qwenCapture)
    #expect(qwen.status == 0, qwen.output.isEmpty ? "no output" : "\(qwen.output)")
    let qwenPayload = try payload(qwenCapture)
    #expect(qwenPayload.user["track_id"] as? String == "qwen-mtp")
    // The short alias resolves to the same policy.
    let aliasCapture = root.appendingPathComponent("alias-request.json").path
    #expect(try review(trackID: "qwen-mtp", capture: aliasCapture).status == 0)
    #expect(try payload(aliasCapture).user["track_id"] as? String == "qwen-mtp")

    // 2. THE CONTROLLING RULE IS THE MTP RULE, and it reaches the system prompt.
    let controlling = try #require(qwenPayload.user["controlling_track_rule"] as? String)
    #expect(controlling.contains("OWN pinned native MTP head"))
    #expect(controlling.contains("longest common prefix"))
    #expect(controlling.contains("every value in the round"))
    #expect(qwenPayload.system.contains(controlling))
    #expect(qwenPayload.system.contains("Controlling Qwen-MTP-track rule:"))
    // The retired serial rule must be explicitly demoted, not silently present.
    #expect(qwenPayload.system.contains("does NOT govern this track"))

    // 3. THE EDITABLE SURFACE THIS TRACK ACTUALLY SHIPS is named, so editing the
    //    MTP seam is not itself read as suspicious, and the oracle sitting in
    //    the same editable directory is called out.
    for surface in [
        "Qwen36MTPBlockSession.swift",
        "Qwen36MTPHeadAttachment.swift",
        "Qwen36MTPReferenceSession.swift",
        "Sources/MLXFastTransform",
        "Qwen35MTP.swift",
        "CompilableRotatingKVCache.swift",
    ] {
        #expect(
            qwenPayload.system.contains(surface),
            "Qwen-MTP static-review prompt does not name editable surface: \(surface)"
        )
    }

    // 4. THE SERIAL CATEGORIES ARE SWAPPED OUT, not merely overridden in prose.
    //    An honest k-test submission drafts from the MTP head and verifies
    //    multiple rows; leaving these in fail_on invites exactly the
    //    false-positive this arm exists to prevent.
    let qwenPolicy = try #require(qwenPayload.user["policy"] as? [String: Any])
    let qwenFailOn = try #require(qwenPolicy["fail_on"] as? [String])
    let qwenAllow = try #require(qwenPolicy["allow"] as? [String])
    for retired in [
        "prompt-lookup decoding, including n-gram",
        "same-target lookahead",
        "multi-row speculative target execution",
        "auxiliary or additional prediction heads",
        "self-drafting with no separate drafter model",
    ] {
        #expect(
            !qwenFailOn.contains { $0.contains(retired) },
            "Qwen-MTP fail_on still carries the serial category: \(retired)"
        )
    }
    for category in [
        "emitting a token that the target verify row for that position did not select",
        "verifying with a degraded target",
        "undeclared head substitution",
        "leaving rejected rows in the caches",
        "fabricating, approximating, replaying, or precomputing any reported ledger value",
        "detecting whether the current phase is the reference replay",
    ] {
        #expect(
            qwenFailOn.contains { $0.contains(category) },
            "Qwen-MTP fail_on is missing category: \(category)"
        )
    }
    // The k=1 submission under test is legal precisely because a candidate owns
    // its own per-round draft schedule. Since 2026-08-14 so is bringing a
    // DECLARED head, and both have to be spelled out for the judge: a policy
    // that fails legal work is as broken as one that passes cheating.
    for allowed in [
        "native-MTP block speculation itself",
        "any per-round draft schedule from 0 to the trusted maximum of 8",
        "bringing a DIFFERENT MTP head, declared in mtp-head.manifest.json",
        "snapshot, trim, restore and re-forward rollback",
    ] {
        #expect(
            qwenAllow.contains { $0.contains(allowed) },
            "Qwen-MTP allow is missing category: \(allowed)"
        )
    }

    // 5. HARNESS FACTS match this track. The serial arm tells the judge that no
    //    MTP track exists; handing that to a Qwen-MTP judge is worse than
    //    useless.
    let qwenHarness = try #require(qwenPayload.user["harness_protocol"] as? [String: Any])
    #expect(qwenHarness["separate_mtp_track"] == nil)
    #expect(qwenHarness["serial_decode"] == nil)
    #expect((qwenHarness["native_mtp_decode"] as? String)?.contains("declared row ledger") == true)
    // Depth 0, not depth 1: the key was renamed on 2026-08-14 along with the
    // description, which had the serial control wrong. Depth 1 drafts.
    #expect(qwenHarness["depth_1_is_the_serial_control"] == nil)
    #expect(qwenHarness["depth_0_is_the_serial_control"] != nil)
    #expect(qwenHarness["ledger_audit"] != nil)
    #expect(qwenHarness["rollback_contract"] != nil)
    // Shared facts survive the swap.
    #expect(qwenHarness["fresh_worker_process_per_phase"] as? Bool == true)
    #expect(qwenHarness["invariant"] != nil)
    let qwenDecision = try #require(qwenPayload.user["decision_test"] as? String)
    #expect(qwenDecision.contains("mtp-head.manifest.json declares"))
    #expect(qwenDecision.contains("if rejected rows survive in any cache"))

    // 6. THE SERIAL ARM IS UNCHANGED. Default (unset) track still ships the
    //    serial policy, so this arm did not widen the retired track.
    let serialCapture = root.appendingPathComponent("serial-request.json").path
    #expect(try review(trackID: nil, capture: serialCapture).status == 0)
    let serialPayload = try payload(serialCapture)
    #expect(serialPayload.user["track_id"] as? String == "serial")
    let serialHarness = try #require(serialPayload.user["harness_protocol"] as? [String: Any])
    #expect(serialHarness["separate_mtp_track"] != nil)
    #expect(serialHarness["native_mtp_decode"] == nil)
    let serialFailOn = try #require(
        (serialPayload.user["policy"] as? [String: Any])?["fail_on"] as? [String])
    #expect(serialFailOn.contains { $0.contains("prompt-lookup decoding, including n-gram") })
    #expect(!serialFailOn.contains { $0.contains("pinned native MTP head") })

    // 7. THE DFLASH ARM IS UNCHANGED and still distinct from this one.
    let dflashCapture = root.appendingPathComponent("dflash-request.json").path
    #expect(try review(trackID: "laguna-xs-2.1-dflash-v1", capture: dflashCapture).status == 0)
    let dflashPayload = try payload(dflashCapture)
    #expect(dflashPayload.user["track_id"] as? String == "dflash")
    #expect(dflashPayload.system.contains("Controlling DFlash-track rule:"))
    #expect(!dflashPayload.system.contains("Controlling Qwen-MTP-track rule:"))
    #expect(
        (dflashPayload.user["controlling_track_rule"] as? String)?
            .contains("organizer-provisioned DFlash drafter") == true
    )

    // 8. THE FAIL-CLOSED DEFAULT SURVIVES. An unknown track -- including the
    //    benchmark NAME and the retired staging-branch name, both plausible
    //    copy-paste mistakes -- must never be reviewed under a neighbouring
    //    policy.
    for unknown in [
        "mlxfast-challenge-dev-qwen38-mtp",
        "qwen36-mtp-track",
        "qwen3.6-27b-mtp-v2",
        "QWEN3.6-27B-MTP-V1",
    ] {
        let refused = try review(
            trackID: unknown, capture: root.appendingPathComponent("unused.json").path)
        #expect(refused.status != 0, "track '\(unknown)' was not refused")
        #expect(refused.output.contains("unsupported submission static-review track"))
    }

    // 9. SET-BUT-EMPTY is the same failure wearing a different hat: the caller
    //    meant to name a track and its lookup produced nothing. `${VAR:-serial}`
    //    would have silently reviewed a Qwen-MTP submission under the serial
    //    policy, which is the exact outcome the allowlist refuses. Only a
    //    genuinely UNSET variable may mean the serial default.
    let emptyTrack = try review(
        trackID: "", capture: root.appendingPathComponent("unused.json").path)
    #expect(emptyTrack.status != 0)
    #expect(emptyTrack.output.contains("MLXFAST_SUBMISSION_TRACK_ID is set but empty"))
}

// The ranked Qwen-MTP pipeline has THREE consumers of an editable surface --
// enforce-modifiable-surface.sh, run-submission-static-review.sh and
// overlay-editable-paths.sh -- and all three default to benchmark.json when
// CONTRACT_PATH is unset. They must never disagree: judging a submission against
// one surface while enforcing and overlaying another is how a submitter's edits
// get accepted by one gate and silently dropped by the next. Pin that the
// workflow names this track's manifest at every one of them, and that the
// manifest it names is the one the static-review arm was authored against.
@Test
func qwenMTPWorkflowNamesTheTrackManifestAtEveryEditableSurfaceConsumer() throws {
    let workflow = try String(
        contentsOfFile: ".github/workflows/qwen-mtp-ranked-benchmark.yml", encoding: .utf8)
    let occurrences = workflow.components(
        separatedBy: "CONTRACT_PATH=\"${MLXFAST_QWEN_MTP_EDITABLE_SURFACE_CONTRACT}\""
    ).count - 1
    let overlayOccurrences = workflow.components(
        separatedBy: "CONTRACT_PATH: ${{ env.MLXFAST_QWEN_MTP_EDITABLE_SURFACE_CONTRACT }}"
    ).count - 1
    // enforce-modifiable-surface.sh and run-submission-static-review.sh are
    // invoked inline (shell assignment); overlay-editable-paths.sh takes it as
    // step env.
    #expect(
        occurrences == 2,
        "expected both inline editable-surface consumers to set CONTRACT_PATH, found \(occurrences)"
    )
    #expect(overlayOccurrences == 1)
    #expect(workflow.contains("MLXFAST_QWEN_MTP_EDITABLE_SURFACE_CONTRACT: benchmark.qwen-mtp.json"))

    // The static-review arm reads editablePaths from whichever manifest it is
    // handed, so the named manifest must actually carry this track's surface.
    let contract = try #require(
        try JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: "benchmark.qwen-mtp.json"))
        ) as? [String: Any]
    )
    #expect(contract["staticReviewTrackId"] as? String == "qwen3.8-27b-mtp-v1")
    let editablePaths = try #require(contract["editablePaths"] as? [String])
    #expect(editablePaths.contains("Sources/MLXFastModel"))
    #expect(editablePaths.contains("Sources/MLXFastTransform"))
    // The k-test seam lives under Sources/MLXFastModel, so the directory entry
    // is what makes an MTP-session edit a legal submission at all.
    #expect(FileManager.default.fileExists(
        atPath: "Sources/MLXFastModel/Qwen36MTPBlockSession.swift"))

    // The stale note claiming the arm does not exist must be gone -- it told a
    // future operator that this step refuses, which is now false. (The unrelated
    // redactor-vocabulary reconciliation item near the end of the workflow is
    // still genuinely open and is deliberately not covered here.)
    #expect(!workflow.contains("so this step refuses today"))
    #expect(!workflow.contains(
        "allowlist accepts only `serial` and `dflash|laguna-xs-2.1-dflash-v1`"))
}

// Behavioral pin for the stale-clone diagnostic in run-submission-static-
// review.sh's diff-only mode. Run 29549885232 (submission cbe5c48e) failed the
// byte cap with only "refusing oversized source that could hide lookup
// tables": the participant's clone predated the #630 Vendor/ editable-surface
// expansion, so the rebuilt submission commit deleted all 89 vendored editable
// files and the deletion hunks alone (~1.39 MB of trusted-base content) blew
// the 1.5 MB cap. The oversize refusal must keep failing closed, but it must
// name the deletions and the re-sync remedy instead of only implying cheating.
@Test
func submissionStaticReviewOversizeFailureExplainsStaleCloneDeletions() throws {
    let fm = FileManager.default
    let scriptPath = URL(fileURLWithPath: fm.currentDirectoryPath)
        .appendingPathComponent(".github/scripts/run-submission-static-review.sh").path

    let root = fm.temporaryDirectory
        .appendingPathComponent("static-review-stale-\(UUID().uuidString)")
    try fm.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? fm.removeItem(at: root) }
    let repo = root.appendingPathComponent("repo").path
    let privatePath = root.appendingPathComponent("private").path
    try fm.createDirectory(atPath: repo, withIntermediateDirectories: true)

    func run(
        _ argv: [String], env extra: [String: String] = [:]
    ) throws -> (status: Int32, output: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = argv
        process.currentDirectoryURL = URL(fileURLWithPath: repo)
        var env = ProcessInfo.processInfo.environment
        let strayKeys = env.keys.filter {
            $0.hasPrefix("MLXFAST_") || $0.hasPrefix("ANTHROPIC_")
        }
        for key in strayKeys { env.removeValue(forKey: key) }
        env.merge(extra) { _, override in override }
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(data: data, encoding: .utf8) ?? "")
    }

    @discardableResult
    func git(_ args: [String]) throws -> String {
        let result = try run(
            ["git", "-c", "user.email=test@test", "-c", "user.name=test",
             "-c", "commit.gpgsign=false"] + args)
        #expect(result.status == 0, "git \(args.joined(separator: " ")): \(result.output)")
        return result.output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    for tool in ["git", "jq", "bash"] {
        guard try run(["sh", "-c", "command -v \(tool)"]).status == 0 else { return }
    }

    // Trusted base: a small hand-written source plus a vendored editable file
    // (stands in for the #630 Vendor/ kernel surface).
    try git(["init", "-q"])
    try #"{"editablePaths":["Sources/MLXFastModel","Vendor/mlx-swift"]}"#
        .write(toFile: repo + "/benchmark.json", atomically: true, encoding: .utf8)
    try fm.createDirectory(
        atPath: repo + "/Sources/MLXFastModel", withIntermediateDirectories: true)
    try fm.createDirectory(
        atPath: repo + "/Vendor/mlx-swift", withIntermediateDirectories: true)
    try "let model = 1\n".write(
        toFile: repo + "/Sources/MLXFastModel/Model.swift", atomically: true, encoding: .utf8)
    try String(repeating: "// vendored kernel line\n", count: 64).write(
        toFile: repo + "/Vendor/mlx-swift/Kernel.metal", atomically: true, encoding: .utf8)
    try git(["add", "."])
    try git(["commit", "-q", "-m", "trusted base with vendored surface"])
    let baseSha = try git(["rev-parse", "HEAD"])

    // Stale-clone shaped head: edits the model source but deletes the vendored
    // file it never had. The deletion hunk (trusted-base content) dominates
    // the reviewed bytes, exactly like run 29549885232.
    try "let model = 2\n".write(
        toFile: repo + "/Sources/MLXFastModel/Model.swift", atomically: true, encoding: .utf8)
    try fm.removeItem(atPath: repo + "/Vendor/mlx-swift/Kernel.metal")
    try git(["add", "-A"])
    try git(["commit", "-q", "-m", "stale-clone submission"])
    let headSha = try git(["rev-parse", "HEAD"])

    let oversize = try run(
        ["bash", scriptPath],
        env: [
            "ANTHROPIC_API_KEY": "test-key-never-sent",
            "MLXFAST_PRIVATE_DIR": privatePath,
            "MLXFAST_SUBMISSION_REVIEW_BASE_SHA": baseSha,
            "HEAD_SHA": headSha,
            // Below the deletion-dominated diff size, so the cap trips before
            // any network call (a real curl would fail on the dummy key).
            "MLXFAST_SUBMISSION_STATIC_REVIEW_MAX_BYTES": "512",
        ])
    #expect(oversize.status != 0)
    #expect(oversize.output.contains("above static review limit 512"))
    #expect(oversize.output.contains("refusing oversized source that could hide lookup tables"))
    #expect(oversize.output.contains("deletes 1 editable file(s) that exist on the trusted base"))
    #expect(oversize.output.contains("Vendor/mlx-swift/Kernel.metal"))
    #expect(oversize.output.contains("stale clone"))
    #expect(oversize.output.contains("yukon sync"))
    #expect(oversize.output.contains("resubmit"))
}

// Mechanical byte budgets for the enlarged editable surface: the per-file cap
// and the base-to-head growth cap must fail closed BEFORE any judge call, so
// a submission cannot smuggle a lookup table under the total cap by touching
// only a few files.
@Test
func submissionStaticReviewCapsFileSizeAndSurfaceGrowth() throws {
    let fm = FileManager.default
    let scriptPath = URL(fileURLWithPath: fm.currentDirectoryPath)
        .appendingPathComponent(".github/scripts/run-submission-static-review.sh").path

    let root = fm.temporaryDirectory
        .appendingPathComponent("static-review-caps-\(UUID().uuidString)")
    try fm.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? fm.removeItem(at: root) }
    let repo = root.appendingPathComponent("repo").path
    let privatePath = root.appendingPathComponent("private").path
    try fm.createDirectory(atPath: repo, withIntermediateDirectories: true)

    func run(
        _ argv: [String], env extra: [String: String] = [:]
    ) throws -> (status: Int32, output: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = argv
        process.currentDirectoryURL = URL(fileURLWithPath: repo)
        var env = ProcessInfo.processInfo.environment
        let strayKeys = env.keys.filter {
            $0.hasPrefix("MLXFAST_") || $0.hasPrefix("ANTHROPIC_")
        }
        for key in strayKeys { env.removeValue(forKey: key) }
        env.merge(extra) { _, override in override }
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(data: data, encoding: .utf8) ?? "")
    }
    for tool in ["git", "jq", "bash"] {
        guard try run(["sh", "-c", "command -v \(tool)"]).status == 0 else { return }
    }

    @discardableResult
    func git(_ args: [String]) throws -> String {
        let result = try run(
            ["git", "-c", "user.email=test@test", "-c", "user.name=test",
             "-c", "commit.gpgsign=false"] + args)
        #expect(result.status == 0, "git \(args.joined(separator: " ")): \(result.output)")
        return result.output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func review(
        base: String, head: String, env extra: [String: String] = [:]
    ) throws -> (status: Int32, output: String) {
        var env: [String: String] = [
            "ANTHROPIC_API_KEY": "test-key-never-sent",
            "MLXFAST_PRIVATE_DIR": privatePath,
            "HEAD_SHA": head,
            "MLXFAST_SUBMISSION_REVIEW_BASE_SHA": base,
        ]
        env.merge(extra) { _, override in override }
        return try run(["bash", scriptPath], env: env)
    }

    try git(["init", "-q"])
    try #"{"editablePaths":["Sources/MLXFastModel"]}"#
        .write(toFile: repo + "/benchmark.json", atomically: true, encoding: .utf8)
    try fm.createDirectory(
        atPath: repo + "/Sources/MLXFastModel", withIntermediateDirectories: true)
    try "let base = 1\n".write(
        toFile: repo + "/Sources/MLXFastModel/Kernel.swift", atomically: true, encoding: .utf8)
    try git(["add", "."])
    try git(["commit", "-q", "-m", "base"])
    let baseSha = try git(["rev-parse", "HEAD"])

    // Head adds ~300 KB of new "table" bytes inside one editable file.
    let table = "// table\n" + String(repeating: "0123456789abcdef", count: 19_000)
    try table.write(
        toFile: repo + "/Sources/MLXFastModel/Kernel.swift", atomically: true, encoding: .utf8)
    try git(["commit", "-q", "-am", "grow"])
    let grownHead = try git(["rev-parse", "HEAD"])

    // Default growth cap (256 KiB) rejects the ~300 KB addition before any
    // network call (no curl shim exists; reaching curl would fail the test
    // differently).
    let growth = try review(base: baseSha, head: grownHead)
    #expect(growth.status != 0)
    #expect(growth.output.contains("above the growth limit"))

    // With the growth cap raised, the per-file cap still rejects the file.
    let perFile = try review(
        base: baseSha, head: grownHead,
        env: [
            "MLXFAST_SUBMISSION_STATIC_REVIEW_MAX_GROWTH_BYTES": "10000000",
            "MLXFAST_SUBMISSION_STATIC_REVIEW_MAX_FILE_BYTES": "200000",
        ])
    #expect(perFile.status != 0)
    #expect(perFile.output.contains("above the per-file static review limit"))

    // Malformed cap knobs fail closed.
    let badKnob = try review(
        base: baseSha, head: grownHead,
        env: ["MLXFAST_SUBMISSION_STATIC_REVIEW_MAX_GROWTH_BYTES": "lots"])
    #expect(badKnob.status != 0)
    #expect(badKnob.output.contains("MLXFAST_SUBMISSION_STATIC_REVIEW_MAX_GROWTH_BYTES must be a positive integer"))
}

// Behavioral pin for download-r2-object.sh's failure path: the script fetches
// RAW hidden material (correctness golden, GPQA reference), so its error
// diagnostics may print the response body ONLY when the final HTTP status
// proves the body is a server error document (>= 400; curl exit 22 under
// --fail-with-body). The dangerous case is a 200 whose transfer died mid-body
// (curl exit 18/56 through every retry): the temp file then holds a prefix of
// the OBJECT ITSELF, and printing a "bounded error body" would put hidden
// golden bytes into the (public) Actions log. Simulated with a curl shim that
// reproduces curl's observable contract: body lands in --output, the
// --write-out '%{http_code}' code is emitted once on stdout even after
// exhausted retries, and the exit code reports the failure class.
@Test
func downloadR2ObjectWithholdsBodyUnlessServerErrorStatus() throws {
    let fm = FileManager.default
    let scriptPath = URL(fileURLWithPath: fm.currentDirectoryPath)
        .appendingPathComponent(".github/scripts/download-r2-object.sh").path

    let root = fm.temporaryDirectory
        .appendingPathComponent("r2-download-\(UUID().uuidString)")
    try fm.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? fm.removeItem(at: root) }

    func run(
        _ argv: [String], env extra: [String: String] = [:]
    ) throws -> (status: Int32, output: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = argv
        process.currentDirectoryURL = root
        var env = ProcessInfo.processInfo.environment
        let strayKeys = env.keys.filter {
            $0.hasPrefix("MLXFAST_") || $0.hasPrefix("ANTHROPIC_") || $0.hasPrefix("R2_")
        }
        for key in strayKeys { env.removeValue(forKey: key) }
        env.merge(extra) { _, override in override }
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(data: data, encoding: .utf8) ?? "")
    }

    guard try run(["sh", "-c", "command -v bash"]).status == 0 else { return }

    // The shim stands in for curl: writes CURL_SHIM_BODY to the --output
    // target, prints CURL_SHIM_HTTP_CODE (the --write-out result) to stdout,
    // exits CURL_SHIM_EXIT.
    let shimDir = root.appendingPathComponent("bin").path
    try fm.createDirectory(atPath: shimDir, withIntermediateDirectories: true)
    let shim = """
    #!/usr/bin/env bash
    set -euo pipefail
    out=""
    prev=""
    for arg in "$@"; do
      if [[ "${prev}" == "--output" ]]; then out="${arg}"; fi
      prev="${arg}"
    done
    if [[ -n "${out}" && -n "${CURL_SHIM_BODY:-}" ]]; then
      printf '%s' "${CURL_SHIM_BODY}" > "${out}"
    fi
    printf '%s' "${CURL_SHIM_HTTP_CODE:-}"
    exit "${CURL_SHIM_EXIT:-0}"
    """
    try shim.write(toFile: shimDir + "/curl", atomically: true, encoding: .utf8)
    try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: shimDir + "/curl")
    let inheritedPath = ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin"

    // Endpoint WITHOUT a bucket path: base_path is empty, so the aws-cli
    // branch is skipped deterministically even on machines with aws
    // installed and every run reaches the curl path under test.
    let baseEnv: [String: String] = [
        "R2_ACCESS_KEY_ID": "AKIDTESTKEY",
        "R2_SECRET_ACCESS_KEY": "testsecret",
        "R2_BUCKET_ENDPOINT": "https://r2.example.test",
        "PATH": shimDir + ":" + inheritedPath,
        // The script retries in bash now (re-signing each attempt; see
        // R2RequestExecutionTests), so a failing case walks the full attempt
        // budget. Keep the attempt count and drop only the delay between them.
        "R2_RETRY_DELAY_SECONDS": "0",
    ]

    // 200-then-truncated (curl exit 18 after all retries): the temp file is
    // a prefix of the hidden object; NOT ONE BYTE of it may print.
    let hiddenObjectPrefix = "HIDDEN-GOLDEN-OBJECT-BYTES-3c1a9f-DO-NOT-LOG"
    let truncatedOut = root.appendingPathComponent("truncated/golden.json").path
    let truncated = try run(
        ["bash", scriptPath, "correctness_prompts/x.json", truncatedOut],
        env: baseEnv.merging([
            "CURL_SHIM_BODY": hiddenObjectPrefix,
            "CURL_SHIM_HTTP_CODE": "200",
            "CURL_SHIM_EXIT": "18",
        ]) { _, new in new })
    #expect(truncated.status == 18)
    #expect(!truncated.output.contains("HIDDEN-GOLDEN"))
    #expect(!truncated.output.contains(hiddenObjectPrefix))
    #expect(truncated.output.contains("HTTP 200"))
    #expect(truncated.output.contains("body withheld"))
    #expect(truncated.output.contains("\(hiddenObjectPrefix.utf8.count) body byte(s) discarded"))
    // The partial object is also gone from disk (trap cleanup), so a later
    // step cannot leak what this one withheld.
    #expect(!fm.fileExists(atPath: truncatedOut))
    #expect(!fm.fileExists(atPath: truncatedOut + ".tmp"))

    // Provable server error (HTTP 403, curl exit 22): the R2 error document
    // still prints, keeping signature/credential failures diagnosable.
    let deniedOut = root.appendingPathComponent("denied/golden.json").path
    let denied = try run(
        ["bash", scriptPath, "correctness_prompts/x.json", deniedOut],
        env: baseEnv.merging([
            "CURL_SHIM_BODY": "<Error><Code>SignatureDoesNotMatch</Code></Error>",
            "CURL_SHIM_HTTP_CODE": "403",
            "CURL_SHIM_EXIT": "22",
        ]) { _, new in new })
    #expect(denied.status == 22)
    #expect(denied.output.contains("SignatureDoesNotMatch"))
    #expect(denied.output.contains("HTTP 403"))

    // Connection-level failure (curl exit 7): no response at all, code 000;
    // nothing to print and nothing printed.
    let refusedOut = root.appendingPathComponent("refused/golden.json").path
    let refused = try run(
        ["bash", scriptPath, "correctness_prompts/x.json", refusedOut],
        env: baseEnv.merging([
            "CURL_SHIM_HTTP_CODE": "000",
            "CURL_SHIM_EXIT": "7",
        ]) { _, new in new })
    #expect(refused.status == 7)
    #expect(refused.output.contains("body withheld"))
    #expect(refused.output.contains("HTTP 000"))
    #expect(refused.output.contains("0 body byte(s) discarded"))

    // The success path is untouched by the gate.
    let okOut = root.appendingPathComponent("ok/golden.json").path
    let ok = try run(
        ["bash", scriptPath, "correctness_prompts/x.json", okOut],
        env: baseEnv.merging([
            "CURL_SHIM_BODY": "{\"ok\":true}",
            "CURL_SHIM_HTTP_CODE": "200",
            "CURL_SHIM_EXIT": "0",
        ]) { _, new in new })
    #expect(ok.status == 0)
    #expect(ok.output.contains("wrote \(okOut)"))
    #expect(try String(contentsOfFile: okOut, encoding: .utf8) == "{\"ok\":true}")

    // Both R2 scripts carry the same status-gated pattern (upload responses
    // are server-authored, but symmetry keeps the next copy-paste between
    // them from reintroducing the download-side leak), and the old comment's
    // false premise -- that a failed transfer's body is never object
    // content -- is gone from both.
    for scriptFile in ["download-r2-object.sh", "upload-r2-object.sh"] {
        let script = try String(
            contentsOfFile: ".github/scripts/" + scriptFile,
            encoding: .utf8
        )
        #expect(script.contains("--write-out '%{http_code}'"), "\(scriptFile)")
        #expect(script.contains("(( 10#${http_status} >= 400 ))"), "\(scriptFile)")
        #expect(
            !script.contains("A failed transfer's body is an R2/S3 error document"),
            "\(scriptFile)"
        )
    }
}


@Test
func editableSurfaceByteBudgetEnforcesCapsAtWorkerLaunch() throws {
    let fm = FileManager.default
    let root = try temporaryDirectory()
    defer { try? fm.removeItem(at: root) }
    let contract = root.appendingPathComponent("benchmark.json")
    let model = root.appendingPathComponent("Sources/MLXFastModel")
    try fm.createDirectory(at: model, withIntermediateDirectories: true)
    try #"{"editablePaths":["Sources/MLXFastModel","kernel.h"]}"#
        .write(to: contract, atomically: true, encoding: .utf8)
    try Data(repeating: 0x61, count: 100).write(to: model.appendingPathComponent("Engine.swift"))
    try Data(repeating: 0x62, count: 50).write(to: root.appendingPathComponent("kernel.h"))
    // A symlink never counts toward the budget.
    try fm.createSymbolicLink(
        atPath: model.appendingPathComponent("alias.swift").path,
        withDestinationPath: "Engine.swift"
    )

    func verdict(maxTotal: Int, maxFile: Int) -> EditableSurfaceBudgetVerification {
        verifyEditableSurfaceByteBudget(
            contractPath: contract.path,
            maxTotalBytes: maxTotal,
            maxFileBytes: maxFile
        )
    }

    #expect(verdict(maxTotal: 1000, maxFile: 500) == .verified(totalBytes: 150, fileCount: 2))

    // Per-file cap.
    guard case .exceeded(let perFileReason) = verdict(maxTotal: 1000, maxFile: 99) else {
        Issue.record("expected per-file cap to fire")
        return
    }
    #expect(perFileReason.contains("per-file static review limit"))

    // Total cap.
    guard case .exceeded(let totalReason) = verdict(maxTotal: 149, maxFile: 500) else {
        Issue.record("expected total cap to fire")
        return
    }
    #expect(totalReason.contains("static review limit"))

    // Missing contract skips (the caller decides whether that is fatal);
    // a malformed contract fails.
    guard case .skipped = verifyEditableSurfaceByteBudget(
        contractPath: root.appendingPathComponent("missing.json").path,
        maxTotalBytes: 1000,
        maxFileBytes: 500
    ) else {
        Issue.record("expected skipped without a contract")
        return
    }
    try #"{"schemaVersion":1}"#.write(to: contract, atomically: true, encoding: .utf8)
    guard case .exceeded(let malformedReason) = verdict(maxTotal: 1000, maxFile: 500) else {
        Issue.record("expected a malformed contract to fail")
        return
    }
    #expect(malformedReason.contains("no usable editablePaths"))
}

// Behavioral pin for run-semantic-gpqa-gate.sh's Opus 4.8 judge request and
// its parse hardening, via a curl shim serving canned Opus-shaped responses
// (thinking block first, verdict in the trailing text block). Covers the
// verdict shapes run 29124417146 tripped over: bare JSON, a ```json fence,
// JSON preceded/followed by prose, a verdict split across text blocks, and
// truncated garbage that must burn all retries and fail just that one case
// without changing gate semantics.
@Test
func semanticGPQAGateParsesOpusVerdictShapesAndFailsUnparseableCaseClosed() throws {
    let fm = FileManager.default
    let scriptPath = URL(fileURLWithPath: fm.currentDirectoryPath)
        .appendingPathComponent(".github/scripts/run-semantic-gpqa-gate.sh").path

    let root = fm.temporaryDirectory
        .appendingPathComponent("semantic-gate-\(UUID().uuidString)")
    try fm.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? fm.removeItem(at: root) }

    func run(_ argv: [String], env extra: [String: String]) throws -> (status: Int32, output: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = argv
        process.currentDirectoryURL = root
        var env = ProcessInfo.processInfo.environment
        let strayKeys = env.keys.filter {
            $0.hasPrefix("MLXFAST_") || $0.hasPrefix("ANTHROPIC_")
        }
        for key in strayKeys { env.removeValue(forKey: key) }
        env.merge(extra) { _, override in override }
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(data: data, encoding: .utf8) ?? "")
    }

    for tool in ["jq", "bash"] {
        guard try run(["sh", "-c", "command -v \(tool)"], env: [:]).status == 0 else { return }
    }

    let answersPath = root.appendingPathComponent("answers.json").path
    func answerCase(_ id: String) -> String {
        #"{"id":"\#(id)","prompt":"marker-\#(id) question","answer_key":"C","reference_answer":"option C","candidate_answer":"C"}"#
    }
    let answers = #"{"version":1,"cases":["# +
        [
            answerCase("case-bare"), answerCase("case-fenced"), answerCase("case-prose"),
            answerCase("case-garbage"), answerCase("case-split"),
        ].joined(separator: ",") + "]}"
    try answers.write(toFile: answersPath, atomically: true, encoding: .utf8)
    let scorePath = root.appendingPathComponent("score.json").path
    try #"{"passed":true,"score":1.0,"metrics":{"error":""}}"#
        .write(toFile: scorePath, atomically: true, encoding: .utf8)

    // Canned Opus-shaped responses: thinking first, verdict in text block(s).
    let shimDir = root.appendingPathComponent("bin").path
    try fm.createDirectory(atPath: shimDir, withIntermediateDirectories: true)
    let responses: [String: String] = [
        "bare": #"{"content":[{"type":"thinking","thinking":"weighing the options"},{"type":"text","text":"{\"passed\":true}"}],"stop_reason":"end_turn"}"#,
        "fenced": #"{"content":[{"type":"thinking","thinking":"comparing answers"},{"type":"text","text":"Here is the verdict:\n```json\n{\"passed\": true}\n```"}],"stop_reason":"end_turn"}"#,
        "prose": #"{"content":[{"type":"text","text":"The candidate names the wrong mechanism, so the verdict is {\"passed\": false}. Final."}],"stop_reason":"end_turn"}"#,
        "garbage": #"{"content":[{"type":"thinking","thinking":"long think"},{"type":"text","text":"The candidate answer begins by discussing the reaction order and"}],"stop_reason":"max_tokens"}"#,
        "split": #"{"content":[{"type":"thinking","thinking":"t"},{"type":"text","text":"Considering the equivalence carefully."},{"type":"text","text":"{\"passed\":true}"}],"stop_reason":"end_turn"}"#,
    ]
    for (name, body) in responses {
        try body.write(toFile: shimDir + "/response-\(name).json", atomically: true, encoding: .utf8)
    }
    let shim = """
    #!/usr/bin/env bash
    set -euo pipefail
    out=""
    data=""
    prev=""
    for arg in "$@"; do
      case "${prev}" in
        --output) out="${arg}" ;;
        --data) data="${arg}" ;;
      esac
      prev="${arg}"
    done
    req="${data#@}"
    cp "${req}" "${SHIM_DIR}/last-request.json"
    for name in bare fenced prose garbage split; do
      if grep -q "marker-case-${name}" "${req}"; then
        if [[ "${name}" == "garbage" ]]; then
          echo attempt >> "${SHIM_DIR}/garbage-attempts"
        fi
        cp "${SHIM_DIR}/response-${name}.json" "${out}"
        # The script reads the HTTP status from --write-out on stdout and
        # only hands a 200 body to the parse logic.
        printf '200'
        exit 0
      fi
    done
    echo "curl shim: unmatched request" >&2
    exit 1
    """
    try shim.write(toFile: shimDir + "/curl", atomically: true, encoding: .utf8)
    try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: shimDir + "/curl")

    let inheritedPath = ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin"
    let resultsPath = root.appendingPathComponent("results.json").path
    let gate = try run(
        ["bash", scriptPath],
        env: [
            "PATH": shimDir + ":" + inheritedPath,
            "SHIM_DIR": shimDir,
            "ANTHROPIC_API_KEY": "test-key-never-sent",
            "MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH": answersPath,
            "MLXFAST_SCORE_PATH": scorePath,
            "MLXFAST_INTEGRITY_PATH": root.appendingPathComponent("absent-integrity.json").path,
            "MLXFAST_SEMANTIC_GPQA_RESULTS_PATH": resultsPath,
            "MLXFAST_PRIVATE_DIR": root.appendingPathComponent("private").path,
            "MLXFAST_SEMANTIC_GPQA_MIN_PASS": "1",
            "MLXFAST_SEMANTIC_GPQA_REQUIRED": "1",
        ])
    #expect(gate.status == 0, gate.output.isEmpty ? "no output" : "\(gate.output)")

    // Every parseable verdict shape lands, prose-wrapped false stays false,
    // and only the garbage case fails -- after burning all three attempts.
    #expect(gate.output.contains("judging 5 hidden cases with claude-opus-4-8"))
    #expect(gate.output.contains("case 1/5 passed=true"))
    #expect(gate.output.contains("case 2/5 passed=true"))
    #expect(gate.output.contains("case 3/5 passed=false"))
    #expect(!gate.output.contains("case 3/5 passed=false reason=invalid_judge_response"))
    #expect(gate.output.contains("case 4/5 judge response was not parseable JSON (stop_reason=max_tokens); retrying"))
    #expect(gate.output.contains("case 4/5 passed=false reason=invalid_judge_response"))
    #expect(gate.output.contains("case 5/5 passed=true"))
    #expect(gate.output.contains("semantic-gpqa: passed 3/5"))
    let garbageAttempts = try String(contentsOfFile: shimDir + "/garbage-attempts", encoding: .utf8)
    #expect(garbageAttempts.components(separatedBy: "\n").filter { !$0.isEmpty }.count == 3)

    // The judge request is the documented Opus 4.8 shape: adaptive thinking
    // at max effort, an untruncatable max_tokens, and none of the fields
    // Opus 4.8 rejects with a 400 (sampling params, assistant prefill).
    let requestData = try Data(contentsOf: URL(fileURLWithPath: shimDir + "/last-request.json"))
    let request = try #require(
        try JSONSerialization.jsonObject(with: requestData) as? [String: Any])
    #expect(request["model"] as? String == "claude-opus-4-8")
    #expect(request["max_tokens"] as? Int == 32000)
    #expect((request["thinking"] as? [String: Any])?["type"] as? String == "adaptive")
    #expect((request["output_config"] as? [String: Any])?["effort"] as? String == "max")
    #expect(request["temperature"] == nil)
    #expect(request["top_p"] == nil)
    #expect(request["top_k"] == nil)
    let messages = try #require(request["messages"] as? [[String: Any]])
    #expect(messages.count == 1)
    #expect(messages[0]["role"] as? String == "user")
    let system = try #require(request["system"] as? String)
    #expect(system.contains("exactly one JSON object"))

    // score.json is patched with the aggregate verdict and the new model pin;
    // the gate stays green because pass_count 3 >= min_pass 1.
    let score = try String(contentsOfFile: scorePath, encoding: .utf8)
    #expect(score.contains("\"semantic_gpqa_passed\": true"))
    #expect(score.contains("\"semantic_gpqa_pass_count\": 3"))
    #expect(score.contains("\"semantic_gpqa_case_count\": 5"))
    #expect(score.contains("\"semantic_gpqa_model\": \"claude-opus-4-8\""))
    let results = try String(contentsOfFile: resultsPath, encoding: .utf8)
    #expect(results.contains("\"error\": \"invalid_judge_response\""))
}

// Behavioral pin for run-semantic-gpqa-gate.sh's judge-API transport
// hardening. Run 30169401200 (submission e2b91595) had passed the public
// gate and the full hidden correctness gates, and had already met min_pass
// (case 1 judged passed=true), when Anthropic returned a burst of HTTP 529s:
// curl exited 22 under `set -e`, the script died mid-judging, and an
// external outage surfaced as a submission failure. Pins, via a curl shim:
// (1) transport-class failures (curl exit codes, HTTP 429/5xx including
// 529) are retried with bounded backoff and Retry-After honored, then
// judging continues normally; (2) a real judged {"passed":false} below
// min_pass still fails the gate exactly as before -- transport handling
// must not mask a semantic rejection; (3) a persistent outage terminates as
// a labeled infra failure that redact-benchmark-failure.sh maps to
// semantic_gpqa_infra_failed with the submission's earned gate flags intact
// -- never as a semantic fail; (4) a deterministic 4xx is not retried; and
// (5) transport logging carries only fixed strings, attempt counts, and
// status codes -- never prompt, answer, or judge-response content.
@Test
func semanticGPQAGateRetriesTransportFailuresAndFailsInfraNotSemantic() throws {
    let fm = FileManager.default
    let gateScript = URL(fileURLWithPath: fm.currentDirectoryPath)
        .appendingPathComponent(".github/scripts/run-semantic-gpqa-gate.sh").path
    let redactScript = URL(fileURLWithPath: fm.currentDirectoryPath)
        .appendingPathComponent(".github/scripts/redact-benchmark-failure.sh").path

    let root = fm.temporaryDirectory
        .appendingPathComponent("semantic-gate-transport-\(UUID().uuidString)")
    try fm.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? fm.removeItem(at: root) }

    func run(_ argv: [String], env extra: [String: String]) throws -> (status: Int32, output: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = argv
        process.currentDirectoryURL = root
        var env = ProcessInfo.processInfo.environment
        let strayKeys = env.keys.filter {
            $0.hasPrefix("MLXFAST_") || $0.hasPrefix("ANTHROPIC_")
        }
        for key in strayKeys { env.removeValue(forKey: key) }
        env.merge(extra) { _, override in override }
        process.environment = env
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(data: data, encoding: .utf8) ?? "")
    }

    for tool in ["jq", "bash"] {
        guard try run(["sh", "-c", "command -v \(tool)"], env: [:]).status == 0 else { return }
    }

    // Curl shim: emits a canned HTTP status (via the script's --write-out
    // contract) and body per case marker, counting attempts per case so the
    // retry budget is observable. Exit 7 simulates a connection-level curl
    // failure; 529 responses carry an optional Retry-After header through
    // the script's --dump-header path.
    let shimDir = root.appendingPathComponent("bin").path
    try fm.createDirectory(atPath: shimDir, withIntermediateDirectories: true)
    let shim = """
    #!/usr/bin/env bash
    set -euo pipefail
    out=""
    data=""
    headers="/dev/null"
    prev=""
    for arg in "$@"; do
      case "${prev}" in
        --output) out="${arg}" ;;
        --data) data="${arg}" ;;
        --dump-header) headers="${arg}" ;;
      esac
      prev="${arg}"
    done
    req="${data#@}"
    pass_body='{"content":[{"type":"thinking","thinking":"t"},{"type":"text","text":"{\\"passed\\":true}"}],"stop_reason":"end_turn"}'
    fail_body='{"content":[{"type":"text","text":"{\\"passed\\":false}"}],"stop_reason":"end_turn"}'
    mode=""
    for name in transport-then-ok overloaded-then-ok always-529 real-fail bad-request plain-ok; do
      if grep -q "marker-case-${name}" "${req}"; then mode="${name}"; break; fi
    done
    if [[ -z "${mode}" ]]; then
      echo "curl shim: unmatched request" >&2
      exit 1
    fi
    count_file="${SHIM_COUNTS}/attempts-${mode}"
    count=$(( $(cat "${count_file}" 2>/dev/null || echo 0) + 1 ))
    printf '%s' "${count}" > "${count_file}"
    case "${mode}" in
      transport-then-ok)
        if [[ "${count}" -eq 1 ]]; then exit 7; fi
        printf '%s' "${pass_body}" > "${out}"
        printf '200' ;;
      overloaded-then-ok)
        if [[ "${count}" -eq 1 ]]; then
          printf 'HTTP/1.1 529 Overloaded\\r\\nRetry-After: 1\\r\\n\\r\\n' > "${headers}"
          printf '{"type":"error","error":{"type":"overloaded_error"}}' > "${out}"
          printf '529'
        else
          printf '%s' "${pass_body}" > "${out}"
          printf '200'
        fi ;;
      always-529)
        printf 'HTTP/1.1 529 Overloaded\\r\\n\\r\\n' > "${headers}"
        printf '{"type":"error","error":{"type":"overloaded_error"}}' > "${out}"
        printf '529' ;;
      real-fail)
        printf '%s' "${fail_body}" > "${out}"
        printf '200' ;;
      bad-request)
        printf '{"type":"error","error":{"type":"invalid_request_error"}}' > "${out}"
        printf '400' ;;
      plain-ok)
        printf '%s' "${pass_body}" > "${out}"
        printf '200' ;;
    esac
    """
    try shim.write(toFile: shimDir + "/curl", atomically: true, encoding: .utf8)
    try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: shimDir + "/curl")
    let inheritedPath = ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin"

    func answerCase(_ id: String) -> String {
        #"{"id":"\#(id)","prompt":"marker-case-\#(id) question","answer_key":"C","reference_answer":"option C","candidate_answer":"C"}"#
    }

    struct GateRun {
        let status: Int32
        let output: String
        let score: String
        let redacted: String
    }

    func attempts(_ scenario: String, _ mode: String) throws -> String {
        try String(
            contentsOfFile: root.appendingPathComponent(scenario)
                .appendingPathComponent("counts/attempts-\(mode)").path,
            encoding: .utf8)
    }

    func runGate(name: String, cases: [String], minPass: Int) throws -> GateRun {
        let dir = root.appendingPathComponent(name)
        let counts = dir.appendingPathComponent("counts").path
        try fm.createDirectory(atPath: counts, withIntermediateDirectories: true)
        let answersPath = dir.appendingPathComponent("answers.json").path
        let answers = #"{"version":1,"cases":["# +
            cases.map(answerCase).joined(separator: ",") + "]}"
        try answers.write(toFile: answersPath, atomically: true, encoding: .utf8)
        let scorePath = dir.appendingPathComponent("score.json").path
        // The score the correctness+gates phase leaves behind: every earned
        // gate flag true, timing not yet run.
        try #"{"passed":true,"score":null,"metrics":{"error":"","passed_correctness":true,"partial_result":true,"benchmark_wall_seconds":30}}"#
            .write(toFile: scorePath, atomically: true, encoding: .utf8)
        let gate = try run(
            ["bash", gateScript],
            env: [
                "PATH": shimDir + ":" + inheritedPath,
                "SHIM_COUNTS": counts,
                "ANTHROPIC_API_KEY": "test-key-never-sent",
                "MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH": answersPath,
                "MLXFAST_SCORE_PATH": scorePath,
                "MLXFAST_INTEGRITY_PATH": dir.appendingPathComponent("absent-integrity.json").path,
                "MLXFAST_SEMANTIC_GPQA_RESULTS_PATH": dir.appendingPathComponent("results.json").path,
                "MLXFAST_PRIVATE_DIR": dir.appendingPathComponent("private").path,
                "MLXFAST_SEMANTIC_GPQA_MIN_PASS": String(minPass),
                "MLXFAST_SEMANTIC_GPQA_REQUIRED": "1",
                "MLXFAST_SEMANTIC_GPQA_TRANSPORT_MAX_ATTEMPTS": "4",
                "MLXFAST_SEMANTIC_GPQA_TRANSPORT_BACKOFF_BASE_SECONDS": "0",
            ])
        // The redacted failure artifact is the operator-facing verdict for a
        // failed run; derive it from the score the gate left behind exactly
        // as the workflow's failure path does.
        let failurePath = dir.appendingPathComponent("benchmark-failure.json").path
        let redact = try run(
            ["bash", redactScript, failurePath, scorePath],
            env: ["MLXFAST_BENCHMARK_MODE": "single-machine"])
        #expect(redact.status == 0, redact.output.isEmpty ? "no output" : "\(redact.output)")
        return GateRun(
            status: gate.status,
            output: gate.output,
            score: try String(contentsOfFile: scorePath, encoding: .utf8),
            redacted: try String(contentsOfFile: failurePath, encoding: .utf8))
    }

    // (1) Recovery: a connection-level curl failure and a 529 with
    // Retry-After are retried (Retry-After: 1 wins over the 0s test
    // backoff) and both cases still judge to completion.
    let recovery = try runGate(
        name: "recovery", cases: ["transport-then-ok", "overloaded-then-ok"], minPass: 2)
    #expect(recovery.status == 0, recovery.output.isEmpty ? "no output" : "\(recovery.output)")
    #expect(recovery.output.contains(
        "case 1/2 judge call attempt 1/4 failed (curl transport error exit=7); retrying in 0s"))
    #expect(recovery.output.contains(
        "case 2/2 judge call attempt 1/4 failed (retryable HTTP status 529); retrying in 1s"))
    #expect(recovery.output.contains("case 1/2 passed=true"))
    #expect(recovery.output.contains("case 2/2 passed=true"))
    #expect(recovery.output.contains("semantic-gpqa: passed 2/2"))
    #expect(recovery.score.contains("\"semantic_gpqa_passed\": true"))
    #expect(try attempts("recovery", "transport-then-ok") == "2")
    #expect(try attempts("recovery", "overloaded-then-ok") == "2")

    // (2) A real judged rejection below min_pass fails the gate exactly as
    // before, on the first response, and is categorized as the judged
    // semantic failure it is -- transport handling must not mask it.
    let rejected = try runGate(name: "rejected", cases: ["real-fail"], minPass: 1)
    #expect(rejected.status != 0)
    #expect(rejected.output.contains("case 1/1 passed=false"))
    #expect(rejected.output.contains("semantic GPQA gate failed pass_count=0/1"))
    #expect(!rejected.output.contains("infra"))
    #expect(try attempts("rejected", "real-fail") == "1")
    #expect(rejected.score.contains("\"semantic_gpqa_passed\": false"))
    #expect(rejected.score.contains("\"passed\": false"))
    #expect(rejected.score.contains("semantic GPQA gate failed"))
    #expect(rejected.redacted.contains("\"failure_category\": \"semantic_gpqa_failed\""))

    // (3) Persistent outage in the exact run-30169401200 shape: min_pass
    // already met by case 1 when case 2's retry budget exhausts. The gate
    // fails as labeled, re-dispatchable infra with every earned gate flag
    // intact -- never as a semantic rejection of the submission.
    let outage = try runGate(name: "outage", cases: ["plain-ok", "always-529"], minPass: 1)
    #expect(outage.status != 0)
    #expect(outage.output.contains("case 1/2 passed=true"))
    #expect(outage.output.contains(
        "case 2/2 judge call attempt 4/4 failed (retryable HTTP status 529); retry budget exhausted"))
    #expect(outage.output.contains("semantic GPQA gate infra failure"))
    #expect(outage.output.contains("re-dispatch the run"))
    #expect(try attempts("outage", "always-529") == "4")
    #expect(outage.score.contains("\"passed\": true"))
    #expect(outage.score.contains("\"passed_correctness\": true"))
    #expect(outage.score.contains("semantic GPQA gate infra failure: judge API unavailable"))
    #expect(!outage.score.contains("semantic_gpqa_passed"))
    #expect(outage.redacted.contains("\"failure_category\": \"semantic_gpqa_infra_failed\""))
    #expect(outage.redacted.contains("\"passed_correctness\": true"))
    // Transport/infra logging never carries prompt, answer, or
    // judge-response content -- only fixed strings, attempt counts, and
    // status codes.
    #expect(!outage.output.contains("marker-case"))
    #expect(!outage.output.contains("option C"))
    #expect(!outage.output.contains("overloaded_error"))

    // (4) A deterministic request rejection (400) is not retried -- and is
    // still labeled infra, because no judge verdict exists.
    let badRequest = try runGate(name: "bad-request", cases: ["bad-request"], minPass: 1)
    #expect(badRequest.status != 0)
    #expect(badRequest.output.contains(
        "case 1/1 judge call attempt 1/4 non-retryable HTTP status 400"))
    #expect(badRequest.output.contains("semantic GPQA gate infra failure"))
    #expect(try attempts("bad-request", "bad-request") == "1")
    #expect(badRequest.redacted.contains("\"failure_category\": \"semantic_gpqa_infra_failed\""))
}

// Ranked validation otherwise exercises only the hidden goldens, so numerics
// drift on prompts they happen not to cover can be promoted into main and then
// break every participant's local public gate (issue #83). The ranked job must
// therefore also run the checked-in public fixture as an independent drift
// tripwire -- before the hidden golden ever enters the bench workspace, so
// honest drift fails fast and the worker running the public case cannot read
// hidden bytes -- with the fixture hash pinned to the actual checked-in file
// so a stale pin cannot silently validate a different oracle.
@Test
func rankedJobRunsPublicBehaviorGateBeforeHiddenGates() throws {
    let workflow = try String(
        contentsOfFile: ".github/workflows/dflash-benchmark.yml",
        encoding: .utf8
    )

    let publicGate = try #require(workflow.range(of: "- name: Public behavior gate"))
    let hiddenGolden = try #require(workflow.range(of: "- name: Prepare hidden correctness golden"))
    let hiddenGates = try #require(workflow.range(of: "- name: Correctness and gates"))
    #expect(publicGate.lowerBound < hiddenGolden.lowerBound)
    #expect(hiddenGolden.lowerBound < hiddenGates.lowerBound)

    let gateBody = String(workflow[publicGate.lowerBound..<hiddenGolden.lowerBound])
    #expect(gateBody.contains("--golden \"${MLXFAST_PUBLIC_GOLDEN}\""))
    #expect(gateBody.contains("public-gate-report.json"))
    // The worker Seatbelt profile denies the fixture under check by literal
    // path, matching the harness's own blockedGoldenPath convention.
    #expect(gateBody.contains("BENCH_GOLDEN_PATH: ${{ env.MLXFAST_JOB_WS }}/correctness_prompts/public_longcopy_gate_english_512_256.json"))
    // Submission branches must suppress the submitted process's own logs, same
    // as every other step that executes submitted model code.
    #expect(gateBody.contains("public-gate-private.log"))

    // The pinned hash must match the actual checked-in fixture, so regenerating
    // the fixture forces the workflow pin to move too.
    //
    // The public fixtures were regenerated with the Qwen 3.6 runtime, so the
    // workflow that mirrors the CURRENT bytes is the Qwen-MTP one; this was a
    // withKnownIssue only for as long as that workflow did not exist. It is now
    // a HARD assertion aimed at it, still derived from the checked-in bytes.
    let fixtureData = try Data(
        contentsOf: URL(fileURLWithPath: "correctness_prompts/public_longcopy_gate_english_512_256.json")
    )
    let fixtureHash = SHA256.hash(data: fixtureData).map { String(format: "%02x", $0) }.joined()
    let qwenWorkflow = try String(
        contentsOfFile: ".github/workflows/qwen-mtp-ranked-benchmark.yml",
        encoding: .utf8
    )
    #expect(
        qwenWorkflow.contains("MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_SHA256: \(fixtureHash)"),
        "the Qwen-MTP workflow does not pin the checked-in 256-step fixture digest \(fixtureHash)"
    )
    // Same for the byte count: the correctness-only validation hard-checks
    // bytes after the SHA, so a stale byte pin fails the correctness-only run
    // even when the hash matches. Deriving it from the fixture forces the pin to
    // move when the fixture is regenerated, instead of silently drifting.
    #expect(
        qwenWorkflow.contains(
            "MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_BYTES: \"\(fixtureData.count)\""
        )
    )
    // The 1024-step variant is the pre-submit fixture and the 256-step golden is
    // a strict greedy prefix of it. The Qwen workflow pins both so the pair
    // cannot drift apart; derive that pin from its own bytes too.
    let longFixtureData = try Data(
        contentsOf: URL(fileURLWithPath: "correctness_prompts/public_longcopy_gate_english_512_1024.json")
    )
    let longFixtureHash = SHA256.hash(data: longFixtureData)
        .map { String(format: "%02x", $0) }.joined()
    #expect(
        qwenWorkflow.contains(
            "MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_1024_SHA256: \(longFixtureHash)"
        )
    )
    #expect(
        qwenWorkflow.contains(
            "MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_1024_BYTES: \"\(longFixtureData.count)\""
        )
    )

    // DFLASH-SIDE COVERAGE, DELIBERATELY KEPT. This test's subject is the DFlash
    // job's public gate, and that job still pins the PRE-Qwen fixture -- a pin no
    // longer derivable from any checked-in file. Assert Laguna's literal rather
    // than dropping the check, so the live ranked track's public-gate pin cannot
    // be edited silently.
    #expect(
        workflow.contains(
            "MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_SHA256: b9509697c08a2cf3c2943a85f0b76e39c485c441794690fa76835b40a58d7a63"
        ),
        """
        the DFlash workflow's public-gate digest changed. It pins the pre-Qwen \
        fixture on purpose; if that track has been repointed at the regenerated \
        fixture, update this literal in the same commit.
        """
    )
    #expect(workflow.contains("MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_BYTES: \"10686\""))
    // The gate re-hashes the fixture at run time against the pinned value --
    // required in BOTH workflows, since that is what turns a stale pin into a
    // failed run rather than a silently different oracle.
    #expect(gateBody.contains("public correctness golden hash mismatch"))
    #expect(qwenWorkflow.contains("public correctness golden hash mismatch"))
}

// The 64-step teacher-forced base case only exercises single-token forwards at
// offsets 512..575, while the timed decode reaches 512..639. attach-free-run-gate
// lets the operator regenerate the private golden with a free-run case whose
// greedy continuation covers the full timed decode offset range with different
// prompt content, so an offset-gated cheap model path has to survive the
// unscored correctness gate too instead of only the LLM static review.
@Test
func cliSupportsFreeRunGateAttachmentCoveringTimedDecodeOffsets() throws {
    let cli = try String(
        contentsOfFile: "Sources/MLXFastCLI/main.swift",
        encoding: .utf8
    )

    #expect(cli.contains("case \"attach-free-run-gate\""))
    #expect(cli.contains("func runAttachFreeRunGate"))
    // Defaults to covering exactly the timed decode step count, capped at the
    // free-run maximum, and FAILS CLOSED when asked for less than full
    // coverage unless the operator explicitly opts in with --allow-partial.
    #expect(cli.contains("options.value(for: \"--steps\", default: \"\\(MLXFastConstants.benchmarkDecodeSteps)\")"))
    #expect(cli.contains("steps <= MLXFastConstants.correctnessMaxFreeRunSteps"))
    #expect(cli.contains("options.hasFlag(\"--allow-partial\")"))
    #expect(cli.contains("pass --allow-partial to write it anyway"))
    // Expected tokens come from actually running the reference model, with the
    // golden blocked from the worker like every other generation tool.
    #expect(cli.contains("runtimeWorkerOptions(blockedGoldenPath: goldenPath)"))
    // The INPUT golden is strict-validated before any generation or write --
    // --output defaults to the input path, so a malformed input must fail
    // before it could be replaced on disk.
    #expect(cli.contains("_ = try loadGoldenFixture(from: goldenPath)"))
    // The merged golden is staged to a temp sibling and re-validated through
    // the strict loader BEFORE it replaces the destination, so a failed
    // validation can never destroy the original golden.
    #expect(cli.contains("func writeValidatedGoldenDocument"))
    #expect(cli.contains("_ = try loadGoldenFixture(from: temporaryURL.path)"))
    #expect(!cli.contains("try outputData.write(to: outputURL, options: [.atomic])"))
    #expect(cli.contains("attach-free-run-gate ["))
}

// Both GPQA capture entry points must frame the hidden question as a ChatML
// turn with thinking pre-closed. Un-framed, the checkpoint (no bos_token, chat
// template shipped as a sidecar so addSpecialTokens adds nothing) received a
// bare question and never answered -- measured 0/9 commitments at a 512-token
// budget with degenerate looping, which is what scored the semantic gate 1/9.
// generate-gpqa-answers exists to reproduce the in-run capture offline, so the
// two framings must not drift apart.
@Test
func gpqaCapturePathsFrameHiddenPromptsAsChatMLTurns() throws {
    let cli = try String(
        contentsOfFile: "Sources/MLXFastCLI/main.swift",
        encoding: .utf8
    )

    // Both entry points wrap, and neither encodes the bare prompt any more.
    let wrapped = "QwenChatTemplate.userTurnDisablingThinking(testCase.prompt)"
    #expect(cli.components(separatedBy: wrapped).count - 1 == 2)
    #expect(!cli.contains("tokenizer.encode(text: testCase.prompt, addSpecialTokens: true)"))
    // The offline verb truncates the finished-turn tail before decoding.
    #expect(cli.contains("QwenChatTemplate.truncatedAtFirstEndOfTurn("))
    #expect(cli.contains("candidateTokens: answerTokens"))

    // The in-run capture truncates too -- in BOTH harness copies.
    for path in [
        "Sources/MLXFastTrustedHarness/QwenRuntimeGPQA.swift",
        "Sources/MLXFastHarness/QwenRuntimeGPQA.swift",
    ] {
        let harness = try String(contentsOfFile: path, encoding: .utf8)
        #expect(harness.contains("QwenChatTemplate.truncatedAtFirstEndOfTurn("))
        #expect(harness.contains("eosTokenId: tokenizer.eosTokenId"))
    }

    // The fixed-length behavior loop must NOT learn to stop early: its step
    // count is predicted statically from the golden and cross-checked against
    // the reported checked_steps.
    let compare = try String(
        contentsOfFile: "Sources/MLXFastTrustedHarness/QwenRuntimeCorrectnessCompare.swift",
        encoding: .utf8
    )
    #expect(compare.contains("while generated.count < testCase.maxNewTokens {"))
}

// QwenRuntime.benchmark refuses a golden with no `.benchmark` oracle even on
// the gates-only phase (CHECK_GATES=1 + SKIP_TIMED=1, where the oracle only
// supplies the baseline placeholders), but generate-golden writes
// `benchmark: nil` and both attach verbs pass the section through -- so
// without attach-benchmark-oracle no in-repo tool could author a hidden
// golden the ranked "Correctness and gates" step accepts.
@Test
func cliSupportsBenchmarkOracleAttachmentForRankedGoldens() throws {
    let cli = try String(
        contentsOfFile: "Sources/MLXFastCLI/main.swift",
        encoding: .utf8
    )

    #expect(cli.contains("case \"attach-benchmark-oracle\""))
    #expect(cli.contains("func runAttachBenchmarkOracle"))
    #expect(cli.contains("attach-benchmark-oracle ["))
    // The derivation itself lives in MLXFastCore so it is unit-testable and
    // so the CLI cannot drift from the precedent rule.
    #expect(cli.contains("goldenDocumentAttachingDerivedBenchmarkOracle(golden)"))
    // Pure file I/O: unlike the other attach verbs this one needs no weights,
    // no tokenizer and no runtime worker, because it only restates tokens the
    // golden already carries.
    #expect(cli.contains("try options.validate(valueOptions: [\"--golden\", \"--output\"])"))
    // The INPUT golden is strict-validated before any write -- --output
    // defaults to the input path, so a malformed input must fail before it
    // could be replaced on disk.
    #expect(cli.contains("_ = try loadGoldenFixture(from: goldenPath)"))
    // Same staged write-and-revalidate path as the other golden writers.
    #expect(cli.contains("func writeValidatedGoldenDocument"))

    let core = try String(
        contentsOfFile: "Sources/MLXFastCore/Golden.swift",
        encoding: .utf8
    )

    #expect(core.contains("func goldenDocumentAttachingDerivedBenchmarkOracle"))
    // No silent overwrite of an oracle that may have been measured.
    #expect(core.contains("already contains a benchmark oracle; refusing to overwrite it"))
    // Derived strictly from the golden's own base case.
    #expect(core.contains("prefillPromptTokens: baseCase.promptTokens"))
    #expect(core.contains("decodeSeedTokens: baseCase.promptTokens"))
    #expect(core.contains("expectedDecodeTokens: Array(baseCase.expectedTokens.dropFirst())"))
    // Validated before it can be written.
    #expect(core.contains("try validateBenchmarkGolden(oracle)"))
    // A hidden correctness golden is NOT a prompt-pool golden: it must not
    // carry per-prompt pool-rotation baselines.
    #expect(!core.contains("baselinePrefillSecondsPerToken: MLXFastConstants"))
}

// The public fixtures under correctness_prompts/ are BASE golden cases (the
// version-1 cases[] shape), not free-run gates, so the operator needs a
// generation path that writes that shape directly from a prompt text file.
// generate-golden mirrors attach-free-run-gate's prompt-file tokenization
// (weights-dir tokenizer, addSpecialTokens: false, first 512 tokens), runs
// the reference model through the same worker isolation, and only writes
// output that passes the strict fixture loader at the generated step count.
@Test
func cliSupportsPublicBaseGoldenGeneration() throws {
    let cli = try String(
        contentsOfFile: "Sources/MLXFastCLI/main.swift",
        encoding: .utf8
    )

    #expect(cli.contains("case \"generate-golden\""))
    #expect(cli.contains("func runGenerateGolden"))
    // Same prompt tokenization convention as attach-free-run-gate's
    // prompt-file path: weights-dir tokenizer, no special tokens, exactly the
    // required correctness prompt token count.
    #expect(cli.contains("tokenizer.encode(text: promptText, addSpecialTokens: false)"))
    #expect(cli.contains("Array(encoded.prefix(requiredPromptTokens))"))
    // A base case shorter than the correctness window would be rejected by
    // every consumer, so the command fails before generating anything.
    #expect(cli.contains("--steps must be >= correctnessSteps"))
    // Expected tokens come from actually running the reference model, with
    // the output fixture blocked from the worker like the other generators.
    #expect(cli.contains("runtimeWorkerOptions(blockedGoldenPath: outputPath)"))
    // The written fixture is staged through the validated-write helper and
    // re-validated at the full generated step count, so a fixture that would
    // fail its consumer's requiredSteps can never land on disk.
    #expect(cli.contains("generate-golden requires --output PATH"))
    #expect(cli.contains("generate-golden requires --name NAME"))
    #expect(cli.contains("generate-golden requires --prompt-file PATH"))
    #expect(cli.contains("requiredSteps: steps,"))
    #expect(cli.contains("generate-golden --prompt-file PATH"))
    #expect(cli.contains("modelProvenance: GoldenModelProvenance("))
    #expect(cli.contains("repository: MLXFastConstants.referenceModelRepository"))
    #expect(cli.contains("revision: MLXFastConstants.referenceModelRevision"))
}

@Test
func cliSupportsHiddenGPQAGateAttachment() throws {
    let cli = try String(
        contentsOfFile: "Sources/MLXFastCLI/main.swift",
        encoding: .utf8
    )
    let package = try String(
        contentsOfFile: "Package.swift",
        encoding: .utf8
    )
    let runtime = try harnessRuntimeSource()

    #expect(package.contains(".product(name: \"Tokenizers\", package: \"swift-transformers\")"))
    #expect(cli.contains("case \"attach-gpqa-gates\""))
    #expect(!cli.contains("case \"calibrate-gpqa-gates\""))
    #expect(!cli.contains("case \"calibrate-private-golden\""))
    #expect(!cli.contains("collectPrivateArtifactCalibration"))
    #expect(!cli.contains("PrivateArtifactCalibrator"))
    #expect(!cli.contains("PrivateArtifactWriter"))
    #expect(!cli.contains("requirePrivateIsolation"))
    #expect(!cli.contains("--gpqa-output"))
    #expect(cli.contains("case \"generate-gpqa-answers\""))
    #expect(!cli.contains("case \"measure-gpqa-ttft\""))
    #expect(cli.contains("AutoTokenizer.from(modelFolder: modelFolder, strict: false)"))
    #expect(cli.contains("acceptedReferenceTokenSequences"))
    #expect(cli.contains("QwenRuntime.generateGreedyTokens"))
    #expect(cli.contains("runtimeWorkerOptions(blockedGoldenPath: gpqaPath)"))
    #expect(!cli.contains("calibrated_reference_outputs"))
    #expect(cli.contains("SemanticGPQAAnswerDocument"))
    #expect(cli.contains("referenceAnswer(for: testCase)"))
    #expect(cli.contains("generate-gpqa-answers requires --output or MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH"))
    #expect(cli.contains("semantic GPQA answer output"))
    #expect(!cli.contains("measure-gpqa-ttft"))
    #expect(!cli.contains("FirstTokenTimingOptions(weightsPath: weightsPath, promptTokenSets: selectedPrompts)"))
    #expect(cli.contains("[\"\\(normalizedKey).\", \"\\(normalizedKey):\", \"\\(normalizedKey))\"]"))
    #expect(!cli.contains("[\"\\(normalizedKey).\", \"\\(normalizedKey):\", \"\\(normalizedKey))\", \"\\(normalizedKey)\"]"))
    #expect(!cli.contains("existingSequences + [generated]"))
    #expect(!cli.contains("accepted_sequences="))
    #expect(cli.contains("accepted_token_sequences or accepted_responses generated from the reference model"))
    #expect(cli.contains("sequence.prefix(maxNewTokens)"))
    #expect(cli.contains("uniqueSortedTokenSequences"))
    #expect(cli.contains("buildGPQABehaviorCaseIfWithinPromptBudget"))
    #expect(cli.contains("skippedOverBudgetGPQACases"))
    #expect(cli.contains("GPQA reference produced"))
    #expect(!cli.contains("gpqa.cases.prefix(caseCount)"))
    #expect(!cli.contains("answerKey ??"))
    #expect(cli.contains("MLXFastConstants.correctnessGPQACaseCount"))
    #expect(cli.contains("MLXFastConstants.correctnessGPQAMaxNewTokens"))
    #expect(!cli.contains("print(testCase.prompt)"))

    #expect(runtime.contains("compareBehaviorFirstToken"))
    #expect(runtime.contains("testCase.maxNewTokens == 1"))
    #expect(runtime.contains("correctnessTokenAccepted("))
    #expect(runtime.contains("kind: \"correctness_begin\""))
    #expect(runtime.contains("kind: \"correctness_step\""))
    #expect(runtime.contains("worker.teacherForcedCorrectnessStep(previousToken: testCase.expectedTokens[step - 1])"))
    #expect(!runtime.contains("correctness_teacher_forced_batch"))
    #expect(!runtime.contains("teacherForcedCorrectnessBatch"))
    #expect(!runtime.contains("let expectedTokens: [Int]?"))

    let workerTeacherForcedStart = try #require(runtime.range(of: "static func compareTeacherForcedWithWorker"))
    let workerAnchorStart = try #require(runtime.range(of: "static func compareAnchorWithWorker"))
    let workerBehaviorStart = try #require(runtime.range(of: "static func compareBehaviorWithWorker"))
    let workerValidationStart = try #require(runtime.range(of: "static func validatedWorkerTopLogits"))
    let workerTeacherForced = String(runtime[workerTeacherForcedStart.lowerBound..<workerAnchorStart.lowerBound])
    let workerAnchor = String(runtime[workerAnchorStart.lowerBound..<workerBehaviorStart.lowerBound])
    let workerBehavior = String(runtime[workerBehaviorStart.lowerBound..<workerValidationStart.lowerBound])
    // Worker/non-worker parity: every worker comparison applies the same
    // acceptance rules as its cached counterpart (top-logit tie tolerance,
    // anchor rank/delta fields), using worker-reported top logits only after
    // validatedWorkerTopLogits vets them -- malformed logits degrade to the
    // strict comparison (try?), never to a wider acceptance. A golden
    // generated on one Apple Silicon generation must not hard-fail another
    // over a true near-tie (issue #83).
    #expect(workerTeacherForced.contains("actualToken != expectedToken"))
    #expect(workerTeacherForced.contains("correctnessTokenAccepted("))
    #expect(workerTeacherForced.contains("try? validatedWorkerTopLogits("))
    #expect(!workerAnchor.contains("topLogits: nil"))
    #expect(workerAnchor.contains("try? validatedWorkerTopLogits(response.topLogits, actualToken: actualToken)"))
    #expect(!workerBehavior.contains("topLogits: nil"))
    #expect(workerBehavior.contains("try? validatedWorkerTopLogits(beginResponse.topLogits, actualToken: firstToken)"))
    #expect(workerBehavior.contains("let usesSemanticJudge = behaviorUsesSemanticJudge(testCase)"))
    #expect(workerBehavior.contains("if !usesSemanticJudge"))
    #expect(workerBehavior.contains("if usesSemanticJudge ||"))
}

@Test
func unsafePrivateCalibrationToolingStaysExcluded() {
    for path in [
        "Sources/MLXFastCore/PrivateArtifactCalibration.swift",
        "Sources/MLXFastTrustedHarness/QwenRuntimePrivateCalibration.swift",
        "Sources/MLXFastTrustedHarness/LagunaRuntimePrivateCalibration.swift",
        "Tests/Fixtures/PrivateArtifactCalibration/synthetic-private-golden-template.json",
        "Tests/Fixtures/PrivateArtifactCalibration/synthetic-private-gpqa-template.json",
        "Tests/MLXFastTests/PrivateArtifactCalibrationTests.swift",
    ] {
        #expect(
            !FileManager.default.fileExists(atPath: path),
            "unsafe participant-worker private calibration tooling must remain excluded: \(path)"
        )
    }
}

@Test
func offlineRunnerProvesNetworkIsBlockedBeforeRunningCommand() throws {
    let runner = try String(
        contentsOfFile: ".github/scripts/run-offline.sh",
        encoding: .utf8
    )

    #expect(runner.contains("(deny network*)"))
    #expect(runner.contains("pwd -P"))
    #expect(runner.contains("cd -P"))
    #expect(runner.contains("(deny process-fork)"))
    #expect(runner.contains("(deny process-exec*)"))
    #expect(runner.contains("(allow process-exec (literal"))
    #expect(runner.contains("if [[ \"${executable}\" == \"${workspace_root}/\"* ]]; then"))
    #expect(runner.contains("(allow process-exec (subpath"))
    #expect(runner.contains("(deny file-write*)"))
    #expect(runner.contains("MLXFAST_OFFLINE_WRITABLE_PATHS"))
    #expect(runner.contains("write_allowed_writes"))
    #expect(runner.contains("-fsS --max-time 10 https://example.com"))
    #expect(runner.contains("sandbox profile did not block network access; refusing to run"))
    #expect(runner.contains("network egress and child process execution are blocked"))
    #expect(runner.contains("export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"))
    #expect(runner.contains("export HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9"))
    #expect(runner.contains("trap cleanup_profiles EXIT"))
    #expect(runner.contains("rm -f \"${curl_profile}\""))
    #expect(runner.contains("rm -f \"${command_profile}\""))
}


// Before any participant worker is spawned, the trusted CLI re-verifies the
// metallib the worker will load against the vendored Metal sources (the
// published mlx.metallib.fingerprint sidecar): official runs fail closed on a
// stale, missing, or unverifiable metallib so a cached artifact can never
// mask kernel edits; local runs warn.
@Test
func cliVerifiesMetallibFingerprintBeforeWorkerLaunch() throws {
    let cli = try String(contentsOfFile: "Sources/MLXFastCLI/main.swift", encoding: .utf8)
    #expect(!cli.contains("TODO(security): Fingerprint the metallib"))
    #expect(!cli.contains("TODO(security): Separate trusted and participant build caches"))
    #expect(cli.contains("try enforceMetallibFingerprint("))
    #expect(cli.contains("VendoredMetalFingerprint.defaultCmlxRelativePath"))
    #expect(cli.contains("verifyMetallibFingerprintRecord("))
    #expect(cli.contains(
        "official benchmark runs require the metallib fingerprint check"
    ))
    #expect(cli.contains("refusing to spawn the participant worker: "))
    #expect(cli.contains("mlxfast-swift: warning: "))

    let fingerprintSource = try String(
        contentsOfFile: "Sources/MLXFastTrustedHarness/VendoredMetalFingerprint.swift",
        encoding: .utf8
    )
    #expect(fingerprintSource.contains("recordPrefix = \"mlxfast-metallib-fingerprint-v1\""))
    #expect(fingerprintSource.contains("fingerprintedSubtrees = [\"mlx\", \"mlx-generated\"]"))
    #expect(fingerprintSource.contains("defaultCmlxRelativePath = \"Vendor/mlx-swift/Source/Cmlx\""))
    // Byte-order path sort and shasum-format lines: the exact contract shared
    // with tools/build-mlx-metallib.sh's compute_vendored_metal_fingerprint.
    #expect(fingerprintSource.contains("lhs.utf8.lexicographicallyPrecedes(rhs.utf8)"))
    #expect(fingerprintSource.contains("\\(hexEncoded(digest))  \\(relativePath)\\n"))
}

@Test
func benchmarkScriptHidesPrivateDirectoryFromRuntimeWorker() throws {
    let benchmark = try String(
        contentsOfFile: "benchmark.sh",
        encoding: .utf8
    )
    let runtime = try harnessRuntimeSource()
    let cli = try String(
        contentsOfFile: "Sources/MLXFastCLI/main.swift",
        encoding: .utf8
    )

    #expect(benchmark.contains("MLXFAST_PRIVATE_DIR"))
    #expect(benchmark.contains("pwd -P"))
    #expect(benchmark.contains("cd -P"))
    #expect(benchmark.contains("RUNTIME_WORKER_BIN=\"${MLXFAST_RUNTIME_WORKER_EXECUTABLE:-.build-worker/release/mlxfast-runtime-worker}\""))
    #expect(benchmark.contains("MLXFAST_RUNTIME_WORKER_EXECUTABLE=\"$(absolute_path \"${RUNTIME_WORKER_BIN}\")\""))
    #expect(benchmark.contains("export MLXFAST_RUNTIME_WORKER_EXECUTABLE"))
    #expect(benchmark.contains("export MLXFAST_REFERENCE_DIR=\"${REFERENCE_PATH}\""))
    #expect(benchmark.contains("(deny file-read* (subpath"))
    #expect(benchmark.contains("(deny file-write* (subpath"))
    #expect(benchmark.contains("(deny process-fork)"))
    #expect(benchmark.contains("(deny process-exec*)"))
    #expect(benchmark.contains("(deny file-write*)"))
    #expect(benchmark.contains("(allow file-write* (literal \"/dev/null\"))"))
    #expect(benchmark.contains("(allow process-exec (literal"))
    #expect(benchmark.contains("worker_absolute=\"$(absolute_path \"${RUNTIME_WORKER_BIN}\")\""))
    #expect(!benchmark.contains("swift_absolute=\"$(absolute_path \"${SWIFT_BIN}\")\""))
    #expect(!benchmark.contains("(allow network* (remote ip \"localhost:*\"))"))
    #expect(!benchmark.contains("(allow network* (local unix-socket))"))
    // Private-material locations and secrets must not reach the worker's
    // environment. The filter is a strict allowlist, so assert the property
    // against the real function instead of pinning denylist source text.
    let workerEnvironment = sanitizedRuntimeWorkerEnvironment([
        "MLXFAST_PRIVATE_DIR": "/private/golden",
        "ANTHROPIC_API_KEY": "secret",
        "MLXFAST_GPQA_REFERENCE_PATH": "/private/golden/gpqa.json",
        "MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH": "/ws/private/semantic_gpqa_answers.json",
        "MLXFAST_SEMANTIC_GPQA_RESULTS_PATH": "/private/semantic_gpqa_results.json",
    ])
    #expect(workerEnvironment["MLXFAST_PRIVATE_DIR"] == nil)
    #expect(workerEnvironment["ANTHROPIC_API_KEY"] == nil)
    #expect(workerEnvironment["MLXFAST_GPQA_REFERENCE_PATH"] == nil)
    #expect(workerEnvironment["MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH"] == nil)
    #expect(workerEnvironment["MLXFAST_SEMANTIC_GPQA_RESULTS_PATH"] == nil)
    #expect(runtime.contains("gpqaTTFT: correctnessResult.gpqaTTFT"))
    #expect(runtime.contains("gpqaTTFTSource: gpqaTTFT.source"))
    #expect(cli.contains("resolvingSymlinksInPath()"))
    #expect(cli.contains("(deny process-fork)"))
    #expect(cli.contains("(deny process-exec*)"))
    #expect(cli.contains("(deny file-write*)"))
    #expect(cli.contains("(allow file-write* (literal \"/dev/null\"))"))
    #expect(cli.contains("(deny file-read* (subpath"))
    #expect(cli.contains("absolutePath(privateDir)"))
    #expect(!cli.contains("(allow network* (remote ip \\\"localhost:*\\\"))"))
}

@Test
func benchmarkScriptAvoidsNestedSandboxWithRuntimeWorker() throws {
    let benchmark = try String(
        contentsOfFile: "benchmark.sh",
        encoding: .utf8
    )

    #expect(benchmark.contains("supported macOS runners reject nested"))
    #expect(benchmark.contains("if [[ \"${USE_RUNTIME_WORKER}\" != \"1\" && \"${MLXFAST_IN_SANDBOX:-0}\" != \"1\" && \"${MLXFAST_NO_SANDBOX:-0}\" != \"1\" ]]; then"))
    #expect(benchmark.contains("run_offline_writable_command \"${TRANSFORM_STAGING_PARENT_OWNED}\""))
    #expect(benchmark.contains("--output \"${staged_weights}\""))
    #expect(benchmark.contains("run_offline_writable_command \"$(absolute_path \"${VERIFY_TRANSFORM_TMP_PARENT_OWNED}\")\""))
    #expect(benchmark.contains("--tmp-parent \"${VERIFY_TRANSFORM_TMP_PARENT_OWNED}\""))
}

@Test
func overlayScriptRejectsDangerousArtifactsAfterCopy() throws {
    let overlay = try String(
        contentsOfFile: ".github/scripts/overlay-editable-paths.sh",
        encoding: .utf8
    )

    #expect(overlay.contains("validate_overlay_tree \"${target_path}\""))
    #expect(overlay.contains("submitted editable path is missing"))
    #expect(overlay.contains("jq -r '.editablePaths[]'"))
    #expect(overlay.contains("overlaid editable paths must not contain symlinks"))
    #expect(overlay.contains("overlaid editable paths must contain only regular files and directories"))
    #expect(overlay.contains("-links +1"))
    #expect(overlay.contains("setuid or setgid"))
}


@Test
func runtimeWorkerBenchmarkDecodeDoesNotReceiveBulkOracle() throws {
    let runtime = try harnessRuntimeSource()

    #expect(runtime.contains("kind: \"decode_begin\""))
    #expect(runtime.contains("kind: \"decode_step\""))
    #expect(runtime.contains("worker.decodeStep(inputToken: inputToken)"))
    #expect(!runtime.contains("expected_seed_token"))
    #expect(!runtime.contains("case decodeSteps = \"decode_steps\""))
    #expect(!runtime.contains("validation_delay_ms"))
    #expect(!runtime.contains("case secondsPerToken = \"seconds_per_token\""))
    #expect(!runtime.contains("case bandwidthGBPerToken = \"bandwidth_gb_per_token\""))
    #expect(!runtime.contains("message += \": expected"))
    #expect(!runtime.contains("expectedToken: mismatch.expectedToken"))
    #expect(!runtime.contains("actualToken: mismatch.actualToken"))
}

@Test
func benchmarkKeepsZeroExpertStreamingSchemaWithoutStreamingMachinery() throws {
    let fileManager = FileManager.default
    // The dense, RAM-resident runtime has no expert-cache/streaming machinery
    // to audit; only the minimal all-zero stats struct remains (kept so the
    // score schema's expert_* fields stay stable across the model migration).
    #expect(!fileManager.fileExists(atPath: "Sources/MLXFastCore/ExpertSlotBank.swift"))
    #expect(!fileManager.fileExists(atPath: "Sources/MLXFastCore/ExpertStreaming.swift"))
    #expect(fileManager.fileExists(atPath: "Sources/MLXFastCore/ExpertStreamingStats.swift"))

    let contract = try String(contentsOfFile: "benchmark.json", encoding: .utf8)
    #expect(!contract.contains("Sources/MLXFastCore"))

    let runtime = try harnessRuntimeSource()
    #expect(runtime.contains("static let bandwidthSource = \"ram_resident_model\""))
    #expect(!runtime.contains("requirePlausibleSeedForwardExpertReads"))
}

@Test
func benchmarkTimingChargesDecodeSetupAndSeparatesWorkers() throws {
    let runtime = try harnessRuntimeSource()
    let workerStart = try #require(runtime.range(of: "static func benchmarkWithWorker"))
    let workerRuntime = String(runtime[workerStart.lowerBound...])

    #expect(workerRuntime.contains("benchmark prefill worker start"))
    #expect(workerRuntime.contains("benchmark decode worker start"))
    #expect(workerRuntime.contains("reported prefill duration as the score source"))
    #expect(workerRuntime.contains("let prefillStart = DispatchTime.now().uptimeNanoseconds"))
    #expect(workerRuntime.contains("let elapsed = secondsSince(prefillStart)"))
    #expect(workerRuntime.contains("runtime worker prefill response missing token"))
    #expect(!workerRuntime.contains("runtime worker prefill response missing token or seconds"))
    #expect(workerRuntime.contains("worker-reported per-step timing"))
    #expect(workerRuntime.contains("let decodePhaseStart = DispatchTime.now().uptimeNanoseconds"))
    #expect(workerRuntime.contains("includes_seed_prefill=true"))
    #expect(workerRuntime.contains("let measuredSeconds = secondsSince(decodePhaseStart)"))
    #expect(workerRuntime.contains("Hidden GPQA TTFT is a timing gate"))
    #expect(workerRuntime.contains("let ttftStart = DispatchTime.now().uptimeNanoseconds"))
    #expect(workerRuntime.contains("let ttftSeconds = secondsSince(ttftStart)"))
    #expect(!workerRuntime.contains("ttftSeconds: beginResponse.seconds"))
    // benchmarkWithWorker's preflight must not call BenchmarkPreflight.check --
    // that helper loaded config/dense-store/weight-loader
    // (editable MLXFastModel code) in this trusted, unsandboxed parent process.
    // checkWorkerBenchmarkInputs is model-free: it only checks that required
    // artifact paths exist, and lets the sandboxed worker itself validate
    // config/dense/expert metadata when it starts.
    #expect(!workerRuntime.contains("_ = try BenchmarkPreflight.check("))
    #expect(workerRuntime.contains("try checkWorkerBenchmarkInputs("))

    let preflightRange = try #require(workerRuntime.range(of: "progress(\"preflight start\")"))
    let timedBenchmarkRange = try #require(workerRuntime.range(of: "progress(\"timed benchmark start\")"))
    let weightsDigestRange = try #require(workerRuntime.range(of: "progress(\"weights digest start\")"))
    let correctnessRange = try #require(workerRuntime.range(of: "progress(\"correctness start cases=\\(golden.totalCorrectnessCaseCount)\")"))
    #expect(preflightRange.lowerBound < weightsDigestRange.lowerBound)
    #expect(weightsDigestRange.lowerBound < timedBenchmarkRange.lowerBound)
    #expect(timedBenchmarkRange.lowerBound < correctnessRange.lowerBound)
}

@Test
func compareTeacherForcedWithWorkerUsesSerialTeacherForcedSteps() throws {
    let runtime = try harnessRuntimeSource()

    #expect(runtime.contains("static func compareTeacherForcedWithWorker("))
    #expect(runtime.contains("try worker.beginTeacherForcedCorrectness(promptTokens: testCase.promptTokens)"))
    #expect(runtime.contains("for step in 0..<steps {"))
    #expect(runtime.contains("teacherForcedCorrectnessStep(previousToken: testCase.expectedTokens[step - 1])"))
    #expect(!runtime.contains("startStep: Int = 0,"))
    #expect(!runtime.contains("for seedStep in 0..<startStep {"))
    #expect(!runtime.contains("testCase.promptTokens + Array(testCase.expectedTokens[0..<startStep])"))
}

@Test
func decodeMeasurementRunsSingleUnmemoizableSeedForward() throws {
    // The decode measurement must run exactly ONE whole-prompt (seed) forward in
    // the timed window. A second identical forward (the warmup this used to run
    // before the seed) let submitted model code memoize one pass and serve the
    // other from that memo, collapsing two decode-charged forwards into one and
    // inflating decode_speedup with no real speedup. Guard both decode paths.
    let worker = try String(
        contentsOfFile: "Sources/MLXFastHarness/QwenRuntimeWorker.swift",
        encoding: .utf8
    )
    let beginStart = try #require(worker.range(of: "case \"decode_begin\":"))
    let beginEnd = try #require(worker.range(of: "case \"decode_step\":"))
    let decodeBegin = String(worker[beginStart.lowerBound..<beginEnd.lowerBound])
    // Exactly one whole-prompt forward, and no warmup pass preceding the seed.
    #expect(!decodeBegin.contains("warmupCache"))
    #expect(!decodeBegin.contains("warmupLogits"))
    #expect(decodeBegin.components(separatedBy: "qwenLogits(").count - 1 == 1)
    #expect(decodeBegin.contains("with NO preceding"))

    let benchmark = try String(
        contentsOfFile: "Sources/MLXFastHarness/QwenRuntimeBenchmark.swift",
        encoding: .utf8
    )
    let decodeStart = try #require(benchmark.range(of: "static func measureDecode("))
    let decodeEnd = try #require(benchmark.range(of: "static func measureWorkerDecode("))
    let measureDecode = String(benchmark[decodeStart.lowerBound..<decodeEnd.lowerBound])
    // No warmup pass before the seed prefill. (Unlike decode_begin, this path
    // inlines the per-step decode loop, which has its own single-token
    // logits call, so we assert on the absence of the warmup rather than a
    // whole-prompt forward count.)
    #expect(!measureDecode.contains("warmupCache"))
    #expect(!measureDecode.contains("decode warmup start"))
}

// steps == 0 skips the base teacher-forced case entirely (still runs anchors/
// free-run/behavior/GPQA/TTFT) for a run whose base case is verified in a
// separate phase of the ranked pipeline. Skipping is a caller decision; the
// harness must never treat a steps=0 run as having verified correctness on
// its own -- only the trusted pipeline that assembles the final score may.
@Test
func benchmarkCorrectnessSupportsZeroStepBaseCaseSkipWithoutSelfCertifying() throws {
    let runtime = try harnessRuntimeSource()

    #expect(runtime.contains("progress?(\"correctness case \\(caseLabel) skipped (steps=0)\")"))
    #expect(runtime.contains("guard steps > 0 else {"))

    // 0 is an explicit, documented allowance for BenchmarkOptions.correctnessSteps.
    #expect(runtime.contains("guard options.correctnessSteps >= 0 else {"))
    #expect(!runtime.contains("guard options.correctnessSteps > 0 else {"))
}

// checkGates: false must skip all three gate loops (anchors/free-run/behavior/
// GPQA) and report caseCount as golden.cases.count alone -- the timing-only
// role that runs against a gate-free oracle golden (see
// BenchmarkOptions.checkGates).
@Test
func correctnessCheckGatesFalseSkipsGateLoopsAndReportsBaseCaseCountAlone() throws {
    let runtime = try harnessRuntimeSource()

    #expect(runtime.contains("checkGates: Bool = true,"))
    #expect(runtime.contains("let caseCount = checkGates ? golden.totalCorrectnessCaseCount : golden.cases.count"))
    #expect(runtime.contains("let gates = checkGates ? golden.correctnessGates : nil"))
}

// Regression test for a bug caught only by a real dispatch (not reproducible
// locally without a live worker + real weights, hence a source-text check
// rather than a behavioral one -- see BenchmarkOptions.checkGates/
// skipTimedBenchmark for the harness-level design these fields support):
// a gates-only run (checkGates: false, so the behavior-gate loop that
// captures semantic GPQA answers never runs) still built a non-nil
// SemanticGPQACaptureOptions whenever MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH was
// set, and then unconditionally required semanticAnswers.count to equal
// caseCount -- turning a correct, nothing-to-capture gates-only run into a
// hard failure ("captured 0 semantic GPQA answers; expected 5").
@Test
func benchmarkGatesOnlyRunSkipsSpuriousSemanticCaptureFailure() throws {
    let runtime = try harnessRuntimeSource()

    #expect(runtime.contains("public let checkGates: Bool"))
    #expect(runtime.contains("public let skipTimedBenchmark: Bool"))
    #expect(runtime.contains("checkGates: Bool = true,"))
    #expect(runtime.contains("skipTimedBenchmark: Bool = false"))
    #expect(runtime.contains("guard options.checkGates || !options.skipTimedBenchmark else {"))

    // The fix: the semantic-capture count guard must not fire when checkGates
    // is false, since nothing was ever captured to check.
    #expect(runtime.contains("if checkGates, let semanticCapture {"))
    #expect(!runtime.contains("if let semanticCapture {\n                guard semanticAnswers.count == semanticCapture.caseCount else {"))

    // Placeholder timing values for the gates-only phase must be the resolved
    // baseline exactly (speedup == 1.0, always finite) -- 0 would divide-by-
    // zero into +Infinity in BenchmarkScore.speedup, and Double.infinity fails
    // JSON encoding outright. "Resolved" means the golden oracle's per-prompt
    // baseline when it carries one, else the calibrated constants.
    #expect(runtime.contains("prefillSecondsPerToken = baselinePrefillSecondsPerToken"))
    #expect(runtime.contains("secondsPerToken: baselineDecodeSecondsPerToken,"))

    let cli = try String(contentsOfFile: "Sources/MLXFastCLI/main.swift", encoding: .utf8)
    #expect(cli.contains("MLXFAST_BENCHMARK_CHECK_GATES"))
    #expect(cli.contains("MLXFAST_BENCHMARK_SKIP_TIMED"))
    #expect(cli.contains("checkGates: checkGates,"))
    #expect(cli.contains("skipTimedBenchmark: skipTimedBenchmark"))
}

@Test
func benchmarkCliSupportsSkippableBenchmarkCorrectness() throws {
    let cli = try String(contentsOfFile: "Sources/MLXFastCLI/main.swift", encoding: .utf8)

    #expect(cli.contains("MLXFAST_BENCHMARK_CORRECTNESS_STEPS"))
    #expect(cli.contains("private static func parseNonNegativeInt(_ rawValue: String, optionName: String) throws -> Int {"))
    #expect(cli.contains("correctnessSteps: correctnessSteps,"))
}


@Test
func hashWeightsDirectoryIsIndependentOfWeightsPathButSensitiveToContent() throws {
    let hashScript = try String(
        contentsOfFile: ".github/scripts/hash-weights-directory.sh",
        encoding: .utf8
    )
    #expect(hashScript.contains("shasum -a 256"))
    #expect(hashScript.contains("LC_ALL=C sort -z"))

    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    func makeWeights(named name: String, content: String, filename: String = "config.json") throws -> URL {
        let dir = root.appendingPathComponent(name).appendingPathComponent("weights")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try content.write(to: dir.appendingPathComponent(filename), atomically: true, encoding: .utf8)
        return dir
    }

    func hash(_ weightsDir: URL) throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [".github/scripts/hash-weights-directory.sh", weightsDir.path]
        process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let pipe = Pipe()
        process.standardOutput = pipe
        try process.run()
        process.waitUntilExit()
        #expect(process.terminationStatus == 0)
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    let rootA = try makeWeights(named: "rootA-\(UUID().uuidString)", content: "identical content")
    let rootB = try makeWeights(named: "rootB-\(UUID().uuidString)", content: "identical content")
    let hashA = try hash(rootA)
    let hashB = try hash(rootB)
    #expect(!hashA.isEmpty)
    #expect(hashA == hashB)
    var expectedTreeBytes = Data("config.json".utf8)
    expectedTreeBytes.append(0)
    expectedTreeBytes.append(contentsOf: SHA256.hash(data: Data("identical content".utf8)))
    expectedTreeBytes.append(0)
    let expectedTreeHash = SHA256.hash(data: expectedTreeBytes)
        .map { String(format: "%02x", $0) }
        .joined()
    #expect(hashA == expectedTreeHash)

    // Exercise the complete framing contract, including lexical path order,
    // with files created in deliberately different and nested order.
    let multi = root.appendingPathComponent("multi-\(UUID().uuidString)/weights")
    try FileManager.default.createDirectory(
        at: multi.appendingPathComponent("nested"),
        withIntermediateDirectories: true
    )
    let multiFiles: [(path: String, contents: String)] = [
        ("z-last.bin", "z"),
        ("nested/a-first.bin", "a"),
        ("middle.txt", "m"),
    ]
    for file in multiFiles {
        try file.contents.write(
            to: multi.appendingPathComponent(file.path),
            atomically: true,
            encoding: .utf8
        )
    }
    var expectedMultiTreeBytes = Data()
    for file in multiFiles.sorted(by: { $0.path < $1.path }) {
        expectedMultiTreeBytes.append(Data(file.path.utf8))
        expectedMultiTreeBytes.append(0)
        expectedMultiTreeBytes.append(
            contentsOf: SHA256.hash(data: Data(file.contents.utf8))
        )
        expectedMultiTreeBytes.append(0)
    }
    let expectedMultiTreeHash = SHA256.hash(data: expectedMultiTreeBytes)
        .map { String(format: "%02x", $0) }
        .joined()
    #expect(try hash(multi) == expectedMultiTreeHash)

    try "ignored".write(
        to: rootA.appendingPathComponent(".benchmark-source.sha256"),
        atomically: true,
        encoding: .utf8
    )
    #expect(try hash(rootA) == hashA)
    let nested = rootA.appendingPathComponent("nested")
    try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
    try "included".write(
        to: nested.appendingPathComponent(".gitkeep"),
        atomically: true,
        encoding: .utf8
    )
    #expect(try hash(rootA) != hashA)

    let differentContent = try makeWeights(named: "different-\(UUID().uuidString)", content: "not the same")
    #expect(try hash(differentContent) != hashA)

    let renamedFile = try makeWeights(named: "renamed-\(UUID().uuidString)", content: "identical content", filename: "renamed.json")
    #expect(try hash(renamedFile) != hashA)
}

// Published score payloads must apply the same diagnostic coarsening whether
// they are written to disk or emitted on stdout.
@Test
func sealedStdoutScoreIsCoarsenedLikeTheWrittenFile() throws {
    // benchmark.sh rebuilds score.json from emitScorePayloadToStdout's output, so
    // that emit -- not the discarded writeScorePayload file -- is the published
    // per-machine artifact. It must apply the same diagnostic coarsening, or the
    // covert-channel mitigation is bypassed on the sealed path.
    let cli = try String(contentsOfFile: "Sources/MLXFastCLI/main.swift", encoding: .utf8)
    let start = try #require(cli.range(of: "private static func emitScorePayloadToStdout"))
    let body = String(cli[start.lowerBound...].prefix(700))
    #expect(body.contains("withCoarsenedPublicDiagnostics()"))
}

@Test
func runtimeWorkerProtocolUsesAuthenticatedPrivateIO() throws {
    let runtime = try String(
        contentsOfFile: "Sources/MLXFastHarness/QwenRuntimeWorker.swift",
        encoding: .utf8
    )

    #expect(runtime.contains("hello = try readResponseLine(validateNonce: false)"))
    #expect(runtime.contains("self.sessionNonce = nonce"))
    #expect(!runtime.contains("RuntimeWorkerRequest(\n            id: id,\n            nonce"))
    #expect(runtime.contains("response.nonce != sessionNonce"))
    #expect(runtime.contains("RuntimeWorkerProtocolIO.isolatingStandardIO()"))
    let protocolIsolation = try #require(runtime.range(of: "RuntimeWorkerProtocolIO.isolatingStandardIO()"))
    let configLoad = try #require(runtime.range(of: "Qwen35Config.load(from: weightsPath)"))
    #expect(protocolIsolation.lowerBound < configLoad.lowerBound)
    #expect(runtime.contains("F_DUPFD_CLOEXEC"))
    #expect(runtime.contains("arc4random_buf(baseAddress, buffer.count)"))
    #expect(runtime.contains("redirectDescriptorToDevNull(STDIN_FILENO, flags: O_RDONLY"))
    #expect(runtime.contains("redirectDescriptorToDevNull(STDOUT_FILENO, flags: O_WRONLY"))
    #expect(runtime.contains("dup2(devNullFD, descriptor)"))
    #expect(runtime.contains("try protocolIO.writeLine(data)"))
    #expect(!runtime.contains("FileHandle.standardOutput.write(data)"))
}

@Test
func runtimeWorkerValidatesTransformedWeightsAtStartup() throws {
    let worker = try String(
        contentsOfFile: "Sources/MLXFastHarness/QwenRuntimeWorker.swift",
        encoding: .utf8
    )
    // The structural validation BenchmarkPreflight.check used to run in the
    // trusted parent (which links editable MLXFastModel code) now runs inside the
    // sandboxed worker at startup, before the protocol hello. This keeps the
    // coverage without executing submitted code in the score-writing process.
    let runWorkerStart = try #require(worker.range(of: "public static func runWorker"))
    let helloAnchor = try #require(worker.range(of: "RuntimeWorkerResponse(\n            id: 0,"))
    let startup = String(worker[runWorkerStart.lowerBound..<helloAnchor.lowerBound])
    #expect(startup.contains("try loader.denseStore.validateReadableByteRanges()"))
    #expect(startup.contains("try loader.validateRequiredMetadata(config: config)"))
    #expect(startup.contains("try weightCache.requireLibraryModel()"))

    // The parent's worker decode path must not read any editable submission
    // hook. The former editable decode-delay knob (Gemma4SubmissionControls) is
    // removed entirely; assert no residual reference survives on this path.
    let runtime = try harnessRuntimeSource()
    let decodeStart = try #require(runtime.range(of: "static func measureWorkerDecode"))
    let decodeEnd = try #require(runtime.range(of: "static let bandwidthSource"))
    let workerDecode = String(runtime[decodeStart.lowerBound..<decodeEnd.lowerBound])
    #expect(!workerDecode.contains("submissionValidationDelayMilliseconds()"))
    #expect(!workerDecode.contains("decode validation delay enabled"))
    #expect(!workerDecode.contains("Gemma4SubmissionControls"))
}


@Test
func benchmarkLocalIterateModeUsesPublicFixtureAndNonOfficialScore() throws {
    let script = try String(contentsOfFile: "benchmark.sh", encoding: .utf8)
    let constants = try String(
        contentsOfFile: "Sources/MLXFastCore/Constants.swift",
        encoding: .utf8
    )
    let cli = try String(
        contentsOfFile: "Sources/MLXFastCLI/main.swift",
        encoding: .utf8
    )
    let runtime = try String(
        contentsOfFile: "Sources/MLXFastHarness/QwenRuntimeLocalIterate.swift",
        encoding: .utf8
    )
    let options = try String(
        contentsOfFile: "Sources/MLXFastHarness/QwenRuntime.swift",
        encoding: .utf8
    )

    #expect(script.contains("if [[ \"${arg}\" == \"--local-iterate\" ]]; then"))
    #expect(script.contains("SCORE_PATH=\"score.local-iterate.json\""))
    #expect(script.contains("GOLDEN_PATH=\"correctness_prompts/public_longcopy_gate_english_512_256.json\""))
    #expect(constants.contains(
        "public static let localIterateBenchmarkDecodeSteps = benchmarkDecodeSteps"
    ))
    #expect(cli.contains("flagOptions: [\"--local-submit\", \"--local-iterate\"]"))
    #expect(cli.contains("QwenRuntime.localIterate("))
    #expect(cli.contains("MLXFastConstants.defaultLocalIterateScorePath"))
    #expect(runtime.contains("runLocalIterateCheckedTimingWithWorker("))
    #expect(runtime.contains("includes_seed_prefill=true"))
    #expect(runtime.contains("\\(modeName) prefill measured start prompt_tokens="))
    #expect(runtime.contains("let prefillWorker = try RuntimeWorkerClient(options: workerOptions, weightsPath: weightsPath)"))
    #expect(runtime.contains("let decodeWorker = try RuntimeWorkerClient(options: workerOptions, weightsPath: weightsPath)"))
    #expect(runtime.contains("try prefillWorker.prefill(promptTokens: testCase.promptTokens)"))
    #expect(runtime.contains("try decodeWorker.beginDecode(seedTokens: testCase.promptTokens)"))
    #expect(runtime.contains("let expectedDecodeTokens = Array(testCase.expectedTokens.dropFirst().prefix(decodeSteps))"))
    #expect(runtime.contains("let inputToken = decodedStep == 0 ? expectedSeedToken : expectedDecodeTokens[decodedStep - 1]"))
    #expect(runtime.contains("try decodeWorker.decodeStep(inputToken: inputToken)"))
    #expect(!runtime.contains("teacherForcedCorrectnessStep(previousToken: testCase.expectedTokens[decodedStep])"))
    #expect(!runtime.contains("topLogits(from:"))
    // Local modes publish the estimated (non-official) score so the Yukon CLI
    // (`yukon run`), which requires a finite numeric score at scorePath, can
    // consume valid local runs; invalid timing produces an explicit failed payload.
    #expect(runtime.contains("let estimatedScore = BenchmarkScore.score("))
    #expect(runtime.contains("error: \"local estimated score is invalid:"))
    #expect(runtime.contains("score: estimatedScore"))
    #expect(options.contains("runtime: String = \"swift-local-iterate\""))
    let prefillStartRange = try #require(runtime.range(of: "\\(modeName) prefill measured start prompt_tokens="))
    let decodeStartRange = try #require(runtime.range(of: "\\(modeName) decode measured start tokens="))
    #expect(prefillStartRange.lowerBound < decodeStartRange.lowerBound)
}

@Test
func benchmarkTransformCacheKeyIgnoresRuntimeModelSources() throws {
    let script = try String(contentsOfFile: "benchmark.sh", encoding: .utf8)

    #expect(script.contains("This hash gates regeneration of weights/"))
    #expect(script.contains("\"Sources/MLXFastCore\""))
    #expect(script.contains("\"Sources/MLXFastTransform\""))
    #expect(script.contains("git ls-files --cached --others --exclude-standard"))
    #expect(!script.contains("\"Sources/MLXFastModel\""))
}

@Test
func benchmarkScriptForwardsLocalSubmitFlagToSwiftBenchmark() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let weights = root.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try "{}".write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let argLog = root.appendingPathComponent("args.txt")
    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    // benchmark.sh now reconstructs score.json from the trusted process stdout
    // (see emitScorePayloadToStdout), so the fake emits the payload there rather
    // than writing the file itself.
    try """
    #!/bin/sh
    printf '%s\\n' "$@" > "\(argLog.path)"
    cat <<'JSON'
    {
      "score": 1,
      "passed": true,
      "metrics": {
        "weights_hash": "fake-weights",
        "weights_file_count": 1,
        "weights_byte_count": 2
      }
    }
    JSON
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: fakeSwift.path
    )

    let score = root.appendingPathComponent("score.json")
    let integrity = root.appendingPathComponent("benchmark-integrity.json")
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["benchmark.sh", "--local-submit"]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    process.environment = benchmarkTestEnvironment([
        "MLXFAST_NO_SANDBOX": "1",
        "MLXFAST_SKIP_TRANSFORM": "1",
        "MLXFAST_SWIFT_BIN": fakeSwift.path,
        "MLXFAST_WEIGHTS_PATH": weights.path,
        "MLXFAST_SCORE_PATH": score.path,
        "MLXFAST_INTEGRITY_PATH": integrity.path,
    ])

    try process.run()
    process.waitUntilExit()

    let args = try String(contentsOf: argLog, encoding: .utf8)
    #expect(process.terminationStatus == 0)
    #expect(args.contains("benchmark\n"))
    #expect(args.contains("--golden\n"))
    #expect(args.contains("correctness_prompts/public_longcopy_gate_english_512_1024.json\n"))
    #expect(args.contains("--local-submit\n"))
}

@Test
func benchmarkScriptForwardsLocalIterateDefaultsToSwiftBenchmark() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let weights = root.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try "{}".write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let argLog = root.appendingPathComponent("args.txt")
    let score = root.appendingPathComponent("score.local-iterate.json")
    let integrity = root.appendingPathComponent("benchmark-integrity.local-iterate.json")
    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    // benchmark.sh now reconstructs score.json from the trusted process stdout
    // (see emitScorePayloadToStdout), so the fake emits the payload there rather
    // than writing the file itself.
    try """
    #!/bin/sh
    printf '%s\\n' "$@" > "\(argLog.path)"
    cat <<'JSON'
    {
      "score": null,
      "passed": true,
      "metrics": {
        "weights_hash": "fake-weights",
        "weights_file_count": 1,
        "weights_byte_count": 2
      }
    }
    JSON
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: fakeSwift.path
    )

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["benchmark.sh", "--local-iterate"]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    process.environment = benchmarkTestEnvironment([
        "MLXFAST_NO_SANDBOX": "1",
        "MLXFAST_SKIP_TRANSFORM": "1",
        "MLXFAST_SWIFT_BIN": fakeSwift.path,
        "MLXFAST_WEIGHTS_PATH": weights.path,
        "MLXFAST_SCORE_PATH": score.path,
        "MLXFAST_INTEGRITY_PATH": integrity.path,
    ])

    try process.run()
    process.waitUntilExit()

    let args = try String(contentsOf: argLog, encoding: .utf8)
    #expect(process.terminationStatus == 0)
    #expect(args.contains("benchmark\n"))
    #expect(args.contains("--golden\n"))
    #expect(args.contains("correctness_prompts/public_longcopy_gate_english_512_256.json\n"))
    #expect(args.contains("--score-path\n"))
    #expect(args.contains("\(score.path)\n"))
    #expect(args.contains("--local-iterate\n"))
}

@Test
func localIterateStreamsLiveNumbersDuringTheRun() throws {
    // The local edit loop must show useful numbers the moment they exist, not
    // only in the final score JSON: per-token prefill status with speedup, the
    // seed prefill charge, per-step running decode numbers with a projected
    // score, heartbeats during long silent forwards, an immediate (redacted)
    // token-mismatch report, and a final summary block.
    let runtime = try String(
        contentsOfFile: "Sources/MLXFastHarness/QwenRuntimeLocalIterate.swift",
        encoding: .utf8
    )

    // Both timing paths (in-process and runtime-worker) stream the same data.
    #expect(runtime.components(separatedBy: "localIteratePrefillStatus(").count >= 4)
    #expect(runtime.components(separatedBy: "decode seed prefill complete ").count >= 3)
    #expect(runtime.components(separatedBy: "localIterateLiveDecodeStatus(").count >= 4)
    #expect(runtime.components(separatedBy: "reportFirstTokenMismatch(").count >= 8)
    #expect(runtime.components(separatedBy: "startPhaseHeartbeat(").count >= 6)
    #expect(runtime.contains("(charged to decode)"))
    #expect(runtime.contains("projected_decode_seconds_per_token="))
    #expect(runtime.contains("projected_score="))
    #expect(runtime.contains("decode_eta_seconds="))
    // The RAM-resident dense runtime has no expert-streaming machinery, so
    // the live decode status no longer reports expert bandwidth/hit-rate.
    #expect(!runtime.contains("expert_gb_per_token="))
    #expect(!runtime.contains("expert_hit_rate="))
    #expect(runtime.contains("emitLocalIterateSummary(modeName: modeName, timing: timing, progress: progress)"))
    // Baseline constants are printed up front so live speedups have context.
    #expect(runtime.contains("official-runner constants; local speedups are directional"))
    // The immediate mismatch line must stay redacted like the shared error
    // path: no expected/actual token values in the progress stream.
    #expect(runtime.contains("expected/actual tokens are in the score JSON"))
}

// The first-run experience on non-M5 Apple Silicon is a deterministic public
// gate failure (M5-generated goldens; near-tie argmax divergence -- observed
// on an M4 Pro at checked_step=17 on unmodified main). Every
// participant-visible failure surface -- the immediate harness FAIL line, the
// shell's failing-score summary, and the standalone correctness command --
// carries the one-sentence caveat pointing at the README callout and naming
// the ranked M5 runner as the source of truth. Messaging only: the gate is
// not weakened and failing runs still exit non-zero.
@Test
func localCorrectnessFailureSurfacesCarryNonM5GoldenCaveat() throws {
    let script = try String(contentsOfFile: "benchmark.sh", encoding: .utf8)
    let runtime = try String(
        contentsOfFile: "Sources/MLXFastHarness/QwenRuntimeLocalIterate.swift",
        encoding: .utf8
    )
    let cli = try String(
        contentsOfFile: "Sources/MLXFastCLI/main.swift",
        encoding: .utf8
    )
    let readme = try String(contentsOfFile: "README.md", encoding: .utf8)

    // Harness: the caveat is emitted with every first-token-mismatch report;
    // both local modes (and both timing paths) route through
    // reportFirstTokenMismatch.
    #expect(runtime.contains("public static let nonM5GoldenMismatchCaveat"))
    #expect(runtime.contains("progress?(\"\\(modeName) \\(nonM5GoldenMismatchCaveat)\")"))
    #expect(runtime.contains("a deterministic near-tie token mismatch is expected for a correct build"))

    // Shell: the failing-score summary is followed by the caveat for
    // local-mode token-mismatch failures only.
    #expect(script.contains("benchmark produced a failing score; see ${SCORE_PATH}"))
    #expect(script.contains("(.metrics.error // \"\") | test(\"token mismatch\")"))
    #expect(script.contains(
        "the public goldens are M5-generated, so on non-M5 Apple Silicon a deterministic "
            + "near-tie token mismatch is expected for a correct build"
    ))
    #expect(script.contains("the ranked M5 runner is the source of truth"))

    // Standalone correctness command: same sentence on stderr for mismatch
    // failures, with the exit code unchanged.
    #expect(cli.contains("if !report.passed, report.error.contains(\"token mismatch\")"))
    #expect(cli.contains("QwenRuntime.nonM5GoldenMismatchCaveat"))
    #expect(cli.contains("return report.passed ? 0 : 1"))

    // The caveat points at a README callout that actually exists.
    #expect(readme.contains("**Correctness fixtures are M5-generated.**"))
}

@Test
func benchmarkScriptPrintsNonM5CaveatOnlyForLocalTokenMismatchFailures() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let weights = root.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try "{}".write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    func runLocalIterate(failureError: String) throws -> (status: Int32, stderr: String) {
        let fakeSwift = root.appendingPathComponent("mlxfast-swift")
        try """
        #!/bin/sh
        cat <<'JSON'
        {
          "score": 0.998,
          "passed": false,
          "metrics": {
            "error": "\(failureError)",
            "weights_hash": "fake-weights",
            "weights_file_count": 1,
            "weights_byte_count": 2
          }
        }
        JSON
        """.write(to: fakeSwift, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: fakeSwift.path
        )
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["benchmark.sh", "--local-iterate"]
        process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        process.environment = benchmarkTestEnvironment([
            "MLXFAST_NO_SANDBOX": "1",
            "MLXFAST_SKIP_TRANSFORM": "1",
            "MLXFAST_SWIFT_BIN": fakeSwift.path,
            "MLXFAST_WEIGHTS_PATH": weights.path,
            "MLXFAST_SCORE_PATH": root.appendingPathComponent("score.local-iterate.json").path,
            "MLXFAST_INTEGRITY_PATH": root.appendingPathComponent("integrity.local-iterate.json").path,
        ])
        let stderrPipe = Pipe()
        process.standardError = stderrPipe
        try process.run()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(data: stderrData, encoding: .utf8) ?? "")
    }

    // A local token-mismatch failure gets the caveat and still fails the run.
    let mismatch = try runLocalIterate(failureError: "local-iterate teacher-forced token mismatch")
    #expect(mismatch.status != 0)
    #expect(mismatch.stderr.contains("benchmark produced a failing score"))
    #expect(mismatch.stderr.contains("the public goldens are M5-generated"))
    #expect(mismatch.stderr.contains("the ranked M5 runner is the source of truth"))

    // Unrelated local failures keep the original message without the caveat.
    let unrelated = try runLocalIterate(failureError: "weights digest failed")
    #expect(unrelated.status != 0)
    #expect(unrelated.stderr.contains("benchmark produced a failing score"))
    #expect(!unrelated.stderr.contains("the public goldens are M5-generated"))
}

// Frozen or near-zero GPU temperature telemetry (observed: macmon 0.7.2 on
// macOS 26 / M4 Pro reporting a constant 3.657C for 20+ minutes) makes the
// local thermal gate pass instantly and silently, so gated timings are
// effectively ungated. The gate prints one calm warning when the reading
// looks implausible -- at/below the 5C plausibility floor or exactly constant
// across the gate's samples -- without failing the run and without touching
// the ranked box-side gate.
@Test
func coolGateWarnsOnceWhenGpuTelemetryLooksImplausible() throws {
    let script = try String(contentsOfFile: "benchmark.sh", encoding: .utf8)
    #expect(script.contains("warn_if_gpu_telemetry_implausible() {"))
    #expect(script.contains("at or below the 5C plausibility floor"))
    #expect(script.contains("has read a constant $(format_temp_c \"${temp}\")C across ${sample_count} samples"))
    #expect(script.contains("gated timings may effectively be ungated"))
    // Warn-once bookkeeping; the warning never changes the gate outcome.
    #expect(script.contains("telemetry_warned=1"))

    func runCoolGateOnly(gpuTempCommand: String) throws -> (status: Int32, stderr: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["benchmark.sh", "--local-cool-gate-only"]
        process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        process.environment = benchmarkTestEnvironment([
            "MLXFAST_GPU_TEMP_CMD": gpuTempCommand,
        ])
        let stderrPipe = Pipe()
        process.standardError = stderrPipe
        try process.run()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(data: stderrData, encoding: .utf8) ?? "")
    }

    // The observed M4 Pro shape: a frozen 3.657C reading passes the gate
    // immediately, so the warning must ride along with the pass.
    let frozen = try runCoolGateOnly(gpuTempCommand: "printf 3.657")
    #expect(frozen.status == 0)
    #expect(frozen.stderr.contains(
        "benchmark.sh: warning: the GPU temperature reads 3.7C, at or below the 5C plausibility floor"
    ))
    #expect(frozen.stderr.contains("temperature reading looks implausible on this hardware/OS"))
    #expect(frozen.stderr.contains("GPU cool-down gate passed"))

    // A plausible reading passes quietly, exactly as before.
    let plausible = try runCoolGateOnly(gpuTempCommand: "printf 39")
    #expect(plausible.status == 0)
    #expect(!plausible.stderr.contains("looks implausible"))
    #expect(plausible.stderr.contains("GPU cool-down gate passed"))
}

// The local cool gate can offer a ONE-TIME fan boost when the GPU sits hot
// without cooling progress: opt-in on the terminal, sudo-gated (SMC fan
// writes are root-only), hard-capped at 70% of each fan's maximum speed, and
// reversible with `./benchmark.sh --fan-speed-normal`, which returns the fans
// to macOS's automatic curve without pinning any RPM. Password-handling
// contract: sudo prompts on the tty itself; the scripts never pipe, read,
// store, or log the password, and the cached credential is dropped with
// `sudo -k` right after the writes.
@Test
func coolGateFanBoostIsOptInSudoSafeAndHardCappedAtSeventyPercent() throws {
    let script = try String(contentsOfFile: "benchmark.sh", encoding: .utf8)
    let fan = try String(contentsOfFile: "tools/fan-control.sh", encoding: .utf8)

    // benchmark.sh wiring: offer threshold constant, a single offer per gate
    // invocation (before the stall abort can fire), and the restore flag that
    // execs the helper's `normal` command.
    #expect(script.contains("readonly COOL_GATE_FAN_OFFER_STALL_SECONDS=60"))
    #expect(script.contains("--fan-speed-normal"))
    #expect(script.contains("exec \"${fan_helper}\" normal"))
    #expect(script.contains("fan_boost_offered=1"))
    #expect(script.contains("if offer_fan_boost \"${temp}\"; then"))
    // A boost that engages resets the stall clock so the fans get a fresh
    // window before the abort.
    #expect(script.contains("last_progress_waited=\"${waited}\""))
    // Interactive-only: without a tty there is no sudo prompt, just a hint.
    #expect(script.contains(": < /dev/tty"))
    #expect(script.contains("read -r -t 30 -p"))
    // The sudo reasoning is printed before any password prompt can appear.
    #expect(script.contains("the password is never read, stored,"))
    // Never wire a password into sudo from the shell.
    #expect(!script.contains("sudo -S"))
    #expect(!script.contains("SUDO_ASKPASS"))

    // Helper contract: hard 70% cap, explained sudo, credential hygiene.
    #expect(fan.contains("readonly FAN_BOOST_PERCENT=70"))
    #expect(fan.contains("sudo -k"))
    #expect(!fan.contains("sudo -S"))
    #expect(!fan.contains("SUDO_ASKPASS"))
    #expect(fan.contains("never reads, stores"))
    #expect(fan.contains("explain_sudo"))
    // `normal` only clears the manual-mode bit (macOS's automatic fan curve
    // takes over); it never pins an RPM of its own, and `boost` writes the
    // target through the same root-only SMC keys.
    #expect(fan.contains("-k \"F${i}Md\" -w 00"))
    #expect(fan.contains("-k \"F${i}Md\" -w 01"))
    #expect(fan.contains("-k \"F${i}Tg\" -w \"${hex}\""))
}

// Robustness contract layered on top of the boost: (1) `smc -w` is
// fire-and-forget, so every boost write is read back and a fan whose write
// did not stick fails the boost loudly instead of reporting a
// silently-ineffective 70%; (2) mode inspection covers EVERY fan (a foreign
// controller or a partially-restored boost can hold any subset manual), and
// `boost` refuses to overwrite another controller's targets; (3) benchmark.sh
// tracks whether THIS run applied a boost so a completed run ends with a
// restore reminder and an INT/TERM abort restores automatic control instead
// of leaving the machine pinned at 70%.
@Test
func fanBoostVerifiesWritesInspectsAllFansAndTracksRunRestoreState() throws {
    let script = try String(contentsOfFile: "benchmark.sh", encoding: .utf8)
    let fan = try String(contentsOfFile: "tools/fan-control.sh", encoding: .utf8)

    // (1) Write verification: both SMC keys are read back per fan, compared
    // within a small float-rounding tolerance, and a non-stuck write names
    // the fan and fails the boost (non-zero exit), rolling every fan back to
    // automatic so no half-boosted state survives a reported failure.
    #expect(fan.contains("verify_fan_boost_write() {"))
    #expect(fan.contains("if verify_fan_boost_write \"${i}\" \"${target}\"; then"))
    #expect(fan.contains("smc_read_number \"F${fan_index}Md\""))
    #expect(fan.contains("smc_read_number \"F${fan_index}Tg\""))
    #expect(fan.contains("readonly FAN_TARGET_VERIFY_TOLERANCE_RPM="))
    #expect(fan.contains("write did not stick"))
    #expect(fan.contains("boost may not have taken effect for fan(s) ${failed_fans}"))
    #expect(fan.contains("if [[ -n \"${failed_fans}\" ]]; then"))

    // (2) Foreign-controller detection: every fan's mode bit is inspected
    // (fan 0 alone no longer decides `status`), and `boost` refuses to
    // overwrite a manual hold it does not own instead of clobbering it.
    #expect(fan.contains("list_manual_fans() {"))
    #expect(fan.contains("fan_mode_is_manual \"${i}\""))
    #expect(fan.contains("manual_fans=\"$(list_manual_fans \"${fan_count}\")\""))
    #expect(!fan.contains("smc_read_number \"F0Md\""))
    #expect(fan.contains("already in manual mode; another fan controller"))
    // benchmark.sh's offer reconciles with that refusal: a hold this run did
    // not create skips the offer with one warning (no sudo prompt destined to
    // be refused, no double-warn) and does not reset the stall clock.
    #expect(script.contains("if fan_boost_recorded; then"))
    #expect(script.contains("another fan controller (or a prior un-restored boost) already holds the fans in manual mode"))

    // (3) Run-scoped restore tracking: the top-level local run owns a state
    // file, gate children record an applied boost into it, a completed run
    // prints the restore reminder (fans intentionally stay forced -- no
    // auto-restore on the normal path), and INT/TERM restores ONLY when this
    // run applied a boost.
    #expect(script.contains("setup_fan_boost_run_tracking() {"))
    #expect(script.contains("export MLXFAST_FAN_BOOST_STATE_FILE="))
    #expect(script.contains("record_fan_boost_applied"))
    #expect(script.contains("report_fan_boost_restore_reminder"))
    #expect(script.contains("still forced to 70% of max"))
    #expect(script.contains("trap 'handle_benchmark_abort_signal INT' INT"))
    #expect(script.contains("trap 'handle_benchmark_abort_signal TERM' TERM"))
    #expect(script.contains("if [[ -n \"${FAN_BOOST_STATE_FILE_OWNED}\" ]] && fan_boost_recorded; then"))
    #expect(script.contains("returning them to macOS automatic control"))
}

// End-to-end over the real script: a local run that recorded a fan boost
// finishes successfully, prints the restore reminder, and does NOT hand the
// fans back automatically -- staying forced through the timed phases (and
// past exit) is the intended behavior; only the reminder tells the user how
// to undo it.
@Test
func benchmarkPrintsFanRestoreReminderAfterBoostWithoutAutoRestoring() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let weights = root.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try "{}".write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let score = root.appendingPathComponent("score.local-iterate.json")
    let integrity = root.appendingPathComponent("benchmark-integrity.local-iterate.json")
    let fanLog = root.appendingPathComponent("fan-helper-invocations.log")
    let fakeFanHelper = root.appendingPathComponent("fake-fan-control.sh")
    try """
    #!/bin/sh
    printf '%s\\n' "$1" >> "\(fanLog.path)"
    if [ "$1" = "status" ]; then echo auto; fi
    exit 0
    """.write(to: fakeFanHelper, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: fakeFanHelper.path
    )

    // The fake benchmark stands in for the cool-gate child that applies a
    // boost mid-run: it records into the state file benchmark.sh exported,
    // then completes normally with a passing score payload.
    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    try """
    #!/bin/sh
    printf 'applied\\n' >> "${MLXFAST_FAN_BOOST_STATE_FILE}"
    cat <<'JSON'
    {
      "score": null,
      "passed": true,
      "metrics": {
        "weights_hash": "fake-weights",
        "weights_file_count": 1,
        "weights_byte_count": 2
      }
    }
    JSON
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: fakeSwift.path
    )

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["benchmark.sh", "--local-iterate"]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    process.environment = benchmarkTestEnvironment([
        "MLXFAST_NO_SANDBOX": "1",
        "MLXFAST_SKIP_TRANSFORM": "1",
        "MLXFAST_SWIFT_BIN": fakeSwift.path,
        "MLXFAST_WEIGHTS_PATH": weights.path,
        "MLXFAST_SCORE_PATH": score.path,
        "MLXFAST_INTEGRITY_PATH": integrity.path,
        "MLXFAST_FAN_CONTROL_HELPER": fakeFanHelper.path,
    ])
    let stderrPipe = Pipe()
    process.standardError = stderrPipe
    try process.run()
    let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    let stderr = String(data: stderrData, encoding: .utf8) ?? ""

    #expect(process.terminationStatus == 0)
    #expect(stderr.contains("this run boosted the fans; they are still forced to 70% of max"))
    #expect(stderr.contains("./benchmark.sh --fan-speed-normal"))
    // No automatic restore on the normal path: the helper is never asked to
    // hand control back (the log records every helper invocation).
    let helperInvocations = (try? String(contentsOf: fanLog, encoding: .utf8)) ?? ""
    #expect(!helperInvocations.contains("normal"))
}

// End-to-end over the real script: TERM during a run that recorded a fan
// boost restores automatic fan control through the same helper path
// --fan-speed-normal uses, says so, and exits with the conventional
// 128+SIGTERM status. Runs that never boosted are untouched by the handler.
@Test
func benchmarkRestoresFansWhenAbortedAfterBoost() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let weights = root.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try "{}".write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let score = root.appendingPathComponent("score.local-iterate.json")
    let integrity = root.appendingPathComponent("benchmark-integrity.local-iterate.json")
    let fanLog = root.appendingPathComponent("fan-helper-invocations.log")
    let fakeFanHelper = root.appendingPathComponent("fake-fan-control.sh")
    try """
    #!/bin/sh
    printf '%s\\n' "$1" >> "\(fanLog.path)"
    if [ "$1" = "status" ]; then echo manual; fi
    exit 0
    """.write(to: fakeFanHelper, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: fakeFanHelper.path
    )

    // The fake benchmark records a boost, signals it is mid-run via a marker
    // file, then idles so the test can deliver TERM while the run is live.
    let marker = root.appendingPathComponent("benchmark-running.marker")
    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    try """
    #!/bin/sh
    printf 'applied\\n' >> "${MLXFAST_FAN_BOOST_STATE_FILE}"
    : > "\(marker.path)"
    sleep 5
    exit 0
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: fakeSwift.path
    )

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["benchmark.sh", "--local-iterate"]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    process.environment = benchmarkTestEnvironment([
        "MLXFAST_NO_SANDBOX": "1",
        "MLXFAST_SKIP_TRANSFORM": "1",
        "MLXFAST_SWIFT_BIN": fakeSwift.path,
        "MLXFAST_WEIGHTS_PATH": weights.path,
        "MLXFAST_SCORE_PATH": score.path,
        "MLXFAST_INTEGRITY_PATH": integrity.path,
        "MLXFAST_FAN_CONTROL_HELPER": fakeFanHelper.path,
    ])
    let stderrPipe = Pipe()
    process.standardError = stderrPipe
    try process.run()

    var markerSeen = false
    for _ in 0..<100 {
        if FileManager.default.fileExists(atPath: marker.path) {
            markerSeen = true
            break
        }
        usleep(100_000)
    }
    #expect(markerSeen)
    process.terminate()

    let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    let stderr = String(data: stderrData, encoding: .utf8) ?? ""

    #expect(process.terminationStatus == 143)
    #expect(stderr.contains("TERM received after this run boosted the fans"))
    #expect(stderr.contains("fans restored to macOS automatic control"))
    let helperInvocations = (try? String(contentsOf: fanLog, encoding: .utf8)) ?? ""
    #expect(helperInvocations.contains("normal"))
}

// The ~21.6 GB RAM-resident model means TWO simultaneous residencies (an
// overlapping second local run, or a new run started while an orphaned
// model-holding worker from an aborted run lingers) can out-of-memory a
// 36 GiB machine. Local modes must therefore (1) serialize runs behind a
// per-user lock, (2) refuse to start while a model-holding process is
// already alive -- warn-and-abort, never auto-kill -- and (3) reap the
// spawned benchmark process tree on INT/TERM/EXIT so aborted runs cannot
// orphan the worker in the first place. All of it is scoped to local modes;
// the ranked --official invocation stays byte-identical foreground.
@Test
func localModesGuardAgainstDoubleModelResidency() throws {
    let script = try String(contentsOfFile: "benchmark.sh", encoding: .utf8)

    // Startup guard: per-user run lock with stale-holder reclaim, plus a
    // resident-model process scan with a documented testing seam.
    #expect(script.contains("local_run_guard_enabled() {"))
    #expect(script.contains("acquire_local_run_lock() {"))
    #expect(script.contains("release_local_run_lock() {"))
    #expect(script.contains("abort_if_model_already_resident() {"))
    #expect(script.contains("list_resident_model_processes() {"))
    #expect(script.contains("readonly RESIDENT_MODEL_PROCESS_PATTERN="))
    #expect(script.contains("runtime-worker[[:space:]]+--weights"))
    #expect(script.contains("MLXFAST_LOCAL_RUN_GUARD"))
    #expect(script.contains("MLXFAST_LOCAL_RUN_LOCK_DIR"))
    #expect(script.contains("MLXFAST_LOCAL_ORPHAN_SCAN_CMD"))
    #expect(script.contains("removing stale local run lock"))
    // Conservative policy: detection warns and aborts with instructions; the
    // script must never kill a process it did not spawn.
    #expect(script.contains("WARN-AND-ABORT only"))
    #expect(script.contains("never auto-kill"))
    // Wired into the local-mode setup block (after fan tracking), so the
    // ranked --official path never runs the guard.
    #expect(script.contains("acquire_local_run_lock\n  abort_if_model_already_resident"))
    let guardScope = try #require(script.range(of: "local_run_guard_enabled() {"))
    let guardScopeBody = String(script[guardScope.lowerBound...].prefix(220))
    #expect(guardScopeBody.contains("LOCAL_ITERATE"))
    #expect(guardScopeBody.contains("LOCAL_SUBMIT"))

    // Guaranteed teardown: local modes run the Swift benchmark as a monitored
    // background child; INT/TERM reap the tree before the fan restore, and
    // the EXIT cleanup reaps last-resort and releases the lock.
    #expect(script.contains("signal_process_tree() {"))
    #expect(script.contains("terminate_benchmark_child_tree() {"))
    #expect(script.contains("BENCHMARK_CHILD_PID=\"$!\""))
    #expect(script.contains("wait \"${BENCHMARK_CHILD_PID}\""))
    #expect(script.contains("FAN_BOOST_ABORT_HANDLED=1\n  terminate_benchmark_child_tree"))
    #expect(script.contains("terminate_benchmark_child_tree\n  release_local_run_lock"))

    // The monitored-background-child launch exists only on the local branch;
    // the official path keeps the original foreground invocation.
    let localLaunch = try #require(
        script.range(of: "${FORWARD_ARGS[@]+\"${FORWARD_ARGS[@]}\"} > \"${score_stdout}\" &")
    )
    let officialLaunch = try #require(
        script.range(
            of: "${FORWARD_ARGS[@]+\"${FORWARD_ARGS[@]}\"} > \"${score_stdout}\"\nfi"
        )
    )
    #expect(localLaunch.lowerBound < officialLaunch.lowerBound)
}

@Test
func localRunLockRejectsOverlappingRunAndReclaimsStaleLock() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let weights = root.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try "{}".write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    let lockRoot = root.appendingPathComponent("locks")
    try FileManager.default.createDirectory(at: lockRoot, withIntermediateDirectories: true)
    let lockPath = lockRoot.appendingPathComponent("mlxfast-local-benchmark-\(getuid()).lock")

    let invocation = root.appendingPathComponent("swift-invoked")
    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    try """
    #!/bin/sh
    : > "\(invocation.path)"
    cat <<'JSON'
    {
      "score": null,
      "passed": true,
      "metrics": {
        "weights_hash": "fake-weights",
        "weights_file_count": 1,
        "weights_byte_count": 2
      }
    }
    JSON
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: fakeSwift.path
    )

    func runLocalIterate() throws -> (status: Int32, stderr: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["benchmark.sh", "--local-iterate"]
        process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        process.environment = benchmarkTestEnvironment([
            "MLXFAST_NO_SANDBOX": "1",
            "MLXFAST_SKIP_TRANSFORM": "1",
            "MLXFAST_SWIFT_BIN": fakeSwift.path,
            "MLXFAST_WEIGHTS_PATH": weights.path,
            "MLXFAST_SCORE_PATH": root.appendingPathComponent("score.json").path,
            "MLXFAST_INTEGRITY_PATH": root.appendingPathComponent("integrity.json").path,
            "MLXFAST_LOCAL_RUN_GUARD": "1",
            "MLXFAST_LOCAL_RUN_LOCK_DIR": lockRoot.path,
            // The resident scan is exercised by its own test; keep this one
            // about the lock alone.
            "MLXFAST_LOCAL_ORPHAN_SCAN_CMD": "true",
        ])
        let stderrPipe = Pipe()
        process.standardError = stderrPipe
        try process.run()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(data: stderrData, encoding: .utf8) ?? "")
    }

    // A lock held by a LIVE pid (this test process) rejects the overlapping
    // run before any benchmark work starts.
    try FileManager.default.createDirectory(at: lockPath, withIntermediateDirectories: true)
    try "\(ProcessInfo.processInfo.processIdentifier)\n".write(
        to: lockPath.appendingPathComponent("pid"),
        atomically: true,
        encoding: .utf8
    )
    let rejected = try runLocalIterate()
    #expect(rejected.status != 0)
    #expect(rejected.stderr.contains("another local benchmark run (pid"))
    #expect(rejected.stderr.contains("two overlapping local runs"))
    #expect(!FileManager.default.fileExists(atPath: invocation.path))

    // A lock whose recorded holder is gone (a killed run never cleans up) is
    // reclaimed instead of wedging the edit loop; the run then completes and
    // the EXIT cleanup removes the lock again.
    try "999999\n".write(
        to: lockPath.appendingPathComponent("pid"),
        atomically: true,
        encoding: .utf8
    )
    let reclaimed = try runLocalIterate()
    #expect(reclaimed.status == 0)
    #expect(reclaimed.stderr.contains("removing stale local run lock"))
    #expect(FileManager.default.fileExists(atPath: invocation.path))
    #expect(!FileManager.default.fileExists(atPath: lockPath.path))
}

@Test
func localRunAbortsWhenAModelHoldingProcessIsAlreadyResident() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let weights = root.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try "{}".write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    let lockRoot = root.appendingPathComponent("locks")
    try FileManager.default.createDirectory(at: lockRoot, withIntermediateDirectories: true)

    let invocation = root.appendingPathComponent("swift-invoked")
    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    try """
    #!/bin/sh
    : > "\(invocation.path)"
    exit 0
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: fakeSwift.path
    )

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["benchmark.sh", "--local-iterate"]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    process.environment = benchmarkTestEnvironment([
        "MLXFAST_NO_SANDBOX": "1",
        "MLXFAST_SKIP_TRANSFORM": "1",
        "MLXFAST_SWIFT_BIN": fakeSwift.path,
        "MLXFAST_WEIGHTS_PATH": weights.path,
        "MLXFAST_SCORE_PATH": root.appendingPathComponent("score.json").path,
        "MLXFAST_INTEGRITY_PATH": root.appendingPathComponent("integrity.json").path,
        "MLXFAST_LOCAL_RUN_GUARD": "1",
        "MLXFAST_LOCAL_RUN_LOCK_DIR": lockRoot.path,
        // Testing seam (same pattern as MLXFAST_GPU_TEMP_CMD): stand in for
        // the pgrep/ps listing with one fake resident worker line.
        "MLXFAST_LOCAL_ORPHAN_SCAN_CMD":
            "printf '12345 1 17301504 mlxfast-runtime-worker runtime-worker --weights weights\\n'",
    ])
    let stderrPipe = Pipe()
    process.standardError = stderrPipe
    try process.run()
    let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    let stderr = String(data: stderrData, encoding: .utf8) ?? ""

    #expect(process.terminationStatus != 0)
    #expect(stderr.contains("a model-holding mlxfast process is already running"))
    #expect(stderr.contains("mlxfast-runtime-worker runtime-worker --weights weights"))
    #expect(stderr.contains("kill <pid>"))
    #expect(stderr.contains("MLXFAST_LOCAL_RUN_GUARD=0"))
    // The guard fires before any benchmark work: the Swift binary never ran,
    // and the aborted run released its lock for the next attempt.
    #expect(!FileManager.default.fileExists(atPath: invocation.path))
    let lockPath = lockRoot.appendingPathComponent("mlxfast-local-benchmark-\(getuid()).lock")
    #expect(!FileManager.default.fileExists(atPath: lockPath.path))
}

@Test
func terminatedLocalRunReapsTheModelHoldingProcessTree() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let weights = root.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try "{}".write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    // The fake benchmark stands in for `mlxfast-swift benchmark`: it spawns a
    // long-lived child (the stand-in for the ~21.6 GB runtime worker), records
    // both pids, then idles like a real run mid-measurement.
    let workerPidFile = root.appendingPathComponent("worker.pid")
    let marker = root.appendingPathComponent("benchmark-running.marker")
    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    try """
    #!/bin/sh
    /bin/sh -c 'echo $$ > "\(workerPidFile.path)"; exec sleep 300' &
    : > "\(marker.path)"
    sleep 300
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: fakeSwift.path
    )

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["benchmark.sh", "--local-iterate"]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    process.environment = benchmarkTestEnvironment([
        "MLXFAST_NO_SANDBOX": "1",
        "MLXFAST_SKIP_TRANSFORM": "1",
        "MLXFAST_SWIFT_BIN": fakeSwift.path,
        "MLXFAST_WEIGHTS_PATH": weights.path,
        "MLXFAST_SCORE_PATH": root.appendingPathComponent("score.json").path,
        "MLXFAST_INTEGRITY_PATH": root.appendingPathComponent("integrity.json").path,
    ])
    let stderrPipe = Pipe()
    process.standardError = stderrPipe
    try process.run()

    var workerPid: Int32 = 0
    for _ in 0..<100 {
        if FileManager.default.fileExists(atPath: marker.path),
           let recorded = try? String(contentsOf: workerPidFile, encoding: .utf8),
           let parsed = Int32(recorded.trimmingCharacters(in: .whitespacesAndNewlines))
        {
            workerPid = parsed
            break
        }
        usleep(100_000)
    }
    #expect(workerPid > 0)
    defer {
        // Never leave the fixture behind if an assertion fails mid-test.
        if workerPid > 0 { _ = kill(workerPid, SIGKILL) }
    }

    // TERM only the top-level benchmark.sh -- exactly what an agent or
    // process manager does to abort a run. Before the teardown fix, bash
    // deferred the trap until the foreground Swift child finished and the
    // worker stand-in survived as an orphan.
    process.terminate()
    let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    let stderr = String(data: stderrData, encoding: .utf8) ?? ""

    #expect(process.terminationStatus == 143)
    #expect(stderr.contains("stopping the in-flight benchmark process tree"))
    var workerReaped = false
    for _ in 0..<100 {
        if kill(workerPid, 0) != 0 {
            workerReaped = true
            break
        }
        usleep(100_000)
    }
    #expect(workerReaped)
}

@Test
func benchmarkScriptPrintsLocalSummaryWithBaselineComparison() throws {
    let script = try String(contentsOfFile: "benchmark.sh", encoding: .utf8)
    #expect(script.contains("report_local_score_summary() {"))
    #expect(script.contains("cat \"${SCORE_PATH}\"\n  report_local_score_summary"))
    // The baseline context prints before the run. The shell exports its trusted
    // thermal helper, and the harness invokes it at each timed phase boundary.
    #expect(script.contains("report_local_baseline_context() {"))
    #expect(script.contains("export MLXFAST_LOCAL_COOL_GATE_HELPER"))
    let baselineContext = try #require(script.range(of: "\nreport_local_baseline_context\n"))
    let swiftInvocation = try #require(script.range(of: "\"${SWIFT_BIN}\" benchmark \\"))
    #expect(baselineContext.lowerBound < swiftInvocation.lowerBound)

    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let weights = root.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try "{}".write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let score = root.appendingPathComponent("score.local-iterate.json")
    let integrity = root.appendingPathComponent("benchmark-integrity.local-iterate.json")
    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    try """
    #!/bin/sh
    cat <<'JSON'
    {
      "score": null,
      "passed": true,
      "metrics": {
        "prefill_seconds_per_token": 0.1,
        "decode_seconds_per_token": 2.0,
        "prefill_speedup": 2.0,
        "decode_speedup": 2.0,
        "weights_hash": "fake-weights",
        "weights_file_count": 1,
        "weights_byte_count": 2
      }
    }
    JSON
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: fakeSwift.path
    )

    func runLocalIterate() throws -> (status: Int32, stderr: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["benchmark.sh", "--local-iterate"]
        process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        process.environment = benchmarkTestEnvironment([
            "MLXFAST_NO_SANDBOX": "1",
            "MLXFAST_SKIP_TRANSFORM": "1",
            "MLXFAST_SWIFT_BIN": fakeSwift.path,
            "MLXFAST_WEIGHTS_PATH": weights.path,
            "MLXFAST_SCORE_PATH": score.path,
            "MLXFAST_INTEGRITY_PATH": integrity.path,
        ])
        let stderrPipe = Pipe()
        process.standardError = stderrPipe
        try process.run()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(data: stderrData, encoding: .utf8) ?? "")
    }

    // First run: no baseline snapshot yet, so the summary prints the current
    // numbers plus a hint about recording one.
    let first = try runLocalIterate()
    #expect(first.status == 0)
    #expect(first.stderr.contains("benchmark.sh: local-iterate summary"))
    #expect(first.stderr.contains("prefill 0.1 s/token  speedup 2x"))
    #expect(first.stderr.contains("decode  2 s/token  speedup 2x"))
    // 2x on both axes -> estimated score 2 under decode^0.75 * prefill^0.25.
    #expect(first.stderr.contains("est score 2 (decode_speedup^0.75 * prefill_speedup^0.25"))
    #expect(first.stderr.contains("no local baseline at"))

    // Second run: with a slower baseline snapshot the summary reports deltas.
    let baseline = root.appendingPathComponent("score.local-iterate.baseline.json")
    try """
    {
      "score": null,
      "passed": true,
      "metrics": {
        "prefill_seconds_per_token": 0.2,
        "decode_seconds_per_token": 4.0,
        "prefill_speedup": 1.0,
        "decode_speedup": 1.0
      }
    }
    """.write(to: baseline, atomically: true, encoding: .utf8)

    let second = try runLocalIterate()
    #expect(second.status == 0)
    // The baseline target is announced BEFORE the run so the live per-token
    // numbers have something to compare against from the first second.
    #expect(second.stderr.contains(
        "benchmark.sh: local baseline to beat (\(baseline.path)): "
            + "prefill 0.2 s/token, decode 4 s/token, est score 1"
    ))
    #expect(second.stderr.contains("benchmark.sh: local-iterate summary"))
    #expect(second.stderr.contains("vs \(baseline.path) (negative s/token deltas = faster)"))
    #expect(second.stderr.contains("prefill 0.2 -> 0.1 s/token (-50%)"))
    #expect(second.stderr.contains("decode  4 -> 2 s/token (-50%)"))
    #expect(second.stderr.contains("est score 1 -> 2 (+100%)"))
    #expect(!second.stderr.contains("no local baseline at"))
}

@Test
func localModesForwardWorkerStderrLiveButOfficialRunsDoNot() throws {
    let cli = try String(
        contentsOfFile: "Sources/MLXFastCLI/main.swift",
        encoding: .utf8
    )
    let worker = try String(
        contentsOfFile: "Sources/MLXFastHarness/QwenRuntimeWorker.swift",
        encoding: .utf8
    )
    let options = try String(
        contentsOfFile: "Sources/MLXFastHarness/QwenRuntime.swift",
        encoding: .utf8
    )

    // Only the local benchmark modes opt in; every other worker-options call
    // site keeps the default (off), and the builder forces the flag off for
    // official runs so submitted code cannot stream hidden-prompt content
    // into CI logs.
    #expect(options.contains("public let forwardsWorkerStderr: Bool"))
    #expect(cli.components(separatedBy: "forwardsWorkerStderr: true").count == 2)
    #expect(cli.contains("forwardsWorkerStderr: forwardsWorkerStderr && !officialRun"))

    // The drain forwards each line with the worker prefix after per-line
    // token redaction, keeps only a capped raw tail for the exit diagnostic,
    // and is attached before the protocol hello so setup output is drained too.
    #expect(worker.contains("final class WorkerStderrDrain"))
    #expect(worker.contains("static let forwardedLinePrefix = \"mlxfast-worker: \""))
    #expect(worker.contains("redactedWorkerStderrLine(line)"))
    #expect(worker.contains("self.stderrDrain = WorkerStderrDrain("))
    #expect(worker.contains("emit: options.forwardsWorkerStderr ? nil : { _ in }"))
    #expect(worker.contains("private let stderrDrain: WorkerStderrDrain"))
    #expect(!worker.contains("private let stderrDrain: WorkerStderrDrain?"))
}

@Test
func benchmarkScriptLocalSummarySkipsQuietlyWithoutTimingMetrics() throws {
    // Failure payloads and minimal fixtures have no positive timing numbers;
    // the summary must skip silently instead of printing zeros or failing the
    // run (it is diagnostic-only output).
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let weights = root.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try "{}".write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let score = root.appendingPathComponent("score.local-iterate.json")
    let integrity = root.appendingPathComponent("benchmark-integrity.local-iterate.json")
    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    try """
    #!/bin/sh
    cat <<'JSON'
    {
      "score": null,
      "passed": true,
      "metrics": {
        "weights_hash": "fake-weights",
        "weights_file_count": 1,
        "weights_byte_count": 2
      }
    }
    JSON
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: fakeSwift.path
    )

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["benchmark.sh", "--local-iterate"]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    process.environment = benchmarkTestEnvironment([
        "MLXFAST_NO_SANDBOX": "1",
        "MLXFAST_SKIP_TRANSFORM": "1",
        "MLXFAST_SWIFT_BIN": fakeSwift.path,
        "MLXFAST_WEIGHTS_PATH": weights.path,
        "MLXFAST_SCORE_PATH": score.path,
        "MLXFAST_INTEGRITY_PATH": integrity.path,
    ])
    let stderrPipe = Pipe()
    process.standardError = stderrPipe
    try process.run()
    let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    let stderr = String(data: stderrData, encoding: .utf8) ?? ""

    #expect(process.terminationStatus == 0)
    #expect(!stderr.contains("local-iterate summary"))
    #expect(!stderr.contains("est score"))
}

@Test
func benchmarkScriptRejectsPathFlagsBeforeForwardingToSwiftBenchmark() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    try """
    #!/bin/sh
    exit 99
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: fakeSwift.path
    )

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["benchmark.sh", "--local-iterate", "--score-path", "custom.json"]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    process.environment = benchmarkTestEnvironment([
        "MLXFAST_NO_SANDBOX": "1",
        "MLXFAST_SWIFT_BIN": fakeSwift.path,
    ])

    let stderr = Pipe()
    process.standardError = stderr
    try process.run()
    process.waitUntilExit()

    let error = String(data: stderr.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    #expect(process.terminationStatus == 1)
    #expect(error.contains("use MLXFAST_WEIGHTS_PATH, MLXFAST_CORRECTNESS_GOLDEN_PATH, or MLXFAST_SCORE_PATH"))
}

@Test
func benchmarkScriptFailsFastWithGuidanceWhenGoldenIsMissing() throws {
    // The official mode needs the private oracle. When it is absent the
    // script must exit BEFORE any build/transform work, with guidance pointing
    // at the local modes -- not let the Swift harness die minutes later on a
    // raw file-not-found error. Bare invocations default to --local-iterate.
    // Run from an empty temp cwd so the repo's own fixtures (or an operator's
    // local golden) cannot leak into the check.
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let benchmarkScript = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appendingPathComponent("benchmark.sh").path

    func runBare(_ arguments: [String], environment: [String: String] = [:]) throws -> (Int32, String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [benchmarkScript] + arguments
        process.currentDirectoryURL = root
        process.environment = benchmarkTestEnvironment(["MLXFAST_NO_SANDBOX": "1"])
            .merging(environment) { _, new in new }
        let stderrPipe = Pipe()
        process.standardError = stderrPipe
        try process.run()
        let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(data: stderrData, encoding: .utf8) ?? "")
    }

    // Explicit --official without the private oracle: fail fast + guide.
    let official = try runBare(["--official"])
    #expect(official.0 != 0)
    #expect(official.1.contains("correctness_golden.json is missing"))
    #expect(official.1.contains("--local-iterate"))
    #expect(official.1.contains("--local-submit"))
    // Failed before doing any work: no weights/ or score artifacts created.
    #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("score.json").path))

    // Bare invocation defaults to --local-iterate; from an empty cwd the
    // public fixture is absent, so it takes the re-sync branch (never the
    // official-oracle guidance).
    let bare = try runBare([])
    #expect(bare.0 != 0)
    #expect(bare.1.contains("correctness golden not found at"))
    #expect(!bare.1.contains("correctness_golden.json is missing"))

    // --official cannot be combined with local modes.
    let combined = try runBare(["--official", "--local-iterate"])
    #expect(combined.0 != 0)
    #expect(combined.1.contains("cannot be combined"))

    // Explicit override pointing at a missing file: name the path, different hint.
    let overridden = try runBare(
        [],
        environment: ["MLXFAST_CORRECTNESS_GOLDEN_PATH": root.appendingPathComponent("nope.json").path]
    )
    #expect(overridden.0 != 0)
    #expect(overridden.1.contains("correctness golden not found at"))
    #expect(overridden.1.contains("MLXFAST_CORRECTNESS_GOLDEN_PATH"))

    // Local mode from a directory without the public fixtures: re-sync hint.
    let localIterate = try runBare(["--local-iterate"])
    #expect(localIterate.0 != 0)
    #expect(localIterate.1.contains("correctness golden not found at"))
    #expect(localIterate.1.contains("correctness_prompts/"))
}

@Test
func benchmarkScriptBareInvocationDefaultsToLocalIterate() throws {
    // With the public fixtures present (repo-root cwd), a bare ./benchmark.sh
    // must resolve to local-iterate and forward that mode to the Swift binary.
    // The trusted-workflow env (MLXFAST_OFFICIAL_BENCHMARK_RUN=1) keeps its
    // official semantics without any flag, so CI needs no changes.
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let weights = root.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try "{}".write(to: weights.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

    let argLog = root.appendingPathComponent("args.txt")
    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    try """
    #!/bin/sh
    printf '%s\\n' "$@" > "\(argLog.path)"
    cat <<'JSON'
    {
      "score": null,
      "passed": true,
      "metrics": {
        "weights_hash": "fake-weights",
        "weights_file_count": 1,
        "weights_byte_count": 2
      }
    }
    JSON
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fakeSwift.path)

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["benchmark.sh"]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    process.environment = benchmarkTestEnvironment([
        "MLXFAST_NO_SANDBOX": "1",
        "MLXFAST_SKIP_TRANSFORM": "1",
        "MLXFAST_SWIFT_BIN": fakeSwift.path,
        "MLXFAST_WEIGHTS_PATH": weights.path,
        "MLXFAST_SCORE_PATH": root.appendingPathComponent("score.json").path,
        "MLXFAST_INTEGRITY_PATH": root.appendingPathComponent("integrity.json").path,
    ])
    let stdoutPipe = Pipe()
    process.standardOutput = stdoutPipe
    try process.run()
    let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    let stdout = String(data: stdoutData, encoding: .utf8) ?? ""

    #expect(process.terminationStatus == 0)
    #expect(stdout.contains("defaulting to --local-iterate"))
    let args = try String(contentsOf: argLog, encoding: .utf8)
    #expect(args.contains("--local-iterate\n"))
    #expect(!args.contains("--official"))
    #expect(args.contains("correctness_prompts/public_longcopy_gate_english_512_256.json\n"))
}

@Test
func benchmarkScriptFailsWhenScorePayloadFails() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let weights = root.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try "{}".write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    let golden = root.appendingPathComponent("correctness_golden.json")
    try "{}".write(to: golden, atomically: true, encoding: .utf8)

    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    // benchmark.sh reconstructs score.json from the trusted process stdout, so a
    // failing payload must be emitted there for the post-hoc "passed": false gate
    // to see it.
    try """
    #!/bin/sh
    cat <<'JSON'
    {
      "score": null,
      "passed": false,
      "metrics": {
        "weights_hash": "fake-weights",
        "weights_file_count": 1,
        "weights_byte_count": 2
      }
    }
    JSON
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: fakeSwift.path
    )

    let score = root.appendingPathComponent("score.json")
    let integrity = root.appendingPathComponent("benchmark-integrity.json")
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["benchmark.sh"]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    process.environment = benchmarkTestEnvironment([
        "MLXFAST_NO_SANDBOX": "1",
        "MLXFAST_SKIP_TRANSFORM": "1",
        "MLXFAST_SWIFT_BIN": fakeSwift.path,
        "MLXFAST_WEIGHTS_PATH": weights.path,
        "MLXFAST_CORRECTNESS_GOLDEN_PATH": golden.path,
        "MLXFAST_SCORE_PATH": score.path,
        "MLXFAST_INTEGRITY_PATH": integrity.path,
    ])

    try process.run()
    process.waitUntilExit()

    #expect(process.terminationStatus != 0)
    #expect(FileManager.default.fileExists(atPath: score.path))
    #expect(FileManager.default.fileExists(atPath: integrity.path))
}

@Test
func benchmarkScriptSealsScoreFromStdoutDiscardingOnDiskTamper() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let weights = root.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try "{}".write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    let golden = root.appendingPathComponent("correctness_golden.json")
    try "{}".write(to: golden, atomically: true, encoding: .utf8)

    // The fake writes a TAMPERED score to the on-disk --score-path (score: 999,
    // simulating an atexit rewrite by editable code running in the unsandboxed
    // benchmark process) but emits the TRUSTED payload (score: 1) on stdout, the
    // way the real emitScorePayloadToStdout does. benchmark.sh must rebuild
    // score.json from stdout AFTER the process exits, so the tamper is discarded.
    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    try """
    #!/bin/sh
    score_path="score.json"
    while [ "$#" -gt 0 ]; do
      if [ "$1" = "--score-path" ]; then
        shift
        score_path="$1"
      fi
      shift || exit 1
    done
    cat > "$score_path" <<'JSON'
    {
      "score": 999,
      "passed": true,
      "metrics": {
        "weights_hash": "tampered",
        "weights_file_count": 1,
        "weights_byte_count": 2
      }
    }
    JSON
    cat <<'JSON'
    {
      "score": 1,
      "passed": true,
      "metrics": {
        "weights_hash": "trusted",
        "weights_file_count": 1,
        "weights_byte_count": 2
      }
    }
    JSON
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: fakeSwift.path
    )

    let score = root.appendingPathComponent("score.json")
    let integrity = root.appendingPathComponent("benchmark-integrity.json")
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["benchmark.sh"]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    process.environment = benchmarkTestEnvironment([
        "MLXFAST_NO_SANDBOX": "1",
        "MLXFAST_SKIP_TRANSFORM": "1",
        "MLXFAST_SWIFT_BIN": fakeSwift.path,
        "MLXFAST_WEIGHTS_PATH": weights.path,
        "MLXFAST_CORRECTNESS_GOLDEN_PATH": golden.path,
        "MLXFAST_SCORE_PATH": score.path,
        "MLXFAST_INTEGRITY_PATH": integrity.path,
    ])

    try process.run()
    process.waitUntilExit()

    #expect(process.terminationStatus == 0)
    let sealed = try String(contentsOf: score, encoding: .utf8)
    #expect(sealed.contains("\"score\": 1"))
    #expect(sealed.contains("\"weights_hash\": \"trusted\""))
    #expect(!sealed.contains("999"))
    #expect(!sealed.contains("tampered"))
}

@Test
func benchmarkScriptRejectsMultipleScoreObjectsOnStdout() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let weights = root.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try "{}".write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )
    let golden = root.appendingPathComponent("correctness_golden.json")
    try "{}".write(to: golden, atomically: true, encoding: .utf8)

    // Two concatenated score objects on stdout model an injected extra write
    // trying to smuggle a second payload past the seal. benchmark.sh requires
    // exactly one object and must fail closed.
    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    try """
    #!/bin/sh
    cat <<'JSON'
    {"score": 1, "passed": true, "metrics": {"weights_hash": "a", "weights_file_count": 1, "weights_byte_count": 2}}
    {"score": 2, "passed": true, "metrics": {"weights_hash": "b", "weights_file_count": 1, "weights_byte_count": 2}}
    JSON
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: fakeSwift.path
    )

    let score = root.appendingPathComponent("score.json")
    let integrity = root.appendingPathComponent("benchmark-integrity.json")
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = ["benchmark.sh"]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    process.environment = benchmarkTestEnvironment([
        "MLXFAST_NO_SANDBOX": "1",
        "MLXFAST_SKIP_TRANSFORM": "1",
        "MLXFAST_SWIFT_BIN": fakeSwift.path,
        "MLXFAST_WEIGHTS_PATH": weights.path,
        "MLXFAST_CORRECTNESS_GOLDEN_PATH": golden.path,
        "MLXFAST_SCORE_PATH": score.path,
        "MLXFAST_INTEGRITY_PATH": integrity.path,
    ])

    let stderr = Pipe()
    process.standardError = stderr
    try process.run()
    process.waitUntilExit()

    let error = String(data: stderr.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
    #expect(process.terminationStatus != 0)
    #expect(error.contains("did not emit a single valid score payload on stdout"))
}

@Test
func benchmarkScriptFallsBackToCacheWhenReferenceSymlinkIsBroken() throws {
    // Regression: benchmark.sh used to hard-default its reference to the
    // reference_weights/ compatibility symlink. When that symlink is broken (or
    // points at a non-directory), the transform failed with ENOTDIR. The fix
    // resolves the reference like setup.sh does, so a broken symlink falls back to
    // the Hugging Face cache. Runs the real benchmark.sh with a fake swift binary
    // that records the --reference it was handed for the transform.
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    // benchmark.sh's transform-cache hash runs `git ls-files`; real usage is
    // always inside the repo, so make the temp working dir a git repo too.
    let gitInit = Process()
    gitInit.executableURL = URL(fileURLWithPath: "/usr/bin/git")
    gitInit.arguments = ["init", "-q", root.path]
    try gitInit.run()
    gitInit.waitUntilExit()

    // A broken compatibility symlink in the working directory.
    let refWeights = root.appendingPathComponent("reference_weights")
    try FileManager.default.createDirectory(at: refWeights, withIntermediateDirectories: true)
    try FileManager.default.createSymbolicLink(
        atPath: refWeights.appendingPathComponent("Qwen3.8-27B-4bit").path,
        withDestinationPath: root.appendingPathComponent("does-not-exist").path
    )

    // A real cache directory holding the checkpoint.
    let cache = root.appendingPathComponent("hfcache/models--EigenLabs--Qwen3.8-27B-4bit/snapshots/eda45ab47f465d08d6558f0353a2346e2eb9d5b3")
    try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
    try "{}".write(to: cache.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

    let weights = root.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    let golden = root.appendingPathComponent("golden.json")
    try "{}".write(to: golden, atomically: true, encoding: .utf8)

    let reflog = root.appendingPathComponent("transform-args.txt")
    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    try """
    #!/bin/sh
    cmd="$1"
    shift
    if [ "$cmd" = "transform" ]; then
      printf '%s\\n' "$@" >> "\(reflog.path)"
      out=""
      while [ "$#" -gt 0 ]; do
        if [ "$1" = "--output" ]; then shift; out="$1"; fi
        shift
      done
      if [ -n "$out" ]; then
        mkdir -p "$out"
        printf '{}' > "$out/config.json"
        printf '%s\n' '{"weight_map":{"tensor":"model.safetensors"}}' > "$out/model.safetensors.index.json"
        printf '\\100\\000\\000\\000\\000\\000\\000\\000' > "$out/model.safetensors"
        printf '%s' '{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}      ' >> "$out/model.safetensors"
        printf '\\001' >> "$out/model.safetensors"
      fi
      exit 0
    fi
    cat <<'JSON'
    {
      "score": 1,
      "passed": true,
      "metrics": {
        "weights_hash": "fake",
        "weights_file_count": 1,
        "weights_byte_count": 2
      }
    }
    JSON
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fakeSwift.path)

    let score = root.appendingPathComponent("score.json")
    let integrity = root.appendingPathComponent("benchmark-integrity.json")
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = [
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("benchmark.sh").path,
    ]
    // CWD is the temp dir so the broken reference_weights/ symlink is the one
    // benchmark.sh resolves; NO_SANDBOX keeps the transform out of run-offline.sh.
    process.currentDirectoryURL = root
    process.environment = benchmarkTestEnvironment([
        "MLXFAST_NO_SANDBOX": "1",
        "MLXFAST_SWIFT_BIN": fakeSwift.path,
        "MLXFAST_WEIGHTS_PATH": weights.path,
        "MLXFAST_REFERENCE_CACHE_DIR": cache.path,
        "MLXFAST_CORRECTNESS_GOLDEN_PATH": golden.path,
        "MLXFAST_SCORE_PATH": score.path,
        "MLXFAST_INTEGRITY_PATH": integrity.path,
    ])
    // Ensure the resolution path (not an override) is exercised.
    process.environment?.removeValue(forKey: "MLXFAST_REFERENCE_DIR")
    process.environment?.removeValue(forKey: "MLXFAST_SKIP_TRANSFORM")

    try process.run()
    process.waitUntilExit()

    #expect(process.terminationStatus == 0)
    let recorded = (try? String(contentsOf: reflog, encoding: .utf8)) ?? ""
    // The transform was handed the real cache dir, not the broken symlink.
    #expect(recorded.contains(cache.path))
    #expect(!recorded.contains("reference_weights/Qwen3.8-27B-4bit"))
}

@Test
func privateArtifactGuardRejectsRenamedGoldenAndPromptFiles() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let golden = root.appendingPathComponent("correctness_golden_512_2048.json")
    let prompts = root.appendingPathComponent("my_private_prompts.json")
    let gpqa = root.appendingPathComponent("gpqa_reference_cases.json")
    let goldenPromptText = root.appendingPathComponent("golden_prompt_benchmark_transcription_gate_english_512.txt")
    let goldenPromptJSON = root.appendingPathComponent("golden_prompt_benchmark_transcription_gate_english_512_256.json")
    try "{}".write(to: golden, atomically: true, encoding: .utf8)
    try "{}".write(to: prompts, atomically: true, encoding: .utf8)
    try "{}".write(to: gpqa, atomically: true, encoding: .utf8)
    try "hidden prompt".write(to: goldenPromptText, atomically: true, encoding: .utf8)
    try "{}".write(to: goldenPromptJSON, atomically: true, encoding: .utf8)

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/bash")
    process.arguments = [
        ".github/scripts/deny-private-artifacts.sh",
        golden.path,
        prompts.path,
        gpqa.path,
        goldenPromptText.path,
        goldenPromptJSON.path,
    ]
    process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    process.environment = ProcessInfo.processInfo.environment.merging([
        "MLXFAST_GITHUB_ANNOTATIONS": "0",
    ]) { _, new in new }

    try process.run()
    process.waitUntilExit()

    #expect(process.terminationStatus != 0)
}



// Execute the real overlay script against sealed-score fixtures to prove the
// commit binding end to end: a candidate timing score stamped with the
// trusted dispatch SHA (full or short prefix) merges, while empty, wrong,
// uppercase, or too-short commits keep failing the pre-merge checks. This is
// the regression test for the ranked-scoring outage where the harness's
// git-in-sandbox commit came back empty and every ranked run was rejected.
@Test
func overlayPairedTimingAcceptsTrustedCommitAndRejectsForgedOrMissingCommit() throws {
    let expectedCommit = "5f95c4bdce07a0ef79ea350c91d9eb0d7476cf2f"
    // The overlay's pre-merge checks require the candidate timing score to
    // repeat the gates score's harness/weights identity, so both fixtures
    // carry matching values (mirrors runPairedTimingOverlay in
    // BenchmarkSafetyTests).
    let harnessHash = String(repeating: "a", count: 64)
    let weightsHash = String(repeating: "b", count: 64)

    func runOverlay(candidateCommit: String) throws -> (status: Int32, stderr: String, merged: String) {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let gatesScore = root.appendingPathComponent("score.json")
        let candidateScore = root.appendingPathComponent("score-candidate.json")
        let results = root.appendingPathComponent("results.json")
        let integrity = root.appendingPathComponent("benchmark-integrity.json")

        try """
        {
          "passed": true,
          "score": null,
          "metrics": {
            "commit": "\(expectedCommit)",
            "error": "",
            "partial_result": true,
            "benchmark_wall_seconds": 2400.0,
            "peak_ram_gb": 21.5,
            "process_resident_memory_gb": 20.5,
            "expert_bytes_read": 0,
            "expert_cache_hits": 0,
            "expert_cache_misses": 0,
            "expert_cache_evictions": 0,
            "expert_read_seconds": 0,
            "expert_peak_cached_tensors": 0,
            "harness_hash": "\(harnessHash)",
            "weights_hash": "\(weightsHash)",
            "weights_file_count": 4,
            "weights_byte_count": 17000000000
          }
        }
        """.write(to: gatesScore, atomically: true, encoding: .utf8)

        try """
        {
          "passed": false,
          "score": null,
          "metrics": {
            "commit": "\(candidateCommit)",
            "error": "performance floor failed (decode_speedup=0.5)",
            "first_failing_case": null,
            "first_failing_step": null,
            "expected_token": null,
            "actual_token": null,
            "expert_bytes_read": 0,
            "expert_cache_hits": 0,
            "expert_cache_misses": 0,
            "expert_cache_evictions": 0,
            "expert_read_seconds": 0,
            "expert_peak_cached_tensors": 0,
            "bandwidth_source": "ram_resident_model",
            "bandwidth_gb_per_token": 0,
            "timed_benchmark_seconds": 42.5,
            "benchmark_wall_seconds": 300.0,
            "peak_ram_gb": 22.0,
            "process_resident_memory_gb": 21.0,
            "harness_hash": "\(harnessHash)",
            "weights_hash": "\(weightsHash)",
            "weights_file_count": 4,
            "weights_byte_count": 17000000000
          }
        }
        """.write(to: candidateScore, atomically: true, encoding: .utf8)

        try """
        {
          "mode": "paired",
          "paired": {
            "decode_speedup": 1.0071896113655208,
            "prefill_speedup": 1.0032593423660188,
            "paired_score": 1.0062056030137094
          },
          "candidate": {
            "decode_seconds_per_token": 0.0438,
            "prefill_seconds_per_token": 0.00162,
            "verdict": "ACCEPT"
          },
          "baseline": {
            "decode_seconds_per_token": 0.0441,
            "prefill_seconds_per_token": 0.001625,
            "verdict": "ACCEPT"
          }
        }
        """.write(to: results, atomically: true, encoding: .utf8)

        try "{\"score_sha256\": \"stale\"}".write(to: integrity, atomically: true, encoding: .utf8)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [".github/scripts/overlay-paired-timing.sh"]
        process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        process.environment = ProcessInfo.processInfo.environment.merging([
            "MLXFAST_SCORE_PATH": gatesScore.path,
            "MLXFAST_MEASURE_RESULTS_PATH": results.path,
            "MLXFAST_CANDIDATE_SCORE_PATH": candidateScore.path,
            "MLXFAST_INTEGRITY_PATH": integrity.path,
            "MLXFAST_DECODE_SPEEDUP_FLOOR": "0.95",
            "MLXFAST_PREFILL_SPEEDUP_FLOOR": "0.95",
            "MLXFAST_EXPECTED_COMMIT": expectedCommit,
        ]) { _, new in new }
        let stderrPipe = Pipe()
        process.standardError = stderrPipe
        process.standardOutput = Pipe()
        try process.run()
        process.waitUntilExit()
        let stderrText = String(
            data: stderrPipe.fileHandleForReading.readDataToEndOfFile(),
            encoding: .utf8
        ) ?? ""
        let merged = (try? String(contentsOf: gatesScore, encoding: .utf8)) ?? ""
        return (process.terminationStatus, stderrText, merged)
    }

    // Trusted full dispatch SHA: merges, applies the paired score, keeps the
    // gates-score commit in the published result.
    let full = try runOverlay(candidateCommit: expectedCommit)
    #expect(full.status == 0, "expected trusted full-SHA commit to merge: \(full.stderr)")
    #expect(full.merged.contains("\"passed\": true"))
    #expect(full.merged.contains("\"commit\": \"\(expectedCommit)\""))
    #expect(full.merged.contains("\"decode_speedup\": 1.0071896113655208"))
    #expect(full.merged.contains("\"partial_result\": false"))

    // A short prefix of the dispatched SHA (git rev-parse --short form from
    // local/dev paths) is still accepted.
    let prefix = try runOverlay(candidateCommit: String(expectedCommit.prefix(12)))
    #expect(prefix.status == 0, "expected short-prefix commit to merge: \(prefix.stderr)")

    // The regression shape: git failing inside the bench sandbox produced an
    // empty commit; that (and any forged/wrong commit) must keep failing.
    for rejected in [
        "",
        "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        expectedCommit.uppercased(),
        "5f95c4",
    ] {
        let outcome = try runOverlay(candidateCommit: rejected)
        #expect(outcome.status != 0, "expected commit \"\(rejected)\" to be rejected")
        #expect(outcome.stderr.contains("candidate timing score failed the pre-merge checks"))
    }
}


@Test
func benchmarkWorkflowFailsClosedWhenRunnerPrivateArtifactCleanupFails() throws {
    let workflow = try String(
        contentsOfFile: ".github/workflows/dflash-benchmark.yml",
        encoding: .utf8
    )
    let cleanupRange = try #require(workflow.range(of: "- name: Cleanup bench workspace"))
    let janitorRange = try #require(workflow.range(of: "- name: Janitor reset and integrity audit"))
    let cleanup = String(workflow[cleanupRange.lowerBound..<janitorRange.lowerBound])

    // Bench-owned residue remains best-effort because the runner uid may not be
    // able to unlink it; runner-owned private roots must be removed and checked.
    #expect(cleanup.contains("find . -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +"))
    #expect(!cleanup.contains("rm -rf ./* ./.??*"))
    #expect(cleanup.contains("rm -rf \"${MLXFAST_JOB_WS}\" || true"))
    #expect(cleanup.contains("for private_path in \\"))
    #expect(cleanup.contains("\"${MLXFAST_PRIVATE_DIR}\" \\"))
    #expect(cleanup.contains("\"${MLXFAST_MEASURE_OUT}\" \\"))
    #expect(cleanup.contains("\"${MLXFAST_ARTIFACT_ROOT}\"; do"))
    #expect(cleanup.contains("private_cleanup_failed=0"))
    #expect(cleanup.contains("if ! rm -rf -- \"${private_path}\"; then"))
    #expect(cleanup.contains("could not remove runner-private artifact path"))
    #expect(cleanup.contains("[[ -e \"${private_path}\" || -L \"${private_path}\" ]]"))
    #expect(cleanup.contains("runner-private artifact cleanup failed"))
    #expect(cleanup.components(separatedBy: "private_cleanup_failed=1").count - 1 == 2)
    #expect(cleanup.contains("exit \"${private_cleanup_failed}\""))
    #expect(!cleanup.contains("rm -rf \"${MLXFAST_PRIVATE_DIR}\" \"${MLXFAST_MEASURE_OUT}\" \"${MLXFAST_ARTIFACT_ROOT}\" || true"))
}



@Test
func staticReviewFailsClosedOnSelfContradictoryPassedTrueWithHighSeverity() throws {
    let staticReview = try String(
        contentsOfFile: ".github/scripts/run-submission-static-review.sh",
        encoding: .utf8
    )

    // The judge's own system prompt instructs it to set passed=false for high/
    // critical severity, but that's policy text sent to the LLM, not something
    // the schema check enforced -- a prompt-injection-influenced response could
    // return a schema-valid but self-contradictory {passed:true, severity:
    // critical, findings:[...]} and previously sailed through the passed-only
    // gate. Confirm both the tightened schema (severity must be one of the five
    // enumerated values, not just typed as a string) and the explicit fail-
    // closed cross-check are present.
    #expect(staticReview.contains("and (.severity | IN(\"none\", \"low\", \"medium\", \"high\", \"critical\"))"))
    let crossCheckRange = try #require(staticReview.range(
        of: "if [[ \"${passed}\" == \"true\" ]] && { [[ \"${severity}\" == \"high\" ]] || [[ \"${severity}\" == \"critical\" ]]; }; then"
    ))
    let finalGateRange = try #require(
        staticReview.range(of: "if [[ \"${passed}\" != \"true\" ]]; then", range: crossCheckRange.lowerBound..<staticReview.endIndex)
    )
    #expect(crossCheckRange.lowerBound < finalGateRange.lowerBound)
    let crossCheckBlock = String(staticReview[crossCheckRange.lowerBound..<finalGateRange.lowerBound])
    #expect(crossCheckBlock.contains("passed=\"false\""))
}

// QwenRuntime was split across QwenRuntime*.swift; concatenate them so
// source-level assertions stay agnostic to which split file the code lives in.
private func harnessRuntimeSource() throws -> String {
    let directory = "Sources/MLXFastHarness"
    let files = try FileManager.default.contentsOfDirectory(atPath: directory)
        .filter { $0.hasPrefix("QwenRuntime") && $0.hasSuffix(".swift") }
        .sorted()
    return try files
        .map { try String(contentsOfFile: "\(directory)/\($0)", encoding: .utf8) }
        .joined(separator: "\n")
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        "mlxfast-benchmark-script-tests-\(UUID().uuidString)"
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private let benchmarkTestMetallibPath: String = {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        "mlxfast-benchmark-script-tests-\(ProcessInfo.processInfo.processIdentifier).metallib"
    )
    _ = FileManager.default.createFile(
        atPath: url.path,
        contents: Data("fixture metallib".utf8)
    )
    return url.path
}()

private func benchmarkTestEnvironment(
    _ overrides: [String: String] = [:]
) -> [String: String] {
    var environment = ProcessInfo.processInfo.environment
    for key in environment.keys.filter({ $0.hasPrefix("MLXFAST_") }) {
        environment.removeValue(forKey: key)
    }
    environment["MLXFAST_GPU_TEMP_CMD"] = "printf 39"
    environment["MLXFAST_MLX_METALLIB"] = benchmarkTestMetallibPath
    // Tests run in parallel and would contend on the real per-user local run
    // lock (and a genuinely resident model on the host would abort unrelated
    // tests), so the local memory guard defaults OFF here; guard-specific
    // tests re-enable it with their own lock dir / scan seam.
    environment["MLXFAST_LOCAL_RUN_GUARD"] = "0"
    var merged = environment.merging(overrides) { _, new in new }
    // Fake trusted CLIs in shell tests do not spawn a model process. Point the
    // compatibility override at that executable so benchmark.sh's paired
    // product readiness check does not attempt a real Swift build.
    if merged["MLXFAST_RUNTIME_WORKER_EXECUTABLE"] == nil,
       let swiftBinary = merged["MLXFAST_SWIFT_BIN"]
    {
        merged["MLXFAST_RUNTIME_WORKER_EXECUTABLE"] = swiftBinary
    }
    return merged
}

// MARK: - The local DFlash runner must speak the CLI that exists

/// `benchmark-dflash.sh` was restored from the retired MTP scaffolding and only
/// partially retargeted: it invoked two subcommands that no longer exist
/// (`mtp-probe`, `mtp-benchmark`), passed five flags the CLI never declared,
/// defaulted its contract to a deleted fixture and stamped the retired MTP track
/// id into the score it wrote. None of that is visible until someone runs it,
/// and every one of those dead references survived review — which is the whole
/// argument for applying grep discipline mechanically instead of by eye.
@Suite
struct LocalDFlashScriptSurfaceTests {
    private typealias S = DFlashGateTextSupport

    private static let scripts = ["benchmark-dflash.sh", "setup-dflash.sh"]

    @Test
    func localDFlashScriptsNameNoRetiredMTPSurface() throws {
        for path in Self.scripts {
            let script = try S.text(path)
            for retired in S.retiredMTPNames {
                #expect(
                    !script.contains(retired),
                    """
                    \(path) references the retired MTP name '\(retired)'. The MTP \
                    track was retired without going live; every one of these is \
                    either a subcommand that does not exist, a file that was \
                    deleted, or a track id nothing will score.
                    """
                )
            }
        }
    }

    @Test
    func localDFlashScriptsInvokeOnlySubcommandsAndFlagsTheCLIDeclares() throws {
        let declared = try S.cliSubcommandOptions()
        // Guard the extractor: if the dispatch parse breaks, the loop below
        // would pass vacuously.
        #expect(declared["dflash-benchmark"]?.contains("--drafter") == true,
                "CLI option extraction looks broken: \(declared)")
        #expect(declared["correctness"] != nil)
        #expect(declared.count >= 15, "CLI dispatch extraction found only \(declared.count) subcommands")

        for path in Self.scripts {
            let script = try S.text(path)
            for (subcommand, command) in S.swiftBinaryInvocations(in: script) {
                let allowed = declared[subcommand]
                #expect(
                    allowed != nil,
                    """
                    \(path) invokes `mlxfast-swift \(subcommand)`, which is not a \
                    case in the CLI dispatch switch — the run aborts with \
                    "unknown command". Declared: \
                    \(declared.keys.sorted().joined(separator: ", "))
                    """
                )
                guard let allowed else { continue }
                for flag in Set(S.captures(#"(\s|^)(--[a-z][a-z0-9-]*)"#, in: command, group: 2)) {
                    #expect(
                        allowed.contains(flag),
                        """
                        \(path) passes \(flag) to `mlxfast-swift \(subcommand)`, \
                        which does not declare it — ParsedOptions.validate \
                        rejects the invocation before any work happens. \
                        Declared for \(subcommand): \
                        \(allowed.sorted().joined(separator: " "))
                        """
                    )
                }
            }
        }
    }

    // The local DFlash path must REUSE the existing correctness gate and the
    // existing cool-down entrypoint rather than growing its own: `correctness`
    // against the checked-in public fixture is the same directional check the
    // serial local path runs, and `benchmark.sh --local-cool-gate-only` is the
    // one implementation of the 40C gate.
    @Test
    func localDFlashRunnerReusesTheCorrectnessGateAndTheCoolDownEntrypoint() throws {
        let script = try S.text("benchmark-dflash.sh")

        let invocations = S.swiftBinaryInvocations(in: script)
        let correctness = invocations.first { $0.0 == "correctness" }
        let correctnessCommand = try #require(
            correctness?.1,
            """
            benchmark-dflash.sh never runs the existing `correctness` \
            subcommand, so a local DFlash iteration has no correctness signal \
            at all. Call the gate that already exists.
            """
        )
        let goldenToken = try #require(
            S.captures(#"--golden\s+(\S+)"#, in: correctnessCommand).first,
            "benchmark-dflash.sh runs `correctness` without a --golden"
        )
        let golden = S.resolveShellValue(goldenToken, in: script)
        #expect(
            golden.contains("correctness_prompts/public_longcopy_gate_english"),
            """
            benchmark-dflash.sh must run `correctness` against the checked-in \
            public fixture (resolved --golden: \(golden)): it is the only golden \
            a local run legitimately has.
            """
        )

        #expect(
            script.contains("--local-cool-gate-only"),
            """
            benchmark-dflash.sh must gate its timed measurement through \
            `./benchmark.sh --local-cool-gate-only` — the single existing \
            implementation of the 40C cool-down. A second cool-gate is a second \
            calibration.
            """
        )
        #expect(script.contains("benchmark.sh"))
    }
}

// The RAM-resident model guard has to SEE a DFlash residency. A DFlash local run
// holds the ~21.6 GB target plus the drafter, so an overlapping serial run is an
// out-of-memory, not a slowdown — but the guard's process pattern only listed
// serial subcommands, so `dflash-benchmark` / `dflash-probe` / `dflash-reference`
// were invisible to it.
@Test
func residentModelGuardPatternCoversTheDFlashSubcommands() throws {
    let script = try String(contentsOfFile: "benchmark.sh", encoding: .utf8)
    let declaration = try #require(
        script.range(of: "readonly RESIDENT_MODEL_PROCESS_PATTERN="),
        "benchmark.sh lost RESIDENT_MODEL_PROCESS_PATTERN"
    )
    let line = String(
        script[declaration.lowerBound...]
            .prefix(while: { $0 != "\n" })
    )
    // The pre-existing serial coverage must survive the widening.
    #expect(line.contains("runtime-worker[[:space:]]+--weights"))
    #expect(line.contains("correctness"))
    for subcommand in ["dflash-benchmark", "dflash-probe", "dflash-reference"] {
        #expect(
            line.contains(subcommand),
            """
            RESIDENT_MODEL_PROCESS_PATTERN does not match `\(subcommand)`, so the \
            local run guard cannot see a DFlash model residency. Two \
            simultaneous residencies (~21.6 GB each, plus the drafter) exceed \
            the documented 36 GiB local minimum. Pattern: \(line)
            """
        )
    }
}

// MARK: - The transform cache must be keyed on WHICH model it transformed

/// benchmark.sh's reference defaults were still the retired Poolside Laguna
/// checkpoint long after setup.sh, `MLXFastConstants` and the track contract
/// had moved to the Qwen 3.8 27B target. The three values below are one
/// identity spelled in four places; these assertions hold benchmark.sh's copy
/// against the other three, and `resolve_reference_path` is where they now
/// live (a single self-contained resolver, so the top-level REFERENCE_PATH and
/// the transform-cache key cannot drift apart).
@Test
func benchmarkScriptResolvesThePinnedQwenReference() throws {
    let script = try String(contentsOfFile: "benchmark.sh", encoding: .utf8)
    #expect(
        script.contains(
            #"reference_repo="${MLXFAST_REFERENCE_MODEL_REPO:-EigenLabs/Qwen3.8-27B-4bit}""#),
        """
        benchmark.sh does not default to the pinned Qwen 3.8 27B reference \
        repository. A stale default here transforms (and caches) a different \
        model than setup.sh downloaded.
        """
    )
    #expect(
        script.contains(
            #"reference_revision="${MLXFAST_REFERENCE_REVISION:-eda45ab47f465d08d6558f0353a2346e2eb9d5b3}""#))
    #expect(script.contains(#"reference_default_dir="reference_weights/Qwen3.8-27B-4bit""#))
    // The compatibility symlink setup.sh creates, and MLXFastConstants'
    // defaultReferencePath, are that same directory.
    #expect(MLXFastConstants.defaultReferencePath == "reference_weights/Qwen3.8-27B-4bit")
    #expect(MLXFastConstants.referenceModelRepository == "EigenLabs/Qwen3.8-27B-4bit")
    #expect(
        MLXFastConstants.referenceModelRevision == "eda45ab47f465d08d6558f0353a2346e2eb9d5b3")
    // One resolver, one caller: the top-level path must be the function's
    // output, not a second copy of the same if/elif chain.
    #expect(script.contains("REFERENCE_PATH=\"$(resolve_reference_path)\""))
}

/// The sibling track runners reuse benchmark.sh's definitions by extracting
/// them with `awk '/^name\(\) \{/,/^\}/'`. That only works while the name sits
/// at column 0 with `() {`, the closing `}` sits at column 0, and no column-0
/// `}` appears inside the body (the range ends at the FIRST one). Since
/// `source_hash` now calls `resolve_reference_path`, a runner that extracts
/// only `source_hash` aborts on an undefined function -- so both scripts must
/// pull both names.
@Test
func theTransformCacheDefinitionsStayExtractable() throws {
    let script = try String(contentsOfFile: "benchmark.sh", encoding: .utf8)
    for name in ["resolve_reference_path", "source_hash", "config_model_family"] {
        let start = try #require(
            script.range(of: "\n\(name)() {\n"),
            "benchmark.sh lost the column-0 `\(name)() {` the runners extract"
        )
        let body = script[start.upperBound...]
        let end = try #require(
            body.range(of: "\n}\n"),
            "\(name)() has no column-0 closing brace"
        )
        // What awk hands the runner: everything up to the FIRST column-0 `}`.
        // If a column-0 `}` ever appears inside the body, that slice stops
        // early -- so require each function's own last statement to still be
        // inside it.
        let extracted = String(body[..<end.lowerBound])
        let tail: String
        switch name {
        case "resolve_reference_path":
            tail = #"printf '%s\n' "${reference_default_dir}""#
        case "source_hash":
            tail = "} | shasum -a 256 | awk '{print $1}'"
        default:
            tail = #"printf '%s\n' "${family}""#
        }
        #expect(
            extracted.contains(tail),
            """
            the awk extraction of \(name)() stops before its last statement, so \
            a column-0 `}` has appeared inside the body and the track runners \
            would evaluate a truncated function.
            """
        )
    }
    // The sanity string both runners check the extraction for.
    #expect(script.contains("shasum -a 256"))

    for runner in ["benchmark-qwen-mtp.sh", "benchmark-dflash.sh"] {
        let body = try String(contentsOfFile: runner, encoding: .utf8)
        for name in ["resolve_reference_path", "source_hash"] {
            #expect(
                body.contains(#"awk '/^\#(name)\(\) \{/,/^\}/' benchmark.sh"#),
                """
                \(runner) does not extract benchmark.sh's \(name)(). \
                source_hash() calls resolve_reference_path(), so extracting \
                one without the other aborts the runner on an undefined \
                function -- or, worse, computes a different digest and reports \
                a permanent false "stale weights".
                """
            )
            #expect(
                body.contains("could not reuse benchmark.sh's ${reused_definition}()"),
                "\(runner) does not fail closed per extracted name")
        }
    }
    // The Qwen runner additionally reuses the family normalization.
    let qwen = try String(contentsOfFile: "benchmark-qwen-mtp.sh", encoding: .utf8)
    #expect(qwen.contains(#"awk '/^config_model_family\(\) \{/,/^\}/' benchmark.sh"#))
}

/// P0.5: the transform-cache key covered the transform CODE and nothing about
/// the transform INPUT, so a `weights/` tree produced from the retired Laguna
/// checkpoint matched the Qwen track's key exactly and was reused. The run then
/// hashed ~15 GB of the wrong model and died in the worker with a null score.
@Test
func theTransformCacheKeyCoversTheReferenceIdentity() throws {
    let script = try String(contentsOfFile: "benchmark.sh", encoding: .utf8)
    let start = try #require(script.range(of: "\nsource_hash() {\n"))
    let body = String(script[start.upperBound...])
    let end = try #require(body.range(of: "\n}\n"))
    let digestBody = String(body[..<end.lowerBound])

    #expect(digestBody.contains(#"printf 'reference-repository\0%s\0' "${reference_repo}""#))
    #expect(digestBody.contains(#"printf 'reference-revision\0%s\0' "${reference_revision}""#))
    #expect(
        digestBody.contains(
            #"printf 'reference-config-sha256\0%s\0' "${reference_config_digest}""#),
        """
        source_hash() no longer folds the reference config.json digest into the \
        key, so a weights/ tree transformed from a different checkpoint can pass \
        as fresh again.
        """
    )
    #expect(digestBody.contains("reference_dir=\"$(resolve_reference_path)\""))
    // Absent reference and empty-config reference must not collide.
    #expect(digestBody.contains(#"reference_config_digest="MISSING""#))
    // The code stream keeps BOTH branches.
    #expect(digestBody.contains("git ls-files --cached --others --exclude-standard -z"))
    #expect(digestBody.contains(#"find "${paths[@]}" -type f"#))
}

/// P0.3: setup.sh provisions the reference checkpoint and never transforms it,
/// so `./setup.sh && ./setup-qwen-mtp.sh` followed by the track runner reached
/// an empty `weights/` on every fresh machine. `--transform-only` is the step
/// that fills it: everything up to and including the transform and its stamp,
/// then exit 0 BEFORE any measurement.
@Test
func benchmarkScriptTransformOnlyStopsBeforeAnyMeasurement() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let gitInit = Process()
    gitInit.executableURL = URL(fileURLWithPath: "/usr/bin/git")
    gitInit.arguments = ["init", "-q", root.path]
    try gitInit.run()
    gitInit.waitUntilExit()

    let reference = root.appendingPathComponent("reference")
    try FileManager.default.createDirectory(at: reference, withIntermediateDirectories: true)
    try #"{"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text"}}"#
        .write(to: reference.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

    let weights = root.appendingPathComponent("weights")
    let golden = root.appendingPathComponent("golden.json")
    try "{}".write(to: golden, atomically: true, encoding: .utf8)

    let calls = root.appendingPathComponent("calls.txt")
    let fakeSwift = root.appendingPathComponent("mlxfast-swift")
    try """
    #!/bin/sh
    printf '%s\\n' "$1" >> "\(calls.path)"
    cmd="$1"
    shift
    if [ "$cmd" = "transform" ]; then
      out=""
      while [ "$#" -gt 0 ]; do
        if [ "$1" = "--output" ]; then shift; out="$1"; fi
        shift
      done
      if [ -n "$out" ]; then
        mkdir -p "$out"
        printf '%s' '{"model_type":"qwen3_5_text","num_hidden_layers":64,"vocab_size":248320,"hidden_size":5120}' > "$out/config.json"
        printf '%s\\n' '{"weight_map":{"tensor":"model.safetensors"}}' > "$out/model.safetensors.index.json"
        printf '\\100\\000\\000\\000\\000\\000\\000\\000' > "$out/model.safetensors"
        printf '%s' '{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}      ' >> "$out/model.safetensors"
        printf '\\001' >> "$out/model.safetensors"
      fi
      exit 0
    fi
    exit 9
    """.write(to: fakeSwift, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755], ofItemAtPath: fakeSwift.path)

    let score = root.appendingPathComponent("score.json")
    func runTransformOnly() throws -> (status: Int32, output: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("benchmark.sh").path,
            "--transform-only",
        ]
        process.currentDirectoryURL = root
        process.environment = benchmarkTestEnvironment([
            "MLXFAST_NO_SANDBOX": "1",
            "MLXFAST_SWIFT_BIN": fakeSwift.path,
            "MLXFAST_WEIGHTS_PATH": weights.path,
            "MLXFAST_REFERENCE_DIR": reference.path,
            "MLXFAST_CORRECTNESS_GOLDEN_PATH": golden.path,
            "MLXFAST_SCORE_PATH": score.path,
            "MLXFAST_INTEGRITY_PATH": root.appendingPathComponent("integrity.json").path,
        ])
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let output = String(
            decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        process.waitUntilExit()
        return (process.terminationStatus, output)
    }

    let first = try runTransformOnly()
    #expect(first.status == 0, "--transform-only failed: \(first.output)")
    #expect(first.output.contains("regenerating weights with Swift transform"))
    #expect(first.output.contains("--transform-only complete"))
    #expect(first.output.contains("no measurement was run"))
    #expect(
        FileManager.default.fileExists(
            atPath: weights.appendingPathComponent(".benchmark-source.sha256").path),
        "--transform-only did not stamp the transform-source digest")
    #expect(
        !FileManager.default.fileExists(atPath: score.path),
        "--transform-only wrote a score; it must exit before every measured phase")
    let invoked = (try? String(contentsOf: calls, encoding: .utf8)) ?? ""
    #expect(
        !invoked.contains("benchmark"),
        """
        --transform-only invoked `mlxfast-swift benchmark`. It exists precisely \
        so a caller that only needs weights/ does not pay for a measurement.
        """
    )

    // Already fresh: the reuse path, still exit 0, still no measurement.
    let second = try runTransformOnly()
    #expect(second.status == 0, "--transform-only failed on the reuse path: \(second.output)")
    #expect(second.output.contains("reusing \(weights.path)/ for unchanged transform source"))
    #expect(second.output.contains("--transform-only complete"))

    // Defense in depth: a cached tree whose stamp still matches but whose
    // config declares a different model family is stale, not fresh.
    try #"{"model_type":"laguna","num_hidden_layers":48}"#
        .write(
            to: weights.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
    let third = try runTransformOnly()
    #expect(third.status == 0, "--transform-only failed after the family swap: \(third.output)")
    #expect(
        third.output.contains("declares model family laguna"),
        """
        benchmark.sh reused a weights/ tree declaring a different model family \
        than the reference it would transform. The digest catches a stale tree; \
        this check is what catches a stamp collision written by an older, \
        code-only cache key. Output: \(third.output)
        """
    )
    #expect(third.output.contains("treating the cached weights as stale"))
    #expect(third.output.contains("regenerating weights with Swift transform"))
}

/// `--transform-only` is a MODE, not a modifier: pairing it with anything that
/// measures would silently drop the measurement the caller asked for.
@Test
func transformOnlyRefusesEveryMeasuringCombination() throws {
    let script = try String(contentsOfFile: "benchmark.sh", encoding: .utf8)
    // Consumed in the arg-parse case like --official, never forwarded to the
    // Swift CLI (which does not declare it) ...
    let argCase = try #require(script.range(of: "    --transform-only)\n"))
    let caseBody = String(script[argCase.upperBound...].prefix(900))
    let caseEnd = try #require(caseBody.range(of: "\n      ;;"))
    let consumed = String(caseBody[..<caseEnd.lowerBound])
    #expect(consumed.contains("TRANSFORM_ONLY=1"))
    #expect(
        consumed.contains("continue"),
        """
        --transform-only is no longer consumed with `continue`, so it would be \
        appended to FORWARD_ARGS and handed to `mlxfast-swift benchmark`, which \
        does not declare it.
        """
    )
    // ... but re-added by name on the legacy sandbox re-exec, or the child
    // would run a full local benchmark instead of a transform.
    #expect(script.contains("RESOLVED_ARGS+=(\"--transform-only\")"))
    for conflicting in ["--official", "--local-submit", "--local-cool-gate-only"] {
        #expect(
            script.contains("benchmark.sh: --transform-only cannot be combined with \(conflicting)"),
            "benchmark.sh does not refuse --transform-only \(conflicting) by name")
    }
    // It is a LOCAL run for every downstream purpose: the run lock and the
    // resident-model scan must cover a step that rewrites weights/ underneath
    // whatever else is running. LOCAL_ITERATE is raised AFTER the refusals
    // above, so `--transform-only --official` reports the conflict it has
    // rather than one for a flag the caller never typed.
    let refusal = try #require(
        script.range(of: "benchmark.sh: --transform-only cannot be combined with --local-cool-gate-only"))
    let afterRefusals = String(script[refusal.upperBound...].prefix(600))
    #expect(
        afterRefusals.contains("  LOCAL_ITERATE=1\n"),
        "--transform-only no longer joins the local-mode branch that takes the run lock")
}

/// P0.3/P1.1 on the runner side: the Qwen MTP runner must refresh the transform
/// itself (the manifest's benchmarkCommand has no manual step in front of it)
/// and must reject a wrong-model `weights/` tree BEFORE the drift tripwire,
/// any weight hashing, or a worker start.
@Test
func theQwenMTPRunnerRefreshesTheTransformAndGatesTheModelFamily() throws {
    let runner = try String(contentsOfFile: "benchmark-qwen-mtp.sh", encoding: .utf8)

    #expect(
        runner.contains(#"transform_command=("./benchmark.sh" "--transform-only")"#),
        """
        benchmark-qwen-mtp.sh no longer refreshes the transform itself. \
        benchmark.json's benchmarkCommand runs this script straight after \
        setup, and setup never creates weights/, so an abort here is a manual \
        step the published sequence does not contain.
        """
    )
    #expect(
        !runner.contains("Produce (or refresh) it first"),
        "benchmark-qwen-mtp.sh still tells the reader to produce weights/ by hand")
    #expect(
        !runner.contains("./benchmark.sh --local-iterate\n"),
        """
        benchmark-qwen-mtp.sh still points at a full local benchmark for its \
        side effect; --transform-only is the step that only transforms.
        """
    )
    // The refresh happens before this script takes its own lock, and
    // --transform-only takes benchmark.sh's lock in its own process.
    #expect(runner.contains("NO DEADLOCK, and the ordering is the reason"))
    let refreshIndex = try #require(runner.range(of: "transform_command[*]}"))
    let lockIndex = try #require(runner.range(of: "\nacquire_local_run_lock\n"))
    #expect(
        refreshIndex.lowerBound < lockIndex.lowerBound,
        """
        the transform refresh now runs after acquire_local_run_lock, so \
        ./benchmark.sh --transform-only would block forever on the per-user \
        lock this script already holds.
        """
    )
    // Still fails closed when the refresh did not help, and says with what.
    #expect(runner.contains("is STILL not usable after"))
    #expect(runner.contains("expected transform-source digest ${wanted_source_hash}"))

    // The track gate: family first, then the pinned geometry, both before the
    // tripwire.
    #expect(runner.contains(#"weights_model_family="$(config_model_family "${weights_path}/config.json")""#))
    #expect(runner.contains(#"if [[ "${weights_model_family}" != "qwen3_5" ]]; then"#))
    #expect(runner.contains("is not this track's target"))
    #expect(runner.contains("num_hidden_layers=\\(.num_hidden_layers) (expected 64)"))
    #expect(runner.contains("vocab_size=\\(.vocab_size) (expected 248320)"))
    #expect(runner.contains("hidden_size=\\(.hidden_size) (expected 5120)"))
    #expect(runner.contains("MLXFAST_FORCE_TRANSFORM=1 ${transform_command[*]}"))
    let gateIndex = try #require(runner.range(of: "is not this track's target"))
    let tripwireIndex = try #require(runner.range(of: "public drift tripwire (correctness against"))
    #expect(
        gateIndex.lowerBound < tripwireIndex.lowerBound,
        """
        the wrong-model gate now runs after the drift tripwire. Its whole point \
        is to abort before ~15 GB is hashed and a worker is started against a \
        model that is not this track's target.
        """
    )
    // The pinned geometry is the contract fixture's, not a second opinion.
    #expect(MLXFastConstants.numHiddenLayers == 64)
    #expect(MLXFastConstants.vocabSize == 248_320)
    #expect(MLXFastConstants.hiddenSize == 5_120)
}
