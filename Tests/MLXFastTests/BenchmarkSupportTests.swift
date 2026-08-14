import Darwin
import CryptoKit
import Foundation
@testable import MLXFastCore
@testable import MLXFastModel
@testable import MLXFastRuntimeWorkerSupport
import Testing

@Test
func runtimeWorkerClientSkipsNonJSONStdoutLines() {
    #expect(runtimeWorkerLineLooksLikeJSONResponse(Data("  {\"id\":1,\"ok\":true}".utf8)))
    #expect(!runtimeWorkerLineLooksLikeJSONResponse(Data("Metal device initialized".utf8)))
    #expect(!runtimeWorkerLineLooksLikeJSONResponse(Data("".utf8)))
}

// metrics.commit must come from the trusted dispatch context when the ranked
// pipeline supplies it: on the ranked box the harness runs as the sandboxed
// bench uid where `git rev-parse` fails (dubious ownership in the runner-owned
// workspace copy), which produced empty commits that failed every ranked
// score's commit binding. git stays the local/dev fallback only.
@Test
func commitIdentifierPrefersTrustedDispatchSHAOverGit() {
    let fullSHA = "5f95c4bdce07a0ef79ea350c91d9eb0d7476cf2f"
    #expect(QwenRuntime.commitIdentifier(environment: ["MLXFAST_COMMIT_SHA": fullSHA]) == fullSHA)
    // Short (rev-parse --short style) values and surrounding whitespace are fine.
    #expect(QwenRuntime.commitIdentifier(environment: ["MLXFAST_COMMIT_SHA": "5f95c4bdce07"]) == "5f95c4bdce07")
    #expect(QwenRuntime.commitIdentifier(environment: ["MLXFAST_COMMIT_SHA": " \(fullSHA)\n"]) == fullSHA)

    // Values that could not satisfy the trusted shell predicates fall back to
    // git instead of being stamped verbatim into the sealed score.
    for invalid in [
        "",
        "5f95c4",
        fullSHA.uppercased(),
        fullSHA + "0",
        "not-a-commit-sha",
        "5f95c4bdce07;rm -rf /",
    ] {
        let fallback = QwenRuntime.commitIdentifier(environment: ["MLXFAST_COMMIT_SHA": invalid])
        #expect(fallback != invalid || invalid.isEmpty)
        #expect(!fallback.contains(";"))
    }

    #expect(QwenRuntime.isCommitSHAHex(fullSHA))
    #expect(QwenRuntime.isCommitSHAHex("abcdef0"))
    #expect(!QwenRuntime.isCommitSHAHex("abcdef"))
    #expect(!QwenRuntime.isCommitSHAHex(String(repeating: "a", count: 41)))
    #expect(!QwenRuntime.isCommitSHAHex("ABCDEF0"))
}

@Test
func runtimeWorkerPrivateDescriptorUsesPortableLowerBoundAndCloseOnExec() throws {
    let pipe = Pipe()
    let descriptor = try duplicatePrivateDescriptor(
        pipe.fileHandleForReading.fileDescriptor,
        label: "test"
    )
    defer { Darwin.close(descriptor) }

    #expect(descriptor >= STDERR_FILENO + 1)
    let flags = fcntl(descriptor, F_GETFD)
    #expect(flags >= 0)
    #expect(flags & FD_CLOEXEC == FD_CLOEXEC)
}

@Test
func runtimeWorkerProtocolDescriptorAllocationClosesInputWhenOutputFails() {
    var attempts: [String] = []
    var closed: [Int32] = []

    #expect(throws: RuntimeWorkerDescriptorTestError.self) {
        _ = try duplicateRuntimeWorkerProtocolDescriptors(
            inputDescriptor: STDIN_FILENO,
            outputDescriptor: STDOUT_FILENO,
            duplicate: { _, label in
                attempts.append(label)
                if label == "stdin" {
                    return 91
                }
                throw RuntimeWorkerDescriptorTestError.expected
            },
            closeDescriptor: { closed.append($0) }
        )
    }
    #expect(attempts == ["stdin", "stdout"])
    #expect(closed == [91])
}

private enum RuntimeWorkerDescriptorTestError: Error {
    case expected
}

@Test
func runtimeWorkerPinnedConfigurationAcceptsQwen35Architecture() throws {
    let data = try JSONSerialization.data(
        withJSONObject: pinnedRuntimeWorkerConfigurationObject()
    )
    try validateRuntimeWorkerPinnedConfigurationData(data)
}

@Test
func runtimeWorkerPinnedConfigurationRejectsWrongQwen35Fields() throws {
    var cases: [(String, [String: Any])] = []

    func addCase(_ name: String, _ mutate: (inout [String: Any]) -> Void) {
        var object = pinnedRuntimeWorkerConfigurationObject()
        mutate(&object)
        cases.append((name, object))
    }

    addCase("model-type") { $0["model_type"] = "other" }
    addCase("vocab") { $0["vocab_size"] = MLXFastConstants.vocabSize - 1 }
    addCase("hidden-size") { $0["hidden_size"] = MLXFastConstants.hiddenSize - 1 }
    addCase("intermediate-size") { $0["intermediate_size"] = MLXFastConstants.intermediateSize - 1 }
    addCase("hidden-layers") { $0["num_hidden_layers"] = MLXFastConstants.numHiddenLayers - 1 }
    addCase("attention-heads") { $0["num_attention_heads"] = MLXFastConstants.attentionHeads - 1 }
    addCase("kv-heads") { $0["num_key_value_heads"] = 8 }
    addCase("head-dim") { $0["head_dim"] = 128 }
    addCase("linear-value-heads") { $0["linear_num_value_heads"] = 24 }
    addCase("linear-key-heads") { $0["linear_num_key_heads"] = 8 }
    addCase("linear-value-head-dim") { $0["linear_value_head_dim"] = 64 }
    addCase("linear-key-head-dim") { $0["linear_key_head_dim"] = 64 }
    addCase("linear-conv-kernel") { $0["linear_conv_kernel_dim"] = 3 }
    addCase("full-attention-interval") { $0["full_attention_interval"] = 8 }
    addCase("rms-norm") { $0["rms_norm_eps"] = 1e-5 }
    addCase("hidden-activation") { $0["hidden_act"] = "gelu" }
    addCase("max-position") { $0["max_position_embeddings"] = 131_072 }
    addCase("attention-bias") { $0["attention_bias"] = true }
    addCase("attention-dropout") { $0["attention_dropout"] = 0.1 }
    addCase("attention-output-gate") { $0["attn_output_gate"] = false }
    addCase("output-gate-type") { $0["output_gate_type"] = "sigmoid" }
    addCase("bos-token") { $0["bos_token_id"] = 1 }
    addCase("eos-token") { $0["eos_token_id"] = 1 }
    addCase("initializer-range") { $0["initializer_range"] = 0.01 }
    addCase("pad-token") { $0["pad_token_id"] = 0 }
    addCase("tie-embeddings") { $0["tie_word_embeddings"] = true }
    addCase("mamba-dtype") { $0["mamba_ssm_dtype"] = "bfloat16" }
    addCase("dtype") { $0["dtype"] = "float16" }
    addCase("use-cache") { $0["use_cache"] = false }
    addCase("top-level-partial-rotary") { $0["partial_rotary_factor"] = 0.5 }
    addCase("layer-pattern") {
        var layerTypes = $0["layer_types"] as! [String]
        layerTypes[0] = "full_attention"
        $0["layer_types"] = layerTypes
    }
    addCase("layer-count") {
        var layerTypes = $0["layer_types"] as! [String]
        layerTypes.removeLast()
        $0["layer_types"] = layerTypes
    }
    addCase("rope-theta") {
        var rope = $0["rope_parameters"] as! [String: Any]
        rope["rope_theta"] = 1_000_000
        $0["rope_parameters"] = rope
    }
    addCase("rope-type") {
        var rope = $0["rope_parameters"] as! [String: Any]
        rope["rope_type"] = "proportional"
        $0["rope_parameters"] = rope
    }
    addCase("rope-partial-rotary") {
        var rope = $0["rope_parameters"] as! [String: Any]
        rope["partial_rotary_factor"] = 0.5
        $0["rope_parameters"] = rope
    }
    addCase("mrope-interleaved") {
        var rope = $0["rope_parameters"] as! [String: Any]
        rope["mrope_interleaved"] = false
        $0["rope_parameters"] = rope
    }
    addCase("mrope-section") {
        var rope = $0["rope_parameters"] as! [String: Any]
        rope["mrope_section"] = [16, 16]
        $0["rope_parameters"] = rope
    }
    addCase("rope-library-type") {
        var rope = $0["rope_parameters"] as! [String: Any]
        rope["type"] = "linear"
        $0["rope_parameters"] = rope
    }
    addCase("rope-library-factor") {
        var rope = $0["rope_parameters"] as! [String: Any]
        rope["factor"] = 2
        $0["rope_parameters"] = rope
    }
    addCase("rope-unknown") {
        var rope = $0["rope_parameters"] as! [String: Any]
        rope["unknown"] = true
        $0["rope_parameters"] = rope
    }
    addCase("quantization-bits") {
        var quantization = $0["quantization"] as! [String: Any]
        quantization["bits"] = 8
        $0["quantization"] = quantization
    }
    addCase("quantization-group") {
        var quantization = $0["quantization"] as! [String: Any]
        quantization["group_size"] = 32
        $0["quantization"] = quantization
    }
    addCase("quantization-mode") {
        var quantization = $0["quantization"] as! [String: Any]
        quantization["mode"] = "symmetric"
        $0["quantization"] = quantization
    }
    addCase("quantization-unknown") {
        var quantization = $0["quantization"] as! [String: Any]
        quantization["unknown"] = true
        $0["quantization"] = quantization
    }
    addCase("mtp-layers") { $0["mtp_num_hidden_layers"] = 0 }
    addCase("mtp-dedicated-embeddings") { $0["mtp_use_dedicated_embeddings"] = true }
    addCase("mtp-enabled") { $0["mtp_enabled"] = true }
    addCase("moe-experts") { $0["num_experts"] = 8 }
    addCase("moe-intermediate") { $0["moe_intermediate_size"] = 1_024 }
    addCase("moe-enable") { $0["enable_moe_block"] = true }
    addCase("unknown-root") { $0["behavior_change"] = true }

    for (name, object) in cases {
        let data = try JSONSerialization.data(withJSONObject: object)
        #expect(throws: MLXFastError.self, "case \(name)") {
            try validateRuntimeWorkerPinnedConfigurationData(data)
        }
    }
}

@Test
func runtimeWorkerPinnedConfigurationRejectsMissingQwen35Fields() throws {
    let requiredTopLevelFields = [
        "model_type", "vocab_size", "hidden_size", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "linear_num_value_heads", "linear_num_key_heads",
        "linear_value_head_dim", "linear_key_head_dim",
        "linear_conv_kernel_dim", "full_attention_interval", "layer_types",
        "rms_norm_eps", "hidden_act", "max_position_embeddings",
        "attention_bias", "attention_dropout", "attn_output_gate",
        "output_gate_type", "bos_token_id", "eos_token_id",
        "initializer_range", "pad_token_id", "tie_word_embeddings", "mamba_ssm_dtype",
        "dtype", "use_cache", "partial_rotary_factor", "rope_parameters",
        "mtp_num_hidden_layers", "mtp_use_dedicated_embeddings",
    ]
    for field in requiredTopLevelFields {
        var object = pinnedRuntimeWorkerConfigurationObject()
        object.removeValue(forKey: field)
        #expect(throws: MLXFastError.self, "missing \(field)") {
            try validateRuntimeWorkerPinnedConfigurationData(
                JSONSerialization.data(withJSONObject: object)
            )
        }
    }

    for field in [
        "rope_theta", "rope_type", "partial_rotary_factor",
        "mrope_interleaved", "mrope_section",
    ] {
        var object = pinnedRuntimeWorkerConfigurationObject()
        var rope = object["rope_parameters"] as! [String: Any]
        rope.removeValue(forKey: field)
        object["rope_parameters"] = rope
        #expect(throws: MLXFastError.self, "missing rope field \(field)") {
            try validateRuntimeWorkerPinnedConfigurationData(
                JSONSerialization.data(withJSONObject: object)
            )
        }
    }

    for field in ["group_size", "bits", "mode"] {
        var object = pinnedRuntimeWorkerConfigurationObject()
        var quantization = object["quantization"] as! [String: Any]
        quantization.removeValue(forKey: field)
        object["quantization"] = quantization
        #expect(throws: MLXFastError.self, "missing quantization field \(field)") {
            try validateRuntimeWorkerPinnedConfigurationData(
                JSONSerialization.data(withJSONObject: object)
            )
        }
    }

    var noQuantization = pinnedRuntimeWorkerConfigurationObject()
    noQuantization.removeValue(forKey: "quantization")
    #expect(throws: MLXFastError.self) {
        try validateRuntimeWorkerPinnedConfigurationData(
            JSONSerialization.data(withJSONObject: noQuantization)
        )
    }
}

@Test
func runtimeWorkerPinnedConfigurationRejectsAlternateQuantizationSchemas() throws {
    var quantizationConfigOnly = pinnedRuntimeWorkerConfigurationObject()
    quantizationConfigOnly["quantization_config"] =
        quantizationConfigOnly.removeValue(forKey: "quantization")
    #expect(throws: MLXFastError.self) {
        try validateRuntimeWorkerPinnedConfigurationData(
            JSONSerialization.data(withJSONObject: quantizationConfigOnly)
        )
    }

    var matchingForms = pinnedRuntimeWorkerConfigurationObject()
    matchingForms["quantization_config"] = matchingForms["quantization"]
    #expect(throws: MLXFastError.self) {
        try validateRuntimeWorkerPinnedConfigurationData(
            JSONSerialization.data(withJSONObject: matchingForms)
        )
    }
}

@Test
func runtimeWorkerPinnedConfigurationPathRejectsUnsafeFilesystemEntries() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let validData = try JSONSerialization.data(
        withJSONObject: pinnedRuntimeWorkerConfigurationObject()
    )

    let validDirectory = root.appendingPathComponent("valid", isDirectory: true)
    try FileManager.default.createDirectory(at: validDirectory, withIntermediateDirectories: true)
    try validData.write(to: validDirectory.appendingPathComponent("config.json"))
    try validateRuntimeWorkerPinnedConfiguration(weightsPath: validDirectory.path)

    let symlinkTarget = root.appendingPathComponent("config-target.json")
    try validData.write(to: symlinkTarget)
    let symlinkDirectory = root.appendingPathComponent("symlink", isDirectory: true)
    try FileManager.default.createDirectory(at: symlinkDirectory, withIntermediateDirectories: true)
    try FileManager.default.createSymbolicLink(
        at: symlinkDirectory.appendingPathComponent("config.json"),
        withDestinationURL: symlinkTarget
    )

    let directoryEntry = root.appendingPathComponent("directory-entry", isDirectory: true)
    try FileManager.default.createDirectory(
        at: directoryEntry.appendingPathComponent("config.json", isDirectory: true),
        withIntermediateDirectories: true
    )

    let emptyDirectory = root.appendingPathComponent("empty", isDirectory: true)
    try FileManager.default.createDirectory(at: emptyDirectory, withIntermediateDirectories: true)
    try Data().write(to: emptyDirectory.appendingPathComponent("config.json"))

    let oversizedDirectory = root.appendingPathComponent("oversized", isDirectory: true)
    try FileManager.default.createDirectory(at: oversizedDirectory, withIntermediateDirectories: true)
    var oversizedData = validData
    oversizedData.append(Data(repeating: 0x20, count: 1 * 1024 * 1024 + 1))
    try oversizedData.write(to: oversizedDirectory.appendingPathComponent("config.json"))

    for directory in [symlinkDirectory, directoryEntry, emptyDirectory, oversizedDirectory] {
        #expect(throws: MLXFastError.self, "unsafe config at \(directory.lastPathComponent)") {
            try validateRuntimeWorkerPinnedConfiguration(weightsPath: directory.path)
        }
    }
}

// The worker environment filter is a strict ALLOWLIST: submitted model code
// executes inside the worker and can read its whole environment, so every
// harness/CI/phase/identity variable must be dropped -- not just a hand-picked
// denylist. Any keep-by-default behavior is a regression: a variable whose
// value differs between the correctness/gates pass and the timed pass (e.g.
// MLXFAST_NOTE, MLXFAST_SCORE_PATH, MLXFAST_INTEGRITY_PATH) is a phase oracle
// that lets a submission serve a correct path while checked and a cheaper
// path while timed.
@Test
func runtimeWorkerEnvironmentDropsEverythingOutsideTheAllowlist() {
    let sanitized = sanitizedRuntimeWorkerEnvironment([
        // Confirmed phase-distinct leaks under the previous denylist: these
        // differ between the gates pass and the timed pass (or exist in only
        // one of them).
        "MLXFAST_NOTE": "ci workflow_dispatch 5f95c4bdce07 mode=single-machine-gates",
        "MLXFAST_SCORE_PATH": "score.gates.json",
        "MLXFAST_INTEGRITY_PATH": "benchmark-integrity.gates.json",
        "MLXFAST_SEMANTIC_GPQA_CASE_COUNT": "8",
        "MLXFAST_SEMANTIC_GPQA_MAX_NEW_TOKENS": "512",
        // Official-run identity: same value in both phases, but submitted
        // code must not observe that it runs under the ranked pipeline.
        "MLXFAST_COMMIT_SHA": "5f95c4bdce07a0ef79ea350c91d9eb0d7476cf2f",
        "MLXFAST_CANDIDATE_SHA": "5f95c4bdce07a0ef79ea350c91d9eb0d7476cf2f",
        "GITHUB_SHA": "5f95c4bdce07a0ef79ea350c91d9eb0d7476cf2f",
        "CI": "true",
        "GITHUB_ACTIONS": "true",
        "GITHUB_RUN_ID": "123",
        "RUNNER_TEMP": "/tmp/runner",
        "BLACKSMITH_RUNNER": "1",
        // An UNKNOWN future harness variable must be dropped by default;
        // this is the property the old denylist could not provide.
        "MLXFAST_FUTURE_PHASE_MARKER": "x",
        // Trusted-shell wiring around the bench bridge, not for the worker.
        "BENCH_GOLDEN_PATH": "/ws/correctness_golden_ranked.json",
        "BENCH_JOB_ID": "ranked-123-1-gates",
        "MJOB_WS": "/Users/Shared/bench-jobs/ranked-123-1",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": "*",
        // Secrets and private-material locations.
        "ANTHROPIC_API_KEY": "secret",
        "R2_ACCESS_KEY_ID": "key",
        "MLXFAST_PRIVATE_DIR": "/private/golden",
        "MLXFAST_CORRECTNESS_GOLDEN_PATH": "/private/golden/correctness_golden.json",
        // Harness configuration the worker does not read (weights arrive via
        // argv; the byte cap is enforced in the trusted parent).
        "MLXFAST_OFFICIAL_BENCHMARK_RUN": "1",
        "MLXFAST_RUN_BENCHMARK": "1",
        "MLXFAST_BENCHMARK_CHECK_GATES": "1",
        "MLXFAST_BENCHMARK_SKIP_TIMED": "1",
        "MLXFAST_BENCHMARK_CORRECTNESS_STEPS": "64",
        "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "0.17",
        "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "3.6",
        "MLXFAST_REFERENCE_DIR": "/private/reference",
        "MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE": "/tmp/profile.sb",
        "MLXFAST_RUNTIME_WORKER_EXECUTABLE": "/ws/.build/release/mlxfast-runtime-worker",
        "MLXFAST_WEIGHTS_PATH": "weights",
        "MLXFAST_MAX_WEIGHTS_BYTES": "42",
        // Allowlisted entries that must survive with their exact values.
        "PATH": "/usr/bin:/bin",
        "HOME": "/Users/bench",
        "TMPDIR": "/var/folders/xx/T/",
        "USER": "bench",
        "LOGNAME": "bench",
        "SHELL": "/bin/zsh",
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "TERM": "xterm-256color",
        // A live SSH agent socket must be dropped: the worker never needs it
        // and forwarding it hands submitted code a usable auth channel wherever
        // the parent holds one.
        "SSH_AUTH_SOCK": "/tmp/ssh-agent.sock",
        "__CF_USER_TEXT_ENCODING": "0x1F5:0x0:0x0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DYLD_LIBRARY_PATH": "/opt/lib",
        "MTL_SHADER_VALIDATION": "0",
        "METAL_DEVICE_WRAPPER_TYPE": "0",
        "MLX_DISABLE_COMPILE": "1",
        "MLX_RESOURCE_LIMIT": "17179869184",
        "DARKBLOOM_COMPILED_DECODE": "1",
    ])

    // NOTHING outside the allowlist survives -- asserted structurally, not
    // via a hand-picked list, so a future leak cannot slip between cases.
    let expected: [String: String] = [
        "PATH": "/usr/bin:/bin",
        "HOME": "/Users/bench",
        "TMPDIR": "/var/folders/xx/T/",
        "USER": "bench",
        "LOGNAME": "bench",
        "SHELL": "/bin/zsh",
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "TERM": "xterm-256color",
        // SSH_AUTH_SOCK is intentionally absent: it is supplied above but must
        // be dropped by the allowlist.
        "__CF_USER_TEXT_ENCODING": "0x1F5:0x0:0x0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DYLD_LIBRARY_PATH": "/opt/lib",
        "MTL_SHADER_VALIDATION": "0",
        "METAL_DEVICE_WRAPPER_TYPE": "0",
        "MLX_DISABLE_COMPILE": "1",
        "MLX_RESOURCE_LIMIT": "17179869184",
        "DARKBLOOM_COMPILED_DECODE": "1",
        "MLXFAST_USE_RUNTIME_WORKER": "0",
    ]
    #expect(sanitized == expected)
}

// "MLX_"-prefixed tuning knobs are allowlisted, but that prefix must never
// admit harness "MLXFAST_*" names ("MLXFAST_" does not start with "MLX_").
@Test
func runtimeWorkerEnvironmentAllowlistPrefixDoesNotAdmitHarnessNames() {
    let sanitized = sanitizedRuntimeWorkerEnvironment([
        "MLX_METAL_FAST_SYNCH": "1",
        "MLXFAST_NOTE": "ci mode=single-machine-gates",
        "MLXFAST_SCORE_PATH": "score.gates.json",
        "MLXFAST_FUTURE_PHASE_MARKER": "x",
    ])
    #expect(sanitized["MLX_METAL_FAST_SYNCH"] == "1")
    #expect(sanitized["MLXFAST_NOTE"] == nil)
    #expect(sanitized["MLXFAST_SCORE_PATH"] == nil)
    #expect(sanitized["MLXFAST_FUTURE_PHASE_MARKER"] == nil)
    #expect(sanitized == [
        "MLX_METAL_FAST_SYNCH": "1",
        "MLXFAST_USE_RUNTIME_WORKER": "0",
    ])
}

// The property that kills the phase oracle: the gates pass and the timed pass
// hand the worker BYTE-IDENTICAL environments, so submitted code cannot tell
// which pipeline phase it is running in. The two parent environments below
// differ exactly the way the ranked pipeline's do (benchmark.yml gates step
// vs. measure-job's timed invocation); shared host basics are identical
// because both phases go through the same bench-exec bridge.
@Test
func runtimeWorkerEnvironmentIsIdenticalAcrossPipelinePhases() {
    let sharedHostEnvironment: [String: String] = [
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": "/Users/bench",
        "TMPDIR": "/var/folders/bench/T/",
        "USER": "bench",
        "LOGNAME": "bench",
        "SHELL": "/bin/bash",
        "__CF_USER_TEXT_ENCODING": "0x1F5:0x0:0x0",
        "MLXFAST_OFFICIAL_BENCHMARK_RUN": "1",
        "MLXFAST_USE_RUNTIME_WORKER": "1",
        "MLXFAST_RUNTIME_WORKER_EXECUTABLE": "/ws/.build/release/mlxfast-runtime-worker",
        "MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE": "/tmp/mlxfast-runtime-worker.gates.sb",
        "MLXFAST_REFERENCE_DIR": "/opt/bench-runner/cache/reference",
        "MLXFAST_COMMIT_SHA": "5f95c4bdce07a0ef79ea350c91d9eb0d7476cf2f",
        "MLXFAST_SKIP_TRANSFORM": "1",
        "MJOB_WS": "/Users/Shared/bench-jobs/ranked-123-1",
    ]

    var gatesEnvironment = sharedHostEnvironment
    gatesEnvironment["MLXFAST_NOTE"] = "ci workflow_dispatch 5f95c4bdce07 mode=single-machine-gates"
    gatesEnvironment["MLXFAST_SCORE_PATH"] = "score.gates.json"
    gatesEnvironment["MLXFAST_INTEGRITY_PATH"] = "benchmark-integrity.gates.json"
    gatesEnvironment["MLXFAST_SEMANTIC_GPQA_CASE_COUNT"] = "5"
    gatesEnvironment["MLXFAST_SEMANTIC_GPQA_MAX_NEW_TOKENS"] = "64"
    gatesEnvironment["MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH"] = "/ws/private/semantic_gpqa_answers.json"
    gatesEnvironment["MLXFAST_BENCHMARK_CHECK_GATES"] = "1"
    gatesEnvironment["MLXFAST_BENCHMARK_SKIP_TIMED"] = "1"
    gatesEnvironment["MLXFAST_BENCHMARK_CORRECTNESS_STEPS"] = "64"
    gatesEnvironment["MLXFAST_CORRECTNESS_GOLDEN_PATH"] = "correctness_golden_ranked.json"
    gatesEnvironment["MLXFAST_WEIGHTS_PATH"] = "weights"
    gatesEnvironment["MLXFAST_PRIVATE_DIR"] = "/ws/private"
    gatesEnvironment["BENCH_GOLDEN_PATH"] = "/ws/correctness_golden_ranked.json"
    gatesEnvironment["BENCH_JOB_ID"] = "ranked-123-1-gates"
    gatesEnvironment["GIT_CONFIG_COUNT"] = "1"
    gatesEnvironment["GIT_CONFIG_KEY_0"] = "safe.directory"
    gatesEnvironment["GIT_CONFIG_VALUE_0"] = "*"

    var timedEnvironment = sharedHostEnvironment
    timedEnvironment["MLXFAST_SCORE_PATH"] = "score.mjob.json"
    timedEnvironment["MLXFAST_INTEGRITY_PATH"] = "benchmark-integrity.mjob.json"
    timedEnvironment["MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE"] = "/tmp/mlxfast-runtime-worker.timed.sb"
    timedEnvironment["BENCH_GOLDEN_PATH"] = "/ws/bench_oracle.json"
    timedEnvironment["BENCH_JOB_ID"] = "ranked-123-1-timing"

    let gatesWorkerEnvironment = sanitizedRuntimeWorkerEnvironment(gatesEnvironment)
    let timedWorkerEnvironment = sanitizedRuntimeWorkerEnvironment(timedEnvironment)
    #expect(gatesWorkerEnvironment == timedWorkerEnvironment)
    #expect(gatesWorkerEnvironment["MLXFAST_USE_RUNTIME_WORKER"] == "0")
    #expect(gatesWorkerEnvironment["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin")
    #expect(gatesWorkerEnvironment["HOME"] == "/Users/bench")
}

@Test
func pairedBaselineOverrideParsesTrustedEnvironmentFailClosed() throws {
    // Absent entirely: no pairing, callers fall back to golden/constants.
    #expect(try PairedBaselineOverride.fromEnvironment([:]) == nil)
    #expect(try PairedBaselineOverride.fromEnvironment([
        "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "",
        "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "  ",
    ]) == nil)

    // Present: both values parsed precisely.
    let override = try #require(try PairedBaselineOverride.fromEnvironment([
        "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "0.16518489738085937",
        "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "3.6977511595078125",
    ]))
    #expect(override.prefillSecondsPerToken == 0.16518489738085937)
    #expect(override.decodeSecondsPerToken == 3.6977511595078125)

    // Half-set pairs and non-positive/non-finite values are operator wiring
    // errors: fail closed rather than silently repricing against constants.
    for badEnvironment: [String: String] in [
        ["MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "0.17"],
        ["MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "3.6"],
        [
            "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "0",
            "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "3.6",
        ],
        [
            "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "0.17",
            "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "-1",
        ],
        [
            "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "inf",
            "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "3.6",
        ],
        [
            "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "fast",
            "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "3.6",
        ],
    ] {
        #expect(throws: MLXFastError.self) {
            _ = try PairedBaselineOverride.fromEnvironment(badEnvironment)
        }
    }
}

@Test
func semanticBehaviorGateRequiresPromptAndReferenceAnswer() {
    let exactOnly = GoldenBehaviorCase(
        name: "exact-only",
        promptTokens: [1],
        acceptedTokenSequences: [[2]],
        maxNewTokens: 1
    )
    let missingReference = GoldenBehaviorCase(
        name: "missing-reference",
        promptTokens: [1],
        acceptedTokenSequences: [[2]],
        maxNewTokens: 1,
        semanticPrompt: "question"
    )
    let semantic = GoldenBehaviorCase(
        name: "semantic",
        promptTokens: [1],
        acceptedTokenSequences: [[2]],
        maxNewTokens: 1,
        semanticPrompt: "question",
        semanticReferenceAnswer: "answer"
    )

    #expect(!QwenRuntime.behaviorUsesSemanticJudge(exactOnly))
    #expect(!QwenRuntime.behaviorUsesSemanticJudge(missingReference))
    #expect(QwenRuntime.behaviorUsesSemanticJudge(semantic))
}

@Test
func correctnessAcceptsOnlyExactTopLogitTies() {
    #expect(correctnessTokenAccepted(
        expectedToken: 30,
        actualToken: 1,
        topLogits: [
            CorrectnessTraceLogit(token: 1, logit: 19.25),
            CorrectnessTraceLogit(token: 30, logit: 19.25),
        ]
    ))
    #expect(!correctnessTokenAccepted(
        expectedToken: 30,
        actualToken: 1,
        topLogits: [
            CorrectnessTraceLogit(token: 1, logit: 19.25),
            CorrectnessTraceLogit(token: 30, logit: 19.0),
        ]
    ))
    #expect(!correctnessTokenAccepted(expectedToken: 30, actualToken: 1, topLogits: nil))
}

@Test
func failedScoreRedactsCorrectnessTokenMismatchByDefault() {
    let report = CorrectnessReport(
        passed: false,
        checkedSteps: 7,
        caseCount: 1,
        firstFailingCase: "local-iterate",
        firstFailingStep: 6,
        expectedToken: 123,
        actualToken: 456,
        goldenHash: "golden",
        error: "token mismatch"
    )

    let payload = QwenRuntime.failedScore(
        error: "token mismatch",
        correctness: report,
        passedCorrectness: false,
        runtime: "swift-local-iterate"
    )

    #expect(payload.score == nil)
    #expect(payload.passed == false)
    #expect(payload.metrics.runtime == "swift-local-iterate")
    #expect(payload.metrics.firstFailingCase == "local-iterate")
    #expect(payload.metrics.firstFailingStep == 6)
    #expect(payload.metrics.expectedToken == nil)
    #expect(payload.metrics.actualToken == nil)
    #expect(payload.metrics.bandwidthGBPerToken == 0)
}

@Test
func failedScorePreservesExplicitPublicMismatchTokensAndRuntimeLabel() {
    let report = CorrectnessReport(
        passed: false,
        checkedSteps: 7,
        caseCount: 1,
        firstFailingCase: "local-iterate",
        firstFailingStep: 6,
        expectedToken: 123,
        actualToken: 456,
        goldenHash: "golden",
        error: "token mismatch"
    )

    let payload = QwenRuntime.failedScore(
        error: "token mismatch",
        correctness: report,
        passedCorrectness: false,
        expectedToken: report.expectedToken,
        actualToken: report.actualToken,
        runtime: "swift-local-iterate"
    )

    #expect(payload.score == nil)
    #expect(payload.passed == false)
    #expect(payload.metrics.runtime == "swift-local-iterate")
    #expect(payload.metrics.firstFailingCase == "local-iterate")
    #expect(payload.metrics.firstFailingStep == 6)
    #expect(payload.metrics.expectedToken == 123)
    #expect(payload.metrics.actualToken == 456)
}

@Test
func decodeTimingPlanStartsAfterSeedPrefill() throws {
    let plan = try DecodeTimingPlan(seedTokenCount: 32, decodeSteps: 4)
    var offsets: [Int] = []
    for step in 0..<plan.decodeSteps {
        offsets.append(try plan.positionOffset(forDecodedStep: step))
    }

    #expect(offsets == [32, 33, 34, 35])
}

@Test
func decodeTimingPlanRejectsInvalidRanges() throws {
    #expect(throws: MLXFastError.self) {
        _ = try DecodeTimingPlan(seedTokenCount: 0, decodeSteps: 4)
    }
    #expect(throws: MLXFastError.self) {
        _ = try DecodeTimingPlan(seedTokenCount: 32, decodeSteps: 0)
    }

    let plan = try DecodeTimingPlan(seedTokenCount: 32, decodeSteps: 4)
    #expect(throws: MLXFastError.self) {
        _ = try plan.positionOffset(forDecodedStep: 4)
    }
}

// The former editable decode-delay knob (Gemma4SubmissionControls
// .measuredDecodeDelayMilliseconds, read via
// QwenRuntime.submissionValidationDelayMilliseconds) was model code invoked by
// trusted code ONLY on the scored decode path. Because submitted model code is
// editable, being invoked only while timed was itself a phase oracle. The hook
// is now removed entirely: the editable file is gone and no trusted harness
// code invokes it. See decodeMeasurementInvokesNoPhaseVaryingEditableHook for
// the structural phase-independence guard.
@Test
func editableDecodeDelayHookIsFullyRemoved() throws {
    #expect(!FileManager.default.fileExists(
        atPath: "Sources/MLXFastModel/Gemma4SubmissionControls.swift"
    ))
    let harnessDirectory = "Sources/MLXFastHarness"
    let harnessFiles = try FileManager.default.contentsOfDirectory(atPath: harnessDirectory)
        .filter { $0.hasSuffix(".swift") }
    for file in harnessFiles {
        let source = try String(contentsOfFile: "\(harnessDirectory)/\(file)", encoding: .utf8)
        #expect(!source.contains("submissionValidationDelayMilliseconds"))
        #expect(!source.contains("measuredDecodeDelayMilliseconds"))
    }
}

@Test
func benchmarkGoldenOracleFieldsAreValidatedOnLoad() throws {
    let prefill = Array(0..<MLXFastConstants.benchmarkPrefillPromptTokens)
    let seed = Array(0..<MLXFastConstants.benchmarkDecodeSeedTokens)
    let decode = Array(repeating: 9, count: MLXFastConstants.benchmarkDecodeSteps)
    let benchmark = BenchmarkGolden(
        prefillPromptTokens: prefill,
        expectedPrefillToken: 17,
        decodeSeedTokens: seed,
        expectedDecodeSeedToken: 23,
        expectedDecodeTokens: decode
    )
    try validateBenchmarkGolden(benchmark)

    #expect(benchmark.prefillPromptTokens == prefill)
    #expect(benchmark.expectedPrefillToken == 17)
    #expect(benchmark.decodeSeedTokens == seed)
    #expect(benchmark.expectedDecodeSeedToken == 23)
    #expect(benchmark.expectedDecodeTokens == decode)
}

@Test
func validateBenchmarkGoldenRejectsMalformedOracle() {
    #expect(throws: MLXFastError.self) {
        try validateBenchmarkGolden(BenchmarkGolden(
            prefillPromptTokens: [1],
            expectedPrefillToken: 7,
            decodeSeedTokens: Array(repeating: 1, count: MLXFastConstants.benchmarkDecodeSeedTokens),
            expectedDecodeSeedToken: 7,
            expectedDecodeTokens: Array(repeating: 7, count: MLXFastConstants.benchmarkDecodeSteps)
        ))
    }
}

@Test
func defaultTransformedWeightsLimitIsTwentyFiveGiB() {
    #expect(MLXFastConstants.defaultMaxTransformedWeightsBytes == 25 * 1024 * 1024 * 1024)
}

@Test
func benchmarkPreflightAcceptsRequiredArtifacts() throws {
    let fixture = try makePreflightFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let report = try checkPreflight(fixture)

    #expect(report.weightsPath == fixture.weights.path)
    #expect(report.goldenPath == fixture.golden.path)
    #expect(report.weightsByteCount > 0)
    #expect(report.maxWeightsByteCount == MLXFastConstants.defaultMaxTransformedWeightsBytes)
}

@Test
func benchmarkPreflightRejectsWeightsAboveDefaultByteLimit() throws {
    let fixture = try makePreflightFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    try writeSparseFile(
        fixture.weights.appendingPathComponent("oversized.bin"),
        byteCount: MLXFastConstants.defaultMaxTransformedWeightsBytes + 1
    )

    #expect(throws: MLXFastError.self) {
        _ = try checkPreflight(fixture)
    }
}

@Test
func benchmarkPreflightHonorsConfiguredWeightsByteLimit() throws {
    let fixture = try makePreflightFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    try writeSparseFile(
        fixture.weights.appendingPathComponent("large-but-allowed.bin"),
        byteCount: MLXFastConstants.defaultMaxTransformedWeightsBytes + 1
    )
    let override = MLXFastConstants.defaultMaxTransformedWeightsBytes * 2

    let report = try checkPreflight(
        fixture,
        environment: [
            "MLXFAST_MAX_WEIGHTS_BYTES": "\(override)",
        ]
    )

    #expect(report.weightsByteCount > MLXFastConstants.defaultMaxTransformedWeightsBytes)
    #expect(report.maxWeightsByteCount == override)
}

@Test
func benchmarkPreflightRejectsMalformedGolden() throws {
    let fixture = try makePreflightFixture(goldenContents: "{}")
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: Error.self) {
        _ = try checkPreflight(fixture)
    }
}

@Test
func benchmarkPreflightRejectsShortBenchmarkPrompt() throws {
    let fixture = try makePreflightFixture(goldenContents: validGoldenJSON(benchmarkPromptTokens: [1]))
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try checkPreflight(fixture)
    }
}

@Test
func benchmarkPreflightRejectsMissingBenchmarkOracle() throws {
    let expected = arrayJSON(Array(repeating: 7, count: MLXFastConstants.correctnessSteps))
    let fixture = try makePreflightFixture(goldenContents: """
    {
      "version": 1,
      "cases": [
        {
          "name": "preflight",
          "prompt_tokens": \(arrayJSON(Array(repeating: 1, count: MLXFastConstants.correctnessPromptTokens))),
          "expected_tokens": \(expected)
        }
      ]
    }
    """)
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try checkPreflight(fixture)
    }
}

@Test
func correctnessPreflightAcceptsPublicGoldenWithoutBenchmarkOracle() throws {
    let fixture = try makePreflightFixture(goldenContents: correctnessOnlyGoldenJSON())
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let report = try checkCorrectnessPreflight(fixture)

    #expect(report.weightsPath == fixture.weights.path)
    #expect(report.goldenPath == fixture.golden.path)
}

@Test
func correctnessPreflightHonorsConfiguredWeightsByteLimit() throws {
    let fixture = try makePreflightFixture(goldenContents: correctnessOnlyGoldenJSON())
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try checkCorrectnessPreflight(
            fixture,
            environment: ["MLXFAST_MAX_WEIGHTS_BYTES": "1"]
        )
    }
}

@Test
func nonWorkerBenchmarkRejectsBehaviorGatesBecauseTTFTRequiresWorker() throws {
    let behaviorGate = """
    {
      "behavior": [
        {
          "name": "gpqa-hidden-ttft",
          "prompt_tokens": \(arrayJSON(Array(repeating: 1, count: MLXFastConstants.correctnessPromptTokens))),
          "accepted_token_sequences": [[7]],
          "max_new_tokens": 1,
          "semantic_prompt": "Hidden short-answer prompt",
          "semantic_reference_answer": "Reference answer"
        }
      ]
    }
    """
    let fixture = try makePreflightFixture(goldenContents: validGoldenJSON(correctnessGates: behaviorGate))
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let score = QwenRuntime.benchmark(
        BenchmarkOptions(
            weightsPath: fixture.weights.path,
            goldenPath: fixture.golden.path,
            correctnessSteps: 1,
            benchmarkDecodeSteps: 1
        ),
        worker: nil
    )

    #expect(score.passed == false)
    #expect(score.metrics.passedCorrectness == false)
    #expect(score.metrics.error == "benchmark behavior and GPQA TTFT gates require runtime worker timing")
    #expect(score.metrics.gpqaTTFTCaseCount == 0)
    #expect(score.metrics.preflightSeconds == 0)
    #expect(score.metrics.weightsByteCount == 0)
    #expect(score.metrics.weightsFileCount == 0)
}

@Test
func qwen35PreflightFixtureHasExactPinnedTensorInventory() {
    let tensors = requiredQwen35DenseTensorFixtures()
    let names = Set(tensors.map(\.name))

    #expect(tensors.count == Qwen35WeightLoader.requiredTensorCount)
    #expect(names.count == Qwen35WeightLoader.requiredTensorCount)
    #expect(
        names.filter { !$0.contains(".layers.") }.count
            == Qwen35WeightLoader.requiredTopLevelTensorCount
    )
    for layerIndex in 0..<MLXFastConstants.numHiddenLayers {
        let prefix = "language_model.model.layers.\(layerIndex)."
        let layerCount = names.filter { $0.hasPrefix(prefix) }.count
        let expected = layerIndex % 4 == 3
            ? Qwen35WeightLoader.requiredFullAttentionLayerTensorCount
            : Qwen35WeightLoader.requiredLinearLayerTensorCount
        #expect(layerCount == expected, "layer \(layerIndex)")
    }
    #expect(names.contains(Qwen35WeightNames.lmHead))
    #expect(names.contains("language_model.lm_head.scales"))
    #expect(names.contains("language_model.lm_head.biases"))
}

@Test
func benchmarkPreflightRejectsMissingSemanticTensor() throws {
    let fixture = try makePreflightFixture(omitDenseTensorName: Qwen35WeightNames.finalNorm)
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try checkPreflight(fixture)
    }
}

@Test
func lagunaWeightLoaderRejectsUnexpectedTensorInventory() throws {
    let fixture = try makeLagunaWeightsFixture(
        extraDenseTensor: TensorFixture(
            name: "model.unexpected.weight",
            dtype: "BF16",
            shape: [1]
        )
    )
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let config = try LagunaConfig.load(from: fixture.weights.path)
    let loader = try LagunaWeightLoader(weightsPath: fixture.weights.path)
    #expect(throws: MLXFastError.self) {
        try loader.validateRequiredMetadata(config: config)
    }
}

@Test
func lagunaWeightContractPinsExactXSHeaderInventory() throws {
    let contractData = try Data(
        contentsOf: URL(
            fileURLWithPath:
                "Tests/Fixtures/PoolsideLagunaXS21NVFP4/header-inventory-contract.json"
        )
    )
    let contract = try JSONDecoder().decode(
        LagunaHeaderInventoryContract.self,
        from: contractData
    )
    let tensors = requiredLagunaDenseTensorFixtures()
    let dtypeCounts = Dictionary(grouping: tensors, by: \.dtype).mapValues(\.count)
    let canonicalInventory = tensors
        .sorted { $0.name < $1.name }
        .map {
            "\($0.name)\t\($0.dtype)\t\($0.shape.map(String.init).joined(separator: ","))"
        }
        .joined(separator: "\n") + "\n"
    let inventoryDigest = SHA256.hash(data: Data(canonicalInventory.utf8))
        .map { String(format: "%02x", $0) }
        .joined()

    #expect(contract.schemaVersion == 1)
    // Pinned to the Laguna literals rather than MLXFastConstants: this
    // contract describes the Poolside checkpoint, while on the
    // qwen36-mtp-track branch those constants name the Qwen 3.6 target.
    // Same anti-drift intent, anchored to the checkpoint it documents
    // (matches pinnedLagunaConfigObject in LagunaConfigTests).
    #expect(contract.source.repository == "poolside/Laguna-XS-2.1-NVFP4-mlx")
    #expect(
        contract.source.revision == "841778bda563a36104dd521e37d99218e46f4f25"
    )
    #expect(contract.canonicalRecordFormat.contains("name<TAB>dtype<TAB>"))
    #expect(tensors.count == contract.tensorCount)
    #expect(contract.tensorCount == LagunaConstants.tensorCount)
    #expect(dtypeCounts == contract.dtypeCounts)
    #expect(contract.dtypeCounts == [
        "BF16": LagunaConstants.bfloat16TensorCount,
        "F32": LagunaConstants.float32TensorCount,
        "U32": LagunaConstants.packedUInt32TensorCount,
        "U8": LagunaConstants.e4m3ScaleUInt8TensorCount,
    ])
    #expect(
        inventoryDigest == contract.canonicalInventorySHA256,
        Comment(rawValue: "actual inventory digest: \(inventoryDigest)")
    )
    for representative in contract.representativeTensors {
        let tensor = try #require(
            tensors.first { $0.name == representative.name },
            Comment(rawValue: representative.name)
        )
        #expect(tensor.dtype == representative.dtype)
        #expect(tensor.shape == representative.shape)
    }
}

@Test
func lagunaWeightLoaderExplicitlyRejectsNonMLXQuantizationSchemas() throws {
    for forbiddenName in [
        "model.layers.1.mlp.switch_mlp.gate_proj.weight_packed",
        "model.layers.1.mlp.switch_mlp.gate_proj.input_global_scale",
        "model.layers.1.mlp.switch_mlp.gate_proj.weight_global_scale",
        "model.layers.1.self_attn.k_scale",
        "model.layers.1.self_attn.v_scale",
        "model.layers.1.mlp.switch_mlp.gate_proj.biases",
    ] {
        let fixture = try makeLagunaWeightsFixture(
            extraDenseTensor: TensorFixture(
                name: forbiddenName,
                dtype: "U8",
                shape: [1]
            )
        )
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        let config = try LagunaConfig.load(from: fixture.weights.path)
        let loader = try LagunaWeightLoader(weightsPath: fixture.weights.path)
        var rejection: MLXFastError?
        do {
            try loader.validateRequiredMetadata(config: config)
        } catch let error as MLXFastError {
            rejection = error
        }
        #expect(
            rejection?.description.contains("must not contain") == true,
            Comment(rawValue: forbiddenName)
        )
    }
}

private struct PreflightFixture {
    let root: URL
    let weights: URL
    let golden: URL
}

private struct TensorFixture {
    let name: String
    let dtype: String
    let shape: [Int]
}

private struct LagunaHeaderInventoryContract: Decodable {
    struct Source: Decodable {
        let repository: String
        let revision: String
    }

    struct RepresentativeTensor: Decodable {
        let name: String
        let dtype: String
        let shape: [Int]
    }

    let schemaVersion: Int
    let source: Source
    let tensorCount: Int
    let dtypeCounts: [String: Int]
    let canonicalRecordFormat: String
    let canonicalInventorySHA256: String
    let representativeTensors: [RepresentativeTensor]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case source
        case tensorCount = "tensor_count"
        case dtypeCounts = "dtype_counts"
        case canonicalRecordFormat = "canonical_record_format"
        case canonicalInventorySHA256 = "canonical_inventory_sha256"
        case representativeTensors = "representative_tensors"
    }
}

private func checkPreflight(
    _ fixture: PreflightFixture,
    environment: [String: String] = [:]
) throws -> BenchmarkPreflightReport {
    try BenchmarkPreflight.check(
        weightsPath: fixture.weights.path,
        goldenPath: fixture.golden.path,
        environment: environment
    )
}

private func checkCorrectnessPreflight(
    _ fixture: PreflightFixture,
    environment: [String: String] = [:]
) throws -> BenchmarkPreflightReport {
    try BenchmarkPreflight.checkCorrectnessArtifacts(
        weightsPath: fixture.weights.path,
        goldenPath: fixture.golden.path,
        environment: environment
    )
}

private func makePreflightFixture(
    goldenContents: String? = nil,
    omitDenseTensorName: String? = nil,
    extraDenseTensor: TensorFixture? = nil
) throws -> PreflightFixture {
    let directory = try temporaryDirectory()
    let weights = directory.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)

    try qwenPreflightConfigJSON().write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    var denseTensors = requiredQwen35DenseTensorFixtures()
    if let omitDenseTensorName {
        denseTensors.removeAll { $0.name == omitDenseTensorName }
    }
    if let extraDenseTensor {
        denseTensors.append(extraDenseTensor)
    }
    let denseShard = "model-00001.safetensors"
    try writeSafetensors(weights.appendingPathComponent(denseShard), tensors: denseTensors)
    try writeIndex(
        weights.appendingPathComponent("model.safetensors.index.json"),
        tensors: denseTensors,
        shardName: denseShard
    )

    let golden = directory.appendingPathComponent("correctness_golden.json")
    try (goldenContents ?? validGoldenJSON()).write(to: golden, atomically: true, encoding: .utf8)

    return PreflightFixture(root: directory, weights: weights, golden: golden)
}

/// The pinned Qwen 3.6 4-bit runtime config as the transform writes it.
private func qwenPreflightConfigJSON() throws -> String {
    let root = pinnedRuntimeWorkerConfigurationObject()
    let data = try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
    return String(decoding: data, as: UTF8.self)
}

private func validGoldenJSON(
    correctnessPromptTokens: [Int] = Array(repeating: 1, count: MLXFastConstants.correctnessPromptTokens),
    benchmarkPromptTokens: [Int] = Array(repeating: 1, count: MLXFastConstants.benchmarkPrefillPromptTokens),
    correctnessGates: String? = nil
) -> String {
    let correctnessPrompt = arrayJSON(correctnessPromptTokens)
    let benchmarkPrompt = arrayJSON(benchmarkPromptTokens)
    let expected = arrayJSON(Array(repeating: 7, count: MLXFastConstants.correctnessSteps))
    let seed = arrayJSON(Array(benchmarkPromptTokens.prefix(MLXFastConstants.benchmarkDecodeSeedTokens)))
    let decode = arrayJSON(Array(repeating: 9, count: MLXFastConstants.benchmarkDecodeSteps))
    let gates = correctnessGates.map { ",\n  \"correctness_gates\": \($0)" } ?? ""
    return """
    {
      "version": 1,
      "cases": [
        {
          "name": "preflight",
          "prompt_tokens": \(correctnessPrompt),
          "expected_tokens": \(expected)
        }
      ],
      "benchmark": {
        "prefill_prompt_tokens": \(benchmarkPrompt),
        "expected_prefill_token": 8,
        "decode_seed_tokens": \(seed),
        "expected_decode_seed_token": 7,
        "expected_decode_tokens": \(decode)
      }\(gates)
    }
    """
}

private func correctnessOnlyGoldenJSON() -> String {
    let expected = arrayJSON(Array(repeating: 7, count: MLXFastConstants.correctnessSteps))
    return """
    {
      "version": 1,
      "cases": [
        {
          "name": "preflight",
          "prompt_tokens": \(arrayJSON(Array(repeating: 1, count: MLXFastConstants.correctnessPromptTokens))),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
}

/// Every tensor `Qwen35WeightLoader.validateRequiredMetadata` requires for the
/// frozen 64-layer hybrid tower: three linear-attention layers followed by one
/// full-attention layer, repeated 16 times. Real byte contents do not matter
/// for preflight, so the fixture writes sparse files with the checkpoint's
/// exact BF16 and affine-4 U32 metadata.
private func requiredQwen35DenseTensorFixtures() -> [TensorFixture] {
    let hidden = MLXFastConstants.hiddenSize
    let intermediate = MLXFastConstants.intermediateSize
    let vocab = MLXFastConstants.vocabSize
    let layers = MLXFastConstants.numHiddenLayers
    let heads = MLXFastConstants.attentionHeads
    let headDim = 256
    let kvHeads = 4
    let linearValueHeads = 48
    let linearKeyHeads = 16
    let linearValueHeadDim = 128
    let linearKeyHeadDim = 128
    let linearConvKernelDim = 4

    func packedCols(_ inFeatures: Int) -> Int {
        inFeatures / 8
    }

    func quantized(
        _ name: String,
        outFeatures: Int,
        inFeatures: Int
    ) -> [TensorFixture] {
        let companionShape = [outFeatures, inFeatures / 64]
        let baseName = String(name.dropLast(".weight".count))
        return [
            TensorFixture(
                name: name,
                dtype: "U32",
                shape: [outFeatures, packedCols(inFeatures)]
            ),
            TensorFixture(
                name: "\(baseName).scales",
                dtype: "BF16",
                shape: companionShape
            ),
            TensorFixture(
                name: "\(baseName).biases",
                dtype: "BF16",
                shape: companionShape
            ),
        ]
    }

    let linearKeySize = linearKeyHeads * linearKeyHeadDim
    let linearValueSize = linearValueHeads * linearValueHeadDim
    let linearConvSize = linearKeySize * 2 + linearValueSize
    let fullQuerySize = heads * headDim * 2
    let fullKVSize = kvHeads * headDim
    let fullOutputSize = heads * headDim

    var tensors: [TensorFixture] = []
    tensors += quantized(
        Qwen35WeightNames.embedTokens,
        outFeatures: vocab,
        inFeatures: hidden
    )
    tensors.append(
        TensorFixture(
            name: Qwen35WeightNames.finalNorm,
            dtype: "BF16",
            shape: [hidden]
        )
    )
    tensors += quantized(
        Qwen35WeightNames.lmHead,
        outFeatures: vocab,
        inFeatures: hidden
    )

    for layerIndex in 0..<layers {
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ] {
            tensors.append(
                TensorFixture(
                    name: Qwen35WeightNames.layer(layerIndex, suffix),
                    dtype: "BF16",
                    shape: [hidden]
                )
            )
        }

        for (suffix, output, input) in [
            ("gate_proj.weight", intermediate, hidden),
            ("up_proj.weight", intermediate, hidden),
            ("down_proj.weight", hidden, intermediate),
        ] {
            tensors += quantized(
                Qwen35WeightNames.mlp(layerIndex, suffix),
                outFeatures: output,
                inFeatures: input
            )
        }

        if layerIndex % 4 == 3 {
            for suffix in ["q_norm.weight", "k_norm.weight"] {
                tensors.append(
                    TensorFixture(
                        name: Qwen35WeightNames.fullAttention(
                            layerIndex,
                            suffix
                        ),
                        dtype: "BF16",
                        shape: [headDim]
                    )
                )
            }
            for (suffix, output, input) in [
                ("q_proj.weight", fullQuerySize, hidden),
                ("k_proj.weight", fullKVSize, hidden),
                ("v_proj.weight", fullKVSize, hidden),
                ("o_proj.weight", hidden, fullOutputSize),
            ] {
                tensors += quantized(
                    Qwen35WeightNames.fullAttention(layerIndex, suffix),
                    outFeatures: output,
                    inFeatures: input
                )
            }
        } else {
            for (suffix, shape) in [
                ("conv1d.weight", [linearConvSize, linearConvKernelDim, 1]),
                ("A_log", [linearValueHeads]),
                ("dt_bias", [linearValueHeads]),
                ("norm.weight", [linearValueHeadDim]),
            ] {
                tensors.append(
                    TensorFixture(
                        name: Qwen35WeightNames.linearAttention(
                            layerIndex,
                            suffix
                        ),
                        dtype: "BF16",
                        shape: shape
                    )
                )
            }
            for (suffix, output, input) in [
                ("in_proj_qkv.weight", linearConvSize, hidden),
                ("in_proj_z.weight", linearValueSize, hidden),
                ("in_proj_b.weight", linearValueHeads, hidden),
                ("in_proj_a.weight", linearValueHeads, hidden),
                ("out_proj.weight", hidden, linearValueSize),
            ] {
                tensors += quantized(
                    Qwen35WeightNames.linearAttention(layerIndex, suffix),
                    outFeatures: output,
                    inFeatures: input
                )
            }
        }
    }

    return tensors
}

/// Every tensor `LagunaWeightLoader.validateRequiredMetadata` requires for a
/// synthetic Laguna-shaped checkpoint built from the pinned geometry (40
/// layers, one full-attention layer per block of four, dense MLP at layer 0,
/// 256-expert MoE with a shared expert elsewhere, per-head attention
/// gating, untied lm_head). Real byte contents do not matter for these
/// preflight tests -- only declared dtype/shape -- so the shard is written
/// sparse: only routed/shared expert projections use Poolside's packed U32
/// NVFP4 layout with U8 group-16 scales; all other matrices are BF16.
private func requiredLagunaDenseTensorFixtures() -> [TensorFixture] {
    // Laguna's own frozen geometry, not MLXFastConstants: on the
    // qwen36-mtp-track branch those constants carry the Qwen 3.6 target
    // identity (64 layers, hidden 5120, vocab 248320), and building a
    // "Laguna" inventory from them produced a checkpoint that exists
    // nowhere -- 1,464 tensors at Qwen shapes -- which no Laguna contract
    // or preflight byte budget can describe.
    let hidden = LagunaConstants.hiddenSize
    let denseIntermediate = LagunaConstants.denseIntermediateSize
    let vocab = LagunaConstants.vocabSize
    let layers = LagunaConstants.numHiddenLayers
    let kvHeads = 8
    let headDim = 128
    let experts = 256
    let expertIntermediate = 512
    let sharedExpertIntermediate = 512
    let groupSize = 16

    var tensors: [TensorFixture] = []

    func appendBFloat16(_ name: String, shape: [Int]) {
        tensors.append(TensorFixture(name: name, dtype: "BF16", shape: shape))
    }

    func appendNVFP4(
        _ name: String,
        leadingShape: [Int],
        inputFeatures: Int
    ) {
        tensors.append(TensorFixture(
            name: name,
            dtype: "U32",
            shape: leadingShape + [inputFeatures * 4 / 32]
        ))
        let stem = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
        let companionShape = leadingShape + [inputFeatures / groupSize]
        tensors.append(TensorFixture(
            name: "\(stem).scales",
            dtype: "U8",
            shape: companionShape
        ))
    }

    appendBFloat16(LagunaWeightNames.embedTokens, shape: [vocab, hidden])
    appendBFloat16(LagunaWeightNames.finalNorm, shape: [hidden])
    appendBFloat16(LagunaWeightNames.lmHead, shape: [vocab, hidden])

    for layerIndex in 0..<layers {
        let layerHeads = layerIndex % 4 == 0 ? 48 : 64

        for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            tensors.append(TensorFixture(
                name: LagunaWeightNames.layer(layerIndex, suffix),
                dtype: "BF16",
                shape: [hidden]
            ))
        }

        appendBFloat16(
            LagunaWeightNames.attention(layerIndex, "q_proj.weight"),
            shape: [layerHeads * headDim, hidden]
        )
        for suffix in ["k_proj.weight", "v_proj.weight"] {
            appendBFloat16(
                LagunaWeightNames.attention(layerIndex, suffix),
                shape: [kvHeads * headDim, hidden]
            )
        }
        appendBFloat16(
            LagunaWeightNames.attention(layerIndex, "o_proj.weight"),
            shape: [hidden, layerHeads * headDim]
        )
        // Per-head attention gating: one gate per query head.
        appendBFloat16(
            LagunaWeightNames.attention(layerIndex, "g_proj.weight"),
            shape: [layerHeads, hidden]
        )
        for suffix in ["q_norm.weight", "k_norm.weight"] {
            tensors.append(TensorFixture(
                name: LagunaWeightNames.attention(layerIndex, suffix),
                dtype: "BF16",
                shape: [headDim]
            ))
        }

        if layerIndex == 0 {
            for suffix in ["gate_proj.weight", "up_proj.weight"] {
                appendBFloat16(
                    LagunaWeightNames.mlp(layerIndex, suffix),
                    shape: [denseIntermediate, hidden]
                )
            }
            appendBFloat16(
                LagunaWeightNames.mlp(layerIndex, "down_proj.weight"),
                shape: [hidden, denseIntermediate]
            )
        } else {
            appendBFloat16(
                LagunaWeightNames.mlp(layerIndex, "gate.weight"),
                shape: [experts, hidden]
            )
            tensors.append(TensorFixture(
                name: LagunaWeightNames.mlp(layerIndex, "gate.e_score_correction_bias"),
                dtype: "F32",
                shape: [experts]
            ))
            for suffix in ["switch_mlp.gate_proj.weight", "switch_mlp.up_proj.weight"] {
                appendNVFP4(
                    LagunaWeightNames.mlp(layerIndex, suffix),
                    leadingShape: [experts, expertIntermediate],
                    inputFeatures: hidden
                )
            }
            appendNVFP4(
                LagunaWeightNames.mlp(layerIndex, "switch_mlp.down_proj.weight"),
                leadingShape: [experts, hidden],
                inputFeatures: expertIntermediate
            )
            for suffix in ["shared_expert.gate_proj.weight", "shared_expert.up_proj.weight"] {
                appendNVFP4(
                    LagunaWeightNames.mlp(layerIndex, suffix),
                    leadingShape: [sharedExpertIntermediate],
                    inputFeatures: hidden
                )
            }
            appendNVFP4(
                LagunaWeightNames.mlp(layerIndex, "shared_expert.down_proj.weight"),
                leadingShape: [hidden],
                inputFeatures: sharedExpertIntermediate
            )
        }
    }

    return tensors
}

/// Laguna-shaped weights directory for the tests that still exercise the
/// Poolside artifact contract and the Laguna weight loader. The shared
/// `makePreflightFixture` now writes the Qwen 3.6 artifact the harness drives,
/// so those tests build their own tree here rather than reading a config that
/// describes a different checkpoint.
private func makeLagunaWeightsFixture(
    omitDenseTensorName: String? = nil,
    extraDenseTensor: TensorFixture? = nil
) throws -> PreflightFixture {
    let directory = try temporaryDirectory()
    let weights = directory.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)

    let configData = try JSONSerialization.data(
        withJSONObject: pinnedLagunaConfigObject(),
        options: [.sortedKeys]
    )
    try configData.write(to: weights.appendingPathComponent("config.json"))

    var denseTensors = requiredLagunaDenseTensorFixtures()
    if let omitDenseTensorName {
        denseTensors.removeAll { $0.name == omitDenseTensorName }
    }
    if let extraDenseTensor {
        denseTensors.append(extraDenseTensor)
    }
    let denseShard = "model-00001.safetensors"
    try writeSafetensors(weights.appendingPathComponent(denseShard), tensors: denseTensors)
    try writeIndex(
        weights.appendingPathComponent("model.safetensors.index.json"),
        tensors: denseTensors,
        shardName: denseShard
    )

    let golden = directory.appendingPathComponent("correctness_golden.json")
    try validGoldenJSON().write(to: golden, atomically: true, encoding: .utf8)

    return PreflightFixture(root: directory, weights: weights, golden: golden)
}

private func writeIndex(_ path: URL, tensors: [TensorFixture], shardName: String) throws {
    let entries = tensors.map { #""\#($0.name)": "\#(shardName)""# }.joined(separator: ",")
    try """
    {
      "weight_map": {
        \(entries)
      }
    }
    """.write(to: path, atomically: true, encoding: .utf8)
}

@Test
func localDecodeProgressIntervalUsesExpectedCadence() {
    // Tiny synthetic runs get a running-number line on every token.
    #expect(QwenRuntime.localIterateDecodeProgressInterval(totalDecodeSteps: 16, timingRepeats: 1) == 1)
    #expect(QwenRuntime.localIterateDecodeProgressInterval(totalDecodeSteps: 32, timingRepeats: 1) == 1)
    // The 128-step local-iterate window and 1023-step local-submit window keep
    // the historical 8-step cadence.
    #expect(QwenRuntime.localIterateDecodeProgressInterval(
        totalDecodeSteps: MLXFastConstants.localIterateBenchmarkDecodeSteps,
        timingRepeats: 1
    ) == 8)
    #expect(QwenRuntime.localIterateDecodeProgressInterval(totalDecodeSteps: 1023, timingRepeats: 1) == 8)
    // Multi-repeat runs keep the sparser 64-step cadence.
    #expect(QwenRuntime.localIterateDecodeProgressInterval(totalDecodeSteps: 512, timingRepeats: 4) == 64)
}

@Test
func localIterateProjectedDecodeSecondsPerTokenConvergesToChargedMean() {
    // Mid-run: charged 20s so far (18s seed + 2s steps), 2 of 16 tokens done at
    // 1s/step mean -> project 14 more step-seconds on top of the charged 20.
    let projected = QwenRuntime.localIterateProjectedDecodeSecondsPerToken(
        chargedSecondsSoFar: 20,
        stepOnlySecondsSoFar: 2,
        decodedTokens: 2,
        totalDecodeTokens: 16
    )
    #expect(abs(projected - (20.0 + 14.0) / 16.0) < 1e-12)

    // Final token: exactly the charged mean the score payload will report.
    let final = QwenRuntime.localIterateProjectedDecodeSecondsPerToken(
        chargedSecondsSoFar: 32,
        stepOnlySecondsSoFar: 14,
        decodedTokens: 16,
        totalDecodeTokens: 16
    )
    #expect(final == 2.0)

    // Guards.
    #expect(
        QwenRuntime.localIterateProjectedDecodeSecondsPerToken(
            chargedSecondsSoFar: 1,
            stepOnlySecondsSoFar: 1,
            decodedTokens: 0,
            totalDecodeTokens: 16
        ) == 0
    )
}

@Test
func localIterateLiveDecodeStatusIncludesProjectedSpeedupAndScore() {
    let status = QwenRuntime.localIterateLiveDecodeStatus(
        lastStepSeconds: 0.9,
        chargedSecondsSoFar: 20,
        stepOnlySecondsSoFar: 2,
        decodedTokens: 2,
        totalDecodeTokens: 16,
        prefillSecondsPerToken: 0.05
    )
    #expect(status.contains("last_step_seconds=0.900000"))
    #expect(status.contains("mean_step_seconds=1.000000"))
    // 14 remaining tokens at 1s mean -> 14s ETA.
    #expect(status.contains("decode_eta_seconds=14.0"))
    #expect(status.contains("projected_decode_seconds_per_token="))
    #expect(status.contains("projected_decode_speedup="))
    #expect(status.contains("projected_score="))
    // The RAM-resident dense runtime reports no expert-bandwidth live fields.
    #expect(!status.contains("expert_gb_per_token="))
    #expect(!status.contains("expert_hit_rate="))

    // The final step has no ETA.
    let finalStep = QwenRuntime.localIterateLiveDecodeStatus(
        lastStepSeconds: 1,
        chargedSecondsSoFar: 32,
        stepOnlySecondsSoFar: 14,
        decodedTokens: 16,
        totalDecodeTokens: 16,
        prefillSecondsPerToken: 0.05
    )
    #expect(!finalStep.contains("decode_eta_seconds="))

    // Before prefill has a positive measurement there is no score estimate.
    let withoutPrefill = QwenRuntime.localIterateLiveDecodeStatus(
        lastStepSeconds: 0.9,
        chargedSecondsSoFar: 20,
        stepOnlySecondsSoFar: 2,
        decodedTokens: 2,
        totalDecodeTokens: 16,
        prefillSecondsPerToken: nil
    )
    #expect(withoutPrefill.contains("projected_decode_speedup="))
    #expect(!withoutPrefill.contains("projected_score="))

    #expect(
        QwenRuntime.localIterateLiveDecodeStatus(
            lastStepSeconds: 0,
            chargedSecondsSoFar: 0,
            stepOnlySecondsSoFar: 0,
            decodedTokens: 0,
            totalDecodeTokens: 16,
            prefillSecondsPerToken: nil
        ).isEmpty
    )
}

@Test
func workerStderrDrainForwardsRedactedLinesAndKeepsTailForDiagnostics() throws {
    final class LineBox: @unchecked Sendable {
        private let lock = NSLock()
        private var lines: [String] = []
        func append(_ line: String) {
            lock.lock()
            defer { lock.unlock() }
            lines.append(line)
        }
        func snapshot() -> [String] {
            lock.lock()
            defer { lock.unlock() }
            return lines
        }
    }

    let pipe = Pipe()
    let box = LineBox()
    let drain = WorkerStderrDrain(
        handle: pipe.fileHandleForReading,
        emit: { box.append($0) }
    )

    // Two complete lines (one needing token redaction) and one unterminated
    // partial line that must still be flushed at EOF.
    try pipe.fileHandleForWriting.write(contentsOf: Data("model debug: layer 3 routed\n".utf8))
    try pipe.fileHandleForWriting.write(contentsOf: Data("expected 5 actual 7\npartial".utf8))
    try pipe.fileHandleForWriting.close()

    let tail = drain.drainedOutput(timeoutSeconds: 5)
    let emitted = box.snapshot()

    #expect(emitted == [
        "mlxfast-worker: model debug: layer 3 routed\n",
        "mlxfast-worker: token-validation-failed\n",
        "mlxfast-worker: partial\n",
    ])
    // The diagnostic tail keeps the RAW content (workerExitDiagnostic applies
    // its own whole-blob sanitization, unchanged from before).
    #expect(tail.contains("model debug: layer 3 routed"))
    #expect(tail.contains("expected 5 actual 7"))
    #expect(tail.contains("partial"))

    // A second read must not block or lose the tail.
    #expect(drain.drainedOutput(timeoutSeconds: 1).contains("partial"))
}

@Test
func workerStderrDrainCapsRetainedTail() throws {
    let pipe = Pipe()
    let drain = WorkerStderrDrain(
        handle: pipe.fileHandleForReading,
        emit: { _ in }
    )

    let filler = String(repeating: "x", count: 1024)
    for index in 0..<128 {
        try pipe.fileHandleForWriting.write(contentsOf: Data("line-\(index) \(filler)\n".utf8))
    }
    try pipe.fileHandleForWriting.write(contentsOf: Data("final-marker\n".utf8))
    try pipe.fileHandleForWriting.close()

    let tail = drain.drainedOutput(timeoutSeconds: 5)
    #expect(tail.utf8.count <= WorkerStderrDrain.tailByteLimit + 16)
    #expect(tail.contains("final-marker"))
    #expect(!tail.contains("line-0 "))
}

@Test
func workerStderrDrainCapsUnterminatedLine() throws {
    let pipe = Pipe()
    let drain = WorkerStderrDrain(
        handle: pipe.fileHandleForReading,
        emit: { _ in }
    )
    try pipe.fileHandleForWriting.write(
        contentsOf: Data(repeating: 0x78, count: WorkerStderrDrain.tailByteLimit * 3)
    )
    try pipe.fileHandleForWriting.close()

    let tail = drain.drainedOutput(timeoutSeconds: 5)
    #expect(tail.contains(WorkerStderrDrain.truncatedLine))
    #expect(tail.utf8.count <= WorkerStderrDrain.tailByteLimit)
}

@Test
func bufferedFileLineReaderPreservesBufferedLinesAndEOFFragment() throws {
    let pipe = Pipe()
    let reader = BufferedFileLineReader(handle: pipe.fileHandleForReading, maximumLineByteCount: 32)
    try pipe.fileHandleForWriting.write(contentsOf: Data("first\nsecond\nthird".utf8))
    try pipe.fileHandleForWriting.close()

    #expect(try reader.readLine() == Data("first".utf8))
    #expect(try reader.readLine() == Data("second".utf8))
    #expect(try reader.readLine() == Data("third".utf8))
    #expect(try reader.readLine() == nil)
}

@Test
func bufferedFileLineReaderRejectsOversizedLine() throws {
    let pipe = Pipe()
    let reader = BufferedFileLineReader(handle: pipe.fileHandleForReading, maximumLineByteCount: 8)
    try pipe.fileHandleForWriting.write(contentsOf: Data("123456789\n".utf8))
    try pipe.fileHandleForWriting.close()

    #expect(throws: MLXFastError.self) {
        _ = try reader.readLine()
    }
}

@Test
func runtimeWorkerClientTimesOutWaitingForHello() throws {
    // The fixture ignores TERM to force the client's SIGKILL escalation, but
    // it must NOT live forever: an interrupted test run cannot reap it, and
    // unbounded `while :; do :; done` fixtures have been observed lingering
    // as ppid-1 orphans burning a CPU core for days. Sleep-loop for a
    // bounded lifetime instead (worst case ~2 minutes if orphaned).
    let executable = try makeRuntimeWorkerScript("""
    #!/bin/sh
    trap '' TERM
    \(boundedFixtureIdleLoop)
    """)
    defer { try? FileManager.default.removeItem(at: executable.deletingLastPathComponent()) }
    let start = Date()
    var message = ""
    do {
        _ = try RuntimeWorkerClient(
            options: shortRuntimeWorkerOptions(executable: executable),
            weightsPath: executable.deletingLastPathComponent().path
        )
    } catch {
        message = String(describing: error)
    }
    #expect(message.contains("timed out waiting for protocol hello"))
    #expect(Date().timeIntervalSince(start) < 3)
}

@Test
func runtimeWorkerWatchdogAtomicCancellationDisarmsTimer() {
    let watchdog = RuntimeWorkerWatchdog(
        process: Process(),
        timeoutSeconds: 0.05,
        terminationGraceSeconds: 0
    )

    #expect(!watchdog.cancelAndReturnDidFire())
    Thread.sleep(forTimeInterval: 0.1)
    #expect(!watchdog.cancelAndReturnDidFire())
}

@Test
func runtimeWorkerClientCancelsSuccessfulRequestWatchdogs() throws {
    // TIMING-SENSITIVE BY CONSTRUCTION, so give the fake worker room. This test
    // asserts that a SUCCEEDED request cancels its watchdog, which it proves by
    // sleeping 0.25s -- 2.5x the 0.1s request timeout -- between two requests and
    // requiring the second to still be served. Under full-suite parallelism the
    // 2s hello timeout was the part that lost the race (observed: 12.1s wall,
    // versus 0.41s isolated), because a `/bin/sh` spawn competing with the rest
    // of the suite can take longer than that to produce its first line.
    //
    // Only the HELLO budget is widened. The request timeout, the sleep and the
    // ordering are untouched, so what the test asserts is unchanged: a stale
    // watchdog would still fire during the sleep and fail the second request no
    // matter how long the handshake took.
    let executable = try makeRuntimeWorkerScript("""
    #!/bin/sh
    printf '%s\\n' '{"id":0,"nonce":"test-nonce","ok":true}'
    IFS= read -r first_request || exit 0
    printf '%s\\n' '{"id":1,"nonce":"test-nonce","ok":true}'
    IFS= read -r second_request || exit 0
    printf '%s\\n' '{"id":2,"nonce":"test-nonce","ok":true}'
    IFS= read -r final_request || exit 0
    """)
    defer { try? FileManager.default.removeItem(at: executable.deletingLastPathComponent()) }
    let client = try RuntimeWorkerClient(
        options: RuntimeWorkerOptions(
            executablePath: executable.path,
            helloTimeoutSeconds: 30,
            requestTimeoutSeconds: 0.1,
            shutdownTimeoutSeconds: 0.1,
            terminationGraceSeconds: 0.05
        ),
        weightsPath: executable.deletingLastPathComponent().path
    )
    defer { client.close() }

    _ = try client.prefill(promptTokens: [1])
    // A stale watchdog from the first request would terminate the worker while
    // it is idle and make this second request fail before receiving id 2.
    Thread.sleep(forTimeInterval: 0.25)
    _ = try client.prefill(promptTokens: [2])
}

/// The DFlash reference-rows request must actually carry the verify block on the
/// wire (contract Amendment 21). It is the only field that lets the reference
/// reach a REJECTED row, and a client that accepted the argument and then dropped
/// it produced exactly one symptom -- the reference "did not return the replay it
/// was asked for" -- with no compile error anywhere. Assert the bytes.
@Test
func dflashReferenceRowsRequestCarriesTheVerifyBlockOnTheWire() throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let requestLog = directory.appendingPathComponent("request.json")
    let executable = try makeRuntimeWorkerScript("""
    #!/bin/sh
    printf '%s\\n' '{"id":0,"nonce":"test-nonce","ok":true}'
    IFS= read -r request || exit 0
    printf '%s' "$request" > "\(requestLog.path)"
    printf '%s\\n' '{"id":1,"nonce":"test-nonce","ok":true,"reference_verify_top2_tokens":[[5,6],[7,8]],"reference_verify_top2_logits":[[2.5,1.5],[2.0,1.0]],"draft_tokens":[11,12]}'
    IFS= read -r final || exit 0
    """)
    let client = try RuntimeWorkerClient(
        options: RuntimeWorkerOptions(
            executablePath: executable.path,
            helloTimeoutSeconds: 2,
            requestTimeoutSeconds: 5,
            shutdownTimeoutSeconds: 0.5,
            terminationGraceSeconds: 0.25
        ),
        weightsPath: directory.path
    )
    defer { client.close() }

    let response = try client.dflashReferenceRows(
        prefixTokens: [1, 2, 3, 4],
        seedTokenCount: 2,
        startOffset: 2,
        rowCount: 1,
        widestFrame: 2,
        verifyBlockTokens: [3, 99]
    )
    let wire = try String(contentsOf: requestLog, encoding: .utf8)
    #expect(
        wire.contains("\"verify_block_tokens\":[3,99]"),
        "the verify block must reach the reference worker: \(wire)"
    )
    // And the reference's answer must decode back into the parent's fields.
    #expect(response.referenceVerifyTop2Tokens == [[5, 6], [7, 8]])
    #expect(response.referenceVerifyTop2Logits == [[2.5, 1.5], [2.0, 1.0]])
    #expect(response.draftTokens == [11, 12])
}

@Test
func runtimeWorkerClientTimesOutStalledRequest() throws {
    // Bounded stall (see runtimeWorkerClientTimesOutWaitingForHello): the
    // client SIGKILLs it within the test, and an orphaned copy self-expires.
    let executable = try makeRuntimeWorkerScript("""
    #!/bin/sh
    trap '' TERM
    printf '%s\\n' '{"id":0,"nonce":"test-nonce","ok":true}'
    IFS= read -r request || exit 0
    \(boundedFixtureIdleLoop)
    """)
    defer { try? FileManager.default.removeItem(at: executable.deletingLastPathComponent()) }
    let client = try RuntimeWorkerClient(
        options: RuntimeWorkerOptions(
            executablePath: executable.path,
            helloTimeoutSeconds: 2,
            requestTimeoutSeconds: 0.1,
            shutdownTimeoutSeconds: 0.1,
            terminationGraceSeconds: 0.05
        ),
        weightsPath: executable.deletingLastPathComponent().path
    )
    defer { client.close() }
    let start = Date()
    var message = ""
    do {
        _ = try client.prefill(promptTokens: [1])
    } catch {
        message = String(describing: error)
    }
    #expect(message.contains("timed out handling request prefill"))
    #expect(Date().timeIntervalSince(start) < 3)
}

@Test
func runtimeWorkerClientCloseEscalatesPastIgnoredTerminate() throws {
    // Bounded stall (see runtimeWorkerClientTimesOutWaitingForHello): the
    // client SIGKILLs it within the test, and an orphaned copy self-expires.
    let executable = try makeRuntimeWorkerScript("""
    #!/bin/sh
    trap '' TERM
    printf '%s\\n' '{"id":0,"nonce":"test-nonce","ok":true}'
    \(boundedFixtureIdleLoop)
    """)
    defer { try? FileManager.default.removeItem(at: executable.deletingLastPathComponent()) }
    let client = try RuntimeWorkerClient(
        options: RuntimeWorkerOptions(
            executablePath: executable.path,
            helloTimeoutSeconds: 2,
            requestTimeoutSeconds: 0.1,
            shutdownTimeoutSeconds: 0.1,
            terminationGraceSeconds: 0.05
        ),
        weightsPath: executable.deletingLastPathComponent().path
    )
    let start = Date()
    client.close()
    #expect(Date().timeIntervalSince(start) < 3)
}

@Test
func runtimeWorkerClientDrainsLargeStderrBeforeHelloWhenForwardingIsOff() throws {
    let executable = try makeRuntimeWorkerScript("""
    #!/bin/sh
    /usr/bin/yes x | /usr/bin/head -c 196608 | /usr/bin/tr -d '\\n' >&2
    printf '%s\\n' '{"id":0,"nonce":"test-nonce","ok":true}'
    IFS= read -r request || exit 0
    """)
    defer { try? FileManager.default.removeItem(at: executable.deletingLastPathComponent()) }
    let client = try RuntimeWorkerClient(
        options: RuntimeWorkerOptions(
            executablePath: executable.path,
            helloTimeoutSeconds: 2,
            requestTimeoutSeconds: 0.1,
            shutdownTimeoutSeconds: 0.1,
            terminationGraceSeconds: 0.05
        ),
        weightsPath: executable.deletingLastPathComponent().path
    )
    client.close()
}

@Test
func runtimeWorkerClientRejectsNonfiniteTimeoutsBeforeLaunch() throws {
    let options = RuntimeWorkerOptions(
        executablePath: "/does/not/exist",
        helloTimeoutSeconds: .infinity
    )
    #expect(throws: MLXFastError.self) {
        _ = try RuntimeWorkerClient(options: options, weightsPath: "/does/not/exist")
    }
}

@Test
func runtimeWorkerStartsTheOrphanReaperBeforeLoadingTheModel() throws {
    // Source-level guard: runWorker must arm the reaper BEFORE any of the
    // model-loading work, or the load window (the exact window stdin-EOF
    // cannot cover) is left unprotected.
    let worker = try String(
        contentsOfFile: "Sources/MLXFastHarness/QwenRuntimeWorker.swift",
        encoding: .utf8
    )
    let reaperCall = try #require(worker.range(of: "startRuntimeWorkerOrphanReaper()"))
    let protocolIO = try #require(worker.range(of: "RuntimeWorkerProtocolIO.isolatingStandardIO()"))
    let weightCache = try #require(worker.range(of: "Qwen35RuntimeWeightCache(loader:"))
    #expect(reaperCall.lowerBound < protocolIO.lowerBound)
    #expect(reaperCall.lowerBound < weightCache.lowerBound)
    #expect(worker.contains("getppid() == 1"))
}

@Test
func localIteratePrefillStatusReportsPerTokenAndSpeedup() {
    let status = QwenRuntime.localIteratePrefillStatus(
        elapsedSeconds: 51.2,
        promptTokens: 512
    )
    #expect(status.contains("seconds=51.2"))
    #expect(status.contains("seconds_per_token=0.100000"))
    let expectedSpeedup = MLXFastConstants.officialBaselinePrefillSecondsPerToken / 0.1
    #expect(status.contains("prefill_speedup=\(String(format: "%.3f", expectedSpeedup))"))

    // Zero-duration or zero-token inputs fall back to the plain seconds field.
    #expect(
        QwenRuntime.localIteratePrefillStatus(elapsedSeconds: 0, promptTokens: 512)
            == "seconds=0.0"
    )
}

@Test
func localIterateSummaryEmitsSpeedupsAndEstimatedScore() {
    let timing = QwenRuntime.LocalIterateTimingResult(
        correctness: QwenRuntime.localIterateCorrectnessReport(
            passed: true,
            checkedSteps: 18,
            caseCount: 1,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil,
            goldenHash: "hash",
            expertStats: .zero,
            error: "",
            modeName: "local-iterate"
        ),
        prefillSecondsPerToken: MLXFastConstants.officialBaselinePrefillSecondsPerToken / 2,
        decode: QwenRuntime.DecodeMeasurement(
            secondsPerToken: MLXFastConstants.officialBaselineDecodeSecondsPerToken / 2,
            bandwidthGBPerToken: 0,
            bandwidthSource: "ram_resident_model"
        ),
        expertStats: .zero,
        peakRamGB: 24.5
    )

    var lines: [String] = []
    QwenRuntime.emitLocalIterateSummary(
        modeName: "local-iterate",
        timing: timing,
        progress: { lines.append($0) }
    )

    let joined = lines.joined(separator: "\n")
    #expect(joined.contains("local-iterate summary prefill_seconds_per_token="))
    #expect(joined.contains("prefill_speedup=2.000"))
    #expect(joined.contains("local-iterate summary decode_seconds_per_token="))
    #expect(joined.contains("decode_speedup=2.000"))
    // 2x on both axes -> estimated score 2 under decode^0.75 * prefill^0.25.
    #expect(joined.contains("est_score=2.000"))
    #expect(joined.contains("published as this run's local estimated score, not a ranked score"))
    #expect(joined.contains("decode_bandwidth_gb_per_token=0"))
    #expect(joined.contains("peak_ram_gb=24.500"))
    #expect(!joined.contains("expert_hit_rate="))
}

// The Yukon participant CLI (`mlxfast run`) executes benchmarkCommand and then
// validates the contract scorePath as `{ "score": <finite number>, ... }`;
// `score: null` fails its schema. Local modes therefore publish the estimated
// score (the same decode_speedup^0.75 * prefill_speedup^0.25 estimate the
// summary prints) as a numeric, CLI-usable score.
@Test
func localIterateScorePublishesCLIUsableEstimatedScore() throws {
    let payload = QwenRuntime.localIterateScore(
        peakRamGB: 24.5,
        bandwidthGBPerToken: 0,
        decodeSecondsPerToken: MLXFastConstants.officialBaselineDecodeSecondsPerToken / 2,
        prefillSecondsPerToken: MLXFastConstants.officialBaselinePrefillSecondsPerToken / 2,
        wallSeconds: 10,
        validationSeconds: 1,
        correctnessSeconds: 5,
        timedSeconds: 5,
        correctness: QwenRuntime.localIterateCorrectnessReport(
            passed: true,
            checkedSteps: 18,
            caseCount: 1,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil,
            goldenHash: "hash",
            expertStats: .zero,
            error: "",
            modeName: "local-iterate"
        ),
        expertStats: .zero,
        bandwidthSource: "ram_resident_model",
        weightsDigest: nil,
        runtime: "swift-local-iterate"
    )

    // 2x on both axes -> estimated score 2 under decode^0.75 * prefill^0.25.
    let estimated = try #require(payload.score)
    #expect(abs(estimated - 2) < 1e-9)
    #expect(payload.passed)
    // The runtime label is the local-mode marker that keeps the estimate
    // clearly distinguishable from a ranked payload (runtime == "swift").
    #expect(payload.metrics.runtime == "swift-local-iterate")

    // The sealed scorePath JSON must carry the score as a finite JSON number
    // -- the exact thing the CLI's schema checks -- and not null.
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let path = directory.appendingPathComponent("score.json").path
    try writeScorePayload(payload, to: path)
    let raw = try String(contentsOfFile: path, encoding: .utf8)
    #expect(!raw.contains("\"score\" : null"))
    let object = try #require(
        try JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any]
    )
    let rawScore = try #require(object["score"] as? Double)
    #expect(rawScore.isFinite)
    let decoded = try JSONDecoder().decode(ScorePayload.self, from: Data(raw.utf8))
    #expect(decoded.score == estimated)
}

/// The MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT override lets a local run publish its
/// timing estimate after a teacher-forced mismatch (the documented non-M5
/// near-tie divergence). The payload must stay CLI-consumable AND
/// self-describing: passed/score keep the run usable, while
/// passed_correctness, the diverging tokens, and metrics.error record exactly
/// what was not verified. Silently reporting passed_correctness=true here
/// would turn the escape hatch into a way to hide a real regression.
@Test
func localIterateScoreUnderGoldenDriftStaysUsableButRecordsTheDivergence() throws {
    let payload = QwenRuntime.localIterateScore(
        peakRamGB: 24.5,
        bandwidthGBPerToken: 0,
        decodeSecondsPerToken: MLXFastConstants.officialBaselineDecodeSecondsPerToken,
        prefillSecondsPerToken: MLXFastConstants.officialBaselinePrefillSecondsPerToken,
        wallSeconds: 10,
        validationSeconds: 1,
        correctnessSeconds: 5,
        timedSeconds: 5,
        correctness: QwenRuntime.localIterateCorrectnessReport(
            passed: false,
            checkedSteps: 7,
            caseCount: 1,
            firstFailingStep: 6,
            expectedToken: 1234,
            actualToken: 4321,
            goldenHash: "hash",
            expertStats: .zero,
            error: "local-iterate teacher-forced token mismatch",
            modeName: "local-iterate"
        ),
        expertStats: .zero,
        bandwidthSource: "ram_resident_model",
        weightsDigest: nil,
        runtime: "swift-local-iterate"
    )

    // Usable: the Yukon CLI rejects score:null, so the estimate must survive.
    let estimated = try #require(payload.score)
    #expect(estimated.isFinite)
    #expect(payload.passed)
    // Honest: the divergence is recorded, not laundered.
    #expect(payload.metrics.passedCorrectness == false)
    #expect(payload.metrics.firstFailingStep == 6)
    #expect(payload.metrics.expectedToken == 1234)
    #expect(payload.metrics.actualToken == 4321)
    #expect(payload.metrics.error.contains(QwenRuntime.localGoldenDriftEnvironmentName))
    #expect(payload.metrics.error.contains("NOT verified"))
}

@Test
func localIterateScoreFailsForUnusableTimings() {
    // Zero/invalid timings make the estimate non-finite. The payload must be a
    // failed run, not a passing result that merely omits its score.
    let payload = QwenRuntime.localIterateScore(
        peakRamGB: 0,
        bandwidthGBPerToken: 0,
        decodeSecondsPerToken: 0,
        prefillSecondsPerToken: 0,
        wallSeconds: 0,
        validationSeconds: 0,
        correctnessSeconds: 0,
        timedSeconds: 0,
        correctness: QwenRuntime.localIterateCorrectnessReport(
            passed: true,
            checkedSteps: 18,
            caseCount: 1,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil,
            goldenHash: "hash",
            expertStats: .zero,
            error: "",
            modeName: "local-iterate"
        ),
        expertStats: .zero,
        bandwidthSource: "ram_resident_model",
        weightsDigest: nil,
        runtime: "swift-local-iterate"
    )

    #expect(payload.score == nil)
    #expect(!payload.passed)
    #expect(payload.metrics.passedCorrectness)
    #expect(payload.metrics.error.contains("timing metrics must be finite and positive"))
}

@Test
func localIterateScoreSanitizesNonfiniteFailureMetricsForJSON() throws {
    let payload = QwenRuntime.localIterateScore(
        peakRamGB: .nan,
        bandwidthGBPerToken: .infinity,
        decodeSecondsPerToken: .nan,
        prefillSecondsPerToken: -.infinity,
        wallSeconds: -.infinity,
        validationSeconds: .nan,
        correctnessSeconds: .infinity,
        timedSeconds: -1,
        correctness: QwenRuntime.localIterateCorrectnessReport(
            passed: true,
            checkedSteps: 18,
            caseCount: 1,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil,
            goldenHash: "hash",
            expertStats: .zero,
            error: "",
            modeName: "local-iterate"
        ),
        expertStats: .zero,
        bandwidthSource: "ram_resident_model",
        weightsDigest: nil,
        runtime: "swift-local-iterate"
    )

    #expect(payload.score == nil)
    #expect(!payload.passed)
    #expect(payload.metrics.peakRamGB == 0)
    #expect(payload.metrics.bandwidthGBPerToken == 0)
    #expect(payload.metrics.decodeSecondsPerToken == 0)
    #expect(payload.metrics.prefillSecondsPerToken == 0)
    #expect(payload.metrics.benchmarkWallSeconds == 0)
    #expect(payload.metrics.preflightSeconds == 0)
    #expect(payload.metrics.correctnessSeconds == 0)
    #expect(payload.metrics.timedBenchmarkSeconds == 0)

    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let path = directory.appendingPathComponent("score.json").path
    try writeScorePayload(payload, to: path)
    let encoded = try Data(contentsOf: URL(fileURLWithPath: path))
    let encodedText = String(decoding: encoded, as: UTF8.self).lowercased()
    #expect(!encodedText.contains("nan"))
    #expect(!encodedText.contains("infinity"))
    let decoded = try JSONDecoder().decode(
        ScorePayload.self,
        from: encoded
    )
    #expect(decoded.score == nil)
    #expect(!decoded.passed)
    #expect(decoded.metrics.peakRamGB == 0)
    #expect(decoded.metrics.bandwidthGBPerToken == 0)
    #expect(decoded.metrics.decodeSecondsPerToken == 0)
    #expect(decoded.metrics.prefillSecondsPerToken == 0)
    #expect(decoded.metrics.benchmarkWallSeconds == 0)
    #expect(decoded.metrics.preflightSeconds == 0)
    #expect(decoded.metrics.correctnessSeconds == 0)
    #expect(decoded.metrics.timedBenchmarkSeconds == 0)
}

// The immediate token-mismatch report carries the non-M5 caveat: the public
// goldens are M5-generated greedy continuations, so near-tie argmaxes diverge
// deterministically on other Apple Silicon generations even for a correct
// build. Messaging only -- the mismatch still fails the gate, and the line
// stays redacted (no expected/actual token values in the progress stream).
@Test
func firstTokenMismatchReportIncludesNonM5GoldenCaveat() {
    var lines: [String] = []
    QwenRuntime.reportFirstTokenMismatch(
        { lines.append($0) },
        modeName: "local-iterate",
        checkedStep: 17
    )

    #expect(lines.count == 2)
    #expect(lines[0].contains("local-iterate correctness FAIL first token mismatch at checked_step=17"))
    #expect(lines[1].contains("local-iterate note: the public goldens are M5-generated"))
    #expect(lines[1].contains("deterministic near-tie token mismatch is expected for a correct build"))
    #expect(lines[1].contains("Correctness fixtures are M5-generated"))
    #expect(lines[1].contains("the ranked M5 runner is the source of truth"))
}

// A local run whose public-gate correctness check failed but whose timings
// were measured (the deterministic non-M5 near-tie divergence is exactly this
// shape) must still publish the finite estimated score: the Yukon CLI
// validates scorePath as `{ "score": <finite number>, ... }`, and score: null
// used to make `mlxfast run` error at the contract layer on every non-M5 box.
// passed stays false and every failure field survives, so the failure remains
// unmistakable.
@Test
func localFailedRunWithMeasuredTimingsStillPublishesNumericEstimatedScore() throws {
    let report = CorrectnessReport(
        passed: false,
        checkedSteps: 17,
        caseCount: 1,
        firstFailingCase: "local-iterate",
        firstFailingStep: 16,
        expectedToken: 236761,
        actualToken: 618,
        goldenHash: "golden",
        error: "local-iterate teacher-forced token mismatch"
    )
    let failed = QwenRuntime.failedScore(
        error: "local-iterate teacher-forced token mismatch",
        correctness: report,
        passedCorrectness: false,
        expectedToken: report.expectedToken,
        actualToken: report.actualToken,
        decodeSecondsPerToken: MLXFastConstants.officialBaselineDecodeSecondsPerToken / 2,
        prefillSecondsPerToken: MLXFastConstants.officialBaselinePrefillSecondsPerToken / 2,
        runtime: "swift-local-iterate"
    )

    var lines: [String] = []
    let payload = QwenRuntime.localModeFailedPayloadWithEstimatedScore(
        failed,
        modeName: "local-iterate",
        progress: { lines.append($0) }
    )

    // 2x on both axes -> estimated score 2 under decode^0.75 * prefill^0.25.
    let estimated = try #require(payload.score)
    #expect(abs(estimated - 2) < 1e-9)
    #expect(!payload.passed)
    #expect(!payload.metrics.passedCorrectness)
    #expect(payload.metrics.error == "local-iterate teacher-forced token mismatch")
    #expect(payload.metrics.firstFailingStep == 16)
    #expect(payload.metrics.expectedToken == 236761)
    #expect(payload.metrics.actualToken == 618)
    #expect(payload.metrics.runtime == "swift-local-iterate")
    #expect(lines.joined(separator: "\n").contains("publishing est_score=2.000 with passed=false"))

    // The sealed scorePath JSON carries the score as a finite JSON number --
    // the exact thing the CLI's schema checks -- while staying a failed run.
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let path = directory.appendingPathComponent("score.json").path
    try writeScorePayload(payload, to: path)
    let raw = try String(contentsOfFile: path, encoding: .utf8)
    #expect(!raw.contains("\"score\" : null"))
    let decoded = try JSONDecoder().decode(ScorePayload.self, from: Data(raw.utf8))
    #expect(decoded.score == estimated)
    #expect(!decoded.passed)
}

// Failure payloads without measured timings (validation errors, missing
// artifacts) cannot carry a meaningful estimate and keep score: null; passing
// payloads are never touched.
@Test
func localFailedPayloadWithoutTimingsKeepsNullScore() {
    let noTimings = QwenRuntime.failedScore(
        error: "local-iterate public golden must contain at least one case",
        correctness: nil,
        passedCorrectness: false,
        runtime: "swift-local-iterate"
    )
    var lines: [String] = []
    let unchanged = QwenRuntime.localModeFailedPayloadWithEstimatedScore(
        noTimings,
        modeName: "local-iterate",
        progress: { lines.append($0) }
    )
    #expect(unchanged.score == nil)
    #expect(!unchanged.passed)
    #expect(lines.isEmpty)

    let passing = ScorePayload(score: 1.5, passed: true, metrics: noTimings.metrics)
    let stillPassing = QwenRuntime.localModeFailedPayloadWithEstimatedScore(
        passing,
        modeName: "local-iterate",
        progress: { lines.append($0) }
    )
    #expect(stillPassing == passing)
    #expect(lines.isEmpty)
}

// Guard the ranked score semantics against the local-mode estimate: the
// official benchmark path still publishes exactly the score it was given on
// pass, and null on failure. Only localIterateScore (reached exclusively via
// --local-iterate/--local-submit) synthesizes an estimate.
@Test
func rankedScoreSemanticsAreUnchangedByLocalEstimatedScore() {
    let correctness = CorrectnessReport(
        passed: true,
        checkedSteps: 513,
        caseCount: 1,
        firstFailingCase: nil,
        firstFailingStep: nil,
        expectedToken: nil,
        actualToken: nil,
        goldenHash: "hash",
        error: ""
    )
    let passed = QwenRuntime.passedScore(
        score: 1.25,
        peakRamGB: 20,
        bandwidthGBPerToken: 0,
        decodeSecondsPerToken: 0.1,
        prefillSecondsPerToken: 0.01,
        benchmarkWallSeconds: 100,
        preflightSeconds: 1,
        correctnessSeconds: 50,
        timedBenchmarkSeconds: 40,
        numLayers: MLXFastConstants.numHiddenLayers,
        correctness: correctness,
        expertStats: .zero,
        bandwidthSource: "ram_resident_model",
        weightsDigest: nil,
        gpqaTTFT: .zero
    )
    #expect(passed.score == 1.25)
    #expect(passed.metrics.runtime == "swift")

    let failed = QwenRuntime.failedScore(
        error: "boom",
        correctness: nil,
        passedCorrectness: false,
        runtime: "swift"
    )
    #expect(failed.score == nil)
    #expect(failed.passed == false)
}

@Test
func localIteratePhaseHeartbeatFiresWhileBlockedAndStopsAfterCancel() throws {
    final class MessageBox: @unchecked Sendable {
        private let lock = NSLock()
        private var messages: [String] = []
        func append(_ message: String) {
            lock.lock()
            defer { lock.unlock() }
            messages.append(message)
        }
        func snapshot() -> [String] {
            lock.lock()
            defer { lock.unlock() }
            return messages
        }
    }

    // No progress sink means no timer at all.
    #expect(QwenRuntime.startPhaseHeartbeat(label: "x", progress: nil) == nil)

    let box = MessageBox()
    let heartbeat = try #require(
        QwenRuntime.startPhaseHeartbeat(
            label: "local-iterate prefill measured",
            intervalSeconds: 0.05,
            progress: { box.append($0) }
        )
    )
    Thread.sleep(forTimeInterval: 0.3)
    let cancellationDrained = DispatchSemaphore(value: 0)
    heartbeat.setCancelHandler {
        cancellationDrained.signal()
    }
    heartbeat.cancel()
    #expect(cancellationDrained.wait(timeout: .now() + 1) == .success)
    let firedWhileRunning = box.snapshot()
    #expect(!firedWhileRunning.isEmpty)
    #expect(
        firedWhileRunning.allSatisfy {
            $0.hasPrefix("local-iterate prefill measured still running phase_seconds=")
        }
    )

    // After cancellation has drained, the heartbeat must stay quiet.
    Thread.sleep(forTimeInterval: 0.2)
    #expect(box.snapshot().count == firedWhileRunning.count)
}

private func writeSafetensors(_ path: URL, tensors: [TensorFixture]) throws {
    var object: [String: Any] = [:]
    var cursor = 0
    for tensor in tensors.sorted(by: { $0.name < $1.name }) {
        let byteCount = try expectedTensorByteCount(
            name: tensor.name,
            dtype: TensorDType.parse(tensor.dtype),
            shape: tensor.shape
        )
        object[tensor.name] = [
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [cursor, cursor + byteCount],
        ]
        cursor += byteCount
    }

    var header = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    while header.count % 8 != 0 {
        header.append(0x20)
    }

    var output = Data()
    var headerLength = UInt64(header.count).littleEndian
    output.append(Data(bytes: &headerLength, count: 8))
    output.append(header)
    try output.write(to: path)

    let handle = try FileHandle(forWritingTo: path)
    defer {
        try? handle.close()
    }
    try handle.truncate(atOffset: UInt64(output.count + cursor))
}

private func arrayJSON(_ values: [Int]) -> String {
    "[\(values.map(String.init).joined(separator: ","))]"
}

private func writeSparseFile(_ path: URL, byteCount: Int) throws {
    _ = FileManager.default.createFile(atPath: path.path, contents: nil)
    let handle = try FileHandle(forWritingTo: path)
    defer {
        try? handle.close()
    }
    try handle.truncate(atOffset: UInt64(byteCount))
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

/// Idle loop for fake workers that must stay unresponsive while the client
/// times out / escalates to SIGKILL, without becoming immortal debris: it
/// ignores nothing itself (TERM handling is up to the fixture), burns no CPU
/// (sleep, not a busy loop), and self-expires after ~2 minutes so a fixture
/// orphaned by an interrupted `swift test` run disappears on its own.
private let boundedFixtureIdleLoop = """
idle_iterations=0
while [ "${idle_iterations}" -lt 1200 ]; do
  sleep 0.1
  idle_iterations=$((idle_iterations + 1))
done
"""

private func makeRuntimeWorkerScript(_ contents: String) throws -> URL {
    let directory = try temporaryDirectory()
    let executable = directory.appendingPathComponent("fake-runtime-worker")
    try contents.write(to: executable, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: executable.path
    )
    return executable
}

/// The pinned Laguna XS 2.1 MoE architecture as the transformed runtime
/// config declares it (mirroring poolside/Laguna-XS-2.1-NVFP4-mlx's
/// config.json): a full-attention layer with 48 query heads and YaRN
/// partial RoPE at layers 0, 4, ..., 36, sliding-window layers with 64
/// query heads and default RoPE elsewhere, a dense MLP only at layer 0,
/// and 256-expert top-8 MoE blocks with a 512-wide shared expert on the
/// other 39 layers.
private func pinnedRuntimeWorkerConfigurationObject() -> [String: Any] {
    [
        "model_type": "qwen3_5_text",
        "vocab_size": MLXFastConstants.vocabSize,
        "hidden_size": MLXFastConstants.hiddenSize,
        "intermediate_size": MLXFastConstants.intermediateSize,
        "num_hidden_layers": MLXFastConstants.numHiddenLayers,
        "num_attention_heads": MLXFastConstants.attentionHeads,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_value_heads": 48,
        "linear_num_key_heads": 16,
        "linear_value_head_dim": 128,
        "linear_key_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
        "layer_types": (0..<MLXFastConstants.numHiddenLayers).map {
            $0 % 4 == 3 ? "full_attention" : "linear_attention"
        },
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "max_position_embeddings": 262_144,
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attn_output_gate": true,
        "output_gate_type": "swish",
        "bos_token_id": 248_044,
        "eos_token_id": 248_044,
        "initializer_range": 0.02,
        "pad_token_id": NSNull(),
        "tie_word_embeddings": false,
        "mamba_ssm_dtype": "float32",
        "dtype": "bfloat16",
        "use_cache": true,
        "partial_rotary_factor": 0.25,
        "rope_parameters": [
            "rope_theta": 10_000_000.0,
            "rope_type": "default",
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
        ],
        "quantization": [
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
        ],
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": false,
    ]
}

private func shortRuntimeWorkerOptions(executable: URL) -> RuntimeWorkerOptions {
    RuntimeWorkerOptions(
        executablePath: executable.path,
        helloTimeoutSeconds: 0.1,
        requestTimeoutSeconds: 0.1,
        shutdownTimeoutSeconds: 0.1,
        terminationGraceSeconds: 0.05
    )
}
