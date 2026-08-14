// Copyright © 2026 Apple Inc.

import Darwin
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXRandom
import MLXSpeculative
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

enum BenchMode: String {
    case mtp
    case dflash
    case gatherQMV = "gather-qmv"

    var envPrefix: String {
        switch self {
        case .mtp: "MTP_BENCH"
        case .dflash: "DFLASH_BENCH"
        case .gatherQMV: "GATHER_QMV_BENCH"
        }
    }
}

struct CLIError: Error, CustomStringConvertible {
    let description: String

    init(_ description: String) {
        self.description = description
    }
}

struct BenchArguments {
    var mode: BenchMode?
    var targetPath: String?
    var drafterPath: String?
    var promptsJSONPath: String?
    var promptsTextJSONPath: String?
    var promptTokens: String?
    var prompt: String?
    var promptFilePath: String?
    var maxTokens: Int?
    var warmupTokens: Int?
    var blockSizes: [Int]?
    var batchSize: Int?
    var useChatTemplate = true
    var phaseTimings = false
    var verifySubphaseTimings = false
    var enableVerifyQMM: Bool?
    var verifyQMMInclude: String?
    var tokenHashes = false
    var printOutput = false
    var stopOnEOS = false
    var answerRegex: String?
    var expectedAnswer: String?
    var repetitions: Int?
    var parityCheck = false
    var parityBaselineScan = false
    var parityMaxFailures: Int?
    var parityLayerDrift = false
    var paritySummaryOnly = false
    var parityTopK: Int?

    static func parse() throws -> BenchArguments {
        var args = BenchArguments()
        let argv = CommandLine.arguments
        var i = 1

        while i < argv.count {
            let arg = argv[i]
            switch arg {
            case "mtp", "dflash", "gather-qmv":
                args.mode = BenchMode(rawValue: arg)
            case "--mode":
                args.mode = BenchMode(rawValue: try value(after: arg, argv: argv, index: &i))
            case "--target", "-t":
                args.targetPath = try value(after: arg, argv: argv, index: &i)
            case "--drafter", "-d":
                args.drafterPath = try value(after: arg, argv: argv, index: &i)
            case "--prompts-json":
                args.promptsJSONPath = try value(after: arg, argv: argv, index: &i)
            case "--prompts-text-json":
                args.promptsTextJSONPath = try value(after: arg, argv: argv, index: &i)
            case "--prompt-tokens":
                args.promptTokens = try value(after: arg, argv: argv, index: &i)
            case "--prompt":
                args.prompt = try value(after: arg, argv: argv, index: &i)
            case "--prompt-file":
                args.promptFilePath = try value(after: arg, argv: argv, index: &i)
            case "--max-tokens":
                let value = try value(after: arg, argv: argv, index: &i)
                guard let parsed = Int(value), parsed > 0 else {
                    throw CLIError("--max-tokens requires a positive integer")
                }
                args.maxTokens = parsed
            case "--warmup-tokens":
                let value = try value(after: arg, argv: argv, index: &i)
                guard let parsed = Int(value), parsed > 0 else {
                    throw CLIError("--warmup-tokens requires a positive integer")
                }
                args.warmupTokens = parsed
            case "--block-sizes":
                args.blockSizes = try parsePositiveIntList(
                    try value(after: arg, argv: argv, index: &i),
                    minimum: 2,
                    optionName: "--block-sizes")
            case "--batch-size":
                let value = try value(after: arg, argv: argv, index: &i)
                guard let parsed = Int(value), parsed > 0 else {
                    throw CLIError("--batch-size requires a positive integer")
                }
                args.batchSize = parsed
            case "--no-chat-template":
                args.useChatTemplate = false
            case "--phase-timings":
                args.phaseTimings = true
            case "--verify-subphases":
                args.verifySubphaseTimings = true
            case "--verify-qmm":
                args.enableVerifyQMM = true
            case "--verify-qmm-include":
                args.verifyQMMInclude = try value(after: arg, argv: argv, index: &i)
            case "--no-verify-qmm":
                args.enableVerifyQMM = false
            case "--token-hashes":
                args.tokenHashes = true
            case "--print-output":
                args.printOutput = true
            case "--stop-on-eos":
                args.stopOnEOS = true
            case "--answer-regex":
                args.answerRegex = try value(after: arg, argv: argv, index: &i)
            case "--expected-answer":
                args.expectedAnswer = try value(after: arg, argv: argv, index: &i)
            case "--repetitions":
                let value = try value(after: arg, argv: argv, index: &i)
                guard let parsed = Int(value), parsed > 0 else {
                    throw CLIError("--repetitions requires a positive integer")
                }
                args.repetitions = parsed
            case "--parity-check":
                args.parityCheck = true
            case "--parity-baseline-scan":
                args.parityCheck = true
                args.parityBaselineScan = true
            case "--parity-max-failures":
                let value = try value(after: arg, argv: argv, index: &i)
                guard let parsed = Int(value), parsed > 0 else {
                    throw CLIError("--parity-max-failures requires a positive integer")
                }
                args.parityMaxFailures = parsed
            case "--parity-layer-drift":
                args.parityLayerDrift = true
            case "--parity-summary-only":
                args.paritySummaryOnly = true
            case "--parity-top-k":
                let value = try value(after: arg, argv: argv, index: &i)
                guard let parsed = Int(value), parsed >= 0 else {
                    throw CLIError("--parity-top-k requires a non-negative integer")
                }
                args.parityTopK = parsed
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                if !arg.hasPrefix("-"), args.mode == nil {
                    args.mode = BenchMode(rawValue: arg)
                } else {
                    throw CLIError("Unknown argument: \(arg)")
                }
            }
            i += 1
        }

        return args
    }

    static func value(after option: String, argv: [String], index: inout Int) throws -> String {
        index += 1
        guard index < argv.count else {
            throw CLIError("\(option) requires a value")
        }
        return argv[index]
    }

    static func printUsage() {
        print("""
        mlx-bench - local MLX speculative decoding throughput benchmark

        USAGE:
          mlx-bench mtp    --target <target-dir> --drafter <assistant-dir> [options]
          mlx-bench dflash --target <target-dir> --drafter <dflash-dir> [options]
          mlx-bench gather-qmv [options]

        OPTIONS:
          --target, -t <path>       Target model directory
          --drafter, -d <path>      Assistant or DFlash drafter directory
          --prompts-json <path>     JSON file containing [[Int]] token IDs
          --prompts-text-json <path>
                                  JSON file containing [String] prompts to tokenize
          --prompt-tokens <ids>     Comma-separated token IDs for one prompt
          --prompt <text>           Prompt text to tokenize with target tokenizer
          --prompt-file <path>      UTF-8 prompt text file to tokenize
          --max-tokens <int>        Generated tokens per prompt
          --warmup-tokens <int>     Generated tokens for warmup
          --block-sizes <list>      Comma-separated speculative block sizes
                                  (DFlash default sweeps 4,5,6,8 plus checkpoint size)
          --batch-size <int>        Run an in-process batched DFlash benchmark
          --no-chat-template        Encode --prompt as plain text
          --phase-timings           Print DFlash diagnostic phase timings
          --verify-subphases        Split DFlash target verify into diagnostic subphases
          --verify-qmm              Enable experimental DFlash target M=16 qmm fast path
          --verify-qmm-include <s>  Comma-separated qmm subset:
                                  all, attn, attn_q/k/v/o, mlp, router
          --no-verify-qmm           Disable DFlash target M=16 qmm fast path
          --token-hashes            Print per-prompt generated-token hash diagnostics
          --print-output            Decode and print generated text for each prompt
          --stop-on-eos             Stop each prompt when an EOS/extra-EOS token is generated
          --answer-regex <regex>    Stop each run after decoded output matches regex
          --expected-answer <text>  Expected answer capture for answer-regex runs
          --repetitions <int>       Repeated answer-regex runs for average tok/s
          --parity-check            Compare target sequential decode with DFlash block verify
          --parity-baseline-scan    Scan sequential baseline stream in DFlash verify blocks
          --parity-max-failures <n> Stop parity diagnostics after n failures
          --parity-layer-drift      Print per-layer hidden drift at parity mismatches
          --parity-summary-only     Suppress per-mismatch detail rows
          --parity-top-k <int>      Top-k logits to print at first parity mismatch
          --help, -h                Show this help

        ENVIRONMENT:
          TARGET_DIR, DRAFTER_DIR
          MTP_BENCH_TARGET_DIR, MTP_BENCH_DRAFTER_DIR
          DFLASH_BENCH_TARGET_DIR, DFLASH_BENCH_DRAFTER_DIR
          PROMPTS_JSON, PROMPTS_TEXT_JSON, PROMPT_TOKENS, PROMPT
          MAX_TOKENS, WARMUP_TOKENS, BLOCK_SIZES, BATCH_SIZE
          DFLASH_BENCH_PHASES=1
          DFLASH_BENCH_VERIFY_SUBPHASES=1
          DFLASH_BENCH_VERIFY_QMM=1
          DFLASH_BENCH_VERIFY_QMM_INCLUDE=attn,mlp,router
          DFLASH_BENCH_PARITY=1
          DFLASH_BENCH_PARITY_BASELINE_SCAN=1
          DFLASH_BENCH_PARITY_MAX_FAILURES=8
          DFLASH_BENCH_PARITY_LAYER_DRIFT=1
          DFLASH_BENCH_PARITY_SUMMARY_ONLY=1
          DFLASH_BENCH_PARITY_TOP_K=5
          DFLASH_BENCH_STOP_ON_EOS=1
          MLX_GEMMA4_DFLASH_OFFICIAL_FAST=1
          MLX_GEMMA4_DFLASH_VERIFY_FUSION_MAX_ROWS=24
          MLX_SWITCH_GLU_SORT_MIN_SIZE=32
          MLX_SWITCH_GLU_GEMMA4_FUSE_GATE_UP=0
          MLX_SWITCH_GLU_GEMMA4_WEIGHTED_FUSE_GATE_UP=0
          MLX_SWITCH_GLU_GEMMA4_WEIGHTED_MAX_ROWS=24
          GATHER_QMV_BENCH_ROWS=128
          GATHER_QMV_BENCH_EXPERTS=128
          GATHER_QMV_BENCH_INPUT_DIMS=2816
          GATHER_QMV_BENCH_OUTPUT_DIMS=8192
          GATHER_QMV_BENCH_ITERATIONS=20
          GATHER_QMV_BENCH_WARMUPS=3
        """)
    }
}

extension MLXBench {
    static func runGatherQMVBenchmark(args: BenchArguments, mode: BenchMode) throws {
        let rows = envInt(names: ["\(mode.envPrefix)_ROWS"], default: 128)
        let experts = envInt(names: ["\(mode.envPrefix)_EXPERTS"], default: 128)
        let inputDimensions = envInt(names: ["\(mode.envPrefix)_INPUT_DIMS"], default: 2816)
        let outputDimensions = envInt(names: ["\(mode.envPrefix)_OUTPUT_DIMS"], default: 8192)
        let iterations =
            args.maxTokens
            ?? envInt(names: ["\(mode.envPrefix)_ITERATIONS"], default: 20)
        let warmups =
            args.warmupTokens
            ?? envInt(names: ["\(mode.envPrefix)_WARMUPS"], default: 3)
        let groupSize = 64
        let bits = 4

        guard rows > 0, experts > 0 else {
            throw CLIError("gather-qmv rows and experts must be positive")
        }
        guard inputDimensions % groupSize == 0 else {
            throw CLIError("gather-qmv input dimensions must be divisible by \(groupSize)")
        }
        guard outputDimensions % 8 == 0 else {
            throw CLIError("gather-qmv output dimensions must be divisible by 8")
        }

        print("")
        print("=== gather-qmv microbenchmark ===")
        print(
            "rows=\(rows), experts=\(experts), input_dims=\(inputDimensions), output_dims=\(outputDimensions), group_size=\(groupSize), bits=\(bits)"
        )
        print("iterations=\(iterations), warmups=\(warmups)")

        MLXRandom.seed(47)
        let x = MLXRandom.normal([rows, 1, inputDimensions]).asType(.bfloat16)
        let w = MLXRandom.normal([experts, outputDimensions, inputDimensions]).asType(.bfloat16)
        let (wq, scales, biases) = MLX.quantized(w, groupSize: groupSize, bits: bits)
        guard let biases else {
            throw CLIError("gather-qmv affine quantization did not return biases")
        }
        let routeIndices = MLXArray((0..<rows).map { Int32($0 % experts) })
        eval(x, wq, scales, biases, routeIndices)

        func runOnce() -> MLXArray {
            MLX.gatherQuantizedMM(
                x,
                wq,
                scales: scales,
                biases: biases,
                rhsIndices: routeIndices,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                sortedIndices: true)
        }

        for _ in 0..<warmups {
            eval(runOnce())
        }
        MLX.Memory.clearCache()

        var last = MLXArray()
        let start = Date()
        for _ in 0..<iterations {
            last = runOnce()
            eval(last)
        }
        let seconds = Date().timeIntervalSince(start)
        let avgMilliseconds = seconds * 1_000.0 / Double(iterations)
        let routedRowsPerSecond = Double(rows * iterations) / seconds
        let outputElementsPerSecond = Double(rows * outputDimensions * iterations) / seconds
        let checksum = last.sum().item(Float.self)

        print(
            "avg_ms=\(String(format: "%.3f", avgMilliseconds)) "
                + "routed_rows_per_s=\(String(format: "%.1f", routedRowsPerSecond)) "
                + "output_melems_per_s=\(String(format: "%.1f", outputElementsPerSecond / 1_000_000.0)) "
                + "checksum=\(String(format: "%.4f", checksum))"
        )
    }


}


func benchmarkPrompts(
    args: BenchArguments,
    mode: BenchMode,
    tokenizer: any MLXLMCommon.Tokenizer
) throws -> [[Int32]] {
    if let path = args.promptsJSONPath
        ?? envString(names: ["PROMPTS_JSON", "\(mode.envPrefix)_PROMPTS", "\(mode.envPrefix)_PROMPTS_JSON"])
    {
        let url = URL(fileURLWithPath: expandPath(path))
        let data = try Data(contentsOf: url)
        let decoded = try JSONDecoder().decode([[Int]].self, from: data)
        return try decoded.map { try int32Tokens($0) }.filter { !$0.isEmpty }
    }

    if let tokenList = args.promptTokens
        ?? envString(names: ["PROMPT_TOKENS", "\(mode.envPrefix)_PROMPT_TOKENS"])
    {
        return [try parseTokenList(tokenList)]
    }

    if let path = args.promptsTextJSONPath
        ?? envString(names: [
            "PROMPTS_TEXT_JSON",
            "\(mode.envPrefix)_PROMPTS_TEXT",
            "\(mode.envPrefix)_PROMPTS_TEXT_JSON",
        ])
    {
        let url = URL(fileURLWithPath: expandPath(path))
        let data = try Data(contentsOf: url)
        let decoded = try JSONDecoder().decode([String].self, from: data)
        return try decoded
            .map { try benchmarkPromptTokens(text: $0, args: args, tokenizer: tokenizer) }
            .filter { !$0.isEmpty }
    }

    let promptFilePath = args.promptFilePath
        ?? envString(names: ["PROMPT_FILE", "\(mode.envPrefix)_PROMPT_FILE"])
    let promptFromFile = try promptFilePath.map {
        try String(contentsOf: URL(fileURLWithPath: expandPath($0)), encoding: .utf8)
    }

    let prompt = promptFromFile
        ?? args.prompt
        ?? envString(names: ["PROMPT", "\(mode.envPrefix)_PROMPT"])
        ?? "Write a concise explanation of speculative decoding and why accept rate matters."

    return [try benchmarkPromptTokens(text: prompt, args: args, tokenizer: tokenizer)]
}

func benchmarkPromptTokens(
    text prompt: String,
    args: BenchArguments,
    tokenizer: any MLXLMCommon.Tokenizer
) throws -> [Int32] {
    if args.useChatTemplate && envString(names: ["BENCH_CHAT_TEMPLATE"]) != "0" {
        do {
            let messages: [[String: any Sendable]] = [
                ["role": "user", "content": prompt]
            ]
            let tokenIDs = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: nil as [[String: any Sendable]]?,
                additionalContext: nil as [String: any Sendable]?)
            return try int32Tokens(tokenIDs)
        } catch {
            eprint("warning: chat template failed, falling back to plain encode: \(error)")
        }
    }

    return try int32Tokens(tokenizer.encode(text: prompt, addSpecialTokens: true))
}

func requiredURL(
    explicit: String?,
    envNames: [String],
    description: String
) throws -> URL {
    guard let path = explicit ?? envString(names: envNames) else {
        throw CLIError("Missing \(description). Pass an argument or set one of: \(envNames.joined(separator: ", "))")
    }
    let url = URL(fileURLWithPath: expandPath(path))
    guard FileManager.default.fileExists(atPath: url.path) else {
        throw CLIError("\(description) not found: \(url.path)")
    }
    return url
}

func expandPath(_ path: String) -> String {
    (path as NSString).expandingTildeInPath
}

func envString(names: [String]) -> String? {
    let env = ProcessInfo.processInfo.environment
    for name in names {
        if let value = env[name], !value.isEmpty {
            return value
        }
    }
    return nil
}

func envInt(names: [String], default defaultValue: Int) -> Int {
    guard let value = envString(names: names), let parsed = Int(value), parsed > 0 else {
        return defaultValue
    }
    return parsed
}

func envIntList(names: [String], default defaultValue: [Int], minimum: Int) -> [Int] {
    guard let value = envString(names: names),
        let parsed = try? parsePositiveIntList(value, minimum: minimum, optionName: names[0]),
        !parsed.isEmpty
    else {
        return defaultValue
    }
    return parsed
}

func parsePositiveIntList(_ value: String, minimum: Int, optionName: String) throws -> [Int] {
    let parsed = value
        .split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .compactMap(Int.init)
        .filter { $0 >= minimum }
    guard !parsed.isEmpty else {
        throw CLIError("\(optionName) requires comma-separated integers >= \(minimum)")
    }
    return parsed
}

func orderedUnique(_ values: [Int]) -> [Int] {
    var seen = Set<Int>()
    var result = [Int]()
    result.reserveCapacity(values.count)
    for value in values where seen.insert(value).inserted {
        result.append(value)
    }
    return result
}

func envBool(names: [String], default defaultValue: Bool) -> Bool {
    envBoolOverride(names: names) ?? defaultValue
}

func envBoolOverride(names: [String]) -> Bool? {
    guard let value = envString(names: names)?.lowercased() else {
        return nil
    }
    switch value {
    case "1", "true", "yes", "on":
        return true
    case "0", "false", "no", "off":
        return false
    default:
        return nil
    }
}

func parseTokenList(_ value: String) throws -> [Int32] {
    let parsed = value
        .split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .compactMap(Int.init)
    return try int32Tokens(parsed)
}

func int32Tokens(_ values: [Int]) throws -> [Int32] {
    var output = [Int32]()
    output.reserveCapacity(values.count)
    for value in values {
        guard value >= Int(Int32.min), value <= Int(Int32.max) else {
            throw CLIError("Token ID is outside Int32 range: \(value)")
        }
        output.append(Int32(value))
    }
    guard !output.isEmpty else {
        throw CLIError("Prompt tokens must not be empty")
    }
    return output
}

func printDecodedOutput(label: String, tokenIds: [Int], tokenizer: MLXLMCommon.Tokenizer) {
    let decoded = tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: true)
        .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
    print("   \(label):")
    if decoded.isEmpty {
        print("      <empty>")
    } else {
        for line in decoded.split(separator: "\n", omittingEmptySubsequences: false) {
            print("      \(line)")
        }
    }
}

func average(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    return values.reduce(0, +) / Double(values.count)
}

func tokenHash(_ tokens: [Int]) -> String {
    var hash: UInt64 = 0xcbf29ce484222325
    for token in tokens {
        var value = UInt64(bitPattern: Int64(token))
        for _ in 0 ..< 8 {
            hash ^= value & 0xff
            hash &*= 0x100000001b3
            value >>= 8
        }
    }
    return String(hash, radix: 16)
}

func firstTokenMismatch(_ baseline: [Int], _ candidate: [Int]) -> String? {
    let common = min(baseline.count, candidate.count)
    for index in 0 ..< common where baseline[index] != candidate[index] {
        return "index=\(index) baseline=\(baseline[index]) candidate=\(candidate[index])"
    }
    guard baseline.count != candidate.count else { return nil }
    return "length baseline=\(baseline.count) candidate=\(candidate.count)"
}

func eprint(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}
