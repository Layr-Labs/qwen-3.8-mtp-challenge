import Darwin
import Foundation
import MLX
@testable import MLXFastServing
import MLXLMServer
import Testing

struct MTPStartupTests {
    @Test("Native MTP CLI retains all ordinary flags and the explicit pinned head")
    func nativeOptions() throws {
        let options = try MTPServerOptions.parse(arguments: ["server", "--model", "/models/qwen",
            "--host", "127.0.0.1", "--port", "9090", "--context-tokens", "8192",
            "--model-type", "qwen3_5", "--tool-call-parser", "auto", "--reasoning-parser", "qwen3",
            "--engine-kind", "native_mtp", "--mtp-head", "/heads/mtp", "--mtp-max-depth", "8"])
        #expect(options.headPath == "/heads/mtp")
        #expect(options.maxDepth == 8)
        guard case .run(let configuration) = options.command else { Issue.record("Missing run command"); return }
        #expect(configuration.model == "/models/qwen")
        #expect(configuration.port == 9090)
        #expect(configuration.contextTokens == 8192)
        #expect(configuration.toolCallParser == "auto")
        #expect(configuration.reasoningParser == .qwen3)
    }

    @Test("Native MTP accepts one absolute target manifest and leaves it out of the ordinary CLI")
    func targetManifestOption() throws {
        let args = ["server", "--model", "/models/qwen", "--engine-kind", "native_mtp",
                    "--mtp-head", "/heads/mtp", "--target-manifest", "/manifests/target.json"]
        let options = try MTPServerOptions.parse(arguments: args)
        #expect(options.targetManifestPath == "/manifests/target.json")
        let pinned = try MTPServerOptions.parse(arguments: Array(args.dropLast(2)))
        #expect(pinned.targetManifestPath == nil)
        #expect(options.command == pinned.command)
    }

    @Test("Absent MTP arguments preserve the ordinary CLI result")
    func ordinaryOptionsUnchanged() throws {
        let args = ["server", "--model", "model", "--engine-kind", "single_request"]
        let options = try MTPServerOptions.parse(arguments: args)
        let ordinary = try MLXServerCLI.parse(arguments: args, environment: [:])
        #expect(options.command == ordinary)
        #expect(options.headPath == nil)
        #expect(options.targetManifestPath == nil)
    }

    @Test("Serving loader uses the ranked allocator cap, including on a small machine")
    func servingMemoryCap() {
        let savedLimit = Memory.cacheLimit
        let savedMegabytes = getenv("MLX_MAX_MB_PER_BUFFER").map { String(cString: $0) }
        let savedOperations = getenv("MLX_MAX_OPS_PER_BUFFER").map { String(cString: $0) }
        defer {
            Memory.cacheLimit = savedLimit
            if let savedMegabytes {
                setenv("MLX_MAX_MB_PER_BUFFER", savedMegabytes, 1)
            } else {
                unsetenv("MLX_MAX_MB_PER_BUFFER")
            }
            if let savedOperations {
                setenv("MLX_MAX_OPS_PER_BUFFER", savedOperations, 1)
            } else {
                unsetenv("MLX_MAX_OPS_PER_BUFFER")
            }
        }
        ServingMemoryPolicy.install(physicalMemoryBytes: 128 << 30)
        #expect(Memory.cacheLimit == 32 << 30)
        #expect(Memory.cacheLimit != 0)
        #expect(String(cString: getenv("MLX_MAX_MB_PER_BUFFER")) == "512")
        #expect(String(cString: getenv("MLX_MAX_OPS_PER_BUFFER")) == "50")
        ServingMemoryPolicy.install(physicalMemoryBytes: 48 << 30)
        #expect(Memory.cacheLimit == 6 << 30)
        #expect(String(cString: getenv("MLX_MAX_MB_PER_BUFFER")) == "128")
        #expect(String(cString: getenv("MLX_MAX_OPS_PER_BUFFER")) == "64")
    }

    @Test("Command-buffer budgets are assigned before the cache cap")
    func commandBufferBudgetsPrecedeCacheLimit() throws {
        let sourceURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXFastServing/ServingMemoryPolicy.swift")
        let source = try String(contentsOf: sourceURL, encoding: .utf8)
        let function = try #require(source.range(of: "static func install"))
        let body = source[function.lowerBound...]
        let megabytes = try #require(body.range(of: "MLX_MAX_MB_PER_BUFFER"))
        let operations = try #require(body.range(of: "MLX_MAX_OPS_PER_BUFFER"))
        let cap = try #require(body.range(of: "Memory.cacheLimit"))
        #expect(megabytes.lowerBound < cap.lowerBound)
        #expect(operations.lowerBound < cap.lowerBound)
    }

    @Test("Native MTP accepts automatic tool choice and rejects the other modes")
    func toolChoiceMatchesAdvertisedModes() throws {
        let base = OpenAIChatCompletionRequest(
            model: "test",
            messages: [.init(role: .user, content: .text("hi"))],
            temperature: 0
        )
        try MLXServerGreedyPolicy.validate(base)
        var automatic = base
        automatic.toolChoice = .mode(.auto)
        try MLXServerGreedyPolicy.validate(automatic)
        for choice in [
            OpenAIToolChoice.mode(.none),
            .mode(.required),
            .function(name: "read"),
        ] {
            var rejected = base
            rejected.toolChoice = choice
            #expect(throws: (any Error).self) {
                try MLXServerGreedyPolicy.validate(rejected)
            }
        }
        var parallel = base
        parallel.parallelToolCalls = false
        #expect(throws: (any Error).self) {
            try MLXServerGreedyPolicy.validate(parallel)
        }
        var strict = base
        strict.tools = [.init(function: .init(name: "read", strict: true))]
        #expect(throws: (any Error).self) {
            try MLXServerGreedyPolicy.validate(strict)
        }
        var thinking = base
        thinking.chatTemplateKwargs = ["enable_thinking": .bool(false)]
        try MLXServerGreedyPolicy.validate(thinking)
        var foreign = base
        foreign.chatTemplateKwargs = ["chat_template": .string("{{ messages }}")]
        #expect(throws: (any Error).self) {
            try MLXServerGreedyPolicy.validate(foreign)
        }
        var formatted = base
        formatted.responseFormat = .jsonObject()
        #expect(throws: (any Error).self) {
            try MLXServerGreedyPolicy.validate(formatted)
        }
        var parser = base
        parser.toolCallParser = "json"
        #expect(throws: (any Error).self) {
            try MLXServerGreedyPolicy.validate(parser)
        }
    }

    @Test("Invalid depth, relative head, target manifest, and incomplete fast mode fail closed",
        arguments: [
            ["--engine-kind", "native_mtp"],
            ["--mtp-head", "/head"],
            ["--engine-kind", "native_mtp", "--mtp-head", "relative"],
            ["--engine-kind", "native_mtp", "--mtp-head", "/head", "--mtp-max-depth", "0"],
            ["--engine-kind", "native_mtp", "--mtp-head", "/head", "--mtp-max-depth", "9"],
            ["--target-manifest", "/manifest.json"],
            ["--engine-kind", "single_request", "--target-manifest", "/manifest.json"],
            ["--engine-kind", "native_mtp", "--mtp-head", "/head", "--target-manifest", "manifest.json"],
            ["--engine-kind", "native_mtp", "--mtp-head", "/head", "--target-manifest"],
            ["--engine-kind", "native_mtp", "--mtp-head", "/head", "--target-manifest", "/a.json", "--target-manifest", "/b.json"],
        ])
    func invalidOptions(args: [String]) {
        #expect(throws: (any Error).self) { try MTPServerOptions.parse(arguments: ["server"] + args) }
    }
}
