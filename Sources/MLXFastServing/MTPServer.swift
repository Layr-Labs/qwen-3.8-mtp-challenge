import Foundation
import MLX
import MLXFastModel
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXLMServer
import Tokenizers

public enum MTPServer {
    public static func run(arguments: [String] = CommandLine.arguments) async throws {
        let options = try MTPServerOptions.parse(arguments: arguments,
                                               environment: ProcessInfo.processInfo.environment)
        switch options.command {
        case .help:
            print(MLXServerCLI.help)
            print("\nNative MTP: --engine-kind native_mtp --mtp-head <absolute path> --mtp-max-depth <1...8> [--target-manifest <absolute JSON file>]\nRequires the pinned Qwen3.8 27B snapshot, or the snapshot files listed in --target-manifest as [{path, sha256, bytes}], and greedy requests (temperature=0, no penalties).")
        case .listRoutes:
            let data = try JSONEncoder.openAIServer.encode(MLXServerRoute.manifest)
            print(String(decoding: data, as: UTF8.self))
        case .run(let configuration):
            // Before any weight or warmup allocation. The ranked worker
            // already does this; leaving the default lets one user's
            // discarded prefill buffers occupy the free pool for everyone.
            ServingMemoryPolicy.install()
            guard let headPath = options.headPath else {
                try await MLXServer.run(configuration: configuration)
                return
            }
            let started = Date()
            let head = try MTPStartupStage.validateHead.perform { try MTPPinnedFiles.headDirectory(path: headPath) }
            guard configuration.model.hasPrefix("/") else {
                throw MTPStartupError.invalid("native_mtp requires an absolute local pinned Qwen3.8 model directory.")
            }
            let target = URL(fileURLWithPath: configuration.model)
            try MTPStartupStage.validateTarget.perform {
                let files = try options.targetManifestPath.map(MTPPinnedFiles.targetManifest(path:))
                    ?? MTPPinnedFiles.compiledTarget()
                try MTPPinnedFiles.verifyTarget(directory: target, files: files)
            }
            let model = try load(target: target, head: head, maxDepth: options.maxDepth)
            let engine = MLXModelContainerEngine(modelID: configuration.model, model: model,
                modelType: configuration.modelType, contextTokens: configuration.contextTokens,
                defaultToolCallParser: configuration.toolCallParser,
                generationDriver: MTPGenerationDriver(maxDepth: options.maxDepth))
            let service = MLXOpenAIService(engine: engine,
                                           defaultReasoningParser: configuration.reasoningParser)
            let app = MLXServerApplication.buildApplication(service: service,
                host: configuration.host, port: configuration.port)
            diagnostic(stage: "initialized", elapsed: Date().timeIntervalSince(started))
            try await app.runService()
        }
    }

    /// The startup task returns only a Sendable container. Head globals are
    /// restored before routes exist; no model loading occurs during requests.
    private static func load(target: URL, head: URL, maxDepth: Int) throws -> ModelContainer {
        try Qwen36MTPHeadAttachment.withHeadAttached(backboneDirectory: target, headDirectory: head) { _ in
            let result = StartupResult()
            let semaphore = DispatchSemaphore(value: 0)
            Task.detached {
                defer { semaphore.signal() }
                do {
                    let start = MTPStartupStage.loadWeights.begin()
                    let context = try await LLMModelFactory.shared.load(from: target,
                                                                        using: #huggingFaceTokenizerLoader())
                    guard let model = context.model as? any Qwen36MTPTarget, model.hasMTPHead else {
                        throw MTPServingError.invalidModel
                    }
                    eval(context.model)
                    MTPStartupStage.loadWeights.complete(started: start)
                    // One input-independent original shape warm, with disposable
                    // caches. No hash, load, or shape warm belongs to the hot path.
                    try MTPStartupStage.warmKernels.perform {
                        try Qwen36MTPBlockSession(model: model, stopTokens: [])
                            .warmAllDepths(maxDepth: maxDepth)
                    }
                    result.store(.success(ModelContainer(context: context)))
                } catch { result.store(.failure(error)) }
            }
            semaphore.wait()
            return try result.get()
        }
    }

    private static func diagnostic(stage: String, elapsed: TimeInterval) {
        FileHandle.standardError.write(Data("native_mtp stage=\(stage) elapsed_ms=\(Int(elapsed * 1000))\n".utf8))
    }
}

/// Fixed startup stages. The injected clock/action/writer seam performs no
/// model work and leaves the existing elapsed-time protocol unchanged.
enum MTPStartupStage: String, CaseIterable {
    case validateHead = "validate_head"
    case validateTarget = "validate_target"
    case loadWeights = "load_weights"
    case warmKernels = "warm_kernels"

    func begin(now: () -> TimeInterval = { Date.timeIntervalSinceReferenceDate },
               write: (String) -> Void = Self.writeStandardError) -> TimeInterval {
        let started = now()
        write("native_mtp stage=\(rawValue) state=started\n")
        return started
    }

    func complete(started: TimeInterval,
                  now: () -> TimeInterval = { Date.timeIntervalSinceReferenceDate },
                  write: (String) -> Void = Self.writeStandardError) {
        write("native_mtp stage=\(rawValue) elapsed_ms=\(Int((now() - started) * 1000))\n")
    }

    func perform<Value>(now: () -> TimeInterval = { Date.timeIntervalSinceReferenceDate },
                        write: (String) -> Void = Self.writeStandardError,
                        action: () throws -> Value) rethrows -> Value {
        let started = begin(now: now, write: write)
        let value = try action()
        complete(started: started, now: now, write: write)
        return value
    }

    private static func writeStandardError(_ line: String) {
        FileHandle.standardError.write(Data(line.utf8))
    }
}

private final class StartupResult: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Result<ModelContainer, any Error>?
    func store(_ result: Result<ModelContainer, any Error>) { lock.withLock { value = result } }
    func get() throws -> ModelContainer {
        guard let result = lock.withLock({ value }) else {
            throw MTPStartupError.invalid("Model initialization returned no result.")
        }
        return try result.get()
    }
}
