// Copyright © 2026 Eigen Labs Inc.

import Foundation
import Hummingbird
import MLXLMCommon

public enum MLXServer {
    public static func run(configuration: MLXServerConfiguration) async throws {
        var primaryModelConfiguration = modelConfiguration(
            for: configuration.model,
            revision: configuration.revision
        )
        if let parser = configuration.toolCallParser {
            primaryModelConfiguration.toolCallFormat = try ServerToolParser.resolve(
                requested: parser,
                modelType: configuration.modelType
            )
        }

        let engine: any MLXServerEngine
        let batchedEngine: MLXBatchedEngineServerEngine?
        switch configuration.engineKind {
        case .batched:
            let context = try await MLXServerModelLoader.loadContext(
                configuration: primaryModelConfiguration
            )
            let batched = try MLXBatchedEngineServerEngine(
                modelID: configuration.model,
                modelContext: context,
                modelType: configuration.modelType,
                configuration: configuration.batchedEngineConfiguration,
                defaultToolCallParser: configuration.toolCallParser
            )
            await batched.start()
            engine = batched
            batchedEngine = batched
        case .singleRequest:
            let model = try await MLXServerModelLoader.load(
                configuration: primaryModelConfiguration
            )
            engine = MLXModelContainerEngine(
                modelID: configuration.model,
                model: model,
                modelType: configuration.modelType,
                defaultToolCallParser: configuration.toolCallParser
            )
            batchedEngine = nil
        }

        // Stop the engine on any exit path. Fire-and-forget is safe because
        // EngineCore.stop is synchronous + idempotent today.
        defer {
            if let batchedEngine {
                Task.detached(priority: .high) { await batchedEngine.stop() }
            }
        }

        let embeddingEngine: MLXEmbedderContainerEngine?
        if let embeddingModelID = configuration.embeddingModel {
            let embeddingModel = try await MLXServerEmbedderLoader.load(
                configuration: modelConfiguration(for: embeddingModelID)
            )
            embeddingEngine = MLXEmbedderContainerEngine(
                modelID: embeddingModelID,
                model: embeddingModel
            )
        } else {
            embeddingEngine = nil
        }
        let service = MLXOpenAIService(
            engine: engine,
            embeddingEngine: embeddingEngine,
            defaultReasoningParser: configuration.reasoningParser
        )
        let app = MLXServerApplication.buildApplication(
            service: service,
            host: configuration.host,
            port: configuration.port
        )

        try await app.runService()
    }

    static func modelConfiguration(for idOrPath: String, revision: String = "main") -> ModelConfiguration {
        let expandedPath: String
        if idOrPath == "~" {
            expandedPath = FileManager.default.homeDirectoryForCurrentUser.path
        } else if idOrPath.hasPrefix("~/") {
            expandedPath = FileManager.default.homeDirectoryForCurrentUser
                .appending(path: String(idOrPath.dropFirst(2)))
                .path
        } else {
            expandedPath = idOrPath
        }

        let url = URL(fileURLWithPath: expandedPath)
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: expandedPath, isDirectory: &isDirectory),
            isDirectory.boolValue
        {
            return .init(directory: url)
        }

        return .init(id: idOrPath, revision: revision)
    }
}
