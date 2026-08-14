import Foundation
import MLX
import MLXFastCore
import MLXLLM
import MLXLMCommon
import MLXNN

public struct LagunaUpstreamEquivalenceStep: Codable, Equatable {
    public let label: String
    public let maximumAbsoluteLogitError: Float
    public let meanAbsoluteLogitError: Float
    public let runtimeToken: Int
    public let upstreamToken: Int

    public var tokensMatch: Bool {
        runtimeToken == upstreamToken
    }
}

public struct LagunaUpstreamEquivalenceReport: Codable, Equatable {
    public let promptTokenCount: Int
    public let decodeTokenCount: Int
    public let steps: [LagunaUpstreamEquivalenceStep]

    public func passes(maximumAbsoluteLogitError: Float) -> Bool {
        steps.allSatisfy {
            $0.tokensMatch
                && $0.maximumAbsoluteLogitError.isFinite
                && $0.maximumAbsoluteLogitError <= maximumAbsoluteLogitError
        }
    }
}

/// Development-only cross-check between the challenge runtime and the
/// vendored upstream Laguna implementation. Both models receive the same
/// checkpoint arrays, prompt, teacher-forced decode tokens, and independent
/// KV caches. The helper is never called by the benchmark protocol or scored
/// path; operators invoke its gated test on the M5 before publishing new
/// correctness artifacts.
public enum LagunaUpstreamEquivalence {
    public static func compare(
        weightsPath: String,
        promptTokens: [Int],
        decodeTokens: [Int]
    ) throws -> LagunaUpstreamEquivalenceReport {
        guard !promptTokens.isEmpty else {
            throw MLXFastError.invalidInput(
                "Laguna upstream equivalence requires at least one prompt token"
            )
        }

        let runtimeConfig = try LagunaConfig.load(from: weightsPath)
        let allTokens = promptTokens + decodeTokens
        guard allTokens.allSatisfy({ $0 >= 0 && $0 < runtimeConfig.vocabSize }) else {
            throw MLXFastError.invalidInput(
                "Laguna upstream equivalence token is outside the model vocabulary"
            )
        }

        let configURL = URL(fileURLWithPath: weightsPath)
            .appendingPathComponent("config.json")
        let upstreamConfig = try JSONDecoder().decode(
            LagunaConfiguration.self,
            from: Data(contentsOf: configURL)
        )
        let loader = try LagunaWeightLoader(weightsPath: weightsPath)
        try loader.denseStore.validateReadableByteRanges()
        try loader.validateRequiredMetadata(config: runtimeConfig)

        // Load the 21.6 GB tensor set once, then install the same MLXArray
        // objects into both module trees. Their random initializer graphs stay
        // lazy and are overwritten before either model is evaluated.
        let loadedWeights = try loadRuntimeWeightArrays(denseStore: loader.denseStore)
        let runtime = LagunaRuntimeModel(runtimeConfig)
        let upstream = LagunaModel(upstreamConfig)
        try runtime.update(
            parameters: ModuleParameters.unflattened(
                runtime.sanitize(weights: loadedWeights)
            ),
            verify: [.all]
        )
        try upstream.update(
            parameters: ModuleParameters.unflattened(
                upstream.sanitize(weights: loadedWeights)
            ),
            verify: [.all]
        )
        eval(runtime)
        eval(upstream)

        let runtimeCache = runtime.newCache(parameters: nil)
        let upstreamCache = upstream.newCache(parameters: nil)
        let prompt = MLXArray(promptTokens.map(Int32.init), [1, promptTokens.count])

        var steps: [LagunaUpstreamEquivalenceStep] = []
        steps.reserveCapacity(1 + decodeTokens.count)
        steps.append(
            try compareLastLogitRow(
                label: "prefill",
                runtime: runtime(prompt, cache: runtimeCache),
                upstream: upstream(prompt, cache: upstreamCache),
                vocabularySize: runtimeConfig.vocabSize
            )
        )

        for (index, token) in decodeTokens.enumerated() {
            let input = MLXArray([Int32(token)], [1, 1])
            steps.append(
                try compareLastLogitRow(
                    label: "decode-\(index)",
                    runtime: runtime(input, cache: runtimeCache),
                    upstream: upstream(input, cache: upstreamCache),
                    vocabularySize: runtimeConfig.vocabSize
                )
            )
        }

        return LagunaUpstreamEquivalenceReport(
            promptTokenCount: promptTokens.count,
            decodeTokenCount: decodeTokens.count,
            steps: steps
        )
    }

    private static func compareLastLogitRow(
        label: String,
        runtime runtimeLogits: MLXArray,
        upstream upstreamLogits: MLXArray,
        vocabularySize: Int
    ) throws -> LagunaUpstreamEquivalenceStep {
        let runtimeLast = runtimeLogits.reshaped([-1, vocabularySize])[-1]
        let upstreamLast = upstreamLogits.reshaped([-1, vocabularySize])[-1]
        guard runtimeLast.shape == upstreamLast.shape else {
            throw MLXFastError.invalidInput(
                "Laguna upstream equivalence \(label) logit shapes differ: "
                    + "\(runtimeLast.shape) vs \(upstreamLast.shape)"
            )
        }

        let runtimeFinite = all(isFinite(runtimeLast))
        let upstreamFinite = all(isFinite(upstreamLast))
        eval(runtimeLast, upstreamLast, runtimeFinite, upstreamFinite)
        let runtimeLogitsAreFinite = runtimeFinite.item(Bool.self)
        let upstreamLogitsAreFinite = upstreamFinite.item(Bool.self)
        guard runtimeLogitsAreFinite, upstreamLogitsAreFinite else {
            throw MLXFastError.invalidInput(
                "Laguna upstream equivalence \(label) produced non-finite logits: "
                    + "runtime_finite=\(runtimeLogitsAreFinite) "
                    + "upstream_finite=\(upstreamLogitsAreFinite)"
            )
        }

        let difference = abs(
            runtimeLast.asType(.float32) - upstreamLast.asType(.float32)
        )
        eval(difference)
        return LagunaUpstreamEquivalenceStep(
            label: label,
            maximumAbsoluteLogitError: difference.max().item(Float.self),
            meanAbsoluteLogitError: difference.mean().item(Float.self),
            runtimeToken: Int(runtimeLast.argMax().item(Int32.self)),
            upstreamToken: Int(upstreamLast.argMax().item(Int32.self))
        )
    }
}
