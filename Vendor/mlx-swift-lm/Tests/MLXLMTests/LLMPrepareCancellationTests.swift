import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

struct LLMPrepareCancellationTests {
    @Test func cancelledShortPromptDoesNotStartModelWork() async {
        let result = await prepareSyntheticPrompt(tokens: 1, cancelBefore: true)
        #expect(result.cancelled)
        #expect(result.forwardedChunks.isEmpty)
        #expect(result.failure == nil)
    }

    @Test func cancellationDuringLastChunkDoesNotReturnRemainingTokens() async {
        let result = await prepareSyntheticPrompt(tokens: 513, cancelDuringForward: true)
        #expect(result.cancelled)
        #expect(result.forwardedChunks == [512])
        #expect(result.remainingTokens.isEmpty)
        #expect(result.failure == nil)
    }

    @Test func cancellationDuringFirstChunkDoesNotStartNextChunk() async {
        let result = await prepareSyntheticPrompt(tokens: 1025, cancelDuringForward: true)
        #expect(result.cancelled)
        #expect(result.forwardedChunks == [512])
        #expect(result.failure == nil)
    }

    @Test func uncancelledPromptPreservesChunksAndRemainingToken() async {
        let result = await prepareSyntheticPrompt(tokens: 1025)
        #expect(!result.cancelled)
        #expect(result.forwardedChunks == [512, 512])
        #expect(result.remainingTokens == [1024])
        #expect(result.failure == nil)
    }
}

private struct PreparationResult: Sendable {
    let cancelled: Bool
    let forwardedChunks: [Int]
    let remainingTokens: [Int32]
    let failure: String?
}

/// The task owns all arrays and the tiny model. No weights, GPU operations,
/// network access, timers, or race-dependent task handles are needed.
private func prepareSyntheticPrompt(
    tokens: Int, cancelBefore: Bool = false, cancelDuringForward: Bool = false
) async -> PreparationResult {
    await Task {
        Device.withDefaultDevice(.cpu) {
            let model = CancellationProbeModel(cancelDuringForward: cancelDuringForward)
            let input = LMInput(tokens: MLXArray((0..<tokens).map { Int32($0) }))
            if cancelBefore {
                withUnsafeCurrentTask { $0?.cancel() }
            }
            do {
                let prepared = try model.prepare(input, cache: [], windowSize: 512)
                guard case .tokens(let remaining) = prepared else {
                    return PreparationResult(
                        cancelled: false, forwardedChunks: model.forwardedChunks,
                        remainingTokens: [], failure: "Unexpected logits from default prepare"
                    )
                }
                return PreparationResult(
                    cancelled: false, forwardedChunks: model.forwardedChunks,
                    remainingTokens: remaining.tokens.asArray(Int32.self), failure: nil
                )
            } catch is CancellationError {
                return PreparationResult(
                    cancelled: true, forwardedChunks: model.forwardedChunks,
                    remainingTokens: [], failure: nil
                )
            } catch {
                return PreparationResult(
                    cancelled: false, forwardedChunks: model.forwardedChunks,
                    remainingTokens: [], failure: String(describing: error)
                )
            }
        }
    }.value
}

private final class CancellationProbeModel: Module, LLMModel {
    let cancelDuringForward: Bool
    private(set) var forwardedChunks: [Int] = []
    var loraLayers: [Module] { [] }

    init(cancelDuringForward: Bool) {
        self.cancelDuringForward = cancelDuringForward
        super.init()
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forwardedChunks.append(inputs.dim(1))
        if cancelDuringForward {
            withUnsafeCurrentTask { $0?.cancel() }
        }
        return inputs
    }
}
