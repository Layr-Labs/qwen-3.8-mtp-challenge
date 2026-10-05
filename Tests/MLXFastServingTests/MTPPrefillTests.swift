import MLX
@testable import MLXFastModel
import MLXLMCommon
import Testing

struct MTPPrefillTests {
    @Test("Serving prefill is bounded and retains every prompt hidden row")
    func boundedPrefillRetainsHiddenHistory() async throws {
        let result = try await Task {
            try Device.withDefaultDevice(.cpu) {
                let target = PrefillTarget()
                let session = try Qwen36MTPBlockSession(model: target, stopTokens: [])
                let hidden = try session.prefillForServing(seedTokens: Array(1...1025))
                return (target.widths, hidden.asArray(Int32.self))
            }
        }.value
        #expect(result.0 == [512, 512, 1])
        #expect(result.1 == Array(1...1025).map(Int32.init))
    }

    @Test("Cancellation during a chunk prevents later forwards or a successful return", arguments: [1, 513, 1025])
    func cancellationStopsChunkedPrefill(tokens: Int) async throws {
        let result = try await Task {
            try Device.withDefaultDevice(.cpu) {
                let target = PrefillTarget(cancelInForward: true)
                let session = try Qwen36MTPBlockSession(model: target, stopTokens: [])
                do {
                    _ = try session.prefillForServing(seedTokens: Array(1...tokens))
                    return (false, target.widths)
                } catch is CancellationError { return (true, target.widths) }
            }
        }.value
        #expect(result.0)
        #expect(result.1 == [min(512, tokens)])
    }

    @Test("An already canceled seed performs no target forward")
    func cancellationBeforeSeed() async throws {
        let widths = try await Task {
            try Device.withDefaultDevice(.cpu) {
                let target = PrefillTarget()
                let session = try Qwen36MTPBlockSession(model: target, stopTokens: [])
                withUnsafeCurrentTask { $0?.cancel() }
                do {
                    _ = try session.prefillForServing(seedTokens: [1])
                    Issue.record("Canceled prefill returned normally")
                } catch is CancellationError {}
                return target.widths
            }
        }.value
        #expect(widths.isEmpty)
    }
}

private final class PrefillTarget: Qwen36MTPTarget {
    var hasMTPHead: Bool { true }
    var widths: [Int] = []
    let cancelInForward: Bool
    init(cancelInForward: Bool = false) { self.cancelInForward = cancelInForward }
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
    func callWithHidden(input: LMInput.Text, cache: [any KVCache], nConfirmed: Int) -> (MLXArray, MLXArray) {
        widths.append(input.tokens.dim(1))
        if cancelInForward { withUnsafeCurrentTask { $0?.cancel() } }
        let hidden = input.tokens.reshaped([1, input.tokens.size, 1])
        return (hidden, hidden)
    }
    func replayRecurrentPrefix(cache: [any KVCache], committedRows: Int) -> Bool { false }
    func mtpForwardWithHidden(hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]) -> (MLXArray, MLXArray) { (hidden, hidden) }
    func mtpHeadHiddenForward(hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]) -> MLXArray { hidden }
    func mtpHeadLastHiddenWithKVOnlyHistory(hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]) -> MLXArray? { nil }
    func applyLMHead(_ x: MLXArray) -> MLXArray { x }
    func applyDraftLMHead(_ x: MLXArray) -> MLXArray { x }
    func mapDraftTokenIds(_ ids: MLXArray) -> MLXArray { ids }
    func draftTokenID(_ x: MLXArray) -> MLXArray { x }
    func makeMTPCache() -> [any KVCache] { [] }
    func applyFinalNorm(_ x: MLXArray) -> MLXArray { x }
}
