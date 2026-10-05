import MLXFastServing
import Testing

struct MTPTokenStreamTests {
    @Test("An accepted MTP block cannot exceed max_tokens")
    func acceptedBlockHonorsCap() {
        let session = ScriptSession(blocks: [[41, 42, 43], [44, 45]])
        var iterator = MTPTokenIterator(session: session, seedTokens: [7, 8], maxTokens: 2)
        #expect(iterator.next() == 41)
        #expect(iterator.next() == 42)
        #expect(iterator.next() == nil)
        #expect(iterator.tokenCount == 2)
    }

    @Test("Cancellation before prefill performs no model work")
    func cancelledBeforeBegin() async {
        let task = Task {
            try? await Task.sleep(for: .seconds(10))
            let session = ScriptSession(blocks: [[41]])
            var iterator = MTPTokenIterator(session: session, seedTokens: [7], maxTokens: 4)
            let token = iterator.next()
            return (token, session.beginCount, session.roundCount)
        }
        task.cancel()
        let result = await task.value
        #expect(result.0 == nil)
        #expect(result.1 == 0)
        #expect(result.2 == 0)
    }

    @Test("Cancellation after prefill prevents a draft round")
    func cancelledDuringBegin() async {
        let result = await Task {
            let session = ScriptSession(blocks: [[41]], cancelInBegin: true)
            var iterator = MTPTokenIterator(session: session, seedTokens: [7], maxTokens: 4)
            let token = iterator.next()
            return (token, session.roundCount)
        }.value
        #expect(result.0 == nil)
        #expect(result.1 == 0)
    }

    @Test("A failed round poisons the iterator and retains its error")
    func failedRoundStops() {
        let session = ScriptSession(blocks: [], failRound: true)
        let failure = MTPGenerationFailure()
        var iterator = MTPTokenIterator(session: session, seedTokens: [7], maxTokens: 4,
                                       failure: failure)
        #expect(iterator.next() == nil)
        #expect(failure.error is ScriptError)
        #expect(iterator.next() == nil)
        #expect(session.roundCount == 1)
    }
}

private enum ScriptError: Error { case failed }
private final class ScriptSession: MTPServingSession {
    private var blocks: [[Int]]
    private let cancelInBegin: Bool
    private let failRound: Bool
    var beginCount = 0
    var roundCount = 0
    init(blocks: [[Int]], cancelInBegin: Bool = false, failRound: Bool = false) {
        self.blocks = blocks
        self.cancelInBegin = cancelInBegin
        self.failRound = failRound
    }
    func begin(seedTokens: [Int]) throws {
        beginCount += 1
        if cancelInBegin { withUnsafeCurrentTask { $0?.cancel() } }
    }
    func round(depth: Int) throws -> [Int] {
        roundCount += 1
        if failRound { throw ScriptError.failed }
        return blocks.isEmpty ? [] : blocks.removeFirst()
    }
}
