import Foundation
import MLX
import MLXFastCore

public struct CorrectnessTokenComparison: Equatable {
    public let passed: Bool
    public let checkedSteps: Int
    public let firstFailingStep: Int?
    public let expectedToken: Int?
    public let actualToken: Int?

    public init(
        passed: Bool,
        checkedSteps: Int,
        firstFailingStep: Int?,
        expectedToken: Int?,
        actualToken: Int?
    ) {
        self.passed = passed
        self.checkedSteps = checkedSteps
        self.firstFailingStep = firstFailingStep
        self.expectedToken = expectedToken
        self.actualToken = actualToken
    }
}

public enum QwenCorrectness {
    public static func generateGreedyNoCache(
        promptTokens: [Int],
        steps: Int = MLXFastConstants.correctnessSteps,
        nextToken: (_ contextTokens: [Int]) throws -> Int
    ) throws -> [Int] {
        guard !promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("greedy correctness prompt must not be empty")
        }
        guard steps >= 0 else {
            throw MLXFastError.invalidInput("greedy correctness steps must be non-negative")
        }

        var context = promptTokens
        var generated: [Int] = []
        generated.reserveCapacity(steps)
        for _ in 0..<steps {
            let token = try nextToken(context)
            generated.append(token)
            context.append(token)
        }
        return generated
    }

    public static func compareTeacherForcedNoCache(
        promptTokens: [Int],
        expectedTokens: [Int],
        steps: Int = MLXFastConstants.correctnessSteps,
        nextToken: (_ contextTokens: [Int]) throws -> Int
    ) throws -> CorrectnessTokenComparison {
        // Reference implementation mirroring compareTeacherForcedWithWorker's
        // teacher-forced seeding in a form that's directly unit-testable without a
        // live worker: every step's context is the known golden prefix, never the
        // model's own prior output.
        guard !promptTokens.isEmpty else {
            throw MLXFastError.invalidInput("teacher-forced correctness prompt must not be empty")
        }
        guard steps >= 0 else {
            throw MLXFastError.invalidInput("teacher-forced correctness steps must be non-negative")
        }

        var context = promptTokens
        for step in 0..<steps {
            guard step < expectedTokens.count else {
                return CorrectnessTokenComparison(
                    passed: false,
                    checkedSteps: step + 1,
                    firstFailingStep: step,
                    expectedToken: nil,
                    actualToken: nil
                )
            }
            let expectedToken = expectedTokens[step]
            let actualToken = try nextToken(context)
            if actualToken != expectedToken {
                return CorrectnessTokenComparison(
                    passed: false,
                    checkedSteps: step + 1,
                    firstFailingStep: step,
                    expectedToken: expectedToken,
                    actualToken: actualToken
                )
            }
            context.append(expectedToken)
        }

        return CorrectnessTokenComparison(
            passed: true,
            checkedSteps: steps,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil
        )
    }

    public static func greedyToken(from logits: MLXArray) throws -> Int {
        guard let vocabSize = logits.shape.last, vocabSize > 0 else {
            throw MLXFastError.invalidInput("greedy logits must have a non-empty vocab dimension")
        }
        let rows = logits.reshaped([-1, vocabSize])
        let last = rows[-1]
        return Int(last.argMax().item(Int32.self))
    }

    public static func compareTokens(
        expected: [Int],
        actual: [Int],
        steps: Int = MLXFastConstants.correctnessSteps
    ) -> CorrectnessTokenComparison {
        guard steps > 0 else {
            return CorrectnessTokenComparison(
                passed: true,
                checkedSteps: 0,
                firstFailingStep: nil,
                expectedToken: nil,
                actualToken: nil
            )
        }

        for step in 0..<steps {
            let expectedToken = step < expected.count ? expected[step] : nil
            let actualToken = step < actual.count ? actual[step] : nil
            guard let expectedToken else {
                return CorrectnessTokenComparison(
                    passed: false,
                    checkedSteps: step + 1,
                    firstFailingStep: step,
                    expectedToken: nil,
                    actualToken: actualToken
                )
            }
            guard let actualToken else {
                return CorrectnessTokenComparison(
                    passed: false,
                    checkedSteps: step + 1,
                    firstFailingStep: step,
                    expectedToken: expectedToken,
                    actualToken: nil
                )
            }
            if actualToken != expectedToken {
                return CorrectnessTokenComparison(
                    passed: false,
                    checkedSteps: step + 1,
                    firstFailingStep: step,
                    expectedToken: expectedToken,
                    actualToken: actualToken
                )
            }
        }

        return CorrectnessTokenComparison(
            passed: true,
            checkedSteps: steps,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil
        )
    }
}
