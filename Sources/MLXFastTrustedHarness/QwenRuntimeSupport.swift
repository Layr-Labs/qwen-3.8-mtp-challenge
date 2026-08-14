import Darwin
import Foundation
#if !MLXFAST_TRUSTED_HARNESS
import MLX
#endif
import MLXFastCore
import Tokenizers

// QwenRuntime is split across QwenRuntime*.swift for auditability.
// Generated split; behavior identical to the original single file.

extension QwenRuntime {
    static func currentResidentMemoryGB() -> Double {
        guard let info = processMemoryInfo() else {
            return 0
        }
        return Double(info.resident_size) / Double(1 << 30)
    }

    static func peakResidentMemoryGB() -> Double {
        guard let info = processMemoryInfo() else {
            return 0
        }
        return Double(info.resident_size_max) / Double(1 << 30)
    }

    private static func processMemoryInfo() -> mach_task_basic_info? {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.stride / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    rebound,
                    &count
                )
            }
        }
        guard result == KERN_SUCCESS else {
            return nil
        }
        return info
    }

    static func secondsSince(_ start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000.0
    }

    static func makeBenchmarkProgressReporter(startedAt: UInt64) -> (String) -> Void {
        { message in
            let elapsed = formatSeconds(secondsSince(startedAt))
            fputs("mlxfast: benchmark elapsed=\(elapsed)s \(message)\n", stderr)
            fflush(stderr)
        }
    }

    static func formatSeconds(_ value: Double) -> String {
        String(format: "%.1f", value)
    }

    static func formatDouble(_ value: Double) -> String {
        String(format: "%.6f", value)
    }

    static func singleLine(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
    }

    static func redactedProgressError(_ value: String) -> String {
        let line = singleLine(value)
        if line.range(of: "expected", options: .caseInsensitive) != nil
            || line.range(of: "actual", options: .caseInsensitive) != nil
        {
            return "token-validation-failed"
        }
        return line
    }

    static func reportProgress(
        step: Int,
        total: Int,
        intervalSteps: Int,
        progress: ((Int, Int) -> Void)?
    ) {
        guard let progress, intervalSteps > 0 else {
            return
        }
        if step == 1 || step == total || step.isMultiple(of: intervalSteps) {
            progress(step, total)
        }
    }

    static func loadLocalTokenizer(at path: String) throws -> any Tokenizer {
        let modelFolder = URL(fileURLWithPath: path).standardizedFileURL
        return try runBlockingAsync {
            try await AutoTokenizer.from(modelFolder: modelFolder, strict: false)
        }
    }

    final class AsyncResultBox<T>: @unchecked Sendable {
        var result: Result<T, Error>?
    }

    static func runBlockingAsync<T>(
        _ body: @escaping @Sendable () async throws -> T
    ) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        let box = AsyncResultBox<T>()
        Task {
            do {
                box.result = .success(try await body())
            } catch {
                box.result = .failure(error)
            }
            semaphore.signal()
        }
        semaphore.wait()
        return try box.result!.get()
    }

    static func requireFile(_ path: String, description: String) throws {
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory), !isDirectory.boolValue else {
            throw MLXFastError.invalidInput("\(description) missing at \(path)")
        }
    }

    static func trimmedNonEmpty(_ value: String?) -> String? {
        let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return trimmed.isEmpty ? nil : trimmed
    }

    static func requireBenchmarkMatch(_ comparison: BenchmarkTokenComparison) throws {
        guard comparison.passed else {
            throw BenchmarkTokenMismatchError(comparison: comparison)
        }
    }

    #if !MLXFAST_TRUSTED_HARNESS
        static func inputIDsArray(_ tokens: [Int]) throws -> MLXArray {
            guard !tokens.isEmpty else {
                throw MLXFastError.invalidInput("input token array must not be empty")
            }
            let values = try tokens.enumerated().map { index, token -> Int32 in
                guard token >= 0, token < MLXFastConstants.vocabSize else {
                    throw MLXFastError.invalidInput(
                        "input token[\(index)]=\(token) is outside Qwen3.6 vocab range 0..<\(MLXFastConstants.vocabSize)"
                    )
                }
                return Int32(token)
            }
            return MLXArray(values, [1, values.count])
        }
    #endif

}

func correctnessTokenAccepted(
    expectedToken: Int,
    actualToken: Int,
    topLogits: [CorrectnessTraceLogit]?
) -> Bool {
    if actualToken == expectedToken {
        return true
    }
    guard let topLogits,
          let topLogit = topLogits.first?.logit,
          let expectedLogit = topLogits.first(where: { $0.token == expectedToken })?.logit
    else {
        return false
    }
    // Some Apple GPU/Metal combinations break exact argmax ties differently.
    // Accept only a true top-logit tie, and keep feeding the golden token.
    return topLogit - expectedLogit <= MLXFastConstants.correctnessLogitTieTolerance
}

struct DecodeTimingPlan: Equatable {
    let seedTokenCount: Int
    let decodeSteps: Int

    init(seedTokenCount: Int, decodeSteps: Int) throws {
        guard seedTokenCount > 0 else {
            throw MLXFastError.invalidInput("benchmark decode seed must not be empty")
        }
        guard decodeSteps > 0 else {
            throw MLXFastError.invalidInput("benchmark decode steps must be positive")
        }
        self.seedTokenCount = seedTokenCount
        self.decodeSteps = decodeSteps
    }

    func positionOffset(forDecodedStep step: Int) throws -> Int {
        guard step >= 0 && step < decodeSteps else {
            throw MLXFastError.invalidInput(
                "decode step \(step) is outside benchmark range 0..<\(decodeSteps)"
            )
        }
        return seedTokenCount + step
    }
}

struct BenchmarkTokenMismatchError: Error, CustomStringConvertible {
    let label: String
    let step: Int?
    let expectedToken: Int?
    let actualToken: Int?

    init(comparison: BenchmarkTokenComparison) {
        self.label = comparison.label
        self.step = comparison.step
        self.expectedToken = comparison.expectedToken
        self.actualToken = comparison.actualToken
    }

    var description: String {
        var message = "\(label) mismatch"
        if let step {
            message += " at step \(step)"
        }
        return message
    }
}
