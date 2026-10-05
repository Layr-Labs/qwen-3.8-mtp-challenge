@testable import MLXFastServing
import Testing

struct MTPStartupProgressTests {
    @Test("Startup progress precedes each action and retains the exact elapsed line",
          arguments: [
            (MTPStartupStage.validateHead, "validate_head"),
            (.validateTarget, "validate_target"),
            (.loadWeights, "load_weights"),
            (.warmKernels, "warm_kernels"),
          ])
    func stageBeginsBeforeAction(stage: MTPStartupStage, name: String) throws {
        var events: [String] = []
        var clock: Double = 12
        let value = stage.perform(now: { clock }, write: { events.append($0) }) {
            events.append("action")
            clock = 12.625
            return 42
        }
        #expect(value == 42)
        #expect(events == ["native_mtp stage=\(name) state=started\n", "action",
                           "native_mtp stage=\(name) elapsed_ms=625\n"])
    }

    @Test("A failed startup action has a begin marker and no false completion",
          arguments: [
            (MTPStartupStage.validateHead, "validate_head"),
            (.validateTarget, "validate_target"),
            (.loadWeights, "load_weights"),
            (.warmKernels, "warm_kernels"),
          ])
    func failedStageNeverCompletes(stage: MTPStartupStage, name: String) {
        var events: [String] = []
        #expect(throws: ProgressProbeError.failed) {
            try stage.perform(now: { 12 }, write: { events.append($0) }) {
                events.append("action")
                throw ProgressProbeError.failed
            }
        }
        #expect(events == ["native_mtp stage=\(name) state=started\n", "action"])
    }
}

private enum ProgressProbeError: Error { case failed }
