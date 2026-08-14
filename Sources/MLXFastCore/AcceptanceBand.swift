import Foundation

/// Two-sided acceptance band for a scored timing metric (prefill or decode
/// seconds-per-token) against the same-VM paired baseline `B`.
///
/// Both prefill and decode are single, noisy measurements (see
/// docs/thermal-variance-investigation.md). Measured against the same-VM paired
/// baseline (which cancels host-speed differences), a run's value must land within
/// `[B*(1-downTolerance), B*(1+upTolerance)]`:
///   - more than `upTolerance` ABOVE `B` (a slowdown / regression)  -> FAIL.
///   - more than `downTolerance` BELOW `B` (an improvement / faster) -> FAIL.
///
/// The down side caps a single submission's gain: a value far below `B` is either
/// a lucky measurement or a jump too big to trust in one shot, so larger wins must
/// be chunked across submissions. Tolerances are per-axis (see MLXFastConstants):
///   - prefill: +/-5% (prefill is not a real optimization axis -> symmetric gate).
///   - decode: +2% regression / -5% gain (tight on regressions; per-submission
///     decode gain capped at 5%).
public enum AcceptanceBand {
    /// Robust reference `B` from a calibration set: drop the single slowest sample,
    /// average the rest. Returns nil if there are too few / invalid samples.
    /// Computed once at calibration time, never per scored run.
    public static func robustReference(samples: [Double]) -> Double? {
        // Need >= 3 so that after dropping one outlier >= 2 remain to average.
        guard samples.count >= 3, samples.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            return nil
        }
        var rest = samples
        if let slowest = rest.indices.max(by: { rest[$0] < rest[$1] }) {
            rest.remove(at: slowest)
        }
        return rest.reduce(0.0, +) / Double(rest.count)
    }

    /// Per-run gate: validate one scored measurement (`value`) against reference `B`.
    public static func check(
        value: Double,
        reference: Double,
        upTolerance: Double,
        downTolerance: Double,
        label: String = "value"
    ) -> AcceptanceBandResult {
        guard value.isFinite, value > 0, reference.isFinite, reference > 0 else {
            return AcceptanceBandResult(
                passed: false,
                reason: "\(label) (\(value)) and reference (\(reference)) must be finite and positive"
            )
        }
        let hi = reference * (1.0 + upTolerance)
        let lo = reference * (1.0 - downTolerance)
        if value > hi {
            return AcceptanceBandResult(
                passed: false,
                reason: "\(label) \(value) exceeds +\(upTolerance * 100)% of reference "
                    + "\(reference) (> \(hi)): slowdown/regression beyond tolerance"
            )
        }
        if value < lo {
            return AcceptanceBandResult(
                passed: false,
                reason: "\(label) \(value) below -\(downTolerance * 100)% of reference "
                    + "\(reference) (< \(lo)): improvement too large for one submission "
                    + "(chunk it) or a suspiciously lucky reading"
            )
        }
        return AcceptanceBandResult(passed: true, reason: "")
    }
}

public struct AcceptanceBandResult: Equatable {
    public let passed: Bool
    /// Empty when `passed`; otherwise a human-readable failure reason.
    public let reason: String

    public init(passed: Bool, reason: String) {
        self.passed = passed
        self.reason = reason
    }
}
