import Foundation
@testable import MLXFastCore
import Testing

// Acceptance bands: a robust reference B is calibrated once from a fleet (drop the
// single slowest, average the rest); each run's single measurement is then checked
// against B with per-axis tolerances. Prefill is +/-5% (symmetric health gate);
// decode is +2% regression / -5% gain (tight on regressions, per-submission gain
// capped). Numbers anchored on the real tenki fresh-VM run 28893815980
// (prefills: 5 tight at ~0.0106, one spike at 0.01532).

private let freshVMPrefills = [
    0.010556244302734375, 0.01071322314453125, 0.010636167642578125,
    0.010626223958984375, 0.01532235343359375, 0.01049330069921875,
]

@Test
func robustReferenceDropsTheSlowestAndAverages() {
    let B = AcceptanceBand.robustReference(samples: freshVMPrefills)
    #expect(B != nil)
    // mean of the 5 after dropping the 0.01532 spike
    #expect(abs((B ?? 0) - 0.0106048) < 1e-5)
}

@Test
func robustReferenceRejectsTooFewOrInvalid() {
    #expect(AcceptanceBand.robustReference(samples: [0.0106, 0.0106]) == nil)
    #expect(AcceptanceBand.robustReference(samples: [0.0106, 0.0, 0.0106]) == nil)
    #expect(AcceptanceBand.robustReference(samples: [0.0106, .nan, 0.0106]) == nil)
}

// MARK: - Prefill band (+/-5% symmetric)

private func checkPrefill(_ value: Double, _ reference: Double) -> AcceptanceBandResult {
    AcceptanceBand.check(
        value: value, reference: reference,
        upTolerance: MLXFastConstants.prefillBandUpTolerance,
        downTolerance: MLXFastConstants.prefillBandDownTolerance, label: "prefill"
    )
}

@Test
func prefillAcceptsAtReferenceAndModestlyFaster() {
    let B = 0.0106
    #expect(checkPrefill(B, B).passed)          // exactly at B
    #expect(checkPrefill(B * 1.04, B).passed)   // +4% slow: ok
    #expect(checkPrefill(B * 0.97, B).passed)   // -3% fast: ok
}

@Test
func prefillFailsWhenMoreThan5PercentSlow() {
    let B = 0.0106
    #expect(checkPrefill(B * 1.04, B).passed)   // +4% slow: still ok
    #expect(!checkPrefill(B * 1.051, B).passed) // +5.1% slow: fail
    #expect(checkPrefill(B * 1.051, B).reason.contains("slowdown"))
    // the real 0.01532 spike vs B=0.010605 is +44% -> fail
    #expect(!checkPrefill(0.01532235343359375, 0.0106048).passed)
}

@Test
func prefillFailsWhenMoreThan5PercentFast() {
    let B = 0.0106
    #expect(!checkPrefill(B * 0.949, B).passed)
    #expect(checkPrefill(B * 0.949, B).reason.contains("chunk"))
}

// MARK: - Decode band (+2% regression / -5% gain)

private func checkDecode(_ value: Double, _ reference: Double) -> AcceptanceBandResult {
    AcceptanceBand.check(
        value: value, reference: reference,
        upTolerance: MLXFastConstants.decodeBandUpTolerance,
        downTolerance: MLXFastConstants.decodeBandDownTolerance, label: "decode"
    )
}

@Test
func decodeAcceptsSmallRegressionAndUpTo5PercentGain() {
    let B = 0.1336
    #expect(checkDecode(B, B).passed)           // exactly at B
    #expect(checkDecode(B * 1.02, B).passed)    // +2% slower: at the regression cap, ok
    #expect(checkDecode(B * 0.96, B).passed)    // -4% faster: within the 5% gain cap, ok
    #expect(checkDecode(B * 0.95, B).passed)    // -5% faster: exactly at the gain cap, ok
}

@Test
func decodeFailsRegressionBeyond2Percent() {
    let B = 0.1336
    #expect(!checkDecode(B * 1.021, B).passed)  // +2.1% slower: fail (decode is the scored axis)
    #expect(checkDecode(B * 1.021, B).reason.contains("slowdown"))
}

@Test
func decodeFailsGainBeyond5Percent() {
    let B = 0.1336
    #expect(!checkDecode(B * 0.949, B).passed)  // -5.1% faster: too big for one submission
    #expect(checkDecode(B * 0.949, B).reason.contains("chunk"))
}

@Test
func checkRejectsNonFinite() {
    #expect(!checkPrefill(0.0, 0.0106).passed)
    #expect(!checkPrefill(0.0106, 0.0).passed)
    #expect(!checkPrefill(.nan, 0.0106).passed)
}

@Test
func bandTolerancesMatchConstants() {
    #expect(MLXFastConstants.prefillBandUpTolerance == 0.05)
    #expect(MLXFastConstants.prefillBandDownTolerance == 0.05)
    #expect(MLXFastConstants.decodeBandUpTolerance == 0.02)
    #expect(MLXFastConstants.decodeBandDownTolerance == 0.05)
}
