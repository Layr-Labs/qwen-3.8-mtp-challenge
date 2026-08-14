import Foundation
import Testing

/// The R2 keys the DFlash goldens actually live under, PROVEN rather than inferred.
///
/// This constant was wrong twice, in opposite directions, because the operator's
/// phrase "gautham-experiments/correctness_prompts/laguna-xs-2.1-dflash/" is
/// ambiguous between a bucket-plus-key and a key: R2_BUCKET_ENDPOINT carries the
/// bucket, and the serial keys are written with no bucket segment. I stripped the
/// segment, then restored it when corrected, then had to strip it again.
///
/// Probe run 30613434387 settled it in about a minute:
///
///     correctness_prompts/laguna-xs-2.1-dflash/dflash_correctness_golden_hidden.json  FOUND 185394b
///     correctness_prompts/laguna-xs-2.1-dflash/dflash_benchmark_golden_hidden.json    FOUND 185433b
///     gautham-experiments/correctness_prompts/...                                     404 NoSuchKey
///
/// Those byte counts are exactly the generated goldens', and the serial control
/// key was FOUND in the same run, so the credentials and bucket were never in
/// question. `gautham-experiments` is the BUCKET.
///
/// A wrong key costs a 30-40 minute ranked dispatch to discover, which is why
/// this is pinned and why dflash-probe-r2-keys.yml exists. If the objects move,
/// probe first, then change this test in the same commit that moves them.
@Suite("DFlash golden R2 keys")
struct DFlashGoldenKeyTests {
    private static let prefix = "correctness_prompts/laguna-xs-2.1-dflash"

    @Test
    func theCorrectnessGoldenKeyMatchesWhereItWasUploaded() throws {
        let wf = try String(
            contentsOfFile: ".github/workflows/dflash-benchmark.yml", encoding: .utf8
        )
        let expected =
            "MLXFAST_DFLASH_CORRECTNESS_GOLDEN_R2_PATH: "
            + "\(Self.prefix)/dflash_correctness_golden_hidden.json"
        #expect(
            wf.contains(expected),
            """
            the correctness golden's R2 key is not the one the objects were \
            uploaded to. Expected the '\(Self.prefix)' prefix; a key missing it \
            fails as 404 NoSuchKey after a full ranked dispatch.
            """
        )
    }

    @Test
    func everyPoolEntryKeyMatchesWhereItWasUploaded() throws {
        let data = try Data(
            contentsOf: URL(fileURLWithPath: "fixtures/laguna_xs_2_1_dflash_track.json")
        )
        let object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let pool = object["timed_prompt_pool"] as? [[String: Any]] ?? []
        #expect(!pool.isEmpty, "the pool is empty; nothing to key-check")
        for (index, entry) in pool.enumerated() {
            let path = entry["r2_path"] as? String ?? ""
            #expect(
                path.hasPrefix(Self.prefix + "/"),
                "pool entry \(index) key '\(path)' is not under \(Self.prefix)"
            )
            // The signer refuses anything outside this set BEFORE signing, so a
            // bad character is a dispatch-time failure, not a run-time one.
            #expect(
                path.allSatisfy { $0.isLetter || $0.isNumber || "._/-".contains($0) },
                "pool entry \(index) key would be refused by the signer's charset guard"
            )
        }
    }
}
