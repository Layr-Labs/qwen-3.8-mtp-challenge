import Foundation
import Testing


// MARK: - Hidden-golden single occupancy in the DFlash job

/// The DFlash job now runs the serial pipeline's hidden gates AND its own hidden
/// parity gate, so TWO hidden goldens exist in one job. The worker Seatbelt
/// profile denies exactly ONE literal path (`BENCH_GOLDEN_PATH`), so if the
/// serial augmented golden is still resident in the bench workspace while the
/// DFlash gate's worker runs — or the DFlash golden is resident while the serial
/// gates' worker runs — submitted model code can read the other track's hidden
/// golden and use it as a token oracle.
///
/// The defense is ORDER, not a new check: each hidden golden is installed only
/// after the previous one has been removed. Order is invisible to every other
/// test, and grep is the only cheap way to keep it from eroding, so it is pinned
/// here.
@Suite
struct DFlashHiddenGoldenOccupancyTests {
    private typealias S = DFlashGateTextSupport

    /// Byte offset of the first line satisfying `predicate`, over the
    /// comment-stripped workflow.
    private func lineOffset(
        _ workflow: String,
        _ label: String,
        where predicate: (String) -> Bool
    ) throws -> Int {
        var cursor = 0
        for raw in workflow.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(raw)
            if !line.trimmingCharacters(in: .whitespaces).hasPrefix("#"), predicate(line) {
                return cursor
            }
            cursor += line.count + 1
        }
        Issue.record("the DFlash workflow has no \(label)")
        throw HiddenGoldenOrderingFailure.missing(label)
    }

    private enum HiddenGoldenOrderingFailure: Error {
        case missing(String)
    }

    // Nothing hidden may enter the workspace before the public tripwire has run:
    // the worker executing the public case must not be able to read any hidden
    // byte, which is exactly why the serial pipeline orders it first.
    @Test
    func thePublicBehaviorGatePrecedesEveryHiddenDownload() throws {
        let workflow = try S.text(S.dflashWorkflowPath)
        let publicGate = try #require(
            S.offset(of: "- name: Public behavior gate\n", in: workflow),
            """
            the DFlash workflow has no 'Public behavior gate' step, so a \
            correctness-only dispatch produces no verdict at all and the hidden \
            downloads are the first thing a worker sees
            """
        )
        let firstHiddenDownload = try lineOffset(
            workflow, "hidden R2 download"
        ) { $0.contains("download-r2-object.sh") }
        #expect(
            publicGate < firstHiddenDownload,
            """
            the public behavior gate must run BEFORE any hidden object is \
            downloaded (public gate at offset \(publicGate), first hidden \
            download at \(firstHiddenDownload)). Ordering is the security \
            property: the worker running the public case must not be able to \
            read a hidden golden.
            """
        )
    }

    // The two hidden goldens must never be workspace-resident at the same time.
    @Test
    func theSerialHiddenGoldenIsScrubbedBeforeTheDFlashGoldenIsInstalled() throws {
        let workflow = try S.text(S.dflashWorkflowPath)

        let dflashInstall = try lineOffset(
            workflow, "DFlash hidden golden install"
        ) {
            $0.contains("install -m 0444") && $0.contains("dflash_correctness_golden.json")
        }

        let serialGoldenRemoved = try lineOffset(
            workflow, "removal of the serial augmented golden"
        ) { $0.contains("rm ") && $0.contains("correctness_golden_ranked.json") }
        #expect(
            serialGoldenRemoved < dflashInstall,
            """
            the serial augmented hidden golden (correctness_golden_ranked.json) \
            is still in the bench workspace when the DFlash hidden golden is \
            installed (removal at \(serialGoldenRemoved), install at \
            \(dflashInstall)). BENCH_GOLDEN_PATH denies exactly one literal \
            path, so the DFlash gate's worker could read the serial golden.
            """
        )

        let privatePurged = try lineOffset(
            workflow, "purge of the bench private/ dir"
        ) { $0.contains("rm -rf") && $0.contains("${MLXFAST_JOB_WS}/private") }
        #expect(
            privatePurged < dflashInstall,
            """
            the gates phase's bench-written private/ capture (which embeds \
            hidden GPQA reference answers) survives into the DFlash gate \
            (purge at \(privatePurged), DFlash install at \(dflashInstall))
            """
        )

        let rankedSrcRemoved = try lineOffset(
            workflow, "removal of the serial .ranked-src staging dir"
        ) { $0.contains("rm -rf") && $0.contains(".ranked-src") && !$0.contains(".dflash-ranked-src") }
        #expect(
            rankedSrcRemoved < dflashInstall,
            """
            the serial .ranked-src staging dir (raw hidden golden + GPQA answer \
            key) is not removed before the DFlash golden is installed
            """
        )
    }

    // ...and the DFlash golden must be in place before the gate that reads it,
    // otherwise the parity gate fails on a missing file rather than on the
    // submission.
    @Test
    func theDFlashGoldenIsInstalledBeforeTheDFlashParityGate() throws {
        let workflow = try S.text(S.dflashWorkflowPath)
        let install = try lineOffset(
            workflow, "DFlash hidden golden install"
        ) {
            $0.contains("install -m 0444") && $0.contains("dflash_correctness_golden.json")
        }
        let gate = try #require(
            S.offset(
                of: "- name: DFlash correctness and parity gate (untimed)\n",
                in: workflow
            ),
            "the DFlash parity gate step is gone"
        )
        #expect(install < gate)

        // Each gate's worker deny is the golden that gate actually reads.
        let dflashGate = try S.stepBody(
            workflow, "DFlash correctness and parity gate (untimed)"
        )
        #expect(dflashGate.contains(
            "BENCH_GOLDEN_PATH: ${{ env.MLXFAST_JOB_WS }}/.dflash-ranked-src/dflash_correctness_golden.json"
        ))
        let serialGates = try S.stepBody(
            workflow, "Correctness and gates (full base case + hidden gates, no timing)"
        )
        #expect(
            serialGates.contains(
                "BENCH_GOLDEN_PATH: ${{ env.MLXFAST_JOB_WS }}/correctness_golden_ranked.json"
            ),
            """
            the reused gates pass must keep the serial worker deny on the \
            augmented golden; without it the worker that runs submitted model \
            code can read the hidden golden it is being teacher-forced against
            """
        )
    }
}
