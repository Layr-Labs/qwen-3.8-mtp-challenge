// CBv2PagedEligibilityTests.swift
//
// Static eligibility of attention shapes for the paged decode kernels.
// The Swift-side threadgroup-memory model (PagedAttentionKernel
// .partThreadgroupBytes / headsPerThreadgroup / ineligibilityReason) must
// match the shader's allocations byte-for-byte — INCLUDING the head split
// (HPT), which is what keeps large head dims under the cap — and
// PagedKVBackend must refuse over-budget shapes at construction:
// dispatching one is an UNCATCHABLE Metal fatal ("Threadgroup memory size
// (32832) exceeds the maximum (32768)"), not a thrown error. History:
// Gemma-4 global layers (headDim 512, GQA 8) passed the old head-dim-only
// eligibility and killed the process on the first dispatch; they are now
// eligible again via the head split, and the formula models it. No model
// weights or GPU dispatch required.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("CBv2PagedEligibility")
struct CBv2PagedEligibilityTests {

    // MARK: - Helpers

    private func fullKind(
        headDim: Int, kvHeads: Int, queryHeads: Int, sinks: Bool = false
    ) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .full, hasSinks: sinks, headDim: headDim, kvHeads: kvHeads,
            queryHeads: queryHeads)
    }

    private func poolConfig() -> PagedKVPoolConfig {
        PagedKVPoolConfig(
            capacityBytes: 8 << 20, maxPrefillChunk: 64,
            nominalMaxSequenceLength: 1024)
    }

    private func expectIneligible(
        _ kinds: [CBv2LayerKind], reasonContains fragment: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        do {
            _ = try PagedKVBackend(layerKinds: kinds, config: poolConfig())
            Issue.record(
                "expected backendIneligible for \(kinds)", sourceLocation: sourceLocation)
        } catch CBv2KVError.backendIneligible(let reason) {
            #expect(
                reason.contains(fragment),
                "reason \"\(reason)\" should mention \"\(fragment)\"",
                sourceLocation: sourceLocation)
        } catch {
            Issue.record(
                "unexpected error: \(error)", sourceLocation: sourceLocation)
        }
    }

    // MARK: - (c) Budget math vs the shader's constants

    @Test func headSplitModel() {
        // HPT = largest divisor of GQA with acc floats (HPT * D / 32)
        // within the 32-float per-thread budget. Fleet d64/d128 shapes are
        // unsplit (HPT == GQA); d256 splits at GQA 8; d512 runs 2 heads
        // per threadgroup (Gemma-4 global layers).
        #expect(PagedAttentionKernel.headsPerThreadgroup(headDim: 64, gqa: 8) == 8)
        #expect(PagedAttentionKernel.headsPerThreadgroup(headDim: 128, gqa: 8) == 8)
        #expect(PagedAttentionKernel.headsPerThreadgroup(headDim: 256, gqa: 8) == 4)
        #expect(PagedAttentionKernel.headsPerThreadgroup(headDim: 512, gqa: 8) == 2)
        #expect(PagedAttentionKernel.headsPerThreadgroup(headDim: 512, gqa: 2) == 2)
        #expect(PagedAttentionKernel.headsPerThreadgroup(headDim: 512, gqa: 1) == 1)
        // Non-divisible caps fall back to the largest divisor.
        #expect(PagedAttentionKernel.headsPerThreadgroup(headDim: 512, gqa: 3) == 1)
        #expect(PagedAttentionKernel.headsPerThreadgroup(headDim: 256, gqa: 6) == 3)

        // The split must always divide GQA and respect the register cap.
        for d in PagedAttentionKernel.supportedHeadDims {
            for g in 1 ... 16 {
                let hpt = PagedAttentionKernel.headsPerThreadgroup(headDim: d, gqa: g)
                #expect(hpt >= 1 && g % hpt == 0, "d=\(d) gqa=\(g)")
                #expect(hpt * d / 32 <= 32 || hpt == 1, "d=\(d) gqa=\(g): acc over budget")
            }
        }
    }

    @Test func budgetFormulaMatchesShaderAllocations() {
        // The shader allocates q_smem[HPT*D] + red_smem[NSG*HPT*(D+2)]
        // floats (see pagedattention.metal "single source of truth"),
        // where HPT is the head split — NOT the full GQA factor.
        // Regression anchor: Gemma-4 global layers (d512, GQA 8) needed
        // (8*512 + 1*8*514) * 4 = 32,832 B unsplit — 64 B over the cap and
        // a process-fatal at first dispatch. Split at HPT=2 they need
        // (2*512 + NSG*2*514) * 4 → 20,544 B at NSG=4.
        #expect(
            PagedAttentionKernel.partThreadgroupBytes(headDim: 512, gqa: 8, simdgroups: 1)
                == 8208)
        #expect(
            PagedAttentionKernel.partThreadgroupBytes(headDim: 512, gqa: 8, simdgroups: 4)
                == 20544)
        #expect(PagedAttentionKernel.threadgroupMemoryLimit == 32768)
        #expect(PagedAttentionKernel.mergeRecordMetaFloats == 2)

        // General formula on a spread of shapes.
        for (d, g, n) in [(64, 8, 8), (128, 8, 4), (256, 2, 2), (512, 1, 8), (512, 4, 2)] {
            let hpt = PagedAttentionKernel.headsPerThreadgroup(headDim: d, gqa: g)
            let expected = (hpt * d + n * hpt * (d + 2)) * 4
            #expect(
                PagedAttentionKernel.partThreadgroupBytes(headDim: d, gqa: g, simdgroups: n)
                    == expected,
                "d=\(d) gqa=\(g) nsg=\(n)")
        }
    }

    /// Drift guard: the generated kernel bodies must still allocate exactly
    /// the buffers the Swift budget models — two float threadgroup buffers
    /// in each pass-A variant (q_smem, red_smem with RSTRIDE = D + 2), none
    /// in pass B or the bulk-write kernel.
    @Test func shaderSourceStillMatchesBudgetModel() {
        for part in [PagedAttentionMSL.partBody, PagedAttentionMSL.partBodyNoWrite] {
            #expect(part.contains("threadgroup float q_smem[HPT * D];"))
            #expect(part.contains("threadgroup float red_smem[NSG * HPT * (D + 2)];"))
            #expect(
                part.components(separatedBy: "threadgroup float").count - 1 == 2,
                "part bodies must allocate exactly the two modeled buffers")
        }
        #expect(
            !PagedAttentionMSL.mergeBody.contains("threadgroup float"),
            "merge body must allocate no threadgroup memory")
        #expect(
            !PagedAttentionMSL.writeBody.contains("threadgroup"),
            "bulk-write body must allocate no threadgroup memory")
        // RSTRIDE in the .metal impl mirrors mergeRecordMetaFloats.
        #expect(PagedAttentionMSL.header.contains("RSTRIDE = D + 2"))
    }

    @Test func simdgroupSelectionRespectsBudget() {
        // Largest NSG whose staging + merge buffers fit 32 KB (with the
        // head split applied). d64/d128 fleet shapes are unsplit and keep
        // their historical NSG; d256/d512 split and gain simdgroups.
        #expect(PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: 64, gqa: 8) == 8)
        #expect(PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: 128, gqa: 8) == 4)
        #expect(PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: 256, gqa: 8) == 4)
        #expect(PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: 512, gqa: 4) == 4)
        // Gemma-4 global layers (d512, GQA 8): unsplit these were over
        // budget even at NSG=1 (the field fatal); split at HPT=2 they run
        // NSG=4 comfortably.
        #expect(PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: 512, gqa: 8) == 4)

        // Whenever an NSG is returned it fits the budget, and the next
        // larger candidate would not (largest-fit invariant).
        let candidates = PagedAttentionKernel.simdgroupCandidates
        for d in PagedAttentionKernel.supportedHeadDims.sorted() {
            for g in [1, 2, 4, 8, 16] {
                guard
                    let n = PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: d, gqa: g)
                else { continue }
                let bytes = PagedAttentionKernel.partThreadgroupBytes(
                    headDim: d, gqa: g, simdgroups: n)
                #expect(
                    bytes <= PagedAttentionKernel.threadgroupMemoryLimit,
                    "d=\(d) gqa=\(g): selected NSG=\(n) over budget")
                if let i = candidates.firstIndex(of: n), i > 0 {
                    let bigger = PagedAttentionKernel.partThreadgroupBytes(
                        headDim: d, gqa: g, simdgroups: candidates[i - 1])
                    #expect(
                        bigger > PagedAttentionKernel.threadgroupMemoryLimit,
                        "d=\(d) gqa=\(g): NSG=\(candidates[i - 1]) would also fit")
                }
            }
        }
    }

    // MARK: - (a) Ineligible shapes are refused at construction

    /// The former field-fatal shape — Gemma-4 global layers (d512, GQA 8)
    /// — is now ELIGIBLE via the head split. Regression anchor for the
    /// original bug: the UNSPLIT budget (GQA in place of HPT) would still
    /// be over the cap, so the formula must keep modelling the split.
    @Test func gemma4GlobalLayerShapeIsEligibleViaHeadSplit() throws {
        #expect(PagedAttentionKernel.ineligibilityReason(headDim: 512, gqa: 8) == nil)
        let unsplit = (8 * 512 + 1 * 8 * (512 + 2)) * 4
        #expect(unsplit > PagedAttentionKernel.threadgroupMemoryLimit)
        #expect(
            PagedAttentionKernel.partThreadgroupBytes(headDim: 512, gqa: 8, simdgroups: 4)
                <= PagedAttentionKernel.threadgroupMemoryLimit)

        // The real Gemma-4-26B global-layer shape (global_head_dim 512,
        // num_global_key_value_heads 2, 16 query heads) constructs.
        let backend = try PagedKVBackend(
            layerKinds: [fullKind(headDim: 512, kvHeads: 2, queryHeads: 16)],
            config: poolConfig())
        #expect(backend.layerKinds.count == 1)
    }

    /// Per-layer-kind: ONE ineligible layer makes the whole model
    /// ineligible, even when every other layer fits.
    @Test func anyIneligibleLayerMakesModelIneligible() {
        expectIneligible(
            [
                fullKind(headDim: 128, kvHeads: 8, queryHeads: 64),
                fullKind(headDim: 96, kvHeads: 1, queryHeads: 8),
            ],
            reasonContains: "layer 1")
    }

    /// Unsupported head dims still refuse through the same kernel-level
    /// eligibility path.
    @Test func unsupportedHeadDimIsIneligible() {
        #expect(PagedAttentionKernel.ineligibilityReason(headDim: 96, gqa: 2) != nil)
        expectIneligible(
            [fullKind(headDim: 96, kvHeads: 2, queryHeads: 4)],
            reasonContains: "headDim 96")
    }

    // MARK: - (b) Fleet shapes remain eligible

    @Test func fleetShapesRemainEligible() throws {
        // d128 / GQA 8 (dense fleet shape).
        #expect(PagedAttentionKernel.ineligibilityReason(headDim: 128, gqa: 8) == nil)
        let dense = try PagedKVBackend(
            layerKinds: [fullKind(headDim: 128, kvHeads: 8, queryHeads: 64)],
            config: poolConfig())
        #expect(dense.layerKinds.count == 1)

        // d64 / GQA 8 with attention sinks (GPT-OSS shape). Sinks are a
        // kernel parameter folded in at merge time — they add no
        // threadgroup memory and must not affect eligibility.
        #expect(PagedAttentionKernel.ineligibilityReason(headDim: 64, gqa: 8) == nil)
        let sinks = try PagedKVBackend(
            layerKinds: [fullKind(headDim: 64, kvHeads: 8, queryHeads: 64, sinks: true)],
            config: poolConfig())
        #expect(sinks.layerKinds.count == 1)

        // d512 stays supported at small GQA (headDim inclusion alone was
        // never the problem)...
        #expect(PagedAttentionKernel.ineligibilityReason(headDim: 512, gqa: 1) == nil)
        // ...and — via the head split — every supported head dim is now
        // eligible at every fleet GQA factor.
        for d in PagedAttentionKernel.supportedHeadDims {
            for g in [1, 2, 4, 8, 16] {
                #expect(
                    PagedAttentionKernel.ineligibilityReason(headDim: d, gqa: g) == nil,
                    "d=\(d) gqa=\(g)")
            }
        }
    }

    // MARK: - Page size ↔ kernel partition alignment (PR#62 review)

    /// The decode kernel preconditions `partitionTokens % pageSize == 0` at
    /// dispatch (an uncatchable trap). A misaligned page size must therefore
    /// fail POOL construction with a clear `backendIneligible`, not the
    /// first decode.
    @Test func misalignedPageSizeIsRejectedAtConstruction() {
        let kinds = [fullKind(headDim: 64, kvHeads: 4, queryHeads: 8)]
        for pageSize in [24, 48, 512] {
            do {
                _ = try PagedKVPool(
                    layerKinds: kinds,
                    config: PagedKVPoolConfig(
                        pageSize: pageSize, capacityBytes: 8 << 20, maxPrefillChunk: 64,
                        nominalMaxSequenceLength: 1024))
                Issue.record("expected backendIneligible for pageSize \(pageSize)")
            } catch CBv2KVError.backendIneligible(let reason) {
                #expect(reason.contains("pageSize \(pageSize)"), "reason: \(reason)")
                #expect(reason.contains("\(PagedAttentionKernel.partitionTokens)"))
            } catch {
                Issue.record("unexpected error for pageSize \(pageSize): \(error)")
            }
        }
    }

    /// Divisors of the kernel partition stay constructible.
    @Test func divisorPageSizesRemainEligible() throws {
        let kinds = [fullKind(headDim: 64, kvHeads: 4, queryHeads: 8)]
        for pageSize in [8, 16, 32, 64, PagedAttentionKernel.partitionTokens] {
            let pool = try PagedKVPool(
                layerKinds: kinds,
                config: PagedKVPoolConfig(
                    pageSize: pageSize, capacityBytes: 8 << 20, maxPrefillChunk: 64,
                    nominalMaxSequenceLength: 1024))
            #expect(pool.config.pageSize == pageSize)
        }
    }
}
