import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Offline equivalence gate for the fused draft-rerank tail
/// (`qwen_mtp_draft_rerank_fused`) against the incumbent five-dispatch tail.
/// Follows the `qwen35VerifyDraftTop32` pattern shipped in Qwen35.swift: a
/// synthetic declared draft head of the live geometry is side-channeled
/// through `sanitize`, and the fused readout is compared against the
/// incumbent tail computed inline on identical tensors. No checkpoint load.
/// Never called on a scored path.
public enum Qwen35DraftRerankFusedVerifier {

    public struct Result: Equatable, Sendable {
        public let trials: Int
        public let mismatches: Int
        public let firstBadTrial: Int
    }

    /// Drives `Qwen35TextModel.draftTokenID` with a synthetic declared
    /// affine-2 draft head and an affine-4 quantized lm_head, comparing the
    /// fused tail against the incumbent five-dispatch tail on `trials`
    /// random inputs. The geometry guard on live shapes (hidden 5120,
    /// 32 candidates) makes the fused path fire on every trial.
    public static func verify(trials: Int = 16, seed: UInt64 = 1) throws -> Result {
        _qwen35MTPEnabled = true
        defer { _qwen35MTPEnabled = false }

        let configJSON = """
        {"hidden_size": 5120, "num_hidden_layers": 1,
         "vocab_size": 248320, "tie_word_embeddings": false,
         "mtp_num_hidden_layers": 1}
        """
        let config = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(configJSON.utf8))
        let model = Qwen35TextModel(config)

        let hidden = 5120
        let groups = 80
        let rows = 98_336
        let candidates = 32
        let prefixCount = 98_304
        let controlStart = 248_044
        let controlEnd = 248_070
        let realCount = prefixCount + controlEnd - controlStart
        let paddingCount = rows - realCount

        MLXRandom.seed(seed)

        // lm_head: full-vocab affine-4 quantized linear from random weights.
        let lmWeight = MLXRandom.normal([248_320, hidden])
            .asType(.float16) * 0.02
        let lmHead = QuantizedLinear(
            weight: lmWeight, bias: nil, groupSize: 64, bits: 4, mode: .affine)
        model.update(modules: ModuleChildren(values: ["lm_head": .value(lmHead)]))

        // Declared coarse draft head: affine-2, live compact-vocab shape.
        let coarseWeight = MLXRandom.normal([rows, hidden])
            .asType(.float16) * 0.02
        let coarseQuant = QuantizedLinear(
            weight: coarseWeight, bias: nil, groupSize: 64, bits: 2, mode: .affine)
        let coarseW = coarseQuant.weight
        let coarseS = coarseQuant.scales
        let coarseZ = coarseQuant.biases!

        _ = model.sanitize(weights: [
            "mtp.draft_lm_head.weight": coarseW,
            "mtp.draft_lm_head.scales": coarseS,
            "mtp.draft_lm_head.biases": coarseZ,
        ])
        eval(model)

        // Compact exact-head rows (the same gather makeCompactDraftHead runs).
        func compactRows(_ array: MLXArray) -> MLXArray {
            let prefix = array[0 ..< prefixCount]
            let controls = array[controlStart ..< controlEnd]
            let padding = array[0 ..< paddingCount]
            return concatenated([prefix, controls, padding], axis: 0)
        }
        let exactW = compactRows(lmHead.weight)
        let exactS = compactRows(lmHead.scales)
        let exactZ = compactRows(lmHead.biases!)

        var bad = 0
        var firstBad = -1
        let kth = realCount - candidates
        for trial in 0 ..< trials {
            MLXRandom.seed(seed &+ UInt64(trial) &* 2_654_435_761)
            let x = MLXRandom.normal([1, 1, hidden]).asType(.bfloat16)

            // Fused path: geometry guard fires on live shapes.
            let fused = model.draftTokenID(x)
            eval(fused)

            // Incumbent path, inline with the same tensors.
            let coarse = MLX.quantizedMatmul(
                x, coarseW, scales: coarseS, biases: coarseZ,
                transpose: true, groupSize: 64, bits: 2, mode: .affine)
            let candidateIDs = MLX.argPartition(
                coarse[0..., 0..., 0 ..< realCount], kth: kth, axis: -1
            )[.ellipsis, (kth)...].reshaped([candidates]).asType(.uint32)

            let gW = MLX.take(exactW, candidateIDs, axis: 0)
            let gS = MLX.take(exactS, candidateIDs, axis: 0)
            let gZ = MLX.take(exactZ, candidateIDs, axis: 0)
            let exactLogits = MLX.quantizedMatmul(
                x, gW, scales: gS, biases: gZ,
                transpose: true, groupSize: 64, bits: 4, mode: .affine)
            let winner = MLX.argMax(
                exactLogits.reshaped([candidates]), axis: -1
            ).asType(.uint32)
            let winnerID = MLX.take(candidateIDs, winner, axis: 0)
            let mapped = which(
                winnerID .< prefixCount, winnerID,
                winnerID + (controlStart - prefixCount))
            let incumbent = mapped.reshaped([1, 1]).asType(.int32)
            eval(incumbent)

            let fusedItem = fused.item(Int.self)
            let incItem = incumbent.item(Int.self)
            if fusedItem != incItem {
                bad += 1
                if firstBad < 0 {
                    firstBad = trial
                    print(
                        "[verify] trial \(trial): fused \(fusedItem) "
                            + "!= incumbent \(incItem)")
                }
            }
        }
        return Result(trials: trials, mismatches: bad, firstBadTrial: firstBad)
    }
}
