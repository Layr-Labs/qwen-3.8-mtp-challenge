// LayerKindDerivation.swift
//
// ContinuousBatchingV2 — WS-F (model adaptation).
//
// Pure derivation of `[CBv2LayerKind]` (model attention structure as data,
// invariant 11 in report 10) from the production models' configuration
// fields. The functions here are deliberately dependency-free: they take the
// raw config values so they can live in MLXLMCommon without importing the
// model modules. `Gemma4TextConfiguration.cbv2LayerKinds` and
// `GPTOSSConfiguration.cbv2LayerKinds` (in MLXLLM) are thin wrappers over
// these functions.
//
// The derivations mirror, field for field, what the model constructors do:
//  - Gemma 4: `Gemma4Attention.init` (head-dim asymmetry, K-eq-V kv-head
//    override) and `Gemma4TextModelInner.init` (the `previousKvs` shared-KV
//    source map).
//  - GPT-OSS: `GPTOSSModelInner.init` (alternating sliding/full fallback).
// Any change to those constructors must be reflected here — the CBv2 layer
// cache validates its storage against these kinds.

import Foundation

public enum CBv2LayerKindDerivation {

    /// Canonical `layer_types` strings (HF config vocabulary).
    public static let slidingAttentionType = "sliding_attention"
    public static let fullAttentionType = "full_attention"

    // MARK: - Shared helpers

    /// Derive `layer_types` from a repeating sliding-window pattern:
    /// `(slidingWindowPattern - 1)` sliding layers followed by one full
    /// layer, repeated and truncated to `numLayers`. Mirrors the fallback in
    /// `Gemma4TextConfiguration.init(from:)` (pattern 5 → 4×sliding + 1×full).
    public static func layerTypes(slidingWindowPattern: Int, numLayers: Int) -> [String] {
        precondition(slidingWindowPattern > 0, "slidingWindowPattern must be positive")
        var pattern = [String]()
        for i in 0 ..< slidingWindowPattern {
            pattern.append(
                i == slidingWindowPattern - 1 ? fullAttentionType : slidingAttentionType)
        }
        var types = [String]()
        while types.count < numLayers {
            types.append(contentsOf: pattern)
        }
        return Array(types.prefix(numLayers))
    }

    // MARK: - Gemma 4

    /// Shared-KV source map for Gemma 4's trailing `num_kv_shared_layers`
    /// block. Entry `j` is the index of the layer whose K/V (and captured
    /// RoPE offsets) layer `j` borrows, or nil when layer `j` owns its KV.
    ///
    /// Mirrors `Gemma4TextModelInner.previousKvs`: each shared layer
    /// references the LAST non-shared layer OF THE SAME TYPE. A shared layer
    /// whose type has no earlier non-shared layer maps to nil here (the model
    /// leaves `previousKvs[j] == j`, the identity, which behaves as
    /// "no source" — that shape only occurs for force-shared drafter trunks,
    /// which are out of scope for CBv2).
    public static func gemma4SharedKVSources(
        layerTypes: [String], numKvSharedLayers: Int
    ) -> [Int?] {
        let numLayers = layerTypes.count
        var sources = [Int?](repeating: nil, count: numLayers)
        guard numKvSharedLayers > 0 else { return sources }

        let firstShared = max(numLayers - numKvSharedLayers, 0)
        var lastByType = [String: Int]()
        for i in 0 ..< firstShared {
            lastByType[layerTypes[i]] = i
        }
        for j in firstShared ..< numLayers {
            sources[j] = lastByType[layerTypes[j]]
        }
        return sources
    }

    /// Derive per-layer attention structure for the Gemma 4 family.
    ///
    /// Field semantics (mirroring `Gemma4Attention.init`):
    ///  - sliding layers use `headDim`, full layers use `globalHeadDim`
    ///    (asymmetric head dims: 256 / 512 on the production configs);
    ///  - K-eq-V (`attention_k_eq_v`, 26B/31B) applies to FULL layers only
    ///    and switches their kv-head count to `numGlobalKeyValueHeads` when
    ///    that field is present;
    ///  - the trailing `numKvSharedLayers` layers own no KV storage and
    ///    borrow from the last non-shared layer of the same type;
    ///  - Gemma 4 has no attention sinks.
    public static func gemma4LayerKinds(
        layerTypes: [String],
        slidingWindow: Int,
        numKvSharedLayers: Int,
        headDim: Int,
        globalHeadDim: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        numGlobalKeyValueHeads: Int?,
        attentionKeqV: Bool
    ) -> [CBv2LayerKind] {
        let sources = gemma4SharedKVSources(
            layerTypes: layerTypes, numKvSharedLayers: numKvSharedLayers)

        return layerTypes.enumerated().map { index, layerType in
            let isSliding = layerType == slidingAttentionType
            let useKeqV = attentionKeqV && !isSliding
            let kvHeads: Int
            if useKeqV, let globalKvHeads = numGlobalKeyValueHeads {
                kvHeads = globalKvHeads
            } else {
                kvHeads = numKeyValueHeads
            }
            return CBv2LayerKind(
                attention: isSliding ? .slidingWindow(slidingWindow) : .full,
                sharesKVWithLayer: sources[index],
                hasSinks: false,
                headDim: isSliding ? headDim : globalHeadDim,
                kvHeads: kvHeads,
                queryHeads: numAttentionHeads
            )
        }
    }

    // MARK: - GPT-OSS

    /// Derive per-layer attention structure for GPT-OSS.
    ///
    /// `layerTypes` is the config's explicit `layer_types` when present;
    /// otherwise the alternating `[sliding, full]` fallback is derived
    /// exactly as `GPTOSSModelInner.init` does. Every layer carries learned
    /// per-head attention sinks (`hasSinks == true`) — a CBv2 backend that
    /// cannot fold sinks into the softmax denominator must refuse these
    /// kinds at engine build (contract requirement).
    public static func gptossLayerKinds(
        layerTypes explicitLayerTypes: [String]?,
        numHiddenLayers: Int,
        slidingWindow: Int,
        headDim: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int
    ) -> [CBv2LayerKind] {
        let types =
            explicitLayerTypes
            ?? Array(
                repeating: [slidingAttentionType, fullAttentionType],
                count: numHiddenLayers / 2
            ).flatMap { $0 }
        precondition(
            types.count == numHiddenLayers,
            "GPT-OSS layer_types count \(types.count) != num_hidden_layers \(numHiddenLayers)")

        return types.map { layerType in
            CBv2LayerKind(
                attention: layerType == slidingAttentionType
                    ? .slidingWindow(slidingWindow) : .full,
                sharesKVWithLayer: nil,
                hasSinks: true,
                headDim: headDim,
                kvHeads: numKeyValueHeads,
                queryHeads: numAttentionHeads
            )
        }
    }
}
