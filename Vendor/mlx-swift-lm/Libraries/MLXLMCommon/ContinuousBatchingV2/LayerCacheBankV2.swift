// LayerCacheBankV2.swift
//
// The engine's `CBv2LayerCacheProvider` over PERSISTENT per-layer
// attending caches (WS-A `CBv2LayerCache` or WS-C `PagedLayerCache`).
//
// The loop asks for layer caches every step (including every chained
// decode step). Rebinding rows costs a host-integer positionOffsets
// rebuild on the contiguous backend, so the bank fingerprints the batch
// composition by row-object identity and calls the contract's canonical
// `setRows(_:)` ONLY when composition actually changed — the chained
// pure-decode hot path performs zero host rebuilds (the caches advance
// their offsets on-device), preserving WS-A's membership-change-only
// discipline (`CBv2CoreInstrumentation.positionOffsetsHostRebuilds`).
//
// Identity safety: the previous batch's row objects stay retained by the
// caches' `rows` arrays until the next `setRows`, so an ObjectIdentifier
// can never alias a deallocated row.
//
// Engine-thread-confined (no locking) — the loop is the only caller.

import Foundation

/// A layer-cache provider whose composition fingerprint can be forced
/// stale. The engine loop invalidates after compiled decode steps advanced
/// rows OUTSIDE the provider's caches, so the next eager bind rebuilds
/// `positionOffsets` from host truth instead of trusting its own stale
/// on-device advance chain.
public protocol CBv2CompositionInvalidating: AnyObject {
    func invalidateBoundComposition()

    /// Drop the provider's strong row bindings entirely (each cache's
    /// `rows` array). The loop calls this when the eager caches will not be
    /// rebound for an unbounded time — the engine went idle, or compiled
    /// decode took over the step stream — so retired rows' KV buffers are
    /// not pinned by a stale binding until some future eager rebind
    /// (PR#62 review). Implies `invalidateBoundComposition()`; must be a
    /// cheap no-op when nothing is bound.
    func releaseBoundRows()
}

public final class CBv2LayerCacheBank: CBv2LayerCacheProvider, CBv2CompositionInvalidating {

    private let caches: [any CBv2AttendingLayerCache]
    private var boundRowIdentity: [ObjectIdentifier] = []
    private var hasBound = false

    /// Wrap pre-built caches — e.g. `model.newCacheV2 { ... }` output (the
    /// GPT-OSS path, which also primes sink activation at build time) or
    /// `PagedKVBackend.makeLayerCaches(attentionSoftcap:)`.
    public init(caches: [any CBv2AttendingLayerCache]) {
        self.caches = caches
    }

    /// Contiguous-backend convenience: one `CBv2LayerCache` per layer kind
    /// (KV-shared layers get a rowless borrowing cache), with the
    /// construction-time attention softcap threaded identically to the
    /// paged backend.
    public convenience init(layerKinds: [CBv2LayerKind], attentionSoftcap: Float? = nil) {
        self.init(
            caches: layerKinds.enumerated().map { index, kind in
                CBv2LayerCache(
                    layerIndex: index, kind: kind, attentionSoftcap: attentionSoftcap)
            })
    }

    /// Force the next `layerCaches` call to rebind rows even when the
    /// composition is identity-identical — required after compiled decode
    /// steps advanced the rows outside these caches (their cached
    /// `positionOffsets` no longer reflect the rows' true positions).
    public func invalidateBoundComposition() {
        hasBound = false
        boundRowIdentity = []
    }

    /// Unbind every storage-owning cache (`setRows([])`) and reset the
    /// fingerprint. Without this, the previous eager batch's row objects —
    /// including RETIRED rows whose backend accounting was already released
    /// — stay strongly retained by the caches' `rows` arrays for as long as
    /// the engine is idle or serving compiled-only steps (PR#62 review).
    /// No-op when nothing is bound, so idle-loop callers pay nothing.
    public func releaseBoundRows() {
        guard hasBound else { return }
        for cache in caches where cache.kind.sharesKVWithLayer == nil {
            cache.setRows([])
        }
        hasBound = false
        boundRowIdentity = []
    }

    /// The construction-time attention softcap shared by every contiguous
    /// (`CBv2LayerCache`) layer in this bank, or `.some(nil)` when they
    /// uniformly have none, or nil when the bank holds no contiguous caches
    /// / they disagree (then no claim is made). `EngineV2` cross-checks the
    /// compiled-decode config against this so the compiled path can never
    /// silently skip a softcap the eager path applies.
    public var uniformAttentionSoftcap: Float?? {
        var softcaps: [Float?] = []
        for cache in caches {
            guard let contiguous = cache as? CBv2LayerCache else { continue }
            softcaps.append(contiguous.attentionSoftcap)
        }
        guard let first = softcaps.first else { return nil }
        guard softcaps.allSatisfy({ $0 == first }) else { return nil }
        return .some(first)
    }

    /// Vision prefill eligibility: every cache in the bank must honor
    /// span-mask contexts (`CBv2SpanMaskBinding` — the contiguous
    /// `CBv2LayerCache` does; the paged `PagedLayerCache` does not), or a
    /// vision chunk's bidirectional-span mask could silently not apply on
    /// some layer. All-or-nothing, checked at submit.
    public var supportsMultimodalSpans: Bool {
        caches.allSatisfy { $0 is CBv2SpanMaskBinding }
    }

    public func layerCaches(rowStates: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache] {
        let identity = rowStates.map { row -> ObjectIdentifier in
            guard let anchor = row.compactMap({ $0 }).first else {
                preconditionFailure("CBv2LayerCacheBank: row owns no storage at any layer")
            }
            return ObjectIdentifier(anchor)
        }
        if !hasBound || identity != boundRowIdentity {
            for (layer, cache) in caches.enumerated() {
                guard cache.kind.sharesKVWithLayer == nil else { continue }
                cache.setRows(
                    rowStates.map { states in
                        guard let state = states[layer] else {
                            preconditionFailure(
                                "CBv2LayerCacheBank: missing sequence state for layer \(layer)")
                        }
                        return state
                    })
            }
            boundRowIdentity = identity
            hasBound = true
        }
        return caches
    }
}
