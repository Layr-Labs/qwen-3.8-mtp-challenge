# Contract issues — WS-D (prefix cache)

Places where the frozen `CBv2Contracts.swift` was insufficient for the WS-D
spec, and the closest conforming shape shipped. None require changes to other
workstreams' code; all are additive protocol gaps to resolve at integration.

## 1. No adoption-release hook on `CBv2PrefixCache`

The spec requires an in-use refcount ("never evict entries currently being
adopted"), so a successful `lookup` pins the matched entry. But the frozen
protocol (`lookup` / `donate` / `evict` / `bytesInUse`) has no way for the
caller to signal that adoption finished, so a protocol-only caller can never
unpin an entry — pinned entries would be immune to eviction forever.

**Shipped shape:** `PrefixCacheV2.endAdoption(tokens:matched:)` on the
concrete type (resolved by chain hash; pinned entries always retain their
index keys, so resolution is exact). Memory-safety is unaffected either way —
returned `MLXArray` views keep their buffers alive via ARC — the pin only
governs eviction ordering and `bytesInUse` truthfulness.

**Integration ask:** add a release hook to the protocol, e.g.
`func endAdoption(tokens: [Int], matched: Int)`, or return an opaque adoption
ticket from `lookup`.

## 2. Tenant `cacheSalt` cannot be per-request

TB-007 scoping is per-request in vLLM (`cache_salt` request field), but the
frozen `lookup(tokens:layerKinds:)` / `donate(tokens:state:layerKinds:)`
signatures carry no salt and `CBv2Request` has no salt field.

**Shipped shape:** the salt is fixed per cache instance
(`CBv2PrefixCacheConfig.cacheSalt`), folded into the first block hash
(length-prefixed, domain-tagged; empty salt ⇒ byte-identical to the legacy
`computeBlockHash` chain). `CBv2BlockHasher` already namespaces per
(modelName, cacheSalt), so per-tenant scoping needs only the plumbing.

**Integration ask:** add `cacheSalt: String` to `CBv2Request` and thread it
through `lookup`/`donate` (or have the provider bridge run one cache per
tenant scope).

## 3. `requiredRecompute` is not reachable through the protocol

The spec says to expose `requiredRecompute(layerKinds:matched:)` "so B's loop
knows to re-prefill that suffix", but the frozen protocol has no such
requirement, so WS-B holding a `CBv2PrefixCache` existential cannot call it.

**Shipped shape:** a pure static (plus instance convenience) on the concrete
type — `PrefixCacheV2.requiredRecompute(layerKinds:matched:)` — callable by
WS-B without holding a cache instance at all.

**Integration ask:** add the method to the protocol (or move it to a shared
free function in the contract, since it is pure model-shape logic).
