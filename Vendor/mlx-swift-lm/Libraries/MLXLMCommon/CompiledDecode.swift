// CompiledDecode: whole-step compiled decode helper.
//
// Ported (trimmed) from osaurus-ai/vmlx-swift-lm
// (Libraries/MLXLMCommon/BatchEngine/BatchCompile.swift `compileForward`).
//
// This wraps MLX `compile(inputs:outputs:)` so a single decode step — the model
// forward plus the per-layer KV-cache write — is captured as one compiled graph,
// collapsing hundreds of FFI crossings into a single compiled call. The tracer
// captures each cache layer's `innerState()`; subsequent invocations mutate the
// captured cache objects in place via `_updateInternal`.
//
// REQUIREMENTS / CONSTRAINTS:
// - Every layer must be a ``CompilableKVCache`` or ``CompilableRotatingKVCache``
//   (fixed-shape, MLXArray offset). Standard `KVCacheSimple` / `RotatingKVCache`
//   change state shape per step and cannot be compile-traced.
// - Mixed caches (e.g. Gemma4: KVCacheSimple for full-attention layers,
//   RotatingKVCache for sliding-window layers) are supported via per-layer
//   promotion in `setupCompiledDecode`.
// - The trace specialises on the token-batch shape it first sees (typically
//   `[B, 1]`). A changing batch size forces a recompile, so the batched decode
//   path needs fixed-size buckets (see the port plan for `GenerationBatch`).
//
// This helper is dependency-free w.r.t. the continuous-batching engine: it can
// be exercised in isolation (see CompilableKVCacheTests) and reused by either a
// single-stream or a batched decode loop once the cache-promotion + bucketing
// plumbing lands.

import Foundation
import MLX
import os

private let compiledDecodeLog = Logger(subsystem: "darkbloom", category: "CompiledDecode")

public enum CompiledDecode {

    /// Compiled decode is ON by default. Set `DARKBLOOM_COMPILED_DECODE=0`
    /// to disable. Guards in GenerationBatch ensure it only activates for
    /// B=1 solo decode with supported cache types (no MTP, no SSM).
    public static let isEnabled: Bool = {
        if let raw = ProcessInfo.processInfo.environment["DARKBLOOM_COMPILED_DECODE"] {
            return !["0", "false", "no", "off"].contains(raw.lowercased())
        }
        return true
    }()

    /// True iff every layer is a compilable cache type and thus
    /// compile-traceable by ``compileForward(model:cacheRef:)``.
    /// Accepts both single-stream (CompilableKVCache, CompilableRotatingKVCache)
    /// and batched (CompilableBatchKVCache, CompilableBatchRotatingKVCache) types.
    public static func eligible(_ cache: [KVCache]) -> Bool {
        !cache.isEmpty && cache.allSatisfy {
            $0 is CompilableKVCache || $0 is CompilableRotatingKVCache
                || $0 is CompilableBatchKVCache || $0 is CompilableBatchRotatingKVCache
        }
    }

    /// Build a compiled forward closure for a decode step.
    ///
    /// The returned closure accepts `[tokens]` (a single `[B, L]` int token
    /// array wrapped in a one-element array) and returns `[logits]` (a single
    /// `[B, L, V]` array). The captured cache layers are mutated in place.
    ///
    /// Supports mixed cache types: single-stream (``CompilableKVCache``,
    /// ``CompilableRotatingKVCache``) and batched (``CompilableBatchKVCache``,
    /// ``CompilableBatchRotatingKVCache``) all expose `innerState()` returning
    /// MLXArrays tracked by `compile(inputs:outputs:)`.
    ///
    /// - Precondition: `cacheRef` is non-empty and every element is a
    ///   compilable cache (see ``eligible(_:)``). Call `eval(cacheRef)`
    ///   before this so no pending tracer ops corrupt state identity.
    ///
    /// - Parameters:
    ///   - model: The language model to trace through.
    ///   - cacheRef: Per-layer compilable cache instances. Captured by the
    ///     returned closure; must not be empty.
    /// - Returns: A `@Sendable` closure mapping `[tokens]` -> `[logits]`.
    public static func compileForward(
        model: any LanguageModel,
        cacheRef: [KVCache]
    ) -> @Sendable ([MLXArray]) -> [MLXArray] {
        precondition(
            eligible(cacheRef),
            "CompiledDecode.compileForward requires a non-empty cache where every "
                + "layer is a compilable type (single-stream or batched).")

        let capturedModel = model
        let captured = cacheRef

        return compile(
            inputs: captured, outputs: captured
        ) { (args: [MLXArray]) -> [MLXArray] in
            let result = capturedModel(
                LMInput.Text(tokens: args[0]),
                cache: captured.isEmpty ? nil : captured,
                state: nil
            )
            return [result.logits]
        }
    }

    /// Attempt to set up compiled decode for a model + cache pair.
    ///
    /// This converts eligible cache layers to their compilable equivalents
    /// and builds a compiled forward closure. Per-layer promotion handles
    /// heterogeneous caches (e.g. Gemma4 with mixed KVCacheSimple +
    /// RotatingKVCache layers).
    ///
    /// The conversion is only performed when ALL of these conditions hold:
    /// - `DARKBLOOM_COMPILED_DECODE=1` env var is set
    /// - `MLXHardwareInfo.isCompiledDecodeSupported` is true
    /// - Every cache layer is either `KVCacheSimple` or `RotatingKVCache`
    ///
    /// - Parameters:
    ///   - model: The language model.
    ///   - cache: Mutable cache array. On success, entries are replaced with
    ///     their compilable equivalents in place.
    ///   - maxCacheLength: Maximum sequence length for the compiled cache buffers
    ///     (applies to CompilableKVCache; RotatingKVCache uses its own maxCacheSize).
    /// - Returns: A compiled forward closure, or `nil` if setup was skipped.
    public static func setupCompiledDecode(
        model: any LanguageModel,
        cache: inout [KVCache],
        maxCacheLength: Int = 4096
    ) -> (@Sendable ([MLXArray]) -> [MLXArray])? {
        guard isEnabled else { return nil }
        guard MLXHardwareInfo.isCompiledDecodeSupported else {
            compiledDecodeLog.info("Compiled decode skipped: hardware not supported")
            return nil
        }

        // Validate all layers are promotable before doing any conversion.
        for layer in cache {
            if !(layer is KVCacheSimple) && !(layer is RotatingKVCache) {
                compiledDecodeLog.info(
                    "Compiled decode skipped: unsupported cache type (\(type(of: layer)))")
                return nil
            }
        }

        // Materialize all pending cache operations before conversion.
        eval(cache)

        // Per-layer promotion: each layer type gets its compilable equivalent.
        var simpleCount = 0
        var rotatingCount = 0
        for i in 0..<cache.count {
            if let rotating = cache[i] as? RotatingKVCache {
                // RotatingKVCache → CompilableRotatingKVCache
                // (uses its own maxCacheSize, not maxCacheLength)
                cache[i] = CompilableRotatingKVCache.promote(from: rotating, maxLength: maxCacheLength)
                rotatingCount += 1
            } else if let simple = cache[i] as? KVCacheSimple {
                // KVCacheSimple → CompilableKVCache
                cache[i] = CompilableKVCache.promote(from: simple, maxLength: maxCacheLength)
                simpleCount += 1
            }
        }

        // Materialize the new compilable cache buffers
        eval(cache)

        let layerCount = cache.count
        compiledDecodeLog.info(
            "Compiled decode enabled: \(layerCount) layers (\(simpleCount) simple + \(rotatingCount) rotating), maxLength=\(maxCacheLength)")

        return compileForward(model: model, cacheRef: cache)
    }

    /// Set up compiled decode for batched caches (B >= 1).
    ///
    /// Promotes ``BatchKVCache`` layers to ``CompilableBatchKVCache`` and
    /// ``BatchRotatingKVCache`` layers to ``CompilableBatchRotatingKVCache``
    /// in place. Layers that are already compilable are kept as-is.
    ///
    /// ArraysCache / MambaCache layers are unsupported — if any are present,
    /// setup is skipped and `nil` is returned.
    ///
    /// - Parameters:
    ///   - model: The language model.
    ///   - cache: Mutable batched-cache array. On success, entries are replaced
    ///     with their compilable equivalents in place.
    ///   - maxCacheLength: Maximum sequence length for full-attention layers.
    ///     Sliding-window layers use their own maxCacheSize.
    /// - Returns: A compiled forward closure, or `nil` if setup was skipped.
    public static func setupBatchCompiledDecode(
        model: any LanguageModel,
        cache: inout [any BatchedCache],
        maxCacheLength: Int = 4096
    ) -> (@Sendable ([MLXArray]) -> [MLXArray])? {
        guard isEnabled else { return nil }
        guard MLXHardwareInfo.isCompiledDecodeSupported else {
            compiledDecodeLog.info("Batch compiled decode skipped: hardware not supported")
            return nil
        }

        // Validate all layers are promotable.
        for layer in cache {
            if layer is CompilableBatchKVCache || layer is CompilableBatchRotatingKVCache {
                continue  // Already compilable
            }
            if !(layer is BatchKVCache) && !(layer is BatchRotatingKVCache) {
                compiledDecodeLog.info(
                    "Batch compiled decode skipped: unsupported cache type (\(type(of: layer)))")
                return nil
            }
        }

        // Materialize all pending cache operations before conversion.
        eval(cache)

        // Per-layer promotion.
        var fullCount = 0
        var rotatingCount = 0
        for i in 0..<cache.count {
            if cache[i] is CompilableBatchKVCache || cache[i] is CompilableBatchRotatingKVCache {
                // Already compilable — count it.
                if cache[i] is CompilableBatchKVCache { fullCount += 1 }
                else { rotatingCount += 1 }
                continue
            }

            if let rotating = cache[i] as? BatchRotatingKVCache {
                cache[i] = CompilableBatchRotatingKVCache.promote(
                    from: rotating, maxLength: maxCacheLength)
                rotatingCount += 1
            } else if let full = cache[i] as? BatchKVCache {
                cache[i] = CompilableBatchKVCache.promote(
                    from: full, maxLength: maxCacheLength)
                fullCount += 1
            }
        }

        // Materialize the new compilable cache buffers.
        eval(cache)

        let layerCount = cache.count
        compiledDecodeLog.info(
            "Batch compiled decode enabled: \(layerCount) layers (\(fullCount) full + \(rotatingCount) rotating), maxLength=\(maxCacheLength)")

        // Build compiled forward with the caches cast to [KVCache].
        let cacheRef = cache.map { $0 as any KVCache }
        return compileForward(model: model, cacheRef: cacheRef)
    }
}
