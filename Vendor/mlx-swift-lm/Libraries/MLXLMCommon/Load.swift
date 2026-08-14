// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

#if canImport(Darwin)
import Darwin

/// Tell the kernel to start prefetching every shard into the unified buffer
/// cache. Non-blocking; the actual reads happen in the SSD controller's own
/// queue. Safe to call even when files are already cached (it's just a hint).
private func prefetchShards(_ urls: [URL]) {
    DispatchQueue.concurrentPerform(iterations: urls.count) { idx in
        let path = urls[idx].path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { return }
        defer { close(fd) }

        var sb = stat()
        guard fstat(fd, &sb) == 0 else { return }

        var ra = radvisory(ra_offset: 0, ra_count: Int32(min(Int(sb.st_size), Int(Int32.max))))
        _ = fcntl(fd, F_RDADVISE, &ra)
    }
}
#else
private func prefetchShards(_ urls: [URL]) { }
#endif

/// Lock-protected scratch space used by the parallel shard reader. Wrapped in
/// a final class so Swift 6 strict concurrency can see we mean to share it
/// across the concurrent closures.
private final class ParallelShardState: @unchecked Sendable {
    typealias ShardResult = (weights: [String: MLXArray], metadata: [String: String])
    let lock = NSLock()
    var results: [ShardResult?]
    var firstError: Error?

    init(shardCount: Int) {
        self.results = Array(repeating: nil, count: shardCount)
    }

    func store(index: Int, result: ShardResult) {
        lock.lock()
        defer { lock.unlock() }
        results[index] = result
    }

    func recordError(_ error: Error) {
        lock.lock()
        defer { lock.unlock() }
        if firstError == nil { firstError = error }
    }
}

/// A separately-pinned weight tree to merge into the next ``loadWeights`` call.
///
/// `keyPrefix` is prepended to every tensor name the tree carries, which is how a
/// checkpoint published with BARE names (`fc.weight`, `layers.0.*`, `norm.weight`)
/// becomes the `mtp.*` namespace the loading model expects.
public struct AdditionalWeightSource: Sendable {
    public let directory: URL
    public let keyPrefix: String

    public init(directory: URL, keyPrefix: String) {
        self.directory = directory
        self.keyPrefix = keyPrefix
    }
}

/// Extra weight trees merged by the next ``loadWeights`` call, before `sanitize`.
///
/// Set immediately before a load and cleared immediately after; it exists because the
/// model factory's `_load` is a fixed protocol requirement that cannot carry a second
/// directory. Same idiom, same lifetime discipline as `_qwen35MTPEnabled`.
public nonisolated(unsafe) var _additionalWeightSources: [AdditionalWeightSource] = []

/// Prefix stripped from the PRIMARY directory's tensor names by the next
/// ``loadWeights`` call, before `sanitize`.
///
/// This exists for one specific, checked-in shape: a checkpoint whose tensors are named
/// for a multimodal WRAPPER (`language_model.model.*`, `language_model.lm_head.weight`)
/// while its `config.json` declares the bare TEXT model. Such a tree is internally
/// consistent only if the loader strips the wrapper prefix — which is exactly what this
/// repository's own eager loader (`RuntimeWeightNameTracker`) already does for the same
/// tree, by the same rule.
///
/// DO NOT SET THIS FOR A WRAPPER MODEL. `Qwen35Model.sanitize` deliberately ADDS
/// `language_model.` so the parameters address its `language_model` child; stripping it
/// first would leave every tensor addressed to nothing. The caller decides from the
/// config's declared `model_type`, which is the only place the answer is written down.
public nonisolated(unsafe) var _primaryWeightKeyPrefixStrip: String?

/// Drop `prefix` from every key that carries it.
///
/// Pure and model-free so the rename rule is testable without loading anything.
/// A collision after the rename means the tree carried BOTH spellings of a
/// tensor and one of them would silently win, so it throws instead — the same
/// stance this repository's eager loader (`RuntimeWeightNameTracker`) takes on
/// the identical rename.
public func strippingWeightKeyPrefix(
    _ prefix: String, from weights: [String: MLXArray]
) throws -> [String: MLXArray] {
    guard !prefix.isEmpty else { return weights }
    var renamed = [String: MLXArray]()
    renamed.reserveCapacity(weights.count)
    for (key, value) in weights {
        let name = key.hasPrefix(prefix)
            ? String(key.dropFirst(prefix.count)) : key
        guard renamed.updateValue(value, forKey: name) == nil else {
            throw ModelFactoryError.configurationFileError(
                "model.safetensors.index.json",
                "weight names collide after stripping '\(prefix)': \(name)",
                NSError(domain: "MLXLMCommon", code: 1)
            )
        }
    }
    return renamed
}

/// Load a directory's safetensors into one dictionary, without any model coupling.
public func loadArrays(directory: URL) throws -> [String: MLXArray] {
    var urls: [URL] = []
    if let enumerator = FileManager.default.enumerator(
        at: directory, includingPropertiesForKeys: nil)
    {
        for case let url as URL in enumerator where url.pathExtension == "safetensors" {
            urls.append(url)
        }
    }
    urls.sort { $0.lastPathComponent < $1.lastPathComponent }
    var out = [String: MLXArray]()
    for url in urls {
        for (key, value) in try loadArrays(url: url) {
            out[key] = value
        }
    }
    return out
}

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    let bench = ProcessInfo.processInfo.environment["BENCH_VERBOSE"] != nil
    var t = CFAbsoluteTimeGetCurrent()
    func mark(_ label: String) {
        if bench {
            let now = CFAbsoluteTimeGetCurrent()
            FileHandle.standardError.write(
                Data("    [stage] \(label): \(String(format: "%.1f", (now - t) * 1000)) ms\n"
                    .utf8))
            t = now
        }
    }

    // Gather the shard URLs first so we can fan them out concurrently.
    var shardURLs: [URL] = []
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator where url.pathExtension == "safetensors" {
        shardURLs.append(url)
    }
    shardURLs.sort { $0.lastPathComponent < $1.lastPathComponent }

    // Hand the kernel a head start on every shard. F_RDADVISE is Darwin's
    // async-prefetch primitive — it issues a non-blocking advisory read into
    // the unified buffer cache, letting the SSD start streaming pages before
    // we ask for them. Net cost is one open/fcntl/close per shard. By the
    // time the DispatchQueue.concurrentPerform tasks below try to read, the
    // pages may already be resident.
    prefetchShards(shardURLs)
    mark("rdadvise")

    // Load shards in parallel. Each task forces eval() on its arrays so MLX
    // actually performs the disk read inside the task rather than deferring all
    // of it to a single later eval(model) call. This lets concurrent shards
    // overlap their I/O.
    typealias ShardResult = (weights: [String: MLXArray], metadata: [String: String])
    let shared = ParallelShardState(shardCount: shardURLs.count)
    let urls = shardURLs  // immutable Sendable snapshot for the closure

    DispatchQueue.concurrentPerform(iterations: urls.count) { idx in
        do {
            let (w, m) = try loadArraysAndMetadata(url: urls[idx])
            if !w.isEmpty {
                eval(Array(w.values))
            }
            shared.store(index: idx, result: (w, m))
        } catch {
            shared.recordError(error)
        }
    }
    if let err = shared.firstError { throw err }

    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    for (i, slot) in shared.results.enumerated() {
        guard let (w, m) = slot else { continue }
        for (key, value) in w { weights[key] = value }
        // Match the original "first iterated shard's metadata" semantics by
        // taking shard 0's, falling back to next non-empty for safety.
        if i == 0 || metadata.isEmpty { metadata = m }
    }
    mark("read shards (parallel)")

    // Merge any separately-pinned weight trees BEFORE sanitize and BEFORE the
    // quantization wiring below. Both orderings are load-bearing: `sanitize` is
    // where a model decides what an extra key MEANS, and `quantize(model:)`
    // decides which submodules become quantized layers by asking whether
    // `weights["<path>.scales"]` exists -- so a tree merged after that walk
    // would arrive as quantized tensors addressed to unquantized layers and
    // fail `update(parameters:verify:)`.
    //
    // WHY A GLOBAL. The factory's `_load` is a protocol requirement with a
    // fixed signature, so an extra directory cannot be threaded through it
    // without changing every conformance. `_qwen35MTPEnabled` (Qwen35MTP.swift)
    // already establishes this exact idiom in this fork: a global set
    // immediately before the load, read during it. Callers must clear it after
    // the load; see `Qwen36MTPHeadAttachment` on the consumer side.
    // Strip the wrapper prefix from the primary tree FIRST, so an additional
    // source merged below lands in the same namespace the model addresses.
    if let strip = _primaryWeightKeyPrefixStrip, !strip.isEmpty {
        weights = try strippingWeightKeyPrefix(strip, from: weights)
        mark("strip \(strip)*")
    }

    for source in _additionalWeightSources {
        let extra = try loadArrays(directory: source.directory)
        for (key, value) in extra {
            weights[source.keyPrefix + key] = value
        }
        mark("merge \(source.keyPrefix)*")
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)
    mark("sanitize")

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }
    mark("quantize wire")

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])
    mark("update params")

    // Drop the staging dictionary before dtype conversion so we don't keep
    // two copies of safetensor arrays alive during the bf16 pass.
    weights.removeAll(keepingCapacity: false)
    MLX.Memory.clearCache()

    // Convert fp16 parameters to bf16 to eliminate AsType cascades.
    // Metal's kernel dispatcher promotes mixed fp16/fp32 operations to full fp32,
    // causing extra kernel dispatches across all layers. bf16 shares fp32's exponent
    // range, so the promotion is eliminated. Quantization scales/biases are also
    // converted — QuantizedMatmul uses scales dtype to determine output dtype.
    //
    // Controlled by DARKBLOOM_BF16_WEIGHTS (default: ON, set to "0" to disable).
    let bf16Env = ProcessInfo.processInfo.environment["DARKBLOOM_BF16_WEIGHTS"] ?? "1"
    if bf16Env == "1" {
        convertToBFloat16(model: model)
        mark("bf16 convert")
    }

    eval(model)
    mark("eval")
}

// MARK: - BFloat16 Weight Conversion

/// Convert float16 model parameters to bfloat16 to prevent AsType cascades.
///
/// Metal's kernel dispatcher promotes mixed float16/float32 operations to full float32,
/// causing speed regression for models where routing/normalization runs at float32.
/// bfloat16 avoids this because it shares float32's exponent range.
///
/// Quantization scales/biases are also converted — QuantizedMatmul uses scales dtype to
/// determine output dtype, so float16 scales → float16 output → AsType when multiplied
/// with bfloat16 norms. Converting scales to bfloat16 eliminates this cascade.
///
/// The conversion is chunked to bound transient memory. Large quantized models carry
/// tens of GB of scale/bias metadata; converting all arrays at once keeps both fp16 and
/// bf16 copies alive until the final eval.
private func convertToBFloat16(model: Module) {
    let t0 = CFAbsoluteTimeGetCurrent()

    let convertibleParams: [(key: String, convertedBytes: Int)] = {
        let flat = model.parameters().flattened()
        return flat.compactMap { key, array in
            guard array.dtype == .float16 else { return nil }
            return (key: key, convertedBytes: estimatedByteCount(array, as: .bfloat16))
        }
    }()

    guard !convertibleParams.isEmpty else { return }

    let totalBytes = convertibleParams.reduce(0) { $0 + $1.convertedBytes }
    let chunkLimit = bfloat16ConversionChunkLimit()
    var index = 0
    var convertedCount = 0

    while index < convertibleParams.count {
        var converted = [String: MLXArray]()
        var chunkBytes = 0

        do {
            let current = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            while index < convertibleParams.count {
                let entry = convertibleParams[index]
                if !converted.isEmpty,
                    chunkBytes + entry.convertedBytes > chunkLimit
                {
                    break
                }

                guard let array = current[entry.key],
                    array.dtype == DType.float16
                else {
                    index += 1
                    continue
                }

                converted[entry.key] = array.asType(DType.bfloat16)
                chunkBytes += entry.convertedBytes
                index += 1
            }
        }

        guard !converted.isEmpty else { continue }

        let values = Array(converted.values)
        MLX.eval(values)
        convertedCount += converted.count

        let params = ModuleParameters.unflattened(converted)
        do {
            try model.update(parameters: params, verify: [])
        } catch {
            FileHandle.standardError.write(
                Data("[bf16] model.update failed: \(error)\n".utf8))
        }
        MLX.Memory.clearCache()
    }

    let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
    let mb = Double(totalBytes) / (1024 * 1024)
    FileHandle.standardError.write(
        Data(
            "[bf16] converted \(convertedCount) params (\(String(format: "%.1f", mb)) MB) fp16→bf16 in \(String(format: "%.0f", elapsed)) ms\n"
                .utf8))
}

/// Maximum bytes of bf16 output to accumulate per conversion chunk.
/// Override with DARKBLOOM_BF16_CHUNK_MB environment variable.
private func bfloat16ConversionChunkLimit() -> Int {
    if let raw = ProcessInfo.processInfo.environment["DARKBLOOM_BF16_CHUNK_MB"],
        let mb = Int(raw), mb > 0
    {
        return mb * 1024 * 1024
    }
    return 256 * 1024 * 1024
}

private func estimatedByteCount(_ array: MLXArray, as dtype: DType) -> Int {
    let elements = array.shape.reduce(1) { partial, dim in
        partial * max(dim, 1)
    }
    return elements * dtypeByteWidth(dtype)
}

private func dtypeByteWidth(_ dtype: DType) -> Int {
    if dtype == .bool || dtype == .int8 || dtype == .uint8 {
        return 1
    }
    if dtype == .float16 || dtype == .bfloat16
        || dtype == .int16 || dtype == .uint16
    {
        return 2
    }
    if dtype == .int64 || dtype == .uint64 {
        return 8
    }
    return 4  // float32, int32, uint32, complex64
}
