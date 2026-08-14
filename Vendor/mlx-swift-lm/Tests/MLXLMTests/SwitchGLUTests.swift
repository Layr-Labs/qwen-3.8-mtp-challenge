import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

// Serialized: SwitchGLU's init probes custom activations through compiled
// functions (`silu(probe)`), and its forward runs compiled GLU kernels.
// Running these tests in parallel can deadlock MLX-Swift's per-
// CompiledFunction locks (one test tracing inside a compiled call while
// another blocks on the same function's lock from init).
@Suite(.serialized)
struct SwitchGLUTests {
    @Test func fusedGateUpMatchesSeparateProjections() {
        let inputDims = 8
        let hiddenDims = 4
        let numExperts = 3

        let separate = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            activation: { $0 }, bias: false)
        let fused = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            activation: { $0 }, bias: false, fuseGateUp: true)

        let gate = values(numExperts * hiddenDims * inputDims)
            .reshaped(numExperts, hiddenDims, inputDims)
            .asType(.float32)
        let up = values(numExperts * hiddenDims * inputDims, offset: 1000)
            .reshaped(numExperts, hiddenDims, inputDims)
            .asType(.float32)
        let down = values(numExperts * inputDims * hiddenDims, offset: 2000)
            .reshaped(numExperts, inputDims, hiddenDims)
            .asType(.float32)

        separate.update(parameters: ModuleParameters.unflattened([
            "gate_proj.weight": gate,
            "up_proj.weight": up,
            "down_proj.weight": down,
        ]))
        fused.update(parameters: ModuleParameters.unflattened([
            "gate_up_proj.weight": concatenated([gate, up], axis: -2),
            "down_proj.weight": down,
        ]))

        let x = values(2 * inputDims, offset: 3000)
            .reshaped(2, inputDims)
            .asType(.float32)
        let indices = MLXArray([Int32(0), 2, 1, 0]).reshaped(2, 2)

        let separateOut = separate(x, indices)
        let fusedOut = fused(x, indices)
        eval(separateOut, fusedOut)

        let maxDiff = max(abs(separateOut - fusedOut)).item(Float.self)
        #expect(maxDiff == 0)
    }

    /// N1 fast-follow: same parity check for the *quantized* fused path. Affine
    /// quantization is independent per output row, so quantizing the fused
    /// `[gate; up]` weight yields gate/up rows bit-identical to quantizing the
    /// two projections separately -- the single gathered quantized matmul + split
    /// must therefore match two separate quantized matmuls exactly.
    @Test func fusedGateUpQuantizedMatchesSeparateProjections() {
        let inputDims = 64
        let hiddenDims = 64
        let numExperts = 3
        let groupSize = 64
        let bits = 4

        let separate = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            activation: { $0 }, bias: false)
        let fused = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            activation: { $0 }, bias: false, fuseGateUp: true)

        let gate = values(numExperts * hiddenDims * inputDims)
            .reshaped(numExperts, hiddenDims, inputDims)
            .asType(.float32)
        let up = values(numExperts * hiddenDims * inputDims, offset: 20000)
            .reshaped(numExperts, hiddenDims, inputDims)
            .asType(.float32)
        let down = values(numExperts * inputDims * hiddenDims, offset: 40000)
            .reshaped(numExperts, inputDims, hiddenDims)
            .asType(.float32)

        separate.update(parameters: ModuleParameters.unflattened([
            "gate_proj.weight": gate,
            "up_proj.weight": up,
            "down_proj.weight": down,
        ]))
        fused.update(parameters: ModuleParameters.unflattened([
            "gate_up_proj.weight": concatenated([gate, up], axis: -2),
            "down_proj.weight": down,
        ]))

        quantize(model: separate, groupSize: groupSize, bits: bits)
        quantize(model: fused, groupSize: groupSize, bits: bits)

        let x = values(2 * inputDims, offset: 60000)
            .reshaped(2, inputDims)
            .asType(.float32)
        let indices = MLXArray([Int32(0), 2, 1, 0]).reshaped(2, 2)

        let separateOut = separate(x, indices)
        let fusedOut = fused(x, indices)
        eval(separateOut, fusedOut)

        let maxDiff = max(abs(separateOut - fusedOut)).item(Float.self)
        #expect(maxDiff == 0)
    }

    /// Regression for the removed runtime fused gate+up cache: SwitchGLU must
    /// retain NO weight copies beyond its parameters. The removed cache
    /// concatenated a full second copy of the quantized gate+up expert weights
    /// onto the module (stored OUTSIDE `parameters()`, so quantize/update never
    /// saw it) on the first forward — ~8 GiB (qat-4bit) / ~15 GiB (8bit)
    /// model-wide on Gemma 4 26B, and the root cause of the v0.7.3 production
    /// incident. This pins both views:
    ///
    ///   * parameter identity — every parameter array is the SAME instance
    ///     after N decode-shaped forwards (nothing re-pointed, nothing added);
    ///   * module-retained bytes — a reflection walk over every MLXArray
    ///     reachable from the module (deduped by instance) measures EXACTLY
    ///     what the module holds. The baseline is taken BEFORE any forward —
    ///     the old cache built in `callAsFunction` ahead of its size gate, so
    ///     any warm-up forward (whatever its shape) would hide a reintroduced
    ///     build; and unlike the process-global `GPU.activeMemory` counter,
    ///     the walk is immune to allocations from concurrently-running suites,
    ///     so the assertion can be exact instead of noise-margined.
    @Test func decodeShapedForwardsRetainNoExtraWeightCopies() {
        let inputDims = 2048
        let hiddenDims = 2048
        let numExperts = 8

        let glu = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts)
        quantize(model: glu, groupSize: 64, bits: 4)
        MLX.eval(glu.parameters().flattened().map(\.1))

        let paramsBefore = Dictionary(
            uniqueKeysWithValues: glu.parameters().flattened())
        let retainedBefore = Self.moduleRetainedBytes(glu)

        // Decode shape: B=1, top_k=8 → indices.size == 8, the exact per-call
        // shape the removed fused dispatch keyed on.
        let x = values(inputDims, offset: 100).reshaped(1, inputDims).asType(.float32)
        let indices = MLXArray((0 ..< 8).map(Int32.init)).reshaped(1, 8)
        for _ in 0 ..< 8 {
            eval(glu(x, indices))
        }

        let retainedAfter = Self.moduleRetainedBytes(glu)
        #expect(
            retainedAfter == retainedBefore,
            Comment(
                rawValue: "8 decode-shaped forwards grew module-retained arrays from "
                    + "\(retainedBefore) to \(retainedAfter) bytes — SwitchGLU is "
                    + "caching weights again"))

        let paramsAfter = Dictionary(
            uniqueKeysWithValues: glu.parameters().flattened())
        #expect(paramsAfter.count == paramsBefore.count)
        for (key, before) in paramsBefore {
            #expect(
                paramsAfter[key] === before,
                Comment(rawValue: "parameter \(key) was replaced during forwards"))
        }
    }

    /// Total bytes of every MLXArray reachable from `root` via reflection,
    /// deduped by array instance. Sees arrays stored in plain (non-parameter)
    /// properties — exactly where the removed cache lived — and only this
    /// module's arrays, so concurrent suites cannot perturb the measurement.
    private static func moduleRetainedBytes(_ root: Any) -> Int {
        var seenArrays = Set<ObjectIdentifier>()
        var seenObjects = Set<ObjectIdentifier>()
        var total = 0
        func walk(_ value: Any) {
            if let array = value as? MLXArray {
                if seenArrays.insert(ObjectIdentifier(array)).inserted {
                    total += array.nbytes
                }
                return
            }
            let mirror = Mirror(reflecting: value)
            if mirror.displayStyle == .class {
                let id = ObjectIdentifier(value as AnyObject)
                if !seenObjects.insert(id).inserted { return }
            }
            for child in mirror.children {
                walk(child.value)
            }
        }
        walk(root)
        return total
    }

    private func values(_ count: Int, offset: Int = 0) -> MLXArray {
        MLXArray((offset ..< offset + count).map { Float($0) / 1000 })
    }
}
