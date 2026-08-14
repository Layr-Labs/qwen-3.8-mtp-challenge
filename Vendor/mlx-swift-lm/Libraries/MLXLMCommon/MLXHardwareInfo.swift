import Foundation

/// Runtime gate for `compile(shapeless: true)` activation micro-fusions
/// (SwiGLU / GeGLU / GELU / softcap).
///
/// Ported from osaurus/main (`Libraries/MLXLMCommon/HardwareInfo.swift`) and
/// renamed to `MLXHardwareInfo` to avoid colliding with ProviderCore's existing
/// `HardwareInfo` type in the provider build.
///
/// ## Background
///
/// `compile(shapeless: true)` wraps a closure in a `CompiledFunction` whose
/// `Compiled::eval_gpu` path can return zero results on some macOS Tahoe Metal
/// JIT drivers (notably M1/M2 A14/A15 GPUs, but also observed on M4). Swift then
/// crashes reading `call(...)[0]` on the empty result. See MLX issues #3329,
/// #3201, #3256.
///
/// The crash was empirically tied to the *whole-model* compile path
/// (`setupCompiledDecode` + `CompilableKVCache`), not to these small fixed-shape
/// activation fusions, so the default is ON. The per-op overhead of the fusions
/// is negligible vs. the model forward pass, so a machine that does hit the JIT
/// bug can opt out via the `MLX_COMPILED_DECODE` env var with no code change.
///
/// This intentionally mirrors the `MLX_COMPILED_DECODE` knob used by the Gemma 4
/// decode helpers (`MLXVLM/Models/Gemma4.swift`) so those file-private fusions can
/// later be collapsed onto this single shared symbol with no behavior change.
public enum MLXHardwareInfo {
    /// `true` when `compile(shapeless: true)` activation fusions are enabled.
    ///
    /// Defaults to `true`. Set `MLX_COMPILED_DECODE` to `0`/`false`/`no`/`off`
    /// to disable compiled fusions on a machine that hits the Tahoe Metal JIT
    /// bug; any other (or absent) value keeps them on.
    public static let isCompiledDecodeSupported: Bool = {
        if let raw = ProcessInfo.processInfo.environment["MLX_COMPILED_DECODE"] {
            return ["1", "true", "yes", "on"].contains(raw.lowercased())
        }
        return true
    }()
}
