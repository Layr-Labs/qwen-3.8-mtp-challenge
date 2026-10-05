import Darwin
import Foundation
import MLX
import MLXFastModel

/// The HTTP loader's allocator cap.
///
/// Ranked workers set this before the first MLX allocation. The serving
/// process used to leave MLX's default in place, which is the memory limit:
/// 1.5 times the device's recommended working set. Discarded prefill buffers
/// from one request then stay in the free-buffer pool and count against every
/// later user. Weights stay outside this cap; the wired-limit ticket still
/// pins the post-warm active set.
///
/// `MLX_MAX_MB_PER_BUFFER` and `MLX_MAX_OPS_PER_BUFFER` are read once, on the
/// first Metal device access. `Memory.cacheLimit` is that access, so the
/// command-buffer budgets are installed first. The cache cap itself is
/// `mlx_set_cache_limit`; an environment variable cannot install it. This
/// function does not call `Memory.clearCache()` and does not set the cap to
/// zero. A trim can release a Metal buffer that an in-flight command buffer
/// still references.
enum ServingMemoryPolicy {
    static func install(
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) {
        let policy = RuntimeStartupMemoryPolicy.resolve(
            physicalMemoryBytes: physicalMemoryBytes
        )
        setenv(
            "MLX_MAX_MB_PER_BUFFER",
            String(policy.maxMegabytesPerCommandBuffer),
            1
        )
        setenv(
            "MLX_MAX_OPS_PER_BUFFER",
            String(policy.maxOperationsPerCommandBuffer),
            1
        )
        Memory.cacheLimit = policy.cacheLimitBytes
    }
}
