// Copyright © 2026 Apple Inc.
//
// Select the JACCL distributed backend the same way mlx's CMake does
// (mlx/mlx/distributed/jaccl/CMakeLists.txt): build the real RDMA-backed
// backend only on macOS when BOTH the SDK and the deployment target are
// >= 26.2, otherwise use the no-op stub. This keeps unsupported Apple targets
// (older macOS, iOS, tvOS, visionOS) on the stub instead of compiling the RDMA
// implementation.
//
// CMake requires `MACOS_SDK_VERSION >= 26.2 AND CMAKE_OSX_DEPLOYMENT_TARGET >=
// 26.2`. We mirror that with two checks:
//   - __MAC_OS_X_VERSION_MAX_ALLOWED  == the SDK version (RDMA APIs available?)
//   - __MAC_OS_X_VERSION_MIN_REQUIRED == the deployment target.
// Without the SDK (MAX_ALLOWED) check, building with an older SDK but a 26.2
// deployment target would pull in the RDMA sources and fail on declarations the
// older SDK does not provide.
//
// On Linux this file is not compiled (mlx-conditional is excluded there); the
// SwiftPM target compiles no_jaccl.cpp directly instead.

#if defined(__APPLE__)
#include <AvailabilityMacros.h> // __MAC_OS_X_VERSION_{MIN_REQUIRED,MAX_ALLOWED}
#include <TargetConditionals.h>
#if TARGET_OS_OSX &&                                  \
    defined(__MAC_OS_X_VERSION_MIN_REQUIRED) &&       \
    (__MAC_OS_X_VERSION_MIN_REQUIRED >= 260200) &&    \
    defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&        \
    (__MAC_OS_X_VERSION_MAX_ALLOWED >= 260200)
#define MLX_JACCL_REAL 1
#endif
#endif

#ifdef MLX_JACCL_REAL
#include "../mlx/mlx/distributed/jaccl/jaccl.cpp"
#include "../mlx/mlx/distributed/jaccl/lib/jaccl/jaccl.cpp"
#include "../mlx/mlx/distributed/jaccl/lib/jaccl/mesh.cpp"
#include "../mlx/mlx/distributed/jaccl/lib/jaccl/rdma.cpp"
#include "../mlx/mlx/distributed/jaccl/lib/jaccl/ring.cpp"
#include "../mlx/mlx/distributed/jaccl/lib/jaccl/tcp.cpp"
#else
#include "../mlx/mlx/distributed/jaccl/no_jaccl.cpp"
#endif
