// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import CompilerPluginSupport
import PackageDescription

let package = Package(
    name: "mlx-swift-lm",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "MLXLLM",
            targets: ["MLXLLM"]),
        .library(
            name: "MLXVLM",
            targets: ["MLXVLM"]),
        .library(
            name: "MLXLMCommon",
            targets: ["MLXLMCommon"]),
        .library(
            name: "MLXEmbedders",
            targets: ["MLXEmbedders"]),
        .library(
            name: "MLXHuggingFace",
            targets: ["MLXHuggingFace"]),
        .library(
            name: "MLXLMServer",
            targets: ["MLXLMServer"]),
        .executable(
            name: "mlx-server",
            targets: ["mlx-server"]),
        .library(
            name: "BenchmarkHelpers",
            targets: ["BenchmarkHelpers"]),
        .library(
            name: "IntegrationTestHelpers",
            targets: ["IntegrationTestHelpers"]),
        .library(
            name: "MLXSpeculative",
            targets: ["MLXSpeculative"]),
        .executable(
            name: "mlx-bench",
            targets: ["mlx-bench"]),
    ],
    dependencies: [
        .package(path: "../mlx-swift"),
        .package(url: "https://github.com/swiftlang/swift-syntax.git", "600.0.0" ..< "604.0.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.23.0"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.2"),
    ],
    targets: [
        .target(
            name: "MLXLLM",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            path: "Libraries/MLXLLM",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "MLXVLM",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            path: "Libraries/MLXVLM",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "MLXLMCommon",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                // Cmlx: low-level mlx_slice_dynamic / mlx_slice_update_dynamic
                // entry points used by DynamicSlice.swift (compiled-decode infra).
                .product(name: "Cmlx", package: "mlx-swift"),
            ],
            path: "Libraries/MLXLMCommon",
            exclude: [
                "README.md"
            ],
            resources: [
                // CBv2 paged-attention MSL source, JIT-compiled at runtime
                // via MLXFast.metalKernel (NOT compiled by SwiftPM).
                .copy("ContinuousBatchingV2/Paged/pagedattention.metal")
            ]
        ),
        .target(
            name: "MLXEmbedders",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .target(name: "MLXLMCommon"),
            ],
            path: "Libraries/MLXEmbedders",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "MLXLMServer",
            dependencies: [
                "MLXLLM",
                "MLXVLM",
                "MLXLMCommon",
                "MLXEmbedders",
                "MLXHuggingFace",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Libraries/MLXLMServer"
        ),
        .executableTarget(
            name: "mlx-server",
            dependencies: ["MLXLMServer"],
            path: "Executables/mlx-server"
        ),
        .target(
            name: "BenchmarkHelpers",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Libraries/BenchmarkHelpers"
        ),
        .target(
            name: "IntegrationTestHelpers",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Libraries/IntegrationTestHelpers",
            exclude: ["README.md"]
        ),
        .testTarget(
            name: "MLXLMTests",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                "MLXSpeculative",
            ],
            path: "Tests/MLXLMTests",
            exclude: [
                "README.md"
            ],
            resources: [
                .process("Resources/1080p_30.mov"),
                .process("Resources/audio_only.mov"),
                .process("Resources/dflash-laguna-xs-2.1-config.json"),
            ]
        ),
        .testTarget(
            name: "MLXLMServerTests",
            dependencies: [
                "MLXLMServer",
                "MLXLMCommon",
                .product(name: "HummingbirdTesting", package: "hummingbird"),
            ],
            path: "Tests/MLXLMServerTests"
        ),
        .macro(
            name: "MLXHuggingFaceMacros",
            dependencies: [
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
            ],
            path: "Libraries/MLXHuggingFaceMacros"
        ),
        .target(
            name: "MLXHuggingFace",
            dependencies: [
                "MLXHuggingFaceMacros",
                "MLXLMCommon",
            ],
            path: "Libraries/MLXHuggingFace"
        ),
        .executableTarget(
            name: "BenchLoad",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "BenchmarkHelpers",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/BenchLoad"
        ),
        // WS-G (engine-v2) benchmark harness: tiny-model v2-style step loop
        // (CI-runnable) + `--model <path>` legacy-engine baseline for real
        // weights. Reports land as markdown next to the other benchmarks/
        // reports. `sources:` keeps the target scoped to the one Swift file.
        .executableTarget(
            name: "CBv2Benchmark",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "benchmarks",
            sources: ["CBv2Benchmark.swift"]
        ),
        // Real-weights validation driver for the CBv2 engine: correctness
        // smoke (batch-composition + chunked-prefill invariance) and the
        // legacy-vs-v2 perf matrix on local model directories.
        .executableTarget(
            name: "BenchCBv2",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXHuggingFace",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/BenchCBv2"
        ),
        // DFlash speculative-decoding library (Laguna XS 2.1 draft-model
        // bring-up; dev-repo / M5-C experimentation only — not in benchmark.json
        // editablePaths, not wired into mlxfast-runtime-worker, unsubmittable).
        .target(
            name: "MLXSpeculative",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Libraries/MLXSpeculative",
            exclude: ["README.md"]
        ),
        // Standalone DFlash benchmark driver (baseline vs block-draft sweep,
        // parity check). Built with `swift build -c release --product mlx-bench`.
        .executableTarget(
            name: "mlx-bench",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXHuggingFace",
                "MLXSpeculative",
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/mlx-bench"
        ),
    ]
)

if Context.environment["MLX_SWIFT_BUILD_DOC"] == "1"
    || Context.environment["SPI_GENERATE_DOCS"] == "1"
{
    package.dependencies.append(
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0")
    )
}
