// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "mlxfast-challenge-dev",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "mlxfast-swift", targets: ["MLXFastCLI"]),
        .executable(
            name: "mlxfast-runtime-worker",
            targets: ["MLXFastRuntimeWorkerCLI"]
        ),
        .library(name: "MLXFastCore", targets: ["MLXFastCore"]),
        .library(name: "MLXFastTransform", targets: ["MLXFastTransform"]),
        .library(name: "MLXFastModel", targets: ["MLXFastModel"]),
        .library(name: "MLXFastHarness", targets: ["MLXFastHarness"]),
    ],
    dependencies: [
        // Exact vendored revisions:
        // mlx-swift df1fdc5f7821a1fabe921fdefbc42ac74dcfb6bc
        // mlx-swift-lm bc1c0ee67d15798343be17c9f8f61f7c0d977149
        .package(path: "Vendor/mlx-swift"),
        .package(path: "Vendor/mlx-swift-lm"),
        // The resolved dependency graph is frozen and asserted before either
        // trusted-harness or participant-worker builds begin: setup.sh and
        // benchmark.sh refuse to build over a Package.swift/Package.resolved
        // that differs from the committed state, the ranked workflows
        // byte-verify both against the trusted ref
        // (.github/scripts/verify-trusted-source-scope.sh), and every build
        // and resolve passes --force-resolved-versions so SwiftPM fails
        // closed instead of silently re-resolving.
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.3"),
    ],
    targets: [
        .target(name: "MLXFastCore"),
        .target(
            name: "MLXFastTransform",
            dependencies: ["MLXFastCore"]
        ),
        .target(
            name: "MLXFastModel",
            dependencies: [
                "MLXFastCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                // DFlash block-decode track (laguna-xs-2.1-dflash-v1):
                // LagunaDFlashBlockSession wraps the vendored speculative round.
                // The serial track does not reach this code.
                .product(name: "MLXSpeculative", package: "mlx-swift-lm"),
            ]
        ),
        .target(
            name: "MLXFastRuntimeWorkerSupport",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastModel",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                // DFlash block-decode worker: loads the organizer-pinned target
                // through the vendored factory (LagunaModel is the type that
                // conforms to DFlashTargetModel) and the DFlash drafter.
                .product(name: "MLXSpeculative", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
            ],
            path: "Sources/MLXFastHarness"
        ),
        // The trusted-harness source scope (this manifest, Package.resolved,
        // Sources/MLXFastCLI, Sources/MLXFastTrustedHarness, and
        // Sources/MLXFastCore) is pinned independently of participant-
        // controlled manifests: the ranked workflows byte-verify it against
        // trusted git content before every build via
        // .github/scripts/verify-trusted-source-scope.sh, so a submission
        // cannot expand or repoint the targets feeding the trusted binary.
        .target(
            name: "MLXFastHarness",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/MLXFastTrustedHarness",
            swiftSettings: [
                .define("MLXFAST_TRUSTED_HARNESS")
            ]
        ),
        .executableTarget(
            name: "MLXFastCLI",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastHarness",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "MLXFastRuntimeWorkerCLI",
            dependencies: [
                "MLXFastCore",
                "MLXFastModel",
                "MLXFastRuntimeWorkerSupport",
            ]
        ),
        .testTarget(
            name: "MLXFastTests",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastModel",
                "MLXFastHarness",
                "MLXFastRuntimeWorkerSupport",
            ]
        ),
    ]
)
