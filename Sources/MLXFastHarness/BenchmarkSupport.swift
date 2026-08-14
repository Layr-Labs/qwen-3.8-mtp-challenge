import Foundation
import MLXFastCore
import MLXFastModel

public struct BenchmarkPreflightReport: Codable, Equatable {
    public let weightsPath: String
    public let goldenPath: String
    public let weightsByteCount: Int
    public let maxWeightsByteCount: Int?

    public init(
        weightsPath: String,
        goldenPath: String,
        weightsByteCount: Int = 0,
        maxWeightsByteCount: Int? = MLXFastConstants.defaultMaxTransformedWeightsBytes
    ) {
        self.weightsPath = weightsPath
        self.goldenPath = goldenPath
        self.weightsByteCount = weightsByteCount
        self.maxWeightsByteCount = maxWeightsByteCount
    }
}

/// Same-session baseline measurement supplied by the trusted workflow (see
/// docs/benchmark-window-freeze.md, "Paired baseline measurement"). The official
/// timing machine measures the pinned reference implementation immediately
/// before the candidate and passes its seconds-per-token through these
/// environment variables, so candidate speedups and floors are computed against
/// the same runner VM, hour, and thermal state instead of a constant measured
/// on a different host weeks earlier. Both variables are stripped from the
/// sandboxed worker environment: submitted model code must not observe them.
///
/// Resolution precedence in the benchmark paths is paired override, then the
/// golden's per-prompt baselines, then the calibrated constants.
public struct PairedBaselineOverride: Equatable {
    public static let prefillEnvironmentKey = "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN"
    public static let decodeEnvironmentKey = "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN"

    public let prefillSecondsPerToken: Double
    public let decodeSecondsPerToken: Double

    public init(prefillSecondsPerToken: Double, decodeSecondsPerToken: Double) {
        self.prefillSecondsPerToken = prefillSecondsPerToken
        self.decodeSecondsPerToken = decodeSecondsPerToken
    }

    /// Fails closed: a half-set pair or a non-positive/non-finite value is an
    /// operator wiring error and must stop the run, never silently degrade to
    /// the constants (which would misprice the whole session).
    public static func fromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> PairedBaselineOverride? {
        let prefillRaw = environment[prefillEnvironmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let decodeRaw = environment[decodeEnvironmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if prefillRaw.isEmpty, decodeRaw.isEmpty {
            return nil
        }
        guard !prefillRaw.isEmpty, !decodeRaw.isEmpty else {
            throw MLXFastError.invalidInput(
                "\(prefillEnvironmentKey) and \(decodeEnvironmentKey) must be provided together"
            )
        }
        guard let prefill = Double(prefillRaw), prefill.isFinite, prefill > 0 else {
            throw MLXFastError.invalidInput(
                "\(prefillEnvironmentKey) must be a finite positive seconds-per-token value"
            )
        }
        guard let decode = Double(decodeRaw), decode.isFinite, decode > 0 else {
            throw MLXFastError.invalidInput(
                "\(decodeEnvironmentKey) must be a finite positive seconds-per-token value"
            )
        }
        return PairedBaselineOverride(
            prefillSecondsPerToken: prefill,
            decodeSecondsPerToken: decode
        )
    }
}

public enum BenchmarkPreflight {
    public static func check(
        weightsPath: String,
        goldenPath: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> BenchmarkPreflightReport {
        try checkArtifacts(
            weightsPath: weightsPath,
            goldenPath: goldenPath,
            requiresBenchmarkOracle: true,
            environment: environment
        )
    }

    public static func checkCorrectnessArtifacts(
        weightsPath: String,
        goldenPath: String,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> BenchmarkPreflightReport {
        try checkArtifacts(
            weightsPath: weightsPath,
            goldenPath: goldenPath,
            requiresBenchmarkOracle: false,
            environment: environment
        )
    }

    private static func checkArtifacts(
        weightsPath: String,
        goldenPath: String,
        requiresBenchmarkOracle: Bool,
        environment: [String: String]
    ) throws -> BenchmarkPreflightReport {
        let requiredFiles = [
            ("\(weightsPath)/config.json", "transformed config"),
            ("\(weightsPath)/model.safetensors.index.json", "dense safetensors index"),
            (goldenPath, "correctness golden file"),
        ]
        for (path, description) in requiredFiles {
            try requireFile(path, description: description)
        }

        let maxWeightsByteCount = try transformedWeightsByteLimit(from: environment)
        let weightsByteCount = try transformedWeightsByteCount(
            weightsPath: weightsPath,
            maxByteCount: maxWeightsByteCount
        )

        let golden = try loadGoldenFixture(from: goldenPath)
        if requiresBenchmarkOracle {
            guard golden.benchmark != nil else {
                throw MLXFastError.invalidInput("benchmark golden file must contain a benchmark oracle")
            }
        }
        let config = try Qwen35Config.load(from: weightsPath)

        let denseStore = try DenseTensorStore(weightsPath: weightsPath)
        try denseStore.validateReadableByteRanges()

        try Qwen35WeightLoader(denseStore: denseStore)
            .validateRequiredMetadata(config: config)

        return BenchmarkPreflightReport(
            weightsPath: weightsPath,
            goldenPath: goldenPath,
            weightsByteCount: weightsByteCount,
            maxWeightsByteCount: maxWeightsByteCount
        )
    }

    private static func transformedWeightsByteCount(
        weightsPath: String,
        maxByteCount: Int?,
        fileManager: FileManager = .default
    ) throws -> Int {
        let root = URL(fileURLWithPath: weightsPath).standardizedFileURL
        let rootValues = try root.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard rootValues.isSymbolicLink != true else {
            throw MLXFastError.invalidInput("transformed weights path must not be a symlink: \(root.path)")
        }
        guard rootValues.isDirectory == true else {
            throw MLXFastError.invalidInput("transformed weights path must be a directory: \(root.path)")
        }

        let rootPrefix = root.path + "/"
        guard let enumerator = fileManager.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey],
            options: []
        ) else {
            throw MLXFastError.missingFile("transformed weights directory not found at \(root.path)")
        }

        var byteCount = 0
        for case let url as URL in enumerator {
            let standardized = url.standardizedFileURL
            let path = standardized.path
            guard path.hasPrefix(rootPrefix) else {
                throw MLXFastError.invalidInput("transformed weights path escaped root: \(path)")
            }
            let relativePath = String(path.dropFirst(rootPrefix.count))
            let values = try standardized.resourceValues(
                forKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey]
            )
            if values.isSymbolicLink == true {
                throw MLXFastError.invalidInput("transformed weights must not contain symlink \(relativePath)")
            }
            if values.isDirectory == true {
                continue
            }
            guard values.isRegularFile == true else {
                throw MLXFastError.invalidInput("transformed weights contains non-regular file \(relativePath)")
            }

            let size = try fileSizeByteCount(
                from: fileManager.attributesOfItem(atPath: standardized.path),
                path: standardized.path
            )
            guard byteCount <= Int.max - size else {
                throw MLXFastError.invalidInput("transformed weights byte count exceeds Int range")
            }
            byteCount += size
            if let maxByteCount, byteCount > maxByteCount {
                throw MLXFastError.invalidInput(
                    "transformed weights are \(byteCount) bytes, above limit \(maxByteCount)"
                )
            }
        }
        return byteCount
    }
}
