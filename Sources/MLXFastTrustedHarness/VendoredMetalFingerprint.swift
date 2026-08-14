import CryptoKit
import Foundation
import MLXFastCore

/// Content fingerprint over every vendored source that can influence the AOT
/// `mlx.metallib` (the whole vendored mlx tree) plus the runtime-effective JIT
/// kernel strings (`mlx-generated`).
///
/// This is the bit-for-bit Swift twin of
/// `tools/build-mlx-metallib.sh`'s `compute_vendored_metal_fingerprint`: the
/// builder publishes the fingerprint as a `mlx.metallib.fingerprint` sidecar
/// at build time, and the trusted CLI recomputes it here immediately before
/// spawning the participant worker, so a cached or stale metallib can never
/// silently mask participant edits to the AOT-served kernel sources (RoPE,
/// RMSNorm, the SDPA vector kernel, arg_reduce). Keep the two implementations
/// in sync: per-file SHA-256 lines in `shasum` format over byte-sorted
/// relative paths, hashed again.
public enum VendoredMetalFingerprint {
    public static let recordPrefix = "mlxfast-metallib-fingerprint-v1"
    public static let fingerprintedSubtrees = ["mlx", "mlx-generated"]
    public static let defaultCmlxRelativePath = "Vendor/mlx-swift/Source/Cmlx"

    public static func compute(cmlxRoot: String) throws -> String {
        let fileManager = FileManager.default
        var relativePaths: [String] = []
        for subtree in fingerprintedSubtrees {
            let subtreeRoot = URL(fileURLWithPath: cmlxRoot)
                .appendingPathComponent(subtree).path
            var isDirectory = ObjCBool(false)
            guard fileManager.fileExists(atPath: subtreeRoot, isDirectory: &isDirectory),
                  isDirectory.boolValue
            else {
                throw MLXFastError.invalidInput(
                    "vendored source tree missing for metallib fingerprint: \(subtreeRoot)"
                )
            }
            guard let enumerator = fileManager.enumerator(atPath: subtreeRoot) else {
                throw MLXFastError.invalidInput(
                    "could not enumerate vendored source tree: \(subtreeRoot)"
                )
            }
            while let entry = enumerator.nextObject() as? String {
                // Regular files only, exactly like `find -type f`: symlinks
                // and directories never contribute to the fingerprint.
                guard let type = enumerator.fileAttributes?[.type] as? FileAttributeType,
                      type == .typeRegular
                else {
                    continue
                }
                relativePaths.append("\(subtree)/\(entry)")
            }
        }
        // LC_ALL=C sort order: raw byte comparison of the relative paths.
        relativePaths.sort { lhs, rhs in
            lhs.utf8.lexicographicallyPrecedes(rhs.utf8)
        }

        var listing = Data()
        for relativePath in relativePaths {
            let fileURL = URL(fileURLWithPath: cmlxRoot)
                .appendingPathComponent(relativePath)
            let contents = try Data(contentsOf: fileURL, options: [.mappedIfSafe])
            let digest = SHA256.hash(data: contents)
            let line = "\(hexEncoded(digest))  \(relativePath)\n"
            listing.append(Data(line.utf8))
        }
        return hexEncoded(SHA256.hash(data: listing))
    }

    public static func recordLine(fingerprint: String) -> String {
        "\(recordPrefix) \(fingerprint)\n"
    }

    private static func hexEncoded(_ digest: SHA256.Digest) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }
}

/// Outcome of checking a published `mlx.metallib.fingerprint` sidecar against
/// the vendored sources actually on disk.
public enum MetallibFingerprintVerification: Equatable {
    case verified
    /// The check could not run at all (no vendored tree / no metallib). The
    /// caller decides whether that is fatal: official runs fail closed, local
    /// runs proceed (other guards report the missing pieces).
    case skipped(reason: String)
    case mismatch(reason: String)
}

/// Verify that the metallib the participant worker will load still
/// corresponds to the vendored Metal sources in `cmlxRoot`.
public func verifyMetallibFingerprintRecord(
    metallibPath: String,
    cmlxRoot: String
) -> MetallibFingerprintVerification {
    let fileManager = FileManager.default
    for subtree in VendoredMetalFingerprint.fingerprintedSubtrees {
        let subtreeRoot = URL(fileURLWithPath: cmlxRoot)
            .appendingPathComponent(subtree).path
        if !fileManager.fileExists(atPath: subtreeRoot) {
            return .skipped(
                reason: "vendored MLX sources are not present at \(cmlxRoot)"
            )
        }
    }
    if !fileManager.fileExists(atPath: metallibPath) {
        return .skipped(reason: "mlx.metallib is not present at \(metallibPath)")
    }

    let recordPath = metallibPath + ".fingerprint"
    guard let recordData = fileManager.contents(atPath: recordPath),
          let record = String(data: recordData, encoding: .utf8)
    else {
        return .mismatch(
            reason: "no fingerprint record at \(recordPath); rebuild the metallib "
                + "with tools/build-mlx-metallib.sh (or rerun ./setup.sh)"
        )
    }
    let fields = record
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .split(separator: " ", omittingEmptySubsequences: true)
    guard fields.count == 2,
          fields[0] == VendoredMetalFingerprint.recordPrefix
    else {
        return .mismatch(
            reason: "fingerprint record at \(recordPath) is malformed or has an "
                + "unsupported version; rebuild the metallib with "
                + "tools/build-mlx-metallib.sh"
        )
    }

    let expected: String
    do {
        expected = try VendoredMetalFingerprint.compute(cmlxRoot: cmlxRoot)
    } catch {
        return .mismatch(
            reason: "could not fingerprint the vendored Metal sources: \(error)"
        )
    }
    if String(fields[1]) != expected {
        return .mismatch(
            reason: "mlx.metallib at \(metallibPath) was built from different "
                + "vendored Metal sources than the ones on disk (recorded "
                + "\(fields[1]), current \(expected)); rebuild it with "
                + "tools/build-mlx-metallib.sh (or rerun ./setup.sh) so kernel "
                + "edits are not masked by a stale metallib"
        )
    }
    return .verified
}
