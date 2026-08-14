import CryptoKit
import Foundation
import MLXFastCore

// Head delivery for the Qwen 3.6 native-MTP track (qwen3.8-27b-mtp-v1).
//
// OPERATOR-RATIFIED 2026-08-14. The MTP head is part of the competitive
// surface: a submission may bring its own. It declares one in
// `mtp-head.manifest.json`, which is an editable path, and the RUNNER resolves
// that declaration pre-sandbox exactly the way it resolves a hidden golden --
// fetch, digest-verify, refuse on mismatch or oversize.
//
// THE SAFETY ARGUMENT, in one line: a head only PROPOSES tokens. The
// organizer-pinned target decides every emitted token and the trusted parent
// re-checks the whole stream against a hidden serial trajectory after the clock
// stops, so a substituted head moves the accept rate -- which is the game --
// and cannot move the output.
//
// THIS FILE IS TRUSTED CODE. It parses the declaration and computes the
// provenance the sealed report carries; it never decides whether a run passes.
// The `head_provenance` block it produces replaced three booleans
// (uses_pinned_mtp_head / uses_native_mtp_head / mtp_head_attached) that used
// to be pass-conditions in the workflow's jq and are now recorded fields.

/// One parsed `mtp-head.manifest.json`.
public struct QwenMTPHeadDeclaration: Equatable, Sendable {
    public enum Source: String, Equatable, CaseIterable, Sendable {
        /// The operator's section-9d pinned head. The default, and what an
        /// ABSENT manifest means.
        case pinned
        /// Fetched by the runner from `sourceURL` (`hf:<repo>@<rev>` or
        /// `r2:<key>`) into the run's private directory.
        case remote
        /// Shipped in the submission under the editable weights directory
        /// named by `path`.
        case inBranch = "in_branch"
    }

    public let source: Source
    public let sourceURL: String?
    public let path: String?
    public let sha256: String?
    public let bytes: Int
    public let maxBytes: Int

    public init(
        source: Source,
        sourceURL: String? = nil,
        path: String? = nil,
        sha256: String? = nil,
        bytes: Int = 0,
        maxBytes: Int = QwenMTPHeadDeclaration.defaultMaxBytes
    ) {
        self.source = source
        self.sourceURL = sourceURL
        self.path = path
        self.sha256 = sha256
        self.bytes = bytes
        self.maxBytes = maxBytes
    }

    /// 2 GiB. Mirrored in `mtp-head.manifest.json` (`max_bytes`) and in the
    /// contract manifest's `editableSurfaceByteBudget.exemptPathMaxBytes`; a
    /// declaration may lower it and may not raise it.
    public static let defaultMaxBytes = 2_147_483_648

    /// The default when no manifest exists at all.
    public static let pinnedDefault = QwenMTPHeadDeclaration(source: .pinned)

    public static let relativePath = "mtp-head.manifest.json"

    /// Parse a declaration, FAILING CLOSED on anything malformed.
    ///
    /// Only two things select the pinned head: the file being absent, and an
    /// explicit `source: "pinned"`. A manifest that is present but unreadable,
    /// unparseable, or internally inconsistent is a REFUSAL -- never a silent
    /// fall back -- because "your head declaration was broken so we quietly
    /// scored you on the pinned head" is exactly the failure mode that makes a
    /// leaderboard number unattributable.
    public static func parse(contentsOf url: URL) throws -> QwenMTPHeadDeclaration {
        guard let data = FileManager.default.contents(atPath: url.path) else {
            throw MLXFastError.invalidInput(
                "the MTP head declaration at \(url.path) exists but could not "
                    + "be read; refusing to fall back to the pinned head")
        }
        return try parse(data: data, origin: url.path)
    }

    public static func parse(
        data: Data,
        origin: String
    ) throws -> QwenMTPHeadDeclaration {
        guard let root = (try? JSONSerialization.jsonObject(with: data))
            as? [String: Any]
        else {
            throw MLXFastError.invalidInput(
                "the MTP head declaration at \(origin) is not a JSON object")
        }
        let rawSource = (root["source"] as? String) ?? Source.pinned.rawValue
        guard let source = Source(rawValue: rawSource) else {
            throw MLXFastError.invalidInput(
                "the MTP head declaration at \(origin) names an unknown source "
                    + "'\(rawSource)'; expected one of "
                    + Source.allCases.map(\.rawValue).joined(separator: ", "))
        }
        let maxBytes = (root["max_bytes"] as? NSNumber)?.intValue
            ?? defaultMaxBytes
        guard maxBytes > 0, maxBytes <= defaultMaxBytes else {
            throw MLXFastError.invalidInput(
                "the MTP head declaration at \(origin) sets max_bytes "
                    + "\(maxBytes); it must be positive and may not exceed the "
                    + "track cap \(defaultMaxBytes)")
        }
        let sourceURL = root["source_url"] as? String
        let path = root["path"] as? String
        let sha256 = (root["sha256"] as? String)?.lowercased()
        let bytes = (root["bytes"] as? NSNumber)?.intValue ?? 0

        if source != .pinned {
            guard let sha256, sha256.count == 64,
                  sha256.allSatisfy({ $0.isHexDigit })
            else {
                throw MLXFastError.invalidInput(
                    "the MTP head declaration at \(origin) selects source "
                        + "'\(source.rawValue)' without a 64-hex-character "
                        + "sha256; an unverifiable head is a refusal")
            }
            guard bytes > 0 else {
                throw MLXFastError.invalidInput(
                    "the MTP head declaration at \(origin) selects source "
                        + "'\(source.rawValue)' without a positive byte count")
            }
            guard bytes <= maxBytes else {
                throw MLXFastError.invalidInput(
                    "the declared MTP head is \(bytes) bytes, above the "
                        + "\(maxBytes)-byte cap in \(origin)")
            }
        }
        switch source {
        case .pinned:
            break
        case .remote:
            guard let sourceURL, !sourceURL.isEmpty,
                  sourceURL.hasPrefix("hf:") || sourceURL.hasPrefix("r2:")
            else {
                throw MLXFastError.invalidInput(
                    "the MTP head declaration at \(origin) selects source "
                        + "'remote' without a source_url of the form "
                        + "hf:<repo>@<revision> or r2:<key>")
            }
        case .inBranch:
            guard let path, !path.isEmpty, !path.hasPrefix("/"),
                  !path.contains(".."), !path.contains("\\")
            else {
                throw MLXFastError.invalidInput(
                    "the MTP head declaration at \(origin) selects source "
                        + "'in_branch' without a safe repo-relative path")
            }
        }
        return QwenMTPHeadDeclaration(
            source: source,
            sourceURL: sourceURL,
            path: path,
            sha256: sha256,
            bytes: bytes,
            maxBytes: maxBytes
        )
    }

    /// Read the declaration next to a contract root, treating ABSENCE as the
    /// pinned default and everything else as parse-or-refuse.
    public static func resolve(contractRoot: URL) throws -> QwenMTPHeadDeclaration {
        let url = contractRoot.appendingPathComponent(relativePath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return pinnedDefault
        }
        return try parse(contentsOf: url)
    }
}

/// Hash a head tree and pair the digest with the declared source.
///
/// TREE DIGEST RULE, stated once and mirrored in `mtp-head/README.md` so a
/// participant can recompute the number they are asked to declare: SHA-256 over
/// the concatenation, in `LC_ALL=C` sorted relative-path order, of
/// `"<hex file sha256>  <relative path>\n"` for every regular file in the tree
/// except a top-level `README.md`. One number describes a multi-file head, and
/// it does not depend on directory-enumeration order.
///
/// Runs OUTSIDE every scored window -- the CLI computes it while assembling the
/// evidence payload, after the parent's clock has stopped.
public func computeQwenMTPHeadProvenance(
    headDirectory: String,
    declaration: QwenMTPHeadDeclaration
) -> QwenMTPHeadProvenance {
    let root = URL(fileURLWithPath: headDirectory)
    var entries: [(path: String, digest: String, bytes: Int)] = []
    let fileManager = FileManager.default
    if let enumerator = fileManager.enumerator(atPath: root.path) {
        while let relative = enumerator.nextObject() as? String {
            guard let type = enumerator.fileAttributes?[.type]
                as? FileAttributeType, type == .typeRegular
            else { continue }
            if relative == "README.md" { continue }
            let fileURL = root.appendingPathComponent(relative)
            guard let digest = sha256OfFile(fileURL) else { continue }
            let bytes = (enumerator.fileAttributes?[.size] as? NSNumber)?
                .intValue ?? 0
            entries.append((relative, digest, bytes))
        }
    }
    entries.sort { $0.path < $1.path }
    var hasher = SHA256()
    var totalBytes = 0
    for entry in entries {
        hasher.update(data: Data("\(entry.digest)  \(entry.path)\n".utf8))
        totalBytes += entry.bytes
    }
    let treeDigest = hasher.finalize()
        .map { String(format: "%02x", $0) }
        .joined()
    let origin: String
    switch declaration.source {
    case .pinned: origin = headDirectory
    case .remote: origin = declaration.sourceURL ?? headDirectory
    case .inBranch: origin = declaration.path ?? headDirectory
    }
    return QwenMTPHeadProvenance(
        source: declaration.source.rawValue,
        sha256: treeDigest,
        bytes: totalBytes,
        fileCount: entries.count,
        origin: origin
    )
}

/// Streamed file digest; a head file is hundreds of megabytes and must not be
/// read into memory whole.
private func sha256OfFile(_ url: URL) -> String? {
    guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
    defer { try? handle.close() }
    var hasher = SHA256()
    while true {
        guard let chunk = (try? handle.read(upToCount: 4 * 1024 * 1024))
            ?? nil, !chunk.isEmpty
        else { break }
        hasher.update(data: chunk)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}
