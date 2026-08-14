import CryptoKit
import Darwin
import Foundation
import MLXFastCore

// QwenRuntime is split across QwenRuntime*.swift for auditability.
// Generated split; behavior identical to the original single file.

extension QwenRuntime {
    /// The commit stamped into the sealed score's `metrics.commit`, which the
    /// trusted workflow binds against the dispatched commit
    /// (MLXFAST_EXPECTED_COMMIT in overlay-paired-timing.sh and
    /// validate-benchmark-artifacts.sh).
    ///
    /// Ranked runs execute this process as the sandboxed `bench` uid inside a
    /// runner-owned workspace copy, where `git rev-parse` fails (dubious
    /// ownership under `env -i`), so git output is not a usable authority
    /// there. Instead the trusted context supplies the dispatched commit via
    /// MLXFAST_COMMIT_SHA: the ranked workflow threads it through the
    /// bench-exec'd gates argv, and benchmark.sh --official recovers it from
    /// the workflow-authored candidate.sha for the measure-job timed path.
    /// `git rev-parse` remains the local/dev fallback only.
    static func commitIdentifier(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String {
        if let supplied = environment["MLXFAST_COMMIT_SHA"] {
            let trimmed = supplied.trimmingCharacters(in: .whitespacesAndNewlines)
            if isCommitSHAHex(trimmed) {
                return trimmed
            }
        }
        return (try? runProcess("/usr/bin/git", arguments: ["rev-parse", "--short", "HEAD"])) ?? ""
    }

    /// Matches the trusted shell predicates' `^[0-9a-f]{7,40}$` (lowercase
    /// hex, short-to-full commit SHA).
    static func isCommitSHAHex(_ value: String) -> Bool {
        guard value.count >= 7, value.count <= 40 else {
            return false
        }
        return value.allSatisfy { ("0"..."9").contains($0) || ("a"..."f").contains($0) }
    }

    static func harnessHash() -> String {
        let roots = [
            "Package.swift",
            "Sources",
            "Tests",
            "benchmark.json",
            "benchmark.sh",
            "setup.sh",
            "tools",
            "README.md",
            "TASK.md",
        ]
        var files: [String] = []
        for root in roots {
            let url = URL(fileURLWithPath: root)
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: root, isDirectory: &isDirectory) else {
                continue
            }
            if isDirectory.boolValue {
                guard let enumerator = FileManager.default.enumerator(
                    at: url,
                    includingPropertiesForKeys: [.isRegularFileKey],
                    options: [.skipsHiddenFiles]
                ) else {
                    continue
                }
                for case let fileURL as URL in enumerator {
                    let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey])
                    if values?.isRegularFile == true {
                        files.append(fileURL.path)
                    }
                }
            } else {
                files.append(url.path)
            }
        }

        var hasher = SHA256()
        for path in files.sorted() {
            guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
                continue
            }
            hasher.update(data: Data(path.utf8))
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    struct DirectoryDigest: Equatable {
        let fileCount: Int
        let byteCount: Int
        let sha256: String
    }

    static func checkWorkerBenchmarkInputs(
        weightsPath: String,
        goldenPath: String
    ) throws {
        try requireDirectory(weightsPath, description: "transformed weights")
        let requiredFiles = [
            ("\(weightsPath)/config.json", "transformed config"),
            ("\(weightsPath)/model.safetensors.index.json", "dense safetensors index"),
            (goldenPath, "correctness golden file"),
        ]
        for (path, description) in requiredFiles {
            try requireRegularFile(path, description: description)
        }
    }

    static func requireDirectory(_ path: String, description: String) throws {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        if values.isSymbolicLink == true {
            throw MLXFastError.invalidInput("\(description) must not be a symlink: \(path)")
        }
        guard values.isDirectory == true else {
            throw MLXFastError.missingFile("\(description) directory missing at \(path)")
        }
    }

    static func requireRegularFile(_ path: String, description: String) throws {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
        if values.isSymbolicLink == true {
            throw MLXFastError.invalidInput("\(description) must not be a symlink: \(path)")
        }
        guard values.isRegularFile == true else {
            throw MLXFastError.missingFile("\(description) missing at \(path)")
        }
        try requireSingleHardLink(path: url.path, description: description)
    }

    /// Reject files with a POSIX link count other than 1 (hardlinks). A
    /// hardlink inside the transform output tree would let a second name --
    /// potentially owned or planted by the sandboxed bench uid outside the
    /// tree -- alias trusted bytes, mirroring the `find ... -type f -links +1`
    /// guard `overlay-editable-paths.sh` already applies to the editable
    /// overlay. Uses `lstat` so a symlink is never dereferenced here.
    static func requireSingleHardLink(path: String, description: String) throws {
        var info = stat()
        guard lstat(path, &info) == 0 else {
            throw MLXFastError.invalidInput("\(description) could not be stat'd: \(path)")
        }
        if (info.st_mode & mode_t(S_IFMT)) == mode_t(S_IFLNK) {
            throw MLXFastError.invalidInput("\(description) must not be a symlink: \(path)")
        }
        if info.st_nlink != 1 {
            throw MLXFastError.invalidInput(
                "\(description) must not be hardlinked (link count \(info.st_nlink)): \(path)"
            )
        }
    }

    static func enforceTransformedWeightsByteLimit(_ byteCount: Int) throws {
        guard let maxByteCount = try transformedWeightsByteLimit() else {
            return
        }
        guard byteCount <= maxByteCount else {
            throw MLXFastError.invalidInput(
                "transformed weights are \(byteCount) bytes, above MLXFAST_MAX_WEIGHTS_BYTES=\(maxByteCount)"
            )
        }
    }

    static func directoryDigest(
        rootPath: String,
        ignoredRelativePaths: Set<String>
    ) throws -> DirectoryDigest {
        let root = URL(fileURLWithPath: rootPath).standardizedFileURL
        let rootPrefix = root.path + "/"
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey],
            options: []
        ) else {
            throw MLXFastError.missingFile("directory not found at \(root.path)")
        }

        var files: [(relativePath: String, url: URL)] = []
        for case let url as URL in enumerator {
            let standardized = url.standardizedFileURL
            let path = standardized.path
            guard path.hasPrefix(rootPrefix) else {
                throw MLXFastError.invalidInput("path escaped digest root: \(path)")
            }
            let relativePath = String(path.dropFirst(rootPrefix.count))
            if ignoredRelativePaths.contains(relativePath) {
                continue
            }

            let values = try standardized.resourceValues(
                forKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey]
            )
            if values.isSymbolicLink == true {
                throw MLXFastError.invalidInput("directory digest rejects symlink \(relativePath)")
            }
            if values.isDirectory == true {
                continue
            }
            guard values.isRegularFile == true else {
                throw MLXFastError.invalidInput("directory digest rejects non-regular file \(relativePath)")
            }
            try requireSingleHardLink(
                path: standardized.path,
                description: "directory digest entry \(relativePath)"
            )
            files.append((relativePath: relativePath, url: standardized))
        }

        var treeHasher = SHA256()
        var byteCount = 0
        for file in files.sorted(by: { $0.relativePath < $1.relativePath }) {
            let size = try fileSizeByteCount(
                from: FileManager.default.attributesOfItem(atPath: file.url.path),
                path: file.url.path
            )
            guard byteCount <= Int.max - size else {
                throw MLXFastError.invalidInput("directory digest byte count exceeds Int range")
            }
            byteCount += size
            let digest = try fileDigest(file.url)
            treeHasher.update(data: Data(file.relativePath.utf8))
            treeHasher.update(data: Data([0]))
            treeHasher.update(data: Data(digest))
            treeHasher.update(data: Data([0]))
        }

        return DirectoryDigest(
            fileCount: files.count,
            byteCount: byteCount,
            sha256: treeHasher.finalize().map { String(format: "%02x", $0) }.joined()
        )
    }

    static func fileDigest(_ url: URL) throws -> SHA256.Digest {
        let handle = try FileHandle(forReadingFrom: url)
        defer {
            try? handle.close()
        }
        // Digest bytes are short-lived. Keeping a second copy in the unified
        // buffer cache while the model is about to load can exhaust 36 GiB
        // machines, so match the runtime weight loader's uncached read path.
        _ = Darwin.fcntl(handle.fileDescriptor, F_NOCACHE, 1)
        _ = Darwin.fcntl(handle.fileDescriptor, F_RDAHEAD, 0)

        var hasher = SHA256()
        let chunkSize = 8 * 1024 * 1024
        while true {
            let reachedEOF = autoreleasepool {
                let data = handle.readData(ofLength: chunkSize)
                if data.isEmpty {
                    return true
                }
                hasher.update(data: data)
                return false
            }
            if reachedEOF {
                return hasher.finalize()
            }
        }
    }

    static func runProcess(_ executable: String, arguments: [String]) throws -> String {
        let process = Process()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.standardOutput = output
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            return ""
        }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

}
