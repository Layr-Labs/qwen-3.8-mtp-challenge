import Foundation
import CryptoKit
import Darwin

enum MTPPinnedFiles {
    static func verifyFile(_ url: URL, bytes: UInt64, sha256: String) throws {
        let descriptor = open(url.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
        guard descriptor >= 0 else { throw MTPStartupError.invalid("Cannot open pinned regular file: \(url.path)") }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? handle.close() }
        var state = stat()
        guard fstat(descriptor, &state) == 0,
            state.st_mode & S_IFMT == S_IFREG,
            state.st_uid == geteuid() || state.st_uid == 0,
            state.st_mode & 0o022 == 0,
            state.st_size >= 0, UInt64(state.st_size) == bytes
        else { throw MTPStartupError.invalid("Pinned file type, owner, or size differs: \(url.path)") }
        try verifyContents(handle, bytes: bytes, sha256: sha256)
    }

    /// The descriptor may grow after the metadata check. Keep content
    /// verification separately bounded by the expected pinned byte count.
    static func verifyContents(_ handle: FileHandle, bytes: UInt64, sha256: String) throws {
        var digest = SHA256()
        var remaining = bytes
        while remaining > 0 {
            let requested = Int(Swift.min(remaining, UInt64(1 << 20)))
            guard let chunk = try handle.read(upToCount: requested), !chunk.isEmpty else {
                throw MTPStartupError.invalid("Pinned file ended before its expected byte count.")
            }
            digest.update(data: chunk)
            remaining -= UInt64(chunk.count)
        }
        guard try handle.read(upToCount: 1)?.isEmpty != false else {
            throw MTPStartupError.invalid("Pinned file exceeds its expected byte count.")
        }
        let actual = digest.finalize().map { String(format: "%02x", $0) }.joined()
        guard actual == sha256 else { throw MTPStartupError.invalid("Pinned SHA256 differs.") }
    }

    static let headSHA256 = "d038fd41e2d5dab1b3905c115d859fdc98dfbfde9862c14ebb82c2b3247ec2f1"
    static let headBytes: UInt64 = 427742600

    static func headDirectory(path: String) throws -> URL {
        guard path.hasPrefix("/") else { throw MTPStartupError.invalid("--mtp-head must be absolute.") }
        let pathURL = URL(fileURLWithPath: path)
        let type = try FileManager.default.attributesOfItem(atPath: path)[.type] as? FileAttributeType
        let directory: URL
        if type == .typeRegular, pathURL.lastPathComponent == "model.safetensors" {
            directory = pathURL.deletingLastPathComponent()
        } else if type == .typeDirectory {
            directory = pathURL
        } else { throw MTPStartupError.invalid("--mtp-head must name an owned regular model.safetensors file or its directory.") }
        let entries = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        try verifyDirectory(directory)
        guard entries == ["model.safetensors"] else {
            throw MTPStartupError.invalid("The pinned head directory must contain only model.safetensors.")
        }
        try verifyFile(directory.appendingPathComponent("model.safetensors"), bytes: headBytes, sha256: headSHA256)
        return directory
    }

    /// One pinned file directly inside the target snapshot directory.
    struct TargetFile: Equatable, Sendable, Decodable {
        let path: String
        let sha256: String
        let bytes: UInt64

        init(path: String, sha256: String, bytes: UInt64) {
            self.path = path
            self.sha256 = sha256
            self.bytes = bytes
        }

        /// Exact keys only: an unknown field may carry intent this verifier ignores.
        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: AnyKey.self)
            guard Set(container.allKeys.map(\.stringValue)) == ["path", "sha256", "bytes"] else {
                throw MTPStartupError.invalid("Target manifest records must contain exactly path, sha256, and bytes.")
            }
            path = try container.decode(String.self, forKey: AnyKey("path"))
            sha256 = try container.decode(String.self, forKey: AnyKey("sha256"))
            bytes = try container.decode(UInt64.self, forKey: AnyKey("bytes"))
        }

        private struct AnyKey: CodingKey {
            let stringValue: String
            var intValue: Int? { nil }
            init(_ stringValue: String) { self.stringValue = stringValue }
            init?(stringValue: String) { self.stringValue = stringValue }
            init?(intValue: Int) { nil }
        }
    }

    /// The raw EigenLabs/Qwen3.8-27B-4bit@eda45ab snapshot used when no
    /// --target-manifest is given.
    static func compiledTarget() throws -> [TargetFile] {
        try validatedTarget(targetManifest.split(separator: "\n").map { line in
            let fields = line.split(separator: " ", omittingEmptySubsequences: false)
            guard fields.count == 3, let bytes = UInt64(fields[1]) else {
                throw MTPStartupError.invalid("Compiled target manifest record is malformed.")
            }
            return TargetFile(path: String(fields[2]), sha256: String(fields[0]), bytes: bytes)
        })
    }

    static let targetManifestMaximumBytes = 1 << 20

    /// The manifest is held to the same open, type, owner, and permission
    /// rules as the files it pins. It must also be a single hard link so a
    /// second name cannot swap the bytes this launch agreed to check.
    static func targetManifest(path: String) throws -> [TargetFile] {
        guard path.hasPrefix("/") else { throw MTPStartupError.invalid("--target-manifest must be absolute.") }
        let descriptor = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
        guard descriptor >= 0 else { throw MTPStartupError.invalid("Cannot open target manifest regular file: \(path)") }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? handle.close() }
        var state = stat()
        guard fstat(descriptor, &state) == 0,
            state.st_mode & S_IFMT == S_IFREG,
            state.st_nlink == 1,
            state.st_uid == geteuid() || state.st_uid == 0,
            state.st_mode & 0o022 == 0,
            state.st_size > 0, state.st_size <= targetManifestMaximumBytes
        else { throw MTPStartupError.invalid("Target manifest type, owner, permissions, link count, or size is unsafe: \(path)") }
        let data = try handle.read(upToCount: targetManifestMaximumBytes + 1) ?? Data()
        guard data.count <= targetManifestMaximumBytes else {
            throw MTPStartupError.invalid("Target manifest exceeds \(targetManifestMaximumBytes) bytes.")
        }
        let files: [TargetFile]
        do { files = try JSONDecoder().decode([TargetFile].self, from: data) } catch {
            throw MTPStartupError.invalid("Target manifest is not a JSON list of {path, sha256, bytes}: \(error.localizedDescription)")
        }
        return try validatedTarget(files)
    }

    /// Names are single plain components so every pin and the unpinned-shard
    /// check address the same directory level.
    static func validatedTarget(_ files: [TargetFile]) throws -> [TargetFile] {
        let hex = Set("0123456789abcdef")
        var names = Set<String>()
        for file in files {
            guard !file.path.isEmpty, !file.path.hasPrefix("."), !file.path.contains("/"),
                file.path.utf8.count <= 255, !file.path.contains("\0"), names.insert(file.path).inserted
            else { throw MTPStartupError.invalid("Target manifest path is invalid or duplicated: \(file.path)") }
            guard file.sha256.count == 64, file.sha256.allSatisfy(hex.contains) else {
                throw MTPStartupError.invalid("Target manifest SHA-256 must be 64 lowercase hex digits: \(file.path)")
            }
        }
        guard files.contains(where: { $0.path.hasSuffix(".safetensors") }) else {
            throw MTPStartupError.invalid("Target manifest pins no safetensors weights.")
        }
        return files
    }

    /// The exact raw EigenLabs/Qwen3.8-27B-4bit@eda45ab snapshot when `files`
    /// is omitted. Nonweight operator metadata is allowed; unknown safetensors
    /// shards are refused.
    static func verifyTarget(directory: URL) throws {
        try verifyTarget(directory: directory, files: try compiledTarget())
    }

    /// Nonweight operator metadata is allowed; unknown safetensors shards are refused.
    static func verifyTarget(directory: URL, files: [TargetFile]) throws {
        try verifyDirectory(directory)
        let names = Set(files.map(\.path))
        let entries = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        guard !entries.contains(where: { $0.hasSuffix(".safetensors") && !names.contains($0) }) else {
            throw MTPStartupError.invalid("The Qwen3.8 backbone contains an unpinned safetensors shard.")
        }
        for file in files {
            try verifyFile(directory.appendingPathComponent(file.path), bytes: file.bytes, sha256: file.sha256)
        }
    }

    /// Other users cannot replace verified files through a writable parent.
    /// Root-owned sticky system temporary directories may occur above the
    /// protected snapshot, but the snapshot itself must never be writable.
    static func verifyDirectory(_ directory: URL) throws {
        let attributes = try FileManager.default.attributesOfItem(atPath: directory.path)
        guard attributes[.type] as? FileAttributeType == .typeDirectory else {
            throw MTPStartupError.invalid("Pinned snapshot directory must not be a symbolic link.")
        }
        guard let resolved = realpath(directory.path, nil) else {
            throw MTPStartupError.invalid("Cannot resolve pinned snapshot directory.")
        }
        var path = String(cString: resolved)
        free(resolved)
        var snapshot = true
        while true {
            var state = stat()
            guard lstat(path, &state) == 0, state.st_mode & S_IFMT == S_IFDIR,
                state.st_uid == geteuid() || state.st_uid == 0
            else { throw MTPStartupError.invalid("Pinned snapshot ancestor is not trusted: \(path)") }
            let writable = state.st_mode & 0o022 != 0
            let rootStickyAncestor = !snapshot && state.st_uid == 0 && state.st_mode & S_ISVTX != 0
            guard !writable || rootStickyAncestor else {
                throw MTPStartupError.invalid("Pinned snapshot directory is writable by another user: \(path)")
            }
            if path == "/" { break }
            path = (path as NSString).deletingLastPathComponent
            snapshot = false
        }
    }

    static let targetManifest = """
    748c964f9b7e5f2c3770ce013bbc0153be7c54311d9692c343b6188eefe77ac6 81 README.md
    c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041 8952 chat_template.jinja
    558cd605a6f1c16c73f4918534d122a943e12e754d38567b3b704acc93596965 4094 config.json
    e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e 202 generation_config.json
    075eac5fbba3951bc4870c1ac65d684c32e0abb24284201bd27f49df56735963 5328325648 model-00001-of-00003.safetensors
    6c99c446987a432beb1f6ef7d6fde6db02682f1ebe1a913760953140e0fa4e47 5354185130 model-00002-of-00003.safetensors
    0e267246064a1e635077dd05e22181f5eda44a1fa8a67dcf46c903f858bbe35b 4450532735 model-00003-of-00003.safetensors
    a2161f6a6c9f7c434145f97a4a4262a74eb3e16a95088e713754eb108b198511 189789 model.safetensors.index.json
    06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523 19989325 tokenizer.json
    95c557768e6b88a7128befc7bfd3c7de50e5d51af9b8b33a9f4dee0e04f99679 1161 tokenizer_config.json
    """
}
