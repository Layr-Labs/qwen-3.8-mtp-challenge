import CryptoKit
import Darwin
import Foundation
@testable import MLXFastServing
import Testing

struct MTPPinnedFilesTests {
    @Test("A FIFO without a writer is rejected without blocking startup")
    func fifoIsRejectedWithoutWriter() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("model.safetensors")
        try #require(mkfifo(path.path, 0o600) == 0)
        let finished = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            defer { finished.signal() }
            do {
                try MTPPinnedFiles.verifyFile(path, bytes: 0, sha256: String(repeating: "0", count: 64))
                Issue.record("A FIFO was accepted as pinned model data")
            } catch {}
        }
        let promptReturn = finished.wait(timeout: .now() + .milliseconds(500))
        if promptReturn == .timedOut {
            // Unblock the vulnerable implementation so the red run itself
            // remains bounded and does not leave a waiting thread behind.
            let writer = open(path.path, O_WRONLY | O_NONBLOCK | O_CLOEXEC)
            if writer >= 0 { close(writer) }
            #expect(finished.wait(timeout: .now() + .seconds(2)) == .success)
        }
        #expect(promptReturn == .success)
    }

    @Test("Content growth after the size check consumes at most one extra byte")
    func grownContentsAreBounded() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("model.safetensors")
        let grown = Data(repeating: 0x61, count: 2 << 20)
        try grown.write(to: path)
        let handle = try FileHandle(forReadingFrom: path)
        defer { try? handle.close() }
        let digest = SHA256.hash(data: grown).map { String(format: "%02x", $0) }.joined()
        // The real descriptor presents a larger body than the metadata snapshot.
        // Even its correct whole-body digest cannot authorize extra bytes.
        #expect(throws: (any Error).self) {
            try MTPPinnedFiles.verifyContents(handle, bytes: 4, sha256: digest)
        }
        #expect(try handle.offset() <= 5)
    }

    @Test("Content verification accepts exact bytes and rejects truncated data", arguments: [0, 4, (1 << 20) + 1])
    func exactAndTruncatedContents(bytes: Int) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("model.safetensors")
        let data = Data(repeating: 0x62, count: bytes)
        try data.write(to: path)
        let handle = try FileHandle(forReadingFrom: path)
        defer { try? handle.close() }
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        try MTPPinnedFiles.verifyContents(handle, bytes: UInt64(bytes), sha256: digest)
        #expect(try handle.offset() == UInt64(bytes))
        try handle.seek(toOffset: 0)
        #expect(throws: (any Error).self) {
            try MTPPinnedFiles.verifyContents(handle, bytes: UInt64(bytes + 1), sha256: digest)
        }
    }

    @Test("Pinned bytes reject mismatched digest, size, and a symbolic link")
    func invalidPinnedFiles() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let data = Data("pinned head fixture".utf8)
        let path = directory.appendingPathComponent("model.safetensors")
        try data.write(to: path)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        try MTPPinnedFiles.verifyFile(path, bytes: UInt64(data.count), sha256: digest)
        #expect(throws: (any Error).self) { try MTPPinnedFiles.verifyFile(path, bytes: UInt64(data.count + 1), sha256: digest) }
        #expect(throws: (any Error).self) { try MTPPinnedFiles.verifyFile(path, bytes: UInt64(data.count), sha256: String(repeating: "0", count: 64)) }
        let alias = directory.appendingPathComponent("alias.safetensors")
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: path)
        #expect(throws: (any Error).self) { try MTPPinnedFiles.verifyFile(alias, bytes: UInt64(data.count), sha256: digest) }
    }

    @Test("A correctly hashed file writable by another user is refused")
    func writablePinIsRejected() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let data = Data("pinned".utf8)
        let path = directory.appendingPathComponent("model.safetensors")
        try data.write(to: path)
        try FileManager.default.setAttributes([.posixPermissions: 0o666], ofItemAtPath: path.path)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        #expect(throws: (any Error).self) { try MTPPinnedFiles.verifyFile(path, bytes: UInt64(data.count), sha256: digest) }
    }

    @Test("Compiled target pin matches all ten frozen published snapshot records")
    func targetManifestMatchesFrozenFixture() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let fixture = try String(contentsOf: root.appendingPathComponent("fixtures/reference_qwen3_8_27b_4bit.sha256"), encoding: .utf8)
        let expected = fixture.split(separator: "\n").filter { !$0.hasPrefix("#") }.map(String.init)
        #expect(MTPPinnedFiles.targetManifest.split(separator: "\n").map(String.init) == expected)
        #expect(expected.count == 10)
    }

    @Test("The compiled default target parses to the ten frozen records")
    func compiledTargetRecords() throws {
        let files = try MTPPinnedFiles.compiledTarget()
        #expect(files.count == 10)
        #expect(files.first == .init(path: "README.md", sha256: "748c964f9b7e5f2c3770ce013bbc0153be7c54311d9692c343b6188eefe77ac6", bytes: 81))
        #expect(files.filter { $0.path.hasSuffix(".safetensors") }.count == 3)
    }

    @Test("A target manifest file selects exactly its verified snapshot files")
    func targetManifestFileVerifiesSnapshot() throws {
        let root = try privateDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let snapshot = root.appendingPathComponent("snapshot")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: false)
        let config = Data("{}".utf8), weights = Data("weights".utf8)
        try config.write(to: snapshot.appendingPathComponent("config.json"))
        try weights.write(to: snapshot.appendingPathComponent("model.safetensors"))
        try Data("operator note".utf8).write(to: snapshot.appendingPathComponent("NOTES.md"))
        let manifest = root.appendingPathComponent("target-manifest.json")
        try manifestJSON([("config.json", sha256(config), config.count),
                          ("model.safetensors", sha256(weights), weights.count)]).write(to: manifest)
        let files = try MTPPinnedFiles.targetManifest(path: manifest.path)
        #expect(files == [.init(path: "config.json", sha256: sha256(config), bytes: 2),
                          .init(path: "model.safetensors", sha256: sha256(weights), bytes: 7)])
        try MTPPinnedFiles.verifyTarget(directory: snapshot, files: files)
    }

    @Test("A manifest-selected target fails closed on a missing, changed, or unpinned file",
          arguments: ["missing", "changed", "resized", "unpinned_shard", "symlink"])
    func targetManifestFailsClosed(kind: String) throws {
        let root = try privateDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let snapshot = root.appendingPathComponent("snapshot")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: false)
        let weights = Data("weights".utf8)
        let shard = snapshot.appendingPathComponent("model.safetensors")
        try weights.write(to: shard)
        let files = [MTPPinnedFiles.TargetFile(path: "model.safetensors", sha256: sha256(weights), bytes: 7)]
        try MTPPinnedFiles.verifyTarget(directory: snapshot, files: files)
        switch kind {
        case "missing": try FileManager.default.removeItem(at: shard)
        case "changed": try Data("weighTs".utf8).write(to: shard)
        case "resized": try Data("weights!".utf8).write(to: shard)
        case "unpinned_shard": try weights.write(to: snapshot.appendingPathComponent("extra.safetensors"))
        default:
            let outside = root.appendingPathComponent("outside.safetensors")
            try FileManager.default.moveItem(at: shard, to: outside)
            try FileManager.default.createSymbolicLink(at: shard, withDestinationURL: outside)
        }
        #expect(throws: (any Error).self) { try MTPPinnedFiles.verifyTarget(directory: snapshot, files: files) }
    }

    @Test("Malformed target manifests are refused",
          arguments: [
            "{}",
            "[]",
            "[{\"path\":\"config.json\",\"sha256\":\"\(String(repeating: "a", count: 64))\",\"bytes\":2}]",
            "[{\"path\":\"model.safetensors\",\"sha256\":\"\(String(repeating: "a", count: 64))\",\"bytes\":2,\"extra\":1}]",
            "[{\"path\":\"model.safetensors\",\"sha256\":\"\(String(repeating: "a", count: 64))\"}]",
            "[{\"path\":\"model.safetensors\",\"sha256\":\"\(String(repeating: "A", count: 64))\",\"bytes\":2}]",
            "[{\"path\":\"model.safetensors\",\"sha256\":\"\(String(repeating: "a", count: 63))\",\"bytes\":2}]",
            "[{\"path\":\"model.safetensors\",\"sha256\":\"\(String(repeating: "a", count: 64))\",\"bytes\":-1}]",
            "[{\"path\":\"model.safetensors\",\"sha256\":\"\(String(repeating: "a", count: 64))\",\"bytes\":1.5}]",
            "[{\"path\":\"sub/model.safetensors\",\"sha256\":\"\(String(repeating: "a", count: 64))\",\"bytes\":2}]",
            "[{\"path\":\"../model.safetensors\",\"sha256\":\"\(String(repeating: "a", count: 64))\",\"bytes\":2}]",
            "[{\"path\":\".hidden.safetensors\",\"sha256\":\"\(String(repeating: "a", count: 64))\",\"bytes\":2}]",
            "[{\"path\":\"\",\"sha256\":\"\(String(repeating: "a", count: 64))\",\"bytes\":2}]",
            "[{\"path\":\"model.safetensors\",\"sha256\":\"\(String(repeating: "a", count: 64))\",\"bytes\":2},{\"path\":\"model.safetensors\",\"sha256\":\"\(String(repeating: "b", count: 64))\",\"bytes\":3}]",
            "not json",
          ])
    func malformedTargetManifest(contents: String) throws {
        let root = try privateDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let manifest = root.appendingPathComponent("target-manifest.json")
        try Data(contents.utf8).write(to: manifest)
        #expect(throws: (any Error).self) { try MTPPinnedFiles.targetManifest(path: manifest.path) }
    }

    @Test("An unsafe target manifest file is refused",
          arguments: ["symlink", "hardlink", "writable", "fifo", "oversized", "relative", "missing"])
    func unsafeTargetManifestFile(kind: String) throws {
        let root = try privateDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let valid = try manifestJSON([("model.safetensors", String(repeating: "a", count: 64), 2)])
        let manifest = root.appendingPathComponent("target-manifest.json")
        var path = manifest.path
        switch kind {
        case "symlink":
            let real = root.appendingPathComponent("real.json")
            try valid.write(to: real)
            try FileManager.default.createSymbolicLink(at: manifest, withDestinationURL: real)
        case "hardlink":
            let real = root.appendingPathComponent("real.json")
            try valid.write(to: real)
            try #require(link(real.path, manifest.path) == 0)
        case "writable":
            try valid.write(to: manifest)
            try FileManager.default.setAttributes([.posixPermissions: 0o666], ofItemAtPath: manifest.path)
        case "fifo":
            try #require(mkfifo(manifest.path, 0o600) == 0)
        case "oversized":
            try Data(repeating: 0x20, count: (1 << 20) + 1).write(to: manifest)
        case "relative":
            try valid.write(to: manifest)
            path = "target-manifest.json"
        default: break
        }
        #expect(throws: (any Error).self) { try MTPPinnedFiles.targetManifest(path: path) }
    }

    @Test("A writable snapshot directory is refused before loading")
    func unsafeDirectoryIsRejected() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        try MTPPinnedFiles.verifyDirectory(directory)
        try FileManager.default.setAttributes([.posixPermissions: 0o777], ofItemAtPath: directory.path)
        #expect(throws: (any Error).self) { try MTPPinnedFiles.verifyDirectory(directory) }
    }

    private func privateDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        return directory
    }

    private func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func manifestJSON(_ records: [(String, String, Int)]) throws -> Data {
        try JSONSerialization.data(withJSONObject: records.map { ["path": $0.0, "sha256": $0.1, "bytes": $0.2] })
    }
}
