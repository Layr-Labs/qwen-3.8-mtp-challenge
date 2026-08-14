import CoreFoundation
import Darwin
import Foundation

public struct SafetensorInfo: Equatable {
    public let name: String
    public let dtype: String
    public let shape: [Int]
    public let dataStart: Int
    public let dataEnd: Int

    public var byteCount: Int {
        dataEnd - dataStart
    }

    public init(name: String, dtype: String, shape: [Int], dataStart: Int, dataEnd: Int) {
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.dataStart = dataStart
        self.dataEnd = dataEnd
    }
}

public struct SafetensorsHeader: Equatable {
    public let headerLength: Int
    public let metadata: [String: String]
    public let tensors: [String: SafetensorInfo]
    let sourceIdentity: SafetensorsFileIdentity?

    public var dataBaseOffset: UInt64 {
        UInt64(max(0, headerLength)) + 8
    }

    public init(
        headerLength: Int,
        metadata: [String: String],
        tensors: [String: SafetensorInfo]
    ) {
        self.headerLength = headerLength
        self.metadata = metadata
        self.tensors = tensors
        self.sourceIdentity = nil
    }

    init(
        headerLength: Int,
        metadata: [String: String],
        tensors: [String: SafetensorInfo],
        sourceIdentity: SafetensorsFileIdentity
    ) {
        self.headerLength = headerLength
        self.metadata = metadata
        self.tensors = tensors
        self.sourceIdentity = sourceIdentity
    }

    public static func == (lhs: SafetensorsHeader, rhs: SafetensorsHeader) -> Bool {
        lhs.headerLength == rhs.headerLength
            && lhs.metadata == rhs.metadata
            && lhs.tensors == rhs.tensors
    }
}

struct SafetensorsFileIdentity: Equatable {
    let device: UInt64
    let inode: UInt64
    let byteCount: Int
    let modificationSeconds: Int64
    let modificationNanoseconds: Int64
    let statusChangeSeconds: Int64
    let statusChangeNanoseconds: Int64
}

public enum Safetensors {
    // Matches the safetensors reference implementation's header limit. Check
    // this before allocating the declared header buffer.
    public static let maximumHeaderByteCount = 100_000_000

    public static func readHeader(_ path: URL) throws -> SafetensorsHeader {
        let handle = try FileHandle(forReadingFrom: path)
        defer {
            try? handle.close()
        }
        let sourceIdentity = try fileIdentity(for: handle, path: path.path)
        let fileByteCount = sourceIdentity.byteCount
        guard fileByteCount >= 8 else {
            throw MLXFastError.invalidInput("safetensors file is too small: \(path.path)")
        }

        let prefix = handle.readData(ofLength: 8)
        guard prefix.count == 8 else {
            throw MLXFastError.invalidInput("safetensors file is too small: \(path.path)")
        }
        let rawHeaderLength = prefix.withUnsafeBytes { raw -> UInt64 in
            raw.loadUnaligned(as: UInt64.self).littleEndian
        }
        guard rawHeaderLength > 0 else {
            throw MLXFastError.invalidInput("safetensors header is empty: \(path.path)")
        }
        guard rawHeaderLength <= UInt64(maximumHeaderByteCount) else {
            throw MLXFastError.invalidInput(
                "safetensors header exceeds \(maximumHeaderByteCount) bytes: \(path.path)"
            )
        }
        guard let headerLength = Int(exactly: rawHeaderLength) else {
            throw MLXFastError.invalidInput("safetensors header exceeds Int range: \(path.path)")
        }
        guard headerLength <= fileByteCount - 8 else {
            throw MLXFastError.invalidInput("truncated safetensors header: \(path.path)")
        }

        let headerData = handle.readData(ofLength: headerLength)
        guard headerData.count == headerLength else {
            throw MLXFastError.invalidInput("truncated safetensors header: \(path.path)")
        }

        let object: Any
        do {
            object = try JSONSerialization.jsonObject(with: headerData)
        } catch {
            throw MLXFastError.invalidInput("invalid safetensors header JSON in \(path.path)")
        }
        guard let dictionary = object as? [String: Any] else {
            throw MLXFastError.invalidInput("safetensors header must be a JSON object: \(path.path)")
        }

        let dataByteCount = fileByteCount - 8 - headerLength
        var metadata: [String: String] = [:]
        var tensors: [String: SafetensorInfo] = [:]
        for (name, value) in dictionary {
            if name == "__metadata__" {
                if let raw = value as? [String: String] {
                    metadata = raw
                }
                continue
            }
            guard let tensor = value as? [String: Any] else {
                continue
            }
            guard let dtype = tensor["dtype"] as? String else {
                throw MLXFastError.invalidInput("invalid tensor header for \(name) in \(path.path)")
            }
            let shape = try jsonIntegerArray(
                tensor["shape"],
                field: "shape",
                tensorName: name,
                path: path.path
            )
            let offsets = try jsonIntegerArray(
                tensor["data_offsets"],
                field: "data_offsets",
                tensorName: name,
                path: path.path
            )
            guard offsets.count == 2 else {
                throw MLXFastError.invalidInput("invalid tensor header for \(name) in \(path.path)")
            }
            guard offsets[0] >= 0, offsets[1] >= offsets[0] else {
                throw MLXFastError.invalidInput("invalid data_offsets for \(name) in \(path.path)")
            }
            guard offsets[1] <= dataByteCount else {
                throw MLXFastError.invalidInput(
                    "data_offsets for \(name) exceed safetensors file size in \(path.path)"
                )
            }
            tensors[name] = SafetensorInfo(
                name: name,
                dtype: dtype,
                shape: shape,
                dataStart: offsets[0],
                dataEnd: offsets[1]
            )
        }

        return SafetensorsHeader(
            headerLength: headerLength,
            metadata: metadata,
            tensors: tensors,
            sourceIdentity: sourceIdentity
        )
    }

    public static func validateSourceIdentity(
        _ path: URL,
        against header: SafetensorsHeader
    ) throws {
        guard let expectedIdentity = header.sourceIdentity else {
            throw MLXFastError.invalidInput(
                "validated safetensors header is not bound to \(path.lastPathComponent)"
            )
        }
        let handle = try FileHandle(forReadingFrom: path)
        defer {
            try? handle.close()
        }
        let currentIdentity = try fileIdentity(for: handle, path: path.path)
        guard currentIdentity == expectedIdentity else {
            throw MLXFastError.invalidInput(
                "safetensors source changed after its header was validated: \(path.path)"
            )
        }
    }

    public static func copySubset(
        from source: URL,
        to destination: URL,
        tensorNames: [String]
    ) throws -> Int {
        let header = try readHeader(source)
        return try copySubset(
            from: source,
            to: destination,
            tensorNames: tensorNames,
            validatedHeader: header
        )
    }

    public static func copySubset(
        from source: URL,
        to destination: URL,
        tensorNames: [String],
        validatedHeader header: SafetensorsHeader
    ) throws -> Int {
        try copySubset(
            from: source,
            to: destination,
            tensorNames: tensorNames,
            validatedHeader: header,
            beforeFinalSourceIdentityValidation: nil
        )
    }

    static func copySubset(
        from source: URL,
        to destination: URL,
        tensorNames: [String],
        validatedHeader header: SafetensorsHeader,
        beforeFinalSourceIdentityValidation: (() throws -> Void)?
    ) throws -> Int {
        let boundHeader: SafetensorsHeader
        if header.sourceIdentity == nil {
            let sourceHeader = try readHeader(source)
            guard sourceHeader == header else {
                throw MLXFastError.invalidInput(
                    "validated safetensors header does not match \(source.lastPathComponent)"
                )
            }
            boundHeader = sourceHeader
        } else {
            boundHeader = header
        }

        guard boundHeader.headerLength > 0,
              boundHeader.headerLength <= maximumHeaderByteCount
        else {
            throw MLXFastError.invalidInput(
                "invalid validated safetensors header length for \(source.lastPathComponent)"
            )
        }
        let (baseOffsetInt, baseOffsetOverflow) = boundHeader.headerLength.addingReportingOverflow(8)
        guard !baseOffsetOverflow,
              let baseOffset = UInt64(exactly: baseOffsetInt)
        else {
            throw MLXFastError.invalidInput(
                "validated safetensors header offset overflows for \(source.lastPathComponent)"
            )
        }
        guard let expectedSourceIdentity = boundHeader.sourceIdentity else {
            throw MLXFastError.invalidInput(
                "validated safetensors header is not bound to \(source.lastPathComponent)"
            )
        }
        let sourceByteCount = expectedSourceIdentity.byteCount
        guard baseOffsetInt <= sourceByteCount else {
            throw MLXFastError.invalidInput(
                "validated safetensors header exceeds file size for \(source.lastPathComponent)"
            )
        }
        let sourceDataByteCount = sourceByteCount - baseOffsetInt

        guard Set(tensorNames).count == tensorNames.count else {
            throw MLXFastError.invalidInput(
                "duplicate tensor name requested while copying \(source.lastPathComponent)"
            )
        }

        var selected: [SafetensorInfo] = []
        selected.reserveCapacity(tensorNames.count)
        for name in tensorNames {
            guard let tensor = boundHeader.tensors[name] else {
                throw MLXFastError.invalidInput(
                    "tensor \(name) requested from \(source.lastPathComponent) but missing from safetensors header"
                )
            }
            let (byteCount, byteCountOverflow) = tensor.dataEnd.subtractingReportingOverflow(
                tensor.dataStart
            )
            guard tensor.name == name,
                  !byteCountOverflow,
                  tensor.dataStart >= 0,
                  tensor.dataEnd >= tensor.dataStart,
                  tensor.dataEnd <= sourceDataByteCount,
                  byteCount >= 0
            else {
                throw MLXFastError.invalidInput(
                    "invalid validated safetensors tensor range for \(name)"
                )
            }
            selected.append(tensor)
        }
        selected.sort { $0.name < $1.name }
        guard !selected.isEmpty else {
            return 0
        }
        try validateCopyDestination(source: source, destination: destination)

        let input = try FileHandle(forReadingFrom: source)
        defer {
            try? input.close()
        }
        // Copied bytes are short-lived. Keeping a second copy of the ~21.6 GB
        // source in the unified buffer cache can exhaust 36 GiB machines, so
        // match the digest path's uncached read treatment (see #636).
        _ = Darwin.fcntl(input.fileDescriptor, F_NOCACHE, 1)
        _ = Darwin.fcntl(input.fileDescriptor, F_RDAHEAD, 0)
        let openedSourceIdentity = try fileIdentity(for: input, path: source.path)
        guard openedSourceIdentity == expectedSourceIdentity else {
            throw MLXFastError.invalidInput(
                "safetensors source changed after its header was validated: \(source.path)"
            )
        }

        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let temporaryDestination = destination.deletingLastPathComponent().appendingPathComponent(
            ".\(destination.lastPathComponent).mlxfast-copy-\(UUID().uuidString)"
        )
        var published = false
        defer {
            if !published {
                try? FileManager.default.removeItem(at: temporaryDestination)
            }
        }
        try Data().write(to: temporaryDestination, options: [.withoutOverwriting])
        let output = try FileHandle(forWritingTo: temporaryDestination)
        // The staged ~18 GB copy is synchronized and reopened later; keeping
        // its dirty pages out of the buffer cache avoids stacking a second
        // tree of cached pages next to a resident model.
        _ = Darwin.fcntl(output.fileDescriptor, F_NOCACHE, 1)

        do {
            let outputHeader = try makeHeaderData(
                tensors: selected,
                metadata: boundHeader.metadata
            )
            var headerLength = UInt64(outputHeader.count).littleEndian
            let prefix = Data(bytes: &headerLength, count: 8)
            try output.write(contentsOf: prefix)
            try output.write(contentsOf: outputHeader)

            for tensor in selected {
                guard let relativeOffset = UInt64(exactly: tensor.dataStart) else {
                    throw MLXFastError.invalidInput(
                        "negative safetensors tensor offset for \(tensor.name)"
                    )
                }
                let (absoluteOffset, overflow) = baseOffset.addingReportingOverflow(
                    relativeOffset
                )
                guard !overflow else {
                    throw MLXFastError.invalidInput(
                        "safetensors tensor offset overflows UInt64 for \(tensor.name)"
                    )
                }
                try copyBytes(
                    from: input,
                    to: output,
                    offset: absoluteOffset,
                    count: tensor.byteCount
                )
            }
            try output.synchronize()
            try output.close()
        } catch {
            try? output.close()
            throw error
        }

        try beforeFinalSourceIdentityValidation?()
        let finalSourceIdentity = try fileIdentity(for: input, path: source.path)
        guard finalSourceIdentity == openedSourceIdentity else {
            throw MLXFastError.invalidInput(
                "safetensors source changed while tensor bytes were copied: \(source.path)"
            )
        }
        try atomicRename(from: temporaryDestination, to: destination)
        published = true

        return selected.count
    }

    static func makeHeaderData(
        tensors: [SafetensorInfo],
        metadata: [String: String]
    ) throws -> Data {
        var object: [String: Any] = [:]
        if !metadata.isEmpty {
            object["__metadata__"] = metadata
        }

        var cursor = 0
        for tensor in tensors.sorted(by: { $0.name < $1.name }) {
            let (end, overflow) = cursor.addingReportingOverflow(tensor.byteCount)
            guard !overflow else {
                throw MLXFastError.invalidInput(
                    "safetensors output byte offsets overflow Int for \(tensor.name)"
                )
            }
            object[tensor.name] = [
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [cursor, end],
            ]
            cursor = end
        }

        var data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys]
        )
        while data.count % 8 != 0 {
            data.append(0x20)
        }
        return data
    }

    private static func validateCopyDestination(source: URL, destination: URL) throws {
        let canonicalSource = source.standardizedFileURL.resolvingSymlinksInPath()
        let canonicalDestination = destination.standardizedFileURL.resolvingSymlinksInPath()
        guard canonicalSource.path != canonicalDestination.path else {
            throw MLXFastError.invalidInput(
                "safetensors source and destination must be different files: \(source.path)"
            )
        }

        var destinationStatus = stat()
        let destinationResult = destination.path.withCString {
            Darwin.lstat($0, &destinationStatus)
        }
        if destinationResult != 0 {
            guard errno == ENOENT else {
                throw MLXFastError.invalidInput(
                    "could not inspect safetensors destination: \(destination.path)"
                )
            }
            return
        }

        guard (destinationStatus.st_mode & S_IFMT) == S_IFREG else {
            throw MLXFastError.invalidInput(
                "safetensors destination exists and is not a regular file: \(destination.path)"
            )
        }

        var sourceStatus = stat()
        let sourceResult = source.path.withCString {
            stat($0, &sourceStatus)
        }
        if sourceResult == 0,
           sourceStatus.st_dev == destinationStatus.st_dev,
           sourceStatus.st_ino == destinationStatus.st_ino
        {
            throw MLXFastError.invalidInput(
                "safetensors source and destination refer to the same file: \(source.path)"
            )
        }
    }

    private static func copyBytes(
        from input: FileHandle,
        to output: FileHandle,
        offset: UInt64,
        count: Int
    ) throws {
        try input.seek(toOffset: offset)
        var remaining = count
        let chunkSize = 8 * 1024 * 1024
        while remaining > 0 {
            // Drain each chunk's autoreleased buffer per iteration so RSS
            // stays bounded over multi-GB copies (same treatment as the #636
            // digest fix; readData(ofLength:) returns autoreleased storage).
            let copied = try autoreleasepool { () throws -> Int in
                let data = input.readData(ofLength: min(chunkSize, remaining))
                if data.isEmpty {
                    throw MLXFastError.invalidInput(
                        "unexpected EOF while copying safetensors tensor data"
                    )
                }
                try output.write(contentsOf: data)
                return data.count
            }
            remaining -= copied
        }
    }

    private static func jsonIntegerArray(
        _ value: Any?,
        field: String,
        tensorName: String,
        path: String
    ) throws -> [Int] {
        guard let values = value as? [Any] else {
            throw MLXFastError.invalidInput(
                "invalid \(field) for \(tensorName) in \(path)"
            )
        }
        return try values.map { value in
            guard let number = value as? NSNumber,
                  CFGetTypeID(number) != CFBooleanGetTypeID(),
                  !CFNumberIsFloatType(number),
                  let integer = Int(number.stringValue)
            else {
                throw MLXFastError.invalidInput(
                    "invalid \(field) for \(tensorName) in \(path)"
                )
            }
            return integer
        }
    }

    private static func fileIdentity(
        for handle: FileHandle,
        path: String
    ) throws -> SafetensorsFileIdentity {
        var status = stat()
        guard Darwin.fstat(handle.fileDescriptor, &status) == 0 else {
            throw MLXFastError.invalidInput("could not inspect safetensors file: \(path)")
        }
        guard (status.st_mode & S_IFMT) == S_IFREG else {
            throw MLXFastError.invalidInput("safetensors source is not a regular file: \(path)")
        }
        guard status.st_size >= 0, let byteCount = Int(exactly: status.st_size) else {
            throw MLXFastError.invalidInput("safetensors file size exceeds Int range: \(path)")
        }
        return SafetensorsFileIdentity(
            device: UInt64(status.st_dev),
            inode: UInt64(status.st_ino),
            byteCount: byteCount,
            modificationSeconds: Int64(status.st_mtimespec.tv_sec),
            modificationNanoseconds: Int64(status.st_mtimespec.tv_nsec),
            statusChangeSeconds: Int64(status.st_ctimespec.tv_sec),
            statusChangeNanoseconds: Int64(status.st_ctimespec.tv_nsec)
        )
    }

    private static func atomicRename(from source: URL, to destination: URL) throws {
        let result = source.path.withCString { sourcePath in
            destination.path.withCString { destinationPath in
                Darwin.rename(sourcePath, destinationPath)
            }
        }
        guard result == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
    }
}
