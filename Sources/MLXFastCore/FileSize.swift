import Foundation

public func fileSizeByteCount(from attributes: [FileAttributeKey: Any], path: String) throws -> Int {
    guard let size = attributes[.size] as? NSNumber else {
        throw MLXFastError.invalidInput("file size is unavailable: \(path)")
    }
    let bytes = size.int64Value
    guard bytes >= 0 else {
        throw MLXFastError.invalidInput("file size is negative: \(path)")
    }
    guard bytes <= Int64(Int.max) else {
        throw MLXFastError.invalidInput("file size exceeds Int range: \(path)")
    }
    return Int(bytes)
}
