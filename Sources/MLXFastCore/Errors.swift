import Foundation

public enum MLXFastError: Error, CustomStringConvertible {
    case missingFile(String)
    case invalidInput(String)

    public var description: String {
        switch self {
        case .missingFile(let message):
            return message
        case .invalidInput(let message):
            return message
        }
    }
}

public func requireFile(_ path: String, description: String) throws {
    if !FileManager.default.fileExists(atPath: path) {
        throw MLXFastError.missingFile("\(description) not found at \(path)")
    }
}
