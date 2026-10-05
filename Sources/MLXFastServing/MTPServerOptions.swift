import Foundation
import MLXLMServer

public struct MTPServerOptions: Sendable {
    public let command: MLXServerCLICommand
    public let headPath: String?
    public let maxDepth: Int
    public let targetManifestPath: String?

    public static func parse(arguments: [String], environment: [String: String] = [:]) throws -> Self {
        if arguments.contains("--help") || arguments.contains("-h") {
            return .init(command: .help, headPath: nil, maxDepth: 8, targetManifestPath: nil)
        }
        if arguments.contains("--list-routes") {
            return .init(command: .listRoutes, headPath: nil, maxDepth: 8, targetManifestPath: nil)
        }
        var standard = [arguments.first ?? "mlxfast-mtp-server"]
        var head: String?
        var depth: Int?
        var targetManifest: String?
        var native = false
        var engineCount = 0
        var index = 1
        func value(_ option: String) throws -> String {
            guard index + 1 < arguments.count else { throw MLXServerCLIError.missingValue(option) }
            index += 1
            return arguments[index]
        }
        while index < arguments.count {
            let option = arguments[index]
            switch option {
            case "--mtp-head":
                guard head == nil else { throw MTPStartupError.invalid("Duplicate --mtp-head.") }
                head = try value(option)
            case "--mtp-max-depth":
                guard depth == nil else { throw MTPStartupError.invalid("Duplicate --mtp-max-depth.") }
                let raw = try value(option)
                guard let parsed = Int(raw), (1...8).contains(parsed) else {
                    throw MTPStartupError.invalid("--mtp-max-depth must be 1...8.")
                }
                depth = parsed
            case "--target-manifest":
                guard targetManifest == nil else { throw MTPStartupError.invalid("Duplicate --target-manifest.") }
                targetManifest = try value(option)
            case "--engine-kind":
                engineCount += 1
                let kind = try value(option)
                native = native || kind == "native_mtp"
                standard += [option, kind == "native_mtp" ? "single_request" : kind]
            default:
                standard.append(option)
            }
            index += 1
        }
        guard native ? (head?.hasPrefix("/") == true && engineCount == 1) : (head == nil && depth == nil)
        else { throw MTPStartupError.invalid("native_mtp requires --engine-kind native_mtp and an absolute --mtp-head path.") }
        guard targetManifest == nil || native && targetManifest?.hasPrefix("/") == true else {
            throw MTPStartupError.invalid("--target-manifest requires native_mtp and an absolute path.")
        }
        let command = try MLXServerCLI.parse(arguments: standard, environment: environment)
        if native, case .run(let configuration) = command, configuration.embeddingModel != nil {
            throw MTPStartupError.invalid("native_mtp does not support --embedding-model in the same process.")
        }
        return .init(command: command, headPath: head, maxDepth: depth ?? 8, targetManifestPath: targetManifest)
    }
}

public enum MTPStartupError: Error, LocalizedError {
    case invalid(String)
    public var errorDescription: String? { if case .invalid(let message) = self { return message }; return nil }
}
