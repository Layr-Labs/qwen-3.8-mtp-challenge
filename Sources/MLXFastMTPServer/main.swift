import MLXFastServing
import Foundation

do {
    try await MTPServer.run()
} catch {
    FileHandle.standardError.write(Data("mlxfast-mtp-server: \(error.localizedDescription)\n".utf8))
    Foundation.exit(1)
}
