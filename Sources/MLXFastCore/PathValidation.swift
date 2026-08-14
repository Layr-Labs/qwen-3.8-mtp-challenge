import Foundation

public func validateSafetensorsShardName(_ shardName: String, context: String) throws {
    guard !shardName.isEmpty else {
        throw MLXFastError.invalidInput("\(context) contains an empty shard name")
    }
    guard shardName.hasSuffix(".safetensors") else {
        throw MLXFastError.invalidInput(
            "\(context) maps tensors to unsupported shard \(shardName); expected safetensors"
        )
    }
    guard shardName != ".", shardName != "..",
          !shardName.contains("/"), !shardName.contains("\\") else {
        throw MLXFastError.invalidInput(
            "\(context) contains unsafe shard name \(shardName); expected a local safetensors filename"
        )
    }
}
