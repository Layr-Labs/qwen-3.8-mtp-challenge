import Foundation
@testable import MLXFastCore
import Testing

private func timedPromptWorkflow() throws -> String {
    try String(contentsOfFile: ".github/workflows/dflash-benchmark.yml", encoding: .utf8)
}

private func timedPromptStep(
    _ workflow: String,
    from startMarker: String,
    to endMarker: String
) throws -> String {
    let start = try #require(workflow.range(of: startMarker))
    let end = try #require(
        workflow.range(of: endMarker, range: start.upperBound..<workflow.endIndex)
    )
    return String(workflow[start.lowerBound..<end.lowerBound])
}



