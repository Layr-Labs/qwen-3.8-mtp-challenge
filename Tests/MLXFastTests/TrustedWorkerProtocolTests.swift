import Foundation
@testable import MLXFastCore
@testable import MLXFastHarness
import Testing

@Test
func trustedPreflightUsesOneShotWorkerProtocolAndPropagatesFailure() throws {
    let root = try trustedWorkerTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let argumentLog = root.appendingPathComponent("arguments.txt")
    let successWorker = root.appendingPathComponent("success-worker")
    try writeExecutable(
        """
        #!/bin/bash
        printf '%s\n' "$*" > \(shellQuote(argumentLog.path))
        printf '%s\n' '{"ok":true}'
        """,
        to: successWorker
    )
    try QwenRuntime.runPreflightWithWorker(
        weightsPath: "/tmp/weights",
        worker: RuntimeWorkerOptions(
            executablePath: successWorker.path,
            requestTimeoutSeconds: 5
        )
    )
    let arguments = try String(
        contentsOf: argumentLog,
        encoding: .utf8
    )
    #expect(arguments.contains("preflight --weights /tmp/weights"))

    let failingWorker = root.appendingPathComponent("failing-worker")
    try writeExecutable(
        """
        #!/bin/bash
        printf '%s\n' '{"ok":false,"error":"invalid tensor metadata"}'
        exit 1
        """,
        to: failingWorker
    )
    #expect(throws: MLXFastError.self) {
        try QwenRuntime.runPreflightWithWorker(
            weightsPath: "/tmp/weights",
            worker: RuntimeWorkerOptions(
                executablePath: failingWorker.path,
                requestTimeoutSeconds: 5
            )
        )
    }
}

@Test
func trustedTracePreservesLargeTopKAndOutOfSubsetExpectedDiagnostics() throws {
    let root = try trustedWorkerTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let requestLog = root.appendingPathComponent("request.json")
    let worker = root.appendingPathComponent("trace-worker")
    let topLogits = (0..<12).map {
        #"{"token":\#($0),"logit":\#(100 - $0)}"#
    }.joined(separator: ",")
    try writeExecutable(
        """
        #!/bin/bash
        printf '%s\n' '{"id":0,"nonce":"trace-nonce","ok":true}'
        IFS= read -r request
        printf '%s\n' "$request" > \(shellQuote(requestLog.path))
        printf '%s\n' '{"id":1,"nonce":"trace-nonce","ok":true,"token":0,"top_logits":[\(topLogits)],"expected_token_logit":1,"expected_token_rank":20,"top_logit_margin":1}'
        """,
        to: worker
    )

    let prompt = Array(
        repeating: 1,
        count: MLXFastConstants.correctnessPromptTokens
    )
    var expected = Array(
        repeating: 19,
        count: MLXFastConstants.correctnessSteps
    )
    expected[0] = 19
    let golden = root.appendingPathComponent("golden.json")
    try """
    {
      "version": 1,
      "cases": [
        {
          "name": "trace",
          "prompt_tokens": \(jsonArray(prompt)),
          "expected_tokens": \(jsonArray(expected))
        }
      ]
    }
    """.write(to: golden, atomically: true, encoding: .utf8)

    let report = try QwenRuntime.traceCorrectness(
        CorrectnessTraceOptions(
            weightsPath: "/tmp/weights",
            goldenPath: golden.path,
            caseName: "trace",
            step: 0,
            topK: 12
        ),
        worker: RuntimeWorkerOptions(
            executablePath: worker.path,
            requestTimeoutSeconds: 5
        )
    )
    #expect(report.topLogits.count == 12)
    #expect(report.expectedToken == 19)
    #expect(report.expectedTokenLogit == 1)
    #expect(report.expectedTokenRank == 20)
    #expect(!report.topLogits.contains { $0.token == 19 })

    let request = try String(
        contentsOf: requestLog,
        encoding: .utf8
    )
    #expect(request.contains(#""top_k":12"#))
    #expect(request.contains(#""expected_token":19"#))
}

@Test
func trustedSandboxRebindsExecPermissionToTheRuntimeWorker() throws {
    let sandboxExecutable = "/usr/bin/sandbox-exec"
    guard FileManager.default.isExecutableFile(atPath: sandboxExecutable) else {
        return
    }
    let root = try trustedWorkerTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }

    let staleExecutable = "/usr/bin/false"
    let runtimeWorkerExecutable = "/usr/bin/true"
    let staleProfile = root.appendingPathComponent("stale-worker.sb")
    try """
    (version 1)
    (allow default)
    (deny network*)
    (deny process-fork)
    (deny process-exec*)
    (allow process-exec (literal "\(staleExecutable)"))
    """.write(to: staleProfile, atomically: true, encoding: .utf8)

    let reboundProfilePath = try runtimeWorkerSandboxProfile(
        rebinding: staleProfile.path,
        toExecutableAt: runtimeWorkerExecutable
    )
    defer { try? FileManager.default.removeItem(atPath: reboundProfilePath) }
    let reboundProfile = try String(
        contentsOfFile: reboundProfilePath,
        encoding: .utf8
    )
    #expect(reboundProfile.contains("(deny network*)"))
    #expect(!reboundProfile.contains(
        "(allow process-exec (literal \"\(staleExecutable)\"))"
    ))
    #expect(reboundProfile.hasSuffix(
        """
        (deny process-exec*)
        (allow process-exec (literal "\(runtimeWorkerExecutable)"))
        """
    ))

    let allowed = Process()
    allowed.executableURL = URL(fileURLWithPath: sandboxExecutable)
    allowed.arguments = ["-f", reboundProfilePath, runtimeWorkerExecutable]
    allowed.standardError = Pipe()
    try allowed.run()
    allowed.waitUntilExit()
    #expect(allowed.terminationStatus == 0)

    let denied = Process()
    denied.executableURL = URL(fileURLWithPath: sandboxExecutable)
    denied.arguments = ["-f", reboundProfilePath, staleExecutable]
    denied.standardError = Pipe()
    try denied.run()
    denied.waitUntilExit()
    // sandbox-exec returns EX_OSERR (71) when its profile rejects exec. If the
    // stale allow still won, /usr/bin/false would run and return 1 instead.
    #expect(denied.terminationStatus == 71)
}

private func trustedWorkerTemporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(
        at: url,
        withIntermediateDirectories: true
    )
    return url
}

private func writeExecutable(_ contents: String, to url: URL) throws {
    try contents.write(to: url, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: url.path
    )
}

private func shellQuote(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
}

private func jsonArray(_ values: [Int]) -> String {
    "[\(values.map(String.init).joined(separator: ","))]"
}
