// Copyright © 2026 Apple Inc.

import Darwin
import Foundation

struct MLXBench {}

do {
    let args = try BenchArguments.parse()
    guard let mode = args.mode else {
        BenchArguments.printUsage()
        exit(1)
    }

    switch mode {
    case .mtp:
        throw CLIError("mtp benchmark was removed from this vendored DFlash build")
    case .dflash:
        try await MLXBench.runDFlashBenchmark(args: args, mode: mode)
    case .gatherQMV:
        try MLXBench.runGatherQMVBenchmark(args: args, mode: mode)
    }
} catch {
    eprint("error: \(error)")
    exit(1)
}
