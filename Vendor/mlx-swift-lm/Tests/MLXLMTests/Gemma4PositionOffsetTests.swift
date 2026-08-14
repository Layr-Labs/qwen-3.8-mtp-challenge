// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import Testing

@Suite("Gemma4.PositionOffset is public and constructible")
struct Gemma4PositionOffsetTests {

    @Test func scalarCase() {
        let offset = Gemma4.PositionOffset.scalar(42)
        if case .scalar(let v) = offset {
            #expect(v == 42)
        } else {
            Issue.record("expected .scalar case")
        }
    }

    @Test func batchCase() {
        let arr = MLXArray([Int32(0), 1, 2, 3])
        let offset = Gemma4.PositionOffset.batch(arr)
        if case .batch(let a) = offset {
            #expect(a.dim(0) == 4)
        } else {
            Issue.record("expected .batch case")
        }
    }
}
