// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLXLMCommon
import Testing

/// Unit tests for `decodeToolCallArguments` — the helper that normalizes an
/// OpenAI tool-call `arguments` JSON string into a chat-template mapping.
///
/// Issue #249: when `arguments` is forwarded as a raw `String`, Gemma's
/// `chat_template.jinja` takes the `is string` branch and emits a corrupting
/// double brace `call:run_terminal{{"command":"ls -la"}}`. Decoding to an object
/// makes the template take the `is mapping` branch (`command:<|"|>ls -la<|"|>`).
struct DecodeToolCallArgumentsTests {
    @Test("JSON object string decodes into a mapping (not a String)")
    func decodesJSONObjectIntoMapping() throws {
        let decoded = decodeToolCallArguments(#"{"command":"ls -la"}"#)
        let mapping = try #require(decoded as? [String: any Sendable])
        #expect(mapping["command"] as? String == "ls -la")
        // The `is string` branch only fires for a Swift String; assert the fix
        // no longer hands the template one.
        #expect(!(decoded is String))
    }

    @Test("Nested objects, arrays, numbers and bools survive the round trip")
    func decodesNestedStructures() throws {
        let raw = #"{"path":"/tmp","limit":5,"deep":{"k":"v"},"flags":[true,false]}"#
        let mapping = try #require(decodeToolCallArguments(raw) as? [String: any Sendable])
        #expect(mapping["path"] as? String == "/tmp")
        #expect((mapping["limit"] as? NSNumber)?.intValue == 5)
        let deep = try #require(mapping["deep"] as? [String: any Sendable])
        #expect(deep["k"] as? String == "v")
        let flags = try #require(mapping["flags"] as? [any Sendable])
        #expect(flags.count == 2)
        #expect((flags.first as? NSNumber)?.boolValue == true)
    }

    @Test("Empty object decodes to an empty mapping")
    func decodesEmptyObject() throws {
        let mapping = try #require(decodeToolCallArguments("{}") as? [String: any Sendable])
        #expect(mapping.isEmpty)
    }

    @Test("Non-JSON arguments fall back to the raw string unchanged")
    func nonJSONFallsBackToString() {
        #expect(decodeToolCallArguments("not json") as? String == "not json")
        #expect(decodeToolCallArguments("") as? String == "")
    }

    @Test("JSON non-objects (arrays, scalars) fall back to the raw string")
    func nonObjectJSONFallsBackToString() {
        // A top-level array/scalar is valid JSON but not an object. The OpenAI
        // arguments contract is always an object, so keep the original wire
        // shape rather than reshaping it.
        #expect(decodeToolCallArguments("[1,2,3]") as? String == "[1,2,3]")
        #expect(decodeToolCallArguments("\"plain\"") as? String == "\"plain\"")
    }
}
