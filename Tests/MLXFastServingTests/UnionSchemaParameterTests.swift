import Foundation
import MLXLMCommon
import Testing

struct UnionSchemaParameterTests {
    private func parser() -> XMLFunctionParser {
        XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
    }

    private func tools(
        _ properties: [String: any Sendable]
    ) -> [[String: any Sendable]] {
        [[
            "function": [
                "name": "write",
                "parameters": [
                    "properties": properties
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]]
    }

    private func argument(
        _ xml: String, tools: [[String: any Sendable]]?, name: String = "offset"
    ) throws -> JSONValue {
        let call = try #require(parser().parse(content: xml, tools: tools))
        return try #require(call.function.arguments[name])
    }

    @Test("A live integer-or-null type array coerces, and invalid text stays a string")
    func integerOrNullArrayCoerces() throws {
        let typeList: [any Sendable] = ["integer", "null"]
        let schemas = tools([
            "offset": ["type": typeList] as [String: any Sendable]
        ])
        #expect(try argument(
            "<function=write><parameter=offset>4</parameter></function>", tools: schemas
        ) == .int(4))
        #expect(try argument(
            "<function=write><parameter=offset>null</parameter></function>", tools: schemas
        ) == .null)
        #expect(try argument(
            "<function=write><parameter=offset> none </parameter></function>", tools: schemas
        ) == .null)
        #expect(try argument(
            "<function=write><parameter=offset>one</parameter></function>", tools: schemas
        ) == .string("one"))
    }

    @Test("anyOf integer-or-null uses the same coercion")
    func anyOfIntegerOrNullCoerces() throws {
        let choices: [any Sendable] = [
            ["type": "integer"] as [String: any Sendable],
            ["type": "null"] as [String: any Sendable],
        ]
        let schemas = tools([
            "offset": ["anyOf": choices] as [String: any Sendable]
        ])
        #expect(try argument(
            "<function=write><parameter=offset>8</parameter></function>", tools: schemas
        ) == .int(8))
        #expect(try argument(
            "<function=write><parameter=offset>nil</parameter></function>", tools: schemas
        ) == .null)
    }

    @Test("A string in the union keeps the raw text, including null")
    func stringUnionStaysText() throws {
        let typeList: [any Sendable] = ["string", "null"]
        let schemas = tools([
            "offset": ["type": typeList] as [String: any Sendable]
        ])
        #expect(try argument(
            "<function=write><parameter=offset>null</parameter></function>", tools: schemas
        ) == .string("null"))
        #expect(try argument(
            "<function=write><parameter=offset>4</parameter></function>", tools: schemas
        ) == .string("4"))
    }

    @Test("Two concrete types and a missing schema stay strings")
    func ambiguousOrMissingStayStrings() throws {
        let choices: [any Sendable] = [
            ["type": "integer"] as [String: any Sendable],
            ["type": "boolean"] as [String: any Sendable],
        ]
        let schemas = tools([
            "offset": ["oneOf": choices] as [String: any Sendable]
        ])
        #expect(try argument(
            "<function=write><parameter=offset>1</parameter></function>", tools: schemas
        ) == .string("1"))
        #expect(try argument(
            "<function=write><parameter=offset>4</parameter></function>", tools: nil
        ) == .string("4"))
    }

    @Test("A single string type still ignores union handling")
    func singleStringTypeStaysOnTheExistingPath() throws {
        let schemas = tools([
            "offset": ["type": "integer"] as [String: any Sendable],
            "path": ["type": "string"] as [String: any Sendable],
        ])
        #expect(try argument(
            "<function=write><parameter=offset>25</parameter></function>", tools: schemas
        ) == .int(25))
        #expect(try argument(
            "<function=write><parameter=offset>one</parameter></function>", tools: schemas
        ) == .string("one"))
        #expect(try argument(
            "<function=write><parameter=path>null</parameter></function>",
            tools: schemas, name: "path"
        ) == .string("null"))
    }
}
