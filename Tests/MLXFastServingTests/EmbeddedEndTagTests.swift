import Foundation
import MLXLMCommon
import Testing

struct EmbeddedEndTagTests {
    private func feed(_ processor: ToolCallProcessor, _ text: String) {
        var rest = text
        while !rest.isEmpty {
            let end = rest.index(rest.startIndex, offsetBy: min(7, rest.count), limitedBy: rest.endIndex)
                ?? rest.endIndex
            _ = processor.processChunk(String(rest[..<end]))
            rest = String(rest[end...])
        }
    }

    @Test("A tool-call end tag inside a parameter stays in the value")
    func endTagInsideParameterKeepsTheCall() throws {
        let processor = ToolCallProcessor(format: .xmlFunction)
        feed(processor, """
            <tool_call>
            <function=write>
            <parameter=content>
            <div></tool_call>
            </parameter>
            </function>
            </tool_call>
            """)
        #expect(processor.toolCalls.count == 1)
        let call = try #require(processor.toolCalls.first)
        #expect(call.function.name == "write")
        #expect(call.function.arguments["content"] == .string("<div></tool_call>"))
    }

    @Test("A parameter end tag inside a value stays in that value")
    func parameterEndTagInsideValueKeepsLaterParameters() throws {
        let parser = XMLFunctionParser(startTag: "<tool_call>", endTag: "</tool_call>")
        let call = try #require(parser.parse(
            content: """
                <function=write>
                <parameter=content>
                see </parameter> in the file
                </parameter>
                <parameter=path>
                notes.txt
                </parameter>
                </function>
                """,
            tools: nil))
        #expect(call.function.arguments["content"] == .string("see </parameter> in the file"))
        #expect(call.function.arguments["path"] == .string("notes.txt"))
    }

    @Test("Two Qwen wrappers still produce two calls")
    func twoWrappersKeepBothArgumentMaps() throws {
        let processor = ToolCallProcessor(format: .xmlFunction)
        feed(processor, """
            <tool_call>
            <function=read>
            <parameter=path>
            a.txt
            </parameter>
            </function>
            </tool_call>
            <tool_call>
            <function=read>
            <parameter=path>
            b.txt
            </parameter>
            </function>
            </tool_call>
            """)
        #expect(processor.toolCalls.count == 2)
        #expect(processor.toolCalls.map(\.function.arguments["path"]) == [.string("a.txt"), .string("b.txt")])
    }

    @Test("Integer conversion still runs on a call the scanner kept whole")
    func integerConversionSurvivesTheScanner() throws {
        let tools: [[String: any Sendable]] = [[
            "function": [
                "name": "read",
                "parameters": [
                    "properties": [
                        "offset": ["type": "integer"] as [String: any Sendable]
                    ]
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]]
        let processor = ToolCallProcessor(format: .xmlFunction, tools: tools)
        feed(processor, """
            <tool_call>
            <function=read>
            <parameter=offset>
            25
            </parameter>
            </function>
            </tool_call>
            """)
        let call = try #require(processor.toolCalls.first)
        #expect(call.function.arguments["offset"] == .int(25))
    }
}
