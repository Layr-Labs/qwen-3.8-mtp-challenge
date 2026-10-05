// Copyright © 2025 Apple Inc.

import Foundation

// MARK: - JSON to Sendable Bridge

/// Convert a JSON-deserialized value to `any Sendable`.
///
/// `JSONSerialization` returns `Any`, but all JSON types it produces
/// (String, NSNumber, NSNull, Array, Dictionary) are Sendable.
func asSendable(_ value: Any) -> any Sendable {
    switch value {
    case let s as String: return s
    case let n as NSNumber: return n
    case let a as [Any]: return a.map(asSendable)
    case let d as [String: Any]: return d.mapValues(asSendable)
    case let null as NSNull: return null
    default: return "\(value)"
    }
}

/// Deserialize JSON data, returning a Sendable value.
func deserializeJSON(_ data: Data) -> (any Sendable)? {
    guard let object = try? JSONSerialization.jsonObject(with: data) else { return nil }
    return asSendable(object)
}

/// Normalize an OpenAI tool-call `arguments` payload for chat-template rendering.
///
/// The OpenAI wire format delivers `function.arguments` as a JSON-encoded
/// *string* (e.g. `#"{"command":"ls -la"}"#`). Gemma's `chat_template.jinja`
/// opens its own `{ … }` block around the call and, on the `is string` branch,
/// dumps that string verbatim — producing a malformed double brace
/// `call:run_terminal{{"command":"ls -la"}}`. The model then imitates that
/// shape on the next turn and the output parser shreds it (splitting the inner
/// object at its first `:`), corrupting `function.arguments`.
///
/// Decoding the string into a `[String: any Sendable]` object makes the
/// template take the `is mapping` branch and emit valid `command:<|"|>ls -la<|"|>`
/// pairs instead. Returns the decoded object when `raw` is a JSON object;
/// otherwise returns `raw` unchanged so non-JSON arguments (and JSON
/// non-objects) keep their original shape. This only normalizes the
/// *template-input* shape — the OpenAI request/response contract still carries
/// `arguments` as a `String`.
public func decodeToolCallArguments(_ raw: String) -> any Sendable {
    guard let data = raw.data(using: .utf8),
        let decoded = deserializeJSON(data),
        let object = decoded as? [String: any Sendable]
    else {
        return raw
    }
    return object
}

// MARK: - Basic Deserialization

/// Deserialize a string value to JSON or return as string.
///
/// Attempts JSON parsing first, falling back to the original string value.
/// Reference: Python's `ast.literal_eval` / `json.loads` pattern
func tryParseJSON(_ value: String) -> (any Sendable)? {
    guard let data = value.data(using: .utf8) else { return nil }
    return deserializeJSON(data)
}

func deserialize(_ value: String) -> any Sendable {
    tryParseJSON(value) ?? value
}

// MARK: - Schema Lookup Functions

/// Check if a parameter is a string type in the tool schema.
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/glm47.py
func isStringType(funcName: String, argName: String, tools: [[String: any Sendable]]?) -> Bool {
    guard let type = getParameterType(funcName: funcName, paramName: argName, tools: tools) else {
        return false
    }
    return type == "string"
}

/// Get the parameter type from tool schema for a specific function and parameter.
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/qwen3_coder.py
func getParameterType(
    funcName: String, paramName: String, tools: [[String: any Sendable]]?
) -> String? {
    guard let tools else { return nil }
    for tool in tools {
        guard let function = tool["function"] as? [String: any Sendable],
            function["name"] as? String == funcName,
            let parameters = function["parameters"] as? [String: any Sendable],
            let properties = parameters["properties"] as? [String: any Sendable],
            let param = properties[paramName] as? [String: any Sendable],
            let type = param["type"] as? String
        else { continue }
        return type
    }
    return nil
}

/// Get parameter configuration for a function from tools schema.
func getParameterConfig(
    funcName: String, tools: [[String: any Sendable]]?
) -> [String: any Sendable] {
    guard let tools else { return [:] }
    for tool in tools {
        guard let function = tool["function"] as? [String: any Sendable],
            function["name"] as? String == funcName,
            let parameters = function["parameters"] as? [String: any Sendable],
            let properties = parameters["properties"] as? [String: any Sendable]
        else { continue }
        return properties
    }
    return [:]
}

/// Parameter object for one function argument.
///
/// `getParameterType` only sees a `type` string. JSON schemas decoded through
/// `JSONValue.sendableValue` store a type array or `anyOf` as `[any Sendable]`,
/// which that cast misses. This returns the property dictionary either way.
func parameterSchema(
    funcName: String, paramName: String, tools: [[String: any Sendable]]?
) -> [String: any Sendable]? {
    guard let tools else { return nil }
    for tool in tools {
        guard let function = tool["function"] as? [String: any Sendable],
            function["name"] as? String == funcName,
            let parameters = function["parameters"] as? [String: any Sendable],
            let properties = parameters["properties"] as? [String: any Sendable],
            let param = properties[paramName] as? [String: any Sendable]
        else { continue }
        return param
    }
    return nil
}

/// Type names declared by `type`, including an array, plus `anyOf` / `oneOf` / `allOf`.
///
/// A `[String]` and an `[any Sendable]` of strings are both accepted. Live tool
/// specs use the second shape, and a Swift dictionary literal often uses the first.
func schemaTypeNames(_ schema: [String: any Sendable]) -> Set<String> {
    var types: Set<String> = []
    if let name = schema["type"] as? String {
        types.insert(name.lowercased())
    } else if let names = schema["type"] as? [String] {
        types.formUnion(names.map { $0.lowercased() })
    } else if let names = schema["type"] as? [any Sendable] {
        for item in names {
            if let name = item as? String {
                types.insert(name.lowercased())
            }
        }
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        let choices: [any Sendable]
        if let typed = schema[key] as? [[String: any Sendable]] {
            choices = typed.map { $0 as any Sendable }
        } else if let untyped = schema[key] as? [any Sendable] {
            choices = untyped
        } else {
            continue
        }
        for choice in choices {
            if let child = choice as? [String: any Sendable] {
                types.formUnion(schemaTypeNames(child))
            }
        }
    }
    return types
}

func declaredParameterTypes(
    funcName: String, paramName: String, tools: [[String: any Sendable]]?
) -> Set<String> {
    guard let schema = parameterSchema(
        funcName: funcName, paramName: paramName, tools: tools)
    else { return [] }
    return schemaTypeNames(schema)
}

// MARK: - Schema Type Extraction

/// Extract types from JSON schema (handles anyOf, oneOf, allOf, enums).
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/minimax_m2.py
func extractTypesFromSchema(_ schema: [String: any Sendable]?) -> [String] {
    guard let schema else { return ["string"] }

    var types: Set<String> = []

    // Handle direct "type" field
    if let typeValue = schema["type"] {
        if let typeString = typeValue as? String {
            types.insert(typeString)
        } else if let typeArray = typeValue as? [String] {
            types.formUnion(typeArray)
        }
    }

    // Handle enum - infer types from enum values
    if let enumValues = schema["enum"] as? [any Sendable], !enumValues.isEmpty {
        for value in enumValues {
            switch value {
            case is NSNull: types.insert("null")
            case is Bool: types.insert("boolean")
            case is Int: types.insert("integer")
            case is Double: types.insert("number")
            case is String: types.insert("string")
            case is [any Sendable]: types.insert("array")
            case is [String: any Sendable]: types.insert("object")
            default: break
            }
        }
    }

    // Handle anyOf, oneOf, allOf - recursively extract types
    for choiceField in ["anyOf", "oneOf", "allOf"] {
        if let choices = schema[choiceField] as? [[String: any Sendable]] {
            for choice in choices {
                types.formUnion(extractTypesFromSchema(choice))
            }
        }
    }

    return types.isEmpty ? ["string"] : Array(types)
}

// MARK: - Type Conversion

/// Convert parameter value based on multiple possible types.
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/minimax_m2.py
func convertValueWithTypes(_ value: String, types: [String]) -> any Sendable {
    let lowerValue = value.lowercased()

    // Handle null values
    if ["null", "none", "nil"].contains(lowerValue) {
        return NSNull()
    }

    let normalizedTypes = Set(types.map { $0.lowercased() })

    // Priority: integer > number > boolean > object > array > string
    let typePriority = [
        "integer", "int", "number", "float", "boolean", "bool",
        "object", "array", "string", "str", "text",
    ]

    for paramType in typePriority {
        guard normalizedTypes.contains(paramType) else { continue }

        switch paramType {
        case "string", "str", "text":
            return value

        case "integer", "int":
            if let intVal = Int(value) {
                return intVal
            }

        case "number", "float":
            if let floatVal = Double(value) {
                let intVal = Int(floatVal)
                return floatVal != Double(intVal) ? floatVal : intVal
            }

        case "boolean", "bool":
            let trimmed = lowerValue.trimmingCharacters(in: .whitespaces)
            if ["true", "1", "yes", "on"].contains(trimmed) {
                return true
            } else if ["false", "0", "no", "off"].contains(trimmed) {
                return false
            }

        case "object", "array":
            if let json = tryParseJSON(value) {
                return json
            }

        default:
            continue
        }
    }

    // Fallback: try JSON parse, then return as string
    return tryParseJSON(value) ?? value
}

/// Convert parameter value based on schema type.
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/qwen3_coder.py
func convertParameterValue(
    _ value: String, paramName: String, funcName: String, tools: [[String: any Sendable]]?
) -> any Sendable {
    if let paramType = getParameterType(funcName: funcName, paramName: paramName, tools: tools) {
        return convertTypedParameterValue(value, type: paramType)
    }
    return convertUnionParameterValue(
        value, paramName: paramName, funcName: funcName, tools: tools)
}

/// Union and `anyOf` coercion used only when `type` is not a single string.
///
/// A declared string keeps the raw XML text, including the word `null`.
/// `NSNull` is returned only when null is declared, string is not, and the
/// text is a null token. One remaining concrete type reuses the single-type
/// converter. Mixed concrete types stay strings. `convertValueWithTypes` is
/// not used: it turns the text `null` into `NSNull` for every schema.
func convertUnionParameterValue(
    _ value: String, paramName: String, funcName: String, tools: [[String: any Sendable]]?
) -> any Sendable {
    let types = declaredParameterTypes(
        funcName: funcName, paramName: paramName, tools: tools)
    if types.isEmpty { return value }
    if types.contains("string") || types.contains("str") || types.contains("text") {
        return value
    }
    let token = value.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
    if types.contains("null"), ["null", "none", "nil"].contains(token) {
        return NSNull()
    }
    let concrete = types.subtracting(["null"])
    guard concrete.count == 1, let only = concrete.first else {
        return value
    }
    return convertTypedParameterValue(value, type: only)
}

func convertTypedParameterValue(_ value: String, type paramType: String) -> any Sendable {
    let type = paramType.lowercased()

    // String types - return as-is
    if ["string", "str", "text", "varchar", "char", "enum"].contains(type) {
        return value
    }

    // Integer types
    if type.hasPrefix("int") || type.hasPrefix("uint")
        || type.hasPrefix("long") || type.hasPrefix("short")
        || type.hasPrefix("unsigned")
    {
        return Int(value) ?? value
    }

    // Float types
    if type.hasPrefix("num") || type.hasPrefix("float") {
        if let floatVal = Double(value) {
            guard floatVal.isFinite else { return value }
            if let intVal = Int(exactly: floatVal) { return intVal }
            return floatVal
        }
        return value
    }

    // Boolean types
    if ["boolean", "bool", "binary"].contains(type) {
        let normalized = value.lowercased().trimmingCharacters(in: .whitespaces)
        if ["true", "1", "yes", "on"].contains(normalized) { return true }
        if ["false", "0", "no", "off"].contains(normalized) { return false }
        return value
    }

    // Object/Array types - JSON decode
    if ["object", "array"].contains(type) || type.hasPrefix("dict") || type.hasPrefix("list") {
        if let json = tryParseJSON(value) {
            return json
        }
    }

    return value
}

// MARK: - String Utilities

/// Extract name from a potentially quoted string.
/// Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tool_parsers/minimax_m2.py
func extractName(_ nameStr: String) -> String {
    let trimmed = nameStr.trimmingCharacters(in: .whitespaces)
    if (trimmed.hasPrefix("\"") && trimmed.hasSuffix("\""))
        || (trimmed.hasPrefix("'") && trimmed.hasSuffix("'"))
    {
        return String(trimmed.dropFirst().dropLast())
    }
    return trimmed
}
