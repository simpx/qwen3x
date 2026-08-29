#include <cstdio>
#include <string>
#include <vector>

#include "render.h"

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "parser-test: %s\n", message);
    ++failures;
}

void test_basic_request() {
    const std::string text = R"({
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hello"}
        ],
        "add_generation_prompt": false,
        "enable_thinking": true
    })";

    q35_render::ChatRequest request;
    const q35_render::Status status =
        q35_render::parse_chat_request(text, request);

    if (!status.ok()) {
        check(false, "basic request should parse");
        return;
    }
    check(request.messages.size() == 2, "basic request message count");
    if (request.messages.size() != 2) return;
    check(request.messages[0].role == q35_render::Role::System,
          "basic request system role");
    check(request.messages[1].content == "hello", "basic request content");
    check(!request.options.add_generation_prompt,
          "basic request generation option");
    check(request.options.enable_thinking, "basic request thinking option");
}

void test_null_content() {
    q35_render::ChatRequest request;
    const q35_render::Status status = q35_render::parse_chat_request(
        R"({"messages":[{"role":"user","content":null}]})", request
    );

    if (!status.ok()) {
        check(false, "null content should parse");
        return;
    }
    check(request.messages.size() == 1, "null content message count");
    if (request.messages.size() != 1) return;
    check(request.messages[0].content_is_null, "null content flag");
}

void test_content_parts() {
    q35_render::ChatRequest request;
    const q35_render::Status status = q35_render::parse_chat_request(
        R"({"messages":[{"role":"user","content":[
            {"text":"compare "},
            {"type":"image"},
            {"video":"unused"}
        ]}]})",
        request
    );

    if (!status.ok()) {
        check(false, "content parts should parse");
        return;
    }
    check(request.messages.size() == 1, "content parts message count");
    if (request.messages.size() != 1) return;
    const std::vector<q35_render::ContentPart>& parts = request.messages[0].parts;
    check(parts.size() == 3, "content parts count");
    if (parts.size() != 3) return;
    check(parts[0].kind == q35_render::ContentKind::Text &&
          parts[0].text == "compare ", "text content part");
    check(parts[1].kind == q35_render::ContentKind::Image,
          "image content part");
    check(parts[2].kind == q35_render::ContentKind::Video,
          "video content part");
}

void test_string_tool_calls() {
    q35_render::ChatRequest request;
    const q35_render::Status status = q35_render::parse_chat_request(
        R"({"messages":[
            {"role":"user","content":"look it up"},
            {"role":"assistant","content":"","tool_calls":[
                {"name":"alpha","arguments":{"city":"Hangzhou"}},
                {"function":{"name":"beta","arguments":{"unit":"C"}}}
            ]}
        ]})",
        request
    );

    if (!status.ok()) {
        check(false, "string tool calls should parse");
        return;
    }
    check(request.messages.size() == 2, "tool call message count");
    if (request.messages.size() != 2) return;
    const std::vector<q35_render::ToolCall>& calls =
        request.messages[1].tool_calls;
    check(calls.size() == 2, "tool call count");
    if (calls.size() != 2) return;
    check(calls[0].name == "alpha", "direct tool call name");
    check(calls[0].arguments.size() == 1 &&
          calls[0].arguments[0].name == "city" &&
          calls[0].arguments[0].text == "Hangzhou",
          "direct tool call argument");
    check(calls[1].name == "beta", "nested function name");
}

void test_scalar_tool_arguments() {
    q35_render::ChatRequest request;
    const q35_render::Status status = q35_render::parse_chat_request(
        R"({"messages":[{"role":"assistant","tool_calls":[{
            "name":"lookup",
            "arguments":{
                "none":null,
                "enabled":true,
                "count":2,
                "offset":-3,
                "ratio":1.5,
                "city":"Hangzhou"
            }
        }]}]})",
        request
    );

    if (!status.ok()) {
        check(false, "scalar tool arguments should parse");
        return;
    }
    check(request.messages.size() == 1, "scalar argument message count");
    if (request.messages.size() != 1) return;
    check(request.messages[0].tool_calls.size() == 1,
          "scalar argument tool call count");
    if (request.messages[0].tool_calls.size() != 1) return;
    const std::vector<q35_render::ToolArgument>& arguments =
        request.messages[0].tool_calls[0].arguments;
    check(arguments.size() == 6, "scalar tool argument count");
    if (arguments.size() != 6) return;
    check(arguments[0].name == "none" && arguments[0].text == "None",
          "null tool argument");
    check(arguments[1].name == "enabled" && arguments[1].text == "True",
          "boolean tool argument");
    check(arguments[2].text == "2", "unsigned tool argument");
    check(arguments[3].text == "-3", "signed tool argument");
    check(arguments[4].text == "1.5", "floating-point tool argument");
    check(arguments[5].text == "Hangzhou", "string tool argument");
}

void test_nested_tool_arguments() {
    q35_render::ChatRequest request;
    const q35_render::Status status = q35_render::parse_chat_request(
        R"({"messages":[{"role":"assistant","tool_calls":[{
            "name":"lookup",
            "arguments":{
                "list":[2,1,true,null],
                "object":{"b":2,"a":"line\n\"quoted\""}
            }
        }]}]})",
        request
    );

    if (!status.ok()) {
        check(false, "nested tool arguments should parse");
        return;
    }
    check(request.messages.size() == 1, "nested argument message count");
    if (request.messages.size() != 1) return;
    check(request.messages[0].tool_calls.size() == 1,
          "nested argument tool call count");
    if (request.messages[0].tool_calls.size() != 1) return;
    const std::vector<q35_render::ToolArgument>& arguments =
        request.messages[0].tool_calls[0].arguments;
    check(arguments.size() == 2, "nested tool argument count");
    if (arguments.size() != 2) return;
    check(arguments[0].name == "list" &&
          arguments[0].text == "[2, 1, true, null]",
          "array tool argument");
    check(arguments[1].name == "object" &&
          arguments[1].text ==
              R"({"b": 2, "a": "line\n\"quoted\""})",
          "object tool argument preserves order and escapes strings");
}

void test_encoded_tool_arguments() {
    q35_render::ChatRequest request;
    const q35_render::Status status = q35_render::parse_chat_request(
        R"({"messages":[{"role":"assistant","tool_calls":[{
            "type":"function",
            "function":{
                "name":"lookup",
                "arguments":"{\"z\":[2,1],\"a\":\"Hangzhou\",\"enabled\":true}"
            }
        }]}]})",
        request
    );

    if (!status.ok()) {
        check(false, "encoded tool arguments should parse");
        return;
    }
    check(request.messages.size() == 1, "encoded argument message count");
    if (request.messages.size() != 1) return;
    check(request.messages[0].tool_calls.size() == 1,
          "encoded argument tool call count");
    if (request.messages[0].tool_calls.size() != 1) return;
    const std::vector<q35_render::ToolArgument>& arguments =
        request.messages[0].tool_calls[0].arguments;
    check(arguments.size() == 3, "encoded tool argument count");
    if (arguments.size() != 3) return;
    check(arguments[0].name == "z" && arguments[0].text == "[2, 1]",
          "encoded arguments preserve first field");
    check(arguments[1].name == "a" && arguments[1].text == "Hangzhou",
          "encoded arguments preserve second field");
    check(arguments[2].name == "enabled" && arguments[2].text == "True",
          "encoded arguments normalize scalar values");
}

void test_tool_schema() {
    q35_render::ChatRequest request;
    const q35_render::Status status = q35_render::parse_chat_request(
        R"({
            "messages":[{"role":"user","content":"weather?"}],
            "tools":[{
                "type":"function",
                "function":{
                    "name":"weather",
                    "parameters":{
                        "type":"object",
                        "required":["city"],
                        "default":1.0,
                        "maximum":18446744073709551615
                    }
                }
            }]
        })",
        request
    );

    if (!status.ok()) {
        check(false, "tool schema should parse");
        return;
    }
    check(request.tools.size() == 1, "tool schema count");
    if (request.tools.size() != 1) return;
    check(request.tools[0] ==
          R"({"type": "function", "function": {"name": "weather", "parameters": {"type": "object", "required": ["city"], "default": 1.0, "maximum": 18446744073709551615}}})",
          "tool schema serialization and field order");
}

void test_error_preserves_output() {
    q35_render::ChatRequest request;
    q35_render::Message existing;
    existing.content = "keep me";
    request.messages.push_back(existing);

    const q35_render::Status status =
        q35_render::parse_chat_request("not json", request);

    check(!status.ok(), "malformed JSON should fail");
    check(status.code() == q35_render::StatusCode::InvalidArgument,
          "malformed JSON status code");
    check(status.message() == "invalid chat request: malformed JSON",
          "malformed JSON message");
    check(request.messages.size() == 1 &&
          request.messages[0].content == "keep me",
          "failed parse should preserve output");
}

void test_invalid_fields() {
    q35_render::ChatRequest request;
    q35_render::Status status = q35_render::parse_chat_request(
        R"({"messages":[{"role":"alien","content":"x"}]})", request
    );
    check(!status.ok(), "unknown role should fail");

    status = q35_render::parse_chat_request(
        R"({"messages":[],"enable_thinking":"yes"})", request
    );
    check(!status.ok(), "non-boolean option should fail");

    status = q35_render::parse_chat_request(
        R"({"messages":[{"role":"user","content":[{"type":"audio"}]}]})",
        request
    );
    check(!status.ok(), "unknown content part should fail");

    status = q35_render::parse_chat_request(
        R"({"messages":[{"role":"user","content":[42]}]})", request
    );
    check(!status.ok(), "non-object content part should fail");

    status = q35_render::parse_chat_request(
        R"({"messages":[{"role":"assistant","tool_calls":{}}]})", request
    );
    check(!status.ok(), "non-array tool calls should fail");

    status = q35_render::parse_chat_request(
        R"({"messages":[{"role":"assistant","tool_calls":[{
            "name":"f","arguments":"not json"
        }]}]})",
        request
    );
    check(!status.ok(), "malformed encoded arguments should fail");

    status = q35_render::parse_chat_request(
        R"({"messages":[],"tools":{}})", request
    );
    check(!status.ok(), "non-array tools should fail");
}

}  // namespace

int main() {
    test_basic_request();
    test_null_content();
    test_content_parts();
    test_string_tool_calls();
    test_scalar_tool_arguments();
    test_nested_tool_arguments();
    test_encoded_tool_arguments();
    test_tool_schema();
    test_error_preserves_output();
    test_invalid_fields();
    if (failures) return 1;
    std::puts("parser-test: ok");
    return 0;
}
