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

    q3x_render::ChatRequest request;
    const q3x_render::Status status =
        q3x_render::parse_chat_request(text, request);

    if (!status.ok()) {
        check(false, "basic request should parse");
        return;
    }
    check(request.messages.size() == 2, "basic request message count");
    if (request.messages.size() != 2) return;
    check(request.messages[0].role == q3x_render::Role::System,
          "basic request system role");
    check(request.messages[1].content == "hello", "basic request content");
    check(!request.options.add_generation_prompt,
          "basic request generation option");
    check(request.options.enable_thinking, "basic request thinking option");
}

void test_null_content() {
    q3x_render::ChatRequest request;
    const q3x_render::Status status = q3x_render::parse_chat_request(
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
    q3x_render::ChatRequest request;
    const q3x_render::Status status = q3x_render::parse_chat_request(
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
    const std::vector<q3x_render::ContentPart>& parts = request.messages[0].parts;
    check(parts.size() == 3, "content parts count");
    if (parts.size() != 3) return;
    check(parts[0].kind == q3x_render::ContentKind::Text &&
          parts[0].text == "compare ", "text content part");
    check(parts[1].kind == q3x_render::ContentKind::Image,
          "image content part");
    check(parts[2].kind == q3x_render::ContentKind::Video,
          "video content part");
}

void test_string_tool_calls() {
    q3x_render::ChatRequest request;
    const q3x_render::Status status = q3x_render::parse_chat_request(
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
    const std::vector<q3x_render::ToolCall>& calls =
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
    q3x_render::ChatRequest request;
    const q3x_render::Status status = q3x_render::parse_chat_request(
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
    const std::vector<q3x_render::ToolArgument>& arguments =
        request.messages[0].tool_calls[0].arguments;
    check(arguments.size() == 6, "scalar tool argument count");
    if (arguments.size() != 6) return;
    check(arguments[0].name == "none" && arguments[0].text == "null",
          "null tool argument");
    check(arguments[1].name == "enabled" && arguments[1].text == "true",
          "boolean tool argument");
    check(arguments[2].text == "2", "unsigned tool argument");
    check(arguments[3].text == "-3", "signed tool argument");
    check(arguments[4].text == "1.5", "floating-point tool argument");
    check(arguments[5].text == "Hangzhou", "string tool argument");
}

void test_nested_tool_arguments() {
    q3x_render::ChatRequest request;
    const q3x_render::Status status = q3x_render::parse_chat_request(
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
    const std::vector<q3x_render::ToolArgument>& arguments =
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
    q3x_render::ChatRequest request;
    const q3x_render::Status status = q3x_render::parse_chat_request(
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
    const std::vector<q3x_render::ToolArgument>& arguments =
        request.messages[0].tool_calls[0].arguments;
    check(arguments.size() == 3, "encoded tool argument count");
    if (arguments.size() != 3) return;
    check(arguments[0].name == "z" && arguments[0].text == "[2, 1]",
          "encoded arguments preserve first field");
    check(arguments[1].name == "a" && arguments[1].text == "Hangzhou",
          "encoded arguments preserve second field");
    check(arguments[2].name == "enabled" && arguments[2].text == "true",
          "encoded arguments normalize scalar values");
}

void test_tool_schema() {
    q3x_render::ChatRequest request;
    const q3x_render::Status status = q3x_render::parse_chat_request(
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
    q3x_render::ChatRequest request;
    q3x_render::Message existing;
    existing.content = "keep me";
    request.messages.push_back(existing);

    const q3x_render::Status status =
        q3x_render::parse_chat_request("not json", request);

    check(!status.ok(), "malformed JSON should fail");
    check(status.code() == q3x_render::StatusCode::InvalidArgument,
          "malformed JSON status code");
    check(status.message() == "invalid chat request: malformed JSON",
          "malformed JSON message");
    check(request.messages.size() == 1 &&
          request.messages[0].content == "keep me",
          "failed parse should preserve output");
}

void test_invalid_fields() {
    q3x_render::ChatRequest request;
    q3x_render::Status status = q3x_render::parse_chat_request(
        R"({"messages":[{"role":"alien","content":"x"}]})", request
    );
    check(!status.ok(), "unknown role should fail");

    status = q3x_render::parse_chat_request(
        R"({"messages":[],"enable_thinking":"yes"})", request
    );
    check(!status.ok(), "non-boolean option should fail");

    status = q3x_render::parse_chat_request(
        R"({"messages":[{"role":"user","content":[{"type":"audio"}]}]})",
        request
    );
    check(!status.ok(), "unknown content part should fail");

    status = q3x_render::parse_chat_request(
        R"({"messages":[{"role":"user","content":[42]}]})", request
    );
    check(!status.ok(), "non-object content part should fail");

    status = q3x_render::parse_chat_request(
        R"({"messages":[{"role":"assistant","tool_calls":{}}]})", request
    );
    check(!status.ok(), "non-array tool calls should fail");

    status = q3x_render::parse_chat_request(
        R"({"messages":[{"role":"assistant","tool_calls":[{
            "name":"f","arguments":"not json"
        }]}]})",
        request
    );
    check(!status.ok(), "malformed encoded arguments should fail");

    status = q3x_render::parse_chat_request(
        R"({"messages":[],"tools":{}})", request
    );
    check(!status.ok(), "non-array tools should fail");
}

void test_completion_request() {
    const std::string text = R"({
        "model":"qwen3.5-0.8b",
        "messages":[
            {"role":"developer","content":"brief"},
            {"role":"user","content":"hello"}
        ],
        "temperature":0.5,
        "top_p":0.9,
        "top_k":20,
        "presence_penalty":1.25,
        "seed":-1,
        "max_completion_tokens":42,
        "stop":["END"],
        "stream":true,
        "stream_options":{"include_usage":true},
        "chat_template_kwargs":{
            "enable_thinking":true,
            "preserve_thinking":false
        }
    })";
    q3x_render::CompletionRequest request;
    const q3x_render::Status status = q3x_render::parse_completion_request(
        text, "qwen3.5-0.8b", 128, request);
    check(status.ok(), "completion request should parse");
    check(request.chat.messages.size() == 2 &&
          request.chat.messages[0].role == q3x_render::Role::System,
          "developer role should map to system");
    check(request.max_tokens == 42 && request.top_k == 20,
          "completion integer options");
    check(request.temperature == 0.5f && request.top_p == 0.9f &&
          request.presence_penalty == 1.25f,
          "completion sampling options");
    check(request.has_seed && request.seed == UINT64_MAX,
          "completion signed seed");
    check(request.stream && request.include_usage &&
          request.chat.options.enable_thinking &&
          !request.chat.options.preserve_thinking,
          "completion stream and template options");
    check(request.stops.size() == 1 && request.stops[0] == "END",
          "completion stop strings");
}

void test_completion_tool_request() {
    const std::string text = R"({
        "model":"qwen3.5-0.8b",
        "messages":[
            {"role":"user","content":"read it"},
            {"role":"assistant","content":null,"tool_calls":[{
                "id":"call_1","type":"function","function":{
                    "name":"read_file","arguments":"{\"path\":\"README.md\"}"
                }
            }]},
            {"role":"tool","tool_call_id":"call_1","content":"hello"}
        ],
        "tools":[{"type":"function","function":{
            "name":"read_file","parameters":{"type":"object"}
        }}],
        "stream":true
    })";
    q3x_render::CompletionRequest request;
    const q3x_render::Status status = q3x_render::parse_completion_request(
        text, "qwen3.5-0.8b", 128, request);
    check(status.ok(), "completion tool request should parse");
    check(request.stream, "completion tool request should allow streaming");
    check(request.chat.tools.size() == 1, "completion tool schema count");
    check(request.chat.messages.size() == 3, "completion tool history count");
    if (request.chat.messages.size() < 2 ||
        request.chat.messages[1].tool_calls.empty()) return;
    check(request.chat.messages[1].tool_calls[0].id == "call_1",
          "completion tool call ID");
}

void test_generated_tool_calls() {
    const std::string text = R"(I will read it.
<tool_call>
<function=read_file>
<parameter=path>
README.md
</parameter>
</function>
</tool_call>
<tool_call>
<function=bash>
<parameter=command>
printf 'hello\nworld'
</parameter>
</function>
</tool_call>)";
    std::string content;
    std::string error;
    std::vector<q3x_render::ToolCall> calls;
    check(q3x_render::parse_generated_tool_calls(
              text, &content, &calls, &error),
          "generated tool calls should parse");
    check(content == "I will read it.", "generated tool preamble");
    check(calls.size() == 2, "generated tool call count");
    if (calls.size() != 2) return;
    check(calls[0].name == "read_file" &&
          calls[0].arguments.size() == 1 &&
          calls[0].arguments[0].name == "path" &&
          calls[0].arguments[0].text == "README.md",
          "generated read call");
    check(calls[1].name == "bash" &&
          calls[1].arguments[0].text == "printf 'hello\\nworld'",
          "generated multiline argument");

    check(q3x_render::parse_generated_tool_calls(
              "plain answer", &content, &calls, &error) &&
          content == "plain answer" && calls.empty(),
          "plain generated content");
    check(q3x_render::parse_generated_tool_calls(
              "<function=bash><parameter=command>pwd</parameter></function>",
              &content, &calls, &error) && calls.size() == 1 &&
          calls[0].name == "bash" && calls[0].arguments[0].text == "pwd",
          "generated function fallback");
    check(!q3x_render::parse_generated_tool_calls(
              "<tool_call><function=bash>", &content, &calls, &error),
          "incomplete generated call should fail");
    check(content.empty() && calls.empty(),
          "incomplete generated call should not become content");

    const std::string partial = R"(Let me check.
<tool_call>
<function=bash>
<parameter=command>pwd</parameter>
</function>
</tool_call>
<tool_call>
<function=read>
<parameter=path>README.md)";
    check(!q3x_render::parse_generated_tool_calls(
              partial, &content, &calls, &error),
          "partially generated tool calls should report failure");
    check(content == "Let me check." && calls.size() == 1 &&
          calls[0].name == "bash" && calls[0].arguments.size() == 1 &&
          calls[0].arguments[0].text == "pwd",
          "complete calls before an incomplete tail should be retained");

    const std::string missing_wrapper = R"(<tool_call>
<function=bash>
<parameter=command>pwd</parameter>
<broken>
<tool_call>
<function=read>
<parameter=path>README.md</parameter>
</function>
</tool_call>)";
    check(!q3x_render::parse_generated_tool_calls(
              missing_wrapper, &content, &calls, &error),
          "a missing function wrapper should report failure");
    check(calls.size() == 2 && calls[0].name == "bash" &&
          calls[0].arguments.size() == 1 &&
          calls[0].arguments[0].text == "pwd" &&
          calls[1].name == "read" && calls[1].arguments.size() == 1 &&
          calls[1].arguments[0].text == "README.md",
          "a new tool wrapper should keep complete calls separate");
}

void test_completion_json() {
    q3x_render::CompletionUsage usage{10, 4, 2};
    std::vector<q3x_render::ToolCall> no_calls;
    const std::string response = q3x_render::completion_json(
        "chatcmpl-1", 123, "qwen3.5-0.8b", "think", "answer", true,
        no_calls, "stop", usage);
    check(response.find("\"reasoning_content\":\"think\"") !=
          std::string::npos, "completion reasoning JSON");
    check(response.find("\"cached_tokens\":4") != std::string::npos,
          "completion cached usage JSON");
    q3x_render::ToolCall call;
    call.id = "call_1";
    call.name = "read_file";
    call.arguments.push_back({"path", "README.md"});
    const std::string tool_response = q3x_render::completion_json(
        "chatcmpl-2", 123, "qwen3.5-0.8b", "", "", false,
        {call}, "tool_calls", usage);
    check(tool_response.find("\"content\":null") != std::string::npos &&
          tool_response.find("\"finish_reason\":\"tool_calls\"") !=
              std::string::npos &&
          tool_response.find("\\\"path\\\":\\\"README.md\\\"") !=
              std::string::npos,
          "completion tool call JSON");
    call.arguments.push_back({"start_line", "201"});
    const std::vector<std::string> tools = {
        R"({"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"start_line":{"type":"integer"}}}}})"
    };
    const std::string typed_tool_response = q3x_render::completion_json(
        "chatcmpl-3", 123, "qwen3.5-0.8b", "", "", false,
        {call}, "tool_calls", usage, &tools);
    check(typed_tool_response.find(
              "\\\"path\\\":\\\"README.md\\\"") != std::string::npos &&
          typed_tool_response.find(
              "\\\"start_line\\\":201") != std::string::npos,
          "generated arguments follow tool JSON schema");
    const std::string tool_chunk = q3x_render::completion_tool_call_chunk_json(
        "chatcmpl-3", 123, "qwen3.5-0.8b", 0, call, &tools);
    check(tool_chunk.find("\"tool_calls\"") != std::string::npos &&
          tool_chunk.find("\"name\":\"read_file\"") != std::string::npos &&
          tool_chunk.find("\"index\":0") != std::string::npos &&
          tool_chunk.find("\\\"start_line\\\":201") != std::string::npos,
          "stream tool call chunk follows schema");
    const std::string chunk = q3x_render::completion_chunk_json(
        "chatcmpl-1", 123, "qwen3.5-0.8b", "content", "你");
    check(chunk.find("\"content\":\"你\"") != std::string::npos,
          "stream chunk keeps UTF-8");
    const std::string error = q3x_render::error_json(
        "bad", "invalid_request_error", "model", "model_not_found");
    check(error.find("\"code\":\"model_not_found\"") != std::string::npos,
          "OpenAI error JSON");
}

void test_invalid_completion_requests() {
    for (const char* text : {
        R"({"model":"other","messages":[{"role":"user","content":"x"}]})",
        R"({"model":"qwen3.5-0.8b","messages":[]})",
        R"({"model":"qwen3.5-0.8b","messages":[{"role":"user","content":null}]})",
        R"({"model":"qwen3.5-0.8b","messages":[{"role":"user","content":"x"}],"top_p":0})",
        R"({"model":"qwen3.5-0.8b","messages":[{"role":"user","content":"x"}],"n":2})",
    }) {
        q3x_render::CompletionRequest request;
        check(!q3x_render::parse_completion_request(
                  text, "qwen3.5-0.8b", 128, request).ok(),
              "invalid completion request should fail");
    }
}

void test_tool_output_limit_error() {
    const std::string truncated = q3x_render::tool_call_error_json(
        "incomplete generated tool call", 4096);
    check(truncated.find("max_tokens=4096") != std::string::npos &&
          truncated.find("\"code\":\"max_tokens_exceeded\"") != std::string::npos &&
          truncated.find("\"param\":\"max_tokens\"") != std::string::npos &&
          truncated.find("please retry") == std::string::npos,
          "a truncated tool call explains its budget instead of retrying unchanged");
    const std::string malformed = q3x_render::tool_call_error_json(
        "incomplete generated tool call", 0);
    check(malformed.find("\"code\":\"incomplete_tool_call\"") != std::string::npos &&
          malformed.find("please retry your request") != std::string::npos,
          "an incomplete tool call without truncation retains retry behavior");
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
    test_completion_request();
    test_completion_tool_request();
    test_generated_tool_calls();
    test_completion_json();
    test_invalid_completion_requests();
    test_tool_output_limit_error();
    if (failures) return 1;
    std::puts("parser-test: ok");
    return 0;
}
