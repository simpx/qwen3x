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
        R"({"messages":[{"role":"assistant","tool_calls":[
            {"name":"f","arguments":{"count":2}}
        ]}]})",
        request
    );
    check(!status.ok(), "non-string tool argument should remain unsupported");
}

}  // namespace

int main() {
    test_basic_request();
    test_null_content();
    test_content_parts();
    test_string_tool_calls();
    test_error_preserves_output();
    test_invalid_fields();
    if (failures) return 1;
    std::puts("parser-test: ok");
    return 0;
}
