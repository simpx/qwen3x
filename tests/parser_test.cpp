#include <cstdio>
#include <string>

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
        R"({"messages":[{"role":"user","content":[]}]})", request
    );
    check(!status.ok(), "content arrays should remain unsupported");
}

}  // namespace

int main() {
    test_basic_request();
    test_null_content();
    test_error_preserves_output();
    test_invalid_fields();
    if (failures) return 1;
    std::puts("parser-test: ok");
    return 0;
}
