#ifndef QWEN35_RENDER_H
#define QWEN35_RENDER_H

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace q35_render {

enum class StatusCode {
    Ok,
    InvalidArgument,
};

class [[nodiscard]] Status {
public:
    Status() = default;

    static Status invalid_argument(std::string message) {
        return Status(StatusCode::InvalidArgument, std::move(message));
    }

    bool ok() const { return code_ == StatusCode::Ok; }
    StatusCode code() const { return code_; }
    const std::string& message() const { return message_; }

private:
    Status(StatusCode code, std::string message)
        : code_(code), message_(std::move(message)) {}

    StatusCode code_ = StatusCode::Ok;
    std::string message_;
};

enum class Role { System, User, Assistant, Tool };
enum class ContentKind { Text, Image, Video };

struct ContentPart {
    ContentKind kind = ContentKind::Text;
    std::string text;
};

// parser.cpp has already converted each argument into the exact text expected
// by the fixed chat template.  render.cpp never needs to know its JSON type.
struct ToolArgument {
    std::string name;
    std::string text;
};

struct ToolCall {
    std::string name;
    std::vector<ToolArgument> arguments;
};

struct Message {
    Role role = Role::User;

    // parts takes precedence over content. Missing content, null and an empty
    // list all render as empty text.
    bool content_is_null = false;
    std::string content;
    std::vector<ContentPart> parts;

    bool has_reasoning = false;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
};

struct ChatOptions {
    bool add_generation_prompt = true;
    bool enable_thinking = false;
    bool preserve_thinking = true;
    bool add_vision_id = false;
};

struct ChatRequest {
    std::vector<Message> messages;

    // One stable, Python-compatible JSON serialization per tool.  JSON parsing
    // and serialization are confined to parser.cpp; the renderer appends these
    // strings as opaque model syntax.
    std::vector<std::string> tools;
    ChatOptions options;
};

struct RenderedPrompt {
    std::string text;
    std::vector<int> tokens;
};

// The implementation and its JSON dependency live only in parser.cpp.
Status parse_chat_request(const std::string& text, ChatRequest& output);

// Fixed Qwen3.5 chat template and ByteLevel-BPE tokenizer. The binary contains
// only read-only tables; template behavior remains visible in render.cpp.
class Renderer {
public:
    // Factory: a null pointer means the render data could not be loaded.
    // The caller owns the returned Renderer.
    static Renderer* create(const std::string& render_bin, std::string* error);

    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    bool render(const ChatRequest& request, RenderedPrompt* output,
                std::string* error) const;
    bool encode(const std::string& text, std::vector<int>* output,
                std::string* error) const;
    bool decode(const std::vector<int>& tokens, bool skip_special_tokens,
                std::string* output, std::string* error) const;

private:
    Renderer();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace q35_render

#endif
