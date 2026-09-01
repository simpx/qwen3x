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
    std::string id;
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

// Plain OpenAI-compatible request/response values. parser.cpp is the only
// translation unit that converts these values to or from JSON text.
struct CompletionRequest {
    ChatRequest chat;
    std::string model;
    int max_tokens = 128;
    float temperature = 1.0f;
    int top_k = 0;
    float top_p = 1.0f;
    float presence_penalty = 0.0f;
    uint64_t seed = 0;
    bool has_seed = false;
    bool stream = false;
    bool include_usage = false;
    std::vector<std::string> stops;
};

struct CompletionUsage {
    int prompt_tokens = 0;
    int cached_tokens = 0;
    int completion_tokens = 0;
};

// The implementation and its JSON dependency live only in parser.cpp.
Status parse_chat_request(const std::string& text, ChatRequest& output);
Status parse_completion_request(const std::string& text,
                                const std::string& served_model,
                                int default_max_tokens,
                                CompletionRequest& output);

// Split Qwen3.5's generated XML tool syntax from ordinary assistant text.
bool parse_generated_tool_calls(const std::string& text,
                                std::string* content,
                                std::vector<ToolCall>* calls,
                                std::string* error);

std::string error_json(const std::string& message,
                       const char* type = "invalid_request_error",
                       const char* param = nullptr,
                       const char* code = nullptr);
std::string models_json(const std::string& model);
std::string completion_json(const std::string& id, int64_t created,
                            const std::string& model,
                            const std::string& reasoning,
                            const std::string& content,
                            bool include_reasoning,
                            const std::vector<ToolCall>& tool_calls,
                            const char* finish_reason,
                            const CompletionUsage& usage,
                            const std::vector<std::string>* tools = nullptr);
std::string completion_chunk_json(const std::string& id, int64_t created,
                                  const std::string& model,
                                  const char* delta_field,
                                  const std::string& delta,
                                  const char* finish_reason = nullptr);
std::string completion_tool_call_chunk_json(
    const std::string& id, int64_t created, const std::string& model,
    int index, const ToolCall& call,
    const std::vector<std::string>* tools = nullptr);
std::string completion_usage_chunk_json(const std::string& id,
                                        int64_t created,
                                        const std::string& model,
                                        const CompletionUsage& usage);

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
