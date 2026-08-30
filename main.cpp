// main.cpp -- one-process Qwen3.5 inference program.
//
//   qwen35 -p "hello"
//   qwen35 -l --host 127.0.0.1 --port 8000
//   qwen35 --bench PREFILL_TOKENS DECODE_TOKENS
//
// Prompt mode constructs a ChatRequest directly. Listen mode adds the HTTP/JSON
// boundary. Bench mode bypasses both renderer and HTTP to measure Session only.
//
// Listen mode exposes:
//   POST /v1/chat/completions
//   Content-Type: application/json
//   {"model":"qwen3.5-0.8b",
//    "messages":[{"role":"user","content":"hello"}],
//    "stream":true}

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#define CPPHTTPLIB_NO_EXCEPTIONS
#include "third_party/httplib/httplib.h"

#include "internal.h"
#include "render.h"

namespace {

constexpr size_t ERROR_SIZE = 512;
constexpr size_t MAX_REQUEST_BYTES = 1024 * 1024;
constexpr const char* DEFAULT_MODEL = "qwen35-0.8b-model.bin";
constexpr const char* DEFAULT_RENDER = "qwen35-0.8b-render.bin";

enum class Mode { None, Prompt, Listen, Bench };

struct Options {
    Mode mode = Mode::None;
    std::string model_path;
    std::string render_path;
    std::string prompt;
    std::string host = "127.0.0.1";
    std::string served_model = "qwen3.5-0.8b";
    std::string log_file;
    q35_log_level log_level = Q35_LOG_ERROR;
    int port = 8000;
    int slots = 2;
    int context = 4096;
    int max_tokens = 128;
    int bench_prefill = 0;
    int bench_decode = 0;
    int request_timeout = 600;
    size_t log_max_bytes = 20 * 1024 * 1024;
    size_t log_backups = 5;
    bool context_set = false;
    bool slots_set = false;
    bool mock = false;
};

struct Runtime {
    Options options;
    q35_engine* engine = nullptr;
    q35_session_manager* manager = nullptr;
    std::unique_ptr<q35_render::Renderer> renderer;
    int think_end_token = -1;

    ~Runtime() {
        if (manager) q35_session_manager_destroy(manager);
        if (engine) q35_engine_destroy(engine);
    }
};

struct SessionLease {
    q35_session_manager* manager = nullptr;
    q35_session* session = nullptr;
    bool keep = false;

    ~SessionLease() {
        if (session) q35_session_manager_release(manager, session, keep);
    }
};

struct GenerationResult {
    std::vector<int> tokens;
    std::string reasoning;
    std::string content;
    std::string finish_reason = "length";
    int cached_tokens = 0;
};

template <typename Integer>
bool integer(const std::string& text, Integer* output) {
    if (text.empty()) return false;
    Integer value{};
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto parsed = std::from_chars(begin, end, value);
    if (parsed.ec != std::errc() || parsed.ptr != end) return false;
    *output = value;
    return true;
}

std::string sibling_file(const char* program, const char* name) {
    std::error_code error;
    std::filesystem::path executable =
        std::filesystem::read_symlink("/proc/self/exe", error);
    if (error) {
        error.clear();
        executable = std::filesystem::absolute(program, error);
        if (error) executable = program;
    }
    return (executable.parent_path() / name).string();
}

bool option_value(int& index, int argc, char** argv,
                  std::string* value, std::string* error) {
    if (++index >= argc) {
        *error = "missing option value";
        return false;
    }
    *value = argv[index];
    return true;
}

bool parse_options(int argc, char** argv, Options* options,
                   std::string* error) {
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--mock") {
            options->mock = true;
            continue;
        }
        if (argument == "-l" || argument == "--listen") {
            if (options->mode != Mode::None) {
                *error = "choose exactly one of --prompt, --listen, or --bench";
                return false;
            }
            options->mode = Mode::Listen;
            continue;
        }
        if (argument == "--bench") {
            if (options->mode != Mode::None) {
                *error = "choose exactly one of --prompt, --listen, or --bench";
                return false;
            }
            if (index + 2 >= argc ||
                !integer(argv[index + 1], &options->bench_prefill) ||
                !integer(argv[index + 2], &options->bench_decode) ||
                options->bench_prefill <= 0 || options->bench_decode <= 0) {
                *error = "--bench requires positive PREFILL_TOKENS and DECODE_TOKENS";
                return false;
            }
            options->mode = Mode::Bench;
            index += 2;
            continue;
        }
        std::string value;
        if (!option_value(index, argc, argv, &value, error)) return false;
        if (argument == "-p" || argument == "--prompt") {
            if (options->mode != Mode::None) {
                *error = "choose exactly one of --prompt, --listen, or --bench";
                return false;
            }
            options->mode = Mode::Prompt;
            options->prompt = value;
        } else if (argument == "-m" || argument == "--model") {
            options->model_path = value;
        } else if (argument == "-r" || argument == "--render") {
            options->render_path = value;
        } else if (argument == "--host") {
            options->host = value;
        } else if (argument == "--served-model-name") {
            options->served_model = value;
        } else if (argument == "--log-file") {
            options->log_file = value;
        } else if (argument == "--port") {
            if (!integer(value, &options->port) ||
                options->port <= 0 || options->port > 65535) {
                *error = "invalid --port";
                return false;
            }
        } else if (argument == "--session-slots") {
            if (!integer(value, &options->slots) || options->slots <= 0) {
                *error = "invalid --session-slots";
                return false;
            }
            options->slots_set = true;
        } else if (argument == "--session-context") {
            if (!integer(value, &options->context) || options->context <= 0) {
                *error = "invalid --session-context";
                return false;
            }
            options->context_set = true;
        } else if (argument == "--max-tokens") {
            if (!integer(value, &options->max_tokens) ||
                options->max_tokens <= 0) {
                *error = "invalid --max-tokens";
                return false;
            }
        } else if (argument == "--request-timeout") {
            if (!integer(value, &options->request_timeout) ||
                options->request_timeout <= 0) {
                *error = "invalid --request-timeout";
                return false;
            }
        } else if (argument == "--log-max-mb") {
            size_t megabytes = 0;
            if (!integer(value, &megabytes) || megabytes == 0) {
                *error = "invalid --log-max-mb";
                return false;
            }
            options->log_max_bytes = megabytes * 1024 * 1024;
        } else if (argument == "--log-backups") {
            if (!integer(value, &options->log_backups) ||
                options->log_backups == 0) {
                *error = "invalid --log-backups";
                return false;
            }
        } else if (argument == "--log-level") {
            if (value == "debug") options->log_level = Q35_LOG_DEBUG;
            else if (value == "info") options->log_level = Q35_LOG_INFO;
            else if (value == "warn" || value == "warning") {
                options->log_level = Q35_LOG_WARN;
            } else if (value == "error") options->log_level = Q35_LOG_ERROR;
            else {
                *error = "invalid --log-level";
                return false;
            }
        } else {
            *error = "unknown option: " + argument;
            return false;
        }
    }
    if (options->model_path.empty()) {
        *error = "-m/--model is required";
        return false;
    }
    if (options->mode == Mode::None) {
        *error = "one of -p/--prompt, -l/--listen, or --bench is required";
        return false;
    }
    if (options->mode != Mode::Bench && options->render_path.empty()) {
        *error = "-r/--render is required";
        return false;
    }
    if (options->mode == Mode::Bench) {
        const int64_t required = static_cast<int64_t>(options->bench_prefill) +
                                 options->bench_decode;
        if (required > q35_backend::max_context()) {
            *error = "benchmark exceeds the model context";
            return false;
        }
        if (!options->context_set) options->context = static_cast<int>(required);
        if (required > options->context) {
            *error = "benchmark exceeds --session-context";
            return false;
        }
        if (!options->slots_set) options->slots = 1;
    }
    return true;
}

void usage(const char* program) {
    std::fprintf(stderr,
        "usage:\n"
        "  %s [-m MODEL] [-r RENDER] -p PROMPT [--max-tokens N]\n"
        "  %s [-m MODEL] [-r RENDER] -l [--host HOST] [--port PORT]\n"
        "  %s [-m MODEL] --bench PREFILL_TOKENS DECODE_TOKENS\n"
        "common: [--session-slots N] [--session-context N] [--mock]\n"
        "        [--log-level debug|info|warn|error] [--log-file PATH]\n"
        "defaults beside qwen35: %s and %s\n", program, program, program,
        DEFAULT_MODEL, DEFAULT_RENDER);
}

std::string make_id(const char* prefix) {
    static const uint64_t random_prefix = [] {
        std::random_device device;
        return (static_cast<uint64_t>(device()) << 32) ^ device();
    }();
    static std::atomic<uint64_t> next{0};
    char text[64];
    std::snprintf(text, sizeof(text), "%s%016llx%016llx", prefix,
                  static_cast<unsigned long long>(random_prefix),
                  static_cast<unsigned long long>(next.fetch_add(1)));
    return text;
}

uint64_t random_seed() {
    std::random_device device;
    return (static_cast<uint64_t>(device()) << 32) ^ device();
}

bool constant_time_equal(const std::string& left, const std::string& right) {
    size_t difference = left.size() ^ right.size();
    const size_t count = std::max(left.size(), right.size());
    for (size_t index = 0; index < count; ++index) {
        const unsigned char a = index < left.size() ? left[index] : 0;
        const unsigned char b = index < right.size() ? right[index] : 0;
        difference |= a ^ b;
    }
    return difference == 0;
}

void json_response(httplib::Response& response, int status,
                   std::string body) {
    response.status = status;
    response.set_content(std::move(body), "application/json; charset=utf-8");
}

void request_id(httplib::Response& response) {
    response.set_header("X-Request-Id", make_id("req-"));
}

void api_error(httplib::Response& response, int status,
               const std::string& message,
               const char* type = "invalid_request_error",
               const char* param = nullptr, const char* code = nullptr) {
    json_response(response, status,
                  q35_render::error_json(message, type, param, code));
}

bool authenticate(const httplib::Request& request,
                  httplib::Response& response) {
    const char* key = std::getenv("QWEN_API_KEY");
    if (!key || !key[0]) return true;
    const std::string expected = std::string("Bearer ") + key;
    if (constant_time_equal(request.get_header_value("Authorization"), expected)) {
        return true;
    }
    api_error(response, 401, "invalid API key", "authentication_error",
              nullptr, "invalid_api_key");
    return false;
}

bool split_output(const Runtime& runtime,
                  const q35_render::CompletionRequest& request,
                  const std::vector<int>& tokens,
                  std::string* reasoning, std::string* content,
                  std::string* error) {
    std::vector<int> reasoning_tokens;
    std::vector<int> content_tokens = tokens;
    if (request.chat.options.enable_thinking) {
        const auto boundary = std::find(tokens.begin(), tokens.end(),
                                        runtime.think_end_token);
        reasoning_tokens.assign(tokens.begin(), boundary);
        if (boundary == tokens.end()) content_tokens.clear();
        else content_tokens.assign(boundary + 1, tokens.end());
    }
    if (!runtime.renderer->decode(reasoning_tokens, true, reasoning, error)) {
        return false;
    }
    if (!runtime.renderer->decode(content_tokens, true, content, error)) {
        return false;
    }
    if (request.chat.options.enable_thinking) {
        while (!reasoning->empty() &&
               std::strchr(" \t\r\n\f\v", reasoning->back())) {
            reasoning->pop_back();
        }
        size_t begin = 0;
        while (begin < content->size() &&
               std::strchr(" \t\r\n\f\v", (*content)[begin])) ++begin;
        content->erase(0, begin);
    }
    return true;
}

std::string stable_text(std::string text) {
    static const std::string replacement = "\xef\xbf\xbd";
    while (text.size() >= replacement.size() &&
           text.compare(text.size() - replacement.size(),
                        replacement.size(), replacement) == 0) {
        text.resize(text.size() - replacement.size());
    }
    return text;
}

std::pair<std::string, bool> stop_view(
        const std::string& text, const std::vector<std::string>& stops,
        bool final) {
    size_t earliest = std::string::npos;
    for (const std::string& stop : stops) {
        earliest = std::min(earliest, text.find(stop));
    }
    if (earliest != std::string::npos) return {text.substr(0, earliest), true};
    if (final || stops.empty()) return {text, false};
    size_t hold = 0;
    for (const std::string& stop : stops) hold = std::max(hold, stop.size() - 1);
    return {text.substr(0, text.size() > hold ? text.size() - hold : 0), false};
}

using Publish = std::function<bool(const char* field, const std::string& text)>;

bool generate(Runtime& runtime, const q35_render::CompletionRequest& request,
              const std::vector<int>& prompt, q35_session* session,
              const Publish& publish, GenerationResult* result,
              std::string* error) {
    char native_error[ERROR_SIZE]{};
    const auto started = std::chrono::steady_clock::now();
    if (q35_session_sync(session, prompt.data(), static_cast<int>(prompt.size()),
                         static_cast<int>(prompt.size()), &result->cached_tokens,
                         native_error, sizeof(native_error)) != Q35_OK) {
        *error = native_error;
        return false;
    }

    uint64_t rng = request.has_seed ? request.seed : random_seed();
    std::vector<int> penalty_tokens;
    std::unordered_set<int> penalty_seen;
    std::string published_reasoning;
    std::string published_content;

    while (static_cast<int>(result->tokens.size()) < request.max_tokens) {
        if (std::chrono::steady_clock::now() - started >
            std::chrono::seconds(runtime.options.request_timeout)) {
            *error = "generation request timed out";
            return false;
        }
        const int token = request.temperature == 0.0f &&
                          request.presence_penalty == 0.0f
            ? q35_session_argmax(session)
            : q35_session_sample(
                session, request.temperature, request.top_k, request.top_p,
                request.presence_penalty,
                penalty_tokens.empty() ? nullptr : penalty_tokens.data(),
                static_cast<int>(penalty_tokens.size()), &rng);
        if (token < 0) {
            *error = "sampling failed";
            return false;
        }
        if (q35_token_is_stop(token)) {
            result->finish_reason = "stop";
            break;
        }

        result->tokens.push_back(token);
        if (penalty_seen.insert(token).second) penalty_tokens.push_back(token);
        if (!split_output(runtime, request, result->tokens,
                          &result->reasoning, &result->content, error)) return false;

        const std::string reasoning = stable_text(result->reasoning);
        const auto visible = stop_view(result->content, request.stops, false);
        const std::string content = stable_text(visible.first);
        if (reasoning.compare(0, published_reasoning.size(),
                              published_reasoning) != 0 ||
            content.compare(0, published_content.size(),
                            published_content) != 0) {
            *error = "tokenizer rewrote streamed text";
            return false;
        }
        if (reasoning.size() > published_reasoning.size() &&
            !publish("reasoning_content",
                     reasoning.substr(published_reasoning.size()))) {
            *error = "client disconnected";
            return false;
        }
        if (content.size() > published_content.size() &&
            !publish("content", content.substr(published_content.size()))) {
            *error = "client disconnected";
            return false;
        }
        published_reasoning = reasoning;
        published_content = content;

        if (visible.second) {
            result->finish_reason = "stop";
            break;
        }
        if (static_cast<int>(result->tokens.size()) == request.max_tokens) break;
        if (q35_session_eval(session, token,
                             native_error, sizeof(native_error)) != Q35_OK) {
            *error = native_error;
            return false;
        }
    }

    if (!split_output(runtime, request, result->tokens,
                      &result->reasoning, &result->content, error)) return false;
    const auto final_content = stop_view(result->content, request.stops, true);
    result->content = final_content.first;
    if (final_content.second) result->finish_reason = "stop";
    if (result->reasoning.compare(0, published_reasoning.size(),
                                  published_reasoning) != 0 ||
        result->content.compare(0, published_content.size(),
                                published_content) != 0) {
        *error = "tokenizer rewrote streamed text";
        return false;
    }
    if (result->reasoning.size() > published_reasoning.size() &&
        !publish("reasoning_content",
                 result->reasoning.substr(published_reasoning.size()))) {
        *error = "client disconnected";
        return false;
    }
    if (result->content.size() > published_content.size() &&
        !publish("content", result->content.substr(published_content.size()))) {
        *error = "client disconnected";
        return false;
    }
    return true;
}

bool prepare_request(Runtime& runtime, const httplib::Request& http,
                     httplib::Response& response,
                     q35_render::CompletionRequest* request,
                     q35_render::RenderedPrompt* prompt) {
    const q35_render::Status status = q35_render::parse_completion_request(
        http.body, runtime.options.served_model,
        runtime.options.max_tokens, *request);
    if (!status.ok()) {
        const bool missing_model = status.message().find("model not found") !=
                                   std::string::npos;
        api_error(response, missing_model ? 404 : 400, status.message(),
                  "invalid_request_error", missing_model ? "model" : nullptr,
                  missing_model ? "model_not_found" : nullptr);
        return false;
    }
    std::string error;
    if (!runtime.renderer->render(request->chat, prompt, &error)) {
        api_error(response, 400, error, "invalid_request_error", "messages");
        return false;
    }
    if (prompt->tokens.empty()) {
        api_error(response, 500, "chat template produced an empty prompt",
                  "server_error", nullptr, "invalid_chat_template");
        return false;
    }
    if (prompt->tokens.size() + static_cast<size_t>(request->max_tokens) >
        static_cast<size_t>(runtime.options.context)) {
        api_error(response, 400, "prompt and completion exceed the Session context",
                  "invalid_request_error", "max_completion_tokens",
                  "context_length_exceeded");
        return false;
    }
    return true;
}

q35_session* acquire(Runtime& runtime, const std::vector<int>& prompt,
                     httplib::Response& response) {
    q35_session* session = nullptr;
    char error[ERROR_SIZE]{};
    const int status = q35_session_manager_acquire(
        runtime.manager, prompt.data(), static_cast<int>(prompt.size()),
        &session, error, sizeof(error));
    if (status == Q35_BUSY) {
        api_error(response, 429, error[0] ? error : "all sessions are busy",
                  "rate_limit_error", nullptr, "engine_busy");
        return nullptr;
    }
    if (status != Q35_OK) {
        api_error(response, 500, error[0] ? error : "cannot acquire Session",
                  "server_error");
        return nullptr;
    }
    return session;
}

void chat(Runtime& runtime, const httplib::Request& http,
          httplib::Response& response) {
    if (!authenticate(http, response)) return;
    q35_render::CompletionRequest request;
    q35_render::RenderedPrompt prompt;
    if (!prepare_request(runtime, http, response, &request, &prompt)) return;

    q35_session* session = acquire(runtime, prompt.tokens, response);
    if (!session) return;
    auto lease = std::make_shared<SessionLease>();
    lease->manager = runtime.manager;
    lease->session = session;

    const std::string completion_id = make_id("chatcmpl-");
    const int64_t created = static_cast<int64_t>(std::time(nullptr));
    LOG_INFO("generation started completion_id=%s prompt_tokens=%zu max_tokens=%d "
             "stream=%d", completion_id.c_str(), prompt.tokens.size(),
             request.max_tokens, request.stream);

    if (!request.stream) {
        GenerationResult result;
        std::string error;
        const Publish discard = [](const char*, const std::string&) { return true; };
        if (!generate(runtime, request, prompt.tokens, session, discard,
                      &result, &error)) {
            api_error(response, 500, error, "server_error");
            return;
        }
        lease->keep = true;
        q35_render::CompletionUsage usage{
            static_cast<int>(prompt.tokens.size()), result.cached_tokens,
            static_cast<int>(result.tokens.size())};
        json_response(response, 200, q35_render::completion_json(
            completion_id, created, runtime.options.served_model,
            result.reasoning, result.content,
            request.chat.options.enable_thinking,
            result.finish_reason.c_str(), usage));
        LOG_INFO("generation completed completion_id=%s finish_reason=%s "
                 "prompt_tokens=%d cache_hit_tokens=%d completion_tokens=%d",
                 completion_id.c_str(), result.finish_reason.c_str(),
                 usage.prompt_tokens, usage.cached_tokens,
                 usage.completion_tokens);
        return;
    }

    response.set_header("Cache-Control", "no-cache");
    response.set_header("X-Accel-Buffering", "no");
    response.status = 200;
    response.set_chunked_content_provider(
        "text/event-stream; charset=utf-8",
        [&runtime, request = std::move(request), prompt = std::move(prompt),
         lease, completion_id, created](size_t offset,
                                        httplib::DataSink& sink) mutable {
            if (offset != 0) return false;
            auto send = [&](const std::string& text) {
                const std::string event = "data: " + text + "\n\n";
                return sink.is_writable() && sink.write(event.data(), event.size());
            };
            if (!send(q35_render::completion_chunk_json(
                    completion_id, created, runtime.options.served_model,
                    "role", "assistant"))) return false;

            GenerationResult result;
            std::string error;
            const Publish publish = [&](const char* field,
                                        const std::string& text) {
                return send(q35_render::completion_chunk_json(
                    completion_id, created, runtime.options.served_model,
                    field, text));
            };
            if (!generate(runtime, request, prompt.tokens, lease->session,
                          publish, &result, &error)) {
                if (error != "client disconnected" && sink.is_writable()) {
                    send(q35_render::error_json(error, "server_error"));
                    send("[DONE]");
                }
                return false;
            }
            if (!send(q35_render::completion_chunk_json(
                    completion_id, created, runtime.options.served_model,
                    nullptr, "", result.finish_reason.c_str()))) return false;
            q35_render::CompletionUsage usage{
                static_cast<int>(prompt.tokens.size()), result.cached_tokens,
                static_cast<int>(result.tokens.size())};
            if (request.include_usage &&
                !send(q35_render::completion_usage_chunk_json(
                    completion_id, created, runtime.options.served_model,
                    usage))) return false;
            if (!send("[DONE]")) return false;
            lease->keep = true;
            sink.done();
            LOG_INFO("generation completed completion_id=%s finish_reason=%s "
                     "prompt_tokens=%d cache_hit_tokens=%d completion_tokens=%d",
                     completion_id.c_str(), result.finish_reason.c_str(),
                     usage.prompt_tokens, usage.cached_tokens,
                     usage.completion_tokens);
            return true;
        });
}

bool initialize(Runtime* runtime, std::string* error) {
    std::string render_error;
    runtime->renderer.reset(q35_render::Renderer::create(
        runtime->options.render_path, &render_error));
    if (!runtime->renderer) {
        *error = render_error;
        return false;
    }
    std::vector<int> think_end;
    if (!runtime->renderer->encode("</think>", &think_end, &render_error) ||
        think_end.size() != 1) {
        *error = "renderer has no single </think> token";
        return false;
    }
    runtime->think_end_token = think_end[0];

    char native_error[ERROR_SIZE]{};
    q35_engine_options engine_options{
        runtime->options.model_path.c_str(), runtime->options.mock};
    if (q35_engine_create(&engine_options, &runtime->engine,
                          native_error, sizeof(native_error)) != Q35_OK) {
        *error = native_error;
        return false;
    }
    if (q35_session_manager_create(
            runtime->engine, runtime->options.slots, runtime->options.context,
            &runtime->manager, native_error, sizeof(native_error)) != Q35_OK) {
        *error = native_error;
        return false;
    }
    return true;
}

int run_prompt(Runtime& runtime, std::string* error) {
    q35_render::CompletionRequest request;
    request.model = runtime.options.served_model;
    request.max_tokens = runtime.options.max_tokens;
    request.temperature = 0.0f;
    q35_render::Message message;
    message.role = q35_render::Role::User;
    message.content = runtime.options.prompt;
    request.chat.messages.push_back(std::move(message));

    q35_render::RenderedPrompt prompt;
    if (!runtime.renderer->render(request.chat, &prompt, error)) return 1;
    if (prompt.tokens.empty()) {
        *error = "chat template produced an empty prompt";
        return 1;
    }
    if (prompt.tokens.size() + static_cast<size_t>(request.max_tokens) >
        static_cast<size_t>(runtime.options.context)) {
        *error = "prompt and completion exceed --session-context";
        return 1;
    }

    char native_error[ERROR_SIZE]{};
    q35_session* session = nullptr;
    if (q35_session_manager_acquire(
            runtime.manager, prompt.tokens.data(),
            static_cast<int>(prompt.tokens.size()), &session,
            native_error, sizeof(native_error)) != Q35_OK) {
        *error = native_error;
        return 1;
    }
    SessionLease lease;
    lease.manager = runtime.manager;
    lease.session = session;

    GenerationResult result;
    const Publish discard = [](const char*, const std::string&) { return true; };
    if (!generate(runtime, request, prompt.tokens, session, discard,
                  &result, error)) return 1;
    if (!result.reasoning.empty()) {
        std::fwrite(result.reasoning.data(), 1, result.reasoning.size(), stdout);
        std::fputc('\n', stdout);
    }
    std::fwrite(result.content.data(), 1, result.content.size(), stdout);
    if (result.content.empty() || result.content.back() != '\n') {
        std::fputc('\n', stdout);
    }
    return 0;
}

int run_benchmark(const Options& options, std::string* error) {
    char native_error[ERROR_SIZE]{};
    q35_engine* engine = nullptr;
    q35_engine_options engine_options{options.model_path.c_str(), options.mock};
    if (q35_engine_create(&engine_options, &engine,
                          native_error, sizeof(native_error)) != Q35_OK) {
        *error = native_error;
        return 1;
    }

    q35_session_manager* manager = nullptr;
    if (q35_session_manager_create(
            engine, options.slots, options.context, &manager,
            native_error, sizeof(native_error)) != Q35_OK) {
        *error = native_error;
        q35_engine_destroy(engine);
        return 1;
    }

    std::vector<int> prompt(static_cast<size_t>(options.bench_prefill));
    for (int index = 0; index < options.bench_prefill; ++index) {
        prompt[index] = 100 + index % 1000;
    }

    q35_session* session = nullptr;
    if (q35_session_manager_acquire(
            manager, prompt.data(), options.bench_prefill, &session,
            native_error, sizeof(native_error)) != Q35_OK) {
        *error = native_error;
        q35_session_manager_destroy(manager);
        q35_engine_destroy(engine);
        return 1;
    }

    const auto prefill_started = std::chrono::steady_clock::now();
    int status = q35_session_sync(
        session, prompt.data(), options.bench_prefill, options.bench_prefill,
        nullptr, native_error, sizeof(native_error));
    const auto decode_started = std::chrono::steady_clock::now();
    if (status == Q35_OK) {
        for (int index = 0; index < options.bench_decode; ++index) {
            status = q35_session_eval(session, 100,
                                      native_error, sizeof(native_error));
            if (status != Q35_OK) break;
        }
    }
    const auto finished = std::chrono::steady_clock::now();

    q35_session_manager_release(manager, session, false);
    q35_session_manager_destroy(manager);
    q35_engine_destroy(engine);
    if (status != Q35_OK) {
        *error = native_error;
        return 1;
    }

    const double prefill_seconds = std::chrono::duration<double>(
        decode_started - prefill_started).count();
    const double decode_seconds = std::chrono::duration<double>(
        finished - decode_started).count();
    std::printf("prefill  %d tokens  %.3f s  %.3f tok/s\n",
                options.bench_prefill, prefill_seconds,
                options.bench_prefill / prefill_seconds);
    std::printf("decode   %d tokens  %.3f s  %.3f tok/s\n",
                options.bench_decode, decode_seconds,
                options.bench_decode / decode_seconds);
    return 0;
}

int serve(Runtime& runtime, std::string* error) {
    httplib::Server server;
    const int workers = std::max(8, runtime.options.slots + 4);
    server.new_task_queue = [workers] {
        return new httplib::ThreadPool(workers, workers);
    };
    server.set_payload_max_length(MAX_REQUEST_BYTES);
    server.set_read_timeout(30, 0);
    server.set_write_timeout(runtime.options.request_timeout, 0);
    server.set_keep_alive_timeout(5);
    server.set_keep_alive_max_count(10);
    server.set_tcp_nodelay(true);

    server.Get("/healthz", [](const httplib::Request&,
                              httplib::Response& response) {
        request_id(response);
        json_response(response, 200, "{\"status\":\"ok\"}");
    });
    server.Get("/readyz", [&](const httplib::Request&,
                              httplib::Response& response) {
        request_id(response);
        const std::string body = "{\"status\":\"ready\",\"slots\":" +
            std::to_string(runtime.options.slots) + ",\"context_size\":" +
            std::to_string(runtime.options.context) +
            ",\"request_timeout\":" +
            std::to_string(runtime.options.request_timeout) +
            ",\"compute\":\"" +
            (runtime.options.mock ? "mock" : "real") + "\"}";
        json_response(response, 200, body);
    });
    server.Get("/v1/models", [&](const httplib::Request& request,
                                 httplib::Response& response) {
        request_id(response);
        if (!authenticate(request, response)) return;
        json_response(response, 200,
                      q35_render::models_json(runtime.options.served_model));
    });
    server.Post("/v1/chat/completions", [&](const httplib::Request& request,
                                            httplib::Response& response) {
        const std::string request_id = make_id("req-");
        response.set_header("X-Request-Id", request_id);
        const auto started = std::chrono::steady_clock::now();
        LOG_INFO("request started request_id=%s method=POST path=%s",
                 request_id.c_str(), request.path.c_str());
        chat(runtime, request, response);
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        LOG_INFO("request accepted request_id=%s status=%d elapsed=%.3fs",
                 request_id.c_str(), response.status, elapsed);
    });
    server.set_error_handler([](const httplib::Request&,
                                httplib::Response& response) {
        request_id(response);
        if (response.body.empty()) {
            api_error(response, response.status, "endpoint not found");
        }
    });

    LOG_INFO("server ready model=%s host=%s port=%d slots=%d context=%d",
             runtime.options.served_model.c_str(), runtime.options.host.c_str(),
             runtime.options.port, runtime.options.slots, runtime.options.context);
    if (!server.listen(runtime.options.host, runtime.options.port)) {
        *error = "cannot listen on " + runtime.options.host + ":" +
                 std::to_string(runtime.options.port);
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    options.model_path = sibling_file(argv[0], DEFAULT_MODEL);
    options.render_path = sibling_file(argv[0], DEFAULT_RENDER);
    std::string error;
    if (!parse_options(argc, argv, &options, &error)) {
        usage(argv[0]);
        std::fprintf(stderr, "qwen35: %s\n", error.c_str());
        return 2;
    }

    char log_error[ERROR_SIZE]{};
    if (!q35_internal::log_configure(
            options.log_level,
            options.log_file.empty() ? nullptr : options.log_file.c_str(),
            options.log_max_bytes, options.log_backups,
            log_error, sizeof(log_error))) {
        std::fprintf(stderr, "qwen35: %s\n", log_error);
        return 1;
    }

    int result = 1;
    if (options.mode == Mode::Bench) {
        result = run_benchmark(options, &error);
    } else {
        Runtime runtime;
        runtime.options = std::move(options);
        if (initialize(&runtime, &error)) {
            if (runtime.options.mode == Mode::Prompt) {
                result = run_prompt(runtime, &error);
            } else {
                result = serve(runtime, &error);
            }
        }
    }
    if (result != 0) LOG_ERROR("%s", error.c_str());
    q35_internal::log_shutdown();
    return result;
}
