// main.cpp -- one-process Qwen3.5 inference program.
//
//   qwen35 -p "raw prompt"
//   qwen35 -c "chat message"
//   qwen35 -l --host 127.0.0.1 --port 8000
//   qwen35 --bench PREFILL_TOKENS DECODE_TOKENS
//
// Prompt mode tokenizes raw text. Chat mode constructs a ChatRequest directly.
// Listen mode adds the HTTP/JSON boundary. Bench mode bypasses both renderer and
// HTTP to measure Session only.
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
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#define CPPHTTPLIB_NO_EXCEPTIONS
#include "third_party/httplib/httplib.h"

#include "internal.h"
#include "render.h"

namespace {

constexpr size_t ERROR_SIZE = 512;
constexpr size_t MAX_REQUEST_BYTES = 16 * 1024 * 1024;
constexpr const char* DEFAULT_MODEL = "qwen35-0.8b-model.bin";
constexpr const char* DEFAULT_RENDER = "qwen35-0.8b-render.bin";

enum class Mode { None, Prompt, Chat, Listen, Bench };

struct Options {
    Mode mode = Mode::None;
    std::string model_path;
    std::string render_path;
    std::string prompt;
    std::string host = "127.0.0.1";
    std::string served_model = "qwen3.5-0.8b";
    std::string log_file;
    std::string audit_file;
    std::string logits_output_dir = "data";
    q35_log_level log_level = Q35_LOG_ERROR;
    int port = 8000;
    int slots = 1;
    int context = 40960;
    int max_tokens = 128;
    int bench_prefill = 0;
    int bench_decode = 0;
    int request_timeout = 600;
    size_t log_max_bytes = 20 * 1024 * 1024;
    size_t log_backups = 5;
    bool context_set = false;
    bool slots_set = false;
    bool log_level_set = false;
    bool served_model_set = false;
    bool save_logits = false;
    bool logits_output_dir_set = false;
    bool mock = false;
};

struct Runtime {
    Options options;
    q35_engine* engine = nullptr;
    q35_session_manager* manager = nullptr;
    std::unique_ptr<q35_render::Renderer> renderer;
    int think_end_token = -1;
    std::atomic<int> inflight{0};

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
    std::string stop_cause = "max_tokens";
    int stop_token = -1;
    bool retryable = false;
    int cached_tokens = 0;
    double prefill_seconds = 0.0;
    double ttft_seconds = 0.0;
    double decode_seconds = 0.0;
    double elapsed_seconds = 0.0;
};

using Clock = std::chrono::steady_clock;

double elapsed_since(Clock::time_point started) {
    return std::chrono::duration<double>(Clock::now() - started).count();
}

std::string retryable_error(const std::string& message) {
    return message + "; please retry your request";
}

void audit_record(const char* event, const std::string& request_id,
                  const std::string& session_id,
                  const std::string& detail = {},
                  const std::string& payload = {}) {
    q35_internal::audit_write(
        event, request_id.c_str(), session_id.empty() ? "-" : session_id.c_str(),
        detail.c_str(), payload.data(), payload.size());
}

void audit_response_end(const std::string& request_id,
                        const std::string& session_id, int status,
                        const char* result, const char* stage,
                        size_t chunks, size_t bytes) {
    char detail[256];
    std::snprintf(detail, sizeof(detail),
                  "status=%d result=%s stage=%s chunks=%zu response_bytes=%zu",
                  status, result, stage, chunks, bytes);
    audit_record("response_end", request_id, session_id, detail);
}

struct AccessLog {
    std::string request_id;
    std::string completion_id = "-";
    std::string session_id;
    std::string remote;
    std::string method;
    std::string path;
    std::string model = "-";
    std::string finish_reason = "-";
    std::string stage = "auth";
    Clock::time_point started;
    std::atomic<int>* inflight = nullptr;
    int inflight_start = 0;
    int slots = 0;
    int status = 500;
    size_t request_bytes = 0;
    size_t response_bytes = 0;
    size_t response_chunks = 0;
    size_t messages = 0;
    size_t prompt_tokens = 0;
    int cached_tokens = 0;
    int completion_tokens = 0;
    bool stream = false;
    double parse_ms = 0.0;
    double render_ms = 0.0;
    double prefill_ms = 0.0;
    double ttft_ms = 0.0;
    double decode_tps = 0.0;
    std::atomic<bool> completed{false};

    void complete(const char* result, const char* stage) {
        if (completed.exchange(true)) return;
        const int inflight_end = inflight->fetch_sub(1) - 1;
        audit_response_end(request_id, session_id, status, result, stage,
                           response_chunks, response_bytes);
        LOG_INFO("access completed request_id=%s completion_id=%s "
                 "session_id=%s remote=%s "
                 "method=%s path=%s status=%d result=%s stage=%s model=%s "
                 "stream=%d inflight_start=%d inflight_end=%d slots=%d "
                 "request_bytes=%zu response_bytes=%zu messages=%zu "
                 "prompt_tokens=%zu cached_tokens=%d completion_tokens=%d "
                 "finish_reason=%s parse_ms=%.3f render_ms=%.3f "
                 "prefill_ms=%.3f ttft_ms=%.3f duration_ms=%.3f "
                 "decode_tps=%.2f",
                 request_id.c_str(), completion_id.c_str(),
                 session_id.empty() ? "-" : session_id.c_str(), remote.c_str(),
                 method.c_str(), path.c_str(), status, result, stage,
                 model.c_str(), stream, inflight_start, inflight_end, slots,
                 request_bytes, response_bytes, messages, prompt_tokens,
                 cached_tokens, completion_tokens, finish_reason.c_str(),
                 parse_ms, render_ms, prefill_ms, ttft_ms,
                 elapsed_since(started) * 1000.0, decode_tps);
    }

    ~AccessLog() {
        complete("aborted", "server");
    }
};

void record_generation(AccessLog* access, const GenerationResult& result) {
    access->cached_tokens = result.cached_tokens;
    access->completion_tokens = static_cast<int>(result.tokens.size());
    access->finish_reason = result.finish_reason;
    access->prefill_ms = result.prefill_seconds * 1000.0;
    access->ttft_ms = result.ttft_seconds * 1000.0;
    access->decode_tps = result.decode_seconds > 0.0
        ? access->completion_tokens / result.decode_seconds : 0.0;
}

void audit_generation(const AccessLog& access, const GenerationResult& result,
                      const std::string& error) {
    char detail[1024];
    std::snprintf(detail, sizeof(detail),
                  "completion_id=%s tokens=%zu stop_cause=%s stop_token=%d "
                  "retryable=%d cached_tokens=%d error=%s reasoning_bytes=%zu "
                  "content_bytes=%zu",
                  access.completion_id.c_str(), result.tokens.size(),
                  result.stop_cause.c_str(), result.stop_token,
                  result.retryable, result.cached_tokens,
                  error.empty() ? "-" : error.c_str(),
                  result.reasoning.size(), result.content.size());
    std::string payload = "reasoning:\n" + result.reasoning +
                          "\ncontent:\n" + result.content;
    audit_record("generation_end", access.request_id, access.session_id,
                 detail, payload);
}

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

std::filesystem::path executable_path(const char* program) {
    std::error_code error;
#if defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> path(size);
    if (_NSGetExecutablePath(path.data(), &size) == 0) {
        const auto resolved = std::filesystem::canonical(path.data(), error);
        return error ? std::filesystem::path(path.data()) : resolved;
    }
    return std::filesystem::absolute(program, error);
#else
    std::filesystem::path executable =
        std::filesystem::read_symlink("/proc/self/exe", error);
    if (error) {
        error.clear();
        executable = std::filesystem::absolute(program, error);
        if (error) executable = program;
    }
    return executable;
#endif
}

std::string sibling_file(const char* program, const char* name) {
    return (executable_path(program).parent_path() / name).string();
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
        if (argument == "--save-logits") {
            options->save_logits = true;
            continue;
        }
        if (argument == "-l" || argument == "--listen") {
            if (options->mode != Mode::None) {
                *error = "choose exactly one of --prompt, --chat, --listen, or --bench";
                return false;
            }
            options->mode = Mode::Listen;
            continue;
        }
        if (argument == "--bench") {
            if (options->mode != Mode::None) {
                *error = "choose exactly one of --prompt, --chat, --listen, or --bench";
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
                *error = "choose exactly one of --prompt, --chat, --listen, or --bench";
                return false;
            }
            options->mode = Mode::Prompt;
            options->prompt = value;
        } else if (argument == "-c" || argument == "--chat") {
            if (options->mode != Mode::None) {
                *error = "choose exactly one of --prompt, --chat, --listen, or --bench";
                return false;
            }
            options->mode = Mode::Chat;
            options->prompt = value;
        } else if (argument == "-m" || argument == "--model") {
            options->model_path = value;
        } else if (argument == "-r" || argument == "--render") {
            options->render_path = value;
        } else if (argument == "--host") {
            options->host = value;
        } else if (argument == "--served-model-name") {
            options->served_model = value;
            options->served_model_set = true;
        } else if (argument == "--log-file") {
            options->log_file = value;
        } else if (argument == "--audit-log") {
            options->audit_file = value;
        } else if (argument == "--logits-output-dir") {
            options->logits_output_dir = value;
            options->logits_output_dir_set = true;
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
            if (value == "trace") options->log_level = Q35_LOG_TRACE;
            else if (value == "debug") options->log_level = Q35_LOG_DEBUG;
            else if (value == "info") options->log_level = Q35_LOG_INFO;
            else if (value == "warn" || value == "warning") {
                options->log_level = Q35_LOG_WARN;
            } else if (value == "error") options->log_level = Q35_LOG_ERROR;
            else {
                *error = "invalid --log-level";
                return false;
            }
            options->log_level_set = true;
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
        *error = "one of -p/--prompt, -c/--chat, -l/--listen, or --bench is required";
        return false;
    }
    if (options->mode != Mode::Bench && options->render_path.empty()) {
        *error = "-r/--render is required";
        return false;
    }
    if (!options->audit_file.empty() && options->mode != Mode::Listen) {
        *error = "--audit-log requires --listen";
        return false;
    }
    if (options->save_logits && options->mode != Mode::Prompt) {
        *error = "--save-logits requires -p/--prompt";
        return false;
    }
    if (options->logits_output_dir_set && !options->save_logits) {
        *error = "--logits-output-dir requires --save-logits";
        return false;
    }
    if (!options->log_level_set && options->mode == Mode::Listen) {
        options->log_level = Q35_LOG_INFO;
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
        "  %s [-m MODEL] [-r RENDER] -c MESSAGE [--max-tokens N]\n"
        "  %s [-m MODEL] [-r RENDER] -p PROMPT --save-logits\n"
        "     [--logits-output-dir DIR]\n"
        "  %s [-m MODEL] [-r RENDER] -l [--host HOST] [--port PORT]\n"
        "  %s [-m MODEL] --bench PREFILL_TOKENS DECODE_TOKENS\n"
        "common: [--session-slots N] [--session-context N] [--mock]\n"
        "        [--log-level trace|debug|info|warn|error] [--log-file PATH]\n"
        "listen: [--audit-log PATH]\n"
        "defaults: 1 session slot, 40960-token context\n"
        "        log default: info for --listen, error otherwise\n"
        "defaults beside qwen35: %s and %s\n", program, program, program,
        program, program,
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

std::string session_id(const q35_session* session) {
    const char* id = q35_session_id(session);
    return id ? id : "-";
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
    if (!response.has_header("X-Request-Id")) {
        response.set_header("X-Request-Id", make_id("req-"));
    }
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
        result->stop_cause = "generation_error";
        *error = native_error;
        return false;
    }
    result->prefill_seconds = elapsed_since(started);

    uint64_t rng = request.has_seed ? request.seed : random_seed();
    std::vector<int> penalty_tokens;
    std::unordered_set<int> penalty_seen;
    std::string published_reasoning;
    std::string published_content;

    while (static_cast<int>(result->tokens.size()) < request.max_tokens) {
        if (std::chrono::steady_clock::now() - started >
            std::chrono::seconds(runtime.options.request_timeout)) {
            result->stop_cause = "timeout";
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
            result->stop_cause = "generation_error";
            *error = "sampling failed";
            return false;
        }
        if (q35_token_is_stop(token)) {
            const bool thinking_open =
                request.chat.options.enable_thinking &&
                std::find(result->tokens.begin(), result->tokens.end(),
                          runtime.think_end_token) == result->tokens.end();
            if (thinking_open || result->tokens.empty() ||
                (request.chat.options.enable_thinking &&
                 result->content.empty())) {
                result->stop_cause = "incomplete_generation";
                result->stop_token = token;
                result->retryable = true;
                *error = "model stopped before completing its response";
                return false;
            }
            result->finish_reason = "stop";
            result->stop_cause = "model_stop_token";
            result->stop_token = token;
            break;
        }

        result->tokens.push_back(token);
        if (result->tokens.size() == 1) {
            result->ttft_seconds = elapsed_since(started);
        }
        if (penalty_seen.insert(token).second) penalty_tokens.push_back(token);
        if (!split_output(runtime, request, result->tokens,
                          &result->reasoning, &result->content, error)) {
            result->stop_cause = "generation_error";
            return false;
        }

        const std::string reasoning = stable_text(result->reasoning);
        const auto visible = stop_view(result->content, request.stops, false);
        const std::string content = stable_text(visible.first);
        if (reasoning.compare(0, published_reasoning.size(),
                              published_reasoning) != 0 ||
            content.compare(0, published_content.size(),
                            published_content) != 0) {
            *error = "tokenizer rewrote streamed text";
            result->stop_cause = "generation_error";
            return false;
        }
        if (reasoning.size() > published_reasoning.size() &&
            !publish("reasoning_content",
                     reasoning.substr(published_reasoning.size()))) {
            *error = "client disconnected";
            result->stop_cause = "client_disconnected";
            return false;
        }
        if (content.size() > published_content.size() &&
            !publish("content", content.substr(published_content.size()))) {
            *error = "client disconnected";
            result->stop_cause = "client_disconnected";
            return false;
        }
        published_reasoning = reasoning;
        published_content = content;

        if (visible.second) {
            result->finish_reason = "stop";
            result->stop_cause = "request_stop_string";
            break;
        }
        if (static_cast<int>(result->tokens.size()) == request.max_tokens) break;
        if (q35_session_eval(session, token,
                             native_error, sizeof(native_error)) != Q35_OK) {
            result->stop_cause = "generation_error";
            *error = native_error;
            return false;
        }
    }

    if (!split_output(runtime, request, result->tokens,
                      &result->reasoning, &result->content, error)) {
        result->stop_cause = "generation_error";
        return false;
    }
    const auto final_content = stop_view(result->content, request.stops, true);
    result->content = final_content.first;
    if (final_content.second) {
        result->finish_reason = "stop";
        result->stop_cause = "request_stop_string";
    }
    if (result->reasoning.compare(0, published_reasoning.size(),
                                  published_reasoning) != 0 ||
        result->content.compare(0, published_content.size(),
                                published_content) != 0) {
        *error = "tokenizer rewrote streamed text";
        result->stop_cause = "generation_error";
        return false;
    }
    if (result->reasoning.size() > published_reasoning.size() &&
        !publish("reasoning_content",
                 result->reasoning.substr(published_reasoning.size()))) {
        *error = "client disconnected";
        result->stop_cause = "client_disconnected";
        return false;
    }
    if (result->content.size() > published_content.size() &&
        !publish("content", result->content.substr(published_content.size()))) {
        *error = "client disconnected";
        result->stop_cause = "client_disconnected";
        return false;
    }
    result->elapsed_seconds = elapsed_since(started);
    result->decode_seconds =
        result->elapsed_seconds - result->prefill_seconds;
    return true;
}

bool prepare_request(Runtime& runtime, AccessLog* access,
                     const httplib::Request& http,
                     httplib::Response& response,
                     q35_render::CompletionRequest* request,
                     q35_render::RenderedPrompt* prompt) {
    const auto started = Clock::now();
    access->stage = "parse";
    const q35_render::Status status = q35_render::parse_completion_request(
        http.body, runtime.options.served_model,
        runtime.options.max_tokens, *request);
    if (!status.ok()) {
        const bool missing_model = status.message().find("model not found") !=
                                   std::string::npos;
        const int http_status = missing_model ? 404 : 400;
        LOG_WARN("request rejected request_id=%s stage=parse status=%d error=%s",
                 access->request_id.c_str(), http_status,
                 status.message().c_str());
        api_error(response, http_status, status.message(),
                  "invalid_request_error", missing_model ? "model" : nullptr,
                  missing_model ? "model_not_found" : nullptr);
        return false;
    }
    const double parse_elapsed = elapsed_since(started);
    access->parse_ms = parse_elapsed * 1000.0;
    access->model = request->model;
    access->messages = request->chat.messages.size();
    access->stream = request->stream;
    LOG_DEBUG("request parsed request_id=%s model=%s messages=%zu tools=%zu "
              "stream=%d max_tokens=%d temperature=%.3f elapsed=%.3fs",
              access->request_id.c_str(), request->model.c_str(),
              request->chat.messages.size(), request->chat.tools.size(),
              request->stream, request->max_tokens, request->temperature,
              parse_elapsed);

    access->stage = "render";
    const auto render_started = Clock::now();
    std::string error;
    const bool rendered = runtime.renderer->render(request->chat, prompt, &error);
    access->render_ms = elapsed_since(render_started) * 1000.0;
    if (!rendered) {
        LOG_WARN("request rejected request_id=%s stage=render status=400 error=%s",
                 access->request_id.c_str(), error.c_str());
        api_error(response, 400, error, "invalid_request_error", "messages");
        return false;
    }
    if (prompt->tokens.empty()) {
        LOG_ERROR("request failed request_id=%s stage=render status=500 "
                  "error=empty_prompt", access->request_id.c_str());
        api_error(response, 500, "chat template produced an empty prompt",
                  "server_error", nullptr, "invalid_chat_template");
        return false;
    }
    access->prompt_tokens = prompt->tokens.size();
    access->stage = "context";
    if (prompt->tokens.size() + static_cast<size_t>(request->max_tokens) >
        static_cast<size_t>(runtime.options.context)) {
        LOG_WARN("request rejected request_id=%s stage=context status=400 "
                 "prompt_tokens=%zu max_tokens=%d context=%d",
                 access->request_id.c_str(), prompt->tokens.size(),
                 request->max_tokens, runtime.options.context);
        api_error(response, 400, "prompt and completion exceed the Session context",
                  "invalid_request_error", "max_completion_tokens",
                  "context_length_exceeded");
        return false;
    }
    LOG_INFO("request prepared request_id=%s messages=%zu prompt_bytes=%zu "
             "prompt_tokens=%zu stream=%d thinking=%d parse_ms=%.3f "
             "render_ms=%.3f elapsed_ms=%.3f",
             access->request_id.c_str(), request->chat.messages.size(),
             prompt->text.size(), prompt->tokens.size(), request->stream,
             request->chat.options.enable_thinking, access->parse_ms,
             access->render_ms,
             elapsed_since(started) * 1000.0);
    access->stage = "prepared";
    return true;
}

q35_session* acquire(Runtime& runtime, const std::string& request_id,
                     const std::vector<int>& prompt,
                     httplib::Response& response) {
    const auto started = Clock::now();
    q35_session* session = nullptr;
    char error[ERROR_SIZE]{};
    const int status = q35_session_manager_acquire(
        runtime.manager, prompt.data(), static_cast<int>(prompt.size()),
        &session, error, sizeof(error));
    if (status == Q35_BUSY) {
        LOG_WARN("request rejected request_id=%s stage=session status=429 "
                 "error=%s", request_id.c_str(),
                 error[0] ? error : "all sessions are busy");
        api_error(response, 429, error[0] ? error : "all sessions are busy",
                  "rate_limit_error", nullptr, "engine_busy");
        return nullptr;
    }
    if (status != Q35_OK) {
        LOG_ERROR("request failed request_id=%s stage=session status=500 "
                  "error=%s", request_id.c_str(),
                  error[0] ? error : "cannot acquire Session");
        api_error(response, 500, error[0] ? error : "cannot acquire Session",
                  "server_error");
        return nullptr;
    }
    LOG_DEBUG("session acquired request_id=%s session_id=%s prompt_tokens=%zu "
              "elapsed=%.3fs",
              request_id.c_str(), session_id(session).c_str(), prompt.size(),
              elapsed_since(started));
    return session;
}

bool chat(Runtime& runtime, const std::shared_ptr<AccessLog>& access,
          const httplib::Request& http, httplib::Response& response) {
    if (!authenticate(http, response)) {
        LOG_WARN("request rejected request_id=%s stage=auth status=401",
                 access->request_id.c_str());
        return false;
    }
    q35_render::CompletionRequest request;
    q35_render::RenderedPrompt prompt;
    if (!prepare_request(runtime, access.get(), http, response,
                         &request, &prompt)) return false;
    audit_record("render", access->request_id, "",
                 "prompt_tokens=" + std::to_string(prompt.tokens.size()),
                 prompt.text);

    access->stage = "session";
    q35_session* session = acquire(
        runtime, access->request_id, prompt.tokens, response);
    if (!session) return false;
    auto lease = std::make_shared<SessionLease>();
    lease->manager = runtime.manager;
    lease->session = session;
    access->session_id = session_id(session);
    response.set_header("X-Session-Id", access->session_id);
    audit_record("session_acquired", access->request_id, access->session_id);

    const std::string completion_id = make_id("chatcmpl-");
    access->completion_id = completion_id;
    access->stage = "generation";
    const int64_t created = static_cast<int64_t>(std::time(nullptr));
    LOG_INFO("generation started request_id=%s completion_id=%s session_id=%s "
             "prompt_tokens=%zu max_tokens=%d stream=%d",
             access->request_id.c_str(), completion_id.c_str(),
             access->session_id.c_str(),
             prompt.tokens.size(), request.max_tokens, request.stream);

    if (!request.stream) {
        GenerationResult result;
        std::string error;
        const Publish discard = [](const char*, const std::string&) { return true; };
        if (!generate(runtime, request, prompt.tokens, session, discard,
                      &result, &error)) {
            audit_generation(*access, result, error);
            record_generation(access.get(), result);
            lease->keep = result.retryable;
            LOG_ERROR("request failed request_id=%s completion_id=%s "
                      "stage=generation status=500 error=%s",
                      access->request_id.c_str(), completion_id.c_str(),
                      error.c_str());
            api_error(response, 500,
                      result.retryable ? retryable_error(error) : error,
                      "server_error", nullptr,
                      result.retryable ? "incomplete_generation" : nullptr);
            return false;
        }
        audit_generation(*access, result, "");
        lease->keep = true;
        q35_render::CompletionUsage usage{
            static_cast<int>(prompt.tokens.size()), result.cached_tokens,
            static_cast<int>(result.tokens.size())};
        std::vector<q35_render::ToolCall> tool_calls;
        std::string content = result.content;
        if (!request.chat.tools.empty()) {
            const bool parsed = q35_render::parse_generated_tool_calls(
                result.content, &content, &tool_calls, &error);
            const std::string detail = "completion_id=" + completion_id +
                " ok=" + (parsed ? "1" : "0") +
                " error=" + (error.empty() ? "-" : error);
            audit_record("tool_parse", access->request_id,
                         access->session_id, detail);
            if (!parsed) {
                LOG_WARN("tool output not parsed request_id=%s completion_id=%s "
                         "error=%s",
                         access->request_id.c_str(), completion_id.c_str(),
                         error.c_str());
                record_generation(access.get(), result);
                api_error(response, 500, retryable_error(error),
                          "server_error", nullptr, "incomplete_tool_call");
                return false;
            }
        }
        for (q35_render::ToolCall& call : tool_calls) {
            call.id = make_id("call_");
        }
        if (!tool_calls.empty()) result.finish_reason = "tool_calls";
        json_response(response, 200, q35_render::completion_json(
            completion_id, created, runtime.options.served_model,
            result.reasoning, content,
            request.chat.options.enable_thinking,
            tool_calls,
            result.finish_reason.c_str(), usage, &request.chat.tools));
        record_generation(access.get(), result);
        return false;
    }

    response.set_header("Cache-Control", "no-cache");
    response.set_header("X-Accel-Buffering", "no");
    response.status = 200;
    response.set_chunked_content_provider(
        "text/event-stream; charset=utf-8",
        [&runtime, request = std::move(request), prompt = std::move(prompt),
         lease, access, completion_id,
         created](size_t offset, httplib::DataSink& sink) mutable {
            if (offset != 0) return false;
            audit_record("response_start", access->request_id,
                         access->session_id, "status=200 stream=1");
            auto release = [&](bool keep) {
                if (!lease->session) return;
                q35_session_manager_release(
                    lease->manager, lease->session, keep);
                lease->session = nullptr;
            };
            auto interrupted = [&](const char* stage) {
                LOG_WARN("request interrupted request_id=%s completion_id=%s "
                         "stage=%s elapsed=%.3fs", access->request_id.c_str(),
                         completion_id.c_str(), stage,
                         elapsed_since(access->started));
                release(false);
                access->status = 200;
                access->stage = stage;
                access->complete("disconnected", stage);
            };
            auto send = [&](const std::string& text) {
                const std::string event = "data: " + text + "\n\n";
                if (!sink.is_writable() ||
                    !sink.write(event.data(), event.size())) {
                    audit_record(
                        "response_write_failed", access->request_id,
                        access->session_id,
                        "seq=" + std::to_string(access->response_chunks),
                        event);
                    return false;
                }
                audit_record(
                    "response_chunk", access->request_id, access->session_id,
                    "seq=" + std::to_string(access->response_chunks), event);
                ++access->response_chunks;
                access->response_bytes += event.size();
                return true;
            };
            if (!send(q35_render::completion_chunk_json(
                    completion_id, created, runtime.options.served_model,
                    "role", "assistant"))) {
                interrupted("stream_start");
                return false;
            }

            GenerationResult result;
            std::string error;
            const bool buffer_for_tools = !request.chat.tools.empty();
            const Publish publish = [&](const char* field,
                                        const std::string& text) {
                if (buffer_for_tools) return sink.is_writable();
                return send(q35_render::completion_chunk_json(
                    completion_id, created, runtime.options.served_model,
                    field, text));
            };
            if (!generate(runtime, request, prompt.tokens, lease->session,
                          publish, &result, &error)) {
                audit_generation(*access, result, error);
                record_generation(access.get(), result);
                if (error == "client disconnected") {
                    interrupted("generation");
                } else {
                    LOG_ERROR("request failed request_id=%s completion_id=%s "
                              "stage=generation status=500 error=%s",
                              access->request_id.c_str(), completion_id.c_str(),
                              error.c_str());
                }
                if (error == "client disconnected") return false;
                if (!sink.is_writable() || !send(q35_render::error_json(
                        result.retryable ? retryable_error(error) : error,
                        "server_error", nullptr,
                        result.retryable ? "incomplete_generation" : nullptr))) {
                    interrupted("stream_error");
                    return false;
                }
                sink.done();
                release(result.retryable);
                access->status = 200;
                access->stage = "generation";
                access->complete("error", "generation");
                return true;
            }
            audit_generation(*access, result, "");
            if (buffer_for_tools) {
                std::vector<q35_render::ToolCall> tool_calls;
                std::string content;
                const bool parsed = q35_render::parse_generated_tool_calls(
                    result.content, &content, &tool_calls, &error);
                const std::string detail = "completion_id=" + completion_id +
                    " ok=" + (parsed ? "1" : "0") +
                    " error=" + (error.empty() ? "-" : error);
                audit_record("tool_parse", access->request_id,
                             access->session_id, detail);
                if (!parsed) {
                    LOG_WARN("stream tool output not parsed request_id=%s "
                             "completion_id=%s error=%s",
                             access->request_id.c_str(), completion_id.c_str(),
                             error.c_str());
                    record_generation(access.get(), result);
                    if (!send(q35_render::error_json(
                            retryable_error(error), "server_error", nullptr,
                            "incomplete_tool_call"))) {
                        interrupted("tool_parse_error");
                        return false;
                    }
                    sink.done();
                    release(true);
                    access->status = 200;
                    access->stage = "tool_parse";
                    access->complete("error", "tool_parse");
                    return true;
                }
                if (request.chat.options.enable_thinking &&
                    !result.reasoning.empty() &&
                    !send(q35_render::completion_chunk_json(
                        completion_id, created, runtime.options.served_model,
                        "reasoning_content", result.reasoning))) {
                    interrupted("stream_reasoning");
                    return false;
                }
                if (!content.empty() &&
                    !send(q35_render::completion_chunk_json(
                        completion_id, created, runtime.options.served_model,
                        "content", content))) {
                    interrupted("stream_content");
                    return false;
                }
                for (size_t index = 0; index < tool_calls.size(); ++index) {
                    tool_calls[index].id = make_id("call_");
                    if (!send(q35_render::completion_tool_call_chunk_json(
                            completion_id, created,
                            runtime.options.served_model,
                            static_cast<int>(index), tool_calls[index],
                            &request.chat.tools))) {
                        interrupted("stream_tool_call");
                        return false;
                    }
                }
                if (!tool_calls.empty()) result.finish_reason = "tool_calls";
            }
            record_generation(access.get(), result);
            if (!send(q35_render::completion_chunk_json(
                    completion_id, created, runtime.options.served_model,
                    nullptr, "", result.finish_reason.c_str()))) {
                interrupted("stream_finish");
                return false;
            }
            q35_render::CompletionUsage usage{
                static_cast<int>(prompt.tokens.size()), result.cached_tokens,
                static_cast<int>(result.tokens.size())};
            if (request.include_usage &&
                !send(q35_render::completion_usage_chunk_json(
                    completion_id, created, runtime.options.served_model,
                    usage))) {
                interrupted("stream_usage");
                return false;
            }
            if (!send("[DONE]")) {
                interrupted("stream_done");
                return false;
            }
            sink.done();
            release(true);
            access->status = 200;
            access->stage = "generation";
            access->complete("ok", "generation");
            return true;
        });
    return true;
}

bool initialize(Runtime* runtime, std::string* error) {
    const auto render_started = Clock::now();
    LOG_INFO("renderer load started bin=%s",
             runtime->options.render_path.c_str());
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
    LOG_INFO("renderer load completed elapsed=%.3fs",
             elapsed_since(render_started));

    char native_error[ERROR_SIZE]{};
    q35_engine_options engine_options{
        runtime->options.model_path.c_str(), runtime->options.mock};
    if (q35_engine_create(&engine_options, &runtime->engine,
                          native_error, sizeof(native_error)) != Q35_OK) {
        *error = native_error;
        return false;
    }
    if (!runtime->options.served_model_set) {
        switch (q35_engine_model_id(runtime->engine)) {
        case 800:
            runtime->options.served_model = "qwen3.5-0.8b";
            break;
        case 4000:
            runtime->options.served_model = "qwen3.5-4b";
            break;
        case 9000:
            runtime->options.served_model = "qwen3.5-9b";
            break;
        default:
            *error = "loaded model has no served model name";
            return false;
        }
    }
    if (q35_session_manager_create(
            runtime->engine, runtime->options.slots, runtime->options.context,
            &runtime->manager, native_error, sizeof(native_error)) != Q35_OK) {
        *error = native_error;
        return false;
    }
    return true;
}

bool save_logits(Runtime& runtime, const q35_render::RenderedPrompt& prompt,
                 q35_session* session, std::string* error) {
    static_assert(sizeof(int) == 4, "token files contain int32 values");
    static_assert(sizeof(float) == 4, "logit files contain float32 values");

    char native_error[ERROR_SIZE]{};
    if (q35_session_sync(
            session, prompt.tokens.data(), static_cast<int>(prompt.tokens.size()),
            -1, nullptr, native_error, sizeof(native_error)) != Q35_OK) {
        *error = native_error;
        return false;
    }

    std::vector<float> logits(static_cast<size_t>(q35_vocab_size()));
    if (q35_session_copy_logits(
            session, logits.data(), static_cast<int>(logits.size()),
            native_error, sizeof(native_error)) != Q35_OK) {
        *error = native_error;
        return false;
    }

    const std::filesystem::path output = runtime.options.logits_output_dir;
    std::error_code filesystem_error;
    std::filesystem::create_directories(output, filesystem_error);
    if (filesystem_error) {
        *error = "cannot create logits output directory: " + output.string();
        return false;
    }
    const std::string model =
        std::filesystem::path(runtime.options.model_path).stem().string();
    const std::filesystem::path base = output / ("qwen3x-" + model);

    {
        const std::filesystem::path path = base.string() + ".bin";
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(logits.data()),
                   static_cast<std::streamsize>(logits.size() * sizeof(float)));
        if (!file) {
            *error = "cannot write logits file: " + path.string();
            return false;
        }
    }
    {
        const std::filesystem::path path = base.string() + "-tokens.bin";
        std::ofstream file(path, std::ios::binary);
        file.write(
            reinterpret_cast<const char*>(prompt.tokens.data()),
            static_cast<std::streamsize>(prompt.tokens.size() * sizeof(int)));
        if (!file) {
            *error = "cannot write token file: " + path.string();
            return false;
        }
    }

    std::printf("logits saved to %s.bin\n", base.c_str());
    std::printf("tokens saved to %s-tokens.bin\n", base.c_str());
    return true;
}

int run_cli(Runtime& runtime, std::string* error) {
    q35_render::CompletionRequest request;
    request.model = runtime.options.served_model;
    request.max_tokens = runtime.options.max_tokens;
    request.temperature = 0.0f;

    q35_render::RenderedPrompt prompt;
    if (runtime.options.mode == Mode::Prompt) {
        prompt.text = runtime.options.prompt;
        if (!runtime.renderer->encode(prompt.text, &prompt.tokens, error)) return 1;
    } else {
        q35_render::Message message;
        message.role = q35_render::Role::User;
        message.content = runtime.options.prompt;
        request.chat.messages.push_back(std::move(message));
        if (!runtime.renderer->render(request.chat, &prompt, error)) return 1;
    }
    if (prompt.tokens.empty()) {
        *error = "prompt produced no tokens";
        return 1;
    }
    const size_t completion_tokens =
        runtime.options.save_logits ? 0 : static_cast<size_t>(request.max_tokens);
    if (prompt.tokens.size() + completion_tokens >
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

    if (runtime.options.save_logits) {
        return save_logits(runtime, prompt, session, error) ? 0 : 1;
    }

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
        auto access = std::make_shared<AccessLog>();
        access->request_id = make_id("req-");
        access->remote = request.remote_addr.empty()
            ? "-" : request.remote_addr;
        access->method = request.method;
        access->path = request.path;
        access->model = runtime.options.served_model;
        access->started = Clock::now();
        access->inflight = &runtime.inflight;
        access->inflight_start = runtime.inflight.fetch_add(1) + 1;
        access->slots = runtime.options.slots;
        access->request_bytes = request.body.size();
        response.set_header("X-Request-Id", access->request_id);
        const std::string audit_detail = "remote=" + access->remote +
            " method=" + access->method + " path=" + access->path;
        audit_record("request", access->request_id, "", audit_detail,
                     request.body);
        LOG_INFO("access started request_id=%s remote=%s method=%s path=%s "
                 "inflight=%d slots=%d request_bytes=%zu",
                 access->request_id.c_str(), access->remote.c_str(),
                 access->method.c_str(), access->path.c_str(),
                 access->inflight_start, access->slots,
                 access->request_bytes);
        const bool streaming = chat(runtime, access, request, response);
        if (streaming) {
            LOG_DEBUG("stream ready request_id=%s status=%d elapsed=%.3fs",
                      access->request_id.c_str(), response.status,
                      elapsed_since(access->started));
        } else {
            access->status = response.status;
            access->response_bytes = response.body.size();
            audit_record("response_start", access->request_id,
                         access->session_id,
                         "status=" + std::to_string(response.status) +
                             " stream=0");
            if (!response.body.empty()) {
                audit_record(
                    "response_chunk", access->request_id, access->session_id,
                    "seq=" + std::to_string(access->response_chunks),
                    response.body);
                ++access->response_chunks;
            }
            const char* result = response.status >= 500 ? "error"
                : response.status >= 400 ? "rejected" : "ok";
            access->complete(result, access->stage.c_str());
        }
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
    if (!options.audit_file.empty() && !q35_internal::audit_configure(
            options.audit_file.c_str(), log_error, sizeof(log_error))) {
        std::fprintf(stderr, "qwen35: %s\n", log_error);
        q35_internal::log_shutdown();
        return 1;
    }
    if (options.mode == Mode::Bench) {
        result = run_benchmark(options, &error);
    } else {
        Runtime runtime;
        runtime.options = std::move(options);
        if (initialize(&runtime, &error)) {
            if (runtime.options.mode == Mode::Prompt ||
                runtime.options.mode == Mode::Chat) {
                result = run_cli(runtime, &error);
            } else {
                result = serve(runtime, &error);
            }
        }
    }
    if (result != 0) LOG_ERROR("%s", error.c_str());
    q35_internal::audit_shutdown();
    q35_internal::log_shutdown();
    return result;
}
