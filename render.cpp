// render.cpp -- dependency-free Qwen3.5 text boundary.
//
// This file deliberately owns both directions around the model:
//
//   messages -> fixed chat template -> UTF-8 -> ByteLevel BPE -> token IDs
//   token IDs -> raw bytes -> UTF-8 text
//
// parser.cpp turns wire JSON into the semantic ChatRequest below this file's
// boundary. scripts/pack_render.py turns the official tokenizer JSON into
// token bytes, merge-ID triples and compact Unicode tables. render.cpp itself
// therefore uses only the C++ standard library: no JSON parser, regex engine,
// ICU, Rust or Jinja interpreter. The implementation is fixed to Qwen3.5's
// tokenizer and to reference/chat_template.jinja, just like
// engine.cpp is fixed to the 0.8B model data flow.

#include "render.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace q35_render {
namespace {

// Fixed Qwen3.5 chat template. Read this section before the tokenizer below.
constexpr char IM_START[] = "<|im_start|>";
constexpr char IM_END[] = "<|im_end|>";
constexpr char THINK_START[] = "<think>";
constexpr char THINK_END[] = "</think>";
constexpr char TOOL_RESPONSE_START[] = "<tool_response>";
constexpr char TOOL_RESPONSE_END[] = "</tool_response>";

constexpr char TOOL_SYSTEM_BEGIN[] = R"QWEN(<|im_start|>system
# Tools

You have access to the following functions:

<tools>)QWEN";

constexpr char TOOL_SYSTEM_END[] = R"QWEN(
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>)QWEN";

// Standalone render.bin header. The same bytes can later live in the model pack.
constexpr std::array<char, 8> RENDER_MAGIC = {
    'Q', '3', '5', 'R', 'N', 'D', '1', '\0'
};
constexpr uint32_t RENDER_VERSION = 1;
constexpr uint32_t RENDER_HEADER_SIZE = 60;
constexpr uint64_t MAX_RENDER_SIZE = 64ull * 1024 * 1024;
constexpr uint8_t LETTER = 1;
constexpr uint8_t MARK = 2;
constexpr uint8_t NUMBER = 4;
constexpr uint8_t SPACE = 8;

uint64_t pair_key(uint32_t left, uint32_t right);
bool fail(std::string* error, const std::string& message);
bool unicode_scalar(uint32_t codepoint);
bool utf8_codepoints(const std::string& text, std::vector<uint32_t>* output,
                     std::string* error);
void append_utf8(std::string& output, uint32_t codepoint);
std::string utf8_text(const std::vector<uint32_t>& codepoints);
std::string lossy_utf8(const std::string& bytes);

}  // namespace

struct Renderer::Impl {
    struct Merge { uint32_t rank = 0, result = 0; };
    struct Range { uint32_t first = 0, last = 0; uint8_t flags = 0; };
    struct Combining { uint32_t codepoint = 0; uint8_t value = 0; };
    struct Decomposition { uint32_t codepoint = 0; std::vector<uint32_t> value; };
    struct Added { uint32_t id = 0; std::string text; bool special = false; };

    uint32_t model_vocab = 0;
    uint32_t base_vocab = 0;
    uint32_t decodable = 0;
    std::string oracle_version;
    std::string tokenizer_sha256;
    std::string config_sha256;
    std::array<uint32_t, 256> byte_ids{};
    std::vector<std::string> token_bytes;
    std::vector<uint8_t> special;
    std::vector<Added> added;
    std::unordered_map<uint64_t, Merge> merges;
    std::vector<Range> ranges;
    std::vector<Combining> combining;
    std::vector<Decomposition> decompositions;
    std::unordered_map<uint64_t, uint32_t> compositions;

    bool load(const std::string& path, std::string* error);

    // Qwen3.5 chat template -------------------------------------------------

    bool render_content(const Message& message, bool count_vision,
                        bool system, int& image_count, int& video_count,
                        bool add_vision_id, std::string* output,
                        std::string* error) const {
        output->clear();
        if (message.parts.empty()) {
            if (!message.content_is_null) *output = message.content;
            return true;
        }
        for (const ContentPart& part : message.parts) {
            if (part.kind == ContentKind::Text) {
                *output += part.text;
            } else if (part.kind == ContentKind::Image) {
                if (system) {
                    return fail(error, "System message cannot contain images.");
                }
                if (count_vision) ++image_count;
                if (add_vision_id) {
                    *output += "Picture " + std::to_string(image_count) + ": ";
                }
                *output += "<|vision_start|><|image_pad|><|vision_end|>";
            } else if (part.kind == ContentKind::Video) {
                if (system) {
                    return fail(error, "System message cannot contain videos.");
                }
                if (count_vision) ++video_count;
                if (add_vision_id) {
                    *output += "Video " + std::to_string(video_count) + ": ";
                }
                *output += "<|vision_start|><|video_pad|><|vision_end|>";
            }
        }
        return true;
    }

    bool render_trimmed_content(const Message& message, bool count_vision,
                                bool system, int& image_count, int& video_count,
                                bool add_vision_id, std::string* output,
                                std::string* error) const {
        std::string raw;
        if (!render_content(message, count_vision, system, image_count,
                            video_count, add_vision_id, &raw, error)) return false;
        return trim(raw, output, error);
    }

    bool append_system(const ChatRequest& request, int& image_count,
                       int& video_count, std::string* output,
                       std::string* error) const {
        const std::vector<Message>& messages = request.messages;
        const ChatOptions& options = request.options;

        if (!request.tools.empty()) {
            *output += TOOL_SYSTEM_BEGIN;
            for (const std::string& tool : request.tools) {
                *output += '\n';
                *output += tool;
            }
            *output += TOOL_SYSTEM_END;
            if (messages[0].role == Role::System) {
                std::string content;
                if (!render_trimmed_content(
                        messages[0], false, true, image_count, video_count,
                        options.add_vision_id, &content, error)) return false;
                if (!content.empty()) {
                    *output += "\n\n";
                    *output += content;
                }
            }
            *output += IM_END;
            *output += '\n';
        } else if (messages[0].role == Role::System) {
            std::string content;
            if (!render_trimmed_content(
                    messages[0], false, true, image_count, video_count,
                    options.add_vision_id, &content, error)) return false;
            *output += IM_START;
            *output += "system\n";
            *output += content;
            *output += IM_END;
            *output += '\n';
        }
        return true;
    }

    bool find_last_query(const ChatRequest& request, int image_count,
                         int video_count, size_t* last_query,
                         std::string* error) const {
        const std::vector<Message>& messages = request.messages;
        const size_t response_start_size = sizeof(TOOL_RESPONSE_START) - 1;
        const size_t response_end_size = sizeof(TOOL_RESPONSE_END) - 1;

        for (size_t reverse = messages.size(); reverse > 0; --reverse) {
            const size_t index = reverse - 1;
            if (messages[index].role != Role::User) continue;

            std::string content;
            if (!render_trimmed_content(
                    messages[index], false, false, image_count, video_count,
                    request.options.add_vision_id, &content, error)) return false;
            const bool tool_response =
                content.size() >= response_start_size + response_end_size &&
                content.compare(
                    0, response_start_size, TOOL_RESPONSE_START
                ) == 0 &&
                content.compare(
                    content.size() - response_end_size,
                    response_end_size, TOOL_RESPONSE_END
                ) == 0;
            if (!tool_response) {
                *last_query = index;
                return true;
            }
        }
        return fail(error, "No user query found in messages.");
    }

    void append_user(const std::string& content, std::string* output) const {
        *output += IM_START;
        *output += "user\n";
        *output += content;
        *output += IM_END;
        *output += '\n';
    }

    void append_tool_calls(const Message& message, bool has_content,
                           std::string* output) const {
        for (size_t index = 0; index < message.tool_calls.size(); ++index) {
            const ToolCall& call = message.tool_calls[index];
            if (index == 0 && has_content) *output += "\n\n";
            else if (index != 0) *output += '\n';

            *output += "<tool_call>\n<function=";
            *output += call.name;
            *output += ">\n";
            for (const ToolArgument& argument : call.arguments) {
                *output += "<parameter=";
                *output += argument.name;
                *output += ">\n";
                *output += argument.text;
                *output += "\n</parameter>\n";
            }
            *output += "</function>\n</tool_call>";
        }
    }

    bool append_assistant(const Message& message, std::string content,
                          size_t index, size_t last_query,
                          const ChatOptions& options, std::string* output,
                          std::string* error) const {
        std::string reasoning;
        if (message.has_reasoning) {
            reasoning = message.reasoning_content;
        } else {
            const size_t close = content.find(THINK_END);
            if (close != std::string::npos) {
                reasoning = content.substr(0, close);
                while (!reasoning.empty() && reasoning.back() == '\n') {
                    reasoning.pop_back();
                }
                const size_t open = reasoning.rfind(THINK_START);
                if (open != std::string::npos) {
                    reasoning.erase(0, open + sizeof(THINK_START) - 1);
                }
                while (!reasoning.empty() && reasoning.front() == '\n') {
                    reasoning.erase(0, 1);
                }

                const size_t last_close = content.rfind(THINK_END);
                content.erase(0, last_close + sizeof(THINK_END) - 1);
                while (!content.empty() && content.front() == '\n') {
                    content.erase(0, 1);
                }
            }
        }

        std::string trimmed_reasoning;
        if (!trim(reasoning, &trimmed_reasoning, error)) return false;

        *output += IM_START;
        *output += "assistant\n";
        if (options.preserve_thinking || index > last_query) {
            *output += THINK_START;
            *output += '\n';
            *output += trimmed_reasoning;
            *output += '\n';
            *output += THINK_END;
            *output += "\n\n";
        }
        *output += content;
        append_tool_calls(message, !content.empty(), output);
        *output += IM_END;
        *output += '\n';
        return true;
    }

    void append_tool_response(const std::vector<Message>& messages,
                              size_t index, const std::string& content,
                              std::string* output) const {
        if (index != 0 && messages[index - 1].role != Role::Tool) {
            *output += IM_START;
            *output += "user";
        }
        *output += "\n";
        *output += TOOL_RESPONSE_START;
        *output += "\n";
        *output += content;
        *output += "\n";
        *output += TOOL_RESPONSE_END;
        if (index + 1 == messages.size() ||
            messages[index + 1].role != Role::Tool) {
            *output += IM_END;
            *output += '\n';
        }
    }

    void append_generation_prompt(const ChatOptions& options,
                                  std::string* output) const {
        if (!options.add_generation_prompt) return;
        *output += IM_START;
        *output += "assistant\n";
        if (options.enable_thinking) {
            *output += THINK_START;
            *output += '\n';
        } else {
            *output += THINK_START;
            *output += "\n\n";
            *output += THINK_END;
            *output += "\n\n";
        }
    }

    bool render_chat(const ChatRequest& request, std::string* output,
                     std::string* error) const {
        output->clear();
        if (request.messages.empty()) {
            return fail(error, "No messages provided.");
        }

        int image_count = 0;
        int video_count = 0;
        std::string rendered;
        if (!append_system(
                request, image_count, video_count, &rendered, error)) {
            return false;
        }

        size_t last_query = 0;
        if (!find_last_query(
                request, image_count, video_count, &last_query, error)) {
            return false;
        }

        const std::vector<Message>& messages = request.messages;
        for (size_t index = 0; index < messages.size(); ++index) {
            const Message& message = messages[index];
            std::string content;
            if (!render_trimmed_content(
                    message, true, false, image_count, video_count,
                    request.options.add_vision_id, &content, error)) {
                return false;
            }

            switch (message.role) {
            case Role::System:
                if (index != 0) {
                    return fail(
                        error, "System message must be at the beginning."
                    );
                }
                break;
            case Role::User:
                append_user(content, &rendered);
                break;
            case Role::Assistant:
                if (!append_assistant(
                        message, std::move(content), index, last_query,
                        request.options, &rendered, error)) return false;
                break;
            case Role::Tool:
                append_tool_response(messages, index, content, &rendered);
                break;
            default:
                return fail(error, "Unexpected message role.");
            }
        }

        append_generation_prompt(request.options, &rendered);
        *output = std::move(rendered);
        return true;
    }

    // Fixed Qwen3.5 tokenizer ----------------------------------------------

    uint8_t flags(uint32_t codepoint) const {
        const auto found = std::upper_bound(
            ranges.begin(), ranges.end(), codepoint,
            [](uint32_t value, const Range& range) { return value < range.first; }
        );
        if (found == ranges.begin()) return 0;
        const Range& range = found[-1];
        return codepoint <= range.last ? range.flags : 0;
    }

    uint8_t ccc(uint32_t codepoint) const {
        const auto found = std::lower_bound(
            combining.begin(), combining.end(), codepoint,
            [](const Combining& value, uint32_t target) { return value.codepoint < target; }
        );
        return found != combining.end() && found->codepoint == codepoint ? found->value : 0;
    }

    const std::vector<uint32_t>* decomposition(uint32_t codepoint) const {
        const auto found = std::lower_bound(
            decompositions.begin(), decompositions.end(), codepoint,
            [](const Decomposition& value, uint32_t target) { return value.codepoint < target; }
        );
        return found != decompositions.end() && found->codepoint == codepoint
            ? &found->value : nullptr;
    }

    void decompose(uint32_t codepoint, std::vector<uint32_t>& output) const {
        constexpr uint32_t SBASE = 0xac00, LBASE = 0x1100, VBASE = 0x1161, TBASE = 0x11a7;
        constexpr uint32_t LCOUNT = 19, VCOUNT = 21, TCOUNT = 28;
        constexpr uint32_t NCOUNT = VCOUNT * TCOUNT, SCOUNT = LCOUNT * NCOUNT;
        if (codepoint >= SBASE && codepoint < SBASE + SCOUNT) {
            const uint32_t index = codepoint - SBASE;
            output.push_back(LBASE + index / NCOUNT);
            output.push_back(VBASE + (index % NCOUNT) / TCOUNT);
            if (index % TCOUNT) output.push_back(TBASE + index % TCOUNT);
            return;
        }
        if (const std::vector<uint32_t>* value = decomposition(codepoint)) {
            for (uint32_t item : *value) decompose(item, output);
        } else {
            output.push_back(codepoint);
        }
    }

    uint32_t compose_pair(uint32_t first, uint32_t second) const {
        constexpr uint32_t SBASE = 0xac00, LBASE = 0x1100, VBASE = 0x1161, TBASE = 0x11a7;
        constexpr uint32_t LCOUNT = 19, VCOUNT = 21, TCOUNT = 28;
        constexpr uint32_t NCOUNT = VCOUNT * TCOUNT, SCOUNT = LCOUNT * NCOUNT;
        if (first >= LBASE && first < LBASE + LCOUNT &&
            second >= VBASE && second < VBASE + VCOUNT) {
            return SBASE + ((first - LBASE) * VCOUNT + second - VBASE) * TCOUNT;
        }
        if (first >= SBASE && first < SBASE + SCOUNT &&
            (first - SBASE) % TCOUNT == 0 &&
            second > TBASE && second < TBASE + TCOUNT) {
            return first + second - TBASE;
        }
        const auto found = compositions.find(pair_key(first, second));
        return found == compositions.end() ? 0 : found->second;
    }

    bool nfc(const std::string& text, std::vector<uint32_t>* output,
             std::string* error) const {
        std::vector<uint32_t> input;
        if (!utf8_codepoints(text, &input, error)) return false;
        std::vector<uint32_t> ordered;
        for (uint32_t codepoint : input) {
            const size_t start = ordered.size();
            decompose(codepoint, ordered);
            for (size_t index = start; index < ordered.size(); ++index) {
                const uint8_t current = ccc(ordered[index]);
                if (!current) continue;
                size_t position = index;
                while (position > 0) {
                    const uint8_t previous = ccc(ordered[position - 1]);
                    if (!previous || previous <= current) break;
                    std::swap(ordered[position], ordered[position - 1]);
                    --position;
                }
            }
        }
        if (ordered.empty()) {
            output->clear();
            return true;
        }

        output->clear();
        output->reserve(ordered.size());
        output->push_back(ordered[0]);
        size_t starter_index = 0;
        uint32_t starter = ordered[0];
        uint8_t previous_ccc = 0;
        for (size_t index = 1; index < ordered.size(); ++index) {
            const uint32_t codepoint = ordered[index];
            const uint8_t current_ccc = ccc(codepoint);
            const uint32_t composite = compose_pair(starter, codepoint);
            if (composite && (previous_ccc < current_ccc || previous_ccc == 0)) {
                (*output)[starter_index] = composite;
                starter = composite;
            } else {
                if (current_ccc == 0) {
                    starter_index = output->size();
                    starter = codepoint;
                }
                output->push_back(codepoint);
                previous_ccc = current_ccc;
            }
        }
        return true;
    }

    bool is_letter(uint32_t cp) const { return flags(cp) & LETTER; }
    bool is_mark(uint32_t cp) const { return flags(cp) & MARK; }
    bool is_number(uint32_t cp) const { return flags(cp) & NUMBER; }
    bool is_space(uint32_t cp) const { return flags(cp) & SPACE; }
    bool is_line(uint32_t cp) const { return cp == '\r' || cp == '\n'; }
    bool is_letter_mark(uint32_t cp) const { return flags(cp) & (LETTER | MARK); }

    size_t contraction(const std::vector<uint32_t>& text, size_t position) const {
        if (text[position] != '\'' || position + 1 >= text.size()) return position;
        static constexpr const char* SUFFIXES[] = {"s", "t", "re", "ve", "m", "ll", "d"};
        for (const char* suffix : SUFFIXES) {
            size_t index = position + 1;
            size_t offset = 0;
            while (suffix[offset] && index < text.size()) {
                uint32_t cp = text[index];
                if (cp >= 'A' && cp <= 'Z') cp += 'a' - 'A';
                if (cp != static_cast<uint32_t>(suffix[offset])) break;
                ++index;
                ++offset;
            }
            if (!suffix[offset]) return index;
        }
        return position;
    }

    size_t piece_end(const std::vector<uint32_t>& text, size_t position) const {
        const size_t size = text.size();
        size_t end = contraction(text, position);
        if (end != position) return end;

        // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        end = position;
        if (!is_line(text[end]) && !is_letter(text[end]) && !is_number(text[end])) ++end;
        const size_t letters = end;
        while (end < size && is_letter_mark(text[end])) ++end;
        if (end > letters) return end;
        end = position;
        while (end < size && is_letter_mark(text[end])) ++end;
        if (end > position) return end;

        // \p{N} -- intentionally one number at a time.
        if (is_number(text[position])) return position + 1;

        // Optional ASCII space, punctuation/symbol run, then CR/LF.
        end = position + (text[position] == ' ' ? 1 : 0);
        const size_t symbols = end;
        while (end < size && !is_space(text[end]) && !is_letter(text[end]) &&
               !is_mark(text[end]) && !is_number(text[end])) ++end;
        if (end > symbols) {
            while (end < size && is_line(text[end])) ++end;
            return end;
        }

        if (is_space(text[position])) {
            size_t whitespace_end = position;
            size_t last_line_end = position;
            while (whitespace_end < size && is_space(text[whitespace_end])) {
                if (is_line(text[whitespace_end])) last_line_end = whitespace_end + 1;
                ++whitespace_end;
            }
            // \s*[\r\n]+ stops after the last newline, leaving later spaces.
            if (last_line_end != position) return last_line_end;
            // \s+(?!\S) consumes all trailing whitespace, otherwise all but
            // the final character before the next non-space.
            if (whitespace_end == size) return whitespace_end;
            if (whitespace_end - position > 1) return whitespace_end - 1;
            return whitespace_end;  // final \s+ alternative
        }
        return position + 1;  // Defensive: the fixed regex should cover every codepoint.
    }

    void bpe(const std::string& bytes, std::vector<int>& output) const {
        if (bytes.empty()) return;
        struct Node { uint32_t token; int previous, next; bool alive = true; };
        struct Event { uint32_t rank; int left; };
        struct Later {
            bool operator()(const Event& a, const Event& b) const {
                return a.rank != b.rank ? a.rank > b.rank : a.left > b.left;
            }
        };

        std::vector<Node> nodes;
        nodes.reserve(bytes.size());
        for (size_t index = 0; index < bytes.size(); ++index) {
            nodes.push_back(Node{
                byte_ids[static_cast<uint8_t>(bytes[index])],
                index ? static_cast<int>(index - 1) : -1,
                index + 1 < bytes.size() ? static_cast<int>(index + 1) : -1,
                true,
            });
        }
        std::priority_queue<Event, std::vector<Event>, Later> queue;
        auto add = [&](int left) {
            if (left < 0 || !nodes[left].alive || nodes[left].next < 0) return;
            const int right = nodes[left].next;
            const auto found = merges.find(pair_key(nodes[left].token, nodes[right].token));
            if (found != merges.end()) queue.push(Event{found->second.rank, left});
        };
        for (size_t index = 0; index + 1 < nodes.size(); ++index) add(static_cast<int>(index));

        while (!queue.empty()) {
            const Event event = queue.top();
            queue.pop();
            const int left = event.left;
            if (left < 0 || !nodes[left].alive || nodes[left].next < 0) continue;
            const int right = nodes[left].next;
            if (!nodes[right].alive) continue;
            const auto found = merges.find(pair_key(nodes[left].token, nodes[right].token));
            if (found == merges.end() || found->second.rank != event.rank) continue;

            nodes[left].token = found->second.result;
            nodes[left].next = nodes[right].next;
            if (nodes[right].next >= 0) nodes[nodes[right].next].previous = left;
            nodes[right].alive = false;
            add(nodes[left].previous);
            add(left);
        }
        for (int node = 0; node >= 0; node = nodes[node].next) {
            output.push_back(static_cast<int>(nodes[node].token));
        }
    }

    bool encode_plain(const std::string& text, std::vector<int>* output,
                      std::string* error) const {
        std::vector<uint32_t> normalized;
        if (!nfc(text, &normalized, error)) return false;
        for (size_t position = 0; position < normalized.size();) {
            const size_t end = piece_end(normalized, position);
            std::string bytes;
            for (size_t index = position; index < end; ++index) append_utf8(bytes, normalized[index]);
            bpe(bytes, *output);
            position = end;
        }
        return true;
    }

    bool encode(const std::string& text, std::vector<int>* output,
                std::string* error) const {
        // AddedToken(normalized=false) strings are extracted before NFC.
        output->clear();
        std::vector<uint32_t> validated;
        if (!utf8_codepoints(text, &validated, error)) return false;
        size_t position = 0;
        while (position < text.size()) {
            size_t match_position = std::string::npos;
            const Added* match = nullptr;
            for (const Added& token : added) {
                const size_t found = text.find(token.text, position);
                if (found == std::string::npos) continue;
                if (!match || found < match_position ||
                    (found == match_position && token.text.size() > match->text.size()) ||
                    (found == match_position && token.text.size() == match->text.size() &&
                     token.id < match->id)) {
                    match_position = found;
                    match = &token;
                }
            }
            if (!match) {
                if (!encode_plain(text.substr(position), output, error)) return false;
                break;
            }
            if (!encode_plain(text.substr(position, match_position - position),
                              output, error)) return false;
            output->push_back(static_cast<int>(match->id));
            position = match_position + match->text.size();
        }
        return true;
    }

    bool trim(const std::string& text, std::string* output,
              std::string* error) const {
        std::vector<uint32_t> codepoints;
        if (!utf8_codepoints(text, &codepoints, error)) return false;
        size_t first = 0;
        size_t last = codepoints.size();
        const auto python_space = [&](uint32_t cp) {
            return is_space(cp) || (cp >= 0x1c && cp <= 0x1f);
        };
        while (first < last && python_space(codepoints[first])) ++first;
        while (last > first && python_space(codepoints[last - 1])) --last;
        *output = utf8_text(std::vector<uint32_t>(codepoints.begin() + first,
                                                  codepoints.begin() + last));
        return true;
    }
};

namespace {

uint64_t pair_key(uint32_t left, uint32_t right) {
    return (static_cast<uint64_t>(left) << 32) | right;
}

bool fail(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

class BinaryReader {
public:
    explicit BinaryReader(const std::string& path) : input_(path, std::ios::binary) {
        if (!input_) {
            set_error("cannot open render data: " + path);
            return;
        }
        input_.seekg(0, std::ios::end);
        const std::streampos end = input_.tellg();
        if (end < 0) {
            set_error("cannot size render data");
            return;
        }
        size_ = static_cast<uint64_t>(end);
        if (size_ > MAX_RENDER_SIZE) {
            set_error("render data is too large");
            return;
        }
        input_.seekg(0, std::ios::beg);
        if (!input_) set_error("cannot read render data");
    }

    uint8_t u8() {
        char byte = 0;
        read(&byte, 1);
        return static_cast<uint8_t>(byte);
    }

    uint32_t u32() {
        std::array<uint8_t, 4> bytes{};
        read(reinterpret_cast<char*>(bytes.data()), bytes.size());
        return static_cast<uint32_t>(bytes[0]) |
               (static_cast<uint32_t>(bytes[1]) << 8) |
               (static_cast<uint32_t>(bytes[2]) << 16) |
               (static_cast<uint32_t>(bytes[3]) << 24);
    }

    uint64_t u64() {
        std::array<uint8_t, 8> bytes{};
        read(reinterpret_cast<char*>(bytes.data()), bytes.size());
        uint64_t value = 0;
        for (size_t index = 0; index < bytes.size(); ++index) {
            value |= static_cast<uint64_t>(bytes[index]) << (index * 8);
        }
        return value;
    }

    std::string bytes(size_t count) {
        if (!require(count)) return {};
        std::string output(count, '\0');
        if (count && !read(&output[0], count)) return {};
        return output;
    }

    std::string string() {
        const uint32_t count = u32();
        return ok() ? bytes(count) : std::string{};
    }

    uint64_t size() const { return size_; }
    uint64_t remaining() const { return size_ - position_; }
    bool ok() const { return error_.empty(); }
    const std::string& error() const { return error_; }

    bool finish() {
        if (!ok()) return false;
        if (position_ != size_) return set_error("render data has trailing bytes");
        return true;
    }

private:
    bool set_error(const std::string& message) {
        if (error_.empty()) error_ = message;
        return false;
    }

    bool require(size_t count) {
        if (!ok()) return false;
        if (static_cast<uint64_t>(count) > size_ - position_) {
            return set_error("truncated render data");
        }
        return true;
    }

    bool read(char* output, size_t count) {
        if (!require(count)) return false;
        if (count > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
            return set_error("render data field is too large");
        }
        input_.read(output, static_cast<std::streamsize>(count));
        if (input_.gcount() != static_cast<std::streamsize>(count)) {
            return set_error("cannot read render data");
        }
        position_ += static_cast<uint64_t>(count);
        return true;
    }

    std::ifstream input_;
    uint64_t size_ = 0;
    uint64_t position_ = 0;
    std::string error_;
};

bool hex_sha256(const std::string& text) {
    if (text.size() != 64) return false;
    return std::all_of(text.begin(), text.end(), [](unsigned char character) {
        return (character >= '0' && character <= '9') ||
               (character >= 'a' && character <= 'f');
    });
}

bool unicode_scalar(uint32_t codepoint) {
    return codepoint <= 0x10ffff && !(codepoint >= 0xd800 && codepoint <= 0xdfff);
}

bool utf8_codepoints(const std::string& text, std::vector<uint32_t>* output,
                     std::string* error) {
    output->clear();
    output->reserve(text.size());
    for (size_t index = 0; index < text.size();) {
        const uint8_t first = static_cast<uint8_t>(text[index]);
        uint32_t value = 0;
        size_t count = 0;
        if (first <= 0x7f) {
            value = first;
            count = 1;
        } else if (first >= 0xc2 && first <= 0xdf) {
            value = first & 0x1f;
            count = 2;
        } else if (first >= 0xe0 && first <= 0xef) {
            value = first & 0x0f;
            count = 3;
        } else if (first >= 0xf0 && first <= 0xf4) {
            value = first & 0x07;
            count = 4;
        } else {
            return fail(error, "text is not valid UTF-8");
        }
        if (index + count > text.size()) {
            return fail(error, "text is not valid UTF-8");
        }
        for (size_t offset = 1; offset < count; ++offset) {
            const uint8_t byte = static_cast<uint8_t>(text[index + offset]);
            if ((byte & 0xc0) != 0x80) {
                return fail(error, "text is not valid UTF-8");
            }
            value = (value << 6) | (byte & 0x3f);
        }
        if ((count == 3 && value < 0x800) ||
            (count == 4 && value < 0x10000) ||
            value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) {
            return fail(error, "text is not valid UTF-8");
        }
        output->push_back(value);
        index += count;
    }
    return true;
}

void append_utf8(std::string& output, uint32_t codepoint) {
    if (codepoint <= 0x7f) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
        output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
}

std::string utf8_text(const std::vector<uint32_t>& codepoints) {
    std::string output;
    for (uint32_t codepoint : codepoints) append_utf8(output, codepoint);
    return output;
}

std::string lossy_utf8(const std::string& bytes) {
    std::string output;
    for (size_t index = 0; index < bytes.size();) {
        const uint8_t first = static_cast<uint8_t>(bytes[index]);
        size_t count = 0;
        uint32_t value = 0;
        if (first <= 0x7f) {
            output.push_back(static_cast<char>(first));
            ++index;
            continue;
        } else if (first >= 0xc2 && first <= 0xdf) {
            count = 2;
            value = first & 0x1f;
        } else if (first >= 0xe0 && first <= 0xef) {
            count = 3;
            value = first & 0x0f;
        } else if (first >= 0xf0 && first <= 0xf4) {
            count = 4;
            value = first & 0x07;
        }

        bool valid = count != 0 && index + count <= bytes.size();
        size_t valid_prefix = count ? 1 : 0;
        if (valid) {
            for (size_t offset = 1; offset < count; ++offset) {
                const uint8_t byte = static_cast<uint8_t>(bytes[index + offset]);
                if ((byte & 0xc0) != 0x80) {
                    valid = false;
                    break;
                }
                value = (value << 6) | (byte & 0x3f);
                ++valid_prefix;
            }
            valid = valid && !(count == 3 && value < 0x800) &&
                    !(count == 4 && value < 0x10000) && value <= 0x10ffff &&
                    !(value >= 0xd800 && value <= 0xdfff);
        }
        if (valid) {
            output.append(bytes, index, count);
            index += count;
        } else {
            output.append("\xef\xbf\xbd");
            if (count && index + count > bytes.size()) break;
            // Match Rust/Python lossy UTF-8: a valid leading byte plus valid
            // continuation prefix is one malformed subsequence. The first
            // non-continuation byte is reconsidered as the next character.
            index += std::max<size_t>(valid_prefix, 1);
        }
    }
    return output;
}

}  // namespace

bool Renderer::Impl::load(const std::string& path, std::string* error) {
    BinaryReader input(path);
    if (!input.ok()) return fail(error, input.error());
    const std::string magic = input.bytes(RENDER_MAGIC.size());
    if (!input.ok()) return fail(error, input.error());
    if (!std::equal(magic.begin(), magic.end(), RENDER_MAGIC.begin())) {
        return fail(error, "invalid render data magic");
    }
    const uint32_t version = input.u32();
    if (!input.ok()) return fail(error, input.error());
    if (version != RENDER_VERSION) {
        return fail(error, "unsupported render data version");
    }
    if (input.u32() != RENDER_HEADER_SIZE) {
        if (!input.ok()) return fail(error, input.error());
        return fail(error, "unsupported render data header");
    }
    if (input.u64() != input.size()) {
        if (!input.ok()) return fail(error, input.error());
        return fail(error, "render data size does not match header");
    }
    model_vocab = input.u32();
    base_vocab = input.u32();
    decodable = input.u32();
    const uint32_t merge_count = input.u32();
    const uint32_t added_count = input.u32();
    const uint32_t range_count = input.u32();
    const uint32_t combining_count = input.u32();
    const uint32_t decomposition_count = input.u32();
    const uint32_t composition_count = input.u32();
    if (!input.ok()) return fail(error, input.error());
    if (model_vocab != 248320 || base_vocab != 248044 ||
        decodable < base_vocab || decodable > model_vocab ||
        merge_count != 247587 || added_count != 33) {
        return fail(error, "render data does not match Qwen3.5");
    }
    if (range_count > 0x110000 || combining_count > 0x110000 ||
        decomposition_count > 0x110000 || composition_count > 0x110000) {
        return fail(error, "render data has impossible Unicode counts");
    }

    oracle_version = input.string();
    tokenizer_sha256 = input.string();
    config_sha256 = input.string();
    if (!input.ok()) return fail(error, input.error());
    if (oracle_version.compare(0, 11, "tokenizers-") != 0 ||
        oracle_version.size() > 128 ||
        !hex_sha256(tokenizer_sha256) || !hex_sha256(config_sha256)) {
        return fail(error, "invalid render data source metadata");
    }
    for (uint32_t& token : byte_ids) {
        token = input.u32();
        if (!input.ok()) return fail(error, input.error());
        if (token >= base_vocab) return fail(error, "invalid byte token ID");
    }

    token_bytes.resize(model_vocab);
    special.assign(model_vocab, false);
    for (uint32_t id = 0; id < base_vocab; ++id) {
        token_bytes[id] = input.string();
        if (!input.ok()) return fail(error, input.error());
        if (token_bytes[id].empty() || token_bytes[id].size() > 1024 * 1024) {
            return fail(error, "invalid base token table");
        }
    }

    merges.reserve(static_cast<size_t>(merge_count) * 2);
    for (uint32_t rank = 0; rank < merge_count; ++rank) {
        const uint32_t left = input.u32();
        const uint32_t right = input.u32();
        const uint32_t result = input.u32();
        if (!input.ok()) return fail(error, input.error());
        if (left >= base_vocab || right >= base_vocab || result >= base_vocab ||
            token_bytes[left] + token_bytes[right] != token_bytes[result] ||
            !merges.emplace(pair_key(left, right), Merge{rank, result}).second) {
            return fail(error, "invalid BPE merge table");
        }
    }

    added.reserve(added_count);
    for (uint32_t index = 0; index < added_count; ++index) {
        Added token;
        token.id = input.u32();
        const uint8_t special_value = input.u8();
        token.special = special_value != 0;
        token.text = input.string();
        if (!input.ok()) return fail(error, input.error());
        if (token.id < base_vocab || token.id >= decodable || token.text.empty() ||
            token.text.size() > 1024 * 1024 || special_value > 1 ||
            (!added.empty() && added.back().id >= token.id) ||
            !token_bytes[token.id].empty()) {
            return fail(error, "invalid added token table");
        }
        token_bytes[token.id] = token.text;
        special[token.id] = token.special;
        added.push_back(std::move(token));
    }

    ranges.reserve(range_count);
    for (uint32_t index = 0; index < range_count; ++index) {
        Range range{input.u32(), input.u32(), input.u8()};
        if (!input.ok()) return fail(error, input.error());
        if (range.first > range.last || range.last > 0x10ffff ||
            range.flags == 0 || (range.flags & ~(LETTER | MARK | NUMBER | SPACE)) != 0 ||
            (!ranges.empty() && ranges.back().last >= range.first)) {
            return fail(error, "invalid Unicode category table");
        }
        ranges.push_back(range);
    }
    combining.reserve(combining_count);
    for (uint32_t index = 0; index < combining_count; ++index) {
        Combining value{input.u32(), input.u8()};
        if (!input.ok()) return fail(error, input.error());
        if ((!combining.empty() && combining.back().codepoint >= value.codepoint) ||
            !unicode_scalar(value.codepoint) || value.value == 0) {
            return fail(error, "invalid Unicode combining table");
        }
        combining.push_back(value);
    }
    decompositions.reserve(decomposition_count);
    for (uint32_t index = 0; index < decomposition_count; ++index) {
        Decomposition value;
        value.codepoint = input.u32();
        const uint8_t count = input.u8();
        if (!input.ok()) return fail(error, input.error());
        if (!unicode_scalar(value.codepoint) || count == 0 ||
            (!decompositions.empty() &&
                           decompositions.back().codepoint >= value.codepoint)) {
            return fail(error, "invalid Unicode decomposition table");
        }
        value.value.reserve(count);
        for (uint8_t item = 0; item < count; ++item) {
            const uint32_t codepoint = input.u32();
            if (!input.ok()) return fail(error, input.error());
            if (!unicode_scalar(codepoint)) {
                return fail(error, "invalid Unicode decomposition table");
            }
            value.value.push_back(codepoint);
        }
        decompositions.push_back(std::move(value));
    }
    compositions.reserve(static_cast<size_t>(composition_count) * 2);
    for (uint32_t index = 0; index < composition_count; ++index) {
        const uint32_t first = input.u32();
        const uint32_t second = input.u32();
        const uint32_t result = input.u32();
        if (!input.ok()) return fail(error, input.error());
        if (!unicode_scalar(first) || !unicode_scalar(second) ||
            !unicode_scalar(result) ||
            !compositions.emplace(pair_key(first, second), result).second) {
            return fail(error, "invalid Unicode composition table");
        }
    }
    if (!input.finish()) return fail(error, input.error());
    return true;
}

bool parse_generated_tool_calls(const std::string& text,
                                std::string* content,
                                std::vector<ToolCall>* calls,
                                std::string* error) {
    if (!content || !calls) return fail(error, "tool output pointer is null");
    content->clear();
    calls->clear();

    constexpr char TOOL_START[] = "<tool_call>";
    constexpr char TOOL_END[] = "</tool_call>";
    constexpr char FUNCTION_START[] = "<function=";
    constexpr char FUNCTION_END[] = "</function>";
    constexpr char PARAMETER_START[] = "<parameter=";
    constexpr char PARAMETER_END[] = "</parameter>";

    size_t first = text.find(TOOL_START);
    if (first == std::string::npos) first = text.find(FUNCTION_START);
    if (first == std::string::npos) {
        *content = text;
        return true;
    }

    *content = text.substr(0, first);
    while (!content->empty() &&
           std::strchr(" \t\r\n\f\v", content->back())) {
        content->pop_back();
    }

    size_t cursor = first;
    while (true) {
        const size_t function = text.find(FUNCTION_START, cursor);
        if (function == std::string::npos) break;
        const size_t name_begin = function + sizeof(FUNCTION_START) - 1;
        const size_t name_end = text.find('>', name_begin);
        const size_t function_end = name_end == std::string::npos
            ? std::string::npos : text.find(FUNCTION_END, name_end + 1);
        if (name_end == std::string::npos || function_end == std::string::npos) {
            return fail(error, "incomplete generated tool call");
        }

        ToolCall call;
        call.name = text.substr(name_begin, name_end - name_begin);
        while (!call.name.empty() && call.name.front() == ' ') {
            call.name.erase(0, 1);
        }
        while (!call.name.empty() && call.name.back() == ' ') {
            call.name.pop_back();
        }
        if (call.name.empty()) return fail(error, "generated tool name is empty");

        size_t argument = name_end + 1;
        while (true) {
            const size_t parameter = text.find(PARAMETER_START, argument);
            if (parameter == std::string::npos || parameter >= function_end) break;
            const size_t parameter_name = parameter +
                                          sizeof(PARAMETER_START) - 1;
            const size_t parameter_name_end = text.find('>', parameter_name);
            if (parameter_name_end == std::string::npos ||
                parameter_name_end >= function_end) {
                return fail(error, "incomplete generated parameter name");
            }
            const size_t value_begin = parameter_name_end + 1;
            const size_t next_parameter = text.find(PARAMETER_START, value_begin);
            const size_t parameter_end = text.find(PARAMETER_END, value_begin);
            size_t value_end = parameter_end;
            if (next_parameter != std::string::npos &&
                next_parameter < function_end &&
                (parameter_end == std::string::npos ||
                 next_parameter < parameter_end)) {
                value_end = next_parameter;
                argument = next_parameter;
            } else {
                if (parameter_end == std::string::npos ||
                    parameter_end > function_end) {
                    return fail(error, "incomplete generated parameter value");
                }
                argument = parameter_end + sizeof(PARAMETER_END) - 1;
            }

            ToolArgument item;
            item.name = text.substr(parameter_name,
                                    parameter_name_end - parameter_name);
            item.text = text.substr(value_begin, value_end - value_begin);
            if (!item.text.empty() && item.text.front() == '\n') {
                item.text.erase(0, 1);
            }
            if (!item.text.empty() && item.text.back() == '\n') {
                item.text.pop_back();
            }
            if (item.name.empty()) {
                return fail(error, "generated parameter name is empty");
            }
            call.arguments.push_back(std::move(item));
        }
        calls->push_back(std::move(call));
        cursor = function_end + sizeof(FUNCTION_END) - 1;
    }

    if (calls->empty()) return fail(error, "generated tool wrapper has no function");
    if (text.compare(first, sizeof(TOOL_START) - 1, TOOL_START) == 0 &&
        text.rfind(TOOL_END) == std::string::npos) {
        return fail(error, "incomplete generated tool wrapper");
    }
    return true;
}

Renderer::Renderer() = default;
Renderer::~Renderer() = default;

Renderer* Renderer::create(const std::string& render_bin, std::string* error) {
    if (error) error->clear();
    std::unique_ptr<Renderer> renderer(new Renderer());
    renderer->impl_.reset(new Impl());
    if (!renderer->impl_->load(render_bin, error)) return nullptr;
    return renderer.release();
}

bool Renderer::render(const ChatRequest& request, RenderedPrompt* output,
                      std::string* error) const {
    if (error) error->clear();
    if (!output) return fail(error, "render output pointer is null");
    *output = RenderedPrompt{};
    if (!impl_->render_chat(request, &output->text, error)) return false;
    if (!impl_->encode(output->text, &output->tokens, error)) {
        *output = RenderedPrompt{};
        return false;
    }
    return true;
}

bool Renderer::encode(const std::string& text, std::vector<int>* output,
                      std::string* error) const {
    if (error) error->clear();
    if (!output) return fail(error, "encode output pointer is null");
    return impl_->encode(text, output, error);
}

bool Renderer::decode(const std::vector<int>& tokens, bool skip_special_tokens,
                      std::string* output, std::string* error) const {
    if (error) error->clear();
    if (!output) return fail(error, "decode output pointer is null");
    output->clear();
    std::string bytes;
    for (int token : tokens) {
        if (token < 0 || static_cast<uint32_t>(token) >= impl_->model_vocab) {
            return fail(error, "token outside model vocabulary");
        }
        if (skip_special_tokens && impl_->special[token]) continue;
        bytes += impl_->token_bytes[token];
    }
    *output = lossy_utf8(bytes);
    return true;
}

}  // namespace q35_render
