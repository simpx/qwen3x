// parser.cpp -- the only JSON-aware part of the C++ text boundary.
//
//   JSON text -> ordered DOM -> ChatRequest
//
// The DOM is local to parse_chat_request(). render.cpp only sees the plain
// structs declared in render.h.

#include "render.h"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#define JSON_NOEXCEPTION
#include "third_party/nlohmann/json.hpp"

namespace q35_render {
namespace {

using Json = nlohmann::ordered_json;

class Parser {
public:
    Status parse(const std::string& text, ChatRequest& output) {
        Json root = Json::parse(text, nullptr, false);
        if (root.is_discarded()) {
            return invalid("malformed JSON");
        }

        ChatRequest request;
        if (!parse_root(root, &request)) return invalid(error_message_);
        output = std::move(request);
        return Status{};
    }

    Status parse_completion(const std::string& text,
                            const std::string& served_model,
                            int default_max_tokens,
                            CompletionRequest& output) {
        Json root = Json::parse(text, nullptr, false);
        if (root.is_discarded()) return invalid("malformed JSON");
        if (!root.is_object()) return invalid("root must be an object");

        CompletionRequest request;
        request.max_tokens = default_max_tokens;
        if (!parse_completion_fields(root, served_model, &request)) {
            return invalid(error_message_);
        }
        if (!parse_root(root, &request.chat)) return invalid(error_message_);
        if (!parse_template_options(root, &request.chat.options)) {
            return invalid(error_message_);
        }
        if (!validate_http_chat(request.chat)) return invalid(error_message_);
        output = std::move(request);
        return Status{};
    }

private:
    bool fail(const std::string& message) {
        error_message_ = message;
        return false;
    }

    Status invalid(const std::string& message) {
        return Status::invalid_argument(
            "invalid chat request: " + message
        );
    }

    // The caller has already checked is_string(). Transfer the DOM string
    // instead of copying it; the DOM is discarded after parsing.
    std::string move_string(Json& value) {
        return std::move(value.get_ref<std::string&>());
    }

    bool float_text(double value, std::string* output) {
        if (!std::isfinite(value)) return fail("non-finite number");
        std::array<char, 64> buffer{};
        const auto result = std::to_chars(
            buffer.data(), buffer.data() + buffer.size(), value,
            std::chars_format::general
        );
        if (result.ec != std::errc()) {
            return fail("cannot render number");
        }
        output->assign(buffer.data(), result.ptr);
        if (output->find_first_of(".eE") == std::string::npos) *output += ".0";
        return true;
    }

    std::string json_string(const std::string& value) {
        constexpr char HEX[] = "0123456789abcdef";
        std::string output = "\"";
        for (uint8_t byte : value) {
            switch (byte) {
                case '"': output += "\\\""; break;
                case '\\': output += "\\\\"; break;
                case '\b': output += "\\b"; break;
                case '\f': output += "\\f"; break;
                case '\n': output += "\\n"; break;
                case '\r': output += "\\r"; break;
                case '\t': output += "\\t"; break;
                default:
                    if (byte < 0x20) {
                        output += "\\u00";
                        output.push_back(HEX[byte >> 4]);
                        output.push_back(HEX[byte & 15]);
                    } else {
                        output.push_back(static_cast<char>(byte));
                    }
            }
        }
        output.push_back('"');
        return output;
    }

    bool json_text(const Json& value, std::string* output) {
        if (value.is_null()) {
            *output = "null";
        } else if (value.is_boolean()) {
            *output = value.get<bool>() ? "true" : "false";
        } else if (value.is_number_unsigned()) {
            *output = std::to_string(value.get<uint64_t>());
        } else if (value.is_number_integer()) {
            *output = std::to_string(value.get<int64_t>());
        } else if (value.is_number_float()) {
            return float_text(value.get<double>(), output);
        } else if (value.is_string()) {
            *output = json_string(value.get_ref<const std::string&>());
        } else if (value.is_array()) {
            *output = "[";
            for (size_t index = 0; index < value.size(); ++index) {
                if (index) *output += ", ";
                std::string item;
                if (!json_text(value[index], &item)) return false;
                *output += item;
            }
            *output += ']';
        } else if (value.is_object()) {
            *output = "{";
            bool first = true;
            for (const auto& item : value.items()) {
                if (!first) *output += ", ";
                first = false;
                *output += json_string(item.key());
                *output += ": ";
                std::string item_text;
                if (!json_text(item.value(), &item_text)) return false;
                *output += item_text;
            }
            *output += '}';
        } else {
            return fail("unsupported nested tool argument");
        }
        return true;
    }

    bool argument_text(Json& value, std::string* output) {
        if (value.is_array() || value.is_object()) {
            return json_text(value, output);
        } else if (value.is_null()) {
            *output = "None";
        } else if (value.is_boolean()) {
            *output = value.get<bool>() ? "True" : "False";
        } else if (value.is_number_unsigned()) {
            *output = std::to_string(value.get<uint64_t>());
        } else if (value.is_number_integer()) {
            *output = std::to_string(value.get<int64_t>());
        } else if (value.is_number_float()) {
            return float_text(value.get<double>(), output);
        } else if (value.is_string()) {
            *output = move_string(value);
        } else {
            return fail("unsupported tool argument");
        }
        return true;
    }

    bool string_field_equals(const Json& object, const char* field,
                             const char* expected) const {
        const auto value = object.find(field);
        return value != object.end() && value->is_string() &&
               value->get_ref<const std::string&>() == expected;
    }

    bool parse_content_part(Json& value, ContentPart* output) {
        if (!value.is_object()) {
            return fail("message.content items must be objects");
        }
        if (value.contains("image") || value.contains("image_url") ||
            string_field_equals(value, "type", "image")) {
            output->kind = ContentKind::Image;
            return true;
        }
        if (value.contains("video") ||
            string_field_equals(value, "type", "video")) {
            output->kind = ContentKind::Video;
            return true;
        }

        auto text = value.find("text");
        if (text == value.end()) {
            return fail("unexpected item type in message.content");
        }
        if (!text->is_string()) return fail("content.text must be a string");
        output->kind = ContentKind::Text;
        output->text = move_string(*text);
        return true;
    }

    bool parse_tool_call(Json& value, ToolCall* output) {
        if (!value.is_object()) {
            return fail("message.tool_calls items must be objects");
        }

        auto id = value.find("id");
        if (id != value.end()) {
            if (!id->is_string()) return fail("tool_call.id must be a string");
            output->id = move_string(*id);
        }

        Json* call = &value;
        auto function = value.find("function");
        if (function != value.end()) {
            if (!function->is_object()) {
                return fail("tool_call.function must be an object");
            }
            call = &*function;
        }

        auto name = call->find("name");
        if (name == call->end()) return fail("tool call is missing name");
        if (!name->is_string()) return fail("tool_call.name must be a string");
        output->name = move_string(*name);

        auto arguments = call->find("arguments");
        if (arguments == call->end()) return true;
        Json parsed_arguments;
        Json* object = &*arguments;
        if (arguments->is_string()) {
            parsed_arguments = Json::parse(
                arguments->get_ref<const std::string&>(), nullptr, false
            );
            if (parsed_arguments.is_discarded()) {
                return fail("tool_call.arguments contains malformed JSON");
            }
            object = &parsed_arguments;
        }
        if (!object->is_object()) {
            return fail(
                "tool_call.arguments must be an object or an encoded object"
            );
        }

        output->arguments.reserve(object->size());
        for (auto& item : object->items()) {
            ToolArgument argument;
            argument.name = item.key();
            if (!argument_text(item.value(), &argument.text)) return false;
            output->arguments.push_back(std::move(argument));
        }
        return true;
    }

    bool parse_role(Json& value, Role* output) {
        if (!value.is_string()) return fail("message.role must be a string");
        const std::string role = move_string(value);

        if (role == "system" || role == "developer") *output = Role::System;
        else if (role == "user") *output = Role::User;
        else if (role == "assistant") *output = Role::Assistant;
        else if (role == "tool") *output = Role::Tool;
        else return fail("unexpected message role: " + role);
        return true;
    }

    bool parse_message(Json& value, Message* output) {
        if (!value.is_object()) return fail("messages items must be objects");

        auto role = value.find("role");
        if (role == value.end()) return fail("message is missing role");
        if (!parse_role(*role, &output->role)) return false;

        auto content = value.find("content");
        if (content == value.end() || content->is_null()) {
            output->content_is_null = true;
        } else if (content->is_string()) {
            output->content = move_string(*content);
        } else if (content->is_array()) {
            output->parts.reserve(content->size());
            for (Json& part_value : *content) {
                ContentPart part;
                if (!parse_content_part(part_value, &part)) return false;
                output->parts.push_back(std::move(part));
            }
        } else {
            return fail("message.content must be a string, array, or null");
        }

        auto reasoning = value.find("reasoning_content");
        if (reasoning != value.end()) {
            if (!reasoning->is_string()) {
                return fail("reasoning_content must be a string");
            }
            output->has_reasoning = true;
            output->reasoning_content = move_string(*reasoning);
        }

        auto calls = value.find("tool_calls");
        if (calls != value.end() && !calls->is_null()) {
            if (!calls->is_array()) {
                return fail("message.tool_calls must be an array");
            }
            output->tool_calls.reserve(calls->size());
            for (Json& call_value : *calls) {
                ToolCall call;
                if (!parse_tool_call(call_value, &call)) return false;
                output->tool_calls.push_back(std::move(call));
            }
        }
        return true;
    }

    bool parse_bool_option(Json& root, const char* name, bool* output) {
        auto value = root.find(name);
        if (value == root.end()) return true;
        if (!value->is_boolean()) {
            return fail(std::string(name) + " must be a boolean");
        }
        *output = value->get<bool>();
        return true;
    }

    bool number(const Json& root, const char* name, double fallback,
                double minimum, double maximum, float* output) {
        auto value = root.find(name);
        if (value == root.end()) {
            *output = static_cast<float>(fallback);
            return true;
        }
        if (!value->is_number() || value->is_boolean()) {
            return fail(std::string(name) + " must be a number");
        }
        const double result = value->get<double>();
        if (!std::isfinite(result) || result < minimum || result > maximum) {
            return fail(std::string(name) + " is outside its allowed range");
        }
        *output = static_cast<float>(result);
        return true;
    }

    bool integer_option(const Json& root, const char* name, int fallback,
                        int minimum, int* output) {
        auto value = root.find(name);
        if (value == root.end()) {
            *output = fallback;
            return true;
        }
        if (!value->is_number_integer() || value->is_boolean()) {
            return fail(std::string(name) + " must be an integer");
        }
        const int64_t result = value->get<int64_t>();
        if (result < minimum || result > std::numeric_limits<int>::max()) {
            return fail(std::string(name) + " is outside its allowed range");
        }
        *output = static_cast<int>(result);
        return true;
    }

    bool parse_template_options(Json& root, ChatOptions* output) {
        auto options = root.find("chat_template_kwargs");
        if (options == root.end() || options->is_null()) return true;
        if (!options->is_object()) {
            return fail("chat_template_kwargs must be an object");
        }
        for (const auto& item : options->items()) {
            if (item.key() != "enable_thinking" &&
                item.key() != "preserve_thinking") {
                return fail("unsupported chat template option: " + item.key());
            }
        }
        return parse_bool_option(*options, "enable_thinking",
                                 &output->enable_thinking) &&
               parse_bool_option(*options, "preserve_thinking",
                                 &output->preserve_thinking);
    }

    bool parse_completion_fields(Json& root, const std::string& served_model,
                                 CompletionRequest* output) {
        auto model = root.find("model");
        if (model == root.end() || !model->is_string()) {
            return fail("model must be a string");
        }
        output->model = model->get_ref<const std::string&>();
        if (output->model != served_model) return fail("model not found");

        auto session = root.find("session_id");
        if (session != root.end()) return fail("session_id is not supported");
        for (const char* name : {"tool_choice", "functions", "function_call",
                                 "response_format"}) {
            auto value = root.find(name);
            if (value != root.end() && !value->is_null() &&
                !(value->is_array() && value->empty()) &&
                !(value->is_string() && value->get<std::string>() == "none")) {
                return fail(std::string(name) + " is not supported");
            }
        }
        int n = 1;
        if (!integer_option(root, "n", 1, 1, &n) || n != 1) {
            return fail("only n=1 is supported");
        }
        if (!number(root, "temperature", 1.0, 0.0, 2.0,
                    &output->temperature)) return false;
        if (!number(root, "top_p", 1.0, 0.0,
                    1.0, &output->top_p) || output->top_p <= 0.0f) {
            return fail("top_p must be in (0, 1]");
        }
        if (!integer_option(root, "top_k", 0, 0, &output->top_k)) return false;
        if (!number(root, "presence_penalty", 0.0, -2.0, 2.0,
                    &output->presence_penalty)) return false;

        float min_p = 0.0f;
        if (!number(root, "min_p", 0.0, 0.0, 1.0, &min_p) || min_p != 0.0f) {
            return fail("only min_p=0 is supported");
        }
        float repetition = 1.0f;
        if (!number(root, "repetition_penalty", 1.0, 0.0, 100.0,
                    &repetition) || repetition != 1.0f) {
            return fail("only repetition_penalty=1 is supported");
        }

        auto maximum = root.find("max_completion_tokens");
        if (maximum == root.end()) maximum = root.find("max_tokens");
        if (maximum != root.end()) {
            if (!maximum->is_number_integer() || maximum->is_boolean()) {
                return fail("max_completion_tokens must be an integer");
            }
            const int64_t value = maximum->get<int64_t>();
            if (value <= 0 || value > std::numeric_limits<int>::max()) {
                return fail("max_completion_tokens must be positive");
            }
            output->max_tokens = static_cast<int>(value);
        }

        auto seed = root.find("seed");
        if (seed != root.end() && !seed->is_null()) {
            if (!seed->is_number_integer() || seed->is_boolean()) {
                return fail("seed must be an integer");
            }
            output->seed = static_cast<uint64_t>(seed->get<int64_t>());
            output->has_seed = true;
        }
        auto stream = root.find("stream");
        if (stream != root.end()) {
            if (!stream->is_boolean()) return fail("stream must be boolean");
            output->stream = stream->get<bool>();
        }
        auto stream_options = root.find("stream_options");
        if (stream_options != root.end() && !stream_options->is_null()) {
            if (!stream_options->is_object()) {
                return fail("stream_options must be an object");
            }
            auto usage = stream_options->find("include_usage");
            if (usage != stream_options->end()) {
                if (!usage->is_boolean()) {
                    return fail("stream_options.include_usage must be boolean");
                }
                output->include_usage = usage->get<bool>();
            }
        }
        auto logprobs = root.find("logprobs");
        if (logprobs != root.end() && !logprobs->is_null() &&
            !(logprobs->is_boolean() && !logprobs->get<bool>())) {
            return fail("logprobs are not implemented");
        }

        auto stop = root.find("stop");
        if (stop != root.end() && !stop->is_null()) {
            if (stop->is_string()) {
                output->stops.push_back(stop->get<std::string>());
            } else if (stop->is_array()) {
                if (stop->size() > 4) return fail("stop accepts at most 4 strings");
                for (const Json& item : *stop) {
                    if (!item.is_string() || item.get_ref<const std::string&>().empty()) {
                        return fail("stop values must be non-empty strings");
                    }
                    output->stops.push_back(item.get<std::string>());
                }
            } else {
                return fail("stop must be a string or array");
            }
            if (output->stops.empty() || output->stops.front().empty()) {
                return fail("stop values must be non-empty strings");
            }
        }
        return true;
    }

    bool validate_http_chat(const ChatRequest& chat) {
        if (chat.messages.empty()) return fail("messages must not be empty");
        bool has_user = false;
        for (size_t index = 0; index < chat.messages.size(); ++index) {
            const Message& message = chat.messages[index];
            if (message.role == Role::System && index != 0) {
                return fail("system/developer message must be first");
            }
            if (!message.tool_calls.empty() && message.role != Role::Assistant) {
                return fail("tool_calls require an assistant message");
            }
            if (message.content_is_null &&
                !(message.role == Role::Assistant &&
                  !message.tool_calls.empty())) {
                return fail("message content must be text");
            }
            for (const ContentPart& part : message.parts) {
                if (part.kind != ContentKind::Text) {
                    return fail("only text message content is supported");
                }
            }
            has_user = has_user || message.role == Role::User;
        }
        if (!has_user) return fail("messages must contain a user message");
        return true;
    }

    bool parse_root(Json& root, ChatRequest* output) {
        if (!root.is_object()) return fail("root must be an object");

        auto messages = root.find("messages");
        if (messages == root.end()) return fail("missing messages");
        if (!messages->is_array()) return fail("messages must be an array");

        output->messages.reserve(messages->size());
        for (Json& value : *messages) {
            Message message;
            if (!parse_message(value, &message)) return false;
            output->messages.push_back(std::move(message));
        }

        auto tools = root.find("tools");
        if (tools != root.end() && !tools->is_null()) {
            if (!tools->is_array()) return fail("tools must be an array");
            output->tools.reserve(tools->size());
            for (const Json& tool : *tools) {
                std::string text;
                if (!json_text(tool, &text)) return false;
                output->tools.push_back(std::move(text));
            }
        }

        if (!parse_bool_option(root, "add_generation_prompt",
                               &output->options.add_generation_prompt)) return false;
        if (!parse_bool_option(root, "enable_thinking",
                               &output->options.enable_thinking)) return false;
        if (!parse_bool_option(root, "preserve_thinking",
                               &output->options.preserve_thinking)) return false;
        if (!parse_bool_option(root, "add_vision_id",
                               &output->options.add_vision_id)) return false;
        return true;
    }

    std::string error_message_;
};

}  // namespace

Status parse_chat_request(const std::string& text, ChatRequest& output) {
    return Parser().parse(text, output);
}

Status parse_completion_request(const std::string& text,
                                const std::string& served_model,
                                int default_max_tokens,
                                CompletionRequest& output) {
    return Parser().parse_completion(
        text, served_model, default_max_tokens, output);
}

namespace {

Json usage_json(const CompletionUsage& usage) {
    return Json{
        {"prompt_tokens", usage.prompt_tokens},
        {"completion_tokens", usage.completion_tokens},
        {"total_tokens", usage.prompt_tokens + usage.completion_tokens},
        {"prompt_tokens_details", {{"cached_tokens", usage.cached_tokens}}},
    };
}

Json chunk_base(const std::string& id, int64_t created,
                const std::string& model) {
    return Json{{"id", id}, {"object", "chat.completion.chunk"},
                {"created", created}, {"model", model}};
}

Json generated_argument(const ToolCall& call, const ToolArgument& argument,
                        const std::vector<std::string>* tools) {
    if (!tools) return argument.text;
    for (const std::string& tool_text : *tools) {
        const Json tool = Json::parse(tool_text, nullptr, false);
        if (tool.is_discarded() || !tool.is_object()) continue;
        const auto function = tool.find("function");
        if (function == tool.end() || !function->is_object()) continue;
        const auto name = function->find("name");
        if (name == function->end() || !name->is_string() ||
            name->get_ref<const std::string&>() != call.name) continue;
        const auto parameters = function->find("parameters");
        if (parameters == function->end() || !parameters->is_object()) break;
        const auto properties = parameters->find("properties");
        if (properties == parameters->end() || !properties->is_object()) break;
        const auto property = properties->find(argument.name);
        if (property == properties->end() || !property->is_object()) break;
        const auto type = property->find("type");
        if (type == property->end() || !type->is_string()) break;
        const std::string& expected = type->get_ref<const std::string&>();
        if (expected == "string") return argument.text;

        Json parsed = Json::parse(argument.text, nullptr, false);
        if (parsed.is_discarded()) return argument.text;
        if ((expected == "integer" && parsed.is_number_integer() &&
             !parsed.is_boolean()) ||
            (expected == "number" && parsed.is_number()) ||
            (expected == "boolean" && parsed.is_boolean()) ||
            (expected == "object" && parsed.is_object()) ||
            (expected == "array" && parsed.is_array()) ||
            (expected == "null" && parsed.is_null())) return parsed;
        return argument.text;
    }
    return argument.text;
}

}  // namespace

std::string error_json(const std::string& message, const char* type,
                       const char* param, const char* code) {
    Json error{{"message", message}, {"type", type ? type : "server_error"},
               {"param", param ? Json(param) : Json(nullptr)},
               {"code", code ? Json(code) : Json(nullptr)}};
    return Json{{"error", std::move(error)}}.dump();
}

std::string models_json(const std::string& model) {
    return Json{{"object", "list"},
                {"data", Json::array({Json{{"id", model}, {"object", "model"},
                                           {"created", 0}, {"owned_by", "qwen3x"}}})}}.dump();
}

std::string completion_json(const std::string& id, int64_t created,
                            const std::string& model,
                            const std::string& reasoning,
                            const std::string& content,
                            bool include_reasoning,
                            const std::vector<ToolCall>& tool_calls,
                            const char* finish_reason,
                            const CompletionUsage& usage,
                            const std::vector<std::string>* tools) {
    Json message{{"role", "assistant"},
                 {"content", tool_calls.empty() || !content.empty()
                     ? Json(content) : Json(nullptr)},
                 {"refusal", nullptr}};
    if (include_reasoning) message["reasoning_content"] = reasoning;
    if (!tool_calls.empty()) {
        message["tool_calls"] = Json::array();
        for (const ToolCall& call : tool_calls) {
            Json arguments = Json::object();
            for (const ToolArgument& argument : call.arguments) {
                arguments[argument.name] = generated_argument(
                    call, argument, tools);
            }
            message["tool_calls"].push_back({
                {"id", call.id},
                {"type", "function"},
                {"function", {
                    {"name", call.name},
                    {"arguments", arguments.dump()},
                }},
            });
        }
    }
    Json choice{{"index", 0}, {"message", std::move(message)},
                {"logprobs", nullptr}, {"finish_reason", finish_reason}};
    return Json{{"id", id}, {"object", "chat.completion"},
                {"created", created}, {"model", model},
                {"choices", Json::array({std::move(choice)})},
                {"usage", usage_json(usage)}}.dump();
}

std::string completion_chunk_json(const std::string& id, int64_t created,
                                  const std::string& model,
                                  const char* delta_field,
                                  const std::string& delta,
                                  const char* finish_reason) {
    Json body = chunk_base(id, created, model);
    Json delta_json = Json::object();
    if (delta_field) delta_json[delta_field] = delta;
    body["choices"] = Json::array({Json{{"index", 0},
                                         {"delta", std::move(delta_json)},
                                         {"logprobs", nullptr},
                                         {"finish_reason", finish_reason
                                             ? Json(finish_reason) : Json(nullptr)}}});
    return body.dump();
}

std::string completion_tool_call_chunk_json(
    const std::string& id, int64_t created, const std::string& model,
    int index, const ToolCall& call, const std::vector<std::string>* tools) {
    Json arguments = Json::object();
    for (const ToolArgument& argument : call.arguments) {
        arguments[argument.name] = generated_argument(call, argument, tools);
    }
    Json body = chunk_base(id, created, model);
    body["choices"] = Json::array({Json{
        {"index", 0},
        {"delta", {{"tool_calls", Json::array({Json{
            {"index", index},
            {"id", call.id},
            {"type", "function"},
            {"function", {
                {"name", call.name},
                {"arguments", arguments.dump()},
            }},
        }})}}},
        {"logprobs", nullptr},
        {"finish_reason", nullptr},
    }});
    return body.dump();
}

std::string completion_usage_chunk_json(const std::string& id,
                                        int64_t created,
                                        const std::string& model,
                                        const CompletionUsage& usage) {
    Json body = chunk_base(id, created, model);
    body["choices"] = Json::array();
    body["usage"] = usage_json(usage);
    return body.dump();
}

}  // namespace q35_render
