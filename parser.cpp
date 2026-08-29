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

        if (role == "system") *output = Role::System;
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

}  // namespace q35_render
