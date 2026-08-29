// parser.cpp -- the only JSON-aware part of the C++ text boundary.
//
//   JSON text -> ordered DOM -> ChatRequest
//
// The DOM is local to parse_chat_request(). render.cpp only sees the plain
// structs declared in render.h.

#include "render.h"

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
        if (arguments->is_string()) {
            return fail("encoded tool_call.arguments are not supported yet");
        }
        if (!arguments->is_object()) {
            return fail("tool_call.arguments must be an object");
        }

        output->arguments.reserve(arguments->size());
        for (auto& item : arguments->items()) {
            if (!item.value().is_string()) {
                return fail("tool argument values must be strings for now");
            }
            ToolArgument argument;
            argument.name = item.key();
            argument.text = move_string(item.value());
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
            for (Json& value : *content) {
                ContentPart part;
                if (!parse_content_part(value, &part)) return false;
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
            for (Json& value : *calls) {
                ToolCall call;
                if (!parse_tool_call(value, &call)) return false;
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
            return fail("tools are not supported yet");
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
