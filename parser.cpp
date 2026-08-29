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
            return fail("message.content arrays are not supported yet");
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
            return fail("message.tool_calls are not supported yet");
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
