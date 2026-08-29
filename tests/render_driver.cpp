#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "render.h"

using q35_render::ChatRequest;
using q35_render::Renderer;
using q35_render::parse_chat_request;

namespace {

int report(const std::string& error) {
    std::cerr << "render-driver: " << error << '\n';
    return 1;
}

bool nibble(char value, int* output) {
    if (value >= '0' && value <= '9') *output = value - '0';
    else if (value >= 'a' && value <= 'f') *output = value - 'a' + 10;
    else if (value >= 'A' && value <= 'F') *output = value - 'A' + 10;
    else return false;
    return true;
}

bool unhex(const std::string& text, std::string* output) {
    if (text.size() % 2) return false;
    output->clear();
    output->reserve(text.size() / 2);
    for (size_t index = 0; index < text.size(); index += 2) {
        int high = 0;
        int low = 0;
        if (!nibble(text[index], &high) || !nibble(text[index + 1], &low)) {
            return false;
        }
        output->push_back(static_cast<char>((high << 4) | low));
    }
    return true;
}

std::string hex(const std::string& text) {
    constexpr char DIGITS[] = "0123456789abcdef";
    std::string output;
    output.reserve(text.size() * 2);
    for (unsigned char byte : text) {
        output.push_back(DIGITS[byte >> 4]);
        output.push_back(DIGITS[byte & 15]);
    }
    return output;
}

void print_tokens(const std::vector<int>& tokens) {
    for (size_t index = 0; index < tokens.size(); ++index) {
        if (index) std::cout << ' ';
        std::cout << tokens[index];
    }
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) return report("usage: render-driver RENDER_BIN MODE");

    std::string error;
    std::unique_ptr<Renderer> renderer(Renderer::create(argv[1], &error));
    if (!renderer) return report(error);

    const std::string mode = argv[2];
    if (mode == "batch-encode") {
        std::string line;
        while (std::getline(std::cin, line)) {
            std::string text;
            if (!unhex(line, &text)) return report("invalid hex input");
            std::vector<int> tokens;
            if (!renderer->encode(text, &tokens, &error)) return report(error);
            print_tokens(tokens);
        }
    } else if (mode == "batch-decode") {
        std::string line;
        while (std::getline(std::cin, line)) {
            std::istringstream input(line);
            int skip = 0;
            if (!(input >> skip) || (skip != 0 && skip != 1)) {
                return report("decode line needs a skip flag");
            }
            std::vector<int> tokens;
            int token = 0;
            while (input >> token) tokens.push_back(token);
            if (!input.eof()) return report("decode line contains an invalid token");
            std::string text;
            if (!renderer->decode(tokens, skip != 0, &text, &error)) {
                return report(error);
            }
            std::cout << hex(text) << '\n';
        }
    } else if (mode == "batch-chat") {
        std::string line;
        while (std::getline(std::cin, line)) {
            ChatRequest request;
            const q35_render::Status status = parse_chat_request(line, request);
            if (!status.ok()) return report(status.message());
            q35_render::RenderedPrompt rendered;
            if (!renderer->render(request, &rendered, &error)) {
                return report(error);
            }
            std::cout << hex(rendered.text) << '\n';
        }
    } else if (mode == "chat" || mode == "chat-ids") {
        const std::string input(
            std::istreambuf_iterator<char>(std::cin),
            std::istreambuf_iterator<char>()
        );
        ChatRequest request;
        const q35_render::Status status = parse_chat_request(input, request);
        if (!status.ok()) return report(status.message());
        q35_render::RenderedPrompt rendered;
        if (!renderer->render(request, &rendered, &error)) return report(error);
        if (mode == "chat") std::cout << rendered.text;
        else print_tokens(rendered.tokens);
    } else {
        return report("unknown mode");
    }
    return 0;
}
