// qwen38_tokenizer.cpp -- Qwen tokenizer adapter using Hugging Face tokenizers.
//
// The dependency is tokenizers-cpp's small Rust C ABI, compiled by the CMake
// build.  Loading the official tokenizer.json keeps Unicode normalization,
// ByteLevel BPE, and added/special tokens exact without making Transformers or
// Python a runtime dependency.

#include "qwen38_tokenizer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "tokenizers_c.h"

namespace qwen38 {
namespace {

[[noreturn]] void tokenizer_fail(const std::string& message) {
    std::fprintf(stderr, "qwen38: tokenizer: %s\n", message.c_str());
    std::exit(1);
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) tokenizer_fail("cannot open " + path.string());
    std::string result(std::istreambuf_iterator<char>(file), {});
    if (result.empty()) tokenizer_fail("tokenizer file is empty: " + path.string());
    return result;
}

}  // namespace

QwenTokenizer::QwenTokenizer(const char* model_directory) {
    if (!model_directory || !model_directory[0]) tokenizer_fail("model directory is empty");
    const std::filesystem::path tokenizer_path = std::filesystem::path(model_directory) / "tokenizer.json";
    const std::string tokenizer_json = read_file(tokenizer_path);
    handle_ = tokenizers_new_from_str(tokenizer_json.data(), tokenizer_json.size());
    if (!handle_) tokenizer_fail("failed to load " + tokenizer_path.string());
}

QwenTokenizer::~QwenTokenizer() {
    if (handle_) tokenizers_free(handle_);
}

std::vector<int> QwenTokenizer::encode(std::string_view text) {
    TokenizerEncodeResult result{};
    tokenizers_encode(handle_, text.data(), text.size(), 0, &result);
    std::vector<int> ids(result.token_ids, result.token_ids + result.len);
    tokenizers_free_encode_results(&result, 1);
    return ids;
}

std::string QwenTokenizer::decode(const std::vector<int>& token_ids) {
    static_assert(sizeof(int) == sizeof(uint32_t), "tokenizers C ABI requires 32-bit token IDs");
    tokenizers_decode(handle_, reinterpret_cast<const uint32_t*>(token_ids.data()), token_ids.size(), 0);
    const char* text = nullptr;
    size_t text_size = 0;
    tokenizers_get_decode_str(handle_, &text, &text_size);
    return std::string(text, text_size);
}

}  // namespace qwen38
