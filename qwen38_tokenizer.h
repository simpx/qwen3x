// qwen38_tokenizer.h -- deliberately thin adapter around a proven tokenizer.
//
// Tokenization is peripheral infrastructure.  The model execution remains in
// qwen38.cpp; this file only exposes text <-> Qwen token IDs.

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace qwen38 {

class QwenTokenizer {
public:
    // Loads <model-directory>/tokenizer.json from the exact Qwen checkpoint.
    explicit QwenTokenizer(const char* model_directory);
    ~QwenTokenizer();

    QwenTokenizer(const QwenTokenizer&) = delete;
    QwenTokenizer& operator=(const QwenTokenizer&) = delete;

    std::vector<int> encode(std::string_view text);
    std::string decode(const std::vector<int>& token_ids);

private:
    void* handle_ = nullptr;
};

}  // namespace qwen38
