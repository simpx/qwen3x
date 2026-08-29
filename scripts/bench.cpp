// CPU baseline for Engine prefill/decode without HTTP or tokenization.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "qwen35.h"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s WEIGHTS PREFILL_TOKENS DECODE_TOKENS\n",
                     argv[0]);
        return 2;
    }
    const int prefill_count = std::atoi(argv[2]);
    const int decode_count = std::atoi(argv[3]);
    if (prefill_count <= 0 || decode_count <= 0) return 2;

    char error[512]{};
    q35_engine* engine = nullptr;
    q35_engine_options options{argv[1], false};
    if (q35_engine_create(&options, &engine, error, sizeof(error)) != Q35_OK) {
        std::fprintf(stderr, "bench: %s\n", error);
        return 1;
    }
    q35_session* session = nullptr;
    if (q35_session_create(engine, prefill_count + decode_count, &session,
                           error, sizeof(error)) != Q35_OK) {
        std::fprintf(stderr, "bench: %s\n", error);
        q35_engine_destroy(engine);
        return 1;
    }
    std::vector<int> prompt(static_cast<size_t>(prefill_count));
    for (int index = 0; index < prefill_count; ++index) {
        prompt[index] = 100 + index % 1000;
    }
    const auto prefill_start = std::chrono::steady_clock::now();
    q35_session_sync(session, prompt.data(), prompt.size(), prompt.size(),
                     nullptr, error, sizeof(error));
    const auto decode_start = std::chrono::steady_clock::now();
    for (int index = 0; index < decode_count; ++index) {
        q35_session_eval(session, 100, error, sizeof(error));
    }
    const auto finished = std::chrono::steady_clock::now();
    const double prefill = std::chrono::duration<double>(
        decode_start - prefill_start).count();
    const double decode = std::chrono::duration<double>(
        finished - decode_start).count();
    std::printf("prefill  %d tokens  %.3f s  %.3f tok/s\n",
                prefill_count, prefill, prefill_count / prefill);
    std::printf("decode   %d tokens  %.3f s  %.3f tok/s\n",
                decode_count, decode, decode_count / decode);
    q35_session_destroy(session);
    q35_engine_destroy(engine);
    return 0;
}
