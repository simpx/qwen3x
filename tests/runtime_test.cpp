#include <cstdio>
#include <vector>

#include "qwen35.h"

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "runtime-test: %s\n", message);
    ++failures;
}

}  // namespace

int main() {
    char error[512]{};
    q35_engine* engine = nullptr;
    q35_engine_options options{"unused-in-mock-mode", true};
    check(q35_engine_create(&options, &engine, error, sizeof(error)) == Q35_OK,
          "create mock Engine");
    if (!engine) return 1;

    q35_session_manager* manager = nullptr;
    check(q35_session_manager_create(engine, 2, 64, &manager,
                                     error, sizeof(error)) == Q35_OK,
          "create SessionManager");
    const std::vector<int> prompt{10, 42, 99, 7};
    q35_session* first = nullptr;
    check(q35_session_manager_acquire(manager, prompt.data(), prompt.size(),
                                      &first, error, sizeof(error)) == Q35_OK,
          "acquire first Session");
    int cached = -1;
    check(q35_session_sync(first, prompt.data(), prompt.size(), prompt.size(),
                           &cached, error, sizeof(error)) == Q35_OK && cached == 0,
          "initial sync");
    const int greedy = q35_session_argmax(first);
    check(greedy >= 0 && greedy < q35_vocab_size(), "mock argmax");
    uint64_t rng_a = 123;
    uint64_t rng_b = 123;
    const int sample_a = q35_session_sample(
        first, 1.0f, 0, 1.0f, 0.0f, nullptr, 0, &rng_a);
    const int sample_b = q35_session_sample(
        first, 1.0f, 0, 1.0f, 0.0f, nullptr, 0, &rng_b);
    check(sample_a == sample_b && rng_a == rng_b,
          "sampling is deterministic for one seed");
    check(q35_session_eval(first, greedy, error, sizeof(error)) == Q35_OK,
          "decode one token");
    q35_session_manager_release(manager, first, true);

    q35_session* reused = nullptr;
    check(q35_session_manager_acquire(manager, prompt.data(), prompt.size(),
                                      &reused, error, sizeof(error)) == Q35_OK,
          "reacquire cached Session");
    check(q35_session_sync(reused, prompt.data(), prompt.size(), prompt.size(),
                           &cached, error, sizeof(error)) == Q35_OK &&
          cached == static_cast<int>(prompt.size()),
          "restore prompt checkpoint");

    q35_session* second = nullptr;
    q35_session* unavailable = nullptr;
    check(q35_session_manager_acquire(manager, prompt.data(), prompt.size(),
                                      &second, error, sizeof(error)) == Q35_OK,
          "acquire second slot");
    check(q35_session_manager_acquire(manager, prompt.data(), prompt.size(),
                                      &unavailable, error, sizeof(error)) == Q35_BUSY,
          "all busy returns Q35_BUSY");
    q35_session_manager_release(manager, second, false);
    q35_session_manager_release(manager, reused, true);

    q35_session_manager_destroy(manager);
    q35_engine_destroy(engine);
    if (failures) return 1;
    std::puts("runtime-test: ok");
    return 0;
}
