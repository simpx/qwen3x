#ifndef QWEN35_INTERNAL_H
#define QWEN35_INTERNAL_H

#include <cassert>
#include <cstdlib>

#include "qwen35.h"

// runtime.cpp 只通过这组不透明操作访问一个计算后端。构建时由根目录
// engine.cpp、arch/x86/engine.cpp、arch/cuda/engine.cu 或 Metal 实现其中一个。
namespace q35_backend {

struct Model;
struct State;

Model* model_create(const char* path, char* err, size_t errlen);
void model_destroy(Model* model);

State* state_create(Model* model, int context_size);
void state_destroy(State* state);
void state_reset(State* state);
void state_forward(Model* model, State* state, int token, bool compute_logits);
void state_checkpoint_save(State* state);
void state_checkpoint_restore(State* state);
int state_argmax(const State* state);
void state_copy_logits(const State* state, float* output);

int vocab_size();
int max_context();
bool token_is_stop(int token);

}  // namespace q35_backend

namespace q35_internal {

// Configure the process logger before worker threads start. stderr is always
// enabled; a non-empty file adds a rotating file sink.
bool log_configure(q35_log_level level, const char* file,
                   size_t max_bytes, size_t backups,
                   char* err, size_t errlen);
void log_shutdown();

// Format one log message and synchronously pass it to the host callback.
void logf(q35_log_level level, const char* file, int line,
          const char* format, ...);

// Print an invariant failure even when no log callback is installed.
void report_assertion(const char* expression, const char* file, int line,
                      const char* format, ...);

}  // namespace q35_internal

// __VA_ARGS__ always contains at least the format string.
#define LOG_TRACE(...) \
    q35_internal::logf(Q35_LOG_TRACE, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_DEBUG(...) \
    q35_internal::logf(Q35_LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_INFO(...) \
    q35_internal::logf(Q35_LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARN(...) \
    q35_internal::logf(Q35_LOG_WARN, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...) \
    q35_internal::logf(Q35_LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)

// Evaluate condition once, report useful context, then retain standard assert
// behavior. abort() keeps invariant failures fatal in NDEBUG builds too.
#define Q35_ASSERT(condition, ...)                                      \
    do {                                                                \
        const bool q35_assert_ok = static_cast<bool>(condition);         \
        if (!q35_assert_ok) {                                           \
            q35_internal::report_assertion(                             \
                #condition, __FILE__, __LINE__, __VA_ARGS__);            \
            assert(q35_assert_ok);                                      \
            std::abort();                                               \
        }                                                               \
    } while (false)

#endif
