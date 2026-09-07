#ifndef QWEN3X_INTERNAL_H
#define QWEN3X_INTERNAL_H

#include <cassert>
#include <cstdlib>

#include "qwen3x.h"

// runtime.cpp 只通过这组不透明操作访问一个计算后端。构建时由根目录
// engine.cpp、arch/cuda/engine.cu 或 arch/metal/engine.mm 实现其中一个。
namespace q3x_backend {

struct Model;
struct State;

Model* model_create(const char* path, char* err, size_t errlen);
void model_destroy(Model* model);

State* state_create(Model* model, int context_size);
void state_destroy(State* state);
void state_reset(State* state);
void state_forward(Model* model, State* state,
                   const int* tokens, int count, bool compute_logits);
void state_checkpoint_save(State* state);
void state_checkpoint_restore(State* state);
int state_argmax(const State* state);
void state_copy_logits(const State* state, float* output);
uint32_t model_id(const Model* model);

int vocab_size();
int max_context();
bool token_is_stop(int token);

}  // namespace q3x_backend

namespace q3x_internal {

// Configure the process logger before worker threads start. An empty file uses
// stderr; a non-empty file replaces stderr with a rotating file sink.
bool log_configure(q3x_log_level level, const char* file,
                   size_t max_bytes, size_t backups,
                   char* err, size_t errlen);
void log_shutdown();

// Full request/response audit is opt-in and always uses a separate file.
bool audit_configure(const char* file, char* err, size_t errlen);
void audit_shutdown();
void audit_write(const char* event, const char* request_id,
                 const char* session_id, const char* detail,
                 const char* payload, size_t payload_size);

// Format one log message and synchronously pass it to the host callback.
void logf(q3x_log_level level, const char* file, int line,
          const char* format, ...);

// Print an invariant failure even when no log callback is installed.
void report_assertion(const char* expression, const char* file, int line,
                      const char* format, ...);

}  // namespace q3x_internal

// __VA_ARGS__ always contains at least the format string.
#define LOG_TRACE(...) \
    q3x_internal::logf(Q3X_LOG_TRACE, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_DEBUG(...) \
    q3x_internal::logf(Q3X_LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_INFO(...) \
    q3x_internal::logf(Q3X_LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARN(...) \
    q3x_internal::logf(Q3X_LOG_WARN, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...) \
    q3x_internal::logf(Q3X_LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)

// Evaluate condition once, report useful context, then retain standard assert
// behavior. abort() keeps invariant failures fatal in NDEBUG builds too.
#define Q3X_ASSERT(condition, ...)                                      \
    do {                                                                \
        const bool q3x_assert_ok = static_cast<bool>(condition);         \
        if (!q3x_assert_ok) {                                           \
            q3x_internal::report_assertion(                             \
                #condition, __FILE__, __LINE__, __VA_ARGS__);            \
            assert(q3x_assert_ok);                                      \
            std::abort();                                               \
        }                                                               \
    } while (false)

#endif
