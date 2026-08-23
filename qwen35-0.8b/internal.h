#ifndef QWEN35_INTERNAL_H
#define QWEN35_INTERNAL_H

#include "qwen35.h"

namespace q35_internal {

// Return the longest complete Session checkpoint that prefixes tokens.
// Zero means this Session has no reusable checkpoint for the request.
int session_reusable_prefix(const q35_session* session,
                            const int* tokens, int count);

// Format one log message and synchronously pass it to the host callback.
void logf(q35_log_level level, const char* file, int line,
          const char* format, ...);

}  // namespace q35_internal

// __VA_ARGS__ always contains at least the format string.
#define LOG_DEBUG(...) \
    q35_internal::logf(Q35_LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_INFO(...) \
    q35_internal::logf(Q35_LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARN(...) \
    q35_internal::logf(Q35_LOG_WARN, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...) \
    q35_internal::logf(Q35_LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)

#endif
