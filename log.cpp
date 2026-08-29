// log.cpp -- process-wide logging shared by every native component.

#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "internal.h"

namespace {

q35_log_callback log_callback = nullptr;
void* log_user_data = nullptr;
q35_log_level log_level = Q35_LOG_INFO;

const char* file_name(const char* path) {
    if (!path) return "?";
    const char* slash = std::strrchr(path, '/');
    const char* backslash = std::strrchr(path, '\\');
    const char* separator = slash;
    if (backslash && (!separator || backslash > separator)) separator = backslash;
    return separator ? separator + 1 : path;
}

}  // namespace

void q35_log_set_callback(q35_log_callback callback,
                          void* user_data,
                          q35_log_level level) {
    log_callback = callback;
    log_user_data = user_data;
    log_level = level >= Q35_LOG_DEBUG && level <= Q35_LOG_ERROR
        ? level : Q35_LOG_INFO;
}

namespace q35_internal {

void logf(q35_log_level level, const char* file, int line,
          const char* format, ...) {
    if (!log_callback || level < log_level) return;

    char message[1024];
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(message, sizeof(message), format, arguments);
    va_end(arguments);

    log_callback(log_user_data, level, file_name(file), line, message);
}

void report_assertion(const char* expression, const char* file, int line,
                      const char* format, ...) {
    char detail[1024];
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(detail, sizeof(detail), format, arguments);
    va_end(arguments);

    char message[1280];
    std::snprintf(message, sizeof(message),
                  "assertion '%s' failed: %s",
                  expression ? expression : "?", detail);
    std::fprintf(stderr, "qwen35: %s:%d: %s\n",
                 file_name(file), line, message);
    std::fflush(stderr);

    if (log_callback && Q35_LOG_ERROR >= log_level) {
        log_callback(log_user_data, Q35_LOG_ERROR,
                     file_name(file), line, message);
    }
}

}  // namespace q35_internal
