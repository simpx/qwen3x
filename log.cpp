// log.cpp -- process-wide logging shared by every native component.

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "spdlog/logger.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "internal.h"

namespace {

q35_log_callback log_callback = nullptr;
void* log_user_data = nullptr;
q35_log_level log_level = Q35_LOG_INFO;
std::shared_ptr<spdlog::logger> process_logger;
std::shared_ptr<spdlog::logger> audit_logger;

const char* file_name(const char* path) {
    if (!path) return "?";
    const char* slash = std::strrchr(path, '/');
    const char* backslash = std::strrchr(path, '\\');
    const char* separator = slash;
    if (backslash && (!separator || backslash > separator)) separator = backslash;
    return separator ? separator + 1 : path;
}

spdlog::level::level_enum spd_level(q35_log_level level) {
    switch (level) {
        case Q35_LOG_TRACE: return spdlog::level::trace;
        case Q35_LOG_DEBUG: return spdlog::level::debug;
        case Q35_LOG_WARN: return spdlog::level::warn;
        case Q35_LOG_ERROR: return spdlog::level::err;
        default: return spdlog::level::info;
    }
}

bool prepare_file(const char* file, const char* kind,
                  char* err, size_t errlen) {
    const std::filesystem::path path(file);
    std::error_code filesystem_error;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(
            path.parent_path(), filesystem_error);
    }
    if (filesystem_error) {
        if (err && errlen) {
            std::snprintf(err, errlen, "cannot create %s directory: %s", kind,
                          path.parent_path().string().c_str());
        }
        return false;
    }
    std::FILE* probe = std::fopen(file, "ab");
    if (!probe) {
        if (err && errlen) {
            std::snprintf(err, errlen, "cannot open %s file: %s", kind, file);
        }
        return false;
    }
    std::fclose(probe);
    return true;
}

}  // namespace

void q35_log_set_callback(q35_log_callback callback,
                          void* user_data,
                          q35_log_level level) {
    log_callback = callback;
    log_user_data = user_data;
    log_level = level >= Q35_LOG_TRACE && level <= Q35_LOG_ERROR
        ? level : Q35_LOG_INFO;
}

namespace q35_internal {

bool log_configure(q35_log_level level, const char* file,
                   size_t max_bytes, size_t backups,
                   char* err, size_t errlen) {
    if (err && errlen) err[0] = '\0';
    if (file && file[0] && (max_bytes == 0 || backups == 0)) {
        if (err && errlen) {
            std::snprintf(err, errlen,
                          "log file requires positive size and backups");
        }
        return false;
    }

    std::vector<spdlog::sink_ptr> sinks;
    if (file && file[0]) {
        // With exceptions disabled spdlog treats an open failure as fatal.
        // Probe it first so ordinary path/permission errors remain recoverable.
        if (!prepare_file(file, "log", err, errlen)) return false;
        sinks.push_back(
            std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                file, max_bytes, backups));
    } else {
        sinks.push_back(std::make_shared<spdlog::sinks::stderr_color_sink_mt>());
    }

    process_logger = std::make_shared<spdlog::logger>(
        "qwen35", sinks.begin(), sinks.end());
    process_logger->set_level(spd_level(level));
    process_logger->set_pattern(
        "[%Y-%m-%d %H:%M:%S.%e] [%t] [%^%l%$] [%s:%#] %v");
    process_logger->flush_on(
        file && file[0] ? spdlog::level::info : spdlog::level::warn);
    process_logger->set_error_handler([](const std::string& message) {
        std::fprintf(stderr, "qwen35: logging error: %s\n", message.c_str());
    });
    log_level = level;
    return true;
}

void log_shutdown() {
    if (process_logger) process_logger->flush();
    process_logger.reset();
}

bool audit_configure(const char* file, char* err, size_t errlen) {
    if (err && errlen) err[0] = '\0';
    if (!file || !file[0]) return true;
    if (!prepare_file(file, "audit", err, errlen)) return false;
    const std::filesystem::path path(file);
    std::error_code filesystem_error;
    std::filesystem::permissions(
        path,
        std::filesystem::perms::owner_read |
            std::filesystem::perms::owner_write,
        std::filesystem::perm_options::replace, filesystem_error);
    if (filesystem_error) {
        if (err && errlen) {
            std::snprintf(err, errlen,
                          "cannot set audit log permissions: %s", file);
        }
        return false;
    }
    auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(file, false);
    audit_logger = std::make_shared<spdlog::logger>("audit", std::move(sink));
    audit_logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] %v");
    audit_logger->set_level(spdlog::level::info);
    audit_logger->flush_on(spdlog::level::info);
    audit_logger->set_error_handler([](const std::string& message) {
        std::fprintf(stderr, "qwen35: audit logging error: %s\n",
                     message.c_str());
    });
    return true;
}

void audit_shutdown() {
    if (audit_logger) audit_logger->flush();
    audit_logger.reset();
}

void audit_write(const char* event, const char* request_id,
                 const char* session_id, const char* detail,
                 const char* payload, size_t payload_size) {
    if (!audit_logger) return;
    char header[2048];
    const int length = std::snprintf(
        header, sizeof(header),
        "event=%s request_id=%s session_id=%s%s%s bytes=%zu",
        event ? event : "-",
        request_id ? request_id : "-", session_id ? session_id : "-",
        detail && detail[0] ? " " : "", detail && detail[0] ? detail : "",
        payload_size);
    if (length < 0) return;
    std::string message(header, static_cast<size_t>(
        std::min<int>(length, sizeof(header) - 1)));
    if (payload_size && payload) {
        message.push_back('\n');
        message.append(payload, payload_size);
    }
    audit_logger->info(message);
}

void logf(q35_log_level level, const char* file, int line,
          const char* format, ...) {
    if ((!process_logger && !log_callback) || level < log_level) return;

    char message[1024];
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(message, sizeof(message), format, arguments);
    va_end(arguments);

    const char* short_file = file_name(file);
    if (process_logger) {
        process_logger->log(
            spdlog::source_loc(short_file, line, ""),
            spd_level(level), message);
    }
    if (log_callback) {
        log_callback(log_user_data, level, short_file, line, message);
    }
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
