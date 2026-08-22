// runtime.cpp -- resident Session selection, identity, prefix reuse and LRU.
//
// This file deliberately knows nothing about Model, State, Work or forward.
// It owns a fixed number of opaque q35_session objects and coordinates which
// request has exclusive access to each one.

#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "internal.h"

namespace {

enum class EntryState {
    FREE,
    IDLE,
    BUSY,
};

struct SessionEntry {
    q35_session* session = nullptr;
    std::string session_id;
    EntryState state = EntryState::FREE;
    uint64_t last_used = 0;
};

void write_error(char* output, size_t capacity, const char* message) {
    if (!output || capacity == 0) return;
    std::snprintf(output, capacity, "%s", message ? message : "unknown error");
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool is_prefix(const q35_session* session, const int* tokens, int count) {
    int live_count = 0;
    const int* live_tokens = q35_internal::session_tokens(session, &live_count);
    if (live_count > count) return false;
    for (int index = 0; index < live_count; ++index) {
        if (live_tokens[index] != tokens[index]) return false;
    }
    return true;
}

}  // namespace

struct q35_session_manager {
    std::mutex mutex;
    std::vector<SessionEntry> entries;
    uint64_t clock = 0;

    ~q35_session_manager() {
        for (SessionEntry& entry : entries) q35_session_destroy(entry.session);
    }
};

namespace {

SessionEntry* find_entry(q35_session_manager& manager, const q35_session* session) {
    for (SessionEntry& entry : manager.entries) {
        if (entry.session == session) return &entry;
    }
    return nullptr;
}

void reset(SessionEntry& entry) {
    q35_session_reset(entry.session, nullptr, 0);
    entry.session_id.clear();
    entry.state = EntryState::FREE;
    entry.last_used = 0;
}

}  // namespace

int q35_session_manager_create(q35_engine* engine, int session_count,
                                            int context_size, q35_session_manager** out,
                                            char* err, size_t errlen) {
    try {
        require(engine, "engine is null");
        require(out, "session manager output pointer is null");
        *out = nullptr;
        require(session_count > 0, "session_count must be positive");

        auto manager = std::make_unique<q35_session_manager>();
        manager->entries.resize(static_cast<size_t>(session_count));
        for (SessionEntry& entry : manager->entries) {
            const int result = q35_session_create(
                engine, context_size, &entry.session, err, errlen
            );
            if (result != Q35_OK) return result;
        }
        *out = manager.release();
        write_error(err, errlen, "");
        return Q35_OK;
    } catch (const std::exception& exception) {
        write_error(err, errlen, exception.what());
        return Q35_ERROR;
    } catch (...) {
        write_error(err, errlen, "unknown C++ exception");
        return Q35_ERROR;
    }
}

void q35_session_manager_destroy(q35_session_manager* manager) {
    delete manager;
}

int q35_session_manager_acquire(q35_session_manager* manager,
                                             const char* session_id,
                                             const int* tokens, int count,
                                             q35_session** out,
                                             char* err, size_t errlen) {
    try {
        require(manager, "session manager is null");
        require(out, "session output pointer is null");
        *out = nullptr;
        require(tokens, "tokens pointer is null");
        require(count > 0, "token sequence is empty");
        if (session_id) require(session_id[0] != '\0', "session_id is empty");

        std::lock_guard<std::mutex> lock(manager->mutex);
        SessionEntry* selected = nullptr;

        // session_id is a routing hint and always wins over prefix lookup.
        if (session_id) {
            for (SessionEntry& entry : manager->entries) {
                if (entry.session_id == session_id) {
                    if (entry.state == EntryState::BUSY) {
                        write_error(err, errlen, "session is already busy");
                        return Q35_BUSY;
                    }
                    selected = &entry;
                    break;
                }
            }
        }

        // Otherwise reuse the longest complete live prefix.
        if (!selected) {
            int best = -1;
            for (SessionEntry& entry : manager->entries) {
                if (entry.state != EntryState::IDLE ||
                    !is_prefix(entry.session, tokens, count)) continue;
                int live_count = 0;
                q35_internal::session_tokens(entry.session, &live_count);
                if (live_count > best) {
                    selected = &entry;
                    best = live_count;
                }
            }
        }

        // No prefix: take an unused Session, then the least recently used idle one.
        if (!selected) {
            for (SessionEntry& entry : manager->entries) {
                if (entry.state == EntryState::FREE) {
                    selected = &entry;
                    break;
                }
            }
        }
        if (!selected) {
            for (SessionEntry& entry : manager->entries) {
                if (entry.state != EntryState::IDLE) continue;
                if (!selected || entry.last_used < selected->last_used) selected = &entry;
            }
        }
        if (!selected) {
            write_error(err, errlen, "all sessions are busy");
            return Q35_BUSY;
        }

        selected->session_id = session_id ? session_id : "";
        selected->state = EntryState::BUSY;
        *out = selected->session;
        write_error(err, errlen, "");
        return Q35_OK;
    } catch (const std::exception& exception) {
        write_error(err, errlen, exception.what());
        return Q35_ERROR;
    } catch (...) {
        write_error(err, errlen, "unknown C++ exception");
        return Q35_ERROR;
    }
}

void q35_session_manager_release(q35_session_manager* manager,
                                              q35_session* session, bool keep) {
    if (!manager || !session) return;
    std::lock_guard<std::mutex> lock(manager->mutex);
    SessionEntry* entry = find_entry(*manager, session);
    if (!entry || entry->state != EntryState::BUSY) return;
    if (keep) {
        entry->state = EntryState::IDLE;
        entry->last_used = ++manager->clock;
    } else {
        reset(*entry);
    }
}

int q35_session_manager_forget(q35_session_manager* manager,
                                            const char* session_id,
                                            char* err, size_t errlen) {
    try {
        require(manager, "session manager is null");
        require(session_id && session_id[0], "session_id is empty");
        std::lock_guard<std::mutex> lock(manager->mutex);
        for (SessionEntry& entry : manager->entries) {
            if (entry.session_id != session_id) continue;
            if (entry.state == EntryState::BUSY) {
                write_error(err, errlen, "session is busy");
                return Q35_BUSY;
            }
            reset(entry);
            write_error(err, errlen, "");
            return Q35_OK;
        }
        write_error(err, errlen, "session not found");
        return Q35_NOT_FOUND;
    } catch (const std::exception& exception) {
        write_error(err, errlen, exception.what());
        return Q35_ERROR;
    } catch (...) {
        write_error(err, errlen, "unknown C++ exception");
        return Q35_ERROR;
    }
}
