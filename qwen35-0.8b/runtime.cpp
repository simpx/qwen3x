// runtime.cpp -- resident Session selection, prefix reuse and LRU.
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
    entry.state = EntryState::FREE;
    entry.last_used = 0;
}

}  // namespace

int q35_session_manager_create(q35_engine* engine, int session_count,
                                            int context_size, q35_session_manager** out,
                                            char* err, size_t errlen) {
    try {
        LOG_DEBUG("session manager create started sessions=%d context_size=%d",
                  session_count, context_size);
        require(engine, "engine is null");
        require(out, "session manager output pointer is null");
        *out = nullptr;
        require(session_count > 0, "session_count must be positive");

        auto manager = std::make_unique<q35_session_manager>();
        manager->entries.resize(static_cast<size_t>(session_count));
        for (size_t slot = 0; slot < manager->entries.size(); ++slot) {
            SessionEntry& entry = manager->entries[slot];
            const int result = q35_session_create(
                engine, context_size, &entry.session, err, errlen
            );
            if (result != Q35_OK) return result;
            LOG_DEBUG("session manager slot created slot=%zu", slot);
        }
        LOG_INFO("session manager created sessions=%d context_size=%d",
                 session_count, context_size);
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
    if (manager) {
        LOG_INFO("session manager closing sessions=%zu", manager->entries.size());
    }
    delete manager;
}

int q35_session_manager_acquire(q35_session_manager* manager,
                                             const int* tokens, int count,
                                             q35_session** out,
                                             char* err, size_t errlen) {
    try {
        require(manager, "session manager is null");
        require(out, "session output pointer is null");
        *out = nullptr;
        require(tokens, "tokens pointer is null");
        require(count > 0, "token sequence is empty");
        LOG_DEBUG("session lookup started prompt_tokens=%d", count);

        std::unique_lock<std::mutex> lock(manager->mutex);
        SessionEntry* selected = nullptr;
        const char* selection = nullptr;
        int selected_cache_hit_tokens = 0;

        // 比较每个空闲 Session 的 live/checkpoint；
        // reusable 是它能完整复用的 token 数，选择其中最大的一个。
        int best = 0;
        for (size_t slot = 0; slot < manager->entries.size(); ++slot) {
            SessionEntry& entry = manager->entries[slot];
            if (entry.state != EntryState::IDLE) continue;
            const int cache_hit_tokens = q35_internal::session_cache_hit_tokens(
                entry.session, tokens, count
            );
            const int live_state_tokens = q35_session_position(entry.session);
            const int checkpoint_state_tokens =
                q35_internal::session_checkpoint_state_tokens(entry.session);
            LOG_DEBUG("session candidate slot=%zu prompt_tokens=%d "
                      "live_state_tokens=%d checkpoint_state_tokens=%d "
                      "cache_hit_tokens=%d",
                      slot, count, live_state_tokens, checkpoint_state_tokens,
                      cache_hit_tokens);
            if (cache_hit_tokens > best) {
                selected = &entry;
                best = cache_hit_tokens;
                selection = "prefix";
                selected_cache_hit_tokens = cache_hit_tokens;
            }
        }

        // No prefix: take an unused Session, then the least recently used idle one.
        if (!selected) {
            for (SessionEntry& entry : manager->entries) {
                if (entry.state == EntryState::FREE) {
                    selected = &entry;
                    selection = "free";
                    break;
                }
            }
        }
        if (!selected) {
            for (SessionEntry& entry : manager->entries) {
                if (entry.state != EntryState::IDLE) continue;
                if (!selected || entry.last_used < selected->last_used) {
                    selected = &entry;
                    selection = "lru";
                }
            }
        }
        if (!selected) {
            write_error(err, errlen, "all sessions are busy");
            lock.unlock();
            LOG_WARN("session lookup failed reason=all_slots_busy");
            return Q35_BUSY;
        }

        const int slot = static_cast<int>(selected - manager->entries.data());
        const int live_state_tokens = q35_session_position(selected->session);
        const int checkpoint_state_tokens =
            q35_internal::session_checkpoint_state_tokens(selected->session);
        const char* cache_result = selected_cache_hit_tokens > 0 ?
                                   (selected_cache_hit_tokens == live_state_tokens ?
                                    "hit_live" : "hit_checkpoint") :
                                   (live_state_tokens == 0 ? "new" : "rebuild");
        const int to_prefill_tokens = count - selected_cache_hit_tokens;
        selected->state = EntryState::BUSY;
        *out = selected->session;
        write_error(err, errlen, "");
        lock.unlock();
        LOG_INFO("session acquire slot=%d selection=%s prompt_tokens=%d "
                 "live_state_tokens=%d checkpoint_state_tokens=%d "
                 "cache_result=%s cache_hit_tokens=%d to_prefill_tokens=%d",
                 slot, selection, count, live_state_tokens, checkpoint_state_tokens,
                 cache_result, selected_cache_hit_tokens, to_prefill_tokens);
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
    std::unique_lock<std::mutex> lock(manager->mutex);
    SessionEntry* entry = find_entry(*manager, session);
    if (!entry || entry->state != EntryState::BUSY) return;
    const int slot = static_cast<int>(entry - manager->entries.data());
    if (keep) {
        entry->state = EntryState::IDLE;
        entry->last_used = ++manager->clock;
    } else {
        reset(*entry);
    }
    const int live_state_tokens = q35_session_position(entry->session);
    const int checkpoint_state_tokens =
        q35_internal::session_checkpoint_state_tokens(entry->session);
    lock.unlock();
    LOG_INFO("session release slot=%d result=%s live_state_tokens=%d "
             "checkpoint_state_tokens=%d",
             slot, keep ? "kept" : "cleared", live_state_tokens,
             checkpoint_state_tokens);
}
