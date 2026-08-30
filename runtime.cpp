// runtime.cpp -- shared Engine/Session runtime and resident Session selection.
//
// This file owns token timelines, prefix/checkpoint policy, sampling,
// SessionManager, FREE/IDLE/BUSY and LRU. It only sees opaque backend Model/State
// handles; model math and the physical State snapshot live in engine.cpp.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

#include "internal.h"

namespace {

constexpr int IM_END_TOKEN = 248046;
constexpr int THINK_START_TOKEN = 248068;
constexpr int THINK_END_TOKEN = 248069;
constexpr int NEWLINE_TOKEN = 198;
constexpr int DOUBLE_NEWLINE_TOKEN = 271;
constexpr int MOCK_REASONING_TOKEN = 26003;
constexpr std::array<int, 15> MOCK_TARGET_TOKENS = {
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, IM_END_TOKEN,
    MOCK_REASONING_TOKEN, NEWLINE_TOKEN, THINK_END_TOKEN, DOUBLE_NEWLINE_TOKEN,
};

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

struct PrefixMatch {
    int common = 0;
    int reused = 0;
    bool use_checkpoint = false;
};

void write_error(char* output, size_t capacity, const char* message) {
    if (!output || capacity == 0) return;
    std::snprintf(output, capacity, "%s", message ? message : "unknown error");
}

int fail(char* err, size_t errlen, const char* message) {
    write_error(err, errlen, message);
    return Q35_ERROR;
}

int succeed(char* err, size_t errlen) {
    write_error(err, errlen, "");
    return Q35_OK;
}

int common_token_prefix(const std::vector<int>& saved,
                        const int* tokens, int count) {
    int common = 0;
    while (common < static_cast<int>(saved.size()) && common < count &&
           saved[common] == tokens[common]) {
        ++common;
    }
    return common;
}

int argmax_host(const std::vector<float>& values) {
    return static_cast<int>(
        std::max_element(values.begin(), values.end()) - values.begin());
}

uint64_t random_u64(uint64_t& state) {
    if (state == 0) state = 0x9e3779b97f4a7c15ULL;
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 0x2545f4914f6cdd1dULL;
}

double random_unit(uint64_t& state) {
    return (random_u64(state) >> 11) * 0x1.0p-53;
}

int sample_host(const std::vector<float>& logits, float temperature, int top_k,
                float top_p, float presence_penalty,
                const int* generated_tokens, int generated_count,
                uint64_t& rng, std::vector<int>& order,
                std::vector<float>& adjusted,
                std::vector<uint8_t>& penalty_marks,
                std::vector<int>& penalty_touched) {
    const int vocab = q35_backend::vocab_size();
    if (!std::isfinite(temperature) || temperature < 0.0f ||
        !std::isfinite(top_p) || top_p <= 0.0f || top_p > 1.0f ||
        !std::isfinite(presence_penalty) || presence_penalty < -2.0f ||
        presence_penalty > 2.0f || generated_count < 0 ||
        (generated_count > 0 && !generated_tokens)) {
        return -1;
    }

    const std::vector<float>* values = &logits;
    if (presence_penalty != 0.0f && generated_count > 0) {
        adjusted = logits;
        penalty_touched.clear();
        for (int index = 0; index < generated_count; ++index) {
            const int token = generated_tokens[index];
            if (token < 0 || token >= vocab) return -1;
        }
        for (int index = 0; index < generated_count; ++index) {
            const int token = generated_tokens[index];
            if (!penalty_marks[token]) {
                penalty_marks[token] = 1;
                penalty_touched.push_back(token);
                adjusted[token] -= presence_penalty;
            }
        }
        for (int token : penalty_touched) penalty_marks[token] = 0;
        values = &adjusted;
    }

    if (temperature == 0.0f) return argmax_host(*values);

    const int limit = top_k <= 0 ? vocab : std::min(top_k, vocab);
    order.resize(vocab);
    for (int token = 0; token < vocab; ++token) order[token] = token;
    if (limit < vocab || top_p < 1.0f) {
        std::partial_sort(order.begin(), order.begin() + limit, order.end(),
                          [&](int left, int right) {
                              return (*values)[left] > (*values)[right];
                          });
    }

    float maximum = -std::numeric_limits<float>::infinity();
    for (int index = 0; index < limit; ++index) {
        maximum = std::max(maximum, (*values)[order[index]]);
    }

    double total = 0.0;
    for (int index = 0; index < limit; ++index) {
        total += std::exp(((*values)[order[index]] - maximum) / temperature);
    }
    if (!(total > 0.0) || !std::isfinite(total)) return -1;

    const double nucleus = total * top_p;
    double kept = 0.0;
    int count = 0;
    do {
        kept += std::exp(((*values)[order[count]] - maximum) / temperature);
        ++count;
    } while (count < limit && kept < nucleus);

    const double target = random_unit(rng) * kept;
    double cumulative = 0.0;
    for (int index = 0; index < count; ++index) {
        cumulative += std::exp(((*values)[order[index]] - maximum) / temperature);
        if (target < cumulative) return order[index];
    }
    return order[count - 1];
}

}  // namespace

struct q35_engine {
    q35_backend::Model* model = nullptr;
    bool mock = false;
    std::vector<std::vector<float>> mock_logits;

    ~q35_engine() {
        q35_backend::model_destroy(model);
    }
};

struct q35_session {
    q35_engine* engine;
    q35_backend::State* state = nullptr;
    int capacity;
    std::vector<int> tokens;
    std::vector<float> logits;
    std::vector<int> sample_order;
    std::vector<float> adjusted_logits;
    std::vector<uint8_t> penalty_marks;
    std::vector<int> penalty_touched;
    std::vector<float> mock_checkpoint_logits;
    bool checkpoint_valid = false;
    int checkpoint_position = 0;
    bool logits_valid = false;

    q35_session(q35_engine* owner, int context_size)
        : engine(owner), capacity(context_size),
          logits(static_cast<size_t>(q35_backend::vocab_size())) {
        tokens.reserve(static_cast<size_t>(context_size));
        sample_order.reserve(static_cast<size_t>(q35_backend::vocab_size()));
        adjusted_logits.reserve(static_cast<size_t>(q35_backend::vocab_size()));
        penalty_marks.assign(static_cast<size_t>(q35_backend::vocab_size()), 0);
        penalty_touched.reserve(static_cast<size_t>(context_size));
        if (!engine->mock) {
            state = q35_backend::state_create(engine->model, context_size);
        }
    }

    ~q35_session() {
        q35_backend::state_destroy(state);
    }

    int position() const {
        return static_cast<int>(tokens.size());
    }

    void reset() {
        if (!engine->mock) q35_backend::state_reset(state);
        tokens.clear();
        checkpoint_valid = false;
        checkpoint_position = 0;
        logits_valid = false;
    }

    void append(const int* input, int count, bool compute_logits = true) {
        Q35_ASSERT(input && count > 0,
                   "session append input=%p count=%d",
                   static_cast<const void*>(input), count);
        if (engine->mock) {
            for (int index = 0; index < count; ++index) {
                const int token = input[index];
                const int previous_token = tokens.empty() ? -1 : tokens.back();
                const int next_position = position() + 1;
                size_t row = static_cast<size_t>(next_position % 10);
                if (previous_token == THINK_START_TOKEN && token == NEWLINE_TOKEN) {
                    row = 11;
                } else if (token == MOCK_REASONING_TOKEN) {
                    row = 12;
                } else if (previous_token == MOCK_REASONING_TOKEN &&
                           token == NEWLINE_TOKEN) {
                    row = 13;
                } else if (token == THINK_END_TOKEN) {
                    row = 14;
                } else if (token >= MOCK_TARGET_TOKENS.front() &&
                           token <= MOCK_TARGET_TOKENS[9]) {
                    row = static_cast<size_t>(
                        token - MOCK_TARGET_TOKENS.front() + 1);
                }
                logits = engine->mock_logits[row];
                tokens.push_back(token);
                LOG_TRACE("mock forward token=%d position=%d logits_row=%zu target=%d",
                          token, next_position, row, MOCK_TARGET_TOKENS[row]);
            }
        } else {
            q35_backend::state_forward(engine->model, state, input, count,
                                       compute_logits);
            tokens.insert(tokens.end(), input, input + count);
        }
        logits_valid = compute_logits || engine->mock;
    }

    void append(int token, bool compute_logits = true) {
        append(&token, 1, compute_logits);
    }

    PrefixMatch match_prefix(const int* request, int count) const {
        PrefixMatch match;
        match.common = common_token_prefix(tokens, request, count);
        const int live = position();
        if (live > 0 && match.common == live) {
            match.reused = live;
        } else if (checkpoint_valid && match.common >= checkpoint_position) {
            match.reused = checkpoint_position;
            match.use_checkpoint = true;
        }
        return match;
    }

    void save_checkpoint() {
        checkpoint_valid = false;
        if (engine->mock) {
            mock_checkpoint_logits = logits;
        } else {
            q35_backend::state_checkpoint_save(state);
        }
        checkpoint_position = position();
        checkpoint_valid = true;
        LOG_DEBUG("session checkpoint saved checkpoint_state_tokens=%d",
                 checkpoint_position);
    }

    void restore_checkpoint() {
        Q35_ASSERT(checkpoint_valid,
                   "restore_checkpoint valid=%d position=%d checkpoint=%d",
                   checkpoint_valid, position(), checkpoint_position);
        const int live_position = position();
        if (engine->mock) {
            logits = mock_checkpoint_logits;
        } else {
            q35_backend::state_checkpoint_restore(state);
        }
        Q35_ASSERT(tokens.size() >= static_cast<size_t>(checkpoint_position),
                   "token history=%zu checkpoint_position=%d",
                   tokens.size(), checkpoint_position);
        tokens.resize(static_cast<size_t>(checkpoint_position));
        logits_valid = true;
        LOG_DEBUG("session checkpoint restored live_state_tokens=%d "
                 "checkpoint_state_tokens=%d discarded_tokens=%d",
                 checkpoint_position, checkpoint_position,
                 live_position - checkpoint_position);
    }

    void copy_logits_to_host() {
        if (!engine->mock) {
            q35_backend::state_copy_logits(state, logits.data());
        }
    }
};

namespace q35_internal {

int session_checkpoint_state_tokens(const q35_session* session) {
    if (!session || !session->checkpoint_valid) return 0;
    return session->checkpoint_position;
}

int session_cache_hit_tokens(const q35_session* session,
                             const int* tokens, int count) {
    if (!session || !tokens || count <= 0) return 0;
    return session->match_prefix(tokens, count).reused;
}

}  // namespace q35_internal

int q35_engine_create(const q35_engine_options* options, q35_engine** out,
                      char* err, size_t errlen) {
    if (!options) return fail(err, errlen, "engine options are null");
    if (!options->bin_path || !options->bin_path[0]) {
        return fail(err, errlen, "bin_path is empty");
    }
    if (!out) return fail(err, errlen, "engine output pointer is null");
    *out = nullptr;

    std::unique_ptr<q35_engine> engine(new q35_engine());
    engine->mock = options->mock;
    if (engine->mock) {
        const int vocab = q35_backend::vocab_size();
        engine->mock_logits.resize(MOCK_TARGET_TOKENS.size());
        for (size_t row = 0; row < engine->mock_logits.size(); ++row) {
            auto& logits = engine->mock_logits[row];
            logits.assign(static_cast<size_t>(vocab), -1000.0f);
            const int target = MOCK_TARGET_TOKENS[row];
            logits[target] = 1000.0f;
            logits[(target + 1) % vocab] = 999.0f;
        }
        LOG_INFO("mock compute enabled logits_rows=%zu",
                 engine->mock_logits.size());
    } else {
        LOG_INFO("model load started bin=%s", options->bin_path);
        const auto started = std::chrono::steady_clock::now();
        engine->model = q35_backend::model_create(options->bin_path, err, errlen);
        if (!engine->model) return Q35_ERROR;
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        LOG_INFO("model load completed elapsed=%.3fs", elapsed);
    }
    LOG_DEBUG("model ready vocab=%d", q35_backend::vocab_size());
    *out = engine.release();
    return succeed(err, errlen);
}

void q35_engine_destroy(q35_engine* engine) {
    LOG_INFO("engine closing");
    delete engine;
}

int q35_session_create(q35_engine* engine, int context_size, q35_session** out,
                       char* err, size_t errlen) {
    if (!engine) return fail(err, errlen, "engine is null");
    if (!out) return fail(err, errlen, "session output pointer is null");
    *out = nullptr;
    if (context_size <= 0 || context_size > q35_backend::max_context()) {
        return fail(err, errlen, "context_size is outside 1..262144");
    }
    *out = new q35_session(engine, context_size);
    LOG_DEBUG("session created context_size=%d", context_size);
    return succeed(err, errlen);
}

void q35_session_destroy(q35_session* session) {
    if (session) LOG_DEBUG("session closing position=%d", session->position());
    delete session;
}

int q35_session_reset(q35_session* session, char* err, size_t errlen) {
    if (!session) return fail(err, errlen, "session is null");
    const int old_position = session->position();
    session->reset();
    LOG_DEBUG("session reset old_position=%d", old_position);
    return succeed(err, errlen);
}

int q35_session_sync(q35_session* session, const int* tokens, int count,
                     int checkpoint_at, int* cached_tokens,
                     char* err, size_t errlen) {
    if (cached_tokens) *cached_tokens = 0;
    if (!session) return fail(err, errlen, "session is null");
    if (!tokens) return fail(err, errlen, "tokens pointer is null");
    if (count <= 0) return fail(err, errlen, "token sequence is empty");
    if (count > session->capacity) {
        return fail(err, errlen, "token sequence exceeds session context");
    }
    if (checkpoint_at != -1 &&
        (checkpoint_at <= 0 || checkpoint_at > count)) {
        return fail(err, errlen, "checkpoint_at must be -1 or in [1,count]");
    }
    const int vocab = q35_backend::vocab_size();
    for (int index = 0; index < count; ++index) {
        if (tokens[index] < 0 || tokens[index] >= vocab) {
            return fail(err, errlen, "token is outside vocabulary");
        }
    }

    const int live = session->position();
    PrefixMatch match = session->match_prefix(tokens, count);
    int reused = match.reused;
    const bool checkpoint_already_saved =
        checkpoint_at > 0 && session->checkpoint_valid &&
        session->checkpoint_position == checkpoint_at &&
        match.common >= checkpoint_at;
    if (checkpoint_at >= 0 && checkpoint_at < reused &&
        !checkpoint_already_saved) {
        reused = 0;
        match.use_checkpoint = false;
    }

    const int tokens_to_prefill = count - reused;
    const char* cache_result =
        reused > 0 ? (match.use_checkpoint ? "hit_checkpoint" : "hit_live")
                   : (live == 0 ? "new" : "rebuild");
    const char* mode =
        tokens_to_prefill == 0 ? "cache_only" : "append_suffix";
    const int checkpoint_position =
        session->checkpoint_valid ? session->checkpoint_position : 0;
    if (cached_tokens) *cached_tokens = reused;
    LOG_DEBUG("session sync prompt_tokens=%d live_state_tokens=%d "
             "checkpoint_state_tokens=%d checkpoint_at=%d "
             "cache_result=%s cache_hit_tokens=%d to_prefill_tokens=%d",
             count, live, checkpoint_position, checkpoint_at,
             cache_result, reused, tokens_to_prefill);

    const auto started = std::chrono::steady_clock::now();
    if (match.use_checkpoint) {
        session->restore_checkpoint();
    } else if (reused == 0) {
        session->reset();
    }

    bool checkpoint_saved = checkpoint_already_saved;
    if (checkpoint_at == reused && checkpoint_at > 0 &&
        !checkpoint_saved) {
        session->save_checkpoint();
        checkpoint_saved = true;
    }
    LOG_DEBUG("session prefill started mode=%s prompt_tokens=%d "
             "cache_hit_tokens=%d to_prefill_tokens=%d",
             mode, count, reused, tokens_to_prefill);
    // checkpoint_at is an exact backend range boundary. Each backend chooses
    // how to traverse a range without crossing the saved snapshot position.
    int cursor = reused;
    if (checkpoint_at > cursor && checkpoint_at < count) {
        session->append(tokens + cursor, checkpoint_at - cursor, true);
        session->save_checkpoint();
        checkpoint_saved = true;
        cursor = checkpoint_at;
    }
    if (cursor < count) {
        session->append(tokens + cursor, count - cursor, true);
        if (session->position() == checkpoint_at) {
            session->save_checkpoint();
            checkpoint_saved = true;
        }
    }
    Q35_ASSERT(checkpoint_at <= 0 || checkpoint_saved,
               "checkpoint_at=%d reused=%d prompt_tokens=%d",
               checkpoint_at, reused, count);

    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    const int final_checkpoint =
        session->checkpoint_valid ? session->checkpoint_position : 0;
    LOG_INFO("session prefill completed prompt_tokens=%d "
             "live_state_tokens=%d checkpoint_state_tokens=%d "
             "cache_hit_tokens=%d to_prefill_tokens=%d elapsed=%.3fs",
             count, count, final_checkpoint, reused, tokens_to_prefill,
             elapsed);
    return succeed(err, errlen);
}

int q35_session_eval(q35_session* session, int token,
                     char* err, size_t errlen) {
    if (!session) return fail(err, errlen, "session is null");
    if (token < 0 || token >= q35_backend::vocab_size()) {
        return fail(err, errlen, "token is outside vocabulary");
    }
    if (session->position() >= session->capacity) {
        return fail(err, errlen, "session context is full");
    }
    const auto started = std::chrono::steady_clock::now();
    session->append(token);
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    LOG_TRACE("session evaluated token=%d position=%d elapsed=%.3fs",
              token, session->position(), elapsed);
    return succeed(err, errlen);
}

int q35_session_position(const q35_session* session) {
    return session ? session->position() : -1;
}

int q35_session_argmax(const q35_session* session) {
    if (!session || !session->logits_valid) return -1;
    const int token = session->engine->mock
        ? argmax_host(session->logits)
        : q35_backend::state_argmax(session->state);
    LOG_TRACE("argmax selected token=%d position=%d",
              token, session->position());
    return token;
}

int q35_session_sample(q35_session* session, float temperature, int top_k,
                       float top_p, float presence_penalty,
                       const int* generated_tokens, int generated_count,
                       uint64_t* rng) {
    if (!session || !session->logits_valid || !rng) return -1;
    session->copy_logits_to_host();
    const int token = sample_host(
        session->logits, temperature, top_k, top_p, presence_penalty,
        generated_tokens, generated_count, *rng, session->sample_order,
        session->adjusted_logits, session->penalty_marks,
        session->penalty_touched);
    LOG_TRACE("sample selected token=%d position=%d temperature=%.3f top_k=%d "
              "top_p=%.3f presence_penalty=%.3f generated_tokens=%d",
              token, session->position(), temperature, top_k, top_p,
              presence_penalty, generated_count);
    return token;
}

bool q35_token_is_stop(int token) {
    const bool stop = q35_backend::token_is_stop(token);
    if (stop) LOG_TRACE("stop token detected token=%d", token);
    return stop;
}

int q35_vocab_size(void) {
    return q35_backend::vocab_size();
}

int q35_session_copy_logits(const q35_session* session, float* output,
                            int capacity, char* err, size_t errlen) {
    if (!session) return fail(err, errlen, "session is null");
    if (!session->logits_valid) {
        return fail(err, errlen,
                    "session has no logits; sync or eval tokens first");
    }
    if (!output) return fail(err, errlen, "logits output pointer is null");
    if (capacity < q35_backend::vocab_size()) {
        return fail(err, errlen, "logits output is smaller than vocabulary");
    }
    if (session->engine->mock) {
        std::memcpy(output, session->logits.data(),
                    sizeof(float) * q35_backend::vocab_size());
    } else {
        q35_backend::state_copy_logits(session->state, output);
    }
    LOG_TRACE("logits copied count=%d position=%d",
              q35_backend::vocab_size(), session->position());
    return succeed(err, errlen);
}

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
    Q35_ASSERT(entry.session, "SessionEntry reset has null session");
    entry.session->reset();
    entry.state = EntryState::FREE;
    entry.last_used = 0;
}

}  // namespace

int q35_session_manager_create(q35_engine* engine, int session_count,
                               int context_size, q35_session_manager** out,
                               char* err, size_t errlen) {
    LOG_DEBUG("session manager create started sessions=%d context_size=%d",
              session_count, context_size);
    if (!engine) return fail(err, errlen, "engine is null");
    if (!out) return fail(err, errlen, "session manager output pointer is null");
    *out = nullptr;
    if (session_count <= 0) {
        return fail(err, errlen, "session_count must be positive");
    }

    std::unique_ptr<q35_session_manager> manager(new q35_session_manager());
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
    return succeed(err, errlen);
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
    if (!manager) return fail(err, errlen, "session manager is null");
    if (!out) return fail(err, errlen, "session output pointer is null");
    *out = nullptr;
    if (!tokens) return fail(err, errlen, "tokens pointer is null");
    if (count <= 0) return fail(err, errlen, "token sequence is empty");
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
    LOG_DEBUG("session release slot=%d result=%s live_state_tokens=%d "
             "checkpoint_state_tokens=%d",
             slot, keep ? "kept" : "cleared", live_state_tokens,
             checkpoint_state_tokens);
}
