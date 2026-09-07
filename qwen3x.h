#ifndef QWEN3X_H
#define QWEN3X_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct q3x_engine q3x_engine;
typedef struct q3x_session q3x_session;
typedef struct q3x_session_manager q3x_session_manager;

enum {
    Q3X_OK = 0,
    Q3X_ERROR = -1,
    Q3X_BUSY = -2,
};

/* Engine: one loaded, read-only supported Qwen text model. */

typedef enum {
    Q3X_LOG_TRACE = -1,
    Q3X_LOG_DEBUG = 0,
    Q3X_LOG_INFO = 1,
    Q3X_LOG_WARN = 2,
    Q3X_LOG_ERROR = 3,
} q3x_log_level;

/*
 * Called synchronously on the thread that produced the log. All strings are
 * valid only during the call. The host owns user_data and the callback's
 * lifetime; callbacks should be fast, must not throw, and must not call back
 * into q3x APIs.
 */
typedef void (*q3x_log_callback)(void* user_data,
                                 q3x_log_level level,
                                 const char* file,
                                 int line,
                                 const char* message);

/*
 * Process-wide logger shared by Engine, Session and SessionManager.
 * Configure it before starting native work. Do not change it until every
 * concurrent q3x call has stopped.
 */
void q3x_log_set_callback(q3x_log_callback callback,
                          void* user_data,
                          q3x_log_level level);

typedef struct {
    const char* bin_path;
    /* Use lightweight State updates and a fixed logits bank instead of model math. */
    bool mock;
} q3x_engine_options;

int q3x_engine_create(const q3x_engine_options* options, q3x_engine** out,
                      char* err, size_t errlen);
void q3x_engine_destroy(q3x_engine* engine);
uint32_t q3x_engine_model_id(const q3x_engine* engine);

/*
 * Session: one mutable token timeline containing State, Work and logits.
 * Engine must outlive every Session created from it. One Session is
 * single-writer and must not be advanced concurrently.
 */

int q3x_session_create(q3x_engine* engine, int context_size, q3x_session** out,
                       char* err, size_t errlen);
void q3x_session_destroy(q3x_session* session);

/* Stable UUID for this Session's lifetime; pointer dies with the Session. */
const char* q3x_session_id(const q3x_session* session);

/* Clear the token timeline while retaining all allocated buffers. */
int q3x_session_reset(q3x_session* session, char* err, size_t errlen);

/*
 * Bring the Session to exactly tokens[count]. If its live State or saved checkpoint
 * is a prefix, only the new suffix runs through forward; otherwise State is
 * reset and the complete sequence is prefetched again. cached_tokens receives
 * the number of prompt tokens whose existing State was reused; it may be NULL.
 * After tokens[0:checkpoint_at] have been evaluated, save that State as the
 * Session's additional checkpoint. checkpoint_at must be in [1,count], or -1
 * to keep the existing checkpoint without creating a new one.
 */
int q3x_session_sync(q3x_session* session, const int* tokens, int count,
                     int checkpoint_at,
                     int* cached_tokens,
                     char* err, size_t errlen);

/* Append one token and update State/logits. */
int q3x_session_eval(q3x_session* session, int token,
                     char* err, size_t errlen);

int q3x_session_position(const q3x_session* session);
int q3x_session_argmax(const q3x_session* session);

/*
 * Sample one token from logits. top_k <= 0 disables top-k filtering.
 * presence_penalty is applied once to each distinct generated token; prompt
 * tokens are deliberately excluded, matching vLLM/OpenAI semantics.
 */
int q3x_session_sample(q3x_session* session, float temperature, int top_k,
                       float top_p, float presence_penalty,
                       const int* generated_tokens, int generated_count,
                       uint64_t* rng);

bool q3x_token_is_stop(int token);
int q3x_vocab_size(void);

/* Copy the current logits[V]. capacity must be at least q3x_vocab_size(). */
int q3x_session_copy_logits(const q3x_session* session, float* output, int capacity,
                            char* err, size_t errlen);

/*
 * SessionManager: a fixed pool of preallocated Sessions.
 *
 * It reuses Sessions by token prefix, tracks FREE/IDLE/BUSY state and applies
 * LRU replacement. It does not know about HTTP, JSON, tokenization, chat
 * formats or disk cache.
 * Engine must outlive its SessionManager.
 */

int q3x_session_manager_create(q3x_engine* engine, int session_count,
                               int context_size, q3x_session_manager** out,
                               char* err, size_t errlen);
/* No Session may remain acquired when the manager is destroyed. */
void q3x_session_manager_destroy(q3x_session_manager* manager);

/*
 * Select and acquire one Session exclusively.
 *
 * 1. Prefer the IDLE Session with the longest reusable prefix of tokens[count].
 * 2. Otherwise use a FREE/LRU Entry.
 *
 * The function only selects the Session and marks it BUSY; the caller must
 * still call q3x_session_sync(). Returns Q3X_BUSY when every Session is BUSY.
 */
int q3x_session_manager_acquire(q3x_session_manager* manager,
                                const int* tokens, int count,
                                q3x_session** out,
                                char* err, size_t errlen);

/*
 * Release exclusive access. keep=true retains the Session as IDLE for future
 * prefix reuse. keep=false resets it and makes the Entry FREE.
 */
void q3x_session_manager_release(q3x_session_manager* manager,
                                 q3x_session* session, bool keep);

#ifdef __cplusplus
}
#endif

#endif
