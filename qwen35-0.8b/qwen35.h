#ifndef QWEN35_H
#define QWEN35_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct q35_engine q35_engine;
typedef struct q35_session q35_session;
typedef struct q35_session_manager q35_session_manager;

enum {
    Q35_OK = 0,
    Q35_ERROR = -1,
    Q35_BUSY = -2,
    Q35_NOT_FOUND = -3,
};

/* Engine: one loaded, read-only Qwen3.5-0.8B model. */

typedef struct {
    const char* weights_path;
} q35_engine_options;

int q35_engine_create(const q35_engine_options* options, q35_engine** out,
                      char* err, size_t errlen);
void q35_engine_destroy(q35_engine* engine);

/*
 * Session: one mutable token timeline containing State, Work and logits.
 * Engine must outlive every Session created from it. One Session is
 * single-writer and must not be advanced concurrently.
 */

int q35_session_create(q35_engine* engine, int context_size, q35_session** out,
                       char* err, size_t errlen);
void q35_session_destroy(q35_session* session);

/* Clear the token timeline while retaining all allocated buffers. */
int q35_session_reset(q35_session* session, char* err, size_t errlen);

/*
 * Bring the Session to exactly tokens[count]. If the live timeline is already
 * a prefix, only the new suffix runs through forward; otherwise State is reset
 * and the complete sequence is prefetched again.
 */
int q35_session_sync(q35_session* session, const int* tokens, int count,
                     char* err, size_t errlen);

/* Append one token and update State/logits. */
int q35_session_eval(q35_session* session, int token,
                     char* err, size_t errlen);

int q35_session_position(const q35_session* session);
int q35_session_argmax(const q35_session* session);

/* Sample one token from logits. top_k <= 0 disables top-k filtering. */
int q35_session_sample(q35_session* session, float temperature, int top_k,
                       float top_p, uint64_t* rng);

bool q35_token_is_stop(int token);
int q35_vocab_size(void);

/* Copy the current logits[V]. capacity must be at least q35_vocab_size(). */
int q35_session_copy_logits(const q35_session* session, float* output, int capacity,
                            char* err, size_t errlen);

/*
 * SessionManager: a fixed pool of preallocated Sessions.
 *
 * It binds optional server-issued session IDs, reuses anonymous Sessions by
 * token prefix, tracks FREE/IDLE/BUSY state and applies LRU replacement. It
 * does not know about HTTP, JSON, tokenization, chat formats or disk cache.
 * Engine must outlive its SessionManager.
 */

int q35_session_manager_create(q35_engine* engine, int session_count,
                               int context_size, q35_session_manager** out,
                               char* err, size_t errlen);
void q35_session_manager_destroy(q35_session_manager* manager);

/*
 * Select and acquire one Session exclusively.
 *
 * 1. Reuse the Entry bound to session_id, when it exists.
 * 2. Otherwise prefer an IDLE Session whose complete token timeline is a
 *    prefix of tokens[count].
 * 3. Otherwise use a FREE/LRU Entry.
 *
 * A non-NULL session_id is copied when a new binding is created. NULL means an
 * anonymous OpenAI-style request. The function only selects the Session and
 * marks it BUSY; the caller must still call q35_session_sync(). Returns
 * Q35_BUSY when the bound Session or every Session is already BUSY.
 */
int q35_session_manager_acquire(q35_session_manager* manager,
                                const char* session_id,
                                const int* tokens, int count,
                                q35_session** out,
                                char* err, size_t errlen);

/*
 * Release exclusive access. keep=true retains either a named or anonymous
 * Session as IDLE for future reuse. keep=false resets it, removes any ID
 * binding and makes the Entry FREE.
 */
void q35_session_manager_release(q35_session_manager* manager,
                                 q35_session* session, bool keep);

/* Forget and reset an IDLE named Session. Returns Q35_NOT_FOUND or Q35_BUSY. */
int q35_session_manager_forget(q35_session_manager* manager,
                               const char* session_id,
                               char* err, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif
