#ifndef QWEN35_ENGINE_H
#define QWEN35_ENGINE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Stable, narrow C ABI.
 *
 * Engine  = one loaded, read-only Qwen3.5-0.8B model.
 * Session = one mutable token timeline: State + Work + logits.
 *
 * Engine must outlive every Session created from it. A Session is single-writer:
 * the runtime must not call sync/eval concurrently on the same handle.
 */
typedef struct q35_engine q35_engine;
typedef struct q35_session q35_session;

enum {
    Q35_OK = 0,
    Q35_ERROR = 1,
};

int q35_engine_create(const char* weights_path, q35_engine** out,
                      char* error, size_t error_capacity);
void q35_engine_destroy(q35_engine* engine);

int q35_session_create(q35_engine* engine, int context_size, q35_session** out,
                       char* error, size_t error_capacity);
void q35_session_destroy(q35_session* session);

/* Clear the token timeline while retaining all allocated buffers. */
int q35_session_reset(q35_session* session, char* error, size_t error_capacity);

/*
 * Bring the Session to exactly tokens[count]. If the live timeline is already
 * a prefix, only the new suffix runs through forward; otherwise State is reset
 * and the complete sequence is prefetched again.
 */
int q35_session_sync(q35_session* session, const int* tokens, int count,
                     char* error, size_t error_capacity);

/* Append one token and update State/logits. */
int q35_session_eval(q35_session* session, int token,
                     char* error, size_t error_capacity);

int q35_session_position(const q35_session* session);
int q35_session_argmax(const q35_session* session);
int q35_session_is_stop_token(int token);
int q35_vocab_size(void);

/* Copy the current logits[V]. capacity must be at least q35_vocab_size(). */
int q35_session_copy_logits(const q35_session* session, float* output, int capacity,
                            char* error, size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
