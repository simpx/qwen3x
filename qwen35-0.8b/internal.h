#ifndef QWEN35_INTERNAL_H
#define QWEN35_INTERNAL_H

#include "qwen35.h"

namespace q35_internal {

// Read-only view of the Session timeline. The pointer remains valid until the
// next sync, eval or reset on this Session.
const int* session_tokens(const q35_session* session, int* count);

}  // namespace q35_internal

#endif
