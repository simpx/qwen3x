#pragma once

#include <algorithm>
#include <dispatch/dispatch.h>
#include <unistd.h>

namespace q3x_cpu::apple {

// GCD owns the threads; this call finishes every row before returning.
template <typename Row>
void parallel_rows(int rows, const Row& row) {
    static const long online = sysconf(_SC_NPROCESSORS_ONLN);
    const size_t tasks = std::min(static_cast<size_t>(rows),
                                  static_cast<size_t>(online > 0 ? online : 1));
    struct Context { int rows; size_t tasks; const Row* row; };
    Context context {rows, tasks, &row};
    dispatch_apply_f(tasks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        &context, [](void* opaque, size_t task) {
            const auto& c = *static_cast<const Context*>(opaque);
            const int begin = static_cast<int>(task * c.rows / c.tasks);
            const int end = static_cast<int>((task + 1) * c.rows / c.tasks);
            for (int i = begin; i < end; ++i) (*c.row)(i);
        });
}

}  // namespace q3x_cpu::apple
