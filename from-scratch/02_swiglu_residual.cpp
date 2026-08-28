// 第 02 课：SwiGLU FFN 与 residual connection。
//
// 阅读路线：
//   已经会：dot 组成 matrix-vector linear；RMSNorm 稳定 hidden 的整体大小。
//   本课只加：SiLU/SwiGLU 为 linear 加入非线性，再用 residual 保留原 hidden。
//   运行后看：hidden 怎样依次经过 norm、三个 linear、SwiGLU 和 residual。
//   下一课：FFN 只处理当前 token；先给之后的 attention 准备“位置”这个信息。
//
// 一个 Qwen pre-norm FFN 分支的完整数据流是：
//   n      = RMSNorm(hidden)
//   gate   = gate_proj @ n
//   up     = up_proj @ n
//   mixed  = SiLU(gate) * up                 // * 是逐元素乘法
//   branch = down_proj @ mixed
//   output = hidden + branch                 // residual
//
// 真实模型中 gate_proj 和 up_proj 都把 [hidden] 投到更宽的 [intermediate]，
// down_proj 再投回 [hidden]。本课明确使用 H=2、I=3：
//   gate_proj / up_proj: W[I,H] @ hidden[H] -> intermediate[I]
//   down_proj:           W[H,I] @ intermediate[I] -> branch[H]
// 这样可以直接看到 FFN 的“升维 -> 逐通道非线性 -> 降维”。
//
// 白话记忆：FFN 不读取别的 token；它是每个 token 自己的“小型非线性思考器”。up
// 提供候选信息，gate 决定每个候选通道开多大，down 再把变宽后的结果压回 hidden。
// residual 的 x + ffn(x) 则表示“保留原句的表示，只加上本分支学到的修正”，不是覆盖 x。

#include <cassert>
#include <cmath>
#include <cstdio>

namespace lesson02 {

constexpr int kHidden = 2;
constexpr int kIntermediate = 3;
constexpr float kEpsilon = 1e-6f;

// 目的/直觉：比较浮点结果时容许微小舍入误差。
// 数学：      close(a,b) = |a-b| < 1e-5。
// 实现：      对差值取绝对值，再与阈值比较。
bool close(float left, float right) { return std::fabs(left - right) < 1e-5f; }

// 目的/直觉：复用第 00 课的点积，为 linear 计算一个输出元素。
// 数学：      dot(a,b) = sum_{i=0}^{N-1}(a_i*b_i) -> scalar。
// 实现：      遍历 count=N 个对应位置；升维时 N=H，降维时 N=I。
float dot(const float* left, const float* right, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) sum += left[i] * right[i];
    return sum;
}

// 目的/直觉：gate_proj/up_proj 把较窄的 hidden[H] 升维成更多 intermediate[I] 通道。
// 数学：      W[I,H] @ x[H] -> y[I]；y_i = dot(W[i],x)，每次 dot 长度为 H。
// 实现：      遍历 I 个输出行，每行与 hidden 做一次第 00 课的 dot。
void linear_hidden_to_intermediate(
        const float weight[kIntermediate][kHidden], const float* input, float* output) {
    for (int output_row = 0; output_row < kIntermediate; ++output_row) {
        output[output_row] = dot(weight[output_row], input, kHidden);
    }
}

// 目的/直觉：down_proj 把 SwiGLU 处理后的 intermediate[I] 降回 hidden[H]，
//             这样 branch 才能与原 hidden 做 residual add。
// 数学：      W[H,I] @ x[I] -> y[H]；y_h = dot(W[h],x)，每次 dot 长度为 I。
// 实现：      遍历 H 个输出行，每行与 intermediate 做一次 dot。
void linear_intermediate_to_hidden(
        const float weight[kHidden][kIntermediate], const float* input, float* output) {
    for (int output_row = 0; output_row < kHidden; ++output_row) {
        output[output_row] = dot(weight[output_row], input, kIntermediate);
    }
}

// 目的/直觉：复用第 01 课，先稳定 hidden 的整体大小，再送入 FFN 的 linear。
// 数学：      m=(1/H)*sum_i(x_i^2)，r=1/sqrt(m+eps)，y_i=x_i*r*(1+w_i)。
// 实现：      第一遍求均方，计算一次 inverse_rms，第二遍逐维乘 (1+weight[i])。
void qwen_rmsnorm(const float* input, const float* weight, float* output) {
    float mean_square = 0.0f;
    for (int i = 0; i < kHidden; ++i) mean_square += input[i] * input[i];
    mean_square /= kHidden;

    const float inverse_rms = 1.0f / std::sqrt(mean_square + kEpsilon);
    for (int i = 0; i < kHidden; ++i) {
        output[i] = input[i] * inverse_rms * (1.0f + weight[i]);
    }
}

// 目的/直觉：把任意实数平滑压到 0..1，作为 SiLU 中的软开关。
// 数学：      sigmoid(x) = 1 / (1 + exp(-x))。
// 实现：      x>=0 时算 exp(-x)，x<0 时改写成 exp(x)/(1+exp(x))，避免指数溢出。
float sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + std::exp(-x));
    const float exp_x = std::exp(x);
    return exp_x / (1.0f + exp_x);
}

// 目的/直觉：给 gate 加入平滑非线性；正值大体通过，负值被压小但不硬截断。
// 数学：      SiLU(x) = x * sigmoid(x)。
// 实现：      调用 sigmoid(x)，再乘回原来的 x。
float silu(float x) { return x * sigmoid(x); }

// 目的/直觉：gate 决定 up 中每个候选通道开多大；gate=0 时该通道输出必为 0。
// 数学：      mixed_i = SiLU(gate_i) * up_i；三个向量 shape 都是 [intermediate]。
// 实现：      遍历每个 intermediate 通道，做 SiLU 后与 up 对应位置逐元素相乘。
void swiglu(const float* gate, const float* up, float* output) {
    for (int i = 0; i < kIntermediate; ++i) output[i] = silu(gate[i]) * up[i];
}

// 目的/直觉：不让 FFN 覆盖原 hidden，只把分支学到的修正量加回去，保留直通路径。
// 数学：      hidden'_i = hidden_i + branch_i；两个输入与输出 shape 都是 [H]。
// 实现：      遍历 H 个位置，原地执行 hidden[i] += branch[i]；不是拼接。
void residual_add(float* hidden, const float* branch) {
    for (int i = 0; i < kHidden; ++i) hidden[i] += branch[i];
}

// 目的/直觉：用一条完整的 pre-norm FFN 数据流，验证新学的 SwiGLU 和 residual
//             确实接在已经学过的 RMSNorm/linear 后面，而不是凭空出现 gate/up。
// 数学：      n=RMSNorm(h)，g=Wg@n，u=Wu@n，z=SiLU(g)*u，b=Wd@z，out=h+b。
// 实现：      依次调用每个小函数；特意让 gate[0]=0，并用 assert 锁定中间值和 residual。
void self_test() {
    const float hidden[kHidden] = {3.0f, 4.0f};
    const float norm_weight[kHidden] = {0.0f, 0.0f};
    const float gate_projection[kIntermediate][kHidden] = {
        {4.0f, -3.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}};
    const float up_projection[kIntermediate][kHidden] = {
        {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
    const float down_projection[kHidden][kIntermediate] = {
        {1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 1.0f}};

    float normalized[kHidden] = {};
    float gate[kIntermediate] = {};
    float up[kIntermediate] = {};
    float mixed[kIntermediate] = {};
    float branch[kHidden] = {};
    float output[kHidden] = {hidden[0], hidden[1]};

    qwen_rmsnorm(hidden, norm_weight, normalized);
    linear_hidden_to_intermediate(gate_projection, normalized, gate);
    linear_hidden_to_intermediate(up_projection, normalized, up);
    swiglu(gate, up, mixed);
    linear_intermediate_to_hidden(down_projection, mixed, branch);
    residual_add(output, branch);

    // normalized=[3r,4r]，所以 gate[0]=4*(3r)-3*(4r)=0，该 intermediate 通道被关闭。
    assert(close(gate[0], 0.0f));
    assert(close(mixed[0], 0.0f));
    assert(close(mixed[1], silu(gate[1]) * up[1]));
    assert(close(branch[0], mixed[1]));
    assert(close(branch[1], mixed[1] + mixed[2]));
    assert(close(output[0], hidden[0] + branch[0]));
    assert(close(output[1], hidden[1] + branch[1]));
}

}  // namespace lesson02

// 目的/直觉：把完整 FFN 的每个中间向量打印出来，读者可以沿数据流逐步核对。
// 数学：      hidden -> norm -> {gate,up} -> SwiGLU -> down -> residual。
// 实现：      重复 self_test 的同一组玩具权重，并按计算顺序打印所有阶段。
int main() {
    lesson02::self_test();

    const float hidden[lesson02::kHidden] = {3.0f, 4.0f};
    const float norm_weight[lesson02::kHidden] = {0.0f, 0.0f};
    const float gate_projection[lesson02::kIntermediate][lesson02::kHidden] = {
        {4.0f, -3.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}};
    const float up_projection[lesson02::kIntermediate][lesson02::kHidden] = {
        {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
    const float down_projection[lesson02::kHidden][lesson02::kIntermediate] = {
        {1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 1.0f}};

    float normalized[lesson02::kHidden] = {};
    float gate[lesson02::kIntermediate] = {};
    float up[lesson02::kIntermediate] = {};
    float mixed[lesson02::kIntermediate] = {};
    float branch[lesson02::kHidden] = {};
    float output[lesson02::kHidden] = {hidden[0], hidden[1]};
    std::printf("output initialized to hidden: [%.6f, %.6f]\n", output[0], output[1]);

    lesson02::qwen_rmsnorm(hidden, norm_weight, normalized);
    lesson02::linear_hidden_to_intermediate(gate_projection, normalized, gate);
    lesson02::linear_hidden_to_intermediate(up_projection, normalized, up);
    lesson02::swiglu(gate, up, mixed);
    lesson02::linear_intermediate_to_hidden(down_projection, mixed, branch);
    lesson02::residual_add(output, branch);

    std::printf("hidden [H=2]:       [%.6f, %.6f]\n", hidden[0], hidden[1]);
    std::printf("RMSNorm [H=2]:      [%.6f, %.6f]\n", normalized[0], normalized[1]);
    std::printf("gate_proj [I=3]:    [%.6f, %.6f, %.6f]\n", gate[0], gate[1], gate[2]);
    std::printf("up_proj [I=3]:      [%.6f, %.6f, %.6f]\n", up[0], up[1], up[2]);
    std::printf("SwiGLU [I=3]:       [%.6f, %.6f, %.6f]\n", mixed[0], mixed[1], mixed[2]);
    std::printf("down_proj [H=2]:    [%.6f, %.6f]\n", branch[0], branch[1]);
    std::printf("after residual [H=2]: [%.6f, %.6f]\n", output[0], output[1]);
}
