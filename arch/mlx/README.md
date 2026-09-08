# MLX C++ backend

This optional backend implements the Qwen3.6-35B-A3B text model using MLX 0.32.2.
The executable keeps the existing C++17 runtime, renderer, tokenizer and HTTP
server. Only the MLX translation unit uses C++20 and exceptions. Python is used
for offline packing and numerical tests, never for production inference.

## Build

The tested machine is Apple M5 Pro (16 GPU cores), macOS 26.3.2, Xcode 26.6.
CMake and the Metal Toolchain are required. With the prepared environment:

```sh
make mlx-deps
make -j4 mlx mlx-library mlx-test
```

The `mlx-deps` target pins MLX v0.32.2 at
`1f8e74e3f12f31365464a6867c6579f0e9b29d85`. It uses precompiled Metal kernels
(`MLX_METAL_JIT=OFF`): the earlier JIT build had BF16 sigmoid rounding differences
from the reference wheel. Re-enabling JIT requires complete numerical validation.
MLX lives under `build/`; the default CPU build needs none of these dependencies.
For another prefix, pass `MLX_ROOT=/absolute/path` to make.

## Weights and runtime files

The input is `mlx-community/Qwen3.6-35B-A3B-4bit`, revision
`38740b847e4cb78f352aba30aa41c76e08e6eb46`. Preserve its mixed affine4/affine8
quantization and BF16 scales/biases. Offline packing only copies text tensor bytes:

```sh
python3 scripts/pack_mlx.py \
  build/models/mlx-community-Qwen3.6-35B-A3B-4bit \
  build/qwen36-35b-a3b-mlx-affine4-model.bin
```

The output is a safetensors container with a `mlx-affine-v1` metadata marker,
model ID, fixed model configuration and per-tensor dtype/shape. Its `.bin` suffix
does not make it interchangeable with the CPU/Metal Q4_0 format. The loader checks
the marker, contiguous ranges, EOF, expected tensors, shapes and quantized dtypes.
The original snapshot directory (or its config.json) is also accepted for testing.

Deployment currently needs four files: `qwen3x-mlx`, the MLX model bin,
`qwen3x-render.bin`, and `mlx.metallib`. The MLX library is linked statically;
the shader resource must still be present. Set `Q3X_MLX_METALLIB` to its absolute
path when moving the executable. The compiled default points to the build prefix.
No venv, Python interpreter, MLX dylib or second server is used by the executable.

## Validation and tuning

`tests/mlx_reference.py` runs the reference and C++ library in separate processes.
Use the pinned development environment: Python 3.12, mlx/ mlx-metal 0.32.2,
mlx-lm 0.31.3, transformers 5.16.1. Reference vectors compare the same input tokens,
weights, dtypes, prefill schedule and last-position-only output projection.
The initial short tests require bit-identical logits and greedy continuations.
Ordinary full-batch vs token-at-a-time BF16 reductions can differ; that difference
is recorded separately, not hidden by an enlarged implementation tolerance.

```sh
build/mlx-venv/bin/python tests/mlx_reference.py reference
build/mlx-venv/bin/python tests/mlx_reference.py check \
  --model build/qwen36-35b-a3b-mlx-affine4-model.bin
```

`Q3X_MLX_CHUNK` selects prefill chunk size (1..4096). Benchmark commands explicitly
record it; decode always evaluates one token. GPU work is serialized within the
backend. Checkpoint arrays retain immutable state; functional slice updates allow
MLX to reuse unshared KV storage without overwriting saved checkpoints.

The test library exports `q3x_mlx_memory_stats` for active/peak/cache allocator
counters. Darwin RSS alone does not fully describe GPU unified-memory usage.
Do not infer model correctness or long-context acceptance from a successful build.

## Correspondence with the CPU baseline

The existing Qwen3.6 branch in `engine.cpp` is the mathematical baseline; this
backend adds no new model semantics and therefore needs no CPU forward change.

| CPU path | MLX path and layout |
| --- | --- |
| `forward`, `embed`, `rms`, `residual_add` | `forward`, `Linear::embed`, `fast::rms_norm`; the same embedding, 40-layer mixer/MoE loop and final head. Attention runs every fourth layer. |
| `attention`, `rope` | `attention`; per-head query/gate split, Q/K normalization, partial RoPE, causal GQA, sigmoid output gate. CPU KV is token-major; MLX KV is `[1, KVH, T, AD]`. |
| `deltanet`, `conv_step`, `delta_rule`, `gated_rms` | `delta` and `delta_source`; depthwise causal convolution, decay, gated state update and output normalization. CPU recurrent state is `[VH, KD, VD]`; MLX stores `[1, VH, VD, KD]`. |
| `router_top_k`, `moe` | `moe`; softmax routing, eight selected experts with renormalized scores, SwiGLU and sigmoid-gated shared expert. CPU stores combined gate/up matrices; MLX keeps separate tensors. |
| checkpoint save/restore | The same prefix and logits semantics; CPU copies buffers, MLX retains immutable arrays and uses functional updates. |

Numerical paths intentionally differ. CPU accumulates in FP32 and reads the
official BF16/Q4_0 layout; MLX preserves the community affine4/affine8 weights and
BF16 activations. The community checkpoint already contains shifted RMS weights
(`1 + w`), whereas CPU adds one when reading the official weights. Delta Q/K
normalization also follows the pinned mlx-lm RMS formulation: its denominator is
equivalent to `sqrt(sum(x*x) + KD*EPS)`, while CPU L2 uses
`sqrt(sum(x*x) + EPS)`. BF16 rounding and reduction schedules differ as well.
These are explicit numerical differences, not a claim of CPU/MLX bit equality.
The exact-logit oracle uses the same community weights and schedule in mlx-lm.

The previous overnight validation recorded exact agreement on short full-logit
fixtures and on 61,415/126,976-token prefix/checkpoint fixtures. That version
included MLX-specific cancellation and recoverable-error paths, removed from
this change during review. Its source hashes and cancellation/soak results do
not describe the current version. Real-model long tests have not been rerun after removal;
build and small GPU fixtures are separate checks.

Interactive Pi testing subsequently exposed incomplete tool calls ending at a
model EOS before the output budget. This remains unresolved, including a case
after compaction with no prefix cache reuse. Successful numerical fixtures and
service soak tests do not establish reliable completion of arbitrary agent tasks.

## Boundaries for tomorrow's review

- Model loading catches exceptions and returns an error through the existing
  factory interface; failed initialization makes the executable exit before
  serving requests. Forward catches execution exceptions, logs the operation and
  reason, then aborts. State creation, reset and checkpoint copies use direct
  operations without per-function catches; uncaught exceptions terminate through
  the C++ runtime, without a guaranteed project log message. Restoring a missing
  checkpoint is an internal invariant failure and uses `Q3X_ASSERT`.
  Recovery has not been established and is deferred in [TODO.md](../../TODO.md).
  No failure-state polling or MLX-specific runtime build is needed. Existing
  request validation still rejects invalid requests before model computation.
- Request cancellation is a separate service capability, deferred in
  [TODO.md](../../TODO.md). The existing HTTP output path can detect disconnects,
  but an in-progress prefill/forward runs until it returns. No request callback
  is stored in MLX state. The public C ABI remains unchanged.
- No MLX types enter the public ABI or the teaching CPU forward.
- Quantized weights use an optional backend-specific container; no generic tensor
  framework was introduced. The dependency build and extra shader file remain
  explicit distribution work.
- The GatedDeltaNet recurrence is adapted from the pinned mlx-lm implementation;
  see NOTICE for attribution and license.
