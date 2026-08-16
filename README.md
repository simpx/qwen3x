# qwen38.cpp

A small, readable Qwen inference engine for CPU, Metal, and CUDA.

`qwen38.cpp` is an executable specification, not a general inference
framework. The core forward remains direct, fixed-model C++ that can be read
top-to-bottom: embedding → norm → DeltaNet or attention → residual → MLP →
logits.

## Teaching rebuild: active

This repository is now being rebuilt as a progressive C++ inference course.
The active reading path is [lessons/](lessons/README.md): every source file
adds exactly one concept, runs with hand-checkable toy weights, and has its own
self-test. The intended final chapter is one text-only CPU implementation
under 1000 lines that runs the fixed official Qwen3.5-0.8B text backbone—a
small model with the same Qwen3.8-style 3 DeltaNet : 1 attention hybrid
pattern.

The existing root-level implementation is preserved as the prototype-v0 git
tag. It remains a useful correctness reference, but it is deliberately not
the first file a learner should read.

Current teaching scope: text-only, batch 1, one process, CPU reference first.
There is no training, quantization, server, continuous batching, tensor
abstraction, generic model loader, or distributed execution.

Run the completed lessons:

~~~
make lesson-test
~~~

不想直接啃两千行 C++ 时，打开本地的
[`docs/qwen38-reading-guide.html`](docs/qwen38-reading-guide.html)：它按真实
forward 顺序解释源码，并附关键行号、陷阱提示和阅读进度。

## Status

The CPU reference path is complete for the fixed Qwen3.8-27B text topology:
safetensors loading, embedding, RMSNorm, RoPE, DeltaNet state and convolution,
GQA cache, MLP, final norm, logits, sampling, prefill, and decode.

For day-to-day work this repository also builds the fixed official
[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) checkpoint.
It is small enough for this WSL machine and provides real end-to-end text
tests. It has the same 3 DeltaNet : 1 attention hybrid family pattern, but it
is **not** a claim that its dimensions, tied LM head, or weights equal the
eventual Qwen3.8-27B target.

| Path | Status |
| --- | --- |
| CPU scalar reference | Complete and checked against official Transformers on 0.8B |
| CUDA correctness path | Complete for tiny fixture and 0.8B; deliberately naive GEMV |
| Metal | Planned; not implemented or claimed working yet |
| Qwen3.8-27B real weights | Loader and scalar forward are fixed for it; this WSL machine cannot hold its BF16 checkpoint |

## Quick start

The single-file CPU numerical reference needs only a C++17 compiler:

```bash
make test
make qwen38_08b
./qwen38_08b --self-test
```

The Makefile takes token IDs. For actual text, use the CMake build. It adds a
thin adapter over the official `tokenizer.json` through the Hugging Face
tokenizers C ABI; model inference itself still has no Python or Transformers
runtime dependency.

```bash
cmake -S . -B build
cmake --build build --target qwen38_08b -j

model=models/Qwen3.5-0.8B
./build/qwen38_08b --inspect "$model"
./build/qwen38_08b --tokenize "$model" '你好，Qwen38!'
./build/qwen38_08b --forward-text "$model" 'Hello'
./build/qwen38_08b --generate-text "$model" 'Hello' 16 --temperature 0 --seed 1
./build/qwen38_08b --generate-chat "$model" '请用一句话解释 DeltaNet。' 64 --temperature 0
```

`--temperature 0` is greedy. Positive temperature enables deterministic
temperature/top-k/top-p sampling when a `--seed` is supplied.

`--generate-chat` is a deliberately small convenience: one user turn, no
tools/vision, and the official Qwen3.5 generation prefix with thinking
disabled. It is regression-tested against the shipped Jinja template; it is
not a general chat-template interpreter.

Download the official checkpoint into `models/Qwen3.5-0.8B` (ignored by Git).
The required files are `config.json`, `tokenizer.json`, and
`model.safetensors-00001-of-00001.safetensors`. The loader validates all 320
text tensor names, shapes, and dtypes before evaluating a token; it ignores the
153 vision tensors and 15 MTP tensors in this text-only checkpoint.

To regression-test the tokenizer against the local official tokenizer:

```bash
make tokenizer-test MODEL=models/Qwen3.5-0.8B
```

## Correctness workflow

`--trace` writes the final token's FP32 checkpoints: embedding; every layer's
input norm, mixer, residual, post norm, MLP, and final residual; final norm;
and logits. `reference/` pins a Transformers revision containing the official
Qwen3.5 implementation and compares the same points.

```bash
python3 -m pip install --target reference/.deps -r reference/requirements.txt
scripts/test_08b_oracle.sh models/Qwen3.5-0.8B
```

The default three-token oracle test compares 147 tensors at an absolute
tolerance of `1e-3`. On the checked WSL machine its largest all-tensor error
was `3.156662e-4` at final norm; logits were `1.223087e-4`, and the greedy
next token matched the official reference.
The Python pieces are development-only and are excluded from the runtime
binary.

## CUDA correctness backend

CUDA is intentionally simple at this stage: one readable BF16 GEMV kernel per
row and explicit state kernels, not cuBLAS or a performance backend. It keeps
the whole fixed model forward on the GPU and compares the final logits with the
scalar CPU oracle.

```bash
make qwen38_tiny_cuda
tiny=$(mktemp -d)
./qwen38_tiny_cuda --make-fake "$tiny"
./qwen38_tiny_cuda --cuda-compare "$tiny" 1,2,3,4

make qwen38_08b_cuda
./qwen38_08b_cuda --cuda-compare models/Qwen3.5-0.8B 248044,198,198
```

For the real 0.8B correctness build, CPU and CUDA use FP32 KV cache values so
both can be checked directly against the FP32 Transformers oracle. The tiny
fixture intentionally retains BF16 cache storage to test that representation.
On an RTX 4080 SUPER, the tested 0.8B CUDA logits differed from scalar CPU by
at most `3.9053e-4` for a one-token prompt and `1.0443e-4` for a three-token
prompt; greedy IDs matched. The accepted CUDA bound is `1e-3`, accounting for
the different floating-point reduction order.

## What is deliberately not here

- Any arbitrary Hugging Face architecture or GGUF compatibility
- Quantization, TP/PP, multiple GPUs, continuous batching, scheduling, or serving
- Vision and MTP execution
- Training, LoRA, speculative decode, and multimodal inputs
- A `Tensor`, `Operator`, `Backend`, virtual dispatch, or computation graph framework

Those omissions are scope control, not missing plumbing. Libraries are used
for peripheral work (tokenization); the Qwen forward is written here.

## Repository map

```text
qwen38.cpp                 # CPU reference, safetensors loader, sampler, trace writer
kernels/qwen38_tiny_cuda.cu # compile-time fixed CUDA correctness implementation
kernels/qwen38_08b_cuda.cu  # 0.8B compile-time configuration wrapper
qwen38_tokenizer.*          # small tokenizer.json adapter, outside the core forward
reference/                  # pinned Transformers oracle; development-only
scripts/test_08b_oracle.sh  # repeatable real-weight CPU oracle test
```

## Machine limits and current validation

This workspace has 15 GiB WSL RAM and an RTX 4080 SUPER with 16 GiB VRAM. The
0.8B checkpoint has 1.40 GiB of text tensors and runs comfortably as an oracle.
The Qwen3.8-27B BF16 text weights are roughly 50 GiB, so real 27B inference is
not a viable test on this machine. It remains the fixed final configuration;
0.8B is the deliberately bounded development fixture.

Current local checks include scalar unit tests, strict safetensors inspection,
real tokenizer round-trip (`你好，Qwen38!`), real greedy text generation,
one- and three-token official traces, and CPU/CUDA logits comparison.

## Before publishing

CI, contribution scope, and reproducible validation instructions are included.
The repository still needs the project owner's explicit code-license choice
before a public release; until then, no license is implied.

[`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) separates the checks that can
run today from the larger-hardware validation required before claiming a fully
verified Qwen3.8-27B / Metal release.
