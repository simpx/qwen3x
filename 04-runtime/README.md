# qwen3x Python runtime

这是面向个人使用的单模型、单并发 OpenAI-compatible runtime。Python 负责 HTTP、鉴权、
chat template、tokenizer 和 SSE；`02` CPU 或 `03` CUDA engine 负责真正的 token forward。

```text
OpenAI client
  -> Python /v1/chat/completions
  -> tokenizer: messages -> prompt_ids
  -> 常驻 C++ --worker: start(prompt_ids), next, next, ...
  -> tokenizer: generated_ids -> text/SSE
```

## 为什么 engine 要有 `--worker`

普通 `--generate` 每次启动都会重新 mmap/upload 权重。runtime 启动 engine 后等待 `ready`，以后
一个 token 只发一条 `start` 或 `next` 命令；模型、Work 和 GPU allocation 始终留在子进程里。
每个 HTTP 请求结束后发送 `reset`，确保 KV/DeltaNet/conv state 不会串到下一个请求。

## 安装与启动

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r 04-runtime/requirements.txt

make -C 02-cpu-0.8b
make -C 02-cpu-0.8b weights MODEL=../models/Qwen3.5-0.8B

export QWEN_API_KEY='换成随机长字符串'
make -C 04-runtime run
```

默认只监听 `127.0.0.1:8000`。不要用 `--no-auth` 暴露到网络；远程使用应放在 Caddy、
Tailscale 或其他 TLS 入口之后。

CUDA engine 使用同一协议，只需替换两个路径：

```sh
python3 04-runtime/runtime.py \
  --engine 03-cuda-0.8b/qwen35_cuda \
  --weights 03-cuda-0.8b/build/qwen35-0.8b.bin
```

WSL 如果同时存在 distro stub driver，需要与 Stage 3 一样显式选择宿主机 driver：

```sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH \
python3 04-runtime/runtime.py \
  --engine 03-cuda-0.8b/qwen35_cuda \
  --weights 03-cuda-0.8b/build/qwen35-0.8b.bin
```

## OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="换成随机长字符串")
for part in client.chat.completions.create(
    model="qwen3.5-0.8b",
    messages=[{"role": "user", "content": "你好"}],
    temperature=0,
    stream=True,
):
    print(part.choices[0].delta.content or "", end="", flush=True)
```

当前明确支持：text-only messages、`n=1`、greedy `temperature=0`、`max_tokens` /
`max_completion_tokens`、最多 4 个 stop strings、streaming 和 `stream_options.include_usage`。
tools、vision、logprobs、JSON schema、非零 temperature/top-p sampling 会返回明确的 400，绝不静默忽略。
请求体、context、单次 engine step 和整个 generation 都有硬上限；对应时间可通过
`--engine-step-timeout`、`--request-timeout` 调整。

## 验证

```sh
make -C 04-runtime test
make -C 04-runtime e2e
```

`test` 不加载模型；`e2e` 会真实加载 tokenizer 和常驻 CPU engine。生产部署示例见
`qwen3x-runtime.service.example`。systemd 只负责进程拉起；TLS、访问控制和公网限流仍应由
反向代理或私有网络承担。

当前生产边界是“个人、单模型、单并发”：忙时返回 429。decode 阶段能在 token 边界发现客户端
断开；一次很长的 C++ prefill 仍是同步命令，不能在中间安全取消。需要多人并发、可抢占 prefill
或高可用多副本时，应继续增加 scheduler/worker 隔离，而不是把这一版伪装成多租户服务。
