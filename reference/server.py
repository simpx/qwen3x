#!/usr/bin/env python3
"""Minimal OpenAI-compatible server backed by official Transformers Qwen3.5.

This is an evaluation oracle, not a production runtime.  It intentionally
implements only the non-streaming chat-completions surface used by EvalScope.
"""

from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache


STOP_TOKENS = {248044, 248046}
THINK_END_TOKEN = 248069


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path, required=True)
    parser.add_argument("--served-model-name", default="qwen3.5-0.8b-reference")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32",
                        help="float32 is the correctness oracle; bfloat16 is faster but may diverge")
    parser.add_argument("--cache", choices=("dynamic", "static"), default="static",
                        help="static avoids one torch.cat allocation per generated token")
    parser.add_argument("--max-context-tokens", type=int, default=40960)
    return parser.parse_args()


def select_token(logits: torch.Tensor, generated_set: set[int], body: dict,
                 generator: torch.Generator) -> int:
    temperature = float(body.get("temperature", 1.0))
    top_p = float(body.get("top_p", 1.0))
    top_k = int(body.get("top_k", 0))
    presence_penalty = float(body.get("presence_penalty", 0.0))
    if not 0 <= temperature <= 2 or not 0 < top_p <= 1 or top_k < 0:
        raise HTTPException(400, "invalid sampling parameters")
    if not -2 <= presence_penalty <= 2:
        raise HTTPException(400, "presence_penalty must be between -2 and 2")

    scores = logits.float().clone()
    if presence_penalty and generated_set:
        # Presence is binary: once a token has appeared, subtract exactly once.
        # Keep this set incrementally instead of rebuilding set(generated) after
        # every token, which becomes quadratic for a 32K response.
        token_ids = torch.tensor(tuple(generated_set), device=scores.device)
        scores[token_ids] -= presence_penalty
    if temperature == 0:
        return int(scores.argmax())
    scores /= temperature

    if top_k > 0 and top_k < scores.numel():
        threshold = torch.topk(scores, top_k).values[-1]
        scores.masked_fill_(scores < threshold, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    if top_p < 1:
        sorted_probabilities, sorted_ids = torch.sort(probabilities, descending=True)
        remove = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities >= top_p
        sorted_probabilities.masked_fill_(remove, 0)
        probabilities.zero_().scatter_(0, sorted_ids, sorted_probabilities)
        probabilities /= probabilities.sum()
    return int(torch.multinomial(probabilities, 1, generator=generator))


def create_app(args: argparse.Namespace) -> FastAPI:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA reference requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.chat_template = args.chat_template.read_text(encoding="utf-8")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation="eager",
    ).eval().to(device)

    app = FastAPI(title="Qwen3.5 Transformers reference")

    @app.get("/readyz")
    def ready():
        return {"status": "ready", "backend": "transformers", "device": str(device),
                "dtype": args.dtype, "cache": args.cache,
                "max_context_tokens": args.max_context_tokens}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": args.served_model_name,
                "object": "model", "created": 0, "owned_by": "transformers"}]}

    @app.post("/v1/chat/completions")
    def chat(body: dict):
        if body.get("stream", False):
            raise HTTPException(400, "reference server does not implement streaming")
        if body.get("model") != args.served_model_name:
            raise HTTPException(404, "model not found")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, "messages must be a non-empty array")
        template_options = body.get("chat_template_kwargs") or {}
        enable_thinking = bool(template_options.get("enable_thinking", False))
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            enable_thinking=enable_thinking,
            preserve_thinking=bool(template_options.get("preserve_thinking", True)),
        )
        max_tokens = int(body.get("max_completion_tokens", body.get("max_tokens", 128)))
        if max_tokens <= 0:
            raise HTTPException(400, "max_tokens must be positive")
        if len(prompt_ids) + max_tokens > args.max_context_tokens:
            raise HTTPException(400, "prompt + max_tokens exceeds max context")
        seed = int(body.get("seed", 0))
        generator = torch.Generator(device=device).manual_seed(seed & ((1 << 63) - 1))

        with torch.inference_mode():
            cache = None
            if args.cache == "static":
                cache = StaticCache(
                    config=model.config,
                    max_cache_len=len(prompt_ids) + max_tokens,
                )
            output = model(
                input_ids=torch.tensor([prompt_ids], device=device, dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            logits = output.logits[0, -1]
            generated: list[int] = []
            generated_set: set[int] = set()
            finish_reason = "length"
            for _ in range(max_tokens):
                token = select_token(logits, generated_set, body, generator)
                if token in STOP_TOKENS:
                    finish_reason = "stop"
                    break
                generated.append(token)
                generated_set.add(token)
                output = model(
                    input_ids=torch.tensor([[token]], device=device, dtype=torch.long),
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = output.past_key_values
                logits = output.logits[0, -1]

        reasoning = None
        content_ids = generated
        if enable_thinking:
            try:
                boundary = generated.index(THINK_END_TOKEN)
            except ValueError:
                boundary = len(generated)
            reasoning = tokenizer.decode(generated[:boundary], skip_special_tokens=True).strip()
            content_ids = generated[boundary + 1:] if boundary < len(generated) else []
        content = tokenizer.decode(content_ids, skip_special_tokens=True).lstrip()
        message = {"role": "assistant", "content": content, "refusal": None}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        return {
            "id": "chatcmpl-ref-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": args.served_model_name,
            "choices": [{"index": 0, "message": message, "logprobs": None,
                         "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(generated),
                "total_tokens": len(prompt_ids) + len(generated),
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }

    return app


def main() -> None:
    args = parse_args()
    uvicorn.run(create_app(args), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
