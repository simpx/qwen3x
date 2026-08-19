"""Official Qwen3.5-0.8B, one-token-at-a-time oracle helpers.

This module deliberately exposes the same state boundary that the future C++ engine uses:
feed one token, retain `past_key_values`, receive logits for the *next* token.  It does not
train, sample probabilistically, or hide the loop behind `generate()`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


VOCAB_SIZE = 248320
STOP_TOKENS = {248044, 248046}


@dataclass
class LoadedReference:
    """Official model plus its selected execution device and tokenizer."""

    model: Any
    tokenizer: Any
    device: torch.device


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device("cuda:0" if requested == "cuda" else "cpu")


def load_reference(model_dir: Path, requested_device: str = "auto") -> LoadedReference:
    """Load exactly the local official checkpoint in FP32/eager mode.

    FP32 is intentional: the CPU lesson applies BF16 *weights* to FP32 activations, and the
    official FP32 execution is its numerical gold standard.  CUDA Stage 3 will document its
    wider BF16-input tolerance separately instead of silently changing this oracle.
    """

    if not model_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {model_dir}")
    config = json.loads((model_dir / "config.json").read_text())
    text_config = config.get("text_config", {})
    if text_config.get("model_type") != "qwen3_5_text":
        raise ValueError("this reference only accepts the official Qwen3.5 text backbone")

    device = choose_device(requested_device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.float32, attn_implementation="eager"
        )
    except (KeyError, ValueError) as error:
        # A common failure is an older Transformers lacking Qwen3.5 support.  Do not fall back
        # to trust_remote_code: vectors must be reproducible from a declared public dependency.
        raise RuntimeError(
            "could not load qwen3_5 with AutoModelForCausalLM; install requirements-dev.txt "
            "(Transformers >= 5.0.0)"
        ) from error
    model.eval().to(device)
    return LoadedReference(model=model, tokenizer=tokenizer, device=device)


def token_tensor(token: int, device: torch.device) -> torch.Tensor:
    if not 0 <= token < VOCAB_SIZE:
        raise ValueError(f"token id {token} is outside Qwen3.5 vocabulary")
    return torch.tensor([[token]], device=device, dtype=torch.long)


def run_tokens(reference: LoadedReference, tokens: Iterable[int]) -> np.ndarray:
    """Feed tokens one by one and return one FP32 `[vocab]` logit row after each input.

    The returned row i predicts the token following input i.  This is intentionally the same
    order as a C++ prefill loop and makes a later decode error observable at its first token.
    """

    cache = None
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for token in tokens:
            output = reference.model(
                input_ids=token_tensor(int(token), reference.device),
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            logits = output.logits[0, -1].float().cpu().numpy().copy()
            if logits.shape != (VOCAB_SIZE,):
                raise RuntimeError(f"official model returned logits shape {logits.shape}, expected {(VOCAB_SIZE,)}")
            rows.append(logits)
    if not rows:
        raise ValueError("a reference case must contain at least one input token")
    return np.stack(rows)


def greedy_from_prefill(reference: LoadedReference, prompt: Iterable[int], max_new_tokens: int) -> list[int]:
    """Run official greedy decode while keeping the same official cache alive."""

    prompt = [int(token) for token in prompt]
    if not prompt:
        raise ValueError("prompt is empty")
    cache = None
    logits = None
    with torch.inference_mode():
        for token in prompt:
            output = reference.model(
                input_ids=token_tensor(token, reference.device), past_key_values=cache, use_cache=True
            )
            cache, logits = output.past_key_values, output.logits[0, -1]
        result: list[int] = []
        for _ in range(max_new_tokens):
            assert logits is not None
            next_token = int(logits.argmax())
            if next_token in STOP_TOKENS:
                break
            result.append(next_token)
            output = reference.model(
                input_ids=token_tensor(next_token, reference.device),
                past_key_values=cache,
                use_cache=True,
            )
            cache, logits = output.past_key_values, output.logits[0, -1]
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(model_dir: Path, tokenizer: Any) -> dict[str, str | int | None]:
    """Describe exactly which official artifacts produced a vector bundle."""

    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    tokenizer_path = model_dir / "tokenizer.json"
    return {
        "config_sha256": file_sha256(config_path),
        "safetensors_index_sha256": file_sha256(index_path),
        "tokenizer_sha256": file_sha256(tokenizer_path) if tokenizer_path.exists() else None,
        "tokenizer_vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "torch_version": torch.__version__,
    }
