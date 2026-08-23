#!/usr/bin/env python3
"""Real-weight Engine/Session ABI regression."""

import argparse
import json
from pathlib import Path

from qwen35 import Engine, EngineError, Q35_LOG_DEBUG, SessionBusy, set_log_callback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--reference", type=Path,
                        default=Path(__file__).with_name("reference.json"))
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text())
    case = next(item for item in reference["cases"] if item["name"] == "lesson_short")
    events = []
    set_log_callback(
        args.library,
        lambda level, file, line, message:
            events.append((level, file, line, message)),
        Q35_LOG_DEBUG,
    )
    with Engine(args.library, args.bin) as engine:
        first = engine.create_session(64)
        second = engine.create_session(64)

        assert first.sync(case["prefill_ids"]) == 0
        assert first.position == len(case["prefill_ids"])
        assert first.argmax() == case["next_token"]
        sampled, _rng = first.sample(0.0, 1.0, 123)
        assert sampled == case["next_token"]
        sampled_a, rng_a = first.sample(1.0, 0.9, 123)
        sampled_b, rng_b = first.sample(1.0, 0.9, 123)
        assert 0 <= sampled_a < engine.vocab_size
        assert (sampled_a, rng_a) == (sampled_b, rng_b)
        logits = first.copy_logits()
        assert abs(logits[case["next_token"]] - case["next_logit"]) <= reference["max_abs_error"]

        # Decode advances the live point, while sync's prompt boundary remains
        # available as this Session's second checkpoint.
        extended = case["prefill_ids"] + case["decode_ids"]
        for token in case["decode_ids"]:
            first.eval(token)
        assert first.position == len(extended)

        # A shorter request can restore the saved checkpoint without
        # rebuilding DeltaNet from token zero.
        assert first.sync(case["prefill_ids"]) == len(case["prefill_ids"])
        assert first.position == len(case["prefill_ids"])
        assert first.argmax() == case["next_token"]
        restored_logits = first.copy_logits()
        assert abs(restored_logits[case["next_token"]] -
                   logits[case["next_token"]]) <= reference["max_abs_error"]

        # checkpoint_at may be earlier than the end of prefill. Engine must
        # capture the State while forward crosses that exact token boundary.
        early = engine.create_session(64)
        checkpoint_at = len(case["prefill_ids"]) - 1
        assert early.sync(case["prefill_ids"], checkpoint_at) == 0
        early.eval(case["decode_ids"][0])
        assert early.sync(case["prefill_ids"][:checkpoint_at]) == checkpoint_at
        assert early.position == checkpoint_at

        # Append-only sync still evaluates just the suffix.
        assert first.sync(extended) == len(case["prefill_ids"])
        assert first.position == len(extended)

        # Independent Session State must remain untouched.
        assert second.position == 0
        assert second.sync(case["prefill_ids"]) == 0
        assert second.argmax() == case["next_token"]

        # Bad requests cross the ABI as Python errors; they must not terminate the host.
        try:
            second.eval(-1)
            raise AssertionError("invalid token was accepted")
        except EngineError as error:
            assert "outside vocabulary" in str(error)
        assert second.position == len(case["prefill_ids"])

        # SessionManager owns fixed Sessions and selects by reusable prefix.
        manager = engine.create_session_manager(2, 64)
        managed = manager.acquire(case["prefill_ids"])
        assert managed.sync(case["prefill_ids"]) == 0
        for token in case["decode_ids"]:
            managed.eval(token)
        manager.release(managed, keep=True)

        # The saved checkpoint makes this Session the longest prefix match even
        # though its live timeline is longer than the request.
        managed = manager.acquire(case["prefill_ids"])
        assert managed.position == len(extended)
        other = manager.acquire(case["prefill_ids"])
        try:
            manager.acquire(case["prefill_ids"])
            raise AssertionError("a third Session was acquired from a two-slot pool")
        except SessionBusy:
            pass
        manager.release(other, keep=False)
        assert managed.sync(case["prefill_ids"]) == len(case["prefill_ids"])
        for token in case["decode_ids"]:
            managed.eval(token)
        manager.release(managed, keep=True)

        # Prefix lookup again finds that same checkpoint without an external ID.
        reused = manager.acquire(case["prefill_ids"])
        assert reused.position == len(extended)
        assert reused.sync(case["prefill_ids"]) == len(case["prefill_ids"])
        manager.release(reused, keep=True)
        manager.close()

        tiny = engine.create_session(3)
        assert tiny.sync(case["prefill_ids"]) == 0
        try:
            tiny.eval(198)
            raise AssertionError("full context accepted another token")
        except EngineError as error:
            assert "context is full" in str(error)
        assert tiny.position == 3

    set_log_callback(args.library, None)

    messages = [message for _level, _file, _line, message in events]
    assert any("model load completed" in message for message in messages)
    assert any("prefill started" in message for message in messages)
    assert any("prefill completed" in message for message in messages)
    assert any("session acquire" in message and "prompt_tokens=" in message
               and "live_state_tokens=" in message
               and "checkpoint_state_tokens=" in message
               and "cache_result=" in message
               and "cache_hit_tokens=" in message
               and "to_prefill_tokens=" in message for message in messages)
    assert any("session release" in message and "live_state_tokens=" in message
               and "checkpoint_state_tokens=" in message for message in messages)
    assert any("session sync" in message and "cache_result=hit_live" in message
               for message in messages)
    assert any("session sync" in message and "cache_result=new" in message
               for message in messages)
    assert any("cache_result=hit_checkpoint" in message for message in messages)
    assert any("session sync" in message and "prompt_tokens=" in message
               and "live_state_tokens=" in message
               and "checkpoint_state_tokens=" in message
               and "checkpoint_at=" in message
               and "cache_hit_tokens=" in message
               and "to_prefill_tokens=" in message for message in messages)
    assert any("session checkpoint restored" in message
               and "live_state_tokens=" in message
               and "checkpoint_state_tokens=" in message for message in messages)
    assert any("session checkpoint saved" in message
               and "checkpoint_state_tokens=" in message for message in messages)
    assert any("engine closing" in message for message in messages)
    assert any("session created" in message for message in messages)
    assert any(file == "engine.cpp" and line > 0 for _, file, line, _ in events)
    assert any(file == "runtime.cpp" and line > 0 for _, file, line, _ in events)

    print("native Engine/Session regression: passed")


if __name__ == "__main__":
    main()
