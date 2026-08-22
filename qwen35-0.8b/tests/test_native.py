#!/usr/bin/env python3
"""Real-weight Engine/Session ABI regression."""

import argparse
import json
from pathlib import Path

from qwen35 import Engine, EngineError, Q35_LOG_INFO, SessionBusy, set_log_callback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
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
        Q35_LOG_INFO,
    )
    with Engine(args.library, args.weights) as engine:
        first = engine.create_session(64)
        second = engine.create_session(64)

        first.sync(case["prefill_ids"])
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

        # Append-only sync evaluates just the suffix.
        extended = case["prefill_ids"] + case["decode_ids"]
        first.sync(extended)
        assert first.position == len(extended)

        # A shorter timeline cannot rewind DeltaNet, so sync rebuilds it exactly.
        first.sync(case["prefill_ids"])
        assert first.position == len(case["prefill_ids"])
        assert first.argmax() == case["next_token"]

        # Independent Session State must remain untouched.
        assert second.position == 0
        second.sync(case["prefill_ids"])
        assert second.argmax() == case["next_token"]

        # Bad requests cross the ABI as Python errors; they must not terminate the host.
        try:
            second.eval(-1)
            raise AssertionError("invalid token was accepted")
        except EngineError as error:
            assert "outside vocabulary" in str(error)
        assert second.position == len(case["prefill_ids"])

        # SessionManager owns fixed Sessions and routes named requests back to
        # the same live state. Acquiring the same ID concurrently is rejected.
        manager = engine.create_session_manager(2, 64)
        managed = manager.acquire("agent-a", case["prefill_ids"])
        managed.sync(case["prefill_ids"])
        manager.release(managed, keep=True)

        managed = manager.acquire("agent-a", case["prefill_ids"])
        assert managed.position == len(case["prefill_ids"])
        try:
            manager.acquire("agent-a", case["prefill_ids"])
            raise AssertionError("busy named session was acquired twice")
        except SessionBusy:
            pass
        manager.release(managed, keep=True)

        # Anonymous lookup can reuse the longest complete token prefix. Once
        # reused anonymously, the old name is intentionally no longer bound.
        anonymous = manager.acquire(None, case["prefill_ids"] + case["decode_ids"])
        assert anonymous.position == len(case["prefill_ids"])
        anonymous.sync(case["prefill_ids"] + case["decode_ids"])
        manager.release(anonymous, keep=True)
        assert not manager.forget("agent-a")
        manager.close()

        tiny = engine.create_session(3)
        tiny.sync(case["prefill_ids"])
        try:
            tiny.eval(198)
            raise AssertionError("full context accepted another token")
        except EngineError as error:
            assert "context is full" in str(error)
        assert tiny.position == 3

    set_log_callback(args.library, None)

    messages = [message for _level, _file, _line, message in events]
    assert any("model load completed" in message for message in messages)
    assert any("session sync completed" in message for message in messages)
    assert any("session acquired" in message for message in messages)
    assert any("session released" in message for message in messages)
    assert any("engine closing" in message for message in messages)
    assert any(file == "engine.cpp" and line > 0 for _, file, line, _ in events)
    assert any(file == "runtime.cpp" and line > 0 for _, file, line, _ in events)

    print("native Engine/Session regression: passed")


if __name__ == "__main__":
    main()
