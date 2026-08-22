#!/usr/bin/env python3
"""Real-weight Engine/Session ABI regression."""

import argparse
import json
from pathlib import Path

from qwen_runtime.binding import Engine, EngineError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--reference", type=Path,
                        default=Path(__file__).with_name("reference.json"))
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text())
    case = next(item for item in reference["cases"] if item["name"] == "lesson_short")
    with Engine(args.library, args.weights) as engine:
        first = engine.create_session(64)
        second = engine.create_session(64)

        first.sync(case["prefill_ids"])
        assert first.position == len(case["prefill_ids"])
        assert first.argmax() == case["next_token"]
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

        tiny = engine.create_session(3)
        tiny.sync(case["prefill_ids"])
        try:
            tiny.eval(198)
            raise AssertionError("full context accepted another token")
        except EngineError as error:
            assert "context is full" in str(error)
        assert tiny.position == 3

    print("native Engine/Session regression: passed")


if __name__ == "__main__":
    main()
