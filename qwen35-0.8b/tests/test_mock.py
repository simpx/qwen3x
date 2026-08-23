#!/usr/bin/env python3
"""Fast regression for the native mock-compute Engine path."""

import argparse
from pathlib import Path

from qwen35 import Engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    args = parser.parse_args()

    with Engine(args.library, args.bin, mock=True) as engine:
        session = engine.create_session(64)
        prompt = [10, 42, 99, 7]

        assert session.sync(prompt, checkpoint_at=3) == 0
        assert session.position == len(prompt)
        first = session.argmax()
        # The last prompt token is not a digit token id. position=4 therefore
        # selects digit 4 (Qwen token id 19), never the stop row.
        assert first == 19
        sampled, _rng = session.sample(1.0, 0.9, 123)
        assert sampled == first

        logits = session.copy_logits()
        assert logits[first] == max(logits)

        # Decode uses the same mock forward as prefill and eventually selects
        # the real model stop token stored in the fixed logits bank.
        stopped = False
        for _ in range(11):
            token = session.argmax()
            if engine.token_is_stop(token):
                stopped = True
                break
            session.eval(token)
        assert stopped

        # The additional checkpoint remains a real restorable State point.
        assert session.sync(prompt[:3]) == 3
        assert session.position == 3

        # Re-evaluating the same suffix selects the same token-indexed logits,
        # independent of whether the path came from cache or reset.
        session.eval(prompt[3])
        assert session.position == len(prompt)
        assert session.argmax() == first

        # Thinking mode uses token-to-token rules too, without a separate
        # prefill/decode branch in the mock Engine.
        thinking = engine.create_session(64)
        thinking.sync([248068, 198])  # <think>\n
        assert thinking.argmax() == 26003  # "think"
        thinking.eval(26003)
        assert thinking.argmax() == 198  # \n
        thinking.eval(198)
        assert thinking.argmax() == 248069  # </think>
        thinking.eval(248069)
        assert thinking.argmax() == 271  # DOUBLE_NEWLINE_TOKEN: \n\n
        thinking.eval(271)
        assert 15 <= thinking.argmax() <= 24

    print("native mock Engine regression: passed")


if __name__ == "__main__":
    main()
