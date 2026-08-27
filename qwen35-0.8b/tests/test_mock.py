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
        sampled, _rng = session.sample(1.0, 0.9, 123, top_k=1)
        assert sampled == first
        nucleus, _rng = session.sample(1.0, 0.5, 999)
        assert nucleus == first
        repeat_a = session.sample(1.0, 1.0, 123)
        repeat_b = session.sample(1.0, 1.0, 123)
        assert repeat_a == repeat_b

        # Replaying a whole request with the same seed must reproduce every
        # sampled token and every intermediate RNG state, not only step zero.
        def sampled_sequence(seed):
            replay = engine.create_session(64)
            try:
                replay.sync(prompt)
                rng = seed
                result = []
                for _ in range(8):
                    token, rng = replay.sample(1.0, 1.0, rng)
                    result.append((token, rng))
                    if engine.token_is_stop(token):
                        break
                    replay.eval(token)
                return result
            finally:
                replay.close()

        assert sampled_sequence(123) == sampled_sequence(123)

        # Presence penalty looks only at generated tokens and subtracts once,
        # even if the same token appears repeatedly.
        penalized, _rng = session.sample(
            0.0, 1.0, 123,
            presence_penalty=2.0,
            generated_tokens=[first, first],
        )
        assert penalized == first + 1

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

        # Leave one idle manager owned only by Engine. Engine.close() must be
        # able to clear it without its __del__ re-entering engine._lock.
        manager = engine.create_session_manager(1, 64)
        managed = manager.acquire(prompt)
        managed.sync(prompt)
        manager.release(managed, keep=True)
        manager = None

    print("native mock Engine regression: passed")


if __name__ == "__main__":
    main()
