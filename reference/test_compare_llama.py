import unittest

from reference.compare_llama import compare_logits, require_matching_tokens, top_ids


class CompareLlamaTest(unittest.TestCase):
    def test_matching_logits_pass(self):
        result = compare_logits([0.0, 1.0, 2.0], [0.01, 1.01, 2.01], "matching")
        self.assertAlmostEqual(result["max_abs_error"], 0.01)
        self.assertEqual(result["argmax"], 2)

    def test_wrong_argmax_fails(self):
        with self.assertRaisesRegex(AssertionError, "argmax"):
            compare_logits([2.0, 1.0, 0.0], [0.0, 1.0, 2.0], "wrong")

    def test_top_ids_are_ordered(self):
        self.assertEqual(top_ids([1.0, 4.0, 3.0, 2.0], 3), [1, 2, 3])

    def test_matching_tokens_pass(self):
        require_matching_tokens([1, 2, 3], [1, 2, 3], "matching")

    def test_token_value_mismatch_fails(self):
        with self.assertRaisesRegex(AssertionError, "token 1 differs"):
            require_matching_tokens([1, 9, 3], [1, 2, 3], "wrong")

    def test_token_count_mismatch_fails(self):
        with self.assertRaisesRegex(AssertionError, "token counts differ"):
            require_matching_tokens([1, 2], [1, 2, 3], "short")


if __name__ == "__main__":
    unittest.main()
