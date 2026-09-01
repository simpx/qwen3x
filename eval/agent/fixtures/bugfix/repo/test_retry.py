import unittest

from retry import retry_delays


class RetryDelaysTest(unittest.TestCase):
    def test_zero_attempts(self):
        self.assertEqual(retry_delays(0, 0.5), [])

    def test_first_delay_starts_at_base(self):
        self.assertEqual(retry_delays(4, 0.5), [0.5, 1.0, 2.0, 4.0])

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            retry_delays(-1, 1.0)
        with self.assertRaises(ValueError):
            retry_delays(1, 0.0)


if __name__ == "__main__":
    unittest.main()
