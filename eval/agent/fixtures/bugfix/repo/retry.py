"""Retry scheduling helper."""


def retry_delays(attempts: int, base_seconds: float) -> list[float]:
    if attempts < 0:
        raise ValueError("attempts must not be negative")
    if base_seconds <= 0:
        raise ValueError("base_seconds must be positive")
    return [base_seconds * (2 ** attempt) for attempt in range(1, attempts + 1)]
