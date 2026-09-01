"""Small arithmetic helpers."""


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        raise ValueError("denominator must not be zero")
    return numerator / denominator


def average(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def clamp(value: float, lower: float, upper: float) -> float:
    if lower > upper:
        raise ValueError("lower must not exceed upper")
    return max(lower, min(value, upper))
