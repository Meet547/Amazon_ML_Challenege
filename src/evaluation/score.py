"""
Competition evaluation utilities.

Do not modify the competition metric definition.
"""

def f05(precision: float, recall: float) -> float:
    """
    F_0.5 score.

    F_0.5 gives greater weight to precision than recall.
    """
    denominator = 0.25 * precision + recall

    if denominator == 0:
        return 0.0

    return (1.25 * precision * recall) / denominator


def precision(tp: int, fp: int) -> float:
    denominator = tp + fp

    if denominator == 0:
        return 0.0

    return tp / denominator


def recall(tp: int, fn: int) -> float:
    denominator = tp + fn

    if denominator == 0:
        return 0.0

    return tp / denominator


def calculate_f05(tp: int, fp: int, fn: int) -> float:
    p = precision(tp, fp)
    r = recall(tp, fn)

    return f05(p, r)
