from src.evaluation.score import f05


def test_perfect_score():
    assert f05(1.0, 1.0) == 1.0


def test_zero_recall():
    assert f05(1.0, 0.0) == 0.0


def test_zero_precision():
    assert f05(0.0, 1.0) == 0.0
