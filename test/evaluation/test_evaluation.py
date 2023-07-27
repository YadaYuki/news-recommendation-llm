from src.evaluation.RecEvaluator import RecEvaluator
import numpy as np
import pytest


def test_dcg_score() -> None:
    y_score = np.asarray([0.3, 0.5, 0.0, 0.0, 0.2])
    y_true = np.asarray([0, 1, 0, 0, 1])
    assert RecEvaluator.dcg_score(y_true, y_score, 5) == 1.5
    assert RecEvaluator.dcg_score(y_true, y_score, 10) == 1.5


def test_ndcg_score() -> None:
    y_score = np.asarray([0.3, 0.5, 0.0, 0.0, 0.2])
    y_true = np.asarray([0, 1, 0, 0, 1])
    assert RecEvaluator.ndcg_score(y_true, y_score, 5) == pytest.approx(0.9197207, 0.001)
    assert RecEvaluator.ndcg_score(y_true, y_score, 10) == pytest.approx(0.9197207, 0.001)


def test_mrr_score() -> None:
    y_score = np.asarray([0.3, 0.5, 0.0, 0.0, 0.2])
    y_true = np.asarray([0, 1, 0, 0, 1])
    assert RecEvaluator.mrr_score(y_true, y_score) == pytest.approx(0.666666, 0.001)
