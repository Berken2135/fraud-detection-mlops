"""Tests for metric computation and threshold optimization."""
import numpy as np

from src.evaluate import compute_metrics, find_best_threshold


def test_compute_metrics_matches_hand_calculation():
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    # threshold=0.5 -> preds = [0, 0, 1, 1, 1] -> TP=2, FP=1, FN=0
    metrics = compute_metrics(y_true, y_proba, threshold=0.5)

    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0
    assert round(metrics["f1"], 4) == round(2 * (2 / 3 * 1.0) / (2 / 3 + 1.0), 4)
    assert metrics["confusion_matrix"].sum() == len(y_true)


def test_compute_metrics_never_includes_accuracy():
    y_true = np.array([0, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.9])
    metrics = compute_metrics(y_true, y_proba, threshold=0.5)
    assert "accuracy" not in metrics


def test_find_best_threshold_maximizes_f1():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 90 + [1] * 10)
    # Well-separated scores so there is a clear best threshold.
    y_proba = np.concatenate([rng.uniform(0, 0.4, 90), rng.uniform(0.6, 1.0, 10)])

    best_threshold, best_metrics = find_best_threshold(y_true, y_proba)

    # Brute-force search over candidate thresholds should not find a strictly better F1.
    for t in np.linspace(0.01, 0.99, 99):
        preds = (y_proba >= t).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        assert f1 <= best_metrics["f1"] + 1e-9

    assert 0.0 < best_threshold < 1.0
