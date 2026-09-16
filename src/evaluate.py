"""Metrics, threshold optimization, and diagnostic plots for imbalanced classification.

Accuracy is deliberately never computed here: with ~0.17% positives it is
meaningless (a constant "not fraud" predictor scores ~99.8%).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    """Computes PR-AUC (threshold-independent) plus precision/recall/F1 at `threshold`."""
    preds = (y_proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, preds),
    }


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, dict]:
    """Finds the threshold maximizing F1 on the given (validation) set."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = int(np.argmax(f1_scores[:-1]))
    best_threshold = float(thresholds[best_idx])
    return best_threshold, compute_metrics(y_true, y_proba, best_threshold)


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, title: str = "Confusion Matrix") -> Path:
    """Saves a confusion matrix heatmap to `out_path` and returns it.

    Uses a log-scaled color norm: with ~99.8% negatives, a linear scale makes
    the true-negative cell dominate and the other three cells look uniformly
    blank regardless of their actual counts.
    """
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", norm=LogNorm(vmin=max(cm.min(), 1), vmax=cm.max()))
    ax.set_xticks([0, 1], labels=["Normal", "Fraud"])
    ax.set_yticks([0, 1], labels=["Normal", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    threshold = cm.max() ** 0.5  # midpoint on the log scale
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=12)
    fig.colorbar(im, ax=ax, label="Count (log scale)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_pr_curve(
    y_true: np.ndarray, y_proba: np.ndarray, out_path: Path, title: str = "Precision-Recall Curve"
) -> Path:
    """Saves a precision-recall curve plot to `out_path` and returns it."""
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recalls, precisions, label=f"AP={pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
