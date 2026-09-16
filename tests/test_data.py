"""Tests for stratified splitting. Uses synthetic data only (no CSV dependency)."""
import numpy as np
import pandas as pd

from src.data import split_data


def _make_imbalanced_df(n_rows: int = 2000, fraud_rate: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_fraud = int(n_rows * fraud_rate)
    y = np.array([1] * n_fraud + [0] * (n_rows - n_fraud))
    X = rng.normal(size=(n_rows, 5))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["Class"] = y
    return df


def test_split_sizes_are_70_15_15():
    df = _make_imbalanced_df(2000)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    total = len(df)
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert abs(len(X_train) / total - 0.70) < 0.02
    assert abs(len(X_val) / total - 0.15) < 0.02
    assert abs(len(X_test) / total - 0.15) < 0.02


def test_split_preserves_class_ratio():
    df = _make_imbalanced_df(2000, fraud_rate=0.02)
    _, _, _, y_train, y_val, y_test = split_data(df)

    overall_rate = df["Class"].mean()
    for y in (y_train, y_val, y_test):
        assert abs(y.mean() - overall_rate) < 0.01


def test_split_no_overlap_between_sets():
    df = _make_imbalanced_df(500)
    X_train, X_val, X_test, *_ = split_data(df)

    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
