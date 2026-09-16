"""Tests that preprocessing is fit on train data only (no leakage)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features import build_pipeline


def test_scaler_is_fit_only_on_training_data():
    rng = np.random.default_rng(1)
    X_train = pd.DataFrame(rng.normal(loc=0.0, scale=1.0, size=(200, 3)), columns=["a", "b", "c"])
    y_train = pd.Series(rng.integers(0, 2, size=200))
    # Deliberately different distribution, to prove it does not influence the fitted scaler.
    X_test = pd.DataFrame(rng.normal(loc=50.0, scale=5.0, size=(50, 3)), columns=["a", "b", "c"])

    pipeline = build_pipeline(LogisticRegression())
    pipeline.fit(X_train, y_train)

    fitted_scaler = pipeline.named_steps["scaler"]
    np.testing.assert_allclose(fitted_scaler.mean_, X_train.mean().values, rtol=1e-6)

    # Transforming test data must not refit the scaler.
    pipeline.predict_proba(X_test)
    np.testing.assert_allclose(fitted_scaler.mean_, X_train.mean().values, rtol=1e-6)


def test_pipeline_predict_proba_shape():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("wxyz"))
    y = pd.Series(rng.integers(0, 2, size=100))

    pipeline = build_pipeline(LogisticRegression())
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)

    assert proba.shape == (100, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
