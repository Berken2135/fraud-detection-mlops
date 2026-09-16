"""Tests for the Evidently-based drift computation. Uses synthetic data only."""
import numpy as np
import pandas as pd

from monitoring.drift_check import compute_drift
from src import config


def _reference_and_current(drifted: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    reference = pd.DataFrame(
        {
            "Amount": rng.normal(50, 20, 500),
            "V1": rng.normal(0, 1, 500),
            "V2": rng.normal(0, 1, 500),
        }
    )
    if drifted:
        current = pd.DataFrame(
            {
                "Amount": rng.normal(500, 100, 200),  # shifted distribution
                "V1": rng.normal(0, 1, 200),
                "V2": rng.normal(0, 1, 200),
            }
        )
    else:
        current = pd.DataFrame(
            {
                "Amount": rng.normal(50, 20, 200),
                "V1": rng.normal(0, 1, 200),
                "V2": rng.normal(0, 1, 200),
            }
        )
    return reference, current


def test_compute_drift_detects_shifted_distribution():
    reference, current = _reference_and_current(drifted=True)
    summary = compute_drift(reference, current)

    # Only "Amount" is shifted (1 of 3 columns) - below Evidently's default
    # 50% dataset_drift threshold, but share_of_drifted_columns (what
    # pipelines.retrain_flow actually acts on) must reflect it.
    assert summary["share_of_drifted_columns"] > 0
    assert summary["number_of_drifted_columns"] >= 1


def test_compute_drift_no_drift_on_same_distribution():
    reference, current = _reference_and_current(drifted=False)
    summary = compute_drift(reference, current)

    assert summary["dataset_drift"] is False
    assert summary["share_of_drifted_columns"] == 0.0


def test_compute_drift_writes_html_report():
    reference, current = _reference_and_current(drifted=True)
    summary = compute_drift(reference, current)

    report_path = config.ARTIFACTS_DIR / "drift_report.html"
    assert report_path.exists()
    assert summary["report_path"] == str(report_path)
