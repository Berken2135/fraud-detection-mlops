"""Tests the drift-threshold decision in the retraining flow, without needing
a real database, MLflow registry, or Prefect server."""
from unittest.mock import MagicMock

from pipelines import retrain_flow
from src import config


def test_retrain_flow_skips_when_drift_below_threshold(monkeypatch):
    monkeypatch.setattr(
        retrain_flow,
        "check_drift",
        lambda: {
            "dataset_drift": False,
            "share_of_drifted_columns": config.DRIFT_SHARE_THRESHOLD - 0.1,
            "number_of_drifted_columns": 1,
            "report_path": "irrelevant.html",
        },
    )
    retrain_mock = MagicMock()
    monkeypatch.setattr(retrain_flow, "retrain", retrain_mock)

    retrain_flow.retrain_flow()

    retrain_mock.assert_not_called()


def test_retrain_flow_retrains_when_drift_exceeds_threshold(monkeypatch):
    monkeypatch.setattr(
        retrain_flow,
        "check_drift",
        lambda: {
            "dataset_drift": True,
            "share_of_drifted_columns": config.DRIFT_SHARE_THRESHOLD + 0.1,
            "number_of_drifted_columns": 5,
            "report_path": "irrelevant.html",
        },
    )
    retrain_mock = MagicMock()
    monkeypatch.setattr(retrain_flow, "retrain", retrain_mock)

    retrain_flow.retrain_flow()

    retrain_mock.assert_called_once()
