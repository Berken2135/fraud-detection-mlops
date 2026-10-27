"""Exported-model path used by the Azure deployment: export to a directory,
then load it in the API via MODEL_PATH — no MLflow server involved."""
import json

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from api import main as api_main
from src import config


def _train_tiny_model() -> LogisticRegression:
    rng = np.random.default_rng(config.RANDOM_STATE)
    X = pd.DataFrame(rng.normal(size=(50, 2)), columns=["Amount", "V1"])
    y = (X["Amount"] > 0).astype(int)
    return LogisticRegression().fit(X, y)


def test_load_exported_model_reads_threshold_and_version(tmp_path):
    target = tmp_path / "model"
    mlflow.sklearn.save_model(_train_tiny_model(), target)
    (target / config.MODEL_META_FILE).write_text(json.dumps({"version": "7", "threshold": 0.42}))

    api_main.load_exported_model(str(target))

    assert api_main.model_state["model"] is not None
    assert api_main.model_state["threshold"] == 0.42
    assert api_main.model_state["version"] == "7"


def test_load_exported_model_missing_dir_leaves_service_degraded(tmp_path):
    api_main.load_exported_model(str(tmp_path / "does-not-exist"))

    assert api_main.model_state["model"] is None
    assert api_main.model_state["version"] is None


def test_load_production_model_prefers_model_path(monkeypatch):
    calls = []
    monkeypatch.setattr(config, "MODEL_PATH", "/app/model")
    monkeypatch.setattr(api_main, "load_exported_model", calls.append)

    api_main.load_production_model()

    assert calls == ["/app/model"]
