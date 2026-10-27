"""Exports the "production" model from the MLflow registry to model/.

The exported directory is baked into the Azure container image
(deploy/azure/Dockerfile), so the cloud API can serve predictions without
an MLflow server. The threshold and registry version travel with it in
MODEL_META_FILE.

Usage: python -m src.export_model
"""
import json
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src import config


def export_production_model(target: Path = config.EXPORT_DIR) -> Path:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    version = client.get_model_version_by_alias(config.REGISTERED_MODEL_NAME, config.PRODUCTION_ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{config.REGISTERED_MODEL_NAME}@{config.PRODUCTION_ALIAS}")

    if target.exists():
        shutil.rmtree(target)
    mlflow.sklearn.save_model(model, target)

    meta = {
        "version": version.version,
        "threshold": float(version.tags.get("threshold", config.DEFAULT_THRESHOLD)),
    }
    (target / config.MODEL_META_FILE).write_text(json.dumps(meta, indent=2))
    print(f"Exported model version {meta['version']} (threshold={meta['threshold']:.4f}) to {target}")
    return target


if __name__ == "__main__":
    export_production_model()
