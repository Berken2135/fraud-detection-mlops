"""FastAPI prediction service.

Loads the "production"-aliased model from the MLflow Model Registry at
startup. If no production model exists yet (e.g. before the first
`python -m src.train` run), the service still starts — /health reports
`model_loaded: false` and /predict returns 503 until a model is promoted
and /admin/reload-model is called (or the service is restarted).
"""
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from api import db
from api.schemas import (
    BatchPredictionResponse,
    BatchRequest,
    HealthResponse,
    PredictionResponse,
    Transaction,
)
from src import config

model_state: dict = {"model": None, "threshold": config.DEFAULT_THRESHOLD, "version": None}


def load_production_model() -> None:
    """Loads the current production model + its stored threshold, if one exists."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    try:
        client = MlflowClient()
        version = client.get_model_version_by_alias(config.REGISTERED_MODEL_NAME, config.PRODUCTION_ALIAS)
        model_uri = f"models:/{config.REGISTERED_MODEL_NAME}@{config.PRODUCTION_ALIAS}"
        model_state["model"] = mlflow.sklearn.load_model(model_uri)
        model_state["threshold"] = float(version.tags.get("threshold", config.DEFAULT_THRESHOLD))
        model_state["version"] = version.version
        print(f"Loaded production model version {version.version} (threshold={model_state['threshold']:.4f})")
    except MlflowException as exc:
        model_state["model"] = None
        model_state["version"] = None
        print(f"WARNING: no production model available yet ({exc}). Run 'make train' first.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    load_production_model()
    yield


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


def _predict_dataframe(df: pd.DataFrame) -> tuple[list[int], list[float]]:
    probas = model_state["model"].predict_proba(df)[:, 1]
    preds = (probas >= model_state["threshold"]).astype(int)
    return preds.tolist(), probas.tolist()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model_state["model"] is not None else "degraded",
        model_loaded=model_state["model"] is not None,
        model_version=str(model_state["version"]) if model_state["version"] is not None else None,
        database_connected=db.is_connected(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction) -> PredictionResponse:
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="No production model loaded yet")

    df = pd.DataFrame([transaction.model_dump()])
    preds, probas = _predict_dataframe(df)
    pred, proba = preds[0], probas[0]

    db.log_predictions(
        [
            {
                "features": transaction.model_dump(),
                "prediction": pred,
                "fraud_probability": proba,
                "threshold": model_state["threshold"],
                "model_version": str(model_state["version"]),
            }
        ]
    )
    return PredictionResponse(
        prediction=pred,
        fraud_probability=proba,
        threshold=model_state["threshold"],
        model_version=str(model_state["version"]),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchRequest) -> BatchPredictionResponse:
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="No production model loaded yet")
    if not batch.transactions:
        raise HTTPException(status_code=400, detail="transactions must not be empty")

    df = pd.DataFrame([t.model_dump() for t in batch.transactions])
    preds, probas = _predict_dataframe(df)

    rows = [
        {
            "features": t.model_dump(),
            "prediction": pred,
            "fraud_probability": proba,
            "threshold": model_state["threshold"],
            "model_version": str(model_state["version"]),
        }
        for t, pred, proba in zip(batch.transactions, preds, probas)
    ]
    db.log_predictions(rows)

    return BatchPredictionResponse(
        predictions=[
            PredictionResponse(
                prediction=r["prediction"],
                fraud_probability=r["fraud_probability"],
                threshold=r["threshold"],
                model_version=r["model_version"],
            )
            for r in rows
        ]
    )


@app.post("/admin/reload-model", response_model=HealthResponse)
def reload_model() -> HealthResponse:
    """Re-reads the "production" alias from the registry without restarting
    the service — call this after a retraining run promotes a new model."""
    load_production_model()
    return health()
