"""API tests with a fake model and a stubbed database — no real MLflow/Postgres needed."""
import numpy as np
from fastapi.testclient import TestClient

from api import db as api_db
from api import main as api_main


class FakeModel:
    """Flags any transaction with Amount >= 1000 as fraud, deterministically."""

    def predict_proba(self, X):
        proba_fraud = (X["Amount"] >= 1000).astype(float).to_numpy()
        return np.column_stack([1 - proba_fraud, proba_fraud])


def _sample_transaction(amount: float = 50.0) -> dict:
    features = {f"V{i}": 0.0 for i in range(1, 29)}
    return {"Time": 0.0, **features, "Amount": amount}


def _client(monkeypatch) -> TestClient:
    monkeypatch.setattr(api_main, "load_production_model", lambda: None)
    monkeypatch.setattr(api_db, "init_db", lambda: None)
    monkeypatch.setattr(api_db, "is_connected", lambda: True)
    monkeypatch.setattr(api_db, "log_predictions", lambda rows: None)
    api_main.model_state.update({"model": FakeModel(), "threshold": 0.5, "version": "1"})
    return TestClient(api_main.app)


def test_health_reports_model_loaded(monkeypatch):
    with _client(monkeypatch) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_loaded"] is True
        assert body["model_version"] == "1"
        assert body["database_connected"] is True


def test_predict_normal_transaction(monkeypatch):
    with _client(monkeypatch) as client:
        resp = client.post("/predict", json=_sample_transaction(amount=50.0))
        assert resp.status_code == 200
        body = resp.json()
        assert body["prediction"] == 0
        assert body["fraud_probability"] == 0.0
        assert body["model_version"] == "1"


def test_predict_flags_high_amount_as_fraud(monkeypatch):
    with _client(monkeypatch) as client:
        resp = client.post("/predict", json=_sample_transaction(amount=5000.0))
        assert resp.status_code == 200
        body = resp.json()
        assert body["prediction"] == 1
        assert body["fraud_probability"] == 1.0


def test_predict_rejects_negative_amount(monkeypatch):
    with _client(monkeypatch) as client:
        resp = client.post("/predict", json=_sample_transaction(amount=-10.0))
        assert resp.status_code == 422


def test_predict_batch(monkeypatch):
    with _client(monkeypatch) as client:
        payload = {"transactions": [_sample_transaction(50.0), _sample_transaction(5000.0)]}
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == 200
        preds = [p["prediction"] for p in resp.json()["predictions"]]
        assert preds == [0, 1]


def test_predict_batch_rejects_empty(monkeypatch):
    with _client(monkeypatch) as client:
        resp = client.post("/predict/batch", json={"transactions": []})
        assert resp.status_code == 400


def test_predict_returns_503_when_model_not_loaded(monkeypatch):
    monkeypatch.setattr(api_main, "load_production_model", lambda: None)
    monkeypatch.setattr(api_db, "init_db", lambda: None)
    monkeypatch.setattr(api_db, "is_connected", lambda: False)
    api_main.model_state.update({"model": None, "threshold": 0.5, "version": None})

    with TestClient(api_main.app) as client:
        resp = client.post("/predict", json=_sample_transaction())
        assert resp.status_code == 503

        health = client.get("/health").json()
        assert health["status"] == "degraded"
        assert health["model_loaded"] is False
