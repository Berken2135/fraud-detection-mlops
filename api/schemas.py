"""Pydantic request/response schemas for the prediction API.

Field order in `Transaction` matches the training feature order exactly
(Time, V1..V28, Amount) — scikit-learn pipelines validate column order
against what they saw during `fit`, so this order must not change.
"""
from pydantic import BaseModel, Field


class Transaction(BaseModel):
    Time: float = Field(..., ge=0, description="Seconds elapsed since the first transaction in the dataset")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount")


class BatchRequest(BaseModel):
    transactions: list[Transaction]


class PredictionResponse(BaseModel):
    prediction: int
    fraud_probability: float
    threshold: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None
    database_connected: bool
