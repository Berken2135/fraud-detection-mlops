"""Central configuration: constants and paths used across the project.

Nothing here should be hardcoded elsewhere — modules import from this file.
"""
import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = PROJECT_ROOT / "data" / "creditcard.csv"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"

RANDOM_STATE: int = 42
TARGET_COL: str = "Class"

# Stratified split: 70% train / 15% val / 15% test.
VAL_TEST_SIZE: float = 0.30
TEST_OF_TEMP_SIZE: float = 0.50

# MLflow
MLFLOW_TRACKING_URI: str = os.environ.get(
    "MLFLOW_TRACKING_URI", f"file:{(PROJECT_ROOT / 'mlruns').as_posix()}"
)
EXPERIMENT_NAME: str = "fraud-detection"
REGISTERED_MODEL_NAME: str = "fraud-xgboost"
PRODUCTION_ALIAS: str = "production"

# XGBoost hyperparameters (winner of the Stage 1 baseline comparison).
XGB_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.1,
    "eval_metric": "aucpr",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

DEFAULT_THRESHOLD: float = 0.5

# API / database
# 127.0.0.1 (not "localhost") avoids a slow IPv6-then-IPv4 dual-stack retry
# when Postgres is unreachable - each address gets its own connect_timeout.
DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/fraud_detection"
)

# Retraining: fraction of features Evidently must flag as drifted to trigger a retrain.
DRIFT_SHARE_THRESHOLD: float = 0.3
