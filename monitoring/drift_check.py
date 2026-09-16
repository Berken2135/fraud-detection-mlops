"""Compares recently logged live predictions against a reference sample of
the training data using Evidently, and reports whether production traffic
has drifted enough to warrant retraining.

Requires predictions already logged in Postgres (run
`monitoring/simulate_traffic.py` first). Run with: python -m monitoring.drift_check
"""
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from sqlalchemy import text

from api.db import engine
from src import config

REPORT_PATH = config.ARTIFACTS_DIR / "drift_report.html"
REFERENCE_SAMPLE_SIZE = 2000
CURRENT_SAMPLE_SIZE = 400


def load_reference(n: int = REFERENCE_SAMPLE_SIZE) -> pd.DataFrame:
    """Samples the original dataset as the "what the model was trained on" baseline."""
    df = pd.read_csv(config.DATA_PATH)
    return df.drop(columns=["Class"]).sample(n=n, random_state=config.RANDOM_STATE)


def load_current(n: int = CURRENT_SAMPLE_SIZE) -> pd.DataFrame:
    """Loads the most recently logged live predictions from Postgres."""
    query = text("SELECT features FROM predictions ORDER BY id DESC LIMIT :n")
    with engine.connect() as conn:
        rows = conn.execute(query, {"n": n}).fetchall()
    if not rows:
        raise RuntimeError("No logged predictions found. Run monitoring/simulate_traffic.py first.")
    return pd.DataFrame([row[0] for row in rows])


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Runs Evidently's data drift preset, saves an HTML report, and returns a summary."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))

    result = report.as_dict()["metrics"][0]["result"]
    return {
        "dataset_drift": result["dataset_drift"],
        "share_of_drifted_columns": result["share_of_drifted_columns"],
        "number_of_drifted_columns": result["number_of_drifted_columns"],
        "report_path": str(REPORT_PATH),
    }


def main() -> dict:
    reference = load_reference()
    current = load_current()
    summary = compute_drift(reference, current)
    print(
        f"dataset_drift={summary['dataset_drift']} "
        f"share_of_drifted_columns={summary['share_of_drifted_columns']:.2%} "
        f"({summary['number_of_drifted_columns']} columns) -> {summary['report_path']}"
    )
    return summary


if __name__ == "__main__":
    main()
