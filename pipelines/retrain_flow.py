"""Prefect flow: checks recent production traffic for feature drift and
retrains the model only when drift crosses a threshold. A retrained model is
promoted to "production" only if it beats the current production PR-AUC —
this reuses `src.train.main` (which calls `promote_if_better` internally) so
the initial training run and this automated flow share one promotion rule.

Requires predictions already logged in Postgres (run
`monitoring/simulate_traffic.py` first). Run with: python -m pipelines.retrain_flow
"""
import random

from prefect import flow, get_run_logger, task

from monitoring.drift_check import compute_drift, load_current, load_reference
from src import config
from src.train import main as train_main


@task
def check_drift() -> dict:
    reference = load_reference()
    current = load_current()
    return compute_drift(reference, current)


@task
def retrain() -> None:
    # A fresh random split stands in for a newly collected batch of labeled
    # data; the fixed project seed is reserved for reproducible baseline runs.
    seed = random.randint(0, 1_000_000)
    train_main(random_state=seed)


@flow(name="fraud-detection-retrain")
def retrain_flow() -> None:
    logger = get_run_logger()
    drift = check_drift()
    logger.info(
        f"Drift check: {drift['share_of_drifted_columns']:.2%} of features drifted "
        f"({drift['number_of_drifted_columns']} columns), report at {drift['report_path']}"
    )

    if drift["share_of_drifted_columns"] >= config.DRIFT_SHARE_THRESHOLD:
        logger.info(
            f"Drift share {drift['share_of_drifted_columns']:.2%} >= threshold "
            f"{config.DRIFT_SHARE_THRESHOLD:.2%} - retraining."
        )
        retrain()
    else:
        logger.info(
            f"Drift share {drift['share_of_drifted_columns']:.2%} below threshold "
            f"{config.DRIFT_SHARE_THRESHOLD:.2%} - skipping retrain."
        )


if __name__ == "__main__":
    retrain_flow()
