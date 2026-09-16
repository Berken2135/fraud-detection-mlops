"""Trains candidate models, logs everything to MLflow, and promotes the best
one to the "production" alias in the Model Registry if it beats the current
production model's test PR-AUC.

Run with: python -m src.train
"""
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config
from src.data import load_data, split_data
from src.evaluate import compute_metrics, find_best_threshold, plot_confusion_matrix, plot_pr_curve
from src.features import build_pipeline

CANDIDATE_PARAMS: dict[str, dict] = {
    "logistic_regression": {"class_weight": "balanced", "max_iter": 1000},
    "xgboost": config.XGB_PARAMS,
}


def build_candidates() -> dict[str, Pipeline]:
    """Builds the (unfit) candidate pipelines compared during training."""
    return {
        "logistic_regression": build_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE)
        ),
        "xgboost": build_pipeline(XGBClassifier(**config.XGB_PARAMS)),
    }


def train_and_log(name: str, pipeline: Pipeline, X_train, y_train, X_val, y_val) -> tuple[str, float]:
    """Fits `pipeline`, logs params/metrics/artifacts/model to MLflow, returns (run_id, val_pr_auc)."""
    with mlflow.start_run(run_name=name) as run:
        pipeline.fit(X_train, y_train)
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_val_proba, config.DEFAULT_THRESHOLD)

        mlflow.log_params({f"clf__{k}": v for k, v in CANDIDATE_PARAMS[name].items()})
        mlflow.log_metrics(
            {
                "val_pr_auc": metrics["pr_auc"],
                "val_precision": metrics["precision"],
                "val_recall": metrics["recall"],
                "val_f1": metrics["f1"],
            }
        )

        cm_path = config.ARTIFACTS_DIR / f"{name}_val_confusion_matrix.png"
        plot_confusion_matrix(metrics["confusion_matrix"], cm_path, title=f"{name} (val, threshold=0.5)")
        mlflow.log_artifact(str(cm_path))

        pr_path = config.ARTIFACTS_DIR / f"{name}_val_pr_curve.png"
        plot_pr_curve(y_val, y_val_proba, pr_path, title=f"{name} (val)")
        mlflow.log_artifact(str(pr_path))

        signature = infer_signature(X_train, pipeline.predict_proba(X_train))
        mlflow.sklearn.log_model(pipeline, artifact_path="model", signature=signature, input_example=X_train.head(3))

        return run.info.run_id, metrics["pr_auc"]


def promote_if_better(client: MlflowClient, new_version: str, new_pr_auc: float) -> bool:
    """Sets the "production" alias on `new_version` only if it beats the
    current production model's logged test PR-AUC (or none exists yet).

    Reused by the retraining pipeline (pipelines/retrain_flow.py) so both the
    initial training run and automated retraining share one promotion rule.
    """
    try:
        current = client.get_model_version_by_alias(config.REGISTERED_MODEL_NAME, config.PRODUCTION_ALIAS)
        current_pr_auc = float(current.tags.get("test_pr_auc", "-1"))
    except MlflowException:
        current_pr_auc = -1.0

    if new_pr_auc > current_pr_auc:
        client.set_registered_model_alias(config.REGISTERED_MODEL_NAME, config.PRODUCTION_ALIAS, new_version)
        print(
            f"Promoted version {new_version} to '{config.PRODUCTION_ALIAS}' "
            f"(PR-AUC {new_pr_auc:.4f} > {current_pr_auc:.4f})"
        )
        return True

    print(f"Not promoted: version {new_version} PR-AUC {new_pr_auc:.4f} <= current production {current_pr_auc:.4f}")
    return False


def main(random_state: int = config.RANDOM_STATE) -> None:
    """Trains, evaluates, registers, and conditionally promotes the best model.

    `random_state` controls the train/val/test split only (see
    `data.split_data`) — the retraining pipeline passes a fresh seed per run
    to sample a different split each time, standing in for newly collected data.
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, random_state=random_state)

    candidates = build_candidates()
    results: dict[str, tuple[str, float]] = {}
    for name, pipeline in candidates.items():
        run_id, val_pr_auc = train_and_log(name, pipeline, X_train, y_train, X_val, y_val)
        results[name] = (run_id, val_pr_auc)
        candidates[name] = pipeline  # already fit in-place
        print(f"{name}: val PR-AUC={val_pr_auc:.4f}")

    best_name = max(results, key=lambda k: results[k][1])
    best_run_id, best_val_pr_auc = results[best_name]
    best_pipeline = candidates[best_name]
    print(f"Best candidate: {best_name} (val PR-AUC={best_val_pr_auc:.4f})")

    y_val_proba = best_pipeline.predict_proba(X_val)[:, 1]
    best_threshold, _ = find_best_threshold(y_val, y_val_proba)

    y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_test_proba, best_threshold)
    print(
        f"Test (optimized threshold={best_threshold:.4f}): "
        f"PR-AUC={test_metrics['pr_auc']:.4f} precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f}"
    )

    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_param("selected_threshold", best_threshold)
        mlflow.log_metrics(
            {
                "test_pr_auc": test_metrics["pr_auc"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
            }
        )
        cm_path = config.ARTIFACTS_DIR / f"{best_name}_test_confusion_matrix.png"
        plot_confusion_matrix(
            test_metrics["confusion_matrix"], cm_path, title=f"{best_name} (test, threshold={best_threshold:.3f})"
        )
        mlflow.log_artifact(str(cm_path))

    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri, config.REGISTERED_MODEL_NAME)

    client = MlflowClient()
    model_name, version = config.REGISTERED_MODEL_NAME, registered.version
    client.set_model_version_tag(model_name, version, "test_pr_auc", str(test_metrics["pr_auc"]))
    client.set_model_version_tag(model_name, version, "threshold", str(best_threshold))
    client.set_model_version_tag(model_name, version, "candidate", best_name)

    promote_if_better(client, registered.version, test_metrics["pr_auc"])


if __name__ == "__main__":
    main()
