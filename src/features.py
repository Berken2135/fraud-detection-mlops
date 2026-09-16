"""Feature preprocessing shared by training and inference.

The scaler must only ever be fit on training data — it is fit inside the
pipeline built here, on whatever X is passed to `Pipeline.fit`, so callers
must pass only the training split.
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor() -> StandardScaler:
    """Returns an unfit StandardScaler for the pipeline's first step."""
    return StandardScaler()


def build_pipeline(classifier) -> Pipeline:
    """Wraps a classifier with the shared preprocessing step.

    Fitting the returned pipeline on X_train fits the scaler on X_train only,
    preventing leakage from validation/test data.
    """
    return Pipeline([("scaler", build_preprocessor()), ("clf", classifier)])
