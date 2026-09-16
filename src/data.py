"""Data loading and stratified train/val/test splitting."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_data(path: str | None = None) -> pd.DataFrame:
    """Loads the raw transactions CSV."""
    return pd.read_csv(path or config.DATA_PATH)


def split_data(
    df: pd.DataFrame, random_state: int = config.RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified 70/15/15 train/val/test split, preserving class ratio in each split.

    `random_state` defaults to the fixed project seed for reproducibility, but
    the retraining pipeline passes a fresh seed per run to sample a
    different split each time it retrains (standing in for newly collected data).
    """
    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=config.VAL_TEST_SIZE, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=config.TEST_OF_TEMP_SIZE,
        stratify=y_temp,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
