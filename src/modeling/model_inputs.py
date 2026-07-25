"""Shared model-input handling: load the model table, encode features.

Both the comparison script and the final trainer use exactly these functions,
so "the model we compared" and "the model we ship" can never quietly diverge.
"""

import pandas as pd

from src.config import DATA_PROCESSED_DIR
from src.features.build_model_table import (CATEGORICAL_FEATURES,
                                            NUMERIC_FEATURES, TARGET)

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_model_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (training window, monitoring stream) from model_table.csv."""
    df = pd.read_csv(DATA_PROCESSED_DIR / "model_table.csv",
                     parse_dates=["listing_date", "sale_date"])
    train = df[df["is_monitoring_stream"] == 0].reset_index(drop=True)
    stream = df[df["is_monitoring_stream"] == 1].reset_index(drop=True)
    return train, stream


def category_levels(train: pd.DataFrame) -> dict[str, list]:
    """Fixed category levels learned from training data — stored in the model
    artifact so serving encodes unseen data identically."""
    return {c: sorted(train[c].astype(str).unique()) for c in CATEGORICAL_FEATURES}


def encode_for_trees(df: pd.DataFrame, levels: dict[str, list]) -> pd.DataFrame:
    """LightGBM/RF input: numerics as-is, categoricals as pandas category dtype
    with the fixed training levels (unseen values become NaN — handled natively)."""
    X = df[ALL_FEATURES].copy()
    for col, cats in levels.items():
        X[col] = pd.Categorical(df[col].astype(str), categories=cats)
    return X
