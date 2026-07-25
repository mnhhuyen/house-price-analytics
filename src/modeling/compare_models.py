"""Module 5 (part 1) — compare regressors with 5-fold CV. Run from repo root:

    python -m src.modeling.compare_models

Compares Linear/Ridge/Lasso, Random Forest, and LightGBM on the training
window (2006-2009), all on log(price) with metrics reported on the price
scale. Saves models/model_comparison.json and residual-analysis figures for
the best model.
"""

import json

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODELS_DIR, RANDOM_SEED, ROOT_DIR
from src.features.build_model_table import (CATEGORICAL_FEATURES,
                                            NUMERIC_FEATURES, TARGET)
from src.modeling.model_inputs import (category_levels, encode_for_trees,
                                       load_model_table)

FIG_DIR = ROOT_DIR / "reports" / "figures"


def linear_pipeline(model) -> Pipeline:
    """Scaling + one-hot encoding for the linear family."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def price_scale_metrics(y_true: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """Metrics on the dollar scale (predictions come out of log space)."""
    pred = np.expm1(y_pred_log)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "mape_pct": float(mean_absolute_percentage_error(y_true, pred) * 100),
        "r2": float(r2_score(y_true, pred)),
    }


def residual_analysis(y_true: pd.Series, y_pred: np.ndarray, name: str) -> None:
    """Check model assumptions: no trend vs fitted values, no systematic
    under/over-prediction by price segment."""
    resid_pct = (y_pred - y_true) / y_true * 100
    decile = pd.qcut(y_true, 10, labels=False) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred / 1000, resid_pct, s=8, alpha=0.4)
    axes[0].axhline(0, color="red", lw=1)
    axes[0].set(xlabel="Predicted price ($k)", ylabel="Residual (% of true price)",
                title=f"{name}: residuals vs fitted (CV predictions)")
    by_decile = resid_pct.groupby(decile).mean()
    axes[1].bar(by_decile.index, by_decile.values, color="#4C72B0")
    axes[1].axhline(0, color="red", lw=1)
    axes[1].set(xlabel="True-price decile (1=cheapest)", ylabel="Mean residual (%)",
                title="Systematic bias by price segment")
    fig.savefig(FIG_DIR / "08_residual_analysis.png", dpi=120,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\nresidual bias by price decile (% of true price):")
    print(by_decile.round(1).to_string())


def main() -> None:
    train, _ = load_model_table()
    y = train[TARGET]
    y_log = np.log1p(y)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    levels = category_levels(train)
    X_linear = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_trees = encode_for_trees(train, levels)
    # sklearn's RandomForest needs plain numerics: ordinal codes are fine for
    # trees (splits are order-agnostic in effect given enough depth)
    X_rf = X_trees.copy()
    for col in CATEGORICAL_FEATURES:
        X_rf[col] = X_rf[col].cat.codes

    candidates = {
        "LinearRegression": (linear_pipeline(LinearRegression()), X_linear),
        "Ridge": (linear_pipeline(Ridge(alpha=10.0)), X_linear),
        "Lasso": (linear_pipeline(Lasso(alpha=0.001, max_iter=50_000)), X_linear),
        "RandomForest": (RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=RANDOM_SEED,
            n_jobs=-1), X_rf),
        "LightGBM": (lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, random_state=RANDOM_SEED, verbose=-1), X_trees),
    }

    results, predictions = {}, {}
    for name, (model, X) in candidates.items():
        pred_log = cross_val_predict(model, X, y_log, cv=cv, n_jobs=1)
        results[name] = price_scale_metrics(y, pred_log)
        predictions[name] = np.expm1(pred_log)
        m = results[name]
        print(f"{name:<18} RMSE ${m['rmse']:>9,.0f}  MAE ${m['mae']:>8,.0f}  "
              f"MAPE {m['mape_pct']:5.2f}%  R2 {m['r2']:.3f}")

    best = min(results, key=lambda k: results[k]["rmse"])
    print(f"\nbest by CV RMSE: {best}")

    MODELS_DIR.mkdir(exist_ok=True)
    out = {"cv_folds": 5, "training_rows": len(train), "best_model": best,
           "metrics": results}
    (MODELS_DIR / "model_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {MODELS_DIR / 'model_comparison.json'}")

    try:
        residual_analysis(y, predictions[best], best)
    except RecursionError:
        # matplotlib 3.x + some Python builds hit MarkerStyle deepcopy recursion
        print("warning: residual figure skipped (matplotlib RecursionError)")


if __name__ == "__main__":
    main()
