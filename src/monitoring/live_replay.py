"""Live monitoring replay + retraining simulation (Module 7).

Unlike ``run_monitoring.py`` (offline Evidently job), this module is what the
Streamlit dashboard calls: it loads the model table, scores the champion on
each rolling window, and can train a challenger when a retrain trigger fires.

Walk-forward retrain protocol (matches "as new sales data arrives"):
  Standing at the end of decision month T (labels for T just became available):
  1. Labeled pool = 2006–2009 train + 2010 sales with month < T (through T−1).
  2. Challenger = LightGBM quantile models refit on that pool.
  3. Evaluate champion vs challenger **only on month T** (the new sales).
  4. Promote only if challenger RMSE on month T beats the champion.
  Month T is never used to train the challenger.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from src.config import (BIAS_ALERT_PCT, COVERAGE_ALERT, MONITORING_REPORTS_DIR,
                        RANDOM_SEED, RMSE_ALERT_RATIO, WINDOW_MONTHS)
from src.features.build_model_table import MONITORING_TARGET
from src.modeling.model_inputs import (category_levels, encode_for_trees,
                                       load_model_table)
from src.modeling.train_final import conformal_offset, predict_interval
from src.modeling.valuation_service import ValuationService

# Faster than production training — enough trees for a fair demo comparison
_CHALLENGER_PARAMS = dict(
    n_estimators=400, learning_rate=0.05, num_leaves=8,
    min_child_samples=15, colsample_bytree=0.7, subsample=0.8,
    subsample_freq=1, reg_lambda=1.0, random_state=RANDOM_SEED, verbose=-1,
)


def _score_window(models, offset, window: pd.DataFrame, levels: dict,
                  y: np.ndarray) -> dict[str, float]:
    X = encode_for_trees(window, levels)
    iv = predict_interval(models, offset, X)
    pred = iv.point.to_numpy()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "bias_pct": float((pred - y).mean() / np.median(y) * 100),
        "coverage": float(((y >= iv.low.to_numpy()) & (y <= iv.high.to_numpy())).mean()),
        "n": int(len(window)),
    }


def _merge_drift_from_csv(summary: pd.DataFrame) -> pd.DataFrame:
    """Attach Evidently T1 columns from the committed offline job when present."""
    csv_path = MONITORING_REPORTS_DIR / "monitoring_summary.csv"
    if not csv_path.exists():
        summary["drift_share"] = np.nan
        summary["t1_drift"] = False
        return summary
    offline = pd.read_csv(csv_path)[["month", "drift_share", "t1_drift"]]
    out = summary.drop(columns=["drift_share", "t1_drift"], errors="ignore")
    return out.merge(offline, on="month", how="left")


def compute_rolling_metrics(
    artifact: dict | None = None,
) -> tuple[pd.DataFrame, float]:
    """Score the champion month-by-month on the 2010 stream (live from data)."""
    train, stream = load_model_table()
    if artifact is None:
        artifact = ValuationService().artifact
    models = artifact["models"]
    offset = artifact["conformal_offset_log"]
    levels = artifact["category_levels"]
    baseline_rmse = float(artifact["baseline"]["rmse"])

    stream = stream.sort_values("sale_date").copy()
    stream["month"] = stream["sale_date"].dt.to_period("M")
    months = sorted(stream["month"].unique())

    rows = []
    for m in months:
        window = stream[(stream["month"] > m - WINDOW_MONTHS) & (stream["month"] <= m)]
        y = window[MONITORING_TARGET].to_numpy()
        metrics = _score_window(models, offset, window, levels, y)
        t2 = metrics["rmse"] > RMSE_ALERT_RATIO * baseline_rmse
        t3 = abs(metrics["bias_pct"]) > BIAS_ALERT_PCT
        t4 = metrics["coverage"] < COVERAGE_ALERT
        rows.append({
            "month": str(m),
            "window_n": metrics["n"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "bias_pct": metrics["bias_pct"],
            "coverage": metrics["coverage"],
            "t2_performance": t2,
            "t3_bias": t3,
            "t4_interval": t4,
        })

    summary = pd.DataFrame(rows)
    summary = _merge_drift_from_csv(summary)
    summary["t1_drift"] = summary["t1_drift"].fillna(False).astype(bool)
    summary["retrain"] = (
        summary["t1_drift"] | summary["t2_performance"]
        | summary["t3_bias"] | summary["t4_interval"]
    )
    summary["drift_pct"] = summary["drift_share"] * 100
    summary["coverage_pct"] = summary["coverage"] * 100
    return summary, baseline_rmse


def _train_challenger(fit_df: pd.DataFrame, levels: dict) -> dict[str, Any]:
    """Fit quantile models + conformal offset on an expanded labeled pool."""
    # Labels: monitoring target so 2010 rebound enters the training signal
    fit_part, cal_part = train_test_split(
        fit_df, test_size=0.20, random_state=RANDOM_SEED)
    X_fit = encode_for_trees(fit_part, levels)
    X_cal = encode_for_trees(cal_part, levels)
    y_fit_log = np.log1p(fit_part[MONITORING_TARGET]).to_numpy()
    y_cal_log = np.log1p(cal_part[MONITORING_TARGET]).to_numpy()

    models = {}
    for name, alpha in {"p10": 0.10, "p50": 0.50, "p90": 0.90}.items():
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **_CHALLENGER_PARAMS)
        m.fit(X_fit, y_fit_log)
        models[name] = m
    offset = conformal_offset(models, X_cal, y_cal_log)
    return {"models": models, "conformal_offset_log": offset,
            "category_levels": levels, "n_train": len(fit_df),
            "n_fit": len(fit_part), "n_cal": len(cal_part)}


def run_retrain_simulation(
    artifact: dict | None = None,
    decision_month: str | None = None,
) -> dict[str, Any]:
    """Walk-forward: train through T−1, score champion vs challenger on month T only."""
    train, stream = load_model_table()
    if artifact is None:
        artifact = ValuationService().artifact
    summary, _ = compute_rolling_metrics(artifact)

    if decision_month is None:
        # Default: first month *after* the first retrain signal (designed: 2010-04)
        fired = summary.loc[summary["retrain"], "month"]
        if fired.empty:
            decision_month = str(summary["month"].iloc[0])
        else:
            first_signal = pd.Period(str(fired.iloc[0]), freq="M")
            later = summary.loc[
                summary["month"].map(lambda m: pd.Period(m, freq="M") > first_signal),
                "month",
            ]
            decision_month = str(later.iloc[0]) if len(later) else str(fired.iloc[0])

    decision_period = pd.Period(decision_month, freq="M")
    stream = stream.sort_values("sale_date").copy()
    stream["month"] = stream["sale_date"].dt.to_period("M")

    # Data available *before* month T closes into the training pool
    labeled_before = stream[stream["month"] < decision_period]
    eval_month = stream[stream["month"] == decision_period]
    if len(eval_month) == 0:
        return {"status": "no_eval",
                "message": f"No labeled sales in {decision_month} to score."}

    pool = pd.concat([train, labeled_before], ignore_index=True)
    levels = category_levels(pool)
    challenger = _train_challenger(pool, levels)

    y_eval = eval_month[MONITORING_TARGET].to_numpy()
    champ = _score_window(
        artifact["models"], artifact["conformal_offset_log"],
        eval_month, artifact["category_levels"], y_eval)
    chal = _score_window(
        challenger["models"], challenger["conformal_offset_log"],
        eval_month, levels, y_eval)

    promote = chal["rmse"] < champ["rmse"]
    labeled_through = (
        str(labeled_before["month"].max()) if len(labeled_before) else "2009-12"
    )
    return {
        "status": "ok",
        "decision_month": decision_month,
        "labeled_through": labeled_through,
        "n_original_train": len(train),
        "n_new_labels": len(labeled_before),
        "n_pool": len(pool),
        "eval_month": decision_month,
        "n_eval": len(eval_month),
        "champion": champ,
        "challenger": chal,
        "promote": promote,
        "decision": (
            f"PROMOTE challenger — better RMSE on {decision_month} new sales"
            if promote else
            f"KEEP champion — challenger did not beat RMSE on {decision_month}"
        ),
    }
