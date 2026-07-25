"""Module 7 — post-deployment monitoring. Run from the repo root:

    python -m src.monitoring.run_monitoring

Replays the 2010 holdout stream as a production system would see it: each
month, the monitor evaluates the most recent 3 months of incoming sales
(a rolling window — single months of 6-48 sales are too small for stable
statistics) against the 2006-2009 training baseline.

Retraining triggers (all defined BEFORE looking at stream results):

  T1  DATA DRIFT       share of drifting input features > 30% (Evidently
                       per-column tests; listing_month is excluded — the
                       calendar "drifts" every month by definition)
  T2  PERFORMANCE      rolling RMSE > 1.25 x training baseline
  T3  SYSTEMATIC BIAS  |rolling mean error| > 5% of the window's median
                       price — the model is consistently mis-pricing in one
                       direction even if RMSE looks acceptable
  T4  INTERVAL HEALTH  observed coverage of the 80% range < 65% — the
                       confidence promise itself is breaking

Outputs: monitoring/reports/drift_report_YYYY-MM.html (Evidently, one per
month), monitoring_summary.csv, reports/figures/09_monitoring_timeline.png.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config import (BIAS_ALERT_PCT, COVERAGE_ALERT, DRIFT_SHARE_ALERT,
                        MONITORING_REPORTS_DIR, RMSE_ALERT_RATIO, ROOT_DIR,
                        WINDOW_MONTHS)
from src.features.build_model_table import (CATEGORICAL_FEATURES,
                                            MONITORING_TARGET,
                                            NUMERIC_FEATURES)
from src.modeling.model_inputs import category_levels, encode_for_trees, load_model_table
from src.modeling.train_final import predict_interval
from src.modeling.valuation_service import ValuationService

MIN_DRIFT_SAMPLE = 40  # below this, per-feature drift tests are noise, not signal

# listing_month excluded: the calendar always "drifts"; that is seasonality,
# not a data-quality or distribution problem
DRIFT_COLUMNS = [c for c in NUMERIC_FEATURES if c != "listing_month"] + CATEGORICAL_FEATURES


def main() -> None:
    train, stream = load_model_table()
    service = ValuationService()
    art = service.artifact
    models, offset = art["models"], art["conformal_offset_log"]
    baseline_rmse = art["baseline"]["rmse"]
    levels = category_levels(train)

    mapping = ColumnMapping(
        numerical_features=[c for c in NUMERIC_FEATURES if c != "listing_month"],
        categorical_features=CATEGORICAL_FEATURES)
    reference = train[DRIFT_COLUMNS]

    stream = stream.sort_values("sale_date")
    stream["month"] = stream["sale_date"].dt.to_period("M")
    months = sorted(stream["month"].unique())

    MONITORING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"training baseline RMSE ${baseline_rmse:,.0f} | "
          f"rolling {WINDOW_MONTHS}-month window over the 2010 stream:\n")
    print(f"{'month':<9}{'n':>4}  {'RMSE':>9}  {'bias':>7}  {'drift':>6}  "
          f"{'coverage':>9}  triggers")

    for m in months:
        window = stream[(stream["month"] > m - WINDOW_MONTHS) & (stream["month"] <= m)]
        X = encode_for_trees(window, levels)
        iv = predict_interval(models, offset, X)
        # Performance simulation uses the separate market-adjusted label.
        # The model itself was trained/evaluated on original Kaggle SalePrice.
        y = window[MONITORING_TARGET].to_numpy()

        rmse = float(np.sqrt(((iv.point.to_numpy() - y) ** 2).mean()))
        bias_pct = float((iv.point.to_numpy() - y).mean() / np.median(y) * 100)
        coverage = float(((y >= iv.low.to_numpy()) & (y <= iv.high.to_numpy())).mean())

        # drift statistics need a minimum sample; with fewer rows the tests
        # flag noise (verified: 10-row windows "drift" on every feature)
        if len(window) >= MIN_DRIFT_SAMPLE:
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference, current_data=window[DRIFT_COLUMNS],
                       column_mapping=mapping)
            drift_share = report.as_dict()["metrics"][0]["result"]["share_of_drifted_columns"]
            report.save_html(str(MONITORING_REPORTS_DIR / f"drift_report_{m}.html"))
            t1 = drift_share > DRIFT_SHARE_ALERT
        else:
            drift_share, t1 = float("nan"), False
        t2 = rmse > RMSE_ALERT_RATIO * baseline_rmse
        t3 = abs(bias_pct) > BIAS_ALERT_PCT
        t4 = coverage < COVERAGE_ALERT
        fired = [t for t, f in zip(("T1", "T2", "T3", "T4"), (t1, t2, t3, t4)) if f]
        rows.append({"month": str(m), "window_n": len(window), "rmse": rmse,
                     "bias_pct": bias_pct, "coverage": coverage,
                     "drift_share": drift_share, "t1_drift": t1,
                     "t2_performance": t2, "t3_bias": t3, "t4_interval": t4,
                     "retrain": bool(fired)})
        drift_txt = f"{drift_share:>5.0%}" if drift_share == drift_share else "  n/a"
        print(f"{str(m):<9}{len(window):>4}  ${rmse:>8,.0f}  {bias_pct:>6.1f}%  "
              f"{drift_txt}  {coverage:>8.0%}  "
              f"{'RETRAIN: ' + '+'.join(fired) if fired else '-'}")

    summary = pd.DataFrame(rows)
    summary.to_csv(MONITORING_REPORTS_DIR / "monitoring_summary.csv", index=False)

    # timeline figure for the report/slides
    try:
        fig, axes = plt.subplots(4, 1, figsize=(9, 9.5), sharex=True)
        x = summary["month"]
        axes[0].plot(x, summary["rmse"] / 1000, "o-")
        axes[0].axhline(baseline_rmse / 1000, ls="--", c="gray", label="training baseline")
        axes[0].axhline(RMSE_ALERT_RATIO * baseline_rmse / 1000, ls="--", c="red",
                        label=f"alert ({RMSE_ALERT_RATIO}x)")
        axes[0].set(ylabel="RMSE ($k)", title=f"Rolling {WINDOW_MONTHS}-mo RMSE on incoming 2010 sales")
        axes[0].legend(fontsize=8)
        axes[1].plot(x, summary["bias_pct"], "o-", color="#8172B2")
        axes[1].axhline(0, c="gray", lw=0.8)
        for s in (BIAS_ALERT_PCT, -BIAS_ALERT_PCT):
            axes[1].axhline(s, ls="--", c="red")
        axes[1].set(ylabel="Mean bias (%)", title="Systematic under/over-prediction "
                    "(negative = model prices below the market)")
        axes[2].bar(x, summary["drift_share"] * 100, color="#4C72B0")
        axes[2].axhline(DRIFT_SHARE_ALERT * 100, ls="--", c="red")
        axes[2].set(ylabel="Drifting features (%)", title="Data drift (Evidently)")
        axes[3].plot(x, summary["coverage"] * 100, "o-", color="#55A868")
        axes[3].axhline(80, ls="--", c="gray")
        axes[3].axhline(COVERAGE_ALERT * 100, ls="--", c="red")
        axes[3].set(ylabel="Coverage (%)", title="Observed 80%-interval coverage",
                    xlabel="2010 month")
        fig.savefig(ROOT_DIR / "reports" / "figures" / "09_monitoring_timeline.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)
    except RecursionError:
        print("warning: monitoring timeline figure skipped (matplotlib RecursionError)")

    fired_months = summary[summary["retrain"]]
    print(f"\nretraining recommended in {len(fired_months)}/{len(summary)} months"
          + (f" (first: {fired_months['month'].iloc[0]})" if len(fired_months) else ""))
    print(f"wrote {len(summary)} Evidently reports + monitoring_summary.csv")


if __name__ == "__main__":
    main()
