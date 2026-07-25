"""Monitoring dashboard — live 2010 replay + retraining simulation.

Scores the champion model directly on ``model_table.csv`` (rolling RMSE/MAE/
bias/coverage). Evidently drift HTML stays as a committed drill-down; T1 flags
are merged from that offline job. Retraining simulation trains a challenger on
labeled data through the first trigger month and promotes only if it beats the
champion on a future holdout never used for training.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit Cloud clones into /mount/src/<repo>. If /mount is on sys.path,
# `import src` resolves to the mount folder, not this project's package.
_ROOT = Path(__file__).resolve().parent.parent
_SHADOW = {"/mount", "/mount/src"}
sys.path[:] = [
    p for p in sys.path
    if str(Path(p).resolve()) not in _SHADOW
]
sys.path.insert(0, str(_ROOT))

from src.config import (BIAS_ALERT_PCT, COVERAGE_ALERT, DRIFT_SHARE_ALERT,
                        MONITORING_REPORTS_DIR, RMSE_ALERT_RATIO, ROOT_DIR,
                        WINDOW_MONTHS)
from src.modeling.valuation_service import ValuationService
from src.monitoring.live_replay import (compute_rolling_metrics,
                                        run_retrain_simulation)

TIMELINE_PNG = ROOT_DIR / "reports" / "figures" / "09_monitoring_timeline.png"


@st.cache_resource
def _champion():
    return ValuationService().artifact


@st.cache_data(show_spinner="Scoring champion on the 2010 stream…")
def _live_summary() -> tuple[pd.DataFrame, float]:
    return compute_rolling_metrics(_champion())


st.title("📈 Model monitoring — the 2010 replay")

df, baseline_rmse = _live_summary()
rmse_alert = RMSE_ALERT_RATIO * baseline_rmse

st.caption(
    f"Metrics below are **computed live** from `model_table.csv` + the "
    f"deployed champion (not a static scorecard). Rolling "
    f"{WINDOW_MONTHS}-month windows on the 175 held-out 2010 sales; labels "
    f"use the market-adjusted monitoring scenario. Drift (T1) still comes "
    f"from the committed Evidently job — Evidently is too heavy for the free "
    f"tier at request time."
)

# ---- headline metrics -----------------------------------------------------
latest = df.iloc[-1]
retrain_months = df.loc[df["retrain"], "month"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rolling RMSE (latest)", f"${latest.rmse:,.0f}",
          delta=f"{latest.rmse / baseline_rmse - 1:+.0%} vs training baseline",
          delta_color="inverse")
c2.metric("Rolling MAE (latest)", f"${latest.mae:,.0f}")
c3.metric("Systematic bias", f"{latest.bias_pct:+.1f}%",
          delta="under-pricing" if latest.bias_pct < 0 else None,
          delta_color="off")
c4.metric("Verdict", f"retrain since {retrain_months.iloc[0]}"
          if len(retrain_months) else "healthy")

# ---- timeline (committed figure, same as slides) --------------------------
st.subheader("Four monitors over the 2010 stream")
if TIMELINE_PNG.exists():
    st.image(str(TIMELINE_PNG), use_container_width=True)
    st.caption(
        f"Gray dashed = training baseline / promised coverage. "
        f"Red dashed = alert thresholds "
        f"(T2 RMSE {RMSE_ALERT_RATIO}×, T3 bias ±{BIAS_ALERT_PCT:.0f}%, "
        f"T1 drift {DRIFT_SHARE_ALERT:.0%}, T4 coverage {COVERAGE_ALERT:.0%})."
    )

# ---- live RMSE / MAE table (no Altair — Cloud /mount + local Py3.14 safe) --
st.subheader("Live rolling error (computed in this app)")
st.dataframe(pd.DataFrame({
    "month": df["month"],
    "window n": df["window_n"],
    "RMSE": df["rmse"].map(lambda x: f"${x:,.0f}"),
    "MAE": df["mae"].map(lambda x: f"${x:,.0f}"),
    "bias %": df["bias_pct"].map(lambda x: f"{x:+.1f}%"),
    "coverage": df["coverage"].map(lambda x: f"{x:.0%}"),
}), hide_index=True, use_container_width=True)
st.caption(
    f"RMSE alert ≈ ${rmse_alert:,.0f} "
    f"({RMSE_ALERT_RATIO}× training baseline ${baseline_rmse:,.0f}). "
    "Values are scored now from the model table + champion artifact."
)

# ---- month drill-down -----------------------------------------------------
st.subheader("Month drill-down")
month = st.selectbox("Select a 2010 month", df["month"].tolist(),
                     index=len(df) - 1)
row = df.loc[df["month"] == month].iloc[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Sales in window", f"{int(row.window_n)}")
m2.metric("RMSE", f"${row.rmse:,.0f}",
          delta=f"alert at ${rmse_alert:,.0f}", delta_color="off")
m3.metric("MAE", f"${row.mae:,.0f}")
m4.metric("Bias", f"{row.bias_pct:+.1f}%",
          delta=f"alert at ±{BIAS_ALERT_PCT:.0f}%", delta_color="off")

triggers = {
    f"T1 input drift (> {DRIFT_SHARE_ALERT:.0%} features)": bool(row.t1_drift),
    f"T2 performance (RMSE > {RMSE_ALERT_RATIO}× baseline)": bool(row.t2_performance),
    f"T3 systematic bias (|bias| > {BIAS_ALERT_PCT:.0f}%)": bool(row.t3_bias),
    f"T4 interval health (coverage < {COVERAGE_ALERT:.0%})": bool(row.t4_interval),
}
fired = [name for name, on in triggers.items() if on]
if fired:
    st.error("**RETRAIN** — " + " · ".join(fired))
else:
    st.success("Healthy — no retraining trigger fired this month.")

# ---- trigger table --------------------------------------------------------
st.subheader("Retraining triggers by month")


def mark(series: pd.Series) -> pd.Series:
    return series.map({True: "FIRED", False: "—"})


st.dataframe(pd.DataFrame({
    "month": df["month"],
    "sales in window": df["window_n"],
    "RMSE": df["rmse"].round(0).map(lambda x: f"${x:,.0f}"),
    "MAE": df["mae"].round(0).map(lambda x: f"${x:,.0f}"),
    f"T1 drift > {DRIFT_SHARE_ALERT:.0%}": mark(df["t1_drift"]),
    f"T2 RMSE > {RMSE_ALERT_RATIO}×": mark(df["t2_performance"]),
    f"T3 |bias| > {BIAS_ALERT_PCT:.0f}%": mark(df["t3_bias"]),
    f"T4 coverage < {COVERAGE_ALERT:.0%}": mark(df["t4_interval"]),
    "verdict": df["retrain"].map({True: "RETRAIN", False: "healthy"}),
}), hide_index=True, use_container_width=True)

# ---- Retraining simulation (walk-forward) ---------------------------------
st.subheader("Retraining simulation")
st.markdown(
    """
Walk-forward — as if you are at the **end of month T**, when that month’s
sale labels just arrived (same spirit as rolling error monitoring):

1. **Today** = decision month **T** (default **2010-04**, the first month
   after the March retrain signal).
2. **Train pool** = 2006–2009 + 2010 sales with month **&lt; T** (through T−1).
3. **Challenger** = refit LightGBM quantile models on that pool.
4. **Score only on month T** — the new sales; T is never used to train.
5. **Promote** only if challenger RMSE on month T beats the champion.
"""
)

# Default: first month after the first retrain signal (March → April)
if len(retrain_months):
    first_signal = pd.Period(str(retrain_months.iloc[0]), freq="M")
    after = [m for m in df["month"].tolist()
             if pd.Period(m, freq="M") > first_signal]
    decision_default = after[0] if after else str(retrain_months.iloc[0])
else:
    decision_default = df["month"].iloc[0]

decision_choice = st.selectbox(
    "Decision month T (today — labels for this month just arrived)",
    df["month"].tolist(),
    index=df["month"].tolist().index(decision_default),
)
st.caption(
    f"Will train through the month before **{decision_choice}**, "
    f"then score both models only on **{decision_choice}**."
)

if st.button("Run retrain simulation", type="primary"):
    with st.spinner(f"Training challenger through month before {decision_choice} "
                    f"and scoring on {decision_choice}…"):
        result = run_retrain_simulation(
            _champion(), decision_month=decision_choice)
    st.session_state["retrain_sim"] = result

result = st.session_state.get("retrain_sim")
if result:
    if result["status"] != "ok":
        st.warning(result.get("message", result["status"]))
    else:
        st.info(
            f"Decision month **{result['decision_month']}** → train pool "
            f"{result['n_original_train']} history + {result['n_new_labels']} "
            f"sales through **{result['labeled_through']}** "
            f"= **{result['n_pool']}** rows. "
            f"Scored only on **{result['eval_month']}** "
            f"(n={result['n_eval']} new sales)."
        )
        left, right = st.columns(2)
        left.metric(f"Champion RMSE ({result['eval_month']})",
                    f"${result['champion']['rmse']:,.0f}",
                    delta=f"MAE ${result['champion']['mae']:,.0f}",
                    delta_color="off")
        right.metric(f"Challenger RMSE ({result['eval_month']})",
                     f"${result['challenger']['rmse']:,.0f}",
                     delta=f"MAE ${result['challenger']['mae']:,.0f}",
                     delta_color="off")
        c_cov, h_cov = st.columns(2)
        c_cov.metric("Champion coverage", f"{result['champion']['coverage']:.0%}")
        h_cov.metric("Challenger coverage", f"{result['challenger']['coverage']:.0%}")

        if result["promote"]:
            st.success(f"**{result['decision']}**")
            st.session_state["production_model"] = "challenger"
        else:
            st.warning(f"**{result['decision']}**")
            st.session_state["production_model"] = "champion"

        active = st.session_state.get("production_model", "champion")
        st.caption(
            f"Session production pointer: **{active}** "
            f"(demo only — the on-disk `valuation_model.pkl` is not overwritten)."
        )

# ---- narrative ------------------------------------------------------------
fired_by = {t: df.loc[df[c], "month"].tolist()
            for t, c in [("T1", "t1_drift"), ("T2", "t2_performance"),
                         ("T3", "t3_bias"), ("T4", "t4_interval")]}
lines = [f"- **{t}** fired in {', '.join(months)}"
         for t, months in fired_by.items() if months]
st.markdown(
    "**Reading the replay:** input drift says the world changed; error and "
    "bias prove that it matters.\n" + "\n".join(lines) + "\n\n"
    "The model, trained on 2006–2009, under-prices progressively as the 2010 "
    "rebound outruns its training window. **Verdict: retrain from March 2010.**"
)

# ---- Evidently drill-down -------------------------------------------------
reports = sorted(MONITORING_REPORTS_DIR.glob("drift_report_*.html"))
if reports:
    with st.expander("Per-month Evidently drift reports (full HTML drill-down)"):
        st.caption("Column-level drift tests behind the T1 numbers above.")
        for p in reports:
            st.download_button(p.name, p.read_bytes(), file_name=p.name,
                               mime="text/html", key=p.name)
