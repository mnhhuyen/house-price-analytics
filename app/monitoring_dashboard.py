"""Monitoring dashboard — Module 7's 2010 replay as a live demo page.

Renders the committed monitoring artifacts produced by
`python -m src.monitoring.run_monitoring` in the dev environment:

  - monitoring/reports/monitoring_summary.csv
  - monitoring/reports/drift_report_YYYY-MM.html
  - reports/figures/09_monitoring_timeline.png

The deployed app only *reads* these files — Evidently never runs on the
free tier, so the lean deploy requirements stay unchanged.

Trigger thresholds come from src/config.py — the same constants the
monitoring job uses, committed before the stream was replayed (D16).
"""

import json
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
                        MODELS_DIR, MONITORING_REPORTS_DIR, RMSE_ALERT_RATIO,
                        ROOT_DIR, WINDOW_MONTHS)

SUMMARY_CSV = MONITORING_REPORTS_DIR / "monitoring_summary.csv"
TIMELINE_PNG = ROOT_DIR / "reports" / "figures" / "09_monitoring_timeline.png"


@st.cache_data
def load_artifacts() -> tuple[pd.DataFrame, float]:
    df = pd.read_csv(SUMMARY_CSV)
    df["drift_pct"] = df["drift_share"] * 100
    df["coverage_pct"] = df["coverage"] * 100
    baseline_rmse = json.loads(
        (MODELS_DIR / "training_summary.json").read_text())["rmse"]
    return df, baseline_rmse


st.title("📈 Model monitoring — the 2010 replay")

if not SUMMARY_CSV.exists():
    st.error("Monitoring artifacts not found. Generate them with "
             "`python -m src.monitoring.run_monitoring` from the repo root.")
    st.stop()

df, baseline_rmse = load_artifacts()
rmse_alert = RMSE_ALERT_RATIO * baseline_rmse

st.caption(
    f"All 175 sales from 2010 were held out of training and replayed "
    f"month-by-month on rolling {WINDOW_MONTHS}-month windows (single months "
    f"hold only 6–48 sales). Performance uses the market-adjusted 2010 "
    f"scenario label — the model, trained on 2006–2009, has never seen the "
    f"designed +9.5% market rebound. Thresholds were committed **before** "
    f"the stream was replayed."
)

# ---- headline metrics -----------------------------------------------------
latest = df.iloc[-1]
retrain_months = df.loc[df["retrain"], "month"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rolling RMSE (latest)", f"${latest.rmse:,.0f}",
          delta=f"{latest.rmse / baseline_rmse - 1:+.0%} vs training baseline",
          delta_color="inverse")
c2.metric("Systematic bias", f"{latest.bias_pct:+.1f}%",
          delta="model prices below the market" if latest.bias_pct < 0 else None,
          delta_color="off")
c3.metric("80%-interval coverage", f"{latest.coverage:.0%}",
          delta=f"{latest.coverage - 0.80:+.0%} vs promised", delta_color="off")
c4.metric("Verdict", f"retrain since {retrain_months.iloc[0]}"
          if len(retrain_months) else "healthy")

# ---- committed timeline figure (same as the slides) -----------------------
st.subheader("Four monitors over the 2010 stream")
if TIMELINE_PNG.exists():
    st.image(str(TIMELINE_PNG), use_container_width=True)
    st.caption(
        f"Gray dashed = training baseline / promised coverage. "
        f"Red dashed = alert thresholds "
        f"(T2 RMSE {RMSE_ALERT_RATIO}× baseline, "
        f"T3 bias ±{BIAS_ALERT_PCT:.0f}%, "
        f"T1 drift {DRIFT_SHARE_ALERT:.0%}, "
        f"T4 coverage {COVERAGE_ALERT:.0%})."
    )
else:
    st.warning("Timeline figure not found — run "
               "`python -m src.monitoring.run_monitoring` to regenerate it.")

# ---- month drill-down -----------------------------------------------------
st.subheader("Month drill-down")
month = st.selectbox("Select a 2010 month", df["month"].tolist(),
                     index=len(df) - 1)
row = df.loc[df["month"] == month].iloc[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Sales in window", f"{int(row.window_n)}")
m2.metric("RMSE", f"${row.rmse:,.0f}",
          delta=f"alert at ${rmse_alert:,.0f}", delta_color="off")
m3.metric("Bias", f"{row.bias_pct:+.1f}%",
          delta=f"alert at ±{BIAS_ALERT_PCT:.0f}%", delta_color="off")
drift_txt = (f"{row.drift_pct:.0f}%" if pd.notna(row.drift_pct)
             else "n/a (<40 sales)")
m4.metric("Drift share", drift_txt,
          delta=f"alert at {DRIFT_SHARE_ALERT:.0%}", delta_color="off")

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


def mark(fired: pd.Series) -> pd.Series:
    return fired.map({True: "FIRED", False: "—"})


st.dataframe(pd.DataFrame({
    "month": df["month"],
    "sales in window": df["window_n"],
    f"T1 drift > {DRIFT_SHARE_ALERT:.0%}": mark(df["t1_drift"]),
    f"T2 RMSE > {RMSE_ALERT_RATIO}×": mark(df["t2_performance"]),
    f"T3 |bias| > {BIAS_ALERT_PCT:.0f}%": mark(df["t3_bias"]),
    f"T4 coverage < {COVERAGE_ALERT:.0%}": mark(df["t4_interval"]),
    "verdict": df["retrain"].map({True: "RETRAIN", False: "healthy"}),
}), hide_index=True, use_container_width=True)

# ---- what happened, in words ----------------------------------------------
fired_by = {t: df.loc[df[c], "month"].tolist()
            for t, c in [("T1", "t1_drift"), ("T2", "t2_performance"),
                         ("T3", "t3_bias"), ("T4", "t4_interval")]}
lines = [f"- **{t}** fired in {', '.join(months)}"
         for t, months in fired_by.items() if months]
st.markdown(
    "**Reading the replay:** input drift says the world changed; error and "
    "bias prove that it matters.\n" + "\n".join(lines) + "\n\n"
    "The model, trained on 2006–2009, under-prices progressively as the 2010 "
    "rebound outruns its training window — an explainable, designed drift "
    "signal, not a random shock. **Verdict: retrain from March 2010.**"
)

# ---- Evidently drill-down -------------------------------------------------
reports = sorted(MONITORING_REPORTS_DIR.glob("drift_report_*.html"))
if reports:
    with st.expander("Per-month Evidently drift reports (full HTML drill-down)"):
        st.caption("Column-level drift tests behind the T1 numbers above. "
                   "Download and open in a browser.")
        for p in reports:
            st.download_button(p.name, p.read_bytes(), file_name=p.name,
                               mime="text/html", key=p.name)
