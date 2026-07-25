"""House-price valuation tool — the primary deployed artifact.

Architecture (DECISIONS.md D2): the trained quantile model runs IN-PROCESS —
the live demo never depends on a second running service. The FastAPI service
(app/api/) exposes the same ValuationService for programmatic access.

The confidence range is a conformally calibrated 80% interval: verified on
held-out data to contain the true sale price ~80% of the time (see
models/training_summary.json). Width varies per home — unusual or high-value
homes get honest, wider ranges.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.modeling.valuation_service import ValuationService

# Valuation is expressed as of the end of the model's training window
VALUATION_YEAR = 2010


@st.cache_resource
def load_service() -> ValuationService:
    return ValuationService()


service = load_service()
NEIGHBORHOODS = service.category_levels["Neighborhood"]
KITCHEN_QUAL = ["Ex", "Gd", "TA", "Fa"]  # excellent .. fair, per Ames coding
st.title("🏠 House Price Valuation Tool")
st.caption(
    "Estimates a home's fair market value in Ames, Iowa with a **calibrated "
    "80% confidence range** — trained on 2006–2009 sales (Kaggle Ames data + "
    "synthetic market context). Business Analytics group project."
)

with st.form("valuation_form"):
    col1, col2 = st.columns(2)
    with col1:
        neighborhood = st.selectbox(
            "Neighborhood", NEIGHBORHOODS,
            index=NEIGHBORHOODS.index("NAmes") if "NAmes" in NEIGHBORHOODS else 0)
        living_area = st.number_input("Above-ground living area (sq ft)",
                                      min_value=300, max_value=6000, value=1500, step=50)
        basement_area = st.number_input("Basement area (sq ft, 0 = none)",
                                        min_value=0, max_value=3000, value=1000, step=50)
        year_built = st.number_input("Year built", min_value=1870,
                                     max_value=VALUATION_YEAR, value=1990)
    with col2:
        quality = st.slider("Overall material & finish quality (1–10)", 1, 10, 5)
        baths = st.number_input("Bathrooms (halves count 0.5)", min_value=1.0,
                                max_value=6.0, value=2.0, step=0.5)
        garage_cars = st.number_input("Garage capacity (cars)", min_value=0,
                                      max_value=4, value=2)
        dist_school = st.number_input("Distance to nearest school (km)",
                                      min_value=0.1, max_value=10.0, value=1.5, step=0.1)

    with st.expander("More details (optional)"):
        condition = st.slider("Overall condition (1–10)", 1, 10, 5)
        lot_area = st.number_input("Lot area (sq ft)", min_value=1000,
                                   max_value=60000, value=9500, step=250)
        kitchen = st.select_slider("Kitchen quality", KITCHEN_QUAL, value="TA")
        central_air = st.checkbox("Central air conditioning", value=True)
        remod_year = st.number_input(
            "Last remodel year (= year built if never remodeled)",
            min_value=1870, max_value=VALUATION_YEAR, value=1990)
        reno_cost = st.number_input("Renovation spend if remodeled (USD)",
                                    min_value=0, max_value=500_000, value=0,
                                    step=5000)

    submitted = st.form_submit_button("Estimate value", type="primary",
                                      use_container_width=True)

if submitted:
    result = service.predict({
        "Neighborhood": neighborhood,
        "GrLivArea": living_area,
        "TotalBsmtSF": basement_area,
        "OverallQual": quality,
        "OverallCond": condition,
        "property_age": max(0, VALUATION_YEAR - year_built),
        "years_since_remodel": max(0, VALUATION_YEAR - max(remod_year, year_built)),
        "total_baths": baths,
        "GarageCars": garage_cars,
        "dist_school_km": dist_school,
        "LotArea": lot_area,
        "KitchenQual": kitchen,
        "CentralAir": "Y" if central_air else "N",
        "renovation_cost_usd": reno_cost,
    })

    st.divider()
    st.subheader(f"Estimated value: ${result['estimate_usd']:,.0f}")
    lo, hi = result["range_low_usd"], result["range_high_usd"]
    # dollar signs escaped: st.markdown treats $...$ as LaTeX math
    st.markdown(f"**80% confidence range: \\${lo:,.0f} – \\${hi:,.0f}**")

    # visualize where the estimate sits inside its range
    span = hi - lo
    st.progress(min(1.0, max(0.0, (result["estimate_usd"] - lo) / span)) if span else 0.5)

    width = result["relative_width_pct"]
    if width < 25:
        confidence_note = ("This is a **common home profile** for the market — "
                           "the model has seen many similar sales, so the range is tight.")
    elif width < 40:
        confidence_note = "Typical estimation uncertainty for this market."
    else:
        confidence_note = ("This home is **unusual or high-value** relative to the "
                           "training data — the honest range is wide. Consider a "
                           "professional appraisal before pricing.")
    st.caption(
        f"Range width: ±{width / 2:.0f}% of the estimate. {confidence_note} "
        "The range is calibrated: verified on held-out sales to contain the "
        "true price ~80% of the time."
    )
    st.caption("Values expressed at the 2010 market level (end of training window).")
