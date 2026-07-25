"""Module 4 (features) — build the modeling table. Run from the repo root:

    python -m src.features.build_model_table

Reads data/processed/listings_clean.csv, engineers derived features, selects
the model feature set, runs a VIF multicollinearity check, and writes
data/processed/model_table.csv (features + target + time split flag).

Feature-selection principles (defended in DECISIONS.md):
- Only information available AT VALUATION TIME enters the model. Excluded:
  days_on_market (an outcome of the listing, unknown when the home is priced)
  and market_price_index (a citywide index is published with a lag — using the
  current month's value would be look-ahead leakage).
- One variable per concept: GarageCars not GarageArea (r=0.88), GrLivArea +
  TotalBsmtSF not TotRmsAbvGrd (r=0.83), bath counts combined into one.
"""

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, MONITORING_HOLDOUT_START

# The model's contract: exactly these columns, in this role.
NUMERIC_FEATURES = [
    "OverallQual", "OverallCond", "GrLivArea", "TotalBsmtSF", "LotArea",
    "property_age", "years_since_remodel", "total_baths", "GarageCars",
    "Fireplaces", "dist_school_km", "dist_hospital_km", "dist_transit_km",
    "renovation_cost_usd", "local_interest_rate_pct", "listing_month",
]
CATEGORICAL_FEATURES = [
    "Neighborhood", "BldgType", "HouseStyle", "ExterQual", "KitchenQual",
    "CentralAir",
]
TARGET = "sale_price"
MONITORING_TARGET = "sale_price_market_adjusted"


def _amenity_score_from_row(df: pd.DataFrame) -> pd.Series:
    """Recompute amenity_score after cleaning (distances may have been fixed)."""
    prox = (
        50.0 / (1.0 + df["dist_school_km"])
        + 30.0 / (1.0 + df["dist_hospital_km"])
        + 20.0 / (1.0 + df["dist_transit_km"])
    )
    cond1 = df["Condition1"].astype(str)
    cond2 = df["Condition2"].astype(str)
    cond_bonus = np.where(cond1.isin({"PosN", "PosA"}) | cond2.isin({"PosN", "PosA"}), 10.0, 0.0)
    cond_bonus = cond_bonus + np.where(
        cond1.isin({"Artery", "Feedr"}) | cond2.isin({"Artery", "Feedr"}), 5.0, 0.0
    )
    zone_bonus = {
        "RH": 10, "RM": 8, "FV": 7, "RL": 3, "RP": 3,
        "C (all)": 5, "C": 5, "A (agr)": 0, "A": 0, "I": 1, "I (all)": 1,
    }
    z = df["MSZoning"].astype(str).map(zone_bonus).fillna(0.0).to_numpy()
    return pd.Series(np.clip(prox + cond_bonus + z, 0, 100).round(1), index=df.index)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns (brief M4: age, renovation flag, amenity score)."""
    out = df.copy()
    out["property_age"] = (out["YrSold"] - out["YearBuilt"]).clip(lower=0)
    out["years_since_remodel"] = (out["YrSold"] - out["YearRemodAdd"]).clip(lower=0)
    out["total_baths"] = (out["FullBath"] + 0.5 * out["HalfBath"]
                          + out["BsmtFullBath"] + 0.5 * out["BsmtHalfBath"])
    out["is_renovated"] = (out["renovated"] == "Yes").astype(int)
    out["renovation_cost_usd"] = out["renovation_cost_usd"].fillna(0)
    out["listing_month"] = out["listing_date"].dt.month
    # amenity_score: hybrid proximity + Condition + Zoning; EDA only (not in
    # NUMERIC_FEATURES — collinear with the three distances already modeled)
    out["amenity_score"] = _amenity_score_from_row(out)
    return out


def vif_table(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Variance inflation factors: VIF_i = 1 / (1 - R²) of feature i regressed
    on all others. > 10 signals problematic multicollinearity."""
    X = df[cols].to_numpy(dtype=float)
    X = (X - X.mean(0)) / X.std(0)
    vifs = {}
    for i, col in enumerate(cols):
        y, others = X[:, i], np.delete(X, i, axis=1)
        beta, *_ = np.linalg.lstsq(others, y, rcond=None)
        r2 = 1 - ((y - others @ beta) ** 2).sum() / (y ** 2).sum()
        vifs[col] = 1 / max(1 - r2, 1e-9)
    return pd.Series(vifs).sort_values(ascending=False)


def main() -> None:
    df = pd.read_csv(DATA_PROCESSED_DIR / "listings_clean.csv",
                     parse_dates=["listing_date", "sale_date"])
    df = engineer_features(df)

    vifs = vif_table(df, NUMERIC_FEATURES)
    print("VIF (numeric model features):")
    print(vifs.round(2).to_string())
    worst = vifs.iloc[0]
    assert worst < 10, f"multicollinearity regression: max VIF {worst:.1f}"
    print(f"max VIF = {worst:.2f} < 10 — feature set is acceptable for linear models\n")

    keep = (["Id", "listing_id", "listing_date", "sale_date",
             "is_renovated", "amenity_score"]
            + NUMERIC_FEATURES + CATEGORICAL_FEATURES
            + [TARGET, MONITORING_TARGET])
    table = df[keep].copy()
    table["is_monitoring_stream"] = (
        table["sale_date"] >= MONITORING_HOLDOUT_START).astype(int)

    table.to_csv(DATA_PROCESSED_DIR / "model_table.csv", index=False)
    n_stream = table["is_monitoring_stream"].sum()
    print(f"model_table.csv: {len(table)} rows "
          f"({len(table) - n_stream} training window, {n_stream} monitoring stream), "
          f"{len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical features")


if __name__ == "__main__":
    main()
