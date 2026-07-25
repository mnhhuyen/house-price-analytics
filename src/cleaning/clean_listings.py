"""Module 3 — data cleaning. Run from the repo root:

    python -m src.cleaning.clean_listings

Reads data/synthetic/ames_extended.csv, fixes every documented data-quality
problem (DATA_DICTIONARY.md "Injected data-quality problems" + Kaggle-side
issues found in EDA), writes data/processed/listings_clean.csv and prints a
before/after evidence table (the course requires documented cleaning
decisions with before/after comparisons).

Every fix targets a diagnosed cause — nothing is dropped or imputed blindly.
"""

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR, DATA_SYNTHETIC_DIR

# Kaggle categoricals where NA means "the home has no such feature" — a valid
# category, not missing data (per data_description.txt)
_NA_MEANS_ABSENT = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
    "GarageCond", "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]

# Known partial/abnormal sales (huge area, far-below-trend price) — the two
# canonical Ames outliers identified in the EDA scatter (Module 2, fig 04)
_PARTIAL_SALE_IDS = [524, 1299]


def _log(step: str, before, after, unit: str) -> dict:
    print(f"  {step:<58} {before:>6} -> {after:<6} {unit}")
    return {"step": step, "before": before, "after": after, "unit": unit}


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    evidence = []
    print("Cleaning steps (before -> after):")

    # 1. Duplicate listings: same home re-posted under a new listing_id.
    # Keep the FIRST posting (the original listing event).
    n0 = df["Id"].duplicated().sum()
    df = df.sort_values("listing_date").drop_duplicates("Id", keep="first")
    evidence.append(_log("duplicate listings removed (same Id, keep earliest)",
                         n0, df["Id"].duplicated().sum(), "dupes"))

    # 2. Renovated labels: free-text variants -> canonical Yes/No
    n0 = (~df["renovated"].isin(["Yes", "No"])).sum()
    df["renovated"] = df["renovated"].str.strip().str[0].str.upper().map(
        {"Y": "Yes", "N": "No"})
    evidence.append(_log("renovated labels normalized to Yes/No",
                         n0, (~df["renovated"].isin(["Yes", "No"])).sum(), "messy"))

    # 3. Days-on-market sentinel 999 = "unknown" -> impute median DOM of the
    # listing quarter (market conditions drive DOM, so impute within-period)
    n0 = (df["days_on_market"] == 999).sum()
    df.loc[df["days_on_market"] == 999, "days_on_market"] = np.nan
    quarter = df["listing_date"].dt.to_period("Q")
    df["days_on_market"] = df["days_on_market"].fillna(
        df.groupby(quarter)["days_on_market"].transform("median")).round()
    # Keep date arithmetic consistent after replacing the sentinel.
    df["listing_date"] = (
        df["sale_date"] - pd.to_timedelta(df["days_on_market"], unit="D")
    )
    evidence.append(_log("DOM sentinel 999 -> quarter-median impute",
                         n0, (df["days_on_market"] == 999).sum(), "rows"))

    # 4. Transit distances entered in meters (values implausibly > 100 km in
    # a town ~10 km across) -> convert back to km
    n0 = (df["dist_transit_km"] > 100).sum()
    df.loc[df["dist_transit_km"] > 100, "dist_transit_km"] /= 1000
    evidence.append(_log("transit meter-entry errors converted to km",
                         n0, (df["dist_transit_km"] > 100).sum(), "rows"))

    # 5. Missing school distance -> neighborhood median (amenity distance is
    # a neighborhood property by construction, so this is the natural fill)
    n0 = df["dist_school_km"].isna().sum()
    df["dist_school_km"] = df["dist_school_km"].fillna(
        df.groupby("Neighborhood")["dist_school_km"].transform("median"))
    evidence.append(_log("missing school distance -> neighborhood median",
                         n0, df["dist_school_km"].isna().sum(), "rows"))

    # 6. Kaggle NA-means-absent categoricals -> explicit "NoFeature" category.
    # ("None" would round-trip back to NaN: pandas.read_csv treats the string
    # "None" as a missing-value marker by default.)
    n0 = int(df[_NA_MEANS_ABSENT].isna().sum().sum())
    df[_NA_MEANS_ABSENT] = df[_NA_MEANS_ABSENT].fillna("NoFeature")
    evidence.append(_log('Kaggle "no such feature" NAs -> "NoFeature" category',
                         n0, int(df[_NA_MEANS_ABSENT].isna().sum().sum()), "cells"))

    # 7. True Kaggle missing values: LotFrontage by neighborhood median
    # (street frontage is set by the subdivision plan); small numeric gaps
    # (GarageYrBlt when no garage, MasVnrArea) -> 0
    n0 = int(df["LotFrontage"].isna().sum())
    df["LotFrontage"] = df["LotFrontage"].fillna(
        df.groupby("Neighborhood")["LotFrontage"].transform("median"))
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
    evidence.append(_log("LotFrontage -> neighborhood median (+small fills)",
                         n0, int(df["LotFrontage"].isna().sum()), "rows"))

    # 8. Partial-sale outliers: two homes > 4000 sqft sold far below trend —
    # not market transactions a valuation tool should learn from
    n0 = len(df)
    df = df[~df["Id"].isin(_PARTIAL_SALE_IDS)]
    evidence.append(_log("partial-sale outliers dropped (Ids 524, 1299)",
                         n0, len(df), "rows"))

    return df.reset_index(drop=True), evidence


def main() -> None:
    df = pd.read_csv(DATA_SYNTHETIC_DIR / "ames_extended.csv",
                     parse_dates=["listing_date", "sale_date"])
    n_in = len(df)
    df, _ = clean(df)

    assert df["renovated"].isin(["Yes", "No"]).all()
    assert not df["Id"].duplicated().any()

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED_DIR / "listings_clean.csv", index=False)
    # the only NAs allowed out of this stage are renovation_cost_usd for
    # never-renovated homes — semantically "no renovation", filled with 0 in
    # feature engineering
    leftover = df.drop(columns=["renovation_cost_usd"]).isna().sum()
    assert int(leftover.sum()) == 0, f"unexpected NAs remain: {leftover[leftover > 0]}"
    print(f"\n{n_in} rows in -> {len(df)} rows out; only by-design NAs remain "
          f"(renovation_cost_usd for {int(df.renovation_cost_usd.isna().sum())} never-renovated homes)")
    print(f"wrote {DATA_PROCESSED_DIR / 'listings_clean.csv'}")


if __name__ == "__main__":
    main()
