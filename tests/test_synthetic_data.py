"""Invariant tests for the generated synthetic data.

These encode the claims made in DATA_DICTIONARY.md — if a change to the
generator breaks a documented property (leakage discipline, dirt placement,
clean monitoring stream), a test fails rather than the claim silently rotting.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_RAW_DIR, DATA_SYNTHETIC_DIR, MONITORING_HOLDOUT_START
from src.data_generation.listing_synthesis import _days_on_market
from src.data_generation.macro_series import build_macro_series
from src.data_generation.neighborhood_profiles import build_neighborhood_profiles


def _load() -> pd.DataFrame:
    return pd.read_csv(
        DATA_SYNTHETIC_DIR / "ames_extended.csv",
        parse_dates=["listing_date", "sale_date"],
    )


def test_time_axis_matches_real_ames_sale_dates():
    df = _load()
    assert (df["sale_date"].dt.year == df["YrSold"]).all()
    assert (df["sale_date"].dt.month == df["MoSold"]).all()
    assert (df["listing_date"] < df["sale_date"]).all()
    elapsed = (df["sale_date"] - df["listing_date"]).dt.days
    untouched = (df["days_on_market"] != 999) & ~df["listing_id"].str.startswith("L-9")
    assert (elapsed[untouched] == df.loc[untouched, "days_on_market"]).all()


def test_monitoring_stream_exists_and_is_clean():
    df = _load()
    stream = df[df["sale_date"] >= MONITORING_HOLDOUT_START]
    assert len(stream) >= 150  # enough rows for monthly rolling metrics
    # no injected dirt in the stream: drift must not be confused with dirt
    assert stream["dist_school_km"].notna().all()
    assert stream["renovated"].isin(["Yes", "No"]).all()
    assert (stream["days_on_market"] != 999).all()
    assert (stream["dist_transit_km"] < 100).all()
    assert not stream["Id"].duplicated().any()


def test_documented_dirt_is_present_in_training_window():
    df = _load()
    train = df[df["sale_date"] < MONITORING_HOLDOUT_START]
    assert train["dist_school_km"].isna().sum() > 0
    assert not train["renovated"].isin(["Yes", "No"]).all()  # messy labels exist
    assert (train["days_on_market"] == 999).sum() > 0
    assert (train["dist_transit_km"] > 100).sum() > 0  # meter-unit errors
    assert train["Id"].duplicated().sum() > 0  # duplicate listings


def test_amenity_profiles_independent_of_saleprice():
    """Shuffling SalePrice must not change neighborhood amenity bases."""
    ames = pd.read_csv(DATA_RAW_DIR / "train.csv")
    p1 = build_neighborhood_profiles(ames)
    shuffled = ames.copy()
    shuffled["SalePrice"] = shuffled["SalePrice"].sample(
        frac=1, random_state=0
    ).to_numpy()
    p2 = build_neighborhood_profiles(shuffled)
    cols = [
        "Neighborhood", "base_dist_school_km", "base_dist_hospital_km",
        "base_dist_transit_km", "access_multiplier",
    ]
    pd.testing.assert_frame_equal(
        p1[cols].sort_values("Neighborhood").reset_index(drop=True),
        p2[cols].sort_values("Neighborhood").reset_index(drop=True),
    )


def test_days_on_market_independent_of_saleprice():
    """DOM uses size/quality + market heat — not the eventual sale price."""
    ames = pd.read_csv(DATA_RAW_DIR / "train.csv")
    macro = build_macro_series()
    # listing months from real sale dates (same construction as synthesis)
    listing_month = (
        pd.to_datetime(dict(year=ames["YrSold"], month=ames["MoSold"], day=1))
        .dt.to_period("M")
        .astype(str)
    )
    import numpy as np
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    d1 = _days_on_market(ames, listing_month, macro, rng1)
    shuffled = ames.copy()
    shuffled["SalePrice"] = shuffled["SalePrice"].sample(
        frac=1, random_state=1
    ).to_numpy()
    d2 = _days_on_market(shuffled, listing_month, macro, rng2)
    pd.testing.assert_series_equal(d1, d2)


def test_amenity_map_artifacts_exist():
    hubs = pd.read_csv(DATA_SYNTHETIC_DIR / "amenity_hubs.csv")
    profiles = pd.read_csv(DATA_SYNTHETIC_DIR / "neighborhood_profiles.csv")
    assert set(hubs["amenity_type"]) == {"school", "hospital", "transit"}
    assert len(profiles) == 25
    assert profiles["Neighborhood"].is_unique


def test_amenity_distances_in_range_and_coherent_by_neighborhood():
    df = _load()
    ok = df["dist_transit_km"] < 100  # exclude meter-unit dirt
    clean = df.loc[ok & df["dist_school_km"].notna()]
    assert clean["dist_school_km"].between(0.1, 20).all()
    assert clean["dist_hospital_km"].between(0.1, 25).all()
    assert clean["dist_transit_km"].between(0.1, 20).all()
    within = clean.groupby("Neighborhood")["dist_school_km"].std().median()
    overall = clean["dist_school_km"].std()
    assert within < overall


def test_primary_target_is_original_and_scenario_carries_market_trend():
    df = _load()
    assert (df["sale_price"] == df["sale_price_kaggle"]).all()
    before = df["sale_date"] < MONITORING_HOLDOUT_START
    assert (
        df.loc[before, "sale_price_market_adjusted"]
        == df.loc[before, "sale_price"]
    ).all()
    factor = df["sale_price_market_adjusted"] / df["sale_price_kaggle"]
    stream = df.loc[~before].copy()
    by_month = factor.loc[~before].groupby(
        stream["sale_date"].dt.to_period("M")
    ).mean()
    assert by_month.iloc[-1] > by_month.iloc[0]  # rebound within 2010 stream


def test_renovation_consistent_with_kaggle_remodel_column():
    df = _load()
    clean = df[df["renovated"].isin(["Yes", "No"])]
    remodeled = clean["YearRemodAdd"] > clean["YearBuilt"]
    assert ((clean["renovated"] == "Yes") == remodeled).all()
    # cost present iff renovated
    assert clean.loc[clean["renovated"] == "Yes", "renovation_cost_usd"].notna().all()
    assert clean.loc[clean["renovated"] == "No", "renovation_cost_usd"].isna().all()
