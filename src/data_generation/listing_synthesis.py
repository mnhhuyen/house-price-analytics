"""Synthesize per-listing contextual fields on top of the Ames base data.

Amenity generation combines three methods (see neighborhood_profiles.py):
  1) fictional map distances
  2) neighborhood access priors (zoning + lot density + hand-crafted urban form)
  3) listing-level Condition1/2 nudges

Leakage discipline: never derive amenity distances or days-on-market from
SalePrice. The primary target remains the original Kaggle SalePrice. A
separate market-adjusted label exists only for the monitoring simulation.
"""

import numpy as np
import pandas as pd

from src.config import MONITORING_HOLDOUT_START, RANDOM_SEED
from src.data_generation.neighborhood_profiles import build_neighborhood_profiles

_NEAR_POSITIVE = {"PosN", "PosA"}
_NEAR_ARTERY = {"Artery", "Feedr"}


def _sale_dates(ames: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Create a sale-date proxy inside the real Ames sale month."""
    day = rng.integers(1, 28, len(ames))
    return pd.to_datetime(dict(year=ames["YrSold"], month=ames["MoSold"], day=day))


def _amenity_distances(ames: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Home-level distances: map × neighborhood prior × noise × Condition nudge."""
    profiles = build_neighborhood_profiles(ames)
    merged = ames[["Neighborhood"]].merge(profiles, on="Neighborhood", how="left")
    out = pd.DataFrame(index=ames.index)

    cond1 = ames["Condition1"].astype(str)
    cond2 = ames["Condition2"].astype(str)
    near_pos = cond1.isin(_NEAR_POSITIVE) | cond2.isin(_NEAR_POSITIVE)
    near_road = cond1.isin(_NEAR_ARTERY) | cond2.isin(_NEAR_ARTERY)

    for amenity in ("school", "hospital", "transit"):
        base = merged[f"base_dist_{amenity}_km"].to_numpy(dtype=float)
        dist = base * rng.lognormal(0, 0.18, len(ames))
        if amenity == "school":
            dist = np.where(near_pos, dist * 0.85, dist)
        elif amenity == "transit":
            dist = np.where(near_road, dist * 0.80, dist)
        out[f"dist_{amenity}_km"] = np.maximum(0.1, dist).round(2)

    return out


def _renovation_fields(ames: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Renovation flag + cost from YearRemodAdd / size / quality — not price."""
    renovated = ames["YearRemodAdd"] > ames["YearBuilt"]
    cost_per_sqft = 12 + 3.2 * ames["OverallQual"]
    cost = ames["GrLivArea"] * cost_per_sqft * rng.lognormal(0, 0.30, len(ames))
    return pd.DataFrame({
        "renovated": np.where(renovated, "Yes", "No"),
        "renovation_cost_usd": np.where(renovated, cost.round(-2), np.nan),
    }, index=ames.index)


def _days_on_market(ames: pd.DataFrame, listing_month: pd.Series,
                    macro: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """DOM from market heat + unusual size/quality vs neighborhood — not SalePrice."""
    nbhd_area = ames.groupby("Neighborhood")["GrLivArea"].transform("median")
    nbhd_qual = ames.groupby("Neighborhood")["OverallQual"].transform("median")
    unusual = (
        (ames["GrLivArea"] / nbhd_area).clip(0.6, 2.0)
        * (ames["OverallQual"] / nbhd_qual).clip(0.6, 2.0)
    )

    idx = macro.set_index("month")["market_price_index"]
    trend = idx.rolling(6, min_periods=1).mean()
    cold = (trend / idx).reindex(listing_month).to_numpy()

    dom = 35 * unusual.to_numpy() ** 1.2 * cold ** 6
    dom *= rng.lognormal(0, 0.35, len(ames))
    return pd.Series(np.maximum(3, dom).round().astype(int), index=ames.index)


def _market_adjusted_price(
    ames: pd.DataFrame,
    sale_month: pd.Series,
    macro: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.Series:
    """Synthetic label used only to replay the 2010 recovery in monitoring.

    Before the holdout it equals the original target. From 2010-01 onward it
    compounds the price-index recovery relative to the first holdout month.
    """
    idx = macro.set_index("month")["market_price_index"]
    holdout_month = str(pd.Period(MONITORING_HOLDOUT_START, freq="M"))
    in_stream = sale_month >= holdout_month
    factor = np.ones(len(ames), dtype=float)
    factor[in_stream.to_numpy()] = (
        idx.reindex(sale_month[in_stream]).to_numpy() / idx.loc[holdout_month]
    )
    noise = np.ones(len(ames), dtype=float)
    noise[in_stream.to_numpy()] = rng.lognormal(0, 0.02, int(in_stream.sum()))
    adjusted = ames["SalePrice"].astype(int).copy()
    adjusted.loc[in_stream] = (
        ames.loc[in_stream, "SalePrice"]
        * factor[in_stream.to_numpy()]
        * noise[in_stream.to_numpy()]
    ).round(-2).astype(int)
    return adjusted


def synthesize_listing_fields(ames: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Return the full synthetic extension, one row per Ames home (keyed by Id)."""
    rng = np.random.default_rng(RANDOM_SEED + 2)

    out = pd.DataFrame({"Id": ames["Id"]})
    out["listing_id"] = "L-" + ames["Id"].astype(str).str.zfill(6)
    out["sale_date"] = _sale_dates(ames, rng)
    sale_month = out["sale_date"].dt.to_period("M").astype(str)

    out = pd.concat([out, _amenity_distances(ames, rng),
                     _renovation_fields(ames, rng)], axis=1)
    out["days_on_market"] = _days_on_market(ames, sale_month, macro, rng)
    out["listing_date"] = (
        out["sale_date"] - pd.to_timedelta(out["days_on_market"], unit="D")
    )

    # Macro context is observed at listing time. A few early listings can
    # predate the synthetic macro axis; use the earliest available month.
    listing_period = out["listing_date"].dt.to_period("M")
    macro_periods = pd.PeriodIndex(macro["month"], freq="M")
    listing_period = listing_period.where(
        listing_period >= macro_periods.min(), macro_periods.min()
    ).where(listing_period <= macro_periods.max(), macro_periods.max())
    listing_month = listing_period.astype(str)
    macro_by_month = macro.set_index("month")
    out["local_interest_rate_pct"] = (
        macro_by_month["local_interest_rate_pct"].reindex(listing_month).to_numpy()
    )
    out["market_price_index"] = (
        macro_by_month["market_price_index"].reindex(listing_month).to_numpy()
    )
    out["sale_price"] = ames["SalePrice"].astype(int)
    out["sale_price_market_adjusted"] = _market_adjusted_price(
        ames, sale_month, macro, rng
    )
    out["sale_price_kaggle"] = ames["SalePrice"]
    return out
