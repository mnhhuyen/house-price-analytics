"""Generate the full synthetic data extension. Run from the repo root:

    python -m src.data_generation.generate

Outputs (all committed for reproducibility; identical on every run — seeded):
    data/synthetic/macro_monthly.csv      one row per month, 2006-01..2010-07
    data/synthetic/amenity_hubs.csv       fictional map hub coordinates
    data/synthetic/neighborhood_profiles.csv map distances + access priors
    data/synthetic/listings_synthetic.csv per-listing synthetic fields (with
                                          injected dirt), keyed to Ames by Id
    data/synthetic/ames_extended.csv      Kaggle train.csv joined with the
                                          synthetic fields — the single input
                                          for all downstream modules
"""

import pandas as pd

from src.config import DATA_RAW_DIR, DATA_SYNTHETIC_DIR, MONITORING_HOLDOUT_START
from src.data_generation.listing_synthesis import synthesize_listing_fields
from src.data_generation.macro_series import build_macro_series
from src.data_generation.neighborhood_profiles import (
    build_amenity_hubs,
    build_neighborhood_profiles,
)
from src.data_generation.quality_imperfections import inject_imperfections

_REQUIRED_COLUMNS = {
    "Id", "SalePrice", "Neighborhood", "MSZoning", "LotArea",
    "Condition1", "Condition2", "YrSold", "MoSold", "YearBuilt",
    "YearRemodAdd", "GrLivArea", "OverallQual",
}


def main() -> None:
    ames = pd.read_csv(DATA_RAW_DIR / "train.csv")
    missing = sorted(_REQUIRED_COLUMNS - set(ames.columns))
    if missing:
        raise ValueError(f"train.csv missing required columns: {missing}")
    if ames["Id"].isna().any() or not ames["Id"].is_unique:
        raise ValueError("raw Kaggle Id must be non-null and unique")

    macro = build_macro_series()
    hubs = build_amenity_hubs()
    profiles = build_neighborhood_profiles(ames, hubs)
    listings = synthesize_listing_fields(ames, macro)
    listings = inject_imperfections(listings)

    extended = listings.merge(ames.drop(columns=["SalePrice"]), on="Id", how="left")
    if extended[["local_interest_rate_pct", "market_price_index"]].isna().any().any():
        raise ValueError("macro join produced missing values")
    if not (extended["listing_date"] < extended["sale_date"]).all():
        raise ValueError("every listing_date must precede sale_date")

    DATA_SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    macro.to_csv(DATA_SYNTHETIC_DIR / "macro_monthly.csv", index=False)
    hubs.to_csv(DATA_SYNTHETIC_DIR / "amenity_hubs.csv", index=False)
    profiles.to_csv(DATA_SYNTHETIC_DIR / "neighborhood_profiles.csv", index=False)
    listings.to_csv(DATA_SYNTHETIC_DIR / "listings_synthetic.csv", index=False)
    extended.to_csv(DATA_SYNTHETIC_DIR / "ames_extended.csv", index=False)

    stream = extended["sale_date"] >= MONITORING_HOLDOUT_START
    print(f"listings: {len(extended)} ({len(extended) - len(ames)} injected duplicates)")
    print(f"training window: {(~stream).sum()} rows | monitoring stream: {stream.sum()} rows")
    print(f"sale dates: {extended['sale_date'].min()} .. {extended['sale_date'].max()}")
    print(f"listing dates: {extended['listing_date'].min()} .. {extended['listing_date'].max()}")
    print(f"wrote 5 files to {DATA_SYNTHETIC_DIR}")


if __name__ == "__main__":
    main()
