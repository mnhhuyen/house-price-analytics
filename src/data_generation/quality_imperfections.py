"""Deliberate, seeded, documented data-quality problems (DECISIONS.md D7).

The Kaggle base data is fairly clean, so Module 3 (cleaning) would have
little real work. We therefore simulate the data-quality problems a real
listing platform faces — but ONLY in our own synthetic fields, never in the
Kaggle columns, and only in rows BEFORE the monitoring holdout (the incoming
stream must stay clean so drift signals aren't confused with dirt).

Every injection is listed in DATA_DICTIONARY.md with its exact rate, and the
cleaning module must detect and fix each one with before/after evidence.
"""

import numpy as np
import pandas as pd

from src.config import MONITORING_HOLDOUT_START, RANDOM_SEED

# (injection, rate) — rates chosen to be realistic annoyances, not data chaos
MISSING_SCHOOL_DIST_RATE = 0.02      # field left blank at listing entry
MESSY_RENOVATED_LABEL_RATE = 0.04    # free-text entry: Y / yes / NO / n ...
DOM_SENTINEL_RATE = 0.004            # legacy system exports 999 = "unknown"
TRANSIT_METERS_RATE = 0.003          # distance entered in meters, not km
N_DUPLICATE_LISTINGS = 12            # same home re-posted under a new listing id

_MESSY_LABELS = {"Yes": ["Y", "yes", "YES"], "No": ["N", "no", "NO"]}


def inject_imperfections(listings: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the synthetic listings with seeded dirt injected."""
    rng = np.random.default_rng(RANDOM_SEED + 3)
    df = listings.copy()

    # dirt only before the monitoring stream
    eligible = df.index[df["sale_date"] < pd.Timestamp(MONITORING_HOLDOUT_START)]

    def _sample(rate_or_n) -> np.ndarray:
        n = rate_or_n if isinstance(rate_or_n, int) else round(len(eligible) * rate_or_n)
        return rng.choice(eligible, size=n, replace=False)

    # 1. missing school distance
    df.loc[_sample(MISSING_SCHOOL_DIST_RATE), "dist_school_km"] = np.nan

    # 2. inconsistent renovated labels
    for row in _sample(MESSY_RENOVATED_LABEL_RATE):
        clean = df.at[row, "renovated"]
        df.at[row, "renovated"] = rng.choice(_MESSY_LABELS[clean])

    # 3. days-on-market sentinel value
    df.loc[_sample(DOM_SENTINEL_RATE), "days_on_market"] = 999

    # 4. transit distance accidentally entered in meters
    meters_rows = _sample(TRANSIT_METERS_RATE)
    df.loc[meters_rows, "dist_transit_km"] = (
        df.loc[meters_rows, "dist_transit_km"] * 1000
    ).round(0)

    # 5. duplicated listings (same home, new listing_id, a few days later)
    dup_rows = df.loc[_sample(N_DUPLICATE_LISTINGS)].copy()
    dup_rows["listing_id"] = "L-9" + dup_rows["Id"].astype(str).str.zfill(5)
    dup_rows["listing_date"] += pd.to_timedelta(
        rng.integers(1, 3, len(dup_rows)), unit="D"
    )
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df.sort_values("listing_date").reset_index(drop=True)
