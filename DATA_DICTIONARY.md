# Data Dictionary — Synthetic Fields

Every synthetically generated field, per the course requirement: column name,
data type, unit, valid range, and the generation logic / business assumption.
The base Kaggle (Ames) fields are documented by Kaggle's own
`data/raw/data_description.txt` and are not duplicated here.

**Reproducibility:** all generation is seeded (`RANDOM_SEED = 42`,
`src/config.py`). `python -m src.data_generation.generate` reproduces the
identical files byte-for-byte (verified).

**Time axis:** `sale_date` is anchored to the *real* Ames sale month
(`YrSold`/`MoSold`, Jan 2006 – Jul 2010). `listing_date` is then calculated as
`sale_date - days_on_market`. All sales in 2010
(Jan–Jul, 175 rows) are the **monitoring holdout stream** — never used in
training, replayed month-by-month in Module 7.

## Files

| File | Grain | Contents |
|---|---|---|
| `data/synthetic/macro_monthly.csv` | 1 row / month | Synthetic macro series |
| `data/synthetic/amenity_hubs.csv` | 1 row / hub | Fictional school/hospital/transit coordinates |
| `data/synthetic/neighborhood_profiles.csv` | 1 row / neighborhood | Map distances, zoning/lot priors, final base distances |
| `data/synthetic/listings_synthetic.csv` | 1 row / listing | Synthetic fields only, keyed to Ames by `Id` |
| `data/synthetic/ames_extended.csv` | 1 row / listing | Kaggle columns + synthetic fields — input for all downstream modules |

## Listing-level fields

| Column | Type | Unit | Valid range | Generation logic / business assumption |
|---|---|---|---|---|
| `listing_id` | str | — | `L-` + 6 digits | Platform listing key. Injected duplicates use prefix `L-9`. |
| `sale_date` | date | — | 2006-01-01 … 2010-07-27 | Sale-date proxy: real Ames sale month (`YrSold`/`MoSold`) + seeded day 1–27. Used for train/monitoring split. |
| `listing_date` | date | — | Before `sale_date` | `sale_date - days_on_market`. This is the valuation-time date; macro context is joined using its month (early dates before the macro axis use the first available month). |
| `dist_school_km` | float | km | 0.1 – ~15 | **Hybrid, leakage-safe.** (1) Fictional map: seeded centroids + school hubs on a ~10×10 km plane → Euclidean nearest-hub distance. (2) Neighborhood prior multiplies map distance: modal `MSZoning` + median `LotArea` density + hand-crafted urban-form priors for selected neighborhoods (OldTown/SWISU closer fabric; ClearCr/Timber farther) — **never `SalePrice`**. (3) Homes with `Condition1/2 ∈ {PosN, PosA}` get a 15% closer nudge. Per-home lognormal scatter (σ=0.18). |
| `dist_hospital_km` | float | km | 0.1 – ~20 | Same hybrid as school (map × neighborhood prior × scatter). |
| `dist_transit_km` | float | km | 0.1 – ~15 | Same hybrid; `Condition1/2 ∈ {Artery, Feedr}` → 20% closer nudge. Transit hubs placed independently of schools/hospitals. |
| `renovated` | str | — | Yes / No | `Yes` iff Kaggle `YearRemodAdd` > `YearBuilt` — consistent with the real remodel column by construction. |
| `renovation_cost_usd` | float | USD | ~$15k – $400k; NaN if not renovated | Living area × ($12 + $3.2 × OverallQual)/sqft × lognormal(σ=0.30), rounded to $100. Uses size/quality only — not price. **Assumption:** remodel cost scales with size and finish quality (~$15–44/sqft). |
| `days_on_market` | int | days | 3 – ~250 | 35 × (size/quality vs neighborhood median)^1.2 × market-coldness^6 × lognormal(σ=0.35). **Leakage-safe:** unusual homes use `GrLivArea`/`OverallQual` vs neighborhood medians — **not** `SalePrice`. Cold markets (index below 6-month trend) lengthen DOM (2008–09). Already excluded from the model (outcome unknown at pricing time). |
| `local_interest_rate_pct` | float | % p.a. | 4.7 – 6.7 | Monthly macro series joined by listing month (see below). |
| `market_price_index` | float | index, Jan 2006 = 100 | ~94 – 106 | Monthly macro series joined by listing month (see below). |
| `sale_price` | int | USD | 34,900 – 755,000 | **Primary modeling target:** original Kaggle `SalePrice`, unchanged. All model comparison/calibration metrics use this field. |
| `sale_price_market_adjusted` | int | USD | ~$34k – $830k | **Monitoring-scenario label only:** equals original `SalePrice` before 2010; in the holdout it is `SalePrice × (index at sale month ÷ index at 2010-01) × lognormal(σ=0.02)`, rounded to $100. This isolates a progressive 2010 recovery and is never a model feature or training target. |
| `sale_price_kaggle` | int | USD | 34,900 – 755,000 | Audit copy of original Kaggle `SalePrice`; exactly equal to primary `sale_price`. |

## Derived after cleaning

| Column | Type | Unit | Valid range | Generation logic / business assumption |
|---|---|---|---|---|
| `amenity_score` | float | score | 0 – 100 | Computed only after distance cleaning: `50/(1+school)+30/(1+hospital)+20/(1+transit)` + Condition bonus (+10 near PosN/PosA, +5 near Artery/Feedr) + zoning bonus. EDA only; not a model input because the three distances are already included. |

## Macro series (`macro_monthly.csv`)

| Column | Type | Unit | Valid range | Generation logic / business assumption |
|---|---|---|---|---|
| `month` | str | YYYY-MM | 2006-01 … 2010-07 | Calendar month. |
| `local_interest_rate_pct` | float | % p.a. | 4.7 – 6.7 | Piecewise-linear through anchors shaped like US 30-yr mortgage rates through the financial crisis (6.25% → peak 6.65% mid-2007 → 4.75% by 2010) + N(0, 0.04) monthly noise. |
| `market_price_index` | float | index | ~94 – 106 | Boom–crisis–rebound shape: +5.5% to mid-2007, −10% peak-to-trough through 2009, **+9.5% rebound during 2010** (sharp post-crisis recoveries of this size occurred in real US metros; sized to be detectable within the 7-month stream). The rebound occurs only inside the monitoring stream, so a model trained on 2006–09 under-predicts 2010 — a designed, explainable performance-drift signal for Module 7. + N(0, 0.35) noise. |

## Injected data-quality problems

Deliberate, seeded, injected **only into synthetic fields** and **only before
2010** (the monitoring stream stays clean so drift is never confused with
dirt). Module 3 must detect and fix each with before/after evidence.
Definitions: `src/data_generation/quality_imperfections.py`.

| # | Field | Problem | Rate / count (realized) |
|---|---|---|---|
| 1 | `dist_school_km` | Missing (blank at listing entry) | 2% (26 rows) |
| 2 | `renovated` | Inconsistent free-text labels: Y, yes, YES, N, no, NO | 4% (51 rows) |
| 3 | `days_on_market` | Legacy sentinel `999` = "unknown" | 0.4% (5 rows) |
| 4 | `dist_transit_km` | Entered in meters instead of km (values > 100) | 0.3% (4 rows) |
| 5 | whole row | Same home re-posted days later under a new `listing_id` (`L-9…`) | 12 rows |
