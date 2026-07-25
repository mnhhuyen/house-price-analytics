# Decisions

Every non-trivial design choice, in plain business language. This file feeds
the final report and the defense to the academic panel. Format: what we chose,
why, and what we rejected.

---

## D1. Grading-driven effort allocation

**Choice:** Invest effort in (1) realistic documented synthetic data,
(2) clear EDA insight, (3) a deployment that reliably runs on a free tier,
(4) monitoring that genuinely detects drift — and deliberately *not* in
squeezing out the last few percent of model accuracy.
**Why:** The course brief states grading "emphasizes the rigor and clarity of
each stage, rather than model accuracy alone." Optimizing RMSE beyond a solid
baseline has near-zero grade return; a broken demo or fake monitoring has
catastrophic grade cost.

## D2. Deployment architecture — Streamlit with the model in-process

**Choice:** The deployed valuation tool is a Streamlit app that loads the
model directly in its own process. The FastAPI service is built and documented
as a separate deliverable, but the demo never depends on it.
**Why:** One running service = one thing that can fail during the live
presentation. Free tiers make multi-service setups fragile (each service can
sleep or run out of RAM independently).
**Rejected:** Streamlit calling the FastAPI backend over HTTP — doubles the
failure surface for zero demo benefit.

## D3. Deploy target — Streamlit Community Cloud (revised 2026-07-11)

**Choice:** Streamlit Community Cloud, deploying directly from our public
GitHub repository.
**Why:** Free, explicitly approved by the course brief, purpose-built for
Streamlit, and it deploys straight from GitHub — a required deliverable
anyway, so no extra infrastructure. Known caveat: apps sleep after ~12 hours
without visitors and take ~1 minute to wake; mitigation is opening the link
shortly before any demo (in the presentation runbook).
**History:** Our first choice was Hugging Face Spaces, but during setup we
found HF had moved all compute Spaces (Streamlit/Gradio/Docker) behind a paid
plan — only static pages remain free. We validated the deploy pipeline against
HF in the project's first week, then pivoted; the app itself needed no changes, which is
exactly why the pipeline was de-risked before modeling (see D4).
**Rejected:** Render free tier — sleeps after only 15 min idle with 30–60 s
cold starts, riskier live; HF PRO — paying for a course project is
unnecessary.

## D4. Deploy pipeline proven before the model exists

**Choice:** Before any real modeling started, we shipped a placeholder
estimator through the full deploy pipeline ("hello world" deploy).
**Why:** Free-tier deployment is the most binary deliverable: it either runs
or it doesn't, and its failure modes (RAM limits, build errors, cold starts)
are unrelated to modeling. Proving the pipeline first means Module 6 is a
low-risk artifact swap instead of a last-minute integration scramble.

## D5. Confidence intervals — LightGBM quantile regression + conformal check

**Choice:** Train quantile models (p10/p50/p90) for an 80% interval, then
verify calibration on held-out data with a conformal-prediction check
(intervals should contain the true price ~80% of the time; widths adjusted if
not).
**Why:** The business question explicitly asks "how confident is that
estimate?" — so intervals are a first-class requirement decided at modeling
time. Quantile regression produces feature-dependent intervals: wide for
rare/expensive homes, narrow for typical ones — behavior we surface in the UI.
The conformal check lets us *prove* the stated 80% is honest, not decorative.
**Rejected:** A fixed ±X% band (ignores that uncertainty varies per home);
bolting intervals on at deployment time (untestable, indefensible).

## D6. Synthetic data carries a real time axis, designed up front

**Choice:** Generated data includes a `sale_date` anchored to 55 real Ames
sale months plus a derived `listing_date = sale_date - days_on_market`, with
realistic
drift baked in (price-index trend, shifting days-on-market, a local
interest-rate series). All 2010 sales are held out as the "incoming stream"
and never used in training.
**Why:** The Kaggle Ames data is a static snapshot, but Module 7 requires
detecting drift "as new sales data arrives." Without a time axis designed into
the data from Module 1, monitoring would be fake. With drift baked in, the
Evidently dashboard genuinely fires and the retraining triggers are testable.

## D7. Known imperfections injected into the synthetic fields

**Choice:** The synthetic generator deliberately injects a small, seeded,
documented dose of data-quality problems (missing values, inconsistent
category labels, a few implausible entries) into the *synthetic* fields only.
**Why:** The Ames data is already fairly clean, so Module 3 (cleaning) would
otherwise have little real work. Injecting known dirt simulates the
data-quality problems a real listing platform faces, and because we know the
ground truth, we can show before/after evidence that the cleaning worked.

## D8. Currency — USD

**Choice:** The demo displays USD.
**Why:** Faithful to the Ames, Iowa training data; no exchange-rate assumption
to defend. (VND localization was considered and rejected as an extra
assumption with no analytical benefit.)

## D9. Python 3.11 + pinned dependencies, numpy held below 2.0

**Choice:** Python 3.11 everywhere (local venv and the Space), every
dependency pinned in `requirements.txt`, numpy pinned to 1.26.4.
**Why:** Reproducibility is an explicit grading criterion. Python 3.11 is the
best-supported version across our stack. The monitoring library (evidently
0.4.x) is incompatible with numpy 2, so numpy is held back repo-wide to keep
one consistent environment.
**Also:** The repo-root `requirements.txt` is the lean deploy set (Streamlit
Community Cloud installs it when building the app) — monitoring/EDA libraries
never ship to production. The full dev environment is `requirements-dev.txt`.

## D10. Raw Kaggle data is not committed to the repo

**Choice:** `data/raw/` is gitignored; each member downloads from Kaggle.
**Why:** Kaggle competition rules prohibit redistributing the data publicly.
README documents the exact download steps. Synthetic and processed data we
generate ourselves *are* committed for reproducibility.

## D11. Time axis anchored to the real Ames sale dates

**Choice:** `sale_date` uses the dataset's own `YrSold`/`MoSold` columns
(Jan 2006 – Jul 2010) plus a seeded day. `listing_date` is then derived by
subtracting `days_on_market`. All 2010 sales (175 homes) form the monitoring
holdout stream.
**Why:** The brief requires a time dimension for monitoring; the honest way to
get one is to use the temporal information the data already contains — no row
is ever internally inconsistent (e.g. "listed" a decade after it sold). The
2006–2010 window also gives us a historically grounded macro narrative (the
financial crisis) instead of an arbitrary trend.
**Rejected:** A fully invented 2018–2020 axis — it would contradict the real
`YrSold` column in every row and be indefensible under scrutiny.

## D12. The 2010 rebound is the designed drift signal

**Choice:** The synthetic market price index falls ~10% through 2008–09, then
rebounds ~9.5% during 2010 — entirely inside the monitoring stream (initially
+5%; strengthened for detectability — see D17). The primary modeling target
`sale_price` remains the original Kaggle label. A separate
`sale_price_market_adjusted` label re-expresses it under the synthetic index
relative to 2010-01 and is used only to replay the monitoring scenario;
before 2010 it is exactly equal to the original target.
**Why:** Module 7 must demonstrate *detectable* drift. A model trained on
2006–09 has never seen a rebound, so it under-predicts 2010 progressively —
rolling error rises for an explainable business reason ("the market recovered
faster than the model's training window"), not because we faked a random
shock. Covariate drift is also present: 2010 interest rates (≈4.7%) sit below
anything in training, and days-on-market shortens.

## D13. Synthetic realism is enforced by tests, not just claimed

**Choice:** Every property claimed in DATA_DICTIONARY.md (amenity/DOM
generation independent of SalePrice, dirt exists only before 2010, renovation
fields agree with the real remodel column, the trend/rebound shape) is encoded
as a pytest invariant (`tests/test_synthetic_data.py`).
**Why:** "Justify the realism of generated features" is an explicit grading
criterion; a documented claim that can silently rot is worth little. The tests
make the documentation binding.

## D18. Amenity distances — hybrid of map + priors + Condition/Zoning (no SalePrice)

**Choice:** Combine three generators: (1) seeded fictional 10×10 km map for
base Euclidean distances; (2) neighborhood access priors from modal MSZoning,
median LotArea density, and a short hand-crafted urban-form table; (3)
listing-level Condition1/2 nudges. After cleaning, `amenity_score` 0–100 is
derived from proximity + Condition + Zoning.
**Why:** The brief asks for contextual amenity scores with business logic.
A pure random map is leakage-safe but hard to narrate; price-ranked bases
leak the target. The hybrid stays leakage-safe while using real Kaggle
columns (zoning, lot size, proximity codes) and documented urban-form priors.
**Rejected:** Rank-by-median-SalePrice bases (target contamination).
`amenity_score` is for EDA only — the model already has the three distances.

## D19. Original SalePrice is the primary target

**Choice:** Train, cross-validate, calibrate and report the model against the
unmodified Kaggle `SalePrice`. Keep the macro-adjusted price under a separate,
explicitly named monitoring-scenario column.
**Why:** Contextual-data generation should not silently redefine ground truth.
Separating the two labels preserves valid model metrics while still allowing
Module 7 to demonstrate a designed market-recovery scenario.

## D14. Deploy the quantile LightGBM even though Ridge wins on CV RMSE

**Choice:** 5-fold CV showed Ridge as the best point model (MAPE 8.99%, R²
0.916) vs LightGBM (9.78%, 0.882). We deploy the quantile LightGBM anyway,
using its median (p50) forecast as the point estimate.
**Why:** The deliverable is a price **with a calibrated confidence range**.
Quantile regression gives feature-dependent intervals natively (wide for
unusual homes, narrow for common ones); one coherent model is simpler and
more robust than stitching a Ridge point onto intervals from another family
(mismatched point/band, two failure modes). The accuracy cost (~0.8 MAPE
points) is immaterial to the tool's purpose, and grading explicitly
de-prioritizes raw accuracy. The comparison is reported openly, not hidden.
**Rejected:** Ridge + constant-percentage conformal band — dollar widths
would scale only with price, not with how unusual the home is.

## D15. Leakage discipline: what the model is not allowed to see

**Choice:** Excluded from model features: `days_on_market` (an outcome of the
listing — unknown on the day the home is priced) and `market_price_index`
(a citywide index is published with a lag; feeding the model the current
month's value would be look-ahead leakage and would trivially encode the
target's trend). Included: `local_interest_rate_pct` (publicly known at
listing time).
**Why:** A valuation tool must only use inputs an agent actually has when
pricing a home. This honesty is also what makes the monitoring module
meaningful: the model cannot see the 2010 rebound through a leaked index, so
the drift is genuine.

## D16. Four retraining triggers, defined before the stream was replayed

**Choice:** T1 input drift (>30% of features, Evidently), T2 rolling RMSE
(>1.25× training baseline), T3 systematic bias (|mean error| >5% of median
price), T4 interval coverage (<65% observed vs 80% promised) — evaluated on
rolling 3-month windows, drift gated at ≥40 rows.
**Why:** Committing to thresholds up front prevents "tuning alerts until they
fire". Rolling windows because single months (6–48 sales) are statistically
unstable — we verified a 10-row window flags every feature as drifted. The
bias trigger matters most for the business: a model can hold acceptable RMSE
while systematically under-pricing every listing (exactly what the 2010
rebound produces: bias reaches −7.9% while RMSE stays near baseline).
**2010 verdict:** retrain from March — T2 fires once, T1 persistently
(rate regime + post-crisis sales mix), T3 from April onward.

## D17. The designed rebound was strengthened from +5% to +9.5%

**Choice:** After the first monitoring replay, the original +5% rebound
produced real but sub-threshold drift (bias ≈ −4% by July, under the 5%
trigger). We strengthened the synthetic rebound to +9.5% over 2010 and
regenerated the entire pipeline, rather than lowering alert thresholds.
**Why:** Lowering thresholds to make alerts fire is tuning the test to the
answer — indefensible. Adjusting the *designed scenario* is legitimate for
synthetic data (the scenario's purpose is to demonstrate detection within a
short 7-month stream) and is documented here and in the data dictionary.
Post-crisis rebounds of ~10%/year occurred in real US metros. Because every
stage is a seeded script, regeneration end-to-end (data → cleaning →
features → model → monitoring, notebook re-executed) took minutes and every
number in the docs was refreshed.
