# House Price Prediction for a Real Estate Listing Platform

End-to-end Business Analytics group project (IT5315E): predict a home's sale
price with a confidence range, deploy the model as a working valuation tool,
and monitor it for data drift after deployment.

**Business question:** *"Given a property's characteristics and location, what
is its estimated market price, and how confident is that estimate?"*

## Live demo

- 🏠 **Valuation + monitoring (Streamlit):** https://house-price-analyticsgit-o5btajapatusvnfdndcsd6.streamlit.app/
- 📈 **Monitoring dashboard:** second page of the same app (sidebar) — live rolling RMSE/MAE on the 2010 stream plus walk-forward retrain simulation
- 🔌 **Prediction API (FastAPI):** run locally with `uvicorn app.api.main:app` → interactive docs at http://localhost:8000/docs (same model, same serving code as the live tool)

**Cloud Python version:** Community Cloud ignores `runtime.txt`. When (re)deploying, open **Advanced settings** and choose **Python 3.11** (or 3.12). If the app was created on 3.14, delete it and redeploy with 3.11 selected — reboot alone cannot change Python.

## Team

| Member | Student ID |
|---|---|
| Đỗ Hoàng Việt | 20252744M |
| Nguyễn Minh Huyền | 20252007M |
| Nguyễn Thị Huyền Trang | 20251322M |
| Bùi Tá Cường | 20242430M |
| Nguyễn Tiến Đức | 20252076M |

## Project documentation

| Doc | Purpose |
|---|---|
| [PROJECT_BRIEF.md](PROJECT_BRIEF.md) | Course specification (source of truth) |
| [DECISIONS.md](DECISIONS.md) | Every design choice, in plain business language |
| [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | Every synthetic field: type, unit, range, generation logic |
| [docs/huong-dan-cho-nhom.md](docs/huong-dan-cho-nhom.md) | Team guide (Vietnamese): goals, what to read, what to run, defense checklist |

## Setup

Requires Python 3.11 and the Kaggle "House Prices — Advanced Regression
Techniques" data (not committed — competition rules forbid redistribution).

```bash
# 1. Environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # full dev env; requirements.txt is the lean deploy set

# 2. Data: download train.csv, test.csv, data_description.txt from
# kaggle.com/competitions/house-prices-advanced-regression-techniques
# and place them in data/raw/

# 3. Run the valuation tool locally
streamlit run app/streamlit_app.py
```

## Pipeline commands

Run from the repo root, in order. Every step is seeded — outputs are
byte-for-byte reproducible.

```bash
python -m src.data_generation.generate     # M1: synthetic data (DATA_DICTIONARY.md)
jupyter nbconvert --execute --inplace notebooks/eda_price_drivers.ipynb  # M2: EDA
python -m src.cleaning.clean_listings      # M3: cleaning w/ before-after evidence
python -m src.features.build_model_table   # M4: features + VIF check
python -m src.modeling.compare_models      # M5: 5-fold CV comparison + residuals
python -m src.modeling.train_final         # M5: quantile model + conformal calibration
python -m src.monitoring.run_monitoring    # M7: drift/performance replay of 2010
pytest tests/                              # invariants + app/API smoke tests
```

Monitoring outputs land in `monitoring/reports/` (per-month Evidently HTML +
`monitoring_summary.csv`); report figures in `reports/figures/`.

## Repository layout

```
data/          raw (Kaggle, gitignored) · synthetic · processed
src/           pipeline code: data_generation · cleaning · features · modeling · monitoring
notebooks/     EDA and residual-analysis notebooks
app/           Streamlit app (valuation tool + monitoring dashboard) + FastAPI service
models/        trained model artifacts
monitoring/    generated Evidently drift & performance reports
reports/       final report (PDF + HTML source), slides (PPTX on the official
               HUST template, built by reports/build/build_slides.py), figures
tests/         smoke tests (data schema, prediction sanity)
```
