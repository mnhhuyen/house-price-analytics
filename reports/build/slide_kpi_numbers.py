"""One-off: numbers for the slide edits.

(a) Naive baseline (neighborhood median $/sqft) under the exact same 5-fold
    CV protocol as compare_models.py.
(b) High-confidence subset MAPE from the shipped quantile model, on the
    calibration split (never seen by the p50 model).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import KFold, train_test_split

from src.config import MODELS_DIR, RANDOM_SEED
from src.features.build_model_table import TARGET
from src.modeling.model_inputs import (category_levels, encode_for_trees,
                                       load_model_table)
from src.modeling.train_final import predict_interval

train, _ = load_model_table()
y = train[TARGET].to_numpy()

# ---------- (a) naive baseline, same CV protocol ----------
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
pred = np.empty(len(train))
for tr_idx, te_idx in cv.split(train):
    tr, te = train.iloc[tr_idx], train.iloc[te_idx]
    ppsf = (tr[TARGET] / tr["GrLivArea"]).groupby(tr["Neighborhood"]).median()
    global_ppsf = (tr[TARGET] / tr["GrLivArea"]).median()
    rate = te["Neighborhood"].map(ppsf).fillna(global_ppsf)
    pred[te_idx] = rate * te["GrLivArea"]

print("=== (a) Naive baseline: neighborhood median $/sqft x area, 5-fold CV ===")
print(f"RMSE ${np.sqrt(mean_squared_error(y, pred)):,.0f}  "
      f"MAE ${mean_absolute_error(y, pred):,.0f}  "
      f"MAPE {mean_absolute_percentage_error(y, pred)*100:.2f}%  "
      f"R2 {r2_score(y, pred):.3f}")

# ---------- (b) high-confidence MAPE on the calibration split ----------
art = joblib.load(MODELS_DIR / "valuation_model.pkl")
models, offset = art["models"], art["conformal_offset_log"]
levels = category_levels(train)

fit_df, cal_df = train_test_split(train, test_size=0.20, random_state=RANDOM_SEED)
X_cal = encode_for_trees(cal_df, levels)
y_cal = cal_df[TARGET].to_numpy()

iv = predict_interval(models, offset, X_cal)
rel_width = (iv.high - iv.low) / iv.point * 100
ape = np.abs(iv.point.to_numpy() - y_cal) / y_cal * 100

print(f"\n=== (b) calibration split n={len(cal_df)} "
      f"(p50 never trained on these rows) ===")
print(f"global MAPE on cal split: {ape.mean():.2f}%")

print("\nMAPE by relative-interval-width tercile:")
terc = pd.qcut(rel_width, 3, labels=["narrow", "mid", "wide"])
df = pd.DataFrame({"rel_width": rel_width, "ape": ape, "terc": terc})
print(df.groupby("terc", observed=True)
        .agg(n=("ape", "size"), mean_rel_width=("rel_width", "mean"),
             mape=("ape", "mean"), coverage_med=("ape", "median"))
        .round(2).to_string())

print("\nthreshold sweep: homes with relative width <= t")
for t in [20, 22, 24, 25, 26, 28, 30, 32, 35]:
    m = rel_width <= t
    if m.sum() == 0:
        continue
    print(f"  width<={t:>2}%: share {m.mean()*100:5.1f}%  n={m.sum():>3}  "
          f"MAPE {ape[m.to_numpy()].mean():5.2f}%")
