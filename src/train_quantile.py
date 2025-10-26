import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor

from load_ausgrid_wide import load_ausgrid_wide
from get_weather import fetch_openmeteo_history
from features import make_features

# ---------- Utilities ----------
def seasonal_naive(y, period: int):
    y = np.asarray(y, dtype=float)
    y_hat = np.roll(y, period)
    y_hat[:period] = np.nan
    return y_hat

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y_true = y_true[mask]; y_pred = y_pred[mask]
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape}

def fit_quantile_models(X_train, y_train, quantiles=(0.1, 0.5, 0.9), n_estimators=800):
    models = {}
    for q in quantiles:
        m = GradientBoostingRegressor(
            loss="quantile", alpha=q,
            learning_rate=0.05, n_estimators=n_estimators,
            max_depth=3, subsample=0.9, random_state=42
        )
        # ensure numeric dtypes
        m.fit(X_train.astype("float32"), y_train.astype("float32"))
        models[f"q{int(q*100)}"] = m
    return models

# ---------- Main run ----------
def run(
    ausgrid_csv: Path = None,
    lat: float = -33.86, lon: float = 151.20,
    start: str = "2012-07-01", end: str = "2013-06-30",
    model_dir: Path = None
):
    ROOT = Path(__file__).resolve().parents[1]
    if ausgrid_csv is None:
        ausgrid_csv = ROOT / "data" / "ausgrid" / "2012-2013 Solar home electricity data v2.csv"
    if model_dir is None:
        model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load and align data
    df_load = load_ausgrid_wide(str(ausgrid_csv), aggregate="sum_all")
    df_load = df_load[(df_load["DATETIME"] >= start) & (df_load["DATETIME"] <= end)]
    df_w = fetch_openmeteo_history(lat, lon, start, end, timezone="Australia/Sydney")

    # 2) Feature engineering
    X, y, df_full = make_features(df_load, df_w)

    # 3) Time split
    split = int(len(X) * 0.8)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]

    # 4) Baseline
    base = seasonal_naive(y.values, period=48)  # 30-min data
    print("Baseline:", metrics(yte.values, base[split:]))

    # 5) Train quantile models
    models = fit_quantile_models(Xtr, ytr, quantiles=(0.1, 0.5, 0.9), n_estimators=800)

    # 6) Evaluate P50
    p50 = models["q50"].predict(Xte)
    print("Model (P50):", metrics(yte.values, p50))

    # 7) Predict all quantiles on test
    p10 = models["q10"].predict(Xte)
    p90 = models["q90"].predict(Xte)
    coverage = float(np.mean((yte.values >= p10) & (yte.values <= p90)) * 100.0)
    print(f"Interval coverage (P10–P90): {coverage:.1f}%")

    # 8) Save models and predictions
    for k, m in models.items():
        joblib.dump(m, str(model_dir / f"{k}.pkl"))
    with open(model_dir / "feature_cols.json", "w") as f:
        json.dump({"feature_cols": list(X.columns)}, f, indent=2)

    # IMPORTANT: Xte no longer has DATETIME; take timestamps from df_full
    dt_test = df_full.iloc[split:]["DATETIME"].reset_index(drop=True)
    out = pd.DataFrame({
        "DATETIME": dt_test,
        "y_true": yte.values,
        "p10": p10,
        "p50": p50,
        "p90": p90
    })
    out.to_csv(model_dir / "test_predictions_netload.csv", index=False)
    print(f"Saved model + predictions to {model_dir}")

if __name__ == "__main__":
    run()