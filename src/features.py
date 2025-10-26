# src/features.py
import numpy as np
import pandas as pd

def time_features(df, ts="DATETIME"):
    df["hour"] = df[ts].dt.hour
    df["dow"] = df[ts].dt.dayofweek
    df["month"] = df[ts].dt.month
    df["sin_hour"] = np.sin(2*np.pi*df["hour"]/24)
    df["cos_hour"] = np.cos(2*np.pi*df["hour"]/24)
    df["sin_dow"] = np.sin(2*np.pi*df["dow"]/7)
    df["cos_dow"] = np.cos(2*np.pi*df["dow"]/7)
    return df

def lag_roll(df, cols, lags=(1,2,3,6,12,24,48), rolls=(3,6,12,24,48)):
    for c in cols:
        for L in lags:
            df[f"lag_{c}_{L}"] = df[c].shift(L)
        for R in rolls:
            df[f"rmean_{c}_{R}"] = df[c].shift(1).rolling(R).mean()
            df[f"rstd_{c}_{R}"] = df[c].shift(1).rolling(R).std()
    return df

def make_features(df_load, df_weather):
    df = pd.merge_asof(
        df_load.sort_values("DATETIME"),
        df_weather.sort_values("DATETIME"),
        on="DATETIME", direction="nearest", tolerance=pd.Timedelta("30m")
    )
    df = time_features(df)
    base_cols = [c for c in ["NET_LOAD_KW","SOLAR_KW","CONSUMPTION_KW"] if c in df.columns]
    df = lag_roll(df, cols=base_cols)

    df["is_day"] = (df.get("shortwave_radiation", pd.Series(0, index=df.index)) > 10).astype(int)
    df = df.dropna().reset_index(drop=True)

    y = df["NET_LOAD_KW"]

    # DROP DATETIME from the model features and keep only numeric columns
    drop_cols = [c for c in ["DATETIME","NET_LOAD_KW","CONSUMPTION_KWH","SOLAR_KWH"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    X = X.select_dtypes(include=[np.number]).copy()

    return X, y, df  # df still contains DATETIME for plotting/saving