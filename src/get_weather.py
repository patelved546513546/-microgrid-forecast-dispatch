# src/get_weather.py
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib

WEATHER_CACHE_DIR = Path("data/weather_cache")

def _cache_path(tag: str, lat: float, lon: float, start: str, end: str, timezone: str):
    WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{tag}|{lat:.3f}|{lon:.3f}|{start}|{end}|{timezone}"
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return WEATHER_CACHE_DIR / f"{tag}_{h}.csv"

def _save_cache(df: pd.DataFrame, path: Path, source="api"):
    df = df.copy()
    df.to_csv(path, index=False)
    df.attrs["source"] = source
    return df

def _read_cache(path: Path):
    df = pd.read_csv(path)
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df.attrs["source"] = "cache"
    return df

def synthesize_weather(lat, lon, start, end, timezone="auto", freq="H"):
    # Simple realistic hourly profile so the model can still run offline
    rng = pd.date_range(start=start, end=end, freq=freq, inclusive="left")
    hr = rng.hour.values

    # Shortwave radiation: bell curve around noon; zero at night
    rad = 800.0 * np.exp(-0.5 * ((hr - 12) / 4.0) ** 2)
    rad = np.where((hr >= 6) & (hr <= 18), rad, 0.0)
    direct = rad * 0.7
    diffuse = rad * 0.3

    temp = 20.0 + 8.0 * np.sin(2 * np.pi * (hr - 8) / 24)
    rh = 60.0 - 10.0 * np.sin(2 * np.pi * (hr - 8) / 24)
    cloud = np.full_like(hr, 40.0, dtype=float)
    wind = 3.0 + 2.0 * np.sin(2 * np.pi * hr / 24)
    precip = np.zeros_like(hr, dtype=float)

    df = pd.DataFrame({
        "DATETIME": rng.tz_localize(None),
        "temperature_2m": temp,
        "relative_humidity_2m": rh,
        "cloud_cover": cloud,
        "shortwave_radiation": rad,
        "direct_radiation": direct,
        "diffuse_radiation": diffuse,
        "wind_speed_10m": wind,
        "precipitation": precip,
    })
    df.attrs["source"] = "synthetic"
    return df

def fetch_openmeteo_history(lat, lon, start, end, timezone="auto",
                            use_cache=True, fallback=True, timeout=60):
    """
    Returns hourly weather with DATETIME column.
    Tries: cache → API → synthetic fallback (if fallback=True).
    df.attrs['source'] in {'cache','api','synthetic'}
    """
    cache_fp = _cache_path("era5", lat, lon, start, end, timezone)
    if use_cache and cache_fp.exists():
        return _read_cache(cache_fp)

    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat, "longitude": lon, "timezone": timezone,
        "start_date": start, "end_date": end,
        "hourly": ",".join([
            "temperature_2m","relative_humidity_2m","cloud_cover",
            "shortwave_radiation","direct_radiation","diffuse_radiation",
            "wind_speed_10m","precipitation"
        ])
    }

    # simple retry loop
    for tmo in (timeout, int(timeout/2), int(timeout/3)):
        try:
            r = requests.get(url, params=params, timeout=tmo)
            r.raise_for_status()
            j = r.json()
            df = pd.DataFrame(j["hourly"])
            df["DATETIME"] = pd.to_datetime(df.pop("time"))
            df.attrs["source"] = "api"
            if use_cache:
                return _save_cache(df, cache_fp, source="api")
            return df
        except Exception:
            continue

    # If we reach here: API failed
    if use_cache and cache_fp.exists():
        return _read_cache(cache_fp)
    if fallback:
        return synthesize_weather(lat, lon, start, end, timezone=timezone)
    raise  # if fallback=False, propagate error

def fetch_openmeteo_forecast(lat, lon, start, end, timezone="auto",
                             use_cache=True, fallback=True, timeout=30):
    # Similar to history but hits the forecast endpoint
    cache_fp = _cache_path("forecast", lat, lon, start, end, timezone)
    if use_cache and cache_fp.exists():
        return _read_cache(cache_fp)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": timezone,
        "start_date": start, "end_date": end,
        "hourly": ",".join([
            "temperature_2m","relative_humidity_2m","cloud_cover",
            "shortwave_radiation","direct_radiation","diffuse_radiation",
            "wind_speed_10m","precipitation"
        ])
    }
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        df = pd.DataFrame(j["hourly"])
        df["DATETIME"] = pd.to_datetime(df.pop("time"))
        df.attrs["source"] = "api"
        if use_cache:
            return _save_cache(df, cache_fp, source="api")
        return df
    except Exception:
        if use_cache and cache_fp.exists():
            return _read_cache(cache_fp)
        if fallback:
            return synthesize_weather(lat, lon, start, end, timezone=timezone)
        raise