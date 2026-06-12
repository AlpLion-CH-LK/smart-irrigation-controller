"""Download 20 years of HOURLY weather data for Uduvil, Jaffna from NASA POWER.

Fetches year by year, saves:
  - data/weather/uduvil_per_hour/raw/uduvil_hourly_<year>.json
  - data/weather/uduvil_per_hour/uduvil_hourly_2004_2024.csv

CSV columns:
  datetime, temperature, humidity_pct, rain_mm, is_raining,
  wind_speed_ms, solar_rad_MJ, et0_mm

Usage:
    python scripts/fetch_weather_hourly.py
    python scripts/fetch_weather_hourly.py --start-year 2010 --end-year 2024
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd


# ---------------------------------------------------------------------------
# Location — Uduvil, Jaffna
# ---------------------------------------------------------------------------
LAT = 9.7432
LON = 80.0076

# ---------------------------------------------------------------------------
# NASA POWER Hourly API
# ---------------------------------------------------------------------------
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

PARAMETERS = ",".join([
    "T2M",               # Temperature at 2m (°C) — hourly
    "RH2M",              # Relative humidity at 2m (%)
    "PRECTOTCORR",       # Precipitation (mm/hr)
    "WS2M",              # Wind speed at 2m (m/s)
    "ALLSKY_SFC_SW_DWN", # Solar radiation (MJ/m²/hr)
])

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "weather" / "uduvil_per_hour" / "raw"
OUTPUT_CSV   = PROJECT_ROOT / "data" / "weather" / "uduvil_per_hour" / "uduvil_hourly_2004_2024.csv"


# ---------------------------------------------------------------------------
# Hourly ET₀ — FAO-56 Penman-Monteith (hourly version)
# Reference: FAO Irrigation and Drainage Paper 56, Chapter 4
# ---------------------------------------------------------------------------

def _saturation_vapour_pressure(temp_c: float) -> float:
    """Saturation vapour pressure in kPa for given temperature."""
    return 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))


def _slope_vapour_pressure(temp_c: float) -> float:
    """Slope of saturation vapour pressure curve (kPa/°C)."""
    es = _saturation_vapour_pressure(temp_c)
    return (4098 * es) / ((temp_c + 237.3) ** 2)


def compute_et0_hourly(
    temp_c: float,
    humidity_pct: float,
    wind_ms: float,
    solar_MJ: float,
    hour: int,
) -> float:
    """Compute FAO-56 Penman-Monteith ET₀ for one hour (mm/hr).

    Args:
        temp_c:       Air temperature (°C)
        humidity_pct: Relative humidity (%)
        wind_ms:      Wind speed at 2m (m/s)
        solar_MJ:     Incoming solar radiation (MJ/m²/hr)
        hour:         Hour of day (0-23), used for soil heat flux sign
    """
    # Psychrometric constant (kPa/°C) at sea level
    gamma = 0.0665

    delta = _slope_vapour_pressure(temp_c)

    es = _saturation_vapour_pressure(temp_c)
    ea = es * (humidity_pct / 100.0)

    # Net shortwave radiation (albedo = 0.23 for reference crop)
    Rns = (1 - 0.23) * solar_MJ

    # Simplified net longwave (outgoing) radiation — constant approximation
    Rnl = -0.05  # MJ/m²/hr (negative = energy leaving surface)

    Rn = Rns + Rnl

    # Soil heat flux — daytime: 10% of Rn, nighttime: 50% of Rn
    G = 0.1 * Rn if 6 <= hour <= 18 else 0.5 * Rn

    # Wind speed correction (already at 2m)
    u2 = max(0.5, wind_ms)  # minimum 0.5 m/s to avoid division issues

    # FAO-56 hourly Penman-Monteith
    numerator = (
        0.408 * delta * (Rn - G)
        + gamma * (37.0 / (temp_c + 273)) * u2 * (es - ea)
    )
    denominator = delta + gamma * (1 + 0.34 * u2)

    et0 = numerator / denominator
    return round(max(0.0, et0), 4)


# ---------------------------------------------------------------------------
# NASA POWER fetch — one year at a time
# ---------------------------------------------------------------------------

def fetch_year(year: int) -> dict:
    """Fetch one year of hourly data from NASA POWER."""
    params = {
        "parameters": PARAMETERS,
        "community":  "AG",
        "longitude":  LON,
        "latitude":   LAT,
        "start":      f"{year}0101",
        "end":        f"{year}1231",
        "format":     "JSON",
    }

    print(f"  Fetching {year} (hourly)...", end=" ", flush=True)
    response = requests.get(NASA_POWER_URL, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()
    print(f"done.")
    return data


def save_raw(year: int, data: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"uduvil_hourly_{year}.json"
    with open(path, "w") as f:
        json.dump(data, f)


def load_raw(year: int) -> dict | None:
    path = RAW_DIR / f"uduvil_hourly_{year}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Parse NASA POWER hourly response → list of hourly records
# ---------------------------------------------------------------------------

def parse_year(data: dict, year: int) -> list[dict]:
    """Parse one year of hourly NASA POWER JSON into a list of dicts.

    NASA POWER hourly keys are formatted as 'YYYYMMDD HH' (e.g. '20040115 06').
    """
    params   = data["properties"]["parameter"]
    temp_d   = params["T2M"]
    rh_d     = params["RH2M"]
    rain_d   = params["PRECTOTCORR"]
    wind_d   = params["WS2M"]
    solar_d  = params["ALLSKY_SFC_SW_DWN"]

    records = []
    current = datetime(year, 1, 1, 0, 0)
    end     = datetime(year, 12, 31, 23, 0)

    while current <= end:
        key = current.strftime("%Y%m%d%H")    # e.g. '2004010106'

        temp   = temp_d.get(key, None)
        rh     = rh_d.get(key, 70.0)
        rain   = rain_d.get(key, 0.0)
        wind   = wind_d.get(key, 2.0)
        solar  = solar_d.get(key, 0.0)

        # Skip rows with NASA missing-value flag
        if temp is None or temp < -900:
            current += timedelta(hours=1)
            continue

        # Replace other missing values with sensible defaults
        if rh    < -900: rh    = 70.0
        if rain  < -900: rain  = 0.0
        if wind  < -900: wind  = 2.0
        if solar < -900: solar = 0.0

        rh    = max(0.0, min(100.0, rh))
        rain  = max(0.0, rain)
        wind  = max(0.0, wind)
        solar = max(0.0, solar)

        et0 = compute_et0_hourly(
            temp_c       = temp,
            humidity_pct = rh,
            wind_ms      = wind,
            solar_MJ     = solar,
            hour         = current.hour,
        )

        records.append({
            "datetime":      current.strftime("%Y-%m-%d %H:%M"),
            "temperature":   round(temp, 2),
            "humidity_pct":  round(rh, 2),
            "rain_mm":       round(rain, 2),
            "is_raining":    int(rain > 0.1),
            "wind_speed_ms": round(wind, 2),
            "solar_rad_MJ":  round(solar, 4),
            "et0_mm":        et0,
        })

        current += timedelta(hours=1)

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(start_year: int = 2004, end_year: int = 2024) -> None:
    print(f"NASA POWER Hourly Weather Fetch — Uduvil, Jaffna")
    print(f"Location : lat={LAT}, lon={LON}")
    print(f"Period   : {start_year} → {end_year}")
    print(f"Output   : {OUTPUT_CSV}\n")

    all_records: list[dict] = []

    for year in range(start_year, end_year + 1):
        raw = load_raw(year)
        if raw is not None:
            print(f"  Year {year}: using cached raw file.")
        else:
            try:
                raw = fetch_year(year)
                save_raw(year, raw)
            except requests.RequestException as e:
                print(f"  Year {year}: FAILED — {e}. Skipping.")
                continue
            time.sleep(1.5)   # polite delay between requests

        records = parse_year(raw, year)
        all_records.extend(records)
        print(f"  Year {year}: {len(records)} hourly records parsed.")

    if not all_records:
        print("\nNo data collected. Check your internet connection.")
        return

    df = pd.DataFrame(all_records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone.")
    print(f"Total hours : {len(df):,}")
    print(f"Date range  : {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"Saved to    : {OUTPUT_CSV}")
    print(f"\nSummary statistics:")
    print(df[["temperature", "humidity_pct", "rain_mm", "wind_speed_ms", "solar_rad_MJ", "et0_mm"]].describe().round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch NASA POWER hourly weather data for Uduvil, Jaffna."
    )
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year",   type=int, default=2024)
    args = parser.parse_args()

    main(args.start_year, args.end_year)
