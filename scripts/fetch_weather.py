"""Download 20 years of daily weather data for Uduvil, Jaffna from NASA POWER.

Fetches year by year to stay within API limits, saves:
  - data/weather/raw/uduvil_<year>.json   (raw API response per year)
  - data/weather/uduvil_weather_2004_2024.csv  (processed, ready to use)

Usage:
    python scripts/fetch_weather.py
    python scripts/fetch_weather.py --start-year 2004 --end-year 2024
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import date, timedelta
from pathlib import Path

import requests
import pandas as pd


# ---------------------------------------------------------------------------
# Location — Uduvil, Jaffna
# ---------------------------------------------------------------------------
LAT = 9.7432
LON = 80.0076

# ---------------------------------------------------------------------------
# NASA POWER API
# ---------------------------------------------------------------------------
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

PARAMETERS = ",".join([
    "T2M_MAX",          # Daily max temperature (°C)
    "T2M_MIN",          # Daily min temperature (°C)
    "RH2M",             # Relative humidity at 2m (%)
    "PRECTOTCORR",      # Precipitation corrected (mm/day)
    "WS2M",             # Wind speed at 2m (m/s)
    "ALLSKY_SFC_SW_DWN", # Solar radiation (MJ/m²/day)
])

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "weather" / "raw"
OUTPUT_CSV = PROJECT_ROOT / "data" / "weather" / "uduvil_weather_2004_2024.csv"


# ---------------------------------------------------------------------------
# ET₀ — Hargreaves-Samani method
# FAO Paper 56, Eq. 52 — suitable for tropical climates like Jaffna
# Requires only T_max, T_min, latitude, and day of year.
# ---------------------------------------------------------------------------

def _extraterrestrial_radiation(doy: int, lat_deg: float) -> float:
    """Return extraterrestrial radiation Ra in MJ/m²/day."""
    lat_rad = math.radians(lat_deg)
    dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365)
    declination = 0.409 * math.sin(2 * math.pi * doy / 365 - 1.39)
    ws = math.acos(-math.tan(lat_rad) * math.tan(declination))
    Ra = (
        (24 * 60 / math.pi)
        * 0.0820
        * dr
        * (
            ws * math.sin(lat_rad) * math.sin(declination)
            + math.cos(lat_rad) * math.cos(declination) * math.sin(ws)
        )
    )
    return max(0.0, Ra)


def compute_et0(t_max: float, t_min: float, doy: int, lat_deg: float) -> float:
    """Compute FAO-56 Hargreaves ET₀ in mm/day.

    ET₀ = 0.0023 × (T_mean + 17.8) × √(T_max - T_min) × Ra
    """
    t_mean = (t_max + t_min) / 2.0
    temp_range = max(0.0, t_max - t_min)
    Ra = _extraterrestrial_radiation(doy, lat_deg)
    et0 = 0.0023 * (t_mean + 17.8) * math.sqrt(temp_range) * Ra
    return round(max(0.0, et0), 2)


# ---------------------------------------------------------------------------
# NASA POWER fetch — one year at a time
# ---------------------------------------------------------------------------

def fetch_year(year: int) -> dict:
    """Fetch one year of daily data from NASA POWER. Returns parsed JSON."""
    start = f"{year}0101"
    end = f"{year}1231"

    params = {
        "parameters": PARAMETERS,
        "community": "AG",
        "longitude": LON,
        "latitude": LAT,
        "start": start,
        "end": end,
        "format": "JSON",
    }

    print(f"  Fetching {year}...", end=" ", flush=True)
    response = requests.get(NASA_POWER_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    print("done.")
    return data


def save_raw(year: int, data: dict) -> None:
    """Save raw JSON response for a year."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"uduvil_{year}.json"
    with open(path, "w") as f:
        json.dump(data, f)


def load_raw(year: int) -> dict | None:
    """Load raw JSON if already downloaded."""
    path = RAW_DIR / f"uduvil_{year}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Parse NASA POWER response → list of daily records
# ---------------------------------------------------------------------------

def parse_year(data: dict, year: int) -> list[dict]:
    """Parse one year of NASA POWER JSON into a list of daily dicts."""
    params = data["properties"]["parameter"]

    t_max_data  = params["T2M_MAX"]
    t_min_data  = params["T2M_MIN"]
    rh_data     = params["RH2M"]
    rain_data   = params["PRECTOTCORR"]
    wind_data   = params["WS2M"]
    solar_data  = params["ALLSKY_SFC_SW_DWN"]

    records = []
    current = date(year, 1, 1)
    end = date(year, 12, 31)

    while current <= end:
        key = current.strftime("%Y%m%d")

        t_max = t_max_data.get(key, None)
        t_min = t_min_data.get(key, None)

        # NASA POWER uses -999 for missing values
        if t_max is None or t_max < -900 or t_min is None or t_min < -900:
            current += timedelta(days=1)
            continue

        t_mean  = round((t_max + t_min) / 2.0, 2)
        rh      = rh_data.get(key, 70.0)
        rain_mm = rain_data.get(key, 0.0)
        wind    = wind_data.get(key, 2.0)
        solar   = solar_data.get(key, 15.0)
        doy     = current.timetuple().tm_yday
        et0     = compute_et0(t_max, t_min, doy, LAT)

        # Replace missing markers with sensible defaults
        if rh < -900:
            rh = 70.0
        if rain_mm < -900:
            rain_mm = 0.0
        if wind < -900:
            wind = 2.0
        if solar < -900:
            solar = 15.0

        records.append({
            "date":         current.isoformat(),
            "t_max":        round(t_max, 2),
            "t_min":        round(t_min, 2),
            "t_mean":       t_mean,
            "humidity_pct": round(max(0.0, min(100.0, rh)), 2),
            "rain_mm":      round(max(0.0, rain_mm), 2),
            "is_raining":   int(rain_mm > 0.5),
            "wind_speed_ms":round(max(0.0, wind), 2),
            "solar_rad_MJ": round(max(0.0, solar), 2),
            "et0_mm":       et0,
        })
        current += timedelta(days=1)

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(start_year: int = 2004, end_year: int = 2024) -> None:
    print(f"NASA POWER Weather Fetch — Uduvil, Jaffna")
    print(f"Location: lat={LAT}, lon={LON}")
    print(f"Period:   {start_year} → {end_year}")
    print(f"Output:   {OUTPUT_CSV}\n")

    all_records: list[dict] = []

    for year in range(start_year, end_year + 1):
        # Use cached raw file if already downloaded
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
            # Be polite to the API
            time.sleep(1.0)

        records = parse_year(raw, year)
        all_records.extend(records)
        print(f"  Year {year}: {len(records)} days parsed.")

    if not all_records:
        print("\nNo data collected. Check your internet connection and try again.")
        return

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDone.")
    print(f"Total days:  {len(df)}")
    print(f"Date range:  {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Saved to:    {OUTPUT_CSV}")
    print(f"\nSummary statistics:")
    print(df[["t_max", "t_min", "humidity_pct", "rain_mm", "et0_mm"]].describe().round(2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NASA POWER weather data for Uduvil.")
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year",   type=int, default=2024)
    args = parser.parse_args()

    main(args.start_year, args.end_year)
