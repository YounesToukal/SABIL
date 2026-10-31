"""Extra open datasets for the forecasting lab (owner: Younes). All free, no API key.

1. ERA5 reanalysis (ECMWF, via the Open-Meteo archive API), hourly, local time:
   cloud cover (total/low/mid/high), dew point, humidity, surface pressure, rain,
   wind speed and direction (Harmattan from the north-east vs monsoon from the south-west),
   vapour-pressure deficit, top-layer soil moisture (monsoon onset), and ERA5's own
   solar radiation (an independent estimate of the sun next to NASA POWER).
   -> data/era5_<year>.csv
2. NOAA climate indices, monthly: ONI (El Nino / La Nina), NAO (North Atlantic
   Oscillation), TNA (Tropical North Atlantic SST). Large-scale drivers of the West
   African monsoon; tested in the ablation even if they turn out not to help.
   -> data/climate_indices.csv
Every source is optional: if a download fails, the lab skips that feature group.

Run:  python src/datasets.py
"""
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd

from params_io import ROOT, load_all

DATA = ROOT / "data"
ERA5_API = "https://archive-api.open-meteo.com/v1/archive"
# canonical column -> (current Open-Meteo name, legacy name)
ERA5_VARS = {
    "t2m": ("temperature_2m", "temperature_2m"),
    "rh": ("relative_humidity_2m", "relativehumidity_2m"),
    "dewpoint": ("dew_point_2m", "dewpoint_2m"),
    "psurf": ("surface_pressure", "surface_pressure"),
    "cloud": ("cloud_cover", "cloudcover"),
    "cloud_low": ("cloud_cover_low", "cloudcover_low"),
    "cloud_mid": ("cloud_cover_mid", "cloudcover_mid"),
    "cloud_high": ("cloud_cover_high", "cloudcover_high"),
    "precip": ("precipitation", "precipitation"),
    "wind": ("wind_speed_10m", "windspeed_10m"),
    "wind_dir": ("wind_direction_10m", "winddirection_10m"),
    "vpd": ("vapour_pressure_deficit", "vapor_pressure_deficit"),
    "soil_moist": ("soil_moisture_0_to_7cm", "soil_moisture_0_to_7cm"),
    "sw_era5": ("shortwave_radiation", "shortwave_radiation"),
    "diffuse_era5": ("diffuse_radiation", "diffuse_radiation"),
}
MINIMAL = ["t2m", "cloud", "precip", "sw_era5"]


def _http(url, params, tries=4):
    import requests
    last = ""
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=180)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code == 400:
                break
        except Exception as e:
            last = repr(e)
        time.sleep(3 * (k + 1))
    raise RuntimeError(last)


def fetch_era5_year(sim, year):
    path = DATA / f"era5_{year}.csv"
    if path.exists():
        return path
    s = sim["site"]
    base = {"latitude": s["latitude"], "longitude": s["longitude"], "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31", "timezone": "Africa/Lagos"}
    attempts = [list(ERA5_VARS), list(ERA5_VARS), MINIMAL]
    for i, keys in enumerate(attempts):
        names = [ERA5_VARS[k][1 if i == 1 else 0] for k in keys]
        try:
            js = _http(ERA5_API, {**base, "hourly": ",".join(names)}).json()
        except RuntimeError as e:
            print(f"  ERA5 {year}: attempt {i + 1} failed ({e})")
            continue
        h = js["hourly"]
        df = pd.DataFrame({"timestamp_local": pd.to_datetime(h["time"])})
        for k, nm in zip(keys, names):
            if nm in h:
                df[k] = pd.to_numeric(pd.Series(h[nm]), errors="coerce")
        df = df.interpolate(limit_direction="both")
        DATA.mkdir(exist_ok=True)
        df.to_csv(path, index=False, float_format="%.3f")
        print(f"  saved {path.name}: {len(df)} hours, {len(df.columns) - 1} variables")
        return path
    return None


def fetch_indices():
    path = DATA / "climate_indices.csv"
    if path.exists():
        return path
    frames = []
    try:   # ONI: 3-month running seasons, centred month
        txt = _http("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", {}).text
        seas = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
        rows = []
        for line in txt.splitlines():
            p = line.split()
            if len(p) == 4 and p[0] in seas:
                rows.append({"year": int(p[1]), "month": seas.index(p[0]) + 1, "oni": float(p[3])})
        frames.append(pd.DataFrame(rows))
    except Exception as e:
        print("  ONI unavailable:", e)
    try:   # NAO monthly
        txt = _http("https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii", {}).text
        rows = [{"year": int(p[0]), "month": int(p[1]), "nao": float(p[2])}
                for p in (l.split() for l in txt.splitlines()) if len(p) == 3]
        frames.append(pd.DataFrame(rows))
    except Exception as e:
        print("  NAO unavailable:", e)
    try:   # TNA monthly (PSL format: year + 12 values)
        txt = _http("https://psl.noaa.gov/data/correlation/tna.data", {}).text
        rows = []
        for l in txt.splitlines():
            p = l.split()
            if len(p) == 13 and p[0].isdigit() and len(p[0]) == 4:
                for m, v in enumerate(p[1:], start=1):
                    v = float(v)
                    if v > -90:
                        rows.append({"year": int(p[0]), "month": m, "tna": v})
        frames.append(pd.DataFrame(rows))
    except Exception as e:
        print("  TNA unavailable:", e)
    if not frames:
        return None
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["year", "month"], how="outer")
    DATA.mkdir(exist_ok=True)
    out.sort_values(["year", "month"]).to_csv(path, index=False)
    print(f"  saved {path.name}: {len(out)} months, columns {list(out.columns[2:])}")
    return path


def load_era5(sim):
    frames = []
    for y in sim["weather_years"]:
        p = DATA / f"era5_{y}.csv"
        if p.exists():
            frames.append(pd.read_csv(p, parse_dates=["timestamp_local"]))
    return pd.concat(frames, ignore_index=True) if frames else None


def load_indices():
    p = DATA / "climate_indices.csv"
    return pd.read_csv(p) if p.exists() else None


def main():
    sim = load_all(verbose=False)["sim"]
    print("ERA5 (Open-Meteo archive):")
    for y in sim["weather_years"]:
        fetch_era5_year(sim, y)
    print("NOAA climate indices:")
    fetch_indices()


if __name__ == "__main__":
    main()
