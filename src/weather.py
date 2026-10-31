"""Hourly weather for the site (owner: Younes).

Source: NASA POWER hourly API, parameters ALLSKY_SFC_SW_DWN (sun) and T2M
(air temperature), downloaded in UTC and shifted by the site's UTC offset so
the year runs 00:00-23:00 local time (WAT = UTC+1). Saved to
data/weather_<year>.csv so the rest of the pipeline (and the judges) can run
offline.

Run:  python src/weather.py            (downloads if the CSV is missing)
      python src/weather.py --force    (downloads again)
"""
import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from params_io import ROOT, load_all

API = "https://power.larc.nasa.gov/api/temporal/hourly/point"
PARAMS = "ALLSKY_SFC_SW_DWN,T2M"


def cache_path(sim, root=ROOT):
    return Path(root) / "data" / f"weather_{sim['site']['weather_year']}.csv"


def _month_chunks(first_day, last_day):
    """Yield (start, end) date pairs, one per calendar month, covering the range."""
    d = first_day
    while d <= last_day:
        nxt = (d.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        end = min(last_day, nxt - dt.timedelta(days=1))
        yield d, end
        d = end + dt.timedelta(days=1)


def _fetch(lat, lon, start, end, tries=4):
    import requests
    q = {
        "parameters": PARAMS, "community": "RE",
        "longitude": lon, "latitude": lat,
        "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
        "format": "JSON", "time-standard": "UTC",
    }
    for k in range(tries):
        try:
            r = requests.get(API, params=q, timeout=120)
            if r.status_code == 200:
                return r.json()
            print(f"  HTTP {r.status_code} for {q['start']}-{q['end']}: {r.text[:300]}")
        except Exception as e:  # network hiccup: retry
            print(f"  error {e!r}, retrying")
        time.sleep(3 * (k + 1))
    raise RuntimeError("NASA POWER download failed. On Kaggle: Settings > Internet must be ON.")


def download(sim, root=ROOT):
    site = sim["site"]
    year, off = site["weather_year"], site["utc_offset_h"]
    # local year [Jan 1 00:00, Dec 31 23:00] in UTC needs one extra day on one side
    first = dt.date(year, 1, 1) - dt.timedelta(days=1 if off > 0 else 0)
    last = dt.date(year, 12, 31) + dt.timedelta(days=1 if off < 0 else 0)
    frames = []
    for s, e in _month_chunks(first, last):
        print(f"  downloading {s} .. {e}")
        js = _fetch(site["latitude"], site["longitude"], s, e)
        p = js["properties"]["parameter"]
        fill = js.get("header", {}).get("fill_value", -999.0)
        df = pd.DataFrame({"ghi_w_m2": pd.Series(p["ALLSKY_SFC_SW_DWN"]),
                           "ambient_c": pd.Series(p["T2M"])})
        df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
        frames.append(df.replace(fill, np.nan))
    utc = pd.concat(frames).sort_index()
    utc = utc[~utc.index.duplicated()]
    n_missing = int(utc.isna().sum().sum())
    utc = utc.interpolate(limit_direction="both")
    utc["ghi_w_m2"] = utc["ghi_w_m2"].clip(lower=0)
    local = utc.copy()
    local.index = local.index + pd.Timedelta(hours=off)
    local = local[local.index.year == year]
    expected = 8784 if pd.Timestamp(year=year, month=12, day=31).dayofyear == 366 else 8760
    assert len(local) == expected, f"got {len(local)} hours, expected {expected}"
    out = local.reset_index().rename(columns={"index": "timestamp_local"})
    path = cache_path(sim, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.3f")
    meta = {"source": "NASA POWER hourly API (power.larc.nasa.gov), community RE",
            "parameters": PARAMS, "time_standard": "UTC shifted to local by utc_offset_h",
            "latitude": site["latitude"], "longitude": site["longitude"],
            "utc_offset_h": off, "year": year, "missing_values_interpolated": n_missing,
            "downloaded_utc": dt.datetime.utcnow().isoformat(timespec="seconds")}
    with open(path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {path} ({len(out)} hours, {n_missing} missing values interpolated)")
    return out


def load_weather(sim, root=ROOT, force=False):
    path = cache_path(sim, root)
    if force or not path.exists():
        download(sim, root)
    w = pd.read_csv(path, parse_dates=["timestamp_local"])
    n_nan = int(w[["ghi_w_m2", "ambient_c"]].isna().sum().sum())
    if n_nan:
        print(f"WARNING - {n_nan} missing weather values, filled by interpolation")
        w[["ghi_w_m2", "ambient_c"]] = w[["ghi_w_m2", "ambient_c"]].interpolate(limit_direction="both")
    w["ghi_w_m2"] = w["ghi_w_m2"].clip(lower=0)
    return w


def describe(w):
    ghi = w["ghi_w_m2"].to_numpy()
    by_hour = w.groupby(w["timestamp_local"].dt.hour)["ghi_w_m2"].mean()
    print(f"hours: {len(w)}")
    print(f"annual irradiation: {ghi.sum() / 1000:.0f} kWh/m2 (Maiduguri should be roughly 2,000-2,300)")
    print(f"air temperature: mean {w['ambient_c'].mean():.1f} C, max {w['ambient_c'].max():.1f} C, "
          f"min {w['ambient_c'].min():.1f} C")
    print(f"sunniest local hour on average: {int(by_hour.idxmax())}:00 (should be about 12:00-13:00)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    P = load_all(verbose=False)
    describe(load_weather(P["sim"], force=a.force))
