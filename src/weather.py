"""Hourly weather for the site (owner: Younes).

Source: NASA POWER hourly API (power.larc.nasa.gov): ALLSKY_SFC_SW_DWN (sun, W/m2),
T2M (air temperature, C) and, when available, RH2M (relative humidity, %), downloaded
in UTC and shifted by the site's UTC offset so each year runs 00:00-23:00 local time.
One file per year: data/weather_<year>.csv (the pipeline and the judges run offline).
A clear-sky irradiance column (Haurwitz model, computed, not downloaded) is added for
the learned forecast's clearness index.

Run:  python src/weather.py           (downloads the missing years)
      python src/weather.py --force   (downloads everything again)
"""
import argparse
import datetime as dt
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from params_io import ROOT, load_all

API = "https://power.larc.nasa.gov/api/temporal/hourly/point"
CORE, EXTRA = "ALLSKY_SFC_SW_DWN,T2M", "RH2M"
COLS = ["timestamp_local", "ghi_w_m2", "ambient_c", "rh_pct", "ghi_clearsky_w_m2"]


def years_needed(sim):
    if sim.get("weather_mode", "multi_year") == "multi_year":
        return list(sim["weather_years"])
    return [sim["repeat_year"]]


def year_path(year, root=ROOT):
    return Path(root) / "data" / f"weather_{year}.csv"


def _month_chunks(first_day, last_day):
    d = first_day
    while d <= last_day:
        nxt = (d.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        end = min(last_day, nxt - dt.timedelta(days=1))
        yield d, end
        d = end + dt.timedelta(days=1)


def _get(lat, lon, start, end, params, tries=5):
    import requests
    q = {"parameters": params, "community": "RE", "longitude": lon, "latitude": lat,
         "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
         "format": "JSON", "time-standard": "UTC"}
    last = ""
    for k in range(tries):
        try:
            r = requests.get(API, params=q, timeout=180)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code in (400, 422):
                break
        except Exception as e:
            last = repr(e)
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"NASA POWER {start}-{end} failed ({last}). On Kaggle: Settings > Internet ON.")


def _fetch_chunk(args):
    lat, lon, s, e = args
    try:
        js = _get(lat, lon, s, e, CORE + "," + EXTRA)
    except RuntimeError:
        js = _get(lat, lon, s, e, CORE)            # RH2M unavailable: core only
    p = js["properties"]["parameter"]
    fill = js.get("header", {}).get("fill_value", -999.0)
    df = pd.DataFrame({"ghi_w_m2": pd.Series(p["ALLSKY_SFC_SW_DWN"]),
                       "ambient_c": pd.Series(p["T2M"]),
                       "rh_pct": pd.Series(p.get("RH2M", {}), dtype=float)})
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
    return df.replace(fill, np.nan)


def clearsky_ghi(ts_local, lat, lon, utc_offset_h):
    """Haurwitz clear-sky GHI (W/m2) at mid-hour, from solar geometry (NOAA approximations)."""
    t = pd.DatetimeIndex(ts_local)
    doy = t.dayofyear.to_numpy()
    hour = t.hour.to_numpy() + 0.5
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24)
    decl = 0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) \
        + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g)
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                    - 0.040849 * np.sin(2 * g))
    solar_time = hour * 60 + eot + 4 * lon - 60 * utc_offset_h
    ha = np.radians(solar_time / 4 - 180)
    phi = np.radians(lat)
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cs = np.where(cosz > 0.01, 1098.0 * cosz * np.exp(-0.059 / np.maximum(cosz, 0.01)), 0.0)
    return cs


def download_year(sim, year, root=ROOT, workers=4):
    site = sim["site"]
    off = site["utc_offset_h"]
    first = dt.date(year, 1, 1) - dt.timedelta(days=1 if off > 0 else 0)
    last = dt.date(year, 12, 31) + dt.timedelta(days=1 if off < 0 else 0)
    jobs = [(site["latitude"], site["longitude"], s, e) for s, e in _month_chunks(first, last)]
    with ThreadPoolExecutor(workers) as ex:
        frames = list(ex.map(_fetch_chunk, jobs))
    utc = pd.concat(frames).sort_index()
    utc = utc[~utc.index.duplicated()]
    n_missing = int(utc[["ghi_w_m2", "ambient_c"]].isna().sum().sum())
    utc[["ghi_w_m2", "ambient_c"]] = utc[["ghi_w_m2", "ambient_c"]].interpolate(limit_direction="both")
    if utc["rh_pct"].notna().any():
        utc["rh_pct"] = utc["rh_pct"].interpolate(limit_direction="both")
    utc["ghi_w_m2"] = utc["ghi_w_m2"].clip(lower=0)
    local = utc.copy()
    local.index = local.index + pd.Timedelta(hours=off)
    local = local[local.index.year == year]
    expected = 8784 if pd.Timestamp(year=year, month=12, day=31).dayofyear == 366 else 8760
    assert len(local) == expected, f"{year}: got {len(local)} hours, expected {expected}"
    out = local.reset_index().rename(columns={"index": "timestamp_local"})
    out["ghi_clearsky_w_m2"] = clearsky_ghi(out["timestamp_local"], site["latitude"], site["longitude"], off)
    path = year_path(year, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    out[COLS].to_csv(path, index=False, float_format="%.3f")
    meta = {"source": "NASA POWER hourly API (power.larc.nasa.gov), community RE",
            "parameters": CORE + ("," + EXTRA if out["rh_pct"].notna().any() else ""),
            "time_standard": "UTC shifted to local by utc_offset_h", "latitude": site["latitude"],
            "longitude": site["longitude"], "utc_offset_h": off, "year": year,
            "missing_values_interpolated": n_missing,
            "downloaded_utc": dt.datetime.utcnow().isoformat(timespec="seconds")}
    json.dump(meta, open(path.with_suffix(".meta.json"), "w"), indent=2)
    print(f"  saved {path.name}: {len(out)} hours, {n_missing} gaps filled")


def _read_year(sim, year, root):
    w = pd.read_csv(year_path(year, root), parse_dates=["timestamp_local"])
    for c in ["rh_pct", "ghi_clearsky_w_m2"]:
        if c not in w:
            w[c] = np.nan
    if w["ghi_clearsky_w_m2"].isna().all():
        s = sim["site"]
        w["ghi_clearsky_w_m2"] = clearsky_ghi(w["timestamp_local"], s["latitude"], s["longitude"], s["utc_offset_h"])
    w[["ghi_w_m2", "ambient_c"]] = w[["ghi_w_m2", "ambient_c"]].interpolate(limit_direction="both")
    w["ghi_w_m2"] = w["ghi_w_m2"].clip(lower=0)
    w["wyear"] = year
    return w[COLS + ["wyear"]]


def load_weather(sim, root=ROOT, force=False):
    """All weather years needed, concatenated in order (column 'wyear' = weather year)."""
    for y in years_needed(sim):
        p = year_path(y, root)
        stale = p.exists() and "rh_pct" not in pd.read_csv(p, nrows=1).columns
        if force or not p.exists() or stale:
            print(f"downloading {y} from NASA POWER ...")
            download_year(sim, y, root)
    return pd.concat([_read_year(sim, y, root) for y in years_needed(sim)], ignore_index=True)


def describe(w):
    print(f"{'year':>6} {'kWh/m2':>8} {'mean C':>7} {'max C':>6} {'peak hour':>9} {'RH %':>6}")
    for y, g in w.groupby("wyear"):
        peak = g.groupby(g["timestamp_local"].dt.hour)["ghi_w_m2"].mean().idxmax()
        rh = g["rh_pct"].mean()
        print(f"{y:>6} {g['ghi_w_m2'].sum() / 1000:>8.0f} {g['ambient_c'].mean():>7.1f} "
              f"{g['ambient_c'].max():>6.1f} {int(peak):>7}:00 {rh:>6.1f}")
    print("check: Maiduguri ~2,000-2,300 kWh/m2 per year; sunniest hour 12:00-13:00 local")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    P = load_all(verbose=False)
    describe(load_weather(P["sim"], force=a.force))
