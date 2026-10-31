"""Sanity checks to run at every sync point (owner: Younes).

Run:  python src/check_results.py
Checks the contract keys and CSV columns exist, served + unmet = load,
SoC stays within limits, and battery energy is conserved hour by hour.
"""
import json
import sys

import numpy as np
import pandas as pd

from params_io import ROOT, load_all

KEYS = ["battery_life_years", "capacity_after_10y_frac", "energy_served_kwh_per_year",
        "energy_unmet_kwh_per_year", "critical_service_frac", "critical_hours_per_year",
        "diesel_avoided_l_per_year", "pv_unused_kwh_per_year", "full_cycles_per_year", "avg_battery_temp_c"]
COLS = ["timestamp_local", "ghi_w_m2", "ambient_c", "pv_kw", "load_kw", "served_kw", "unmet_kw",
        "battery_kw", "soc", "battery_temp_c", "capacity_frac"]


def main():
    P = load_all(verbose=False)
    b = P["battery"]
    ok = True
    for tag in ["baseline", "smart"]:
        s = json.load(open(ROOT / "results" / f"summary_{tag}.json"))
        missing = [k for k in KEYS if k not in s]
        h = pd.read_csv(ROOT / "results" / f"hourly_{tag}.csv")
        miss_cols = [c for c in COLS if c not in h.columns]
        bal = np.abs(h["served_kw"] + h["unmet_kw"] - h["load_kw"]).max()
        over = (h["served_kw"] - h["load_kw"]).max()
        soc_lo, soc_hi = h["soc"].min(), h["soc"].max()
        # battery energy: stored change vs what went in / out (hours where capacity did not jump)
        cap = b["capacity_kwh"] * h["capacity_frac"].to_numpy()
        stored = h["soc"].to_numpy() * cap
        bk = h["battery_kw"].to_numpy()
        flow = np.where(bk > 0, bk * b["charge_eff"], bk / b["discharge_eff"]) * P["sim"]["dt_h"]
        d_stored = np.diff(stored)
        cap_change = np.abs(np.diff(cap)) > 1e-3
        err = np.abs(d_stored - flow[1:])[~cap_change]
        checks = {
            "contract keys present": not missing,
            "hourly columns present": not miss_cols,
            "served + unmet = load": bal < 1e-6,
            "never serve more than load": over < 1e-6,
            f"soc within [{b['soc_min_baseline'] - 1e-6:.2f}, 1]": soc_lo >= b["soc_min_baseline"] - 1e-6 and soc_hi <= 1 + 1e-9,
            "battery energy conserved": float(np.percentile(err, 99.9)) < 0.01,
        }
        print(f"\n{tag}:")
        for name, passed in checks.items():
            print(f"  [{'ok' if passed else 'FAIL'}] {name}")
            ok &= passed
        if missing:
            print("   missing keys:", missing)
        if miss_cols:
            print("   missing columns:", miss_cols)
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
