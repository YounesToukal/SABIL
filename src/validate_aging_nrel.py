"""SABIL - checks of the NREL prismatic LFP ageing model (src/aging_nrel.py) and first physics results (owner: Youcef).

Run from the repo root:
    python src/validate_aging.py                      # model checks + Maiduguri scenarios
    python src/validate_aging.py --blast-path DIR     # also cross-check against NREL BLAST-Lite
                                                      # (DIR = folder containing the `blast` package)
Writes results/aging_validation_nrel.json and results/figures/capacity_fade_scenarios.png
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import aging_nrel as am  # noqa: E402

P, R = ROOT / "params", ROOT / "results"

# Monthly mean, mean max and mean min air temperature (C). 1991-2020 normals, Diffa station
# (~180 km from Maiduguri), climate-zone.com. Replaced by NASA POWER hourly data in simulate.py.
T_MEAN = [22.2, 25.6, 30.0, 33.3, 34.4, 33.3, 30.6, 28.3, 30.0, 30.6, 27.2, 23.3]
T_MAX = [30.6, 33.9, 38.9, 41.7, 42.2, 40.0, 36.1, 33.3, 36.1, 38.9, 36.1, 31.7]
T_MIN = [13.9, 16.7, 21.1, 24.4, 26.1, 27.2, 25.0, 23.3, 23.9, 22.2, 18.3, 15.0]
DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def fresh_battery(**overrides):
    b = json.loads((P / "battery.json").read_text())
    b.update(overrides)
    am.reset_state(b)
    return b


def ambient(month, hour):
    """Hourly air temperature: cosine between the monthly mean min and mean max, peak at 15:00."""
    lo, hi = T_MIN[month], T_MAX[month]
    return (lo + hi) / 2 + (hi - lo) / 2 * math.cos(2 * math.pi * (hour - 15) / 24)


STRATEGIES = {
    # name: (charge ceiling, charging starts, ceiling reached)  - illustrative, NOT the controller
    "standard": (1.00, 7, 12),   # fills to 100% by noon, sits full through the afternoon heat
    "late_95": (0.95, 11, 17),   # same energy, charged later, small ceiling cut
    "capped": (None, 11, 17),    # ceiling = what tonight needs + reserve (sunny day ahead)
}
RESERVE_SOC = 0.25  # kept in the battery at dawn for critical loads (cloudy day / line outage)


def daily_soc(strategy, dod):
    """Illustrative 24-h SoC path (hour 0..23): charge, hold, discharge 18:00-06:00."""
    top, start, full = STRATEGIES[strategy]
    if top is None:
        top = min(1.0, RESERVE_SOC + dod)
    low = top - dod
    soc = []
    for h in range(24):
        if start <= h < full:
            soc.append(low + (top - low) * (h - start + 1) / (full - start))
        elif full <= h < 18:
            soc.append(top)
        else:  # discharge 18..23, 0..6 (13 h), back to `low` at 07:00
            k = (h - 18) % 24 + 1
            soc.append(max(top - dod * k / 13, low))
    return soc


def run(battery, soc_day, years=10, temp_fn=None, power_scale=True):
    """Hourly integration over `years`; returns yearly capacity fractions and final state."""
    e_nom = battery["capacity_kwh"]
    caps, prev = [1.0], soc_day[-1]
    for _ in range(years):
        for m in range(12):
            for _d in range(DAYS[m]):
                for h in range(24):
                    soc = soc_day[h]
                    q = am.get_state(battery)["capacity_frac"]
                    thr = abs(soc - prev) * e_nom * q          # kWh through the cells this hour
                    p_kw = (soc - prev) * e_nom * q              # + charge / - discharge
                    t_amb = temp_fn(m, h) if temp_fn else ambient(m, h)
                    t_bat = am.battery_temp(t_amb, p_kw, battery)
                    am.step_loss(1.0, (soc + prev) / 2, t_bat, thr, e_nom * q, battery)
                    prev = soc
        caps.append(am.get_state(battery)["capacity_frac"])
    return caps, am.get_state(battery)


def years_to(caps, level=0.8):
    for y in range(1, len(caps)):
        if caps[y] <= level:
            f = (caps[y - 1] - level) / (caps[y - 1] - caps[y])
            return y - 1 + f
    return None


def check_closed_form():
    """Constant conditions, hourly steps: must equal k * t**p exactly (tests the time integration)."""
    b = fresh_battery()
    a = b["aging"]
    for _ in range(365 * 24):
        am.step_loss(1.0, 0.5, 25.0, 0.0, b["capacity_kwh"], b)
    expected = am.k_calendar(25.0, 0.5, a) * 365 ** a["p_cal"]
    got = am.get_state(b)["q_cal"]
    # changing conditions: 1 year at 45 C after 1 year at 25 C must equal 'virtual time' result
    for _ in range(365 * 24):
        am.step_loss(1.0, 0.5, 45.0, 0.0, b["capacity_kwh"], b)
    k45 = am.k_calendar(45.0, 0.5, a)
    t_eq = (expected / k45) ** (1 / a["p_cal"])
    expected2 = k45 * (t_eq + 365) ** a["p_cal"]
    got2 = am.get_state(b)["q_cal"]
    return {"calendar_1y_25C_soc50": {"model": got, "closed_form": expected,
                                      "rel_error": abs(got - expected) / expected},
            "then_1y_45C": {"model": got2, "virtual_time_closed_form": expected2,
                            "rel_error": abs(got2 - expected2) / expected2}}


def check_storage_table():
    """1-year calendar fade (% of capacity) vs storage temperature and SoC."""
    b = fresh_battery()
    a = b["aging"]
    out = {}
    for T in (25, 35, 45):
        out[f"{T}C"] = {f"soc{int(s*100)}": round(100 * am.k_calendar(T, s, a) * 365 ** a["p_cal"], 2)
                        for s in (0.3, 0.5, 0.8, 1.0)}
    return out


def check_datasheet_like():
    """25 C, one 80%-DoD cycle per day at ~0.25 C: years and cycles to 80% capacity."""
    b = fresh_battery()
    soc = []
    for h in range(24):  # charge 10-13 (to 0.9), discharge 19-22 (to 0.1): 0.2 C each way
        if 10 <= h < 14:
            soc.append(0.1 + 0.8 * (h - 9) / 4)
        elif 14 <= h < 19:
            soc.append(0.9)
        elif 19 <= h < 23:
            soc.append(0.9 - 0.8 * (h - 18) / 4)
        else:
            soc.append(0.1)
    caps, st = run(b, soc, years=20, temp_fn=lambda m, h: 25.0 - b["temp_offset_c"])
    y = years_to(caps)
    return {"years_to_80pct": round(y, 1) if y else None,
            "efc_to_80pct": round(y * 365 * 0.8) if y else None,
            "split_after_20y": {"calendar": round(st["q_cal"], 3), "cycle": round(st["q_cyc"], 3)}}


def scenarios():
    sys_r = json.loads((P / "system.json").read_text())["sizing_results"]
    dod_std = sys_r["nightly_dod_new_battery"]
    dod_smart = dod_std * sys_r["night_load_wh_with_smart_shift"] / sys_r["night_load_wh"]
    res, curves = {}, {}
    hot25 = lambda m, h: 25.0  # noqa: E731
    cases = {
        # one lever at a time, so each effect can be attributed
        "A_standard": ("standard", dod_std, None, {}),
        "B_charge_later_95": ("late_95", dod_std, None, {}),
        "C_ceiling_capped": ("capped", dod_std, None, {}),
        "D_capped_plus_load_shift": ("capped", dod_smart, None, {}),
        "E_standard_at_25C_reference": ("standard", dod_std, hot25, {"temp_offset_c": 0.0}),
        "F_standard_battery_offset_0C": ("standard", dod_std, None, {"temp_offset_c": 0.0}),
        "G_standard_battery_offset_6C": ("standard", dod_std, None, {"temp_offset_c": 6.0}),
        "H_standard_lowgrade_cells_x2": ("standard", dod_std, None, {"x2": True}),
        "I_smart_lowgrade_cells_x2": ("capped", dod_smart, None, {"x2": True}),
    }
    years = 15
    for name, (strat, dod, tf, ov) in cases.items():
        ov = dict(ov)
        b = fresh_battery(**{k: v for k, v in ov.items() if k != "x2"})
        if ov.get("x2"):
            b["aging"]["degradation_scalar"] = 2.0
        soc = daily_soc(strat, dod)
        caps, st = run(b, soc, years=years, temp_fn=tf)
        curves[name] = caps
        y80 = years_to(caps)
        res[name] = {"ceiling_soc": round(max(soc), 3), "dawn_soc": round(min(soc), 3),
                     "mean_soc": round(sum(soc) / 24, 3),
                     "capacity_after_10y": round(caps[10], 3),
                     "years_to_80pct": round(y80, 1) if y80 else f">{years}",
                     "calendar_share_of_fade": round(st["q_cal"] / (st["q_cal"] + st["q_cyc"]), 2),
                     "efc_per_year": round(st["efc"] / years)}
    a, d = res["A_standard"]["capacity_after_10y"], res["D_capped_plus_load_shift"]["capacity_after_10y"]
    res["summary"] = {
        "fade_10y_standard_pct": round(100 * (1 - a), 1),
        "fade_10y_smart_pct": round(100 * (1 - d), 1),
        "fade_reduction_pct": round(100 * (1 - (1 - d) / (1 - a)), 0),
        "dod_standard": round(dod_std, 3), "dod_smart": round(dod_smart, 3),
        "reserve_soc_at_dawn_smart": RESERVE_SOC,
    }
    return res, curves


def cross_check_blast(blast_dir):
    """Same SoC/temperature inputs through NREL's own implementation (constant 30 C)."""
    import types
    import numpy as np
    sys.path.insert(0, blast_dir)
    # Stubs for packages BLAST-Lite imports only for its weather-data download helpers
    sys.modules.setdefault("h5pyd", types.ModuleType("h5pyd"))
    geo = types.ModuleType("geopy.geocoders")
    geo.Nominatim = object
    sys.modules.setdefault("geopy", types.ModuleType("geopy"))
    sys.modules.setdefault("geopy.geocoders", geo)
    if not hasattr(np, "trapz"):  # removed in recent NumPy; BLAST-Lite still calls it
        np.trapz = np.trapezoid
    from blast.models.lfp_gr_250AhPrismatic_2019 import Lfp_Gr_250AhPrismatic

    dod = json.loads((P / "system.json").read_text())["sizing_results"]["nightly_dod_new_battery"]
    soc_day = daily_soc("standard", dod)
    days = 365
    soc = np.array(soc_day * days + [soc_day[0]], dtype=float)
    t = np.arange(len(soc), dtype=float) * 3600
    temp = np.full(len(soc), 30.0)
    ref = Lfp_Gr_250AhPrismatic()
    ref.simulate_battery_life({"Time_s": t, "SOC": soc, "Temperature_C": temp},
                              threshold_time=10, is_conserve_energy_throughput=False)
    t_years = np.array(ref.stressors["t_days"]) / 365
    q_ref = np.array(ref.outputs["q"])

    b = fresh_battery(temp_offset_c=0.0, thermal={"r_th_k_per_w": 0.0})
    caps, _ = run(b, soc_day, years=10, temp_fn=lambda m, h: 30.0)
    out = {}
    for y in (1, 5, 10):
        q_at = float(np.interp(y, t_years, q_ref))
        out[f"year_{y}"] = {"blast_lite": round(q_at, 4), "ours": round(caps[y], 4),
                            "diff_pct_points": round(100 * (caps[y] - q_at), 2)}
    return out


def plot(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    years = list(range(len(next(iter(curves.values())))))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), layout="constrained", sharey=True)
    panels = [
        ("Premium LFP cells (as tested by NREL)",
         [("A_standard", "Standard controller", "#D55E00", "-", "o"),
          ("D_capped_plus_load_shift", "Battery-aware controller", "#0072B2", "-", "s"),
          ("E_standard_at_25C_reference", "Standard, 25 °C reference", "#555555", "--", None)]),
        ("Sensitivity: cells ageing 2x faster",
         [("H_standard_lowgrade_cells_x2", "Standard controller", "#D55E00", "-", "o"),
          ("I_smart_lowgrade_cells_x2", "Battery-aware controller", "#0072B2", "-", "s")]),
    ]
    for ax, (title, lines) in zip(axes, panels):
        for k, lab, c, ls, mk in lines:
            ax.plot(years, [100 * v for v in curves[k]], color=c, ls=ls, marker=mk, ms=3.5, label=lab)
        ax.axhline(80, color="#999999", lw=1, ls=":")
        ax.text(0.3, 80.5, "End of life (80%)", color="#555555", fontsize=8.5)
        ax.set(title=title, xlabel="Years in service", xlim=(0, years[-1]), ylim=(60, 100.5))
        ax.title.set_fontsize(10)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    axes[0].set_ylabel("Usable capacity (% of new)")
    fig.suptitle("Maiduguri climate, 1.28 kWh LFP, illustrative daily cycle (not the final simulation)",
                 fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blast-path", default=None)
    args = ap.parse_args()

    report = {"closed_form": check_closed_form(),
              "calendar_fade_1y_pct": check_storage_table(),
              "datasheet_like_25C": check_datasheet_like()}
    report["scenarios"], curves = scenarios()
    if args.blast_path:
        report["cross_check_blast_lite_30C"] = cross_check_blast(args.blast_path)

    R.mkdir(exist_ok=True)
    (R / "aging_validation_nrel.json").write_text(json.dumps(report, indent=2) + "\n")
    plot(curves, R / "figures" / "capacity_fade_scenarios.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
