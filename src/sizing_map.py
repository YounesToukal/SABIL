"""SABIL - reliability-based, ageing-aware sizing map (owner: Youcef).

Classical stand-alone PV sizing picks the PV and battery sizes that reach a target
loss-of-load probability (LLP). Here the LLP is computed on the CRITICAL loads only
(what Brief 13 measures) and on the AGED system, year by year over 10 years, for the
standard and the SABIL controller. Same house, same NASA POWER weather (the 10 real years
2014-2023 in sequence), two ageing models fitted on different datasets (TU Munich, NREL).

    critical LLP = share of hours in which the critical loads are not fully powered

Run from the repo root:  python src/sizing_map.py
Writes results/sizing_map.csv and results/figures/sizing_map.png
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate  # noqa: E402
from params_io import ROOT, load_all  # noqa: E402
from weather import load_weather  # noqa: E402

PV_WP = [300, 350, 400, 500]
BATTERY_KWH = [1.28, 1.92, 2.56, 3.84]   # 25.6 V modules of 50, 75, 100 and 150 Ah
CONTROLLERS = [("baseline", "none"), ("smart", "persistence")]
AGEING = ["naumann_sony", "nrel_prismatic"]  # two ageing models fitted on different datasets


def scale_ageing(battery, scalar):
    """Cells ageing `scalar` times faster: scale the fade coefficients of the active model."""
    a = battery["aging"]
    if a.get("model", "naumann_sony") == "naumann_sony":
        for k in ("cal_k_ref_per_sqrt_s", "cyc_c_a", "cyc_c_b"):
            a[k] *= scalar
    else:
        a["degradation_scalar"] = scalar


def run_case(P0, w, pv_wp, kwh, ctrl, model):
    P = copy.deepcopy(P0)
    P["system"]["pv_kwp"] = pv_wp / 1000
    P["battery"]["capacity_kwh"] = kwh
    P["battery"]["aging"]["model"] = model
    mode = "persistence" if ctrl == "smart" else "perfect"
    s, hr = simulate.run(ctrl, P, w, forecast_mode=mode)
    py = s["per_year"]
    llp = [1 - y["critical_service_frac"] for y in py]
    return {
        "controller": ctrl, "ageing_model": model, "pv_wp": pv_wp,
        "battery_kwh": kwh, "crit_llp_year1": llp[0], "crit_llp_year10": llp[-1],
        "crit_hours_lost_year1": py[0]["critical_demand_hours"] - py[0]["critical_hours"],
        "crit_hours_lost_year10": py[-1]["critical_demand_hours"] - py[-1]["critical_hours"],
        "crit_hours_lost_worst_year": max(y["critical_demand_hours"] - y["critical_hours"] for y in py),
        "unmet_kwh_year10": py[-1]["energy_unmet_kwh"],
        "capacity_after_10y": s["capacity_after_10y_frac"],
        "pv_unused_kwh_per_year": s["pv_unused_kwh_per_year"],
        "mean_soc": float(hr["soc"].mean()),
    }


def plot(df, path, model="naumann_sony"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {300: "#D55E00", 350: "#E69F00", 400: "#0072B2", 500: "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), layout="constrained", sharey=True)
    for ax, (ctrl, title) in zip(axes, [("baseline", "Standard controller"), ("smart", "SABIL controller")]):
        sub = df[(df.controller == ctrl) & (df.ageing_model == model)]
        for pv in PV_WP:
            d = sub[sub.pv_wp == pv].sort_values("battery_kwh")
            ax.plot(d.battery_kwh, d.crit_hours_lost_worst_year, marker="o", ms=4, color=colors[pv],
                    label=f"{pv} Wp")
        ax.set(title=title, xlabel="Battery size (kWh, new)", xticks=BATTERY_KWH)
        ax.title.set_fontsize(10)
        ax.set_yscale("symlog", linthresh=1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Hours without critical power, worst of 10 years")
    axes[0].legend(frameon=False, fontsize=8.5, title="PV array", title_fontsize=8.5)
    label = {"naumann_sony": "TU Munich ageing data", "nrel_prismatic": "NREL prismatic ageing data"}[model]
    fig.suptitle(f"Critical-load reliability of the aged system, {label} (Maiduguri, NASA POWER 2014-2023)",
                 fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)


def main():
    if "--plot-only" in sys.argv:
        df = pd.read_csv(ROOT / "results" / "sizing_map.csv")
        plot(df, ROOT / "results" / "figures" / "sizing_map.png")
        return
    P0 = load_all(verbose=False)
    w = load_weather(P0["sim"])
    rows = []
    for model in AGEING:
        for ctrl, _ in CONTROLLERS:
            for pv in PV_WP:
                for kwh in BATTERY_KWH:
                    rows.append(run_case(P0, w, pv, kwh, ctrl, model))
                    r = rows[-1]
                    print(f"{r['ageing_model']:14s} {ctrl:8s} {pv:4d} Wp {kwh:4.2f} kWh  "
                          f"lost h y1={r['crit_hours_lost_year1']:4d} y10={r['crit_hours_lost_year10']:4d} "
                          f"worst={r['crit_hours_lost_worst_year']:4d}  "
                          f"cap10={r['capacity_after_10y']:.3f}")
    df = pd.DataFrame(rows)
    out = ROOT / "results"
    df.round(5).to_csv(out / "sizing_map.csv", index=False)
    plot(df, out / "figures" / "sizing_map.png")


if __name__ == "__main__":
    main()
