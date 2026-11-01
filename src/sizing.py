"""SABIL - step 1: household load profile and kit sizing (owner: Youcef).

Reads  params/household.json, params/system.json and, when present,
       data/weather_<year>.csv (NASA POWER hourly, downloaded by src/weather.py)
Writes results/load_profile_24h.csv, system.json["sizing_results"],
       the design sun hours into system.json["design_inputs"],
       and the battery capacity into params/battery.json

Run from the repo root:  python src/sizing.py
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "params"
R = ROOT / "results"

# LFP cell-level efficiencies (typical round trip ~94%). Used only if battery.json lacks them.
DEFAULT_CHARGE_EFF = 0.97
DEFAULT_DISCHARGE_EFF = 0.97


def load(name):
    return json.loads((P / name).read_text())


def save(name, data):
    (P / name).write_text(json.dumps(data, indent=2) + "\n")


def hourly_profile(household, smart=False):
    """24 hourly values in W, per appliance and in total.

    smart=False: everyone uses appliances at their usual hours (standard controller).
    smart=True : flexible loads are moved, with the same energy, into their allowed sunny hours.
    """
    night = set(household["night_hours"])
    per_app = {}
    for a in household["appliances"]:
        w = a["power_w"] * a["quantity"]
        prof = [0.0] * 24
        for h in a["use_hours"]:
            prof[h] += w
        if smart and a.get("flexible"):
            moved = sum(prof[h] for h in range(24) if h in night)
            for h in range(24):
                if h in night:
                    prof[h] = 0.0
            allowed = a["allowed_hours"]
            for h in allowed:
                prof[h] += moved / len(allowed)
        per_app[a["name"]] = prof
    total = [sum(p[h] for p in per_app.values()) for h in range(24)]
    return per_app, total


def split(household, per_app, total):
    night = set(household["night_hours"])
    crit_names = {a["name"] for a in household["appliances"] if a["critical"]}
    e_day = sum(total)
    e_night = sum(total[h] for h in night)
    e_crit = sum(sum(per_app[n]) for n in crit_names)
    e_crit_night = sum(per_app[n][h] for n in crit_names for h in night)
    peak_w = max(total)
    return dict(e_day_wh=e_day, e_night_wh=e_night, e_sun_wh=e_day - e_night,
                e_critical_wh=e_crit, e_critical_night_wh=e_crit_night, peak_w=peak_w)


def sun_from_nasa(design):
    """Long-term monthly irradiation from the NASA POWER hourly years in data/, if present.

    The PV model of the simulation uses horizontal GHI, so sizing uses horizontal GHI too.
    design_psh_h = lowest long-term monthly mean (kWh/m2/day) over all available years
    (classical design-month method); worst_single_month_kwh_m2_day is reported as a check.
    """
    sim_path = P / "sim.json"
    sim = json.loads(sim_path.read_text()) if sim_path.exists() else {}
    years = sim.get("weather_years") or [sim.get("site", {}).get("weather_year", 2023)]
    by_month, days, single = {}, {}, {}
    used = []
    for year in years:
        path = ROOT / "data" / f"weather_{year}.csv"
        if not path.exists():
            continue
        used.append(year)
        ym = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                m, d = int(row["timestamp_local"][5:7]), row["timestamp_local"][:10]
                e = float(row["ghi_w_m2"]) / 1000.0
                by_month[m] = by_month.get(m, 0.0) + e
                days.setdefault(m, set()).add(d)
                ym.setdefault(m, [0.0, set()])
                ym[m][0] += e
                ym[m][1].add(d)
        for m, (e, dd) in ym.items():
            single[(year, m)] = e / len(dd)
    if not used:
        return None
    monthly = {m: by_month[m] / len(days[m]) for m in sorted(by_month)}
    worst = min(monthly, key=monthly.get)
    annual = sum(by_month.values()) / sum(len(v) for v in days.values())
    (wy, wm), wv = min(single.items(), key=lambda kv: kv[1])
    design["design_psh_h"] = round(monthly[worst], 2)
    design["annual_psh_h"] = round(annual, 2)
    design["worst_single_month_kwh_m2_day"] = {"year": wy, "month": wm, "value": round(wv, 2)}
    design["sun_source"] = (f"NASA POWER hourly GHI, Maiduguri, {used[0]}-{used[-1]} ({len(used)} years, data/): "
                            f"lowest long-term monthly mean = month {worst} at {monthly[worst]:.2f} kWh/m2/day; "
                            f"annual mean {annual:.2f}; worst single month {wm}/{wy} at {wv:.2f}. "
                            "Horizontal irradiance, as in the PV model.")
    design["monthly_ghi_kwh_m2_day"] = {str(m): round(v, 2) for m, v in monthly.items()}
    return monthly


def size(system, battery, s):
    d = system["design_inputs"]
    eta_dist = system["inverter_eff"]
    eta_c = battery.get("charge_eff") or DEFAULT_CHARGE_EFF
    eta_d = battery.get("discharge_eff") or DEFAULT_DISCHARGE_EFF

    # --- PV array: energy it must deliver on the worst-season day ---
    e_pv_direct = s["e_sun_wh"] / eta_dist                       # used as it is produced
    e_pv_stored = s["e_night_wh"] / (eta_dist * eta_c * eta_d)   # goes through the battery
    e_pv_needed = e_pv_direct + e_pv_stored
    pr = system["pv_derate"] * (1 - d["pv_temp_loss_design"])
    wp_raw = e_pv_needed / (d["design_psh_h"] * pr) * d["pv_oversize_factor"]
    step = d["pv_rounding_wp"]
    wp = math.ceil(wp_raw / step) * step

    # --- Battery: still covers one night at end of life ---
    lo, hi = d["battery_window_soc"]
    window = hi - lo
    e_from_batt = s["e_night_wh"] / eta_dist                     # at battery terminals
    kwh_raw = e_from_batt / (eta_d * window * d["battery_eol_capacity_frac"]) / 1000
    n_mod = math.ceil(kwh_raw / d["battery_module_kwh"] - 1e-9)
    kwh = n_mod * d["battery_module_kwh"]

    # --- Checks ---
    dod_new = e_from_batt / eta_d / (kwh * 1000)                  # nightly depth of discharge, new battery
    c_rate_peak = (s["peak_w"] / eta_dist) / (kwh * 1000)          # 1/h
    yield_design = wp / 1000 * d["design_psh_h"] * pr * 1000       # Wh/day
    yield_annual = wp / 1000 * d["annual_psh_h"] * pr * 1000
    return dict(
        pr_design=pr, e_pv_needed_wh=e_pv_needed, pv_wp_raw=wp_raw, pv_wp=wp,
        battery_kwh_raw=kwh_raw, battery_modules=n_mod, battery_kwh=kwh,
        nightly_dod_new=dod_new, nightly_dod_end_of_life=dod_new / d["battery_eol_capacity_frac"],
        peak_c_rate=c_rate_peak, pv_yield_design_wh=yield_design,
        pv_yield_annual_wh=yield_annual,
        pv_margin_design=yield_design / e_pv_needed - 1,
        pv_surplus_annual_wh=yield_annual - e_pv_needed,
    )


def main():
    household, system = load("household.json"), load("system.json")
    sun_from_nasa(system["design_inputs"])
    bpath = P / "battery.json"
    battery = load("battery.json") if bpath.exists() else {}

    per_app, total = hourly_profile(household, smart=False)
    s = split(household, per_app, total)
    z = size(system, battery, s)

    _, total_smart = hourly_profile(household, smart=True)
    per_s, _ = hourly_profile(household, smart=True)
    s_smart = split(household, per_s, total_smart)

    # ---- outputs ----
    R.mkdir(exist_ok=True)
    with open(R / "load_profile_24h.csv", "w", newline="") as f:
        w = csv.writer(f)
        names = list(per_app)
        w.writerow(["hour"] + names + ["total_w", "total_w_smart_shift"])
        for h in range(24):
            w.writerow([h] + [round(per_app[n][h], 2) for n in names]
                       + [round(total[h], 2), round(total_smart[h], 2)])

    # PV: energy balance gives the minimum; the reliability map may select a larger array
    system["pv_kwp"] = (system["design_inputs"].get("pv_wp_selected") or z["pv_wp"]) / 1000
    system["sizing_results"] = {
        "daily_load_wh": round(s["e_day_wh"]),
        "night_load_wh": round(s["e_night_wh"]),
        "sun_hours_load_wh": round(s["e_sun_wh"]),
        "critical_load_wh": round(s["e_critical_wh"]),
        "critical_night_load_wh": round(s["e_critical_night_wh"]),
        "peak_load_w": round(s["peak_w"]),
        "pv_wp": z["pv_wp"],
        "pv_wp_selected": round(system["pv_kwp"] * 1000),
        "battery_kwh": z["battery_kwh"],
        "battery_modules": z["battery_modules"],
        "nightly_dod_new_battery": round(z["nightly_dod_new"], 3),
        "nightly_dod_end_of_life": round(z["nightly_dod_end_of_life"], 3),
        "peak_c_rate_per_h": round(z["peak_c_rate"], 3),
        "pv_margin_worst_season": round(z["pv_margin_design"], 3),
        "night_load_wh_with_smart_shift": round(s_smart["e_night_wh"]),
        "generated_by": "src/sizing.py",
    }
    save("system.json", system)

    if not bpath.exists():
        battery = {"chemistry": "LFP", "charge_eff": DEFAULT_CHARGE_EFF,
                   "discharge_eff": DEFAULT_DISCHARGE_EFF,
                   "aging": {"status": "TO FILL in step 2"}}
    # The energy-balance rule above is only a lower bound. The selected size comes from the
    # reliability map (src/sizing_map.py: critical loss of load of the AGED system, 10 years).
    selected = system["design_inputs"].get("battery_kwh_selected")
    battery["capacity_kwh"] = selected or z["battery_kwh"]
    system["sizing_results"]["battery_kwh_energy_balance_minimum"] = z["battery_kwh"]
    system["sizing_results"]["battery_kwh_selected"] = battery["capacity_kwh"]
    if battery["capacity_kwh"] != z["battery_kwh"]:   # depth of discharge of the selected battery
        ratio = z["battery_kwh"] / battery["capacity_kwh"]
        for k in ("nightly_dod_new", "nightly_dod_end_of_life", "peak_c_rate"):
            z[k] *= ratio
        system["sizing_results"]["nightly_dod_new_battery"] = round(z["nightly_dod_new"], 3)
        system["sizing_results"]["nightly_dod_end_of_life"] = round(z["nightly_dod_end_of_life"], 3)
        system["sizing_results"]["peak_c_rate_per_h"] = round(z["peak_c_rate"], 3)
    save("system.json", system)
    battery.setdefault("eol_capacity_frac", system["design_inputs"]["battery_eol_capacity_frac"])
    save("battery.json", battery)

    # ---- report ----
    print("=== Household load (5 people, Maiduguri) ===")
    print(f"Daily energy        : {s['e_day_wh']:.0f} Wh/day")
    print(f"  in sun hours      : {s['e_sun_wh']:.0f} Wh   at night: {s['e_night_wh']:.0f} Wh "
          f"({s['e_night_wh']/s['e_day_wh']:.0%})")
    print(f"Critical loads      : {s['e_critical_wh']:.0f} Wh/day "
          f"({s['e_critical_night_wh']:.0f} Wh at night)")
    print(f"Peak load           : {s['peak_w']:.0f} W")
    print(f"Night load with smart shifting: {s_smart['e_night_wh']:.0f} Wh "
          f"({s_smart['e_night_wh']/s['e_night_wh']-1:+.0%})")
    print("\n=== Sizing (worst season) ===")
    print(f"PV energy needed    : {z['e_pv_needed_wh']:.0f} Wh/day, PR = {z['pr_design']:.3f}")
    print(f"PV array            : {z['pv_wp_raw']:.0f} Wp raw -> {z['pv_wp']} Wp installed")
    print(f"Battery             : {z['battery_kwh_raw']:.2f} kWh raw -> "
          f"{z['battery_modules']} x {system['design_inputs']['battery_module_label']} = {z['battery_kwh']:.2f} kWh")
    print("\n=== Checks ===")
    print(f"Energy balance worst season : yield {z['pv_yield_design_wh']:.0f} Wh vs need "
          f"{z['e_pv_needed_wh']:.0f} Wh -> margin {z['pv_margin_design']:+.0%}")
    print(f"Annual average surplus      : {z['pv_surplus_annual_wh']:.0f} Wh/day")
    print(f"Nightly DoD                 : {z['nightly_dod_new']:.0%} new, "
          f"{z['nightly_dod_end_of_life']:.0%} at end of life (window {system['design_inputs']['battery_window_soc']})")
    print(f"Peak C-rate                 : {z['peak_c_rate']:.2f} C")
    if system["design_inputs"].get("battery_kwh_selected"):
        print(f"\nSelected kit (reliability map, aged system): {system['pv_kwp'] * 1000:.0f} Wp, "
              f"{battery['capacity_kwh']} kWh - {system['design_inputs'].get('battery_selection_reason', '')}")


if __name__ == "__main__":
    main()
