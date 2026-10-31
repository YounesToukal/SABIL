"""Hourly simulation engine (owner: Younes).

Same house, same weather, same battery: only the controller changes, so any
difference in the results comes from the controller alone.

Weather year repeated `years` times (params/sim.json). Each hour:
  PV (DC) -> loads (through the inverter) -> surplus charges the battery -> rest is unused
  shortfall -> battery (within the controller's limits) -> what is left is unmet
Battery ageing comes only from src/aging_model.py (Youcef): battery_temp() and step_loss().

Run:
  python src/simulate.py                       # baseline + smart (both forecasts) + charts
  python src/simulate.py --controller baseline
  python src/simulate.py --years 1 --no-plots  # quick check
Outputs (results/): summary_baseline.json, summary_smart.json (headline forecast),
summary_smart_perfect.json, summary_smart_persistence.json, hourly_baseline.csv,
hourly_smart.csv, figures/*.png
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import aging_model
from controllers import Context, get_controller
from load_profile import build
from params_io import ROOT, load_all
from weather import load_weather

RESULTS = ROOT / "results"
HOURLY_COLS = ["timestamp_local", "ghi_w_m2", "ambient_c", "pv_kw", "load_kw", "served_kw",
               "unmet_kw", "battery_kw", "soc", "battery_temp_c", "capacity_frac"]


def pv_dc_kw(ghi, amb, system):
    """PV output at the DC bus, kW. Cell temperature from the NOCT model."""
    t_cell = amb + (system["noct_c"] - 20.0) / 800.0 * ghi
    p = system["pv_kwp"] * ghi / 1000.0 * system["pv_derate"] * (1.0 + system["pv_temp_coeff_per_c"] * (t_cell - 25.0))
    return np.maximum(0.0, p)


def run(name, P, weather, forecast_mode="perfect", years=None):
    sim, bat, system = P["sim"], P["battery"], P["system"]
    years = years or sim["years"]
    dt = sim["dt_h"]
    n1 = len(weather)
    n = n1 * years
    ghi = np.tile(weather["ghi_w_m2"].to_numpy(float), years)
    amb = np.tile(weather["ambient_c"].to_numpy(float), years)
    hod = np.tile(weather["timestamp_local"].dt.hour.to_numpy(), years)
    pv = pv_dc_kw(ghi, amb, system)
    L = build(P["household"], hod)
    ctx = Context(pv, amb, L["critical_kw"], L["noncritical_kw"], hod, L["flex"], bat, system, sim)
    ctrl = get_controller(name, ctx, forecast_mode)

    inv, ce, de = system["inverter_eff"], bat["charge_eff"], bat["discharge_eff"]
    cap0, eol, max_c = bat["capacity_kwh"], bat["eol_capacity_frac"], bat["max_c_rate"]
    replace = sim["replace_battery_at_eol"]
    eps = sim["critical_eps_kw"]

    cap_frac, soc = 1.0, bat["soc_max_baseline"]
    life_h, replacements = None, 0
    state = {"last_battery_temp_c": aging_model.battery_temp(amb[0], 0.0, bat)}

    out = {k: np.zeros(n) for k in ["load", "served", "unmet", "bat_kw", "soc", "temp", "cap",
                                     "crit_dem", "crit_srv", "curt", "dis_dc", "chg_dc"]}
    fc = fnc = np.zeros(24)
    day_t0 = 0
    for t in range(n):
        if t == 0 or hod[t] == 0:
            fc, fnc = ctrl.plan_day(t, ctx)
            day_t0 = t
        h = t - day_t0
        crit = ctx.crit_fixed[t] + fc[h]
        nonc = ctx.nonc_fixed[t] + fnc[h]
        load = crit + nonc
        cap = cap0 * cap_frac
        state["capacity_kwh"], state["soc"] = cap, soc
        lim = ctrl.decide(t, state, ctx)

        # PV straight to the loads
        pv_ac = pv[t] * inv
        if lim["priority"]:
            pv_c = min(pv_ac, crit)
            pv_n = min(pv_ac - pv_c, nonc)
        else:
            share = min(pv_ac, load) / load if load > 0 else 0.0
            pv_c, pv_n = crit * share, nonc * share
        surplus_dc = pv[t] - (pv_c + pv_n) / inv
        need_c, need_n = crit - pv_c, nonc - pv_n
        soc0 = soc
        chg = dis = curt = 0.0
        sup_c = sup_n = 0.0

        if surplus_dc > 1e-12:  # charge
            room = max(0.0, lim["soc_max"] - soc) * cap
            chg = min(surplus_dc, lim["chg_c"] * cap, room / (ce * dt))
            soc += chg * ce * dt / cap
            curt = surplus_dc - chg

        def available_ac(floor, c_rate, used_dc):
            stored = max(0.0, soc - floor) * cap
            dc = min(max(0.0, c_rate * cap - used_dc), stored * de / dt)
            return dc * inv

        if need_c + need_n > 1e-12:  # discharge
            if lim["priority"]:
                sup_c = min(need_c, available_ac(lim["soc_floor_crit"], lim["dis_c_crit"], 0.0))
                dis += sup_c / inv
                soc -= sup_c / inv / de * dt / cap
                rate_n = min(lim["dis_c_nonc"], max_c)
                sup_n = min(need_n, available_ac(lim["soc_floor_nonc"], rate_n, 0.0),
                            max(0.0, max_c * cap - dis) * inv)
                dis += sup_n / inv
                soc -= sup_n / inv / de * dt / cap
            else:
                need = need_c + need_n
                sup = min(need, available_ac(lim["soc_floor_crit"], lim["dis_c_crit"], 0.0))
                sup_c, sup_n = need_c * sup / need, need_n * sup / need
                dis = sup / inv
                soc -= sup / inv / de * dt / cap

        bat_kw = chg - dis
        temp = aging_model.battery_temp(amb[t], bat_kw, bat)
        loss = aging_model.step_loss(dt, 0.5 * (soc0 + soc), temp, abs(bat_kw) * dt, cap, bat)
        cap_frac -= max(0.0, loss)
        if cap_frac <= eol:
            if life_h is None:
                life_h = (t + 1) * dt
            if replace:
                replacements += 1
                cap_frac = 1.0
        state["last_battery_temp_c"] = temp

        served_c, served_n = pv_c + sup_c, pv_n + sup_n
        o = out
        o["load"][t], o["served"][t] = load, served_c + served_n
        o["unmet"][t] = load - o["served"][t]
        o["bat_kw"][t], o["soc"][t], o["temp"][t], o["cap"][t] = bat_kw, soc, temp, cap_frac
        o["crit_dem"][t], o["crit_srv"][t] = crit, served_c
        o["curt"][t], o["dis_dc"][t], o["chg_dc"][t] = curt, dis, chg

    hourly = pd.DataFrame({
        "timestamp_local": pd.date_range(weather["timestamp_local"].iloc[0], periods=n, freq="h"),
        "ghi_w_m2": ghi, "ambient_c": amb, "pv_kw": pv, "load_kw": out["load"],
        "served_kw": out["served"], "unmet_kw": out["unmet"], "battery_kw": out["bat_kw"],
        "soc": out["soc"], "battery_temp_c": out["temp"], "capacity_frac": out["cap"]})
    summary = summarise(name, forecast_mode, out, P, n1, years, life_h, replacements, eps)
    return summary, hourly


def summarise(name, mode, o, P, n1, years, life_h, replacements, eps):
    dt = P["sim"]["dt_h"]
    eol = P["battery"]["eol_capacity_frac"]
    l_per_kwh = P["costs"]["generator_l_per_kwh"]
    hours_per_year = n1 * dt
    crit_need = o["crit_dem"] > eps
    crit_ok = crit_need & (o["crit_srv"] >= o["crit_dem"] - 1e-9)

    per_year = []
    for y in range(years):
        s = slice(y * n1, (y + 1) * n1)
        served = o["served"][s].sum() * dt
        per_year.append({
            "year": y + 1,
            "energy_served_kwh": round(served, 2),
            "energy_unmet_kwh": round(o["unmet"][s].sum() * dt, 2),
            "critical_hours": int(crit_ok[s].sum()),
            "critical_demand_hours": int(crit_need[s].sum()),
            "critical_service_frac": round(float(crit_ok[s].sum() / max(1, crit_need[s].sum())), 4),
            "diesel_avoided_l": round(served * l_per_kwh, 1),
            "capacity_end_frac": round(float(o["cap"][s][-1]), 4),
        })

    cap_end = float(o["cap"][-1])
    if life_h is not None:
        life = life_h / hours_per_year
        life_note = "reached in simulation"
    else:
        start_last = 1.0 if years == 1 else float(o["cap"][(years - 1) * n1 - 1])
        fade_last = start_last - cap_end
        if fade_last > 0:
            life = years + (cap_end - eol) / fade_last
            life_note = "extrapolated with the last simulated year's fade rate"
        else:
            life, life_note = None, "no capacity fade (ageing model is still the stub)"

    total_served = o["served"].sum() * dt
    return {
        "controller": name,
        "forecast": mode if name == "smart" else "none",
        "years_simulated": years,
        "battery_life_years": None if life is None else round(life, 2),
        "battery_life_note": life_note,
        "capacity_after_10y_frac": round(cap_end, 4),
        "energy_served_kwh_per_year": round(total_served / years, 2),
        "energy_unmet_kwh_per_year": round(o["unmet"].sum() * dt / years, 2),
        "critical_service_frac": round(float(crit_ok.sum() / max(1, crit_need.sum())), 4),
        "critical_hours_per_year": round(crit_ok.sum() / years, 1),
        "critical_demand_hours_per_year": round(crit_need.sum() / years, 1),
        "diesel_avoided_l_per_year": round(total_served / years * l_per_kwh, 1),
        "pv_unused_kwh_per_year": round(o["curt"].sum() * dt / years, 2),
        "full_cycles_per_year": round(o["dis_dc"].sum() * dt / P["battery"]["capacity_kwh"] / years, 1),
        "avg_battery_temp_c": round(float(o["temp"].mean()), 2),
        "battery_replacements_in_sim": replacements,
        "per_year": per_year,
        "definitions": {
            "averages": "per-year values are averages over all simulated years (ageing included)",
            "critical_service_frac": "hours where every critical load was fully powered / hours with critical demand",
            "full_cycles_per_year": "energy discharged at the battery terminals / initial capacity, per year",
            "pv_unused_kwh_per_year": "DC solar energy available but not used (battery at its ceiling, no load)",
        },
        "params_from_placeholders": P["_used_defaults"],
    }


def save(tag, summary, hourly=None):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"summary_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    if hourly is not None:
        hourly.to_csv(RESULTS / f"hourly_{tag}.csv", index=False, float_format="%.4f",
                      date_format="%Y-%m-%d %H:%M")


def print_table(rows):
    keys = ["battery_life_years", "capacity_after_10y_frac", "critical_service_frac",
            "critical_hours_per_year", "energy_served_kwh_per_year", "energy_unmet_kwh_per_year",
            "diesel_avoided_l_per_year", "pv_unused_kwh_per_year", "full_cycles_per_year", "avg_battery_temp_c"]
    names = [f"{r['controller']}/{r['forecast']}" for r in rows]
    print("\n" + "".ljust(30) + "".join(nm.rjust(22) for nm in names))
    for k in keys:
        print(k.ljust(30) + "".join(str(r[k]).rjust(22) for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["baseline", "smart", "all"], default="all")
    ap.add_argument("--forecast", choices=["perfect", "persistence", "both"], default="both")
    ap.add_argument("--years", type=int, default=None)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    P = load_all()
    w = load_weather(P["sim"])
    headline = P["sim"]["headline_forecast"]
    rows = []
    if a.controller in ("baseline", "all"):
        t0 = time.time()
        s, hr = run("baseline", P, w, years=a.years)
        save("baseline", s, hr)
        rows.append(s)
        print(f"baseline done in {time.time() - t0:.1f} s")
    if a.controller in ("smart", "all"):
        modes = ["perfect", "persistence"] if a.forecast == "both" else [a.forecast]
        for m in modes:
            t0 = time.time()
            s, hr = run("smart", P, w, forecast_mode=m, years=a.years)
            save(f"smart_{m}", s)
            if m == headline or len(modes) == 1:
                save("smart", s, hr)
            rows.append(s)
            print(f"smart ({m}) done in {time.time() - t0:.1f} s")
    print_table(rows)
    print(f"\nheadline smart forecast: {headline} (params/sim.json)")
    if not a.no_plots:
        import plots
        plots.make_all()


if __name__ == "__main__":
    main()
