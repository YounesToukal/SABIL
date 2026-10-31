"""Hourly simulation engine (owner: Younes).

Same house, same weather, same battery: only the controller changes, so any
difference in the results comes from the controller alone.

Weather: params/sim.json "weather_mode" = "multi_year" (each real year 2014-2023 once,
real calendar, leap years included) or "repeat" (repeat_year repeated `years` times,
as in the original contract). Each hour:
  PV (DC) -> loads (through the inverter) -> surplus charges the battery -> rest unused
  shortfall -> battery (within the controller's limits) -> what is left is unmet
Battery ageing comes only from src/aging_model.py: battery_temp() and step_loss(), plus
the optional hooks reset() and observe_soc() when the model provides them.

Run:
  python src/simulate.py                       # baseline + smart (all forecasts) + charts
  python src/simulate.py --controller baseline
  python src/simulate.py --no-plots
"""
import argparse
import copy
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


def weather_series(P, weather):
    """Hourly arrays for the whole horizon + a year index per hour."""
    sim = P["sim"]
    if sim.get("weather_mode", "multi_year") == "multi_year":
        w = weather.reset_index(drop=True)
        year_idx = w["wyear"].rank(method="dense").astype(int).to_numpy() - 1
        ts = w["timestamp_local"]
    else:
        reps = sim["years"]
        one = weather[weather["wyear"] == sim["repeat_year"]].reset_index(drop=True)
        w = pd.concat([one] * reps, ignore_index=True)
        year_idx = np.repeat(np.arange(reps), len(one))
        ts = pd.Series(pd.date_range(one["timestamp_local"].iloc[0], periods=len(w), freq="h"))
    return w, year_idx, ts.reset_index(drop=True)


def learned_pv_forecast(P, w, risk=False, ghi_forecast=None):
    """Hourly PV forecasts issued at 00:00 for the same day and for the next day (learned model).
    risk=True uses the conformal P10 (pessimistic) forecast. ghi_forecast=(same, next) GHI arrays
    overrides the file (used by the forecasting lab to test every model)."""
    amb_persist = np.roll(w["ambient_c"].to_numpy(float), 24)
    if ghi_forecast is not None:
        g0, g1 = ghi_forecast
    else:
        path = ROOT / "results" / "forecast_learned_hourly.csv"
        if not path.exists():
            return None
        f = pd.read_csv(path, parse_dates=["timestamp_local"]).set_index("timestamp_local").reindex(w["timestamp_local"])
        sfx = "_p10" if risk else ""
        if f"ghi_fc_sameday{sfx}" not in f:
            return None
        g0, g1 = f[f"ghi_fc_sameday{sfx}"].fillna(0).to_numpy(), f[f"ghi_fc_nextday{sfx}"].fillna(0).to_numpy()
    return pv_dc_kw(np.asarray(g0, float), amb_persist, P["system"]), pv_dc_kw(np.asarray(g1, float), amb_persist, P["system"])


def run(name, P, weather, forecast_mode="perfect", ghi_forecast=None):
    sim, bat, system = P["sim"], P["battery"], P["system"]
    dt = sim["dt_h"]
    w, year_idx, ts = weather_series(P, weather)
    n = len(w)
    ghi, amb = w["ghi_w_m2"].to_numpy(float), w["ambient_c"].to_numpy(float)
    hod = ts.dt.hour.to_numpy()
    pv = pv_dc_kw(ghi, amb, system)
    L = build(P["household"], hod, P["sim"].get("demand"))
    ctx = Context(pv, amb, L["critical_kw"], L["noncritical_kw"], hod, L["flex"], bat, system, sim)
    if forecast_mode in ("learned", "learned_risk"):
        fc = learned_pv_forecast(P, w, risk=forecast_mode == "learned_risk", ghi_forecast=ghi_forecast)
        if fc is None:
            raise RuntimeError("run src/forecast_lab.py first (results/forecast_learned_hourly.csv missing)")
        ctx.pv_fc_same, ctx.pv_fc_next = fc
    ctrl = get_controller(name, ctx, "learned" if forecast_mode == "learned_risk" else forecast_mode)

    inv, ce, de = system["inverter_eff"], bat["charge_eff"], bat["discharge_eff"]
    cap0, eol, max_c = bat["capacity_kwh"], bat["eol_capacity_frac"], bat["max_c_rate"]
    replace = sim["replace_battery_at_eol"]
    has_observe = hasattr(aging_model, "observe_soc")

    cap_frac, soc = 1.0, bat["soc_max_baseline"]
    if hasattr(aging_model, "reset"):
        aging_model.reset()
    if has_observe:
        aging_model.observe_soc(soc, dt)
    life_t, replacements = None, 0
    est = getattr(aging_model, "battery_temp_estimate", None)
    state = {"last_battery_temp_c": est(amb[0], bat) if est else aging_model.battery_temp(amb[0], 0.0, bat)}

    out = {k: np.zeros(n) for k in ["load", "served", "unmet", "bat_kw", "soc", "temp", "cap",
                                     "crit_dem", "crit_srv", "curt", "dis_dc", "chg_dc"]}
    fcd = fncd = np.zeros(24)
    day_t0 = 0
    for t in range(n):
        if t == 0 or hod[t] == 0:
            fcd, fncd = ctrl.plan_day(t, ctx)
            day_t0 = t
        h = t - day_t0
        crit = ctx.crit_fixed[t] + fcd[h]
        nonc = ctx.nonc_fixed[t] + fncd[h]
        load = crit + nonc
        cap = cap0 * cap_frac
        state["capacity_kwh"], state["soc"] = cap, soc
        lim = ctrl.decide(t, state, ctx)

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
        chg = dis = curt = sup_c = sup_n = 0.0

        if surplus_dc > 1e-12:
            room = max(0.0, lim["soc_max"] - soc) * cap
            chg = min(surplus_dc, lim["chg_c"] * cap, room / (ce * dt))
            soc += chg * ce * dt / cap
            curt = surplus_dc - chg

        if need_c + need_n > 1e-12:
            def available_ac(floor, c_rate):
                stored = max(0.0, soc - floor) * cap
                return min(c_rate * cap, stored * de / dt) * inv
            if lim["priority"]:
                sup_c = min(need_c, available_ac(lim["soc_floor_crit"], lim["dis_c_crit"]))
                dis += sup_c / inv
                soc -= sup_c / inv / de * dt / cap
                sup_n = min(need_n, available_ac(lim["soc_floor_nonc"], min(lim["dis_c_nonc"], max_c)),
                            max(0.0, max_c * cap - dis) * inv)
                dis += sup_n / inv
                soc -= sup_n / inv / de * dt / cap
            else:
                need = need_c + need_n
                sup = min(need, available_ac(lim["soc_floor_crit"], lim["dis_c_crit"]))
                sup_c, sup_n = need_c * sup / need, need_n * sup / need
                dis = sup / inv
                soc -= sup / inv / de * dt / cap

        bat_kw = chg - dis
        temp = aging_model.battery_temp(amb[t], bat_kw, bat)
        if has_observe:
            aging_model.observe_soc(soc, dt)
        loss = aging_model.step_loss(dt, 0.5 * (soc0 + soc), temp, abs(bat_kw) * dt, cap, bat)
        cap_frac -= max(0.0, loss)
        if cap_frac <= eol:
            if life_t is None:
                life_t = t
            if replace:
                replacements += 1
                cap_frac = 1.0
                if hasattr(aging_model, "reset"):
                    aging_model.reset()
                    if has_observe:
                        aging_model.observe_soc(soc, dt)
        state["last_battery_temp_c"] = temp

        o = out
        o["load"][t], o["served"][t] = load, pv_c + pv_n + sup_c + sup_n
        o["unmet"][t] = load - o["served"][t]
        o["bat_kw"][t], o["soc"][t], o["temp"][t], o["cap"][t] = bat_kw, soc, temp, cap_frac
        o["crit_dem"][t], o["crit_srv"][t] = crit, pv_c + sup_c
        o["curt"][t], o["dis_dc"][t], o["chg_dc"][t] = curt, dis, chg

    breakdown = aging_model.breakdown() if hasattr(aging_model, "breakdown") else None
    hourly = pd.DataFrame({"timestamp_local": ts, "ghi_w_m2": ghi, "ambient_c": amb, "pv_kw": pv,
                           "load_kw": out["load"], "served_kw": out["served"], "unmet_kw": out["unmet"],
                           "battery_kw": out["bat_kw"], "soc": out["soc"], "battery_temp_c": out["temp"],
                           "capacity_frac": out["cap"]})
    summary = summarise(name, forecast_mode, out, P, year_idx, ts, life_t, replacements, breakdown,
                        sorted(w["wyear"].unique().tolist()))
    return summary, hourly


def summarise(name, mode, o, P, year_idx, ts, life_t, replacements, breakdown, wyears):
    sim = P["sim"]
    dt, eps = sim["dt_h"], sim["critical_eps_kw"]
    eol = P["battery"]["eol_capacity_frac"]
    l_per_kwh = P["costs"]["generator_l_per_kwh"]
    co2 = sim["diesel_kg_co2_per_l"]
    years = int(year_idx.max()) + 1
    crit_need = o["crit_dem"] > eps
    crit_ok = crit_need & (o["crit_srv"] >= o["crit_dem"] - 1e-9)
    elapsed_y = ((ts - ts.iloc[0]).dt.total_seconds().to_numpy() / 3600.0 + dt) / (365.25 * 24)

    per_year = []
    for y in range(years):
        m = year_idx == y
        served = o["served"][m].sum() * dt
        per_year.append({
            "year": y + 1,
            "weather_year": int(wyears[y]) if len(wyears) == years else int(wyears[0]),
            "energy_served_kwh": round(served, 2),
            "energy_unmet_kwh": round(o["unmet"][m].sum() * dt, 2),
            "critical_hours": int(crit_ok[m].sum()),
            "critical_demand_hours": int(crit_need[m].sum()),
            "critical_service_frac": round(float(crit_ok[m].sum() / max(1, crit_need[m].sum())), 4),
            "diesel_avoided_l": round(served * l_per_kwh, 1),
            "capacity_end_frac": round(float(o["cap"][m][-1]), 4),
        })

    cap_end = float(o["cap"][-1])
    if life_t is not None:
        life, note = float(elapsed_y[life_t]), "reached in simulation"
    else:
        last = year_idx == years - 1
        start_last = 1.0 if years == 1 else float(o["cap"][year_idx == years - 2][-1])
        fade_last = start_last - cap_end
        span = float(elapsed_y[last][-1] - (elapsed_y[year_idx == years - 2][-1] if years > 1 else 0.0))
        if fade_last > 0:
            life = float(elapsed_y[-1]) + (cap_end - eol) / fade_last * span
            note = "extrapolated with the last simulated year's fade rate (conservative for square-root ageing)"
        else:
            life, note = None, "no capacity fade (ageing model is still the stub)"

    total_served = o["served"].sum() * dt
    served_py = total_served / years
    diesel_py = served_py * l_per_kwh
    s = {
        "controller": name,
        "forecast": mode if name == "smart" else "none",
        "features": P["sim"]["smart"]["features"] if name == "smart" else None,
        "years_simulated": years,
        "weather": f"{sim.get('weather_mode', 'multi_year')}: {wyears[0]}-{wyears[-1]}" if len(wyears) > 1
                   else f"repeat {wyears[0]}",
        "battery_life_years": None if life is None else round(life, 2),
        "battery_life_note": note,
        "capacity_after_10y_frac": round(cap_end, 4),
        "energy_served_kwh_per_year": round(served_py, 2),
        "energy_unmet_kwh_per_year": round(o["unmet"].sum() * dt / years, 2),
        "critical_service_frac": round(float(crit_ok.sum() / max(1, crit_need.sum())), 4),
        "critical_hours_per_year": round(crit_ok.sum() / years, 1),
        "critical_demand_hours_per_year": round(crit_need.sum() / years, 1),
        "diesel_avoided_l_per_year": round(diesel_py, 1),
        "co2_avoided_t_per_year": round(diesel_py * co2 / 1000.0, 3),
        "diesel_avoided_l_per_battery_life": None if life is None else round(diesel_py * life, 0),
        "pv_unused_kwh_per_year": round(o["curt"].sum() * dt / years, 2),
        "full_cycles_per_year": round(o["dis_dc"].sum() * dt / P["battery"]["capacity_kwh"] / years, 1),
        "avg_battery_temp_c": round(float(o["temp"].mean()), 2),
        "avg_soc": round(float(o["soc"].mean()), 4),
        "avg_soc_when_hot": round(float(o["soc"][o["temp"] >= sim["smart"]["hot_threshold_c"]].mean()), 4)
        if (o["temp"] >= sim["smart"]["hot_threshold_c"]).any() else None,
        "ageing_breakdown": breakdown,
        "battery_replacements_in_sim": replacements,
        "per_year": per_year,
        "definitions": {
            "averages": "per-year values are averages over all simulated years (ageing included)",
            "critical_service_frac": "hours where every critical load was fully powered / hours with critical demand",
            "full_cycles_per_year": "energy discharged at the battery terminals / initial capacity, per year",
            "pv_unused_kwh_per_year": "DC solar energy available but not used (battery at its ceiling, no load)",
            "diesel_avoided_l_per_battery_life": "diesel_avoided_l_per_year x battery_life_years",
            "co2_avoided_t_per_year": "diesel avoided x diesel_kg_co2_per_l (IPCC 2006 default)",
        },
        "params_from_placeholders": P["_used_defaults"],
    }
    return s


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
            "diesel_avoided_l_per_year", "diesel_avoided_l_per_battery_life", "full_cycles_per_year",
            "avg_soc_when_hot", "avg_battery_temp_c"]
    names = [f"{r['controller']}/{r['forecast']}" for r in rows]
    print("\n" + "".ljust(34) + "".join(nm.rjust(20) for nm in names))
    for k in keys:
        print(k.ljust(34) + "".join(str(r[k]).rjust(20) for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["baseline", "smart", "all"], default="all")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    P = load_all()
    w = load_weather(P["sim"])
    headline = P["sim"]["headline_forecast"]
    rows = []
    if a.controller in ("baseline", "all"):
        t0 = time.time()
        s, hr = run("baseline", P, w)
        save("baseline", s, hr)
        rows.append(s)
        print(f"baseline done in {time.time() - t0:.1f} s")
    if a.controller in ("smart", "all"):
        modes = ["perfect", "persistence"]
        fpath = ROOT / "results" / "forecast_learned_hourly.csv"
        if fpath.exists():
            modes.append("learned")
            if "ghi_fc_sameday_p10" in pd.read_csv(fpath, nrows=1).columns:
                modes.append("learned_risk")
        for m in modes:
            t0 = time.time()
            s, hr = run("smart", copy.deepcopy(P), w, forecast_mode=m)
            save(f"smart_{m}", s)
            if m == headline:
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
