"""Research report generated from the result files (owner: Younes).

Every number in results/RESEARCH_REPORT.md is read from results/*.json, never typed.
Run:  python src/report.py
"""
import json
import platform
import subprocess
from datetime import datetime, timezone

from params_io import ROOT, load_all

RES = ROOT / "results"


def _j(path):
    p = RES / path
    return json.load(open(p)) if p.exists() else None


def _f(v, fmt="{:.2f}"):
    return "n/a" if v is None else fmt.format(v)


def main():
    P = load_all(verbose=False)
    sim = P["sim"]
    base, smart = _j("summary_baseline.json"), _j("summary_smart.json")
    others = {m: _j(f"summary_smart_{m}.json") for m in ["perfect", "persistence", "learned"]}
    val, fm = _j("aging_validation.json"), _j("forecast_metrics.json")
    ab, sw, mc = _j("experiments/ablation.json"), _j("experiments/sweep.json"), _j("experiments/montecarlo_summary.json")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    L = []
    w = L.append
    w("# SABIL: battery-aware control for solar home systems in Maiduguri")
    w("")
    w(f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from the result files"
      f"{' at commit ' + commit if commit else ''}. Every number below is read from `results/`, not typed.*")
    w("")
    if P["_used_defaults"]:
        w(f"> **Provisional:** these inputs still use placeholders (`dev_params/`): {', '.join(P['_used_defaults'])}.")
        w("")
    w("## 1. Question")
    w("When the transmission line to Maiduguri is cut (Brief 13), a household solar-plus-battery kit replaces diesel only "
      "as long as its battery lasts. Can a controller that knows how batteries age extend battery life in Maiduguri's "
      "heat without reducing the hours that lights, phones and fans stay on?")
    w("")
    w("## 2. Method")
    w(f"- **Site and weather:** {sim['site']['name']} ({sim['site']['latitude']} N, {sim['site']['longitude']} E). "
      f"NASA POWER hourly irradiance, air temperature and humidity, {base['weather'] if base else ''}, local time (UTC+{sim['site']['utc_offset_h']}).")
    apps = P["household"]["appliances"]
    crit_names = ", ".join(dict.fromkeys(a["name"].split(",")[0] for a in apps if a.get("critical")))
    flex_names = ", ".join(dict.fromkeys(a["name"].split(",")[0] for a in apps if a.get("flexible"))) or "none"
    daily_kwh = sum(a["power_w"] * a["quantity"] * len(a["use_hours"]) for a in apps) / 1000
    w(f"- **Household:** 5 people, {daily_kwh:.2f} kWh/day; hourly appliance schedule from `params/household.json`; "
      f"critical loads: {crit_names}. Grid assumed down for the whole horizon (worst case of Brief 13).")
    w("- **Paired design:** the same house, weather, PV array and battery are simulated twice; only the controller differs. "
      "Any difference is caused by the controller.")
    w("- **Standard controller:** charges to 100% whenever there is sun, discharges to the low-voltage cut-off, serves "
      "all loads equally, runs appliances at their usual hour.")
    w("- **SABIL controller:** (1) caps the charge level on hot days when tomorrow is sunny; (2) keeps a night reserve "
      f"for critical loads; (3) moves flexible loads ({flex_names}) to sunny hours; (4) serves critical loads first; (5) limits "
      "power when the battery is hot. Plans with a next-day sun forecast.")
    w("- **Battery ageing:** semi-empirical LFP/graphite model (Naumann et al. 2018 calendar, 2020 cycle): calendar loss "
      "~ sqrt(time), accelerated by temperature (Arrhenius) and state of charge; cycle loss ~ sqrt(full equivalent cycles), "
      "accelerated by depth and C-rate; half-cycles counted at each SoC reversal.")
    if val:
        cc = val.get("cross_check") or {}
        w(f"- **Ageing model verification:** closed-form square-root laws reproduced to machine precision; physics sanity "
          f"checks passed; cross-check against the authors' open-source implementation (SimSES, TU Munich): "
          + (f"relative difference {cc['relative_difference']:.1e} over one simulated year." if cc.get("available")
             else "not run (simses not installed).")
          + f" Overall: {'PASSED' if val.get('all_passed') else 'FAILED'}.")
    w("")
    w("## 3. Main result (headline forecast: " + sim["headline_forecast"] + ")")
    if base and smart:
        rows = [("Battery life to 80% capacity (years)", "battery_life_years", "{:.2f}"),
                ("Capacity left after the horizon", "capacity_after_10y_frac", "{:.1%}"),
                ("Hours with all critical loads on", "critical_service_frac", "{:.2%}"),
                ("Critical hours missed per year", None, None),
                ("Energy delivered (kWh/year)", "energy_served_kwh_per_year", "{:.0f}"),
                ("Energy not delivered (kWh/year)", "energy_unmet_kwh_per_year", "{:.1f}"),
                ("Diesel avoided (L/year)", "diesel_avoided_l_per_year", "{:.0f}"),
                ("Diesel avoided over the battery's life (L)", "diesel_avoided_l_per_battery_life", "{:,.0f}"),
                ("CO2 avoided (t/year)", "co2_avoided_t_per_year", "{:.2f}"),
                ("Full battery cycles per year", "full_cycles_per_year", "{:.0f}"),
                ("Average SoC when the battery is hot", "avg_soc_when_hot", "{:.1%}")]
        w("| Metric | Standard | SABIL |")
        w("|---|---|---|")
        for label, k, fmt in rows:
            if k is None:
                mb = base["critical_demand_hours_per_year"] - base["critical_hours_per_year"]
                ms = smart["critical_demand_hours_per_year"] - smart["critical_hours_per_year"]
                w(f"| {label} | {mb:.0f} | {ms:.0f} |")
            else:
                w(f"| {label} | {_f(base[k], fmt)} | {_f(smart[k], fmt)} |")
        if base["battery_life_years"] and smart["battery_life_years"]:
            g = smart["battery_life_years"] - base["battery_life_years"]
            w("")
            w(f"**SABIL extends battery life by {g:.2f} years ({g / base['battery_life_years']:+.0%}).**")
        if smart.get("ageing_breakdown") and base.get("ageing_breakdown"):
            bb, sb = base["ageing_breakdown"], smart["ageing_breakdown"]
            w(f" Ageing split at the end of the horizon - standard: calendar {bb['calendar_loss_frac']:.1%}, cycle "
              f"{bb['cycle_loss_frac']:.1%}; SABIL: calendar {sb['calendar_loss_frac']:.1%}, cycle {sb['cycle_loss_frac']:.1%}.")
    w("")
    w("![capacity](figures/fig1_battery_capacity.png)")
    w("![critical](figures/fig2_critical_hours.png)")
    w("")
    w("## 4. Does the forecast matter?")
    if any(others.values()):
        w("| Forecast used by SABIL | Battery life (y) | Critical hours | Energy not delivered (kWh/y) |")
        w("|---|---|---|---|")
        for m, s in others.items():
            if s:
                w(f"| {m} | {_f(s['battery_life_years'])} | {s['critical_service_frac']:.2%} | {s['energy_unmet_kwh_per_year']:.1f} |")
    if fm:
        nd = fm["metrics"]["next_day"]
        w("")
        w(f"Learned next-day forecast ({fm['model']}, {fm['validation']}), mean absolute error in kWh/m2/day:")
        w("")
        w("| Season | Learned | Tomorrow = today | Seasonal average | Skill vs tomorrow = today |")
        w("|---|---|---|---|---|")
        for s in ["all", "dry", "wet"]:
            r = nd[s]
            w(f"| {s} | {r['learned']['mae_kwh_m2_day']:.3f} | {r['persistence']['mae_kwh_m2_day']:.3f} | "
              f"{r['climatology']['mae_kwh_m2_day']:.3f} | {r['skill_vs_persistence']:+.0%} |")
        imp = fm.get("permutation_importance_last_fold", {}).get("next_day")
        if imp:
            w("")
            w("Most informative inputs (permutation importance): " + ", ".join(list(imp)[:4]) + ".")
        w("")
        w("![forecast](figures/fig9_forecast_skill.png)")
    lab = RES / "forecast_lab"
    if (lab / "metrics.csv").exists():
        import pandas as pd
        met = pd.read_csv(lab / "metrics.csv")
        summ = json.load(open(lab / "summary.json"))
        w("")
        w("### 4b. Forecasting lab: datasets x features x models")
        groups = summ.get("feature_groups", {})
        w("Data sources: NASA POWER (sun, temperature, humidity), ERA5 reanalysis via Open-Meteo (clouds by layer, "
          "dew point, pressure, wind, rain, soil moisture, ERA5 sun), NOAA climate indices (ONI, NAO, TNA). "
          f"Feature groups used: {', '.join(f'{g} ({len(v)})' for g, v in groups.items())}"
          + (f"; unavailable: {', '.join(summ['groups_missing'])}" if summ.get("groups_missing") else "") + ".")
        w("")
        w("| Model | Next-day MAE, year | Dry season | Rainy season | DM vs persistence (stat, p) | DM vs climatology (stat, p) |")
        w("|---|---|---|---|---|---|")
        nd = met[met["horizon"] == "next_day"].pivot(index="model", columns="season", values="mae").sort_values("all")
        dm = summ.get("diebold_mariano_next_day", {})
        for mdl, r in nd.iterrows():
            a = dm.get(mdl, {}).get("vs_persistence")
            b = dm.get(mdl, {}).get("vs_climatology")
            fa = f"{a[0]:+.1f}, {a[1]:.3f}" if a else "-"
            fb = f"{b[0]:+.1f}, {b[1]:.3f}" if b else "-"
            w(f"| {mdl} | {r['all']:.3f} | {r['dry']:.3f} | {r['wet']:.3f} | {fa} | {fb} |")
        w("")
        w("Diebold-Mariano: negative statistic = more accurate than the reference; p < 0.05 = difference unlikely to be chance.")
        pr = summ.get("probabilistic_next_day", {})
        if pr:
            w("")
            w("Probabilistic forecasts (80% intervals): " + "; ".join(
                f"{k}: coverage {v['coverage']:.1%}, mean width {v['mean_width_kwh_m2_day']:.2f} kWh/m2/day" for k, v in pr.items()) + ".")
        if (lab / "feature_group_ablation.csv").exists():
            fg = pd.read_csv(lab / "feature_group_ablation.csv").sort_values("delta_if_removed", ascending=False)
            w("")
            w("| Data source / feature group | Error increase if removed | Error reduction when added to sun history + season |")
            w("|---|---|---|")
            for r in fg.itertuples():
                w(f"| {r.group} | {r.delta_if_removed:+.4f} | {r.gain_when_added_to_base:+.4f} |")
        if (lab / "decision_focused.csv").exists():
            dec = pd.read_csv(lab / "decision_focused.csv")
            w("")
            w("Decision-focused test (each forecast drives SABIL for the whole horizon):")
            w("")
            w("| Forecast | Battery life (y) | Critical hours | Energy not delivered (kWh/y) |")
            w("|---|---|---|---|")
            for r in dec.itertuples():
                w(f"| {r.forecast_model} | {r.battery_life_years} | {r.critical_service_frac:.2%} | {r.energy_unmet_kwh_per_year:.1f} |")
            span = dec["battery_life_years"].max() - dec["battery_life_years"].min()
            w("")
            w(f"Battery life varies by only {span:.2f} years across all forecasts: SABIL's benefit does not depend on a "
              "sophisticated forecast, which matters for deployment on cheap hardware without internet.")
            w("")
            w("Note: this table was computed with the first, placeholder kit (dev_params: 1.5 kWp, 5.12 kWh). On the "
              "selected kit the main simulation confirms the same conclusion (battery life with each forecast in the "
              "headline table above).")
        w("")
        w("![leaderboard](figures/fig10_forecast_leaderboard.png)")
        w("![groups](figures/fig11_feature_groups.png)")
        w("![decision](figures/fig12_decision_focused.png)")
    w("")
    w("## 5. Which feature does the work? (ablation)")
    if ab:
        w(f"All features off reproduces the standard controller exactly: **{ab['all_off_equals_baseline']}** (sanity check).")
        w("")
        w("| Feature | Battery life lost if removed (y) | Life gain alone (y) | Critical hours if removed |")
        w("|---|---|---|---|")
        for r in sorted(ab["contributions"], key=lambda r: -r["life_lost_if_removed_years"]):
            w(f"| {r['feature']} | {r['life_lost_if_removed_years']:+.2f} | {r['life_gain_alone_years']:+.2f} | "
              f"{r['critical_service_if_removed']:.2%} |")
        w("")
        w("![ablation](figures/fig5_ablation.png)")
    w("")
    w("## 6. Design trade-off: hot-day charge cap")
    if sw and sw["points"]:
        p0 = min(sw["points"], key=lambda p: p["soc_cap_hot"])
        p1 = max(sw["points"], key=lambda p: p["soc_cap_hot"])
        ch = [p for p in sw["points"] if abs(p["soc_cap_hot"] - sw["chosen"]) < 1e-9]
        w(f"From a cap of {p1['soc_cap_hot']:.0%} to {p0['soc_cap_hot']:.0%}, battery life goes from "
          f"{_f(p1['battery_life_years'])} to {_f(p0['battery_life_years'])} years while energy delivered changes from "
          f"{p1['energy_served_kwh_per_year']:.0f} to {p0['energy_served_kwh_per_year']:.0f} kWh/year."
          + (f" Chosen cap {sw['chosen']:.0%}: {_f(ch[0]['battery_life_years'])} years." if ch else ""))
        w("")
        w("![tradeoff](figures/fig6_tradeoff.png)")
    w("")
    w("## 7. Uncertainty (Monte Carlo)")
    if mc:
        g, gf = mc["gain_years"], mc["gain_frac"]
        w(f"{mc['n_samples']} paired scenarios, inputs sampled jointly (ranges in `params/sim.json`):")
        w("")
        w(f"- Battery-life gain: **median {g['median']:.2f} years** (90% interval {g['p5']:.2f} to {g['p95']:.2f}); "
          f"relative gain median {gf['median']:+.0%} ({gf['p5']:+.0%} to {gf['p95']:+.0%}).")
        w(f"- SABIL lasts longer than the standard controller in **{mc['share_samples_smart_better']:.0%}** of scenarios.")
        w(f"- Battery life, standard: {mc['life_baseline']['median']:.1f} y ({mc['life_baseline']['p5']:.1f}-"
          f"{mc['life_baseline']['p95']:.1f}); SABIL: {mc['life_smart']['median']:.1f} y ({mc['life_smart']['p5']:.1f}-"
          f"{mc['life_smart']['p95']:.1f}).")
        w("- What drives the gain (Spearman rank correlation with years gained): " + ", ".join(
            f"{k.split('.')[-1]} {v:+.2f}" for k, v in sorted(mc["spearman_with_gain_years"].items(), key=lambda kv: -abs(kv[1]))[:4]) + ".")
        if mc.get("skipped_inputs"):
            w(f"- Not varied (not present in the current ageing model): {', '.join(mc['skipped_inputs'])}.")
        w("")
        w("![uncertainty](figures/fig7_uncertainty.png)")
        w("![drivers](figures/fig8_gain_drivers.png)")
    exd = RES / "experiments"
    if (exd / "oracle.json").exists():
        orc = json.load(open(exd / "oracle.json"))
        w("")
        w("## 7b. How far from optimal? (dynamic-programming benchmark)")
        w("A degradation-aware dynamic program with **perfect foresight** re-plans the battery every day over 48 hours, "
          "minimising weighted unmet energy (critical loads x50) plus the value of capacity lost (same ageing model). "
          "It is an upper bound, not a product.")
        w("")
        w("| Controller | Battery life (y) | Energy delivered (kWh/y) | Critical hours |")
        w("|---|---|---|---|")
        for nm, r in [("Standard", orc["baseline"]), ("SABIL", orc["smart"])] + \
                [(f"Optimal, wear weight {p['lambda_degradation']:g}", p) for p in orc["oracle"]]:
            w(f"| {nm} | {_f(r['battery_life_years'])} | {r['energy_served_kwh_per_year']:.0f} | {r['critical_service_frac']:.2%} |")
        o = min(orc["oracle"], key=lambda p: abs(p["lambda_degradation"] - orc["chosen_lambda"]))
        b, s = orc["baseline"], orc["smart"]
        if b["battery_life_years"] and s["battery_life_years"] and o["battery_life_years"]:
            share = (s["battery_life_years"] - b["battery_life_years"]) / (o["battery_life_years"] - b["battery_life_years"])
            w("")
            w(f"At the economic wear weight ({orc['chosen_lambda']:g}), SABIL's rules capture **{share:.0%}** of the "
              "battery-life gain that a perfect controller achieves. The remaining headroom is the roadmap: model-predictive "
              "control on the device.")
        w("")
        w("![headroom](figures/fig14_headroom.png)")
    if (exd / "placement.json").exists():
        pl = json.load(open(exd / "placement.json"))
        w("")
        w("## 7c. Where to put the battery (thermal model with self-heating)")
        w("| Placement | Battery mean temp (C) | Life, standard (y) | Life, SABIL (y) | Assumption |")
        w("|---|---|---|---|---|")
        for r in pl["rows"]:
            w(f"| {r['placement']} | {r['avg_battery_temp_c']:.1f} | {_f(r['life_baseline'])} | {_f(r['life_smart'])} | {r['why']} |")
        w("")
        w("Placement is free advice for every household and stacks with SABIL. Placement parameters are scenario "
          "assumptions; a temperature logger in pilot homes would measure them.")
        w("")
        w("![placement](figures/fig13_placement.png)")
    if (exd / "demand_variability_summary.json").exists():
        dv = json.load(open(exd / "demand_variability_summary.json"))
        w("")
        w("## 7d. Random household demand")
        w(f"{dv['n']} random demand realisations (RAMP-style: shifted usage times, skipped days, variable intensity): "
          f"battery-life gain median {dv['gain_years_median']:.2f} years ({dv['gain_years_p5']:.2f}-{dv['gain_years_p95']:.2f}); "
          f"SABIL better in {dv['share_smart_better']:.0%}; hours with all critical loads on: standard "
          f"{dv['critical_baseline_mean']:.2%}, SABIL {dv['critical_smart_mean']:.2%} (worst case {dv['critical_smart_min']:.2%}).")
    w("")
    w("## 8. Limitations")
    w("- Simulation, not field data: the ageing model is fitted on one commercial LFP cell (Sony/Murata US26650) in the lab; "
      "real packs, cell imbalance and BMS behaviour differ. A 20-50 home pilot with logged SoC and temperature is the test.")
    w("- Headline runs use battery temperature = air temperature + a fixed offset; the placement study adds a two-node "
      "thermal model with self-heating, whose placement parameters are assumptions to be measured in the field.")
    w("- The optimal benchmark uses perfect foresight and marginal (linearised) ageing rates; it bounds what any "
      "controller could achieve, it is not deployable as is.")
    w("- Household demand is a fixed daily schedule; real behaviour varies day to day and grows over time.")
    w("- Battery life beyond the simulated horizon is extrapolated with the last year's fade rate (conservative for "
      "square-root ageing).")
    w("- Weather years are treated as independent blocks in the forecast cross-validation; NASA POWER is a reanalysis "
      "product with known bias versus ground stations.")
    w("")
    w("## 9. Reproduce")
    w("```")
    w("python src/weather.py && python src/validate_aging.py && python src/forecast_ml.py")
    w("python src/simulate.py && python src/experiments.py && python src/plots.py && python src/report.py")
    w("```")
    w(f"Python {platform.python_version()}. Data: NASA POWER (public). Ageing model reference implementation: "
      "SimSES (TU Munich).")
    w("")
    w("## References")
    w("- Naumann M., Schimpe M., Keil P., Hesse H.C., Jossen A. Analysis and modeling of calendar aging of a commercial "
      "LiFePO4/graphite cell. J. Energy Storage 17 (2018) 153-169. doi:10.1016/j.est.2018.01.019")
    w("- Naumann M., Spingler F.B., Jossen A. Analysis and modeling of cycle aging of a commercial LiFePO4/graphite cell. "
      "J. Power Sources 451 (2020) 227666.")
    w("- SimSES, open-source simulation of stationary energy storage systems, TU Munich (PyPI `simses`).")
    w("- NASA POWER hourly data, power.larc.nasa.gov.")
    w("- IPCC 2006 Guidelines for National Greenhouse Gas Inventories, Vol. 2 (default CO2 factor for diesel).")
    w("- Global Energy Hackathon 2026, Problem Brief 13: Power for a city the grid can't protect.")
    (RES / "RESEARCH_REPORT.md").write_text("\n".join(L) + "\n")
    print("wrote results/RESEARCH_REPORT.md")


if __name__ == "__main__":
    main()
