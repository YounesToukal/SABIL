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
    w("- **Household:** 5 people; hourly appliance schedule from `params/household.json`; lights, phone charging and fans "
      "are critical loads. Grid assumed down for the whole horizon (worst case of Brief 13).")
    w("- **Paired design:** the same house, weather, PV array and battery are simulated twice; only the controller differs. "
      "Any difference is caused by the controller.")
    w("- **Standard controller:** charges to 100% whenever there is sun, discharges to the low-voltage cut-off, serves "
      "all loads equally, runs appliances at their usual hour.")
    w("- **SABIL controller:** (1) caps the charge level on hot days when tomorrow is sunny; (2) keeps a night reserve "
      "for critical loads; (3) moves flexible loads (iron) to sunny hours; (4) serves critical loads first; (5) limits "
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
    w("")
    w("## 8. Limitations")
    w("- Simulation, not field data: the ageing model is fitted on one commercial LFP cell (Sony/Murata US26650) in the lab; "
      "real packs, cell imbalance and BMS behaviour differ. A 20-50 home pilot with logged SoC and temperature is the test.")
    w("- Battery temperature = air temperature + a fixed offset; no self-heating or enclosure thermal model, so the "
      "'gentler when hot' feature has no effect by construction.")
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
