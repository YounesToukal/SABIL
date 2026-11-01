"""SABIL - do the controller's results depend on one battery dataset? (owner: Youcef)

Runs the selected kit with the standard and the SABIL controller under BOTH ageing models
of src/aging_model.py, each fitted on different experimental data:
  naumann_sony   TU Munich, Naumann et al. 2018/2020 (3 Ah cylindrical cells)
  nrel_prismatic NREL BLAST-Lite, large-format prismatic cells
for cells as tested, plus NREL cells ageing 2x faster. Same house, same NASA POWER weather.

Run from the repo root:  python src/robustness_aging.py
Writes results/aging_robustness.json
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate  # noqa: E402
from params_io import ROOT, load_all  # noqa: E402
from sizing_map import scale_ageing  # noqa: E402
from weather import load_weather  # noqa: E402


def main():
    P0 = load_all(verbose=False)
    w = load_weather(P0["sim"])
    mode = P0["sim"]["headline_forecast"]
    report = {"kit": {"pv_kwp": P0["system"]["pv_kwp"], "battery_kwh": P0["battery"]["capacity_kwh"]},
              "forecast": mode}
    # "2x faster" is applied to the slow-fading NREL model only: the TU Munich model already
    # fades fast, and doubling it would double-count the pessimism
    for model, scalar in (("naumann_sony", 1.0), ("nrel_prismatic", 1.0), ("nrel_prismatic", 2.0)):
        if True:
            res = {}
            for ctrl in ("baseline", "smart"):
                P = copy.deepcopy(P0)
                P["battery"]["aging"]["model"] = model
                scale_ageing(P["battery"], scalar)
                s, _ = simulate.run(ctrl, P, w, forecast_mode=mode if ctrl == "smart" else "perfect")
                py = s["per_year"]
                res[ctrl] = {"capacity_after_10y": s["capacity_after_10y_frac"],
                             "battery_life_years": s["battery_life_years"],
                             "battery_life_note": s["battery_life_note"],
                             "critical_hours_lost_worst_year": max(y["critical_demand_hours"] - y["critical_hours"]
                                                                   for y in py)}
            fb, fs = 1 - res["baseline"]["capacity_after_10y"], 1 - res["smart"]["capacity_after_10y"]
            res["fade_reduction_pct"] = round(100 * (1 - fs / fb), 1)
            key = f"{model}, cells {'as tested' if scalar == 1 else 'x2 faster'}"
            report[key] = res
            print(key, json.dumps(res))
    (ROOT / "results" / "aging_robustness.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
