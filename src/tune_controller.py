"""SABIL - tune the controller's rules by Bayesian optimization, checked on unseen weather (owner: Youcef).

Why: the perfect-foresight benchmark (results/experiments/oracle.json) shows battery life can go well
beyond what the hand-set rules reach, so the rules' thresholds are worth tuning.

How:
  - Search space: the six numbers of the SABIL controller in params/sim.json -> smart
    (hot-day threshold, hot-day charge cap, sunny-day ratio, reserve margin, hot charge rate,
    cloudy-day buffer).
  - Optimizer: Optuna's TPE sampler (tree-structured Parzen estimator, a Bayesian optimization
    method; Bergstra et al., NeurIPS 2011), fixed seed.
  - Objective: capacity left after 10 years (no extrapolation), with penalties if the tuned rules
    lose more critical hours or deliver less energy than the hand-set rules on the same weather.
  - Honesty: tuned on 2014-2018 weather only (played twice to make 10 years), tested on 2019-2023
    (played twice), which the optimizer never sees. Both the hand-set and tuned rules are reported
    on the test years, next to the standard controller.

Run from the repo root:  python src/tune_controller.py [--trials 80] [--apply]
Writes results/controller_tuning.json (figure: src/exhibits.py).
--apply writes the tuned values into params/sim.json (only if they win on the test years).
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulate  # noqa: E402
from params_io import ROOT, load_all  # noqa: E402
from weather import load_weather  # noqa: E402

TRAIN = [2014, 2015, 2016, 2017, 2018] * 2
TEST = [2019, 2020, 2021, 2022, 2023] * 2
SPACE = {  # name: (low, high)
    "hot_threshold_c": (22.0, 36.0),
    "soc_cap_hot": (0.35, 1.0),
    "sunny_ratio": (0.6, 1.4),
    "reserve_margin": (0.0, 0.5),
    "c_rate_hot": (0.1, 0.5),
    "cloudy_day_buffer": (0.0, 1.0),
}


def evaluate(P0, w, ctrl, smart=None):
    P = copy.deepcopy(P0)
    if smart:
        P["sim"]["smart"].update(smart)
    s, _ = simulate.run(ctrl, P, w, forecast_mode="persistence" if ctrl == "smart" else "perfect")
    py = s["per_year"]
    out = {"capacity_after_10y": s["capacity_after_10y_frac"]}
    if len(set(P["sim"]["weather_years"])) == len(P["sim"]["weather_years"]):   # life is only defined on real 10-year runs
        out.update(battery_life_years=s["battery_life_years"], battery_life_note=s.get("battery_life_note"))
    return {**out,
            "critical_hours_lost": int(sum(y["critical_demand_hours"] - y["critical_hours"] for y in py)),
            "critical_hours_lost_worst_year": int(max(y["critical_demand_hours"] - y["critical_hours"] for y in py)),
            "energy_unmet_kwh_per_year": s["energy_unmet_kwh_per_year"],
            "energy_served_kwh_per_year": s["energy_served_kwh_per_year"]}


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plots
    plots._style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), layout="constrained", sharey=True)
    h = out["history"]
    ax = axes[0]
    ax.scatter([t["critical_hours_lost"] for t in h], [100 * t["capacity_after_10y"] for t in h], s=18,
               color=plots.BLUE, alpha=0.45, linewidths=0, label="80 rule settings tried")
    for key, lab, c, mk in (("sabil_hand_set", "hand-set rules", plots.INK, "o"), ("sabil_tuned", "tuned rules", plots.ORANGE, "D")):
        r = out["train"][key]
        ax.scatter(r["critical_hours_lost"], 100 * r["capacity_after_10y"], s=64, color=c, marker=mk, zorder=3)
        ax.annotate(lab, (r["critical_hours_lost"], 100 * r["capacity_after_10y"]), xytext=(8, -4),
                    textcoords="offset points", fontsize=9, color=plots.INK)
    ax.set(xlabel="Critical hours lost over 10 years", ylabel="Capacity left after 10 years (%)", xscale="symlog")
    plots._subtitle(ax, "Search on 2014-2018 weather (seen by the optimizer)")
    ax = axes[1]
    for key, lab, c, mk in (("standard", "standard controller", plots.GREY, "s"), ("sabil_hand_set", "SABIL, hand-set rules", plots.INK, "o"),
                            ("sabil_tuned", "SABIL, tuned rules", plots.ORANGE, "D")):
        r = out["test_unseen_weather"][key]
        ax.scatter(r["critical_hours_lost"], 100 * r["capacity_after_10y"], s=70, color=c, marker=mk, zorder=3)
        ax.annotate(f"{lab}\n{r['critical_hours_lost']} h lost, {100 * r['capacity_after_10y']:.1f}%",
                    (r["critical_hours_lost"], 100 * r["capacity_after_10y"]), xytext=(8, -14), textcoords="offset points",
                    fontsize=8.5, color=plots.INK)
    ax.set(xlabel="Critical hours lost over 10 years", xscale="symlog")
    ax.set_xlim(5, 400)
    plots._subtitle(ax, "Test on 2019-2023 weather (never seen)")
    fig.savefig(ROOT / "results" / "figures" / "fig17_tuning.png", dpi=200)
    plt.close(fig)


def weather_for(P0, years):
    P = copy.deepcopy(P0)
    P["sim"]["weather_years"] = list(years)
    P["sim"]["years"] = len(years)
    return P, load_weather(P["sim"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    P0 = load_all(verbose=False)
    hand = {k: P0["sim"]["smart"][k] for k in SPACE}
    Ptr, wtr = weather_for(P0, TRAIN)
    Pte, wte = weather_for(P0, TEST)
    ref = evaluate(Ptr, wtr, "smart")

    def objective(trial):
        x = {k: trial.suggest_float(k, lo, hi) for k, (lo, hi) in SPACE.items()}
        r = evaluate(Ptr, wtr, "smart", x)
        pen = 0.002 * max(0, r["critical_hours_lost"] - ref["critical_hours_lost"]) \
            + 0.05 * max(0.0, r["energy_unmet_kwh_per_year"] - ref["energy_unmet_kwh_per_year"] - 0.5)
        for k, v in r.items():
            trial.set_user_attr(k, v)
        return r["capacity_after_10y"] - pen

    t0 = time.time()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=2026, multivariate=True, n_startup_trials=16))
    study.enqueue_trial(hand)          # the hand-set rules are the first point
    study.optimize(objective, n_trials=a.trials)
    best = {k: round(v, 3) for k, v in study.best_params.items()}

    test = {"standard": evaluate(Pte, wte, "baseline"), "sabil_hand_set": evaluate(Pte, wte, "smart"),
            "sabil_tuned": evaluate(Pte, wte, "smart", best)}
    full_P, full_w = weather_for(P0, P0["sim"]["weather_years"])
    full = {"standard": evaluate(full_P, full_w, "baseline"), "sabil_hand_set": evaluate(full_P, full_w, "smart"),
            "sabil_tuned": evaluate(full_P, full_w, "smart", best)}
    tr = {"sabil_hand_set": ref, "sabil_tuned": evaluate(Ptr, wtr, "smart", best)}

    def gain(d):
        b = 1 - d["standard"]["capacity_after_10y"]
        return {k: round(100 * (1 - (1 - d[k]["capacity_after_10y"]) / b), 1) for k in ("sabil_hand_set", "sabil_tuned")}
    wins = (test["sabil_tuned"]["capacity_after_10y"] > test["sabil_hand_set"]["capacity_after_10y"]
            and test["sabil_tuned"]["critical_hours_lost"] <= test["sabil_hand_set"]["critical_hours_lost"]
            and test["sabil_tuned"]["energy_unmet_kwh_per_year"] <= test["sabil_hand_set"]["energy_unmet_kwh_per_year"] + 0.5)
    out = {
        "method": "Optuna TPE (Bayesian optimization), seed 2026, %d trials, tuned on 2014-2018 weather, "
                  "tested on 2019-2023 (each played twice = 10 years of ageing)" % a.trials,
        "search_space": SPACE, "hand_set": hand, "tuned": best,
        "train_years": TRAIN, "test_years": TEST,
        "train": tr, "test_unseen_weather": test, "full_2014_2023": full,
        "fade_reduction_pct_vs_standard": {"test_unseen_weather": gain(test), "full_2014_2023": gain(full)},
        "tuned_wins_on_unseen_weather": bool(wins),
        "history": [{"trial": t.number, "value": t.value, **t.user_attrs} for t in study.trials],
        "runtime_s": round(time.time() - t0, 1),
    }
    (ROOT / "results" / "controller_tuning.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"{a.trials} trials in {out['runtime_s']} s. Tuned: {best}")
    for name, d in (("TRAIN 2014-18", tr), ("TEST 2019-23 (unseen)", test), ("FULL 2014-23", full)):
        print(name)
        for k, r in d.items():
            print(f"  {k:<16} capacity 10y {r['capacity_after_10y']:.4f}  life {r.get('battery_life_years', '-')} y  "
                  f"critical h lost {r['critical_hours_lost']} (worst year {r['critical_hours_lost_worst_year']})  "
                  f"unmet {r['energy_unmet_kwh_per_year']} kWh/y")
    print("fade reduction vs standard:", out["fade_reduction_pct_vs_standard"], "| tuned wins on unseen weather:", wins)

    if "--old-plots" in sys.argv:   # figures are drawn by src/exhibits.py
        plot(out)

    if a.apply and wins:
        p = ROOT / "params" / "sim.json"
        s = json.loads(p.read_text())
        s["smart"].update(best)
        s["smart_tuning"] = {"hand_set": hand, "source": "src/tune_controller.py (results/controller_tuning.json)"}
        p.write_text(json.dumps(s, indent=2) + "\n")
        print("applied to params/sim.json")


if __name__ == "__main__":
    main()
