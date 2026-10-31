"""Experiments (owner: Younes): ablation, design trade-off, uncertainty.

1. Ablation: run SABIL with each feature removed ("leave-one-out") and with each feature
   alone ("only"), plus all features off (must equal the standard controller: a sanity check).
2. Trade-off sweep: battery life vs energy delivered as the hot-day charge cap varies.
3. Monte Carlo: paired runs (standard and SABIL with the same sampled inputs) over the
   uncertain inputs listed in params/sim.json -> experiments.uncertain_inputs. Reports the
   distribution of the battery-life gain and Spearman rank correlations (which input
   drives the gain).
All runs use the headline forecast, the same weather and the same house.

Run:  python src/experiments.py                  (all three, parallel)
      python src/experiments.py --only mc --samples 32
Outputs: results/experiments/*.json, montecarlo.csv
"""
import argparse
import copy
import json
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

from params_io import ROOT, load_all

OUT = ROOT / "results" / "experiments"
FEATURES = ["charge_cap", "night_reserve", "load_shift", "heat_gentle", "critical_priority"]
KEEP = ["avg_battery_temp_c", "battery_life_years", "capacity_after_10y_frac", "critical_service_frac", "critical_hours_per_year",
        "energy_served_kwh_per_year", "energy_unmet_kwh_per_year", "diesel_avoided_l_per_year",
        "diesel_avoided_l_per_battery_life", "full_cycles_per_year", "avg_soc_when_hot"]

_W = None


def _init():
    global _W
    from weather import load_weather
    _W = load_weather(load_all(verbose=False)["sim"])


def _run(job):
    """job = (tag, controller, P, forecast) -> (tag, compact summary)."""
    from simulate import run
    tag, ctrl, P, mode = job
    s, _ = run(ctrl, P, _W, forecast_mode=mode)
    return tag, {k: s[k] for k in KEEP}


def _pool_map(jobs, workers):
    if workers <= 1:
        _init()
        return [_run(j) for j in jobs]
    with get_context("fork").Pool(workers, initializer=_init) as pool:
        return pool.map(_run, jobs, chunksize=1)


def _with_features(P, on):
    Q = copy.deepcopy(P)
    Q["sim"]["smart"]["features"] = {f: (f in on) for f in FEATURES}
    return Q


def ablation_jobs(P, mode):
    jobs = [("baseline", "baseline", P, "none"), ("full", "smart", _with_features(P, FEATURES), mode),
            ("all_off", "smart", _with_features(P, []), mode)]
    for f in FEATURES:
        jobs.append((f"without_{f}", "smart", _with_features(P, [g for g in FEATURES if g != f]), mode))
        jobs.append((f"only_{f}", "smart", _with_features(P, [f]), mode))
    return jobs


def sweep_jobs(P, mode):
    jobs = []
    for v in P["sim"]["experiments"]["sweep_soc_cap_hot"]:
        Q = copy.deepcopy(P)
        Q["sim"]["smart"]["soc_cap_hot"] = v
        jobs.append((f"cap_{v:.2f}", "smart", Q, mode))
    return jobs


def _get(P, path):
    d = P
    for k in path.split("."):
        d = d[k]
    return d


def _set(P, path, value):
    *head, last = path.split(".")
    d = P
    for k in head:
        d = d[k]
    d[last] = value


def sample_inputs(P, n, seed):
    rng = np.random.default_rng(seed)
    specs, skipped = [], []
    for u in P["sim"]["experiments"]["uncertain_inputs"]:
        path = u["path"]
        try:
            base = 1.0 if path == "household.load_scale" else _get(P, path)
            specs.append((u, base))
        except (KeyError, TypeError):
            skipped.append(path)        # e.g. Youcef's ageing model uses other keys
    rows = []
    for i in range(n):
        row = {}
        for u, base in specs:
            if u["dist"] == "lognormal_scale":
                v = base * float(rng.lognormal(0.0, u["sigma"]))
            elif u["dist"] == "uniform_scale":
                v = base * float(rng.uniform(u["low"], u["high"]))
            elif u["dist"] == "uniform_set":
                v = float(rng.uniform(u["low"], u["high"]))
            else:
                raise ValueError(u["dist"])
            row[u["path"]] = v
        rows.append(row)
    return rows, skipped


def mc_jobs(P, rows, mode):
    jobs = []
    for i, row in enumerate(rows):
        Q = copy.deepcopy(P)
        for path, v in row.items():
            _set(Q, path, v)
        jobs.append((f"mc{i}_baseline", "baseline", Q, "none"))
        jobs.append((f"mc{i}_smart", "smart", Q, mode))
    return jobs


def spearman(x, y):
    rx, ry = pd.Series(x).rank(), pd.Series(y).rank()
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ablation", "sweep", "mc", "placement", "demand", "oracle"], default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    P = load_all()
    ex = P["sim"]["experiments"]
    mode = P["sim"]["headline_forecast"]
    workers = a.workers or ex["workers"]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if a.only in (None, "ablation"):
        res = dict(_pool_map(ablation_jobs(P, mode), workers))
        base, full = res["baseline"], res["full"]
        if full["battery_life_years"] is None:
            print("ablation: ageing model is the stub (no fade) - battery life cannot be compared")
        table = []
        for f in FEATURES:
            wo, only = res[f"without_{f}"], res[f"only_{f}"]
            lf = lambda r: r["battery_life_years"] or 0.0
            table.append({"feature": f,
                          "life_lost_if_removed_years": round(lf(full) - lf(wo), 2),
                          "life_gain_alone_years": round(lf(only) - lf(base), 2),
                          "critical_service_if_removed": wo["critical_service_frac"],
                          "unmet_kwh_if_removed": wo["energy_unmet_kwh_per_year"]})
        check = abs((res["all_off"]["battery_life_years"] or 0) - (base["battery_life_years"] or 0)) < 1e-6 \
            and abs(res["all_off"]["energy_served_kwh_per_year"] - base["energy_served_kwh_per_year"]) < 1e-6
        json.dump({"runs": res, "contributions": table, "all_off_equals_baseline": bool(check),
                   "forecast": mode}, open(OUT / "ablation.json", "w"), indent=2)
        print(f"\nABLATION (forecast: {mode})  all features off == standard controller: {check}")
        print(f"{'feature':<20}{'life lost if removed':>22}{'life gain alone':>18}{'critical if removed':>21}")
        for r in table:
            print(f"{r['feature']:<20}{r['life_lost_if_removed_years']:>20.2f} y{r['life_gain_alone_years']:>16.2f} y"
                  f"{r['critical_service_if_removed']:>21.2%}")

    if a.only in (None, "sweep"):
        res = dict(_pool_map(sweep_jobs(P, mode), workers))
        base = dict(_pool_map([("baseline", "baseline", P, "none")], 1))["baseline"]
        pts = [{"soc_cap_hot": float(k.split("_")[1]), **v} for k, v in res.items()]
        pts.sort(key=lambda r: r["soc_cap_hot"])
        json.dump({"points": pts, "baseline": base, "chosen": P["sim"]["smart"]["soc_cap_hot"],
                   "forecast": mode}, open(OUT / "sweep.json", "w"), indent=2)
        print("\nTRADE-OFF SWEEP: hot-day charge cap")
        print(f"{'cap':>6}{'life (y)':>10}{'served kWh/y':>14}{'unmet kWh/y':>13}{'critical':>10}")
        for r in pts:
            print(f"{r['soc_cap_hot']:>6.2f}{(r['battery_life_years'] or 0):>10.2f}{r['energy_served_kwh_per_year']:>14.1f}"
                  f"{r['energy_unmet_kwh_per_year']:>13.1f}{r['critical_service_frac']:>10.2%}")

    if a.only in (None, "mc"):
        n = a.samples or ex["monte_carlo_samples"]
        rows, skipped = sample_inputs(P, n, ex["seed"])
        if skipped:
            print("Monte Carlo: inputs not present in these params, skipped:", skipped)
        res = dict(_pool_map(mc_jobs(P, rows, mode), workers))
        recs = []
        for i, row in enumerate(rows):
            b, s = res[f"mc{i}_baseline"], res[f"mc{i}_smart"]
            if b["battery_life_years"] is None or s["battery_life_years"] is None:
                continue
            recs.append({**{f"in:{k}": v for k, v in row.items()},
                         "life_baseline": b["battery_life_years"], "life_smart": s["battery_life_years"],
                         "gain_years": s["battery_life_years"] - b["battery_life_years"],
                         "gain_frac": s["battery_life_years"] / b["battery_life_years"] - 1,
                         "critical_baseline": b["critical_service_frac"], "critical_smart": s["critical_service_frac"],
                         "unmet_extra_kwh": s["energy_unmet_kwh_per_year"] - b["energy_unmet_kwh_per_year"],
                         "diesel_per_battery_extra_l": (s["diesel_avoided_l_per_battery_life"] or 0)
                         - (b["diesel_avoided_l_per_battery_life"] or 0)})
        df = pd.DataFrame(recs)
        df.to_csv(OUT / "montecarlo.csv", index=False)
        if len(df):
            q = lambda c, p: float(np.percentile(df[c], p))
            summ = {"n_samples": len(df), "forecast": mode, "skipped_inputs": skipped,
                    "gain_years": {"median": q("gain_years", 50), "p5": q("gain_years", 5), "p95": q("gain_years", 95),
                                   "mean": float(df["gain_years"].mean())},
                    "gain_frac": {"median": q("gain_frac", 50), "p5": q("gain_frac", 5), "p95": q("gain_frac", 95)},
                    "share_samples_smart_better": float((df["gain_years"] > 0).mean()),
                    "life_baseline": {"median": q("life_baseline", 50), "p5": q("life_baseline", 5), "p95": q("life_baseline", 95)},
                    "life_smart": {"median": q("life_smart", 50), "p5": q("life_smart", 5), "p95": q("life_smart", 95)},
                    "critical_smart_min": float(df["critical_smart"].min()),
                    "critical_baseline_min": float(df["critical_baseline"].min()),
                    "spearman_with_gain_years": {c[3:]: spearman(df[c], df["gain_years"]) for c in df if c.startswith("in:")},
                    "spearman_with_gain_frac": {c[3:]: spearman(df[c], df["gain_frac"]) for c in df if c.startswith("in:")}}
            json.dump(summ, open(OUT / "montecarlo_summary.json", "w"), indent=2)
            g = summ["gain_years"]
            print(f"\nMONTE CARLO ({len(df)} paired samples): battery-life gain median {g['median']:.2f} y "
                  f"(90% interval {g['p5']:.2f} to {g['p95']:.2f}); SABIL better in "
                  f"{summ['share_samples_smart_better']:.0%} of samples")
            print("what drives the gain (Spearman rank correlation):")
            for k, v in sorted(summ["spearman_with_gain_years"].items(), key=lambda kv: -abs(kv[1])):
                print(f"   {k:<40}{v:+.2f}")

    if a.only in (None, "placement"):
        import aging_model
        st = ex.get("placement_study")
        if st and hasattr(aging_model, "battery_temp_estimate"):
            jobs = []
            for name in st["placements"]:
                Q = copy.deepcopy(P)
                Q["battery"]["thermal"] = {**st["thermal"], "placement": name, "placements": st["placements"]}
                jobs += [(f"{name}|baseline", "baseline", Q, "none"), (f"{name}|smart", "smart", Q, mode)]
            res = dict(_pool_map(jobs, workers))
            rows = []
            for name in st["placements"]:
                b, s_ = res[f"{name}|baseline"], res[f"{name}|smart"]
                rows.append({"placement": name, "why": st["placements"][name].get("why", ""),
                             "avg_battery_temp_c": b["avg_battery_temp_c"],
                             "life_baseline": b["battery_life_years"], "life_smart": s_["battery_life_years"],
                             "critical_baseline": b["critical_service_frac"], "critical_smart": s_["critical_service_frac"]})
            json.dump({"rows": rows, "forecast": mode, "sources": st.get("sources", "")},
                      open(OUT / "placement.json", "w"), indent=2)
            print("\nBATTERY PLACEMENT (thermal model with self-heating)")
            print(f"{'placement':<22}{'mean battery C':>15}{'life standard':>15}{'life SABIL':>12}")
            for r in rows:
                print(f"{r['placement']:<22}{r['avg_battery_temp_c']:>15.1f}{(r['life_baseline'] or 0):>15.2f}{(r['life_smart'] or 0):>12.2f}")
        else:
            print("placement study skipped (needs the interim ageing model's thermal option)")

    if a.only in (None, "demand"):
        n = ex.get("demand_seeds", 0)
        if n:
            jobs = []
            for i in range(n):
                Q = copy.deepcopy(P)
                Q["sim"]["demand"] = {**P["sim"]["demand"], "stochastic": True, "seed": 1000 + i}
                jobs += [(f"d{i}|baseline", "baseline", Q, "none"), (f"d{i}|smart", "smart", Q, mode)]
            res = dict(_pool_map(jobs, workers))
            recs = []
            for i in range(n):
                b, s_ = res[f"d{i}|baseline"], res[f"d{i}|smart"]
                if b["battery_life_years"] and s_["battery_life_years"]:
                    recs.append({"seed": 1000 + i, "life_baseline": b["battery_life_years"], "life_smart": s_["battery_life_years"],
                                 "gain_years": s_["battery_life_years"] - b["battery_life_years"],
                                 "critical_baseline": b["critical_service_frac"], "critical_smart": s_["critical_service_frac"],
                                 "unmet_baseline": b["energy_unmet_kwh_per_year"], "unmet_smart": s_["energy_unmet_kwh_per_year"]})
            df = pd.DataFrame(recs)
            df.to_csv(OUT / "demand_variability.csv", index=False)
            if len(df):
                summ = {"n": len(df), "gain_years_median": float(df["gain_years"].median()),
                        "gain_years_p5": float(df["gain_years"].quantile(0.05)), "gain_years_p95": float(df["gain_years"].quantile(0.95)),
                        "share_smart_better": float((df["gain_years"] > 0).mean()),
                        "critical_baseline_mean": float(df["critical_baseline"].mean()),
                        "critical_smart_mean": float(df["critical_smart"].mean()),
                        "critical_smart_min": float(df["critical_smart"].min()),
                        "unmet_baseline_mean": float(df["unmet_baseline"].mean()), "unmet_smart_mean": float(df["unmet_smart"].mean())}
                json.dump(summ, open(OUT / "demand_variability_summary.json", "w"), indent=2)
                print(f"\nRANDOM DEMAND ({len(df)} realisations): gain median {summ['gain_years_median']:.2f} y "
                      f"({summ['gain_years_p5']:.2f}-{summ['gain_years_p95']:.2f}), SABIL better in {summ['share_smart_better']:.0%}; "
                      f"critical hours standard {summ['critical_baseline_mean']:.2%} vs SABIL {summ['critical_smart_mean']:.2%}")

    if a.only in (None, "oracle"):
        import aging_model
        if "oracle" in P["sim"] and hasattr(aging_model, "breakdown") and "cal_k_ref_per_sqrt_s" in P["battery"].get("aging", {}):
            lams = P["sim"]["oracle"]["lambda_degradation_sweep"]
            jobs = [(f"oracle:{l}", f"oracle:{l}", P, "perfect") for l in lams]
            jobs += [("baseline", "baseline", P, "none"), ("smart", "smart", P, mode)]
            res = dict(_pool_map(jobs, workers))
            pts = [{"lambda_degradation": l, **res[f"oracle:{l}"]} for l in lams]
            json.dump({"oracle": pts, "baseline": res["baseline"], "smart": res["smart"], "forecast": mode,
                       "chosen_lambda": P["sim"]["oracle"]["lambda_degradation"]},
                      open(OUT / "oracle.json", "w"), indent=2)
            sm = res["smart"]
            print("\nOPTIMAL BENCHMARK (dynamic programming, perfect foresight)")
            print(f"{'controller':<22}{'life (y)':>10}{'served kWh/y':>14}{'unmet kWh/y':>13}{'critical':>10}")
            for name, r in [("standard", res["baseline"]), (f"SABIL ({mode})", sm)] + \
                           [(f"oracle lambda={p['lambda_degradation']:g}", p) for p in pts]:
                print(f"{name:<22}{(r['battery_life_years'] or 0):>10.2f}{r['energy_served_kwh_per_year']:>14.1f}"
                      f"{r['energy_unmet_kwh_per_year']:>13.1f}{r['critical_service_frac']:>10.2%}")
        else:
            print("oracle benchmark skipped (needs the interim ageing model)")
    print(f"\nexperiments done in {time.time() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
