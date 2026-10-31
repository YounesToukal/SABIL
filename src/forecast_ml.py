"""Learned next-day solar forecast, evaluated honestly (owner: Younes).

Target: daily clearness index kt = daily GHI / daily clear-sky GHI, for the same day
(issued at 00:00) and the next day. Features known at 00:00: kt of the last 3 days and
7-day mean, yesterday's mean temperature, temperature range and humidity, the 3-day
humidity trend, and the season (day of year, sin/cos).
Model: gradient-boosted trees (scikit-learn HistGradientBoostingRegressor).
Validation: leave-one-year-out (train on 9 weather years, predict the 10th, rotate),
so every forecast used in the simulation is out-of-sample.
Baselines: persistence (kt tomorrow = kt yesterday) and climatology (mean kt for the
same time of year in the training years).
Hourly forecast = predicted kt x clear-sky GHI of that hour.

Run:  python src/forecast_ml.py
Outputs: results/forecast_metrics.json, results/forecast_learned_hourly.csv
"""
import json

import numpy as np
import pandas as pd

from params_io import ROOT, load_all
from weather import load_weather

FEATS = ["kt_l1", "kt_l2", "kt_l3", "kt_m7", "t_l1", "trange_l1", "rh_l1", "rh_trend", "doy_sin", "doy_cos"]


def daily_table(w):
    d = w.assign(date=w["timestamp_local"].dt.normalize()).groupby("date").agg(
        ghi=("ghi_w_m2", "sum"), cs=("ghi_clearsky_w_m2", "sum"), t=("ambient_c", "mean"),
        tmax=("ambient_c", "max"), tmin=("ambient_c", "min"), rh=("rh_pct", "mean"))
    d["ghi"], d["cs"] = d["ghi"] / 1000.0, d["cs"] / 1000.0          # kWh/m2/day
    d["kt"] = (d["ghi"] / d["cs"]).clip(0, 1.2)
    d["trange"] = d["tmax"] - d["tmin"]
    d["kt_l1"], d["kt_l2"], d["kt_l3"] = d["kt"].shift(1), d["kt"].shift(2), d["kt"].shift(3)
    d["kt_m7"] = d["kt"].shift(1).rolling(7, min_periods=3).mean()
    d["t_l1"], d["trange_l1"], d["rh_l1"] = d["t"].shift(1), d["trange"].shift(1), d["rh"].shift(1)
    d["rh_trend"] = d["rh"].shift(1) - d["rh"].shift(4)
    d["year"] = d.index.year
    d["doy"] = d.index.dayofyear
    return d


def with_target_season(d, h):
    X = d.copy()
    doy = (X["doy"] + h - 1) % 366 + 1
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    X["y"] = X["kt"].shift(-h)
    X["cs_target"] = X["cs"].shift(-h)
    return X


def make_model(cfg):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=cfg["max_iter"], learning_rate=cfg["learning_rate"],
                                             max_leaf_nodes=cfg["max_leaf_nodes"], random_state=0), "gradient_boosting"
    except ImportError:
        return None, "ridge"


def fit_predict(cfg, Xtr, ytr, Xte):
    model, kind = make_model(cfg)
    if model is not None:
        model.fit(Xtr, ytr)
        return model.predict(Xte), model
    A = np.c_[np.nan_to_num(Xtr), np.ones(len(Xtr))]
    coef = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ ytr)
    return np.c_[np.nan_to_num(Xte), np.ones(len(Xte))] @ coef, None


def climatology(train, test_doy):
    clim = train.groupby("doy")["y"].mean()
    full = clim.reindex(range(1, 367)).interpolate(limit_direction="both")
    smooth = pd.concat([full.iloc[-7:], full, full.iloc[:7]]).rolling(15, center=True).mean().iloc[7:-7]
    smooth.index = range(1, 367)
    return smooth.reindex(test_doy).to_numpy()


def main():
    P = load_all(verbose=False)
    sim, cfg = P["sim"], P["sim"]["forecast_ml"]
    w = load_weather(sim)
    d = daily_table(w)
    feats = [f for f in FEATS if not (f.startswith("rh") and d["rh"].isna().all())]
    years = sorted(d["year"].unique())
    preds, metrics, importance = {}, {}, {}
    for h, name in [(0, "same_day"), (1, "next_day")]:
        X = with_target_season(d, h)
        X["doy_t"] = (X["doy"] + h - 1) % 366 + 1
        out = pd.DataFrame(index=X.index, columns=["learned", "persistence", "climatology"], dtype=float)
        for y in years:
            tr = X[(X["year"] != y)].dropna(subset=feats + ["y"])
            te = X[X["year"] == y]
            p, model = fit_predict(cfg, tr[feats].to_numpy(), tr["y"].to_numpy(), te[feats].to_numpy())
            out.loc[te.index, "learned"] = np.clip(p, 0, 1.1)
            out.loc[te.index, "climatology"] = climatology(tr.assign(doy=tr["doy_t"]), te["doy_t"])
            out.loc[te.index, "persistence"] = te["kt_l1"].to_numpy()
            if model is not None and y == years[-1]:
                try:
                    from sklearn.inspection import permutation_importance
                    ok = te.dropna(subset=feats + ["y"])
                    pi = permutation_importance(model, ok[feats].to_numpy(), ok["y"].to_numpy(),
                                                n_repeats=5, random_state=0, scoring="neg_mean_absolute_error")
                    importance[name] = dict(sorted(zip(feats, pi.importances_mean.tolist()), key=lambda kv: -kv[1]))
                except Exception:
                    pass
        out = out.fillna(out["climatology"].mean())
        ev = X.join(out).dropna(subset=["y", "cs_target"])
        m = {}
        for season, months in [("all", list(range(1, 13))), ("dry", cfg["dry_months"]), ("wet", cfg["wet_months"])]:
            e = ev[ev.index.month.isin(months)]
            res = {}
            for k in ["learned", "persistence", "climatology"]:
                err = (e[k] - e["y"]) * e["cs_target"]                  # kWh/m2/day of sunshine
                res[k] = {"mae_kwh_m2_day": float(err.abs().mean()), "rmse_kwh_m2_day": float(np.sqrt((err ** 2).mean())),
                          "bias_kwh_m2_day": float(err.mean())}
            res["skill_vs_persistence"] = 1 - res["learned"]["mae_kwh_m2_day"] / res["persistence"]["mae_kwh_m2_day"]
            res["skill_vs_climatology"] = 1 - res["learned"]["mae_kwh_m2_day"] / res["climatology"]["mae_kwh_m2_day"]
            res["mean_daily_sun_kwh_m2"] = float((e["y"] * e["cs_target"]).mean())
            m[season] = res
        metrics[name] = m
        preds[name] = out["learned"]

    # hourly forecasts: issued at 00:00 for the same day, and at 00:00 the day before for the next day
    w = w.copy()
    date = w["timestamp_local"].dt.normalize()
    kt0 = preds["same_day"].reindex(date).to_numpy()
    kt1 = preds["next_day"].shift(1).reindex(date).to_numpy()        # issued yesterday for today
    kt1 = np.where(np.isnan(kt1), kt0, kt1)
    hourly = pd.DataFrame({"timestamp_local": w["timestamp_local"],
                           "ghi_fc_sameday": kt0 * w["ghi_clearsky_w_m2"].to_numpy(),
                           "ghi_fc_nextday": kt1 * w["ghi_clearsky_w_m2"].to_numpy()})
    (ROOT / "results").mkdir(exist_ok=True)
    hourly.to_csv(ROOT / "results" / "forecast_learned_hourly.csv", index=False, float_format="%.2f",
                  date_format="%Y-%m-%d %H:%M")
    _, kind = make_model(cfg)
    json.dump({"model": kind, "validation": "leave-one-year-out over " + f"{years[0]}-{years[-1]}",
               "features": feats, "metrics": metrics, "permutation_importance_last_fold": importance},
              open(ROOT / "results" / "forecast_metrics.json", "w"), indent=2)

    print(f"learned forecast ({kind}), leave-one-year-out {years[0]}-{years[-1]}")
    print(f"{'horizon':<10}{'season':<6}{'learned':>9}{'persist.':>10}{'climat.':>9}{'skill vs persist.':>19}  (MAE, kWh/m2/day)")
    for name, m in metrics.items():
        for season, r in m.items():
            print(f"{name:<10}{season:<6}{r['learned']['mae_kwh_m2_day']:>9.3f}{r['persistence']['mae_kwh_m2_day']:>10.3f}"
                  f"{r['climatology']['mae_kwh_m2_day']:>9.3f}{r['skill_vs_persistence']:>18.0%}")
    if importance:
        print("most useful inputs (next day):", list(importance.get("next_day", {}))[:4])


if __name__ == "__main__":
    main()
