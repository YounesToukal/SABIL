"""Forecasting lab (owner: Younes): datasets x features x models, evaluated honestly.

Question: which day-ahead solar forecast should a SABIL controller use, and does a
better forecast actually give the family a longer-lasting battery or fewer blackouts?

Target: daily clearness index kt = GHI / clear-sky GHI (NASA POWER), for the same day
(issued 00:00) and the next day (issued 00:00 the day before). Every feature is known at
issue time (yesterday's values or earlier).

Feature groups (ablation units):
  history    sun of the last days: kt lags, 3/7/14-day means, 7-day spread, trend
  season     day of year (2 harmonics)
  nasa_met   NASA POWER temperature, temperature range, humidity and its trend
  era5_cloud ERA5 cloud cover total/low/mid/high, evening cloud, rain
  era5_dyn   ERA5 dew-point depression, pressure and tendency, wind speed and direction
             (Harmattan vs monsoon), vapour-pressure deficit, soil moisture and trend
  era5_sun   ERA5's own solar radiation (independent estimate)
  climate    NOAA ONI, NAO, TNA (previous month)
Models: persistence, climatology, ridge, random forest, extra trees, HistGradientBoosting,
LightGBM, XGBoost, MLP, an ensemble of the tabular learners, a GRU sequence model
(PyTorch, last 14 days), and Chronos-Bolt (pretrained time-series foundation model,
zero-shot). Probabilistic: quantile gradient boosting + conformalized quantile regression
(Romano et al. 2019) for calibrated 80% intervals.
Validation: leave-one-year-out (train 9 years, test the 10th, rotate). Diebold-Mariano
tests (Newey-West variance) against persistence and climatology. Decision-focused test:
each forecast drives the SABIL controller for 10 years (battery life, unmet energy).

Run:  python src/forecast_lab.py
Outputs: results/forecast_lab/*, results/forecast_learned_hourly.csv (best model + P10),
         results/forecast_metrics.json (summary used by the report)
"""
import json
import math
import time
import warnings

import numpy as np
import pandas as pd

from datasets import load_era5, load_indices
from forecast_ml import climatology
from params_io import ROOT, load_all
from weather import load_weather

warnings.filterwarnings("ignore")
OUT = ROOT / "results" / "forecast_lab"


# ---------------------------------------------------------------- features
def build_daily(w, era, idx):
    d = w.assign(date=w["timestamp_local"].dt.normalize()).groupby("date").agg(
        ghi=("ghi_w_m2", "sum"), cs=("ghi_clearsky_w_m2", "sum"), t=("ambient_c", "mean"),
        tmax=("ambient_c", "max"), tmin=("ambient_c", "min"), rh=("rh_pct", "mean"))
    d["ghi"], d["cs"] = d["ghi"] / 1000.0, d["cs"] / 1000.0
    d["kt"] = (d["ghi"] / d["cs"]).clip(0, 1.2)
    G = {}
    kt1 = d["kt"].shift(1)
    d["kt_l1"], d["kt_l2"], d["kt_l3"] = kt1, d["kt"].shift(2), d["kt"].shift(3)
    d["kt_m3"] = kt1.rolling(3, min_periods=2).mean()
    d["kt_m7"] = kt1.rolling(7, min_periods=3).mean()
    d["kt_m14"] = kt1.rolling(14, min_periods=5).mean()
    d["kt_s7"] = kt1.rolling(7, min_periods=3).std()
    d["kt_trend"] = d["kt_l1"] - d["kt_m7"]
    G["history"] = ["kt_l1", "kt_l2", "kt_l3", "kt_m3", "kt_m7", "kt_m14", "kt_s7", "kt_trend"]
    G["season"] = ["doy_s1", "doy_c1", "doy_s2", "doy_c2"]          # filled per horizon
    d["t_l1"], d["tmax_l1"], d["tmin_l1"] = d["t"].shift(1), d["tmax"].shift(1), d["tmin"].shift(1)
    d["trange_l1"] = d["tmax_l1"] - d["tmin_l1"]
    G["nasa_met"] = ["t_l1", "tmax_l1", "tmin_l1", "trange_l1"]
    if d["rh"].notna().any():
        d["rh_l1"] = d["rh"].shift(1)
        d["rh_trend"] = d["rh_l1"] - d["rh"].shift(4)
        d["rh_m7"] = d["rh"].shift(1).rolling(7, min_periods=3).mean()
        G["nasa_met"] += ["rh_l1", "rh_trend", "rh_m7"]
    if era is not None:
        e = era.assign(date=era["timestamp_local"].dt.normalize(), hour=era["timestamp_local"].dt.hour)
        agg = {}
        for c in ["cloud", "cloud_low", "cloud_mid", "cloud_high", "dewpoint", "t2m", "psurf", "wind", "vpd", "soil_moist"]:
            if c in e:
                agg[c] = e.groupby("date")[c].mean()
        if "precip" in e:
            agg["precip"] = e.groupby("date")["precip"].sum()
        if "cloud" in e:
            agg["cloud_eve"] = e[e["hour"] >= 18].groupby("date")["cloud"].mean()
        if "wind_dir" in e:
            rad = np.radians(e["wind_dir"])
            agg["wdir_s"] = e.assign(s=np.sin(rad)).groupby("date")["s"].mean()
            agg["wdir_c"] = e.assign(c=np.cos(rad)).groupby("date")["c"].mean()
        if "sw_era5" in e:
            agg["sw_era5"] = e.groupby("date")["sw_era5"].sum() / 1000.0
        a = pd.DataFrame(agg).reindex(d.index)
        cl = [c for c in ["cloud", "cloud_low", "cloud_mid", "cloud_high", "cloud_eve", "precip"] if c in a]
        for c in cl:
            d[f"{c}_l1"] = a[c].shift(1)
        if "cloud" in a:
            d["cloud_m3"] = a["cloud"].shift(1).rolling(3, min_periods=2).mean()
            cl_f = [f"{c}_l1" for c in cl] + ["cloud_m3"]
        else:
            cl_f = [f"{c}_l1" for c in cl]
        if "precip" in a:
            d["precip_m7"] = a["precip"].shift(1).rolling(7, min_periods=3).sum()
            cl_f.append("precip_m7")
        G["era5_cloud"] = cl_f
        dyn = []
        if "dewpoint" in a and "t2m" in a:
            d["dewdep_l1"] = (a["t2m"] - a["dewpoint"]).shift(1)
            d["dewpoint_l1"] = a["dewpoint"].shift(1)
            dyn += ["dewdep_l1", "dewpoint_l1"]
        if "psurf" in a:
            d["psurf_l1"] = a["psurf"].shift(1)
            d["psurf_tend"] = a["psurf"].shift(1) - a["psurf"].shift(2)
            dyn += ["psurf_l1", "psurf_tend"]
        for c in ["wind", "wdir_s", "wdir_c", "vpd"]:
            if c in a:
                d[f"{c}_l1"] = a[c].shift(1)
                dyn.append(f"{c}_l1")
        if "soil_moist" in a:
            d["soil_l1"] = a["soil_moist"].shift(1)
            d["soil_trend"] = a["soil_moist"].shift(1) - a["soil_moist"].shift(8)
            dyn += ["soil_l1", "soil_trend"]
        if dyn:
            G["era5_dyn"] = dyn
        if "sw_era5" in a:
            k_era = (a["sw_era5"] / d["cs"]).clip(0, 1.2)
            d["kt_era5_l1"] = k_era.shift(1)
            d["kt_era5_m3"] = k_era.shift(1).rolling(3, min_periods=2).mean()
            G["era5_sun"] = ["kt_era5_l1", "kt_era5_m3"]
    if idx is not None and len(idx):
        prev = d.index.to_period("M") - 1                                 # previous month (published with lag)
        per = pd.PeriodIndex([f"{int(y)}-{int(mo):02d}" for y, mo in zip(idx["year"], idx["month"])], freq="M")
        m = idx.assign(period=per).drop_duplicates("period").set_index("period")
        cols = [c for c in ["oni", "nao", "tna"] if c in m]
        for c in cols:
            d[c] = m[c].reindex(prev).to_numpy()
        if cols:
            G["climate"] = cols
    d["year"], d["doy"] = d.index.year, d.index.dayofyear
    return d, G


def for_horizon(d, h):
    X = d.copy()
    doy = (X["doy"] + h - 1) % 366 + 1
    for k in (1, 2):
        X[f"doy_s{k}"], X[f"doy_c{k}"] = np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)
    X["doy_t"] = doy
    X["y"], X["cs_t"] = X["kt"].shift(-h), X["cs"].shift(-h)
    return X


# ---------------------------------------------------------------- models
def tabular_models(cfg):
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    imp = lambda: SimpleImputer(strategy="median")
    m = {
        "ridge": lambda: make_pipeline(imp(), StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": lambda: make_pipeline(imp(), RandomForestRegressor(n_estimators=cfg["trees"], min_samples_leaf=5, max_features=0.5,
                                                                            n_jobs=-1, random_state=0)),
        "extra_trees": lambda: make_pipeline(imp(), ExtraTreesRegressor(n_estimators=cfg["trees"], min_samples_leaf=3, max_features=0.7,
                                                                        n_jobs=-1, random_state=0)),
        "hist_gb": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                         l2_regularization=1.0, random_state=0),
        "mlp": lambda: make_pipeline(imp(), StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                                                           max_iter=600, random_state=0)),
    }
    try:
        import lightgbm as lgb
        m["lightgbm"] = lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, subsample=0.8,
                                                  subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
    except ImportError:
        pass
    try:
        import xgboost as xgb
        m["xgboost"] = lambda: xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                                                colsample_bytree=0.8, random_state=0, n_jobs=-1)
    except ImportError:
        pass
    return m


def loyo(X, feats, make, years):
    pred = pd.Series(np.nan, index=X.index)
    for y in years:
        tr = X[(X["year"] != y)].dropna(subset=["y"])
        te = X[X["year"] == y]
        mdl = make()
        mdl.fit(tr[feats].to_numpy(float), tr["y"].to_numpy())
        pred.loc[te.index] = mdl.predict(te[feats].to_numpy(float))
    return pred.clip(0, 1.1)


def gru_loyo(X, feats, years, cfg):
    try:
        import torch
        from torch import nn
    except ImportError:
        return None
    torch.manual_seed(0)
    L = cfg["gru_window"]
    F = X[feats].to_numpy(float)
    yv = X["y"].to_numpy(float)
    yr = X["year"].to_numpy()
    n = len(X)
    pred = np.full(n, np.nan)

    class Net(nn.Module):
        def __init__(self, k):
            super().__init__()
            self.gru = nn.GRU(k, cfg["gru_hidden"], batch_first=True)
            self.head = nn.Sequential(nn.Linear(cfg["gru_hidden"], 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out[:, -1]).squeeze(-1)

    idx_all = np.arange(L - 1, n)
    for y in years:
        tr_mask = (yr != y) & ~np.isnan(yv)
        mu, sd = np.nanmean(F[tr_mask], 0), np.nanstd(F[tr_mask], 0) + 1e-6
        Z = np.nan_to_num((F - mu) / sd)
        seq = np.stack([Z[i - L + 1:i + 1] for i in idx_all])
        tr_i = idx_all[tr_mask[idx_all]]
        te_i = idx_all[yr[idx_all] == y]
        Xtr = torch.tensor(seq[tr_i - (L - 1)], dtype=torch.float32)
        ytr = torch.tensor(yv[tr_i], dtype=torch.float32)
        net = Net(Z.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=cfg["gru_lr"], weight_decay=1e-4)
        loss_fn = nn.L1Loss()
        for _ in range(cfg["gru_epochs"]):
            perm = torch.randperm(len(Xtr))
            for b in range(0, len(Xtr), 128):
                j = perm[b:b + 128]
                opt.zero_grad()
                loss = loss_fn(net(Xtr[j]), ytr[j])
                loss.backward()
                opt.step()
        with torch.no_grad():
            pred[te_i] = net(torch.tensor(seq[te_i - (L - 1)], dtype=torch.float32)).numpy()
    return pd.Series(np.clip(pred, 0, 1.1), index=X.index)


def chronos_forecast(d, cfg):
    """Zero-shot Chronos-Bolt on the daily kt series: returns {h: (median, q10, q90)}."""
    try:
        import torch
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained(cfg["chronos_model"], device_map="cpu", torch_dtype=torch.float32)
    except Exception as e:
        print(f"  Chronos skipped ({type(e).__name__}: {str(e)[:120]})")
        return None
    kt = d["kt"].to_numpy(float)
    C = cfg["chronos_context"]
    n = len(kt)
    out = {0: [np.full(n, np.nan) for _ in range(3)], 1: [np.full(n, np.nan) for _ in range(3)]}
    starts = list(range(C, n))
    for b in range(0, len(starts), 256):
        chunk = starts[b:b + 256]
        ctx = [torch.tensor(kt[i - C:i], dtype=torch.float32) for i in chunk]   # history to yesterday
        q, _ = pipe.predict_quantiles(ctx, prediction_length=2, quantile_levels=[0.1, 0.5, 0.9])
        q = q.numpy()
        for j, i in enumerate(chunk):
            for h in (0, 1):
                out[h][0][i], out[h][1][i], out[h][2][i] = q[j, h, 1], q[j, h, 0], q[j, h, 2]
    return {h: tuple(pd.Series(np.clip(a, 0, 1.1), index=d.index) for a in v) for h, v in out.items()}


def conformal_quantiles(X, feats, years, alpha):
    """Quantile gradient boosting (q10, q90), conformalized with a held-out calibration year (CQR)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    lo, hi = pd.Series(np.nan, index=X.index), pd.Series(np.nan, index=X.index)
    q_lo, q_hi = alpha / 2, 1 - alpha / 2
    mk = lambda q: HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=250, learning_rate=0.05,
                                                 max_leaf_nodes=15, random_state=0)
    for y in years:
        train_years = [t for t in years if t != y]
        cal_y = train_years[-1]
        tr = X[X["year"].isin(train_years[:-1])].dropna(subset=["y"])
        cal = X[X["year"] == cal_y].dropna(subset=["y"])
        te = X[X["year"] == y]
        ml, mh = mk(q_lo).fit(tr[feats], tr["y"]), mk(q_hi).fit(tr[feats], tr["y"])
        e = np.maximum(ml.predict(cal[feats]) - cal["y"], cal["y"] - mh.predict(cal[feats]))
        k = min(1.0, (1 - alpha) * (1 + 1 / len(e)))
        qhat = float(np.quantile(e, k))
        lo.loc[te.index] = ml.predict(te[feats]) - qhat
        hi.loc[te.index] = mh.predict(te[feats]) + qhat
    return lo.clip(0, 1.1), hi.clip(0, 1.1)


# ---------------------------------------------------------------- evaluation
def scores(y, p, cs):
    e = (p - y) * cs                                # kWh/m2/day
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean()), "n": int(len(e))}


def diebold_mariano(e1, e2, h=1):
    """DM test on absolute errors; H0 equal accuracy. Negative stat = model 1 better."""
    dlt = np.abs(e1) - np.abs(e2)
    n = len(dlt)
    dbar = dlt.mean()
    g = [np.mean((dlt[k:] - dbar) * (dlt[:n - k] - dbar)) for k in range(h + 1)]
    var = g[0] + 2 * sum((1 - k / (h + 1)) * g[k] for k in range(1, h + 1))
    stat = dbar / math.sqrt(var / n) if var > 0 else 0.0
    p = math.erfc(abs(stat) / math.sqrt(2))
    return float(stat), float(p)


def main():
    t_start = time.time()
    P = load_all(verbose=False)
    sim = P["sim"]
    cfg, seasons_cfg = sim["forecast_lab"], sim["forecast_ml"]
    OUT.mkdir(parents=True, exist_ok=True)
    w = load_weather(sim)
    era, idx = load_era5(sim), load_indices()
    d, G = build_daily(w, era, idx)
    years = sorted(d["year"].unique())
    groups_all = [g for g in ["history", "season", "nasa_met", "era5_cloud", "era5_dyn", "era5_sun", "climate"] if g in G]
    feats_all = sum((G[g] for g in groups_all), [])
    print(f"days {len(d)}, years {years[0]}-{years[-1]}, feature groups: " +
          ", ".join(f"{g} ({len(G[g])})" for g in groups_all))
    missing = [g for g in ["era5_cloud", "era5_dyn", "era5_sun", "climate"] if g not in G]
    if missing:
        print("  not available (download failed or not run): " + ", ".join(missing))

    seasons = {"all": list(range(1, 13)), "dry": seasons_cfg["dry_months"], "wet": seasons_cfg["wet_months"]}
    preds, probs, rows = {}, {}, []
    ch = chronos_forecast(d, cfg) if cfg.get("use_chronos", True) else None
    for h, hname in [(1, "next_day"), (0, "same_day")]:
        X = for_horizon(d, h)
        P_h = {"persistence": X["kt_l1"].clip(0, 1.1), "climatology": pd.Series(np.nan, index=X.index)}
        for y in years:
            tr = X[X["year"] != y].dropna(subset=["y"])
            te = X[X["year"] == y]
            P_h["climatology"].loc[te.index] = climatology(tr.assign(doy=tr["doy_t"]), te["doy_t"])
        models = tabular_models(cfg)
        for name, make in models.items():
            t0 = time.time()
            P_h[name] = loyo(X, feats_all, make, years)
            print(f"  [{hname}] {name:<14} {time.time() - t0:5.1f} s")
        P_h["ensemble"] = pd.concat([P_h[m] for m in models], axis=1).mean(axis=1)
        t0 = time.time()
        g = gru_loyo(X, feats_all, years, cfg)
        if g is not None:
            P_h["gru"] = g
            print(f"  [{hname}] {'gru':<14} {time.time() - t0:5.1f} s")
        if ch is not None:
            P_h["chronos_bolt"] = ch[h][0]
        lo, hi = conformal_quantiles(X, feats_all, years, cfg["conformal_alpha"])
        probs[hname] = {"cqr": (lo, hi)}
        if ch is not None:
            probs[hname]["chronos_bolt"] = (ch[h][1], ch[h][2])
        preds[hname] = P_h
        ok = X.dropna(subset=["y", "cs_t", "kt_l1"])
        for m, p in P_h.items():
            pm = p.reindex(ok.index).fillna(P_h["climatology"].reindex(ok.index))
            for s, months in seasons.items():
                sel = ok.index.month.isin(months)
                r = scores(ok["y"][sel], pm[sel], ok["cs_t"][sel])
                rows.append({"horizon": hname, "model": m, "season": s, **r})
        preds[hname]["_eval_index"] = ok.index
        preds[hname]["_X"] = X

    met = pd.DataFrame(rows)
    for (hz, s), g in met.groupby(["horizon", "season"]):
        pers = g.loc[g["model"] == "persistence", "mae"].iloc[0]
        clim = g.loc[g["model"] == "climatology", "mae"].iloc[0]
        met.loc[g.index, "skill_vs_persistence"] = 1 - g["mae"] / pers
        met.loc[g.index, "skill_vs_climatology"] = 1 - g["mae"] / clim
    met.to_csv(OUT / "metrics.csv", index=False)

    # best learned model (next day, whole year)
    learned = [m for m in preds["next_day"] if not m.startswith("_") and m not in ("persistence", "climatology")]
    nd = met[(met["horizon"] == "next_day") & (met["season"] == "all")].set_index("model")
    best = nd.loc[learned, "mae"].idxmin()

    # Diebold-Mariano: every model vs persistence and vs climatology (next day)
    X1, ok1 = preds["next_day"]["_X"], preds["next_day"]["_eval_index"]
    y1, cs1 = X1.loc[ok1, "y"], X1.loc[ok1, "cs_t"]
    err = {m: ((preds["next_day"][m].reindex(ok1).fillna(preds["next_day"]["climatology"].reindex(ok1)) - y1) * cs1).to_numpy()
           for m in learned + ["persistence", "climatology"]}
    dm = {m: {"vs_persistence": diebold_mariano(err[m], err["persistence"], 1),
              "vs_climatology": diebold_mariano(err[m], err["climatology"], 1)} for m in learned}

    # probabilistic: coverage, width, pinball
    prob_out = {}
    for name, (lo, hi) in probs["next_day"].items():
        lo, hi = lo.reindex(ok1), hi.reindex(ok1)
        good = lo.notna() & hi.notna()
        yy = y1[good]
        cover = float(((yy >= lo[good]) & (yy <= hi[good])).mean())
        width = float(((hi[good] - lo[good]) * cs1[good]).mean())
        a = cfg["conformal_alpha"]
        pin = lambda q, tau: float(np.mean(np.maximum(tau * (yy - q), (tau - 1) * (yy - q)) * cs1[good]))
        prob_out[name] = {"target_coverage": 1 - a, "coverage": cover, "mean_width_kwh_m2_day": width,
                          "pinball_lo": pin(lo[good], a / 2), "pinball_hi": pin(hi[good], 1 - a / 2)}

    # feature-group ablation (next day, hist_gb)
    abl = []
    X = preds["next_day"]["_X"]
    make = tabular_models(cfg)[cfg["ablation_model"]]
    base_groups = ["history", "season"]

    def mae_of(groups):
        fs = sum((G[g] for g in groups), [])
        p = loyo(X, fs, make, years).reindex(ok1)
        return float((np.abs(p - y1) * cs1).mean())

    full = mae_of(groups_all)
    base_mae = mae_of(base_groups)
    for g in groups_all:
        drop = mae_of([x for x in groups_all if x != g]) if len(groups_all) > 1 else full
        add = mae_of(base_groups + [g]) if g not in base_groups else base_mae
        abl.append({"group": g, "n_features": len(G[g]), "mae_without": drop, "delta_if_removed": drop - full,
                    "mae_base_plus_group": add, "gain_when_added_to_base": base_mae - add})
    abl_df = pd.DataFrame(abl)
    abl_df.to_csv(OUT / "feature_group_ablation.csv", index=False)

    # decision-focused test: each forecast drives the controller for 10 years
    from simulate import run
    w_sim = w
    clear = w["ghi_clearsky_w_m2"].to_numpy()
    date = w["timestamp_local"].dt.normalize()

    def hourly_from(pred_same, pred_next):
        k0 = pred_same.reindex(date).to_numpy()
        k1 = pred_next.shift(1).reindex(date).to_numpy()
        k0 = np.where(np.isnan(k0), k1, k0)
        k1 = np.where(np.isnan(k1), k0, k1)
        return np.nan_to_num(k0, nan=0.7) * clear, np.nan_to_num(k1, nan=0.7) * clear

    decision = []
    cand = ["persistence", "climatology"] + learned
    for m in cand:
        same = preds["same_day"].get(m, preds["same_day"]["persistence"])
        nxt = preds["next_day"][m]
        s, _ = run("smart", P, w_sim, forecast_mode="learned", ghi_forecast=hourly_from(same, nxt))
        decision.append({"forecast_model": m, "battery_life_years": s["battery_life_years"],
                         "critical_service_frac": s["critical_service_frac"],
                         "energy_unmet_kwh_per_year": s["energy_unmet_kwh_per_year"],
                         "next_day_mae": float(nd.loc[m, "mae"])})
    lo1, _ = probs["next_day"]["cqr"]
    lo0, _ = probs["same_day"]["cqr"]
    s, _ = run("smart", P, w_sim, forecast_mode="learned", ghi_forecast=hourly_from(lo0, lo1))
    decision.append({"forecast_model": "risk_aware_p10", "battery_life_years": s["battery_life_years"],
                     "critical_service_frac": s["critical_service_frac"],
                     "energy_unmet_kwh_per_year": s["energy_unmet_kwh_per_year"], "next_day_mae": None})
    pd.DataFrame(decision).to_csv(OUT / "decision_focused.csv", index=False)

    # hourly forecast for the simulation: best model + conformal P10
    g0, g1 = hourly_from(preds["same_day"][best], preds["next_day"][best])
    r0, r1 = hourly_from(lo0, lo1)
    pd.DataFrame({"timestamp_local": w["timestamp_local"], "ghi_fc_sameday": g0, "ghi_fc_nextday": g1,
                  "ghi_fc_sameday_p10": r0, "ghi_fc_nextday_p10": r1}).to_csv(
        ROOT / "results" / "forecast_learned_hourly.csv", index=False, float_format="%.2f", date_format="%Y-%m-%d %H:%M")

    # summary in the schema the report already reads
    def block(hz):
        out = {}
        for s in seasons:
            g = met[(met["horizon"] == hz) & (met["season"] == s)].set_index("model")
            out[s] = {k: {"mae_kwh_m2_day": float(g.loc[k, "mae"]), "rmse_kwh_m2_day": float(g.loc[k, "rmse"]),
                          "bias_kwh_m2_day": float(g.loc[k, "bias"])} for k in ["persistence", "climatology"]}
            out[s]["learned"] = {"mae_kwh_m2_day": float(g.loc[best, "mae"]), "rmse_kwh_m2_day": float(g.loc[best, "rmse"]),
                                 "bias_kwh_m2_day": float(g.loc[best, "bias"])}
            out[s]["skill_vs_persistence"] = float(g.loc[best, "skill_vs_persistence"])
            out[s]["skill_vs_climatology"] = float(g.loc[best, "skill_vs_climatology"])
        return out

    summary = {"model": best, "validation": f"leave-one-year-out over {years[0]}-{years[-1]}",
               "features": feats_all, "feature_groups": {g: G[g] for g in groups_all},
               "groups_missing": missing, "models_compared": learned,
               "metrics": {"next_day": block("next_day"), "same_day": block("same_day")},
               "diebold_mariano_next_day": dm, "probabilistic_next_day": prob_out,
               "permutation_importance_last_fold": {}, "runtime_s": round(time.time() - t_start, 1)}
    json.dump(summary, open(ROOT / "results" / "forecast_metrics.json", "w"), indent=2)
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)

    # print
    print(f"\nNEXT-DAY LEADERBOARD (MAE kWh/m2/day, leave-one-year-out; skill vs persistence / vs climatology)")
    for s in ["all", "dry", "wet"]:
        g = met[(met["horizon"] == "next_day") & (met["season"] == s)].sort_values("mae")
        print(f" {s}: " + " | ".join(f"{r.model} {r.mae:.3f}" for r in g.head(6).itertuples()))
    print(f"\nbest learned model: {best}")
    print("Diebold-Mariano (next day, negative = better than reference; p-value):")
    for m in learned:
        a, b = dm[m]["vs_persistence"], dm[m]["vs_climatology"]
        print(f"   {m:<14} vs persistence {a[0]:+6.2f} (p={a[1]:.3f})   vs climatology {b[0]:+6.2f} (p={b[1]:.3f})")
    print("\nprobabilistic 80% intervals (next day):")
    for k, v in prob_out.items():
        print(f"   {k:<14} coverage {v['coverage']:.1%} (target {v['target_coverage']:.0%}), width {v['mean_width_kwh_m2_day']:.2f} kWh/m2/day")
    print("\nfeature-group ablation (hist_gb, next day, MAE change if the group is removed; >0 = group helps):")
    for r in abl_df.sort_values("delta_if_removed", ascending=False).itertuples():
        print(f"   {r.group:<11} {r.delta_if_removed:+.4f}   (gain when added to history+season: {r.gain_when_added_to_base:+.4f})")
    print("\ndecision-focused (each forecast drives SABIL for 10 years):")
    for r in decision:
        print(f"   {r['forecast_model']:<16} life {r['battery_life_years']} y, critical {r['critical_service_frac']:.2%}, "
              f"unmet {r['energy_unmet_kwh_per_year']:.1f} kWh/y")
    print(f"\nforecast lab done in {time.time() - t_start:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
