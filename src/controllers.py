"""The two controllers (owner: Younes).

Both controllers expose the same two methods and are picked by name with
get_controller("baseline" | "smart", ...):

  plan_day(t0, ctx)  -> (flex_critical_kw[24], flex_noncritical_kw[24])
      called at local 00:00; places the flexible loads for the day.
  decide(t, state, ctx) -> dict
      called every hour; returns the limits the battery must respect:
        soc_max           charge ceiling (fraction of current capacity)
        soc_floor_crit    lowest SoC allowed when serving critical loads
        soc_floor_nonc    lowest SoC allowed when serving non-critical loads
        chg_c, dis_c_crit, dis_c_nonc   power limits as C-rates
        priority          True = serve critical loads first; False = cut everything
                          together (what a low-voltage disconnect does)

Standard controller (what cheap kits do): charge to soc_max_baseline whenever
there is sun, discharge to soc_min_baseline, flexible loads at their usual hour,
no priority, heat ignored, no forecast.

SABIL smart controller (ours), using a next-day sun and load forecast:
  - caps the charge level on hot days when tomorrow is sunny (just enough for
    tonight + margin, never below soc_cap_hot); charges full before a cloudy day
  - keeps a reserve sized for the critical loads until the sun is back
  - moves flexible loads to the sunniest allowed hours
  - serves critical loads first; non-critical loads only above the reserve
  - limits charge / non-critical discharge power when the battery is hot

Forecasts: "perfect" = the real future data (upper bound);
"persistence" = same hour on the most recent past day (no look-ahead, realistic).
"""
import numpy as np

import aging_model


class Context:
    """Arrays shared by the engine and the controllers (one entry per simulated hour)."""

    def __init__(self, pv_dc, ambient_c, crit_fixed, nonc_fixed, hod, flex, battery, system, sim):
        self.pv_dc, self.ambient_c = pv_dc, ambient_c
        self.crit_fixed, self.nonc_fixed = crit_fixed, nonc_fixed
        self.hod, self.flex = hod, flex
        self.battery, self.system, self.sim = battery, system, sim
        self.n = len(pv_dc)
        self.inv = system["inverter_eff"]


def forecast(arr, t, horizon, mode):
    """Forecast of arr[t : t+horizon] issued at hour t."""
    n = len(arr)
    k = np.arange(horizon)
    if mode == "perfect":
        idx = t + k
    elif mode == "persistence":
        idx = t + k - 24 * (k // 24 + 1)  # same hour, most recent day already observed
    else:
        raise ValueError(f"unknown forecast mode {mode!r}")
    return arr[idx % n]


class Baseline:
    name = "baseline"

    def __init__(self, ctx):
        b = ctx.battery
        self.limits = dict(soc_max=b["soc_max_baseline"], soc_floor_crit=b["soc_min_baseline"],
                           soc_floor_nonc=b["soc_min_baseline"], chg_c=b["max_c_rate"],
                           dis_c_crit=b["max_c_rate"], dis_c_nonc=b["max_c_rate"], priority=False)
        self.usual = self._usual(ctx)

    @staticmethod
    def _usual(ctx):
        c, nc = np.zeros(24), np.zeros(24)
        for f in ctx.flex:
            for h in f["use_hours"]:
                (c if f["critical"] else nc)[h] += f["kw"]
        return c, nc

    def plan_day(self, t0, ctx):
        return self.usual

    def decide(self, t, state, ctx):
        return self.limits


class Smart:
    name = "smart"

    def __init__(self, ctx, mode):
        self.mode = mode
        self.s = ctx.sim["smart"]
        b = ctx.battery
        self.soc_min, self.soc_max_b = b["soc_min_baseline"], b["soc_max_baseline"]
        self.max_c = b["max_c_rate"]
        self.eta_out = b["discharge_eff"] * ctx.inv  # stored kWh -> AC kWh
        self.f = self.s.get("features", {})
        self.today = None

    def _on(self, feature):
        return self.f.get(feature, True)

    def _pv_forecast(self, ctx, t0, H):
        if self.mode == "learned":
            idx = (t0 + np.arange(H)) % ctx.n
            return np.concatenate([ctx.pv_fc_same[idx[:24]], ctx.pv_fc_next[idx[24:]]])
        return forecast(ctx.pv_dc, t0, H, self.mode)

    # ---------- daily plan ----------
    def plan_day(self, t0, ctx):
        H = 48
        pv_ac = self._pv_forecast(ctx, t0, H) * ctx.inv
        amb = forecast(ctx.ambient_c, t0, 24, "persistence" if self.mode == "learned" else self.mode)
        crit = forecast(ctx.crit_fixed, t0, H, "perfect")  # household schedule, known
        nonc = forecast(ctx.nonc_fixed, t0, H, "perfect")

        # 1. flexible loads -> sunniest allowed hours (greedy on forecast surplus)
        fc, fnc = np.zeros(24), np.zeros(24)
        surplus = pv_ac[:24] - crit[:24] - nonc[:24]
        for f in sorted(ctx.flex, key=lambda f: -f["kw"]):
            hours = (f["allowed_hours"] or f["use_hours"]) if self._on("load_shift") else f["use_hours"]
            best = sorted(hours, key=lambda h: -surplus[h])[: f["n_hours"]]
            for h in best:
                (fc if f["critical"] else fnc)[h] += f["kw"]
                surplus[h] -= f["kw"]
        crit[:24] += fc
        nonc[:24] += fnc
        load = crit + nonc

        # 2. hot day? tomorrow sunny?
        est = getattr(aging_model, "battery_temp_estimate", None)   # stateless if the model provides it
        temp_fc = np.array([est(a, ctx.battery) if est else aging_model.battery_temp(a, 0.0, ctx.battery) for a in amb])
        hot = float(np.mean(temp_fc)) >= self.s["hot_threshold_c"]
        sunny_tomorrow = pv_ac[24:48].sum() >= self.s["sunny_ratio"] * load[24:48].sum()

        # 3. energy tonight: from the first evening hour the sun no longer covers the load
        #    until the sun covers it again tomorrow (AC kWh)
        deficit = np.maximum(0.0, load - pv_ac)
        start = next((k for k in range(12, H) if pv_ac[k] < load[k]), H)
        end = next((k for k in range(max(start, 24), H) if pv_ac[k] >= load[k]), H)
        e_night = deficit[start:end].sum()

        # 4. reserve: critical energy needed over the next dark period after hour h
        #    (during the day this protects tonight's lights, phones and fan)
        crit_def = np.maximum(0.0, crit - pv_ac)
        sun_ok = (pv_ac > 0) & (pv_ac >= crit)
        rem_crit = np.zeros(24)
        for h in range(24):
            k = h + 1
            while k < H and sun_ok[k]:      # skip the rest of today's sunny hours
                k += 1
            e = 0.0
            while k < H and not sun_ok[k]:  # sum the dark period until the sun is back
                e += crit_def[k]
                k += 1
            rem_crit[h] = e

        self.today = dict(t0=t0, hot=hot, sunny_tomorrow=sunny_tomorrow, e_night=e_night, rem_crit=rem_crit)
        return fc, fnc

    # ---------- hourly decision ----------
    def decide(self, t, state, ctx):
        d, s = self.today, self.s
        cap = state["capacity_kwh"]
        m = 1.0 + s["reserve_margin"]
        h = t - d["t0"]

        need_soc = self.soc_min + d["e_night"] * m / (cap * self.eta_out)
        if self._on("charge_cap") and d["hot"] and d["sunny_tomorrow"]:
            soc_max = min(self.soc_max_b, max(s["soc_cap_hot"], need_soc))
        else:
            soc_max = self.soc_max_b

        if self._on("night_reserve"):
            reserve = min(soc_max, self.soc_min + d["rem_crit"][h] * m / (cap * self.eta_out))
        else:
            reserve = self.soc_min

        hot_now = state["last_battery_temp_c"] >= s["hot_threshold_c"]
        gentle = s["c_rate_hot"] if (hot_now and self._on("heat_gentle")) else self.max_c
        return dict(soc_max=soc_max, soc_floor_crit=self.soc_min, soc_floor_nonc=reserve,
                    chg_c=gentle, dis_c_crit=self.max_c, dis_c_nonc=gentle,
                    priority=self._on("critical_priority"),
                    hot=d["hot"], sunny_tomorrow=d["sunny_tomorrow"])


class Oracle:
    """Benchmark, not a product: degradation-aware dynamic programming with PERFECT foresight.
    Every day at 00:00 it solves, over the next `horizon_h` hours, the battery schedule that
    minimises  lambda_c * unmet critical kWh + lambda_n * unmet other kWh + lambda_deg * capacity lost,
    with capacity loss from the same ageing model (marginal sqrt-law rates at the current age).
    Flexible loads are placed exactly as SABIL places them. Answers: how far are SABIL's simple
    rules from the best achievable schedule?"""
    name = "oracle"

    def __init__(self, ctx, lam_deg):
        self.o = ctx.sim["oracle"]
        self.lam_deg = lam_deg
        self.helper = Smart(ctx, "perfect")
        b = ctx.battery
        self.b = b
        self.grid = np.linspace(b["soc_min_baseline"], b["soc_max_baseline"], self.o["grid_levels"])
        self.plan = None

    def _deg_rates(self, temp, s_avg, dsoc, cap0):
        import aging_model as am
        a = self.b["aging"]
        st = am.breakdown() if hasattr(am, "breakdown") else {"calendar_loss_frac": 0.01, "cycle_loss_frac": 0.001}
        q_cal = max(st["calendar_loss_frac"], 1e-3)
        q_cyc_pct = max(st["cycle_loss_frac"] * 100.0, 0.05)
        T = temp + 273.15
        k = a["cal_k_ref_per_sqrt_s"] * np.exp(-a["cal_ea_j_mol"] / am.R_GAS * (1 / T - 1 / a["cal_t_ref_k"])) \
            * (a["cal_soc_a"] * (s_avg - 0.5) ** 3 + a["cal_soc_b"])
        d_cal = k ** 2 * 3600.0 / (2.0 * q_cal)
        kc = (a["cyc_c_a"] * np.abs(dsoc) + a["cyc_c_b"]) * a["cyc_dod_d"]
        d_cyc = kc ** 2 / (2.0 * q_cyc_pct) * (np.abs(dsoc) / 2.0) / 100.0
        return d_cal + d_cyc

    def plan_day(self, t0, ctx):
        fc, fnc = self.helper.plan_day(t0, ctx)
        H = self.o["horizon_h"]
        idx = (t0 + np.arange(H)) % ctx.n
        pv = ctx.pv_dc[idx]
        crit = ctx.crit_fixed[idx] + np.r_[fc, fc][:H]
        nonc = ctx.nonc_fixed[idx] + np.r_[fnc, fnc][:H]
        est = getattr(__import__("aging_model"), "battery_temp_estimate", None)
        temps = np.array([est(a, self.b) if est else a + self.b["temp_offset_c"] for a in ctx.ambient_c[idx]])
        inv, ce, de = ctx.inv, self.b["charge_eff"], self.b["discharge_eff"]
        cap = self._cap if hasattr(self, "_cap") else self.b["capacity_kwh"]
        g = self.grid
        S, S2 = g[:, None], g[None, :]
        dS = S2 - S
        lam_c, lam_n = self.o["lambda_critical"], self.o["lambda_noncritical"]
        step_e = (g[1] - g[0]) * cap
        V = np.zeros(len(g))
        choice = np.zeros((H, len(g)), int)
        for j in range(H - 1, -1, -1):
            pv_ac = pv[j] * inv
            pv_c = min(pv_ac, crit[j]); pv_n = min(pv_ac - pv_c, nonc[j])
            c_def, n_def = crit[j] - pv_c, nonc[j] - pv_n
            surplus = max(0.0, pv[j] - (pv_c + pv_n) / inv)
            e_in = np.where(dS > 0, dS * cap / ce, 0.0)
            e_out = np.where(dS < 0, -dS * cap * de * inv, 0.0)
            feas = (e_in <= min(surplus, self.b["max_c_rate"] * cap) + 1e-9) & \
                   (e_out <= min(c_def + n_def + step_e * de * inv, self.b["max_c_rate"] * cap * inv) + 1e-9)
            deliv = np.minimum(e_out, c_def + n_def)
            un_c = np.maximum(0.0, c_def - deliv)
            un_n = np.maximum(0.0, n_def - np.maximum(0.0, deliv - c_def))
            deg = self._deg_rates(temps[j], (S + S2) / 2.0, dS, self.b["capacity_kwh"])
            cost = lam_c * un_c + lam_n * un_n + self.lam_deg * deg + V[None, :]
            cost = np.where(feas, cost, np.inf)
            choice[j] = np.argmin(cost, axis=1)
            V = cost[np.arange(len(g)), choice[j]]
        self.choice, self.t0 = choice, t0
        return fc, fnc

    def decide(self, t, state, ctx):
        """Follow the plan's direction, not its rounded level: charge freely unless the plan
        holds the battery below its current level; discharge for non-critical loads only when
        the plan discharges, down to the planned level (relative to the actual SoC)."""
        self._cap = state["capacity_kwh"]
        h = t - self.t0
        soc = state["soc"]
        k = int(np.argmin(np.abs(self.grid - soc)))
        d_plan = self.grid[self.choice[h, k]] - self.grid[k]
        floor, top = self.b["soc_min_baseline"], self.b["soc_max_baseline"]
        step = self.grid[1] - self.grid[0]
        if d_plan > 0:
            soc_max, floor_n = min(top, soc + d_plan + step), soc
        elif d_plan < 0:
            soc_max, floor_n = soc, max(floor, soc + d_plan - step / 2)
        else:
            soc_max, floor_n = min(top, soc + step), soc
        return dict(soc_max=soc_max, soc_floor_crit=floor, soc_floor_nonc=floor_n,
                    chg_c=self.b["max_c_rate"], dis_c_crit=self.b["max_c_rate"], dis_c_nonc=self.b["max_c_rate"],
                    priority=True)


def get_controller(name, ctx, forecast_mode="perfect"):
    if name == "baseline":
        return Baseline(ctx)
    if name == "smart":
        return Smart(ctx, forecast_mode)
    if name.startswith("oracle"):
        lam = float(name.split(":")[1]) if ":" in name else ctx.sim["oracle"]["lambda_degradation"]
        return Oracle(ctx, lam)
    raise ValueError(f"unknown controller {name!r} (use 'baseline' or 'smart')")
