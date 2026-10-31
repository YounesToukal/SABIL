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
        temp_fc = np.array([aging_model.battery_temp(a, 0.0, ctx.battery) for a in amb])
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


def get_controller(name, ctx, forecast_mode="perfect"):
    if name == "baseline":
        return Baseline(ctx)
    if name == "smart":
        return Smart(ctx, forecast_mode)
    raise ValueError(f"unknown controller {name!r} (use 'baseline' or 'smart')")
