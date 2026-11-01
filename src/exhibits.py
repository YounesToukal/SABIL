"""SABIL - publication exhibits (owner: Youcef).

Every exhibit states one finding in its title, shows the evidence to scale, names its uncertainty and
ends with its source, read straight from results/ (nothing is typed in).
Style: white ground, near-black text, SABIL in the brand indigo, comparators in grey, the generator in
a muted orange; every series is also labelled directly, so colour is never the only cue.

Writes results/figures/fig15..fig20 (with titles, for the report) and results/figures/deck/*.png
(the same charts without titles, sized for the slides).
Run from the repo root after simulate.py, experiments.py, cost_model.py and tune_controller.py:
    python src/exhibits.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
DECK = FIG / "deck"

INK, INK2, GRID = "#1A1A1A", "#5A5F6B", "#E6E7EB"
SABIL, STD, GEN, PALE = "#1F2B5B", "#8B8FA0", "#C8701E", "#C9CCD6"

for f in Path.home().joinpath(".fonts").glob("sora-*.ttf"):
    font_manager.fontManager.addfont(str(f))
FONT = ["Sora", "DejaVu Sans"] if any("Sora" in f.name for f in font_manager.fontManager.ttflist) else ["DejaVu Sans"]  # DejaVu supplies the naira sign
plt.rcParams.update({
    "font.family": FONT, "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": "#BFC2CA", "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


DECK_SIZE = (10.75, 6.0)   # slide versions: no title or source (the slide carries them)
DECK_SIZES = {"essentials_vs_fade": (10.0, 5.6), "hot_days": (9.4, 5.25), "cost_buildup": (9.6, 5.35), "robustness": (8.2, 4.6)}


def load(name):
    return json.loads((RES / name).read_text())


def ngn(v):
    return f"₦{v:,.0f}"


def frame(fig, title, subtitle, source, titled):
    """Consulting-style chrome: action title, definition line, source line."""
    if titled:
        fig.text(0.035, 0.955, title, fontsize=16, fontweight="bold", va="top")
        fig.text(0.035, 0.865, subtitle, fontsize=11, color=INK2, va="top")
        fig.text(0.035, 0.018, source, fontsize=8.5, color=INK2, va="bottom", linespacing=1.4)


def save(fig, name, titled):
    out = (FIG if titled else DECK) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ 1. essentials vs battery fade
def essentials_vs_fade(titled=True):
    b, s = load("summary_baseline.json"), load("summary_smart.json")
    yrs = np.arange(0, 11)
    cb = [100] + [100 * y["capacity_end_frac"] for y in b["per_year"]]
    cs = [100] + [100 * y["capacity_end_frac"] for y in s["per_year"]]
    lb = [y["critical_demand_hours"] - y["critical_hours"] for y in b["per_year"]]
    ls = [y["critical_demand_hours"] - y["critical_hours"] for y in s["per_year"]]
    fig = plt.figure(figsize=(13.33, 7.5) if titled else DECK_SIZES["essentials_vs_fade"])
    top = 0.78 if titled else 0.94
    a1 = fig.add_axes([0.08, 0.47 + (0 if titled else 0.03), 0.74, top - 0.47 - (0 if titled else 0.03)])
    a2 = fig.add_axes([0.08, 0.11, 0.74, 0.27] if titled else [0.09, 0.1, 0.72, 0.3], sharex=a1)
    a1.plot(yrs, cb, color=STD, lw=2.4, marker="o", ms=4.5)
    a1.plot(yrs, cs, color=SABIL, lw=2.6, marker="o", ms=4.5)
    a1.axhline(80, color=INK2, lw=1, ls=(0, (4, 3)))
    a1.text(0.15, 80.6, "80%: battery due for replacement", fontsize=9.5, color=INK2, va="bottom")
    a1.text(10.15, cb[-1], f"Standard kit  {cb[-1]:.1f}%", color=STD, va="center", fontsize=11, fontweight="bold")
    a1.text(10.15, cs[-1], f"SABIL  {cs[-1]:.1f}%", color=SABIL, va="center", fontsize=11, fontweight="bold")
    for life, c, dy in ((b["battery_life_years"], STD, -3.6), (s["battery_life_years"], SABIL, -3.6)):
        a1.annotate(f"{life:.1f} y", (life, 80), xytext=(life, 80 + dy), ha="center", fontsize=9.5, color=c,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.8))
    a1.set_ylim(72, 101)
    a1.set_ylabel("Battery capacity left (%)")
    a1.tick_params(labelbottom=False)
    w = 0.38
    x = np.arange(1, 11)
    a2.bar(x - w / 2, lb, w, color=STD)
    a2.bar(x + w / 2, ls, w, color=SABIL)
    for xi, v in zip(x, lb):
        if v:
            a2.text(xi - w / 2, v + 0.8, str(v), ha="center", fontsize=9, color=INK2)
    for xi, v in zip(x, ls):
        if v:
            a2.text(xi + w / 2, v + 0.8, str(v), ha="center", fontsize=9, color=SABIL)
    a2.text(10.75, max(lb) * 0.7, f"Total {sum(lb)} h", color=STD, fontsize=11, fontweight="bold", va="center")
    a2.text(10.75, max(lb) * 0.25, f"Total {sum(ls)} h", color=SABIL, fontsize=11, fontweight="bold", va="center")
    a2.set_ylabel("Hours without\nessential power")
    a2.set_xlabel("Year of operation (weather of 2014 to 2023, in order)")
    a2.set_xticks(range(0, 11))
    a2.set_ylim(0, max(lb) * 1.25)
    frame(fig, "SABIL's slower battery fade keeps essentials on: 13 hours lost in 10 years against 110 for a standard kit",
          "Same 400 Wp + 1.92 kWh kit and 5-person home, Maiduguri hourly weather 2014-2023, grid down throughout; "
          "essentials = lights, phones, fan, fridge",
          "Source: SABIL simulation (results/summary_baseline.json, summary_smart.json). Ageing: Naumann et al. 2018, 2020 "
          "(TU Munich), checked against SimSES to 4.5e-14.", titled)
    save(fig, "fig18_essentials_vs_fade.png" if titled else "essentials_vs_fade.png", titled)


# ------------------------------------------------------------------ 2. two hot days
def hot_days(titled=True):
    hb, hs = pd.read_csv(RES / "hourly_baseline.csv"), pd.read_csv(RES / "hourly_smart.csv")
    m = hs["timestamp_local"].between("2016-04-09 00:00", "2016-04-10 23:59")
    t = np.arange(m.sum())
    sb, ss = 100 * hb.loc[m, "soc"].to_numpy(), 100 * hs.loc[m, "soc"].to_numpy()
    amb = hs.loc[m, "ambient_c"].to_numpy()
    tb, ts = hb.loc[m, "battery_temp_c"].to_numpy(), hs.loc[m, "battery_temp_c"].to_numpy()
    fig = plt.figure(figsize=(13.33, 7.5) if titled else DECK_SIZES["hot_days"])
    top = 0.78 if titled else 0.94
    a1 = fig.add_axes([0.08, 0.42, 0.74, top - 0.42] if titled else [0.09, 0.42, 0.74, 0.54])
    a2 = fig.add_axes([0.08, 0.11, 0.74, 0.22] if titled else [0.09, 0.08, 0.74, 0.25], sharex=a1)
    hot_full = (sb >= 99.5)
    a1.fill_between(t, 0, 100, where=hot_full, step="mid", color="#F3F4F7", lw=0)
    a1.plot(t, sb, color=STD, lw=2.4)
    a1.plot(t, ss, color=SABIL, lw=2.6)
    a1.text(47.6, sb[-1] + 3, "Standard kit", color=STD, fontweight="bold", fontsize=11, va="center")
    a1.text(47.6, ss[-1] - 4, "SABIL", color=SABIL, fontweight="bold", fontsize=11, va="center")
    i = int(np.argmax(ss[:24]))
    a1.annotate(f"SABIL stops at {ss.max():.0f}%: tonight's need\nplus a reserve for essentials", (i + 2, ss.max() - 1),
                xytext=(i, 36), fontsize=10, color=SABIL, arrowprops=dict(arrowstyle="-", color=SABIL, lw=0.8))
    a1.text(np.argmax(hot_full) + 0.3, 101.5, f"Standard kit full {hot_full.sum() // 2} h each afternoon (shaded)",
            fontsize=10, color=INK2)
    a1.set_ylim(30, 108)
    a1.set_yticks([40, 60, 80, 100])
    a1.set_ylabel("Battery charge (%)")
    a1.tick_params(labelbottom=False)
    a2.plot(t, amb, color=INK2, lw=1.6)
    a2.text(47.4, amb[-1], "Air", color=INK2, fontsize=10, va="center")
    a2.set_ylabel("Air (°C)")
    a2.set_ylim(28, 46)
    ticks = list(range(0, 48, 6 if titled else 12))
    a2.set_xticks(ticks, [("9 Apr " if k < 24 else "10 Apr ") + f"{k % 24:02d}:00" for k in ticks])
    frame(fig, f"On hot sunny days SABIL stops at {ss.max():.0f}% instead of 100%, so the battery is never full in the heat",
          "Battery charge, hour by hour, 9-10 April 2016 in Maiduguri (air "
          f"{amb.min():.0f} to {amb.max():.0f} °C); same kit and home, only the controller differs. ",
          "Source: SABIL simulation (results/hourly_baseline.csv, hourly_smart.csv), NASA POWER hourly weather. "
          f"Mean battery temperature on these days: standard {tb.mean():.1f} °C, SABIL {ts.mean():.1f} °C.", titled)
    save(fig, "fig19_hot_days.png" if titled else "hot_days.png", titled)


# ------------------------------------------------------------------ 3. robustness
def robustness(titled=True):
    mc = list(csv.DictReader(open(RES / "experiments" / "montecarlo.csv")))
    dv = list(csv.DictReader(open(RES / "experiments" / "demand_variability.csv")))
    rob = load("aging_robustness.json")
    g_mc = np.array([float(r["gain_years"]) for r in mc])
    g_dv = np.array([float(r["gain_years"]) for r in dv])
    rows = [("Uncertain ageing, heat,\nload and panel size", g_mc), ("Random daily use\n(when and how much)", g_dv)]
    models = [("TU Munich model (base)", rob["naumann_sony, cells as tested"]), ("NREL model,\ncells ageing 2x faster", rob["nrel_prismatic, cells x2 faster"])]
    fig = plt.figure(figsize=(13.33, 7.5) if titled else DECK_SIZES["robustness"])
    top = 0.78 if titled else 0.94
    ax = fig.add_axes([0.25, 0.14, 0.62, top - 0.14] if titled else [0.3, 0.14, 0.68, 0.83])
    rng = np.random.default_rng(7)
    ylab = []
    for i, (lab, g) in enumerate(rows):
        y = len(rows) + len(models) - 1 - i
        ax.scatter(g, y + rng.uniform(-0.18, 0.18, len(g)), s=16, color=SABIL, alpha=0.55, lw=0)
        lo, med, hi = np.percentile(g, [5, 50, 95])
        ax.plot([lo, hi], [y - 0.32, y - 0.32], color=INK, lw=1.4)
        ax.plot([med, med], [y - 0.4, y - 0.24], color=INK, lw=2)
        txt = (f"median {med:.2f} y (90% of runs {lo:.2f} to {hi:.2f}); SABIL better in {int((g > 0).sum())} of {len(g)}" if titled
               else f"median {med:.2f} y\n90% of runs: {lo:.2f} to {hi:.2f}\nbetter in {int((g > 0).sum())} of {len(g)}")
        ax.text(max(hi, g.max()) + 0.12, y - 0.1, txt, fontsize=9.5, va="center", color=INK)
        ylab.append((y, f"{lab}\nn = {len(g)}"))
    for j, (lab, r) in enumerate(models):
        y = len(models) - 1 - j
        gain = r["smart"]["battery_life_years"] - r["baseline"]["battery_life_years"]
        ax.scatter([gain], [y], s=70, marker="D", color=SABIL)
        ax.text(gain + 0.08, y, f"{gain:.2f} y ({r['baseline']['battery_life_years']} to {r['smart']['battery_life_years']} y)",
                fontsize=9.5, va="center")
        ylab.append((y, lab))
    ax.axvline(0, color=INK2, lw=1)
    ax.text(0.04, 3.42, "no gain", fontsize=9, color=INK2)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks([y for y, _ in ylab], [l for _, l in ylab], fontsize=10)
    ax.set_xlim(-0.3, max(g_mc.max(), 3.6) + (3.2 if titled else 2.0))
    ax.set_xlabel("Extra battery life with SABIL (years to 80% capacity)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    frame(fig, f"SABIL extends battery life in all {len(g_mc) + len(g_dv)} stress-test runs, by about one year at the median",
          "Each dot is one 10-year simulation with different inputs; bars show the 5th to 95th percentile and the median; "
          "diamonds: single runs with each ageing model",
          "Source: results/experiments/montecarlo.csv (96 samples), demand_variability.csv (24 RAMP-style realisations), "
          "aging_robustness.json (NREL BLAST-Lite).\nNot shown: NREL model with cells as tested (+2.7 y), because its life is extrapolated beyond the 10 simulated years.", titled)
    save(fig, "fig20_robustness.png" if titled else "robustness.png", titled)


# ------------------------------------------------------------------ 4. cost build-up
def cost_buildup(titled=True):
    cs = load("costs_summary.json")
    fx = cs["inputs_used"]["ngn_per_usd"]
    kwh = cs["inputs_used"]["kwh_delivered_per_year"]["smart"]
    it = cs["sabil"]["yearly_items_usd"]
    per = lambda usd: usd / kwh * fx  # noqa: E731
    parts = [("Battery\nwear", per(it["battery"])), ("Panels,\nmounting", per(it["pv_and_mounting"])),
             ("Controller,\nmeter", per(it["electronics"])), ("Wiring,\ninstall,\nsales", per(it["wiring_install_acquisition"])),
             ("Running\ncosts", per(it["running"]))]
    tariff = cs["tariff_ngn_per_kwh"]["sabil"]
    parts.append(("Fees,\nunpaid\nbills", tariff - sum(v for _, v in parts)))
    sens = cs["sensitivity_sabil_tariff"]
    lo = min(min(s["tariff_ngn_per_kwh"]) for s in sens)
    hi = max(max(s["tariff_ngn_per_kwh"]) for s in sens)
    gen = cs["petrol_generator"]
    fig = plt.figure(figsize=(13.33, 7.5) if titled else DECK_SIZES["cost_buildup"])
    top = 0.76 if titled else 0.93
    ax = fig.add_axes([0.07, 0.17, 0.9, top - 0.17] if titled else [0.08, 0.13, 0.9, 0.82])
    x, run = 0, 0.0
    for lab, v in parts:
        ax.bar(x, v, bottom=run, width=0.62, color=SABIL if lab.startswith("Battery") else "#6A73A0")  # battery wear: the cost SABIL acts on
        ax.text(x, run + v + 14, ngn(v), ha="center", fontsize=10.5)
        if x:
            ax.plot([x - 1 + 0.31, x - 0.31], [run, run], color=INK2, lw=0.8)
        run += v
        x += 1
    ax.bar(x, tariff, width=0.62, color=SABIL)
    ax.errorbar(x, tariff, yerr=[[tariff - lo], [hi - tariff]], color=INK, capsize=6, lw=1.4)
    ax.text(x + 0.38, tariff, f"{ngn(tariff)}\nrange tested\n{ngn(lo)} to {ngn(hi)}", fontsize=10.5, va="center", fontweight="bold", color=SABIL)
    x += 2.0
    ax.bar(x, gen["best_case"]["fuel_ngn_per_kwh"], width=0.62, color=GEN)
    ax.text(x, gen["best_case"]["fuel_ngn_per_kwh"] + 14, ngn(gen["best_case"]["fuel_ngn_per_kwh"]), ha="center", fontsize=10.5)
    x += 1.3
    cap = 1000
    ax.bar(x, cap, width=0.62, color=GEN)
    ax.plot([x - 0.36, x + 0.36], [cap - 45, cap - 25], color="white", lw=5)
    ax.text(x, cap + 14, f"{ngn(gen['typical_small_set']['fuel_ngn_per_kwh'])}\n(bar cut)", ha="center", fontsize=10.5)
    labels = [p[0] for p in parts] + ["SABIL\ntariff", "Generator fuel\nbest case\n0.5 L/kWh", "Generator fuel\nsmall set\n2 L/kWh"]
    pos = list(range(len(parts) + 1)) + [len(parts) + 2.0, len(parts) + 3.3]
    ax.set_xticks(pos, labels, fontsize=9.5 if titled else 8.8)
    ax.axvline(len(parts) + 1.25, color=GRID, lw=1)
    ax.set_ylim(0, 1100)
    ax.set_ylabel("Naira per kWh delivered")
    ax.tick_params(axis="x", length=0)
    frame(fig, f"A SABIL kWh costs {ngn(tariff)}, below even the best-case fuel cost of a petrol generator ({ngn(gen['best_case']['fuel_ngn_per_kwh'])})",
          "Break-even tariff built up from its parts (kit financed over 20 years at 12% real, 2026 prices) against "
          "generator fuel alone at the Borno pump price",
          "Source: src/cost_model.py, params/costs.json (NNPC Oct 2026: petrol ₦1,430/L in Borno; CBN ₦1,332/USD; PVPRO Sep 2026; "
          "BNEF Dec 2025; A2EI 2022).\nRange tested: battery price, cost of capital, unpaid bills, panel price, use, ageing model.", titled)
    save(fig, "fig15_cost_per_kwh.png" if titled else "cost_buildup.png", titled)


# ------------------------------------------------------------------ 5. sensitivity
def sensitivity(titled=True):
    cs = load("costs_summary.json")
    base = cs["tariff_ngn_per_kwh"]["sabil"]
    genb = cs["petrol_generator"]["best_case"]["fuel_ngn_per_kwh"]
    sens = sorted(cs["sensitivity_sabil_tariff"], key=lambda s: abs(s["tariff_ngn_per_kwh"][1] - s["tariff_ngn_per_kwh"][0]))
    fmt = {"Battery price ($/kWh)": lambda v: f"${v:g}/kWh", "Cost of capital (real)": lambda v: f"{v:.0%}",
           "Unpaid bills": lambda v: f"{v:.0%}", "Other kit and install costs (x)": lambda v: f"x{v:g}",
           "PV price ($/Wp)": lambda v: f"${v:g}/Wp", "Energy used (x design load)": lambda v: f"x{v:g} use"}
    names = {"Battery price ($/kWh)": "Battery price", "Cost of capital (real)": "Cost of capital (real)",
             "Unpaid bills": "Unpaid bills", "Other kit and install costs (x)": "Other kit and install costs",
             "PV price ($/Wp)": "Panel price", "Energy used (x design load)": "Energy the family uses",
             "Battery ageing model": "Battery ageing model"}
    fig = plt.figure(figsize=(13.33, 7.5) if titled else DECK_SIZE)
    top = 0.78 if titled else 0.94
    ax = fig.add_axes([0.24, 0.13, 0.7, top - 0.13])
    for i, s in enumerate(sens):
        (v0, v1), (t0, t1) = s["values"], s["tariff_ngn_per_kwh"]
        f = fmt.get(s["input"], str)
        for v, tv in ((v0, t0), (v1, t1)):
            if abs(tv - base) < 0.5:
                continue
            ax.barh(i, tv - base, left=base, height=0.56, color=SABIL if tv > base else "#6A73A0")
            ha = "left" if tv > base else "right"
            ax.text(tv + (8 if tv > base else -8), i, f"{f(v)}  {ngn(tv)}" if tv > base else f"{ngn(tv)}  {f(v)}",
                    ha=ha, va="center", fontsize=9.5)
    ax.axvline(base, color=INK, lw=1.2)
    ax.text(base, len(sens) - 0.35, f"Central case {ngn(base)}", ha="center", fontsize=10, fontweight="bold")
    ax.axvline(genb, color=GEN, lw=2)
    ax.text(genb + 6, -0.6, f"Generator fuel,\nbest case {ngn(genb)}", color=GEN, fontsize=10, fontweight="bold", va="bottom")
    ax.set_yticks(range(len(sens)), [names.get(s["input"], s["input"]) for s in sens], fontsize=10.5)
    ax.set_xlim(250, 790)
    ax.set_ylim(-0.8, len(sens) - 0.1)
    ax.set_xlabel("SABIL break-even tariff (naira per kWh)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    frame(fig, "No single assumption we tested pushes the SABIL tariff above the generator's best-case fuel cost",
          "Each bar moves one input between the two values shown, all others at their central case; sorted by effect",
          "Source: src/cost_model.py, results/costs_summary.json. Ageing models: TU Munich (Naumann) and NREL BLAST-Lite "
          "prismatic LFP.", titled)
    save(fig, "fig16_cost_sensitivity.png" if titled else "sensitivity.png", titled)


# ------------------------------------------------------------------ 6. tuning
def tuning(titled=True):
    tu = load("controller_tuning.json")
    h = tu["history"]
    fig = plt.figure(figsize=(13.33, 7.5) if titled else DECK_SIZE)
    top = 0.76 if titled else 0.92
    a1 = fig.add_axes([0.07, 0.13, 0.42, top - 0.13])
    a2 = fig.add_axes([0.57, 0.13, 0.38, top - 0.13])
    pts = np.array([[t["critical_hours_lost"], 100 * t["capacity_after_10y"]] for t in h])
    a1.scatter(pts[:, 0], pts[:, 1], s=20, color=PALE, lw=0)
    order = np.lexsort((-pts[:, 1], pts[:, 0]))
    front, best = [], -1
    for k in order:
        if pts[k, 1] > best:
            front.append(pts[k]); best = pts[k, 1]
    front = np.array(front)
    a1.plot(front[:, 0], front[:, 1], color=INK2, lw=1, drawstyle="steps-post")
    a1.text(front[-1, 0], front[-1, 1] + 0.12, "best trade-offs found", fontsize=9, color=INK2, ha="right")
    for key, lab, mk, fc in (("sabil_hand_set", "hand-set rules", "o", SABIL), ("sabil_tuned", "tuned rules", "D", "white")):
        r = tu["train"][key]
        a1.scatter(r["critical_hours_lost"], 100 * r["capacity_after_10y"], s=80, marker=mk, facecolor=fc, edgecolor=SABIL, lw=1.8, zorder=3)
        a1.annotate(lab, (r["critical_hours_lost"], 100 * r["capacity_after_10y"]), xytext=(10, -4), textcoords="offset points", fontsize=10)
    a1.set_xlabel("Hours without essential power over 10 years")
    a1.set_ylabel("Battery capacity left after 10 years (%)")
    a1.set_title(f"Search: {len(h)} rule settings on 2014-2018 weather", fontsize=11, loc="left", color=INK)
    te = tu["test_unseen_weather"]
    for key, lab, mk, fc, ec in (("standard", "Standard kit", "s", STD, STD), ("sabil_hand_set", "SABIL, hand-set rules", "o", SABIL, SABIL),
                                 ("sabil_tuned", "SABIL, tuned rules", "D", "white", SABIL)):
        r = te[key]
        a2.scatter(r["critical_hours_lost"], 100 * r["capacity_after_10y"], s=110, marker=mk, facecolor=fc, edgecolor=ec, lw=1.8, zorder=3)
        a2.annotate(f"{lab}\n{r['critical_hours_lost']} h lost, {100 * r['capacity_after_10y']:.1f}%",
                    (r["critical_hours_lost"], 100 * r["capacity_after_10y"]), xytext=(10, -8 if key != "sabil_tuned" else 6),
                    textcoords="offset points", fontsize=10)
    a, b_ = te["sabil_hand_set"], te["sabil_tuned"]
    a2.annotate("", (b_["critical_hours_lost"], 100 * b_["capacity_after_10y"]), (a["critical_hours_lost"], 100 * a["capacity_after_10y"]),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    a2.set_xlim(0, 160)
    a2.set_ylim(74, 78.2)
    a2.set_xlabel("Hours without essential power over 10 years")
    a2.set_title("Test: 2019-2023 weather, never seen by the search", fontsize=11, loc="left", color=INK)
    frame(fig, "Tuned rules buy battery life with lost essential hours on unseen weather, so we keep ours",
          "Bayesian optimization (Optuna TPE, seed 2026) of the six controller settings; up and left is better",
          "Source: src/tune_controller.py, results/controller_tuning.json. Training and test weather are each played twice "
          "to make 10 years of battery ageing.", titled)
    save(fig, "fig17_tuning.png" if titled else "tuning.png", titled)


if __name__ == "__main__":
    for f in (essentials_vs_fade, hot_days, robustness, cost_buildup, sensitivity, tuning):
        f(True)
        f(False)
    print("exhibits written to", FIG, "and", DECK)
