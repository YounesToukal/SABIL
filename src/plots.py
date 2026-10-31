"""Charts for the deck and the demo video (owner: Younes).

Reads results/summary_*.json and results/hourly_*.csv, writes results/figures/*.png.
Run:  python src/plots.py
Style: standard controller in grey, SABIL in blue (the bar that matters is the
coloured one), one axis per chart, headlines are sentences.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from params_io import ROOT, load_all

RES = ROOT / "results"
FIG = RES / "figures"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
GREY, BLUE, ORANGE = "#9a9993", "#2a78d6", "#eb6834"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.titlesize": 13, "axes.titlelocation": "left", "font.size": 10.5,
        "legend.frameon": False, "lines.linewidth": 2.0, "figure.dpi": 110})


def _load(tag):
    p = RES / f"summary_{tag}.json"
    return json.load(open(p)) if p.exists() else None


def _hourly(tag):
    p = RES / f"hourly_{tag}.csv"
    return pd.read_csv(p, parse_dates=["timestamp_local"]) if p.exists() else None


def _subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom")


def fig_capacity(base, smart, alt, hb, hs, headline):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    n_years = base["years_simulated"]
    for h, color, label in [(hb, GREY, "Standard controller"), (hs, BLUE, f"SABIL ({headline} forecast)")]:
        d = h.set_index("timestamp_local")["capacity_frac"].resample("D").last().to_numpy()
        x = np.arange(len(d)) / (len(d) / n_years)
        ax.plot(x, d, color=color, label=label)
        ax.annotate(f"{d[-1]:.0%}", (x[-1], d[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=10)
    if alt:
        caps = [1.0] + [y["capacity_end_frac"] for y in alt["per_year"]]
        ax.plot(range(len(caps)), caps, color=BLUE, ls="--", lw=1.4, marker="o", ms=4,
                label=f"SABIL ({alt['forecast']} forecast)")
    eol = load_all(verbose=False)["battery"]["eol_capacity_frac"]
    ax.axhline(eol, color=INK2, lw=1, ls=":")
    ax.text(n_years, eol, f"end of life ({eol:.0%}) ", va="bottom", ha="right", color=INK2, fontsize=9)
    lo = min(float(hb["capacity_frac"].min()), float(hs["capacity_frac"].min()), eol)
    ax.set_ylim(lo - 0.03, 1.01)
    ax.set_xlabel("Years in service")
    ax.set_ylabel("Battery capacity left")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    lb, ls_ = base["battery_life_years"], smart["battery_life_years"]
    if lb and ls_:
        title = f"SABIL battery lasts {ls_:.1f} years vs {lb:.1f} with a standard controller"
    else:
        title = "Battery capacity over 10 years (waiting for the real ageing model)"
    ax.set_title(title, pad=22)
    _subtitle(ax, f"Same house, same Maiduguri weather (NASA POWER, {base.get('weather', '')}), same battery; "
                  "only the controller changes")
    ax.legend(loc="lower left" if lo < eol - 0.01 else "center left")
    ax.set_xlim(0, n_years + 0.8)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_battery_capacity.png")
    plt.close(fig)


def fig_critical(base, smart, headline):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    yb = np.array([y["critical_service_frac"] for y in base["per_year"]])
    ys = np.array([y["critical_service_frac"] for y in smart["per_year"]])
    x = np.arange(1, len(yb) + 1)
    ax.plot(x, yb, color=GREY, marker="o", ms=8, mec=SURFACE, mew=2, label="Standard controller")
    ax.plot(x, ys, color=BLUE, marker="o", ms=8, mec=SURFACE, mew=2, label=f"SABIL ({headline} forecast)")
    for arr in (yb, ys):
        ax.annotate(f"{arr[-1]:.0%}", (x[-1], arr[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=10)
    ax.set_xticks(x)
    ax.set_xlim(0.6, len(x) + 0.8)
    ax.set_xlabel("Year in service")
    ax.set_ylabel("Share of hours with lights, phones, fan on")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    lo = float(min(yb.min(), ys.min()))
    ax.set_ylim(min(lo - 0.02, 0.9), 1.01)
    lost = (smart["critical_hours_per_year"] - base["critical_hours_per_year"]) * base["years_simulated"]
    ax.set_title(f"{lost:,.0f} more hours of critical service per home over 10 years", pad=22)
    _subtitle(ax, "Hours where every critical load was fully powered / hours they were needed; grid down all year")
    ax.legend(loc="lower left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_critical_hours.png")
    plt.close(fig)


def fig_days(hb, hs, headline):
    n1 = len(hb) // max(1, round(len(hb) / 8760))
    y1b, y1s = hb.iloc[:n1], hs.iloc[:n1]
    daily = y1b.set_index("timestamp_local")["ambient_c"].resample("D").mean()
    start = daily.rolling(3).mean().idxmax() - pd.Timedelta(days=2)
    sel = (y1b["timestamp_local"] >= start) & (y1b["timestamp_local"] < start + pd.Timedelta(days=3))
    b, s = y1b[sel], y1s[sel]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    a1.fill_between(b["timestamp_local"], b["pv_kw"], color=ORANGE, alpha=0.25, lw=0, label="Solar available")
    a1.plot(b["timestamp_local"], b["load_kw"], color=GREY, lw=1.4, label="Load, standard")
    a1.plot(s["timestamp_local"], s["load_kw"], color=BLUE, lw=1.4, label="Load, SABIL (iron moved to noon)")
    a1.set_ylabel("kW")
    a1.legend(loc="upper right", fontsize=8.5, ncol=3)
    a1.set_title("Hot days: SABIL holds the battery lower, the lights stay on", pad=8)
    a2.plot(b["timestamp_local"], b["soc"], color=GREY, label="Standard: charges to the top every day")
    a2.plot(s["timestamp_local"], s["soc"], color=BLUE, label=f"SABIL ({headline} forecast)")
    a2.set_ylabel("Battery state of charge")
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a2.set_ylim(0, 1.05)
    a2.legend(loc="lower left", fontsize=8.5)
    a2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b %H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "fig3_hot_days.png")
    plt.close(fig)


def fig_headline(base, smart, alt, headline):
    rows = [("Standard controller", base, GREY), (f"SABIL, {headline} forecast", smart, BLUE)]
    if alt:
        rows.append((f"SABIL, {alt['forecast']} forecast", alt, "#7fb0ea"))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    metrics = [("battery_life_years", "Battery life (years)", "{:.1f}"),
               ("critical_service_frac", "Critical hours covered", "{:.1%}"),
               ("diesel_avoided_l_per_year", "Diesel avoided (L/year)", "{:,.0f}")]
    for ax, (k, label, fmt) in zip(axes, metrics):
        vals = [r[1][k] if r[1][k] is not None else 0 for r in rows]
        y = np.arange(len(rows))[::-1]
        ax.barh(y, vals, color=[r[2] for r in rows], height=0.6)
        for yi, v, r in zip(y, vals, rows):
            txt = fmt.format(v) if r[1][k] is not None else "n/a"
            ax.text(v, yi, " " + txt, va="center", color=INK, fontsize=10)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows] if ax is axes[0] else [])
        ax.set_title(label, fontsize=11)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.35 if max(vals) > 0 else 1)
        ax.set_xticks([])
        ax.tick_params(left=False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    fig.suptitle("SABIL vs a standard controller: one household in Maiduguri, 10 years, grid down",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_headline_numbers.png")
    plt.close(fig)


FEATURE_LABEL = {"charge_cap": "Lower charge cap on hot, sunny days", "night_reserve": "Night reserve for critical loads",
                 "load_shift": "Move flexible loads to noon", "heat_gentle": "Gentler power when hot",
                 "critical_priority": "Critical loads served first"}


def fig_ablation(ab):
    rows = ab["contributions"]
    rows = sorted(rows, key=lambda r: r["life_lost_if_removed_years"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(rows))
    vals = [r["life_lost_if_removed_years"] for r in rows]
    ax.barh(y, vals, color=[BLUE if v > 0.05 else GREY for v in vals], height=0.6)
    for yi, v, r in zip(y, vals, rows):
        crit = r["critical_service_if_removed"]
        ax.text(max(v, 0), yi, f"  {v:+.2f} y   (lights on {crit:.1%} without it)", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([FEATURE_LABEL.get(r["feature"], r["feature"]) for r in rows])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Battery life lost when this feature is switched off (years)")
    ax.set_xlim(min(0, min(vals)) - 0.2, max(vals) * 1.9 + 0.1)
    ax.grid(axis="y", visible=False)
    ax.set_title("What each SABIL feature is worth", pad=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_ablation.png")
    plt.close(fig)


def fig_tradeoff(sw):
    pts = [p for p in sw["points"] if p["battery_life_years"]]
    if not pts:
        return
    base = sw["baseline"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = [p["energy_served_kwh_per_year"] for p in pts]
    y = [p["battery_life_years"] for p in pts]
    ax.plot(x, y, color=BLUE, marker="o", ms=7, mec=SURFACE, mew=2, label="SABIL, charge cap from 40% to 100%")
    for p in pts:
        if abs(p["soc_cap_hot"] - sw["chosen"]) < 1e-9:
            ax.plot(p["energy_served_kwh_per_year"], p["battery_life_years"], "o", ms=13, mfc="none", mec=INK, mew=1.5)
            ax.annotate(f"chosen: cap {p['soc_cap_hot']:.0%}", (p["energy_served_kwh_per_year"], p["battery_life_years"]),
                        xytext=(10, 8), textcoords="offset points", fontsize=9.5)
    for p in (pts[0], pts[-1]):
        ax.annotate(f"{p['soc_cap_hot']:.0%}", (p["energy_served_kwh_per_year"], p["battery_life_years"]),
                    xytext=(-6, -14), textcoords="offset points", fontsize=8.5, color=INK2)
    if base["battery_life_years"]:
        ax.plot(base["energy_served_kwh_per_year"], base["battery_life_years"], "o", color=GREY, ms=9,
                label="Standard controller")
    ax.set_xlabel("Energy delivered to the home (kWh per year)")
    ax.set_ylabel("Battery life (years)")
    ax.set_title("The charge cap buys years of battery life for very little energy", pad=22)
    _subtitle(ax, "Each point is a full 10-year simulation with a different hot-day charge cap")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_tradeoff.png")
    plt.close(fig)


def fig_uncertainty(mc_df, mc):
    fig, ax = plt.subplots(figsize=(8, 4))
    g = mc_df["gain_years"]
    bins = np.linspace(min(0, g.min()), g.max() * 1.05, 18)
    ax.hist(g, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=2)
    q = mc["gain_years"]
    for v, ls in [(q["p5"], ":"), (q["median"], "-"), (q["p95"], ":")]:
        ax.axvline(v, color=INK, lw=1.2, ls=ls)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Extra battery life with SABIL (years)")
    ax.set_ylabel("Number of simulations")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"SABIL adds {q['median']:.1f} years (90% range {q['p5']:.1f}-{q['p95']:.1f}) "
                 f"in {mc['share_samples_smart_better']:.0%} of {mc['n_samples']} scenarios", pad=22)
    _subtitle(ax, "Monte Carlo over uncertain battery ageing, heat, household load and PV size; same inputs for both controllers")
    fig.tight_layout()
    fig.savefig(FIG / "fig7_uncertainty.png")
    plt.close(fig)

    corr = mc["spearman_with_gain_years"]
    names = {"battery.aging.cal_k_ref_per_sqrt_s": "Calendar ageing rate", "battery.aging.cal_ea_j_mol": "Heat sensitivity of ageing",
             "battery.aging.cal_soc_a": "Extra ageing when full", "battery.aging.cyc_c_b": "Cycle ageing rate",
             "battery.temp_offset_c": "Battery warmer than air", "household.load_scale": "Household load",
             "system.pv_kwp": "Solar array size"}
    items = sorted(corr.items(), key=lambda kv: abs(kv[1]))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(items))
    ax.barh(y, [v for _, v in items], color=[BLUE if v > 0 else GREY for _, v in items], height=0.6)
    for yi, (_, v) in zip(y, items):
        ax.text(v, yi, f" {v:+.2f} " if v >= 0 else f" {v:+.2f} ", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([names.get(k, k) for k, _ in items])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-1.15, 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Rank correlation with SABIL's battery-life gain (years)")
    ax.set_title("What the size of the gain depends on", pad=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig8_gain_drivers.png")
    plt.close(fig)


def fig_forecast(fm):
    m = fm["metrics"]["next_day"]
    seasons = [("all", "Whole year"), ("dry", "Dry season"), ("wet", "Rainy season")]
    kinds = [("persistence", "Tomorrow = today", GREY), ("climatology", "Seasonal average", "#c9c8c2"),
             ("learned", "Learned (gradient boosting)", BLUE)]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(seasons))
    wdt = 0.26
    for i, (k, label, color) in enumerate(kinds):
        vals = [m[s][k]["mae_kwh_m2_day"] for s, _ in seasons]
        ax.bar(x + (i - 1) * (wdt + 0.02), vals, wdt, color=color, label=label)
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 1) * (wdt + 0.02), v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s[1] for s in seasons])
    ax.set_ylabel("Forecast error, kWh/m2 per day (lower is better)")
    ax.grid(axis="x", visible=False)
    sk = m["wet"]["skill_vs_persistence"]
    ax.set_title(f"Learned sun forecast cuts rainy-season error by {sk:.0%}" if sk > 0
                 else "Learned sun forecast does not beat 'tomorrow = today'", pad=22)
    _subtitle(ax, "Out-of-sample: each year predicted by a model trained on the other 9 years (NASA POWER)")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig9_forecast_skill.png")
    plt.close(fig)


def make_all():
    _style()
    FIG.mkdir(parents=True, exist_ok=True)
    headline = load_all(verbose=False)["sim"]["headline_forecast"]
    other = "perfect" if headline == "persistence" else "persistence"
    base, smart, alt = _load("baseline"), _load("smart"), _load(f"smart_{other}")
    hb, hs = _hourly("baseline"), _hourly("smart")
    if base is None or smart is None:
        print("run simulate.py first")
        return
    fig_capacity(base, smart, alt, hb, hs, headline)
    fig_critical(base, smart, headline)
    fig_days(hb, hs, headline)
    fig_headline(base, smart, alt, headline)
    ex = RES / "experiments"
    if (ex / "ablation.json").exists():
        fig_ablation(json.load(open(ex / "ablation.json")))
    if (ex / "sweep.json").exists():
        fig_tradeoff(json.load(open(ex / "sweep.json")))
    if (ex / "montecarlo_summary.json").exists():
        fig_uncertainty(pd.read_csv(ex / "montecarlo.csv"), json.load(open(ex / "montecarlo_summary.json")))
    if (RES / "forecast_metrics.json").exists():
        fig_forecast(json.load(open(RES / "forecast_metrics.json")))
    print(f"charts saved in {FIG}")


if __name__ == "__main__":
    make_all()
