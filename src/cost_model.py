"""SABIL - cost model (owner: Youcef).

What does one kWh from a SABIL kit cost, what tariff covers it, what does a family pay a month,
how does that compare with a petrol generator, and what is the controller worth in money?

Method: equivalent annual cost. Each part of the kit is turned into a yearly payment with the
capital recovery factor CRF(r, n) = r / (1 - (1 + r)^-n), where n is the part's life. The
battery's life is not assumed: it is the simulated time to 80% capacity (results/summary_*.json),
so the controller's effect on battery life flows straight into the cost of energy.

    cost per kWh  = (sum of yearly payments + running costs) / kWh delivered per year
    tariff        = cost per kWh / (1 - collection fees - bad debt)

Same house, same weather, same kit: only the controller differs between "standard" and "SABIL".

Reads  params/costs.json, params/system.json, params/battery.json, params/household.json,
       results/summary_baseline.json, results/summary_smart.json, results/aging_robustness.json
Writes results/costs_summary.json, (figures: src/exhibits.py)
Run from the repo root:  python src/cost_model.py
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from params_io import ROOT, load_all  # noqa: E402

RES = ROOT / "results"


def crf(r, n):
    """Capital recovery factor: the yearly payment that repays 1 over n years at rate r."""
    return 1.0 / n if r == 0 else r / (1.0 - (1.0 + r) ** -n)


def kit_items(C, P, controller):
    """Up-front cost of each part of the kit (USD) and the lifetime key that applies to it."""
    k = C["kit"]
    pv_wp = P["system"]["pv_kwp"] * 1000.0
    kwh = P["battery"]["capacity_kwh"]
    extra = k["sabil_controller_extra_usd"] if controller == "smart" else 0.0
    return {
        "pv_and_mounting": (pv_wp * k["pv_usd_per_wp"] + k["pv_mounting_usd"], "pv_and_mounting"),
        "battery": (kwh * k["battery_usd_per_kwh"], "battery"),
        "electronics": (k["charge_controller_usd"] + k["meter_comms_usd"] + extra, "electronics"),
        "wiring_install_acquisition": (k["wiring_protection_usd"] + k["install_logistics_usd"]
                                       + k["customer_acquisition_usd"], "wiring_install_acquisition"),
    }


def levelised(C, P, controller, battery_life_y, kwh_per_year, grant_usd=0.0):
    """Yearly cost, cost per kWh and break-even tariff of one kit."""
    r = C["finance"]["real_discount_rate"]
    life = C["lifetimes_years"]
    items = kit_items(C, P, controller)
    capex = sum(c for c, _ in items.values())
    yearly = {name: cost * crf(r, battery_life_y if name == "battery" else life[key])
              for name, (cost, key) in items.items()}
    o = C["opex"]
    yearly["running"] = (o["maintenance_usd_per_year"] + o["data_platform_usd_per_year"]
                         + o["insurance_theft_frac_of_capex_per_year"] * capex)
    if grant_usd:
        yearly["grant"] = -grant_usd * crf(r, C["finance"]["horizon_years"])
    total = sum(yearly.values())
    per_kwh = total / kwh_per_year
    rl = C["revenue_losses"]
    tariff = per_kwh / (1.0 - rl["collection_fee_frac"] - rl["bad_debt_frac"])
    return {"capex_usd": capex, "capex_items_usd": {n: c for n, (c, _) in items.items()},
            "yearly_cost_usd": total, "yearly_items_usd": yearly,
            "cost_usd_per_kwh": per_kwh, "tariff_usd_per_kwh": tariff}


def critical_kwh_per_day(household):
    return sum(a["power_w"] * a["quantity"] * len(a["use_hours"])
               for a in household["appliances"] if a.get("critical")) / 1000.0


def basic_kwh_per_day(household):
    """Lights, phone charging and fans only: the essential loads without the fridge."""
    return sum(a["power_w"] * a["quantity"] * len(a["use_hours"]) for a in household["appliances"]
               if a.get("critical") and "refrigerator" not in a["name"].lower()) / 1000.0


def total_kwh_per_day(household):
    return sum(a["power_w"] * a["quantity"] * len(a["use_hours"]) for a in household["appliances"]) / 1000.0


def analyse(C, P, sums, rob):
    ngn = C["currency"]["ngn_per_usd"]
    life = {c: sums[c]["battery_life_years"] for c in ("baseline", "smart")}
    kwh = {c: sums[c]["energy_served_kwh_per_year"] for c in ("baseline", "smart")}
    std = levelised(C, P, "baseline", life["baseline"], kwh["baseline"])
    sab = levelised(C, P, "smart", life["smart"], kwh["smart"])
    sab_grant = levelised(C, P, "smart", life["smart"], kwh["smart"], C["subsidy"]["dares_grant_usd_per_system"])

    # --- the family's bill ---
    months = 12.0
    bill = lambda t, k: t * k / months * ngn  # noqa: E731
    crit_day, all_day = critical_kwh_per_day(P["household"]), total_kwh_per_day(P["household"])
    tariff_ngn = sab["tariff_usd_per_kwh"] * ngn
    inc = C["income"]
    family = {
        "kwh_per_month": kwh["smart"] / months,
        "bill_ngn_per_month": bill(sab["tariff_usd_per_kwh"], kwh["smart"]),
        "bill_ngn_per_month_with_dares_grant": bill(sab_grant["tariff_usd_per_kwh"], kwh["smart"]),
        "essentials_only_kwh_per_month": crit_day * 365 / months,
        "essentials_only_bill_ngn_per_month": tariff_ngn * crit_day * 365 / months,
        "lights_phones_fans_kwh_per_month": basic_kwh_per_day(P["household"]) * 365 / months,
        "lights_phones_fans_bill_ngn_per_month": tariff_ngn * basic_kwh_per_day(P["household"]) * 365 / months,
        "share_of_minimum_wage": bill(sab["tariff_usd_per_kwh"], kwh["smart"]) / inc["minimum_wage_ngn_per_month"],
        "essentials_share_of_minimum_wage": tariff_ngn * crit_day * 365 / months / inc["minimum_wage_ngn_per_month"],
        "note": "Tariff set so that a home using the designed load (~%.2f kWh/day) pays for its kit. "
                "Essentials only = lights, security light, phone charging, night fan and fridge." % all_day,
    }

    # --- the same energy from a petrol generator (fuel only: no generator purchase or repairs) ---
    f = C["fuel"]
    gen = {}
    for label, lpk in (("best_case", C["generator_l_per_kwh"]), ("typical_small_set", f["small_generator_l_per_kwh"])):
        per_kwh_ngn = f["petrol_ngn_per_l_borno"] * lpk
        gen[label] = {"l_per_kwh": lpk, "fuel_ngn_per_kwh": per_kwh_ngn,
                      "fuel_ngn_per_month_same_energy": per_kwh_ngn * kwh["smart"] / months,
                      "litres_avoided_per_year": lpk * kwh["smart"],
                      "sabil_cheaper_by_frac": 1.0 - tariff_ngn / per_kwh_ngn}

    # --- what the controller is worth ---
    r, H = C["finance"]["real_discount_rate"], C["finance"]["horizon_years"]
    save = std["yearly_cost_usd"] - sab["yearly_cost_usd"]
    bat_save = std["yearly_items_usd"]["battery"] - sab["yearly_items_usd"]["battery"]
    extra = C["kit"]["sabil_controller_extra_usd"]
    module_kg = (P["sim"].get("experiments", {}).get("placement_study", {}).get("thermal", {}).get("mass_kg", 18.0))
    batt_std, batt_sab = H / life["baseline"], H / life["smart"]
    controller = {
        "extra_hardware_usd": extra,
        "yearly_saving_usd_per_home": save,
        "battery_yearly_saving_usd_per_home": bat_save,
        "payback_of_extra_hardware_years": extra / bat_save if bat_save > 0 else None,
        "npv_usd_per_home": save / crf(r, H),
        "npv_usd_per_1000_homes": 1000 * save / crf(r, H),
        "battery_modules_per_home_over_horizon": {"standard": batt_std, "sabil": batt_sab},
        "battery_modules_avoided_per_1000_homes": 1000 * (batt_std - batt_sab),
        "battery_waste_avoided_t_per_1000_homes": 1000 * (batt_std - batt_sab) * module_kg / 1000.0,
        "cost_per_kwh_reduction_frac": 1.0 - sab["cost_usd_per_kwh"] / std["cost_usd_per_kwh"],
    }

    # --- sensitivity of the SABIL tariff (NGN/kWh) ---
    S = C["sensitivity"]
    base_t = tariff_ngn

    def tariff_with(mod, life_s=None, kwh_scale=1.0):
        C2 = copy.deepcopy(C)
        mod(C2)
        return levelised(C2, P, "smart", life_s or life["smart"], kwh["smart"] * kwh_scale)["tariff_usd_per_kwh"] * ngn

    def setk(section, key, v):
        return lambda c: c[section].__setitem__(key, v)

    def scale_other(s):
        def f2(c):
            for key in ("pv_mounting_usd", "charge_controller_usd", "meter_comms_usd", "wiring_protection_usd",
                        "install_logistics_usd", "customer_acquisition_usd"):
                c["kit"][key] *= s
        return f2

    sens = [
        ("Battery price ($/kWh)", S["battery_usd_per_kwh"], [tariff_with(setk("kit", "battery_usd_per_kwh", v)) for v in S["battery_usd_per_kwh"]]),
        ("Cost of capital (real)", S["real_discount_rate"], [tariff_with(setk("finance", "real_discount_rate", v)) for v in S["real_discount_rate"]]),
        ("Unpaid bills", S["bad_debt_frac"], [tariff_with(setk("revenue_losses", "bad_debt_frac", v)) for v in S["bad_debt_frac"]]),
        ("Other kit and install costs (x)", S["capex_other_scale"], [tariff_with(scale_other(v)) for v in S["capex_other_scale"]]),
        ("PV price ($/Wp)", S["pv_usd_per_wp"], [tariff_with(setk("kit", "pv_usd_per_wp", v)) for v in S["pv_usd_per_wp"]]),
        ("Energy used (x design load)", S["energy_scale"], [tariff_with(lambda c: None, kwh_scale=v) for v in S["energy_scale"]]),
    ]
    nrel = rob.get("nrel_prismatic, cells as tested", {}).get("smart", {}).get("battery_life_years")
    if nrel:
        sens.append(("Battery ageing model", ["TU Munich (base)", "NREL prismatic"],
                     [base_t, tariff_with(lambda c: None, life_s=nrel)]))
    sensitivity = [{"input": n, "values": v, "tariff_ngn_per_kwh": t} for n, v, t in sens]

    return {
        "kit": {"pv_wp": P["system"]["pv_kwp"] * 1000, "battery_kwh": P["battery"]["capacity_kwh"]},
        "inputs_used": {"ngn_per_usd": ngn, "battery_life_years": life, "kwh_delivered_per_year": kwh,
                        "real_discount_rate": r, "horizon_years": H},
        "standard": std, "sabil": sab, "sabil_with_dares_grant": sab_grant,
        "tariff_ngn_per_kwh": {"standard": std["tariff_usd_per_kwh"] * ngn, "sabil": base_t,
                               "sabil_with_dares_grant": sab_grant["tariff_usd_per_kwh"] * ngn},
        "family": family, "petrol_generator": gen, "controller_value": controller,
        "sensitivity_sabil_tariff": sensitivity,
        "sources": "params/costs.json",
    }


def plot(res, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plots
    plots._style()
    t = res["tariff_ngn_per_kwh"]
    g = res["petrol_generator"]
    rows = [
        ("SABIL kit, break-even tariff", t["sabil"], plots.BLUE),
        ("SABIL kit with DARES grant", t["sabil_with_dares_grant"], plots.BLUE),
        ("Standard kit, break-even tariff", t["standard"], plots.GREY),
        ("Petrol generator, fuel only, best case (%.1f L/kWh)" % g["best_case"]["l_per_kwh"], g["best_case"]["fuel_ngn_per_kwh"], plots.ORANGE),
        ("Petrol generator, fuel only, small set (%.0f L/kWh)" % g["typical_small_set"]["l_per_kwh"], g["typical_small_set"]["fuel_ngn_per_kwh"], plots.ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 3.6), layout="constrained")
    y = list(range(len(rows)))[::-1]
    for yi, (lab, v, c) in zip(y, rows):
        ax.barh(yi, v, height=0.62, color=c)
        ax.text(v + 40, yi, f"N{v:,.0f}", va="center", fontsize=9.5, color=plots.INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9.5)
    ax.set_xlabel("Naira per kWh delivered")
    ax.set_xlim(0, max(r[1] for r in rows) * 1.15)
    ax.grid(axis="y", visible=False)
    plots._subtitle(ax, "Cost of one kWh in Maiduguri (petrol N%s/L, Borno, Oct 2026; N%s/USD)"
                    % (f"{int(res_fuel_price(res)):,}", f"{res['inputs_used']['ngn_per_usd']:,}"))
    fig.savefig(outdir / "fig15_cost_per_kwh.png", dpi=200)
    plt.close(fig)

    # tornado
    sens = res["sensitivity_sabil_tariff"]
    base = t["sabil"]
    sens = sorted(sens, key=lambda s: abs(s["tariff_ngn_per_kwh"][1] - s["tariff_ngn_per_kwh"][0]))
    fig, ax = plt.subplots(figsize=(9.2, 3.8), layout="constrained")
    for i, s in enumerate(sens):
        lo, hi = s["tariff_ngn_per_kwh"]
        a, b = min(lo, hi), max(lo, hi)
        ax.barh(i, b - a, left=a, height=0.55, color=plots.BLUE)
        fmt = lambda v: (f"{v:g}" if not isinstance(v, str) else v)  # noqa: E731
        ax.text(a - 8, i, f"{fmt(s['values'][0] if lo <= hi else s['values'][1])}  N{a:,.0f}", ha="right",
                va="center", fontsize=8.5, color=plots.INK2)
        ax.text(b + 8, i, f"N{b:,.0f}  {fmt(s['values'][1] if lo <= hi else s['values'][0])}", ha="left",
                va="center", fontsize=8.5, color=plots.INK2)
    ax.axvline(base, color=plots.INK, lw=1)
    ax.set_yticks(range(len(sens)), [s["input"] for s in sens], fontsize=9.5)
    lo_all = min(min(s["tariff_ngn_per_kwh"]) for s in sens)
    hi_all = max(max(s["tariff_ngn_per_kwh"]) for s in sens)
    ax.set_xlim(lo_all - 160, hi_all + 160)
    ax.set_xlabel("SABIL break-even tariff, naira per kWh")
    ax.grid(axis="y", visible=False)
    plots._subtitle(ax, f"What moves the tariff (base N{base:,.0f}/kWh, vertical line)")
    fig.savefig(outdir / "fig16_cost_sensitivity.png", dpi=200)
    plt.close(fig)


def res_fuel_price(res):
    return res["_fuel_price"]


def main():
    P = load_all(verbose=False)
    C = P["costs"]
    sums = {c: json.loads((RES / f"summary_{c}.json").read_text()) for c in ("baseline", "smart")}
    rp = RES / "aging_robustness.json"
    rob = json.loads(rp.read_text()) if rp.exists() else {}
    res = analyse(C, P, sums, rob)
    res["_fuel_price"] = C["fuel"]["petrol_ngn_per_l_borno"]
    if "--old-plots" in sys.argv:   # figures are drawn by src/exhibits.py
        plot(res, RES / "figures")
    res.pop("_fuel_price")

    def rnd(o):
        if isinstance(o, float):
            return round(o, 4)
        if isinstance(o, dict):
            return {k: rnd(v) for k, v in o.items()}
        if isinstance(o, list):
            return [rnd(v) for v in o]
        return o
    (RES / "costs_summary.json").write_text(json.dumps(rnd(res), indent=2) + "\n")

    t, f, g, c = res["tariff_ngn_per_kwh"], res["family"], res["petrol_generator"], res["controller_value"]
    ngn = C["currency"]["ngn_per_usd"]
    print(f"Kit: {res['kit']['pv_wp']:.0f} Wp, {res['kit']['battery_kwh']} kWh | up-front cost: standard "
          f"${res['standard']['capex_usd']:.0f}, SABIL ${res['sabil']['capex_usd']:.0f}")
    print(f"Cost per kWh: standard ${res['standard']['cost_usd_per_kwh']:.3f}, SABIL ${res['sabil']['cost_usd_per_kwh']:.3f} "
          f"({c['cost_per_kwh_reduction_frac']:.1%} lower)")
    print(f"Break-even tariff: standard N{t['standard']:,.0f}/kWh, SABIL N{t['sabil']:,.0f}/kWh, "
          f"SABIL with DARES grant N{t['sabil_with_dares_grant']:,.0f}/kWh")
    print(f"Family bill: N{f['bill_ngn_per_month']:,.0f}/month for {f['kwh_per_month']:.1f} kWh "
          f"({f['share_of_minimum_wage']:.0%} of the minimum wage); essentials incl. fridge N{f['essentials_only_bill_ngn_per_month']:,.0f}/month; "
          f"lights, phones and fans only N{f['lights_phones_fans_bill_ngn_per_month']:,.0f}/month")
    for k, v in g.items():
        print(f"Petrol generator ({k}, {v['l_per_kwh']} L/kWh): fuel N{v['fuel_ngn_per_kwh']:,.0f}/kWh, "
              f"N{v['fuel_ngn_per_month_same_energy']:,.0f}/month for the same energy; SABIL is {v['sabil_cheaper_by_frac']:.0%} cheaper; "
              f"{v['litres_avoided_per_year']:.0f} L/year avoided")
    print(f"Controller: +${c['extra_hardware_usd']} hardware, saves ${c['yearly_saving_usd_per_home']:.2f}/home/year "
          f"(payback {c['payback_of_extra_hardware_years']:.1f} y), NPV ${c['npv_usd_per_1000_homes']:,.0f} per 1,000 homes, "
          f"{c['battery_modules_avoided_per_1000_homes']:.0f} battery modules ({c['battery_waste_avoided_t_per_1000_homes']:.1f} t) "
          f"avoided per 1,000 homes over {C['finance']['horizon_years']} years")
    for s in res["sensitivity_sabil_tariff"]:
        print(f"  {s['input']:<34} {s['values']} -> N{s['tariff_ngn_per_kwh'][0]:,.0f} / N{s['tariff_ngn_per_kwh'][1]:,.0f}")
    print(f"(N{ngn:,}/USD)")


if __name__ == "__main__":
    main()
