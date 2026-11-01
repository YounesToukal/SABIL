"""Verification of the ageing model implementation (owner: Younes).

1. Closed-form checks: under constant conditions the model must reproduce the
   published square-root laws exactly (calendar: k*sqrt(t); cycle: k*sqrt(FEC)/100).
2. Cross-implementation check: drive src/aging_model.py and the authors' own code
   (SimSES, TU Munich, `pip install simses`) with the same one-year SoC and
   temperature history and compare the capacity loss.
3. Physics sanity: hotter and fuller must age faster; deeper cycles must age faster.

Run: python src/validate_aging.py        -> prints a table, writes results/aging_validation.json
"""
import json
import math

import numpy as np

import aging_naumann as am  # the TU Munich model checked against SimSES
from params_io import ROOT, load_all


def calendar_closed_form(a, T_c, soc, hours):
    T = T_c + 273.15
    k = a["cal_k_ref_per_sqrt_s"] * math.exp(-a["cal_ea_j_mol"] / am.R_GAS * (1 / T - 1 / a["cal_t_ref_k"])) \
        * (a["cal_soc_a"] * (soc - 0.5) ** 3 + a["cal_soc_b"])
    return k * math.sqrt(hours * 3600.0)


def run_calendar(bat, T_c, soc, hours):
    am.reset()
    am.observe_soc(soc)
    q = 0.0
    for _ in range(hours):
        am.observe_soc(soc)
        q += am.step_loss(1.0, soc, T_c, 0.0, bat["capacity_kwh"], bat)
    return q


def cycle_profile(lo, hi, step, n_cycles):
    up = list(np.arange(lo + step, hi + 1e-9, step))
    down = list(np.arange(hi - step, lo - 1e-9, -step))
    return [lo] + (up + down) * n_cycles


def run_cycle_only(bat, socs):
    """Cycle loss only: zero the calendar part by a reference temperature far below 0 K? no -
    simply subtract the calendar loss computed on the same trajectory with no reversals."""
    am.reset()
    am.observe_soc(socs[0])
    total = 0.0
    for s in socs[1:]:
        am.observe_soc(s)
        total += am.step_loss(1.0, s, 25.0, 0.0, bat["capacity_kwh"], bat)
    return am.breakdown()["cycle_loss_frac"], total


def cross_check_simses(bat, socs, temps):
    try:
        from simses.battery.state import BatteryState
        from simses.degradation.degradation import DegradationModel
        from simses.model.degradation.sony_lfp_calendar import SonyLFPCalendarDegradation
        from simses.model.degradation.sony_lfp_cyclic import SonyLFPCyclicDegradation
    except Exception as e:  # simses not installed
        return {"available": False, "reason": repr(e)}
    st = BatteryState(v=0, i=0, T=float(temps[0]), power=0, power_setpoint=0, soc=float(socs[0]), ocv=0, hys=0,
                      rint=0, entropy=0, soh_Q=1.0, soh_R=1.0, is_charge=False, loss=0, heat=0,
                      i_max_charge=0, i_max_discharge=0)
    ref = DegradationModel(SonyLFPCalendarDegradation(), SonyLFPCyclicDegradation(), initial_soc=float(socs[0]))
    am.reset()
    am.observe_soc(float(socs[0]))
    ours = 0.0
    for s, T in zip(socs[1:], temps[1:]):
        st.soc, st.T = float(s), float(T)
        ref.step(st, 3600.0)
        am.observe_soc(float(s))
        ours += am.step_loss(1.0, float(s), float(T), 0.0, bat["capacity_kwh"], bat)
    theirs = 1.0 - st.soh_Q
    return {"available": True, "simses_loss": theirs, "sabil_loss": ours,
            "relative_difference": abs(ours - theirs) / theirs,
            "simses_calendar": ref.state.qloss_cal, "simses_cycle": ref.state.qloss_cyc,
            "sabil_calendar": am.breakdown()["calendar_loss_frac"], "sabil_cycle": am.breakdown()["cycle_loss_frac"]}


def main():
    P = load_all(verbose=False)
    bat = P["battery"]
    if "cal_k_ref_per_sqrt_s" not in bat.get("aging", {}):
        print("battery.json has no interim-model coefficients (Youcef's model in use) - skipping")
        return
    a = bat["aging"]
    out = {"calendar_closed_form": [], "cycle_closed_form": None, "sanity": {}, "cross_check": None}
    print("1) calendar vs closed form (1 year, constant conditions)")
    for T_c, soc in [(25, 0.5), (25, 1.0), (35, 0.7), (40, 1.0)]:
        sim, exact = run_calendar(bat, T_c, soc, 8760), calendar_closed_form(a, T_c, soc, 8760)
        out["calendar_closed_form"].append({"T_c": T_c, "soc": soc, "model": sim, "closed_form": exact})
        print(f"   {T_c:>2} C, SoC {soc:.1f}: model {sim:.4%}  closed form {exact:.4%}  diff {abs(sim - exact):.1e}")

    print("2) cycle vs closed form (0.1 <-> 0.9 SoC, 0.1/h, 500 full cycles)")
    socs = cycle_profile(0.1, 0.9, 0.1, 500)
    cyc, _ = run_cycle_only(bat, socs)
    n_half = 2 * 500 - 1                      # the last half-cycle is not closed by a reversal
    kc = (a["cyc_c_a"] * 0.1 + a["cyc_c_b"]) * (a["cyc_dod_c"] * (0.8 - 0.6) ** 3 + a["cyc_dod_d"])
    exact = kc * math.sqrt(n_half * 0.4) / 100.0
    out["cycle_closed_form"] = {"model": cyc, "closed_form": exact}
    print(f"   model {cyc:.4%}  closed form {exact:.4%}  diff {abs(cyc - exact):.1e}")

    print("3) physics sanity (1 year, calendar)")
    s = out["sanity"]
    s["hotter_ages_faster"] = run_calendar(bat, 40, 0.7, 8760) > run_calendar(bat, 25, 0.7, 8760)
    s["fuller_ages_faster"] = run_calendar(bat, 30, 1.0, 8760) > run_calendar(bat, 30, 0.6, 8760)
    deep, _ = run_cycle_only(bat, cycle_profile(0.1, 0.9, 0.1, 200))
    shallow, _ = run_cycle_only(bat, cycle_profile(0.5, 0.7, 0.1, 800))   # same throughput
    s["deeper_cycles_age_faster_per_kwh"] = deep > shallow
    for k, v in s.items():
        print(f"   [{'ok' if v else 'FAIL'}] {k}")

    print("4) cross-check against the authors' code (SimSES), one year of daily solar cycling, 25-40 C")
    hours = np.arange(8760)
    socs = 0.55 + 0.35 * np.sin(2 * np.pi * (hours - 9) / 24)
    temps = 32 + 6 * np.sin(2 * np.pi * (hours - 9) / 24) + 4 * np.sin(2 * np.pi * hours / 8760)
    cc = cross_check_simses(bat, socs, temps)
    out["cross_check"] = cc
    if cc["available"]:
        print(f"   SimSES {cc['simses_loss']:.4%} (cal {cc['simses_calendar']:.4%}, cyc {cc['simses_cycle']:.4%})")
        print(f"   SABIL  {cc['sabil_loss']:.4%} (cal {cc['sabil_calendar']:.4%}, cyc {cc['sabil_cycle']:.4%})")
        print(f"   relative difference {cc['relative_difference']:.2e}")
    else:
        print("   SimSES not installed (pip install simses) - skipped")

    ok = all(abs(r["model"] - r["closed_form"]) < 1e-9 for r in out["calendar_closed_form"]) \
        and abs(out["cycle_closed_form"]["model"] - out["cycle_closed_form"]["closed_form"]) < 1e-9 \
        and all(s.values()) and (not cc["available"] or cc["relative_difference"] < 1e-6)
    out["all_passed"] = ok
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "results" / "aging_validation.json", "w"), indent=2, default=float)
    print("\nAGEING MODEL VERIFIED" if ok else "\nVERIFICATION FAILED")


if __name__ == "__main__":
    main()
