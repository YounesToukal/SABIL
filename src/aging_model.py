# src/aging_model.py                    (owner: Youcef)
#
# INTERIM MODEL written by Younes so results exist before Youcef's file arrives.
# Same two functions and arguments as the integration contract, plus two optional
# hooks that simulate.py calls only if they exist: reset() at the start of a run and
# observe_soc(soc_end) once per step (the true end-of-step SoC, for cycle counting).
# Youcef: review it, or replace this whole file with yours. Nothing else needs to change.
#
# Semi-empirical LFP/graphite model (Sony/Murata US26650FTC1), superposition of:
#   calendar:  q_cal = k_T(T) * k_SoC(SoC) * t^0.5               t in seconds, q as a fraction
#              k_T   = k_ref * exp(-Ea/R * (1/T - 1/T_ref)),  k_SoC = a (SoC - 0.5)^3 + b
#              Naumann et al., J. Energy Storage 17 (2018) 153-169, doi:10.1016/j.est.2018.01.019
#   cycle:     q_cyc = k_C(C) * k_DOD(DOD) * FEC^0.5 / 100        per half-cycle
#              k_C = a C + b,  k_DOD = c (DOD - 0.6)^3 + d,  FEC of a half-cycle = DOD / 2
#              Naumann et al., J. Power Sources 451 (2020) 227666
# Implementation mirrors the authors' open-source SimSES (TU Munich, PyPI 'simses' 2.1.1,
# simses/model/degradation/sony_lfp_calendar.py and sony_lfp_cyclic.py,
# simses/degradation/cycle_detector.py): identical coefficients, "virtual time" / "virtual
# FEC" continuation under changing stress, and half-cycles closed at each SoC reversal.
# Verified by src/validate_aging.py. Coefficients live in battery.json -> "aging".
import math

R_GAS = 8.3144598
_s = {}


def reset():
    """Fresh battery: no accumulated loss, empty half-cycle detector."""
    _s.clear()
    _s.update(q_cal=0.0, q_cyc=0.0, fec=0.0, prev=None, start=None, direction=0,
              hours=0.0, pending=[])


reset()


def observe_soc(soc_end, dt_h=1.0):
    """Half-cycle detector (SimSES HalfCycleDetector): closes a half-cycle at each reversal."""
    s = _s
    if s["prev"] is None:
        s["prev"] = s["start"] = soc_end
        return
    delta = soc_end - s["prev"]
    if abs(delta) < 1e-12:
        return                                   # rest: no cycle, no elapsed time
    d = 1 if delta > 0 else -1
    if s["direction"] == 0 or d == s["direction"]:
        s["direction"] = d
        s["hours"] += dt_h
        s["prev"] = soc_end
        return
    dod = abs(s["prev"] - s["start"])            # reversal: close the half-cycle
    c_rate = dod / s["hours"] if s["hours"] > 0 else 0.0
    s["pending"].append((dod, c_rate))
    s["start"], s["direction"], s["hours"], s["prev"] = s["prev"], d, dt_h, soc_end


def battery_temp(ambient_c, power_kw, battery):
    """Battery temperature during one hour, in °C: indoor air + a fixed offset."""
    return ambient_c + battery["temp_offset_c"]


def step_loss(dt_h, soc, temp_c, throughput_kwh, capacity_kwh, battery):
    """Capacity lost during one step, as a fraction of the INITIAL capacity (>= 0)."""
    a, s = battery["aging"], _s
    # calendar, every step
    T, T_ref = temp_c + 273.15, a["cal_t_ref_k"]
    k = a["cal_k_ref_per_sqrt_s"] * math.exp(-a["cal_ea_j_mol"] / R_GAS * (1.0 / T - 1.0 / T_ref)) \
        * (a["cal_soc_a"] * (soc - 0.5) ** 3 + a["cal_soc_b"])
    d_cal = 0.0
    if k > 0:
        t_virtual = (s["q_cal"] / k) ** 2
        d_cal = k * math.sqrt(t_virtual + dt_h * 3600.0) - s["q_cal"]
        s["q_cal"] += d_cal
    # cycle, for every half-cycle closed since the last step
    d_cyc = 0.0
    for dod, c_rate in s["pending"]:
        dfec = dod / 2.0
        kc = (a["cyc_c_a"] * c_rate + a["cyc_c_b"]) * (a["cyc_dod_c"] * (dod - 0.6) ** 3 + a["cyc_dod_d"])
        if dfec > 0 and kc > 0:
            fec_virtual = (s["q_cyc"] * 100.0 / kc) ** 2
            dq = kc * math.sqrt(fec_virtual + dfec) / 100.0 - s["q_cyc"]
            s["q_cyc"] += dq
            d_cyc += dq
            s["fec"] += dfec
    s["pending"].clear()
    return max(0.0, d_cal) + max(0.0, d_cyc)


def breakdown():
    """Accumulated calendar and cycle losses (fractions of initial capacity) and FEC."""
    return {"calendar_loss_frac": _s["q_cal"], "cycle_loss_frac": _s["q_cyc"], "fec": _s["fec"]}
