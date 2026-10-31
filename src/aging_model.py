# src/aging_model.py                    (owner: Youcef)
#
# INTERIM MODEL written by Younes so results exist before Youcef's file arrives.
# Same two functions and arguments as the integration contract, plus an optional
# reset() that simulate.py calls at the start of each run.
# Youcef: review it, or replace this whole file with yours. Nothing else needs to change.
#
# Semi-empirical LFP/graphite model, superposition of:
#   calendar:  q_cal = k_T(T) * k_SoC(SoC) * t^0.5          (t in seconds, q as a fraction)
#              k_T   = k_ref * exp(-Ea/R * (1/T - 1/T_ref))
#              k_SoC = a * (SoC - 0.5)^3 + b
#     after Naumann et al., J. Energy Storage 17 (2018) 153-169, doi:10.1016/j.est.2018.01.019
#   cycle:     q_cyc = k_C(C) * k_DOD(DOD) * FEC^0.5          (q in percent)
#              k_C   = a * C + b ;  k_DOD = c * (DOD - 0.6)^3 + d
#     after Naumann et al., J. Power Sources 451 (2020) 227666, doi:10.1016/j.jpowsour.2019.227666
# Both laws are square-root laws; under changing conditions each is advanced with the
# "virtual time" method (the time that would have produced today's loss under today's stress).
# DOD is estimated as the SoC swing over the last 24 hours (daily solar cycling).
# All coefficients live in battery.json -> "aging" (TO CONFIRM against the papers).
import math
from collections import deque

R_GAS = 8.314
_state = {}


def reset():
    _state.clear()
    _state.update(q_cal=0.0, q_cyc_pct=0.0, fec=0.0, soc_hist=deque(maxlen=24))


reset()


def battery_temp(ambient_c, power_kw, battery):
    """Battery temperature during one hour, in °C: indoor air + a fixed offset."""
    return ambient_c + battery["temp_offset_c"]


def step_loss(dt_h, soc, temp_c, throughput_kwh, capacity_kwh, battery):
    """Capacity lost during one step, as a fraction of the INITIAL capacity (>= 0)."""
    a = battery["aging"]
    s = _state

    # ---- calendar ----
    T = temp_c + 273.15
    k_T = a["cal_k_ref_per_sqrt_s"] * math.exp(-a["cal_ea_j_mol"] / R_GAS * (1.0 / T - 1.0 / a["cal_t_ref_k"]))
    k_soc = a["cal_soc_a"] * (soc - 0.5) ** 3 + a["cal_soc_b"]
    k_cal = max(1e-15, k_T * k_soc)
    t_eq = (s["q_cal"] / k_cal) ** 2
    q_cal_new = k_cal * math.sqrt(t_eq + dt_h * 3600.0)
    d_cal = q_cal_new - s["q_cal"]
    s["q_cal"] = q_cal_new

    # ---- cycle ----
    s["soc_hist"].append(soc)
    d_cyc = 0.0
    if throughput_kwh > 0:
        cap0 = battery["capacity_kwh"]
        dfec = throughput_kwh / (2.0 * cap0)                  # charge + discharge = 2 x one full cycle
        c_rate = throughput_kwh / dt_h / cap0
        dod = max(a["cyc_dod_min"], max(s["soc_hist"]) - min(s["soc_hist"]))
        k_c = a["cyc_c_a"] * c_rate + a["cyc_c_b"]
        k_dod = a["cyc_dod_c"] * (dod - 0.6) ** 3 + a["cyc_dod_d"]
        k_cyc = max(1e-12, k_c * k_dod)
        fec_eq = (s["q_cyc_pct"] / k_cyc) ** 2
        q_new = k_cyc * math.sqrt(fec_eq + dfec)
        d_cyc = (q_new - s["q_cyc_pct"]) / 100.0
        s["q_cyc_pct"] = q_new
        s["fec"] += dfec

    return max(0.0, d_cal) + max(0.0, d_cyc)
