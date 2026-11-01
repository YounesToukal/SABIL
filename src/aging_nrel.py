"""SABIL - NREL large-format prismatic LFP ageing model (owner: Youcef).

Selected through src/aging_model.py (battery.json aging.model = "nrel_prismatic").

Semi-empirical capacity-fade model, two mechanisms that add up:

  calendar fade  q_cal = k_cal(T, Ua(SoC)) * t_days ** p_cal
                 k_cal = p1 * exp(p2 / T) * exp(p3 * Ua / T)
                 -> faster when hot, and when the SoC is high (low graphite potential Ua)

  cycle fade     q_cyc = k_cyc(T, DoD, C) * EFC ** p_cyc
                 k_cyc = (p4 + p5*DoD + p6*C) * (exp(p7 / T) + exp(-p8 / T))
                 -> grows with charge throughput (equivalent full cycles), depth of
                    discharge and C-rate

  capacity / initial capacity = 1 - q_cal - q_cyc

Equations and parameters: NREL BLAST-Lite model "Lfp_Gr_250AhPrismatic" (large-format
prismatic LFP cells, the cell type used in stationary storage modules), fitted on the
ageing data reported in J. Energy Storage, doi:10.1016/j.est.2023.109042.
Graphite potential Ua(SoC): Safari & Delacourt, J. Electrochem. Soc. 158(5) A562 (2011),
doi:10.1149/1.3567007, as used in BLAST-Lite.

Time integration ("virtual time"): power laws like t**0.526 are not additive step by step
when conditions change. Each step we find the equivalent time (or EFC) that would give the
damage already accumulated under the CURRENT conditions, then advance it by the step:
    x_eq = (q / k) ** (1 / p);   dq = k * ((x_eq + dx) ** p - x_eq ** p)

Validity: calendar and cycling tests 10-45 C, DoD 0.8-1, charge <= 0.65 C. Our home use
(0.1 C, 30-45 C) is inside the temperature range; shallow-DoD cycling is an extrapolation.

Interface fixed by the team integration contract:
    battery_temp(ambient_c, power_kw, battery) -> degC
    step_loss(dt_h, soc, temp_c, throughput_kwh, capacity_kwh, battery) -> fraction of
        INITIAL capacity lost in this step (>= 0)
The model keeps its running state inside the `battery` dict (key "_aging_state").
Start each simulation run with reset() (no argument; used by simulate.py) or
reset_state(battery). battery_temp() is stateless, so controllers can call it for forecasts.
"""
import math
from collections import deque

_HISTORY_H = 24  # window used to measure DoD and average C-rate (one daily cycle)
_RUN = 0         # run counter: reset() starts a new run, any battery dict then starts fresh
_LAST = None     # battery dict of the current run, for breakdown()


# --------------------------------------------------------------------------- physics
def graphite_potential(soc):
    """Graphite anode potential vs Li/Li+ (V) from SoC (Safari & Delacourt 2011)."""
    xa = 8.5e-3 + soc * (0.78 - 8.5e-3)  # lithiation fraction of the anode
    return (0.6379
            + 0.5416 * math.exp(-305.5309 * xa)
            + 0.044 * math.tanh(-(xa - 0.1958) / 0.1088)
            - 0.1978 * math.tanh((xa - 1.0571) / 0.0854)
            - 0.6875 * math.tanh((xa + 0.0117) / 0.0529)
            - 0.0175 * math.tanh((xa - 0.5692) / 0.0875))


def k_calendar(temp_c, soc, a):
    """Calendar fade coefficient; `degradation_scalar` scales it for lower-grade cells."""
    T = temp_c + 273.15
    ua = graphite_potential(soc)
    return a.get("degradation_scalar", 1.0) * abs(a["p1"]) * math.exp(a["p2"] / T) * math.exp(a["p3"] * ua / T)


def k_cycle(temp_c, dod, c_rate, a):
    """Cycle fade coefficient; `degradation_scalar` scales it for lower-grade cells."""
    T = temp_c + 273.15
    return a.get("degradation_scalar", 1.0) * (a["p4"] + a["p5"] * dod + a["p6"] * c_rate) * (
        math.exp(a["p7"] / T) + math.exp(-a["p8"] / T))


def _advance_power_law(q, k, p, dx):
    """Damage added by advancing x by dx on q = k * x**p, starting from damage q."""
    if dx <= 0 or k <= 0:
        return 0.0
    x_eq = (q / k) ** (1.0 / p) if q > 0 else 0.0
    return k * ((x_eq + dx) ** p - x_eq ** p)


def battery_temp(ambient_c, power_kw, battery):
    """Battery temperature during one hour, in degC.

    ambient_c : outdoor air temperature, degC
    power_kw  : battery power, + charging / - discharging, kW
    Indoor offset + steady-state self-heating from conversion losses:
        T = ambient + offset + R_th * P_loss,  P_loss = |P| * (1 - efficiency)
    """
    th = battery.get("thermal", {})
    eff = battery["charge_eff"] if power_kw >= 0 else battery["discharge_eff"]
    p_loss_w = abs(power_kw) * 1000.0 * (1.0 - eff)
    return ambient_c + battery.get("temp_offset_c", 0.0) + th.get("r_th_k_per_w", 0.0) * p_loss_w


# --------------------------------------------------------------------------- state
def reset():
    """New simulation run: the next step_loss() call starts from a new battery."""
    global _RUN
    _RUN += 1


def reset_state(battery):
    """Start a fresh, new battery in this dict."""
    battery["_aging_state"] = {
        "run": _RUN, "q_cal": 0.0, "q_cyc": 0.0, "t_days": 0.0, "efc": 0.0,
        "soc_hist": deque(maxlen=_HISTORY_H),
        "crate_hist": deque(maxlen=_HISTORY_H),
    }
    return battery["_aging_state"]


def _state(battery):
    s = battery.get("_aging_state")
    if s is None or s["run"] != _RUN:
        s = reset_state(battery)
    return s


def get_state(battery):
    """Current ageing state: calendar part, cycle part, days, equivalent full cycles."""
    s = _state(battery)
    return {"q_cal": s["q_cal"], "q_cyc": s["q_cyc"], "q_total": s["q_cal"] + s["q_cyc"],
            "capacity_frac": 1.0 - s["q_cal"] - s["q_cyc"],
            "t_days": s["t_days"], "efc": s["efc"]}


# --------------------------------------------------------------------------- main API
def step_loss(dt_h, soc, temp_c, throughput_kwh, capacity_kwh, battery):
    """Capacity lost during one step, as a fraction of the INITIAL capacity (>= 0).

    dt_h           : step length, hours (1.0)
    soc            : average state of charge in the step, 0-1
    temp_c         : battery temperature from battery_temp(), degC
    throughput_kwh : energy charged + discharged in the step, kWh (>= 0)
    capacity_kwh   : current usable capacity, kWh (not needed by this model; kept for the contract)
    battery        : the battery.json dict (also holds the running state)
    """
    global _LAST
    _LAST = battery
    a = battery["aging"]
    s = _state(battery)
    e_nom = battery["capacity_kwh"]
    soc = min(max(soc, 0.0), 1.0)

    # stressors over the last 24 h (one daily cycle)
    s["soc_hist"].append(soc)
    c_now = throughput_kwh / (dt_h * e_nom)          # |current| / nominal capacity, 1/h
    s["crate_hist"].append(c_now if c_now >= 1e-2 else 0.0)
    dod = max(s["soc_hist"]) - min(s["soc_hist"])
    c_rate = sum(s["crate_hist"]) / len(s["crate_hist"])

    # calendar ageing: time
    dt_days = dt_h / 24.0
    dq_cal = _advance_power_law(s["q_cal"], k_calendar(temp_c, soc, a), a["p_cal"], dt_days)

    # cycle ageing: charge throughput as equivalent full cycles (nominal capacity)
    d_efc = throughput_kwh / (2.0 * e_nom)
    dq_cyc = _advance_power_law(s["q_cyc"], k_cycle(temp_c, dod, c_rate, a), a["p_cyc"], d_efc)

    s["q_cal"] += dq_cal
    s["q_cyc"] += dq_cyc
    s["t_days"] += dt_days
    s["efc"] += d_efc
    return dq_cal + dq_cyc


def breakdown():
    """Accumulated calendar and cycle losses (fractions of initial capacity) and FEC."""
    if _LAST is None:
        return {"calendar_loss_frac": 0.0, "cycle_loss_frac": 0.0, "fec": 0.0}
    st = get_state(_LAST)
    return {"calendar_loss_frac": st["q_cal"], "cycle_loss_frac": st["q_cyc"], "fec": st["efc"]}
