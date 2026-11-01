"""Battery ageing entry point used by the simulation (owner: Youcef).

Two LFP/graphite ageing models, fitted on different experimental datasets, behind the
interface fixed by the integration contract. battery.json -> aging.model picks one:

  "naumann_sony"   (default) TU Munich data, Naumann et al. 2018/2020; implementation
                   checked against the authors' SimSES code (src/aging_naumann.py, Younes)
  "nrel_prismatic" large-format prismatic cells, NREL BLAST-Lite model; implementation
                   checked against NREL's code (src/aging_nrel.py, Youcef)

The default is the more conservative of the two (faster fade). Running both is how the
robustness of every ageing conclusion is checked (src/robustness_aging.py).

Battery temperature (shared by both models):
  default   outdoor air + indoor offset + steady-state self-heating from conversion losses,
            T = T_air + offset + R_th * |P| * (1 - efficiency)                    (Youcef)
  "rc"      if battery.json -> thermal.model is "rc": two-node lumped model of the room and
            the pack, with the placement (outdoor box, room under a metal roof...) setting the
            room time constant and offset; used by the placement study            (Younes)
"""
import aging_naumann
import aging_nrel

R_GAS = aging_naumann.R_GAS             # used by the optimal-control benchmark (controllers.Oracle)
_MODELS = {"naumann_sony": aging_naumann, "nrel_prismatic": aging_nrel}
_active = aging_naumann
_th = {"t_room": None, "t_bat": None}     # state of the RC thermal model


def _model(battery):
    return _MODELS[battery.get("aging", {}).get("model", "naumann_sony")]


def reset():
    """Start a new battery in every model (called by simulate.py at the start of a run)."""
    aging_naumann.reset()
    aging_nrel.reset()
    _th.update(t_room=None, t_bat=None)


def observe_soc(soc_end, dt_h=1.0):
    """End-of-step SoC for half-cycle counting (used by the TU Munich model)."""
    aging_naumann.observe_soc(soc_end, dt_h)


def _rc(battery):
    th = battery.get("thermal") or {}
    return th if th.get("model") == "rc" else None


def battery_temp(ambient_c, power_kw, battery):
    """Battery temperature during one hour, in degC.
    Default: stateless (outdoor air + offset + self-heating).
    RC model: room  T_room += dt/tau_room * (T_out + room_offset - T_room)
              pack  C dT/dt = UA (T_room - T_bat) + I^2 R,  I = P / V_nominal
    The RC version keeps state, so it must be called exactly once per simulated hour."""
    th = _rc(battery)
    if th is None:
        return aging_nrel.battery_temp(ambient_c, power_kw, battery)
    pl = th["placements"][th["placement"]]
    if _th["t_room"] is None:
        _th["t_room"] = _th["t_bat"] = ambient_c + pl["room_offset_c"]
    _th["t_room"] += min(1.0, 1.0 / pl["tau_room_h"]) * (ambient_c + pl["room_offset_c"] - _th["t_room"])
    current = abs(power_kw) * 1000.0 / th["v_nominal"]
    q_heat = current ** 2 * th["r_internal_ohm"]
    c_j_k = th["mass_kg"] * th["cp_j_kg_k"]
    _th["t_bat"] += 3600.0 / c_j_k * (pl["ua_w_k"] * (_th["t_room"] - _th["t_bat"]) + q_heat)
    return _th["t_bat"]


def battery_temp_estimate(ambient_c, battery):
    """Stateless estimate for the controller's planning (does not advance the thermal state)."""
    th = _rc(battery)
    if th is None:
        return aging_nrel.battery_temp(ambient_c, 0.0, battery)
    return ambient_c + th["placements"][th["placement"]]["room_offset_c"]


def step_loss(dt_h, soc, temp_c, throughput_kwh, capacity_kwh, battery):
    """Capacity lost during one step, as a fraction of the INITIAL capacity (>= 0)."""
    global _active
    _active = _model(battery)
    return _active.step_loss(dt_h, soc, temp_c, throughput_kwh, capacity_kwh, battery)


def breakdown():
    """Calendar and cycle losses of the model used in the current run."""
    return _active.breakdown()
