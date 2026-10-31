# src/aging_model.py                    (owner: Youcef)
#
# STUB written by Younes so the simulation runs before the real model arrives.
# Same two functions and arguments as the integration contract.
# Youcef: replace this whole file with the real one. Nothing else needs to change.


def battery_temp(ambient_c, power_kw, battery):
    """Battery temperature during one hour, in °C.
    ambient_c : air temperature, °C
    power_kw  : battery power, + charging / - discharging, kW
    battery   : the whole battery.json as a dict
    First version: return ambient_c + battery["temp_offset_c"]
    """
    return ambient_c + battery["temp_offset_c"]


def step_loss(dt_h, soc, temp_c, throughput_kwh, capacity_kwh, battery):
    """Capacity lost during one step, as a fraction of the INITIAL capacity (>= 0).
    STUB: no ageing until Youcef's model replaces this file.
    """
    return 0.0
