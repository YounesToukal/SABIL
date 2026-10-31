"""Hourly household load from params/household.json (owner: Younes).

Each appliance runs at power_w x quantity during its use_hours (local hours).
Fixed loads are split into critical (lights, phones, fan...) and non-critical.
Flexible appliances (flexible=true with allowed_hours) are returned separately:
the standard controller runs them at use_hours, the smart controller may move
them to any allowed hour.

Run:  python src/load_profile.py   -> prints a daily summary, writes data/load_profile.csv
"""
import numpy as np
import pandas as pd

from params_io import ROOT, load_all


def build(household, hours_of_day):
    """hours_of_day: int array (0-23), one entry per simulated hour."""
    n = len(hours_of_day)
    crit, nonc, flex = np.zeros(n), np.zeros(n), []
    scale = household.get("load_scale", 1.0)   # set only by the uncertainty analysis
    for a in household["appliances"]:
        kw = a["power_w"] * a["quantity"] / 1000.0 * scale
        if a.get("flexible") and a.get("allowed_hours"):
            flex.append({"name": a["name"], "kw": kw, "n_hours": len(a["use_hours"]),
                         "use_hours": list(a["use_hours"]), "allowed_hours": list(a["allowed_hours"]),
                         "critical": bool(a.get("critical"))})
        else:
            mask = np.isin(hours_of_day, a["use_hours"])
            (crit if a.get("critical") else nonc)[mask] += kw
    return {"critical_kw": crit, "noncritical_kw": nonc, "flex": flex}


def flex_at_usual_hours(flex, hours_of_day):
    """Flexible loads placed at their usual use_hours: (critical_kw, noncritical_kw)."""
    c, nc = np.zeros(len(hours_of_day)), np.zeros(len(hours_of_day))
    for f in flex:
        mask = np.isin(hours_of_day, f["use_hours"])
        (c if f["critical"] else nc)[mask] += f["kw"]
    return c, nc


def summary(household):
    hod = np.arange(24)
    L = build(household, hod)
    fc, fnc = flex_at_usual_hours(L["flex"], hod)
    total = L["critical_kw"] + L["noncritical_kw"] + fc + fnc
    night = np.isin(hod, list(range(18, 24)) + list(range(0, 6)))
    print(f"daily energy: {total.sum():.2f} kWh "
          f"(critical {L['critical_kw'].sum() + fc.sum():.2f}, non-critical {L['noncritical_kw'].sum() + fnc.sum():.2f})")
    print(f"night share (18:00-06:00): {total[night].sum() / total.sum():.2f}")
    print(f"peak hour: {int(total.argmax())}:00 at {total.max():.2f} kW")
    for f in L["flex"]:
        print(f"flexible: {f['name']} {f['kw']:.2f} kW x {f['n_hours']} h, usual {f['use_hours']}, "
              f"allowed {f['allowed_hours']}")
    return total


if __name__ == "__main__":
    P = load_all()
    summary(P["household"])
    from weather import load_weather
    w = load_weather(P["sim"])
    hod = w["timestamp_local"].dt.hour.to_numpy()
    L = build(P["household"], hod)
    fc, fnc = flex_at_usual_hours(L["flex"], hod)
    w = w[w["wyear"] == w["wyear"].iloc[-1]].reset_index(drop=True)   # one year is enough
    hod = w["timestamp_local"].dt.hour.to_numpy()
    L = build(P["household"], hod)
    fc, fnc = flex_at_usual_hours(L["flex"], hod)
    out = pd.DataFrame({"timestamp_local": w["timestamp_local"],
                        "critical_kw": L["critical_kw"] + fc,
                        "noncritical_kw": L["noncritical_kw"] + fnc})
    out["load_kw"] = out["critical_kw"] + out["noncritical_kw"]
    out.to_csv(ROOT / "data" / "load_profile.csv", index=False, float_format="%.4f")
    print("saved data/load_profile.csv (usual hours, before any smart shifting)")
