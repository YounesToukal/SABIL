"""Check that the interface's engine gives the same results as the simulation.

Run from the repository root (needs Node.js, which GitHub Codespaces has):
    python app/build/check_engine.py

Both engines run 2023 Maiduguri weather with the learned forecast, the selected kit and both
controllers, then the script prints them side by side.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import simulate  # noqa: E402
from build import weather_data  # noqa: E402
from params_io import load_all  # noqa: E402

JS = """
const S = require(process.argv[1]);
S.setWeather(JSON.parse(require("fs").readFileSync(0, "utf8")));
const out = {};
for (const kind of ["standard", "sabil"]) {
  const k = new S.Kit({ pvKwp: S.P.sys.pv_kwp, loadScale: 1 }, kind); let ts = 0;
  for (let t = 0; t < S.W.n; t++) ts += k.step({ ghi: S.W.ghi[t], amb: S.W.amb[t] }).temp;
  out[kind] = { capacity_end: k.capFrac, served_kwh: k.stats.served, unmet_kwh: k.stats.unmet, full_cycles: k.age.fec, avg_battery_temp_c: ts / S.W.n };
}
console.log(JSON.stringify(out));
"""


def main():
    P = load_all()
    w = pd.read_csv(ROOT / "data" / "weather_2023.csv", parse_dates=["timestamp_local"])
    w["wyear"] = 2023
    js = json.loads(subprocess.run(["node", "-e", JS, str(HERE / "engine.js")], input=json.dumps(weather_data()),
                                   capture_output=True, text=True, check=True).stdout)
    print(f"{'2023, learned forecast':34s}{'Python':>12s}{'Browser':>12s}")
    for py_name, js_name in [("baseline", "standard"), ("smart", "sabil")]:
        s, hr = simulate.run(py_name, P, w, "learned")
        py = {"capacity_end": float(hr.capacity_frac.iloc[-1]), "served_kwh": s["energy_served_kwh_per_year"],
              "unmet_kwh": s["energy_unmet_kwh_per_year"], "full_cycles": s["ageing_breakdown"]["fec"],
              "avg_battery_temp_c": s["avg_battery_temp_c"]}
        for key, v in py.items():
            d = 6 if key == "capacity_end" else 2
            print(f"{js_name + ', ' + key:34s}{v:12.{d}f}{js[js_name][key]:12.{d}f}")


if __name__ == "__main__":
    main()
