"""Load the params/ files (owner: Younes).

Reads params/<name>.json. If Youcef has not pushed a file yet, falls back to
dev_params/<name>.json (placeholders marked TO CONFIRM) and says so loudly.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YOUCEF_FILES = ["household", "system", "battery", "costs"]


def _read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all(root=ROOT, verbose=True):
    root = Path(root)
    params, used_defaults = {}, []
    for name in YOUCEF_FILES:
        real = root / "params" / f"{name}.json"
        default = root / "dev_params" / f"{name}.json"
        if real.exists():
            params[name] = _read(real)
        else:
            params[name] = _read(default)
            used_defaults.append(name)
    params["sim"] = _read(root / "params" / "sim.json")
    if verbose:
        if used_defaults:
            print("WARNING - using TO CONFIRM placeholders for:", ", ".join(used_defaults))
        else:
            print("All params come from params/ (Youcef's files).")
        for name in YOUCEF_FILES:
            if "TO CONFIRM" in json.dumps(params[name]):
                print(f"  note: {name}.json still contains 'TO CONFIRM'")
    params["_used_defaults"] = used_defaults
    return params
