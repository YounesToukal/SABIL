"""Build app/index.html: one offline page with the engine, the data, the logo and the fonts inside.

Run from the repository root:
    python app/build/build.py

Inputs (all in the repo):
    data/weather_2023.csv                    NASA POWER hourly weather, Maiduguri, 2023
    results/forecast_learned_hourly.csv      learned sun forecast from src/forecast_lab.py
    app/build/engine.js                      the simulation engine (port of src/simulate.py, controllers.py,
                                             aging_naumann.py, aging_model.py, load_profile.py)
    app/build/page_*.html                    page layout, logic and views
    app/build/fonts/, logo_short_reversed.png, favicon-32.png
"""
import base64
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "app" / "index.html"


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def weather_data():
    """2023 hourly sun and temperature, and the learned forecasts, as compact integer lists."""
    w = pd.read_csv(ROOT / "data" / "weather_2023.csv", parse_dates=["timestamp_local"])
    f = pd.read_csv(ROOT / "results" / "forecast_learned_hourly.csv", parse_dates=["timestamp_local"])
    f = f.set_index("timestamp_local").reindex(w["timestamp_local"]).fillna(0)
    assert len(w) == 8760, f"expected 8760 hours in 2023, got {len(w)}"
    return {
        "start": "2023-01-01T00:00",
        "ghi": w.ghi_w_m2.round().astype(int).tolist(),
        "amb10": (w.ambient_c * 10).round().astype(int).tolist(),
        "fc0": f.ghi_fc_sameday.round().astype(int).tolist(),
        "fc1": f.ghi_fc_nextday.round().astype(int).tolist(),
        "fc1p10": f.ghi_fc_nextday_p10.round().astype(int).tolist(),
        "fc0p10": f.ghi_fc_sameday_p10.round().astype(int).tolist(),
    }


def main():
    fonts = ""
    for family, weight, name in [("Sora", 400, "sora-latin-400-normal.woff2"), ("Sora", 600, "sora-latin-600-normal.woff2"),
                                 ("Atkinson Hyperlegible", 400, "atkinson-hyperlegible-latin-400-normal.woff2"),
                                 ("Atkinson Hyperlegible", 700, "atkinson-hyperlegible-latin-700-normal.woff2")]:
        fonts += (f'@font-face{{font-family:"{family}";font-style:normal;font-weight:{weight};font-display:swap;'
                  f'src:url(data:font/woff2;base64,{b64(HERE / "fonts" / name)}) format("woff2")}}\n')
    engine = (HERE / "engine.js").read_text().replace('if (typeof module !== "undefined") module.exports = SABIL;', "")
    page = ((HERE / "page_head.html").read_text() + (HERE / "page_logic.js.html").read_text() + (HERE / "page_ui.js.html").read_text())
    page = (page.replace("/*__FONTS__*/", fonts).replace("/*__ENGINE__*/", engine)
                .replace("__DATA__", json.dumps(weather_data(), separators=(",", ":")))
                .replace("__LOGO_SHORT_REV__", "data:image/png;base64," + b64(HERE / "logo_short_reversed.png")))
    cut = page.index("</style>") + len("</style>")
    html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<meta name="theme-color" content="#1F2B5B">\n'
            f'<link rel="icon" type="image/png" href="data:image/png;base64,{b64(HERE / "favicon-32.png")}">\n'
            + page[:cut] + "\n</head>\n<body>\n" + page[cut:] + "\n</body>\n</html>\n")
    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(html) / 1024:.0f} KB), works offline")


if __name__ == "__main__":
    main()
