# SABIL interface

Open `index.html` in any browser. It works offline: the engine, the 2023 weather, the
learned sun forecasts, the logo and the fonts are all inside the one file.

Three views:

- **Family**: what the home sees. Battery level, the part kept for essentials (lights,
  phones, night fan, fridge), credit in kWh and naira, and the best hours to charge phones.
- **Controller**: the live readings (sun, air and battery temperature, charge, battery
  voltage and current, home use), each decision of the hour with its reason, alerts, the
  last and next 24 hours, the battery health model next to a standard kit, and the sun
  forecast. "Try a situation" adds a heat wave, a cloudy tomorrow or dusty panels.
- **Fleet**: twelve example homes with the selected kit, sorted by what needs attention.

Data sources: a replay of NASA POWER hourly weather for Maiduguri (2023), sliders, or
readings from a controller (paste one JSON reading, load an hourly CSV log, or read a URL on
the home network).

## Rebuild and check it

```bash
python app/build/build.py          # rebuilds app/index.html from the repo's data and app/build/
python app/build/check_engine.py   # runs the Python simulation and the page's engine on 2023, side by side (needs Node.js)
```

`app/build/` holds the page source: `engine.js` (the engine), `page_head.html` (layout and
style), `page_logic.js.html` (fleet, forecasts, charts), `page_ui.js.html` (views and controls),
the fonts (Sora and Atkinson Hyperlegible, SIL Open Font License) and the logo.

## Same engine as the simulation

`build/engine.js` (inlined in `index.html`) is a line-by-line port of `src/simulate.py`,
`src/controllers.py`, `src/aging_naumann.py`, the default battery temperature of
`src/aging_model.py` and `src/load_profile.py`, with the selected kit from `params/`:
24 V DC, 400 Wp, LFP 25.6 V 75 Ah (1.92 kWh), household 1,164 Wh a day, SABIL rules with
the cloudy-day buffer.

Check: both engines run on the 2023 weather with the learned forecast.

| End of 2023 | Python | Browser |
|---|---|---|
| Standard kit, capacity left | 0.918949 | 0.918948 |
| SABIL, capacity left | 0.930452 | 0.930451 |
| SABIL, energy served (kWh) | 424.29 | 424.29 |
| SABIL, full cycles | 135.89 | 135.88 |
| SABIL, mean battery temperature (°C) | 33.48 | 33.48 |

Battery life figures quoted by the team come from the 10-year study (`results/summary_*.json`):
7.5 years with SABIL against 6.2 with a standard kit. The interface's per-home projection uses
one home's data so far and is labelled as such.
