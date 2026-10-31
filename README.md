# SABIL

A home solar-plus-storage kit whose controller is designed to make the battery last,
sold pay-per-kWh with battery health guaranteed by the operator.

Global Energy Hackathon 2026 · Brief 13 "Power for a city the grid can't protect"
(Maiduguri, Nigeria) · Clean Energy Solutions · Residential.

## Proof of concept

One-year hourly simulation of a 5-person household in Maiduguri on real weather
(NASA POWER 2023), repeated over 10 years for battery ageing. Same house, same
weather, same battery: a standard controller (what cheap kits do) vs the SABIL
smart controller. Only the decisions change.

## Run it

```bash
pip install numpy pandas matplotlib requests
python src/weather.py          # downloads NASA POWER once, saves data/weather_2023.csv
python src/load_profile.py     # hourly household load from params/household.json
python src/simulate.py         # baseline + smart (perfect and persistence forecasts) + charts
python src/check_results.py    # contract keys, energy balance, SoC limits
```

Or open `notebooks/sabil_kaggle.ipynb` on Kaggle (Internet ON).

## Layout (integration contract)

| Path | Owner | What |
|---|---|---|
| `params/household.json`, `system.json`, `battery.json`, `costs.json` | Youcef | inputs with sources |
| `params/sim.json` | Younes | site, years, forecast choice, smart-controller settings |
| `dev_params/` | Younes | TO CONFIRM placeholders, used only while a `params/` file is missing |
| `src/aging_model.py`, `src/cost_model.py` | Youcef | ageing and cost models |
| `src/weather.py`, `load_profile.py`, `controllers.py`, `simulate.py`, `plots.py`, `check_results.py`, `params_io.py` | Younes | data, simulation, controllers, charts |
| `results/` | Younes (summaries, hourly CSVs, figures), Youcef (`costs_summary.json`) | outputs |

No number is typed into the code: every value lives in `params/`.

## Forecasts

The smart controller plans with a next-day sun and load forecast:

- `perfect`: tomorrow's real data (upper bound)
- `persistence`: same hour on the most recent past day (realistic, no look-ahead)

Both always run. `params/sim.json → headline_forecast` picks which one becomes
`summary_smart.json`.
