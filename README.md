<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/sabil-horizontal-reversed.svg">
    <img src="assets/brand/sabil-horizontal-colour.svg" alt="SABIL: Solar Access Builds Independent Lives" width="420">
  </picture>
</p>

# SABIL

A home solar-plus-storage kit whose controller is designed to make the battery last,
sold pay-per-kWh with battery health guaranteed by the operator.

Global Energy Hackathon 2026 · Brief 13 "Power for a city the grid can't protect"
(Maiduguri, Nigeria) · Clean Energy Solutions · Residential.

## Proof of concept

Ten-year hourly simulation of a 5-person household in Maiduguri on real weather
(NASA POWER 2014-2023, each year once), with the battery ageing hour by hour. Same house,
same weather, same battery: a standard controller (what cheap kits do) vs the SABIL
smart controller. Only the decisions change.

The kit (400 Wp PV, 1.92 kWh LFP, 24 V DC) is chosen by the critical-load reliability of
the AGED system over the 10 weather years (`src/sizing_map.py`), and every ageing result is
checked with two LFP ageing models fitted on different datasets (`src/robustness_aging.py`).

## Run it

```bash
pip install numpy pandas matplotlib requests
python src/weather.py            # downloads NASA POWER 2014-2023 once into data/
python src/sizing.py             # household load and PV/battery sizing from the NASA years
python src/validate_aging.py     # TU Munich ageing model vs closed form and the authors' SimSES code
python src/validate_aging_nrel.py  # NREL prismatic ageing model vs closed form (and NREL's code, optional)
python src/load_profile.py       # hourly household load from params/household.json
python src/forecast_lab.py       # forecasting lab: ERA5 + NOAA indices, 12 models (GRU and Chronos need torch), learned forecast for the simulation
python src/simulate.py           # baseline + smart (perfect, persistence, learned forecasts) + charts
python src/check_results.py      # contract keys, energy balance, SoC limits
python src/sizing_map.py         # reliability of the aged system: 4 PV x 4 battery sizes x 2 controllers x 2 ageing models
python src/robustness_aging.py   # controller results under both ageing models
python src/experiments.py        # ablation, charge-cap sweep, Monte Carlo, battery placement, random demand, optimal benchmark
python src/plots.py              # all figures
python src/cost_model.py         # cost per kWh, family bill, generator comparison, value of the controller
python src/tune_controller.py    # Bayesian optimization of the controller rules, tested on unseen weather (needs optuna)
python src/exhibits.py           # report and slide exhibits (fig15-fig20, results/figures/deck/)
python src/report.py             # results/RESEARCH_REPORT.md
```

## Layout (integration contract)

| Path | Owner | What |
|---|---|---|
| `params/household.json`, `system.json`, `battery.json`, `costs.json` | Youcef | inputs with sources |
| `params/sim.json` | Younes | site, years, forecast choice, smart-controller settings |
| `dev_params/` | Younes | TO CONFIRM placeholders, used only while a `params/` file is missing |
| `src/sizing.py`, `src/sizing_map.py` | Youcef | household load, kit sizing from NASA POWER, reliability map of the aged system |
| `src/aging_model.py`, `src/aging_nrel.py`, `src/validate_aging_nrel.py`, `src/robustness_aging.py`, `src/cost_model.py`, `src/tune_controller.py`, `src/exhibits.py` | Youcef | ageing entry point (picks the model in `battery.json`), NREL prismatic LFP model and its checks, robustness across ageing datasets, cost model |
| `src/aging_naumann.py`, `src/validate_aging.py` | Younes | TU Munich LFP ageing model (default) and its check against SimSES |
| `src/weather.py`, `load_profile.py`, `controllers.py`, `simulate.py`, `plots.py`, `check_results.py`, `params_io.py`, `forecast_ml.py`, `forecast_lab.py`, `datasets.py`, `experiments.py`, `report.py` | Younes | data, simulation, controllers, forecasts and forecasting lab, experiments, charts, report |
| `assets/brand/` | Youcef | logo (Solar Access Builds Independent Lives) |
| `results/` | Younes (summaries, hourly CSVs, experiments, figures, report), Youcef (`sizing_map.csv`, `aging_robustness.json`, `aging_validation_nrel.json`, `load_profile_24h.csv`, `costs_summary.json`) | outputs |

No number is typed into the code: every value lives in `params/`.

## Forecasts

The smart controller plans with a next-day sun and load forecast:

- `perfect`: tomorrow's real data (upper bound)
- `persistence`: same hour on the most recent past day (realistic, no look-ahead)

Both always run. `params/sim.json → headline_forecast` picks which one becomes
`summary_smart.json`.

## Key results (400 Wp, 1.92 kWh, NASA POWER 2014-2023, realistic forecast)

| | Standard | SABIL |
|---|---|---|
| Critical hours lost, worst year (TU Munich / NREL ageing) | 32 / 13 | 9 / 9 |
| Battery life to 80% capacity, TU Munich ageing (years) | 6.24 | 7.51 |
| Capacity after 10 years, TU Munich / NREL ageing | 74.6% / 87.4% | 76.6% / 88.7% |

Monte Carlo over ageing, temperature, load and PV uncertainty (96 paired samples): SABIL lengthens
battery life in every sample (`results/experiments/montecarlo_summary.json`).

## Main references

- NASA POWER hourly data, Maiduguri 2014-2023
- Naumann et al., J. Energy Storage 17 (2018) 153-169; J. Power Sources 451 (2020) 227666 (TU Munich LFP ageing; SimSES)
- NREL BLAST-Lite, large-format prismatic LFP model (data in J. Energy Storage, doi:10.1016/j.est.2023.109042)
- Narayan et al., Applied Energy 240 (2019) 907-917 (solar home system sizing for the Multi-Tier Framework)
- CLASP / Efficiency for Access appliance technology briefs (2021)
