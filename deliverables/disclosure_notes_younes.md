# Disclosure notes (Younes's part) — for the portal form

## What existed before the event
- No prior code. Prior know-how only.

## Built during the sprint (Oct 30 – Nov 1, 2026)
- Weather download and processing (`src/weather.py`)
- Hourly household load profile (`src/load_profile.py`)
- Simulation engine, standard and smart controllers, forecasts (`src/simulate.py`, `src/controllers.py`)
- Result checks and charts (`src/check_results.py`, `src/plots.py`)
- Kaggle notebook (`notebooks/sabil_kaggle.ipynb`)

## Data
- NASA POWER hourly API (power.larc.nasa.gov), ALLSKY_SFC_SW_DWN and T2M,
  Maiduguri 11.85° N, 13.16° E, year 2023. Public.

## Open-source libraries
- Python, NumPy, pandas, Matplotlib, Requests

## Tools
- Kaggle Notebooks (compute)
- GitHub (repository)
- AI assistant: Claude (Anthropic) used to help draft and review code and charts

## Headline numbers use
- Smart controller forecast: see `params/sim.json → headline_forecast`
  (persistence = realistic; perfect = upper bound, also reported)
