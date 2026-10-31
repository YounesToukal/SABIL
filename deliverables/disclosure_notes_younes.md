# Disclosure notes (Younes's part) — for the portal form

## What existed before the event
- No prior code. Prior know-how only.

## Built during the sprint (Oct 30 – Nov 1, 2026)
- Weather download and processing (`src/weather.py`)
- Hourly household load profile (`src/load_profile.py`)
- Simulation engine, standard and smart controllers, forecasts (`src/simulate.py`, `src/controllers.py`)
- Result checks and charts (`src/check_results.py`, `src/plots.py`)
- Kaggle notebook (`notebooks/sabil_kaggle.ipynb`)

- Ageing model verification, learned forecast, experiments (ablation, trade-off sweep,
  Monte Carlo) and the auto-generated research report (`src/validate_aging.py`,
  `src/forecast_ml.py`, `src/experiments.py`, `src/report.py`)

## Data
- NASA POWER hourly API (power.larc.nasa.gov), ALLSKY_SFC_SW_DWN, T2M, RH2M,
  Maiduguri 11.85° N, 13.16° E, years 2014–2023. Public.

- ERA5 reanalysis (ECMWF / Copernicus) via the Open-Meteo archive API, hourly 2014–2023:
  cloud cover by layer, dew point, humidity, pressure, wind, rain, soil moisture, radiation.
- NOAA climate indices: ONI (CPC), NAO (CPC), TNA (PSL).

## Pretrained models
- Chronos-Bolt (Amazon, `amazon/chronos-bolt-small`), used zero-shot as a forecasting baseline.

## Published models
- Battery ageing: Naumann et al. 2018 (calendar) and 2020 (cycle), LFP/graphite;
  coefficients and method checked against the authors' open-source SimSES (TU Munich).

## Open-source libraries
- Python, NumPy, pandas, Matplotlib, Requests, scikit-learn, LightGBM, XGBoost, PyTorch,
  chronos-forecasting, SimSES (verification only)

## Tools
- Kaggle Notebooks (compute)
- GitHub (repository)
- AI assistant: Claude (Anthropic) used to help draft and review code and charts

## Headline numbers use
- Smart controller forecast: see `params/sim.json → headline_forecast`
  (persistence = realistic; perfect = upper bound, also reported)
