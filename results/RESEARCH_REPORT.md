# SABIL: battery-aware control for solar home systems in Maiduguri

*Generated 2026-11-01 09:07 UTC from the result files at commit 49399d2. Every number below is read from `results/`, not typed.*

> **Provisional:** these inputs still use placeholders (`dev_params/`): costs.

## 1. Question
When the transmission line to Maiduguri is cut (Brief 13), a household solar-plus-battery kit replaces diesel only as long as its battery lasts. Can a controller that knows how batteries age extend battery life in Maiduguri's heat without reducing the hours that lights, phones and fans stay on?

## 2. Method
- **Site and weather:** Maiduguri, Borno State, Nigeria (11.85 N, 13.16 E). NASA POWER hourly irradiance, air temperature and humidity, multi_year: 2014-2023, local time (UTC+1).
- **Household:** 5 people, 1.16 kWh/day; hourly appliance schedule from `params/household.json`; critical loads: LED lights, LED security light, Phone charging, DC fan, DC refrigerator ~100 L. Grid assumed down for the whole horizon (worst case of Brief 13).
- **Paired design:** the same house, weather, PV array and battery are simulated twice; only the controller differs. Any difference is caused by the controller.
- **Standard controller:** charges to 100% whenever there is sun, discharges to the low-voltage cut-off, serves all loads equally, runs appliances at their usual hour.
- **SABIL controller:** (1) caps the charge level on hot days when tomorrow is sunny; (2) keeps a night reserve for critical loads; (3) moves flexible loads (Phone charging, DC refrigerator ~100 L) to sunny hours; (4) serves critical loads first; (5) limits power when the battery is hot. Plans with a next-day sun forecast.
- **Battery ageing:** semi-empirical LFP/graphite model (Naumann et al. 2018 calendar, 2020 cycle): calendar loss ~ sqrt(time), accelerated by temperature (Arrhenius) and state of charge; cycle loss ~ sqrt(full equivalent cycles), accelerated by depth and C-rate; half-cycles counted at each SoC reversal.
- **Ageing model verification:** closed-form square-root laws reproduced to machine precision; physics sanity checks passed; cross-check against the authors' open-source implementation (SimSES, TU Munich): relative difference 4.5e-14 over one simulated year. Overall: PASSED.

## 3. Main result (headline forecast: persistence)
| Metric | Standard | SABIL |
|---|---|---|
| Battery life to 80% capacity (years) | 6.24 | 7.51 |
| Capacity left after the horizon | 74.6% | 76.6% |
| Hours with all critical loads on | 99.87% | 99.99% |
| Critical hours missed per year | 11 | 1 |
| Energy delivered (kWh/year) | 425 | 424 |
| Energy not delivered (kWh/year) | 0.3 | 0.7 |
| Diesel avoided (L/year) | 212 | 212 |
| Diesel avoided over the battery's life (L) | 1,326 | 1,594 |
| CO2 avoided (t/year) | 0.57 | 0.57 |
| Full battery cycles per year | 155 | 126 |
| Average SoC when the battery is hot | 83.2% | 79.6% |

**SABIL extends battery life by 1.27 years (+20%).**
 Ageing split at the end of the horizon - standard: calendar 20.5%, cycle 4.8%; SABIL: calendar 19.2%, cycle 4.2%.

![capacity](figures/fig1_battery_capacity.png)
![critical](figures/fig2_critical_hours.png)

## 4. Does the forecast matter?
| Forecast used by SABIL | Battery life (y) | Critical hours | Energy not delivered (kWh/y) |
|---|---|---|---|
| perfect | 7.51 | 99.99% | 0.5 |
| persistence | 7.51 | 99.99% | 0.7 |
| learned | 7.59 | 99.98% | 0.7 |

Learned next-day forecast (ensemble, leave-one-year-out over 2014-2023), mean absolute error in kWh/m2/day:

| Season | Learned | Tomorrow = today | Seasonal average | Skill vs tomorrow = today |
|---|---|---|---|---|
| all | 0.612 | 0.796 | 0.639 | +23% |
| dry | 0.377 | 0.461 | 0.424 | +18% |
| wet | 0.970 | 1.329 | 0.973 | +27% |

![forecast](figures/fig9_forecast_skill.png)

### 4b. Forecasting lab: datasets x features x models
Data sources: NASA POWER (sun, temperature, humidity), ERA5 reanalysis via Open-Meteo (clouds by layer, dew point, pressure, wind, rain, soil moisture, ERA5 sun), NOAA climate indices (ONI, NAO, TNA). Feature groups used: history (8), season (4), nasa_met (7), era5_cloud (8), era5_dyn (10), era5_sun (2), climate (3).

| Model | Next-day MAE, year | Dry season | Rainy season | DM vs persistence (stat, p) | DM vs climatology (stat, p) |
|---|---|---|---|---|---|
| ensemble | 0.612 | 0.377 | 0.970 | -15.1, 0.000 | -5.9, 0.000 |
| ridge | 0.614 | 0.378 | 0.978 | -15.5, 0.000 | -5.6, 0.000 |
| xgboost | 0.615 | 0.382 | 0.972 | -14.9, 0.000 | -4.7, 0.000 |
| random_forest | 0.617 | 0.386 | 0.970 | -14.8, 0.000 | -5.1, 0.000 |
| extra_trees | 0.618 | 0.386 | 0.973 | -14.6, 0.000 | -5.3, 0.000 |
| lightgbm | 0.618 | 0.383 | 0.979 | -14.5, 0.000 | -4.0, 0.000 |
| hist_gb | 0.622 | 0.388 | 0.983 | -14.0, 0.000 | -3.1, 0.002 |
| gru | 0.631 | 0.389 | 0.977 | -12.2, 0.000 | -1.1, 0.272 |
| climatology | 0.639 | 0.424 | 0.973 | - | - |
| chronos_bolt | 0.648 | 0.411 | 1.003 | -13.5, 0.000 | +1.2, 0.222 |
| mlp | 0.794 | 0.523 | 1.175 | -0.1, 0.898 | +15.2, 0.000 |
| persistence | 0.796 | 0.461 | 1.329 | - | - |

Diebold-Mariano: negative statistic = more accurate than the reference; p < 0.05 = difference unlikely to be chance.

Probabilistic forecasts (80% intervals): cqr: coverage 76.3%, mean width 1.69 kWh/m2/day; chronos_bolt: coverage 78.6%, mean width 1.98 kWh/m2/day.

| Data source / feature group | Error increase if removed | Error reduction when added to sun history + season |
|---|---|---|
| history | +0.0080 | +0.0000 |
| season | +0.0067 | +0.0000 |
| climate | +0.0019 | -0.0001 |
| era5_dyn | +0.0017 | +0.0149 |
| era5_sun | +0.0015 | -0.0021 |
| era5_cloud | +0.0014 | +0.0041 |
| nasa_met | -0.0017 | +0.0091 |

Decision-focused test (each forecast drives SABIL for the whole horizon):

| Forecast | Battery life (y) | Critical hours | Energy not delivered (kWh/y) |
|---|---|---|---|
| persistence | 8.19 | 99.99% | 17.4 |
| climatology | 8.21 | 100.00% | 17.8 |
| ridge | 8.22 | 100.00% | 17.8 |
| random_forest | 8.21 | 100.00% | 17.7 |
| extra_trees | 8.21 | 100.00% | 17.7 |
| hist_gb | 8.21 | 100.00% | 17.7 |
| mlp | 8.2 | 100.00% | 17.7 |
| lightgbm | 8.21 | 100.00% | 17.7 |
| xgboost | 8.21 | 100.00% | 17.7 |
| ensemble | 8.21 | 100.00% | 17.7 |
| gru | 8.22 | 100.00% | 17.9 |
| chronos_bolt | 8.22 | 99.99% | 17.9 |
| risk_aware_p10 | 8.16 | 100.00% | 16.6 |

Battery life varies by only 0.06 years across all forecasts: SABIL's benefit does not depend on a sophisticated forecast, which matters for deployment on cheap hardware without internet.

Note: this table was computed with the first, placeholder kit (dev_params: 1.5 kWp, 5.12 kWh). On the selected kit the main simulation confirms the same conclusion (battery life with each forecast in the headline table above).

![leaderboard](figures/fig10_forecast_leaderboard.png)
![groups](figures/fig11_feature_groups.png)
![decision](figures/fig12_decision_focused.png)

## 5. Which feature does the work? (ablation)
All features off reproduces the standard controller exactly: **True** (sanity check).

| Feature | Battery life lost if removed (y) | Life gain alone (y) | Critical hours if removed |
|---|---|---|---|
| charge_cap | +1.24 | +0.27 | 99.99% |
| load_shift | +1.00 | +0.03 | 99.96% |
| night_reserve | +0.00 | +0.00 | 99.93% |
| heat_gentle | +0.00 | +0.00 | 99.99% |
| critical_priority | +0.00 | +0.00 | 99.92% |

![ablation](figures/fig5_ablation.png)

## 6. Design trade-off: hot-day charge cap
From a cap of 100% to 40%, battery life goes from 6.27 to 7.51 years while energy delivered changes from 425 to 424 kWh/year. Chosen cap 70%: 7.51 years.

![tradeoff](figures/fig6_tradeoff.png)

## 7. Uncertainty (Monte Carlo)
96 paired scenarios, inputs sampled jointly (ranges in `params/sim.json`):

- Battery-life gain: **median 1.08 years** (90% interval 0.26 to 3.37); relative gain median +19% (+4% to +52%).
- SABIL lasts longer than the standard controller in **100%** of scenarios.
- Battery life, standard: 6.1 y (2.7-11.0); SABIL: 7.4 y (3.3-12.9).
- What drives the gain (Spearman rank correlation with years gained): load_scale -0.80, cal_k_ref_per_sqrt_s -0.40, temp_offset_c +0.32, cyc_c_b -0.16.

![uncertainty](figures/fig7_uncertainty.png)
![drivers](figures/fig8_gain_drivers.png)

## 7b. How far from optimal? (dynamic-programming benchmark)
A degradation-aware dynamic program with **perfect foresight** re-plans the battery every day over 48 hours, minimising weighted unmet energy (critical loads x50) plus the value of capacity lost (same ageing model). It is an upper bound, not a product.

| Controller | Battery life (y) | Energy delivered (kWh/y) | Critical hours |
|---|---|---|---|
| Standard | 6.24 | 425 | 99.87% |
| SABIL | 7.51 | 424 | 99.99% |
| Optimal, wear weight 0 | 10.28 | 425 | 99.96% |
| Optimal, wear weight 1500 | 10.23 | 425 | 99.96% |
| Optimal, wear weight 3700 | 10.26 | 425 | 99.96% |
| Optimal, wear weight 10000 | 10.34 | 425 | 99.93% |
| Optimal, wear weight 30000 | 10.60 | 423 | 99.80% |
| Optimal, wear weight 100000 | 10.94 | 421 | 99.47% |

At the economic wear weight (1400), SABIL's rules capture **32%** of the battery-life gain that a perfect controller achieves. The remaining headroom is the roadmap: model-predictive control on the device.

![headroom](figures/fig14_headroom.png)

## 7c. Where to put the battery (thermal model with self-heating)
| Placement | Battery mean temp (C) | Life, standard (y) | Life, SABIL (y) | Assumption |
|---|---|---|---|---|
| outdoor_shaded_box | 31.5 | 6.41 | 7.75 | metal box outside in the shade: follows air temperature, enclosure adds heat |
| indoor_metal_roof | 30.6 | 6.85 | 7.96 | room under a corrugated metal roof: some damping, roof heat gain |
| indoor_masonry | 28.6 | 7.54 | 8.37 | thick mud-brick or block room: daily swing damped, same mean |
| cool_interior_floor | 26.6 | 8.30 | 8.88 | on the floor of the coolest interior room, ground-coupled |

Placement is free advice for every household and stacks with SABIL. Placement parameters are scenario assumptions; a temperature logger in pilot homes would measure them.

![placement](figures/fig13_placement.png)

## 7d. Random household demand
24 random demand realisations (RAMP-style: shifted usage times, skipped days, variable intensity): battery-life gain median 1.25 years (1.23-1.29); SABIL better in 100%; hours with all critical loads on: standard 99.88%, SABIL 99.98% (worst case 99.98%).

## 8. Limitations
- Simulation, not field data: the ageing model is fitted on one commercial LFP cell (Sony/Murata US26650) in the lab; real packs, cell imbalance and BMS behaviour differ. A 20-50 home pilot with logged SoC and temperature is the test.
- Headline runs use battery temperature = air temperature + a fixed offset; the placement study adds a two-node thermal model with self-heating, whose placement parameters are assumptions to be measured in the field.
- The optimal benchmark uses perfect foresight and marginal (linearised) ageing rates; it bounds what any controller could achieve, it is not deployable as is.
- Household demand is a fixed daily schedule; real behaviour varies day to day and grows over time.
- Battery life beyond the simulated horizon is extrapolated with the last year's fade rate (conservative for square-root ageing).
- Weather years are treated as independent blocks in the forecast cross-validation; NASA POWER is a reanalysis product with known bias versus ground stations.

## 9. Reproduce
```
python src/weather.py && python src/validate_aging.py && python src/forecast_ml.py
python src/simulate.py && python src/experiments.py && python src/plots.py && python src/report.py
```
Python 3.13.18. Data: NASA POWER (public). Ageing model reference implementation: SimSES (TU Munich).

## References
- Naumann M., Schimpe M., Keil P., Hesse H.C., Jossen A. Analysis and modeling of calendar aging of a commercial LiFePO4/graphite cell. J. Energy Storage 17 (2018) 153-169. doi:10.1016/j.est.2018.01.019
- Naumann M., Spingler F.B., Jossen A. Analysis and modeling of cycle aging of a commercial LiFePO4/graphite cell. J. Power Sources 451 (2020) 227666.
- SimSES, open-source simulation of stationary energy storage systems, TU Munich (PyPI `simses`).
- NASA POWER hourly data, power.larc.nasa.gov.
- IPCC 2006 Guidelines for National Greenhouse Gas Inventories, Vol. 2 (default CO2 factor for diesel).
- Global Energy Hackathon 2026, Problem Brief 13: Power for a city the grid can't protect.
