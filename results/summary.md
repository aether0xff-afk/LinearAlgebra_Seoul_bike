# Reproducible analysis summary

Data: UCI Seoul Bike Sharing Demand (CC BY 4.0), 2017-12-01 to 2018-11-30.
Original rows: 8760
Non-functioning rows removed: 295
Analyzed rows: 8465

Date-level, stratified, deterministic split; season and weekend strata.
Seed: 42; no single date appears in multiple partitions.

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| mean | 508.296 | 633.756 | -0.000 |
| harmonic_24h | 430.716 | 565.230 | 0.204 |
| harmonic_24h_12h | 414.978 | 541.564 | 0.270 |
| harmonic_weekend | 405.584 | 528.761 | 0.304 |

MAE reduction from 24h-only to 24h+12h: 3.65%.

Interpretation: errors refer to held-out dates WITHIN the source year.
No claim is made about forecasting subsequent years or causal effects.
Linear least squares predictions are not forced to be nonnegative.

See metrics.csv, coefficients.csv, hourly_means.csv and figures/.
