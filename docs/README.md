# Documentation

| Page | What it covers |
|---|---|
| [1. Data and stations](01-data.md) | IEM and GHCN sources, how 39 of 133 candidate stations were chosen, METAR cleaning, the missing-precipitation problem, data availability |
| [2. Defining hourly snowfall](02-target.md) | Building an hourly target from gauge and present-weather reports, validation against official daily snowfall, the season split |
| [3. Feature engineering](03-features.md) | The leakage-safe time window, 1,627 features, spatial aggregates, what the model actually uses (SHAP), and a validation-only ablation |
| [4. Model and training](04-model.md) | The two-stage hurdle model, season-grouped tuning, calibration, the decision threshold |
| [5. Test results](05-results.md) | Held-out results scored once, reliability, amounts, events, case studies, limitations |
| [6. Reproducing](06-reproduce.md) | Environment, pipeline, runtimes, regenerating figures |

**Short on time?** Read [feature engineering](03-features.md), then [results](05-results.md).

## Glossary

| Term | Meaning |
|---|---|
| ASOS / AWOS | Automated airport weather stations (US) |
| METAR | The coded hourly airport weather report. *Routine* reports come once an hour; *specials* are issued when conditions change |
| `p01i` | Precipitation in the last hour (inches of liquid) from the station's heated gauge |
| SWE | Snow water equivalent: the liquid depth of melted snow |
| SLR | Snow-to-liquid ratio, used to convert SWE to snow depth |
| Trace | Precipitation reported but too little to measure (< 0.01") |
| Lead *L* | How far ahead the forecast is. Inputs end at hour *t − L* |
| Persistence | Baseline forecast: "whatever happened at *t − L* happens at *t*" |
| Climatology | Baseline forecast: the training-season snow rate for that calendar month |
| POD / FAR | Probability of detection (hit rate) / false-alarm ratio |
| CSI | Critical success index = hits / (hits + misses + false alarms); ignores correct "no snow" |
| Bias | Frequency bias = forecast yes / observed yes (1 is unbiased) |
| Brier / BSS | Mean squared error of probabilities / its skill relative to climatology (1 is perfect, 0 is no better) |
| ROC-AUC / PR-AUC | Area under the ROC / precision–recall curve. PR-AUC is the stricter one for rare events |
| Isotonic calibration | A monotone mapping that makes predicted probabilities match observed frequencies |
| τ | Probability threshold for turning P(snow) into a yes/no forecast |
| Hurdle model | Separate models for "does it happen?" and "how much, if it does?" |
| Tweedie | A loss for non-negative, skewed amounts |
| SHAP | Per-prediction attribution of the model output to each feature |

[← Project README](../README.md)
