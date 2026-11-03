# Derived summary tables

Small tables computed from local pipeline outputs by `python -m src.viz --summaries`
([`src/viz/summaries.py`](../../src/viz/summaries.py)). They let every docs figure be redrawn from a
clean clone with no `data/`. They are aggregates, not datasets.

| File | Computed from | Used by |
|---|---|---|
| `availability_kdtw.csv` | `hourly.parquet`, KDTW: share of target hours with a routine ob / raw `p01i` / usable `p01i`, by season × month | F02 |
| `availability_stations.csv` | `hourly.parquet`: share of target hours with an ob, by station × season | F03 |
| `target_hist.csv`, `target_monthly.csv` | `target.parquet`: hours by intensity class; snow-hour rate by month, per split | F05 |
| `ghcn_daily.csv` | `target.parquet` + GHCN-Daily via `src.target.daily_compare`: one row per compared LST day | F06–F08 |
| `example_hour.csv` | `hourly.parquet`, 40 stations at the hour of the largest test storm with the most stations reporting snow, plus the plane-fit pressure gradient and vector-mean wind | F11, F12 |
| `shap_importance_L{1,3}.csv` | SHAP TreeExplainer on the frozen Stage 1 model, the same 5,000 random test hours as `src.evaluate` (seed 42): mean \|SHAP\| per feature, with family / station / variable / lag | F13–F15 |
| `shap_dependence_L{1,3}.parquet` | The same SHAP run: feature value and SHAP for 4 features per lead | F16 |

SHAP uses no labels and does not re-score the test seasons.
