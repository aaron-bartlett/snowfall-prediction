# KDTW Hourly Snowfall Model

Nowcasts of hourly snowfall at Detroit Metro Airport (KDTW) **1 h and 3 h ahead**, from lagged
surface observations at KDTW and 39 surrounding ASOS stations (≤ 300 km). For each lead there is one
two-stage LightGBM *hurdle* model:

- **Stage 1** classifier: P(falling snow, trace or more), isotonic-calibrated.
- **Stage 2** regressor (Tweedie): E[SWE | snow], trained on all snow hours with trace = 0.
- Outputs: `P(snow)`, expected SWE = P × amount, deterministic SWE (amount if P ≥ τ, else 0), and
  depth = expected SWE × a temperature-dependent snow-to-liquid ratio.

Data: IEM ASOS/AWOS archive (2005–06 to 2025–26 cold seasons, Oct–Apr), with GHCN-Daily `SNOW` at
`USW00094847` used as a cross-check. Whole seasons are randomly split 15 train / 3 val / 3 test
(`meta/splits.json`, seed 42). See [`project_plan.md`](project_plan.md) for the design,
[`execution_plan.md`](execution_plan.md) for the step-by-step procedure, and
[`reports/execution_log.md`](reports/execution_log.md) for what was actually run.

## Setup

```bash
conda env create -f environment.yml          # or: conda env create -f environment.lock.yml (exact versions)
conda activate snowfall
python -m pytest                             # offline unit tests (< 1 min)
```

## Pipeline (run from the project root, in order)

| Step | Command | Output |
|---|---|---|
| 0. Season split (once, never regenerated) | `python -m src.splits` | `meta/splits.json` |
| 1. Station selection | `python -m src.stations --map` | `meta/stations.csv`, `station_map.png` |
| 2. Download IEM + GHCN | `python -m src.download --all` | `data/raw/` |
| 3. Clean / QC to hourly | `python -m src.clean` | `data/processed/hourly.parquet` |
| 4. Target + GHCN check | `python -m src.target` | `data/processed/target.parquet`, `reports/target_validation.md` |
| 5. Feature matrices | `python -m src.features --lead all` | `data/processed/features_L{1,3}.parquet` |
| 6. Tune, fit, calibrate | `python -m src.train --lead all` (≈ 3.5 h) | `models/lead_{1,3}h/` |
| 7. Test evaluation (once) | `python -m src.evaluate --final` | `reports/test_report.md` |

Each script prints `DONE <script>` on success and `FAILED <script>: <reason>` (exit 1) on failure.
Per-phase data checks: `python -m pytest -m data tests/data_checks/test_phaseN.py`.

`src.evaluate --final` refuses to run again once `reports/TEST_EVALUATED.lock` exists (use `--force`
only if you deliberately accept re-scoring the test seasons).

## Model weights

`models/lead_1h/` and `models/lead_3h/` each contain:

- `stage1_classifier.txt`, `stage2_regressor.txt`: LightGBM native model files (the weights)
- `calibrator.json`: isotonic X/Y thresholds for Stage 1
- `metadata.json`: ordered feature list, parameters, Stage 2 variant, τ, seasons, versions, validation metrics

Load them with `src.model.HurdleModel.load("models/lead_1h")`; `predict(df)` selects the features by
name in the stored order and raises if any is missing.

## Predicting

```bash
python -m src.predict --lead 1 --time 2018-01-12T15:00Z
```

This reads that hour's feature row from `data/processed/features_L1.parquet` and prints `p_snow`,
`swe_expected_in`, `swe_deterministic_in` and `depth_expected_in`. The time is the target hour-ending
(UTC). The 1 h model uses observations up to t − 1 h, the 3 h model up to t − 3 h.

## Test results

Held-out test seasons **2015-16, 2020-21, 2021-22** (15,271 hours, 1,280 snow hours), scored once.
Full tables, stratifications and figures: [`reports/test_report.md`](reports/test_report.md).

| lead | system | Brier | BSS vs clim | ROC-AUC | PR-AUC | POD | FAR | CSI | RMSE SWE (in) |
|---|---|---|---|---|---|---|---|---|---|
| 1 h | **model** | **0.0281** | **0.613** | **0.982** | **0.834** | 0.757 | 0.220 | **0.624** | **0.0033** |
| 1 h | persistence | 0.0432 | 0.406 | 0.859 | 0.573 | 0.742 | 0.257 | 0.590 | 0.0043 |
| 3 h | **model** | **0.0392** | **0.460** | **0.963** | **0.705** | 0.676 | 0.359 | **0.490** | **0.0041** |
| 3 h | persistence | 0.0744 | −0.024 | 0.758 | 0.346 | 0.556 | 0.444 | 0.385 | 0.0062 |
| — | climatology | 0.0727 | 0 | 0.734 | 0.164 | 0 | — | 0 | 0.0053 |

- Test skill is a little below validation (val BSS 0.666 / 0.503, CSI 0.666 / 0.530), as expected
  since τ and the calibrator were fit on the validation seasons. The model still beats persistence on
  every headline score at both leads, and the margin is much larger at 3 h.
- Calibrated probabilities are close to reliable on test (see `reports/figures/reliability_L*.png`).
- Weak spots: amounts are underpredicted in heavy hours (≥ 0.06 in/h: mean forecast 0.050 vs 0.070
  observed at 1 h, 0.030 at 3 h), so event totals come out 12% (1 h) and 30% (3 h) low. On the
  event-total error metric, persistence, which simply shifts the observed series, does better than the
  model. At 0.01 in SWE the 1 h model's CSI (0.445) is slightly below persistence (0.459).
- The derived-depth target itself captures about 82% of GHCN `SNOW` on test days (gauge undercatch).
  The model's daily depth captures 77% (1 h) and 74% (3 h).
