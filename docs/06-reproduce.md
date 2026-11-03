# 6. Reproducing

The repo ships code, configuration, the frozen station list and season split, trained weights,
every generated report, and the stored validation and test predictions. **It does not ship data.**
Raw and processed data are rebuilt by the pipeline from public sources.

## Environment

```bash
conda env create -f environment.yml          # Python 3.11, LightGBM, pandas, Optuna, SHAP, cartopy …
conda activate snowfall
python -m pytest                             # offline unit tests, under a minute
```

`environment.lock.yml` pins the exact versions used (lightgbm 4.7.0, pandas 3.1.0, numpy 2.4.6,
optuna 5.0.0, shap 0.51.0, scikit-learn 1.9.1). It was exported on macOS, so on Linux use
`environment.yml`.

## What works on a fresh clone (no data)

| Command | What it does |
|---|---|
| `python -m pytest` | unit tests, including leakage tests on synthetic data |
| `python -m src.viz` | redraws every figure in `docs/figures/` from committed files |
| `HurdleModel.load("models/lead_1h")` | loads the trained weights ([page 4](04-model.md)) |

## Full pipeline

The pipeline runs in eight **phases**. Each one is a module run from the project root. Each prints
`DONE <script>` on success or `FAILED <script>: <reason>` (with exit code 1) on failure.

| Phase | Command | Output | Time |
|---|---|---|---|
| 0. Season split (once, never regenerated) | `python -m src.splits` | `meta/splits.json` (committed) | instant |
| 1. Station selection | `python -m src.stations --map` | `meta/stations.csv` (committed) | ~15 min |
| 2. Download IEM + GHCN | `python -m src.download --all` | `data/raw/` (0.94 GB) | ~25 min |
| 3. Clean / QC to hourly | `python -m src.clean` | `data/processed/hourly.parquet` | ~1 min |
| 4. Target + GHCN check | `python -m src.target` | `target.parquet`, `reports/target_validation.md` | seconds |
| 5. Feature matrices | `python -m src.features --lead all` | `features_L{1,3}.parquet` (~124 MB each) | ~1–2 min |
| 6. Tune, fit, calibrate | `python -m src.train --lead all` | `models/lead_{1,3}h/` | ~3.5 h |
| 7. Test evaluation (once) | `python -m src.evaluate --final` | `reports/test_report.md` | ~1 min |

Phases 0–1 are optional because their outputs are committed and frozen. To reproduce the
published results, start at Phase 2. Phase 6 only needs re-running if you want to retrain.

**Per-phase data checks** run against your local outputs:
`python -m pytest -m data tests/data_checks/test_phaseN.py`. The unit tests in
`tests/unit/test_phaseN.py` follow the same numbering.

> [!WARNING]
> **The test lock.** `src.evaluate --final` refuses to run once `reports/TEST_EVALUATED.lock`
> exists. The test seasons were scored once and that result is the published one. `--force` exists,
> but using it means accepting a second look at the test set.

## Docs figures and the ablation

```bash
python -m src.viz --summaries     # recompute reports/summaries/* (needs data/processed + raw GHCN)
python -m src.viz                 # redraw docs/figures/*
python -m src.ablation --lead all # validation-only feature ablation, about 10 min per lead (needs features)
```

`reports/summaries/` holds small derived tables (under 1.5 MB in total) so that figures needing the
full dataset can still be redrawn without it.

## Reproducibility notes

- **Determinism.** LightGBM runs with `deterministic=True`, `force_row_wise=True`, 4 threads and
  seed 42. Optuna uses a seeded TPE sampler.
- **Feature files across machines.** Training ran in a cloud container (pandas 3.0.5 / numpy 2.5.3)
  and evaluation ran locally (pandas 3.1.0 / numpy 2.4.6). The rebuilt feature parquet differs
  byte for byte, so its sha256 no longer matches `metadata.json`. The *values* match: the saved
  models reproduce the stored validation predictions to within 1.1e-16.
- **Leakage tests.** These are an index-encoding test (every feature value encodes its source hour)
  and future-poisoning tests (overwrite everything after *t − L* and check the features don't
  change), run on synthetic data and on a real January 2014 slice
  ([`tests/unit/test_phase5.py`](../tests/unit/test_phase5.py)).

## Repository map

```
├── README.md                 landing page
├── docs/                     these pages + figures/
├── config.yaml               every tunable setting (stations, radius, seasons, lags, training budget)
├── meta/                     splits.json, stations.csv, station_candidates.csv, feature lists
├── src/                      pipeline modules (python -m src.<module>)
│   ├── splits, stations, download, clean, target, features, train, evaluate, predict
│   ├── model.py, metrics.py, geo.py, timeutil.py, config.py
│   ├── ablation.py           validation-only feature-family ablation
│   └── viz/                  docs figures and summary tables
├── models/lead_{1,3}h/       trained weights, calibrator, metadata
├── reports/                  generated reports, metrics JSON, frozen predictions, summaries/, figures/
└── tests/                    unit/ (offline) and data_checks/ (need pipeline outputs)
```

---
[← Results](05-results.md) · [Docs index](README.md)
