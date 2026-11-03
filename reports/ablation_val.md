# Feature-family ablation (validation seasons only)

Stage 1 occurrence classifier refit on nested feature sets with hyperparameters frozen at the shipped model's values. CV = 5-fold season GroupKFold on the 15 train seasons; val = the 3 validation seasons (raw, uncalibrated probabilities unless noted). Test seasons were not loaded.

Caveats: hyperparameters were tuned for the full set (slightly favours D); only 3 validation seasons, so differences of a few thousandths are within noise; `val_csi_best_tau` fits isotonic calibration and τ on the same validation seasons, so it is optimistic (equally for every set).

## Lead 1 h

Shipped model CV log-loss: 0.09800

| set | n_features | cv_logloss | n_estimators | val_logloss_raw | val_brier_raw | val_roc_auc | val_pr_auc | val_csi_best_tau |
|---|---|---|---|---|---|---|---|---|
| A: KDTW only | 267 | 0.1215 | 792 | 0.1228 | 0.0356 | 0.9716 | 0.8214 | 0.6290 |
| C: KDTW + spatial aggregates | 379 | 0.1096 | 379 | 0.0999 | 0.0298 | 0.9833 | 0.8823 | 0.6799 |
| B: KDTW + neighbor raw lags | 1,515 | 0.0990 | 575 | 0.0961 | 0.0286 | 0.9833 | 0.8861 | 0.6747 |
| D: full | 1,627 | 0.0980 | 539 | 0.0973 | 0.0292 | 0.9834 | 0.8842 | 0.6665 |

## Lead 3 h

Shipped model CV log-loss: 0.12669

| set | n_features | cv_logloss | n_estimators | val_logloss_raw | val_brier_raw | val_roc_auc | val_pr_auc | val_csi_best_tau |
|---|---|---|---|---|---|---|---|---|
| A: KDTW only | 267 | 0.1550 | 175 | 0.1664 | 0.0504 | 0.9402 | 0.6501 | 0.4630 |
| C: KDTW + spatial aggregates | 379 | 0.1293 | 209 | 0.1378 | 0.0419 | 0.9625 | 0.7510 | 0.5251 |
| B: KDTW + neighbor raw lags | 1,515 | 0.1278 | 270 | 0.1370 | 0.0414 | 0.9621 | 0.7566 | 0.5239 |
| D: full | 1,627 | 0.1267 | 272 | 0.1350 | 0.0410 | 0.9639 | 0.7584 | 0.5303 |
