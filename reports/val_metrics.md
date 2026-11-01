# Validation metrics (validation seasons only)

Calibration and the decision threshold τ were fit on these same seasons, so these numbers are optimistic; the held-out test seasons are scored once in Phase 7.

## Lead 1 h

- Val seasons: 2017-18, 2019-20, 2025-26; rows 15,247
- Stage 2 variant: tweedie (val MAE on snow hours: tweedie 0.0045, log1p_l2 0.0048)
- τ = 0.39; Stage 1 n_estimators 539, CV log-loss 0.09800 (13 trials)

| system | brier | bss_clim | roc_auc | pr_auc | pod | far | csi | bias | mae | rmse | mae_snow_hours |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model | 0.0270 | 0.6663 | 0.9840 | 0.8778 | 0.7943 | 0.1945 | 0.6665 | 0.9861 | 0.0005 | 0.0044 | 0.0046 |
| persistence | 0.0411 | 0.4924 | 0.8793 | 0.6331 | 0.7812 | 0.2161 | 0.6427 | 0.9965 | 0.0005 | 0.0050 | 0.0050 |
| climatology | 0.0810 | 0.0000 | 0.7261 | 0.1700 | 0.0000 | nan | 0.0000 | 0.0000 | 0.0012 | 0.0063 | 0.0077 |

## Lead 3 h

- Val seasons: 2017-18, 2019-20, 2025-26; rows 15,247
- Stage 2 variant: tweedie (val MAE on snow hours: tweedie 0.0064, log1p_l2 0.0067)
- τ = 0.33; Stage 1 n_estimators 272, CV log-loss 0.12669 (16 trials)

| system | brier | bss_clim | roc_auc | pr_auc | pod | far | csi | bias | mae | rmse | mae_snow_hours |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model | 0.0402 | 0.5032 | 0.9647 | 0.7514 | 0.7334 | 0.3431 | 0.5303 | 1.1163 | 0.0007 | 0.0053 | 0.0063 |
| persistence | 0.0731 | 0.0981 | 0.7860 | 0.4131 | 0.6122 | 0.3853 | 0.4424 | 0.9958 | 0.0009 | 0.0070 | 0.0072 |
| climatology | 0.0810 | 0.0000 | 0.7261 | 0.1700 | 0.0000 | nan | 0.0000 | 0.0000 | 0.0012 | 0.0063 | 0.0077 |

