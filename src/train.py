"""Tune, fit, calibrate and save the two-stage hurdle models.

Usage: python -m src.train --lead 1|3|all [--smoke]
Output: models/lead_{L}h/{stage1_classifier.txt, stage2_regressor.txt, calibrator.json, metadata.json,
        optuna.db}, reports/val_predictions_L{L}.parquet, reports/val_metrics.md

Tuning uses 5-fold GroupKFold by season within the train seasons only. The validation seasons are
used only to pick the Stage 2 variant, fit the isotonic calibrator and choose the threshold tau.
Test seasons are never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_tweedie_deviance
from sklearn.model_selection import GroupKFold

from src.config import ROOT, load_config
from src.features import TARGET_COLS, feature_list_path, features_path
from src.metrics import apply_climatology, csi, forecast_summary, mae, monthly_climatology
from src.model import ARTIFACTS, HurdleModel, VARIANTS
from src.splits import load_splits

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
NUM_THREADS = 4
TAU_GRID = np.round(np.arange(0.05, 0.9501, 0.01), 2)
TWEEDIE_EVAL_POWER = 1.5     # fixed power for the CV deviance so trials with different powers compare fairly
SMOKE_ROWS = 20_000


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def model_dir(lead: int, root=MODELS) -> Path:
    return Path(root) / f"lead_{lead}h"


def season_folds(groups, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    g = np.asarray(groups)
    return [(np.sort(tr), np.sort(va)) for tr, va in GroupKFold(n_splits=n_folds).split(g, groups=g)]


def base_params(seed: int) -> dict:
    return {"verbosity": -1, "num_threads": NUM_THREADS, "seed": seed, "deterministic": True,
            "force_row_wise": True, "feature_pre_filter": False}


def suggest_tree_params(trial: optuna.Trial) -> dict:
    depth = trial.suggest_int("max_depth", 3, 12)   # 3 encodes "no limit" (-1); 4-12 are real depths
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": -1 if depth == 3 else depth,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
    }


def make_dataset(X: np.ndarray, y: np.ndarray) -> lgb.Dataset:
    return lgb.Dataset(X, label=y, params={"feature_pre_filter": False, "verbosity": -1},
                       free_raw_data=False).construct()


def cv_run(params, ds, X, y_eval, folds, score_fn, tcfg, trial=None):
    """Fit one model per fold with early stopping on the held-out fold. Returns (mean score, best iters)."""
    scores, iters = [], []
    for i, (tr, va) in enumerate(folds):
        b = lgb.train(params, ds.subset(tr), num_boost_round=tcfg["max_trees"],
                      valid_sets=[ds.subset(va)],
                      callbacks=[lgb.early_stopping(tcfg["early_stopping_rounds"], verbose=False)])
        pred = b.predict(X[va], num_iteration=b.best_iteration)
        scores.append(score_fn(y_eval[va], pred))
        iters.append(int(b.best_iteration or b.current_iteration()))
        if trial is not None:
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(scores)), iters


def logloss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def run_study(name, db: Path, objective, n_trials: int, timeout_min: float, seed: int) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(storage=f"sqlite:///{db}", study_name=name, direction="minimize",
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1))
    done = [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE,
                                                   optuna.trial.TrialState.PRUNED)]
    remaining = n_trials - len(done)
    if remaining > 0:
        def cb(st, t):
            v = "pruned" if t.value is None else f"{t.value:.5f}"
            best = st.best_value if any(x.value is not None for x in st.trials) else float("nan")
            log(f"{name} trial {t.number}: {v} (best {best:.5f})")
        study.optimize(objective, n_trials=remaining, timeout=timeout_min * 60, callbacks=[cb])
    else:
        log(f"{name}: {len(done)} trials already done, resuming from storage")
    return study


def tune_stage1(X, y, groups, tcfg, db, n_trials, timeout_min):
    ds = make_dataset(X, y)
    folds = season_folds(groups, tcfg["cv_folds"])

    def objective(trial):
        params = {**base_params(tcfg["seed"]), **suggest_tree_params(trial),
                  "objective": "binary", "metric": "binary_logloss",
                  "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0)}
        score, iters = cv_run(params, ds, X, y, folds, logloss, tcfg, trial)
        trial.set_user_attr("lgb_params", params)
        trial.set_user_attr("best_iters", iters)
        return score

    study = run_study("stage1", db, objective, n_trials, timeout_min, tcfg["seed"])
    return study, ds


def tune_stage2(X, y, groups, variant, tcfg, db, n_trials, timeout_min):
    label = y if variant == "tweedie" else np.log1p(y)
    ds = make_dataset(X, label)
    folds = season_folds(groups, tcfg["cv_folds"])
    if variant == "tweedie":
        def score(yt, pred):
            return float(mean_tweedie_deviance(yt, np.clip(pred, 1e-9, None), power=TWEEDIE_EVAL_POWER))
    else:
        def score(yt, pred):
            return mae(np.clip(np.expm1(pred), 0, None), yt)

    def objective(trial):
        params = {**base_params(tcfg["seed"]), **suggest_tree_params(trial)}
        if variant == "tweedie":
            params.update(objective="tweedie", metric="tweedie",
                          tweedie_variance_power=trial.suggest_float("tweedie_variance_power", 1.1, 1.9))
        else:
            params.update(objective="regression", metric="l2")
        s, iters = cv_run(params, ds, X, y, folds, score, tcfg, trial)
        trial.set_user_attr("lgb_params", params)
        trial.set_user_attr("best_iters", iters)
        return s

    study = run_study(f"stage2_{variant}", db, objective, n_trials, timeout_min, tcfg["seed"])
    return study, ds


def best_of(study: optuna.Study) -> tuple[dict, int, float]:
    t = study.best_trial
    return dict(t.user_attrs["lgb_params"]), int(np.median(t.user_attrs["best_iters"])), float(t.value)


def fit_final(params: dict, ds: lgb.Dataset, n_estimators: int) -> lgb.Booster:
    return lgb.train(params, ds, num_boost_round=max(1, n_estimators))


def choose_tau(p_cal, obs) -> float:
    vals = [csi(obs, p_cal >= t) for t in TAU_GRID]
    return float(TAU_GRID[int(np.nanargmax(vals))])


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def versions() -> dict:
    import sklearn
    return {"python": sys.version.split()[0], "lightgbm": lgb.__version__, "optuna": optuna.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__}


def validation_metrics(preds: pd.DataFrame, val: pd.DataFrame, tau: float, clim: dict, clim_swe: dict) -> dict:
    occ = val.snow_occ.to_numpy(float)
    swe = val.y_swe.to_numpy(float)
    months = val.hour_ending.dt.month.to_numpy()
    p_clim = apply_climatology(months, clim)
    pers_occ = val.persist_occ.fillna(0.0).to_numpy(float)
    pers_swe = val.persist_swe.fillna(0.0).to_numpy(float)
    out = {
        "model": forecast_summary(preds.p_cal, preds.p_cal >= tau, preds.swe_expected, occ, swe, p_clim),
        "persistence": forecast_summary(pers_occ, pers_occ >= 0.5, pers_swe, occ, swe, p_clim),
        "climatology": forecast_summary(p_clim, p_clim >= 0.5, apply_climatology(months, clim_swe), occ, swe,
                                        p_clim),
    }
    out["model_raw_prob"] = {"brier": float(np.mean((preds.p_raw - occ) ** 2))}
    return out


def run_lead(df: pd.DataFrame, features: list[str], lead: int, cfg, splits: dict, out_models: Path,
             out_reports: Path, smoke: bool = False, feature_file: Path | None = None) -> dict:
    tcfg = dict(cfg.train)
    n1, n2 = (2, 2) if smoke else (tcfg["n_trials_stage1"], tcfg["n_trials_stage2"])
    to1, to2 = (5, 5) if smoke else (tcfg["timeout_min_stage1"], tcfg["timeout_min_stage2"])
    mdir = model_dir(lead, out_models)
    mdir.mkdir(parents=True, exist_ok=True)
    db = mdir / "optuna.db"

    train = df[df.split == "train"].reset_index(drop=True)
    val = df[df.split == "val"].reset_index(drop=True)
    if smoke and len(train) > SMOKE_ROWS:
        train = train.sample(SMOKE_ROWS, random_state=tcfg["seed"]).sort_values("hour_ending").reset_index(drop=True)
    if set(df.split) - {"train", "val"}:
        raise RuntimeError("test-season rows passed to training")
    X = train[features].to_numpy(np.float32)
    y1 = train.snow_occ.to_numpy(float)
    groups = train.season.to_numpy()
    log(f"lead {lead}: {len(train):,} train rows ({int(y1.sum()):,} snow), {len(val):,} val rows, "
        f"{len(features)} features")

    t0 = time.time()
    st1, ds1 = tune_stage1(X, y1, groups, tcfg, db, n1, to1)
    p1, n_est1, cv1 = best_of(st1)
    log(f"lead {lead} stage1 best CV logloss {cv1:.5f}, n_estimators {n_est1} ({time.time() - t0:.0f}s)")
    stage1 = fit_final(p1, ds1, n_est1)
    del ds1

    snow = y1 == 1
    Xs, ys, gs = X[snow], train.y_swe.to_numpy(float)[snow], groups[snow]
    stage2 = {}
    for variant in VARIANTS:
        st2, ds2 = tune_stage2(Xs, ys, gs, variant, tcfg, db, n2, to2)
        p2, n_est2, cv2 = best_of(st2)
        log(f"lead {lead} stage2 {variant} best CV {cv2:.5f}, n_estimators {n_est2}")
        stage2[variant] = {"booster": fit_final(p2, ds2, n_est2), "params": p2, "n_estimators": n_est2,
                           "cv_score": cv2, "n_trials": len(st2.trials)}

    # validation seasons: pick variant, calibrate, choose tau
    Xv = val[features].to_numpy(np.float32)
    occ_v = val.snow_occ.to_numpy(float)
    snow_v = occ_v == 1
    val_mae = {}
    for variant, s in stage2.items():
        amt = s["booster"].predict(Xv[snow_v])
        if variant == "log1p_l2":
            amt = np.expm1(amt)
        val_mae[variant] = mae(np.clip(amt, 0, None), val.y_swe.to_numpy(float)[snow_v])
    variant = min(val_mae, key=val_mae.get)
    p_raw_v = stage1.predict(Xv)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_raw_v, occ_v)
    model = HurdleModel(stage1, stage2[variant]["booster"], variant, features, iso.X_thresholds_,
                        iso.y_thresholds_, tau=0.5)
    tau = choose_tau(model.calibrate(p_raw_v), occ_v)
    model.tau = tau
    preds = model.predict(val)

    clim = monthly_climatology(train.hour_ending.dt.month, train.snow_occ)
    clim_swe = monthly_climatology(train.hour_ending.dt.month, train.y_swe)
    metrics = validation_metrics(preds, val, tau, clim, clim_swe)
    log(f"lead {lead}: stage2 variant {variant} (val MAE {val_mae}), tau {tau:.2f}")
    m, p = metrics["model"], metrics["persistence"]
    log(f"lead {lead} val: model Brier {m['brier']:.5f} BSS {m['bss_clim']:.3f} CSI {m['csi']:.3f} "
        f"RMSE {m['rmse']:.5f} | persistence Brier {p['brier']:.5f} CSI {p['csi']:.3f} RMSE {p['rmse']:.5f}")

    model.metadata = {
        "lead_hours": lead,
        "stage1": {"params": p1, "n_estimators": n_est1, "cv_logloss": cv1, "n_trials": len(st1.trials)},
        "stage2": {v: {k: s[k] for k in ("params", "n_estimators", "cv_score", "n_trials")}
                   for v, s in stage2.items()},
        "stage2_val_mae_snow_hours": val_mae,
        "train_seasons": sorted(splits["train"]), "val_seasons": sorted(splits["val"]),
        "n_train_rows": int(len(train)), "n_val_rows": int(len(val)),
        "n_train_snow_rows": int(snow.sum()), "smoke": smoke,
        "features_parquet": str(feature_file.relative_to(ROOT)) if feature_file else None,
        "features_sha256": file_sha256(feature_file) if feature_file else None,
        "climatology_snow_rate_by_month": clim,
        "versions": versions(), "num_threads": NUM_THREADS,
        "git_commit": git_commit(), "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tweedie_cv_eval_power": TWEEDIE_EVAL_POWER,
        "val_metrics": metrics,
    }
    model.save(mdir)
    out_reports.mkdir(parents=True, exist_ok=True)
    vp = pd.concat([val[["hour_ending", "season", "snow_occ", "y_swe", "y_depth", "persist_occ", "persist_swe"]],
                    preds], axis=1)
    path = out_reports / f"val_predictions_L{lead}.parquet"
    tmp = path.with_suffix(".parquet.part")
    vp.to_parquet(tmp, index=False)
    tmp.rename(path)
    missing = [a for a in ARTIFACTS if not (mdir / a).exists()]
    if missing:
        raise RuntimeError(f"artifacts not written: {missing}")
    return model.metadata


def write_val_report(out_models: Path, out_reports: Path, leads) -> None:
    rows = ["# Validation metrics (validation seasons only)", "",
            "Calibration and the decision threshold τ were fit on these same seasons, so these numbers are "
            "optimistic; the held-out test seasons are scored once in Phase 7.", ""]
    keys = ["brier", "bss_clim", "roc_auc", "pr_auc", "pod", "far", "csi", "bias", "mae", "rmse", "mae_snow_hours"]
    for lead in leads:
        f = model_dir(lead, out_models) / "metadata.json"
        if not f.exists():
            continue
        meta = json.loads(f.read_text())
        vm = meta["val_metrics"]
        rows += [f"## Lead {lead} h", "",
                 f"- Val seasons: {', '.join(meta['val_seasons'])}; rows {meta['n_val_rows']:,}",
                 f"- Stage 2 variant: {meta['stage2_variant']} (val MAE on snow hours: "
                 + ", ".join(f"{k} {v:.4f}" for k, v in meta["stage2_val_mae_snow_hours"].items()) + ")",
                 f"- τ = {meta['tau']:.2f}; Stage 1 n_estimators {meta['stage1']['n_estimators']}, "
                 f"CV log-loss {meta['stage1']['cv_logloss']:.5f} ({meta['stage1']['n_trials']} trials)", "",
                 "| system | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
        for system in ("model", "persistence", "climatology"):
            rows.append(f"| {system} | " + " | ".join(f"{vm[system][k]:.4f}" for k in keys) + " |")
        rows.append("")
    (out_reports / "val_metrics.md").write_text("\n".join(rows) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lead", default="all", choices=["all", "1", "3"])
    ap.add_argument("--smoke", action="store_true", help="2 trials per stage on 20,000 sampled train rows")
    args = ap.parse_args(argv)
    try:
        cfg = load_config()
        splits = load_splits()
        leads = list(cfg.lead_hours) if args.lead == "all" else [int(args.lead)]
        out_models = MODELS / "smoke" if args.smoke else MODELS
        out_reports = REPORTS / "smoke" if args.smoke else REPORTS
        for lead in leads:
            features = json.loads(feature_list_path(lead).read_text())
            fpath = features_path(lead)
            df = pd.read_parquet(fpath, columns=TARGET_COLS + features,
                                 filters=[("split", "in", ["train", "val"])])
            run_lead(df, features, lead, cfg, splits, out_models, out_reports, smoke=args.smoke,
                     feature_file=fpath)
            del df
            write_val_report(out_models, out_reports, cfg.lead_hours)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAILED train: {e}")
        return 1
    print("DONE train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
