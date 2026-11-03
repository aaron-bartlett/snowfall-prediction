"""Validation-only feature-family ablation for the Stage 1 (occurrence) classifier.

Usage: python -m src.ablation --lead all|1|3
Output: reports/ablation_val.json, reports/ablation_val.md

Stage 1 hyperparameters are frozen at the shipped model's best values (models/lead_{L}h/metadata.json).
For each nested feature set: 5-fold season GroupKFold CV on the train seasons with early stopping
(same harness as src.train), refit on all train seasons, score raw probabilities on the validation
seasons. Test seasons are never loaded.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.config import ROOT, load_config
from src.features import feature_list_path, features_path
from src.metrics import brier, csi, pr_auc, roc_auc
from src.train import TAU_GRID, cv_run, fit_final, logloss, make_dataset, model_dir, season_folds

REPORTS = ROOT / "reports"
JSON_PATH = REPORTS / "ablation_val.json"
MD_PATH = REPORTS / "ablation_val.md"
SETS = {
    "A: KDTW only": ["dtw", "time"],
    "C: KDTW + spatial aggregates": ["dtw", "time", "agg"],
    "B: KDTW + neighbor raw lags": ["dtw", "time", "neighbor"],
    "D: full": ["dtw", "time", "agg", "neighbor"],
}


def family(name: str, target: str = "DTW") -> str:
    if name.startswith(f"{target}_"):
        return "dtw"
    if name.startswith("agg_"):
        return "agg"
    if name.startswith(("hod_", "doy_")):
        return "time"
    return "neighbor"


def feature_sets(features: list[str]) -> dict[str, list[str]]:
    fam = {f: family(f) for f in features}
    return {name: [f for f in features if fam[f] in fams] for name, fams in SETS.items()}


def run_lead(lead: int, cfg) -> dict:
    features = json.loads(feature_list_path(lead).read_text())
    meta = json.loads((model_dir(lead) / "metadata.json").read_text())
    params = dict(meta["stage1"]["params"])
    tcfg = dict(cfg.train)
    df = pd.read_parquet(features_path(lead), columns=["season", "split", "snow_occ"] + features,
                         filters=[("split", "in", ["train", "val"])])
    if set(df.split) - {"train", "val"}:
        raise RuntimeError("test-season rows loaded")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    y, yv = tr.snow_occ.to_numpy(float), va.snow_occ.to_numpy(float)
    folds = season_folds(tr.season.to_numpy(), tcfg["cv_folds"])
    out = {"lead": lead, "params_from": f"models/lead_{lead}h/metadata.json",
           "shipped_cv_logloss": meta["stage1"]["cv_logloss"], "sets": []}
    for name, cols in feature_sets(features).items():
        t0 = time.time()
        X = tr[cols].to_numpy(np.float32)
        ds = make_dataset(X, y)
        cv, iters = cv_run(params, ds, X, y, folds, logloss, tcfg)
        n_est = int(np.median(iters))
        b = fit_final(params, ds, n_est)
        p = b.predict(va[cols].to_numpy(np.float32))
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, yv)
        pc = iso.predict(p)
        csis = [csi(yv, pc >= t) for t in TAU_GRID]
        row = {"set": name, "n_features": len(cols), "cv_logloss": cv, "n_estimators": n_est,
               "val_logloss_raw": logloss(yv, p), "val_brier_raw": brier(p, yv),
               "val_roc_auc": roc_auc(p, yv), "val_pr_auc": pr_auc(p, yv),
               "val_csi_best_tau": float(np.nanmax(csis)),
               "val_best_tau": float(TAU_GRID[int(np.nanargmax(csis))]), "seconds": round(time.time() - t0)}
        out["sets"].append(row)
        print(f"lead {lead} {name}: {len(cols)} feats, CV {cv:.5f}, val PR-AUC {row['val_pr_auc']:.4f} "
              f"({row['seconds']}s)", flush=True)
        del X, ds, b
    return out


def write_md(res: dict) -> None:
    L = ["# Feature-family ablation (validation seasons only)", "",
         "Stage 1 occurrence classifier refit on nested feature sets with hyperparameters frozen at the "
         "shipped model's values. CV = 5-fold season GroupKFold on the 15 train seasons; val = the 3 "
         "validation seasons (raw, uncalibrated probabilities unless noted). Test seasons were not loaded.", "",
         "Caveats: hyperparameters were tuned for the full set (slightly favours D); only 3 validation "
         "seasons, so differences of a few thousandths are within noise; `val_csi_best_tau` fits isotonic "
         "calibration and τ on the same validation seasons, so it is optimistic (equally for every set).", ""]
    keys = ["set", "n_features", "cv_logloss", "n_estimators", "val_logloss_raw", "val_brier_raw",
            "val_roc_auc", "val_pr_auc", "val_csi_best_tau"]
    for lead, r in res["leads"].items():
        L += [f"## Lead {lead} h", "", f"Shipped model CV log-loss: {r['shipped_cv_logloss']:.5f}", "",
              "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        for s in r["sets"]:
            L.append("| " + " | ".join(f"{s[k]:.4f}" if isinstance(s[k], float) else f"{s[k]:,}"
                                       if isinstance(s[k], int) else str(s[k]) for k in keys) + " |")
        L.append("")
    MD_PATH.write_text("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lead", default="all", choices=["all", "1", "3"])
    args = ap.parse_args(argv)
    try:
        cfg = load_config()
        leads = list(cfg.lead_hours) if args.lead == "all" else [int(args.lead)]
        res = json.loads(JSON_PATH.read_text()) if JSON_PATH.exists() else {"leads": {}}
        for lead in leads:
            res["leads"][str(lead)] = run_lead(lead, cfg)
            JSON_PATH.write_text(json.dumps(res, indent=2) + "\n")
        write_md(res)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAILED ablation: {e}")
        return 1
    print("DONE ablation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
