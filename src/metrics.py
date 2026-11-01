"""Verification metrics for hourly snow forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def contingency(obs, pred) -> dict:
    o = np.asarray(obs).astype(bool)
    f = np.asarray(pred).astype(bool)
    hits = int(np.sum(o & f))
    misses = int(np.sum(o & ~f))
    fa = int(np.sum(~o & f))
    cn = int(np.sum(~o & ~f))
    return {"hits": hits, "misses": misses, "false_alarms": fa, "correct_negatives": cn}


def scores(table: dict) -> dict:
    h, m, fa = table["hits"], table["misses"], table["false_alarms"]
    nan = float("nan")
    return {
        "pod": h / (h + m) if h + m else nan,
        "far": fa / (h + fa) if h + fa else nan,
        "csi": h / (h + m + fa) if h + m + fa else nan,
        "bias": (h + fa) / (h + m) if h + m else nan,
    }


def csi(obs, pred) -> float:
    return scores(contingency(obs, pred))["csi"]


def brier(p, obs) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(obs, float)) ** 2))


def brier_skill(p, obs, p_ref) -> float:
    """BSS = 1 - BS / BS_ref."""
    ref = brier(p_ref, obs)
    return float(1.0 - brier(p, obs) / ref) if ref > 0 else float("nan")


def monthly_climatology(months, obs) -> dict[int, float]:
    """Event rate by calendar month (fit on train seasons)."""
    s = pd.Series(np.asarray(obs, float)).groupby(np.asarray(months)).mean()
    return {int(k): float(v) for k, v in s.items()}


def apply_climatology(months, clim: dict) -> np.ndarray:
    overall = float(np.mean(list(clim.values())))
    return np.array([clim.get(int(m), overall) for m in np.asarray(months)], dtype=float)


def roc_auc(p, obs) -> float:
    o = np.asarray(obs)
    return float(roc_auc_score(o, p)) if 0 < o.sum() < len(o) else float("nan")


def pr_auc(p, obs) -> float:
    o = np.asarray(obs)
    return float(average_precision_score(o, p)) if o.sum() > 0 else float("nan")


def mae(pred, obs) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(obs, float))))


def rmse(pred, obs) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, float) - np.asarray(obs, float)) ** 2)))


def forecast_summary(p, occ_pred, amount, obs_occ, obs_amount, p_clim) -> dict:
    """Probability, categorical and amount scores for one forecast system."""
    out = {"brier": brier(p, obs_occ), "bss_clim": brier_skill(p, obs_occ, p_clim),
           "roc_auc": roc_auc(p, obs_occ), "pr_auc": pr_auc(p, obs_occ)}
    out.update(scores(contingency(obs_occ, occ_pred)))
    out["mae"] = mae(amount, obs_amount)
    out["rmse"] = rmse(amount, obs_amount)
    snow = np.asarray(obs_occ).astype(bool)
    out["mae_snow_hours"] = mae(np.asarray(amount)[snow], np.asarray(obs_amount)[snow]) if snow.any() else float("nan")
    return out
