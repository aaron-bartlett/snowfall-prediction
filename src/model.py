"""Two-stage LightGBM hurdle model: P(snow) classifier x E[SWE | snow] regressor."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import load_config
from src.target import slr

STAGE1_FILE = "stage1_classifier.txt"
STAGE2_FILE = "stage2_regressor.txt"
CALIB_FILE = "calibrator.json"
META_FILE = "metadata.json"
ARTIFACTS = (STAGE1_FILE, STAGE2_FILE, CALIB_FILE, META_FILE)
VARIANTS = ("tweedie", "log1p_l2")


class HurdleModel:
    def __init__(self, stage1: lgb.Booster, stage2: lgb.Booster, variant: str, features: list[str],
                 calib_x, calib_y, tau: float, metadata: dict | None = None,
                 tmpf_feature: str = "DTW_tmpf_lag1"):
        if variant not in VARIANTS:
            raise ValueError(f"unknown stage 2 variant {variant}")
        self.stage1, self.stage2, self.variant = stage1, stage2, variant
        self.features = list(features)
        self.calib_x = np.asarray(calib_x, dtype=float)
        self.calib_y = np.asarray(calib_y, dtype=float)
        self.tau = float(tau)
        self.metadata = dict(metadata or {})
        self.tmpf_feature = tmpf_feature

    # -- prediction ---------------------------------------------------------------------------
    def matrix(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise KeyError(f"{len(missing)} model features missing from input, e.g. {missing[:5]}")
        return df[self.features].to_numpy(dtype=np.float32)

    def calibrate(self, p_raw) -> np.ndarray:
        """Isotonic mapping (piecewise linear between thresholds, clipped outside)."""
        return np.clip(np.interp(np.asarray(p_raw, float), self.calib_x, self.calib_y), 0.0, 1.0)

    def amount(self, X: np.ndarray) -> np.ndarray:
        raw = self.stage2.predict(X)
        if self.variant == "log1p_l2":
            raw = np.expm1(raw)
        return np.clip(raw, 0.0, None)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.matrix(df)
        p_raw = self.stage1.predict(X)
        p_cal = self.calibrate(p_raw)
        amount_cond = self.amount(X)
        swe_expected = p_cal * amount_cond
        cfg = load_config()
        t = df[self.tmpf_feature].to_numpy(dtype=float) if self.tmpf_feature in df else np.full(len(df), np.nan)
        return pd.DataFrame({
            "p_raw": p_raw, "p_cal": p_cal, "amount_cond": amount_cond, "swe_expected": swe_expected,
            "swe_det": np.where(p_cal >= self.tau, amount_cond, 0.0),
            "depth": swe_expected * slr(t, cfg.slr["table"], cfg.slr["fallback"]),
        }, index=df.index)

    # -- persistence --------------------------------------------------------------------------
    def save(self, directory) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.stage1.save_model(str(d / STAGE1_FILE))
        self.stage2.save_model(str(d / STAGE2_FILE))
        (d / CALIB_FILE).write_text(json.dumps(
            {"type": "isotonic", "out_of_bounds": "clip",
             "x_thresholds": self.calib_x.tolist(), "y_thresholds": self.calib_y.tolist()}) + "\n")
        meta = dict(self.metadata)
        meta.update({"features": self.features, "stage2_variant": self.variant, "tau": self.tau,
                     "tmpf_feature": self.tmpf_feature})
        (d / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")

    @classmethod
    def load(cls, directory) -> "HurdleModel":
        d = Path(directory)
        meta = json.loads((d / META_FILE).read_text())
        cal = json.loads((d / CALIB_FILE).read_text())
        return cls(lgb.Booster(model_file=str(d / STAGE1_FILE)), lgb.Booster(model_file=str(d / STAGE2_FILE)),
                   meta["stage2_variant"], meta["features"], cal["x_thresholds"], cal["y_thresholds"],
                   meta["tau"], meta, meta.get("tmpf_feature", "DTW_tmpf_lag1"))
