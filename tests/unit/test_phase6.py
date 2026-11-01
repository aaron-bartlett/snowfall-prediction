"""Phase 6: metrics, hurdle model, training loop (offline, synthetic data)."""
import dataclasses
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.metrics import (apply_climatology, brier, brier_skill, contingency, monthly_climatology,
                         scores)
from src.model import ARTIFACTS, HurdleModel
from src.train import base_params, choose_tau, run_lead, season_folds

FEATURES = ["DTW_tmpf_lag1"] + [f"f{i}" for i in range(7)]
TRAIN_SEASONS = [f"20{y:02d}-{y + 1:02d}" for y in range(5, 11)]
VAL_SEASONS = ["2011-12", "2012-13"]


def synthetic(n=3000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES))).astype(np.float32)
    X[:, 0] = rng.uniform(5, 40, n)
    p = 1 / (1 + np.exp(-(2.0 * X[:, 1] - 2.5)))
    occ = (rng.random(n) < p).astype(float)
    swe = np.where(occ == 1, np.round(np.clip(0.02 * np.exp(0.5 * X[:, 2]) - 0.01, 0, None), 2), 0.0)
    seasons = np.array(TRAIN_SEASONS + VAL_SEASONS)[np.arange(n) % 8]
    hours = pd.date_range("2006-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(X, columns=FEATURES)
    meta = pd.DataFrame({
        "hour_ending": hours, "season": seasons,
        "split": np.where(np.isin(seasons, VAL_SEASONS), "val", "train"),
        "snow_occ": occ, "y_swe": swe, "y_depth": swe * 10, "wx_mixed": 0.0,
        "persist_occ": np.roll(occ, 1), "persist_swe": np.roll(swe, 1)})
    return pd.concat([meta, df], axis=1)


# -- metrics ---------------------------------------------------------------------------------

def test_contingency_scores():
    obs = [1, 1, 1, 0, 0]
    pred = [1, 1, 0, 1, 0]
    t = contingency(obs, pred)
    assert (t["hits"], t["misses"], t["false_alarms"], t["correct_negatives"]) == (2, 1, 1, 1)
    s = scores(t)
    assert s["pod"] == pytest.approx(0.667, abs=1e-3)
    assert s["far"] == pytest.approx(0.333, abs=1e-3)
    assert s["csi"] == pytest.approx(0.5)
    assert s["bias"] == pytest.approx(1.0)


def test_brier_and_bss():
    obs = np.array([0, 1, 1, 0, 1, 0])
    assert brier(obs, obs) == 0.0
    months = np.array([1, 1, 1, 2, 2, 2])
    clim = apply_climatology(months, monthly_climatology(months, obs))
    assert brier_skill(clim, obs, clim) == pytest.approx(0.0)


# -- training pieces ---------------------------------------------------------------------------

def test_cv_folds_disjoint_by_season():
    groups = np.repeat(TRAIN_SEASONS * 3, 20)
    folds = season_folds(groups, 5)
    assert len(folds) == 5
    for tr, va in folds:
        assert not set(groups[tr]) & set(groups[va])
        assert len(tr) + len(va) == len(groups)


def fit_small(df, seed=42):
    params = {**base_params(seed), "objective": "binary", "num_leaves": 15, "learning_rate": 0.1,
              "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1}
    return lgb.train(params, lgb.Dataset(df[FEATURES].to_numpy(np.float32), label=df.snow_occ), 30)


def test_same_seed_same_model():
    df = synthetic()
    assert fit_small(df).model_to_string() == fit_small(df).model_to_string()


@pytest.fixture
def hurdle():
    df = synthetic()
    s1 = fit_small(df)
    snow = df[df.snow_occ == 1]
    s2 = lgb.train({**base_params(42), "objective": "tweedie", "num_leaves": 7},
                   lgb.Dataset(snow[FEATURES].to_numpy(np.float32), label=snow.y_swe), 20)
    return HurdleModel(s1, s2, "tweedie", FEATURES, [0.0, 0.5, 1.0], [0.0, 0.4, 1.0], tau=0.3,
                       metadata={"lead_hours": 1}), df


def test_save_load_roundtrip(hurdle, tmp_path):
    model, df = hurdle
    before = model.predict(df)
    model.save(tmp_path)
    assert all((tmp_path / a).exists() for a in ARTIFACTS)
    loaded = HurdleModel.load(tmp_path)
    after = loaded.predict(df)
    assert np.max(np.abs(before.to_numpy() - after.to_numpy())) <= 1e-12
    shuffled = df[list(reversed(df.columns))]
    np.testing.assert_array_equal(loaded.predict(shuffled).to_numpy(), after.to_numpy())
    with pytest.raises(KeyError):
        loaded.predict(df.drop(columns=["f3"]))
    p = after
    np.testing.assert_allclose(p.swe_expected, p.p_cal * p.amount_cond)
    assert ((p.swe_det == 0) | (p.p_cal >= 0.3)).all()


def test_calibrator_monotonic(hurdle):
    model, _ = hurdle
    grid = np.linspace(-0.5, 1.5, 201)
    c = model.calibrate(grid)
    assert np.all(np.diff(c) >= 0) and c.min() >= 0 and c.max() <= 1


def test_tau_argmax():
    obs = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.62, 0.6, 0.3, 0.2, 0.1, 0.05])
    # p >= 0.61 or 0.62 gives CSI 1.0; the lowest such grid value is returned
    assert choose_tau(p, obs) == pytest.approx(0.61)


def test_end_to_end_train(tmp_path):
    cfg = load_config()
    cfg = dataclasses.replace(cfg, train={**cfg.train, "max_trees": 40, "early_stopping_rounds": 10})
    splits = {"train": TRAIN_SEASONS, "val": VAL_SEASONS, "test": []}
    df = synthetic()
    for lead in (1, 3):
        meta = run_lead(df, FEATURES, lead, cfg, splits, tmp_path / "models", tmp_path / "reports", smoke=True)
        d = tmp_path / "models" / f"lead_{lead}h"
        assert all((d / a).exists() for a in ARTIFACTS)
        assert (tmp_path / "reports" / f"val_predictions_L{lead}.parquet").exists()
        m = json.loads((d / "metadata.json").read_text())
        assert m["features"] == FEATURES and 0 < m["tau"] < 1
        assert m["n_train_rows"] == int((df.split == "train").sum()) == meta["n_train_rows"]
        assert m["stage2_variant"] in ("tweedie", "log1p_l2")
