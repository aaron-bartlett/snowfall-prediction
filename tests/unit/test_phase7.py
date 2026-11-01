"""Phase 7: test-evaluation building blocks (offline, synthetic data)."""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src import evaluate as ev
from src.config import load_config
from src.model import HurdleModel


def hours(n, start="2016-01-01 00:00"):
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


# -- event segmentation ------------------------------------------------------------------------

def test_segment_events_toy():
    occ = [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1]
    #      .  A  A  A  A  A  .  .  .  B  B  B    gap of 2 joins, gap of 3 splits
    ids = ev.segment_events(hours(len(occ)), occ)
    assert ids.tolist() == [-1, 0, 0, 0, 0, 0, -1, -1, -1, 1, 1, 1]


def test_segment_events_missing_hours_count_as_gap():
    t = hours(10)[[0, 1, 5, 6]]          # 3 missing hours between rows 1 and 2
    ids = ev.segment_events(t, [1, 1, 1, 1])
    assert ids.tolist() == [0, 0, 1, 1]
    t2 = hours(10)[[0, 1, 4, 5]]         # 2 missing hours -> same event
    assert ev.segment_events(t2, [1, 1, 1, 1]).tolist() == [0, 0, 0, 0]


def test_segment_events_edge_cases():
    assert ev.segment_events(hours(3), [0, 0, 0]).tolist() == [-1, -1, -1]
    assert ev.segment_events(hours(1), [1]).tolist() == [0]
    assert ev.segment_events(hours(4), [1, 0, 0, 1], max_gap=1).tolist() == [0, -1, -1, 1]


# -- lock guard --------------------------------------------------------------------------------

def test_lock_guard_blocks_second_run(tmp_path):
    lock = tmp_path / "TEST_EVALUATED.lock"
    ev.check_lock(lock, force=False)                      # no lock yet: allowed
    ev.write_lock(lock, {"evaluated_at": "x"})
    assert lock.exists() and not (tmp_path / "TEST_EVALUATED.lock.part").exists()
    with pytest.raises(ev.LockError):
        ev.check_lock(lock, force=False)
    ev.check_lock(lock, force=True)                       # --force overrides
    with pytest.raises(ev.LockError):                     # run() checks the lock before touching any data
        ev.run(reports=tmp_path)


def test_main_requires_final(capsys):
    assert ev.main([]) == 1
    assert "FAILED evaluate" in capsys.readouterr().out


# -- helpers -----------------------------------------------------------------------------------

def test_intensity_bin():
    assert ev.intensity_bin([0.0, 0.01, 0.02, 0.03, 0.05, 0.06, 0.3]).tolist() == \
        ["trace", "0.01-0.02", "0.01-0.02", "0.03-0.05", "0.03-0.05", ">=0.06", ">=0.06"]


def test_wind_sector():
    # wind from W: blows toward east -> u > 0, v = 0
    u = np.array([10.0, 0.0, -10.0, 0.0, 0.1, np.nan])
    v = np.array([0.0, -10.0, 0.0, 10.0, 0.0, 1.0])
    assert ev.wind_sector(u, v).tolist() == ["W", "N", "E", "S", "calm", "missing"]


# -- end to end on synthetic data -------------------------------------------------------------

FEATURES = ["DTW_tmpf_lag1", "DTW_u_lag1", "DTW_v_lag1", "f0", "f1", "f2"]


def synthetic(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES))).astype(np.float32)
    X[:, 0] = rng.uniform(5, 40, n)
    X[:, 1:3] *= 8
    p = 1 / (1 + np.exp(-(2.0 * X[:, 3] - 2.5)))
    occ = (rng.random(n) < p).astype(float)
    swe = np.where(occ == 1, np.round(np.clip(0.02 * np.exp(0.5 * X[:, 4]) - 0.01, 0, None), 2), 0.0)
    t = pd.date_range("2016-10-01", periods=n, freq="h", tz="UTC")
    meta = pd.DataFrame({"hour_ending": t, "season": "2016-17", "split": "test", "snow_occ": occ,
                         "y_swe": swe, "y_depth": swe * 12, "wx_mixed": 0.0,
                         "persist_occ": np.roll(occ, 1), "persist_swe": np.roll(swe, 1)})
    return pd.concat([meta, pd.DataFrame(X, columns=FEATURES)], axis=1)


def tiny_model(df):
    X = df[FEATURES].to_numpy(np.float32)
    prm = {"verbosity": -1, "num_threads": 1, "seed": 0, "deterministic": True, "force_row_wise": True}
    s1 = lgb.train({**prm, "objective": "binary"}, lgb.Dataset(X, df.snow_occ), 20)
    snow = df.snow_occ.to_numpy() == 1
    s2 = lgb.train({**prm, "objective": "tweedie"}, lgb.Dataset(X[snow], df.y_swe[snow]), 20)
    return HurdleModel(s1, s2, "tweedie", FEATURES, [0.0, 1.0], [0.0, 1.0], tau=0.4)


def test_evaluate_lead_synthetic(tmp_path):
    df = synthetic()
    model = tiny_model(df)
    cfg = load_config()
    clims = ({m: 0.08 for m in cfg.target_months}, {m: 0.001 for m in cfg.target_months},
             {m: 0.012 for m in cfg.target_months})
    dates = pd.Series(pd.date_range("2016-09-30", "2017-01-15", freq="D")).dt.date
    ghcn = pd.DataFrame({"snow": np.linspace(0, 2, len(dates))}, index=pd.Index(dates, name="date"))
    res, out, totals = ev.evaluate_lead(1, model, df, clims, ghcn, cfg, tmp_path)
    for f in ("reliability_L1.png", "shap_L1.png"):
        assert (tmp_path / f).stat().st_size > 0
    for s in ev.SYSTEMS:
        for k in ("brier", "csi", "mae", "rmse", "mae_snow"):
            assert np.isfinite(res["headline"][s][k]), (s, k)
    for table in ("thresholds", "events", "by_month", "by_intensity", "by_wind"):
        assert {r["system"] for r in res[table]} == set(ev.SYSTEMS), table
    assert {"model", "persistence", "climatology"} <= {r["system"] for r in res["daily"]}
    assert len(res["shap_top"]) == 6 and res["shap_top"][0][0] == "f0"
    # event totals conserve the observed SWE inside events
    in_ev = totals.obs_swe.sum()
    assert in_ev == pytest.approx(df.y_swe.sum())
    assert len(out) == len(df) and np.allclose(out.swe_expected, out.p_cal * out.amount_cond)
    json.dumps(res, default=str)   # serialisable
