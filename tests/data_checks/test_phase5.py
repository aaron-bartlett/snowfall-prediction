import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.clean import HOURLY_PATH
from src.config import load_config
from src.features import (NEEDED_VARS, STATIONS_PATH, TARGET_COLS, feature_frame, feature_list_path,
                          features_path, load_target)

pytestmark = pytest.mark.data
LEADS = [1, 3]
WX = {"wx_snow", "wx_rain", "wx_fzra", "wx_pl", "wx_fog_br", "wx_blsn"}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def target(cfg):
    hourly = pd.read_parquet(HOURLY_PATH, filters=[("station", "==", cfg.target_station)])
    return load_target(cfg, hourly)


@pytest.fixture(scope="module", params=LEADS)
def feats(request):
    return request.param, pd.read_parquet(features_path(request.param))


def test_rows(feats, target):
    lead, df = feats
    assert len(df) == int(target.target_available.sum())
    assert df.hour_ending.is_unique


def test_feature_columns(feats):
    lead, df = feats
    fl = json.loads(feature_list_path(lead).read_text())
    assert fl == [c for c in df.columns if c not in TARGET_COLS]
    n = len(fl)
    print(f"\nlead {lead}: {n} features")
    assert 1200 <= n <= 2000, f"STOP: feature count {n} outside 1,200-2,000"
    X = df[fl]
    assert all(X.dtypes == np.float32)
    all_nan = X.columns[X.isna().all()].tolist()
    assert not all_nan, f"all-NaN columns: {all_nan}"
    assert X.DTW_tmpf_lag1.notna().mean() >= 0.95


def test_split_shares(feats):
    lead, df = feats
    share = df.split.value_counts(normalize=True)
    rate = df.groupby("split").snow_occ.mean()
    print(f"\nlead {lead} split shares {share.round(4).to_dict()}; snow rate {rate.round(4).to_dict()}")
    for name, n in (("train", 15), ("val", 3), ("test", 3)):
        assert abs(share[name] - n / 21) <= 0.03


def test_size():
    for lead in LEADS:
        assert features_path(lead).stat().st_size < 2e9
        assert pq.ParquetFile(features_path(lead)).metadata.num_rows > 0


@pytest.mark.parametrize("lead", LEADS)
def test_real_data_future_poisoning(lead, cfg):
    stations = pd.read_csv(STATIONS_PATH)
    lo, hi = pd.Timestamp("2014-01-01 01:00", tz="UTC"), pd.Timestamp("2014-01-31 00:00", tz="UTC")
    hourly = pd.read_parquet(HOURLY_PATH, columns=["station", "hour_ending"] + NEEDED_VARS)
    hourly = hourly[(hourly.hour_ending >= lo) & (hourly.hour_ending <= hi)
                    & hourly.station.astype(str).isin(set(stations.sid))].reset_index(drop=True)
    for v in NEEDED_VARS:
        hourly[v] = hourly[v].astype(float)
    index = pd.date_range(lo, hi, freq="h")
    hours = index[24:]
    kw = dict(lags_dtw=cfg.lags["target_station"], lags_nb=tuple(cfg.lags["neighbor"]), index=index)
    base = feature_frame(hourly, stations, cfg.target_station, lead, hours, **kw)
    rng = np.random.default_rng(lead)
    for t0 in rng.choice(hours, 20, replace=False):
        t0 = pd.Timestamp(t0)
        p = hourly.copy()
        fut = p.hour_ending > t0 - pd.Timedelta(hours=lead)
        for v in NEEDED_VARS:
            p.loc[fut, v] = 1.0 if v in WX or v == "has_ob" else 1e6
        got = feature_frame(p, stations, cfg.target_station, lead, pd.DatetimeIndex([t0]), **kw)
        assert np.array_equal(base.loc[t0].to_numpy(), got.loc[t0].to_numpy(), equal_nan=True), \
            f"leakage at {t0} lead {lead}"
