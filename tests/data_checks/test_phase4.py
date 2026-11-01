import json

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.download import RAW_GHCN
from src.target import CALIB_PATH, FIG_PATH, REPORT_PATH, TARGET_PATH, daily_compare, parse_ghcn, season_table

pytestmark = pytest.mark.data
TOTAL_TARGET_HOURS = 106_968


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def target():
    return pd.read_parquet(TARGET_PATH)


@pytest.fixture(scope="module")
def daily(target, cfg):
    return daily_compare(target, parse_ghcn(RAW_GHCN / f"{cfg.ghcn_id}.csv.gz"), cfg.lst_offset_hours)


def test_rows_and_availability(target):
    assert len(target) == TOTAL_TARGET_HOURS
    assert target.hour_ending.is_unique
    avail = target.target_available.mean()
    print(f"\ntarget available {avail:.4f}")
    assert avail >= 0.90, f"STOP: target availability {avail:.3f}"
    assert avail >= 0.95


def test_snow_rate(target):
    rate = target.loc[target.target_available, "snow_occ"].mean()
    print(f"\nsnow_occ rate {rate:.4f}")
    assert 0.03 <= rate <= 0.20, f"STOP: snow-hour rate {rate:.3f}"


def test_columns_consistent(target):
    av = target[target.target_available]
    assert av[["snow_occ", "y_swe", "y_depth", "split"]].notna().all().all()
    assert (av.loc[av.snow_occ == 0, "y_swe"] == 0).all()
    assert (av.y_swe >= 0).all()
    assert set(target.split) == {"train", "val", "test"}


def test_season_depth_ratio(daily):
    s = season_table(daily)
    print("\nper-season depth ratios:", s.depth_ratio.round(2).to_dict())
    m = s.depth_ratio.mean()
    assert 0.4 <= m <= 1.6, f"STOP: mean depth/GHCN ratio {m:.2f}"


def test_daily_correlation(daily):
    big = daily[(daily.snow >= 1) | (daily.y_depth >= 1)]
    r = np.corrcoef(big.snow, big.y_depth)[0, 1]
    print(f"\ndaily r {r:.3f} on {len(big)} days")
    assert r >= 0.6, f"daily r {r:.2f} < 0.6: show scatter and ask before continuing"


def test_top10_days_have_snow_hours(daily):
    top = daily.sort_values("snow", ascending=False).head(10)
    bad = top[top.snow_hours < 3]
    assert bad.empty, f"top GHCN days with < 3 snow hours: {bad.index.tolist()}"


def test_outputs_exist():
    assert REPORT_PATH.exists() and FIG_PATH.exists()
    c = json.loads(CALIB_PATH.read_text())
    assert c["fit_on"] == "train seasons" and c["depth_correction_factor"] > 0
