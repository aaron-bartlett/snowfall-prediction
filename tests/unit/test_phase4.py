import gzip

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.target import build_target, daily_compare, parse_ghcn, slr

CFG = load_config()
TABLE, FALLBACK = CFG.slr["table"], CFG.slr["fallback"]


def test_slr_boundaries():
    got = slr([33.9, 34.0, 14.9, np.nan, 24.9, 25.0, 50.0], TABLE, FALLBACK)
    assert list(got) == [10, 8, 18, 10, 15, 12, 8]


def make_hourly(rows):
    """rows: (hour, has_ob, p01i, wx_snow, wx_mixed, trace, tmpf)"""
    base = pd.Timestamp("2015-01-02 00:00", tz="UTC")
    df = pd.DataFrame(rows, columns=["h", "has_ob", "p01i", "wx_snow", "wx_mixed", "p01i_trace", "tmpf"])
    df["hour_ending"] = base + pd.to_timedelta(df.h, unit="h")
    df["station"] = "DTW"
    df["season"] = "2014-15"
    return df.drop(columns="h")


SPLITS = {"train": ["2014-15"], "val": [], "test": []}


@pytest.fixture
def tgt():
    h = make_hourly([
        (1, True, 0.05, 1, 0, 0, 28.0),     # SN with 0.05 -> 0.05, depth x12
        (2, True, 0.05, 0, 0, 0, 40.0),     # RA -> 0
        (3, True, 0.00, 0, 0, 0, 20.0),     # BLSN only (wx_snow 0) -> 0, no snow
        (4, True, 0.00, 1, 0, 1, 20.0),     # -SN trace -> 0, snow_occ
        (5, True, 0.02, 1, 1, 0, 33.0),     # RASN -> mixed
        (6, True, np.nan, 1, 0, 0, 30.0),   # p01i NaN -> unavailable
        (7, False, np.nan, np.nan, np.nan, np.nan, np.nan),  # no ob
    ])
    return build_target(h, CFG, SPLITS).set_index(pd.Index(range(1, 8)))


def test_y_swe_cases(tgt):
    assert tgt.loc[1, "y_swe"] == pytest.approx(0.05) and tgt.loc[1, "snow_occ"] == 1
    assert tgt.loc[1, "y_depth"] == pytest.approx(0.05 * 12)
    assert tgt.loc[2, "y_swe"] == 0 and tgt.loc[2, "snow_occ"] == 0
    assert tgt.loc[3, "y_swe"] == 0 and tgt.loc[3, "snow_occ"] == 0
    assert tgt.loc[4, "y_swe"] == 0 and tgt.loc[4, "snow_occ"] == 1
    assert tgt.loc[5, "wx_mixed"] == 1 and tgt.loc[5, "y_swe"] == pytest.approx(0.02)
    for i in (6, 7):
        assert not tgt.loc[i, "target_available"]
        assert np.isnan(tgt.loc[i, "y_swe"]) and np.isnan(tgt.loc[i, "snow_occ"])
    assert (tgt.split == "train").all()


def test_parse_ghcn(tmp_path):
    lines = ["USW00094847,20150201,SNOW,424,,,X,",        # 424 mm -> 16.69 in
             "USW00094847,20150201,PRCP,269,,,X,",        # 26.9 mm -> 1.059 in
             "USW00094847,20150202,SNOW,-9999,,,X,",      # missing
             "USW00094847,20150203,SNOW,25,,I,X,",        # QC flagged -> dropped
             "USW00094847,20150203,TMAX,10,,,X,"]
    p = tmp_path / "g.csv.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    g = parse_ghcn(p)
    d = pd.Timestamp("2015-02-01").date()
    assert g.loc[d, "snow"] == pytest.approx(424 / 25.4)
    assert g.loc[d, "prcp"] == pytest.approx(26.9 / 25.4)
    assert pd.Timestamp("2015-02-02").date() not in g.index or np.isnan(g.loc[pd.Timestamp("2015-02-02").date(), "snow"])
    assert pd.Timestamp("2015-02-03").date() not in g.index
    assert "tmax" not in g.columns


def test_daily_aggregation_lst_boundary():
    # 24 available hours: hour-ending 2015-01-02 05Z .. 2015-01-03 04Z = LST Jan 1 23-24h + Jan 2 00-23h
    hours = pd.date_range("2015-01-02 05:00", periods=25, freq="h", tz="UTC")
    t = pd.DataFrame({"hour_ending": hours, "season": "2014-15", "split": "train",
                      "target_available": True, "snow_occ": 0.0, "y_swe": 0.0, "y_depth": 0.0})
    t.loc[0, ["y_depth", "snow_occ"]] = [5.0, 1.0]   # 05Z belongs to LST Jan 1
    t.loc[1, ["y_depth", "snow_occ"]] = [2.0, 1.0]   # 06Z belongs to LST Jan 2
    g = pd.DataFrame({"snow": [0.0, 2.0], "prcp": [0.0, 0.2], "snwd": [0.0, 0.0]},
                     index=[pd.Timestamp("2015-01-01").date(), pd.Timestamp("2015-01-02").date()])
    d = daily_compare(t, g, -5)
    jan2 = pd.Timestamp("2015-01-02").date()
    assert list(d.index) == [jan2]                    # Jan 1 has only 1 hour -> excluded
    assert d.loc[jan2, "y_depth"] == 2.0 and d.loc[jan2, "hours"] == 24
