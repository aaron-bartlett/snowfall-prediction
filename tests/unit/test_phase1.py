import pandas as pd
import pytest

from src.geo import bearing_deg, haversine_km, local_xy_km
from src.stations import coverage_from_csv, total_target_hours
from src.timeutil import assign_hour_ending, keep_latest_per_hour, lst_date, season_of, target_hours


def ts(s):
    return pd.Timestamp(s, tz="UTC")


# --- geo -------------------------------------------------------------------------------------
def test_haversine_one_degree_equator():
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.19, abs=0.1)
    assert haversine_km(42, -83, 42, -83) == pytest.approx(0.0)


def test_bearing_cardinal():
    assert bearing_deg(0, 0, 1, 0) == pytest.approx(0.0, abs=1e-6)
    assert bearing_deg(0, 0, 0, 1) == pytest.approx(90.0, abs=1e-6)
    assert bearing_deg(0, 0, -1, 0) == pytest.approx(180.0, abs=1e-6)
    assert bearing_deg(0, 0, 0, -1) == pytest.approx(270.0, abs=1e-6)


def test_local_xy():
    x, y = local_xy_km(1.0, 0.0, 0.0, 0.0)
    assert x == pytest.approx(0.0) and y == pytest.approx(111.19, abs=0.1)


# --- time conventions ---------------------------------------------------------------------------
def test_assign_hour_ending():
    v = pd.Series([ts("2015-01-01 14:53"), ts("2015-01-01 15:05"), ts("2015-01-01 14:20"),
                   ts("2015-01-01 14:40"), ts("2015-01-01 15:10"), ts("2015-01-01 15:11")])
    he = assign_hour_ending(v)
    assert he[0] == ts("2015-01-01 15:00")
    assert he[1] == ts("2015-01-01 15:00")
    assert pd.isna(he[2])
    assert he[3] == ts("2015-01-01 15:00")
    assert he[4] == ts("2015-01-01 15:00")
    assert pd.isna(he[5])


def test_keep_latest_per_hour():
    df = pd.DataFrame({"station": ["A", "A", "A"],
                       "valid": [ts("2015-01-01 14:55"), ts("2015-01-01 14:45"), ts("2015-01-01 14:20")],
                       "x": [2, 1, 0]})
    df["hour_ending"] = assign_hour_ending(df.valid)
    out = keep_latest_per_hour(df)
    assert len(out) == 1 and out.iloc[0].x == 2


def test_target_hour_counts():
    assert sum(len(target_hours(y)) for y in range(2005, 2026)) == 106_968
    assert sum(len(target_hours(y)) for y in range(2005, 2025)) == 101_880
    th = target_hours(2005)
    assert th[0] == ts("2005-10-01 00:00") and th[-1] == ts("2006-04-30 23:00")
    assert total_target_hours(range(2005, 2026)) == 106_968


def test_lst_date():
    d = lst_date(pd.Series([ts("2015-01-02 05:00"), ts("2015-01-02 06:00")]))
    assert str(d[0]) == "2015-01-01" and str(d[1]) == "2015-01-02"


def test_season_of():
    s = season_of(pd.Series([ts("2005-09-01 01:00"), ts("2005-12-31 23:00"), ts("2006-04-30 23:00"),
                             ts("2006-05-01 00:00")]))
    assert list(s) == [2005, 2005, 2005, 2005]


# --- coverage ---------------------------------------------------------------------------------
def test_coverage_from_csv():
    # Season 2005 only -> denominator = len(target_hours(2005)).
    rows = ["station,valid,tmpf,p01i",
            "X,2005-10-01 00:53,50,0.00",   # hour 01Z: tmpf + p01i
            "X,2005-10-01 00:20,50,0.00",   # dropped (intra-hour)
            "X,2005-10-01 01:45,M,0.00",    # hour 02Z: p01i only
            "X,2005-10-01 01:55,51,M",      # hour 02Z: tmpf only (same hour, counted once each)
            "X,2005-10-01 02:53,52,M",      # hour 03Z: tmpf only
            "X,2005-09-15 12:53,60,0.00",   # out of season -> ignored
            "X,2006-05-01 03:53,60,0.00"]   # out of season -> ignored
    t, p = coverage_from_csv("\n".join(rows), [2005])
    n = len(target_hours(2005))
    assert t == pytest.approx(3 / n)
    assert p == pytest.approx(2 / n)


def test_coverage_empty():
    assert coverage_from_csv("station,valid,tmpf,p01i\n", [2005]) == (0.0, 0.0)
