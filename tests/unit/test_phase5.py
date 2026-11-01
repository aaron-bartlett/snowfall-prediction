"""Phase 5: feature construction and leakage tests (offline, synthetic data)."""
import numpy as np
import pandas as pd
import pytest

from src.features import (NEEDED_VARS, TARGET_COLS, feature_frame, hours_since_snow, idw_mean,
                          persistence, plane_fit_gradient, rolling_sum, snow_streak, upstream_mask)

FLAG_VARS = {"wx_snow", "wx_rain", "wx_fzra", "wx_pl", "wx_fog_br", "wx_blsn", "has_ob"}
T0 = pd.Timestamp("2015-01-01 01:00", tz="UTC")
N_HOURS = 240   # 10 days


def stations():
    # DTW at the origin, one neighbor due west (10 km), one due east (20 km)
    return pd.DataFrame({
        "sid": ["DTW", "AAA", "BBB"],
        "dist_km": [0.0, 10.0, 20.0],
        "bearing_deg": [0.0, 270.0, 90.0],
        "x_km": [0.0, -10.0, 20.0],
        "y_km": [0.0, 0.0, 0.0],
    })


def index():
    return pd.date_range(T0, periods=N_HOURS, freq="h")


def long_from(fn):
    """Long hourly table where var values come from fn(var, hour_idx, station_idx) -> array."""
    idx = index()
    frames = []
    for s_i, sid in enumerate(stations().sid):
        h = np.arange(N_HOURS)
        df = pd.DataFrame({"station": sid, "hour_ending": idx})
        for var in NEEDED_VARS:
            df[var] = fn(var, h, s_i).astype(float)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def index_encoded(var, h, s_i):
    if var == "has_ob":
        return np.ones_like(h)
    return h.copy()


def random_data(seed):
    rng = np.random.default_rng(seed)

    def fn(var, h, s_i):
        if var in FLAG_VARS:
            x = (rng.random(len(h)) < 0.3).astype(float)
            if var == "has_ob":
                return np.ones_like(x)
        else:
            x = rng.normal(10, 5, len(h))
        x[rng.random(len(h)) < 0.05] = np.nan
        return x
    return fn


def build(hourly, lead, hours):
    return feature_frame(hourly, stations(), "DTW", lead, hours, index=index())


WINDOWED = ("_tend", "_swe_sum", "_hours_since_snow", "_snow_streak")


@pytest.mark.parametrize("lead", [1, 3])
def test_index_encoding(lead):
    hourly = long_from(index_encoded)
    hours = index()[20:]
    f = build(hourly, lead, hours)
    tpos = np.arange(20, N_HOURS)
    lag_cols = [c for c in f.columns if "_lag" in c]
    assert len(lag_cols) > 100
    for c in lag_cols:
        k = int(c.rsplit("_lag", 1)[1])
        if c.startswith("agg_"):
            if c.split("_lag")[0] in {"agg_idw_tmpf", "agg_idw_dwpf", "agg_idw_mslp", "agg_idw_p01i"}:
                np.testing.assert_array_equal(f[c].to_numpy(), tpos - lead - (k - 1), err_msg=c)
        else:
            np.testing.assert_array_equal(f[c].to_numpy(), tpos - lead - (k - 1), err_msg=c)
    vals = f[lag_cols].to_numpy(dtype=float)
    assert np.all(np.nan_to_num(vals, nan=-1).max(axis=1) <= tpos - lead)
    # windowed target-station features are read at t - L
    np.testing.assert_array_equal(f["DTW_tmpf_tend3"].to_numpy(), np.full(len(tpos), 3.0))


@pytest.mark.parametrize("lead", [1, 3])
@pytest.mark.parametrize("seed", [0, 1])
def test_future_poisoning(lead, seed):
    hourly = long_from(random_data(seed))
    hours = index()[30:]
    base = build(hourly, lead, hours)
    rng = np.random.default_rng(100 + seed)
    for t0 in rng.choice(hours, 6, replace=False):
        t0 = pd.Timestamp(t0)
        cutoff = t0 - pd.Timedelta(hours=lead)
        p = hourly.copy()
        fut = p.hour_ending > cutoff
        for var in NEEDED_VARS:
            p.loc[fut, var] = 1.0 if var in FLAG_VARS else 1e6
        got = build(p, lead, pd.DatetimeIndex([t0]))
        a = base.loc[t0].to_numpy()
        b = got.loc[t0].to_numpy()
        assert np.array_equal(a, b, equal_nan=True), \
            [c for c, x, y in zip(base.columns, a, b) if not (x == y or (np.isnan(x) and np.isnan(y)))]


def test_target_isolation_and_persistence():
    hourly = long_from(random_data(3))
    hours = index()[30:]
    f = build(hourly, 1, hours)
    assert not [c for c in f.columns if c.startswith(("y_", "snow_occ", "persist_"))]
    assert not set(TARGET_COLS) & set(f.columns)
    assert all(f.dtypes == np.float32)
    for lead in (1, 3):
        p = persistence(hourly, "DTW", lead, hours, index())
        d = hourly[hourly.station == "DTW"].set_index("hour_ending")
        avail = (d.has_ob == 1) & d.p01i.notna()
        occ = np.where(avail, (d.wx_snow == 1).astype(float), np.nan)
        np.testing.assert_array_equal(p.persist_occ.to_numpy(), occ[30 - lead:N_HOURS - lead])


def test_idw():
    assert idw_mean(np.array([0.0, 30.0]), np.array([10.0, 20.0])) == pytest.approx(6.0)
    got = idw_mean(np.array([[0.0, 30.0], [np.nan, 30.0], [np.nan, np.nan]]), np.array([10.0, 20.0]))
    assert got[0] == pytest.approx(6.0) and got[1] == pytest.approx(30.0) and np.isnan(got[2])


def test_upstream():
    m = upstream_mask(270.0, [270.0, 230.0, 90.0, 320.0])
    assert m.tolist() == [[True, True, False, False]]


def test_plane_fit():
    xs, ys = np.meshgrid(np.linspace(-200, 200, 4), np.linspace(-150, 150, 3))
    x, y = xs.ravel(), ys.ravel()
    p = (1000 + 0.01 * x)[None, :]
    mag, direction = plane_fit_gradient(p, x, y)
    assert mag[0] == pytest.approx(1.0) and direction[0] == pytest.approx(90.0)
    few = p.copy()
    few[0, 7:] = np.nan
    assert np.isnan(plane_fit_gradient(few, x, y)[0][0])


def test_time_features_use_target_hour():
    hourly = long_from(random_data(4))
    hours = index()[30:60]
    f = build(hourly, 3, hours)
    np.testing.assert_allclose(f.hod_sin, np.sin(2 * np.pi * hours.hour / 24), atol=1e-6)


def test_tendency():
    def fn(var, h, s_i):
        if var == "has_ob":
            return np.ones_like(h)
        x = np.full(len(h), np.nan)
        if var == "mslp" and s_i == 0:
            x[:4] = [1000, 1002, 1005, 1009]
        return x
    hourly = long_from(fn)
    f = build(hourly, 1, index()[[3, 4]])
    assert f.DTW_mslp_tend1.tolist() == [3.0, 4.0]
    assert np.isnan(f.DTW_mslp_tend3.iloc[0]) and f.DTW_mslp_tend3.iloc[1] == 9.0


def test_windowed_helpers():
    s = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=float)
    assert snow_streak(s).tolist() == [0, 1, 2, 0, 0, 1, 2, 3, 0]
    assert hours_since_snow(s, cap=72).tolist() == [72, 0, 0, 1, 2, 0, 0, 0, 1]
    assert rolling_sum(np.array([1.0, np.nan, 2.0, 3.0]), 2).tolist() == [1.0, 1.0, 2.0, 5.0]
