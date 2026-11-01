"""Lagged + spatial feature matrices for the 1 h and 3 h lead models.

Usage: python -m src.features --lead all|1|3
Output: data/processed/features_L{L}.parquet, meta/feature_list_L{L}.json

Time cutoff: a feature for target hour t at lead L uses only hours <= t - L. Every variable is laid
out on a continuous hourly index, the per-hour aggregates use same-hour values only, and a lag-k
feature reads position t - L - (k - 1) of that index. Windowed features (sums, tendencies, streaks)
are computed on the unshifted series and then read at t - L, so the window ends at t - L.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

from src.clean import HOURLY_PATH, PROCESSED
from src.config import ROOT, load_config
from src.splits import load_splits
from src.target import TARGET_PATH, build_target

STATIONS_PATH = ROOT / "meta" / "stations.csv"
INDEX_START = pd.Timestamp("2005-09-01 01:00", tz="UTC")
INDEX_END = pd.Timestamp("2026-05-01 00:00", tz="UTC")

DTW_VARS = ["tmpf", "dwpf", "relh", "dewpoint_depression", "u", "v", "sknt", "gust", "mslp", "alti",
            "vsby", "ceiling_ft", "sky_cover", "p01i", "wx_snow", "snow_intensity", "wx_rain", "wx_fzra",
            "wx_pl", "wx_fog_br", "wx_blsn"]
NEIGHBOR_VARS = ["tmpf", "dwpf", "u", "v", "mslp", "p01i", "wx_snow", "vsby"]
NEEDED_VARS = sorted(set(DTW_VARS) | set(NEIGHBOR_VARS) | {"has_ob"})
IDW_VARS = ["tmpf", "dwpf", "mslp", "p01i"]
RINGS = [(0, 100), (100, 200), (200, 300)]
SECTORS = {"W": 270.0, "SW": 225.0, "NW": 315.0, "N": 0.0, "S": 180.0, "E": 90.0}
SECTOR_HALF_WIDTH = 22.5
UPSTREAM_HALF_WIDTH = 45.0
CALM_KT = 1.0               # vector-mean wind below this is treated as calm (no upstream direction)
MIN_PLANE_STATIONS = 8
HOURS_SINCE_CAP = 72
TEND_STEPS = [1, 3, 6]
SWE_WINDOWS = [3, 6, 12]
TARGET_COLS = ["hour_ending", "season", "split", "snow_occ", "y_swe", "y_depth", "wx_mixed",
               "persist_occ", "persist_swe"]


def features_path(lead: int):
    return PROCESSED / f"features_L{lead}.parquet"


def feature_list_path(lead: int):
    return ROOT / "meta" / f"feature_list_L{lead}.json"


# ---------------------------------------------------------------------------------------------
# Per-hour aggregates (same-hour values only)
# ---------------------------------------------------------------------------------------------

def angle_diff(a, b):
    """Smallest absolute angular difference in degrees."""
    return np.abs((np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0)


def idw_mean(values: np.ndarray, dist_km: np.ndarray, power: float = 2.0) -> np.ndarray:
    """Inverse-distance-weighted mean across the station axis (last axis), ignoring NaNs."""
    values = np.asarray(values, dtype=float)
    w = 1.0 / np.asarray(dist_km, dtype=float) ** power
    ok = ~np.isnan(values)
    num = np.where(ok, values, 0.0) @ w if values.ndim > 1 else np.sum(np.where(ok, values, 0.0) * w)
    den = ok.astype(float) @ w if values.ndim > 1 else np.sum(ok * w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def wind_from_deg(u, v):
    """Meteorological direction the wind blows from, given u/v components (u = -s·sin d, v = -s·cos d)."""
    return (np.degrees(np.arctan2(-np.asarray(u, float), -np.asarray(v, float))) + 360.0) % 360.0


def upstream_mask(from_deg, bearings) -> np.ndarray:
    """[hours, stations] True where the station bearing is within ±45° of the wind-from direction."""
    f = np.atleast_1d(np.asarray(from_deg, dtype=float))[:, None]
    return angle_diff(f, np.asarray(bearings, dtype=float)[None, :]) <= UPSTREAM_HALF_WIDTH


def plane_fit_gradient(p: np.ndarray, x: np.ndarray, y: np.ndarray, min_n: int = MIN_PLANE_STATIONS):
    """Per-hour least squares p = a + b·x + c·y. Returns (magnitude hPa/100 km, direction toward higher
    pressure in degrees clockwise from north). NaN when fewer than min_n stations report."""
    p = np.atleast_2d(np.asarray(p, dtype=float))
    ok = ~np.isnan(p)
    m = ok.astype(float)
    pz = np.where(ok, p, 0.0)
    n = m.sum(1)
    sx, sy = m @ x, m @ y
    sxx, syy, sxy = m @ (x * x), m @ (y * y), m @ (x * y)
    sp, sxp, syp = pz.sum(1), pz @ x, pz @ y
    A = np.stack([np.stack([n, sx, sy], -1), np.stack([sx, sxx, sxy], -1), np.stack([sy, sxy, syy], -1)], -2)
    rhs = np.stack([sp, sxp, syp], -1)
    good = n >= min_n
    A[~good] = np.eye(3)
    rhs[~good] = 0.0
    det = np.linalg.det(A)
    good &= np.abs(det) > 1e-9
    A[~good] = np.eye(3)
    rhs[~good] = 0.0
    coef = np.linalg.solve(A, rhs[..., None])[..., 0]
    b, c = coef[:, 1], coef[:, 2]
    mag = np.where(good, np.hypot(b, c) * 100.0, np.nan)
    direction = np.where(good, (np.degrees(np.arctan2(b, c)) + 360.0) % 360.0, np.nan)
    return mag, direction


def _frac(flag: np.ndarray, cols: np.ndarray):
    """Count and fraction of reporting stations (in cols) with flag == 1."""
    sub = flag[:, cols]
    rep = (~np.isnan(sub)).sum(1)
    cnt = np.nansum(sub == 1, axis=1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(rep > 0, cnt / np.maximum(rep, 1), np.nan)
    return np.where(rep > 0, cnt, np.nan), frac


def _nansum_or_nan(x: np.ndarray) -> np.ndarray:
    rep = (~np.isnan(x)).sum(1)
    return np.where(rep > 0, np.nansum(x, axis=1), np.nan)


def hourly_aggregates(W: dict, st: pd.DataFrame, target: str) -> dict:
    """Per-hour spatial aggregates from same-hour values. W: var -> [hours, stations] array."""
    nb = (st.sid != target).to_numpy()
    nbi = np.flatnonzero(nb)
    dist = st.dist_km.to_numpy(float)
    bear = st.bearing_deg.to_numpy(float)
    agg = {}
    for var in IDW_VARS:
        agg[f"idw_{var}"] = idw_mean(W[var][:, nbi], dist[nbi])
    snow = W["wx_snow"]
    for lo, hi in RINGS:
        cols = nbi[(dist[nbi] >= lo) & ((dist[nbi] < hi) if hi < 300 else (dist[nbi] <= hi))]
        agg[f"snow_count_r{lo}_{hi}"], agg[f"snow_frac_r{lo}_{hi}"] = _frac(snow, cols)
    for name, center in SECTORS.items():
        cols = nbi[angle_diff(bear[nbi], center) <= SECTOR_HALF_WIDTH]
        agg[f"sector_{name}_snow_frac"] = _frac(snow, cols)[1]
        agg[f"sector_{name}_p01i_sum"] = _nansum_or_nan(W["p01i"][:, cols])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN hours -> NaN
        u_bar = np.nanmean(W["u"], axis=1)
        v_bar = np.nanmean(W["v"], axis=1)
    speed = np.hypot(u_bar, v_bar)
    from_deg = wind_from_deg(u_bar, v_bar)
    valid = ~np.isnan(speed) & (speed >= CALM_KT)
    up = upstream_mask(np.where(valid, from_deg, 0.0), bear[nbi])
    up_snow = np.where(up, snow[:, nbi], np.nan)
    up_frac = _frac(up_snow, np.arange(len(nbi)))[1]
    up_p = np.where(up, W["p01i"][:, nbi], np.nan)
    agg["upstream_snow_frac"] = np.where(valid, up_frac, np.nan)
    agg["upstream_idw_p01i"] = np.where(valid, idw_mean(up_p, dist[nbi]), np.nan)
    agg["mslp_grad_mag"], agg["mslp_grad_dir"] = plane_fit_gradient(
        W["mslp"], st.x_km.to_numpy(float), st.y_km.to_numpy(float))
    with np.errstate(invalid="ignore"):
        rep_t = (~np.isnan(W["tmpf"])).sum(1) > 0
        tmax = np.nanmax(np.where(np.isnan(W["tmpf"]), -np.inf, W["tmpf"]), axis=1)
        tmin = np.nanmin(np.where(np.isnan(W["tmpf"]), np.inf, W["tmpf"]), axis=1)
    agg["tmpf_spread"] = np.where(rep_t, tmax - tmin, np.nan)
    agg["n_reporting"] = np.nansum(W["has_ob"] == 1, axis=1).astype(float)
    return agg


# ---------------------------------------------------------------------------------------------
# Target-station windowed features (computed unshifted, read at t - L)
# ---------------------------------------------------------------------------------------------

def rolling_sum(x: np.ndarray, k: int) -> np.ndarray:
    """Sum over the k hours ending at each hour (NaNs skipped; NaN if all missing)."""
    return pd.Series(x).rolling(k, min_periods=1).sum().to_numpy()


def hours_since_snow(snow: np.ndarray, cap: int = HOURS_SINCE_CAP) -> np.ndarray:
    idx = np.arange(len(snow))
    is_snow = snow == 1
    last = np.maximum.accumulate(np.where(is_snow, idx, -10 ** 9))
    return np.minimum(idx - last, cap).astype(float)


def snow_streak(snow: np.ndarray) -> np.ndarray:
    idx = np.arange(len(snow))
    last_dry = np.maximum.accumulate(np.where(snow == 1, -1, idx))
    return (idx - last_dry).astype(float)


# ---------------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------------

def to_wide(hourly: pd.DataFrame, sids: list[str], index: pd.DatetimeIndex) -> dict:
    """Long hourly table -> {var: float array [len(index), len(sids)]}, NaN where no row."""
    pos_t = ((hourly.hour_ending - index[0]) // pd.Timedelta(hours=1)).to_numpy()
    smap = {s: i for i, s in enumerate(sids)}
    pos_s = hourly.station.astype(str).map(smap).to_numpy()
    keep = (pos_t >= 0) & (pos_t < len(index)) & ~pd.isna(pos_s)
    pos_t, pos_s = pos_t[keep], pos_s[keep].astype(int)
    W = {}
    for var in NEEDED_VARS:
        a = np.full((len(index), len(sids)), np.nan)
        a[pos_t, pos_s] = hourly[var].to_numpy(dtype=float)[keep]
        W[var] = a
    return W


def feature_frame(hourly: pd.DataFrame, stations: pd.DataFrame, target: str, lead: int,
                  hours: pd.DatetimeIndex, lags_dtw: int = 12, lags_nb=(1, 2, 3, 6),
                  index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Feature matrix (float32) for target hours `hours` at lead `lead`."""
    if index is None:
        index = pd.date_range(hourly.hour_ending.min(), hourly.hour_ending.max(), freq="h")
    st = stations.reset_index(drop=True)
    sids = st.sid.tolist()
    W = to_wide(hourly, sids, index)
    rows = ((pd.DatetimeIndex(hours) - index[0]) // pd.Timedelta(hours=1)).to_numpy()
    if (rows < 0).any() or (rows >= len(index)).any():
        raise ValueError("target hours outside the feature index")

    def at(x: np.ndarray, offset: int) -> np.ndarray:
        """x read at row t - offset (NaN before the start of the index)."""
        src = rows - offset
        out = np.full(len(rows), np.nan)
        ok = src >= 0
        out[ok] = x[src[ok]]
        return out

    cols: dict[str, np.ndarray] = {}
    d = sids.index(target)
    for var in DTW_VARS:
        x = W[var][:, d]
        for k in range(1, lags_dtw + 1):
            cols[f"{target}_{var}_lag{k}"] = at(x, lead + k - 1)
    for var in ("mslp", "tmpf"):
        x = W[var][:, d]
        for k in TEND_STEPS:
            cols[f"{target}_{var}_tend{k}"] = at(x, lead) - at(x, lead + k)
    snow_d = W["wx_snow"][:, d]
    swe = np.where(snow_d == 1, W["p01i"][:, d], np.where(np.isnan(snow_d), np.nan, 0.0))
    swe = np.where(np.isnan(W["p01i"][:, d]), np.nan, swe)
    for k in SWE_WINDOWS:
        cols[f"{target}_swe_sum{k}"] = at(rolling_sum(swe, k), lead)
    cols[f"{target}_hours_since_snow"] = at(hours_since_snow(snow_d), lead)
    cols[f"{target}_snow_streak"] = at(snow_streak(snow_d), lead)

    for i, sid in enumerate(sids):
        if sid == target:
            continue
        for var in NEIGHBOR_VARS:
            x = W[var][:, i]
            for k in lags_nb:
                cols[f"{sid}_{var}_lag{k}"] = at(x, lead + k - 1)

    for name, x in hourly_aggregates(W, st, target).items():
        for k in lags_nb:
            cols[f"agg_{name}_lag{k}"] = at(x, lead + k - 1)

    t = pd.DatetimeIndex(hours)
    hod = 2 * np.pi * t.hour.to_numpy() / 24.0
    doy = 2 * np.pi * (t.dayofyear.to_numpy() - 1) / 365.25
    cols["hod_sin"], cols["hod_cos"] = np.sin(hod), np.cos(hod)
    cols["doy_sin"], cols["doy_cos"] = np.sin(doy), np.cos(doy)

    return pd.DataFrame({k: v.astype(np.float32) for k, v in cols.items()}, index=t)


def persistence(hourly: pd.DataFrame, target: str, lead: int, hours: pd.DatetimeIndex,
                index: pd.DatetimeIndex) -> pd.DataFrame:
    """snow_occ and y_swe (Phase 4 definitions) at t - L, from the target station's hourly record."""
    h = hourly[hourly.station.astype(str) == target].set_index("hour_ending").reindex(index)
    avail = h.has_ob.fillna(False).astype(bool) & h.p01i.notna()
    snow = (h.wx_snow == 1) & avail
    occ = np.where(avail, snow.astype(float), np.nan)
    swe = np.where(avail, np.where(snow, h.p01i.fillna(0.0), 0.0), np.nan)
    src = ((pd.DatetimeIndex(hours) - index[0]) // pd.Timedelta(hours=1)).to_numpy() - lead
    ok = src >= 0
    po, ps = np.full(len(src), np.nan), np.full(len(src), np.nan)
    po[ok], ps[ok] = occ[src[ok]], swe[src[ok]]
    return pd.DataFrame({"persist_occ": po, "persist_swe": ps})


def load_target(cfg, hourly: pd.DataFrame) -> pd.DataFrame:
    if TARGET_PATH.exists():
        return pd.read_parquet(TARGET_PATH)
    return build_target(hourly[hourly.station.astype(str) == cfg.target_station], cfg, load_splits())


def build_lead(hourly, stations, target_df, cfg, lead: int) -> pd.DataFrame:
    index = pd.date_range(INDEX_START, INDEX_END, freq="h")
    tg = target_df[target_df.target_available].sort_values("hour_ending").reset_index(drop=True)
    hours = pd.DatetimeIndex(tg.hour_ending)
    feats = feature_frame(hourly, stations, cfg.target_station, lead, hours,
                          lags_dtw=cfg.lags["target_station"], lags_nb=tuple(cfg.lags["neighbor"]),
                          index=index).reset_index(drop=True)
    meta = tg[["hour_ending", "season", "split", "snow_occ", "y_swe", "y_depth", "wx_mixed"]].copy()
    meta = pd.concat([meta, persistence(hourly, cfg.target_station, lead, hours, index)], axis=1)
    return pd.concat([meta, feats], axis=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lead", default="all", choices=["all", "1", "3"])
    args = ap.parse_args(argv)
    try:
        cfg = load_config()
        leads = list(cfg.lead_hours) if args.lead == "all" else [int(args.lead)]
        stations = pd.read_csv(STATIONS_PATH)
        hourly = pd.read_parquet(HOURLY_PATH, columns=["station", "hour_ending", "season", "wx_mixed", "p01i_trace"] + NEEDED_VARS)
        hourly = hourly[hourly.station.astype(str).isin(set(stations.sid))]
        target_df = load_target(cfg, hourly)
        for lead in leads:
            df = build_lead(hourly, stations, target_df, cfg, lead)
            feat_cols = [c for c in df.columns if c not in TARGET_COLS]
            out = features_path(lead)
            tmp = out.with_suffix(".parquet.part")
            df.to_parquet(tmp, index=False)
            tmp.rename(out)
            fl = feature_list_path(lead)
            tmpj = fl.with_suffix(".json.part")
            tmpj.write_text(json.dumps(feat_cols, indent=0) + "\n")
            tmpj.rename(fl)
            print(f"lead {lead}: {len(df):,} rows, {len(feat_cols):,} features, "
                  f"{out.stat().st_size / 1e6:.0f} MB")
            del df
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAILED features: {e}")
        return 1
    print("DONE features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
