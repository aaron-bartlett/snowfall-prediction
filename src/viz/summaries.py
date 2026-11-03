"""Tier B: compute small derived tables from local pipeline outputs into reports/summaries/.

Needs data/processed/{hourly,target,features_L*}.parquet and data/raw/ghcnd. Run once; every docs
figure is then drawn from committed files only.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.clean import HOURLY_PATH
from src.config import ROOT, load_config
from src.evaluate import SEED, SHAP_ROWS
from src.features import feature_list_path, features_path, plane_fit_gradient, wind_from_deg
from src.model import HurdleModel
from src.target import RAW_GHCN, TARGET_PATH, daily_compare, parse_ghcn
from src.train import model_dir
from src.viz.names import frame
from src.viz.style import SUMMARIES

DEPENDENCE = {1: ["DTW_ceiling_ft_lag1", "agg_snow_frac_r0_100_lag1", "agg_upstream_snow_frac_lag1", "AZO_tmpf_lag1"],
              3: ["SBN_tmpf_lag1", "agg_snow_frac_r100_200_lag1", "agg_sector_W_snow_frac_lag1", "DTW_ceiling_ft_lag1"]}
STORM = ("2016-02-24 13:00", "2016-02-25 18:00")


def _write(df: pd.DataFrame, name: str) -> None:
    path = SUMMARIES / name
    (df.to_parquet(path, index=False) if name.endswith(".parquet") else df.to_csv(path, index=False))
    print(f"  wrote reports/summaries/{name} ({path.stat().st_size / 1e3:.0f} KB)", flush=True)


def availability(hourly: pd.DataFrame, cfg) -> None:
    h = hourly[hourly.hour_ending.dt.month.isin(cfg.target_months)].copy()
    h["month"] = h.hour_ending.dt.month
    h["raw_p01i"] = h.p01i.notna() & ~h.p01i_filled.fillna(False).astype(bool)
    h["usable_p01i"] = h.p01i.notna()
    h["has_ob"] = h.has_ob.fillna(False).astype(bool)
    k = h[h.station.astype(str) == cfg.target_station]
    _write(k.groupby(["season", "month"])[["has_ob", "raw_p01i", "usable_p01i"]].mean().reset_index(),
           "availability_kdtw.csv")
    _write(h.groupby(["station", "season"], observed=True)[["has_ob", "usable_p01i"]].mean().reset_index(),
           "availability_stations.csv")


def target_tables(target: pd.DataFrame, cfg) -> None:
    t = target[target.target_available].copy()
    y = t.y_swe.to_numpy()
    t["cls"] = np.select([t.snow_occ == 0, y <= 0, y <= 0.02, y <= 0.05], ["no snow", "trace", "0.01-0.02", "0.03-0.05"], ">=0.06")
    _write(t.groupby(["split", "cls"]).size().rename("hours").reset_index(), "target_hist.csv")
    t["month"] = t.hour_ending.dt.month
    _write(t.groupby(["split", "month"]).snow_occ.agg(["mean", "size"]).reset_index()
           .rename(columns={"mean": "snow_rate", "size": "hours"}), "target_monthly.csv")
    ghcn = parse_ghcn(RAW_GHCN / f"{cfg.ghcn_id}.csv.gz")
    d = daily_compare(target, ghcn, cfg.lst_offset_hours).reset_index()
    _write(d[["date", "season", "split", "hours", "snow_hours", "y_swe", "y_depth", "snow", "prcp"]], "ghcn_daily.csv")


def example_hour(hourly: pd.DataFrame, stations: pd.DataFrame) -> None:
    h = hourly[(hourly.hour_ending >= pd.Timestamp(STORM[0], tz="UTC")) &
               (hourly.hour_ending <= pd.Timestamp(STORM[1], tz="UTC"))]
    h = h[h.station.astype(str).isin(set(stations.sid))]
    best = h.groupby("hour_ending").wx_snow.sum().idxmax()
    x = h[h.hour_ending == best].assign(sid=lambda d: d.station.astype(str))
    x = stations[["sid", "lat", "lon", "dist_km", "bearing_deg", "x_km", "y_km"]].merge(
        x[["sid", "hour_ending", "tmpf", "mslp", "u", "v", "wx_snow", "p01i"]], on="sid", how="left")
    p = x.mslp.to_numpy(float)[None, :]
    mag, direc = plane_fit_gradient(p, x.x_km.to_numpy(float), x.y_km.to_numpy(float))
    ok = x.sid != "DTW"
    wfrom = wind_from_deg(np.nanmean(x.u[ok]), np.nanmean(x.v[ok]))
    x["hour_ending"] = best
    x["grad_mag"], x["grad_dir"], x["wind_from"] = float(mag[0]), float(direc[0]), float(wfrom)
    _write(x, "example_hour.csv")


def shap_tables(lead: int) -> None:
    import shap
    model = HurdleModel.load(model_dir(lead))
    feats = json.loads(feature_list_path(lead).read_text())
    df = pd.read_parquet(features_path(lead), columns=["hour_ending"] + feats,
                         filters=[("split", "==", "test")]).sort_values("hour_ending").reset_index(drop=True)
    rng = np.random.default_rng(SEED)            # same rows as src.evaluate.shap_figure
    idx = np.sort(rng.choice(len(df), size=min(SHAP_ROWS, len(df)), replace=False))
    X = model.matrix(df.iloc[idx])
    sv = shap.TreeExplainer(model.stage1).shap_values(X)
    sv = np.asarray(sv[-1] if isinstance(sv, list) else sv)
    sv = sv[..., -1] if sv.ndim == 3 else sv
    imp = frame(model.features).assign(mean_abs_shap=np.abs(sv).mean(0))
    _write(imp, f"shap_importance_L{lead}.csv")
    cols = {}
    for f in DEPENDENCE[lead]:
        j = model.features.index(f)
        cols[f"{f}__value"], cols[f"{f}__shap"] = X[:, j], sv[:, j]
    _write(pd.DataFrame(cols).astype("float32"), f"shap_dependence_L{lead}.parquet")


def run() -> None:
    cfg = load_config()
    SUMMARIES.mkdir(parents=True, exist_ok=True)
    stations = pd.read_csv(ROOT / "meta" / "stations.csv")
    hourly = pd.read_parquet(HOURLY_PATH, columns=["station", "hour_ending", "season", "has_ob", "p01i", "p01i_filled",
                                                   "tmpf", "mslp", "u", "v", "wx_snow"])
    availability(hourly, cfg)
    example_hour(hourly, stations)
    del hourly
    target_tables(pd.read_parquet(TARGET_PATH), cfg)
    for lead in cfg.lead_hours:
        shap_tables(lead)
