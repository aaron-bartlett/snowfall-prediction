"""Select input stations: IEM network metadata -> candidates within radius -> coverage filter.

Usage: python -m src.stations [--map]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from src.config import ROOT, load_config
from src.geo import bearing_deg, haversine_km, local_xy_km
from src.timeutil import assign_hour_ending, target_hours

GEOJSON_URL = "https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"
ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
NET_DIR = ROOT / "data" / "meta" / "networks"
COV_DIR = ROOT / "data" / "meta" / "coverage"
CANDIDATES_PATH = ROOT / "meta" / "station_candidates.csv"
STATIONS_PATH = ROOT / "meta" / "stations.csv"
MAP_PATH = ROOT / "reports" / "figures" / "station_map.png"

STATION_COLS = ["sid", "name", "network", "lat", "lon", "elev_m", "dist_km", "bearing_deg",
                "x_km", "y_km", "temp_cov", "precip_cov"]


def fetch_networks(networks) -> pd.DataFrame:
    NET_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for net in networks:
        path = NET_DIR / f"{net}.geojson"
        if not path.exists():
            r = requests.get(GEOJSON_URL.format(network=net), timeout=120)
            r.raise_for_status()
            path.write_text(r.text)
        for f in json.loads(path.read_text())["features"]:
            p = f["properties"]
            lon, lat = f["geometry"]["coordinates"][:2]
            rows.append({"sid": p["sid"], "name": (p.get("sname") or "").strip(), "network": net,
                         "lat": lat, "lon": lon, "elev_m": p.get("elevation"),
                         "archive_begin": p.get("archive_begin"), "archive_end": p.get("archive_end")})
    return pd.DataFrame(rows).drop_duplicates("sid")


def add_geometry(df: pd.DataFrame, target: str) -> pd.DataFrame:
    t = df.loc[df.sid == target].iloc[0]
    df = df.copy()
    df["dist_km"] = haversine_km(t.lat, t.lon, df.lat, df.lon).round(1)
    df["bearing_deg"] = bearing_deg(t.lat, t.lon, df.lat, df.lon).round(1)
    df.loc[df.sid == target, "bearing_deg"] = 0.0
    x, y = local_xy_km(df.lat, df.lon, t.lat, t.lon)
    df["x_km"], df["y_km"] = x.round(2), y.round(2)
    return df


def total_target_hours(season_years) -> int:
    return sum(len(target_hours(y)) for y in season_years)


def coverage_from_csv(text: str, season_years) -> tuple[float, float]:
    """Fraction of target hours with a valid tmpf / p01i (CSV with columns station,valid,tmpf,p01i)."""
    df = pd.read_csv(io.StringIO(text), dtype=str)
    if df.empty:
        return 0.0, 0.0
    he = assign_hour_ending(pd.to_datetime(df["valid"], utc=True))
    df = df.assign(hour_ending=he)
    df = df[df.hour_ending.notna()]
    lo = pd.Timestamp(f"{min(season_years)}-10-01", tz="UTC")
    hi = pd.Timestamp(f"{max(season_years) + 1}-05-01", tz="UTC")
    df = df[(df.hour_ending >= lo) & (df.hour_ending < hi)
            & df.hour_ending.dt.month.isin([10, 11, 12, 1, 2, 3, 4])]
    total = total_target_hours(season_years)
    t = df.loc[df.tmpf.notna() & (df.tmpf != "M"), "hour_ending"].nunique()
    p = df.loc[df.p01i.notna() & (df.p01i != "M"), "hour_ending"].nunique()
    return t / total, p / total


def coverage_url_params(sid: str, season_years) -> dict:
    first, last = min(season_years), max(season_years)
    return {"station": sid, "data": "tmpf,p01i",
            "year1": first, "month1": 9, "day1": 30,
            "year2": last + 1, "month2": 5, "day2": 1,
            "tz": "Etc/UTC", "format": "onlycomma", "missing": "M", "trace": "T", "report_type": 3}


def station_coverage(sid: str, season_years, retries: int = 4) -> dict:
    COV_DIR.mkdir(parents=True, exist_ok=True)
    path = COV_DIR / f"{sid}.json"
    if path.exists():
        return json.loads(path.read_text())
    err = None
    for attempt in range(retries):
        try:
            r = requests.get(ASOS_URL, params=coverage_url_params(sid, season_years), timeout=900)
            r.raise_for_status()
            if not r.text.startswith("station,valid"):
                raise RuntimeError(f"unexpected response: {r.text[:80]!r}")
            t, p = coverage_from_csv(r.text, season_years)
            res = {"sid": sid, "temp_cov": t, "precip_cov": p}
            path.write_text(json.dumps(res))
            return res
        except Exception as e:  # noqa: BLE001 - retry any network/parse failure
            err = e
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"coverage failed for {sid}: {err}")


def plot_map(cand: pd.DataFrame, sel: pd.DataFrame, target: str, radius_km: float, period: str):
    import matplotlib
    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from cartopy.geodesic import Geodesic

    tgt = cand[cand.sid == target].iloc[0]
    keep = sel[sel.sid != target]
    drop = cand[~cand.sid.isin(sel.sid)]
    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 11), dpi=150)
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=tgt.lon, central_latitude=tgt.lat))
    ax.set_extent([-88.3, -78.6, 39.2, 45.2], crs=pc)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f4f1ea")
    ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="#cfe3f1", edgecolor="#7aa6c2", lw=0.5)
    ax.add_feature(cfeature.STATES.with_scale("10m"), edgecolor="#888", lw=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="#444", lw=0.9)
    ring = Geodesic().circle(lon=tgt.lon, lat=tgt.lat, radius=radius_km * 1000, n_samples=360)
    ax.plot(ring[:, 0], ring[:, 1], transform=pc, ls="--", color="#555", lw=1, label=f"{radius_km:.0f} km radius")
    ax.scatter(drop.lon, drop.lat, transform=pc, marker="x", s=22, color="#9a9a9a", lw=1,
               label=f"Excluded: below coverage thresholds ({len(drop)})", zorder=3)
    colors = {"MI_ASOS": "#2f6db3", "OH_ASOS": "#c0392b", "IN_ASOS": "#2e8b57", "CA_ON_ASOS": "#8e44ad"}
    names = {"MI_ASOS": "Michigan", "OH_ASOS": "Ohio", "IN_ASOS": "Indiana", "CA_ON_ASOS": "Ontario"}
    for net, g in keep.groupby("network"):
        ax.scatter(g.lon, g.lat, transform=pc, s=34, color=colors.get(net, "k"), edgecolor="white", lw=0.6,
                   label=f"Input station - {names.get(net, net)} ({len(g)})", zorder=4)
        for _, r in g.iterrows():
            ax.text(r.lon + 0.05, r.lat + 0.04, r.sid, transform=pc, fontsize=6.5, color="#222", zorder=5)
    ax.scatter([tgt.lon], [tgt.lat], transform=pc, marker="*", s=420, color="#f1c40f",
               edgecolor="black", lw=1, label=f"Target: K{target}", zorder=6)
    ax.text(tgt.lon - 0.62, tgt.lat - 0.22, f"K{target}", transform=pc, fontsize=10, weight="bold", zorder=6)
    ax.set_title(f"Snowfall prediction stations: target K{target} and {len(keep)} input stations\n"
                 f"ASOS/AWOS within {radius_km:.0f} km, ≥80% temperature & ≥75% hourly precip coverage, "
                 f"{period} (source: IEM)", fontsize=11)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
    tmp = MAP_PATH.with_name(MAP_PATH.stem + ".part.png")
    plt.savefig(tmp, bbox_inches="tight")
    plt.close(fig)
    tmp.rename(MAP_PATH)


def write_csv(df: pd.DataFrame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.part")
    df.to_csv(tmp, index=False)
    tmp.rename(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", action="store_true", help="also redraw station_map.png")
    args = ap.parse_args(argv)
    try:
        cfg = load_config()
        years = cfg.season_years
        allst = add_geometry(fetch_networks(cfg.networks), cfg.target_station)
        cand = allst[allst.dist_km <= cfg.radius_km].sort_values("dist_km").reset_index(drop=True)
        print(f"{len(cand) - 1} candidates within {cfg.radius_km:.0f} km (+ target)", flush=True)

        done = 0

        def run(sid):
            nonlocal done
            res = station_coverage(sid, years, cfg.download["retries"])
            done += 1
            print(f"[{done}/{len(cand)}] {sid} temp={res['temp_cov']:.3f} precip={res['precip_cov']:.3f}", flush=True)
            return res

        with ThreadPoolExecutor(4) as ex:
            cov = pd.DataFrame(list(ex.map(run, cand.sid)))
        cand = cand.merge(cov, on="sid")
        write_csv(cand[STATION_COLS + ["archive_begin", "archive_end"]], CANDIDATES_PATH)

        ok = (cand.temp_cov >= cfg.coverage["min_temp"]) & (cand.precip_cov >= cfg.coverage["min_precip"])
        sel = cand[ok | (cand.sid == cfg.target_station)].sort_values("dist_km").reset_index(drop=True)
        write_csv(sel[STATION_COLS], STATIONS_PATH)
        print(f"selected {len(sel) - 1} input stations + target; by network: "
              f"{sel[sel.sid != cfg.target_station].network.value_counts().to_dict()}")
        if args.map:
            period = f"Oct–Apr {years[0]}–{years[-1] + 1}"
            plot_map(cand, sel, cfg.target_station, cfg.radius_km, period)
            print(f"map written to {MAP_PATH}")
    except Exception as e:
        print(f"FAILED stations: {e}")
        return 1
    print("DONE stations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
