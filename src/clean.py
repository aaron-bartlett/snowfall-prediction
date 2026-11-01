"""Parse, clean and QC raw IEM station-season CSVs into one hourly long table.

Usage: python -m src.clean
Output: data/processed/hourly.parquet, reports/p01i_missing_diagnostic.md, reports/qc_summary.csv
"""
from __future__ import annotations

import re
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import ROOT, load_config, season_label
from src.download import asos_path
from src.stations import STATIONS_PATH
from src.timeutil import assign_hour_ending, keep_latest_per_hour

PROCESSED = ROOT / "data" / "processed"
HOURLY_PATH = PROCESSED / "hourly.parquet"
DIAG_PATH = ROOT / "reports" / "p01i_missing_diagnostic.md"
QC_PATH = ROOT / "reports" / "qc_summary.csv"
PCOUNTS_PATH = ROOT / "reports" / "p01i_missing_counts.csv"

NUMERIC = ["tmpf", "dwpf", "relh", "drct", "sknt", "gust", "mslp", "alti", "vsby", "p01i", "snowdepth"]
WX_FLAGS = ["wx_snow", "wx_blsn", "wx_rain", "wx_fzra", "wx_pl", "wx_up", "wx_fog_br", "wx_ts",
            "wx_mixed", "wx_precip", "snow_intensity"]
SKY_ORD = {"CLR": 0, "SKC": 0, "NSC": 0, "NCD": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4, "VV": 4, "VV ": 4}
NO_CEILING_FT = 20000.0
RANGES = {"tmpf": (-40, 110), "sknt": (0, 100), "mslp": (950, 1060), "alti": (27.5, 31.5),
          "vsby": (0, 10), "p01i": (0, 3)}
SPIKE = {"tmpf": 15.0, "mslp": 8.0}
PNO_RE = re.compile(r"\bPNO\b")
PGROUP_RE = re.compile(r"\sP(\d{4})(?=\s|\$|$)")
OUT_COLS = (["station", "hour_ending", "season", "has_ob"]
            + ["tmpf", "dwpf", "relh", "dewpoint_depression", "drct", "sknt", "gust", "u", "v",
               "mslp", "alti", "vsby", "sky_cover", "ceiling_ft", "p01i", "p01i_trace", "p01i_filled",
               "snowdepth"] + WX_FLAGS)


# --- present weather ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def parse_wx(wx: str) -> tuple:
    """Parse an IEM `wxcodes` string into flags (order = WX_FLAGS)."""
    f = dict.fromkeys(WX_FLAGS, 0)
    for tok in (wx or "").split():
        if tok.startswith("VC"):
            continue
        sign = tok[0] if tok[0] in "+-" else ""
        body = tok.lstrip("+-")
        if body.startswith("VC"):
            continue
        if "BLSN" in body or "DRSN" in body:
            f["wx_blsn"] = 1
            body = body.replace("BLSN", "").replace("DRSN", "")
        if "SN" in body or "SG" in body:
            f["wx_snow"] = 1
            f["snow_intensity"] = max(f["snow_intensity"], {"-": 1, "": 2, "+": 3}[sign])
        if "FZRA" in body or "FZDZ" in body:
            f["wx_fzra"] = 1
        elif "RA" in body or "DZ" in body:
            f["wx_rain"] = 1
        if "PL" in body or "GS" in body or "GR" in body:
            f["wx_pl"] = 1
        if "UP" in body:
            f["wx_up"] = 1
        if "FG" in body or "BR" in body:
            f["wx_fog_br"] = 1
        if "TS" in body:
            f["wx_ts"] = 1
        if any(p in body for p in ("RA", "SN", "DZ", "PL", "GR", "GS", "SG", "IC", "UP")):
            f["wx_precip"] = 1
    f["wx_mixed"] = int(f["wx_snow"] and (f["wx_rain"] or f["wx_fzra"] or f["wx_pl"]))
    return tuple(f[k] for k in WX_FLAGS)


def wx_frame(wxcodes: pd.Series) -> pd.DataFrame:
    """Flags for each row. `wxcodes` of 'M' on an existing ob means no weather."""
    codes = wxcodes.fillna("M").replace("M", "")
    uniq = pd.Series(codes.unique())
    table = pd.DataFrame([parse_wx(c) for c in uniq], columns=WX_FLAGS, index=uniq)
    return table.loc[codes.values].set_index(codes.index)


# --- wind & sky ---------------------------------------------------------------------------------
def wind_uv(drct: pd.Series, sknt: pd.Series) -> tuple[pd.Series, pd.Series]:
    """u = -s sin(d), v = -s cos(d) (knots). Calm -> 0. Missing direction with wind -> NaN."""
    rad = np.radians(drct)
    u = -sknt * np.sin(rad)
    v = -sknt * np.cos(rad)
    calm = sknt == 0
    u[calm], v[calm] = 0.0, 0.0
    return u.round(6), v.round(6)


def sky_fields(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    covers = [df[f"skyc{i}"].where(df[f"skyc{i}"] != "M") for i in range(1, 5)]
    heights = [pd.to_numeric(df[f"skyl{i}"].replace("M", np.nan), errors="coerce") for i in range(1, 5)]
    ords = pd.concat([c.str.strip().map(SKY_ORD) for c in covers], axis=1)
    cover = ords.max(axis=1)
    ceil_layers = pd.concat([h.where(c.str.strip().isin(["BKN", "OVC", "VV"])) for c, h in zip(covers, heights)],
                            axis=1)
    ceiling = ceil_layers.min(axis=1)
    has_sky = covers[0].notna()
    ceiling = ceiling.where(ceiling.notna() | ~has_sky, NO_CEILING_FT)
    return cover.where(has_sky), ceiling.where(has_sky)


# --- p01i ---------------------------------------------------------------------------------------
def fill_p01i(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fill missing p01i on existing obs using the raw METAR (categories a-d, e = no METAR)."""
    raw = df["p01i_raw"]
    trace = raw == "T"
    p = pd.to_numeric(raw.replace({"M": np.nan, "T": 0.0}), errors="coerce")
    missing = p.isna() & ~trace
    metar = df["metar"].fillna("M")
    rmk = metar.str.split("RMK", n=1).str[1].fillna("")
    has_metar = metar != "M"
    pno = rmk.str.contains(PNO_RE)
    pgrp = rmk.str.extract(PGROUP_RE, expand=False)
    precip_wx = df["wx_precip"] == 1

    cat_a = missing & has_metar & pno
    cat_b = missing & has_metar & ~pno & pgrp.notna()
    cat_c = missing & has_metar & ~pno & pgrp.isna() & precip_wx
    cat_d = missing & has_metar & ~pno & pgrp.isna() & ~precip_wx
    cat_e = missing & ~has_metar

    filled = pd.Series(False, index=df.index)
    bval = pd.to_numeric(pgrp, errors="coerce") / 100.0
    p[cat_b] = bval[cat_b]
    trace = trace | (cat_b & (bval == 0))
    p[cat_c] = 0.0
    trace = trace | cat_c
    p[cat_d] = 0.0
    filled[cat_b | cat_c | cat_d] = True
    df = df.assign(p01i=p, p01i_trace=trace.astype(float), p01i_filled=filled.astype(float))
    counts = {"obs": len(df), "raw_missing": int(missing.sum()), "a_pno": int(cat_a.sum()),
              "b_pgroup_parsed": int(cat_b.sum()), "c_precip_wx_no_pgroup": int(cat_c.sum()),
              "d_dry_no_pgroup": int(cat_d.sum()), "e_no_metar": int(cat_e.sum())}
    return df, counts


# --- QC -----------------------------------------------------------------------------------------
def qc(grid: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Range checks, dewpoint check, and spike checks on a complete hourly grid (one station-season)."""
    stats = {}
    g = grid.copy()
    for col, (lo, hi) in RANGES.items():
        if col == "vsby":
            g[col] = g[col].clip(upper=10)
        bad = g[col].notna() & ((g[col] < lo) | (g[col] > hi))
        stats[f"range_{col}"] = int(bad.sum())
        g.loc[bad, col] = np.nan
    bad = g.dwpf.notna() & g.tmpf.notna() & (g.dwpf > g.tmpf + 1)
    stats["dwpf_gt_tmpf"] = int(bad.sum())
    g.loc[bad, "dwpf"] = np.nan
    for col, thr in SPIKE.items():
        x = g[col]
        d_prev = (x - x.shift(1)).abs()
        d_next = (x - x.shift(-1)).abs()
        # A spike differs sharply from both neighbors; with one neighbor we can't tell which is bad.
        spike = (d_prev > thr) & (d_next > thr)
        stats[f"spike_{col}"] = int(spike.sum())
        g.loc[spike, col] = np.nan
    return g, stats


def season_grid(year: int) -> pd.DatetimeIndex:
    """Hour-ending grid for a download window: Sep 1 01Z (year) .. May 1 00Z (year + 1)."""
    return pd.date_range(f"{year}-09-01 01:00", f"{year + 1}-05-01 00:00", freq="h", tz="UTC")


def clean_file(path, sid: str, year: int) -> tuple[pd.DataFrame, dict, dict]:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw["valid"] = pd.to_datetime(raw["valid"], utc=True)
    raw["station"] = sid
    raw["hour_ending"] = assign_hour_ending(raw["valid"])
    df = keep_latest_per_hour(raw).reset_index(drop=True)
    df["p01i_raw"] = df["p01i"]
    for c in NUMERIC:
        if c == "p01i":
            continue
        df[c] = pd.to_numeric(df[c].replace({"M": np.nan, "T": np.nan, "": np.nan}), errors="coerce")
    df = pd.concat([df, wx_frame(df["wxcodes"])], axis=1)
    df, pcounts = fill_p01i(df)
    df["u"], df["v"] = wind_uv(df["drct"], df["sknt"])
    df["sky_cover"], df["ceiling_ft"] = sky_fields(df)
    df["has_ob"] = True

    grid = pd.DataFrame(index=season_grid(year))
    grid.index.name = "hour_ending"
    keep = [c for c in OUT_COLS if c in df.columns and c not in ("station", "season", "hour_ending")]
    grid = grid.join(df.set_index("hour_ending")[keep])
    grid["has_ob"] = grid["has_ob"].fillna(False).astype(bool)
    grid, qstats = qc(grid)
    grid["dewpoint_depression"] = grid["tmpf"] - grid["dwpf"]
    grid = grid.reset_index()
    grid["station"] = sid
    grid["season"] = season_label(year)
    for c in OUT_COLS:
        if c not in grid.columns:
            grid[c] = np.nan
    grid = grid[OUT_COLS]
    num = [c for c in OUT_COLS if c not in ("station", "hour_ending", "season", "has_ob")]
    grid[num] = grid[num].astype("float32")
    pcounts.update(station=sid, season=season_label(year))
    qstats.update(station=sid, season=season_label(year), obs=int(grid.has_ob.sum()))
    return grid, pcounts, qstats


def write_diagnostic(pc: pd.DataFrame, target: str) -> None:
    by_st = pc.groupby("station")[[c for c in pc.columns if c not in ("station", "season")]].sum().astype(int)
    by_st["raw_missing_pct"] = 100 * by_st.raw_missing / by_st.obs
    t = by_st.loc[target]
    lines = ["# p01i missing-value diagnostic", "",
             "Categories for obs whose raw `p01i` was missing (`M`):",
             "- (a) `PNO` in remarks → left NaN (gauge not operating)",
             "- (b) `Prrrr` group present → parsed from the METAR",
             "- (c) precipitation in `wxcodes` but no `P` group → 0.0, trace",
             "- (d) no precipitation and no `P` group → 0.0 (dry hour)",
             "- (e) no raw METAR available → left NaN", "",
             f"## Target station {target}", "",
             "| obs | raw missing | (a) PNO | (b) P group | (c) precip wx | (d) dry | (e) no METAR |",
             "|---|---|---|---|---|---|---|",
             f"| {int(t.obs):,} | {int(t.raw_missing):,} ({t.raw_missing_pct:.1f}%) | {int(t.a_pno):,} | "
             f"{int(t.b_pgroup_parsed):,} | {int(t.c_precip_wx_no_pgroup):,} | {int(t.d_dry_no_pgroup):,} | "
             f"{int(t.e_no_metar):,} |", "",
             f"### {target} by season", "",
             "| season | obs | raw missing | (a) | (b) | (c) | (d) | (e) |", "|---|---|---|---|---|---|---|---|"]
    for r in pc[pc.station == target].itertuples():
        lines.append(f"| {r.season} | {r.obs} | {r.raw_missing} | {r.a_pno} | {r.b_pgroup_parsed} | "
                     f"{r.c_precip_wx_no_pgroup} | {r.d_dry_no_pgroup} | {r.e_no_metar} |")
    lines += ["", "## All stations", "",
              "| station | obs | raw missing % | (a) | (b) | (c) | (d) | (e) |", "|---|---|---|---|---|---|---|---|"]
    for sid, r in by_st.iterrows():
        lines.append(f"| {sid} | {int(r.obs):,} | {r.raw_missing_pct:.1f} | {int(r.a_pno):,} | "
                     f"{int(r.b_pgroup_parsed):,} | {int(r.c_precip_wx_no_pgroup):,} | {int(r.d_dry_no_pgroup):,} | "
                     f"{int(r.e_no_metar):,} |")
    DIAG_PATH.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    try:
        cfg = load_config()
        sids = pd.read_csv(STATIONS_PATH).sid.tolist()
        frames, pcs, qcs = [], [], []
        jobs = [(s, y) for s in sids for y in cfg.season_years]
        for i, (sid, y) in enumerate(jobs, 1):
            path = asos_path(sid, y)
            if not path.exists():
                print(f"  missing raw file {path.name}; station-season left empty", flush=True)
                continue
            g, pc, q = clean_file(path, sid, y)
            frames.append(g)
            pcs.append(pc)
            qcs.append(q)
            if i % 50 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] cleaned", flush=True)
        hourly = pd.concat(frames, ignore_index=True)
        hourly["station"] = hourly["station"].astype("category")
        hourly["season"] = hourly["season"].astype("category")
        PROCESSED.mkdir(parents=True, exist_ok=True)
        tmp = HOURLY_PATH.with_suffix(".parquet.part")
        hourly.to_parquet(tmp, index=False)
        tmp.rename(HOURLY_PATH)
        write_diagnostic(pd.DataFrame(pcs), cfg.target_station)
        pd.DataFrame(pcs).to_csv(PCOUNTS_PATH, index=False)
        pd.DataFrame(qcs).to_csv(QC_PATH, index=False)
        print(f"hourly rows: {len(hourly):,}; stations: {hourly.station.nunique()}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAILED clean: {e}")
        return 1
    print("DONE clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
