"""Build the KDTW hourly snowfall target and validate it against GHCN-Daily snowfall.

Usage: python -m src.target
Output: data/processed/target.parquet, meta/target_calibration.json,
        reports/target_validation.md, reports/figures/target_vs_ghcn.png
"""
from __future__ import annotations

import gzip
import json
import sys

import numpy as np
import pandas as pd

from src.clean import HOURLY_PATH, PROCESSED
from src.config import ROOT, load_config
from src.download import RAW_GHCN
from src.splits import load_splits, season_to_split
from src.timeutil import TARGET_MONTHS, lst_date

TARGET_PATH = PROCESSED / "target.parquet"
CALIB_PATH = ROOT / "meta" / "target_calibration.json"
REPORT_PATH = ROOT / "reports" / "target_validation.md"
FIG_PATH = ROOT / "reports" / "figures" / "target_vs_ghcn.png"
MM_PER_IN = 25.4
MIN_HOURS_PER_DAY = 20   # days with fewer available target hours are excluded from GHCN comparisons
GHCN_COLS = ["id", "date", "element", "value", "mflag", "qflag", "sflag", "obstime"]


def slr(tmpf, table, fallback) -> np.ndarray:
    """Snow-to-liquid ratio from surface temperature (°F): first bucket with T < bound; NaN -> fallback."""
    t = np.asarray(tmpf, dtype=float)
    out = np.full(t.shape, float(fallback))
    assigned = np.isnan(t)
    for bound, ratio in table:
        m = ~assigned & (t < bound)
        out[m] = ratio
        assigned |= m
    return out


def build_target(hourly: pd.DataFrame, cfg, splits: dict) -> pd.DataFrame:
    t = hourly[(hourly.station == cfg.target_station)
               & hourly.hour_ending.dt.month.isin(TARGET_MONTHS)].copy()
    t = t.sort_values("hour_ending").reset_index(drop=True)
    avail = t.has_ob & t.p01i.notna()
    snow = (t.wx_snow == 1) & avail
    y_swe = np.where(snow, t.p01i.fillna(0.0), 0.0)
    out = pd.DataFrame({
        "hour_ending": t.hour_ending,
        "season": t.season.astype(str),
        "target_available": avail.to_numpy(),
        "snow_occ": np.where(avail, snow.astype(float), np.nan),
        "y_swe": np.where(avail, y_swe, np.nan),
        "tmpf": t.tmpf.to_numpy(dtype=float),
        "wx_mixed": np.where(avail, t.wx_mixed, np.nan),
        "p01i_trace": np.where(avail, t.p01i_trace, np.nan),
    })
    out["y_depth"] = out.y_swe * slr(out.tmpf, cfg.slr["table"], cfg.slr["fallback"])
    out["split"] = out.season.map(season_to_split(splits))
    if out.split.isna().any():
        raise ValueError(f"seasons without a split: {out.loc[out.split.isna(), 'season'].unique()}")
    return out


def parse_ghcn(path) -> pd.DataFrame:
    """Daily SNOW, SNWD (mm) and PRCP (tenths of mm) -> inches; -9999 and QC-flagged values dropped."""
    with gzip.open(path, "rt") as fh:
        g = pd.read_csv(fh, header=None, names=GHCN_COLS, dtype=str, keep_default_na=False)
    g = g[g.element.isin(["SNOW", "SNWD", "PRCP"])]
    g = g[(g.value != "-9999") & (g.qflag.str.strip() == "")]
    g["value"] = g.value.astype(float)
    g.loc[g.element == "PRCP", "value"] /= 10.0
    g["value"] /= MM_PER_IN
    wide = g.pivot_table(index="date", columns="element", values="value", aggfunc="first")
    wide.index = pd.to_datetime(wide.index, format="%Y%m%d").date
    wide.index.name = "date"
    return wide.rename(columns=str.lower)


def daily_compare(target: pd.DataFrame, ghcn: pd.DataFrame, offset_hours: int) -> pd.DataFrame:
    t = target.assign(date=lst_date(target.hour_ending, offset_hours))
    d = t.groupby("date").agg(
        hours=("target_available", "sum"), y_swe=("y_swe", "sum"), y_depth=("y_depth", "sum"),
        snow_hours=("snow_occ", "sum"), season=("season", "first"), split=("split", "first"))
    d = d.join(ghcn, how="inner")
    return d[(d.hours >= MIN_HOURS_PER_DAY) & d.snow.notna()]


def season_table(d: pd.DataFrame) -> pd.DataFrame:
    snowdays = d[d.snow >= 0.1]
    s = d.groupby("season").agg(split=("split", "first"), days=("hours", "size"),
                                y_depth=("y_depth", "sum"), ghcn_snow=("snow", "sum"))
    s["depth_ratio"] = s.y_depth / s.ghcn_snow
    sd = snowdays.groupby("season").agg(y_swe_snowdays=("y_swe", "sum"), ghcn_prcp_snowdays=("prcp", "sum"))
    s = s.join(sd)
    s["swe_ratio"] = s.y_swe_snowdays / s.ghcn_prcp_snowdays
    return s


def plot(d: pd.DataFrame, r: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=130)
    ax = axes[0]
    m = (d.snow > 0) | (d.y_depth > 0)
    ax.scatter(d.snow[m], d.y_depth[m], s=8, alpha=0.4, color="#2f6db3", edgecolor="none")
    lim = max(d.snow.max(), d.y_depth.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#555", lw=1, ls="--", label="1:1")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("GHCN-Daily snowfall (in)"); ax.set_ylabel("Σ hourly derived snow depth (in)")
    ax.set_title(f"Daily totals, KDTW (LST days)\nPearson r = {r:.2f} on days where either ≥ 1\"")
    ax.legend(loc="upper left")
    ax = axes[1]
    m = (d.snow >= 0.1) & d.prcp.notna()
    ax.scatter(d.prcp[m], d.y_swe[m], s=8, alpha=0.4, color="#c0392b", edgecolor="none")
    lim = max(d.prcp[m].max(), d.y_swe[m].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#555", lw=1, ls="--", label="1:1")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("GHCN-Daily precipitation (in), snow days"); ax.set_ylabel("Σ hourly SWE target (in)")
    ax.set_title("Liquid equivalent on days with GHCN snowfall ≥ 0.1\"\n(SWE counts only hours with snow reported)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH)
    plt.close(fig)


def write_report(target, d, s, r, factor, top10) -> None:
    av = target[target.target_available]
    rates = av.groupby("split").snow_occ.mean()
    lines = ["# Target validation: KDTW hourly snowfall vs GHCN-Daily", "",
             f"- Target hours: {len(target):,}; available: {len(av):,} ({len(av) / len(target):.2%})",
             f"- Snow-occurring hours (falling snow reported): {av.snow_occ.mean():.2%} of available hours",
             "- Snow-hour rate by split: " + ", ".join(f"{k} {v:.2%}" for k, v in rates.items()),
             f"- Measurable-snow hours (SWE ≥ 0.01\"): {(av.y_swe >= 0.01).mean():.2%}; "
             f"mixed-precip share of snow hours: {av.loc[av.snow_occ == 1, 'wx_mixed'].mean():.2%}",
             f"- Days compared (LST, ≥ {MIN_HOURS_PER_DAY} available hours, GHCN SNOW present): {len(d):,}",
             f"- Mean per-season ratio Σ derived depth / Σ GHCN snowfall: {s.depth_ratio.mean():.2f} "
             f"(range {s.depth_ratio.min():.2f}–{s.depth_ratio.max():.2f})",
             f"- Mean per-season ratio Σ SWE / Σ GHCN precipitation on snow days: {s.swe_ratio.mean():.2f}",
             f"- Daily Pearson r (days where either value ≥ 1\"): {r:.3f}",
             f"- Depth correction factor (Σ GHCN SNOW / Σ derived depth, train seasons only): {factor:.3f} "
             "(informational; not applied to the training target)", "",
             "![Target vs GHCN](figures/target_vs_ghcn.png)", "",
             "## Per-season totals", "",
             "| season | split | days | Σ derived depth (in) | Σ GHCN snow (in) | depth ratio | Σ SWE on snow days | Σ GHCN precip on snow days | SWE ratio |",
             "|---|---|---|---|---|---|---|---|---|"]
    for season, x in s.iterrows():
        lines.append(f"| {season} | {x.split} | {x.days} | {x.y_depth:.1f} | {x.ghcn_snow:.1f} | {x.depth_ratio:.2f} | "
                     f"{x.y_swe_snowdays:.2f} | {x.ghcn_prcp_snowdays:.2f} | {x.swe_ratio:.2f} |")
    lines += ["", "## Ten largest GHCN snowfall days", "",
              "| date | GHCN snow (in) | snow hours | Σ SWE (in) | Σ derived depth (in) |", "|---|---|---|---|---|"]
    for date, x in top10.iterrows():
        lines.append(f"| {date} | {x.snow:.1f} | {int(x.snow_hours)} | {x.y_swe:.2f} | {x.y_depth:.1f} |")
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    try:
        cfg = load_config()
        hourly = pd.read_parquet(HOURLY_PATH, filters=[("station", "==", cfg.target_station)])
        target = build_target(hourly, cfg, load_splits())
        tmp = TARGET_PATH.with_suffix(".parquet.part")
        target.to_parquet(tmp, index=False)
        tmp.rename(TARGET_PATH)

        ghcn = parse_ghcn(RAW_GHCN / f"{cfg.ghcn_id}.csv.gz")
        d = daily_compare(target, ghcn, cfg.lst_offset_hours)
        s = season_table(d)
        big = d[(d.snow >= 1) | (d.y_depth >= 1)]
        r = float(np.corrcoef(big.snow, big.y_depth)[0, 1])
        tr = d[d.split == "train"]
        factor = float(tr.snow.sum() / tr.y_depth.sum())
        CALIB_PATH.write_text(json.dumps({
            "depth_correction_factor": round(factor, 4), "fit_on": "train seasons",
            "definition": "sum(GHCN SNOW) / sum(derived hourly depth) over LST days with >= "
                          f"{MIN_HOURS_PER_DAY} available target hours", "applied_to_target": False}, indent=2) + "\n")
        top10 = d.sort_values("snow", ascending=False).head(10)
        plot(d, r)
        write_report(target, d, s, r, factor, top10)
        av = target[target.target_available]
        print(f"available {av.shape[0]:,}/{len(target):,}; snow_occ rate {av.snow_occ.mean():.4f}; "
              f"depth ratio mean {s.depth_ratio.mean():.3f}; daily r {r:.3f}; factor {factor:.3f}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAILED target: {e}")
        return 1
    print("DONE target")
    return 0


if __name__ == "__main__":
    sys.exit(main())
