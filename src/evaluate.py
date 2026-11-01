"""Single held-out evaluation of the hurdle models on the test seasons.

Usage: python -m src.evaluate --final [--force]
Output: reports/test_report.md, reports/test_metrics.json, reports/test_predictions_L{L}.parquet,
        reports/figures/{reliability_L{L}, shap_L{L}, storm_event}.png, reports/TEST_EVALUATED.lock

The test seasons are scored exactly once. The run refuses to start if the lock file exists, unless
--force is passed. Nothing here feeds back into training: the models, calibrators and tau are loaded
from models/ unchanged. Climatology is fit on the train seasons only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import ROOT, load_config  # noqa: E402
from src.features import TARGET_COLS, feature_list_path, features_path, wind_from_deg  # noqa: E402
from src.metrics import (apply_climatology, brier, brier_skill, contingency, mae,  # noqa: E402
                         monthly_climatology, pr_auc, rmse, roc_auc, scores)
from src.model import HurdleModel  # noqa: E402
from src.splits import load_splits  # noqa: E402
from src.target import MIN_HOURS_PER_DAY, RAW_GHCN, TARGET_PATH, parse_ghcn, slr  # noqa: E402
from src.timeutil import lst_date  # noqa: E402
from src.train import file_sha256, git_commit, model_dir  # noqa: E402

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
LOCK_PATH = REPORTS / "TEST_EVALUATED.lock"
SYSTEMS = ("model", "persistence", "climatology")
SWE_THRESHOLDS = {"trace": 0.0, "0.01in": 0.01, "0.05in": 0.05}
DEPTH_THRESHOLDS = {"0.1in": 0.1, "0.5in": 0.5}
INTENSITY_BINS = [(-0.001, 0.0, "trace"), (0.0, 0.02, "0.01-0.02"), (0.02, 0.05, "0.03-0.05"),
                  (0.05, 10.0, ">=0.06")]
EVENT_MAX_GAP = 2           # up to 2 non-snow hours inside one event
SHAP_ROWS = 5000
SEED = 42


class LockError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------
# Building blocks (unit tested)
# ---------------------------------------------------------------------------------------------

def segment_events(times, occ, max_gap: int = EVENT_MAX_GAP) -> np.ndarray:
    """Event id per row (-1 outside events).

    An event is a run of snow hours (occ == 1) where consecutive snow hours are at most
    `max_gap` + 1 hours apart in time (i.e. up to `max_gap` non-snow or missing hours between them).
    Non-snow hours inside such a gap belong to the event. `times` must be sorted.
    """
    t = pd.DatetimeIndex(times)
    o = np.asarray(occ, float) == 1
    hrs = ((t - t[0]) // pd.Timedelta(hours=1)).to_numpy() if len(t) else np.array([], int)
    ev = np.full(len(o), -1, dtype=int)
    snow_idx = np.flatnonzero(o)
    if not len(snow_idx):
        return ev
    eid, start = 0, snow_idx[0]
    for prev, cur in zip(snow_idx[:-1], snow_idx[1:]):
        if hrs[cur] - hrs[prev] - 1 > max_gap:
            ev[start:prev + 1] = eid
            eid += 1
            start = cur
    ev[start:snow_idx[-1] + 1] = eid
    return ev


def check_lock(lock: Path, force: bool) -> None:
    if lock.exists() and not force:
        raise LockError(f"{lock.name} exists: the test seasons were already evaluated "
                        f"({lock.read_text().strip()[:200]}). Pass --force to re-run.")


def write_lock(lock: Path, info: dict) -> None:
    tmp = lock.with_suffix(".lock.part")
    tmp.write_text(json.dumps(info, indent=2) + "\n")
    tmp.rename(lock)


def intensity_bin(y_swe) -> np.ndarray:
    y = np.asarray(y_swe, float)
    out = np.full(len(y), "", dtype=object)
    for lo, hi, name in INTENSITY_BINS:
        out[(y > lo) & (y <= hi)] = name
    return out


def wind_sector(u, v, calm_kt: float = 1.0) -> np.ndarray:
    """Quadrant the KDTW wind blows from (N: 315-45, E: 45-135, S: 135-225, W: 225-315), or calm/missing."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    d = wind_from_deg(u, v)
    spd = np.hypot(u, v)
    out = np.full(len(u), "missing", dtype=object)
    ok = np.isfinite(d) & np.isfinite(spd)
    out[ok & (spd < calm_kt)] = "calm"
    live = ok & (spd >= calm_kt)
    q = np.array(["N", "E", "S", "W"])[(((d[live] + 45.0) % 360.0) // 90.0).astype(int) % 4]
    out[live] = q
    return out


def system_columns(pred: pd.DataFrame, df: pd.DataFrame, tau: float, clim: dict, clim_swe: dict,
                   clim_depth: dict, slr_table, slr_fallback) -> dict:
    """Per-system probability, amount and deterministic columns, aligned with df."""
    months = df.hour_ending.dt.month.to_numpy()
    t_lag = df["DTW_tmpf_lag1"].to_numpy(float)
    ratio = slr(t_lag, slr_table, slr_fallback)
    pers_occ = df.persist_occ.fillna(0.0).to_numpy(float)
    pers_swe = df.persist_swe.fillna(0.0).to_numpy(float)
    p_clim = apply_climatology(months, clim)
    swe_clim = apply_climatology(months, clim_swe)
    swe_det = pred.swe_det.to_numpy(float)
    return {
        "model": {"p": pred.p_cal.to_numpy(float), "occ": pred.p_cal.to_numpy(float) >= tau,
                  "swe": pred.swe_expected.to_numpy(float), "swe_det": swe_det,
                  "depth": pred.depth.to_numpy(float), "depth_det": swe_det * ratio},
        "persistence": {"p": pers_occ, "occ": pers_occ >= 0.5, "swe": pers_swe, "swe_det": pers_swe,
                        "depth": pers_swe * ratio, "depth_det": pers_swe * ratio},
        "climatology": {"p": p_clim, "occ": p_clim >= 0.5, "swe": swe_clim, "swe_det": swe_clim,
                        "depth": apply_climatology(months, clim_depth),
                        "depth_det": apply_climatology(months, clim_depth)},
    }


def headline(cols: dict, df: pd.DataFrame, p_clim) -> dict:
    occ = df.snow_occ.to_numpy(float)
    swe = df.y_swe.to_numpy(float)
    snow = occ == 1
    out = {}
    for s, c in cols.items():
        r = {"brier": brier(c["p"], occ), "bss_clim": brier_skill(c["p"], occ, p_clim),
             "roc_auc": roc_auc(c["p"], occ), "pr_auc": pr_auc(c["p"], occ)}
        r.update(scores(contingency(occ, c["occ"])))
        r.update({"mae": mae(c["swe"], swe), "rmse": rmse(c["swe"], swe),
                  "mae_snow": mae(c["swe"][snow], swe[snow]), "rmse_snow": rmse(c["swe"][snow], swe[snow]),
                  "mae_det": mae(c["swe_det"], swe), "rmse_det": rmse(c["swe_det"], swe)})
        out[s] = r
    for s in ("model", "climatology"):
        out[s]["mae_skill_vs_persistence"] = 1.0 - out[s]["mae"] / out["persistence"]["mae"]
        out[s]["csi_minus_persistence"] = out[s]["csi"] - out["persistence"]["csi"]
    return out


def threshold_table(cols: dict, df: pd.DataFrame) -> list[dict]:
    rows = []
    obs_swe = df.y_swe.to_numpy(float)
    obs_depth = df.y_depth.to_numpy(float)
    for name, thr in SWE_THRESHOLDS.items():
        obs = df.snow_occ.to_numpy(float) == 1 if thr == 0 else obs_swe >= thr - 1e-9
        for s, c in cols.items():
            fc = c["occ"] if thr == 0 else c["swe_det"] >= thr - 1e-9
            rows.append({"quantity": "SWE", "threshold": name, "system": s, "n_obs": int(obs.sum()),
                         **scores(contingency(obs, fc))})
    for name, thr in DEPTH_THRESHOLDS.items():
        obs = obs_depth >= thr - 1e-9
        for s, c in cols.items():
            rows.append({"quantity": "depth", "threshold": name, "system": s, "n_obs": int(obs.sum()),
                         **scores(contingency(obs, c["depth_det"] >= thr - 1e-9))})
    return rows


def event_table(cols: dict, df: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    ev_all = []
    for season, g in df.groupby("season", sort=True):
        ids = segment_events(g.hour_ending, g.snow_occ)
        ev_all.append(pd.Series(np.where(ids >= 0, [f"{season}#{i}" for i in ids], ""), index=g.index))
    ev = pd.concat(ev_all).reindex(df.index)
    in_ev = ev != ""
    frame = pd.DataFrame({"event": ev[in_ev], "obs_swe": df.y_swe[in_ev], "obs_depth": df.y_depth[in_ev],
                          "start": df.hour_ending[in_ev]})
    for s, c in cols.items():
        frame[f"{s}_swe"] = c["swe"][in_ev.to_numpy()]
        frame[f"{s}_depth"] = c["depth"][in_ev.to_numpy()]
    tot = frame.groupby("event").agg(start=("start", "min"), end=("start", "max"), hours=("obs_swe", "size"),
                                     **{k: (k, "sum") for k in frame.columns if k not in ("event", "start")})
    rows = []
    for s in cols:
        for q in ("swe", "depth"):
            err = tot[f"{s}_{q}"] - tot[f"obs_{q}"]
            r = float(np.corrcoef(tot[f"{s}_{q}"], tot[f"obs_{q}"])[0, 1]) if tot[f"{s}_{q}"].std() > 0 else float("nan")
            rows.append({"quantity": q, "system": s, "n_events": int(len(tot)), "mae": float(err.abs().mean()),
                         "bias": float(err.mean()), "ratio_total": float(tot[f"{s}_{q}"].sum() / tot[f"obs_{q}"].sum()),
                         "r": r})
    return rows, tot.sort_values("obs_swe", ascending=False)


def stratified_table(cols: dict, df: pd.DataFrame, key: np.ndarray, p_clim) -> list[dict]:
    occ_all = df.snow_occ.to_numpy(float)
    swe_all = df.y_swe.to_numpy(float)
    rows = []
    for k in pd.unique(key):
        m = key == k
        occ, swe = occ_all[m], swe_all[m]
        for s, c in cols.items():
            r = {"group": k, "system": s, "n": int(m.sum()), "snow_rate": float(occ.mean()),
                 "brier": brier(c["p"][m], occ), "bss_clim": brier_skill(c["p"][m], occ, p_clim[m]),
                 "mae": mae(c["swe"][m], swe), "rmse": rmse(c["swe"][m], swe)}
            r.update({k2: v for k2, v in scores(contingency(occ, c["occ"][m])).items() if k2 in ("pod", "far", "csi")})
            rows.append(r)
    return rows


def intensity_table(cols: dict, df: pd.DataFrame) -> list[dict]:
    snow = df.snow_occ.to_numpy(float) == 1
    b = intensity_bin(df.y_swe.to_numpy(float))
    swe = df.y_swe.to_numpy(float)
    rows = []
    for _, _, name in INTENSITY_BINS:
        m = snow & (b == name)
        if not m.any():
            continue
        for s, c in cols.items():
            rows.append({"bin": name, "system": s, "n": int(m.sum()), "mean_obs": float(swe[m].mean()),
                         "mean_pred": float(c["swe"][m].mean()), "mean_p": float(c["p"][m].mean()),
                         "pod": float(c["occ"][m].mean()), "mae": mae(c["swe"][m], swe[m]),
                         "bias": float(c["swe"][m].mean() - swe[m].mean())})
    return rows


def daily_ghcn_table(cols: dict, df: pd.DataFrame, ghcn: pd.DataFrame, offset_hours: int) -> tuple[list[dict], pd.DataFrame]:
    d = pd.DataFrame({"date": lst_date(df.hour_ending, offset_hours).to_numpy(), "obs_target": df.y_depth.to_numpy()})
    for s, c in cols.items():
        d[s] = c["depth"]
    daily = d.groupby("date").agg(hours=("obs_target", "size"), **{k: (k, "sum") for k in d.columns if k != "date"})
    daily = daily.join(ghcn[["snow"]], how="inner")
    daily = daily[(daily.hours >= MIN_HOURS_PER_DAY) & daily.snow.notna()]
    rows = []
    for s in ("obs_target",) + tuple(cols):
        x = daily[s]
        big = (daily.snow >= 1) | (x >= 1)
        rows.append({"system": "observed hourly target" if s == "obs_target" else s, "n_days": int(len(daily)),
                     "sum_pred": float(x.sum()), "sum_ghcn": float(daily.snow.sum()),
                     "ratio": float(x.sum() / daily.snow.sum()), "mae": mae(x, daily.snow),
                     "r_all": float(np.corrcoef(x, daily.snow)[0, 1]) if x.std() > 0 else float("nan"),
                     "r_big": float(np.corrcoef(x[big], daily.snow[big])[0, 1]) if x[big].std() > 0 else float("nan"),
                     "n_big": int(big.sum())})
    return rows, daily


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------

def reliability_figure(lead: int, p_raw, p_cal, occ, p_clim, pers, path: Path) -> None:
    bins = np.linspace(0, 1, 11)
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="perfect")
    for p, lab, col in ((p_cal, "model (calibrated)", "C0"), (p_raw, "model (raw)", "C1"),
                        (p_clim, "climatology (train, by month)", "C2"), (pers, "persistence", "C3")):
        idx = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
        g = pd.DataFrame({"b": idx, "p": p, "o": occ}).groupby("b").agg(p=("p", "mean"), o=("o", "mean"), n=("o", "size"))
        g = g[g.n >= 20]
        ax.plot(g.p, g.o, marker="o", ms=4, color=col, label=lab, lw=1.5 if "calibrated" in lab else 1)
        if "calibrated" in lab:
            axh.bar(bins[:-1] + 0.05, np.bincount(idx, minlength=10), width=0.09, color=col, log=True)
    ax.set_ylabel("observed snow frequency")
    ax.set_title(f"Reliability, test seasons, lead {lead} h")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    axh.set_xlabel("forecast probability")
    axh.set_ylabel("hours (model)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def shap_figure(lead: int, model: HurdleModel, df: pd.DataFrame, path: Path) -> list[tuple[str, float]]:
    import shap
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=min(SHAP_ROWS, len(df)), replace=False)
    X = model.matrix(df.iloc[np.sort(idx)])
    sv = shap.TreeExplainer(model.stage1).shap_values(X)
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., -1]
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)[::-1][:20]
    top = [(model.features[i], float(imp[i])) for i in order]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.barh([n for n, _ in top][::-1], [v for _, v in top][::-1], color="C0")
    ax.set_xlabel("mean |SHAP| (log-odds), Stage 1")
    ax.set_title(f"Top 20 Stage 1 features, lead {lead} h ({len(idx):,} random test hours)")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return top


def storm_figure(event: pd.Series, preds: dict, path: Path, pad_h: int = 6) -> None:
    t0 = event.start - pd.Timedelta(hours=pad_h)
    t1 = event.end + pd.Timedelta(hours=pad_h)
    fig, (ax, axp) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    first = True
    for i, (lead, p) in enumerate(sorted(preds.items())):
        w = p[(p.hour_ending >= t0) & (p.hour_ending <= t1)]
        if first:
            ax.bar(w.hour_ending, w.y_swe, width=1 / 24 * 0.9, color="0.75", label="observed SWE")
            first = False
        ax.plot(w.hour_ending, w.swe_expected, color=f"C{i}", lw=1.8, label=f"model E[SWE], {lead} h")
        ax.step(w.hour_ending, w.persist_swe.fillna(0), where="mid", color=f"C{i}", lw=0.9, ls=":",
                label=f"persistence, {lead} h")
        axp.plot(w.hour_ending, w.p_cal, color=f"C{i}", lw=1.5, label=f"P(snow), {lead} h")
        axp.axhline(w.tau.iloc[0], color=f"C{i}", lw=0.7, ls="--")
    obs = preds[min(preds)]
    w = obs[(obs.hour_ending >= t0) & (obs.hour_ending <= t1)]
    axp.scatter(w.hour_ending, w.snow_occ * 1.02, s=6, color="k", label="observed snow hour", zorder=3)
    ax.set_ylabel("SWE (in / h)")
    ax.set_title(f"Largest test-season event: {event.start:%Y-%m-%d %HZ} to {event.end:%Y-%m-%d %HZ} "
                 f"(obs {event.obs_swe:.2f} in SWE, {event.obs_depth:.1f} in derived depth)")
    ax.legend(fontsize=8, ncol=2)
    axp.set_ylabel("P(snow)")
    axp.set_ylim(-0.02, 1.08)
    axp.legend(fontsize=8, ncol=3, loc="lower right")
    axp.set_xlabel("hour ending (UTC); dashed = τ")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------

def fmt(v, nd=4) -> str:
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, float):
        return "nan" if not np.isfinite(v) else f"{v:.{nd}f}"
    return str(v)


def md_table(rows: list[dict], keys: list[str], nd=4) -> list[str]:
    out = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    out += ["| " + " | ".join(fmt(r[k], nd) for k in keys) + " |" for r in rows]
    return out + [""]


def lead_section(lead: int, res: dict) -> list[str]:
    h = res["headline"]
    hk = ["brier", "bss_clim", "roc_auc", "pr_auc", "pod", "far", "csi", "bias", "mae", "rmse", "mae_snow",
          "rmse_snow", "mae_det", "rmse_det"]
    L = [f"## Lead {lead} h", "",
         f"- Test rows: {res['n_rows']:,} hours ({res['n_snow']:,} snow hours, rate {res['snow_rate']:.4f}). "
         f"Stage 2 variant `{res['variant']}`, τ = {res['tau']:.2f} (both fixed in Phase 6).",
         f"- Skill vs. persistence: MAE skill {h['model']['mae_skill_vs_persistence']:+.3f}, "
         f"CSI {h['model']['csi']:.3f} vs {h['persistence']['csi']:.3f} "
         f"({h['model']['csi_minus_persistence']:+.3f}).", "",
         "### Headline (hourly, all test hours)", "",
         "`mae`/`rmse` use the expected SWE; `_snow` restricts to observed snow hours; `_det` uses the "
         "deterministic SWE (amount if P ≥ τ, else 0). Persistence probability is 0/1. Categorical scores "
         "use P ≥ τ (model) or ≥ 0.5 (persistence, climatology).", ""]
    L += md_table([{"system": s, **h[s]} for s in SYSTEMS], ["system"] + hk)
    L += ["### Occurrence by threshold", "",
          "Trace uses the occurrence forecast; amount thresholds use the deterministic SWE (or depth = "
          "deterministic SWE × SLR(T at t−L)). Observed depth is the derived hourly target.", ""]
    L += md_table(res["thresholds"], ["quantity", "threshold", "system", "n_obs", "pod", "far", "csi", "bias"])
    L += [f"### Event totals ({res['events'][0]['n_events']} events; snow-hour runs with gaps ≤ {EVENT_MAX_GAP} h)", "",
          "Error of the event-total SWE / derived depth (forecast − observed), summed from the expected values.", ""]
    L += md_table(res["events"], ["quantity", "system", "n_events", "mae", "bias", "ratio_total", "r"])
    L += ["### Daily depth vs. GHCN `SNOW`", "",
          f"LST days with ≥ {MIN_HOURS_PER_DAY} test hours. `r_big` uses days where either value is ≥ 1\". The "
          "observed hourly target row shows how much of the gap to GHCN is gauge undercatch rather than model error "
          f"(train-season depth correction factor {res['depth_factor']:.3f}, not applied).", ""]
    L += md_table(res["daily"], ["system", "n_days", "sum_pred", "sum_ghcn", "ratio", "mae", "r_all", "r_big", "n_big"], 3)
    L += ["### By month", ""]
    L += md_table(res["by_month"], ["group", "system", "n", "snow_rate", "brier", "bss_clim", "pod", "far", "csi", "mae", "rmse"])
    L += ["### By observed intensity (snow hours only)", "",
          "`pod` = share of the bin's hours forecast as snow; `mean_p` = mean forecast probability.", ""]
    L += md_table(res["by_intensity"], ["bin", "system", "n", "mean_obs", "mean_pred", "mean_p", "pod", "mae", "bias"])
    L += ["### By KDTW wind quadrant at t − L (synoptic vs. lake-enhanced proxy)", ""]
    L += md_table(res["by_wind"], ["group", "system", "n", "snow_rate", "brier", "bss_clim", "pod", "far", "csi", "mae", "rmse"])
    L += ["### Top 20 Stage 1 features (mean |SHAP|, log-odds)", "",
          ", ".join(f"`{n}` {v:.3f}" for n, v in res["shap_top"]), "",
          f"![reliability L{lead}](figures/reliability_L{lead}.png)", "",
          f"![shap L{lead}](figures/shap_L{lead}.png)", ""]
    return L


def write_report(results: dict, splits: dict, storm: pd.Series, path: Path) -> None:
    L = ["# Test-season evaluation (held out, scored once)", "",
         f"- Test seasons: {', '.join(sorted(splits['test']))}",
         f"- Evaluated: {results['evaluated_at']} at git commit `{results['git_commit'][:7]}`",
         "- Models, isotonic calibrators, Stage 2 variants and τ are loaded unchanged from `models/lead_{1,3}h/`. "
         "Climatology = train-season snow rate / mean SWE / mean depth by calendar month. Persistence = the "
         "observed KDTW value at t − L.",
         "- There are no pass/fail gates here; numbers are reported as they came out.", ""]
    L += ["## Summary", ""]
    sk = ["brier", "bss_clim", "roc_auc", "pr_auc", "pod", "far", "csi", "bias", "mae", "rmse", "rmse_snow"]
    srows = []
    for lead, res in sorted(results["leads"].items()):
        for s in SYSTEMS:
            srows.append({"lead": f"{lead} h", "system": s, **res["headline"][s]})
    L += md_table(srows, ["lead", "system"] + sk)
    L += ["Validation-season numbers for comparison (calibration and τ were fit on those seasons): see "
          "`reports/val_metrics.md`.", ""]
    for lead, res in sorted(results["leads"].items()):
        L += lead_section(lead, res)
    L += ["## Largest test-season storm", "",
          f"{storm.start:%Y-%m-%d %H}Z – {storm.end:%Y-%m-%d %H}Z, {int(storm.hours)} h, observed {storm.obs_swe:.2f}\" SWE "
          f"({storm.obs_depth:.1f}\" derived depth).", "",
          "![storm event](figures/storm_event.png)", ""]
    tmp = path.with_suffix(".md.part")
    tmp.write_text("\n".join(L) + "\n")
    tmp.rename(path)


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------

def train_climatology(target: pd.DataFrame) -> tuple[dict, dict, dict]:
    tr = target[(target.split == "train") & target.target_available]
    m = tr.hour_ending.dt.month
    return (monthly_climatology(m, tr.snow_occ), monthly_climatology(m, tr.y_swe),
            monthly_climatology(m, tr.y_depth))


def evaluate_lead(lead: int, model: HurdleModel, df: pd.DataFrame, clims, ghcn, cfg, figures: Path) -> tuple[dict, pd.DataFrame]:
    clim, clim_swe, clim_depth = clims
    pred = model.predict(df)
    cols = system_columns(pred, df, model.tau, clim, clim_swe, clim_depth, cfg.slr["table"], cfg.slr["fallback"])
    p_clim = cols["climatology"]["p"]
    months = df.hour_ending.dt.month.to_numpy()
    sector = wind_sector(df.DTW_u_lag1, df.DTW_v_lag1)
    events, totals = event_table(cols, df)
    daily, _ = daily_ghcn_table(cols, df, ghcn, cfg.lst_offset_hours)
    month_order = {m: i for i, m in enumerate(cfg.target_months)}
    by_month = sorted(stratified_table(cols, df, months, p_clim), key=lambda r: (month_order[r["group"]], SYSTEMS.index(r["system"])))
    for r in by_month:
        r["group"] = pd.Timestamp(2000, int(r["group"]), 1).strftime("%b")
    by_wind = sorted(stratified_table(cols, df, sector, p_clim),
                     key=lambda r: (["N", "E", "S", "W", "calm", "missing"].index(r["group"]), SYSTEMS.index(r["system"])))
    occ = df.snow_occ.to_numpy(float)
    reliability_figure(lead, pred.p_raw.to_numpy(), pred.p_cal.to_numpy(), occ, p_clim,
                       cols["persistence"]["p"], figures / f"reliability_L{lead}.png")
    shap_top = shap_figure(lead, model, df, figures / f"shap_L{lead}.png")
    res = {"n_rows": int(len(df)), "n_snow": int(occ.sum()), "snow_rate": float(occ.mean()),
           "variant": model.variant, "tau": model.tau, "headline": headline(cols, df, p_clim),
           "thresholds": threshold_table(cols, df), "events": events, "daily": daily,
           "by_month": by_month, "by_intensity": intensity_table(cols, df), "by_wind": by_wind,
           "shap_top": shap_top, "largest_event": totals.iloc[0][["start", "end", "hours", "obs_swe", "obs_depth"]]
           .astype(str).to_dict()}
    out = pd.concat([df[["hour_ending", "season", "snow_occ", "y_swe", "y_depth", "persist_occ", "persist_swe"]],
                     pred], axis=1).assign(tau=model.tau)
    return res, out, totals


def run(force: bool = False, reports: Path = REPORTS, models_root=None, lock: Path | None = None) -> dict:
    lock = lock or reports / LOCK_PATH.name
    check_lock(lock, force)
    cfg = load_config()
    splits = load_splits()
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    target = pd.read_parquet(TARGET_PATH)
    clims = train_climatology(target)
    ghcn = parse_ghcn(RAW_GHCN / f"{cfg.ghcn_id}.csv.gz")
    results = {"evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "git_commit": git_commit(), "test_seasons": sorted(splits["test"]), "leads": {}}
    preds, storm = {}, None
    for lead in cfg.lead_hours:
        mdir = model_dir(lead) if models_root is None else model_dir(lead, models_root)
        model = HurdleModel.load(mdir)
        features = json.loads(feature_list_path(lead).read_text())
        if features != model.features:
            raise RuntimeError(f"lead {lead}: feature list differs from the model metadata")
        df = pd.read_parquet(features_path(lead), columns=TARGET_COLS + features,
                             filters=[("split", "==", "test")]).sort_values("hour_ending").reset_index(drop=True)
        if sorted(df.season.unique()) != sorted(splits["test"]):
            raise RuntimeError(f"lead {lead}: test seasons in features {sorted(df.season.unique())} != splits")
        clim_meta = {int(k): v for k, v in model.metadata.get("climatology_snow_rate_by_month", {}).items()}
        if clim_meta and any(abs(clim_meta[m] - clims[0][m]) > 1e-9 for m in clim_meta):
            raise RuntimeError("train climatology differs from the model metadata")
        print(f"lead {lead}: {len(df):,} test rows, {int(df.snow_occ.sum()):,} snow hours", flush=True)
        res, out, totals = evaluate_lead(lead, model, df, clims, ghcn, cfg, figures)
        res["depth_factor"] = json.loads((ROOT / "meta" / "target_calibration.json").read_text())["depth_correction_factor"]
        res["features_sha256"] = file_sha256(features_path(lead))
        results["leads"][lead] = res
        preds[lead] = out
        if storm is None:
            storm = totals.iloc[0]
        p = reports / f"test_predictions_L{lead}.parquet"
        tmp = p.with_suffix(".parquet.part")
        out.to_parquet(tmp, index=False)
        tmp.rename(p)
        h = res["headline"]
        print(f"lead {lead}: model Brier {h['model']['brier']:.5f} BSS {h['model']['bss_clim']:.3f} "
              f"CSI {h['model']['csi']:.3f} RMSE {h['model']['rmse']:.5f} | persistence Brier "
              f"{h['persistence']['brier']:.5f} CSI {h['persistence']['csi']:.3f} RMSE {h['persistence']['rmse']:.5f}",
              flush=True)
        del df
    storm_figure(storm, preds, figures / "storm_event.png")
    write_report(results, splits, storm, reports / "test_report.md")
    mpath = reports / "test_metrics.json"
    tmp = mpath.with_suffix(".json.part")
    tmp.write_text(json.dumps(results, indent=2, default=str) + "\n")
    tmp.rename(mpath)
    write_lock(lock, {"evaluated_at": results["evaluated_at"], "git_commit": results["git_commit"],
                      "test_seasons": results["test_seasons"], "forced": bool(force)})
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--final", action="store_true", help="required: run the one-time test-season evaluation")
    ap.add_argument("--force", action="store_true", help="re-run even though the lock file exists")
    args = ap.parse_args(argv)
    try:
        if not args.final:
            raise RuntimeError("pass --final to run the one-time test-season evaluation")
        run(force=args.force)
    except Exception as e:
        if not isinstance(e, LockError):
            import traceback
            traceback.print_exc()
        print(f"FAILED evaluate: {e}")
        return 1
    print("DONE evaluate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
