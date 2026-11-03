"""Docs figures. Every function here reads committed files only (meta/, models/*/*.json, reports/)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, Wedge

from src.config import ROOT
from src.evaluate import segment_events
from src.features import RINGS, SECTOR_HALF_WIDTH, SECTORS, UPSTREAM_HALF_WIDTH
from src.metrics import pr_auc, roc_auc
from src.train import TAU_GRID
from src.viz import style as S
from src.viz.names import FAMILY_ORDER, frame

META, REPORTS, MODELS, SUM = ROOT / "meta", ROOT / "reports", ROOT / "models", S.SUMMARIES
LEADS = (1, 3)
SYS = ["model", "persistence", "climatology"]
MONTHS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]


def _metrics():
    return json.loads((REPORTS / "test_metrics.json").read_text())["leads"]


def _preds(lead, split="test"):
    return pd.read_parquet(REPORTS / f"{split}_predictions_L{lead}.parquet").sort_values("hour_ending").reset_index(drop=True)


def _sys_color(system, lead):
    return S.lead_color(lead) if system == "model" else S.C[system]


# ------------------------------------------------------------------ maps
def _map_axes(fig, pos, extent=(-87.6, -79.6, 39.2, 45.2)):
    import cartopy.crs as ccrs
    import cartopy.feature as cf
    ax = fig.add_subplot(*pos, projection=ccrs.LambertConformal(central_longitude=-83.3, central_latitude=42.2))
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cf.LAND.with_scale("50m"), facecolor="#f4f3ef", edgecolor="none")
    ax.add_feature(cf.LAKES.with_scale("50m"), facecolor="#dbe8f5", edgecolor="#b9cbe0", linewidth=0.5)
    ax.add_feature(cf.STATES.with_scale("50m"), edgecolor="#bdbcb6", linewidth=0.5)
    ax.add_feature(cf.BORDERS.with_scale("50m"), edgecolor="#8a8985", linewidth=0.7)
    return ax, ccrs.PlateCarree()


def _circle(lat0, lon0, r_km, n=180):
    b = np.radians(np.linspace(0, 360, n))
    d = r_km / 6371.0
    p0, l0 = np.radians(lat0), np.radians(lon0)
    p = np.arcsin(np.sin(p0) * np.cos(d) + np.cos(p0) * np.sin(d) * np.cos(b))
    l = l0 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(p0), np.cos(d) - np.sin(p0) * np.sin(p))
    return np.degrees(l), np.degrees(p)


def _dest(lat0, lon0, bearing, r_km):
    d, b = r_km / 6371.0, np.radians(bearing)
    p0, l0 = np.radians(lat0), np.radians(lon0)
    p = np.arcsin(np.sin(p0) * np.cos(d) + np.cos(p0) * np.sin(d) * np.cos(b))
    l = l0 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(p0), np.cos(d) - np.sin(p0) * np.sin(p))
    return np.degrees(l), np.degrees(p)


def _cone(ax, tr, lat0, lon0, center, half, r_km, **kw):
    bs = np.linspace(center - half, center + half, 40)
    pts = [_dest(lat0, lon0, b, r_km) for b in bs]
    xs = [lon0] + [p[0] for p in pts] + [lon0]
    ys = [lat0] + [p[1] for p in pts] + [lat0]
    ax.fill(xs, ys, transform=tr, **kw)


def f01_stations(out=None):
    st, cand = pd.read_csv(META / "stations.csv"), pd.read_csv(META / "station_candidates.csv")
    t = st[st.sid == "DTW"].iloc[0]
    fig = plt.figure(figsize=(S.FULL, 6.4))
    ax, tr = _map_axes(fig, (1, 1, 1))
    for r in (100, 200, 300):
        x, y = _circle(t.lat, t.lon, r)
        ax.plot(x, y, transform=tr, color=S.GRAY, lw=0.8, ls="--")
        lx, ly = _dest(t.lat, t.lon, 200, r)
        ax.text(lx, ly, f"{r} km", transform=tr, fontsize=7, color=S.INK2)
    drop = cand[~cand.sid.isin(st.sid)]
    ax.scatter(drop.lon, drop.lat, transform=tr, marker="x", s=14, color=S.GRAY, lw=0.8,
               label=f"Excluded candidate ({len(drop)})")
    nb = st[st.sid != "DTW"]
    for state in ("MI", "OH", "IN"):
        s = nb[nb.network.str.startswith(state)]
        ax.scatter(s.lon, s.lat, transform=tr, s=34, color=S.C[state], edgecolor="white", lw=1, zorder=4,
                   label=f"Input station, {state} ({len(s)})")
        for _, r in s.iterrows():
            ax.text(r.lon + 0.06, r.lat + 0.05, r.sid, transform=tr, fontsize=6.5, color=S.INK2, zorder=5)
    ax.scatter([t.lon], [t.lat], transform=tr, marker="*", s=260, color=S.YELLOW, edgecolor=S.INK, lw=0.8,
               zorder=6, label="Target: KDTW")
    ax.legend(loc="lower left", fontsize=8, frameon=True, facecolor="white", edgecolor="none")
    ax.set_title("39 input stations within 300 km of Detroit Metro (KDTW)")
    S.save(fig, "f01_stations.png", out)


def f11_spatial_aggregates(out=None):
    st, x = pd.read_csv(META / "stations.csv"), pd.read_csv(SUM / "example_hour.csv")
    t = st[st.sid == "DTW"].iloc[0]
    wf = x.wind_from.iloc[0]
    fig = plt.figure(figsize=(S.FULL, 4.4))
    ax1, tr = _map_axes(fig, (1, 2, 1))
    ax2, _ = _map_axes(fig, (1, 2, 2))
    for ax in (ax1, ax2):
        for r in (100, 200, 300):
            cx, cy = _circle(t.lat, t.lon, r)
            ax.plot(cx, cy, transform=tr, color=S.GRAY, lw=0.7, ls="--")
    for name, c in SECTORS.items():
        _cone(ax1, tr, t.lat, t.lon, c, SECTOR_HALF_WIDTH, 300, facecolor=S.BLUE if name in ("W", "SW", "NW") else S.LIGHTGRAY,
              alpha=0.12, edgecolor=S.BLUE, lw=0.6)
        lx, ly = _dest(t.lat, t.lon, c, 320)
        ax1.text(lx, ly, name, transform=tr, fontsize=8, ha="center", va="center", color=S.INK)
    _cone(ax2, tr, t.lat, t.lon, wf, UPSTREAM_HALF_WIDTH, 300, facecolor=S.ORANGE, alpha=0.15, edgecolor=S.ORANGE, lw=0.8)
    nb = x[x.sid != "DTW"]
    for ax in (ax1, ax2):
        sn = nb.wx_snow == 1
        ax.scatter(nb.lon[~sn], nb.lat[~sn], transform=tr, s=22, facecolor="white", edgecolor=S.INK2, lw=0.8, zorder=4,
                   label="no snow reported")
        ax.scatter(nb.lon[sn], nb.lat[sn], transform=tr, s=30, color=S.BLUE, edgecolor="white", lw=0.8, zorder=4,
                   label="snow reported")
        ax.scatter([t.lon], [t.lat], transform=tr, marker="*", s=180, color=S.YELLOW, edgecolor=S.INK, lw=0.7, zorder=6)
    hr = pd.Timestamp(x.hour_ending.iloc[0]).strftime("%Y-%m-%d %H UTC")
    ax1.set_title("Distance rings and 45° sectors", fontsize=10)
    ax2.set_title(f"Upstream cone: ±45° of wind from {wf:.0f}°", fontsize=10)
    ax1.legend(loc="lower left", fontsize=7, frameon=True, facecolor="white", edgecolor="none")
    fig.suptitle(f"Spatial aggregates on a real storm hour ({hr})", x=0.02, ha="left", fontweight="bold")
    S.save(fig, "f11_spatial_aggregates.png", out)


def f14_shap_geo(out=None):
    st = pd.read_csv(META / "stations.csv")
    fig = plt.figure(figsize=(S.FULL, 4.3))
    for i, lead in enumerate(LEADS):
        imp = pd.read_csv(SUM / f"shap_importance_L{lead}.csv")
        by = imp[imp.family == "Neighbor raw lags"].groupby("station").mean_abs_shap.sum()
        local = imp[imp.family.str.startswith("KDTW")].mean_abs_shap.sum()
        ax, tr = _map_axes(fig, (1, 2, i + 1))
        s = st.set_index("sid").join(by.rename("imp")).dropna(subset=["imp"])
        ax.scatter(s.lon, s.lat, transform=tr, s=s.imp * 900, color=S.lead_color(lead), alpha=0.75,
                   edgecolor="white", lw=0.8, zorder=4)
        for sid, r in s.sort_values("imp").tail(5).iterrows():
            ax.text(r.lon + 0.1, r.lat + 0.1, sid, transform=tr, fontsize=7.5, color=S.INK, zorder=5)
        t = st[st.sid == "DTW"].iloc[0]
        ax.scatter([t.lon], [t.lat], transform=tr, marker="*", s=170, color=S.YELLOW, edgecolor=S.INK, lw=0.7, zorder=6)
        ax.set_title(f"Lead {lead} h  (KDTW's own features: Σ {local:.2f})", fontsize=10)
    fig.suptitle("Which neighbor stations the occurrence model listens to (circle area = Σ mean |SHAP|)",
                 x=0.02, ha="left", fontweight="bold")
    S.save(fig, "f14_shap_geo.png", out)


# ------------------------------------------------------------------ data / target
def f02_availability_kdtw(out=None):
    a = pd.read_csv(SUM / "availability_kdtw.csv")
    order = [10, 11, 12, 1, 2, 3, 4]
    fig, axs = plt.subplots(1, 3, figsize=(S.FULL, 4.6), sharey=True)
    for ax, col, title in zip(axs, ["has_ob", "raw_p01i", "usable_p01i"],
                              ["Routine METAR present", "Raw p01i reported", "p01i usable after fill"]):
        m = a.pivot(index="season", columns="month", values=col)[order] * 100
        im = ax.imshow(m.values, aspect="auto", cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(7), MONTHS, fontsize=8)
        ax.set_yticks(range(len(m)), m.index, fontsize=7)
        ax.set_title(title, fontsize=9.5)
        ax.grid(False)
    fig.colorbar(im, ax=axs, shrink=0.8, label="% of target hours")
    fig.suptitle("KDTW data availability by season and month", x=0.02, ha="left", fontweight="bold")
    S.save(fig, "f02_availability_kdtw.png", out)


def f03_availability_stations(out=None):
    a = pd.read_csv(SUM / "availability_stations.csv")
    st = pd.read_csv(META / "stations.csv").sort_values("dist_km")
    m = a.pivot(index="station", columns="season", values="has_ob").reindex(st.sid) * 100
    fig, ax = plt.subplots(figsize=(S.FULL, 7))
    im = ax.imshow(m.values, aspect="auto", cmap="Blues", vmin=50, vmax=100)
    ax.set_yticks(range(len(m)), m.index, fontsize=7)
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=90, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.6, label="% of target hours with a routine ob (floor 50%)")
    ax.set_title("Observation coverage, 40 stations (sorted by distance from KDTW) × 21 seasons")
    S.save(fig, "f03_availability_stations.png", out)


def f04_p01i_fill(out=None):
    p = pd.read_csv(REPORTS / "p01i_missing_counts.csv")
    p = p[p.station == "DTW"].set_index("season")
    cats = [("d_dry_no_pgroup", "dry hour, no P group → 0", S.LIGHTGRAY), ("a_pno", "PNO (gauge out) → missing", S.ORANGE),
            ("c_precip_wx_no_pgroup", "precip wx, no P group → trace", S.BLUE)]
    fig, ax = plt.subplots(figsize=(S.FULL, 3.4))
    bottom = np.zeros(len(p))
    for col, lab, c in cats:
        ax.bar(p.index, p[col], bottom=bottom, color=c, label=lab, edgecolor="white", lw=1)
        bottom += p[col].to_numpy()
    ax.set_ylabel("KDTW obs with raw p01i missing")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.legend(loc="upper right")
    ax.set_title("Missing hourly precipitation at KDTW is almost all dry hours, and stops after 2011-12")
    S.save(fig, "f04_p01i_fill.png", out)


def f05_target_distribution(out=None):
    h = pd.read_csv(SUM / "target_hist.csv").groupby("cls").hours.sum()
    cls = ["no snow", "trace", "0.01-0.02", "0.03-0.05", ">=0.06"]
    h = h.reindex(cls)
    m = pd.read_csv(SUM / "target_monthly.csv")
    m = m.assign(n=m.snow_rate * m.hours).groupby("month")[["n", "hours"]].sum()
    m = (m.n / m.hours).reindex([10, 11, 12, 1, 2, 3, 4]) * 100
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL, 3.3))
    a.bar(range(5), h.values, color=[S.LIGHTGRAY] + [S.BLUE] * 4, edgecolor="white")
    a.set_yscale("log")
    a.set_xticks(range(5), ["no snow", "trace", "0.01–0.02", "0.03–0.05", "≥0.06"], fontsize=8)
    a.set_xlabel("hourly SWE (in)")
    a.set_ylabel("hours (log scale)")
    for i, v in enumerate(h.values):
        a.text(i, v * 1.15, f"{v / h.sum() * 100:.1f}%", ha="center", fontsize=8, color=S.INK2)
    a.set_title("91.5% of cold-season hours are dry", fontsize=10)
    b.bar(MONTHS, m.values, color=S.BLUE, edgecolor="white")
    b.set_ylabel("% of hours with falling snow")
    b.set_title("Snow-hour rate by month (all seasons)", fontsize=10)
    S.save(fig, "f05_target_distribution.png", out)


def f06_ghcn_scatter(out=None):
    d = pd.read_csv(SUM / "ghcn_daily.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL, 3.8))
    a.scatter(d.snow, d.y_depth, s=6, color=S.BLUE, alpha=0.45, lw=0)
    big = (d.snow >= 1) | (d.y_depth >= 1)
    r = np.corrcoef(d.snow[big], d.y_depth[big])[0, 1]
    lim = max(d.snow.max(), d.y_depth.max()) * 1.05
    a.plot([0, lim], [0, lim], color=S.GRAY, ls="--", lw=1)
    a.set(xlim=(0, lim), ylim=(0, lim), xlabel="GHCN daily snowfall (in)", ylabel="Σ hourly derived depth (in)")
    a.set_title(f"Daily snowfall, r = {r:.2f} on days ≥ 1\"", fontsize=10)
    s = d[d.snow >= 0.1]
    b.scatter(s.prcp, s.y_swe, s=6, color=S.ORANGE, alpha=0.45, lw=0)
    lim = max(s.prcp.max(), s.y_swe.max()) * 1.05
    b.plot([0, lim], [0, lim], color=S.GRAY, ls="--", lw=1)
    b.set(xlim=(0, lim), ylim=(0, lim), xlabel="GHCN daily precipitation (in)", ylabel="Σ hourly SWE target (in)")
    b.set_title("Liquid equivalent on snow days (points fall below 1:1)", fontsize=10)
    S.save(fig, "f06_ghcn_scatter.png", out)


def _season_totals():
    d = pd.read_csv(SUM / "ghcn_daily.csv")
    return d.groupby("season").agg(split=("split", "first"), depth=("y_depth", "sum"), ghcn=("snow", "sum"))


def f07_ghcn_season_ratio(out=None):
    s = _season_totals()
    ratio = s.depth / s.ghcn
    fig, ax = plt.subplots(figsize=(S.FULL, 3.3))
    ax.bar(s.index, ratio, color=[S.C[x] for x in s.split], edgecolor="white")
    ax.axhline(1, color=S.GRAY, ls="--", lw=1)
    ax.set_ylabel("Σ derived depth / Σ GHCN snowfall")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.legend(handles=[Patch(color=S.C[k], label=k) for k in ("train", "val", "test")], loc="upper left", ncol=3)
    ax.set_title("The hourly gauge captures 53–103% of official seasonal snowfall")
    S.save(fig, "f07_ghcn_season_ratio.png", out)


def f08_season_split(out=None):
    s = _season_totals()
    fig, ax = plt.subplots(figsize=(S.FULL, 2.8))
    ax.bar(s.index, s.ghcn, color=[S.C[x] for x in s.split], edgecolor="white")
    ax.set_ylabel("GHCN seasonal snowfall (in)")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.legend(handles=[Patch(color=S.C[k], label=f"{k} ({(s.split == k).sum()})") for k in ("train", "val", "test")],
              loc="upper right", ncol=3)
    ax.set_title("Whole winters assigned at random (seed 42): 15 train / 3 val / 3 test")
    S.save(fig, "f08_season_split.png", out)


# ------------------------------------------------------------------ features
def f09_lead_lag_timeline(out=None):
    fig, axs = plt.subplots(2, 1, figsize=(S.FULL, 2.6), sharex=True)
    for ax, lead in zip(axs, LEADS):
        for k in range(1, 13):
            off = lead + k - 1
            ax.add_patch(Rectangle((-off - 0.45, 0.2), 0.9, 0.6, color=S.lead_color(lead), alpha=1 - (k - 1) * 0.06))
        for nk in (1, 2, 3, 6):
            off = lead + nk - 1
            ax.plot(-off, 1.15, "v", color=S.INK2, ms=6)
        for off in range(0, lead):
            ax.add_patch(Rectangle((-off - 0.45, 0.2), 0.9, 0.6, facecolor="white", edgecolor=S.GRAY, hatch="///", lw=0.5))
        ax.add_patch(Rectangle((-0.45, 0.2), 0.9, 0.6, facecolor=S.YELLOW, edgecolor=S.INK, lw=0.8))
        ax.text(0, 0.5, "t", ha="center", va="center", fontweight="bold")
        ax.set_ylim(0, 1.45)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.text(-16.6, 0.5, f"L = {lead} h", va="center", fontweight="bold", color=S.lead_color(lead))
    axs[1].set_xticks(range(-15, 1), [f"t{x}" if x else "t" for x in range(-15, 1)], fontsize=7.5)
    axs[1].set_xlim(-17, 0.8)
    axs[0].set_title("Input window: KDTW lags 1–12 (boxes), neighbor lags 1/2/3/6 (▼); hatched hours are never seen")
    S.save(fig, "f09_lead_lag_timeline.png", out)


def f10_feature_inventory(out=None):
    f = frame(json.loads((META / "feature_list_L1.json").read_text()))
    n = f.family.value_counts().reindex(FAMILY_ORDER)
    colors = [S.BLUE, S.BLUE, S.ORANGE, S.AQUA, S.GRAY]
    fig, ax = plt.subplots(figsize=(S.FULL, 2.6))
    ax.barh(n.index[::-1], n.values[::-1], color=colors[::-1], edgecolor="white")
    for i, v in enumerate(n.values[::-1]):
        ax.text(v + 15, i, f"{v:,}", va="center", fontsize=9)
    ax.set_xlabel("number of features")
    ax.set_xlim(0, n.max() * 1.12)
    ax.set_title(f"{len(f):,} features per lead, by family")
    S.save(fig, "f10_feature_inventory.png", out)


def f12_pressure_gradient(out=None):
    x = pd.read_csv(SUM / "example_hour.csv").dropna(subset=["mslp"])
    A = np.c_[np.ones(len(x)), x.x_km, x.y_km]
    coef = np.linalg.lstsq(A, x.mslp, rcond=None)[0]
    gx, gy = np.meshgrid(np.linspace(-330, 330, 60), np.linspace(-330, 330, 60))
    fig, ax = plt.subplots(figsize=(S.HALF * 1.5, 5))
    cs = ax.contour(gx, gy, coef[0] + coef[1] * gx + coef[2] * gy, levels=12, colors=S.GRAY, linewidths=0.7)
    ax.clabel(cs, fontsize=7, fmt="%.0f")
    sc = ax.scatter(x.x_km, x.y_km, c=x.mslp, cmap="Blues_r", s=40, edgecolor=S.INK2, lw=0.5, zorder=3)
    fig.colorbar(sc, ax=ax, shrink=0.8, label="station MSLP (hPa)")
    m, d = x.grad_mag.iloc[0], x.grad_dir.iloc[0]
    ax.annotate("", xy=(120 * np.sin(np.radians(d)), 120 * np.cos(np.radians(d))), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=S.ORANGE, lw=2.5))
    ax.plot(0, 0, "*", ms=14, color=S.YELLOW, mec=S.INK, zorder=5)
    ax.set(xlabel="km east of KDTW", ylabel="km north of KDTW", aspect="equal")
    ax.set_title(f"Plane fit: {m:.2f} hPa/100 km toward {d:.0f}°", fontsize=10)
    S.save(fig, "f12_pressure_gradient.png", out)


def _imp(lead):
    return pd.read_csv(SUM / f"shap_importance_L{lead}.csv")


def f13_shap_by_family(out=None):
    fig, ax = plt.subplots(figsize=(S.FULL, 3.0))
    w = 0.38
    for i, lead in enumerate(LEADS):
        s = _imp(lead).groupby("family").mean_abs_shap.sum().reindex(FAMILY_ORDER)
        share = s / s.sum() * 100
        y = np.arange(len(s)) + (i - 0.5) * w
        ax.barh(y, share.values, height=w - 0.04, color=S.lead_color(lead), label=f"lead {lead} h")
        for yy, v in zip(y, share.values):
            ax.text(v + 0.6, yy, f"{v:.0f}%", va="center", fontsize=8)
    ax.set_yticks(range(len(FAMILY_ORDER)), FAMILY_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("share of total mean |SHAP| (Stage 1)")
    ax.legend(loc="lower right")
    ax.set_title("At 3 h, weight moves from KDTW's own obs to the spatial aggregates")
    S.save(fig, "f13_shap_by_family.png", out)


def f15_shap_lag_profile(out=None):
    fig, axs = plt.subplots(1, 2, figsize=(S.FULL, 3.0), sharey=True)
    groups = [("KDTW lags", S.BLUE), ("Neighbor raw lags", S.ORANGE), ("Spatial aggregates", S.AQUA)]
    for ax, lead in zip(axs, LEADS):
        f = _imp(lead)
        tot = f.mean_abs_shap.sum()
        for fam, c in groups:
            s = f[f.family == fam].groupby("lag").mean_abs_shap.sum() / tot * 100
            ax.plot(s.index, s.values, marker="o", ms=5, color=c, label=fam)
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("lag k (reads hour t − L − (k − 1))")
        ax.set_title(f"Lead {lead} h", fontsize=10)
    axs[0].set_ylabel("% of total mean |SHAP|")
    axs[0].legend()
    fig.suptitle("The most recent observed hour carries the most signal; older lags fade fast", x=0.02, ha="left", fontweight="bold")
    S.save(fig, "f15_shap_lag_profile.png", out)


def f16_shap_dependence(out=None):
    fig, axs = plt.subplots(2, 4, figsize=(S.FULL * 1.15, 4.8))
    for r, lead in enumerate(LEADS):
        d = pd.read_parquet(SUM / f"shap_dependence_L{lead}.parquet")
        feats = [c[:-7] for c in d.columns if c.endswith("__value")]
        for ax, f in zip(axs[r], feats):
            v, s = d[f + "__value"], d[f + "__shap"]
            ax.scatter(v, s, s=3, alpha=0.3, color=S.lead_color(lead), lw=0)
            ax.axhline(0, color=S.GRAY, lw=0.8)
            ax.set_title(f.replace("_lag1", " (lag 1)"), fontsize=7.5)
            ax.tick_params(labelsize=7)
            if "ceiling" in f:
                ax.set_xscale("log"); ax.set_xlim(80, 3e4)
        axs[r][0].set_ylabel(f"L{lead}: SHAP (log-odds)", fontsize=8)
    fig.suptitle("SHAP dependence: low ceilings, nearby snow and sub-freezing upstream air raise P(snow)",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    S.save(fig, "f16_shap_dependence.png", out)


def f17_ablation(out=None):
    res = json.loads((REPORTS / "ablation_val.json").read_text())["leads"]
    order = ["A: KDTW only", "C: KDTW + spatial aggregates", "B: KDTW + neighbor raw lags", "D: full"]
    fig, axs = plt.subplots(1, 2, figsize=(S.FULL, 3.2))
    w = 0.38
    for i, lead in enumerate(LEADS):
        sets = {s["set"]: s for s in res[str(lead)]["sets"]}
        y = np.arange(len(order)) + (i - 0.5) * w
        for ax, key in zip(axs, ["val_pr_auc", "cv_logloss"]):
            vals = [sets[o][key] for o in order]
            ax.barh(y, vals, height=w - 0.04, color=S.lead_color(lead), label=f"lead {lead} h")
            for yy, v in zip(y, vals):
                ax.text(v, yy, f" {v:.3f}", va="center", fontsize=7.5)
    for ax, lab in zip(axs, ["validation PR-AUC (higher is better)", "train-season CV log-loss (lower is better)"]):
        ax.set_yticks(range(len(order)), [o.split(": ")[1] for o in order] if ax is axs[0] else [""] * 4)
        ax.invert_yaxis()
        ax.set_xlabel(lab)
    axs[0].set_xlim(0.4, 1.0)
    lo = min(s["cv_logloss"] for r in res.values() for s in r["sets"])
    hi = max(s["cv_logloss"] for r in res.values() for s in r["sets"])
    axs[1].set_xlim(lo * 0.9, hi * 1.08)
    axs[1].legend(loc="lower right")
    fig.suptitle("Ablation (validation seasons only): what each feature family adds", x=0.02, ha="left", fontweight="bold")
    S.save(fig, "f17_ablation.png", out)


# ------------------------------------------------------------------ model
def f18_tau_selection(out=None):
    fig, axs = plt.subplots(1, 2, figsize=(S.FULL, 3.0), sharey=True)
    for ax, lead in zip(axs, LEADS):
        v = _preds(lead, "val")
        o = v.snow_occ.to_numpy() == 1
        pod, far, csi = [], [], []
        for t in TAU_GRID:
            f = v.p_cal.to_numpy() >= t
            h, m, fa = (f & o).sum(), (~f & o).sum(), (f & ~o).sum()
            pod.append(h / (h + m)); far.append(fa / max(h + fa, 1)); csi.append(h / (h + m + fa))
        tau = TAU_GRID[int(np.argmax(csi))]
        ax.plot(TAU_GRID, csi, color=S.lead_color(lead), label="CSI")
        ax.plot(TAU_GRID, pod, color=S.INK2, ls="--", lw=1.2, label="POD")
        ax.plot(TAU_GRID, far, color=S.INK2, ls=":", lw=1.2, label="FAR")
        ax.axvline(tau, color=S.lead_color(lead), lw=1)
        ax.text(tau + 0.02, 0.05, f"τ = {tau:.2f}", color=S.INK, fontsize=9)
        ax.set_xlabel("threshold τ on calibrated P(snow)")
        ax.set_title(f"Lead {lead} h", fontsize=10)
    axs[0].legend()
    fig.suptitle("τ maximizes CSI on the validation seasons", x=0.02, ha="left", fontweight="bold")
    S.save(fig, "f18_tau_selection.png", out)


def f19_calibration_map(out=None):
    fig, ax = plt.subplots(figsize=(S.HALF * 1.2, 4))
    ax.plot([0, 1], [0, 1], color=S.GRAY, ls="--", lw=1, label="identity")
    for lead in LEADS:
        c = json.loads((MODELS / f"lead_{lead}h" / "calibrator.json").read_text())
        ax.step(c["x_thresholds"], c["y_thresholds"], where="post", color=S.lead_color(lead), label=f"lead {lead} h")
    ax.set(xlabel="raw Stage 1 probability", ylabel="calibrated P(snow)", xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.legend(loc="upper left")
    ax.set_title("Isotonic calibration maps", fontsize=10)
    S.save(fig, "f19_calibration_map.png", out)


# ------------------------------------------------------------------ results
def f20_skill_summary(out=None):
    M = _metrics()
    keys = [("bss_clim", "Brier skill score"), ("pr_auc", "PR-AUC"), ("csi", "CSI (snow / no snow)")]
    fig, axs = plt.subplots(1, 3, figsize=(S.FULL, 3.0), sharey=True)
    for ax, (k, lab) in zip(axs, keys):
        for i, lead in enumerate(LEADS):
            h = M[str(lead)]["headline"]
            for j, s in enumerate(SYS):
                x = i * 4 + j
                v = h[s][k]
                ax.bar(x, v, color=_sys_color(s, lead), edgecolor="white",
                       hatch=None if s == "model" else ("..." if s == "persistence" else None))
                ax.text(x, (v + 0.02) if v >= 0 else (v - 0.07), f"{v:.2f}" if s != "climatology" or k == "pr_auc" else "0", ha="center", fontsize=7.5)
        ax.set_xticks([1, 5], ["lead 1 h", "lead 3 h"])
        ax.axhline(0, color=S.INK2, lw=0.8)
        ax.set_title(lab, fontsize=9.5)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylim(-0.15, 1.05)
    axs[0].legend(handles=[Patch(color=S.BLUE, label="model 1 h"), Patch(color=S.ORANGE, label="model 3 h"),
                           Patch(color=S.AQUA, label="persistence"), Patch(color=S.GRAY, label="climatology")],
                  loc="upper right", fontsize=7.5)
    fig.suptitle("Held-out test seasons: the model beats both baselines at both leads", x=0.02, y=1.04, ha="left", fontweight="bold")
    S.save(fig, "f20_skill_summary.png", out)


def f21_reliability(out=None):
    bins = np.linspace(0, 1, 11)
    fig, axs = plt.subplots(2, 2, figsize=(S.FULL, 5.6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    for c, lead in enumerate(LEADS):
        p = _preds(lead)
        ax, axh = axs[0, c], axs[1, c]
        ax.plot([0, 1], [0, 1], color=S.GRAY, ls="--", lw=1)
        for col, ls, lab in (("p_cal", "-", "calibrated"), ("p_raw", ":", "raw")):
            idx = np.clip(np.digitize(p[col], bins) - 1, 0, 9)
            g = pd.DataFrame({"i": idx, "p": p[col], "o": p.snow_occ}).groupby("i").agg(p=("p", "mean"), o=("o", "mean"), n=("o", "size"))
            g = g[g.n >= 10]
            ax.plot(g.p, g.o, ls=ls, marker="o", ms=4, color=S.lead_color(lead), label=f"model ({lab})")
        ax.set_title(f"Lead {lead} h", fontsize=10)
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left", fontsize=8)
        axh.hist(p.p_cal, bins=bins, color=S.lead_color(lead), edgecolor="white")
        axh.set_yscale("log")
        axh.set_xlabel("forecast probability")
    axs[0, 0].set_ylabel("observed snow frequency")
    axs[1, 0].set_ylabel("hours")
    fig.suptitle("Reliability on test seasons (bins with ≥ 10 hours)", x=0.02, ha="left", fontweight="bold")
    S.save(fig, "f21_reliability.png", out)


def f22_roc_pr(out=None):
    from sklearn.metrics import precision_recall_curve, roc_curve
    fig, (a, b) = plt.subplots(1, 2, figsize=(S.FULL, 3.7))
    for lead in LEADS:
        p = _preds(lead)
        y, s = p.snow_occ.to_numpy(), p.p_cal.to_numpy()
        fpr, tpr, _ = roc_curve(y, s)
        a.plot(fpr, tpr, color=S.lead_color(lead), label=f"model {lead} h (AUC {roc_auc(s, y):.3f})")
        pr, rc, _ = precision_recall_curve(y, s)
        b.plot(rc, pr, color=S.lead_color(lead), label=f"model {lead} h (AUC {pr_auc(s, y):.3f})")
        q = p.persist_occ.to_numpy() == 1
        o = y == 1
        tp = (q & o).sum()
        a.plot((q & ~o).sum() / (~o).sum(), tp / o.sum(), "D", color=S.AQUA, mec=S.lead_color(lead), mew=1.5, ms=7)
        b.plot(tp / o.sum(), tp / q.sum(), "D", color=S.AQUA, mec=S.lead_color(lead), mew=1.5, ms=7)
    a.plot([0, 1], [0, 1], color=S.GRAY, ls="--", lw=1)
    a.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC")
    b.set(xlabel="recall (POD)", ylabel="precision (1 − FAR)", title="Precision–recall")
    a.plot([], [], "D", color=S.AQUA, label="persistence (edge = lead)")
    a.legend(loc="lower right", fontsize=8)
    b.legend(loc="lower left", fontsize=8)
    S.save(fig, "f22_roc_pr.png", out)


def f23_csi_by_threshold(out=None):
    M = _metrics()
    labels = [("SWE", "trace"), ("SWE", "0.01in"), ("SWE", "0.05in"), ("depth", "0.1in"), ("depth", "0.5in")]
    fig, axs = plt.subplots(1, 2, figsize=(S.FULL, 3.0), sharey=True)
    for ax, lead in zip(axs, LEADS):
        t = pd.DataFrame(M[str(lead)]["thresholds"])
        for j, s in enumerate(["model", "persistence"]):
            v = [t[(t.quantity == q) & (t.threshold == th) & (t.system == s)].csi.iloc[0] for q, th in labels]
            x = np.arange(len(labels)) + (j - 0.5) * 0.38
            ax.bar(x, v, width=0.36, color=_sys_color(s, lead), edgecolor="white", label=s)
        ax.set_xticks(range(len(labels)), [f"{q}\n≥ {th}" for q, th in labels], fontsize=8)
        ax.set_title(f"Lead {lead} h", fontsize=10)
        ax.legend(fontsize=8)
    axs[0].set_ylabel("CSI")
    fig.suptitle("CSI by threshold: the 3 h margin over persistence holds at every threshold", x=0.02, ha="left", fontweight="bold")
    S.save(fig, "f23_csi_by_threshold.png", out)


def f24_intensity_bias(out=None):
    M = _metrics()
    bins = ["trace", "0.01-0.02", "0.03-0.05", ">=0.06"]
    fig, ax = plt.subplots(figsize=(S.FULL, 3.2))
    x = np.arange(len(bins))
    obs = pd.DataFrame(M["1"]["by_intensity"]).query("system == 'model'").set_index("bin").mean_obs.reindex(bins)
    ax.bar(x - 0.3, obs, width=0.28, color=S.LIGHTGRAY, label="observed", edgecolor="white")
    for k, lead in enumerate(LEADS):
        d = pd.DataFrame(M[str(lead)]["by_intensity"]).query("system == 'model'").set_index("bin").reindex(bins)
        ax.bar(x + k * 0.3, d.mean_pred, width=0.28, color=S.lead_color(lead), label=f"model {lead} h", edgecolor="white")
    ax.set_xticks(x, ["trace", "0.01–0.02", "0.03–0.05", "≥ 0.06"])
    ax.set_xlabel("observed hourly SWE bin (in)")
    ax.set_ylabel("mean SWE (in/h)")
    ax.legend()
    ax.set_title("Expected SWE is pulled toward the mean: heavy hours are underpredicted")
    S.save(fig, "f24_intensity_bias.png", out)


def _events(p):
    ev = segment_events(p.hour_ending, p.snow_occ)
    g = p.assign(ev=ev)[ev >= 0].groupby("ev")
    return g.agg(start=("hour_ending", "first"), end=("hour_ending", "last"), obs=("y_swe", "sum"),
                 model=("swe_expected", "sum"), pers=("persist_swe", "sum"))


def f25_event_totals(out=None):
    fig, axs = plt.subplots(1, 2, figsize=(S.FULL, 3.8), sharey=True)
    for ax, lead in zip(axs, LEADS):
        e = _events(_preds(lead))
        lim = max(e.obs.max(), e.model.max()) * 1.05
        ax.plot([0, lim], [0, lim], color=S.GRAY, ls="--", lw=1)
        ax.scatter(e.obs, e.model, s=14, color=S.lead_color(lead), alpha=0.7, lw=0)
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="observed event-total SWE (in)", aspect="equal")
        ax.set_title(f"Lead {lead} h: Σ forecast / Σ observed = {e.model.sum() / e.obs.sum():.2f}", fontsize=10)
    axs[0].set_ylabel("forecast event-total SWE (in)")
    fig.suptitle(f"{len(e)} test-season snow events", x=0.02, ha="left", fontweight="bold")
    S.save(fig, "f25_event_totals.png", out)


def f26_stratified(out=None):
    M = _metrics()
    fig, axs = plt.subplots(1, 2, figsize=(S.FULL, 3.2), gridspec_kw={"width_ratios": [6, 5]}, sharey=True)
    for ax, key, groups in ((axs[0], "by_month", MONTHS[1:]), (axs[1], "by_wind", ["N", "E", "S", "W", "calm"])):
        for lead in LEADS:
            d = pd.DataFrame(M[str(lead)][key])
            for s in ("model", "persistence"):
                v = d[d.system == s].set_index("group").csi.reindex(groups)
                ax.plot(groups, v.values, marker="o", ms=5, color=S.lead_color(lead), ls=S.LS[s],
                        label=f"{s} {lead} h")
        ax.set_title("by month (Oct has no test snow)" if key == "by_month" else "by KDTW wind quadrant at t − L", fontsize=10)
    axs[0].set_ylabel("CSI")
    axs[1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Skill is stable across months and wind regimes", x=0.02, ha="left", fontweight="bold")
    S.save(fig, "f26_stratified.png", out)


def _case(start, end, title, name, out, pad=6):
    t0, t1 = pd.Timestamp(start) - pd.Timedelta(hours=pad), pd.Timestamp(end) + pd.Timedelta(hours=pad)
    fig, (ax, axp) = plt.subplots(2, 1, figsize=(S.FULL, 4.8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for lead in LEADS:
        p = _preds(lead)
        w = p[(p.hour_ending >= t0) & (p.hour_ending <= t1)]
        if lead == 1:
            ax.bar(w.hour_ending, w.y_swe, width=1 / 24 * 0.9, color=S.LIGHTGRAY, label="observed SWE")
            axp.scatter(w.hour_ending[w.snow_occ == 1], np.full((w.snow_occ == 1).sum(), 1.05), s=8, color=S.INK, label="observed snow hour")
        ax.plot(w.hour_ending, w.swe_expected, color=S.lead_color(lead), label=f"model E[SWE], {lead} h")
        axp.plot(w.hour_ending, w.p_cal, color=S.lead_color(lead), label=f"P(snow), {lead} h")
        axp.axhline(w.tau.iloc[0], color=S.lead_color(lead), ls="--", lw=0.8)
    ax.set_ylabel("SWE (in/h)")
    ax.legend(fontsize=8, loc="upper right")
    axp.set(ylabel="P(snow)", ylim=(-0.03, 1.12), xlabel="hour ending (UTC); dashed lines = τ")
    axp.legend(fontsize=7.5, loc="upper left", ncol=3, bbox_to_anchor=(0, 0.92))
    ax.set_title(title)
    fig.autofmt_xdate()
    S.save(fig, name, out)


def f27_case_largest_storm(out=None):
    e = _metrics()["1"]["largest_event"]
    _case(e["start"], e["end"], f"Largest test storm, {e['start'][:10]}: {float(e['obs_swe']):.2f}\" SWE "
          f"(≈ {float(e['obs_depth']):.1f}\" snow) — timing right, peak rates too low", "f27_case_largest_storm.png", out)


def f28_cases(out=None):
    p = _preds(1)
    e = _events(p)
    e = e[e.obs >= 0.10].assign(ratio=lambda d: d.model / d.obs)   # events with >= 0.10" SWE (~1" snow)
    miss = e.sort_values("ratio").iloc[0]
    _case(miss.start, miss.end, f"Worst relative miss at 1 h (events ≥ 0.10\" SWE): obs {miss.obs:.2f}\", "
          f"forecast {miss.model:.2f}\"", "f28_case_miss.png", out)
    fa = (p.p_cal >= p.tau) & (p.snow_occ == 0)
    run = (fa != fa.shift()).cumsum()
    runs = p[fa].groupby(run[fa]).hour_ending.agg(["first", "last", "size"]).sort_values("size", ascending=False)
    snow_t = p.hour_ending[p.snow_occ == 1]
    for _, r in runs.iterrows():   # longest run with no observed snow within 3 h either side
        near = snow_t[(snow_t >= r["first"] - pd.Timedelta(hours=3)) & (snow_t <= r["last"] + pd.Timedelta(hours=3))]
        if near.empty:
            break
    _case(r["first"], r["last"], f"Longest isolated false alarm at 1 h: {r['size']} h with P ≥ τ and no snow within 3 h",
          "f28_case_false_alarm.png", out)


TIER_A = [f01_stations, f04_p01i_fill, f09_lead_lag_timeline, f10_feature_inventory, f18_tau_selection,
          f19_calibration_map, f20_skill_summary, f21_reliability, f22_roc_pr, f23_csi_by_threshold,
          f24_intensity_bias, f25_event_totals, f26_stratified, f27_case_largest_storm, f28_cases]
FROM_SUMMARIES = [f02_availability_kdtw, f03_availability_stations, f05_target_distribution, f06_ghcn_scatter,
                  f07_ghcn_season_ratio, f08_season_split, f11_spatial_aggregates, f12_pressure_gradient,
                  f13_shap_by_family, f14_shap_geo, f15_shap_lag_profile, f16_shap_dependence]
FROM_ABLATION = [f17_ablation]
