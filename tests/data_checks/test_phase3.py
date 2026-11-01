import pandas as pd
import pytest

from src.clean import DIAG_PATH, HOURLY_PATH, PCOUNTS_PATH, QC_PATH, season_grid
from src.config import load_config
from src.timeutil import TARGET_MONTHS

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def hourly():
    return pd.read_parquet(HOURLY_PATH)


@pytest.fixture(scope="module")
def tgt(hourly, cfg):
    t = hourly[hourly.station == cfg.target_station]
    return t[t.hour_ending.dt.month.isin(TARGET_MONTHS)]


def test_unique_and_complete_grid(hourly, cfg):
    assert not hourly.duplicated(["station", "hour_ending"]).any()
    expected = sum(len(season_grid(y)) for y in cfg.season_years)
    counts = hourly.groupby("station", observed=True).size()
    short = counts[counts != expected]
    assert short.empty, f"stations with incomplete grids: {short.to_dict()} (expected {expected})"


def test_target_ob_coverage(tgt):
    has_ob = tgt.has_ob.mean()
    tmpf = tgt.tmpf.notna().mean()
    print(f"\ntarget has_ob {has_ob:.4f}, tmpf non-null {tmpf:.4f}")
    assert has_ob >= 0.95 and tmpf >= 0.95


def test_target_p01i_after_fill(tgt):
    obs = tgt[tgt.has_ob]
    avail = obs.p01i.notna().mean()
    print(f"\ntarget p01i available after fill: {avail:.4f}")
    assert avail >= 0.90, f"STOP: target p01i availability {avail:.3f} < 0.90"
    if avail < 0.95:
        print("NOTE: target p01i availability between 90% and 95%")


def test_pgroup_fallback_rare(cfg):
    assert "Target station" in DIAG_PATH.read_text()
    pc = pd.read_csv(PCOUNTS_PATH)
    t = pc[pc.station == cfg.target_station][["obs", "b_pgroup_parsed"]].sum()
    frac_b = t.b_pgroup_parsed / t.obs
    print(f"\ncategory (b) share of target obs: {frac_b:.4f}")
    assert frac_b < 0.01, "investigate p01i parsing: many P-group fallbacks"


def test_qc_rates(cfg):
    q = pd.read_csv(QC_PATH)
    cols = [c for c in q.columns if c.startswith(("range_", "spike_", "dwpf_"))]
    by = q.groupby("station")[cols + ["obs"]].sum()
    rates = by[cols].div(by.obs, axis=0)
    high = rates[(rates > 0.01).any(axis=1)]
    if not high.empty:
        print("\nWARNING stations with >1% QC nulling:\n", high.round(4).to_string())
    t = rates.loc[cfg.target_station]
    assert (t <= 0.05).all(), f"STOP: target QC nulling {t[t > 0.05].to_dict()}"


def test_physical_sanity(tgt):
    jan = tgt[tgt.hour_ending.dt.month == 1]
    t = jan.tmpf.mean()
    s = jan.loc[jan.has_ob, "wx_snow"].mean()
    print(f"\ntarget Jan mean tmpf {t:.1f} F, Jan snow-hour freq {s:.3f}")
    assert 15 <= t <= 35
    assert 0.05 <= s <= 0.35
