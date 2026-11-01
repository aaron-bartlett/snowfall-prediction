import pandas as pd
import pytest

from src.config import load_config
from src.stations import CANDIDATES_PATH, MAP_PATH, STATIONS_PATH

pytestmark = pytest.mark.data

PREVIOUS_36 = ["YIP", "DET", "ARB", "PTK", "ADG", "TDZ", "TOL", "FNT", "JXN", "LAN", "LPR", "CLE",
               "MBS", "BKL", "BTL", "MFD", "MNN", "AZO", "GRR", "AKR", "FWA", "CAK", "HZY", "BIV",
               "OSU", "YNG", "CMH", "PHD", "SBN", "BEH", "VTA", "MKG", "HTL", "SGH", "LHQ", "ZZV"]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def cand():
    return pd.read_csv(CANDIDATES_PATH)


@pytest.fixture(scope="module")
def sel():
    return pd.read_csv(STATIONS_PATH)


def test_candidate_count(cand, cfg):
    n = (cand.sid != cfg.target_station).sum()
    assert 125 <= n <= 145, f"{n} candidates"


def test_target_meets_thresholds(sel, cfg):
    t = sel[sel.sid == cfg.target_station]
    assert len(t) == 1
    t = t.iloc[0]
    assert t.temp_cov >= cfg.coverage["min_temp"], f"STOP: target temp coverage {t.temp_cov:.3f}"
    assert t.precip_cov >= cfg.coverage["min_precip"], f"STOP: target precip coverage {t.precip_cov:.3f}"


def test_selected_count_and_diff(sel, cfg):
    nb = sel[sel.sid != cfg.target_station]
    added = sorted(set(nb.sid) - set(PREVIOUS_36))
    removed = sorted(set(PREVIOUS_36) - set(nb.sid))
    print(f"\nselected {len(nb)}; added vs previous 36: {added}; removed: {removed}")
    assert 25 <= len(nb) <= 50, f"STOP: {len(nb)} stations selected"


def test_selected_satisfy_criteria(sel, cfg):
    assert sel.sid.is_unique
    assert (sel.dist_km <= cfg.radius_km).all()
    assert (sel.temp_cov >= cfg.coverage["min_temp"]).all()
    assert (sel.precip_cov >= cfg.coverage["min_precip"]).all()
    assert sel.iloc[0].sid == cfg.target_station and sel.iloc[0].dist_km == 0


def test_map_exists():
    assert MAP_PATH.exists() and MAP_PATH.stat().st_size > 100_000
