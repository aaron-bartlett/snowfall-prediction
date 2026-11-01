import gzip

import pandas as pd
import pytest

from src.config import ROOT, load_config, season_label
from src.download import MANIFEST_PATH, RAW_GHCN, sha256, validate_file
from src.stations import STATIONS_PATH
from src.timeutil import target_hours

pytestmark = pytest.mark.data
HOURS_PER_WINDOW = 5808  # Sep 1 - Apr 30, non-leap


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def man():
    return pd.read_csv(MANIFEST_PATH)


def test_manifest_complete(man, cfg):
    n_st = len(pd.read_csv(STATIONS_PATH))
    assert len(man) == n_st * len(cfg.season_years)
    ok = (man.status == "ok").mean()
    print(f"\nok fraction {ok:.4f}; failed: {man[man.status != 'ok'][['sid', 'season']].values.tolist()}")
    assert ok >= 0.98, f"STOP: only {ok:.1%} of downloads ok"


def test_files_valid_and_hashes(man, cfg):
    bad = []
    for r in man[man.status == "ok"].itertuples():
        p = ROOT / r.path
        if not validate_file(p, cfg.asos_columns)[0] or sha256(p) != r.sha256:
            bad.append(r.path)
    assert not bad, f"invalid or changed files: {bad[:10]}"


def test_target_all_seasons(man, cfg):
    t = man[man.sid == cfg.target_station]
    assert (t.status == "ok").sum() == len(cfg.season_years), "STOP: target station missing seasons"
    assert (t.rows >= 5000).all(), t[t.rows < 5000][["season", "rows"]]


def test_rows_per_station_season(man):
    ok = man[man.status == "ok"]
    frac = (ok.rows >= 0.8 * HOURS_PER_WINDOW).mean()
    print(f"\nstation-seasons with >= 80% rows: {frac:.3f}")
    assert frac >= 0.90


def test_ghcn(cfg):
    path = RAW_GHCN / f"{cfg.ghcn_id}.csv.gz"
    with gzip.open(path, "rt") as fh:
        g = pd.read_csv(fh, header=None,
                        names=["id", "date", "element", "value", "mflag", "qflag", "sflag", "obstime"],
                        dtype={"date": str})
    snow = g[g.element == "SNOW"]
    dates = set(snow.date)
    days = {d.strftime("%Y%m%d") for y in cfg.season_years
            for d in pd.DatetimeIndex(target_hours(y).date).unique()}
    cov = len(days & dates) / len(days)
    print(f"\nGHCN SNOW coverage of target days: {cov:.4f}")
    assert cov >= 0.95
