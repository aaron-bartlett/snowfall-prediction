"""Docs figures and the validation-only ablation harness."""
import json

import pytest

from src.ablation import feature_sets, family
from src.config import ROOT
from src.viz import figures as F
from src.viz import style
from src.viz.names import FAMILY_ORDER, frame

FEATURES = json.loads((ROOT / "meta" / "feature_list_L1.json").read_text())


def test_family_counts():
    n = frame(FEATURES).family.value_counts()
    assert dict(n) == {"Neighbor raw lags": 1248, "KDTW lags": 252, "Spatial aggregates": 112,
                       "KDTW windowed": 11, "Time": 4}
    assert set(n.index) == set(FAMILY_ORDER)


def test_ablation_sets_are_nested_partitions():
    sets = feature_sets(FEATURES)
    sizes = {k.split(":")[0]: len(v) for k, v in sets.items()}
    assert sizes == {"A": 267, "C": 379, "B": 1515, "D": 1627}
    assert sets["D: full"] == FEATURES
    assert {family(f) for f in FEATURES} == {"dtw", "time", "agg", "neighbor"}


@pytest.mark.parametrize("fn", [f for f in F.TIER_A if f is not F.f01_stations], ids=lambda f: f.__name__)
def test_tier_a_figures_render(fn, tmp_path):
    style.apply()
    fn(tmp_path)
    assert any(tmp_path.glob("*.png"))
