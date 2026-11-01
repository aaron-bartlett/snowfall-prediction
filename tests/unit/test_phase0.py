import re

import numpy as np
import pytest
import yaml

from src.config import ConfigError, DEFAULT_PATH, load_config, season_label, validate
from src.splits import make_splits


def test_imports_and_lightgbm_toy():
    import cartopy, optuna, pandas, pyarrow, shap, sklearn  # noqa: F401
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(int)
    m = lgb.LGBMClassifier(n_estimators=20, verbose=-1).fit(X, y)
    assert m.predict_proba(X).shape == (200, 2)


def test_config_loads():
    cfg = load_config()
    assert cfg.target_station == "DTW"
    assert len(cfg.season_years) == 21
    assert set(cfg.lead_hours) == {1, 3}


def _raw():
    with open(DEFAULT_PATH) as fh:
        return yaml.safe_load(fh)


def test_config_rejects_bad_fractions():
    raw = _raw()
    raw["split"]["train"] = 0.8
    with pytest.raises(ConfigError):
        validate(raw)


def test_config_rejects_bad_lead():
    raw = _raw()
    raw["lead_hours"] = [1, 2]
    with pytest.raises(ConfigError):
        validate(raw)


def test_config_rejects_missing_key():
    raw = _raw()
    del raw["radius_km"]
    with pytest.raises(ConfigError):
        validate(raw)


def test_season_label():
    assert season_label(2005) == "2005-06"
    assert season_label(2099) == "2099-00"


def test_splits_sizes_disjoint_deterministic():
    years = list(range(2005, 2026))
    a = make_splits(years, 42, 0.15, 0.15)
    b = make_splits(years, 42, 0.15, 0.15)
    assert a == b
    assert (len(a["train"]), len(a["val"]), len(a["test"])) == (15, 3, 3)
    tr, va, te = set(a["train"]), set(a["val"]), set(a["test"])
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == {season_label(y) for y in years}
    assert all(re.fullmatch(r"\d{4}-\d{2}", s) for s in tr | va | te)
