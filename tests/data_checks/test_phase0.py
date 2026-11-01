import pytest

from src.splits import SPLITS_PATH, load_splits

pytestmark = pytest.mark.data


def test_splits_file():
    assert SPLITS_PATH.exists(), "meta/splits.json missing"
    sp = load_splits()
    assert sp["seed"] == 42
    assert (len(sp["train"]), len(sp["val"]), len(sp["test"])) == (15, 3, 3)
