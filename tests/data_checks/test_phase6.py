import json
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.config import ROOT
from src.features import feature_list_path, features_path
from src.model import ARTIFACTS, CALIB_FILE, STAGE1_FILE, STAGE2_FILE, HurdleModel
from src.splits import load_splits
from src.train import REPORTS, file_sha256, model_dir

pytestmark = pytest.mark.data
LEADS = [1, 3]


@pytest.fixture(scope="module", params=LEADS)
def lead(request):
    return request.param


@pytest.fixture(scope="module")
def meta(lead):
    return json.loads((model_dir(lead) / "metadata.json").read_text())


@pytest.fixture(scope="module")
def val_preds(lead):
    return pd.read_parquet(REPORTS / f"val_predictions_L{lead}.parquet")


def test_artifacts(lead):
    d = model_dir(lead)
    for a in ARTIFACTS:
        assert (d / a).exists(), a
    for f in (STAGE1_FILE, STAGE2_FILE):
        assert lgb.Booster(model_file=str(d / f)).num_trees() > 0
    c = json.loads((d / CALIB_FILE).read_text())
    assert len(c["x_thresholds"]) == len(c["y_thresholds"]) > 1
    assert (REPORTS / "val_metrics.md").exists()


def test_metadata_consistent(lead, meta):
    sp = load_splits()
    assert meta["features"] == json.loads(feature_list_path(lead).read_text())
    assert meta["train_seasons"] == sorted(sp["train"])
    assert meta["val_seasons"] == sorted(sp["val"])
    assert meta["features_sha256"] == file_sha256(features_path(lead))
    assert 0 < meta["tau"] < 1
    assert meta["lead_hours"] == lead and not meta["smoke"]


def test_no_test_rows(lead, meta, val_preds):
    sp = load_splits()
    df = pd.read_parquet(features_path(lead), columns=["season", "split"])
    assert meta["n_train_rows"] == int((df.split == "train").sum())
    assert meta["n_val_rows"] == len(val_preds) == int((df.split == "val").sum())
    assert not set(val_preds.season) & set(sp["test"])
    assert set(val_preds.season) == set(sp["val"])


def test_reload_reproduces(lead, val_preds):
    model = HurdleModel.load(model_dir(lead))
    df = pd.read_parquet(features_path(lead), filters=[("split", "==", "val")])
    df = df.set_index("hour_ending").loc[val_preds.hour_ending].reset_index()
    p = model.predict(df)
    for c in ("p_raw", "p_cal", "amount_cond", "swe_expected", "swe_det", "depth"):
        assert np.max(np.abs(p[c].to_numpy() - val_preds[c].to_numpy())) <= 1e-9, c


def test_validation_gates(lead, meta):
    m, p = meta["val_metrics"]["model"], meta["val_metrics"]["persistence"]
    print(f"\nlead {lead}: model {json.dumps({k: round(v, 5) for k, v in m.items()})}")
    print(f"lead {lead}: persistence {json.dumps({k: round(v, 5) for k, v in p.items()})}")
    fails = []
    if not m["bss_clim"] > 0:
        fails.append(f"BSS vs climatology {m['bss_clim']:.4f} <= 0")
    if not m["brier"] < p["brier"]:
        fails.append(f"Brier {m['brier']:.5f} >= persistence {p['brier']:.5f}")
    if not m["csi"] >= p["csi"]:
        fails.append(f"CSI {m['csi']:.4f} < persistence {p['csi']:.4f}")
    if not m["rmse"] < p["rmse"]:
        fails.append(f"RMSE {m['rmse']:.5f} >= persistence {p['rmse']:.5f}")
    assert not fails, f"STOP: lead {lead} validation gates failed: {fails}"


def test_predict_cli(lead, val_preds):
    t = val_preds.loc[val_preds.snow_occ == 1, "hour_ending"].min()
    r = subprocess.run([sys.executable, "-m", "src.predict", "--lead", str(lead), "--time", t.isoformat()],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    out = json.loads(r.stdout.split("DONE predict")[0])
    assert all(np.isfinite(v) for v in out.values() if isinstance(v, float))
