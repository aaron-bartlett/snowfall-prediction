"""Phase 7 checks: the one-time test-season report, lock file and figures."""
import json
import math
import re

import pandas as pd
import pytest

from src.evaluate import FIGURES, LOCK_PATH, REPORTS, SYSTEMS
from src.features import features_path
from src.splits import load_splits

pytestmark = pytest.mark.data
LEADS = [1, 3]
REPORT = REPORTS / "test_report.md"


@pytest.fixture(scope="module")
def report():
    return REPORT.read_text()


@pytest.fixture(scope="module")
def metrics():
    return json.loads((REPORTS / "test_metrics.json").read_text())


def tables(md: str) -> list[tuple[str, list[list[str]]]]:
    """(preceding heading, rows) for every markdown table."""
    out, heading, cur = [], "", None
    for line in md.splitlines():
        if line.startswith("#"):
            heading = line
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cur is None:
                cur = (heading, [])
                out.append(cur)
            if not all(re.fullmatch(r"-+", c) for c in cells):
                cur[1].append(cells)
        else:
            cur = None
    return out


def test_section_per_lead(report):
    for lead in LEADS:
        assert f"## Lead {lead} h" in report


def test_headline_metrics_finite(metrics):
    for lead in LEADS:
        h = metrics["leads"][str(lead)]["headline"]
        for s in SYSTEMS:
            for k in ("brier", "bss_clim", "roc_auc", "pr_auc", "pod", "csi", "mae", "rmse", "mae_snow", "rmse_snow"):
                assert math.isfinite(h[s][k]), (lead, s, k)


def test_every_table_has_reference_rows(report):
    ts = tables(report)
    assert len(ts) >= 1 + 7 * len(LEADS)
    for heading, rows in ts:
        header = rows[0]
        assert "system" in header, f"table under {heading!r} has no system column"
        col = header.index("system")
        systems = {r[col] for r in rows[1:]}
        assert {"model", "persistence", "climatology"} <= systems, heading


def test_lock_and_test_seasons(report, metrics):
    assert LOCK_PATH.exists()
    test = sorted(load_splits()["test"])
    m = re.search(r"^- Test seasons: (.+)$", report, re.M)
    assert m and [s.strip() for s in m.group(1).split(",")] == test
    assert metrics["test_seasons"] == test
    assert json.loads(LOCK_PATH.read_text())["test_seasons"] == test


def test_predictions_cover_test_rows_only():
    test = set(load_splits()["test"])
    for lead in LEADS:
        p = pd.read_parquet(REPORTS / f"test_predictions_L{lead}.parquet")
        assert set(p.season) == test
        n = len(pd.read_parquet(features_path(lead), columns=["split"], filters=[("split", "==", "test")]))
        assert len(p) == n
        assert p[["p_cal", "swe_expected", "depth"]].notna().all().all()


def test_figures_exist():
    names = [f"reliability_L{lead}.png" for lead in LEADS] + [f"shap_L{lead}.png" for lead in LEADS] + ["storm_event.png"]
    for n in names:
        assert (FIGURES / n).stat().st_size > 10_000, n
