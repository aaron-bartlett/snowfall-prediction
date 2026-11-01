from types import SimpleNamespace

import pytest

import src.download as dl
from src.config import load_config


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def tmp_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "RAW_ASOS", tmp_path / "asos")
    return tmp_path


def good_body(cfg, rows=3):
    header = ",".join(["station", "valid", *cfg.asos_columns])
    vals = ",".join(["M"] * len(cfg.asos_columns))
    return header + "\n" + "\n".join(f"DTW,2024-10-01 0{i}:53,{vals}" for i in range(rows)) + "\n"


def test_build_asos_url(cfg):
    url = dl.build_asos_url("DTW", 2005, cfg.asos_columns)
    assert "year1=2005&month1=9&day1=1&year2=2006&month2=5&day2=1" in url
    assert "station=DTW" in url and "report_type=3" in url and "tz=Etc%2FUTC" in url
    for c in cfg.asos_columns:
        assert c in url


def test_asos_path():
    assert dl.asos_path("DTW", 2005).name == "DTW_2005-06.csv"


def test_validate_file(cfg, tmp_path):
    good = tmp_path / "good.csv"
    good.write_text(good_body(cfg))
    assert dl.validate_file(good, cfg.asos_columns) == (True, 3)

    wrong = tmp_path / "wrong.csv"
    wrong.write_text("station,valid,tmpf\nDTW,2024-10-01 00:53,50\n")
    assert dl.validate_file(wrong, cfg.asos_columns)[0] is False

    html = tmp_path / "html.csv"
    html.write_text("<html><body>Error</body></html>\n")
    assert dl.validate_file(html, cfg.asos_columns)[0] is False

    header_only = tmp_path / "h.csv"
    header_only.write_text(good_body(cfg, rows=0).split("\n")[0] + "\n")
    assert dl.validate_file(header_only, cfg.asos_columns)[0] is False

    part = tmp_path / "x.csv.part"
    part.write_text(good_body(cfg))
    assert dl.validate_file(part, cfg.asos_columns)[0] is False


def test_skip_existing_valid(cfg, tmp_raw):
    p = dl.asos_path("DTW", 2024)
    p.parent.mkdir(parents=True)
    p.write_text(good_body(cfg))

    def get(*a, **k):
        raise AssertionError("no HTTP call expected")

    assert dl.download_one("DTW", 2024, cfg, get=get, sleep=lambda s: None) == "skipped"


def test_retry_then_success(cfg, tmp_raw):
    calls = []

    def get(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            return SimpleNamespace(status_code=503, text="busy")
        return SimpleNamespace(status_code=200, text=good_body(cfg))

    assert dl.download_one("DTW", 2024, cfg, get=get, sleep=lambda s: None) == "ok"
    assert len(calls) == 2
    assert dl.validate_file(dl.asos_path("DTW", 2024), cfg.asos_columns) == (True, 3)
    assert not list(dl.asos_path("DTW", 2024).parent.glob("*.part"))


def test_all_retries_fail(cfg, tmp_raw):
    def get(url, timeout):
        return SimpleNamespace(status_code=500, text="err")

    assert dl.download_one("DTW", 2024, cfg, get=get, sleep=lambda s: None) == "failed"


def test_html_body_is_retried_and_fails(cfg, tmp_raw):
    def get(url, timeout):
        return SimpleNamespace(status_code=200, text="<html>rate limited</html>")

    assert dl.download_one("DTW", 2024, cfg, get=get, sleep=lambda s: None) == "failed"
    assert not dl.asos_path("DTW", 2024).exists()


def test_window_hours():
    assert dl.window_hours(2005) == 5808
    assert dl.window_hours(2007) == 5832  # Feb 2008 is a leap month


def test_backfill_routine(cfg, tmp_raw):
    header = ",".join(["station", "valid", *cfg.asos_columns])
    vals = ",".join(["M"] * len(cfg.asos_columns))
    p = dl.asos_path("DTW", 2006)
    p.parent.mkdir(parents=True)
    p.write_text(header + "\n" + f"DTW,2006-12-31 23:54,{vals}\n")  # only one routine row
    unfiltered = [header] + [f"DTW,2007-01-01 {h:02d}:54,{vals}" for h in range(10)] \
        + [f"DTW,2007-01-01 0{h}:17,{vals}" for h in range(3)] + [f"DTW,2006-12-31 23:54,{vals}"]

    def get(url, timeout):
        assert "report_type" not in url
        return SimpleNamespace(status_code=200, text="\n".join(unfiltered) + "\n")

    assert dl.backfill_routine("DTW", 2006, cfg, get=get, sleep=lambda s: None) == "backfilled"
    ok, rows = dl.validate_file(p, cfg.asos_columns)
    assert ok and rows == 11  # 1 existing + 10 at :54 (duplicate merged, :17 specials dropped)
    lines = p.read_text().splitlines()[1:]
    assert all(":54," in line for line in lines)
    assert lines == sorted(lines)


def test_backfill_not_needed(cfg, tmp_raw):
    header = ",".join(["station", "valid", *cfg.asos_columns])
    vals = ",".join(["M"] * len(cfg.asos_columns))
    p = dl.asos_path("DTW", 2005)
    p.parent.mkdir(parents=True)
    p.write_text(header + "\n" + "\n".join(f"DTW,2005-10-01 00:{i % 60:02d},{vals}" for i in range(5000)) + "\n")

    def get(*a, **k):
        raise AssertionError("no HTTP call expected")

    assert dl.backfill_routine("DTW", 2005, cfg, get=get, sleep=lambda s: None) == "not_needed"


def test_backfill_uses_existing_routine_minute(cfg, tmp_raw):
    header = ",".join(["station", "valid", *cfg.asos_columns])
    vals = ",".join(["M"] * len(cfg.asos_columns))
    p = dl.asos_path("HTL", 2023)
    p.parent.mkdir(parents=True)
    p.write_text(header + "\n" + "\n".join(f"HTL,2023-10-01 {h:02d}:55,{vals}" for h in range(3)) + "\n")
    # unfiltered: more :45 obs than :55 obs
    unf = [header] + [f"HTL,2023-10-02 {h:02d}:45,{vals}" for h in range(20)] \
        + [f"HTL,2023-10-02 {h:02d}:55,{vals}" for h in range(10)]

    def get(url, timeout):
        return SimpleNamespace(status_code=200, text="\n".join(unf) + "\n")

    assert dl.backfill_routine("HTL", 2023, cfg, get=get, sleep=lambda s: None) == "backfilled"
    lines = p.read_text().splitlines()[1:]
    assert len(lines) == 13 and all(":55," in line for line in lines)
