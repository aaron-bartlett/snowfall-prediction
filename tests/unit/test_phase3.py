import numpy as np
import pandas as pd
import pytest

from src.clean import WX_FLAGS, clean_file, parse_wx, season_grid
from src.config import load_config

COLS = load_config().asos_columns


def wx(code):
    return dict(zip(WX_FLAGS, parse_wx(code)))


@pytest.mark.parametrize("code,expect", [
    ("-SN", {"wx_snow": 1, "snow_intensity": 1, "wx_precip": 1}),
    ("+SN", {"wx_snow": 1, "snow_intensity": 3}),
    ("SN", {"wx_snow": 1, "snow_intensity": 2}),
    ("SHSN", {"wx_snow": 1}),
    ("BLSN", {"wx_snow": 0, "wx_blsn": 1, "wx_precip": 0}),
    ("-SN BLSN", {"wx_snow": 1, "wx_blsn": 1}),
    ("-RASN", {"wx_snow": 1, "wx_rain": 1, "wx_mixed": 1}),
    ("SG", {"wx_snow": 1}),
    ("VCSH", {"wx_snow": 0, "wx_rain": 0, "wx_precip": 0}),
    ("FZRA", {"wx_fzra": 1, "wx_rain": 0, "wx_snow": 0, "wx_precip": 1}),
    ("UP", {"wx_up": 1, "wx_precip": 1, "wx_snow": 0}),
    ("-SN BR", {"wx_snow": 1, "wx_fog_br": 1}),
    ("-FZRASN", {"wx_snow": 1, "wx_fzra": 1, "wx_mixed": 1}),
    ("", {"wx_snow": 0, "wx_precip": 0}),
])
def test_parse_wx(code, expect):
    got = wx(code)
    for k, v in expect.items():
        assert got[k] == v, (code, k, got)


def row(valid, tmpf="50", dwpf="40", drct="270", sknt="10", sky=("BKN", "OVC"), lvl=("1500", "3000"),
        wxc="M", p01i="0.00", rmk="AO2"):
    vals = dict.fromkeys(COLS, "M")
    vals.update(tmpf=tmpf, dwpf=dwpf, relh="68", drct=drct, sknt=sknt, mslp="1015.0", alti="30.00",
                vsby="10.00", wxcodes=wxc, p01i=p01i,
                metar=f"KDTW 010053Z 27010KT 10SM RMK {rmk} $")
    for i, (c, h) in enumerate(zip(sky, lvl), 1):
        vals[f"skyc{i}"], vals[f"skyl{i}"] = c, h
    return ",".join(["DTW", valid] + [vals[c] for c in COLS])


@pytest.fixture
def cleaned(tmp_path):
    lines = [",".join(["station", "valid", *COLS]),
             row("2015-10-01 00:53"),                                        # 01Z normal, ceiling 1500
             row("2015-10-01 01:20", tmpf="99"),                              # dropped (intra-hour)
             row("2015-10-01 01:45", tmpf="98"),                              # superseded by 01:55
             row("2015-10-01 01:55", tmpf="51"),                              # 02Z
             row("2015-10-01 02:53", wxc="-SN", p01i="T", sky=("FEW", "SCT"), lvl=("2000", "5000")),  # 03Z
             row("2015-10-01 03:53", wxc="-SN", p01i="M", rmk="AO2 PNO"),    # 04Z cat a
             row("2015-10-01 04:53", wxc="SN", p01i="M", rmk="AO2 P0012"),   # 05Z cat b
             row("2015-10-01 05:53", wxc="-RASN", p01i="M"),                  # 06Z cat c
             row("2015-10-01 06:53", wxc="M", p01i="M", drct="0", sknt="0"),  # 07Z cat d, calm
             row("2015-10-01 07:53", tmpf="150", dwpf="40"),                  # 08Z range fail
             row("2015-10-01 08:53", tmpf="51"),                              # 09Z
             row("2015-10-01 09:53", tmpf="75"),                              # 10Z spike
             row("2015-10-01 10:53", tmpf="52"),                              # 11Z
             row("2015-10-01 11:53", drct="M", sknt="5"),                     # 12Z variable wind
             row("2015-10-01 12:53", drct="180", sknt="10")]                  # 13Z south wind
    p = tmp_path / "DTW_2015-16.csv"
    p.write_text("\n".join(lines) + "\n")
    g, pc, q = clean_file(p, "DTW", 2015)
    return g.set_index("hour_ending"), pc, q


def at(g, hhmm):
    return g.loc[pd.Timestamp(f"2015-10-01 {hhmm}", tz="UTC")]


def test_grid_complete(cleaned):
    g, _, _ = cleaned
    assert len(g) == len(season_grid(2015))
    assert g.index.is_unique
    assert g.has_ob.sum() == 13
    missing = at(g, "14:00")
    assert not missing.has_ob and np.isnan(missing.tmpf)


def test_hour_assignment_and_dedupe(cleaned):
    g, _, _ = cleaned
    assert at(g, "02:00").tmpf == 51


def test_trace_and_wx(cleaned):
    g, _, _ = cleaned
    r = at(g, "03:00")
    assert r.p01i == 0.0 and r.p01i_trace == 1 and r.wx_snow == 1 and r.snow_intensity == 1
    r = at(g, "01:00")
    assert r.wx_snow == 0 and r.wx_precip == 0 and r.p01i == 0.0  # wxcodes "M" = no weather


def test_p01i_categories(cleaned):
    g, pc, _ = cleaned
    assert np.isnan(at(g, "04:00").p01i)                                   # a: PNO
    assert at(g, "05:00").p01i == pytest.approx(0.12)                      # b: P group parsed
    assert at(g, "05:00").p01i_filled == 1
    r = at(g, "06:00")
    assert r.p01i == 0.0 and r.p01i_trace == 1 and r.wx_mixed == 1         # c
    r = at(g, "07:00")
    assert r.p01i == 0.0 and r.p01i_trace == 0 and r.p01i_filled == 1      # d
    assert (pc["a_pno"], pc["b_pgroup_parsed"], pc["c_precip_wx_no_pgroup"], pc["d_dry_no_pgroup"]) == (1, 1, 1, 1)


def test_wind(cleaned):
    g, _, _ = cleaned
    r = at(g, "01:00")
    assert r.u == pytest.approx(10) and r.v == pytest.approx(0, abs=1e-5)
    r = at(g, "13:00")
    assert r.u == pytest.approx(0, abs=1e-5) and r.v == pytest.approx(10)
    r = at(g, "07:00")
    assert r.u == 0 and r.v == 0
    r = at(g, "12:00")
    assert np.isnan(r.u) and np.isnan(r.v) and r.sknt == 5


def test_sky(cleaned):
    g, _, _ = cleaned
    r = at(g, "01:00")
    assert r.ceiling_ft == 1500 and r.sky_cover == 4
    r = at(g, "03:00")
    assert r.ceiling_ft == 20000 and r.sky_cover == 2


def test_qc(cleaned):
    g, _, q = cleaned
    assert np.isnan(at(g, "08:00").tmpf)
    assert np.isnan(at(g, "10:00").tmpf)
    assert at(g, "09:00").tmpf == 51 and at(g, "11:00").tmpf == 52
    assert q["range_tmpf"] == 1 and q["spike_tmpf"] == 1
    assert at(g, "01:00").dewpoint_depression == pytest.approx(10)
