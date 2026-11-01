"""Download IEM ASOS station-season CSVs and the KDTW GHCN-Daily file.

Usage:
  python -m src.download --all
  python -m src.download --asos [--station DTW --season 2024]
  python -m src.download --ghcn
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

from src.config import ROOT, load_config, season_label
from src.stations import STATIONS_PATH

ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
GHCN_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{ghcn_id}.csv.gz"
RAW_ASOS = ROOT / "data" / "raw" / "asos"
RAW_GHCN = ROOT / "data" / "raw" / "ghcnd"
MANIFEST_PATH = ROOT / "data" / "raw" / "manifest.csv"
RETRY_WAITS = (5, 10, 20, 40)


def build_asos_url(sid: str, year: int, columns, window=("09-01", "05-01")) -> str:
    m1, d1 = (int(x) for x in window[0].split("-"))
    m2, d2 = (int(x) for x in window[1].split("-"))
    params = {"station": sid, "data": ",".join(columns),
              "year1": year, "month1": m1, "day1": d1,
              "year2": year + 1, "month2": m2, "day2": d2,
              "tz": "Etc/UTC", "format": "onlycomma", "missing": "M", "trace": "T", "report_type": 3}
    return f"{ASOS_URL}?{urlencode(params, safe=',')}"


def asos_path(sid: str, year: int) -> Path:
    return RAW_ASOS / sid / f"{sid}_{season_label(year)}.csv"


def expected_header(columns) -> set[str]:
    return {"station", "valid", *columns}


def header_ok(first_line: str, columns) -> bool:
    cols = first_line.strip().split(",")
    return cols[:2] == ["station", "valid"] and set(cols) == expected_header(columns)


def validate_file(path: Path, columns) -> tuple[bool, int]:
    """Valid = correct header (station, valid, then exactly the requested columns) and >= 1 data row."""
    path = Path(path)
    if path.suffix == ".part" or not path.exists() or path.stat().st_size == 0:
        return False, 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
        if not header_ok(first, columns):
            return False, 0
        rows = sum(1 for line in fh if line.strip())
    return rows >= 1, rows


def download_one(sid: str, year: int, cfg, get=requests.get, sleep=time.sleep) -> str:
    """Download one station-season file. Returns 'skipped', 'ok', or 'failed'."""
    path = asos_path(sid, year)
    cols = cfg.asos_columns
    if validate_file(path, cols)[0]:
        return "skipped"
    path.parent.mkdir(parents=True, exist_ok=True)
    url = build_asos_url(sid, year, cols, (cfg.download_window["start"], cfg.download_window["end"]))
    waits = RETRY_WAITS[: cfg.download["retries"]]
    for attempt in range(len(waits) + 1):
        try:
            r = get(url, timeout=cfg.download["timeout_s"])
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            text = r.text
            if not header_ok(text.split("\n", 1)[0], cols):
                raise RuntimeError(f"bad header/body: {text[:80]!r}")
            tmp = path.with_name(path.name + ".part")
            tmp.write_text(text, encoding="utf-8")
            tmp.rename(path)
            sleep(cfg.download["sleep_s"])
            return "ok"
        except Exception as e:  # noqa: BLE001 - retry any network/HTTP/body failure
            if attempt < len(waits):
                sleep(waits[attempt])
            else:
                print(f"  failed {sid} {season_label(year)}: {e}", flush=True)
    return "failed"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(sids, years, cfg) -> pd.DataFrame:
    rows = []
    for sid in sids:
        for y in years:
            p = asos_path(sid, y)
            ok, n = validate_file(p, cfg.asos_columns)
            rows.append({"sid": sid, "season": season_label(y), "path": str(p.relative_to(ROOT)),
                         "rows": n, "bytes": p.stat().st_size if p.exists() else 0,
                         "sha256": sha256(p) if ok else "", "status": "ok" if ok else "failed",
                         "downloaded_at": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()
                         if p.exists() else ""})
    df = pd.DataFrame(rows)
    tmp = MANIFEST_PATH.with_suffix(".csv.part")
    df.to_csv(tmp, index=False)
    tmp.rename(MANIFEST_PATH)
    return df


def download_asos(sids, years, cfg) -> pd.DataFrame:
    jobs = [(s, y) for s in sids for y in years]
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()

    def run(job):
        res = download_one(*job, cfg)
        with lock:
            counts[res] += 1
            n = sum(counts.values())
            if n % 25 == 0 or n == len(jobs):
                print(f"[{n}/{len(jobs)}] {counts}", flush=True)
        return res

    with ThreadPoolExecutor(cfg.download["workers"]) as ex:
        list(ex.map(run, jobs))
    return write_manifest(sids, years, cfg)


def download_ghcn(cfg) -> Path:
    RAW_GHCN.mkdir(parents=True, exist_ok=True)
    path = RAW_GHCN / f"{cfg.ghcn_id}.csv.gz"
    r = requests.get(GHCN_URL.format(ghcn_id=cfg.ghcn_id), timeout=cfg.download["timeout_s"])
    r.raise_for_status()
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(r.content)
    tmp.rename(path)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--asos", action="store_true")
    ap.add_argument("--ghcn", action="store_true")
    ap.add_argument("--station")
    ap.add_argument("--season", type=int, help="season start year, e.g. 2024 for 2024-25")
    args = ap.parse_args(argv)
    try:
        cfg = load_config()
        if args.asos or args.all:
            if args.station:
                years = [args.season] if args.season else cfg.season_years
                for y in years:
                    print(f"{args.station} {season_label(y)}: {download_one(args.station, y, cfg)}")
                    print(f"  rows={validate_file(asos_path(args.station, y), cfg.asos_columns)[1]}")
            else:
                sids = pd.read_csv(STATIONS_PATH).sid.tolist()
                man = download_asos(sids, cfg.season_years, cfg)
                print(f"manifest: {man.status.value_counts().to_dict()}, "
                      f"{man.bytes.sum() / 1e9:.2f} GB")
        if args.ghcn or args.all:
            print(f"GHCN written to {download_ghcn(cfg)}")
    except Exception as e:
        print(f"FAILED download: {e}")
        return 1
    print("DONE download")
    return 0


if __name__ == "__main__":
    sys.exit(main())
