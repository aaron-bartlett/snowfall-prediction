"""Randomly assign whole winter seasons to train/val/test (generated once, never regenerated)."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.config import ROOT, load_config, season_label

SPLITS_PATH = ROOT / "meta" / "splits.json"


def make_splits(years: list[int], seed: int, val_frac: float, test_frac: float) -> dict:
    perm = np.random.default_rng(seed).permutation(years)
    n = len(years)
    n_test = round(test_frac * n)
    n_val = round(val_frac * n)
    test = sorted(int(y) for y in perm[:n_test])
    val = sorted(int(y) for y in perm[n_test:n_test + n_val])
    train = sorted(int(y) for y in perm[n_test + n_val:])
    return {
        "seed": seed,
        "train": [season_label(y) for y in train],
        "val": [season_label(y) for y in val],
        "test": [season_label(y) for y in test],
    }


def load_splits(path=SPLITS_PATH) -> dict:
    with open(path) as fh:
        return json.load(fh)


def season_to_split(splits: dict) -> dict[str, str]:
    return {s: name for name in ("train", "val", "test") for s in splits[name]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite an existing splits.json")
    args = ap.parse_args(argv)
    try:
        if SPLITS_PATH.exists() and not args.force:
            raise RuntimeError(f"{SPLITS_PATH} exists; splits are frozen (use --force to overwrite)")
        cfg = load_config()
        sp = make_splits(cfg.season_years, cfg.split["seed"], cfg.split["val"], cfg.split["test"])
        SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = SPLITS_PATH.with_suffix(".json.part")
        tmp.write_text(json.dumps(sp, indent=2) + "\n")
        tmp.rename(SPLITS_PATH)
        for k in ("train", "val", "test"):
            print(f"{k} ({len(sp[k])}): {', '.join(sp[k])}")
    except Exception as e:
        print(f"FAILED splits: {e}")
        return 1
    print("DONE splits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
