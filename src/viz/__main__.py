"""Draw the docs figures.

Usage:
  python -m src.viz                 # draw every figure whose inputs are committed (works on a clean clone)
  python -m src.viz --summaries     # first recompute reports/summaries/* from local data/processed (author only)
  python -m src.viz --check --out DIR   # CI smoke build into DIR
"""
from __future__ import annotations

import argparse
import sys
import traceback

from src.viz import figures as F
from src.viz import style


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summaries", action="store_true")
    ap.add_argument("--check", action="store_true", help="skip figures whose inputs are missing instead of failing")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", help="figure function names, e.g. f20_skill_summary")
    args = ap.parse_args(argv)
    try:
        if args.summaries:
            from src.viz import summaries
            summaries.run()
        style.apply()
        fns = F.TIER_A + F.FROM_SUMMARIES + F.FROM_ABLATION
        if args.only:
            fns = [f for f in fns if f.__name__ in args.only]
        for fn in fns:
            try:
                fn(args.out)
            except FileNotFoundError as e:
                if not args.check:
                    raise
                print(f"  skipped {fn.__name__}: {e}")
    except Exception as e:
        traceback.print_exc()
        print(f"FAILED viz: {e}")
        return 1
    print("DONE viz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
