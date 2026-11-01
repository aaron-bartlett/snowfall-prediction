"""Predict one hour from the saved model and the features parquet.

Usage: python -m src.predict --lead 1 --time 2018-01-12T15:00Z
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from src.features import features_path
from src.model import HurdleModel
from src.train import MODELS, model_dir


def predict_hour(lead: int, when: str, models_root=MODELS) -> pd.Series:
    t = pd.Timestamp(when)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    model = HurdleModel.load(model_dir(lead, models_root))
    df = pd.read_parquet(features_path(lead), filters=[("hour_ending", "==", t)])
    if df.empty:
        raise LookupError(f"no feature row for {t} at lead {lead} (outside Oct-Apr or target unavailable)")
    return model.predict(df).iloc[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lead", type=int, required=True, choices=[1, 3])
    ap.add_argument("--time", required=True, help="target hour-ending, ISO format (UTC if no zone)")
    args = ap.parse_args(argv)
    try:
        r = predict_hour(args.lead, args.time)
        out = {"lead_hours": args.lead, "hour_ending": str(pd.Timestamp(args.time)),
               "p_snow": float(r.p_cal), "swe_expected_in": float(r.swe_expected),
               "swe_deterministic_in": float(r.swe_det), "depth_expected_in": float(r.depth)}
        if not all(np.isfinite(v) for k, v in out.items() if isinstance(v, float)):
            raise ValueError(f"non-finite prediction: {out}")
        print(json.dumps(out, indent=2))
    except Exception as e:
        print(f"FAILED predict: {e}")
        return 1
    print("DONE predict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
