"""Parse feature names into family / station / variable / lag."""
from __future__ import annotations

import re

import pandas as pd

LAG_RE = re.compile(r"_lag(\d+)$")
WINDOWED = ("_tend", "_swe_sum", "_hours_since_snow", "_snow_streak")
FAMILY_ORDER = ["KDTW lags", "KDTW windowed", "Neighbor raw lags", "Spatial aggregates", "Time"]


def parse(name: str, target: str = "DTW") -> dict:
    m = LAG_RE.search(name)
    lag = int(m.group(1)) if m else None
    base = LAG_RE.sub("", name)
    if name.startswith(("hod_", "doy_")):
        return {"feature": name, "family": "Time", "station": None, "var": name, "lag": None}
    if name.startswith("agg_"):
        return {"feature": name, "family": "Spatial aggregates", "station": None, "var": base[4:], "lag": lag}
    sid, var = base.split("_", 1)
    if sid == target:
        fam = "KDTW windowed" if lag is None or any(w in name for w in WINDOWED) else "KDTW lags"
        return {"feature": name, "family": fam, "station": sid, "var": var, "lag": lag}
    return {"feature": name, "family": "Neighbor raw lags", "station": sid, "var": var, "lag": lag}


def frame(names) -> pd.DataFrame:
    return pd.DataFrame([parse(n) for n in names])
