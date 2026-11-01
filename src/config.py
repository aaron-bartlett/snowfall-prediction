"""Load and validate config.yaml."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "config.yaml"

REQUIRED_KEYS = [
    "target_station", "ghcn_id", "networks", "radius_km", "coverage", "seasons",
    "target_months", "download_window", "split", "lead_hours", "lags", "lst_offset_hours",
    "slr", "asos_columns", "download", "train",
]


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    target_station: str
    ghcn_id: str
    networks: tuple
    radius_km: float
    coverage: dict
    seasons: dict
    target_months: tuple
    download_window: dict
    split: dict
    lead_hours: tuple
    lags: dict
    lst_offset_hours: int
    slr: dict
    asos_columns: tuple
    download: dict
    train: dict

    @property
    def season_years(self) -> list[int]:
        return list(range(self.seasons["first"], self.seasons["last"] + 1))


def season_label(year: int) -> str:
    """2005 -> '2005-06' (Oct 2005 - Apr 2006)."""
    return f"{year}-{(year + 1) % 100:02d}"


def label_to_year(label: str) -> int:
    return int(label[:4])


def validate(raw: dict) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ConfigError(f"missing config keys: {missing}")
    fr = raw["split"]
    total = fr["train"] + fr["val"] + fr["test"]
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ConfigError(f"split fractions sum to {total}, expected 1.0")
    leads = list(raw["lead_hours"])
    if not leads or not set(leads) <= {1, 3}:
        raise ConfigError(f"lead_hours must be a non-empty subset of {{1, 3}}, got {leads}")
    if raw["seasons"]["first"] > raw["seasons"]["last"]:
        raise ConfigError("seasons.first must be <= seasons.last")
    for k in ("min_temp", "min_precip"):
        if not 0 < raw["coverage"][k] <= 1:
            raise ConfigError(f"coverage.{k} must be in (0, 1]")


def load_config(path: str | Path = DEFAULT_PATH) -> Config:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    validate(raw)
    return Config(
        target_station=raw["target_station"],
        ghcn_id=raw["ghcn_id"],
        networks=tuple(raw["networks"]),
        radius_km=float(raw["radius_km"]),
        coverage=dict(raw["coverage"]),
        seasons=dict(raw["seasons"]),
        target_months=tuple(raw["target_months"]),
        download_window=dict(raw["download_window"]),
        split=dict(raw["split"]),
        lead_hours=tuple(raw["lead_hours"]),
        lags=dict(raw["lags"]),
        lst_offset_hours=int(raw["lst_offset_hours"]),
        slr=dict(raw["slr"]),
        asos_columns=tuple(raw["asos_columns"]),
        download=dict(raw["download"]),
        train=dict(raw["train"]),
    )
