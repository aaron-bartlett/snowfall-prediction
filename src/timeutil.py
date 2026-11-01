"""Shared time conventions: hour-ending assignment, target hours, LST dates.

Used by both station coverage (Phase 1) and cleaning (Phase 3) so the two agree.
"""
from __future__ import annotations

import pandas as pd

TARGET_MONTHS = (10, 11, 12, 1, 2, 3, 4)


def assign_hour_ending(valid: pd.Series) -> pd.Series:
    """Map ob times (UTC) to the top of the hour they report for.

    minute 40-59 -> next top of the hour (e.g. 14:53 -> 15:00)
    minute 0-10  -> same top of the hour (e.g. 15:05 -> 15:00)
    minute 11-39 -> NaT (intra-hour AWOS reports are dropped)
    """
    valid = pd.to_datetime(valid, utc=True)
    minute = valid.dt.minute
    floor = valid.dt.floor("h")
    out = pd.Series(pd.NaT, index=valid.index, dtype=floor.dtype)
    late = minute >= 40
    early = minute <= 10
    out[late] = floor[late] + pd.Timedelta(hours=1)
    out[early] = floor[early]
    return out


def keep_latest_per_hour(df: pd.DataFrame, keys=("station", "hour_ending"), time_col="valid") -> pd.DataFrame:
    """Drop rows without an hour_ending and keep the latest ob for each key."""
    df = df[df["hour_ending"].notna()]
    df = df.sort_values(list(keys) + [time_col], kind="stable")
    return df.drop_duplicates(list(keys), keep="last")


def target_hours(season_year: int, months=TARGET_MONTHS) -> pd.DatetimeIndex:
    """Hour-ending timestamps for a season: Oct 1 00Z (year) through Apr 30 23Z (year + 1)."""
    idx = pd.date_range(f"{season_year}-10-01 00:00", f"{season_year + 1}-05-01 00:00",
                        freq="h", inclusive="left", tz="UTC")
    return idx[idx.month.isin(months)]


def season_of(hour_ending: pd.Series) -> pd.Series:
    """Season start year of a timestamp: Sep-Dec -> same year, Jan-Aug -> previous year."""
    he = pd.to_datetime(hour_ending, utc=True)
    return (he.dt.year - (he.dt.month < 9).astype(int)).astype("Int64")


def lst_date(hour_ending: pd.Series, offset_hours: int = -5) -> pd.Series:
    """Local-standard-time calendar date that an hour-ending timestamp belongs to.

    Hour-ending 05Z = 00:00 LST covers 23:00-24:00 LST of the previous day.
    """
    he = pd.to_datetime(hour_ending, utc=True)
    return (he + pd.Timedelta(hours=offset_hours) - pd.Timedelta(minutes=1)).dt.date
