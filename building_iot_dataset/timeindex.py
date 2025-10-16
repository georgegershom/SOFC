from __future__ import annotations
import pandas as pd
import pytz
from typing import Literal


def build_time_index(year: int, tz: str, interval: Literal["15min", "1h"] = "15min") -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz=pytz.UTC)
    end = pd.Timestamp(f"{year}-12-31 23:59:59", tz=pytz.UTC)
    # build at UTC then convert; we will align to exact intervals in local tz to avoid DST issues
    naive_local_start = pd.Timestamp(f"{year}-01-01 00:00:00").tz_localize(tz)
    naive_local_end = pd.Timestamp(f"{year}-12-31 23:59:59").tz_localize(tz)
    freq = interval
    # Generate in local tz with closed='left' to produce exact bin starts
    index = pd.date_range(start=naive_local_start, end=naive_local_end, freq=freq, inclusive="both")
    return index
