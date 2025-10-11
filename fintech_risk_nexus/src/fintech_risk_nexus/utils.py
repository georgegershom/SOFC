from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def get_last_complete_month(today: Optional[date] = None) -> date:
    if today is None:
        today = date.today()
    first_of_this_month = today.replace(day=1)
    last_month_end = first_of_this_month - timedelta(days=1)
    return last_month_end


def month_range(start: date, end: date) -> List[pd.Timestamp]:
    start_ts = pd.Timestamp(year=start.year, month=start.month, day=1)
    end_ts = pd.Timestamp(year=end.year, month=end.month, day=1)
    months = pd.period_range(start=start_ts, end=end_ts, freq="M").to_timestamp()
    return list(months)


def stable_int_from_key(key: str, max_value: int = 2**31 - 1) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max_value


def make_rng_for_key(key: str, seed: int) -> np.random.Generator:
    combined = stable_int_from_key(f"{key}-{seed}")
    return np.random.default_rng(combined)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class TimeSpan:
    start: date
    end: date

    @staticmethod
    def for_years(start_year: int, end_year: int) -> "TimeSpan":
        return TimeSpan(date(start_year, 1, 1), date(end_year, 12, 31))

    def months(self) -> List[pd.Timestamp]:
        return month_range(self.start, self.end)

    def years(self) -> List[int]:
        return list(range(self.start.year, self.end.year + 1))
