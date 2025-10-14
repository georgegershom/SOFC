import csv
import os
import random
from datetime import datetime, timedelta
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, headers: List[str], rows: Iterable[Dict]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in headers})


def seed_everything(seed: int) -> None:
    random.seed(seed)


def daterange(start: datetime, end: datetime, step_minutes: int) -> Iterator[datetime]:
    current = start
    delta = timedelta(minutes=step_minutes)
    while current <= end:
        yield current
        current += delta


def to_iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def seasonal_temperature(day_of_year: int, latitude: float) -> float:
    # Simple sinusoidal annual temperature baseline model
    amp = 12.0 + min(15.0, abs(latitude) * 0.15)
    mean = 12.0 if abs(latitude) < 25 else (9.0 if abs(latitude) < 45 else 6.0)
    phase_shift = 200  # northern hemisphere warmest ~ day 200
    return mean + amp * __import__("math").sin(2 * __import__("math").pi * (day_of_year - phase_shift) / 365.0)


def diurnal_profile(hour: int, low: float, high: float, start_peak: int, end_peak: int) -> float:
    if start_peak <= hour <= end_peak:
        return high
    if hour < start_peak:
        return low + (high - low) * max(0.0, (hour - (start_peak - 3)) / 3.0)
    # after peak
    return low + (high - low) * max(0.0, ((end_peak + 3) - hour) / 3.0)


def choose_weighted(options: Sequence[Tuple[str, float]]) -> str:
    labels, weights = zip(*options)
    total = sum(weights)
    r = random.uniform(0, total)
    upto = 0.0
    for label, w in options:
        if upto + w >= r:
            return label
        upto += w
    return labels[-1]
