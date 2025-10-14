import random
from datetime import datetime
from typing import Dict, Iterable, Iterator, List

from ..utils import daterange, seasonal_temperature, to_iso


def generate_weather(
    building_rows: List[Dict],
    start: datetime,
    end: datetime,
    step_minutes: int = 60,
) -> Iterator[Dict]:
    for b in building_rows:
        lat = float(b["latitude"])
        for ts in daterange(start, end, step_minutes):
            doy = int(ts.strftime("%j"))
            base_temp = seasonal_temperature(doy, lat)
            hour = ts.hour

            # Diurnal variation and noise
            diurnal = 4.0 * __import__("math").sin(2 * __import__("math").pi * (hour - 6) / 24.0)
            temp = base_temp + diurnal + random.gauss(0, 1.2)
            rh = max(20.0, min(100.0, 60.0 - (temp - base_temp) * 1.2 + random.gauss(0, 7)))

            ghi = max(0.0, 700.0 * __import__("math").sin(__import__("math").pi * (hour / 24.0)))
            wind = max(0.0, random.gauss(3.5, 1.5))

            yield {
                "timestamp": to_iso(ts),
                "building_id": b["building_id"],
                "dry_bulb_C": round(temp, 2),
                "relative_humidity_pct": round(rh, 1),
                "global_horizontal_irradiance_W_m2": round(ghi, 1),
                "wind_speed_m_s": round(wind, 2),
                "co2_outdoor_ppm": round(random.gauss(420, 20), 0),
                "pm25_outdoor_ug_m3": round(max(2.0, random.gauss(12, 8)), 1),
                "tvoc_outdoor_ppb": round(max(20.0, random.gauss(120, 60)), 0),
            }
