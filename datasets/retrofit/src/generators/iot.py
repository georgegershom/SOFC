import random
from datetime import datetime
from typing import Dict, Iterator, List

from ..utils import daterange, diurnal_profile, to_iso


def _end_use_split(function: str) -> Dict[str, float]:
    if function.startswith("residential"):
        return {"hvac": 0.35, "lighting": 0.12, "plug": 0.33, "other": 0.2}
    if function == "office":
        return {"hvac": 0.4, "lighting": 0.25, "plug": 0.25, "other": 0.1}
    if function == "education":
        return {"hvac": 0.38, "lighting": 0.23, "plug": 0.29, "other": 0.1}
    if function == "healthcare":
        return {"hvac": 0.45, "lighting": 0.22, "plug": 0.23, "other": 0.1}
    if function == "retail":
        return {"hvac": 0.36, "lighting": 0.34, "plug": 0.2, "other": 0.1}
    if function == "industrial":
        return {"hvac": 0.3, "lighting": 0.22, "plug": 0.38, "other": 0.1}
    return {"hvac": 0.35, "lighting": 0.22, "plug": 0.33, "other": 0.1}


def generate_iot(
    building_rows: List[Dict],
    start: datetime,
    end: datetime,
    step_minutes: int = 15,
) -> Iterator[Dict]:
    for b in building_rows:
        function = str(b["function"])  # type: ignore
        area = float(b["footprint_area_m2"]) * float(b["floors"]) * 0.8
        end_use = _end_use_split(function)
        base_intensity_wh_m2_h = 10.0 if function.startswith("residential") else 18.0

        for ts in daterange(start, end, step_minutes):
            hour = ts.hour
            # Occupancy profile
            occ_peak = 7 if function.startswith("residential") else 50
            occ_low = 2 if function.startswith("residential") else 5
            occ_profile = diurnal_profile(hour, occ_low, occ_peak, 7 if not function.startswith("residential") else 19, 18 if not function.startswith("residential") else 22)
            occupancy = int(max(0, random.gauss(occ_profile, 3)))

            # Energy use profile in Wh per step
            peak_factor = 2.2 if not function.startswith("residential") else 1.8
            low_factor = 0.5 if not function.startswith("residential") else 0.6
            intensity = diurnal_profile(hour, low_factor, peak_factor, 8, 17)
            wh_per_hour = base_intensity_wh_m2_h * area * intensity
            wh_step = wh_per_hour * (step_minutes / 60.0)

            hvac_wh = wh_step * end_use["hvac"] * random.uniform(0.9, 1.2)
            lighting_wh = wh_step * end_use["lighting"] * random.uniform(0.9, 1.2)
            plug_wh = wh_step * end_use["plug"] * random.uniform(0.9, 1.2)
            other_wh = wh_step * end_use["other"] * random.uniform(0.9, 1.2)

            total_e_wh = hvac_wh + lighting_wh + plug_wh + other_wh
            gas_wh = total_e_wh * (0.15 if function.startswith("residential") else 0.25) * random.uniform(0.8, 1.2)

            # Indoor environmental quality
            base_co2 = 420 + occupancy * random.uniform(8, 15)
            co2_ppm = base_co2 + random.gauss(0, 30)
            tvoc = max(50, random.gauss(120 + occupancy * 2, 40))
            pm25 = max(3, random.gauss(8 + occupancy * 0.1, 3))

            # Indoor temperature and humidity influenced by occupancy
            indoor_temp = 20.5 + (0.015 * occupancy) + random.gauss(0, 0.6)
            indoor_rh = max(20, min(70, 45 + random.gauss(0, 5) - (indoor_temp - 21.0)))

            yield {
                "timestamp": to_iso(ts),
                "building_id": b["building_id"],
                "electricity_wh": round(total_e_wh, 2),
                "gas_wh": round(gas_wh, 2),
                "hvac_wh": round(hvac_wh, 2),
                "lighting_wh": round(lighting_wh, 2),
                "plug_wh": round(plug_wh, 2),
                "co2_ppm": round(co2_ppm, 0),
                "tvoc_ppb": round(tvoc, 0),
                "pm25_ug_m3": round(pm25, 1),
                "indoor_temp_C": round(indoor_temp, 2),
                "indoor_rh_pct": round(indoor_rh, 1),
                "occupancy_count": occupancy,
            }
