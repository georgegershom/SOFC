from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import List, Dict


@dataclass
class SensorMeta:
    name: str
    unit: str
    description: str
    category: str
    location: str


def build_metadata(columns: List[str]) -> Dict[str, dict]:
    # Simple heuristics to assign units and categories
    result = {}
    for col in columns:
        if col.endswith("_temp_c"):
            unit = "C"
            cat = "IEQ"
        elif col.endswith("_rh_pct") or col == "ambient_rh_pct":
            unit = "%"
            cat = "IEQ"
        elif col.endswith("_co2_ppm"):
            unit = "ppm"
            cat = "IEQ"
        elif col.endswith("_tvoc_ppb"):
            unit = "ppb"
            cat = "IEQ"
        elif col.endswith("_pm25_ugm3") or col.endswith("_pm10_ugm3"):
            unit = "ug/m3"
            cat = "IEQ"
        elif col.endswith("_lux"):
            unit = "lux"
            cat = "IEQ"
        elif col.endswith("_noise_dba"):
            unit = "dBA"
            cat = "IEQ"
        elif col.endswith("_motion"):
            unit = "binary"
            cat = "Occupancy"
        elif col.endswith("_kw"):
            unit = "kW"
            cat = "Energy"
        elif col.endswith("_speed_pct"):
            unit = "%"
            cat = "HVAC"
        elif col.endswith("_air_temp_c") and col.startswith("ahu_"):
            unit = "C"
            cat = "HVAC"
        elif col in ("chiller_status", "boiler_status", "occupied"):
            unit = "binary"
            cat = "HVAC"
        elif col.endswith("_m3ph"):
            unit = "m3/h"
            cat = "Water"
        elif col.endswith("_pos") or col.endswith("_frac"):
            unit = "[0-1]"
            cat = "Controls"
        elif col.endswith("_occupancy") or col.endswith("_count"):
            unit = "count"
            cat = "Occupancy"
        elif col in ("wind_speed_mps",):
            unit = "m/s"
            cat = "Weather"
        elif col in ("wind_dir_deg",):
            unit = "deg"
            cat = "Weather"
        elif col in ("solar_irradiance_wm2",):
            unit = "W/m2"
            cat = "Weather"
        elif col in ("rain_mm",):
            unit = "mm"
            cat = "Weather"
        elif col in ("cloud_cover_pct",):
            unit = "%"
            cat = "Weather"
        elif col in ("ambient_temp_c",):
            unit = "C"
            cat = "Weather"
        else:
            unit = ""
            cat = "Other"

        location = "building" if "zone_" not in col else col.split("_")[0]
        result[col] = asdict(SensorMeta(name=col, unit=unit, description=col.replace("_", " "), category=cat, location=location))
    return result
