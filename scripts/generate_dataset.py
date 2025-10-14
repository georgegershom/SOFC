#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# -------------------------
# Configuration structures
# -------------------------

CLIMATE_ZONES = [
    {
        "id": "CZ-1",
        "name": "Cold-Temperate",
        "avg_temp_by_month": [-5, -3, 2, 8, 14, 18, 21, 20, 15, 8, 2, -3],
        "avg_rh_by_month": [70, 68, 65, 60, 58, 60, 65, 68, 70, 72, 73, 72],
    },
    {
        "id": "CZ-2",
        "name": "Marine-Mild",
        "avg_temp_by_month": [5, 6, 8, 10, 13, 16, 18, 18, 16, 12, 8, 6],
        "avg_rh_by_month": [78, 78, 76, 75, 75, 73, 72, 73, 75, 77, 78, 79],
    },
    {
        "id": "CZ-3",
        "name": "Hot-Humid",
        "avg_temp_by_month": [15, 17, 20, 24, 27, 29, 31, 31, 29, 25, 20, 16],
        "avg_rh_by_month": [65, 65, 66, 68, 70, 74, 76, 76, 74, 70, 68, 66],
    },
]

BUILDING_TYPES = [
    {"type": "residential", "heating_fuel": ["gas", "electric"], "weekday_hours": (6, 9, 17, 23)},
    {"type": "office", "heating_fuel": ["electric", "district"], "weekday_hours": (7, 18)},
    {"type": "educational", "heating_fuel": ["district", "gas"], "weekday_hours": (7, 16)},
    {"type": "retail", "heating_fuel": ["electric", "gas"], "weekday_hours": (9, 21)},
]

STYLES = ["modernist", "art_deco", "vernacular", "contemporary", "brutalist", "postmodern"]
QUALITIES = ["low", "medium", "high"]
ENVELOPE_MATERIALS = ["brick", "concrete", "wood", "steel", "glass", "composite"]

# LCA materials: name, density (kg/m3), embodied carbon (kgCO2e/kg)
LCA_MATERIAL_LIBRARY = [
    {"material": "concrete", "density": 2400, "ec_intensity": 0.12},
    {"material": "brick", "density": 1800, "ec_intensity": 0.22},
    {"material": "timber", "density": 600, "ec_intensity": 0.05},
    {"material": "glass", "density": 2500, "ec_intensity": 1.15},
    {"material": "mineral_wool", "density": 80, "ec_intensity": 1.00},
    {"material": "xps_insulation", "density": 35, "ec_intensity": 3.20},
    {"material": "gypsum_board", "density": 800, "ec_intensity": 0.20},
    {"material": "steel", "density": 7850, "ec_intensity": 2.00},
]

EU_RATINGS = ["A", "B", "C", "D", "E", "F", "G"]

@dataclass
class Building:
    building_id: str
    climate_zone_id: str
    building_type: str
    function: str
    style: str
    quality: str
    year_built: int
    floors: int
    floor_area_m2: float
    height_m: float
    rooftop_area_m2: float
    volume_m3: float
    wall_u_value: float
    roof_u_value: float
    window_u_value: float
    wall_r_value: float
    roof_r_value: float
    window_r_value: float
    primary_heating_fuel: str
    energy_rating: str

# -------------------------
# Helper functions
# -------------------------


def daterange(start: datetime, end: datetime, step_hours: int = 1):
    cur = start
    while cur <= end:
        yield cur
        cur = cur + timedelta(hours=step_hours)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def seasonal_hourly_profile(month: int, hour: int) -> float:
    # diurnal variation: peak mid-afternoon
    diurnal = math.sin((hour - 7) / 24 * 2 * math.pi)
    # mild monthly shape
    monthly = 0.2 * math.sin((month - 1) / 12 * 2 * math.pi)
    return diurnal + monthly


def eu_rating_from_eui(eui_kwh_per_m2y: float) -> str:
    thresholds = [75, 100, 150, 200, 250, 350]  # A .. F; else G
    for idx, thr in enumerate(thresholds):
        if eui_kwh_per_m2y <= thr:
            return EU_RATINGS[idx]
    return "G"


def random_u_value(year_built: int, element: str) -> float:
    # Very rough synthetic mapping of code evolution
    base = {
        "wall": 1.5,
        "roof": 1.0,
        "window": 3.0,
    }[element]
    improvement_factor = 1.0
    if year_built >= 2010:
        improvement_factor = 0.45
    elif year_built >= 2000:
        improvement_factor = 0.6
    elif year_built >= 1990:
        improvement_factor = 0.75
    elif year_built >= 1980:
        improvement_factor = 0.9
    value = base * improvement_factor * random.uniform(0.9, 1.1)
    return round(value, 3)


def generate_buildings(n: int, rng: random.Random) -> List[Building]:
    buildings: List[Building] = []

    for i in range(n):
        btype_info = rng.choice(BUILDING_TYPES)
        btype = btype_info["type"]
        climate = rng.choice(CLIMATE_ZONES)["id"]
        style = rng.choice(STYLES)
        quality = rng.choices(QUALITIES, weights=[0.3, 0.5, 0.2])[0]
        year_built = rng.randint(1965, 2022)
        floors = rng.randint(1, 12) if btype != "office" else rng.randint(3, 25)
        floor_area_m2 = rng.uniform(800, 6000) if btype != "residential" else rng.uniform(60, 2500)
        height_m = floors * rng.uniform(2.7, 3.5)
        rooftop_area_m2 = floor_area_m2 / floors
        volume_m3 = floor_area_m2 * height_m / floors
        wall_u = random_u_value(year_built, "wall")
        roof_u = random_u_value(year_built, "roof")
        window_u = random_u_value(year_built, "window")
        wall_r = round(1.0 / wall_u, 3)
        roof_r = round(1.0 / roof_u, 3)
        window_r = round(1.0 / window_u, 3)
        heating_fuel = rng.choice(btype_info["heating_fuel"])

        # rough EUI baseline by type and climate (kWh/m2-yr)
        climate_bias = {"CZ-1": 1.15, "CZ-2": 1.0, "CZ-3": 1.25}[climate]
        type_base = {"residential": 100, "office": 160, "educational": 140, "retail": 180}[btype]
        envelope_factor = (1.0 + 0.04 * (wall_u - 0.6)) + (0.03 * (roof_u - 0.3)) + (0.02 * (window_u - 1.5))
        eui = type_base * climate_bias * envelope_factor * rng.uniform(0.9, 1.1)
        rating = eu_rating_from_eui(eui)

        buildings.append(
            Building(
                building_id=f"BLD-{i+1:05d}",
                climate_zone_id=climate,
                building_type=btype,
                function=btype,  # alias for now
                style=style,
                quality=quality,
                year_built=year_built,
                floors=floors,
                floor_area_m2=round(floor_area_m2, 2),
                height_m=round(height_m, 2),
                rooftop_area_m2=round(rooftop_area_m2, 2),
                volume_m3=round(volume_m3, 2),
                wall_u_value=wall_u,
                roof_u_value=roof_u,
                window_u_value=window_u,
                wall_r_value=wall_r,
                roof_r_value=roof_r,
                window_r_value=window_r,
                primary_heating_fuel=heating_fuel,
                energy_rating=rating,
            )
        )
    return buildings


def generate_weather(
    start: datetime, end: datetime, zones: List[Dict], rng: random.Random
) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for zone in zones:
        zid = zone["id"]
        out[zid] = []
        for ts in daterange(start, end, 1):
            m_idx = ts.month - 1
            base_temp = zone["avg_temp_by_month"][m_idx]
            base_rh = zone["avg_rh_by_month"][m_idx]
            diurnal = 6.0 * math.sin((ts.hour - 7) / 24 * 2 * math.pi)
            temp = base_temp + diurnal + rng.uniform(-1.2, 1.2)
            rh = clamp(base_rh + rng.uniform(-5, 5) + (-0.2 * diurnal), 25, 95)
            wind = clamp(3.0 + rng.uniform(-1.5, 3.0), 0.0, 15.0)
            out[zid].append(
                {
                    "timestamp": ts.isoformat(),
                    "climate_zone_id": zid,
                    "outdoor_temp_c": round(temp, 2),
                    "outdoor_rh_pct": round(rh, 1),
                    "wind_m_s": round(wind, 2),
                }
            )
    return out


def occupancy_profile(btype: str, ts: datetime, rng: random.Random) -> float:
    wd = ts.weekday()  # 0 Monday
    hr = ts.hour
    # Base schedules by type
    if btype == "office":
        active = wd < 5 and 8 <= hr <= 18
        peak = 11 <= hr <= 14
        base = 0.05
        return base + (0.8 if active else 0.2) + (0.15 if peak else 0.0) + rng.uniform(-0.05, 0.05)
    if btype == "educational":
        active = wd < 5 and 8 <= hr <= 16
        base = 0.05
        return base + (0.85 if active else 0.15) + rng.uniform(-0.05, 0.05)
    if btype == "retail":
        active = 9 <= hr <= 20
        weekend_boost = 0.15 if wd >= 5 else 0.0
        base = 0.1
        return base + (0.7 if active else 0.2) + weekend_boost + rng.uniform(-0.05, 0.05)
    # residential
    morning = 6 <= hr <= 9
    evening = 17 <= hr <= 23
    night = hr <= 5 or hr >= 23
    return (0.2 + (0.5 if morning else 0) + (0.6 if evening else 0) + (0.9 if night else 0) + rng.uniform(-0.05, 0.05))


def compute_energy_enduses(
    building: Building, outdoor_temp_c: float, occupancy_norm: float, rng: random.Random
) -> Dict[str, float]:
    # Setpoints and simple load model
    heat_sp = 20.0
    cool_sp = 24.0
    envelope_leak = 0.6 + 0.2 * (building.wall_u_value + building.roof_u_value + building.window_u_value)
    area = building.floor_area_m2

    # Heating and cooling loads (kWh) scaled by area
    heat_load = 0.0
    cool_load = 0.0
    if outdoor_temp_c < heat_sp:
        delta = heat_sp - outdoor_temp_c
        heat_load = max(0.0, 0.008 * area * delta * (1.0 + 0.3 * envelope_leak))
    elif outdoor_temp_c > cool_sp:
        delta = outdoor_temp_c - cool_sp
        cool_load = max(0.0, 0.006 * area * delta * (1.0 + 0.25 * envelope_leak))

    # Occupancy-driven internal loads and ventilation
    occ_effect = clamp(occupancy_norm, 0.0, 1.5)
    lighting = 0.0018 * area * (0.6 + 0.8 * occ_effect)
    plug = 0.0022 * area * (0.5 + 0.7 * occ_effect)

    # Partition HVAC energy across fuels
    if building.primary_heating_fuel == "gas":
        gas = heat_load * 1.05  # boiler inefficiency
        elec_for_heat = 0.0
    elif building.primary_heating_fuel == "district":
        gas = 0.0
        elec_for_heat = heat_load * 0.2
    else:  # electric
        gas = 0.0
        elec_for_heat = heat_load * 0.35  # heat pump COP ~ 2.8 synthetic

    cooling_elec = cool_load * 0.35

    # Distribution and fans
    fans = 0.0007 * area * (0.6 + 0.6 * occ_effect)

    electricity = lighting + plug + fans + elec_for_heat + cooling_elec
    total = electricity + gas

    enduses = {
        "electricity_kwh": round(electricity, 3),
        "gas_kwh": round(gas, 3),
        "heating_kwh": round(heat_load, 3),
        "cooling_kwh": round(cool_load, 3),
        "lighting_kwh": round(lighting, 3),
        "plug_load_kwh": round(plug, 3),
        "fans_kwh": round(fans, 3),
        "total_kwh": round(total, 3),
    }
    return enduses


def compute_indoor_air_quality(
    building: Building,
    outdoor_temp_c: float,
    outdoor_rh_pct: float,
    occupancy_norm: float,
    rng: random.Random,
) -> Tuple[float, float, float, float, float]:
    # Simple synthetic IAQ model
    base_co2 = 420
    co2 = base_co2 + occupancy_norm * 350 + rng.uniform(-30, 30)

    base_tvoc = 120
    tvoc = base_tvoc + occupancy_norm * 60 + rng.uniform(-40, 40)

    base_pm25 = 8 + (35 - outdoor_temp_c) * 0.05 if outdoor_temp_c < 20 else 10 + (outdoor_temp_c - 20) * 0.15
    pm25 = clamp(base_pm25 + occupancy_norm * 2 + rng.uniform(-3, 6), 2, 75)

    indoor_temp = clamp(21.0 + (outdoor_temp_c - 21.0) * 0.1 + rng.uniform(-0.8, 0.8), 17, 28)
    indoor_rh = clamp(outdoor_rh_pct * 0.7 + rng.uniform(-5, 5), 25, 70)

    return (
        round(indoor_temp, 2),
        round(indoor_rh, 1),
        round(co2, 0),
        round(tvoc, 0),
        round(pm25, 1),
    )


def write_csv(path: str, fieldnames: List[str], rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def generate_lca_for_buildings(buildings: List[Building], rng: random.Random) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    materials_rows: List[Dict] = []
    building_totals_rows: List[Dict] = []
    retrofit_rows: List[Dict] = []

    material_lookup = {m["material"]: m for m in LCA_MATERIAL_LIBRARY}

    for b in buildings:
        wall_area = 2.2 * (b.floor_area_m2)  # perimeter * height rough proxy
        roof_area = b.rooftop_area_m2
        window_area = 0.18 * wall_area

        wall_layer = rng.choice(["brick", "concrete", "timber"])
        insulation = rng.choice(["mineral_wool", "xps_insulation"])
        glazing = "glass"

        thicknesses_m = {
            "brick": 0.24,
            "concrete": 0.2,
            "timber": 0.16,
            "mineral_wool": 0.12,
            "xps_insulation": 0.08,
            "glass": 0.006,
            "gypsum_board": 0.012,
        }

        def layer_mass_kg(area: float, material: str) -> float:
            density = material_lookup[material]["density"]
            thickness = thicknesses_m[material]
            return area * thickness * density

        wall_mass = layer_mass_kg(wall_area, wall_layer) + 0.5 * layer_mass_kg(wall_area, insulation)
        roof_mass = 0.6 * layer_mass_kg(roof_area, insulation) + 0.4 * layer_mass_kg(roof_area, "gypsum_board") if "gypsum_board" in material_lookup else 0.6 * layer_mass_kg(roof_area, insulation)
        window_mass = layer_mass_kg(window_area, glazing)

        def ec_for(material: str, mass: float) -> float:
            return material_lookup[material]["ec_intensity"] * mass

        ec_wall = ec_for(wall_layer, wall_mass)
        ec_insul = ec_for(insulation, 0.5 * layer_mass_kg(wall_area, insulation) + 0.6 * layer_mass_kg(roof_area, insulation))
        ec_glass = ec_for(glazing, window_mass)
        total_ec = ec_wall + ec_insul + ec_glass

        materials_rows.extend(
            [
                {
                    "building_id": b.building_id,
                    "material": wall_layer,
                    "component": "wall",
                    "mass_kg": round(wall_mass, 1),
                    "embodied_carbon_kgco2e": round(ec_wall, 1),
                    "epd_id": f"EPD-{abs(hash(wall_layer)) % 10000:04d}",
                },
                {
                    "building_id": b.building_id,
                    "material": insulation,
                    "component": "envelope_insulation",
                    "mass_kg": round(0.5 * layer_mass_kg(wall_area, insulation) + 0.6 * layer_mass_kg(roof_area, insulation), 1),
                    "embodied_carbon_kgco2e": round(ec_insul, 1),
                    "epd_id": f"EPD-{abs(hash(insulation)) % 10000:04d}",
                },
                {
                    "building_id": b.building_id,
                    "material": glazing,
                    "component": "window_glazing",
                    "mass_kg": round(window_mass, 1),
                    "embodied_carbon_kgco2e": round(ec_glass, 1),
                    "epd_id": f"EPD-{abs(hash(glazing)) % 10000:04d}",
                },
            ]
        )

        building_totals_rows.append(
            {
                "building_id": b.building_id,
                "wall_area_m2": round(wall_area, 1),
                "roof_area_m2": round(roof_area, 1),
                "window_area_m2": round(window_area, 1),
                "embodied_carbon_total_kgco2e": round(total_ec, 1),
            }
        )

        # Retrofit scenario: insulation upgrade + window replacement
        add_r_value = random.uniform(1.0, 3.0)
        new_window_u = clamp(b.window_u_value - random.uniform(0.5, 1.5), 0.7, b.window_u_value)
        hvac_gain_pct = random.uniform(5, 20)
        predicted_savings_pct = clamp(5 + 6 * add_r_value + (b.window_u_value - new_window_u) * 4 + hvac_gain_pct * 0.3, 8, 55)
        retrofit_ec = ec_for("mineral_wool", 0.2 * layer_mass_kg(wall_area + roof_area, "mineral_wool")) + ec_for("glass", 0.4 * window_mass)
        capex = 30 * (wall_area + roof_area) + 250 * (0.4 * window_mass / 15.0)
        payback_years = clamp(capex / (predicted_savings_pct / 100.0 * 25.0 * b.floor_area_m2), 2.0, 25.0)

        retrofit_rows.append(
            {
                "scenario_id": f"RET-{b.building_id}",
                "building_id": b.building_id,
                "measure_type": "envelope+windows+hvac",
                "added_insulation_r_value": round(add_r_value, 2),
                "replacement_window_u_value": round(new_window_u, 2),
                "hvac_efficiency_gain_pct": round(hvac_gain_pct, 1),
                "predicted_savings_pct": round(predicted_savings_pct, 1),
                "embodied_carbon_kgco2e": round(retrofit_ec, 1),
                "capex_usd": int(capex),
                "payback_years": round(payback_years, 1),
            }
        )

    return materials_rows, building_totals_rows, retrofit_rows


def generate(
    output_root: str,
    num_buildings: int,
    start_date: str,
    end_date: str,
    seed: int,
):
    rng = random.Random(seed)
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)

    # 1) Buildings
    buildings = generate_buildings(num_buildings, rng)

    # 2) Weather per climate zone
    weather = generate_weather(start, end, CLIMATE_ZONES, rng)

    # 3) IoT time series
    iot_rows: List[Dict] = []
    occupancy_rows: List[Dict] = []
    energy_ts_rows: List[Dict] = []

    # For aggregation
    agg_energy_by_building_month = defaultdict(lambda: defaultdict(lambda: {"elec": 0.0, "gas": 0.0, "total": 0.0}))

    weather_idx: Dict[str, Dict[str, Dict]] = {}
    for zid, wz in weather.items():
        weather_idx[zid] = {w["timestamp"]: w for w in wz}

    for b in buildings:
        for ts in daterange(start, end, 1):
            ts_iso = ts.isoformat()
            w = weather_idx[b.climate_zone_id][ts_iso]
            occ_norm = clamp(occupancy_profile(b.building_type, ts, rng), 0.0, 2.0)

            enduses = compute_energy_enduses(b, w["outdoor_temp_c"], occ_norm, rng)
            indoor_temp_c, indoor_rh_pct, co2_ppm, tvoc_ppb, pm25 = compute_indoor_air_quality(
                b, w["outdoor_temp_c"], w["outdoor_rh_pct"], occ_norm, rng
            )

            occupants = max(0, int(occ_norm * (b.floor_area_m2 / (12 if b.building_type != "residential" else 35)) + rng.uniform(-2, 2)))

            iot_rows.append(
                {
                    "timestamp": ts_iso,
                    "building_id": b.building_id,
                    "electricity_kwh": enduses["electricity_kwh"],
                    "gas_kwh": enduses["gas_kwh"],
                    "heating_kwh": enduses["heating_kwh"],
                    "cooling_kwh": enduses["cooling_kwh"],
                    "lighting_kwh": enduses["lighting_kwh"],
                    "plug_load_kwh": enduses["plug_load_kwh"],
                    "fans_kwh": enduses["fans_kwh"],
                    "total_kwh": enduses["total_kwh"],
                    "indoor_temp_c": indoor_temp_c,
                    "indoor_rh_pct": indoor_rh_pct,
                    "co2_ppm": co2_ppm,
                    "tvoc_ppb": tvoc_ppb,
                    "pm25_ug_m3": pm25,
                }
            )

            occupancy_rows.append(
                {
                    "timestamp": ts_iso,
                    "building_id": b.building_id,
                    "occupancy_count": occupants,
                }
            )

            energy_ts_rows.append(
                {
                    "timestamp": ts_iso,
                    "building_id": b.building_id,
                    "electricity_kwh": enduses["electricity_kwh"],
                    "gas_kwh": enduses["gas_kwh"],
                    "total_kwh": enduses["total_kwh"],
                }
            )

            mkey = f"{ts.year}-{ts.month:02d}"
            agg = agg_energy_by_building_month[b.building_id][mkey]
            agg["elec"] += enduses["electricity_kwh"]
            agg["gas"] += enduses["gas_kwh"]
            agg["total"] += enduses["total_kwh"]

    # 4) LCA
    lca_material_rows, lca_building_totals, retrofit_rows = generate_lca_for_buildings(buildings, rng)

    # 5) Write outputs
    buildings_path = os.path.join(output_root, "buildings", "buildings.csv")
    write_csv(
        buildings_path,
        [
            "building_id",
            "climate_zone_id",
            "building_type",
            "function",
            "style",
            "quality",
            "year_built",
            "floors",
            "floor_area_m2",
            "height_m",
            "rooftop_area_m2",
            "volume_m3",
            "wall_u_value",
            "roof_u_value",
            "window_u_value",
            "wall_r_value",
            "roof_r_value",
            "window_r_value",
            "primary_heating_fuel",
            "energy_rating",
        ],
        [b.__dict__ for b in buildings],
    )

    # Weather
    weather_rows: List[Dict] = []
    for zid, wz in weather.items():
        weather_rows.extend(wz)
    weather_path = os.path.join(output_root, "iot", "weather_timeseries.csv")
    write_csv(
        weather_path,
        ["timestamp", "climate_zone_id", "outdoor_temp_c", "outdoor_rh_pct", "wind_m_s"],
        weather_rows,
    )

    # IoT energy+IEQ
    iot_path = os.path.join(output_root, "iot", "iot_timeseries.csv")
    write_csv(
        iot_path,
        [
            "timestamp",
            "building_id",
            "electricity_kwh",
            "gas_kwh",
            "heating_kwh",
            "cooling_kwh",
            "lighting_kwh",
            "plug_load_kwh",
            "fans_kwh",
            "total_kwh",
            "indoor_temp_c",
            "indoor_rh_pct",
            "co2_ppm",
            "tvoc_ppb",
            "pm25_ug_m3",
        ],
        iot_rows,
    )

    # Occupancy
    occ_path = os.path.join(output_root, "iot", "occupancy_timeseries.csv")
    write_csv(occ_path, ["timestamp", "building_id", "occupancy_count"], occupancy_rows)

    # Energy (subset, for convenience)
    energy_ts_path = os.path.join(output_root, "energy", "energy_timeseries.csv")
    write_csv(energy_ts_path, ["timestamp", "building_id", "electricity_kwh", "gas_kwh", "total_kwh"], energy_ts_rows)

    # Energy performance monthly and annual
    monthly_rows: List[Dict] = []
    annual_rows: List[Dict] = []
    for bid, months in agg_energy_by_building_month.items():
        total_year_elec = 0.0
        total_year_gas = 0.0
        total_year_total = 0.0
        for mkey, vals in sorted(months.items()):
            monthly_rows.append(
                {
                    "building_id": bid,
                    "month": mkey,
                    "electricity_kwh": round(vals["elec"], 2),
                    "gas_kwh": round(vals["gas"], 2),
                    "total_kwh": round(vals["total"], 2),
                }
            )
            total_year_elec += vals["elec"]
            total_year_gas += vals["gas"]
            total_year_total += vals["total"]
        area = next(b.floor_area_m2 for b in buildings if b.building_id == bid)
        eui = total_year_total / max(1.0, area)
        annual_rows.append(
            {
                "building_id": bid,
                "year": start.year,
                "electricity_kwh": round(total_year_elec, 1),
                "gas_kwh": round(total_year_gas, 1),
                "total_kwh": round(total_year_total, 1),
                "eui_kwh_per_m2": round(eui, 1),
                "energy_rating": eu_rating_from_eui(eui),
            }
        )

    monthly_path = os.path.join(output_root, "energy", "energy_monthly.csv")
    write_csv(monthly_path, ["building_id", "month", "electricity_kwh", "gas_kwh", "total_kwh"], monthly_rows)

    annual_path = os.path.join(output_root, "energy", "energy_annual.csv")
    write_csv(
        annual_path,
        ["building_id", "year", "electricity_kwh", "gas_kwh", "total_kwh", "eui_kwh_per_m2", "energy_rating"],
        annual_rows,
    )

    # LCA
    lca_materials_path = os.path.join(output_root, "lca", "lca_materials.csv")
    write_csv(
        lca_materials_path,
        ["building_id", "material", "component", "mass_kg", "embodied_carbon_kgco2e", "epd_id"],
        lca_material_rows,
    )

    lca_building_totals_path = os.path.join(output_root, "lca", "lca_building_totals.csv")
    write_csv(
        lca_building_totals_path,
        ["building_id", "wall_area_m2", "roof_area_m2", "window_area_m2", "embodied_carbon_total_kgco2e"],
        lca_building_totals,
    )

    retrofit_path = os.path.join(output_root, "lca", "retrofit_scenarios.csv")
    write_csv(
        retrofit_path,
        [
            "scenario_id",
            "building_id",
            "measure_type",
            "added_insulation_r_value",
            "replacement_window_u_value",
            "hvac_efficiency_gain_pct",
            "predicted_savings_pct",
            "embodied_carbon_kgco2e",
            "capex_usd",
            "payback_years",
        ],
        retrofit_rows,
    )

    # Schema index
    schema = {
        "version": "0.1.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "time_range": {"start": start.isoformat(), "end": end.isoformat()},
        "counts": {
            "buildings": len(buildings),
            "iot_rows": len(iot_rows),
            "occupancy_rows": len(occupancy_rows),
            "energy_ts_rows": len(energy_ts_rows),
        },
        "files": {
            "buildings": buildings_path,
            "weather": weather_path,
            "iot_timeseries": iot_path,
            "occupancy_timeseries": occ_path,
            "energy_timeseries": energy_ts_path,
            "energy_monthly": monthly_path,
            "energy_annual": annual_path,
            "lca_materials": lca_materials_path,
            "lca_building_totals": lca_building_totals_path,
            "retrofit_scenarios": retrofit_path,
        },
        "notes": "Synthetic dataset for research prototyping; values are plausible ranges, not real measurements.",
    }
    schema_path = os.path.join(output_root, "dataset_index.json")
    os.makedirs(os.path.dirname(schema_path), exist_ok=True)
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return schema_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic integrated building retrofit dataset")
    parser.add_argument("--output-root", default=os.path.join("/workspace", "data", "synthetic"), help="Root output directory")
    parser.add_argument("--num-buildings", type=int, default=30, help="Number of buildings")
    parser.add_argument("--start-date", default="2024-01-01T00:00:00", help="Start datetime ISO8601")
    parser.add_argument("--end-date", default="2024-03-31T23:00:00", help="End datetime ISO8601")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    schema_path = generate(
        output_root=args.output_root,
        num_buildings=args.num_buildings,
        start_date=args.start_date,
        end_date=args.end_date,
        seed=args.seed,
    )
    print(f"Generated dataset index: {schema_path}")


if __name__ == "__main__":
    main()
