#!/usr/bin/env python3
import argparse
import csv
import gzip
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import List, Tuple, Dict, Optional
import math
import random


# -----------------------------
# Utility and domain constants
# -----------------------------
FUNCTION_TYPES = [
    "residential",
    "office",
    "school",
    "retail",
    "hospital",
    "warehouse",
]

BUILDING_STYLES = [
    "traditional",
    "modernist",
    "brutalist",
    "contemporary",
    "postmodern",
]

BUILD_QUALITY = [
    "poor",
    "average",
    "good",
    "excellent",
]

WALL_MATERIALS = [
    "brick",
    "concrete",
    "wood",
    "steel",
    "aerated_concrete",
]

ROOF_MATERIALS = [
    "bitumen",
    "metal",
    "tile",
    "membrane",
]

WINDOW_GLAZING = [
    "single",
    "double",
    "triple",
]

RETROFIT_SCOPES = [
    "envelope",
    "hvac",
    "lighting",
    "controls",
]

# Embodied carbon intensities (very approximate, illustrative only)
# Units: kgCO2e per kg of material
EPD_FACTORS = {
    "brick": 0.22,
    "concrete": 0.12,
    "wood": 0.05,
    "steel": 1.9,
    "aerated_concrete": 0.18,
    "bitumen": 0.5,
    "metal": 2.0,
    "tile": 0.2,
    "membrane": 2.5,
    "glass": 1.4,
    "insulation_mineral_wool": 1.2,
    "insulation_eps": 2.6,
}

# Energy intensity (kWh/m2/year) ranges by function and vintage band (approx.)
ENERGY_INTENSITY_TABLE = {
    # values are tuples (low, high)
    ("residential", "pre1980"): (150, 260),
    ("residential", "1980_2000"): (120, 180),
    ("residential", "post2000"): (70, 130),
    ("office", "pre1980"): (200, 320),
    ("office", "1980_2000"): (160, 240),
    ("office", "post2000"): (110, 180),
    ("school", "pre1980"): (160, 280),
    ("school", "1980_2000"): (130, 200),
    ("school", "post2000"): (90, 150),
    ("retail", "pre1980"): (220, 360),
    ("retail", "1980_2000"): (170, 260),
    ("retail", "post2000"): (130, 200),
    ("hospital", "pre1980"): (280, 420),
    ("hospital", "1980_2000"): (220, 340),
    ("hospital", "post2000"): (160, 260),
    ("warehouse", "pre1980"): (100, 180),
    ("warehouse", "1980_2000"): (80, 140),
    ("warehouse", "post2000"): (60, 110),
}

# Map energy intensity (kWh/m2/yr) to EU-like A-G rating (illustrative)
RATING_BINS = [50, 90, 130, 170, 210, 260, math.inf]
RATING_LABELS = ["A", "B", "C", "D", "E", "F", "G"]


@dataclass
class GenerationConfig:
    num_buildings: int
    iot_fraction: float
    iot_days: int
    iot_interval_minutes: int
    seed: int
    start_date: str


def ensure_dirs() -> Dict[str, str]:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    data_raw = os.path.join(base, "data", "raw")
    data_processed = os.path.join(base, "data", "processed")
    os.makedirs(data_raw, exist_ok=True)
    os.makedirs(data_processed, exist_ok=True)
    return {"base": base, "raw": data_raw, "processed": data_processed}


def random_from_range(low: float, high: float, rng: random.Random) -> float:
    return float(rng.uniform(low, high))


def pick_weighted(options: List[str], weights: List[float], rng: random.Random) -> str:
    total = float(sum(weights))
    x = rng.uniform(0.0, total)
    c = 0.0
    for opt, w in zip(options, weights):
        c += w
        if x <= c:
            return str(opt)
    return str(options[-1])


def assign_vintage(construction_year: int) -> str:
    if construction_year < 1980:
        return "pre1980"
    if construction_year <= 2000:
        return "1980_2000"
    return "post2000"


def derive_energy_intensity(function_type: str, vintage: str, rng: random.Random) -> float:
    low, high = ENERGY_INTENSITY_TABLE[(function_type, vintage)]
    return random_from_range(low, high, rng)


def derive_rating(energy_intensity: float) -> str:
    # Pandas cut-like rating for a single value
    for threshold, label in zip(RATING_BINS, RATING_LABELS):
        if energy_intensity <= threshold:
            return label
    return RATING_LABELS[-1]


def derive_u_values(construction_year: int, quality: str, glazing: str, rng: random.Random) -> Tuple[float, float, float]:
    # Coarse, illustrative U-values (W/m2K) based on vintage and quality
    # Walls, roof, windows
    vintage = assign_vintage(construction_year)
    base = {
        "pre1980": (1.2, 0.9, 4.5),
        "1980_2000": (0.8, 0.6, 3.0),
        "post2000": (0.4, 0.25, 1.6),
    }[vintage]

    quality_adjust = {
        "poor": 1.2,
        "average": 1.0,
        "good": 0.85,
        "excellent": 0.7,
    }[quality]

    glazing_adjust = {
        "single": 1.4,
        "double": 1.0,
        "triple": 0.6,
    }[glazing]

    wall_u = base[0] * quality_adjust * random_from_range(0.95, 1.05, rng)
    roof_u = base[1] * quality_adjust * random_from_range(0.95, 1.05, rng)
    window_u = base[2] * glazing_adjust * random_from_range(0.95, 1.05, rng)
    return wall_u, roof_u, window_u


def generate_buildings_table(cfg: GenerationConfig, dirs: Dict[str, str]) -> List[Dict[str, object]]:
    rng = random.Random(cfg.seed)

    buildings: List[Dict[str, object]] = []
    for i in range(cfg.num_buildings):
        building_id = f"B{str(i+1).zfill(5)}"

        # Location within a broad Europe-like bounding box
        lat = rng.uniform(35.0, 60.0)
        lon = rng.uniform(-10.0, 25.0)

        construction_year = rng.randint(1950, 2022)
        function_type = pick_weighted(
            FUNCTION_TYPES, [0.45, 0.2, 0.12, 0.12, 0.06, 0.05], rng
        )
        style = rng.choice(BUILDING_STYLES)
        build_quality = pick_weighted(BUILD_QUALITY, [0.15, 0.5, 0.25, 0.10], rng)

        # Geometry
        floors = rng.randint(1, 20)
        floor_height_m = rng.uniform(2.8, 3.6)
        rooftop_area_m2 = rng.uniform(60.0, 6000.0)
        footprint_area_m2 = rooftop_area_m2
        gross_floor_area_m2 = footprint_area_m2 * floors
        height_m = floors * floor_height_m
        volume_m3 = gross_floor_area_m2 * rng.uniform(2.5, 3.2)

        wall_material = rng.choice(WALL_MATERIALS)
        roof_material = rng.choice(ROOF_MATERIALS)
        window_glazing = rng.choice(WINDOW_GLAZING)

        wall_u, roof_u, window_u = derive_u_values(
            construction_year, build_quality, window_glazing, rng
        )
        vintage = assign_vintage(construction_year)
        baseline_eui = derive_energy_intensity(function_type, vintage, rng)
        rating = derive_rating(baseline_eui)

        monitored_iot = (rng.random() < cfg.iot_fraction)

        buildings.append(
            {
                "building_id": building_id,
                "latitude": lat,
                "longitude": lon,
                "function_type": function_type,
                "style": style,
                "build_quality": build_quality,
                "construction_year": construction_year,
                "num_floors": floors,
                "floor_height_m": floor_height_m,
                "rooftop_area_m2": rooftop_area_m2,
                "footprint_area_m2": footprint_area_m2,
                "gross_floor_area_m2": gross_floor_area_m2,
                "height_m": height_m,
                "volume_m3": volume_m3,
                "wall_material": wall_material,
                "roof_material": roof_material,
                "window_glazing": window_glazing,
                "wall_u_value_w_m2k": wall_u,
                "roof_u_value_w_m2k": roof_u,
                "window_u_value_w_m2k": window_u,
                "baseline_energy_intensity_kwh_m2y": baseline_eui,
                "energy_rating": rating,
                "monitored_iot": monitored_iot,
            }
        )

    out_path = os.path.join(dirs["processed"], "buildings.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(buildings[0].keys()),
        )
        writer.writeheader()
        writer.writerows(buildings)

    return buildings


def generate_epd_factors_table(dirs: Dict[str, str]) -> List[Dict[str, object]]:
    rows = [
        {
            "material": mat,
            "kgco2e_per_kg": factor,
            "source_note": "illustrative_factor_not_for_design_use",
        }
        for mat, factor in EPD_FACTORS.items()
    ]
    out_path = os.path.join(dirs["processed"], "epd_factors.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["material", "kgco2e_per_kg", "source_note"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def generate_material_inventory(buildings: List[Dict[str, object]], rng: random.Random, dirs: Dict[str, str]) -> List[Dict[str, object]]:
    # Very coarse bill of quantities by building, inferred from geometry
    inventories: List[Dict[str, object]] = []

    # Material densities (kg/m2 for layers or kg/m3 simplified) – illustrative
    density_map: Dict[str, float] = {
        "brick": 170,  # kg/m2 of wall surface (layered)
        "concrete": 240,  # wall surface
        "wood": 60,
        "steel": 35,
        "aerated_concrete": 120,
        "bitumen": 5,  # kg/m2 roof cover
        "metal": 15,
        "tile": 45,
        "membrane": 4,
        "glass": 25,  # kg/m2 window
        "insulation_mineral_wool": 8,
        "insulation_eps": 6,
    }

    for row in buildings:
        building_id = str(row["building_id"])
        wall_area_m2 = float(row["gross_floor_area_m2"]) * 1.2  # perimeter factor
        roof_area_m2 = float(row["rooftop_area_m2"]) * 1.0
        window_area_m2 = float(row["gross_floor_area_m2"]) * 0.15

        wall_mat = str(row["wall_material"])
        roof_mat = str(row["roof_material"])

        # Choose insulation type
        insulation = pick_weighted([
            "insulation_mineral_wool", "insulation_eps"
        ], [0.6, 0.4], rng)

        # Mass estimates
        wall_mass_kg = wall_area_m2 * density_map[wall_mat]
        roof_mass_kg = roof_area_m2 * density_map[roof_mat]
        window_mass_kg = window_area_m2 * density_map["glass"]
        insulation_mass_kg = (wall_area_m2 + roof_area_m2) * density_map[insulation] * rng.uniform(0.8, 1.2)

        inventories.extend(
            [
                {
                    "building_id": building_id,
                    "component": "walls",
                    "material": wall_mat,
                    "mass_kg": wall_mass_kg,
                },
                {
                    "building_id": building_id,
                    "component": "roof",
                    "material": roof_mat,
                    "mass_kg": roof_mass_kg,
                },
                {
                    "building_id": building_id,
                    "component": "windows",
                    "material": "glass",
                    "mass_kg": window_mass_kg,
                },
                {
                    "building_id": building_id,
                    "component": "envelope_insulation",
                    "material": insulation,
                    "mass_kg": insulation_mass_kg,
                },
            ]
        )

    out_path = os.path.join(dirs["processed"], "lca_material_inventory.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["building_id", "component", "material", "mass_kg"]
        )
        writer.writeheader()
        writer.writerows(inventories)

    return inventories


def compute_lca_results(inventory: List[Dict[str, object]], epd_rows: List[Dict[str, object]], dirs: Dict[str, str]) -> List[Dict[str, object]]:
    factors: Dict[str, float] = {row["material"]: float(row["kgco2e_per_kg"]) for row in epd_rows}
    by_building: Dict[str, float] = {}
    for item in inventory:
        mat = str(item["material"])
        mass = float(item["mass_kg"])
        factor = factors.get(mat, 0.0)
        by_building[item["building_id"]] = by_building.get(item["building_id"], 0.0) + mass * factor

    results = [
        {"building_id": b, "embodied_carbon_kgco2e": value}
        for b, value in sorted(by_building.items())
    ]

    out_path = os.path.join(dirs["processed"], "lca_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["building_id", "embodied_carbon_kgco2e"])
        writer.writeheader()
        writer.writerows(results)

    return results


def seasonal_multiplier(month: int) -> float:
    # Simple sinusoid to emulate heating/cooling loads
    # Peak in January and July
    winter = 1.0 + 0.25 * math.cos((month - 1) / 12.0 * 2.0 * math.pi)
    summer = 1.0 + 0.15 * math.cos((month - 7) / 12.0 * 2.0 * math.pi)
    return 0.5 * (winter + summer)


def generate_energy_performance(buildings: List[Dict[str, object]], rng: random.Random, dirs: Dict[str, str]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []

    start_year = 2022
    months: List[Tuple[int, int, date]] = []
    # 4 years inclusive
    for year in range(start_year, start_year + 4):
        for month in range(1, 13):
            # end-of-month date
            if month == 12:
                period_end = date(year, 12, 31)
            else:
                period_end = (date(year, month + 1, 1) - timedelta(days=1))
            months.append((year, month, period_end))

    for row in buildings:
        building_id = str(row["building_id"])
        area = float(row["gross_floor_area_m2"])
        baseline_eui = float(row["baseline_energy_intensity_kwh_m2y"])  # kWh/m2/yr
        baseline_annual_kwh = baseline_eui * area

        # Assign retrofit status and date
        is_retrofit = (rng.random() < 0.45)
        retrofit_end_date: Optional[date] = None
        scopes: List[str] = []
        savings_fraction = 0.0
        if is_retrofit:
            # Choose a retrofit month in the second year (mid series)
            retrofit_idx = rng.randint(12, 23)
            retrofit_end_date = months[retrofit_idx][2]
            # Pick scopes
            pick = [rng.random() < p for p in (0.6, 0.5, 0.5, 0.4)]
            scopes = [s for s, m in zip(RETROFIT_SCOPES, pick) if m]
            # Savings based on scopes
            savings_fraction = (
                (0.18 if "envelope" in scopes else 0.0)
                + (0.12 if "hvac" in scopes else 0.0)
                + (0.08 if "lighting" in scopes else 0.0)
                + (0.05 if "controls" in scopes else 0.0)
            )
            savings_fraction = float(min(0.5, savings_fraction * rng.uniform(0.8, 1.1)))

        for (year, month, period_end) in months:
            seasonal = seasonal_multiplier(month)
            noise = rng.gauss(1.0, 0.05)
            monthly_share = 1.0 / 12.0 * seasonal * noise
            monthly_kwh = baseline_annual_kwh * monthly_share

            if is_retrofit and retrofit_end_date is not None and period_end > retrofit_end_date:
                monthly_kwh *= (1.0 - savings_fraction)

            records.append(
                {
                    "building_id": building_id,
                    "period": str(period_end),
                    "year": year,
                    "month": month,
                    "monthly_energy_kwh": monthly_kwh,
                    "retrofit": is_retrofit,
                    "retrofit_date": str(retrofit_end_date) if retrofit_end_date is not None else "",
                    "retrofit_scopes": ",".join(scopes) if scopes else "",
                    "savings_fraction_post": (savings_fraction if (is_retrofit and retrofit_end_date is not None and period_end > retrofit_end_date) else 0.0),
                }
            )

    out_path = os.path.join(dirs["processed"], "energy_performance_monthly.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "building_id",
                "period",
                "year",
                "month",
                "monthly_energy_kwh",
                "retrofit",
                "retrofit_date",
                "retrofit_scopes",
                "savings_fraction_post",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    return records


def simulate_outdoor_weather_series(start_ts: datetime, end_ts: datetime, lat: float, rng: random.Random, freq_minutes: int) -> List[Tuple[datetime, float, float]]:
    """Return list of (timestamp, outdoor_temp_c, outdoor_rh_pct)."""
    timestamps: List[datetime] = []
    t = start_ts
    while t < end_ts:
        timestamps.append(t)
        t = t + timedelta(minutes=freq_minutes)

    # Lat influences seasonal amplitude
    lat_norm = (lat - 35.0) / (60.0 - 35.0)
    base_temp = 12.0 + (1 - lat_norm) * 6.0
    seasonal_amp = 10.0 + lat_norm * 8.0

    out: List[Tuple[datetime, float, float]] = []
    samples_per_day = int(24 * 60 // freq_minutes)
    for idx, ts in enumerate(timestamps):
        days_since_start = idx / samples_per_day
        hour_in_day = idx % samples_per_day
        seasonal_component = seasonal_amp * math.sin(2 * math.pi * days_since_start / 365.0 - 0.5 * math.pi)
        diurnal_component = 3.5 * math.sin(2 * math.pi * (hour_in_day / samples_per_day) - math.pi / 2)
        noise = rng.gauss(0.0, 1.2)
        temp_c = base_temp + seasonal_component + diurnal_component + noise
        rh = 60.0 + 15.0 * math.sin(2 * math.pi * days_since_start / 7.0) + rng.gauss(0.0, 5.0)
        rh = max(25.0, min(95.0, rh))
        out.append((ts, temp_c, rh))
    return out


def generate_iot_timeseries(buildings: List[Dict[str, object]], cfg: GenerationConfig, rng: random.Random, dirs: Dict[str, str]) -> None:
    start_ts = datetime.fromisoformat(cfg.start_date)
    end_ts = start_ts + timedelta(days=cfg.iot_days)

    monitored = [b for b in buildings if bool(b["monitored_iot"]) ]
    if not monitored:
        return None

    out_path = os.path.join(dirs["processed"], "iot_timeseries.csv.gz")
    with gzip.open(out_path, "wt", newline="") as gz:
        writer = csv.writer(gz)
        writer.writerow([
            "timestamp",
            "building_id",
            "energy_wh_total",
            "energy_wh_hvac",
            "energy_wh_lighting",
            "co2_ppm",
            "tvoc_ppb",
            "pm25_ugm3",
            "indoor_temp_c",
            "indoor_rh_pct",
            "outdoor_temp_c",
            "outdoor_rh_pct",
            "occupancy_count",
        ])

        for row in monitored:
            building_id = str(row["building_id"])
            area = float(row["gross_floor_area_m2"])
            lat = float(row["latitude"])
            function_type = str(row["function_type"])
            baseline_eui = float(row["baseline_energy_intensity_kwh_m2y"])  # kWh/m2/yr

            weather = simulate_outdoor_weather_series(start_ts, end_ts, lat, rng, cfg.iot_interval_minutes)

            samples_per_day = int(24 * 60 // cfg.iot_interval_minutes)
            annual_kwh = baseline_eui * area
            N = len(weather)
            # average per-interval allocation (rough)
            base_wh_per_interval = (annual_kwh * 1000.0) / max(1, N)

            # Occupancy baseline by function type
            base_occ_per_1000m2 = {
                "residential": 25,
                "office": 55,
                "school": 60,
                "retail": 35,
                "hospital": 45,
                "warehouse": 10,
            }.get(function_type, 30)

            # Initialize IEQ dynamics
            prev_co2 = 420.0
            vent_decay = 0.02 * (cfg.iot_interval_minutes / 15.0)
            co2_emission_per_person = 8.0 * (cfg.iot_interval_minutes / 15.0)

            for idx, (ts, out_temp, out_rh) in enumerate(weather):
                hour = ts.hour
                dow = ts.weekday()

                # Schedule factor by function type
                base = 0.2
                if function_type == "residential":
                    base = 0.35
                    if 6 <= hour <= 8 or 18 <= hour <= 23:
                        base = 0.75
                    if dow in (5, 6):
                        base = 0.65
                elif function_type == "office":
                    base = 0.1
                    if 8 <= hour <= 18 and dow < 5:
                        base = 0.8
                elif function_type == "school":
                    base = 0.1
                    if 8 <= hour <= 15 and dow < 5:
                        base = 0.85
                elif function_type == "retail":
                    base = 0.2
                    if 10 <= hour <= 20:
                        base = 0.7
                    if dow in (5, 6):
                        base = 0.85
                elif function_type == "hospital":
                    base = 0.6
                elif function_type == "warehouse":
                    base = 0.15
                    if 7 <= hour <= 17 and dow < 6:
                        base = 0.55

                occ = int(max(0, round((area / 1000.0) * base_occ_per_1000m2 * base * rng.uniform(0.8, 1.2))))

                # Energy allocation
                seasonal_wt = 1.0 + 0.25 * math.sin(2 * math.pi * (idx / (samples_per_day * 365.0)))
                schedule_wt = 0.4 + 0.6 * base
                total_wh = base_wh_per_interval * seasonal_wt * schedule_wt

                # End-use shares
                hvac_share = 0.45 + 0.15 * (1 + math.sin(2 * math.pi * (out_temp - 5.0) / 35.0)) / 2.0
                hvac_share = min(0.65, max(0.35, hvac_share))
                lighting_share = min(0.25, max(0.10, 0.12 + 0.08 * schedule_wt))
                plug_share = min(0.5, max(0.2, 1.0 - hvac_share - lighting_share))
                total = hvac_share + lighting_share + plug_share
                hvac_share /= total
                lighting_share /= total
                plug_share /= total

                energy_wh_hvac = total_wh * hvac_share
                energy_wh_lighting = total_wh * lighting_share
                energy_wh_plug = total_wh * plug_share
                energy_wh_total = energy_wh_hvac + energy_wh_lighting + energy_wh_plug

                # IEQ dynamics
                gen = occ * co2_emission_per_person
                co2 = max(400.0, prev_co2 * (1.0 - vent_decay) + gen)
                if co2 > 2000:
                    co2 -= 0.3 * (co2 - 1000)
                prev_co2 = co2

                tvoc = max(50.0, min(1200.0, 80.0 + 2.0 * occ + rng.gauss(0.0, 20.0)))
                pm25 = max(3.0, min(120.0, 10.0 + 0.3 * max(0.0, out_rh - 70.0) + rng.gauss(0.0, 3.0)))
                # random indoor event spikes
                if rng.random() < (1.0 / (samples_per_day * 7)):
                    pm25 += rng.uniform(10.0, 40.0)

                heating_set = 21.0
                cooling_set = 24.0
                indoor_temp = max(18.0, min(27.0, (
                    heating_set if out_temp < heating_set else (
                        cooling_set if out_temp > cooling_set else out_temp
                    )
                ) + rng.gauss(0.0, 0.6)))
                indoor_rh = max(20.0, min(70.0, out_rh - rng.gauss(5.0, 3.0)))

                writer.writerow([
                    ts.isoformat(),
                    building_id,
                    round(energy_wh_total, 3),
                    round(energy_wh_hvac, 3),
                    round(energy_wh_lighting, 3),
                    round(co2, 2),
                    round(tvoc, 2),
                    round(pm25, 2),
                    round(indoor_temp, 2),
                    round(indoor_rh, 2),
                    round(out_temp, 2),
                    round(out_rh, 2),
                    int(occ),
                ])

    return None


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic multi-faceted building retrofit dataset")
    parser.add_argument("--num-buildings", type=int, default=200)
    parser.add_argument("--iot-fraction", type=float, default=0.3)
    parser.add_argument("--iot-days", type=int, default=90)
    parser.add_argument("--iot-interval-minutes", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", type=str, default="2024-01-01")

    args = parser.parse_args()
    cfg = GenerationConfig(
        num_buildings=args.num_buildings,
        iot_fraction=args.iot_fraction,
        iot_days=args.iot_days,
        iot_interval_minutes=args.iot_interval_minutes,
        seed=args.seed,
        start_date=args.start_date,
    )

    dirs = ensure_dirs()
    rng = random.Random(cfg.seed)

    buildings = generate_buildings_table(cfg, dirs)
    epd_rows = generate_epd_factors_table(dirs)
    inventory = generate_material_inventory(buildings, rng, dirs)
    compute_lca_results(inventory, epd_rows, dirs)
    generate_energy_performance(buildings, rng, dirs)
    generate_iot_timeseries(buildings, cfg, rng, dirs)

    print("Generated dataset under:", dirs["processed"]) 


if __name__ == "__main__":
    main()
