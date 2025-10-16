#!/usr/bin/env python3
import json
import csv
import math
import random
import string
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError

# ---------------------------- Config ---------------------------- #
DATASET_ROOT = Path("/workspace/datasets/building_dna")
SAMPLES_IFC_DIR = DATASET_ROOT / "geometry" / "samples" / "ifc"
SAMPLES_LIDAR_DIR = DATASET_ROOT / "geometry" / "samples" / "lidar"
BUILDINGS_DIR = DATASET_ROOT / "buildings"
METADATA_DIR = DATASET_ROOT / "metadata"
SVG_FLOORPLAN_DIRNAME = "floorplans"

# Candidate URLs for open IFC/LiDAR samples (try in order until one succeeds)
IFC_SAMPLE_CANDIDATES = {
    # Popular samples from various public repositories (raw content URLs)
    "AC20-FZK-Haus.ifc": [
        "https://raw.githubusercontent.com/ifcjs/test-ifc-files/main/IFC/IFC2X3/AC20-FZK-Haus.ifc",
        "https://raw.githubusercontent.com/IFCjs/test-ifc-files/main/IFC/IFC2X3/AC20-FZK-Haus.ifc",
    ],
    "AC-20-Smiley-West.ifc": [
        "https://raw.githubusercontent.com/buildingSMART/Sample-Test-Files/main/IFC%204/Architecture/AC-20-Smiley-West.ifc",
    ],
    "AC11-Institute-Var-2.ifc": [
        "https://raw.githubusercontent.com/buildingSMART/Sample-Test-Files/main/IFC%202x3/AC11-Institute-Var-2.ifc",
    ],
    "Duplex_A_20110505.ifc": [
        "https://raw.githubusercontent.com/ifcjs/test-ifc-files/main/IFC/IFC2X3/Duplex_A_20110505.ifc",
        "https://raw.githubusercontent.com/IfcOpenShell/ifcopenshell/master/test/files/Duplex_A_20110505.ifc",
    ],
}

LIDAR_SAMPLE_CANDIDATES = {
    "autzen.laz": [
        # PDAL test data (small sample)
        "https://raw.githubusercontent.com/PDAL/data/master/autzen/autzen.laz",
        "https://raw.githubusercontent.com/hobuinc/laz-perf/master/autzen/autzen.laz",
    ],
    "autzen.las": [
        "https://raw.githubusercontent.com/LAStools/LAStools/master/data/autzen.las",
    ],
}

# ---------------------------- Utilities ---------------------------- #

def ensure_dirs() -> None:
    for p in [DATASET_ROOT, SAMPLES_IFC_DIR, SAMPLES_LIDAR_DIR, BUILDINGS_DIR, METADATA_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def download_if_missing(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"Downloading {url} -> {dest}")
    urlretrieve(url, dest)


def download_from_any(urls: List[str], dest: Path) -> bool:
    """Try multiple URLs until one succeeds. Returns True on success."""
    for u in urls:
        try:
            download_if_missing(u, dest)
            # If file is present and non-empty, success
            if dest.exists() and dest.stat().st_size > 0:
                return True
        except (HTTPError, URLError) as e:
            print(f"Warning: failed to download {u}: {e}")
        except Exception as e:
            print(f"Warning: unexpected error downloading {u}: {e}")
    return dest.exists() and dest.stat().st_size > 0


# ---------------------------- Fabrication Logic ---------------------------- #

MATERIAL_DB = {
    # name: (conductivity W/mK, density kg/m3, specific_heat J/kgK)
    "brick": (0.77, 1800, 840),
    "gypsum_board": (0.25, 800, 1090),
    "mineral_wool": (0.04, 45, 840),
    "eps": (0.036, 20, 1300),
    "xps": (0.029, 35, 1400),
    "concrete": (1.4, 2400, 880),
    "wood": (0.12, 600, 1700),
    "asphalt_shingle": (0.75, 1200, 920),
    "plywood": (0.13, 540, 1500),
}

WINDOW_FRAME_TYPES = ["aluminum", "uPVC", "wood", "fiberglass"]
GAS_FILLS = ["air", "argon", "krypton"]
GLAZING_TYPES = ["single", "double", "triple"]
FUEL_TYPES = ["electricity", "natural_gas", "fuel_oil", "district_heat"]
LIGHTING_TYPES = ["LED", "CFL", "Incandescent"]
LIGHTING_CONTROLS = ["switch", "dimmer", "occupancy_sensor", "daylight_sensor"]
PV_ORIENTATIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def random_name(rng: random.Random) -> str:
    adjectives = ["Oak", "Maple", "Cedar", "Pine", "Elm", "River", "Sunset", "Hill", "Lake", "Valley"]
    nouns = ["House", "Heights", "Plaza", "Center", "Point", "Court", "Terrace", "Tower", "Commons", "Works"]
    return f"{rng.choice(adjectives)} {rng.choice(nouns)}"


def random_address(rng: random.Random) -> str:
    streets = ["Main St", "Broadway", "Elm St", "Oak Ave", "2nd Ave", "Maple Rd", "Cedar Blvd", "Pine Ln"]
    return f"{rng.randint(100, 9999)} {rng.choice(streets)}, City {rng.randint(1, 200)}, ST {rng.randint(1, 50):02d}"


def rnd_gauss_pos(rng: random.Random, mu: float, sigma: float, min_val: float, max_val: float) -> float:
    v = max(min_val, min(max_val, rng.gauss(mu, sigma)))
    return round(v, 3)


def gen_wall_assembly(rng: random.Random) -> Dict:
    # Typical exterior wall: cladding, sheathing, insulation, structure, gypsum
    layers = []
    # Exterior cladding
    layers.append({
        "material": "brick",
        "thickness_m": round(rng.uniform(0.07, 0.12), 3),
    })
    # Sheathing / air gap
    layers.append({
        "material": "plywood",
        "thickness_m": round(rng.uniform(0.01, 0.02), 3),
    })
    # Insulation
    insulation = rng.choice(["mineral_wool", "eps", "xps"])
    layers.append({
        "material": insulation,
        "thickness_m": round(rng.uniform(0.05, 0.18), 3),
    })
    # Structure
    layers.append({
        "material": "wood",
        "thickness_m": round(rng.uniform(0.09, 0.15), 3),
    })
    # Interior finish
    layers.append({
        "material": "gypsum_board",
        "thickness_m": round(rng.uniform(0.012, 0.016), 3),
    })
    u_value, r_value, areal_heat_cap = compute_u_value(layers)
    return {
        "type": "exterior_wall",
        "layers": enrich_layers(layers),
        "overall_u_W_m2K": round(u_value, 3),
        "overall_r_m2K_W": round(r_value, 3),
        "areal_heat_capacity_kJ_m2K": round(areal_heat_cap / 1000.0, 2),
    }


def gen_roof_assembly(rng: random.Random) -> Dict:
    layers = []
    layers.append({"material": "asphalt_shingle", "thickness_m": round(rng.uniform(0.01, 0.02), 3)})
    layers.append({"material": "plywood", "thickness_m": round(rng.uniform(0.015, 0.025), 3)})
    insulation = rng.choice(["mineral_wool", "eps", "xps"])
    layers.append({"material": insulation, "thickness_m": round(rng.uniform(0.12, 0.25), 3)})
    layers.append({"material": "wood", "thickness_m": round(rng.uniform(0.1, 0.2), 3)})
    u_value, r_value, areal_heat_cap = compute_u_value(layers)
    return {
        "type": "roof",
        "layers": enrich_layers(layers),
        "overall_u_W_m2K": round(u_value, 3),
        "overall_r_m2K_W": round(r_value, 3),
        "areal_heat_capacity_kJ_m2K": round(areal_heat_cap / 1000.0, 2),
    }


def gen_floor_assembly(rng: random.Random) -> Dict:
    layers = []
    layers.append({"material": "plywood", "thickness_m": round(rng.uniform(0.015, 0.025), 3)})
    layers.append({"material": "wood", "thickness_m": round(rng.uniform(0.12, 0.22), 3)})
    layers.append({"material": "concrete", "thickness_m": round(rng.uniform(0.1, 0.2), 3)})
    u_value, r_value, areal_heat_cap = compute_u_value(layers)
    return {
        "type": "floor",
        "layers": enrich_layers(layers),
        "overall_u_W_m2K": round(u_value, 3),
        "overall_r_m2K_W": round(r_value, 3),
        "areal_heat_capacity_kJ_m2K": round(areal_heat_cap / 1000.0, 2),
    }


def enrich_layers(layers: List[Dict]) -> List[Dict]:
    enriched = []
    for layer in layers:
        name = layer["material"]
        k, rho, c = MATERIAL_DB[name]
        t = float(layer["thickness_m"])
        enriched.append({
            "material": name,
            "thickness_m": t,
            "conductivity_W_mK": k,
            "density_kg_m3": rho,
            "specific_heat_J_kgK": c,
            "r_layer_m2K_W": round(t / k, 4),
            "areal_heat_capacity_J_m2K": round(rho * t * c, 1),
        })
    return enriched


def compute_u_value(layers: List[Dict]) -> Tuple[float, float, float]:
    # Sum of thermal resistances + film coefficients (simplified):
    r_si = 0.13  # interior surface resistance
    r_se = 0.04  # exterior surface resistance
    r_layers = 0.0
    areal_heat_cap = 0.0
    for layer in layers:
        k, rho, c = MATERIAL_DB[layer["material"]]
        t = float(layer["thickness_m"])
        r_layers += t / k
        areal_heat_cap += rho * t * c
    r_total = r_si + r_layers + r_se
    u_value = 1.0 / r_total
    return u_value, r_total, areal_heat_cap


def gen_windows_doors(rng: random.Random) -> Dict:
    num_windows = rng.randint(6, 30)
    windows = []
    for _ in range(num_windows):
        glazing = rng.choice(GLAZING_TYPES)
        u = {
            "single": rng.uniform(4.8, 6.0),
            "double": rng.uniform(1.6, 3.0),
            "triple": rng.uniform(0.8, 1.4),
        }[glazing]
        shgc = {
            "single": rng.uniform(0.65, 0.85),
            "double": rng.uniform(0.45, 0.65),
            "triple": rng.uniform(0.35, 0.55),
        }[glazing]
        vt = rng.uniform(0.5, 0.8)
        windows.append({
            "area_m2": round(rng.uniform(0.8, 3.2), 2),
            "u_W_m2K": round(u, 2),
            "shgc": round(shgc, 2),
            "vt": round(vt, 2),
            "glazing": glazing,
            "frame_type": rng.choice(WINDOW_FRAME_TYPES),
            "gas_fill": rng.choice(GAS_FILLS),
            "age_years": rng.randint(0, 40),
            "condition": rng.choice(["excellent", "good", "fair", "poor"]),
        })
    doors = [{
        "area_m2": round(rng.uniform(1.8, 2.4), 2),
        "u_W_m2K": round(rng.uniform(1.2, 2.0), 2),
        "type": rng.choice(["insulated", "solid_wood", "metal"]),
        "age_years": rng.randint(1, 40),
        "condition": rng.choice(["excellent", "good", "fair", "poor"]),
    } for _ in range(rng.randint(1, 4))]
    return {"windows": windows, "doors": doors}


def gen_air_tightness(rng: random.Random) -> Dict:
    return {
        "ach50": round(rng.uniform(1.0, 10.0), 2),
        "measured_year": rng.choice([None] + list(range(2000, date.today().year + 1))),
        "test_method": rng.choice(["blower_door", "estimated"]),
    }


def gen_hvac(rng: random.Random) -> List[Dict]:
    systems = []
    archetypes = [
        "ashep_split",
        "gas_furnace_ac",
        "boiler_radiator",
        "vrf_heat_pump",
    ]
    arche = rng.choice(archetypes)
    if arche == "ashep_split":
        systems.append({
            "type": "air_source_heat_pump_split",
            "make": rng.choice(["Daikin", "Mitsubishi", "Carrier", "LG"]),
            "model": f"HP-{rng.randint(9000, 59999)}",
            "fuel": "electricity",
            "age_years": rng.randint(0, 20),
            "cooling_SEER": round(rng.uniform(13.0, 21.5), 1),
            "heating_HSPF": round(rng.uniform(8.0, 12.0), 1),
            "rated_capacity_kW": round(rng.uniform(5.0, 25.0), 1),
            "maintenance_history": gen_maintenance(rng),
        })
    elif arche == "gas_furnace_ac":
        systems.append({
            "type": "gas_furnace",
            "make": rng.choice(["Trane", "Lennox", "Goodman", "Rheem"]),
            "model": f"GF-{rng.randint(9000, 59999)}",
            "fuel": "natural_gas",
            "age_years": rng.randint(0, 30),
            "efficiency_AFUE": round(rng.uniform(0.78, 0.97), 2),
            "rated_capacity_kW": round(rng.uniform(10.0, 40.0), 1),
            "maintenance_history": gen_maintenance(rng),
        })
        systems.append({
            "type": "dx_ac",
            "make": rng.choice(["Trane", "Lennox", "Goodman", "Rheem"]),
            "model": f"AC-{rng.randint(9000, 59999)}",
            "fuel": "electricity",
            "age_years": rng.randint(0, 20),
            "efficiency_SEER": round(rng.uniform(13.0, 18.0), 1),
            "rated_capacity_kW": round(rng.uniform(5.0, 30.0), 1),
            "maintenance_history": gen_maintenance(rng),
        })
    elif arche == "boiler_radiator":
        systems.append({
            "type": "boiler",
            "make": rng.choice(["Weil-McLain", "Navien", "Viessmann"]),
            "model": f"BLR-{rng.randint(1000, 9999)}",
            "fuel": rng.choice(["natural_gas", "fuel_oil", "district_heat"]),
            "age_years": rng.randint(0, 40),
            "efficiency_thermal": round(rng.uniform(0.75, 0.92), 2),
            "rated_capacity_kW": round(rng.uniform(20.0, 200.0), 1),
            "maintenance_history": gen_maintenance(rng),
        })
    else:  # vrf_heat_pump
        systems.append({
            "type": "vrf_heat_pump",
            "make": rng.choice(["Daikin", "Mitsubishi", "Toshiba"]),
            "model": f"VRF-{rng.randint(1000, 9999)}",
            "fuel": "electricity",
            "age_years": rng.randint(0, 20),
            "cooling_EER": round(rng.uniform(9.0, 14.0), 1),
            "heating_COP": round(rng.uniform(2.5, 4.5), 2),
            "rated_capacity_kW": round(rng.uniform(20.0, 150.0), 1),
            "maintenance_history": gen_maintenance(rng),
        })
    return systems


def gen_dhw(rng: random.Random) -> Dict:
    arche = rng.choice(["heat_pump", "gas_tank", "electric_tank", "tankless_gas"])
    base = {
        "make": rng.choice(["AO Smith", "Rheem", "Bradford White", "Navien"]),
        "model": f"DHW-{rng.randint(1000, 9999)}",
        "age_years": rng.randint(0, 25),
        "tank_volume_l": round(rng.uniform(100, 300), 1),
        "maintenance_history": gen_maintenance(rng),
    }
    if arche == "heat_pump":
        base.update({"type": "heat_pump_dhw", "fuel": "electricity", "efficiency_COP": round(rng.uniform(2.0, 3.0), 2)})
    elif arche == "gas_tank":
        base.update({"type": "gas_tank", "fuel": "natural_gas", "efficiency_EF": round(rng.uniform(0.55, 0.7), 2)})
    elif arche == "electric_tank":
        base.update({"type": "electric_tank", "fuel": "electricity", "efficiency_EF": round(rng.uniform(0.85, 0.95), 2)})
    else:
        base.update({"type": "tankless_gas", "fuel": "natural_gas", "efficiency_UEF": round(rng.uniform(0.8, 0.96), 2)})
    return base


def gen_lighting(rng: random.Random) -> Dict:
    fixtures = []
    for kind in LIGHTING_TYPES:
        fixtures.append({
            "type": kind,
            "count": rng.randint(10, 400),
            "watt_per_fixture": {
                "LED": rng.randint(6, 18),
                "CFL": rng.randint(9, 26),
                "Incandescent": rng.randint(40, 100),
            }[kind],
            "controls": rng.sample(LIGHTING_CONTROLS, k=rng.randint(1, 3)),
        })
    return {"fixtures": fixtures}


def gen_renewables(rng: random.Random) -> List[Dict]:
    systems = []
    if rng.random() < 0.6:
        systems.append({
            "type": "solar_pv",
            "dc_capacity_kW": round(rng.uniform(3.0, 100.0), 1),
            "module_efficiency": round(rng.uniform(0.16, 0.22), 2),
            "orientation": rng.choice(PV_ORIENTATIONS),
            "tilt_deg": round(rng.uniform(5, 35), 1),
            "age_years": rng.randint(0, 20),
            "inverter_efficiency": round(rng.uniform(0.94, 0.99), 2),
        })
    if rng.random() < 0.15:
        systems.append({
            "type": "solar_thermal",
            "collector_area_m2": round(rng.uniform(4.0, 40.0), 1),
            "orientation": rng.choice(PV_ORIENTATIONS),
            "tilt_deg": round(rng.uniform(10, 45), 1),
            "age_years": rng.randint(0, 25),
        })
    return systems


def gen_maintenance(rng: random.Random) -> List[Dict]:
    n = rng.randint(0, 4)
    records = []
    for _ in range(n):
        d = date(rng.randint(2010, date.today().year), rng.randint(1, 12), rng.randint(1, 28))
        records.append({"date": d.isoformat(), "action": rng.choice(["filter_change", "annual_service", "repair", "inspection"])})
    return records


def gen_floorplans_svg(rng: random.Random, out_dir: Path, num_storeys: int, width_m: float, depth_m: float) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_paths = []
    for level in range(1, num_storeys + 1):
        svg_path = out_dir / f"level_{level}.svg"
        svg_paths.append(str(svg_path.relative_to(DATASET_ROOT)))
        # Simple rectangular plan subdivided into rooms
        rooms_x = rng.randint(2, 5)
        rooms_y = rng.randint(2, 5)
        room_w = width_m / rooms_x
        room_h = depth_m / rooms_y
        svg_w = int(width_m * 50)  # pixels per meter
        svg_h = int(depth_m * 50)
        svg_elems = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{svg_w}' height='{svg_h}' viewBox='0 0 {svg_w} {svg_h}'>",
            "<rect x='0' y='0' width='100%' height='100%' fill='white' stroke='black' stroke-width='3' />",
        ]
        # draw grid rooms
        for ix in range(rooms_x):
            for iy in range(rooms_y):
                x = int(ix * room_w * 50)
                y = int(iy * room_h * 50)
                w = int(room_w * 50)
                h = int(room_h * 50)
                svg_elems.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' fill='none' stroke='gray' stroke-width='1' />")
        # dims and labels
        svg_elems.append(
            f"<text x='{svg_w/2}' y='{20}' text-anchor='middle' font-family='Arial' font-size='14'>Level {level} Plan ({width_m}m x {depth_m}m)</text>"
        )
        svg_elems.append("</svg>")
        svg_content = "\n".join(svg_elems)
        svg_path.write_text(svg_content, encoding="utf-8")
    return svg_paths


def gen_geometry(rng: random.Random, building_id: str, num_storeys: int) -> Dict:
    width_m = round(rng.uniform(10.0, 60.0), 1)
    depth_m = round(rng.uniform(10.0, 60.0), 1)
    storey_height_m = round(rng.uniform(2.8, 3.6), 2)
    footprint_area_m2 = round(width_m * depth_m, 1)
    gross_floor_area_m2 = round(footprint_area_m2 * num_storeys, 1)
    volume_m3 = round(gross_floor_area_m2 * storey_height_m, 1)
    # Envelope area rough estimate (perimeter * height + roof)
    perimeter_m = 2.0 * (width_m + depth_m)
    envelope_area_m2 = round(perimeter_m * (num_storeys * storey_height_m) + footprint_area_m2, 1)

    floorplan_dir = DATASET_ROOT / "geometry" / "buildings" / building_id / SVG_FLOORPLAN_DIRNAME
    svg_rel = gen_floorplans_svg(rng, floorplan_dir, num_storeys, width_m, depth_m)

    # external sample references (list only files that actually exist after download)
    ifc_refs = []
    for name in IFC_SAMPLE_CANDIDATES.keys():
        p = (SAMPLES_IFC_DIR / name)
        if p.exists():
            ifc_refs.append(str(p.relative_to(DATASET_ROOT)))
    lidar_refs = []
    for name in LIDAR_SAMPLE_CANDIDATES.keys():
        p = (SAMPLES_LIDAR_DIR / name)
        if p.exists():
            lidar_refs.append(str(p.relative_to(DATASET_ROOT)))

    return {
        "footprint_area_m2": footprint_area_m2,
        "gross_floor_area_m2": gross_floor_area_m2,
        "envelope_area_m2": envelope_area_m2,
        "volume_m3": volume_m3,
        "storey_height_m": storey_height_m,
        "num_storeys": num_storeys,
        "bounding_box_m": {
            "width": width_m,
            "depth": depth_m,
            "height": round(num_storeys * storey_height_m, 2),
        },
        "floorplans_svg": svg_rel,
        "samples": {
            "ifc": ifc_refs,
            "lidar": lidar_refs,
        },
    }


def fabricate_building(rng: random.Random, climate_zone: str) -> Dict:
    building_uuid = str(uuid.uuid4())
    name = random_name(rng)
    num_storeys = rng.randint(1, 12)
    year_built = rng.randint(1950, 2024)
    orientation_azimuth_deg = round(rng.uniform(0, 360), 1)

    geom = gen_geometry(rng, building_uuid, num_storeys)

    walls = gen_wall_assembly(rng)
    roof = gen_roof_assembly(rng)
    floor = gen_floor_assembly(rng)
    openings = gen_windows_doors(rng)

    hvac = gen_hvac(rng)
    dhw = gen_dhw(rng)
    lighting = gen_lighting(rng)
    renewables = gen_renewables(rng)

    air_tightness = gen_air_tightness(rng)

    thermal_zones = rng.randint(1, max(1, num_storeys // 2 + 1))

    # LCA placeholders (static/fabric focus; values are fabricated but plausible magnitudes)
    lca = {
        "embodied_carbon_kgCO2e_m2": round(rng.uniform(150, 700), 1),
        "service_life_years": rng.randint(30, 90),
        "end_of_life_scenario": rng.choice(["recycle_partial", "landfill", "reuse_structure"]),
    }

    building = {
        "schema_version": "1.0.0",
        "topic": "Dynamic Digital Twin for Retrofit Optimization (Static Building DNA)",
        "building_id": building_uuid,
        "name": name,
        "address": random_address(rng),
        "climate_zone": climate_zone,
        "year_built": year_built,
        "orientation_main_azimuth_deg": orientation_azimuth_deg,
        "geometry": geom,
        "fabric": {
            "exterior_wall_assembly": walls,
            "roof_assembly": roof,
            "floor_assembly": floor,
            "openings": openings,
            "air_tightness": air_tightness,
        },
        "systems": {
            "hvac": hvac,
            "dhw": dhw,
            "lighting": lighting,
            "renewables": renewables,
        },
        "operations_assumptions": {
            "occupancy_density_m2_per_person": round(rng.uniform(8.0, 25.0), 1),
            "setpoints_C": {"heating": round(rng.uniform(19.0, 22.0), 1), "cooling": round(rng.uniform(23.0, 26.0), 1)},
            "ventilation_Lps_per_person": round(rng.uniform(5.0, 12.0), 1),
        },
        "lca_summary": lca,
        "provenance": {
            "generated_by": "generate_building_dna.py",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "random_seed": rng.randint(0, 10**9),
        },
    }
    return building


def write_building(building: Dict) -> None:
    bldg_id = building["building_id"]
    out_json = BUILDINGS_DIR / f"{bldg_id}.json"
    save_json(out_json, building)


def write_index(buildings: List[Dict]) -> None:
    rows = []
    for b in buildings:
        geo = b["geometry"]
        rows.append({
            "building_id": b["building_id"],
            "name": b["name"],
            "climate_zone": b["climate_zone"],
            "year_built": b["year_built"],
            "gfa_m2": geo["gross_floor_area_m2"],
            "volume_m3": geo["volume_m3"],
            "n_storeys": geo["num_storeys"],
            "wall_U": b["fabric"]["exterior_wall_assembly"]["overall_u_W_m2K"],
            "roof_U": b["fabric"]["roof_assembly"]["overall_u_W_m2K"],
            "floor_U": b["fabric"]["floor_assembly"]["overall_u_W_m2K"],
            "ach50": b["fabric"]["air_tightness"]["ach50"],
            "has_pv": any(s.get("type") == "solar_pv" for s in b["systems"]["renewables"]),
        })
    save_csv(DATASET_ROOT / "buildings" / "index.csv", rows)


def write_schema() -> None:
    schema = {
        "title": "Building Static & Fabric Data (DNA)",
        "version": "1.0.0",
        "description": "Schema for static building DNA dataset supporting digital-twin retrofit studies.",
        "units": {
            "length": "m",
            "area": "m2",
            "volume": "m3",
            "u_value": "W/m2K",
        },
        "sections": [
            "identity", "geometry", "fabric", "systems", "operations_assumptions", "lca_summary", "provenance"
        ],
    }
    save_json(METADATA_DIR / "schema.json", schema)


def download_samples() -> None:
    # Try IFCs
    for name, urls in IFC_SAMPLE_CANDIDATES.items():
        dest = SAMPLES_IFC_DIR / name
        ok = download_from_any(urls, dest)
        if not ok:
            print(f"Warning: could not download any IFC for {name}")
    # Try LiDAR
    for name, urls in LIDAR_SAMPLE_CANDIDATES.items():
        dest = SAMPLES_LIDAR_DIR / name
        ok = download_from_any(urls, dest)
        if not ok:
            print(f"Warning: could not download any LiDAR sample for {name}")


# ---------------------------- CLI ---------------------------- #

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fabricate static building DNA dataset and fetch sample geometry assets.")
    parser.add_argument("--num-buildings", type=int, default=5, help="Number of synthetic buildings to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--climate-zones", type=str, nargs="*", default=["ASHRAE 4A", "ASHRAE 5A", "ASHRAE 3C"], help="Climate zones to sample")
    args = parser.parse_args()

    ensure_dirs()
    rng = random.Random(args.seed)

    print("Downloading sample geometry assets (IFC, LiDAR)...")
    download_samples()

    print(f"Generating {args.num_buildings} buildings...")
    buildings = []
    for i in range(args.num_buildings):
        cz = rng.choice(args.climate_zones)
        b = fabricate_building(rng, cz)
        write_building(b)
        buildings.append(b)

    write_index(buildings)
    write_schema()

    print("Done.")
    print(f"- Buildings written to: {BUILDINGS_DIR}")
    print(f"- Index CSV: {BUILDINGS_DIR / 'index.csv'}")
    print(f"- Sample IFC: {SAMPLES_IFC_DIR}")
    print(f"- Sample LiDAR: {SAMPLES_LIDAR_DIR}")


if __name__ == "__main__":
    main()
