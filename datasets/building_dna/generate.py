#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import os
import random
import string
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import numpy as np
except Exception:  # lazy import guard, we will install if missing in __main__
    np = None  # type: ignore


DATASET_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = DATASET_ROOT / "output"


def ensure_numpy():
    global np
    if np is None:
        raise RuntimeError("numpy not available. Run via `python -m building_dna.generate` after installing requirements, or let the runner auto-install.")


def rng(seed: int) -> random.Random:
    r = random.Random()
    r.seed(seed)
    return r


@dataclass
class MaterialLayer:
    material: str
    thickness_m: float
    thermal_conductivity_W_mK: float
    density_kg_m3: float
    specific_heat_J_kgK: float


@dataclass
class Assembly:
    assembly: str
    layers: List[MaterialLayer]

    @property
    def r_value_m2K_W(self) -> float:
        # Sum thickness / conductivity; add small surface film resistances
        r_layers = sum(layer.thickness_m / max(layer.thermal_conductivity_W_mK, 1e-6) for layer in self.layers)
        r_si = 0.13  # internal surface film
        r_so = 0.04  # external surface film
        return r_layers + r_si + r_so

    @property
    def u_value_W_m2K(self) -> float:
        r = max(self.r_value_m2K_W, 1e-6)
        return 1.0 / r


USE_TYPES = [
    "residential_single_family",
    "residential_multi_family",
    "office",
    "retail",
    "school",
    "hospital",
]

CLIMATE_ZONES = [
    "ASHRAE_2A",
    "ASHRAE_3A",
    "ASHRAE_3B",
    "ASHRAE_4A",
    "ASHRAE_5A",
    "ASHRAE_5B",
    "ASHRAE_6A",
]


MATERIAL_LIBRARY = {
    "brick": dict(k=0.60, rho=1700, cp=840),
    "concrete": dict(k=1.40, rho=2200, cp=880),
    "gypsum": dict(k=0.17, rho=800, cp=1090),
    "mineral_wool": dict(k=0.04, rho=40, cp=840),
    "polyiso": dict(k=0.028, rho=32, cp=1400),
    "plywood": dict(k=0.12, rho=550, cp=1500),
    "air_gap": dict(k=0.024, rho=1.2, cp=1000),
    "glass": dict(k=1.0, rho=2500, cp=800),
    "aluminum": dict(k=205, rho=2700, cp=900),
}


def random_string(r: random.Random, n: int = 6) -> str:
    return "".join(r.choices(string.ascii_lowercase + string.digits, k=n))


def make_wall_assembly(r: random.Random, target_u: float) -> Assembly:
    # Simple stochastic layer stack that targets a U-value range
    layers: List[MaterialLayer] = []
    # Exterior cladding brick or plywood
    cladding = r.choice(["brick", "plywood", "concrete"])
    cladding_thk = r.uniform(0.02, 0.10)
    m = MATERIAL_LIBRARY[cladding]
    layers.append(MaterialLayer(cladding, cladding_thk, m["k"], m["rho"], m["cp"]))

    # Insulation layer thickness to roughly hit target U
    ins = r.choice(["mineral_wool", "polyiso"])
    m = MATERIAL_LIBRARY[ins]
    # back-calc required R ~ 1/target_u minus films and cladding
    r_target = 1.0 / max(target_u, 0.1)
    r_films_and_clad = 0.17 + 0.04 + cladding_thk / max(MATERIAL_LIBRARY[cladding]["k"], 1e-6)
    r_needed = max(r_target - r_films_and_clad, 0.5)
    thk_needed = r_needed * m["k"]
    thk = max(min(thk_needed * r.uniform(0.7, 1.3), 0.25), 0.03)
    layers.append(MaterialLayer(ins, thk, m["k"], m["rho"], m["cp"]))

    # Interior gypsum
    m = MATERIAL_LIBRARY["gypsum"]
    layers.append(MaterialLayer("gypsum", r.uniform(0.012, 0.02), m["k"], m["rho"], m["cp"]))

    return Assembly(assembly=f"wall_{cladding}_{ins}", layers=layers)


def make_roof_assembly(r: random.Random, target_u: float) -> Assembly:
    deck = r.choice(["plywood", "concrete"])
    m = MATERIAL_LIBRARY[deck]
    layers = [MaterialLayer(deck, r.uniform(0.018, 0.05), m["k"], m["rho"], m["cp"])]
    ins = r.choice(["mineral_wool", "polyiso"])
    m = MATERIAL_LIBRARY[ins]
    r_target = 1.0 / max(target_u, 0.08)
    r_films = 0.17 + 0.04
    r_needed = max(r_target - r_films - layers[0].thickness_m / MATERIAL_LIBRARY[deck]["k"], 1.0)
    thk = max(min(r_needed * m["k"] * r.uniform(0.8, 1.2), 0.40), 0.05)
    layers.append(MaterialLayer(ins, thk, m["k"], m["rho"], m["cp"]))
    return Assembly(assembly=f"roof_{deck}_{ins}", layers=layers)


def make_floor_assembly(r: random.Random, target_u: float) -> Assembly:
    slab = r.choice(["concrete", "plywood"])  # simplified
    m = MATERIAL_LIBRARY[slab]
    layers = [MaterialLayer(slab, r.uniform(0.05, 0.2), m["k"], m["rho"], m["cp"])]
    ins = r.choice(["mineral_wool", "polyiso"]) if slab == "plywood" else "mineral_wool"
    m = MATERIAL_LIBRARY[ins]
    r_target = 1.0 / max(target_u, 0.15)
    r_films = 0.17 + 0.04
    r_needed = max(r_target - r_films - layers[0].thickness_m / MATERIAL_LIBRARY[slab]["k"], 0.5)
    thk = max(min(r_needed * m["k"] * r.uniform(0.7, 1.3), 0.30), 0.02)
    layers.append(MaterialLayer(ins, thk, m["k"], m["rho"], m["cp"]))
    return Assembly(assembly=f"floor_{slab}_{ins}", layers=layers)


def make_windows_doors(r: random.Random, perimeter_m: float, ww_ratio: float) -> List[Dict]:
    openings: List[Dict] = []
    window_area_total = perimeter_m * 3.0 * ww_ratio  # approx facade height 3m
    # Create a mix of windows and doors
    remaining = window_area_total
    while remaining > 0.2:
        if r.random() < 0.1:
            area = r.uniform(1.5, 2.5)
            openings.append({
                "type": "door",
                "u_value_W_m2K": r.uniform(1.2, 2.0),
                "shgc": r.uniform(0.2, 0.5),
                "visible_transmittance": r.uniform(0.4, 0.7),
                "frame_type": r.choice(["aluminum", "uPVC", "wood"]),
                "gas_fill": "air",
                "area_m2": area,
                "age_years": r.randint(0, 30),
                "condition": r.choice(["poor", "fair", "good"])})
            remaining -= area
        else:
            area = r.uniform(0.8, 2.0)
            openings.append({
                "type": "window",
                "u_value_W_m2K": r.uniform(0.9, 2.5),
                "shgc": r.uniform(0.25, 0.6),
                "visible_transmittance": r.uniform(0.5, 0.8),
                "frame_type": r.choice(["aluminum", "uPVC", "wood"]),
                "gas_fill": r.choice(["air", "argon", "krypton"]),
                "area_m2": area,
                "age_years": r.randint(0, 30),
                "condition": r.choice(["poor", "fair", "good"])})
            remaining -= area
    return openings


def generate_floorplan_svg(r: random.Random, width_m: float, depth_m: float, cols: int, rows: int) -> str:
    # Simple grid-based plan with rooms
    room_w = width_m / cols
    room_d = depth_m / rows
    scale = 20  # px per meter
    svg_w = int(math.ceil(width_m * scale + 20))
    svg_h = int(math.ceil(depth_m * scale + 20))
    rects = []
    for i in range(cols):
        for j in range(rows):
            x = 10 + i * room_w * scale
            y = 10 + j * room_d * scale
            rects.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{room_w*scale-2:.1f}" height="{room_d*scale-2:.1f}" fill="none" stroke="#000" stroke-width="1"/>')
    outline = f'<rect x="10" y="10" width="{width_m*scale:.1f}" height="{depth_m*scale:.1f}" fill="none" stroke="#333" stroke-width="2"/>'
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='{svg_w}' height='{svg_h}' viewBox='0 0 {svg_w} {svg_h}'>\n{outline}\n" + "\n".join(rects) + "\n</svg>"
    return svg


def generate_obj_mass_model(width_m: float, depth_m: float, height_m: float) -> str:
    # Very simple extruded block OBJ
    x0, y0, z0 = 0.0, 0.0, 0.0
    x1, y1, z1 = width_m, depth_m, height_m
    vertices = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)
    ]
    faces = [
        (1, 2, 3, 4),  # bottom
        (5, 6, 7, 8),  # top
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 4, 8, 7),
        (4, 1, 5, 8)
    ]
    lines = ["o BuildingMass"]
    for v in vertices:
        lines.append(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}")
    for f in faces:
        lines.append("f " + " ".join(str(i) for i in f))
    return "\n".join(lines) + "\n"


def generate_site_context_geojson(center: Tuple[float, float], width_m: float, depth_m: float) -> Dict:
    lon, lat = center
    # Fake rectangular footprint polygon (WGS84 for simplicity)
    dx = width_m * 1e-5
    dy = depth_m * 1e-5
    coords = [
        [lon, lat], [lon + dx, lat], [lon + dx, lat + dy], [lon, lat + dy], [lon, lat]
    ]
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "building_footprint"},
            "geometry": {"type": "Polygon", "coordinates": [coords]}
        }]
    }


def generate_lidar_point_cloud(r: random.Random, width_m: float, depth_m: float, height_m: float, density_per_m2: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    ensure_numpy()
    area = width_m * depth_m
    n = int(area * density_per_m2)
    xs = r.random()
    # Sample ground and roof points
    x = np.random.rand(n) * width_m
    y = np.random.rand(n) * depth_m
    z_ground = np.random.normal(loc=0.0, scale=0.05, size=n)
    z_roof = height_m + np.random.normal(loc=0.0, scale=0.05, size=n)
    # half points on ground, half on roof
    half = n // 2
    z = np.concatenate([z_ground[:half], z_roof[half:]])
    x = np.concatenate([x[:half], x[half:]])
    y = np.concatenate([y[:half], y[half:]])
    intensity = np.clip(np.random.normal(loc=0.6, scale=0.15, size=n), 0.0, 1.0)
    points = np.stack([x, y, z], axis=1)
    return points, intensity


def write_ply(path: Path, points: np.ndarray, intensity: np.ndarray) -> None:
    ensure_numpy()
    header = """ply\nformat ascii 1.0\n"""
    header += f"element vertex {points.shape[0]}\n"
    header += "property float x\nproperty float y\nproperty float z\nproperty float intensity\nend_header\n"
    with path.open("w") as f:
        f.write(header)
        for (x, y, z), i in zip(points, intensity):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {i:.3f}\n")


def pick_hvac(r: random.Random, use_type: str) -> Dict:
    options = [
        {"system_type": "ASHP_split", "fuel_type": "electric", "efficiency": {"seer": r.uniform(13, 22), "cop": r.uniform(2.5, 4.0)}},
        {"system_type": "GSHP", "fuel_type": "electric", "efficiency": {"cop": r.uniform(3.0, 5.0)}},
        {"system_type": "Furnace_AC", "fuel_type": "natural_gas", "efficiency": {"afue": r.uniform(0.8, 0.98), "seer": r.uniform(13, 18)}}
    ]
    hvac = r.choice(options)
    hvac.update({
        "make": r.choice(["Carrier", "Trane", "Daikin", "Lennox", "Mitsubishi"]),
        "model": f"{random_string(r, 4).upper()}-{r.randint(100,999)}",
        "age_years": r.randint(0, 25),
        "rated_capacity_kW": r.uniform(5, 300),
        "maintenance_history": [f"{y}: {r.choice(['filter change','coil clean','refrigerant top-up','belt replace'])}" for y in range(2015, 2025) if r.random() < 0.3]
    })
    return hvac


def pick_dhw(r: random.Random) -> Dict:
    options = [
        {"system_type": "tank_storage", "fuel_type": r.choice(["electric", "natural_gas"]), "efficiency": {"uef": r.uniform(0.6, 0.95)}, "storage_volume_liters": r.uniform(120, 1000)},
        {"system_type": "heat_pump_water_heater", "fuel_type": "electric", "efficiency": {"cop": r.uniform(2.0, 3.5)}, "storage_volume_liters": r.uniform(120, 300)},
        {"system_type": "tankless", "fuel_type": r.choice(["natural_gas", "electric"]), "efficiency": {"uef": r.uniform(0.8, 0.98)}}
    ]
    dhw = r.choice(options)
    dhw.update({
        "make": r.choice(["Rheem", "AO Smith", "Navien", "Bosch", "Stiebel Eltron"]),
        "model": f"{random_string(r, 3).upper()}-{r.randint(10,99)}",
        "age_years": r.randint(0, 20),
    })
    return dhw


def pick_lighting(r: random.Random, floor_area_m2: float) -> Dict:
    fixtures = []
    # Aim for ~8-12 W/m2 if LED; higher if CFL/incandescent
    target_w_per_m2 = r.uniform(7, 15)
    total_w = target_w_per_m2 * floor_area_m2
    remaining = total_w
    while remaining > 10:
        lamp_type = r.choices(["LED", "CFL", "Incandescent"], weights=[0.7, 0.2, 0.1])[0]
        watt = r.choice([8, 12, 18, 24, 36, 48]) if lamp_type != "Incandescent" else r.choice([40, 60, 75])
        count = r.randint(1, 12)
        fixtures.append({
            "location": r.choice(["open_office", "private_office", "corridor", "classroom", "living_room", "kitchen"]),
            "lamp_type": lamp_type,
            "wattage_W": watt,
            "control": r.choice(["switch", "dimmer", "occupancy_sensor"]),
            "count": count
        })
        remaining -= watt * count
    return {"fixtures": fixtures}


def pick_renewables(r: random.Random, roof_area_m2: float) -> Dict:
    renewables = {}
    if r.random() < 0.6:
        capacity = max(0.0, r.gauss(mu=roof_area_m2 * 0.12 / 8.0, sigma=3.0))  # ~12% roof coverage, 1 kWp ~ 8 m2
        renewables["solar_pv"] = {
            "capacity_kWp": round(capacity, 2),
            "inverter_efficiency": round(r.uniform(0.94, 0.99), 3),
            "age_years": r.randint(0, 15),
            "azimuth_deg": r.uniform(-180, 180),
            "tilt_deg": r.uniform(0, 45)
        }
    if r.random() < 0.2:
        renewables["solar_thermal"] = {
            "area_m2": round(r.uniform(2, max(2.0, roof_area_m2 * 0.1)), 2),
            "efficiency": round(r.uniform(0.45, 0.7), 2),
            "age_years": r.randint(0, 20),
            "azimuth_deg": r.uniform(-180, 180),
            "tilt_deg": r.uniform(0, 60)
        }
    return renewables or {"solar_pv": {"capacity_kWp": 0.0, "inverter_efficiency": 0.0, "age_years": 0, "azimuth_deg": 0.0, "tilt_deg": 0.0}}


def generate_building(r: random.Random, idx: int) -> None:
    building_id = f"bldg_{idx:03d}_{random_string(r, 4)}"
    out_dir = OUTPUT_ROOT / building_id
    (out_dir / "geometry").mkdir(parents=True, exist_ok=True)
    (out_dir / "lidar").mkdir(parents=True, exist_ok=True)
    (out_dir / "fabric").mkdir(parents=True, exist_ok=True)
    (out_dir / "systems").mkdir(parents=True, exist_ok=True)

    # Metadata
    num_floors = r.randint(1, 12)
    width_m = r.uniform(12, 60)
    depth_m = r.uniform(10, 50)
    floor_height_m = r.uniform(3.0, 3.6)
    height_m = num_floors * floor_height_m
    floor_area_m2 = width_m * depth_m * num_floors
    metadata = {
        "id": building_id,
        "name": f"Synthetic Building {idx}",
        "address": f"{r.randint(10, 9999)} {r.choice(['Main St','Broadway','1st Ave','Oak St','Maple Rd'])}",
        "year_built": r.randint(1920, 2022),
        "climate_zone": r.choice(CLIMATE_ZONES),
        "num_floors": num_floors,
        "gross_floor_area_m2": round(floor_area_m2, 1),
        "use_type": r.choice(USE_TYPES),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Geometry artifacts
    for level in range(1, num_floors + 1):
        cols = r.randint(2, 5)
        rows = r.randint(2, 5)
        svg = generate_floorplan_svg(r, width_m, depth_m, cols, rows)
        (out_dir / "geometry" / f"floorplan_level_{level}.svg").write_text(svg)
    obj = generate_obj_mass_model(width_m, depth_m, height_m)
    (out_dir / "geometry" / "mass_model.obj").write_text(obj)
    center = (-73.9857 + r.uniform(-0.01, 0.01), 40.7484 + r.uniform(-0.01, 0.01))
    site_geojson = generate_site_context_geojson(center, width_m, depth_m)
    (out_dir / "geometry" / "site_context.geojson").write_text(json.dumps(site_geojson))

    # Fabric
    target_u_wall = r.uniform(0.2, 0.6)
    target_u_roof = r.uniform(0.1, 0.3)
    target_u_floor = r.uniform(0.2, 0.4)
    wall = make_wall_assembly(r, target_u_wall)
    roof = make_roof_assembly(r, target_u_roof)
    floor = make_floor_assembly(r, target_u_floor)

    perimeter_m = 2 * (width_m + depth_m)
    ww_ratio = r.uniform(0.15, 0.35)
    openings = make_windows_doors(r, perimeter_m, ww_ratio)

    airtightness = {
        "blower_door_ach50": round(max(0.6, r.gauss(5.0, 2.0)), 2),
        "test_year": r.randint(2000, 2025)
    }

    def assembly_to_json(a: Assembly) -> Dict:
        return {
            "assembly": a.assembly,
            "layers": [asdict(l) for l in a.layers],
            "u_value_W_m2K": round(a.u_value_W_m2K, 3),
            "r_value_m2K_W": round(a.r_value_m2K_W, 3)
        }

    fabric = {
        "walls": [assembly_to_json(wall)],
        "roofs": [assembly_to_json(roof)],
        "floors": [assembly_to_json(floor)],
        "windows_doors": openings,
        "airtightness": airtightness
    }
    (out_dir / "fabric" / "walls.json").write_text(json.dumps(fabric["walls"], indent=2))
    (out_dir / "fabric" / "roofs.json").write_text(json.dumps(fabric["roofs"], indent=2))
    (out_dir / "fabric" / "floors.json").write_text(json.dumps(fabric["floors"], indent=2))
    (out_dir / "fabric" / "windows_doors.json").write_text(json.dumps(fabric["windows_doors"], indent=2))
    (out_dir / "fabric" / "airtightness.json").write_text(json.dumps(fabric["airtightness"], indent=2))

    # Systems
    hvac = pick_hvac(r, metadata["use_type"])
    dhw = pick_dhw(r)
    lighting = pick_lighting(r, floor_area_m2)
    renew = pick_renewables(r, roof_area_m2=width_m * depth_m)

    (out_dir / "systems" / "hvac.json").write_text(json.dumps(hvac, indent=2))
    (out_dir / "systems" / "dhw.json").write_text(json.dumps(dhw, indent=2))
    (out_dir / "systems" / "lighting.json").write_text(json.dumps(lighting, indent=2))
    (out_dir / "systems" / "renewables.json").write_text(json.dumps(renew, indent=2))

    # LiDAR
    try:
        points, intensity = generate_lidar_point_cloud(r, width_m, depth_m, height_m)
        pcsv = np.concatenate([points, intensity.reshape(-1, 1)], axis=1)
        np.savetxt(out_dir / "lidar" / "point_cloud.csv", pcsv, delimiter=",", header="x,y,z,intensity", comments="")
        write_ply(out_dir / "lidar" / "point_cloud.ply", points, intensity)
    except RuntimeError:
        # If numpy missing, write placeholder small CSV
        (out_dir / "lidar" / "point_cloud.csv").write_text("x,y,z,intensity\n0,0,0,0\n")



def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Building DNA dataset")
    parser.add_argument("--count", type=int, default=3, help="Number of buildings to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Best-effort install numpy if missing and we have pip
    global np
    if np is None:
        try:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "--quiet"])  # noqa: E402
            import numpy as np2  # type: ignore
            np = np2
        except Exception:
            pass

    r = rng(args.seed)
    for i in range(1, args.count + 1):
        generate_building(r, i)

    print(f"Generated {args.count} buildings at {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
