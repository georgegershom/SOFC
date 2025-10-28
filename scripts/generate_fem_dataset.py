#!/usr/bin/env python3
"""
Synthetic multi-physics FEM-like dataset generator.

Generates a structured dataset with:
- Inputs: mesh, boundary conditions, material models, transient thermal profiles
- Outputs: stress/strain fields, damage evolution, temperature/voltage fields, delamination predictions

No third-party dependencies are required. Outputs are CSV/JSON files for easy inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple


# ------------------------------ Data Models ------------------------------


@dataclass
class MeshConfig:
    grid_nx: int
    grid_ny: int
    element_type: str  # "quad4" or "tri3"
    nominal_element_size: float
    interface_refinement_factor: float


@dataclass
class BoundaryConditions:
    temperature_bc: Dict[str, float]
    displacement_bc: Dict[str, float]
    voltage_bc: Dict[str, float]


@dataclass
class MaterialModels:
    elastic_modulus_gpa: float
    poisson_ratio: float
    yield_strength_mpa: float
    creep_coeff: float
    thermal_expansion_per_c: float
    electrochemical_coeff: float


@dataclass
class ThermalProfile:
    # time_seconds -> dict with temperature_c, voltage_v
    times: List[float]
    temperatures_c: List[float]
    voltages_v: List[float]


# ------------------------------ Utilities ------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, header: List[str], rows: List[List[float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(format(v, ".6f") if isinstance(v, (int, float)) else str(v) for v in row) + "\n")


def linspace(start: float, stop: float, num: int) -> List[float]:
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def smooth_step(x: float) -> float:
    # 0->1 smoothstep
    return x * x * (3.0 - 2.0 * x)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ------------------------------ Generators ------------------------------


def generate_mesh(cfg: MeshConfig) -> Tuple[List[Tuple[int, float, float]], List[List[int]]]:
    """
    Generate a structured 2D grid mesh on [0,1]x[0,1].

    - nodes: list of (node_id, x, y)
    - elements: connectivity list of node ids

    For simplicity, we keep uniform spacing and simulate interface effects in the fields.
    """
    nx = cfg.grid_nx
    ny = cfg.grid_ny
    xs = linspace(0.0, 1.0, nx + 1)
    ys = linspace(0.0, 1.0, ny + 1)

    nodes: List[Tuple[int, float, float]] = []
    node_id = 1
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            nodes.append((node_id, x, y))
            node_id += 1

    def node_index(i: int, j: int) -> int:
        return j * (nx + 1) + i + 1  # 1-based ids

    elements: List[List[int]] = []
    elem_id = 1
    if cfg.element_type == "quad4":
        for j in range(ny):
            for i in range(nx):
                n1 = node_index(i, j)
                n2 = node_index(i + 1, j)
                n3 = node_index(i + 1, j + 1)
                n4 = node_index(i, j + 1)
                elements.append([elem_id, n1, n2, n3, n4])
                elem_id += 1
    else:  # tri3; split each quad into 2 triangles
        for j in range(ny):
            for i in range(nx):
                n1 = node_index(i, j)
                n2 = node_index(i + 1, j)
                n3 = node_index(i + 1, j + 1)
                n4 = node_index(i, j + 1)
                elements.append([elem_id, n1, n2, n3])
                elem_id += 1
                elements.append([elem_id, n1, n3, n4])
                elem_id += 1

    return nodes, elements


def generate_bcs(rng: random.Random) -> BoundaryConditions:
    # Temperature BC: left/right edges at different fixed temperatures
    t_left = rng.uniform(20.0, 80.0)
    t_right = t_left + rng.uniform(10.0, 40.0)
    # Displacement BC: clamp bottom in y and left in x
    u_x_left = 0.0
    u_y_bottom = 0.0
    # Voltage BC: left electrode 0-0.5 V, right 0.8-1.5 V
    v_left = rng.uniform(0.0, 0.5)
    v_right = rng.uniform(0.8, 1.5)
    return BoundaryConditions(
        temperature_bc={"left_c": t_left, "right_c": t_right},
        displacement_bc={"ux_left": u_x_left, "uy_bottom": u_y_bottom},
        voltage_bc={"left_v": v_left, "right_v": v_right},
    )


def generate_materials(rng: random.Random) -> MaterialModels:
    # Choose plausible ranges
    elastic_modulus_gpa = rng.uniform(50.0, 210.0)
    poisson_ratio = rng.uniform(0.25, 0.35)
    yield_strength_mpa = rng.uniform(150.0, 600.0)
    creep_coeff = rng.uniform(1e-6, 1e-4)
    thermal_expansion_per_c = rng.uniform(8e-6, 22e-6)
    electrochemical_coeff = rng.uniform(0.1, 1.0)
    return MaterialModels(
        elastic_modulus_gpa=elastic_modulus_gpa,
        poisson_ratio=poisson_ratio,
        yield_strength_mpa=yield_strength_mpa,
        creep_coeff=creep_coeff,
        thermal_expansion_per_c=thermal_expansion_per_c,
        electrochemical_coeff=electrochemical_coeff,
    )


def generate_thermal_profile(rng: random.Random, duration_min: float) -> ThermalProfile:
    # Heating/cooling rate: 1–10 °C/min; create a cycle up and down
    base_temp = rng.uniform(20.0, 40.0)
    peak_temp = base_temp + rng.uniform(20.0, 80.0)
    rate_c_per_min = rng.uniform(1.0, 10.0)

    total_time_s = int(duration_min * 60)
    step_s = 10
    times = list(range(0, total_time_s + 1, step_s))
    temperatures: List[float] = []
    voltages: List[float] = []
    half = len(times) // 2
    for idx, t in enumerate(times):
        if idx <= half:
            frac = idx / max(1, half)
            temp = base_temp + frac * (peak_temp - base_temp)
        else:
            frac = (idx - half) / max(1, len(times) - half - 1)
            temp = peak_temp - frac * (peak_temp - base_temp)
        temperatures.append(temp)
        # Voltage ramps with some noise
        v = 0.1 + 1.2 * (idx / max(1, len(times) - 1)) + rng.uniform(-0.03, 0.03)
        voltages.append(max(0.0, v))

    # Adjust duration to roughly match heating/cooling rate
    # Not enforcing exact rate; recorded in metadata for context
    return ThermalProfile(times=[float(t) for t in times], temperatures_c=temperatures, voltages_v=voltages)


def gaussian(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def compute_fields(
    nodes: List[Tuple[int, float, float]],
    elements: List[List[int]],
    cfg: MeshConfig,
    bcs: BoundaryConditions,
    mats: MaterialModels,
    profile: ThermalProfile,
    rng: random.Random,
) -> Dict[str, Dict[str, List[List[float]]]]:
    """
    Compute synthetic fields for 5 snapshots across the transient.
    Returns a dict keyed by snapshot label containing CSV rows for various fields.
    """
    # Snapshot indices at 0%, 25%, 50%, 75%, 100%
    idxs = [0, int(0.25 * (len(profile.times) - 1)), int(0.5 * (len(profile.times) - 1)), int(0.75 * (len(profile.times) - 1)), len(profile.times) - 1]
    labels = ["t0", "t25", "t50", "t75", "t100"]

    # Precompute node maps
    node_id_to_xy: Dict[int, Tuple[float, float]] = {nid: (x, y) for nid, x, y in nodes}

    # Helper functions to generate smooth spatial fields
    def base_spatial_field(x: float, y: float) -> float:
        # Combination of gradients and a central interface at y=0.5
        interface = gaussian(y, 0.5, 0.08) * 1.5
        waves = 0.5 * math.sin(2 * math.pi * x) * math.cos(2 * math.pi * y)
        grad = 0.8 * x + 0.6 * y
        return interface + waves + grad

    # Prepare per-element and per-node storage
    per_snapshot: Dict[str, Dict[str, List[List[float]]]] = {}

    # Young's modulus in MPa
    E = mats.elastic_modulus_gpa * 1000.0
    nu = mats.poisson_ratio
    yield_mises = mats.yield_strength_mpa

    # Interface for interfacial shear: elements whose centroid y near 0.5
    def element_centroid(elem: List[int]) -> Tuple[float, float]:
        xy = [node_id_to_xy[n] for n in elem[1:]]
        cx = sum(p[0] for p in xy) / len(xy)
        cy = sum(p[1] for p in xy) / len(xy)
        return cx, cy

    interface_band = 0.12

    for label, idx in zip(labels, idxs):
        temp = profile.temperatures_c[idx]
        volt = profile.voltages_v[idx]
        time_s = profile.times[idx]

        # Temperature field at nodes: blend BCs horizontally and add smooth variations
        temp_rows: List[List[float]] = []
        volt_rows: List[List[float]] = []
        for nid, x, y in nodes:
            t_left = bcs.temperature_bc["left_c"]
            t_right = bcs.temperature_bc["right_c"]
            t_lin = t_left * (1 - x) + t_right * x
            t_field = 0.7 * t_lin + 0.3 * temp * (0.5 + 0.5 * math.sin(2 * math.pi * x) * math.sin(2 * math.pi * y))
            temp_rows.append([nid, t_field])

            v_left = bcs.voltage_bc["left_v"]
            v_right = bcs.voltage_bc["right_v"]
            v_lin = v_left * (1 - x) + v_right * x
            v_field = 0.8 * v_lin + 0.2 * volt * (0.5 + 0.5 * math.cos(2 * math.pi * x) * math.sin(2 * math.pi * y))
            volt_rows.append([nid, v_field])

        # Stress/strain per element: synthesize from base spatial field and temperature
        stress_vm_rows: List[List[float]] = []
        stress_p1_rows: List[List[float]] = []
        shear_interface_rows: List[List[float]] = []
        strain_el_rows: List[List[float]] = []
        strain_pl_rows: List[List[float]] = []
        strain_cr_rows: List[List[float]] = []
        strain_th_rows: List[List[float]] = []
        damage_rows: List[List[float]] = []

        for elem in elements:
            eid = elem[0]
            cx, cy = element_centroid(elem)
            base = base_spatial_field(cx, cy)
            thermal_strain = mats.thermal_expansion_per_c * (temp - 20.0)
            elastic_strain = 0.0025 * base + 0.5 * thermal_strain
            creep_strain = mats.creep_coeff * time_s * (0.3 + 0.7 * base)
            plastic_strain = max(0.0, (elastic_strain + creep_strain) - (yield_mises / (E + 1e-9)))

            # Aggregate strain components
            strain_th_rows.append([eid, thermal_strain])
            strain_el_rows.append([eid, elastic_strain])
            strain_cr_rows.append([eid, creep_strain])
            strain_pl_rows.append([eid, plastic_strain])

            # Stress via Hookean mapping (very simplified)
            axial_stress = (elastic_strain - plastic_strain) * E / (1 - nu**2)
            principal1 = axial_stress * (0.6 + 0.4 * math.sin(2 * math.pi * cx))
            principal2 = axial_stress * (0.5 + 0.5 * math.cos(2 * math.pi * cy))
            von_mises = math.sqrt(max(0.0, principal1**2 - principal1 * principal2 + principal2**2))
            stress_vm_rows.append([eid, von_mises])
            stress_p1_rows.append([eid, principal1])

            # Interfacial shear peaks near y=0.5
            shear_val = 0.8 * von_mises * gaussian(cy, 0.5, interface_band)
            shear_interface_rows.append([eid, shear_val])

            # Damage variable D in [0, 1): grows with time, temperature, and shear
            alpha = 1e-5 * (1.0 + 0.5 * (temp / 100.0))
            damage = 1.0 - math.exp(-alpha * time_s * (0.5 + 1.5 * shear_val / (yield_mises + 1e-6)))
            damage = clamp(damage, 0.0, 0.999)
            damage_rows.append([eid, damage])

        # Delamination/crack flags for elements near interface
        delam_rows: List[List[float]] = []
        crack_rows: List[List[float]] = []
        # Thresholds derived from yield
        shear_thresh = 0.35 * yield_mises
        damage_thresh = 0.6
        for s, d in zip(shear_interface_rows, damage_rows):
            eid = s[0]
            shear_val = s[1]
            damage_val = d[1]
            delam = 1.0 if (shear_val > shear_thresh and damage_val > damage_thresh) else 0.0
            crack = 1.0 if (damage_val > 0.8 and shear_val > 0.25 * yield_mises) else 0.0
            delam_rows.append([eid, delam])
            crack_rows.append([eid, crack])

        per_snapshot[label] = {
            "temperature_node": temp_rows,
            "voltage_node": volt_rows,
            "stress_vm_elem": stress_vm_rows,
            "stress_p1_elem": stress_p1_rows,
            "shear_interface_elem": shear_interface_rows,
            "strain_el_elem": strain_el_rows,
            "strain_pl_elem": strain_pl_rows,
            "strain_cr_elem": strain_cr_rows,
            "strain_th_elem": strain_th_rows,
            "damage_elem": damage_rows,
            "delamination_elem": delam_rows,
            "crack_init_elem": crack_rows,
        }

    return per_snapshot


# ------------------------------ IO and Orchestration ------------------------------


def save_mesh(out_dir: Path, nodes: List[Tuple[int, float, float]], elements: List[List[int]], element_type: str) -> None:
    write_csv(out_dir / "nodes.csv", ["node_id", "x", "y"], [[nid, x, y] for nid, x, y in nodes])
    if element_type == "quad4":
        write_csv(out_dir / "elements.csv", ["elem_id", "n1", "n2", "n3", "n4"], elements)
    else:
        write_csv(out_dir / "elements.csv", ["elem_id", "n1", "n2", "n3"], elements)


def save_inputs(out_dir: Path, cfg: MeshConfig, bcs: BoundaryConditions, mats: MaterialModels, profile: ThermalProfile) -> None:
    write_json(out_dir / "mesh.json", asdict(cfg))
    write_json(out_dir / "boundary_conditions.json", asdict(bcs))
    write_json(out_dir / "material_models.json", asdict(mats))
    write_csv(out_dir / "thermal_profile.csv", ["time_s", "temperature_c", "voltage_v"], [[t, T, V] for t, T, V in zip(profile.times, profile.temperatures_c, profile.voltages_v)])


def save_outputs(out_dir: Path, per_snapshot: Dict[str, Dict[str, List[List[float]]]]) -> None:
    for label, fields in per_snapshot.items():
        snap_dir = out_dir / f"snapshot_{label}"
        ensure_dir(snap_dir)
        for name, rows in fields.items():
            write_csv(snap_dir / f"{name}.csv", ["id", "value"], rows)


def write_metadata(out_dir: Path, seed: int, notes: str) -> None:
    write_json(out_dir / "metadata.json", {"random_seed": seed, "notes": notes})


def write_summary(out_dir: Path, per_snapshot: Dict[str, Dict[str, List[List[float]]]]) -> None:
    def stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"min": 0.0, "max": 0.0, "mean": 0.0}
        vmin = min(values)
        vmax = max(values)
        mean = sum(values) / len(values)
        return {"min": vmin, "max": vmax, "mean": mean}

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, fields in per_snapshot.items():
        label_stats: Dict[str, Dict[str, float]] = {}
        for name, rows in fields.items():
            values = [float(r[1]) for r in rows]
            label_stats[name] = stats(values)
        summary[label] = label_stats

    write_json(out_dir / "summary.json", summary)


def generate_one_sim(
    sim_out_dir: Path,
    mesh_cfg: MeshConfig,
    duration_min: float,
    seed: int,
) -> None:
    rng = random.Random(seed)

    nodes, elements = generate_mesh(mesh_cfg)
    bcs = generate_bcs(rng)
    mats = generate_materials(rng)
    profile = generate_thermal_profile(rng, duration_min=duration_min)
    fields = compute_fields(nodes, elements, mesh_cfg, bcs, mats, profile, rng)

    ensure_dir(sim_out_dir)
    save_mesh(sim_out_dir, nodes, elements, mesh_cfg.element_type)
    save_inputs(sim_out_dir, mesh_cfg, bcs, mats, profile)
    save_outputs(sim_out_dir, fields)
    write_metadata(sim_out_dir, seed=seed, notes="Synthetic FEM-like dataset. Fields are plausible but not from a solver.")
    write_summary(sim_out_dir, fields)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic FEM multi-physics dataset")
    parser.add_argument("--output-dir", type=str, default="/workspace/data/simulations", help="Directory to write simulation datasets")
    parser.add_argument("--num-sims", type=int, default=3, help="Number of simulations to generate")
    parser.add_argument("--grid-nx", type=int, default=30, help="Number of elements along x (for quad4) or divisions for tri3")
    parser.add_argument("--grid-ny", type=int, default=15, help="Number of elements along y (for quad4) or divisions for tri3")
    parser.add_argument("--element-type", type=str, choices=["quad4", "tri3"], default="quad4", help="Element type")
    parser.add_argument("--element-size", type=float, default=0.033, help="Nominal element size (units arbitrary)")
    parser.add_argument("--interface-refinement", type=float, default=1.0, help="Interface refinement factor (metadata only)")
    parser.add_argument("--duration-min", type=float, default=20.0, help="Transient duration in minutes")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    out_root = Path(args.output_dir)
    ensure_dir(out_root)

    for i in range(args.num_sims):
        sim_name = f"sim_{i:03d}"
        sim_dir = out_root / sim_name
        seed = args.seed + i
        mesh_cfg = MeshConfig(
            grid_nx=args.grid_nx,
            grid_ny=args.grid_ny,
            element_type=args.element_type,
            nominal_element_size=args.element_size,
            interface_refinement_factor=args.interface_refinement,
        )
        generate_one_sim(sim_dir, mesh_cfg, duration_min=args.duration_min, seed=seed)

    print(f"Generated {args.num_sims} simulations under {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

