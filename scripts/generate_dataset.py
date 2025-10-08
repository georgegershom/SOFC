#!/usr/bin/env python3
import os
import csv
import json
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
from datetime import datetime, timedelta


# -----------------------------
# Utilities
# -----------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: str, header: List[str], rows: List[List]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def seeded_random(seed: int) -> random.Random:
    rng = random.Random()
    rng.seed(seed)
    return rng


# -----------------------------
# Macro-scale data
# -----------------------------


def generate_material_properties(seed: int) -> Dict:
    rng = seeded_random(seed)
    # Representative ceramic electrolyte properties (fabricated but plausible)
    E_GPa = round(rng.uniform(185.0, 215.0), 2)
    poisson_nu = round(rng.uniform(0.22, 0.31), 3)
    cte_per_K = round(rng.uniform(9.0e-6, 12.5e-6), 8)
    density_kg_m3 = round(rng.uniform(4500.0, 5600.0), 1)
    thermal_conductivity_W_mK = round(rng.uniform(1.5, 3.5), 2)
    specific_heat_J_kgK = round(rng.uniform(350.0, 600.0), 1)
    return {
        "material": "Fabricated Electrolyte Ceramic",
        "youngs_modulus_GPa": E_GPa,
        "poisson_ratio": poisson_nu,
        "coefficient_thermal_expansion_per_K": cte_per_K,
        "density_kg_per_m3": density_kg_m3,
        "thermal_conductivity_W_per_mK": thermal_conductivity_W_mK,
        "specific_heat_J_per_kgK": specific_heat_J_kgK,
    }


def generate_cell_dimensions(seed: int) -> Dict:
    rng = seeded_random(seed + 1)
    length_mm = round(rng.uniform(8.0, 12.0), 3)
    width_mm = round(rng.uniform(8.0, 12.0), 3)
    thickness_mm = round(rng.uniform(0.3, 0.8), 3)
    return {
        "length_mm": length_mm,
        "width_mm": width_mm,
        "thickness_mm": thickness_mm,
        "coordinate_frame": {
            "origin": "lower-south-west corner",
            "axes": {
                "x": "length",
                "y": "width",
                "z": "thickness"
            }
        }
    }


def generate_sintering_profile(seed: int, start_temp_C: float = 25.0) -> List[Tuple[float, float]]:
    rng = seeded_random(seed + 2)
    # Build a ramp-soak-cool profile
    peak_temp_C = rng.uniform(1150.0, 1300.0)
    ramp_rate_C_per_min = rng.uniform(3.0, 8.0)
    cool_rate_C_per_min = rng.uniform(2.0, 6.0)
    soak_minutes = rng.randint(60, 180)

    # Ramp
    delta_T_ramp = peak_temp_C - start_temp_C
    ramp_minutes = int(math.ceil(delta_T_ramp / ramp_rate_C_per_min))
    # Cool
    delta_T_cool = peak_temp_C - start_temp_C
    cool_minutes = int(math.ceil(delta_T_cool / cool_rate_C_per_min))

    rows: List[Tuple[float, float]] = []

    def append_segment(minutes: int, start_T: float, rate: float) -> float:
        T = start_T
        for m in range(minutes):
            rows.append((m_total + len(rows), T))
            T += rate
        return T

    m_total = 0
    # Ramp up
    T = start_temp_C
    for m in range(ramp_minutes):
        rows.append((m_total + m, T))
        T += ramp_rate_C_per_min
    m_total += ramp_minutes
    # Soak
    for m in range(soak_minutes):
        rows.append((m_total + m, peak_temp_C))
    m_total += soak_minutes
    # Cool down
    T = peak_temp_C
    for m in range(cool_minutes):
        rows.append((m_total + m, T))
        T -= cool_rate_C_per_min
    return rows


def save_macro_data(base_dir: str, seed: int) -> None:
    processed = os.path.join(base_dir, "processed")

    write_json(
        os.path.join(processed, "material_properties.json"),
        generate_material_properties(seed),
    )
    write_json(
        os.path.join(processed, "cell_dimensions.json"),
        generate_cell_dimensions(seed),
    )
    sinter = generate_sintering_profile(seed)
    write_csv(
        os.path.join(processed, "sintering_profile.csv"),
        ["time_min", "temperature_C"],
        [[t, round(T, 2)] for (t, T) in sinter],
    )


# -----------------------------
# Experimental residual stress
# -----------------------------


def generate_macro_residual_stress_surface(seed: int, dims: Dict, n_x: int = 25, n_y: int = 25) -> List[List]:
    rng = seeded_random(seed)
    length = dims["length_mm"]
    width = dims["width_mm"]
    # Compressively stressed center, relaxing to edges; add noise
    rows: List[List] = []
    for i in range(n_x):
        for j in range(n_y):
            x = (i + 0.5) * length / n_x
            y = (j + 0.5) * width / n_y
            dx = (x - 0.5 * length) / (0.5 * length)
            dy = (y - 0.5 * width) / (0.5 * width)
            r2 = dx * dx + dy * dy
            sigma0 = -180.0 * math.exp(-1.8 * r2)  # MPa compressive at center
            anisotropy = 1.0 + 0.15 * math.cos(2.0 * math.atan2(dy, dx) if r2 > 1e-8 else 0.0)
            sigma_xx = sigma0 * anisotropy + rng.gauss(0.0, 8.0)
            sigma_yy = sigma0 * (2.0 - anisotropy) + rng.gauss(0.0, 8.0)
            sigma_xy = rng.gauss(0.0, 12.0) * (1.0 - min(1.0, 1.2 * math.sqrt(r2)))
            uncertainty = abs(rng.gauss(5.0, 1.2))
            rows.append([
                round(x, 3), round(y, 3), 0.0, "XRD",
                round(sigma_xx, 2), round(sigma_yy, 2), round(sigma_xy, 2), round(uncertainty, 2)
            ])
    return rows


def generate_meso_residual_stress_bulk(seed: int, dims: Dict, n_x: int = 10, n_y: int = 10, n_z: int = 6) -> List[List]:
    rng = seeded_random(seed + 11)
    length = dims["length_mm"]
    width = dims["width_mm"]
    thickness = dims["thickness_mm"]
    rows: List[List] = []
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                x = (i + 0.5) * length / n_x
                y = (j + 0.5) * width / n_y
                z = (k + 0.5) * thickness / n_z
                z_norm = z / thickness
                # Depth relaxation of compressive stress; slight bending across x
                base = -150.0 * (1.0 - 0.7 * z_norm) + 12.0 * math.sin(2.0 * math.pi * x / max(1e-6, length))
                sigma_xx = base + rng.gauss(0.0, 10.0)
                sigma_yy = 0.85 * base + rng.gauss(0.0, 10.0)
                sigma_zz = 0.4 * base + rng.gauss(0.0, 6.0)
                sigma_xy = rng.gauss(0.0, 8.0) * (1.0 - z_norm)
                sigma_yz = rng.gauss(0.0, 6.0) * (1.0 - 0.5 * z_norm)
                sigma_xz = rng.gauss(0.0, 6.0) * (1.0 - 0.5 * z_norm)
                rows.append([
                    round(x, 3), round(y, 3), round(z, 3), "SynchrotronXRD",
                    round(sigma_xx, 2), round(sigma_yy, 2), round(sigma_zz, 2),
                    round(sigma_xy, 2), round(sigma_yz, 2), round(sigma_xz, 2)
                ])
    return rows


def save_experimental_residual_stress(base_dir: str, seed: int) -> None:
    processed = os.path.join(base_dir, "processed")
    dims = json.load(open(os.path.join(processed, "cell_dimensions.json"), "r", encoding="utf-8"))

    macro_rows = generate_macro_residual_stress_surface(seed, dims)
    write_csv(
        os.path.join(processed, "residual_stress_surface_XRD.csv"),
        ["x_mm", "y_mm", "z_mm", "method", "sigma_xx_MPa", "sigma_yy_MPa", "sigma_xy_MPa", "uncertainty_MPa"],
        macro_rows,
    )

    meso_rows = generate_meso_residual_stress_bulk(seed, dims)
    write_csv(
        os.path.join(processed, "residual_stress_bulk_synchrotron.csv"),
        [
            "x_mm", "y_mm", "z_mm", "method",
            "sigma_xx_MPa", "sigma_yy_MPa", "sigma_zz_MPa",
            "sigma_xy_MPa", "sigma_yz_MPa", "sigma_xz_MPa",
        ],
        meso_rows,
    )


# -----------------------------
# Crack initiation and propagation
# -----------------------------


def generate_microcrack_locations(seed: int, n_cracks: int = 80) -> List[List]:
    rng = seeded_random(seed)
    rows: List[List] = []
    for idx in range(n_cracks):
        # Cross-section in micrometers
        x = rng.uniform(0.0, 500.0)
        y = rng.uniform(0.0, 500.0)
        feature = rng.choices(["grain_boundary", "pore_edge", "triple_junction", "intra_granular"], weights=[0.45, 0.35, 0.15, 0.05])[0]
        orientation_deg = rng.uniform(0.0, 180.0)
        length_um = max(0.2, rng.lognormvariate(math.log(3.0), 0.7))
        rows.append([idx + 1, round(x, 2), round(y, 2), feature, round(orientation_deg, 1), round(length_um, 2)])
    return rows


def generate_cracking_tests(seed: int, n_tests: int = 20) -> Tuple[List[List], Dict[str, float]]:
    rng = seeded_random(seed + 7)
    # Define deterministic material-like thresholds
    critical_temp_C = rng.uniform(780.0, 920.0)
    critical_load_MPa = rng.uniform(12.0, 28.0)

    rows: List[List] = []
    for t in range(n_tests):
        max_temp = rng.uniform(700.0, 1000.0)
        load = rng.uniform(5.0, 35.0)
        rate = rng.uniform(2.0, 10.0)
        cracked = (max_temp >= critical_temp_C) or (load >= critical_load_MPa)
        crack_temp = max(critical_temp_C, rng.uniform(critical_temp_C - 50.0, critical_temp_C + 30.0)) if cracked else ""
        rows.append([
            t + 1, round(max_temp, 1), round(load, 2), round(rate, 2), "yes" if cracked else "no", crack_temp if crack_temp == "" else round(crack_temp, 1)
        ])
    thresholds = {
        "estimated_critical_temperature_C": round(critical_temp_C, 1),
        "estimated_critical_load_MPa": round(critical_load_MPa, 2),
    }
    return rows, thresholds


def save_cracking_data(base_dir: str, seed: int) -> None:
    processed = os.path.join(base_dir, "processed")
    microcracks = generate_microcrack_locations(seed)
    write_csv(
        os.path.join(processed, "microcrack_locations_SEM.csv"),
        ["crack_id", "x_um", "y_um", "feature", "orientation_deg", "length_um"],
        microcracks,
    )
    tests, thresholds = generate_cracking_tests(seed)
    write_csv(
        os.path.join(processed, "fracture_tests.csv"),
        ["test_id", "max_temperature_C", "applied_load_MPa", "heating_rate_C_per_min", "cracked", "crack_temperature_C"],
        tests,
    )
    write_json(
        os.path.join(processed, "critical_thresholds.json"), thresholds
    )


# -----------------------------
# Simulation: full-field and collocation data
# -----------------------------


@dataclass
class Node:
    node_id: int
    x_mm: float
    y_mm: float
    z_mm: float
    T_C: float
    Ux_um: float
    Uy_um: float
    Uz_um: float
    sigma_xx: float
    sigma_yy: float
    sigma_zz: float
    sigma_xy: float
    sigma_yz: float
    sigma_xz: float
    eps_xx: float
    eps_yy: float
    eps_zz: float
    eps_xy: float
    eps_yz: float
    eps_xz: float


def generate_full_field(seed: int, dims: Dict, macro: Dict, nx: int = 20, ny: int = 20, nz: int = 12) -> List[Node]:
    rng = seeded_random(seed)
    alpha = macro["coefficient_thermal_expansion_per_K"]  # per K
    length = dims["length_mm"]
    width = dims["width_mm"]
    thickness = dims["thickness_mm"]

    nodes: List[Node] = []
    node_id = 1
    # Assume a modest in-plane gradient and through-thickness gradient from sintering
    peak_T = 1200.0
    ambient_T = 25.0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = (i + 0.5) * length / nx
                y = (j + 0.5) * width / ny
                z = (k + 0.5) * thickness / nz
                # Temperature field: hotter near center and at mid-thickness
                cx = (x - 0.5 * length) / (0.5 * length)
                cy = (y - 0.5 * width) / (0.5 * width)
                cz = (z - 0.5 * thickness) / (0.5 * thickness)
                r2 = cx * cx + cy * cy
                T = ambient_T + (peak_T - ambient_T) * math.exp(-1.2 * r2) * (1.0 - 0.35 * cz * cz)
                # Thermal strain (small) and fictitious fixed corner causing displacements
                # Convert mm position to m for strain-displacement consistency in microns
                # Use alpha * (T - T_ref) * position as a proxy for displacement relative to a clamped corner
                dT = T - ambient_T
                Ux = 1e3 * (alpha * dT * (x / 1000.0)) * 1000.0
                Uy = 1e3 * (alpha * dT * (y / 1000.0)) * 1000.0
                Uz = 1e3 * (alpha * dT * (z / 1000.0)) * 1000.0
                # Stress tensor: compressive where T is higher, with bending terms
                base = -0.18 * dT  # MPa per deg C proxy
                sigma_xx = base * (1.0 - 0.2 * cx) + rng.gauss(0.0, 4.0)
                sigma_yy = base * (1.0 + 0.2 * cy) + rng.gauss(0.0, 4.0)
                sigma_zz = 0.35 * base * (1.0 - 0.1 * cz) + rng.gauss(0.0, 2.0)
                sigma_xy = rng.gauss(0.0, 2.5) * (1.0 - min(1.0, 0.9 * math.sqrt(r2)))
                sigma_yz = rng.gauss(0.0, 2.0) * (1.0 - 0.5 * abs(cz))
                sigma_xz = rng.gauss(0.0, 2.0) * (1.0 - 0.5 * abs(cz))
                # Strain from stress (very rough proxy using E and nu)
                E = macro["youngs_modulus_GPa"] * 1e3  # MPa
                nu = macro["poisson_ratio"]
                # Isotropic linear elastic approximation for principal components
                eps_xx = (sigma_xx - nu * (sigma_yy + sigma_zz)) / E + alpha * dT
                eps_yy = (sigma_yy - nu * (sigma_xx + sigma_zz)) / E + alpha * dT
                eps_zz = (sigma_zz - nu * (sigma_xx + sigma_yy)) / E + alpha * dT
                eps_xy = sigma_xy / (E / (2.0 * (1.0 + nu)))
                eps_yz = sigma_yz / (E / (2.0 * (1.0 + nu)))
                eps_xz = sigma_xz / (E / (2.0 * (1.0 + nu)))

                nodes.append(Node(
                    node_id=node_id,
                    x_mm=round(x, 3), y_mm=round(y, 3), z_mm=round(z, 3),
                    T_C=round(T, 2),
                    Ux_um=round(Ux, 3), Uy_um=round(Uy, 3), Uz_um=round(Uz, 3),
                    sigma_xx=round(sigma_xx, 3), sigma_yy=round(sigma_yy, 3), sigma_zz=round(sigma_zz, 3),
                    sigma_xy=round(sigma_xy, 3), sigma_yz=round(sigma_yz, 3), sigma_xz=round(sigma_xz, 3),
                    eps_xx=round(eps_xx, 8), eps_yy=round(eps_yy, 8), eps_zz=round(eps_zz, 8),
                    eps_xy=round(eps_xy, 8), eps_yz=round(eps_yz, 8), eps_xz=round(eps_xz, 8),
                ))
                node_id += 1
    return nodes


def save_full_field(base_dir: str, nodes: List[Node]) -> None:
    processed = os.path.join(base_dir, "processed")
    header = [
        "node_id", "x_mm", "y_mm", "z_mm", "T_C", "Ux_um", "Uy_um", "Uz_um",
        "sigma_xx_MPa", "sigma_yy_MPa", "sigma_zz_MPa", "sigma_xy_MPa", "sigma_yz_MPa", "sigma_xz_MPa",
        "epsilon_xx", "epsilon_yy", "epsilon_zz", "epsilon_xy", "epsilon_yz", "epsilon_xz",
    ]
    rows: List[List] = []
    for n in nodes:
        rows.append([
            n.node_id, n.x_mm, n.y_mm, n.z_mm, n.T_C, n.Ux_um, n.Uy_um, n.Uz_um,
            n.sigma_xx, n.sigma_yy, n.sigma_zz, n.sigma_xy, n.sigma_yz, n.sigma_xz,
            n.eps_xx, n.eps_yy, n.eps_zz, n.eps_xy, n.eps_yz, n.eps_xz,
        ])
    write_csv(os.path.join(processed, "full_field.csv"), header, rows)


def save_collocation_subset(base_dir: str, nodes: List[Node], dims: Dict, seed: int, n_points: int = 350) -> None:
    rng = seeded_random(seed)
    processed = os.path.join(base_dir, "processed")
    # Identify candidate nodes near features: free surfaces (z near 0 or thickness), edges (x~0/length or y~0/width)
    length = dims["length_mm"]
    width = dims["width_mm"]
    thickness = dims["thickness_mm"]

    def is_near_edge(n: Node) -> bool:
        near_x = (n.x_mm < 0.05 * length) or (n.x_mm > 0.95 * length)
        near_y = (n.y_mm < 0.05 * width) or (n.y_mm > 0.95 * width)
        return near_x or near_y

    def is_near_surface(n: Node) -> bool:
        return (n.z_mm < 0.08 * thickness) or (n.z_mm > 0.92 * thickness)

    candidates = [n for n in nodes if is_near_edge(n) or is_near_surface(n)]
    if len(candidates) < n_points:
        # Fallback to include all nodes if not enough candidates
        candidates = nodes
    subset = rng.sample(candidates, k=min(n_points, len(candidates)))

    header = [
        "node_id", "feature_tag", "x_mm", "y_mm", "z_mm", "T_C", "Ux_um", "Uy_um", "Uz_um",
        "sigma_xx_MPa", "sigma_yy_MPa", "sigma_zz_MPa", "sigma_xy_MPa", "sigma_yz_MPa", "sigma_xz_MPa",
        "epsilon_xx", "epsilon_yy", "epsilon_zz", "epsilon_xy", "epsilon_yz", "epsilon_xz",
    ]
    rows: List[List] = []
    for n in subset:
        tag = []
        if (n.x_mm < 0.05 * length) or (n.x_mm > 0.95 * length):
            tag.append("free_edge")
        if (n.y_mm < 0.05 * width) or (n.y_mm > 0.95 * width):
            tag.append("free_edge")
        if (n.z_mm < 0.08 * thickness) or (n.z_mm > 0.92 * thickness):
            tag.append("free_surface")
        if not tag:
            tag.append("bulk")
        rows.append([
            n.node_id, ",".join(sorted(set(tag))), n.x_mm, n.y_mm, n.z_mm, n.T_C, n.Ux_um, n.Uy_um, n.Uz_um,
            n.sigma_xx, n.sigma_yy, n.sigma_zz, n.sigma_xy, n.sigma_yz, n.sigma_xz,
            n.eps_xx, n.eps_yy, n.eps_zz, n.eps_xy, n.eps_yz, n.eps_xz,
        ])
    write_csv(os.path.join(processed, "collocation_subset.csv"), header, rows)


# -----------------------------
# Meso-scale microstructure data
# -----------------------------


def generate_grain_size_distribution(seed: int, n_grains: int = 1000) -> List[List]:
    rng = seeded_random(seed)
    rows: List[List] = []
    # Log-normal distribution around 10 um
    for gid in range(1, n_grains + 1):
        size_um = max(0.2, rng.lognormvariate(math.log(10.0), 0.35))
        rows.append([gid, round(size_um, 3)])
    return rows


def generate_porosity_map(seed: int, nx: int = 80, ny: int = 80, pitch_um: float = 2.0) -> List[List]:
    rng = seeded_random(seed + 3)
    rows: List[List] = []
    # Spatially varying porosity using smooth fields
    for i in range(nx):
        for j in range(ny):
            x_um = (i + 0.5) * pitch_um
            y_um = (j + 0.5) * pitch_um
            s = 0.08 + 0.06 * math.sin(2.0 * math.pi * i / nx) * math.sin(2.0 * math.pi * j / ny)
            s += rng.gauss(0.0, 0.01)
            s = min(0.45, max(0.0, s))
            rows.append([round(x_um, 2), round(y_um, 2), round(s, 4)])
    return rows


def save_meso_microstructure(base_dir: str, seed: int) -> None:
    processed = os.path.join(base_dir, "processed")
    grain_sizes = generate_grain_size_distribution(seed)
    write_csv(
        os.path.join(processed, "grain_size_distribution.csv"),
        ["grain_id", "equivalent_diameter_um"],
        grain_sizes,
    )
    porosity = generate_porosity_map(seed)
    write_csv(
        os.path.join(processed, "porosity_map.csv"),
        ["x_um", "y_um", "porosity_fraction"],
        porosity,
    )


# -----------------------------
# Micro-scale EBSD and grain boundary properties
# -----------------------------


def generate_ebsd_map(seed: int, n_grains: int = 600) -> List[List]:
    rng = seeded_random(seed)
    rows: List[List] = []
    # Simple tiling of grains across a 2D RVE with random Euler angles (degrees)
    for gid in range(1, n_grains + 1):
        x_um = rng.uniform(0.0, 500.0)
        y_um = rng.uniform(0.0, 500.0)
        phi1 = rng.uniform(0.0, 360.0)
        Phi = rng.uniform(0.0, 180.0)
        phi2 = rng.uniform(0.0, 360.0)
        rows.append([gid, round(x_um, 2), round(y_um, 2), round(phi1, 2), round(Phi, 2), round(phi2, 2)])
    return rows


def generate_grain_boundary_properties(seed: int, n_grains: int = 600) -> List[List]:
    rng = seeded_random(seed + 5)
    rows: List[List] = []
    for gid in range(1, n_grains + 1):
        # Aggregate GB property per grain (simplified)
        gb_strength = rng.uniform(60.0, 180.0)  # MPa
        gb_energy = rng.uniform(0.25, 1.2)  # J/m^2
        gb_diff = 10 ** rng.uniform(-18.0, -14.5)  # m^2/s
        rows.append([gid, round(gb_strength, 2), round(gb_energy, 3), float(f"{gb_diff:.3e}")])
    return rows


def save_micro_scale(base_dir: str, seed: int) -> None:
    processed = os.path.join(base_dir, "processed")
    ebsd = generate_ebsd_map(seed)
    write_csv(
        os.path.join(processed, "ebsd_map.csv"),
        ["grain_id", "x_um", "y_um", "phi1_deg", "Phi_deg", "phi2_deg"],
        ebsd,
    )
    gb = generate_grain_boundary_properties(seed, n_grains=len(ebsd))
    write_csv(
        os.path.join(processed, "grain_boundary_properties.csv"),
        ["grain_id", "gb_strength_MPa", "gb_energy_J_per_m2", "gb_diffusivity_m2_per_s"],
        gb,
    )


# -----------------------------
# Orchestration
# -----------------------------


def main(seed: int = 42) -> None:
    base_data_dir = "/workspace/data"
    macro_dir = os.path.join(base_data_dir, "macro")
    meso_dir = os.path.join(base_data_dir, "meso")
    micro_dir = os.path.join(base_data_dir, "micro")
    sim_dir = os.path.join(base_data_dir, "simulation")

    # Ensure directory structure exists
    for d in [macro_dir, meso_dir, micro_dir, sim_dir]:
        ensure_dir(os.path.join(d, "raw"))
        ensure_dir(os.path.join(d, "processed"))

    # Macro-scale: properties, dimensions, sintering profile
    save_macro_data(macro_dir, seed)

    # Experimental residual stress (macro + meso)
    save_experimental_residual_stress(macro_dir, seed + 10)

    # Crack initiation & propagation
    save_cracking_data(meso_dir, seed + 20)

    # Simulation full-field and collocation subset
    macro_props = json.load(open(os.path.join(macro_dir, "processed", "material_properties.json"), "r", encoding="utf-8"))
    dims = json.load(open(os.path.join(macro_dir, "processed", "cell_dimensions.json"), "r", encoding="utf-8"))
    nodes = generate_full_field(seed + 30, dims, macro_props)
    save_full_field(sim_dir, nodes)
    save_collocation_subset(sim_dir, nodes, dims, seed + 31)

    # Meso-scale microstructure
    save_meso_microstructure(meso_dir, seed + 40)

    # Micro-scale data
    save_micro_scale(micro_dir, seed + 50)

    # Metadata manifest
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "datasets": {
            "macro": [
                "material_properties.json",
                "cell_dimensions.json",
                "sintering_profile.csv",
                "residual_stress_surface_XRD.csv",
                "residual_stress_bulk_synchrotron.csv",
            ],
            "meso": [
                "microcrack_locations_SEM.csv",
                "fracture_tests.csv",
                "critical_thresholds.json",
                "grain_size_distribution.csv",
                "porosity_map.csv",
            ],
            "micro": [
                "ebsd_map.csv",
                "grain_boundary_properties.csv",
            ],
            "simulation": [
                "full_field.csv",
                "collocation_subset.csv",
            ],
        }
    }
    write_json(os.path.join(base_data_dir, "manifest.json"), manifest)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fabricate multi-scale FEM validation dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(seed=args.seed)

