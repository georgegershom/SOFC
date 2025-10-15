#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import shutil
import uuid
from collections import deque
from typing import Dict, List, Tuple

import numpy as np

# -----------------------------
# Utility and sampling helpers
# -----------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def latin_hypercube_sample(num_samples: int, param_bounds: Dict[str, Tuple[float, float]], seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    keys = list(param_bounds.keys())
    d = len(keys)
    # LHS on [0,1]
    U = np.zeros((num_samples, d), dtype=float)
    for j in range(d):
        bins = (np.arange(num_samples) + rng.random(num_samples)) / num_samples
        rng.shuffle(bins)
        U[:, j] = bins
    # Scale to bounds
    data: Dict[str, np.ndarray] = {}
    for j, k in enumerate(keys):
        low, high = param_bounds[k]
        data[k] = low + U[:, j] * (high - low)
    return data


# -----------------------------
# Physics-inspired derived values
# -----------------------------
FARADAY_C_PER_MOL = 96485.33212
R_J_PER_MOLK = 8.314462618


def compute_gas_flows_from_current(current_density_a_per_cm2: np.ndarray,
                                    cell_active_area_cm2: np.ndarray,
                                    fuel_utilization: np.ndarray,
                                    oxidant_utilization: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    total_current_a = current_density_a_per_cm2 * cell_active_area_cm2
    h2_consumption_mol_s = total_current_a / (2.0 * FARADAY_C_PER_MOL)
    o2_consumption_mol_s = total_current_a / (4.0 * FARADAY_C_PER_MOL)
    fuel_inlet_mol_s = h2_consumption_mol_s / np.clip(fuel_utilization, 1e-6, None)
    air_inlet_mol_s = (o2_consumption_mol_s / np.clip(oxidant_utilization, 1e-6, None)) * 4.76
    return fuel_inlet_mol_s, air_inlet_mol_s


# -----------------------------
# Microstructure synthesis and analysis (NumPy-only)
# -----------------------------

def gaussian_kernel_1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=float)
    radius = max(1, int(math.ceil(truncate * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k


def convolve_along_axis_reflect(volume: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    # Convolve 1D separably along the specified axis with reflect padding
    if kernel.size == 1:
        return volume
    pad = kernel.size // 2
    vol = np.swapaxes(volume, 0, axis)
    out = np.empty_like(vol)
    # Iterate rows (remaining dims flattened as first dimension)
    leading = vol.shape[0]
    rest_shape = vol.shape[1:]
    vol2 = vol.reshape((leading, -1))
    out2 = out.reshape((leading, -1))
    length = vol2.shape[1]
    for i in range(leading):
        row = vol2[i]
        # Reflect pad
        left = row[1:pad+1][::-1] if pad > 0 else np.empty(0, dtype=row.dtype)
        right = row[-pad-1:-1][::-1] if pad > 0 else np.empty(0, dtype=row.dtype)
        padded = np.concatenate([left, row, right])
        conv = np.convolve(padded, kernel, mode='valid')
        out2[i] = conv
    return np.swapaxes(out2.reshape((vol.shape[0],) + rest_shape), 0, axis)


def gaussian_filter_3d(volume: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return volume
    k = gaussian_kernel_1d(sigma)
    v = convolve_along_axis_reflect(volume, k, axis=0)
    v = convolve_along_axis_reflect(v, k, axis=1)
    v = convolve_along_axis_reflect(v, k, axis=2)
    return v


def generate_three_phase_smooth_threshold(volume_size: int, voxel_size_um: float,
                                          porosity_fraction: float,
                                          solid_phase_1_fraction: float,
                                          gaussian_sigma_vox: float,
                                          seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vol = rng.standard_normal((volume_size, volume_size, volume_size)).astype(np.float32)
    if gaussian_sigma_vox > 0:
        vol = gaussian_filter_3d(vol, gaussian_sigma_vox).astype(np.float32)
    q_void = np.quantile(vol, porosity_fraction)
    solid_fraction = 1.0 - porosity_fraction
    phase1_fraction_total = solid_fraction * solid_phase_1_fraction
    q_phase1 = np.quantile(vol, porosity_fraction + phase1_fraction_total)
    three_phase = np.zeros_like(vol, dtype=np.uint8)
    three_phase[vol <= q_void] = 0
    mask_solid = vol > q_void
    three_phase[np.logical_and(mask_solid, vol <= q_phase1)] = 1
    three_phase[vol > q_phase1] = 2
    return three_phase


def generate_two_phase_smooth_threshold(volume_size: int, voxel_size_um: float,
                                        porosity_fraction: float,
                                        gaussian_sigma_vox: float,
                                        seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vol = rng.standard_normal((volume_size, volume_size, volume_size)).astype(np.float32)
    if gaussian_sigma_vox > 0:
        vol = gaussian_filter_3d(vol, gaussian_sigma_vox).astype(np.float32)
    q_void = np.quantile(vol, porosity_fraction)
    two_phase = (vol > q_void).astype(np.uint8)
    return two_phase


def compute_specific_surface_area(mask: np.ndarray, voxel_size_um: float) -> float:
    # Count face interfaces in 6-neighborhood and convert to area per volume
    diff_x = np.sum(mask[1:, :, :] != mask[:-1, :, :])
    diff_y = np.sum(mask[:, 1:, :] != mask[:, :-1, :])
    diff_z = np.sum(mask[:, :, 1:] != mask[:, :, :-1])
    total_faces = diff_x + diff_y + diff_z
    face_area_um2 = (voxel_size_um ** 2)
    volume_um3 = mask.size * (voxel_size_um ** 3)
    return float((total_faces * face_area_um2) / max(volume_um3, 1e-12))


def connected_component_sizes_3d(mask: np.ndarray) -> List[int]:
    # 6-connected components for performance
    nx, ny, nz = mask.shape
    mask_flat = mask.ravel()
    visited = np.zeros(mask_flat.shape[0], dtype=bool)
    sizes: List[int] = []
    stride_x = ny * nz
    stride_y = nz
    stride_z = 1
    true_indices = np.flatnonzero(mask_flat)
    for start in true_indices:
        if visited[start]:
            continue
        # BFS
        q = deque([start])
        visited[start] = True
        comp = 0
        while q:
            idx = q.popleft()
            comp += 1
            x = idx // stride_x
            rem = idx - x * stride_x
            y = rem // stride_y
            z = rem - y * stride_y
            # neighbors
            if x > 0:
                n = idx - stride_x
                if not visited[n] and mask_flat[n]:
                    visited[n] = True
                    q.append(n)
            if x < nx - 1:
                n = idx + stride_x
                if not visited[n] and mask_flat[n]:
                    visited[n] = True
                    q.append(n)
            if y > 0:
                n = idx - stride_y
                if not visited[n] and mask_flat[n]:
                    visited[n] = True
                    q.append(n)
            if y < ny - 1:
                n = idx + stride_y
                if not visited[n] and mask_flat[n]:
                    visited[n] = True
                    q.append(n)
            if z > 0:
                n = idx - stride_z
                if not visited[n] and mask_flat[n]:
                    visited[n] = True
                    q.append(n)
            if z < nz - 1:
                n = idx + stride_z
                if not visited[n] and mask_flat[n]:
                    visited[n] = True
                    q.append(n)
        sizes.append(comp)
    return sizes


def compute_phase_morphology_stats(mask: np.ndarray, voxel_size_um: float) -> Dict[str, float]:
    vol_fraction = float(mask.mean())
    ssa = compute_specific_surface_area(mask, voxel_size_um)
    sizes = connected_component_sizes_3d(mask)
    if sizes:
        largest_fraction = float(max(sizes)) / float(mask.size)
        volumes_um3 = np.array(sizes, dtype=float) * (voxel_size_um ** 3)
        diam_um = 2.0 * ((3.0 * volumes_um3 / (4.0 * math.pi)) ** (1.0 / 3.0))
        eq_diam_mean = float(diam_um.mean())
        eq_diam_p90 = float(np.quantile(diam_um, 0.9))
    else:
        largest_fraction = 0.0
        eq_diam_mean = 0.0
        eq_diam_p90 = 0.0
    return {
        "volume_fraction": vol_fraction,
        "specific_surface_area_per_um": float(ssa),
        "largest_cluster_fraction": float(largest_fraction),
        "equivalent_diameter_mean_um": float(eq_diam_mean),
        "equivalent_diameter_p90_um": float(eq_diam_p90),
    }


# -----------------------------
# Dataset generation (NumPy-only, CSV writer)
# -----------------------------

def write_csv(dict_of_cols: Dict[str, List], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    keys = list(dict_of_cols.keys())
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(','.join(keys) + '\n')
        n = len(dict_of_cols[keys[0]])
        for i in range(n):
            row = []
            for k in keys:
                v = dict_of_cols[k][i]
                if isinstance(v, str):
                    row.append('"' + v.replace('"', '""') + '"')
                else:
                    row.append(str(v))
            f.write(','.join(row) + '\n')


def generate_lf_dataset(n_rows: int, seed: int) -> Dict[str, List]:
    bounds = {
        "temperature_K": (973.0, 1173.0),
        "pressure_bar": (1.0, 3.0),
        "current_density_A_per_cm2": (0.05, 1.0),
        "voltage_V": (0.7, 1.1),
        "fuel_utilization": (0.5, 0.9),
        "oxidant_utilization": (0.1, 0.3),
        "num_startup_shutdown_cycles": (1.0, 200.0),
        "num_load_cycles": (100.0, 10000.0),
        "thermal_ramp_rate_K_per_min": (1.0, 10.0),
        "load_ramp_rate_fraction_per_s": (0.001, 0.02),
        "avg_cycle_duration_min": (10.0, 60.0),
        "thermal_cycle_temp_swing_K": (100.0, 300.0),
    }
    data = latin_hypercube_sample(n_rows, bounds, seed)
    # Integers
    data["num_startup_shutdown_cycles"] = np.rint(data["num_startup_shutdown_cycles"]).astype(int)
    data["num_load_cycles"] = np.rint(data["num_load_cycles"]).astype(int)
    # Flows with reference area
    reference_area_cm2 = 100.0
    fuel_inlet_mol_s, air_inlet_mol_s = compute_gas_flows_from_current(
        current_density_a_per_cm2=data["current_density_A_per_cm2"],
        cell_active_area_cm2=np.full(n_rows, reference_area_cm2, dtype=float),
        fuel_utilization=data["fuel_utilization"],
        oxidant_utilization=data["oxidant_utilization"],
    )
    # Build columns
    cols: Dict[str, List] = {"lf_id": [str(uuid.uuid4()) for _ in range(n_rows)]}
    for k in [
        "temperature_K","pressure_bar","current_density_A_per_cm2","voltage_V","fuel_utilization","oxidant_utilization",
        "num_startup_shutdown_cycles","num_load_cycles","thermal_ramp_rate_K_per_min","load_ramp_rate_fraction_per_s",
        "avg_cycle_duration_min","thermal_cycle_temp_swing_K"
    ]:
        vals = data[k]
        if isinstance(vals, np.ndarray):
            vals = vals.tolist()
        cols[k] = [int(v) if "num_" in k else float(v) for v in vals]
    cols["fuel_inlet_mol_s"] = fuel_inlet_mol_s.tolist()
    cols["air_inlet_mol_s"] = air_inlet_mol_s.tolist()
    return cols


def generate_mf_dataset(n_rows: int, lf_cols: Dict[str, List], seed: int) -> Dict[str, List]:
    rng = np.random.default_rng(seed)
    # Geometry
    geom_bounds = {
        "cell_active_area_cm2": (25.0, 200.0),
        "thickness_anode_um": (200.0, 800.0),
        "thickness_cathode_um": (30.0, 80.0),
        "thickness_electrolyte_um": (5.0, 30.0),
        "thickness_interconnect_um": (300.0, 1000.0),
        "channel_width_mm": (0.5, 2.0),
        "channel_height_mm": (0.5, 2.0),
        "channel_length_mm": (50.0, 100.0),
    }
    geom = latin_hypercube_sample(n_rows, geom_bounds, seed + 10)

    # Anode (Ni-YSZ)
    anode_bounds = {
        "anode_porosity": (0.25, 0.45),
        "anode_tortuosity": (2.0, 5.0),
        "anode_ni_particle_size_um": (0.1, 2.0),
        "anode_ni_volume_fraction_in_solid": (0.35, 0.6),
        "anode_ionic_conductivity_S_per_m": (5.0, 80.0),
        "anode_electronic_conductivity_S_per_m": (5e5, 1e6),
    }
    anode = latin_hypercube_sample(n_rows, anode_bounds, seed + 20)

    # Cathode (LSCF)
    cathode_bounds = {
        "cathode_porosity": (0.25, 0.4),
        "cathode_tortuosity": (2.0, 5.0),
        "cathode_particle_size_um": (0.2, 1.5),
        "cathode_ionic_conductivity_S_per_m": (5.0, 120.0),
        "cathode_electronic_conductivity_S_per_m": (2e3, 2e4),
        "cathode_chemical_expansion_per_K": (1e-6, 2e-5),
    }
    cathode = latin_hypercube_sample(n_rows, cathode_bounds, seed + 30)

    # Electrolyte (YSZ)
    temperature_k_samples = rng.uniform(973.0, 1173.0, size=n_rows)
    sigma0_samples = rng.uniform(2.5e5, 4.5e5, size=n_rows)
    Ea_ev_samples = rng.uniform(0.75, 0.9, size=n_rows)
    ysz_sigma = [s0 * math.exp(-(Ea * 1.602176634e-19) / (R_J_PER_MOLK * T)) for T, s0, Ea in zip(temperature_k_samples, sigma0_samples, Ea_ev_samples)]
    electrolyte = {
        "ysz_ionic_conductivity_S_per_m": np.array(ysz_sigma),
        "ysz_youngs_modulus_GPa": rng.uniform(170.0, 220.0, size=n_rows),
        "ysz_poisson_ratio": rng.uniform(0.22, 0.30, size=n_rows),
        "ysz_cte_per_K": rng.uniform(9e-6, 11e-6, size=n_rows),
    }

    # Interconnect
    interconnect = {
        "interconnect_cte_per_K": rng.uniform(11e-6, 13e-6, size=n_rows),
        "interconnect_creep_A": rng.uniform(1e-21, 1e-19, size=n_rows),
        "interconnect_creep_n": rng.uniform(3.5, 6.5, size=n_rows),
        "interconnect_oxide_kp_m2_per_s": rng.uniform(5e-17, 5e-15, size=n_rows),
    }

    cols: Dict[str, List] = {
        "mf_id": [str(uuid.uuid4()) for _ in range(n_rows)],
        "parent_lf_id": list(np.random.default_rng(seed + 99).choice(lf_cols["lf_id"], size=n_rows, replace=True)),
    }

    # Combine
    for block in (geom, anode, cathode, electrolyte, interconnect):
        for k, v in block.items():
            cols[k] = v.tolist() if isinstance(v, np.ndarray) else list(v)

    # TPB density heuristic
    solid_fraction = 1.0 - np.array(cols["anode_porosity"])
    tpb_scale = 1e-3
    tpb_density = tpb_scale * (solid_fraction * np.array(cols["anode_ni_volume_fraction_in_solid"])) / np.clip(np.array(cols["anode_ni_particle_size_um"]), 1e-3, None)
    cols["anode_tpb_density_per_um2"] = tpb_density.tolist()

    # Recompute flows using area and LF linkages
    lf_index = {lf_id: i for i, lf_id in enumerate(lf_cols["lf_id"]) }
    jd = np.array([lf_cols["current_density_A_per_cm2"][lf_index[pid]] for pid in cols["parent_lf_id"]], dtype=float)
    fu = np.array([lf_cols["fuel_utilization"][lf_index[pid]] for pid in cols["parent_lf_id"]], dtype=float)
    ou = np.array([lf_cols["oxidant_utilization"][lf_index[pid]] for pid in cols["parent_lf_id"]], dtype=float)
    area = np.array(cols["cell_active_area_cm2"], dtype=float)
    fuel_mol_s, air_mol_s = compute_gas_flows_from_current(jd, area, fu, ou)
    cols["fuel_inlet_mol_s_estimate"] = fuel_mol_s.tolist()
    cols["air_inlet_mol_s_estimate"] = air_mol_s.tolist()
    return cols


def generate_hf_dataset(n_rows: int, mf_cols: Dict[str, List], seed: int,
                        micro_size: int, out_micro_dir: str) -> Dict[str, List]:
    rng = np.random.default_rng(seed)
    cols: Dict[str, List] = {
        "hf_id": [],
        "parent_mf_id": [],
        "parent_lf_id": [],
        "anode_microstructure_path": [],
        "cathode_microstructure_path": [],
        "anode_void_fraction": [],
        "anode_ni_fraction": [],
        "anode_ysz_fraction": [],
        "anode_void_ssa_1_per_um": [],
        "anode_ni_ssa_1_per_um": [],
        "anode_ysz_ssa_1_per_um": [],
        "anode_ni_largest_cluster_fraction": [],
        "anode_ysz_largest_cluster_fraction": [],
        "anode_ni_eq_diam_mean_um": [],
        "anode_ni_eq_diam_p90_um": [],
        "anode_ysz_eq_diam_mean_um": [],
        "anode_ysz_eq_diam_p90_um": [],
        "cathode_porosity_from_micro": [],
        "cathode_solid_ssa_1_per_um": [],
        "cathode_solid_largest_cluster_fraction": [],
        "cathode_solid_eq_diam_mean_um": [],
        "cathode_solid_eq_diam_p90_um": [],
        "anode_voxel_size_um": [],
        "cathode_voxel_size_um": [],
    }

    ensure_dir(out_micro_dir)
    n_mf = len(mf_cols["mf_id"])
    for i in range(n_rows):
        mf_idx = int(rng.integers(0, n_mf))
        mf_id = mf_cols["mf_id"][mf_idx]
        lf_id = mf_cols["parent_lf_id"][mf_idx]

        # Anode spec
        anode_porosity = float(mf_cols["anode_porosity"][mf_idx])
        ni_frac_solid = float(mf_cols["anode_ni_volume_fraction_in_solid"][mf_idx])
        anode_particle_um = float(mf_cols["anode_ni_particle_size_um"][mf_idx])
        voxel_size_um = max(anode_particle_um * 0.5, 0.05)
        sigma_vox = max(anode_particle_um / voxel_size_um / 3.0, 1.0)
        anode_vol = generate_three_phase_smooth_threshold(
            volume_size=micro_size,
            voxel_size_um=voxel_size_um,
            porosity_fraction=anode_porosity,
            solid_phase_1_fraction=ni_frac_solid,
            gaussian_sigma_vox=sigma_vox,
            seed=seed + i
        )

        # Cathode spec
        cathode_porosity = float(mf_cols["cathode_porosity"][mf_idx])
        cathode_particle_um = float(mf_cols["cathode_particle_size_um"][mf_idx])
        cathode_voxel_size_um = max(cathode_particle_um * 0.5, 0.05)
        cathode_sigma_vox = max(cathode_particle_um / cathode_voxel_size_um / 3.0, 1.0)
        cathode_vol = generate_two_phase_smooth_threshold(
            micro_size, cathode_voxel_size_um,
            cathode_porosity, cathode_sigma_vox, seed=seed + 1000 + i
        )

        # Morphology stats
        anode_void_mask = (anode_vol == 0)
        anode_ni_mask = (anode_vol == 1)
        anode_ysz_mask = (anode_vol == 2)
        void_stats = compute_phase_morphology_stats(anode_void_mask, voxel_size_um)
        ni_stats = compute_phase_morphology_stats(anode_ni_mask, voxel_size_um)
        ysz_stats = compute_phase_morphology_stats(anode_ysz_mask, voxel_size_um)

        cathode_solid_stats = compute_phase_morphology_stats(cathode_vol.astype(bool), cathode_voxel_size_um)
        cathode_void_stats = compute_phase_morphology_stats(~cathode_vol.astype(bool), cathode_voxel_size_um)

        # Save volumes
        hf_id = str(uuid.uuid4())
        anode_path = os.path.join(out_micro_dir, f"{hf_id}_anode.npz")
        cathode_path = os.path.join(out_micro_dir, f"{hf_id}_cathode.npz")
        np.savez_compressed(anode_path, volume=anode_vol, voxel_size_um=voxel_size_um)
        np.savez_compressed(cathode_path, volume=cathode_vol, voxel_size_um=cathode_voxel_size_um)

        cols["hf_id"].append(hf_id)
        cols["parent_mf_id"].append(mf_id)
        cols["parent_lf_id"].append(lf_id)
        cols["anode_microstructure_path"].append(anode_path)
        cols["cathode_microstructure_path"].append(cathode_path)
        cols["anode_void_fraction"].append(void_stats["volume_fraction"])
        cols["anode_ni_fraction"].append(ni_stats["volume_fraction"])
        cols["anode_ysz_fraction"].append(ysz_stats["volume_fraction"])
        cols["anode_void_ssa_1_per_um"].append(void_stats["specific_surface_area_per_um"])
        cols["anode_ni_ssa_1_per_um"].append(ni_stats["specific_surface_area_per_um"])
        cols["anode_ysz_ssa_1_per_um"].append(ysz_stats["specific_surface_area_per_um"])
        cols["anode_ni_largest_cluster_fraction"].append(ni_stats["largest_cluster_fraction"])
        cols["anode_ysz_largest_cluster_fraction"].append(ysz_stats["largest_cluster_fraction"])
        cols["anode_ni_eq_diam_mean_um"].append(ni_stats["equivalent_diameter_mean_um"])
        cols["anode_ni_eq_diam_p90_um"].append(ni_stats["equivalent_diameter_p90_um"])
        cols["anode_ysz_eq_diam_mean_um"].append(ysz_stats["equivalent_diameter_mean_um"])
        cols["anode_ysz_eq_diam_p90_um"].append(ysz_stats["equivalent_diameter_p90_um"])
        cols["cathode_porosity_from_micro"].append(cathode_void_stats["volume_fraction"])
        cols["cathode_solid_ssa_1_per_um"].append(cathode_solid_stats["specific_surface_area_per_um"])
        cols["cathode_solid_largest_cluster_fraction"].append(cathode_solid_stats["largest_cluster_fraction"])
        cols["cathode_solid_eq_diam_mean_um"].append(cathode_solid_stats["equivalent_diameter_mean_um"])
        cols["cathode_solid_eq_diam_p90_um"].append(cathode_solid_stats["equivalent_diameter_p90_um"])
        cols["anode_voxel_size_um"].append(voxel_size_um)
        cols["cathode_voxel_size_um"].append(cathode_voxel_size_um)

    return cols


# -----------------------------
# Schema emission and saving helpers
# -----------------------------

def write_schema(schema_path: str) -> None:
    schema: Dict[str, Dict] = {
        "description": "Multi-fidelity SOFC dataset schema. LF: system-level operating conditions; MF: adds geometry/material properties; HF: microstructural descriptors and file paths.",
        "fidelity_levels": ["LF", "MF", "HF"],
        "tables": {
            "lf": {
                "primary_key": "lf_id",
                "fields": {
                    "lf_id": {"type": "string"},
                    "temperature_K": {"type": "float", "units": "K"},
                    "pressure_bar": {"type": "float", "units": "bar"},
                    "current_density_A_per_cm2": {"type": "float", "units": "A/cm^2"},
                    "voltage_V": {"type": "float", "units": "V"},
                    "fuel_utilization": {"type": "float"},
                    "oxidant_utilization": {"type": "float"},
                    "num_startup_shutdown_cycles": {"type": "integer"},
                    "num_load_cycles": {"type": "integer"},
                    "thermal_ramp_rate_K_per_min": {"type": "float"},
                    "load_ramp_rate_fraction_per_s": {"type": "float"},
                    "avg_cycle_duration_min": {"type": "float", "units": "min"},
                    "thermal_cycle_temp_swing_K": {"type": "float", "units": "K"},
                    "fuel_inlet_mol_s": {"type": "float", "units": "mol/s"},
                    "air_inlet_mol_s": {"type": "float", "units": "mol/s"}
                }
            },
            "mf": {
                "primary_key": "mf_id",
                "foreign_keys": {"parent_lf_id": "lf.lf_id"},
                "fields": {
                    "mf_id": {"type": "string"},
                    "parent_lf_id": {"type": "string"},
                    "cell_active_area_cm2": {"type": "float", "units": "cm^2"},
                    "thickness_anode_um": {"type": "float", "units": "um"},
                    "thickness_cathode_um": {"type": "float", "units": "um"},
                    "thickness_electrolyte_um": {"type": "float", "units": "um"},
                    "thickness_interconnect_um": {"type": "float", "units": "um"},
                    "channel_width_mm": {"type": "float", "units": "mm"},
                    "channel_height_mm": {"type": "float", "units": "mm"},
                    "channel_length_mm": {"type": "float", "units": "mm"},
                    "anode_porosity": {"type": "float"},
                    "anode_tortuosity": {"type": "float"},
                    "anode_ni_particle_size_um": {"type": "float", "units": "um"},
                    "anode_ni_volume_fraction_in_solid": {"type": "float"},
                    "anode_ionic_conductivity_S_per_m": {"type": "float", "units": "S/m"},
                    "anode_electronic_conductivity_S_per_m": {"type": "float", "units": "S/m"},
                    "anode_tpb_density_per_um2": {"type": "float", "units": "1/um^2"},
                    "cathode_porosity": {"type": "float"},
                    "cathode_tortuosity": {"type": "float"},
                    "cathode_particle_size_um": {"type": "float", "units": "um"},
                    "cathode_ionic_conductivity_S_per_m": {"type": "float", "units": "S/m"},
                    "cathode_electronic_conductivity_S_per_m": {"type": "float", "units": "S/m"},
                    "cathode_chemical_expansion_per_K": {"type": "float", "units": "1/K"},
                    "ysz_ionic_conductivity_S_per_m": {"type": "float", "units": "S/m"},
                    "ysz_youngs_modulus_GPa": {"type": "float", "units": "GPa"},
                    "ysz_poisson_ratio": {"type": "float"},
                    "ysz_cte_per_K": {"type": "float", "units": "1/K"},
                    "interconnect_cte_per_K": {"type": "float", "units": "1/K"},
                    "interconnect_creep_A": {"type": "float"},
                    "interconnect_creep_n": {"type": "float"},
                    "interconnect_oxide_kp_m2_per_s": {"type": "float", "units": "m^2/s"},
                    "fuel_inlet_mol_s_estimate": {"type": "float", "units": "mol/s"},
                    "air_inlet_mol_s_estimate": {"type": "float", "units": "mol/s"}
                }
            },
            "hf": {
                "primary_key": "hf_id",
                "foreign_keys": {"parent_mf_id": "mf.mf_id", "parent_lf_id": "lf.lf_id"},
                "fields": {
                    "hf_id": {"type": "string"},
                    "parent_mf_id": {"type": "string"},
                    "parent_lf_id": {"type": "string"},
                    "anode_microstructure_path": {"type": "string"},
                    "cathode_microstructure_path": {"type": "string"},
                    "anode_void_fraction": {"type": "float"},
                    "anode_ni_fraction": {"type": "float"},
                    "anode_ysz_fraction": {"type": "float"},
                    "anode_void_ssa_1_per_um": {"type": "float"},
                    "anode_ni_ssa_1_per_um": {"type": "float"},
                    "anode_ysz_ssa_1_per_um": {"type": "float"},
                    "anode_ni_largest_cluster_fraction": {"type": "float"},
                    "anode_ysz_largest_cluster_fraction": {"type": "float"},
                    "anode_ni_eq_diam_mean_um": {"type": "float"},
                    "anode_ni_eq_diam_p90_um": {"type": "float"},
                    "anode_ysz_eq_diam_mean_um": {"type": "float"},
                    "anode_ysz_eq_diam_p90_um": {"type": "float"},
                    "cathode_porosity_from_micro": {"type": "float"},
                    "cathode_solid_ssa_1_per_um": {"type": "float"},
                    "cathode_solid_largest_cluster_fraction": {"type": "float"},
                    "cathode_solid_eq_diam_mean_um": {"type": "float"},
                    "cathode_solid_eq_diam_p90_um": {"type": "float"},
                    "anode_voxel_size_um": {"type": "float", "units": "um"},
                    "cathode_voxel_size_um": {"type": "float", "units": "um"}
                }
            }
        }
    }
    with open(schema_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-fidelity SOFC dataset (NumPy-only)")
    parser.add_argument("--output-dir", default="datasets/sofc_dataset", help="Output directory for dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lf-samples", type=int, default=1500)
    parser.add_argument("--mf-samples", type=int, default=600)
    parser.add_argument("--hf-samples", type=int, default=30)
    parser.add_argument("--micro-size", type=int, default=64)
    args = parser.parse_args()

    set_global_seed(args.seed)

    out_root = args.output_dir
    out_lf_dir = os.path.join(out_root, "lf")
    out_mf_dir = os.path.join(out_root, "mf")
    out_hf_dir = os.path.join(out_root, "hf")
    out_micro_dir = os.path.join(out_hf_dir, "microstructures")
    ensure_dir(out_root)

    # LF
    lf_cols = generate_lf_dataset(args.lf_samples, seed=args.seed)
    lf_csv = os.path.join(out_lf_dir, "lf.csv")
    write_csv(lf_cols, lf_csv)

    # MF
    mf_cols = generate_mf_dataset(args.mf_samples, lf_cols, seed=args.seed)
    mf_csv = os.path.join(out_mf_dir, "mf.csv")
    write_csv(mf_cols, mf_csv)

    # HF
    hf_cols = generate_hf_dataset(args.hf_samples, mf_cols, seed=args.seed, micro_size=args.micro_size, out_micro_dir=out_micro_dir)
    hf_csv = os.path.join(out_hf_dir, "hf.csv")
    write_csv(hf_cols, hf_csv)

    # Schema and zip
    write_schema(os.path.join(out_root, "schema.json"))
    zip_path = os.path.join(os.path.dirname(out_root), "sofc_multifidelity_dataset.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.make_archive(zip_path[:-4], 'zip', root_dir=out_root)

    print(f"Dataset generated:\n - Root: {out_root}\n - LF rows: {len(lf_cols['lf_id'])}\n - MF rows: {len(mf_cols['mf_id'])}\n - HF rows: {len(hf_cols['hf_id'])}\n - Zip: {zip_path}")


if __name__ == "__main__":
    main()
