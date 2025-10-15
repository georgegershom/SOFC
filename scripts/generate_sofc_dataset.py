#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import shutil
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import qmc
from tqdm import tqdm


# -----------------------------
# Utility and sampling helpers
# -----------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def latin_hypercube_sample(num_samples: int, param_bounds: Dict[str, Tuple[float, float]], seed: int) -> pd.DataFrame:
    keys = list(param_bounds.keys())
    bounds = np.array([param_bounds[k] for k in keys], dtype=float)
    # LHS in unit cube
    sampler = qmc.LatinHypercube(d=len(keys), seed=seed)
    unit = sampler.random(n=num_samples)
    # Scale to bounds
    scaled = qmc.scale(unit, bounds[:, 0], bounds[:, 1])
    data = {k: scaled[:, i] for i, k in enumerate(keys)}
    return pd.DataFrame(data)


# -----------------------------
# Physics-inspired derived values
# -----------------------------
FARADAY_C_PER_MOL = 96485.33212
R_J_PER_MOLK = 8.314462618


def compute_gas_flows_from_current(current_density_a_per_cm2: np.ndarray,
                                    cell_active_area_cm2: np.ndarray,
                                    fuel_utilization: np.ndarray,
                                    oxidant_utilization: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Total current in Amps
    total_current_a = current_density_a_per_cm2 * cell_active_area_cm2
    # H2 consumption rate mol/s = I / (2F)
    h2_consumption_mol_s = total_current_a / (2.0 * FARADAY_C_PER_MOL)
    # O2 consumption rate mol/s = I / (4F)
    o2_consumption_mol_s = total_current_a / (4.0 * FARADAY_C_PER_MOL)
    # Inlet molar flow based on utilization
    fuel_inlet_mol_s = h2_consumption_mol_s / np.clip(fuel_utilization, 1e-3, None)
    air_inlet_mol_s = (o2_consumption_mol_s / np.clip(oxidant_utilization, 1e-3, None)) * 4.76  # approximate air factor (O2 fraction ~21%)
    return fuel_inlet_mol_s, air_inlet_mol_s


def ysz_ionic_conductivity_s_per_m(temperature_k: np.ndarray) -> np.ndarray:
    # Arrhenius-like relation: sigma = sigma0 * exp(-Ea/(R*T))
    # Parameters loosely representative for YSZ; randomized later for MF variability
    sigma0 = 3.5e5  # S/m
    Ea_ev = 0.85
    Ea_j = Ea_ev * 1.602176634e-19
    return sigma0 * np.exp(-Ea_j / (R_J_PER_MOLK * temperature_k))


# -----------------------------
# Microstructure synthesis and analysis
# -----------------------------
@dataclass
class MicrostructureSpec:
    volume_size: int
    voxel_size_um: float
    porosity_fraction: float
    solid_phase_1_fraction: float  # fraction of solid that is phase 1 (e.g., Ni or LSCF)
    gaussian_sigma_vox: float


def generate_three_phase_smooth_threshold(spec: MicrostructureSpec, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Single noise field smoothed to impart correlation length ~ gaussian_sigma_vox
    volume = rng.standard_normal((spec.volume_size, spec.volume_size, spec.volume_size), dtype=np.float32)
    if spec.gaussian_sigma_vox > 0:
        volume = ndi.gaussian_filter(volume, sigma=spec.gaussian_sigma_vox, truncate=3.0)
    # Map to three phases based on quantiles
    q_void = np.quantile(volume, spec.porosity_fraction)
    solid_fraction = 1.0 - spec.porosity_fraction
    phase1_fraction_total = solid_fraction * spec.solid_phase_1_fraction
    q_phase1 = np.quantile(volume, spec.porosity_fraction + phase1_fraction_total)
    # 0 = void, 1 = phase1 (e.g., Ni/LSCF), 2 = phase2 (e.g., YSZ)
    three_phase = np.zeros_like(volume, dtype=np.uint8)
    three_phase[volume <= q_void] = 0
    mask_solid = volume > q_void
    three_phase[np.logical_and(mask_solid, volume <= q_phase1)] = 1
    three_phase[volume > q_phase1] = 2
    return three_phase


def generate_two_phase_smooth_threshold(volume_size: int, voxel_size_um: float,
                                        porosity_fraction: float,
                                        gaussian_sigma_vox: float,
                                        seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    volume = rng.standard_normal((volume_size, volume_size, volume_size), dtype=np.float32)
    if gaussian_sigma_vox > 0:
        volume = ndi.gaussian_filter(volume, sigma=gaussian_sigma_vox, truncate=3.0)
    q_void = np.quantile(volume, porosity_fraction)
    two_phase = (volume > q_void).astype(np.uint8)  # 1 = solid (e.g., LSCF), 0 = pore
    return two_phase


@dataclass
class PhaseMorphologyStats:
    volume_fraction: float
    specific_surface_area_per_um: float
    largest_cluster_fraction: float
    equivalent_diameter_mean_um: float
    equivalent_diameter_p90_um: float


def compute_binary_phase_stats(mask: np.ndarray, voxel_size_um: float) -> PhaseMorphologyStats:
    # Volume fraction
    vol_fraction = float(mask.mean())
    # Specific surface area approximation using 6-neighborhood differences
    # Count face differences along axes and convert to area; divide by total volume
    diff_x = np.sum(mask[1:, :, :] != mask[:-1, :, :])
    diff_y = np.sum(mask[:, 1:, :] != mask[:, :-1, :])
    diff_z = np.sum(mask[:, :, 1:] != mask[:, :, :-1])
    total_interface_faces = (diff_x + diff_y + diff_z)
    face_area_um2 = (voxel_size_um ** 2)
    volume_um3 = mask.size * (voxel_size_um ** 3)
    ssa = (total_interface_faces * face_area_um2) / max(volume_um3, 1e-12)
    # Connectivity via largest component fraction
    labeled, num = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if num > 0:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0  # background
        largest = counts.max()
        largest_fraction = float(largest) / float(mask.size)
    else:
        largest_fraction = 0.0
    # Particle size via equivalent spherical diameter
    # Compute component sizes in voxels; convert to diameter in micrometers: d = 2 * (3V/4π)^(1/3)
    if num > 0:
        volumes_vox = counts[counts > 0]
        volumes_um3 = volumes_vox * (voxel_size_um ** 3)
        diam_um = 2.0 * ((3.0 * volumes_um3 / (4.0 * math.pi)) ** (1.0 / 3.0))
        eq_diam_mean = float(np.mean(diam_um)) if diam_um.size > 0 else 0.0
        eq_diam_p90 = float(np.quantile(diam_um, 0.9)) if diam_um.size > 0 else 0.0
    else:
        eq_diam_mean = 0.0
        eq_diam_p90 = 0.0
    return PhaseMorphologyStats(
        volume_fraction=vol_fraction,
        specific_surface_area_per_um=float(ssa),
        largest_cluster_fraction=float(largest_fraction),
        equivalent_diameter_mean_um=float(eq_diam_mean),
        equivalent_diameter_p90_um=float(eq_diam_p90),
    )


# -----------------------------
# Dataset generation
# -----------------------------

def generate_lf_dataset(n_rows: int, seed: int) -> pd.DataFrame:
    param_bounds = {
        "temperature_K": (973.0, 1173.0),
        "pressure_bar": (1.0, 3.0),
        "current_density_A_per_cm2": (0.05, 1.0),
        "voltage_V": (0.7, 1.1),
        "fuel_utilization": (0.5, 0.9),
        "oxidant_utilization": (0.1, 0.3),
        # Transient descriptors
        "num_startup_shutdown_cycles": (1.0, 200.0),
        "num_load_cycles": (100.0, 10000.0),
        "thermal_ramp_rate_K_per_min": (1.0, 10.0),
        "load_ramp_rate_fraction_per_s": (0.001, 0.02),
        "avg_cycle_duration_min": (10.0, 60.0),
        "thermal_cycle_temp_swing_K": (100.0, 300.0),
    }
    df = latin_hypercube_sample(n_rows, param_bounds, seed)
    # Integers where appropriate
    df["num_startup_shutdown_cycles"] = df["num_startup_shutdown_cycles"].round().astype(int)
    df["num_load_cycles"] = df["num_load_cycles"].round().astype(int)
    # Flows are included in LF; approximate using a reference area so that LF remains self-contained
    reference_area_cm2 = 100.0
    fuel_inlet_mol_s, air_inlet_mol_s = compute_gas_flows_from_current(
        current_density_a_per_cm2=df["current_density_A_per_cm2"].to_numpy(),
        cell_active_area_cm2=np.full(n_rows, reference_area_cm2, dtype=float),
        fuel_utilization=df["fuel_utilization"].to_numpy(),
        oxidant_utilization=df["oxidant_utilization"].to_numpy(),
    )
    df["fuel_inlet_mol_s"] = fuel_inlet_mol_s
    df["air_inlet_mol_s"] = air_inlet_mol_s
    df.insert(0, "lf_id", [str(uuid.uuid4()) for _ in range(n_rows)])
    return df


def generate_mf_dataset(n_rows: int, lf_df: pd.DataFrame, seed: int) -> pd.DataFrame:
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
    geom_df = latin_hypercube_sample(n_rows, geom_bounds, seed + 10)

    # Anode (Ni-YSZ)
    anode_bounds = {
        "anode_porosity": (0.25, 0.45),
        "anode_tortuosity": (2.0, 5.0),
        "anode_ni_particle_size_um": (0.1, 2.0),
        "anode_ni_volume_fraction_in_solid": (0.35, 0.6),
        "anode_ionic_conductivity_S_per_m": (5.0, 80.0),
        "anode_electronic_conductivity_S_per_m": (5e5, 1e6),
    }
    anode_df = latin_hypercube_sample(n_rows, anode_bounds, seed + 20)

    # Cathode (LSCF)
    cathode_bounds = {
        "cathode_porosity": (0.25, 0.4),
        "cathode_tortuosity": (2.0, 5.0),
        "cathode_particle_size_um": (0.2, 1.5),
        "cathode_ionic_conductivity_S_per_m": (5.0, 120.0),
        "cathode_electronic_conductivity_S_per_m": (2e3, 2e4),
        "cathode_chemical_expansion_per_K": (1e-6, 2e-5),
    }
    cathode_df = latin_hypercube_sample(n_rows, cathode_bounds, seed + 30)

    # Electrolyte (YSZ)
    # Temperature-aware ionic conductivity; sample sigma0 and Ea variations per sample
    temperature_k_samples = rng.uniform(973.0, 1173.0, size=n_rows)
    sigma0_samples = rng.uniform(2.5e5, 4.5e5, size=n_rows)
    Ea_ev_samples = rng.uniform(0.75, 0.9, size=n_rows)
    sigma_ysz = []
    for T, s0, Ea_ev in zip(temperature_k_samples, sigma0_samples, Ea_ev_samples):
        sigma = s0 * math.exp(-(Ea_ev * 1.602176634e-19) / (R_J_PER_MOLK * T))
        sigma_ysz.append(sigma)
    electrolyte_df = pd.DataFrame({
        "ysz_ionic_conductivity_S_per_m": sigma_ysz,
        "ysz_youngs_modulus_GPa": rng.uniform(170.0, 220.0, size=n_rows),
        "ysz_poisson_ratio": rng.uniform(0.22, 0.30, size=n_rows),
        "ysz_cte_per_K": rng.uniform(9e-6, 11e-6, size=n_rows),
    })

    # Interconnect (Crofer 22APU)
    interconnect_df = pd.DataFrame({
        "interconnect_cte_per_K": rng.uniform(11e-6, 13e-6, size=n_rows),
        "interconnect_creep_A": rng.uniform(1e-21, 1e-19, size=n_rows),
        "interconnect_creep_n": rng.uniform(3.5, 6.5, size=n_rows),
        "interconnect_oxide_kp_m2_per_s": rng.uniform(5e-17, 5e-15, size=n_rows),
    })

    # Combine blocks
    df = pd.concat([geom_df, anode_df, cathode_df, electrolyte_df, interconnect_df], axis=1)

    # TPB density heuristic (per um^2) for anode
    solid_fraction = 1.0 - df["anode_porosity"].to_numpy()
    tpb_scale = 1e-3  # heuristic scale factor
    tpb_density = tpb_scale * (solid_fraction * df["anode_ni_volume_fraction_in_solid"].to_numpy()) / np.clip(df["anode_ni_particle_size_um"].to_numpy(), 1e-3, None)
    df["anode_tpb_density_per_um2"] = tpb_density

    # Parent linkage to LF
    parent_lf_ids = rng.choice(lf_df["lf_id"].to_numpy(), size=n_rows, replace=True)
    df.insert(0, "mf_id", [str(uuid.uuid4()) for _ in range(n_rows)])
    df.insert(1, "parent_lf_id", parent_lf_ids)

    # Recompute system flows more accurately using sampled area
    # Pull parent LF rows to get current density and utilizations
    lf_lookup = lf_df.set_index("lf_id")
    jd = lf_lookup.loc[parent_lf_ids, "current_density_A_per_cm2"].to_numpy()
    fu = lf_lookup.loc[parent_lf_ids, "fuel_utilization"].to_numpy()
    ou = lf_lookup.loc[parent_lf_ids, "oxidant_utilization"].to_numpy()
    fuel_mol_s, air_mol_s = compute_gas_flows_from_current(jd, df["cell_active_area_cm2"].to_numpy(), fu, ou)
    df["fuel_inlet_mol_s_estimate"] = fuel_mol_s
    df["air_inlet_mol_s_estimate"] = air_mol_s
    return df


def generate_hf_dataset(n_rows: int, mf_df: pd.DataFrame, seed: int,
                        micro_size: int, base_voxel_per_um: float = 0.5,
                        out_micro_dir: str = "") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: List[Dict] = []
    mf_lookup = mf_df.set_index("mf_id")
    parent_mf_ids = rng.choice(mf_df["mf_id"].to_numpy(), size=n_rows, replace=True)

    for idx in tqdm(range(n_rows), desc="Synthesizing HF microstructures"):
        sample_seed = int(seed + idx)
        mf_id = parent_mf_ids[idx]
        row = mf_lookup.loc[mf_id]

        # Anode microstructure spec (3-phase: 0=void,1=Ni,2=YSZ)
        anode_porosity = float(row["anode_porosity"])  # target porosity
        ni_frac_solid = float(row["anode_ni_volume_fraction_in_solid"])  # of solid
        ni_fraction_total = (1.0 - anode_porosity) * ni_frac_solid
        # Correlation length via gaussian sigma in voxels
        anode_particle_um = float(row["anode_ni_particle_size_um"])  # proxy for feature scale
        voxel_size_um = max(anode_particle_um * 0.5, 0.05)
        gaussian_sigma_vox = max(anode_particle_um / voxel_size_um / 3.0, 1.0)
        anode_spec = MicrostructureSpec(
            volume_size=micro_size,
            voxel_size_um=voxel_size_um,
            porosity_fraction=anode_porosity,
            solid_phase_1_fraction=ni_frac_solid,
            gaussian_sigma_vox=gaussian_sigma_vox,
        )
        anode_vol = generate_three_phase_smooth_threshold(anode_spec, seed=sample_seed)

        # Cathode microstructure spec (2-phase: 0=void,1=LSCF)
        cathode_porosity = float(row["cathode_porosity"])  # target porosity
        cathode_particle_um = float(row["cathode_particle_size_um"])  # proxy for feature scale
        cathode_voxel_size_um = max(cathode_particle_um * 0.5, 0.05)
        cathode_sigma_vox = max(cathode_particle_um / cathode_voxel_size_um / 3.0, 1.0)
        cathode_vol = generate_two_phase_smooth_threshold(micro_size, cathode_voxel_size_um,
                                                          cathode_porosity, cathode_sigma_vox,
                                                          seed=sample_seed + 1000)

        # Compute morphology stats
        anode_void_mask = (anode_vol == 0)
        anode_ni_mask = (anode_vol == 1)
        anode_ysz_mask = (anode_vol == 2)
        anode_void_stats = compute_binary_phase_stats(anode_void_mask, anode_spec.voxel_size_um)
        anode_ni_stats = compute_binary_phase_stats(anode_ni_mask, anode_spec.voxel_size_um)
        anode_ysz_stats = compute_binary_phase_stats(anode_ysz_mask, anode_spec.voxel_size_um)

        cathode_solid_stats = compute_binary_phase_stats(cathode_vol.astype(bool), cathode_voxel_size_um)
        cathode_void_stats = compute_binary_phase_stats(~cathode_vol.astype(bool), cathode_voxel_size_um)

        # Save volumes
        hf_id = str(uuid.uuid4())
        if out_micro_dir:
            ensure_dir(out_micro_dir)
            anode_path = os.path.join(out_micro_dir, f"{hf_id}_anode.npz")
            cathode_path = os.path.join(out_micro_dir, f"{hf_id}_cathode.npz")
            np.savez_compressed(anode_path, volume=anode_vol, voxel_size_um=anode_spec.voxel_size_um)
            np.savez_compressed(cathode_path, volume=cathode_vol, voxel_size_um=cathode_voxel_size_um)
        else:
            anode_path = ""
            cathode_path = ""

        record = {
            "hf_id": hf_id,
            "parent_mf_id": mf_id,
            "parent_lf_id": mf_lookup.loc[mf_id, "parent_lf_id"],
            # Paths to microstructures
            "anode_microstructure_path": anode_path,
            "cathode_microstructure_path": cathode_path,
            # Anode morphology
            "anode_void_fraction": anode_void_stats.volume_fraction,
            "anode_ni_fraction": anode_ni_stats.volume_fraction,
            "anode_ysz_fraction": anode_ysz_stats.volume_fraction,
            "anode_void_ssa_1_per_um": anode_void_stats.specific_surface_area_per_um,
            "anode_ni_ssa_1_per_um": anode_ni_stats.specific_surface_area_per_um,
            "anode_ysz_ssa_1_per_um": anode_ysz_stats.specific_surface_area_per_um,
            "anode_ni_largest_cluster_fraction": anode_ni_stats.largest_cluster_fraction,
            "anode_ysz_largest_cluster_fraction": anode_ysz_stats.largest_cluster_fraction,
            "anode_ni_eq_diam_mean_um": anode_ni_stats.equivalent_diameter_mean_um,
            "anode_ni_eq_diam_p90_um": anode_ni_stats.equivalent_diameter_p90_um,
            "anode_ysz_eq_diam_mean_um": anode_ysz_stats.equivalent_diameter_mean_um,
            "anode_ysz_eq_diam_p90_um": anode_ysz_stats.equivalent_diameter_p90_um,
            # Cathode morphology
            "cathode_porosity_from_micro": cathode_void_stats.volume_fraction,
            "cathode_solid_ssa_1_per_um": cathode_solid_stats.specific_surface_area_per_um,
            "cathode_solid_largest_cluster_fraction": cathode_solid_stats.largest_cluster_fraction,
            "cathode_solid_eq_diam_mean_um": cathode_solid_stats.equivalent_diameter_mean_um,
            "cathode_solid_eq_diam_p90_um": cathode_solid_stats.equivalent_diameter_p90_um,
            # Voxel sizes
            "anode_voxel_size_um": anode_spec.voxel_size_um,
            "cathode_voxel_size_um": cathode_voxel_size_um,
        }
        records.append(record)

    return pd.DataFrame.from_records(records)


# -----------------------------
# Schema emission and saving helpers
# -----------------------------

def write_schema(schema_path: str) -> None:
    schema: Dict[str, Dict] = {
        "description": "Multi-fidelity SOFC dataset schema. LF: system-level operating conditions; MF: adds geometry/material properties; HF: adds microstructural descriptors and microstructure file paths.",
        "fidelity_levels": ["LF", "MF", "HF"],
        "tables": {
            "lf": {
                "primary_key": "lf_id",
                "fields": {
                    "lf_id": {"type": "string", "description": "Unique LF sample id"},
                    "temperature_K": {"type": "float", "units": "K"},
                    "pressure_bar": {"type": "float", "units": "bar"},
                    "current_density_A_per_cm2": {"type": "float", "units": "A/cm^2"},
                    "voltage_V": {"type": "float", "units": "V"},
                    "fuel_utilization": {"type": "float", "units": "fraction"},
                    "oxidant_utilization": {"type": "float", "units": "fraction"},
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
                    # geometry
                    "cell_active_area_cm2": {"type": "float", "units": "cm^2"},
                    "thickness_anode_um": {"type": "float", "units": "um"},
                    "thickness_cathode_um": {"type": "float", "units": "um"},
                    "thickness_electrolyte_um": {"type": "float", "units": "um"},
                    "thickness_interconnect_um": {"type": "float", "units": "um"},
                    "channel_width_mm": {"type": "float", "units": "mm"},
                    "channel_height_mm": {"type": "float", "units": "mm"},
                    "channel_length_mm": {"type": "float", "units": "mm"},
                    # anode
                    "anode_porosity": {"type": "float"},
                    "anode_tortuosity": {"type": "float"},
                    "anode_ni_particle_size_um": {"type": "float", "units": "um"},
                    "anode_ni_volume_fraction_in_solid": {"type": "float"},
                    "anode_ionic_conductivity_S_per_m": {"type": "float", "units": "S/m"},
                    "anode_electronic_conductivity_S_per_m": {"type": "float", "units": "S/m"},
                    "anode_tpb_density_per_um2": {"type": "float", "units": "1/um^2"},
                    # cathode
                    "cathode_porosity": {"type": "float"},
                    "cathode_tortuosity": {"type": "float"},
                    "cathode_particle_size_um": {"type": "float", "units": "um"},
                    "cathode_ionic_conductivity_S_per_m": {"type": "float", "units": "S/m"},
                    "cathode_electronic_conductivity_S_per_m": {"type": "float", "units": "S/m"},
                    "cathode_chemical_expansion_per_K": {"type": "float", "units": "1/K"},
                    # electrolyte (YSZ)
                    "ysz_ionic_conductivity_S_per_m": {"type": "float", "units": "S/m"},
                    "ysz_youngs_modulus_GPa": {"type": "float", "units": "GPa"},
                    "ysz_poisson_ratio": {"type": "float"},
                    "ysz_cte_per_K": {"type": "float", "units": "1/K"},
                    # interconnect
                    "interconnect_cte_per_K": {"type": "float", "units": "1/K"},
                    "interconnect_creep_A": {"type": "float"},
                    "interconnect_creep_n": {"type": "float"},
                    "interconnect_oxide_kp_m2_per_s": {"type": "float", "units": "m^2/s"},
                    # flows recomputed with area
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
                    # anode stats
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
                    # cathode stats
                    "cathode_porosity_from_micro": {"type": "float"},
                    "cathode_solid_ssa_1_per_um": {"type": "float"},
                    "cathode_solid_largest_cluster_fraction": {"type": "float"},
                    "cathode_solid_eq_diam_mean_um": {"type": "float"},
                    "cathode_solid_eq_diam_p90_um": {"type": "float"},
                    # voxel size metadata
                    "anode_voxel_size_um": {"type": "float", "units": "um"},
                    "cathode_voxel_size_um": {"type": "float", "units": "um"}
                }
            }
        }
    }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)


def save_table(df: pd.DataFrame, out_dir: str, name: str) -> None:
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    parquet_path = os.path.join(out_dir, f"{name}.parquet")
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        # Fallback if parquet engine missing
        pass


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-fidelity SOFC dataset with LHS and synthetic microstructures")
    parser.add_argument("--output-dir", default="datasets/sofc_dataset", help="Output directory for dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lf-samples", type=int, default=1500)
    parser.add_argument("--mf-samples", type=int, default=600)
    parser.add_argument("--hf-samples", type=int, default=60)
    parser.add_argument("--micro-size", type=int, default=64, help="Edge length of cubic microstructure volume (voxels)")
    args = parser.parse_args()

    set_global_seed(args.seed)

    out_root = args.output_dir
    out_lf_dir = os.path.join(out_root, "lf")
    out_mf_dir = os.path.join(out_root, "mf")
    out_hf_dir = os.path.join(out_root, "hf")
    out_micro_dir = os.path.join(out_hf_dir, "microstructures")
    ensure_dir(out_root)

    # LF
    lf_df = generate_lf_dataset(args.lf_samples, seed=args.seed)
    save_table(lf_df, out_lf_dir, "lf")

    # MF
    mf_df = generate_mf_dataset(args.mf_samples, lf_df, seed=args.seed)
    save_table(mf_df, out_mf_dir, "mf")

    # HF
    hf_df = generate_hf_dataset(args.hf_samples, mf_df, seed=args.seed, micro_size=args.micro_size, out_micro_dir=out_micro_dir)
    save_table(hf_df, out_hf_dir, "hf")

    # Schema
    write_schema(os.path.join(out_root, "schema.json"))

    # Zip bundle for download
    zip_path = os.path.join(os.path.dirname(out_root), "sofc_multifidelity_dataset.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.make_archive(zip_path[:-4], 'zip', root_dir=out_root)

    print(f"\nDataset generated:\n - Root: {out_root}\n - LF rows: {len(lf_df)}\n - MF rows: {len(mf_df)}\n - HF rows: {len(hf_df)}\n - Zip: {zip_path}")


if __name__ == "__main__":
    main()
