#!/usr/bin/env python3

import argparse
import json
import math
import os
import zipfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.ndimage import gaussian_filter, label
from tqdm import tqdm


@dataclass
class ParamMeta:
    name: str
    unit: str
    low: float
    high: float
    distribution: str  # 'uniform' | 'loguniform' | 'normal' (use low/high as min/max for bounds)
    scale: str
    category: str
    fidelities: List[str]
    description: str


SEED_DEFAULT = 42


def get_rng(seed: int):
    return np.random.default_rng(seed)


def latin_hypercube_unit(n_samples: int, n_dims: int, seed: int) -> np.ndarray:
    engine = qmc.LatinHypercube(d=n_dims, seed=seed)
    return engine.random(n=n_samples)


def transform_from_unit(u: np.ndarray, meta: ParamMeta) -> np.ndarray:
    if meta.distribution == "uniform":
        return meta.low + u * (meta.high - meta.low)
    if meta.distribution == "loguniform":
        log_low = math.log(meta.low)
        log_high = math.log(meta.high)
        return np.exp(log_low + u * (log_high - log_low))
    if meta.distribution == "normal":
        # Interpret low/high as 3-sigma bounds around the midpoint
        mu = 0.5 * (meta.low + meta.high)
        sigma = (meta.high - meta.low) / 6.0
        # Clamp to [low, high]
        from scipy.stats import norm  # local import to avoid global cost
        vals = norm.ppf(u, loc=mu, scale=sigma)
        return np.clip(vals, meta.low, meta.high)
    raise ValueError(f"Unknown distribution: {meta.distribution}")


def dirichlet_composition(names: List[str], alpha: List[float], n_samples: int, rng: np.random.Generator) -> pd.DataFrame:
    comps = rng.dirichlet(alpha, size=n_samples)
    df = pd.DataFrame(comps, columns=names)
    return df


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def define_parameter_space() -> Dict[str, ParamMeta]:
    # Note: Bounds are plausible engineering ranges for synthetic data; adjust as needed per study
    params: Dict[str, ParamMeta] = {}
    def add(meta: ParamMeta):
        params[meta.name] = meta

    # System — Operating Conditions (LF, MF, HF)
    add(ParamMeta("Uf", "fraction", 0.60, 0.90, "uniform", "System", "Operating Conditions", ["LF", "MF", "HF"], "Fuel utilization fraction"))
    add(ParamMeta("Ou", "fraction", 0.10, 0.60, "uniform", "System", "Operating Conditions", ["LF", "MF", "HF"], "Oxidant utilization fraction"))
    add(ParamMeta("current_density_Acm2", "A/cm^2", 0.10, 1.50, "uniform", "System", "Operating Conditions", ["LF", "MF", "HF"], "Applied current density"))
    add(ParamMeta("voltage_V", "V", 0.60, 1.10, "uniform", "System", "Operating Conditions", ["LF", "MF", "HF"], "Cell voltage"))
    add(ParamMeta("temperature_K", "K", 923.0, 1173.0, "uniform", "System", "Operating Conditions", ["LF", "MF", "HF"], "Operating temperature"))
    add(ParamMeta("pressure_bar", "bar", 1.0, 3.0, "uniform", "System", "Operating Conditions", ["LF", "MF", "HF"], "Operating pressure"))
    add(ParamMeta("fuel_flow_SLPM", "SLPM", 0.10, 10.0, "loguniform", "System", "Operating Conditions", ["LF", "MF", "HF"], "Fuel flow rate at STP"))
    add(ParamMeta("air_flow_SLPM", "SLPM", 0.50, 50.0, "loguniform", "System", "Operating Conditions", ["LF", "MF", "HF"], "Air flow rate at STP"))

    # Transient Cycling (LF, MF, HF)
    add(ParamMeta("startup_ramp_K_per_min", "K/min", 1.0, 5.0, "uniform", "System", "Transient Cycles", ["LF", "MF", "HF"], "Thermal ramp during startup"))
    add(ParamMeta("shutdown_ramp_K_per_min", "K/min", 1.0, 5.0, "uniform", "System", "Transient Cycles", ["LF", "MF", "HF"], "Thermal ramp during shutdown"))
    add(ParamMeta("load_ramp_Acm2_per_min", "A/(cm^2·min)", 0.01, 0.50, "loguniform", "System", "Transient Cycles", ["LF", "MF", "HF"], "Load-following ramp rate"))
    add(ParamMeta("load_cycle_amplitude_Acm2", "A/cm^2", 0.05, 0.50, "uniform", "System", "Transient Cycles", ["LF", "MF", "HF"], "Amplitude of load cycling"))
    add(ParamMeta("load_cycle_period_min", "min", 5.0, 60.0, "loguniform", "System", "Transient Cycles", ["LF", "MF", "HF"], "Period of load cycling"))
    add(ParamMeta("startups_per_day", "/day", 0.0, 1.0, "uniform", "System", "Transient Cycles", ["LF", "MF", "HF"], "Number of startups per day"))
    add(ParamMeta("thermal_cycles_per_day", "/day", 0.0, 2.0, "uniform", "System", "Transient Cycles", ["LF", "MF", "HF"], "Thermal cycles per day"))
    add(ParamMeta("operating_hours", "h", 100.0, 10000.0, "loguniform", "System", "Transient Cycles", ["LF", "MF", "HF"], "Total hours on stream"))
    add(ParamMeta("air_humidity_frac", "fraction", 0.0, 0.10, "uniform", "System", "Operating Conditions", ["LF", "MF", "HF"], "Air relative humidity fraction at inlet"))

    # Geometry (MF, HF)
    add(ParamMeta("active_area_cm2", "cm^2", 1.0, 400.0, "loguniform", "Cell/Stack", "Geometry", ["MF", "HF"], "Cell active area"))
    add(ParamMeta("anode_thickness_um", "um", 200.0, 1500.0, "uniform", "Cell/Stack", "Geometry", ["MF", "HF"], "Anode thickness"))
    add(ParamMeta("cathode_thickness_um", "um", 20.0, 200.0, "uniform", "Cell/Stack", "Geometry", ["MF", "HF"], "Cathode thickness"))
    add(ParamMeta("electrolyte_thickness_um", "um", 5.0, 50.0, "uniform", "Cell/Stack", "Geometry", ["MF", "HF"], "Electrolyte thickness"))
    add(ParamMeta("interconnect_thickness_um", "um", 200.0, 3000.0, "uniform", "Cell/Stack", "Geometry", ["MF", "HF"], "Interconnect thickness"))
    add(ParamMeta("channel_width_mm", "mm", 0.5, 3.0, "uniform", "Cell/Stack", "Geometry", ["MF", "HF"], "Gas channel width (full stack)"))
    add(ParamMeta("channel_height_mm", "mm", 0.5, 3.0, "uniform", "Cell/Stack", "Geometry", ["MF", "HF"], "Gas channel height (full stack)"))
    add(ParamMeta("rib_width_mm", "mm", 0.5, 3.0, "uniform", "Cell/Stack", "Geometry", ["MF", "HF"], "Interconnect rib width (full stack)"))

    # Anode (Ni-YSZ) properties (MF, HF)
    add(ParamMeta("anode_porosity", "fraction", 0.20, 0.50, "uniform", "Material", "Anode", ["MF", "HF"], "Anode porosity"))
    add(ParamMeta("anode_tortuosity", "-", 1.5, 5.0, "uniform", "Material", "Anode", ["MF", "HF"], "Anode tortuosity (effective)"))
    add(ParamMeta("anode_ni_particle_size_um", "um", 0.20, 5.0, "loguniform", "Material", "Anode", ["MF", "HF"], "Ni particle size"))
    add(ParamMeta("anode_tpb_density_um_inv2", "um^-2", 0.10, 10.0, "loguniform", "Material", "Anode", ["MF", "HF"], "Triple-phase boundary length density"))
    add(ParamMeta("anode_ionic_conductivity_S_per_m", "S/m", 10.0, 1000.0, "loguniform", "Material", "Anode", ["MF", "HF"], "Effective ionic conductivity (YSZ network)"))
    add(ParamMeta("anode_electronic_conductivity_S_per_m", "S/m", 1.0e4, 1.0e7, "loguniform", "Material", "Anode", ["MF", "HF"], "Effective electronic conductivity (Ni network)"))

    # Cathode (LSCF) properties (MF, HF)
    add(ParamMeta("cathode_porosity", "fraction", 0.25, 0.45, "uniform", "Material", "Cathode", ["MF", "HF"], "Cathode porosity"))
    add(ParamMeta("cathode_tortuosity", "-", 1.5, 5.0, "uniform", "Material", "Cathode", ["MF", "HF"], "Cathode tortuosity (effective)"))
    add(ParamMeta("cathode_particle_size_um", "um", 0.20, 5.0, "loguniform", "Material", "Cathode", ["MF", "HF"], "Cathode particle size"))
    add(ParamMeta("cathode_chem_expansion_coeff", "-", 0.0, 0.02, "uniform", "Material", "Cathode", ["MF", "HF"], "Chemical expansion coefficient proxy"))
    add(ParamMeta("cathode_ionic_conductivity_S_per_m", "S/m", 10.0, 500.0, "loguniform", "Material", "Cathode", ["MF", "HF"], "Effective ionic conductivity"))
    add(ParamMeta("cathode_electronic_conductivity_S_per_m", "S/m", 1.0e4, 1.0e6, "loguniform", "Material", "Cathode", ["MF", "HF"], "Effective electronic conductivity"))

    # Electrolyte (YSZ) properties (MF, HF)
    add(ParamMeta("electrolyte_ionic_conductivity_S_per_m", "S/m", 10.0, 300.0, "loguniform", "Material", "Electrolyte", ["MF", "HF"], "Ionic conductivity of YSZ"))
    add(ParamMeta("electrolyte_youngs_modulus_GPa", "GPa", 150.0, 220.0, "uniform", "Material", "Electrolyte", ["MF", "HF"], "Young's modulus"))
    add(ParamMeta("electrolyte_poissons_ratio", "-", 0.20, 0.35, "uniform", "Material", "Electrolyte", ["MF", "HF"], "Poisson's ratio"))
    add(ParamMeta("electrolyte_cte_per_K", "1/K", 8.0e-6, 11.0e-6, "uniform", "Material", "Electrolyte", ["MF", "HF"], "Thermal expansion coefficient"))
    add(ParamMeta("electrolyte_fracture_toughness_MPa_sqrt_m", "MPa·m^0.5", 1.0, 4.0, "uniform", "Material", "Electrolyte", ["MF", "HF"], "Fracture toughness"))

    # Interconnect (Crofer 22APU) properties (MF, HF)
    add(ParamMeta("interconnect_cte_per_K", "1/K", 10.0e-6, 13.0e-6, "uniform", "Material", "Interconnect", ["MF", "HF"], "CTE of interconnect steel"))
    add(ParamMeta("interconnect_creep_A_1_over_MPa_pow_n_s", "1/(MPa^n·s)", 1e-25, 1e-18, "loguniform", "Material", "Interconnect", ["MF", "HF"], "Norton creep A parameter"))
    add(ParamMeta("interconnect_creep_n", "-", 3.0, 8.0, "uniform", "Material", "Interconnect", ["MF", "HF"], "Norton creep stress exponent n"))
    add(ParamMeta("interconnect_oxide_kp_m2_per_s", "m^2/s", 1e-15, 1e-12, "loguniform", "Material", "Interconnect", ["MF", "HF"], "Parabolic oxide growth rate k_p"))

    return params


def build_data_dictionary(params: Dict[str, ParamMeta]) -> pd.DataFrame:
    rows = []
    for name, meta in params.items():
        rows.append({
            "name": name,
            "unit": meta.unit,
            "low": meta.low,
            "high": meta.high,
            "distribution": meta.distribution,
            "scale": meta.scale,
            "category": meta.category,
            "fidelities": ",".join(meta.fidelities),
            "description": meta.description,
        })
    # Add composition entries
    rows.append({
        "name": "fuel_composition_{H2,CO,CH4,CO2,H2O,N2}",
        "unit": "mole fraction",
        "low": 0.0,
        "high": 1.0,
        "distribution": "dirichlet",
        "scale": "System",
        "category": "Operating Conditions",
        "fidelities": "LF,MF,HF",
        "description": "Fuel composition at inlet; components sum to 1.0",
    })
    rows.append({
        "name": "air_composition_{O2,N2}",
        "unit": "mole fraction",
        "low": 0.0,
        "high": 1.0,
        "distribution": "dirichlet",
        "scale": "System",
        "category": "Operating Conditions",
        "fidelities": "LF,MF,HF",
        "description": "Air composition at inlet; components sum to 1.0",
    })
    rows.append({
        "name": "microstructure_{anode,cathode}",
        "unit": "voxels",
        "low": 0,
        "high": 1,
        "distribution": "synthetic level-set",
        "scale": "Microstructural",
        "category": "Properties",
        "fidelities": "HF",
        "description": "3D voxelized phases generated by thresholded smoothed noise; saved as .npz",
    })
    return pd.DataFrame(rows)


def sample_params(param_names: List[str], params: Dict[str, ParamMeta], n: int, seed: int) -> pd.DataFrame:
    metas = [params[name] for name in param_names]
    u = latin_hypercube_unit(n, len(metas), seed)
    data = {}
    for j, meta in enumerate(metas):
        data[meta.name] = transform_from_unit(u[:, j], meta)
    return pd.DataFrame(data)


def generate_system_level(n: int, params: Dict[str, ParamMeta], seed: int) -> pd.DataFrame:
    # Main scalar parameters for LF
    lf_names = [
        "Uf", "Ou", "current_density_Acm2", "voltage_V", "temperature_K", "pressure_bar",
        "fuel_flow_SLPM", "air_flow_SLPM",
        "startup_ramp_K_per_min", "shutdown_ramp_K_per_min", "load_ramp_Acm2_per_min",
        "load_cycle_amplitude_Acm2", "load_cycle_period_min", "startups_per_day",
        "thermal_cycles_per_day", "operating_hours", "air_humidity_frac",
    ]
    df = sample_params(lf_names, params, n, seed)

    # Fuel composition via Dirichlet; typical syngas/hydrogen-rich
    rng = get_rng(seed + 101)
    fuel_names = ["fuel_H2", "fuel_CO", "fuel_CH4", "fuel_CO2", "fuel_H2O", "fuel_N2"]
    alpha_fuel = [3.0, 1.0, 0.8, 0.5, 1.5, 0.5]  # bias toward H2 and H2O
    df_fuel = dirichlet_composition(fuel_names, alpha_fuel, n, rng)

    # Air composition; near standard but allow slight variation
    air_names = ["air_O2", "air_N2"]
    alpha_air = [2.1, 7.9]
    df_air = dirichlet_composition(air_names, alpha_air, n, rng)

    out = pd.concat([df, df_fuel, df_air], axis=1)

    # Some light constraints/derived sanity checks
    out["Uf"] = np.clip(out["Uf"], 0.55, 0.95)
    out["Ou"] = np.clip(out["Ou"], 0.05, 0.70)

    # Ensure compositions sum exactly to 1 (numerical safety)
    fuel_sum = out[fuel_names].sum(axis=1).values
    out[fuel_names] = out[fuel_names].div(fuel_sum, axis=0)
    air_sum = out[air_names].sum(axis=1).values
    out[air_names] = out[air_names].div(air_sum, axis=0)

    out["fidelity"] = "LF"
    return out


def generate_mf(n: int, params: Dict[str, ParamMeta], seed: int) -> pd.DataFrame:
    base = generate_system_level(n, params, seed)
    mf_names = [
        "active_area_cm2", "anode_thickness_um", "cathode_thickness_um", "electrolyte_thickness_um",
        "interconnect_thickness_um", "channel_width_mm", "channel_height_mm", "rib_width_mm",
        "anode_porosity", "anode_tortuosity", "anode_ni_particle_size_um", "anode_tpb_density_um_inv2",
        "anode_ionic_conductivity_S_per_m", "anode_electronic_conductivity_S_per_m",
        "cathode_porosity", "cathode_tortuosity", "cathode_particle_size_um", "cathode_chem_expansion_coeff",
        "cathode_ionic_conductivity_S_per_m", "cathode_electronic_conductivity_S_per_m",
        "electrolyte_ionic_conductivity_S_per_m", "electrolyte_youngs_modulus_GPa",
        "electrolyte_poissons_ratio", "electrolyte_cte_per_K", "electrolyte_fracture_toughness_MPa_sqrt_m",
        "interconnect_cte_per_K", "interconnect_creep_A_1_over_MPa_pow_n_s", "interconnect_creep_n",
        "interconnect_oxide_kp_m2_per_s",
    ]
    df_mf = sample_params(mf_names, params, n, seed + 7)

    out = pd.concat([base.drop(columns=["fidelity"]), df_mf], axis=1)
    out["fidelity"] = "MF"
    return out


def synthesize_three_phase_microstructure(shape: Tuple[int, int, int], phase_fracs: Tuple[float, float, float], grain_sigma_vox: float, rng: np.random.Generator) -> np.ndarray:
    # Generate smoothed noise field and threshold to match exact fractions
    field = gaussian_filter(rng.standard_normal(size=shape), sigma=grain_sigma_vox)
    # Compute thresholds for desired quantiles
    p1, p2, p3 = phase_fracs
    assert abs(p1 + p2 + p3 - 1.0) < 1e-6
    t1 = np.quantile(field, p1)
    t2 = np.quantile(field, p1 + p2)
    labels = np.zeros(shape, dtype=np.uint8)
    labels[field < t1] = 0  # phase 0
    labels[(field >= t1) & (field < t2)] = 1  # phase 1
    labels[field >= t2] = 2  # phase 2
    return labels


def estimate_specific_surface_area(labels: np.ndarray, voxel_size_um: float) -> float:
    # Simple isotropic surface area proxy: count interfaces along axes
    total_interfaces = 0
    for axis in range(3):
        shifted = np.roll(labels, shift=-1, axis=axis)
        total_interfaces += np.sum(labels != shifted)
    # Each interface spans voxel_size^2 area; count unique once
    area_um2 = (total_interfaces / 2.0) * (voxel_size_um ** 2)
    volume_um3 = np.prod(labels.shape) * (voxel_size_um ** 3)
    ssa = area_um2 / volume_um3
    return float(ssa)


def largest_cluster_fraction(binary: np.ndarray) -> float:
    lab, nlab = label(binary)
    if nlab == 0:
        return 0.0
    counts = np.bincount(lab.ravel())
    counts[0] = 0  # background label
    return float(counts.max() / binary.size)


def estimate_tortuosity_from_porosity(porosity: float, rng: np.random.Generator) -> float:
    # Simple empirical proxy: tau = 1 + a*(1-porosity) + noise
    base = 1.0 + 1.2 * (1.0 - porosity)
    noise = rng.normal(0.0, 0.05)
    return float(max(1.0, base + noise))


def generate_hf(n: int, params: Dict[str, ParamMeta], seed: int, out_dir: str, micro_size: int, voxel_size_um: float, save_npz: bool) -> Tuple[pd.DataFrame, List[str], List[str]]:
    base = generate_mf(n, params, seed)
    rng = get_rng(seed + 999)

    anode_paths: List[str] = []
    cathode_paths: List[str] = []

    micro_dir_anode = os.path.join(out_dir, "microstructures", "anode")
    micro_dir_cathode = os.path.join(out_dir, "microstructures", "cathode")
    ensure_dir(micro_dir_anode)
    ensure_dir(micro_dir_cathode)

    # For each sample, generate microstructures consistent with porosity and random phase splits
    records = []
    for i in tqdm(range(n), desc="HF microstructures"):
        record = {}
        # Anode phases: 0=pore, 1=Ni, 2=YSZ
        anode_por = float(base.loc[i, "anode_porosity"]) if "anode_porosity" in base.columns else float(rng.uniform(0.2, 0.5))
        ni_frac_in_solid = float(rng.uniform(0.3, 0.7))
        anode_fracs = (anode_por, (1 - anode_por) * ni_frac_in_solid, (1 - anode_por) * (1 - ni_frac_in_solid))
        anode_sigma = float(np.interp(anode_por, [0.2, 0.5], [1.5, 3.0]))  # coarser with higher porosity
        anode_labels = synthesize_three_phase_microstructure((micro_size, micro_size, micro_size), anode_fracs, anode_sigma, rng)
        anode_ssa = estimate_specific_surface_area(anode_labels, voxel_size_um)
        anode_pore_lcf = largest_cluster_fraction(anode_labels == 0)
        anode_tau_est = estimate_tortuosity_from_porosity(anode_por, rng)

        anode_path = os.path.join(micro_dir_anode, f"anode_{i:05d}.npz") if save_npz else ""
        if save_npz:
            np.savez_compressed(
                anode_path,
                voxels=anode_labels,
                voxel_size_um=voxel_size_um,
                phase_order=np.array(["pore", "Ni", "YSZ"]) 
            )
            anode_paths.append(anode_path)

        # Cathode phases: 0=pore, 1=LSCF, 2=GDC (infiltrate)
        cath_por = float(base.loc[i, "cathode_porosity"]) if "cathode_porosity" in base.columns else float(rng.uniform(0.25, 0.45))
        gdc_frac_in_solid = float(rng.uniform(0.0, 0.20))
        cathode_fracs = (cath_por, (1 - cath_por) * (1 - gdc_frac_in_solid), (1 - cath_por) * gdc_frac_in_solid)
        cath_sigma = float(np.interp(cath_por, [0.25, 0.45], [1.5, 2.5]))
        cath_labels = synthesize_three_phase_microstructure((micro_size, micro_size, micro_size), cathode_fracs, cath_sigma, rng)
        cath_ssa = estimate_specific_surface_area(cath_labels, voxel_size_um)
        cath_pore_lcf = largest_cluster_fraction(cath_labels == 0)
        cath_tau_est = estimate_tortuosity_from_porosity(cath_por, rng)

        cath_path = os.path.join(micro_dir_cathode, f"cathode_{i:05d}.npz") if save_npz else ""
        if save_npz:
            np.savez_compressed(
                cath_path,
                voxels=cath_labels,
                voxel_size_um=voxel_size_um,
                phase_order=np.array(["pore", "LSCF", "GDC"]) 
            )
            cathode_paths.append(cath_path)

        record.update({
            "anode_phase_fraction_pore": anode_fracs[0],
            "anode_phase_fraction_Ni": anode_fracs[1],
            "anode_phase_fraction_YSZ": anode_fracs[2],
            "anode_ssa_um2_per_um3": anode_ssa,
            "anode_pore_largest_cluster_fraction": anode_pore_lcf,
            "anode_tortuosity_estimated": anode_tau_est,
            "anode_microstructure_path": anode_path,
            "cathode_phase_fraction_pore": cathode_fracs[0],
            "cathode_phase_fraction_LSCF": cathode_fracs[1],
            "cathode_phase_fraction_GDC": cathode_fracs[2],
            "cathode_ssa_um2_per_um3": cath_ssa,
            "cathode_pore_largest_cluster_fraction": cath_pore_lcf,
            "cathode_tortuosity_estimated": cath_tau_est,
            "cathode_microstructure_path": cath_path,
        })
        records.append(record)

    df_hf_extra = pd.DataFrame.from_records(records)
    out = pd.concat([base.drop(columns=["fidelity"]).reset_index(drop=True), df_hf_extra], axis=1)
    out["fidelity"] = "HF"
    return out, anode_paths, cathode_paths


def write_tables(df: pd.DataFrame, path_csv: str, path_parquet: str):
    df.to_csv(path_csv, index=False)
    try:
        df.to_parquet(path_parquet, index=False)
    except Exception as e:
        # Fallback if pyarrow/fastparquet missing
        print(f"Warning: writing parquet failed: {e}")


def write_data_dictionary(dict_df: pd.DataFrame, out_dir: str):
    dict_df.to_csv(os.path.join(out_dir, "data_dictionary.csv"), index=False)
    dict_df.to_json(os.path.join(out_dir, "data_dictionary.json"), orient="records", indent=2)


def write_readme(out_dir: str, lf_n: int, mf_n: int, hf_n: int, micro_size: int):
    text = f"""
SOFC Multi-Fidelity Input Dataset
=================================

This bundle contains synthetic, design-of-experiments datasets for a Multi-Fidelity Digital Twin of SOFCs. It covers:
- LF (Low Fidelity): System-level operating conditions and transients
- MF (Medium Fidelity): Adds cell/stack geometry and bulk material properties
- HF (High Fidelity): Adds microstructural summaries and synthetic 3D voxel microstructures for anode and cathode

Generation method
-----------------
- Latin Hypercube Sampling (LHS) is used across continuous parameters.
- Fuel and air compositions are sampled from Dirichlet distributions and normalized.
- Microstructures are synthesized by thresholding smoothed Gaussian noise to match target phase fractions, then summarized.

Defaults used (can be changed via CLI):
- LF samples: {lf_n}
- MF samples: {mf_n}
- HF samples: {hf_n}
- Microstructure size: {micro_size}^3 voxels (voxel_size_um saved in each .npz)

Caution
-------
- This data is synthetic and for research/prototyping only. Ranges are plausible but not tied to any specific system. Validate and tailor before use.

File layout
-----------
- lf.csv, lf.parquet
- mf.csv, mf.parquet
- hf.csv, hf.parquet
- data_dictionary.csv, data_dictionary.json
- microstructures/
  - anode/anode_XXXXX.npz
  - cathode/cathode_XXXXX.npz

"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)


def zip_bundle(out_dir: str, zip_path: str):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(out_dir):
            for fn in files:
                if fn == os.path.basename(zip_path):
                    continue
                fp = os.path.join(root, fn)
                arcname = os.path.relpath(fp, start=os.path.dirname(out_dir))
                zf.write(fp, arcname)


def main():
    parser = argparse.ArgumentParser(description="Generate SOFC multi-fidelity synthetic input dataset")
    parser.add_argument("--out-dir", default="datasets/sofc_multifidelity", help="Output directory")
    parser.add_argument("--lf-samples", type=int, default=1000)
    parser.add_argument("--mf-samples", type=int, default=500)
    parser.add_argument("--hf-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--micro-size", type=int, default=64, help="Edge length of cubic microstructure")
    parser.add_argument("--voxel-size-um", type=float, default=0.2, help="Voxel size in micrometers")
    parser.add_argument("--no-micro-npz", action="store_true", help="Do not save .npz microstructures (only summaries)")
    parser.add_argument("--no-zip", action="store_true", help="Do not create a final zip bundle")
    args = parser.parse_args()

    out_dir = args.out_dir
    ensure_dir(out_dir)

    params = define_parameter_space()

    # Data dictionary
    dict_df = build_data_dictionary(params)
    write_data_dictionary(dict_df, out_dir)

    # LF
    lf_df = generate_system_level(args.lf_samples, params, args.seed)
    write_tables(lf_df, os.path.join(out_dir, "lf.csv"), os.path.join(out_dir, "lf.parquet"))

    # MF
    mf_df = generate_mf(args.mf_samples, params, args.seed + 1)
    write_tables(mf_df, os.path.join(out_dir, "mf.csv"), os.path.join(out_dir, "mf.parquet"))

    # HF
    hf_df, anode_paths, cathode_paths = generate_hf(
        args.hf_samples, params, args.seed + 2, out_dir, args.micro_size, args.voxel_size_um, save_npz=(not args.no_micro_npz)
    )
    write_tables(hf_df, os.path.join(out_dir, "hf.csv"), os.path.join(out_dir, "hf.parquet"))

    # README
    write_readme(out_dir, args.lf_samples, args.mf_samples, args.hf_samples, args.micro_size)

    # ZIP
    if not args.no_zip:
        zip_path = os.path.join("datasets", "sofc_multifidelity_bundle.zip")
        ensure_dir(os.path.dirname(zip_path))
        zip_bundle(out_dir, zip_path)
        print(f"Wrote zip bundle to: {zip_path}")

    print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
