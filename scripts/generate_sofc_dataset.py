#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import math
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def latin_hypercube_sampling(num_samples: int, num_dims: int, rng: np.random.Generator) -> np.ndarray:
    """Classic LHS in [0,1]^d."""
    cut = np.linspace(0, 1, num_samples + 1)
    u = rng.uniform(size=(num_samples, num_dims))
    a = cut[:num_samples]
    b = cut[1:num_samples + 1]
    rdpoints = u * (b - a)[:, None] + a[:, None]
    H = np.zeros_like(rdpoints)
    for j in range(num_dims):
        order = rng.permutation(num_samples)
        H[:, j] = rdpoints[order, 0]  # use one column of rdpoints per dim via permutation
    return H


def choose_enum(options: List[str], t: float) -> str:
    idx = min(len(options) - 1, int(t * len(options)))
    return options[idx]


def map_to_range(t: float, lo: float, hi: float) -> float:
    return lo + (hi - lo) * t


def build_param_space() -> Tuple[List[Tuple[str, Tuple[float, float]]], Dict[str, Any]]:
    """Return flat scalar ranges and structured metadata for rebuild/flattening."""
    ranges: List[Tuple[str, Tuple[float, float]]] = [
        ("operating_voltage_V", (0.6, 1.1)),
        ("current_density_A_per_cm2", (0.2, 2.0)),
        ("air_flow_rate_slpm", (1.0, 20.0)),
        ("fuel_flow_rate_slpm", (0.2, 5.0)),
        ("inlet_temp_fuel_K", (900.0, 1150.0)),
        ("inlet_temp_air_K", (900.0, 1150.0)),
        ("active_area_cm2", (25.0, 100.0)),
        # thickness (um)
        ("thickness_um.anode", (200.0, 1000.0)),
        ("thickness_um.electrolyte", (5.0, 50.0)),
        ("thickness_um.cathode", (20.0, 100.0)),
        ("thickness_um.interconnect", (500.0, 2000.0)),
        ("thickness_um.sealant", (100.0, 500.0)),
        # porosity (dimensionless)
        ("porosity.anode", (0.25, 0.50)),
        ("porosity.cathode", (0.25, 0.50)),
        # permeability (m^2)
        ("permeability.anode", (1e-14, 1e-12)),
        ("permeability.cathode", (1e-15, 1e-13)),
        # ionic conductivity (S/m)
        ("ionic_cond.anode", (0.01, 1.0)),
        ("ionic_cond.electrolyte", (0.5, 8.0)),
        ("ionic_cond.cathode", (0.01, 1.0)),
        # electronic conductivity (S/m)
        ("electronic_cond.anode", (1e4, 2e5)),
        ("electronic_cond.electrolyte", (1e-8, 1e-3)),
        ("electronic_cond.cathode", (1e3, 1e5)),
        ("electronic_cond.interconnect", (1e6, 1e7)),
        ("electronic_cond.sealant", (1e-8, 1e-6)),
        # Young's modulus (GPa)
        ("youngs_modulus_GPa.anode", (50.0, 150.0)),
        ("youngs_modulus_GPa.electrolyte", (150.0, 300.0)),
        ("youngs_modulus_GPa.cathode", (100.0, 200.0)),
        ("youngs_modulus_GPa.interconnect", (150.0, 240.0)),
        ("youngs_modulus_GPa.sealant", (20.0, 80.0)),
        # CTE (1/K)
        ("cte_per_K.anode", (10e-6, 13e-6)),
        ("cte_per_K.electrolyte", (9e-6, 11e-6)),
        ("cte_per_K.cathode", (11e-6, 14e-6)),
        ("cte_per_K.interconnect", (12e-6, 14e-6)),
        ("cte_per_K.sealant", (8e-6, 10e-6)),
        # dummy to reserve space for enum mapping of flow channel
        ("flow_channel_design_code", (0.0, 0.999999)),
    ]

    meta = {
        "flow_channel_designs": ["parallel", "serpentine", "interdigitated"],
        "units": {
            "operating_voltage_V": "V",
            "current_density_A_per_cm2": "A/cm^2",
            "air_flow_rate_slpm": "slpm",
            "fuel_flow_rate_slpm": "slpm",
            "inlet_temp_fuel_K": "K",
            "inlet_temp_air_K": "K",
            "active_area_cm2": "cm^2",
            "thickness_um.*": "um",
            "porosity.*": "1",
            "permeability.*": "m^2",
            "ionic_cond.*": "S/m",
            "electronic_cond.*": "S/m",
            "youngs_modulus_GPa.*": "GPa",
            "cte_per_K.*": "1/K",
        },
    }
    return ranges, meta


def sample_inputs(num_runs: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    ranges, meta = build_param_space()
    H = latin_hypercube_sampling(num_runs, len(ranges), rng)
    samples: List[Dict[str, Any]] = []
    for i in range(num_runs):
        s: Dict[str, Any] = {}
        for j, (name, (lo, hi)) in enumerate(ranges):
            t = float(H[i, j])
            s[name] = map_to_range(t, lo, hi)
        s["flow_channel_design"] = choose_enum(meta["flow_channel_designs"], s.pop("flow_channel_design_code"))
        samples.append(s)
    return samples


def compute_layer_indices(thickness_um: Dict[str, float], nz: int) -> Dict[str, Tuple[int, int]]:
    order = ["anode", "electrolyte", "cathode", "interconnect", "sealant"]
    total_um = sum(thickness_um[layer] for layer in order)
    cum = 0.0
    bounds: Dict[str, Tuple[int, int]] = {}
    z = 0
    for k, layer in enumerate(order):
        frac = thickness_um[layer] / total_um
        dz = nz - z if k == len(order) - 1 else max(1, int(round(frac * nz)))
        z_start = z
        z_end = min(nz, z_start + dz)
        bounds[layer] = (z_start, z_end)
        z = z_end
    # Ensure full coverage
    if bounds[order[-1]][1] < nz:
        bounds[order[-1]] = (bounds[order[-1]][0], nz)
    return bounds


def channel_mask(x: np.ndarray, y: np.ndarray, design: str) -> np.ndarray:
    if design == "parallel":
        # stripes along x (vary in y)
        return 0.5 * (1.0 + np.sign(np.sin(2 * np.pi * 6 * y)))
    if design == "serpentine":
        return 0.5 * (1.0 + np.sign(np.sin(2 * np.pi * (4 * y + 1.5 * x))))
    if design == "interdigitated":
        return 0.5 * (1.0 + np.sign(np.sin(2 * np.pi * (5 * y)) * np.cos(2 * np.pi * (5 * x))))
    return np.zeros_like(x)


def smooth3d(field: np.ndarray, iterations: int = 2) -> np.ndarray:
    f = field.copy()
    for _ in range(iterations):
        # 6-neighbor average (axis-aligned)
        acc = np.zeros_like(f)
        count = np.zeros_like(f)
        for axis in range(3):
            acc += np.roll(f, 1, axis=axis)
            acc += np.roll(f, -1, axis=axis)
            count += 1
            count += 1
        f = 0.5 * f + 0.5 * (acc / np.maximum(count, 1))
    return f


def generate_fields(params: Dict[str, Any], grid: Tuple[int, int, int], seed: int) -> Dict[str, np.ndarray]:
    nx, ny, nz = grid
    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    z = np.linspace(0.0, 1.0, nz, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Layer structure
    thickness_um = {
        "anode": float(params["thickness_um.anode"]),
        "electrolyte": float(params["thickness_um.electrolyte"]),
        "cathode": float(params["thickness_um.cathode"]),
        "interconnect": float(params["thickness_um.interconnect"]),
        "sealant": float(params["thickness_um.sealant"]),
    }
    layer_bounds = compute_layer_indices(thickness_um, nz)

    # Interface z-positions (as indices midpoint across boundaries)
    z_anode_el = layer_bounds["anode"][1] - 0.5
    z_el_cath = layer_bounds["electrolyte"][1] - 0.5
    sigma_z = max(1.0, 0.08 * nz)

    # Operating & material
    j_set_A_per_cm2 = float(params["current_density_A_per_cm2"])  # A/cm^2
    j_set_A_per_m2 = j_set_A_per_cm2 * 1e4

    cond_el_ion = float(params["ionic_cond.electrolyte"])  # S/m
    cond_an_el = float(params["electronic_cond.anode"])    # S/m
    cond_ca_el = float(params["electronic_cond.cathode"])  # S/m

    t_an_m = thickness_um["anode"] * 1e-6
    t_el_m = thickness_um["electrolyte"] * 1e-6
    t_ca_m = thickness_um["cathode"] * 1e-6

    # Ohmic-like aggregate (not unit-accurate; scaled)
    r_ohm = (t_el_m / max(cond_el_ion, 1e-6)) + (t_an_m / max(cond_an_el, 1e-6)) + (t_ca_m / max(cond_ca_el, 1e-6))
    r_ohm_norm = r_ohm / (1e-6)  # scale to O(1)

    # Flow & temperature
    fuel_flow = float(params["fuel_flow_rate_slpm"])  # proxy for depletion
    air_flow = float(params["air_flow_rate_slpm"])    # proxy for cooling/oxygen
    Tin_fuel = float(params["inlet_temp_fuel_K"])
    Tin_air = float(params["inlet_temp_air_K"])
    Tavg_in = 0.5 * (Tin_fuel + Tin_air)

    # Channel effects
    chan = channel_mask(X, Y, params["flow_channel_design"]).astype(np.float32)
    chan = (chan - chan.min()) / max(1e-6, chan.max() - chan.min())  # normalize 0..1

    # Fuel depletion factor along x
    depletion_k = np.clip(0.3 * j_set_A_per_cm2 / max(fuel_flow, 1e-3), 0.02, 1.5)
    fuel_avail = np.exp(-depletion_k * X)
    oxy_avail = np.exp(-0.2 * X * (j_set_A_per_cm2 / max(air_flow, 1e-3)))

    # Overpotential baseline
    eta0 = np.clip(0.12 + 0.38 * (j_set_A_per_cm2 - 0.2) / (2.0 - 0.2) + 0.06 * r_ohm_norm, 0.05, 0.6)
    eta = eta0 * (1.0 + 0.35 * (1 - fuel_avail) + 0.10 * (chan - 0.5))

    # Current density distribution, concentrated near interfaces
    Gz = np.exp(-0.5 * ((Z * nz - z_anode_el) / sigma_z) ** 2) + np.exp(-0.5 * ((Z * nz - z_el_cath) / sigma_z) ** 2)
    Gz = Gz / (Gz.max() + 1e-6)
    j_field = j_set_A_per_m2 * (0.75 * fuel_avail + 0.25 * oxy_avail) * (0.8 + 0.2 * (chan)) * Gz
    j_field = j_field.astype(np.float32)

    # Temperature field: inlet mix + ohmic heating proxy + convective cooling
    # Heating proxy from j^2 / sigma_equiv (use electrolyte ionic conductivity)
    q_proxy = (j_field ** 2) / max(cond_el_ion, 1e-6)
    q_proxy = q_proxy / (q_proxy.max() + 1e-12)
    T_field = (Tin_fuel * (1 - X) + Tin_air * X) * 0.5 + 120.0 * q_proxy - 40.0 * (chan - 0.5)
    T_field = smooth3d(T_field.astype(np.float32), iterations=2)

    # Species in anode region (simple proxy)
    z_start_an, z_end_an = layer_bounds["anode"]
    an_mask = np.zeros((1, 1, nz), dtype=np.float32)
    an_mask[:, :, z_start_an:z_end_an] = 1.0
    an_mask = an_mask + 0.0 * X  # broadcast to (nx, ny, nz)

    c_h2_in = 1.0
    k_h2 = np.clip(0.8 * j_set_A_per_cm2 / max(fuel_flow, 1e-3), 0.05, 2.0)
    c_h2 = c_h2_in * np.exp(-k_h2 * X) * an_mask
    c_h2 = smooth3d(c_h2.astype(np.float32), iterations=1)

    c_h2o = (c_h2_in - c_h2) * an_mask

    # Strain/stress/displacement (thermal mismatch proxy)
    nu = 0.25  # Poisson approx for all layers

    def layer_fill_scalar(value_map: Dict[str, float]) -> np.ndarray:
        out = np.zeros((nx, ny, nz), dtype=np.float32)
        for layer, (zs, ze) in layer_bounds.items():
            out[:, :, zs:ze] = value_map[layer]
        return out

    cte_map = {
        "anode": float(params["cte_per_K.anode"]),
        "electrolyte": float(params["cte_per_K.electrolyte"]),
        "cathode": float(params["cte_per_K.cathode"]),
        "interconnect": float(params["cte_per_K.interconnect"]),
        "sealant": float(params["cte_per_K.sealant"]),
    }
    E_map = {
        "anode": float(params["youngs_modulus_GPa.anode"]) * 1e9,
        "electrolyte": float(params["youngs_modulus_GPa.electrolyte"]) * 1e9,
        "cathode": float(params["youngs_modulus_GPa.cathode"]) * 1e9,
        "interconnect": float(params["youngs_modulus_GPa.interconnect"]) * 1e9,
        "sealant": float(params["youngs_modulus_GPa.sealant"]) * 1e9,
    }

    CTE = layer_fill_scalar(cte_map)
    E = layer_fill_scalar(E_map)

    Tref = Tavg_in - 50.0
    eps_th = CTE * (T_field - Tref)

    # Constraint factor: more constraint near electrolyte due to stiffness mismatch
    stiffness_norm = E / (E.max() + 1e-6)
    constraint = 0.6 + 0.4 * stiffness_norm

    # Principal strains (proxy), shear ~ 0
    exx = 0.4 * eps_th * constraint
    eyy = 0.4 * eps_th * constraint * (1.0 + 0.05 * np.sin(2 * np.pi * Y))
    ezz = 0.2 * eps_th * constraint

    # Stress using Hooke's law proxy (plane-stress-like scale)
    sigma_x = E * exx / (1 - nu ** 2)
    sigma_y = E * eyy / (1 - nu ** 2)
    sigma_z = E * ezz / (1 - nu ** 2)

    # Von Mises (approx)
    vm = np.sqrt(0.5 * ((sigma_x - sigma_y) ** 2 + (sigma_y - sigma_z) ** 2 + (sigma_z - sigma_x) ** 2))

    # Displacement by integrating strain across z (very rough proxy)
    dz = 1.0 / max(nz - 1, 1)
    uz = np.cumsum(ezz, axis=2) * dz
    ux = np.cumsum(exx, axis=0) * (1.0 / max(nx - 1, 1)) * 0.1
    uy = np.cumsum(eyy, axis=1) * (1.0 / max(ny - 1, 1)) * 0.1

    # Overpotential 3D: distribute eta to active layers and decay away
    eta3d = eta * (0.2 + 0.8 * Gz)

    # Final packing as float32
    out = {
        "current_density_A_per_m2": j_field.astype(np.float32),
        "overpotential_V": eta3d.astype(np.float32),
        "temperature_K": T_field.astype(np.float32),
        "von_mises_stress_Pa": vm.astype(np.float32),
        "strain_tensor": np.stack([exx, eyy, ezz, np.zeros_like(exx), np.zeros_like(exx), np.zeros_like(exx)], axis=-1).astype(np.float32),
        "displacement_m": np.stack([ux, uy, uz], axis=-1).astype(np.float32),
        "species_H2": c_h2.astype(np.float32),
        "species_H2O": c_h2o.astype(np.float32),
        # Coordinates for reference
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "z": z.astype(np.float32),
    }
    return out


def flatten_params(params: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, (float, int, str)):
            flat[k] = v
        else:
            flat[k] = json.dumps(v)
    return flat


def save_run(output_dir: str, run_id: int, fields: Dict[str, np.ndarray], params: Dict[str, Any]) -> Tuple[str, str]:
    run_dir = os.path.join(output_dir, "runs")
    ensure_dir(run_dir)
    base = os.path.join(run_dir, f"run_{run_id:05d}")
    npz_path = base + ".npz"
    json_path = base + ".json"

    # Save arrays
    np.savez_compressed(npz_path, **fields)

    # Save metadata
    meta = {
        "run_id": run_id,
        "flow_channel_design": params["flow_channel_design"],
        "inputs": {k: float(v) if isinstance(v, (float, int)) else v for k, v in params.items() if k != "flow_channel_design"},
        "shapes": {k: list(v.shape) for k, v in fields.items()},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return npz_path, json_path


def write_manifest(output_dir: str, records: List[Dict[str, Any]]) -> str:
    manifest_path = os.path.join(output_dir, "manifest.csv")
    if not records:
        return manifest_path
    # Collect all keys
    keys: List[str] = sorted({k for r in records for k in r.keys()})
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow(r)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic high-fidelity-like SOFC multi-physics dataset (npz + json)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory (will create runs/)")
    parser.add_argument("--runs", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "runs"))

    rng = np.random.default_rng(args.seed)
    samples = sample_inputs(args.runs, rng)

    manifest_records: List[Dict[str, Any]] = []

    for i, params in enumerate(samples):
        fields = generate_fields(params, (args.nx, args.ny, args.nz), seed=args.seed + i)
        npz_path, json_path = save_run(args.output_dir, i, fields, params)
        rec = {
            "run_id": i,
            "npz_path": os.path.relpath(npz_path, args.output_dir),
            "json_path": os.path.relpath(json_path, args.output_dir),
            "flow_channel_design": params["flow_channel_design"],
        }
        rec.update({k: v for k, v in flatten_params(params).items()})
        manifest_records.append(rec)
        print(f"[OK] Generated run {i:05d} -> {npz_path}")

    manifest_path = write_manifest(args.output_dir, manifest_records)
    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
