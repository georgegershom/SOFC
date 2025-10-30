#!/usr/bin/env python3
"""
SOFC High-Fidelity Synthetic Dataset Generator

This script fabricates multi-physics 3D fields for SOFC (single cell or stack) using
Latin Hypercube Sampling (LHS) over operating conditions, material properties, and
geometric parameters. It saves each simulation sample as an HDF5 file and writes a
CSV index for inputs and file paths. Optionally zips the dataset directory.

Note: This is a physics-inspired synthetic generator, not a full CFD/FEA solver.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import h5py
import numpy as np
from tqdm import tqdm


@dataclass
class Grid:
    nx: int
    ny: int
    nz: int

    def coords(self, Lx: float, Ly: float, thickness_m: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(0.0, Lx, self.nx, dtype=np.float64)
        y = np.linspace(0.0, Ly, self.ny, dtype=np.float64)
        z = np.linspace(0.0, thickness_m, self.nz, dtype=np.float64)
        return x, y, z


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def latin_hypercube(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a Latin Hypercube Sampling matrix in [0,1]^d."""
    # Divide [0,1] into n_samples strata for each dimension
    cut = np.linspace(0, 1, n_samples + 1)
    u = rng.random((n_samples, n_dims))
    a = cut[:n_samples]
    b = cut[1:n_samples + 1]
    # Points are uniformly sampled within each stratum
    rdpoints = u * (b - a)[:, None] + a[:, None]
    # Randomly permute within each dimension
    H = np.zeros_like(rdpoints)
    for j in range(n_dims):
        order = rng.permutation(n_samples)
        H[:, j] = rdpoints[order, j]
    return H


def smooth3d(field: np.ndarray, iterations: int = 2, kernel_size: int = 3) -> np.ndarray:
    """Cheap separable smoothing (mean filter) to mimic diffusion-like spreading."""
    assert kernel_size % 2 == 3 % 2
    out = field.copy()
    pad = kernel_size // 2
    for _ in range(iterations):
        # x-direction
        tmp = np.pad(out, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        out = (tmp[:-2, :, :] + tmp[1:-1, :, :] + tmp[2:, :, :]) / 3.0
        # y-direction
        tmp = np.pad(out, ((0, 0), (pad, pad), (0, 0)), mode="edge")
        out = (tmp[:, :-2, :] + tmp[:, 1:-1, :] + tmp[:, 2:, :]) / 3.0
        # z-direction
        tmp = np.pad(out, ((0, 0), (0, 0), (pad, pad)), mode="edge")
        out = (tmp[:, :, :-2] + tmp[:, :, 1:-1] + tmp[:, :, 2:]) / 3.0
    return out


def finite_gradient(u: np.ndarray, spacing: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute gradients along x,y,z using central differences (numpy.gradient)."""
    return np.gradient(u, *spacing, edge_order=2)


def build_parameter_space() -> List[Tuple[str, float, float]]:
    params: List[Tuple[str, float, float]] = []

    # Operating conditions
    params += [
        ("oper.voltage_V", 0.6, 1.1),
        ("oper.current_density_A_per_cm2", 0.1, 2.0),
        ("oper.air_flow_slpm", 1.0, 20.0),
        ("oper.fuel_flow_slpm", 0.5, 10.0),
        ("oper.air_inlet_temp_C", 600.0, 900.0),
        ("oper.fuel_inlet_temp_C", 600.0, 900.0),
    ]

    # Geometric parameters
    params += [
        ("geom.active_area_cm2", 4.0, 100.0),  # assume square active area
        ("geom.channel_aspect", 0.2, 2.0),
    ]

    # Layer thicknesses (meters)
    params += [
        ("thickness.anode_um", 200.0, 1000.0),
        ("thickness.electrolyte_um", 5.0, 50.0),
        ("thickness.cathode_um", 10.0, 100.0),
        ("thickness.interconnect_um", 1000.0, 3000.0),  # 1-3 mm
        ("thickness.sealant_um", 100.0, 1000.0),
    ]

    # Material properties (per layer)
    layers = ["anode", "electrolyte", "cathode", "interconnect", "sealant"]
    for layer in layers:
        # Some properties may not strictly apply to all layers, but kept for generality
        params += [
            (f"mat.{layer}.porosity", 0.05, 0.5),
            (f"mat.{layer}.permeability_m2", 1e-16, 1e-12),
            (f"mat.{layer}.ionic_cond_S_per_m", 0.1 if layer != "electrolyte" else 1.0, 10.0),
            (f"mat.{layer}.electronic_cond_S_per_m", 10.0, 1e5),
            (f"mat.{layer}.youngs_modulus_GPa", 50.0, 220.0),
            (f"mat.{layer}.cte_per_C", 8e-6, 14e-6),
        ]

    return params


def vector_to_params(sample_vec: np.ndarray, space: List[Tuple[str, float, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for val, (name, lo, hi) in zip(sample_vec, space):
        out[name] = lo + val * (hi - lo)
    return out


def compute_effective_properties(p: Dict[str, float]) -> Dict[str, float]:
    """Aggregate key effective properties for simple models."""
    # Convert thicknesses to meters
    t_an = p["thickness.anode_um"] * 1e-6
    t_el = p["thickness.electrolyte_um"] * 1e-6
    t_ca = p["thickness.cathode_um"] * 1e-6
    t_ic = p["thickness.interconnect_um"] * 1e-6
    t_se = p["thickness.sealant_um"] * 1e-6

    # Crude effective conductivity estimations (series model)
    # Use ionic for electrolyte path, electronic otherwise; mix by porosity
    def layer_resistivity(layer: str, thickness: float) -> float:
        ionic = p[f"mat.{layer}.ionic_cond_S_per_m"]
        elec = p[f"mat.{layer}.electronic_cond_S_per_m"]
        por = p[f"mat.{layer}.porosity"]
        cond = 0.5 * ((1 - por) * elec + por * ionic)
        cond = max(cond, 1e-6)
        return thickness / cond

    r_eff = (
        layer_resistivity("anode", t_an)
        + layer_resistivity("electrolyte", t_el)
        + layer_resistivity("cathode", t_ca)
        + layer_resistivity("interconnect", t_ic)
        + layer_resistivity("sealant", t_se)
    )

    # Effective CTE and modulus (Voigt average)
    def layer_cte(layer: str) -> float:
        return p[f"mat.{layer}.cte_per_C"]

    def layer_E(layer: str) -> float:
        return p[f"mat.{layer}.youngs_modulus_GPa"] * 1e9

    thicknesses = np.array([t_an, t_el, t_ca, t_ic, t_se])
    weights = thicknesses / np.sum(thicknesses)
    ctes = np.array([layer_cte(l) for l in ["anode", "electrolyte", "cathode", "interconnect", "sealant"]])
    Es = np.array([layer_E(l) for l in ["anode", "electrolyte", "cathode", "interconnect", "sealant"]])

    alpha_eff = float(np.sum(weights * ctes))
    E_eff = float(np.sum(weights * Es))
    nu = 0.28  # Poisson's ratio (assumed)

    return {
        "r_eff": r_eff,
        "alpha_eff": alpha_eff,
        "E_eff": E_eff,
        "nu": nu,
        "total_thickness_m": float(np.sum(thicknesses)),
    }


def synthesize_fields(
    grid: Grid,
    params: Dict[str, float],
    eff: Dict[str, float],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Build synthetic 3D fields with multi-physics couplings."""
    area_cm2 = params["geom.active_area_cm2"]
    area_m2 = area_cm2 * 1e-4
    Lxy = math.sqrt(area_m2)
    Lx = Lxy
    Ly = Lxy
    thickness_m = eff["total_thickness_m"]

    x, y, z = grid.coords(Lx=Lx, Ly=Ly, thickness_m=thickness_m)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")  # shapes (nx, ny, nz)

    # Operating conditions and derived quantities
    j_set_Acm2 = params["oper.current_density_A_per_cm2"]
    voltage = params["oper.voltage_V"]
    air_flow = params["oper.air_flow_slpm"]
    fuel_flow = params["oper.fuel_flow_slpm"]
    T_air_in = params["oper.air_inlet_temp_C"] + 273.15
    T_fuel_in = params["oper.fuel_inlet_temp_C"] + 273.15

    # Base current density (favor set current density; softly influenced by voltage and R_eff)
    r_eff = eff["r_eff"]
    j0 = j_set_Acm2  # A/cm^2
    j0 *= float(np.clip(1.0 + 0.15 * (1.0 - (voltage - 0.6) / 0.5), 0.7, 1.3))

    # Fuel utilization proxy (higher with higher j and lower flow)
    util = float(np.clip(0.05 + 0.35 * (j0 / 1.5) * (5.0 / max(fuel_flow, 1e-3)), 0.02, 0.85))

    # Channel effect across y and reaction zone decay in z
    b = float(np.clip(0.15 * params["geom.channel_aspect"], 0.05, 0.45))
    z_scale = 0.3 * thickness_m

    # Current density distribution j(x,y,z) in A/cm^2
    j = j0 * (1.0 - util * (X / max(Lx, 1e-9))) * (1.0 + b * np.sin(2 * np.pi * Y / max(Ly, 1e-9)) * np.exp(-Z / max(z_scale, 1e-9)))
    j = np.clip(j, 0.02 * j0, None)

    # Overpotential via Tafel-like relation
    T_ref = 800.0 + 273.15
    T_avg_in = 0.5 * (T_air_in + T_fuel_in)
    A = 0.06 * (T_ref / max(T_avg_in, 1.0))  # heuristic scaling
    j_ref = 0.1  # A/cm^2
    eta = A * np.log1p(np.maximum(j / j_ref, 1e-6))  # V

    # Heat generation density proxy Q ~ j*eta (scaled)
    Q = j * eta  # arbitrary units
    Q = smooth3d(Q, iterations=2, kernel_size=3)

    # Temperature field: inlet mixing + heating with diffusion-like smoothing
    T0 = 0.6 * T_air_in + 0.4 * T_fuel_in
    T = T0 + 120.0 * smooth3d(Q / (np.mean(Q) + 1e-9), iterations=2, kernel_size=3)

    # Mechanical: thermal strain and displacement
    alpha_eff = eff["alpha_eff"]
    E_eff = eff["E_eff"]
    nu = eff["nu"]

    thermal_strain = alpha_eff * (T - T_ref)
    # Simple displacement field dominated by out-of-plane expansion
    u_z = thermal_strain * Z
    u_x = 0.05 * thermal_strain * X
    u_y = 0.05 * thermal_strain * Y

    # Strains from displacement gradients
    dx = x[1] - x[0] if grid.nx > 1 else 1.0
    dy = y[1] - y[0] if grid.ny > 1 else 1.0
    dz = z[1] - z[0] if grid.nz > 1 else 1.0

    dux_dx, dux_dy, dux_dz = finite_gradient(u_x, (dx, dy, dz))
    duy_dx, duy_dy, duy_dz = finite_gradient(u_y, (dx, dy, dz))
    duz_dx, duz_dy, duz_dz = finite_gradient(u_z, (dx, dy, dz))

    # Small strain tensor components
    exx = dux_dx
    eyy = duy_dy
    ezz = duz_dz
    exy = 0.5 * (dux_dy + duy_dx)
    exz = 0.5 * (dux_dz + duz_dx)
    eyz = 0.5 * (duy_dz + duz_dy)

    # Isotropic linear elastic stress (Lamé parameters)
    lam = (E_eff * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E_eff / (2 * (1 + nu))
    tr = exx + eyy + ezz
    s_xx = lam * tr + 2 * mu * exx
    s_yy = lam * tr + 2 * mu * eyy
    s_zz = lam * tr + 2 * mu * ezz
    s_xy = 2 * mu * exy
    s_xz = 2 * mu * exz
    s_yz = 2 * mu * eyz

    # Von Mises stress (3D)
    # sigma_vm = sqrt(0.5*((s_xx-s_yy)^2+(s_yy-s_zz)^2+(s_zz-s_xx)^2 + 6*(s_xy^2+s_yz^2+s_xz^2)))
    sigma_vm = np.sqrt(
        0.5
        * (
            (s_xx - s_yy) ** 2
            + (s_yy - s_zz) ** 2
            + (s_zz - s_xx) ** 2
            + 6.0 * (s_xy**2 + s_yz**2 + s_xz**2)
        )
    )

    # Species: H2 decays along x, H2O increases, modulated by channels and z diffusion
    k_util = 1.5 * util
    g_y = 0.5 * (1.0 + np.sin(2 * np.pi * Y / max(Ly, 1e-9)))
    g_z = np.exp(-Z / max(z_scale, 1e-9))
    c_H2_in = 1.0  # normalized
    c_H2 = c_H2_in * np.exp(-k_util * (X / max(Lx, 1e-9))) * (0.7 + 0.3 * g_y) * (0.6 + 0.4 * g_z)
    c_H2O = np.clip(1.2 - c_H2, 0.0, None)

    # Package displacement as vector and strain in Voigt notation
    displacement = np.stack([u_x, u_y, u_z], axis=-1)  # (nx,ny,nz,3)
    strain_voigt = np.stack([exx, eyy, ezz, exy, exz, eyz], axis=-1)

    fields = {
        "current_density_A_per_cm2": j.astype(np.float32),
        "overpotential_V": eta.astype(np.float32),
        "temperature_K": T.astype(np.float32),
        "von_mises_stress_Pa": sigma_vm.astype(np.float32),
        "displacement_m": displacement.astype(np.float32),
        "strain_tensor_voigt": strain_voigt.astype(np.float32),
        "species_H2": c_H2.astype(np.float32),
        "species_H2O": c_H2O.astype(np.float32),
    }

    return fields


def write_h5(path: str, fields: Dict[str, np.ndarray], params: Dict[str, float]) -> None:
    with h5py.File(path, "w") as f:
        for k, v in fields.items():
            dset = f.create_dataset(k, data=v, compression="gzip")
            dset.attrs["units"] = {
                "current_density_A_per_cm2": "A/cm^2",
                "overpotential_V": "V",
                "temperature_K": "K",
                "von_mises_stress_Pa": "Pa",
                "displacement_m": "m",
                "strain_tensor_voigt": "-",
                "species_H2": "normalized",
                "species_H2O": "normalized",
            }[k]
        # Store inputs as JSON attribute
        f.attrs["input_parameters_json"] = json.dumps(params)


def main() -> None:
    parser = argparse.ArgumentParser(description="SOFC synthetic high-fidelity dataset generator")
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--grid", type=int, nargs=3, default=[24, 24, 12], metavar=("NX", "NY", "NZ"))
    parser.add_argument("--output-dir", type=str, default="/workspace/data/sofc_high_fidelity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zip", action="store_true", help="Zip the output directory after generation")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    grid = Grid(nx=args.grid[0], ny=args.grid[1], nz=args.grid[2])

    ensure_dir(args.output_dir)
    samples_dir = os.path.join(args.output_dir, "samples")
    ensure_dir(samples_dir)

    # Parameter space and LHS sampling
    space = build_parameter_space()
    H = latin_hypercube(args.n_samples, len(space), rng)

    index_csv_path = os.path.join(args.output_dir, "index.csv")
    header = ["sample_id", "h5_path"] + [name for name, _, _ in space]

    with open(index_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for i in tqdm(range(args.n_samples), desc="Generating samples"):
            vec = H[i]
            params = vector_to_params(vec, space)
            eff = compute_effective_properties(params)

            fields = synthesize_fields(grid, params, eff, rng)

            h5_name = f"sample_{i:05d}.h5"
            h5_path = os.path.join(samples_dir, h5_name)
            write_h5(h5_path, fields, params)

            row = [i, h5_path] + [params[name] for name, _, _ in space]
            writer.writerow(row)

    if args.zip:
        zip_path = os.path.join(args.output_dir + ".zip")
        # Create/overwrite zip
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(args.output_dir):
                for fn in files:
                    # Skip the zip inside itself (if exists in the same dir)
                    src = os.path.join(root, fn)
                    if os.path.abspath(src) == os.path.abspath(zip_path):
                        continue
                    arcname = os.path.relpath(src, os.path.dirname(args.output_dir))
                    zf.write(src, arcname=arcname)
        print(f"Zipped dataset -> {zip_path}")

    print("Done.")


if __name__ == "__main__":
    main()
