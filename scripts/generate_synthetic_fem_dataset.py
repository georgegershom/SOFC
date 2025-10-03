#!/usr/bin/env python3
"""
Synthetic multi-physics FEM-like dataset generator.

This script fabricates inputs and outputs reminiscent of results from
COMSOL/ABAQUS-style multiphysics simulations. It does not solve PDEs;
instead it generates physically inspired fields with plausible units,
scales, and couplings across thermal, mechanical, and electrochemical
domains.

Generated per-sample artifacts:
  - inputs.json: Mesh, boundary conditions, materials, and time info
  - time.csv: Time axis and boundary signal summaries per step
  - fields_timeseries.npz: Packed arrays for fields over time
      Arrays (all shapes: [time_steps, ny, nx] unless noted):
        * temperature_C
        * voltage_V
        * stress_vm_MPa
        * stress_principal1_MPa
        * stress_principal2_MPa
        * shear_tau_xy_MPa
        * strain_elastic_xx
        * strain_elastic_yy
        * strain_thermal_iso  (isotropic thermal strain scalar)
        * strain_plastic_eq
        * strain_creep_eq
        * damage_D           (0..1)
        * interfacial_tau_xy_MPa  (shape: [time_steps, nx])
        * delamination_init_step_by_x (shape: [nx], int, -1 if none)
        * crack_init_step_mask (shape: [time_steps, ny, nx], bool)

At dataset root:
  - dataset.json: Schema, units, shapes, and sample index
  - README.md: Human-readable documentation

Usage:
  python scripts/generate_synthetic_fem_dataset.py \
    --out /workspace/data/num-sim-001 \
    --num-samples 5 --nx 32 --ny 32 --time-steps 60 --seed 42

"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np


# ----------------------------- Data Models ----------------------------- #


@dataclass
class MeshSpec:
    nx: int
    ny: int
    element_type: str
    element_size_x_mm: float
    element_size_y_mm: float
    interface_y_fraction: float
    interface_refinement_level: int


@dataclass
class ThermalBC:
    t_initial_C: float
    t_peak_C: float
    heating_rate_C_per_min: float
    hold_minutes: float
    cooling_rate_C_per_min: float
    deltaT_top_minus_bottom_C: float


@dataclass
class MechanicalBC:
    uy_top_um: float
    ux_left_um: float
    shear_amplitude: float  # dimensionless gamma amplitude baseline


@dataclass
class ElectricalBC:
    v_top_V: float
    v_bottom_V: float


@dataclass
class MaterialLayer:
    name: str
    thickness_fraction: float
    youngs_modulus_GPa: float
    poisson_ratio: float
    thermal_expansion_1_per_K: float
    thermal_conductivity_W_per_mK: float
    heat_capacity_J_per_kgK: float
    density_kg_per_m3: float
    yield_strength_MPa: float
    hardening_GPa: float
    creep_A_per_Pa_pow_n_per_min: float  # A in Norton law per minute
    creep_n: float
    elec_conductivity_S_per_m: float


@dataclass
class MaterialsSpec:
    layers: List[MaterialLayer]
    interface_tau_crit_MPa: float
    damage_coupling_coeff: float
    crack_sigma1_crit_MPa: float


@dataclass
class TimeSpec:
    total_minutes: float
    time_steps: int


@dataclass
class SampleInputs:
    mesh: MeshSpec
    bc_thermal: ThermalBC
    bc_mechanical: MechanicalBC
    bc_electrical: ElectricalBC
    materials: MaterialsSpec
    time: TimeSpec


# ----------------------------- Utilities ------------------------------ #


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def lerp(a: np.ndarray | float, b: np.ndarray | float, w: np.ndarray | float) -> np.ndarray:
    return (1.0 - w) * np.asarray(a) + w * np.asarray(b)


def create_grid(nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create unit-square grid coordinates X in [0,1] across x, Y across y.
    Shapes: (ny, nx).
    """
    xs = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, ny, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def smooth_interface_weight(Y: np.ndarray, interface_y: float, width: float) -> np.ndarray:
    """Compute a smooth step weight ~0 for bottom layer, ~1 for top layer.
    width controls the transition sharpness.
    """
    return 0.5 * (1.0 + np.tanh((Y - interface_y) / max(width, 1e-6)))


def generate_inputs(rng: np.random.Generator, nx: int, ny: int) -> SampleInputs:
    # Mesh
    element_type = rng.choice(["quad4", "quad8", "tri3"])  # keep it simple
    interface_y_fraction = 0.5
    interface_refinement_level = int(rng.integers(1, 4))

    # Assume physical size ~ 32 mm x 32 mm for scale; element size is that / count
    size_x_mm = 32.0
    size_y_mm = 32.0
    element_size_x_mm = size_x_mm / nx
    element_size_y_mm = size_y_mm / ny

    mesh = MeshSpec(
        nx=nx,
        ny=ny,
        element_type=str(element_type),
        element_size_x_mm=float(element_size_x_mm),
        element_size_y_mm=float(element_size_y_mm),
        interface_y_fraction=float(interface_y_fraction),
        interface_refinement_level=int(interface_refinement_level),
    )

    # Thermal BCs (heating/cooling rates 1..10 C/min)
    t_initial_C = float(rng.uniform(15.0, 30.0))
    t_peak_C = float(rng.uniform(80.0, 180.0))
    heating_rate = float(rng.uniform(1.0, 10.0))
    cooling_rate = float(rng.uniform(1.0, 10.0))
    hold_minutes = float(rng.uniform(5.0, 20.0))
    deltaT_top_minus_bottom = float(rng.uniform(5.0, 30.0))
    bc_thermal = ThermalBC(
        t_initial_C=t_initial_C,
        t_peak_C=t_peak_C,
        heating_rate_C_per_min=heating_rate,
        hold_minutes=hold_minutes,
        cooling_rate_C_per_min=cooling_rate,
        deltaT_top_minus_bottom_C=deltaT_top_minus_bottom,
    )

    # Mechanical BCs
    uy_top_um = float(rng.uniform(0.0, 100.0))
    ux_left_um = float(rng.uniform(0.0, 50.0))
    shear_amplitude = float(rng.uniform(2e-4, 2e-3))  # small base shear strain
    bc_mech = MechanicalBC(
        uy_top_um=uy_top_um,
        ux_left_um=ux_left_um,
        shear_amplitude=shear_amplitude,
    )

    # Electrical BCs
    v_top = float(rng.uniform(0.0, 5.0))
    v_bottom = 0.0
    bc_elec = ElectricalBC(v_top_V=v_top, v_bottom_V=v_bottom)

    # Material layers (two-layer system)
    def random_layer(name: str, thickness_fraction: float) -> MaterialLayer:
        E = float(rng.uniform(5.0, 200.0))  # GPa
        nu = float(rng.uniform(0.20, 0.40))
        alpha = float(rng.uniform(5e-6, 25e-6))
        k = float(rng.uniform(0.2, 200.0))
        cp = float(rng.uniform(400.0, 1000.0))
        rho = float(rng.uniform(900.0, 8900.0))
        sig_y = float(rng.uniform(80.0, 800.0))  # MPa
        H = float(rng.uniform(0.1, 5.0))  # GPa
        A = 10 ** float(rng.uniform(-24.0, -16.0))  # per Pa^n per min
        n = float(rng.uniform(3.0, 7.0))
        sigma_e = float(rng.uniform(1e3, 5e6))
        return MaterialLayer(
            name=name,
            thickness_fraction=thickness_fraction,
            youngs_modulus_GPa=E,
            poisson_ratio=nu,
            thermal_expansion_1_per_K=alpha,
            thermal_conductivity_W_per_mK=k,
            heat_capacity_J_per_kgK=cp,
            density_kg_per_m3=rho,
            yield_strength_MPa=sig_y,
            hardening_GPa=H,
            creep_A_per_Pa_pow_n_per_min=A,
            creep_n=n,
            elec_conductivity_S_per_m=sigma_e,
        )

    layer1 = random_layer("bottom_layer", thickness_fraction=0.5)
    layer2 = random_layer("top_layer", thickness_fraction=0.5)
    interface_tau_crit_MPa = float(rng.uniform(5.0, 40.0))
    damage_coupling_coeff = float(rng.uniform(0.5, 3.0))
    crack_sigma1_crit_MPa = float(rng.uniform(150.0, 900.0))
    materials = MaterialsSpec(
        layers=[layer1, layer2],
        interface_tau_crit_MPa=interface_tau_crit_MPa,
        damage_coupling_coeff=damage_coupling_coeff,
        crack_sigma1_crit_MPa=crack_sigma1_crit_MPa,
    )

    # Time discretization
    heating_minutes = (t_peak_C - t_initial_C) / max(heating_rate, 1e-6)
    cooling_minutes = (t_peak_C - t_initial_C) / max(cooling_rate, 1e-6)
    total_minutes = heating_minutes + hold_minutes + cooling_minutes
    time_steps = int(rng.integers(40, 90))
    time = TimeSpec(total_minutes=float(total_minutes), time_steps=time_steps)

    return SampleInputs(
        mesh=mesh,
        bc_thermal=bc_thermal,
        bc_mechanical=bc_mech,
        bc_electrical=bc_elec,
        materials=materials,
        time=time,
    )


def build_time_axis(inputs: SampleInputs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_min, T_top, T_bottom).
    Heating to peak, hold, then cooling to initial.
    """
    t_total = inputs.time.total_minutes
    n = inputs.time.time_steps
    t = np.linspace(0.0, t_total, n, dtype=np.float64)

    ti = inputs.bc_thermal.t_initial_C
    tp = inputs.bc_thermal.t_peak_C
    rate_up = inputs.bc_thermal.heating_rate_C_per_min
    rate_down = inputs.bc_thermal.cooling_rate_C_per_min
    hold = inputs.bc_thermal.hold_minutes

    t_heat = (tp - ti) / max(rate_up, 1e-6)
    t_cool = (tp - ti) / max(rate_down, 1e-6)
    t_hold_start = t_heat
    t_hold_end = t_heat + hold
    t_end = t_total

    T_top = np.empty_like(t)
    for i, ti_min in enumerate(t):
        if ti_min <= t_hold_start:
            T_top[i] = ti + rate_up * ti_min
        elif ti_min <= t_hold_end:
            T_top[i] = tp
        else:
            T_top[i] = tp - rate_down * (ti_min - t_hold_end)
    T_bottom = T_top - inputs.bc_thermal.deltaT_top_minus_bottom_C

    return t, T_top, T_bottom


def assemble_material_maps(inputs: SampleInputs, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return maps over the grid for E(GPa), nu, alpha(1/K), yield(MPa), H(GPa), creep (A, n).
    For simplicity, the layer transition is sharp at interface_y_fraction.
    """
    yif = inputs.mesh.interface_y_fraction
    layer_bottom, layer_top = inputs.materials.layers

    mask_top = (Y >= yif)
    mask_bottom = ~mask_top

    E = np.empty_like(Y)
    nu = np.empty_like(Y)
    alpha = np.empty_like(Y)
    yield_MPa = np.empty_like(Y)
    H_GPa = np.empty_like(Y)
    creep_A = np.empty_like(Y)
    creep_n = np.empty_like(Y)

    E[mask_bottom] = layer_bottom.youngs_modulus_GPa
    E[mask_top] = layer_top.youngs_modulus_GPa
    nu[mask_bottom] = layer_bottom.poisson_ratio
    nu[mask_top] = layer_top.poisson_ratio
    alpha[mask_bottom] = layer_bottom.thermal_expansion_1_per_K
    alpha[mask_top] = layer_top.thermal_expansion_1_per_K
    yield_MPa[mask_bottom] = layer_bottom.yield_strength_MPa
    yield_MPa[mask_top] = layer_top.yield_strength_MPa
    H_GPa[mask_bottom] = layer_bottom.hardening_GPa
    H_GPa[mask_top] = layer_top.hardening_GPa
    creep_A[mask_bottom] = layer_bottom.creep_A_per_Pa_pow_n_per_min
    creep_A[mask_top] = layer_top.creep_A_per_Pa_pow_n_per_min
    creep_n[mask_bottom] = layer_bottom.creep_n
    creep_n[mask_top] = layer_top.creep_n

    return E, nu, alpha, yield_MPa, H_GPa, creep_A, creep_n


def compute_fields_for_sample(rng: np.random.Generator, inputs: SampleInputs, out_dir: Path) -> None:
    nx, ny = inputs.mesh.nx, inputs.mesh.ny
    X, Y = create_grid(nx, ny)

    t_min, T_top, T_bottom = build_time_axis(inputs)
    nsteps = t_min.shape[0]

    # Spatial noise fields (static over time) for heterogeneity
    spatial_noise_T = (rng.normal(0.0, 0.5, size=(ny, nx))).astype(np.float64)
    spatial_noise_V = (rng.normal(0.0, 0.02, size=(ny, nx))).astype(np.float64)
    spatial_noise_gamma = (rng.normal(0.0, 2e-5, size=(ny, nx))).astype(np.float64)

    # Material maps
    E_GPa_map, nu_map, alpha_map, yield_MPa_map, H_GPa_map, creep_A_map, creep_n_map = assemble_material_maps(inputs, Y)

    # Pre-allocate arrays
    shape = (nsteps, ny, nx)
    temperature_C = np.zeros(shape, dtype=np.float64)
    voltage_V = np.zeros(shape, dtype=np.float64)
    stress_vm_MPa = np.zeros(shape, dtype=np.float64)
    stress_p1_MPa = np.zeros(shape, dtype=np.float64)
    stress_p2_MPa = np.zeros(shape, dtype=np.float64)
    shear_tau_xy_MPa = np.zeros(shape, dtype=np.float64)
    strain_el_xx = np.zeros(shape, dtype=np.float64)
    strain_el_yy = np.zeros(shape, dtype=np.float64)
    strain_th_iso = np.zeros(shape, dtype=np.float64)
    strain_pl_eq = np.zeros(shape, dtype=np.float64)
    strain_cr_eq = np.zeros(shape, dtype=np.float64)
    damage_D = np.zeros(shape, dtype=np.float64)
    crack_init_mask = np.zeros(shape, dtype=bool)

    interfacial_tau_xy_MPa = np.zeros((nsteps, nx), dtype=np.float64)

    # Interface row index
    yif = inputs.mesh.interface_y_fraction
    y_idx_if = int(np.clip(round(yif * (ny - 1)), 0, ny - 1))

    # Constants and conversions
    E_Pa_map = E_GPa_map * 1e9
    H_Pa_map = H_GPa_map * 1e9
    G_Pa_map = E_Pa_map / (2.0 * (1.0 + nu_map))

    # Mechanical BCs converted to strain-like quantities
    # Assume thickness 1 mm -> 1e-3 m; uniform epsilon_yy from uy_top
    thickness_m = 1e-3
    epsilon_yy_mech = (inputs.bc_mechanical.uy_top_um * 1e-6) / thickness_m
    epsilon_xx_mech = -nu_map * epsilon_yy_mech

    gamma_xy_base = inputs.bc_mechanical.shear_amplitude
    gamma_xy_shape = np.sin(np.pi * X) * np.exp(-((Y - yif) ** 2) / (2.0 * (0.07 ** 2)))
    gamma_xy_field = gamma_xy_base * gamma_xy_shape + spatial_noise_gamma

    # Electrical potential (linear in y with small heterogeneity)
    v_top = inputs.bc_electrical.v_top_V
    v_bottom = inputs.bc_electrical.v_bottom_V

    # Plastic and creep accumulators
    eq_plastic_prev = np.zeros((ny, nx), dtype=np.float64)
    eq_creep_prev = np.zeros((ny, nx), dtype=np.float64)
    damage_prev = np.zeros((ny, nx), dtype=np.float64)

    # Time stepping
    dt_min = float(inputs.time.total_minutes / max(inputs.time.time_steps - 1, 1))
    for k in range(nsteps):
        T_top_k = T_top[k]
        T_bot_k = T_bottom[k]

        # Temperature field (linear in y + heterogeneous noise)
        T_field = lerp(T_bot_k, T_top_k, Y) + spatial_noise_T
        temperature_C[k] = T_field

        # Voltage field (linear in y + small heterogeneity + x-shape)
        V_field = lerp(v_bottom, v_top, Y) + 0.03 * (np.sin(2 * np.pi * X) * (v_top - v_bottom)) + spatial_noise_V
        voltage_V[k] = V_field

        # Thermal strain (isotropic scalar) per cell
        epsilon_th = alpha_map * (T_field - inputs.bc_thermal.t_initial_C)
        strain_th_iso[k] = epsilon_th

        # Elastic mechanical strain components (base from BC)
        # Subtract thermal part when computing stress
        epsilon_xx_total = epsilon_xx_mech + epsilon_th
        epsilon_yy_total = epsilon_yy_mech + epsilon_th
        gamma_xy_total = gamma_xy_field

        # Stress under plane stress: sigma = C:(epsilon - epsilon_th)
        # Using E, nu maps per cell
        # sigma_xx = E/(1-nu^2) * [(eps_xx - eps_th) + nu*(eps_yy - eps_th)]
        # sigma_yy = E/(1-nu^2) * [(eps_yy - eps_th) + nu*(eps_xx - eps_th)]
        # tau_xy = G * gamma_xy
        denom = (1.0 - np.square(nu_map))
        term_x = (epsilon_xx_total - epsilon_th) + nu_map * (epsilon_yy_total - epsilon_th)
        term_y = (epsilon_yy_total - epsilon_th) + nu_map * (epsilon_xx_total - epsilon_th)
        sigma_xx_Pa = (E_Pa_map / denom) * term_x
        sigma_yy_Pa = (E_Pa_map / denom) * term_y
        tau_xy_Pa = G_Pa_map * gamma_xy_total

        # Von Mises for plane stress: sqrt(sx^2 - sx*sy + sy^2 + 3*tau^2)
        sx = sigma_xx_Pa
        sy = sigma_yy_Pa
        txy = tau_xy_Pa
        vm_Pa = np.sqrt(np.maximum(sx * sx - sx * sy + sy * sy + 3.0 * (txy * txy), 0.0))

        # Principal stresses
        avg = 0.5 * (sx + sy)
        R = np.sqrt(np.maximum(np.square(0.5 * (sx - sy)) + np.square(txy), 0.0))
        s1_Pa = avg + R
        s2_Pa = avg - R

        # Convert to MPa for storage
        stress_vm_MPa[k] = vm_Pa / 1e6
        stress_p1_MPa[k] = s1_Pa / 1e6
        stress_p2_MPa[k] = s2_Pa / 1e6
        shear_tau_xy_MPa[k] = tau_xy_Pa / 1e6

        # Elastic strain approximation: Take total minus thermal minus plastic/creep increments (lagged)
        # For simplicity, distribute plastic/creep equally to xx, yy when present
        strain_el_xx[k] = epsilon_xx_total
        strain_el_yy[k] = epsilon_yy_total

        # Plastic update (simple isotropic model driven by von Mises)
        yield_MPa = yield_MPa_map
        H_Pa = H_Pa_map
        over_yield_MPa = np.maximum(stress_vm_MPa[k] - yield_MPa, 0.0)
        plastic_rate = over_yield_MPa / (np.maximum(H_Pa, 1e6) / 1e6)  # dimensionless per minute
        plastic_inc = 5e-5 * plastic_rate * dt_min  # tuned scale
        eq_plastic = eq_plastic_prev + plastic_inc
        strain_pl_eq[k] = eq_plastic

        # Creep update: Norton law epsilon_dot = A * sigma^n (sigma in Pa), A per min
        A = creep_A_map
        n_exp = creep_n_map
        creep_rate = A * np.power(np.maximum(vm_Pa, 0.0), n_exp)
        creep_inc = creep_rate * dt_min
        eq_creep = eq_creep_prev + creep_inc
        strain_cr_eq[k] = eq_creep

        # Damage update: D_dot ~ coeff * (plastic_rate + creep_rate_normalized)
        coeff = inputs.materials.damage_coupling_coeff
        creep_rate_norm = creep_rate / (np.maximum(np.max(creep_rate), 1e-30))
        D_inc = coeff * (plastic_inc + 0.2 * creep_rate_norm * dt_min)
        D_new = np.clip(damage_prev + D_inc, 0.0, 1.0)
        damage_D[k] = D_new

        # Crack initiation where sigma1 exceeds critical and damage is high
        crack_mask = (stress_p1_MPa[k] >= inputs.materials.crack_sigma1_crit_MPa * (1.0 - 0.5 * D_new)) & (D_new > 0.2)
        crack_init_mask[k] = crack_mask

        # Interfacial shear record
        interfacial_tau_xy_MPa[k] = shear_tau_xy_MPa[k, y_idx_if]

        # Prepare for next step accumulation
        eq_plastic_prev = eq_plastic
        eq_creep_prev = eq_creep
        damage_prev = D_new

    # Delamination prediction along interface
    tau_if = interfacial_tau_xy_MPa  # [t, nx]
    tau_crit = inputs.materials.interface_tau_crit_MPa
    D_if = damage_D[:, y_idx_if, :]  # [t, nx]
    delam_threshold = tau_crit * (1.0 - 0.5 * np.clip(D_if, 0.0, 1.0))
    exceed = tau_if >= delam_threshold
    init_step_by_x = np.argmax(exceed, axis=0)
    has_event = exceed.any(axis=0)
    init_step_by_x = np.where(has_event, init_step_by_x, -1)

    # Save time summary CSV
    time_csv_path = out_dir / "time.csv"
    with time_csv_path.open("w", encoding="utf-8") as f:
        f.write("step,t_min,T_top_C,T_bottom_C,V_top_V,V_bottom_V\n")
        for i in range(nsteps):
            f.write(f"{i},{t_min[i]:.6f},{T_top[i]:.6f},{T_bottom[i]:.6f},{v_top:.6f},{v_bottom:.6f}\n")

    # Save arrays as NPZ
    np.savez_compressed(
        out_dir / "fields_timeseries.npz",
        temperature_C=temperature_C,
        voltage_V=voltage_V,
        stress_vm_MPa=stress_vm_MPa,
        stress_principal1_MPa=stress_p1_MPa,
        stress_principal2_MPa=stress_p2_MPa,
        shear_tau_xy_MPa=shear_tau_xy_MPa,
        strain_elastic_xx=strain_el_xx,
        strain_elastic_yy=strain_el_yy,
        strain_thermal_iso=strain_th_iso,
        strain_plastic_eq=strain_pl_eq,
        strain_creep_eq=strain_cr_eq,
        damage_D=damage_D,
        interfacial_tau_xy_MPa=interfacial_tau_xy_MPa,
        delamination_init_step_by_x=init_step_by_x,
        crack_init_step_mask=crack_init_mask,
    )


def write_sample_inputs(inputs: SampleInputs, out_dir: Path) -> None:
    obj = {
        "mesh": asdict(inputs.mesh),
        "boundary_conditions": {
            "thermal": asdict(inputs.bc_thermal),
            "mechanical": asdict(inputs.bc_mechanical),
            "electrical": asdict(inputs.bc_electrical),
        },
        "materials": {
            "layers": [asdict(layer) for layer in inputs.materials.layers],
            "interface_tau_crit_MPa": inputs.materials.interface_tau_crit_MPa,
            "damage_coupling_coeff": inputs.materials.damage_coupling_coeff,
            "crack_sigma1_crit_MPa": inputs.materials.crack_sigma1_crit_MPa,
        },
        "time": asdict(inputs.time),
        "units": {
            "length": "mm",
            "temperature": "C",
            "stress": "MPa",
            "voltage": "V",
            "strain": "dimensionless",
            "time": "minutes",
        },
        "notes": "Synthetic data; not from a PDE solver.",
    }
    with (out_dir / "inputs.json").open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_dataset_docs(root_out: Path, samples: List[str]) -> None:
    # dataset.json
    dataset_json = {
        "name": "synthetic_multiphysics_fem_dataset",
        "version": 1,
        "description": "Fabricated multi-physics FEM-like dataset (thermal-mech-electro) with fields over time.",
        "root": str(root_out),
        "num_samples": len(samples),
        "samples": samples,
        "artifacts_per_sample": [
            "inputs.json",
            "time.csv",
            "fields_timeseries.npz",
        ],
        "fields_timeseries_keys": [
            "temperature_C",
            "voltage_V",
            "stress_vm_MPa",
            "stress_principal1_MPa",
            "stress_principal2_MPa",
            "shear_tau_xy_MPa",
            "strain_elastic_xx",
            "strain_elastic_yy",
            "strain_thermal_iso",
            "strain_plastic_eq",
            "strain_creep_eq",
            "damage_D",
            "interfacial_tau_xy_MPa",
            "delamination_init_step_by_x",
            "crack_init_step_mask",
        ],
        "units": {
            "temperature_C": "C",
            "voltage_V": "V",
            "stress_vm_MPa": "MPa",
            "stress_principal1_MPa": "MPa",
            "stress_principal2_MPa": "MPa",
            "shear_tau_xy_MPa": "MPa",
            "strain_elastic_xx": "dimensionless",
            "strain_elastic_yy": "dimensionless",
            "strain_thermal_iso": "dimensionless",
            "strain_plastic_eq": "dimensionless",
            "strain_creep_eq": "dimensionless",
            "damage_D": "0..1",
            "interfacial_tau_xy_MPa": "MPa",
            "delamination_init_step_by_x": "time step index; -1 if never",
            "crack_init_step_mask": "bool",
        },
    }
    with (root_out / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)

    # README.md
    readme = f"""
Synthetic Multi-Physics FEM-Like Dataset
=======================================

This dataset contains fabricated, physically inspired time-series fields and
input descriptors resembling outputs from multi-physics FEM solvers
(thermal, mechanical, electrochemical). No PDEs were solved. Values and
units are plausible but synthetic.

Directory Layout
----------------

root: {root_out}

- dataset.json: Machine-readable schema and units
- README.md: This file
- samples/
  - sample_XXX/
    - inputs.json: Mesh, BCs, materials, time, units
    - time.csv: per-step summary (t_min, T_top, T_bottom, V_top, V_bottom)
    - fields_timeseries.npz: compressed arrays

Key Arrays in fields_timeseries.npz
-----------------------------------

- temperature_C [t, y, x] (C)
- voltage_V [t, y, x] (V)
- stress_vm_MPa [t, y, x] (MPa)
- stress_principal1_MPa [t, y, x] (MPa)
- stress_principal2_MPa [t, y, x] (MPa)
- shear_tau_xy_MPa [t, y, x] (MPa)
- strain_elastic_xx [t, y, x] (-)
- strain_elastic_yy [t, y, x] (-)
- strain_thermal_iso [t, y, x] (-)  isotropic thermal strain
- strain_plastic_eq [t, y, x] (-)  equivalent plastic strain
- strain_creep_eq [t, y, x] (-)   equivalent creep strain
- damage_D [t, y, x] (0..1)
- interfacial_tau_xy_MPa [t, x] (MPa) shear at the material interface
- delamination_init_step_by_x [x] (int) first step where delamination predicted; -1 if none
- crack_init_step_mask [t, y, x] (bool) crack initiation prediction mask

Notes
-----
- Interface at y = inputs.json:mesh.interface_y_fraction (default 0.5).
- Delamination is predicted when interfacial shear exceeds a threshold reduced by damage.
- Crack initiation is predicted when principal stress exceeds a threshold reduced by damage.

How to Load with NumPy
----------------------

"""
    readme += """
```python
import numpy as np
import json
from pathlib import Path

root = Path("." ) / "samples" / "sample_000"
with open(root / "inputs.json") as f:
    meta = json.load(f)
npz = np.load(root / "fields_timeseries.npz")
print(npz["temperature_C"].shape)
```
"""
    with (root_out / "README.md").open("w", encoding="utf-8") as f:
        f.write(readme)


def generate_dataset(out_dir: Path, num_samples: int, nx: int, ny: int, seed: int | None) -> None:
    ensure_dir(out_dir)
    samples_dir = out_dir / "samples"
    ensure_dir(samples_dir)

    rng = np.random.default_rng(seed)

    sample_names: List[str] = []
    for i in range(num_samples):
        sample_name = f"sample_{i:03d}"
        sample_names.append(sample_name)
        sample_dir = samples_dir / sample_name
        ensure_dir(sample_dir)

        # Per-sample RNG (deterministic w.r.t. seed and index)
        sample_rng = np.random.default_rng(None if seed is None else seed + i)

        # Inputs
        inputs = generate_inputs(sample_rng, nx=nx, ny=ny)
        write_sample_inputs(inputs, sample_dir)

        # Fields
        compute_fields_for_sample(sample_rng, inputs, sample_dir)

    # Top-level docs
    write_dataset_docs(out_dir, sample_names)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic multi-physics FEM-like dataset generator")
    p.add_argument("--out", dest="out", type=str, default="/workspace/data/num-sim-001", help="Output dataset directory")
    p.add_argument("--num-samples", dest="num_samples", type=int, default=5, help="Number of samples to generate")
    p.add_argument("--nx", dest="nx", type=int, default=32, help="Grid cells in x")
    p.add_argument("--ny", dest="ny", type=int, default=32, help="Grid cells in y")
    p.add_argument("--time-steps", dest="time_steps", type=int, default=None, help="Override time steps (else random 40-90)")
    p.add_argument("--seed", dest="seed", type=int, default=42, help="RNG seed (None for random)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    # If time_steps override requested, we will monkey patch in generate_inputs via closure
    time_steps_override = args.time_steps

    # Wrap generate_inputs to accept time_steps override
    original_generate_inputs = generate_inputs

    def generate_inputs_override(rng: np.random.Generator, nx: int, ny: int) -> SampleInputs:
        inputs = original_generate_inputs(rng, nx, ny)
        if time_steps_override is not None:
            inputs.time.time_steps = int(time_steps_override)
        return inputs

    # Monkey patch within this scope
    globals()["generate_inputs"] = generate_inputs_override

    generate_dataset(out_dir=out_dir, num_samples=args.num_samples, nx=args.nx, ny=args.ny, seed=None if args.seed == -1 else args.seed)


if __name__ == "__main__":
    main()

