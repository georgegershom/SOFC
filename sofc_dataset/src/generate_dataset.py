#!/usr/bin/env python3
import os
import json
import math
import time
import argparse
import random
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import h5py
import yaml
from scipy.stats import qmc
from scipy.signal import welch
from tqdm import tqdm


@dataclass
class GridSpec:
    nx: int
    ny: int
    nz: int


@dataclass
class ParamRanges:
    current_density: Tuple[float, float]
    fuel_utilization: Tuple[float, float]
    air_utilization: Tuple[float, float]
    inlet_fuel_temp_c: Tuple[float, float]
    inlet_air_temp_c: Tuple[float, float]
    fuel_composition_h2: Tuple[float, float]
    fuel_composition_h2o: Tuple[float, float]
    fuel_composition_co: Tuple[float, float]
    fuel_composition_ch4: Tuple[float, float]
    electrode_porosity: Tuple[float, float]
    electrode_tortuosity: Tuple[float, float]
    ionic_conductivity: Tuple[float, float]
    electronic_conductivity: Tuple[float, float]
    electrode_thickness_um: Tuple[float, float]
    electrolyte_thickness_um: Tuple[float, float]
    interconnect_cte: Tuple[float, float]
    interconnect_youngs_gpa: Tuple[float, float]
    crack_length_mm: Tuple[float, float]
    porosity_variation: Tuple[float, float]


@dataclass
class NoiseLevels:
    voltage_v: float
    temperature_c: float
    flow_sccm: float
    strain_micro: float
    thermal_image_c: float


@dataclass
class Inputs:
    current_density: float
    fuel_utilization: float
    air_utilization: float
    inlet_fuel_temp_c: float
    inlet_air_temp_c: float
    fuel_composition_h2: float
    fuel_composition_h2o: float
    fuel_composition_co: float
    fuel_composition_ch4: float
    electrode_porosity: float
    electrode_tortuosity: float
    ionic_conductivity: float
    electronic_conductivity: float
    electrode_thickness_um: float
    electrolyte_thickness_um: float
    interconnect_cte: float
    interconnect_youngs_gpa: float
    crack_length_mm: float
    porosity_variation: float


# ---------------------
# Utility / Sampling
# ---------------------

def load_config(cfg_path: str) -> Tuple[GridSpec, ParamRanges, NoiseLevels, Dict]:
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    grid = GridSpec(**cfg['grid'])
    r = cfg['ranges']
    ranges = ParamRanges(
        current_density=tuple(r['current_density']),
        fuel_utilization=tuple(r['fuel_utilization']),
        air_utilization=tuple(r['air_utilization']),
        inlet_fuel_temp_c=tuple(r['inlet_fuel_temp_c']),
        inlet_air_temp_c=tuple(r['inlet_air_temp_c']),
        fuel_composition_h2=tuple(r['fuel_composition_h2']),
        fuel_composition_h2o=tuple(r['fuel_composition_h2o']),
        fuel_composition_co=tuple(r['fuel_composition_co']),
        fuel_composition_ch4=tuple(r['fuel_composition_ch4']),
        electrode_porosity=tuple(r['electrode_porosity']),
        electrode_tortuosity=tuple(r['electrode_tortuosity']),
        ionic_conductivity=tuple(r['ionic_conductivity']),
        electronic_conductivity=tuple(r['electronic_conductivity']),
        electrode_thickness_um=tuple(r['electrode_thickness_um']),
        electrolyte_thickness_um=tuple(r['electrolyte_thickness_um']),
        interconnect_cte=tuple(r['interconnect_cte']),
        interconnect_youngs_gpa=tuple(r['interconnect_youngs_gpa']),
        crack_length_mm=tuple(r['crack_length_mm']),
        porosity_variation=tuple(r['porosity_variation']),
    )
    nl = cfg['noise_levels']['measurement_sigma']
    noise = NoiseLevels(
        voltage_v=float(nl['voltage_v']),
        temperature_c=float(nl['temperature_c']),
        flow_sccm=float(nl['flow_sccm']),
        strain_micro=float(nl['strain_micro']),
        thermal_image_c=float(nl['thermal_image_c']),
    )
    return grid, ranges, noise, cfg


def lhs_samples(n: int, ranges: ParamRanges, seed: int) -> List[Inputs]:
    fields = list(ParamRanges.__annotations__.keys())
    dim = len(fields)
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    u = sampler.random(n)
    mins = np.array([getattr(ranges, f)[0] for f in fields], dtype=float)
    maxs = np.array([getattr(ranges, f)[1] for f in fields], dtype=float)
    x = qmc.scale(u, mins, maxs)

    inputs = []
    for row in x:
        kwargs = {fields[i]: float(row[i]) for i in range(dim)}
        # Normalize fuel composition to <= 1 with remainder as N2/others (implicit)
        total_fuel = kwargs['fuel_composition_h2'] + kwargs['fuel_composition_h2o'] + kwargs['fuel_composition_co'] + kwargs['fuel_composition_ch4']
        if total_fuel > 0.98:
            scale = 0.98 / total_fuel
            kwargs['fuel_composition_h2'] *= scale
            kwargs['fuel_composition_h2o'] *= scale
            kwargs['fuel_composition_co'] *= scale
            kwargs['fuel_composition_ch4'] *= scale
        inputs.append(Inputs(**kwargs))
    return inputs


# ---------------------
# Physics-inspired models (fast surrogates)
# ---------------------

def _make_grid(grid: GridSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, grid.nx)
    y = np.linspace(0.0, 1.0, grid.ny)
    z = np.linspace(0.0, 0.2, grid.nz)  # thin through-thickness
    return np.meshgrid(x, y, z, indexing='ij')


def compute_fields(inputs: Inputs, grid: GridSpec, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    X, Y, Z = _make_grid(grid)

    # Effective temperature baseline
    t_fuel = inputs.inlet_fuel_temp_c
    t_air = inputs.inlet_air_temp_c
    base_temp = 0.5 * (t_fuel + t_air)

    # Current density baseline and spatial modulation due to species depletion along x
    i0 = inputs.current_density  # A/cm^2
    species_depletion = np.exp(-2.0 * X)  # less fuel downstream
    porosity_map = inputs.electrode_porosity * (1.0 + inputs.porosity_variation * (rng.standard_normal(X.shape) * 0.2))
    tortuosity_factor = np.clip(1.0 / inputs.electrode_tortuosity, 0.2, 1.0)

    # Local current density field (arbitrary scaling to keep in bounds)
    i_field = i0 * species_depletion * np.clip(0.7 + 0.6 * (porosity_map - 0.2), 0.3, 1.3) * (0.7 + 0.3 * np.exp(-Z / 0.05))

    # Ohmic heating proportional to i^2 / sigma_eff
    sigma_eff = 0.5 * (inputs.ionic_conductivity + inputs.electronic_conductivity)  # S/m surrogate
    q_gen = (i_field ** 2) / (1e-2 + sigma_eff)  # W surrogate

    # Thermal field: baseline + heating + convective cooling along x; add z gradient
    temp = base_temp + 40.0 * (q_gen / (np.max(q_gen) + 1e-6))
    temp = temp - 15.0 * X + 5.0 * (1.0 - np.exp(-Z / 0.04))
    temp += rng.normal(0.0, 0.5, size=temp.shape)

    # Species: simple balances (scaled 0-1 mol fraction fields)
    H2 = np.clip(0.8 * np.exp(-1.5 * X) * (1.0 - 0.2 * Z) * (inputs.fuel_composition_h2 / 0.95), 0.0, 1.0)
    H2O = np.clip(0.2 + 0.5 * (1.0 - np.exp(-1.5 * X)) * (1.0 - 0.1 * Z) * (inputs.fuel_composition_h2o / 0.5), 0.0, 1.0)
    O2 = np.clip(0.21 * (1.0 - 0.1 * X) * (1.0 + 0.1 * Z), 0.0, 1.0)

    # Voltage: Nernst-like drop with ohmic and activation components
    open_circuit_v = 1.1 - 0.0002 * (base_temp - 700.0)
    ohmic_drop = (i0 / (sigma_eff + 1e-6)) * (inputs.electrolyte_thickness_um / 50.0)
    activation_drop = 0.08 * math.log(1.0 + i0)
    voltage = max(0.2, open_circuit_v - 0.3 * i0 - ohmic_drop - activation_drop)

    # Thermoelastic displacement (very simplified expansion constrained by boundaries)
    alpha_th = 12e-6  # 1/K typical
    T_ref = 650.0
    dT = temp - T_ref
    Ux = alpha_th * dT * (X - 0.5)
    Uy = alpha_th * dT * (Y - 0.5)
    Uz = 0.5 * alpha_th * dT * (Z - 0.1)

    # Strain tensor from displacement gradients (small strain)
    def grad(f, axis):
        return np.gradient(f, axis=axis, edge_order=2)

    exx = grad(Ux, 0)
    eyy = grad(Uy, 1)
    ezz = grad(Uz, 2)
    exy = 0.5 * (grad(Ux, 1) + grad(Uy, 0))
    exz = 0.5 * (grad(Ux, 2) + grad(Uz, 0))
    eyz = 0.5 * (grad(Uy, 2) + grad(Uz, 1))

    # Isotropic linear elasticity
    E = inputs.interconnect_youngs_gpa * 1e9
    nu = 0.28
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    tr = exx + eyy + ezz
    sxx = lam * tr + 2 * mu * exx
    syy = lam * tr + 2 * mu * eyy
    szz = lam * tr + 2 * mu * ezz
    sxy = 2 * mu * exy
    sxz = 2 * mu * exz
    syz = 2 * mu * eyz

    # Von Mises
    von_mises = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * (sxy ** 2 + sxz ** 2 + syz ** 2))

    # Fracture metrics at a synthetic crack tip region near mid-plane
    a = max(1e-6, inputs.crack_length_mm) / 1000.0  # m
    sigma_nom = float(np.percentile(von_mises, 95))  # Pa
    Yf = 1.12
    K_I = Yf * sigma_nom * math.sqrt(math.pi * a)
    K_II = 0.2 * K_I
    K_III = 0.1 * K_I
    G = (1 - nu ** 2) / E * (K_I ** 2 + K_II ** 2 + K_III ** 2)

    # Creep damage surrogate using Norton law over arbitrary horizon (t_h = 1000 s)
    A = 1e-25 * np.exp(0.015 * (np.mean(temp) - 650.0))
    n = 3.5
    t_h = 1000.0
    creep_strain = A * (sigma_nom ** n) * t_h

    return {
        'T': temp.astype(np.float32),
        'i': i_field.astype(np.float32),
        'H2': H2.astype(np.float32),
        'H2O': H2O.astype(np.float32),
        'O2': O2.astype(np.float32),
        'Ux': Ux.astype(np.float32),
        'Uy': Uy.astype(np.float32),
        'Uz': Uz.astype(np.float32),
        'sxx': sxx.astype(np.float32),
        'syy': syy.astype(np.float32),
        'szz': szz.astype(np.float32),
        'sxy': sxy.astype(np.float32),
        'syz': syz.astype(np.float32),
        'sxz': sxz.astype(np.float32),
        'exx': exx.astype(np.float32),
        'eyy': eyy.astype(np.float32),
        'ezz': ezz.astype(np.float32),
        'exy': exy.astype(np.float32),
        'eyz': eyz.astype(np.float32),
        'exz': exz.astype(np.float32),
        'von_mises': von_mises.astype(np.float32),
        'voltage': np.array([voltage], dtype=np.float32),
        'K_I': np.array([K_I], dtype=np.float32),
        'K_II': np.array([K_II], dtype=np.float32),
        'K_III': np.array([K_III], dtype=np.float32),
        'G': np.array([G], dtype=np.float32),
        'creep_strain': np.array([creep_strain], dtype=np.float32),
    }


# ---------------------
# Dataset 2: experimental-like synthetic
# ---------------------

def generate_experiment_run(duration_s: int, noise: NoiseLevels, base_v: float, base_i: float, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    t = np.arange(duration_s, dtype=float)
    # Drift and noise in V and I
    V = base_v - 0.02 * (t / duration_s) + rng.normal(0, noise.voltage_v, size=t.shape)
    I = base_i * (1.0 - 0.05 * (t / duration_s)) + rng.normal(0, 0.01 * base_i, size=t.shape)
    Power = V * I
    Tin = 650 + 10 * np.sin(2 * np.pi * t / 600.0) + rng.normal(0, noise.temperature_c, size=t.shape)
    FuelFlow = 100 + 10 * np.sin(2 * np.pi * t / 120.0) + rng.normal(0, noise.flow_sccm, size=t.shape)
    AirFlow = 500 + 30 * np.cos(2 * np.pi * t / 300.0) + rng.normal(0, noise.flow_sccm, size=t.shape)

    # EIS: Randles-like arc
    freqs = np.logspace(5, -1, 60)
    Rs = 0.15
    Rct = 0.3 + 0.05 * rng.random()
    Cdl = 1e-2
    omega = 2 * np.pi * freqs
    Z = Rs + Rct / (1 + 1j * omega * Rct * Cdl)

    # Thermal image (single snapshot near mid transients)
    img_size = 96
    x = np.linspace(-1, 1, img_size)
    Xg, Yg = np.meshgrid(x, x)
    center_hot = 650 + 20 * np.exp(-(Xg ** 2 + Yg ** 2) / 0.2)
    thermal_img = center_hot + rng.normal(0, noise.thermal_image_c, size=center_hot.shape)

    # Strain gauge points (4 points)
    gauge_positions = np.array([[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8]], dtype=float)
    strain_series = np.zeros((len(t), gauge_positions.shape[0]), dtype=float)
    for idx, (gx, gy) in enumerate(gauge_positions):
        strain_series[:, idx] = 200 + 50 * np.sin(2 * np.pi * t / (400 + 100 * idx)) + rng.normal(0, noise.strain_micro, size=t.shape)

    # Acoustic emission events: Poisson process with burst amplitudes
    rate = 0.001  # events per sec
    n_events = rng.poisson(rate * duration_s)
    event_times = np.sort(rng.integers(0, duration_s, size=n_events))
    event_amplitudes = rng.lognormal(mean=1.0, sigma=0.5, size=n_events)

    return {
        't': t.astype(np.float32),
        'voltage': V.astype(np.float32),
        'current': I.astype(np.float32),
        'power': Power.astype(np.float32),
        'fuel_flow_sccm': FuelFlow.astype(np.float32),
        'air_flow_sccm': AirFlow.astype(np.float32),
        'inlet_temp_c': Tin.astype(np.float32),
        'eis_freq_hz': freqs.astype(np.float32),
        'eis_Z_real': np.real(Z).astype(np.float32),
        'eis_Z_imag': np.imag(Z).astype(np.float32),
        'thermal_image_c': thermal_img.astype(np.float32),
        'strain_positions_xy': gauge_positions.astype(np.float32),
        'strain_micro': strain_series.astype(np.float32),
        'ae_event_times_s': event_times.astype(np.int32),
        'ae_event_amplitude': event_amplitudes.astype(np.float32),
    }


# ---------------------
# Dataset 3: real-time stream (synthetic)
# ---------------------

def generate_stream(steps: int, base_v: float, base_i: float, noise: NoiseLevels, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    t = np.arange(steps, dtype=float)
    V = base_v - 0.0005 * t + rng.normal(0, noise.voltage_v, size=t.shape)
    I = base_i * (1.0 - 1e-4 * t) + rng.normal(0, 0.01 * base_i, size=t.shape)
    Tin = 650 + 5 * np.sin(2 * np.pi * t / 1800.0) + rng.normal(0, noise.temperature_c, size=t.shape)
    Fin = 120 + 5 * np.cos(2 * np.pi * t / 900.0) + rng.normal(0, noise.flow_sccm, size=t.shape)
    return {
        't': t.astype(np.float32),
        'voltage': V.astype(np.float32),
        'current': I.astype(np.float32),
        'inlet_temp_c': Tin.astype(np.float32),
        'inlet_flow_sccm': Fin.astype(np.float32),
    }


# ---------------------
# Writers
# ---------------------

def write_dataset1_h5(path: str, inputs_list: List[Inputs], fields_list: List[Dict[str, np.ndarray]]):
    with h5py.File(path, 'w') as f:
        f.attrs['dataset'] = 'SOFC Dataset 1 - High-Fidelity Synthetic Surrogates'
        for idx, (inp, fld) in enumerate(zip(inputs_list, fields_list)):
            g = f.create_group(f'sample_{idx:05d}')
            g.attrs['inputs'] = json.dumps(asdict(inp))
            for k, v in fld.items():
                g.create_dataset(k, data=v, compression='gzip', compression_opts=4)


def write_dataset2_h5(path: str, runs: List[Dict[str, np.ndarray]]):
    with h5py.File(path, 'w') as f:
        f.attrs['dataset'] = 'SOFC Dataset 2 - Experimental-like Synthetic'
        for idx, run in enumerate(runs):
            g = f.create_group(f'run_{idx:03d}')
            for k, v in run.items():
                g.create_dataset(k, data=v, compression='gzip', compression_opts=4)


def write_dataset3_h5(path: str, stream: Dict[str, np.ndarray]):
    with h5py.File(path, 'w') as f:
        f.attrs['dataset'] = 'SOFC Dataset 3 - Real-time Stream Synthetic'
        for k, v in stream.items():
            f.create_dataset(k, data=v, compression='gzip', compression_opts=4)


# ---------------------
# Orchestrator / CLI
# ---------------------

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic SOFC multi-fidelity/multi-physics datasets')
    parser.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'config', 'default.yaml'))
    parser.add_argument('--outdir', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'output'))
    parser.add_argument('--d1-samples', type=int, default=None)
    parser.add_argument('--d2-runs', type=int, default=None)
    parser.add_argument('--d3-steps', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    grid, ranges, noise, cfg = load_config(os.path.abspath(args.config))
    d1_n = args.d1_samples if args.d1_samples is not None else int(cfg['sampling']['dataset1_samples'])
    d2_n = args.d2_runs if args.d2_runs is not None else int(cfg['sampling']['dataset2_runs'])
    d3_steps = args.d3_steps if args.d3_steps is not None else int(cfg['sampling']['dataset3_steps'])

    seed = args.seed if args.seed is not None else int(cfg.get('random_seed', 123))
    rng = np.random.default_rng(seed)

    # Dataset 1
    inputs_list = lhs_samples(d1_n, ranges, seed)
    fields_list: List[Dict[str, np.ndarray]] = []
    for inp in tqdm(inputs_list, desc='Dataset1 samples'):
        fields = compute_fields(inp, grid, rng)
        fields_list.append(fields)

    d1_path = os.path.join(outdir, 'dataset1_simulated.h5')
    write_dataset1_h5(d1_path, inputs_list, fields_list)

    # Dataset 2
    runs: List[Dict[str, np.ndarray]] = []
    for k in tqdm(range(d2_n), desc='Dataset2 runs'):
        base_v = float(np.mean([fld['voltage'][0] for fld in fields_list]))
        base_i = float(np.mean([inp.current_density for inp in inputs_list]))
        run = generate_experiment_run(duration_s=3600, noise=noise, base_v=base_v, base_i=base_i, rng=rng)
        runs.append(run)
    d2_path = os.path.join(outdir, 'dataset2_experimental.h5')
    write_dataset2_h5(d2_path, runs)

    # Dataset 3
    base_v = float(np.mean([fld['voltage'][0] for fld in fields_list]))
    base_i = float(np.mean([inp.current_density for inp in inputs_list]))
    stream = generate_stream(d3_steps, base_v, base_i, noise, rng)
    d3_path = os.path.join(outdir, 'dataset3_stream.h5')
    write_dataset3_h5(d3_path, stream)

    # Manifest
    manifest = {
        'dataset1_path': os.path.abspath(d1_path),
        'dataset2_path': os.path.abspath(d2_path),
        'dataset3_path': os.path.abspath(d3_path),
        'grid': asdict(grid),
        'counts': {
            'dataset1_samples': d1_n,
            'dataset2_runs': d2_n,
            'dataset3_steps': d3_steps,
        },
        'variables': {
            'dataset1': ['T','i','H2','H2O','O2','Ux','Uy','Uz','sxx','syy','szz','sxy','syz','sxz','exx','eyy','ezz','exy','eyz','exz','von_mises','voltage','K_I','K_II','K_III','G','creep_strain'],
            'dataset2': ['t','voltage','current','power','fuel_flow_sccm','air_flow_sccm','inlet_temp_c','eis_freq_hz','eis_Z_real','eis_Z_imag','thermal_image_c','strain_positions_xy','strain_micro','ae_event_times_s','ae_event_amplitude'],
            'dataset3': ['t','voltage','current','inlet_temp_c','inlet_flow_sccm'],
        },
    }
    with open(os.path.join(outdir, 'manifest.json'), 'w') as mf:
        json.dump(manifest, mf, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
