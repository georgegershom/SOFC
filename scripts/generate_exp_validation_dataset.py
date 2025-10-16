#!/usr/bin/env python3
"""
Experimental Validation Dataset (Reality Check) generator.

Produces a compact, realistic synthetic dataset mimicking experimental measurements
used for validating ML-augmented inverse modeling of SOFC plate warpage and
residual stresses. The dataset includes:

Per-sample:
- Warp surface height map (npy, microns)
- Downsampled 3D point cloud (PLY ASCII, mm + microns)
- Curvature-based (Stoney) inferred film stress CSV
- Layer removal curvatures and estimated stress gradient CSV
- Ground-truth through-thickness stress profile CSV (MPa)
- XRD-like localized surface stress map CSV
- Raman-like localized surface stress map CSV

Additionally a top-level manifest.json describing the dataset.

This script uses only Python stdlib and numpy for portability.
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

import numpy as np


# ----------------------------- Utility classes ----------------------------- #

@dataclass
class Layer:
    name: str
    thickness_um: float
    porosity: float
    youngs_modulus_gpa: float
    poisson: float
    cte_per_k: float  # 1/K
    shrinkage_strain: float  # negative for shrinkage

    def effective_modulus_gpa(self) -> float:
        # Simple Gibson–Ashby inspired scaling: E_eff = E_bulk * (1 - phi)^n
        # n ~ 2 works reasonably for ceramics in this range
        return self.youngs_modulus_gpa * (max(0.0, 1.0 - self.porosity) ** 2.0)


@dataclass
class SampleParameters:
    anode: Layer
    electrolyte: Layer
    cathode: Layer
    sinter_peak_c: float
    dwell_minutes: float
    gradient_factor: float  # controls through-thickness stress gradient
    anisotropy_strength: float  # splits Kx, Ky
    anisotropy_angle_rad: float  # orientation for anisotropy
    local_waviness_um: float
    measurement_noise_level: float


@dataclass
class Curvature:
    kx: float  # 1/m
    ky: float  # 1/m
    kxy: float  # 1/m, saddle/twist term


# ----------------------------- Sampling helpers ---------------------------- #

def latin_hypercube(n_samples: int, bounds: List[Tuple[float, float]], rng: random.Random) -> np.ndarray:
    """Generate Latin Hypercube samples in [0,1]^d then scale to bounds."""
    d = len(bounds)
    result = np.zeros((n_samples, d), dtype=float)
    for j in range(d):
        perm = list(range(n_samples))
        rng.shuffle(perm)
        for i in range(n_samples):
            u = (perm[i] + rng.random()) / n_samples
            low, high = bounds[j]
            result[i, j] = low + u * (high - low)
    return result


def choose_extremes_and_center(bounds: Dict[str, Tuple[float, float]], rng: random.Random) -> List[Dict[str, float]]:
    keys = list(bounds.keys())
    # pick extremes for a subset of influential dims to keep count reasonable
    extreme_dims = [
        'anode_thickness_um', 'electrolyte_thickness_um', 'cathode_thickness_um',
        'sinter_peak_c', 'gradient_factor', 'anisotropy_strength', 'local_waviness_um'
    ]
    # 2^n grows fast; instead enumerate a curated set of 12 extremes
    extreme_sets: List[Dict[str, float]] = []
    combos = [
        {'gradient_factor': bounds['gradient_factor'][0], 'anisotropy_strength': bounds['anisotropy_strength'][0]},
        {'gradient_factor': bounds['gradient_factor'][1], 'anisotropy_strength': bounds['anisotropy_strength'][1]},
        {'gradient_factor': bounds['gradient_factor'][0], 'anisotropy_strength': bounds['anisotropy_strength'][1]},
        {'gradient_factor': bounds['gradient_factor'][1], 'anisotropy_strength': bounds['anisotropy_strength'][0]},
    ]
    thicks = [
        {
            'anode_thickness_um': bounds['anode_thickness_um'][0],
            'electrolyte_thickness_um': bounds['electrolyte_thickness_um'][1],
            'cathode_thickness_um': bounds['cathode_thickness_um'][1],
        },
        {
            'anode_thickness_um': bounds['anode_thickness_um'][1],
            'electrolyte_thickness_um': bounds['electrolyte_thickness_um'][0],
            'cathode_thickness_um': bounds['cathode_thickness_um'][0],
        },
    ]
    others = [
        {'sinter_peak_c': bounds['sinter_peak_c'][0], 'local_waviness_um': bounds['local_waviness_um'][0]},
        {'sinter_peak_c': bounds['sinter_peak_c'][1], 'local_waviness_um': bounds['local_waviness_um'][1]},
    ]
    for c in combos:
        for t in thicks:
            for o in others:
                extreme = {k: (bounds[k][0] + bounds[k][1]) / 2.0 for k in keys}
                extreme.update(c)
                extreme.update(t)
                extreme.update(o)
                # randomized anisotropy angle at extremes
                extreme['anisotropy_angle_rad'] = rng.random() * 2.0 * math.pi
                extreme_sets.append(extreme)
                if len(extreme_sets) >= 12:
                    break
            if len(extreme_sets) >= 12:
                break
        if len(extreme_sets) >= 12:
            break
    # center point
    center = {k: (bounds[k][0] + bounds[k][1]) / 2.0 for k in keys}
    center['anisotropy_angle_rad'] = 0.0
    extreme_sets.append(center)
    return extreme_sets


# ----------------------------- Physics surrogates -------------------------- #

def compute_ground_truth_stress_profile(params: SampleParameters, num_depth_points: int = 101) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a plausible through-thickness stress profile sigma(z).

    Returns:
        z_um: depths from top surface (0) to bottom (t_total)
        sigma_mpa: axial stress (MPa), positive tensile
    """
    t_a = params.anode.thickness_um
    t_e = params.electrolyte.thickness_um
    t_c = params.cathode.thickness_um
    t_total = t_c + t_e + t_a

    # Effective moduli (GPa) for each layer considering porosity
    E_a = params.anode.effective_modulus_gpa()
    E_e = params.electrolyte.effective_modulus_gpa()
    E_c = params.cathode.effective_modulus_gpa()
    nu_a = params.anode.poisson
    nu_e = params.electrolyte.poisson
    nu_c = params.cathode.poisson

    # Thermal mismatch based on cooling from sinter peak to 25C
    delta_T = params.sinter_peak_c - 25.0
    eps_a = params.anode.shrinkage_strain + (params.anode.cte_per_k * delta_T)
    eps_e = params.electrolyte.shrinkage_strain + (params.electrolyte.cte_per_k * delta_T)
    eps_c = params.cathode.shrinkage_strain + (params.cathode.cte_per_k * delta_T)

    # Plane stress effective stiffness Y = E / (1 - nu)
    Y_a = E_a / (1.0 - nu_a)
    Y_e = E_e / (1.0 - nu_e)
    Y_c = E_c / (1.0 - nu_c)

    # Compute self-equilibrated mean stresses per layer enforcing net force ~ 0
    # Solve for layer stresses s_a, s_e, s_c minimizing mismatch subject to force balance.
    # Here, approximate by weighted deviation from a common Strain_ref such that sum(Y_i * t_i * (eps_i - eps_ref)) = 0
    eps_ref = (
        Y_a * t_a * eps_a + Y_e * t_e * eps_e + Y_c * t_c * eps_c
    ) / (Y_a * t_a + Y_e * t_e + Y_c * t_c)
    s_a = Y_a * (eps_a - eps_ref)
    s_e = Y_e * (eps_e - eps_ref)
    s_c = Y_c * (eps_c - eps_ref)

    # Impose a gradient across thickness to create bending moment
    # gradient_factor in [-0.4, 0.4] scales a normalized linear gradient
    z_um = np.linspace(0.0, t_total, num_depth_points)
    sigma = np.zeros_like(z_um)
    # Layer piecewise constants with gradient
    def layer_sigma(base: float, z0: float, z1: float) -> np.ndarray:
        z = np.clip((z_um - z0) / max(1e-9, (z1 - z0)), 0.0, 1.0)
        grad = (z - 0.5) * 2.0  # -1 to 1
        return base * (1.0 + params.gradient_factor * grad)

    # z=0 top, order: cathode [0, t_c], electrolyte [t_c, t_c+t_e], anode [t_c+t_e, t_total]
    sigma += layer_sigma(s_c, 0.0, t_c)
    sigma += layer_sigma(s_e, t_c, t_c + t_e)
    sigma += layer_sigma(s_a, t_c + t_e, t_total)

    # Convert GPa to MPa for output
    sigma_mpa = sigma * 1000.0
    return z_um, sigma_mpa


def curvature_from_profile(z_um: np.ndarray, sigma_mpa: np.ndarray, params: SampleParameters) -> Curvature:
    """Map stress profile to plate curvatures (kx, ky, kxy) in 1/m.

    Approximate: K ~ M / (D), where bending moment M = ∫ sigma(z)*z dz (SI),
    D ~ E_eq * t^3 / (12(1 - nu^2)). We compute a scalar K_iso then split into Kx, Ky
    using anisotropy_strength and angle; add a small twist term.
    """
    # Convert units
    z_m = z_um * 1e-6
    sigma_pa = sigma_mpa * 1e6

    # Compute bending moment per unit width: M = ∫ sigma(z) * (z - z0) dz, choose mid-plane as reference
    t_total_m = float(z_m[-1] - z_m[0])
    z_mid = z_m[0] + 0.5 * t_total_m
    M = np.trapz(sigma_pa * (z_m - z_mid), z_m)  # N/m

    # Equivalent plate stiffness: weighted by layer moduli and thickness
    t_a = params.anode.thickness_um * 1e-6
    t_e = params.electrolyte.thickness_um * 1e-6
    t_c = params.cathode.thickness_um * 1e-6
    t_total = t_a + t_e + t_c

    # Use a simple equivalent E and nu
    E_eq = (
        params.anode.effective_modulus_gpa() * t_a +
        params.electrolyte.effective_modulus_gpa() * t_e +
        params.cathode.effective_modulus_gpa() * t_c
    ) / max(1e-12, t_total) * 1e9  # Pa
    nu_eq = (params.anode.poisson + params.electrolyte.poisson + params.cathode.poisson) / 3.0
    D = E_eq * (t_total ** 3) / (12.0 * (1.0 - nu_eq * nu_eq) + 1e-12)

    K_iso = float(M / max(1e-9, D))  # 1/m

    # Anisotropy split
    a = max(-0.95, min(0.95, params.anisotropy_strength))
    k_major = K_iso * (1.0 + a)
    k_minor = K_iso * (1.0 - a)

    # Map principal curvatures to x,y via rotation
    theta = params.anisotropy_angle_rad
    c, s = math.cos(theta), math.sin(theta)
    # Curvature tensor in principal axes: diag(k_major, k_minor)
    # Rotate to xy: K = R diag R^T
    kx = k_major * (c * c) + k_minor * (s * s)
    ky = k_major * (s * s) + k_minor * (c * c)
    kxy = (k_major - k_minor) * s * c

    return Curvature(kx=kx, ky=ky, kxy=kxy * 0.1)  # smaller twist


def generate_height_map(curv: Curvature, grid_size: int, plate_size_mm: float, waviness_um: float, rng: np.random.Generator) -> np.ndarray:
    """Generate warp surface height map (microns) over a square plate.

    Coordinates:
      - x,y in [-L/2, L/2] with L = plate_size_mm (mm)
      - height in microns
    """
    L_m = plate_size_mm / 1000.0
    x = np.linspace(-0.5 * L_m, 0.5 * L_m, grid_size, dtype=np.float64)
    y = np.linspace(-0.5 * L_m, 0.5 * L_m, grid_size, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing='xy')

    # Base quadratic surface from curvature: w = 0.5*(kx x^2 + ky y^2) + kxy x y
    w_m = 0.5 * (curv.kx * xx * xx + curv.ky * yy * yy) + curv.kxy * xx * yy

    # Add low-frequency waviness (micron-scale) and a tiny random roughness
    # Use a few Fourier modes to emulate process fingerprints
    def sin2d(ax, ay, phase):
        return np.sin(ax * xx + ay * yy + phase)

    modes = [
        (2.0 * math.pi / (0.5 * L_m), 2.0 * math.pi / (0.5 * L_m), rng.random()),
        (2.0 * math.pi / (0.33 * L_m), 0.0, rng.random()),
        (0.0, 2.0 * math.pi / (0.25 * L_m), rng.random()),
        (2.0 * math.pi / (0.18 * L_m), 2.0 * math.pi / (0.21 * L_m), rng.random()),
    ]
    waviness_m = (waviness_um * 1e-6) * (0.35 * sin2d(*modes[0]) + 0.25 * sin2d(*modes[1]) + 0.25 * sin2d(*modes[2]) + 0.15 * sin2d(*modes[3]))

    fine_noise_m = (waviness_um * 1e-6) * 0.05 * rng.normal(size=w_m.shape)

    w_total_m = w_m + waviness_m + fine_noise_m

    # Subtract mean to center around zero; convert to microns
    w_um = (w_total_m - float(np.mean(w_total_m))) * 1e6
    return w_um.astype(np.float32)


def estimate_curvature_from_height_map(height_um: np.ndarray, plate_size_mm: float) -> Curvature:
    """Fit a quadratic surface to estimate curvature from height map."""
    n = height_um.shape[0]
    L_m = plate_size_mm / 1000.0
    x = np.linspace(-0.5 * L_m, 0.5 * L_m, n, dtype=np.float64)
    y = np.linspace(-0.5 * L_m, 0.5 * L_m, n, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    zz_m = height_um.astype(np.float64) * 1e-6

    # Fit z = a x^2 + b y^2 + c x y + d x + e y + f
    X = np.column_stack([
        xx.ravel() ** 2,
        yy.ravel() ** 2,
        (xx.ravel() * yy.ravel()),
        xx.ravel(),
        yy.ravel(),
        np.ones_like(xx).ravel(),
    ])
    yv = zz_m.ravel()

    coeffs, *_ = np.linalg.lstsq(X, yv, rcond=None)
    a, b, c, _, _, _ = coeffs
    # For small deflections, curvature ~ 2a (since z ~ 0.5*kx x^2 => a = 0.5*kx)
    kx = 2.0 * a
    ky = 2.0 * b
    kxy = c
    return Curvature(kx=float(kx), ky=float(ky), kxy=float(kxy))


def write_ply_point_cloud(path: Path, height_um: np.ndarray, plate_size_mm: float, downsample: int = 4) -> None:
    """Write an ASCII PLY point cloud using a downsampled grid."""
    n = height_um.shape[0]
    step = max(1, downsample)
    sel = np.s_[::step, ::step]
    hh = height_um[sel]

    L_mm = plate_size_mm
    x = np.linspace(-0.5 * L_mm, 0.5 * L_mm, n, dtype=np.float64)[::step]
    y = np.linspace(-0.5 * L_mm, 0.5 * L_mm, n, dtype=np.float64)[::step]
    xx, yy = np.meshgrid(x, y, indexing='xy')

    points = np.column_stack([xx.ravel(), yy.ravel(), hh.ravel()])

    with path.open('w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {points.shape[0]}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('comment units: x,y in mm; z in microns\n')
        f.write('end_header\n')
        for px, py, pz in points:
            f.write(f"{px:.6f} {py:.6f} {pz:.6f}\n")


# ----------------------------- Measurement simulators ---------------------- #

def simulate_stoney(anode: Layer, film: Layer, curvature: Curvature, noise_level: float, rng: np.random.Generator) -> Dict[str, float]:
    """Compute film stress via Stoney's formula along x and y, add noise."""
    # Stoney: sigma_f = (E_s * t_s^2) / (6 * (1 - nu_s) * t_f) * (1/R)
    E_s = anode.effective_modulus_gpa() * 1e9
    nu_s = anode.poisson
    t_s = anode.thickness_um * 1e-6
    t_f = film.thickness_um * 1e-6

    kx = curvature.kx
    ky = curvature.ky

    def noisy(val: float) -> float:
        return float(val * (1.0 + noise_level * rng.normal()))

    sigma_x = (E_s * (t_s ** 2) / (6.0 * (1.0 - nu_s) * max(t_f, 1e-9))) * kx
    sigma_y = (E_s * (t_s ** 2) / (6.0 * (1.0 - nu_s) * max(t_f, 1e-9))) * ky

    return {
        'sigma_film_x_mpa': noisy(sigma_x) / 1e6,
        'sigma_film_y_mpa': noisy(sigma_y) / 1e6,
        'curvature_x_1_per_m': noisy(kx),
        'curvature_y_1_per_m': noisy(ky),
        'substrate_E_gpa': anode.effective_modulus_gpa(),
        'substrate_nu': nu_s,
        'substrate_thickness_um': anode.thickness_um,
        'film_thickness_um': film.thickness_um,
    }


def simulate_layer_removal_curvatures(curv_full: Curvature, params: SampleParameters, noise_level: float, rng: np.random.Generator) -> Dict[str, float]:
    """Approximate curvatures after removing layers (cathode, then electrolyte)."""
    # Removing top layers reduces bending moment; scale curvatures accordingly using thickness ratios.
    t_a = params.anode.thickness_um
    t_e = params.electrolyte.thickness_um
    t_c = params.cathode.thickness_um
    t_total = t_a + t_e + t_c

    def noisy(k: float) -> float:
        return float(k * (1.0 + noise_level * rng.normal()))

    # Simple scaling model: curvature ~ 1 / t_total^2 times moment fraction remaining
    k0x, k0y = curv_full.kx, curv_full.ky

    # After removing cathode
    frac1 = ((t_a + t_e) / t_total) ** 2
    k1x, k1y = noisy(k0x / max(frac1, 1e-6)), noisy(k0y / max(frac1, 1e-6))

    # After removing electrolyte as well
    frac2 = (t_a / t_total) ** 2
    k2x, k2y = noisy(k0x / max(frac2, 1e-6)), noisy(k0y / max(frac2, 1e-6))

    return {
        'curvature_full_x_1_per_m': noisy(k0x),
        'curvature_full_y_1_per_m': noisy(k0y),
        'curvature_after_remove_cathode_x_1_per_m': k1x,
        'curvature_after_remove_cathode_y_1_per_m': k1y,
        'curvature_after_remove_electrolyte_x_1_per_m': k2x,
        'curvature_after_remove_electrolyte_y_1_per_m': k2y,
    }


def estimate_stress_gradient_from_layer_removal(z_um: np.ndarray, sigma_true_mpa: np.ndarray, noise_level: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Produce a noisy, smoothed estimate of through-thickness stress from destructive test."""
    # Convolve with a smoothing kernel and add noise to emulate inverse problem uncertainty
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=float)
    kernel /= kernel.sum()
    sigma_padded = np.pad(sigma_true_mpa, (2, 2), mode='edge')
    smoothed = np.convolve(sigma_padded, kernel, mode='valid')
    noisy = smoothed + noise_level * np.std(smoothed) * rng.normal(size=smoothed.shape)
    return z_um.copy(), noisy.astype(np.float64)


def simulate_localized_stress_map(height_um: np.ndarray, params: SampleParameters, modality: str, rng: np.random.Generator) -> np.ndarray:
    """Simulate localized surface stress measurements (MPa) for XRD/Raman.

    Samples a small ROI and maps stress to curvature-derived strain plus noise.
    Returns array of shape (N, 5): x_mm, y_mm, sigma_xx_mpa, sigma_yy_mpa, uncertainty_mpa
    """
    n = height_um.shape[0]
    L_mm = 50.0
    # Sample grid in a central patch
    grid_n = 20 if modality == 'xrd' else 12
    xs = np.linspace(-5.0, 5.0, grid_n)
    ys = np.linspace(-5.0, 5.0, grid_n)
    xx, yy = np.meshgrid(xs, ys, indexing='xy')

    # Derive local curvature gradients from height map via finite differences
    # Compute second derivatives to estimate local curvature, then map to stress with E,nu
    height_m = height_um.astype(np.float64) * 1e-6
    dx_m = (L_mm / (n - 1)) / 1000.0
    dy_m = dx_m

    # Simple Laplacian-based curvature estimator in center region
    def curv_est(i: int, j: int) -> Tuple[float, float]:
        # second derivatives at (i,j)
        zxx = (height_m[i, j+1] - 2.0 * height_m[i, j] + height_m[i, j-1]) / (dx_m ** 2)
        zyy = (height_m[i+1, j] - 2.0 * height_m[i, j] + height_m[i-1, j]) / (dy_m ** 2)
        return zxx, zyy

    # Map measurement points to indices
    center = n // 2
    span = min(center - 2, 30)
    points = []
    for x_mm, y_mm in zip(xx.ravel(), yy.ravel()):
        ii = center + int((y_mm / (L_mm * 0.5)) * span)
        jj = center + int((x_mm / (L_mm * 0.5)) * span)
        ii = max(2, min(n - 3, ii))
        jj = max(2, min(n - 3, jj))
        zxx, zyy = curv_est(ii, jj)
        # Hooke: sigma ~ D * k, fold constants into a scale factor
        # Use electrolyte properties for XRD (surface layer)
        E = params.electrolyte.effective_modulus_gpa() * 1e3  # MPa
        nu = params.electrolyte.poisson
        scale = E / (1.0 - nu)
        sigma_xx = scale * zxx
        sigma_yy = scale * zyy
        # Add modality-specific noise
        if modality == 'xrd':
            unc = 15.0 + 5.0 * abs(np.random.normal())
            noise = np.random.normal(scale=unc, size=2)
        else:  # Raman
            unc = 25.0 + 10.0 * abs(np.random.normal())
            noise = np.random.normal(scale=unc, size=2)
        points.append([x_mm, y_mm, sigma_xx + noise[0], sigma_yy + noise[1], unc])

    return np.array(points, dtype=np.float64)


# ----------------------------- Main generation ----------------------------- #

def build_parameter_bounds() -> Dict[str, Tuple[float, float]]:
    return {
        'anode_thickness_um': (700.0, 1000.0),
        'electrolyte_thickness_um': (5.0, 20.0),
        'cathode_thickness_um': (30.0, 100.0),
        'anode_porosity': (0.15, 0.35),
        'cathode_porosity': (0.10, 0.30),
        'E_anode_gpa': (90.0, 120.0),
        'E_electrolyte_gpa': (180.0, 210.0),
        'E_cathode_gpa': (130.0, 170.0),
        'nu_anode': (0.25, 0.32),
        'nu_electrolyte': (0.25, 0.31),
        'nu_cathode': (0.25, 0.32),
        'CTE_anode': (11.0e-6, 13.5e-6),
        'CTE_electrolyte': (10.5e-6, 11.5e-6),
        'CTE_cathode': (12.0e-6, 14.0e-6),
        'shrink_anode': (-0.016, -0.010),
        'shrink_electrolyte': (-0.012, -0.006),
        'shrink_cathode': (-0.012, -0.006),
        'sinter_peak_c': (1250.0, 1400.0),
        'dwell_minutes': (30.0, 180.0),
        'gradient_factor': (-0.4, 0.4),
        'anisotropy_strength': (-0.35, 0.35),
        'anisotropy_angle_rad': (0.0, 2.0 * math.pi),
        'local_waviness_um': (0.0, 8.0),
        'measurement_noise_level': (0.005, 0.02),
    }


def sample_parameters(n: int, rng: random.Random) -> List[SampleParameters]:
    bounds = build_parameter_bounds()
    # Prepare extremes + center
    focus_bounds = {k: bounds[k] for k in [
        'anode_thickness_um', 'electrolyte_thickness_um', 'cathode_thickness_um',
        'sinter_peak_c', 'gradient_factor', 'anisotropy_strength', 'local_waviness_um', 'anisotropy_angle_rad'
    ]}
    extreme_sets = choose_extremes_and_center(focus_bounds, rng)

    num_ext = min(len(extreme_sets), max(0, n // 3))
    chosen_extremes = extreme_sets[:num_ext]

    # Latin hypercube for the rest
    keys = list(bounds.keys())
    d = len(keys)
    lhs = latin_hypercube(n - num_ext, [bounds[k] for k in keys], rng)

    samples: List[SampleParameters] = []
    def to_layer(prefix: str, values: Dict[str, float]) -> Layer:
        return Layer(
            name=prefix,
            thickness_um=values[f'{prefix}_thickness_um'],
            porosity=values[f'{prefix}_porosity'] if f'{prefix}_porosity' in values else (0.25 if prefix == 'anode' else 0.2),
            youngs_modulus_gpa=values[f'E_{prefix}_gpa'] if f'E_{prefix}_gpa' in values else (100.0 if prefix == 'anode' else (200.0 if prefix == 'electrolyte' else 150.0)),
            poisson=values[f'nu_{prefix}'] if f'nu_{prefix}' in values else 0.28,
            cte_per_k=values[f'CTE_{prefix}'] if f'CTE_{prefix}' in values else (12e-6),
            shrinkage_strain=values[f'shrink_{prefix}'] if f'shrink_{prefix}' in values else (-0.012),
        )

    # Convert extremes to full param dicts by sampling remaining dims at center
    def fill_missing(base: Dict[str, float]) -> Dict[str, float]:
        complete = {k: (bounds[k][0] + bounds[k][1]) / 2.0 for k in bounds.keys()}
        complete.update(base)
        return complete

    # Build from extremes
    for ext in chosen_extremes:
        vals = fill_missing(ext)
        samples.append(SampleParameters(
            anode=to_layer('anode', vals),
            electrolyte=to_layer('electrolyte', vals),
            cathode=to_layer('cathode', vals),
            sinter_peak_c=vals['sinter_peak_c'],
            dwell_minutes=vals['dwell_minutes'],
            gradient_factor=vals['gradient_factor'],
            anisotropy_strength=vals['anisotropy_strength'],
            anisotropy_angle_rad=vals['anisotropy_angle_rad'],
            local_waviness_um=vals['local_waviness_um'],
            measurement_noise_level=vals['measurement_noise_level'],
        ))

    # Build from LHS
    for i in range(lhs.shape[0]):
        vals = {keys[j]: float(lhs[i, j]) for j in range(d)}
        samples.append(SampleParameters(
            anode=to_layer('anode', vals),
            electrolyte=to_layer('electrolyte', vals),
            cathode=to_layer('cathode', vals),
            sinter_peak_c=vals['sinter_peak_c'],
            dwell_minutes=vals['dwell_minutes'],
            gradient_factor=vals['gradient_factor'],
            anisotropy_strength=vals['anisotropy_strength'],
            anisotropy_angle_rad=vals['anisotropy_angle_rad'],
            local_waviness_um=vals['local_waviness_um'],
            measurement_noise_level=vals['measurement_noise_level'],
        ))

    return samples[:n]


def write_csv(path: Path, header: List[str], rows: np.ndarray | List[List[float]] | List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            # Dict rows
            header = list(rows[0].keys())
            f.write(','.join(header) + '\n')
            for r in rows:
                f.write(','.join(str(r[h]) for h in header) + '\n')
        else:
            f.write(','.join(header) + '\n')
            if isinstance(rows, np.ndarray):
                for r in rows:
                    f.write(','.join(str(x) for x in r) + '\n')
            else:
                for r in rows:
                    f.write(','.join(str(x) for x in r) + '\n')


def generate_dataset(output_dir: Path, num_samples: int, grid_size: int, plate_size_mm: float, seed: int) -> Dict:
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = sample_parameters(num_samples, rng_py)

    manifest: Dict = {
        'dataset_name': 'Experimental Validation Dataset (Reality Check)',
        'version': 1,
        'plate_size_mm': plate_size_mm,
        'grid_size': grid_size,
        'num_samples': num_samples,
        'units': {'x_mm': 'mm', 'y_mm': 'mm', 'z_um': 'micron', 'stress_mpa': 'MPa'},
        'files': [],
        'modalities': ['warp_height_map', 'point_cloud_ply', 'curvature_stoney', 'layer_removal', 'stress_profile_ground_truth', 'xrd_surface_map', 'raman_surface_map'],
        'notes': 'Synthetic dataset emulating experimental validation measurements for SOFC plates. Use warp maps as ML inputs and compare predicted stresses to destructive/point-wise measurements.',
    }

    for idx, sp in enumerate(samples):
        sample_id = f'sample_{idx:03d}'
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 1) Ground-truth stress profile
        z_um, sigma_mpa = compute_ground_truth_stress_profile(sp)

        # 2) Curvatures from profile
        curv_true = curvature_from_profile(z_um, sigma_mpa, sp)

        # 3) Height map and point cloud
        height_um = generate_height_map(curv_true, grid_size=grid_size, plate_size_mm=plate_size_mm, waviness_um=sp.local_waviness_um, rng=rng_np)
        # Persist height
        np.save(sample_dir / 'warp_height_map_um.npy', height_um)
        # Point cloud
        write_ply_point_cloud(sample_dir / 'warp_point_cloud.ply', height_um, plate_size_mm, downsample=4)

        # 4) Estimate curvature from the height map (as an experimental estimator)
        curv_est = estimate_curvature_from_height_map(height_um, plate_size_mm)

        # 5) Stoney measurement using anode as substrate and electrolyte as film
        stoney = simulate_stoney(sp.anode, sp.electrolyte, curv_est, noise_level=sp.measurement_noise_level, rng=rng_np)
        write_csv(sample_dir / 'curvature_stoney.csv', header=[], rows=[stoney])

        # 6) Layer removal curvatures and estimated stress gradient
        lr_curvs = simulate_layer_removal_curvatures(curv_est, sp, noise_level=sp.measurement_noise_level, rng=rng_np)
        write_csv(sample_dir / 'layer_removal_curvatures.csv', header=[], rows=[lr_curvs])

        z_est_um, sigma_est_mpa = estimate_stress_gradient_from_layer_removal(z_um, sigma_mpa, noise_level=3.0 * sp.measurement_noise_level, rng=rng_np)
        write_csv(sample_dir / 'layer_removal_estimated_stress_profile.csv', header=['z_um', 'sigma_mpa'], rows=np.column_stack([z_est_um, sigma_est_mpa]))

        # 7) Ground-truth stress profile file
        write_csv(sample_dir / 'ground_truth_stress_profile.csv', header=['z_um', 'sigma_mpa'], rows=np.column_stack([z_um, sigma_mpa]))

        # 8) Localized maps (XRD & Raman)
        xrd = simulate_localized_stress_map(height_um, sp, modality='xrd', rng=rng_np)
        write_csv(sample_dir / 'xrd_surface_map.csv', header=['x_mm', 'y_mm', 'sigma_xx_mpa', 'sigma_yy_mpa', 'uncertainty_mpa'], rows=xrd)

        raman = simulate_localized_stress_map(height_um, sp, modality='raman', rng=rng_np)
        write_csv(sample_dir / 'raman_surface_map.csv', header=['x_mm', 'y_mm', 'sigma_xx_mpa', 'sigma_yy_mpa', 'uncertainty_mpa'], rows=raman)

        # Save parameters for traceability
        params_json = {
            'anode': asdict(sp.anode),
            'electrolyte': asdict(sp.electrolyte),
            'cathode': asdict(sp.cathode),
            'sinter_peak_c': sp.sinter_peak_c,
            'dwell_minutes': sp.dwell_minutes,
            'gradient_factor': sp.gradient_factor,
            'anisotropy_strength': sp.anisotropy_strength,
            'anisotropy_angle_rad': sp.anisotropy_angle_rad,
            'local_waviness_um': sp.local_waviness_um,
            'measurement_noise_level': sp.measurement_noise_level,
        }
        with (sample_dir / 'parameters.json').open('w') as f:
            json.dump(params_json, f, indent=2)

        manifest['files'].append({
            'sample_id': sample_id,
            'paths': {
                'warp_height_map_um_npy': str((sample_dir / 'warp_height_map_um.npy').relative_to(output_dir)),
                'warp_point_cloud_ply': str((sample_dir / 'warp_point_cloud.ply').relative_to(output_dir)),
                'curvature_stoney_csv': str((sample_dir / 'curvature_stoney.csv').relative_to(output_dir)),
                'layer_removal_curvatures_csv': str((sample_dir / 'layer_removal_curvatures.csv').relative_to(output_dir)),
                'layer_removal_estimated_stress_profile_csv': str((sample_dir / 'layer_removal_estimated_stress_profile.csv').relative_to(output_dir)),
                'ground_truth_stress_profile_csv': str((sample_dir / 'ground_truth_stress_profile.csv').relative_to(output_dir)),
                'xrd_surface_map_csv': str((sample_dir / 'xrd_surface_map.csv').relative_to(output_dir)),
                'raman_surface_map_csv': str((sample_dir / 'raman_surface_map.csv').relative_to(output_dir)),
                'parameters_json': str((sample_dir / 'parameters.json').relative_to(output_dir)),
            }
        })

    with (output_dir / 'manifest.json').open('w') as f:
        json.dump(manifest, f, indent=2)

    return manifest


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Generate Experimental Validation Dataset (Reality Check)')
    ap.add_argument('--output-dir', type=str, default=str(Path(__file__).resolve().parents[1] / 'datasets' / 'experimental_validation_dataset_v1'))
    ap.add_argument('--num-samples', type=int, default=40)
    ap.add_argument('--grid-size', type=int, default=256)
    ap.add_argument('--plate-size-mm', type=float, default=50.0)
    ap.add_argument('--seed', type=int, default=1337)
    return ap.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = generate_dataset(out_dir, args.num_samples, args.grid_size, args.plate_size_mm, args.seed)
    print(json.dumps({
        'output_dir': str(out_dir),
        'num_samples': manifest['num_samples'],
        'grid_size': manifest['grid_size'],
        'plate_size_mm': manifest['plate_size_mm'],
        'manifest_file': str(out_dir / 'manifest.json'),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
