from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np


@dataclass
class Layer:
    name: str
    thickness: float  # m
    E: float  # Pa
    nu: float
    alpha: float  # CTE, 1/K
    eigenstrain: float  # free shrinkage strain (isotropic in-plane)


@dataclass
class LaminateSpec:
    layers: List[Layer]


@dataclass
class LaminateResult:
    # Mid-plane strains and curvatures (classical laminate theory)
    eps0: np.ndarray  # shape (3,)
    kappa: np.ndarray  # shape (3,)
    # Per-layer in-plane stresses (averaged through thickness)
    layer_stress_xy: Dict[str, np.ndarray]  # name -> (sigma_x, sigma_y, tau_xy)


def build_ABD(layers: List[Layer]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Plane stress orthotropic reduced stiffness Q for isotropic material
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    z_bot = -0.5 * sum([la.thickness for la in layers])
    z = z_bot
    for la in layers:
        t = la.thickness
        E = la.E
        nu = la.nu
        denom = 1.0 - nu * nu
        Q11 = E / denom
        Q22 = E / denom
        Q12 = nu * E / denom
        Q66 = E / (2.0 * (1.0 + nu))
        Q = np.array([[Q11, Q12, 0.0], [Q12, Q22, 0.0], [0.0, 0.0, Q66]])

        z_top = z + t
        A += Q * (z_top - z)
        B += 0.5 * Q * (z_top**2 - z**2)
        D += (1.0 / 3.0) * Q * (z_top**3 - z**3)
        z = z_top
    ABD = np.block([[A, B], [B, D]])
    return A, B, D, ABD


def thermal_eigen_loads(layers: List[Layer], deltaT: float) -> Tuple[np.ndarray, np.ndarray]:
    # Integrate thermal/eigen strains to get resultant force/moment vectors
    N_th = np.zeros(3)
    M_th = np.zeros(3)
    z_bot = -0.5 * sum([la.thickness for la in layers])
    z = z_bot
    for la in layers:
        t = la.thickness
        E = la.E
        nu = la.nu
        denom = 1.0 - nu * nu
        Q11 = E / denom
        Q22 = E / denom
        Q12 = nu * E / denom
        Q66 = E / (2.0 * (1.0 + nu))
        Q = np.array([[Q11, Q12, 0.0], [Q12, Q22, 0.0], [0.0, 0.0, Q66]])

        alpha = la.alpha
        eps_th = np.array([alpha * deltaT + la.eigenstrain, alpha * deltaT + la.eigenstrain, 0.0])
        z_top = z + t
        N_th += Q @ eps_th * (z_top - z)
        M_th += 0.5 * Q @ eps_th * (z_top**2 - z**2)
        z = z_top
    return N_th, M_th


def solve_laminate(layers: List[Layer], deltaT: float) -> LaminateResult:
    A, B, D, ABD = build_ABD(layers)
    N_th, M_th = thermal_eigen_loads(layers, deltaT)
    # Resultants equilibrate: [A B; B D] [eps0; kappa] = [N_th; M_th]
    rhs = np.concatenate([N_th, M_th])
    sol = np.linalg.solve(ABD, rhs)
    eps0 = sol[:3]
    kappa = sol[3:]

    # Average in-plane stresses per layer
    z_bot = -0.5 * sum([la.thickness for la in layers])
    z = z_bot
    layer_stress_xy: Dict[str, np.ndarray] = {}
    for la in layers:
        t = la.thickness
        E = la.E
        nu = la.nu
        denom = 1.0 - nu * nu
        Q11 = E / denom
        Q22 = E / denom
        Q12 = nu * E / denom
        Q66 = E / (2.0 * (1.0 + nu))
        Q = np.array([[Q11, Q12, 0.0], [Q12, Q22, 0.0], [0.0, 0.0, Q66]])

        z_top = z + t
        z_mid = 0.5 * (z + z_top)
        alpha = la.alpha
        eps_th = np.array([alpha * deltaT + la.eigenstrain, alpha * deltaT + la.eigenstrain, 0.0])
        eps = eps0 + z_mid * kappa
        sigma = Q @ (eps - eps_th)
        layer_stress_xy[la.name] = sigma
        z = z_top

    return LaminateResult(eps0=eps0, kappa=kappa, layer_stress_xy=layer_stress_xy)


def warp_surface_from_kappa(kappa: np.ndarray, plate_size: Tuple[float, float], grid_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Lx, Ly = plate_size
    nx, ny = grid_shape
    x = np.linspace(-0.5 * Lx, 0.5 * Lx, nx)
    y = np.linspace(-0.5 * Ly, 0.5 * Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    kx, ky, kxy = kappa
    # Small-slope approximation: w(x,y) ~ 0.5*(kx*x^2 + ky*y^2) + kxy*x*y
    W = 0.5 * (kx * X**2 + ky * Y**2) + kxy * X * Y
    return X, Y, W
