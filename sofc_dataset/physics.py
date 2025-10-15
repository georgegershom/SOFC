from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class LayerProps:
    name: str
    thickness: float  # meters
    youngs_modulus: float  # Pa
    poissons_ratio: float  # unitless
    cte: float  # 1/K
    sinter_eigenstrain: float  # unitless (compressive is positive magnitude)
    relaxation: float  # [0,1] 1=no relaxation, 0=fully relaxed

    def reduced_stiffness(self) -> np.ndarray:
        E = self.youngs_modulus
        nu = self.poissons_ratio
        q11 = E / (1 - nu**2)
        q12 = (nu * E) / (1 - nu**2)
        q66 = E / (2 * (1 + nu))
        Q = np.array(
            [
                [q11, q12, 0.0],
                [q12, q11, 0.0],
                [0.0, 0.0, q66],
            ]
        )
        return Q


@dataclass
class PlateDims:
    length_x: float  # meters
    length_y: float  # meters


@dataclass
class LaminateResult:
    midplane_strain: np.ndarray  # (3,)
    curvature: np.ndarray  # (3,)
    A: np.ndarray  # (3,3)
    B: np.ndarray  # (3,3)
    D: np.ndarray  # (3,3)


ThermalEigen = Tuple[np.ndarray, np.ndarray]  # (N_T(3,), M_T(3,))


def _layer_bounds(layers: List[LayerProps]) -> List[Tuple[float, float]]:
    h_total = sum(l.thickness for l in layers)
    z_bot = -0.5 * h_total
    bounds = []
    for l in layers:
        z_top = z_bot + l.thickness
        bounds.append((z_bot, z_top))
        z_bot = z_top
    return bounds


def _A_B_D(layers: List[LayerProps]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    for l, (z0, z1) in zip(layers, _layer_bounds(layers)):
        Q = l.reduced_stiffness()
        dz = z1 - z0
        A += Q * dz
        B += 0.5 * Q * (z1**2 - z0**2)
        D += (1.0 / 3.0) * Q * (z1**3 - z0**3)
    return A, B, D


def _thermal_eigen_resultants(
    layers: List[LayerProps],
    delta_T: float,
) -> ThermalEigen:
    N_T = np.zeros(3)
    M_T = np.zeros(3)
    for l, (z0, z1) in zip(layers, _layer_bounds(layers)):
        Q = l.reduced_stiffness()
        eps_th = (l.cte * delta_T + l.relaxation * l.sinter_eigenstrain) * np.array(
            [1.0, 1.0, 0.0]
        )
        dz = z1 - z0
        N_T += Q @ eps_th * dz
        M_T += Q @ eps_th * 0.5 * (z1**2 - z0**2)
    return N_T, M_T


def solve_laminate(
    layers: List[LayerProps],
    delta_T: float,
) -> LaminateResult:
    A, B, D = _A_B_D(layers)
    N_T, M_T = _thermal_eigen_resultants(layers, delta_T)
    K = np.block([[A, B], [B, D]])
    rhs = np.concatenate([N_T, M_T])
    # Solve K [eps0, kappa] = N_T, M_T
    sol = np.linalg.solve(K, rhs)
    eps0 = sol[:3]
    kappa = sol[3:]
    return LaminateResult(eps0, kappa, A, B, D)


def plate_warp_height_map(
    curvature: np.ndarray,
    grid_shape: Tuple[int, int],
    dims: PlateDims,
    thickness_total: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx, ny = grid_shape
    x = np.linspace(-0.5 * dims.length_x, 0.5 * dims.length_x, nx)
    y = np.linspace(-0.5 * dims.length_y, 0.5 * dims.length_y, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    kx, ky, kxy = curvature
    w = 0.5 * (kx * X**2 + ky * Y**2 + 2.0 * kxy * X * Y)
    z_top = +0.5 * thickness_total + w
    z_bot = -0.5 * thickness_total + w
    return X, Y, w, z_top, z_bot


def voxelize_stress_field(
    layers: List[LayerProps],
    result: LaminateResult,
    grid_shape: Tuple[int, int],
    num_z: int,
    delta_T: float,
) -> Tuple[np.ndarray, np.ndarray]:
    nx, ny = grid_shape
    # Through-thickness grid
    bounds = _layer_bounds(layers)
    z_min = bounds[0][0]
    z_max = bounds[-1][1]
    z = np.linspace(z_min, z_max, num_z)

    # Precompute which layer each z belongs to
    layer_indices = np.zeros(num_z, dtype=int)
    for i, zi in enumerate(z):
        for li, (z0, z1) in enumerate(bounds):
            if zi >= z0 - 1e-12 and zi <= z1 + 1e-12:
                layer_indices[i] = li
                break

    # Stress components: [sxx, syy, sxy, szz, sxz, syz]
    stress = np.zeros((num_z, ny, nx, 6), dtype=np.float32)
    eps0 = result.midplane_strain
    kappa = result.curvature

    for zi_idx, zi in enumerate(z):
        li = layer_indices[zi_idx]
        l = layers[li]
        Q = l.reduced_stiffness()
        # Include both thermal and sintering eigenstrain contributions
        eps_th = (l.cte * delta_T + l.relaxation * l.sinter_eigenstrain) * np.array(
            [1.0, 1.0, 0.0]
        )
        eps_mech = eps0 + zi * kappa - eps_th
        s_inplane = Q @ eps_mech  # (3,)
        # Broadcast into (ny, nx)
        stress[zi_idx, :, :, 0] = s_inplane[0]
        stress[zi_idx, :, :, 1] = s_inplane[1]
        stress[zi_idx, :, :, 2] = s_inplane[2]
        # Plane stress assumption
        # Out-of-plane and shear set to zero in this simplified model
        # szz, sxz, syz already zero

    return z.astype(np.float32), stress
