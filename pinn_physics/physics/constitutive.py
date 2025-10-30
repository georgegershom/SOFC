from __future__ import annotations
from typing import Tuple
import numpy as np
from .constants import MaterialProperties, PhysicalConstants, ElectrochemKinetics


def fourier_heat_flux(temperature_gradient: np.ndarray, thermal_conductivity: float) -> np.ndarray:
    """q = -k * grad(T)
    temperature_gradient: (..., 2)
    returns: (..., 2)
    """
    return -thermal_conductivity * temperature_gradient


def hooke_stress_from_displacement_gradient(
    displacement_gradient: np.ndarray,
    temperature: np.ndarray,
    materials: MaterialProperties,
    plane_stress: bool = True,
) -> np.ndarray:
    """Compute Cauchy stress tensor sigma in 2D from grad u and T using linear thermoelasticity.
    displacement_gradient: (..., 2, 2) with components [du_i/dx_j]
    temperature: (...,)
    returns sigma: (..., 2, 2)
    """
    E = materials.youngs_modulus_E
    nu = materials.poissons_ratio_nu
    alpha_T = materials.thermal_expansion_alpha_T
    T_ref = materials.reference_temperature_T_ref

    # Small-strain tensor: eps = sym(grad u)
    eps = 0.5 * (displacement_gradient + np.swapaxes(displacement_gradient, -1, -2))  # (...,2,2)

    # Thermal strain
    delta_T = np.expand_dims(temperature - T_ref, axis=(-1, -2))  # (...,1,1)
    eps_th = alpha_T * delta_T * np.eye(2)[None, ...]  # (...,2,2)

    # Mechanical strain
    eps_mech = eps - eps_th

    if plane_stress:
        coef = E / (1.0 - nu * nu)
        D = coef * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ])
    else:  # plane strain
        coef = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        D = coef * np.array([
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
        ])

    # Voigt mapping
    eps_xx = eps_mech[..., 0, 0]
    eps_yy = eps_mech[..., 1, 1]
    eps_xy = eps_mech[..., 0, 1]  # = eps_yx
    eps_voigt = np.stack([eps_xx, eps_yy, eps_xy], axis=-1)  # (...,3)

    sig_voigt = eps_voigt @ D.T  # (...,3)

    sigma = np.zeros_like(eps_mech)
    sigma[..., 0, 0] = sig_voigt[..., 0]
    sigma[..., 1, 1] = sig_voigt[..., 1]
    sigma[..., 0, 1] = sig_voigt[..., 2]
    sigma[..., 1, 0] = sig_voigt[..., 2]
    return sigma


def butler_volmer_current_density(
    overpotential_eta: np.ndarray,
    temperature: np.ndarray,
    kinetics: ElectrochemKinetics,
    constants: PhysicalConstants,
) -> np.ndarray:
    """j = j0 * [exp(alpha_a F eta / RT) - exp(-alpha_c F eta / RT)]
    overpotential_eta: (...,)
    temperature: (...,)
    returns j: (...,) A/m^2 (positive anodic by convention)
    """
    R = constants.gas_constant_R
    F = constants.faraday_F
    a_a = kinetics.alpha_anodic
    a_c = kinetics.alpha_cathodic
    j0 = kinetics.exchange_current_density_j0
    coef_a = (a_a * F) / (R * temperature)
    coef_c = (a_c * F) / (R * temperature)
    return j0 * (np.exp(coef_a * overpotential_eta) - np.exp(-coef_c * overpotential_eta))


def joule_heating_density(phi_gradient: np.ndarray, conductivity: float) -> np.ndarray:
    """Q_J = sigma * ||grad(phi)||^2
    phi_gradient: (..., 2)
    returns: (...,)
    """
    return conductivity * np.sum(phi_gradient * phi_gradient, axis=-1)
