from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from .constants import MaterialProperties, PhysicalConstants, ElectrochemKinetics
from .constitutive import fourier_heat_flux, joule_heating_density, hooke_stress_from_displacement_gradient


def residual_charge_conservation(
    phi_gradient: np.ndarray,
    conductivity: float,
) -> np.ndarray:
    """r = div( conductivity * grad(phi) )
    For PINN, this is evaluated by auto-diff. Here we provide the divergence-of-flux form
    so that downstream can compute divergence by AD on the flux function.
    Returns flux for convenience; divergence should be taken externally.
    phi_gradient: (...,2)
    returns J: (...,2) current density vector = -conductivity * grad(phi)
    """
    # Ohm's law: J = - sigma grad(phi)
    return -conductivity * phi_gradient


def residual_energy_conservation(
    temperature: np.ndarray,
    temperature_t: np.ndarray | None,
    temperature_gradient: np.ndarray,
    phi_e_gradient: np.ndarray,
    phi_i_gradient: np.ndarray,
    materials: MaterialProperties,
    include_transient: bool = True,
    reaction_heat_density: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Energy equation residual in flux form.
    rho c_p dT/dt = div(k grad T) + Q_J + Q_rxn
    Returns:
      q_T: (...,2) = -k grad T (heat flux)
      s_T: (...,) source term density = Q_J + Q_rxn - rho c_p dT/dt (moved to RHS)
    Downstream should enforce div(-q_T) - s_T = 0.
    """
    k = materials.thermal_conductivity_k
    rho = materials.density_rho
    cp = materials.heat_capacity_cp

    q_T = fourier_heat_flux(temperature_gradient, k)  # (...,2)

    Q_joule = (
        joule_heating_density(phi_e_gradient, materials.electronic_conductivity_sigma_e)
        + joule_heating_density(phi_i_gradient, materials.ionic_conductivity_kappa_i)
    )

    if reaction_heat_density is None:
        reaction_heat_density = 0.0

    if include_transient and temperature_t is not None:
        s_T = Q_joule + reaction_heat_density - rho * cp * temperature_t
    else:
        s_T = Q_joule + reaction_heat_density

    return q_T, s_T


def residual_linear_momentum(
    displacement_gradient: np.ndarray,
    temperature: np.ndarray,
    materials: MaterialProperties,
) -> np.ndarray:
    """Return Cauchy stress sigma; PINN should enforce div(sigma) + b = 0.
    b (body force) is 0 by default in this low-fidelity setup.
    """
    sigma = hooke_stress_from_displacement_gradient(displacement_gradient, temperature, materials)
    return sigma


def butler_volmer_interface_flux(
    phi_e: np.ndarray,
    phi_i: np.ndarray,
    temperature: np.ndarray,
    kinetics: ElectrochemKinetics,
    constants: PhysicalConstants,
    U_eq_override: float | None = None,
) -> np.ndarray:
    """Interfacial reaction current density (A/m^2), positive anodic.
    Enforces: n·(-sigma_e grad phi_e) = j_rxn, and n·(-kappa_i grad phi_i) = -j_rxn on the interface.
    Here we return j_rxn given local overpotential eta.
    eta = (phi_e - phi_i) - U_eq(T)
    """
    U_eq = kinetics.open_circuit_potential_U_eq if U_eq_override is None else U_eq_override
    eta = (phi_e - phi_i) - U_eq

    # Use constitutive function (duplicated here to avoid circular import)
    R = constants.gas_constant_R
    F = constants.faraday_F
    a_a = kinetics.alpha_anodic
    a_c = kinetics.alpha_cathodic
    j0 = kinetics.exchange_current_density_j0

    coef_a = (a_a * F) / (R * temperature)
    coef_c = (a_c * F) / (R * temperature)
    j_rxn = j0 * (np.exp(coef_a * eta) - np.exp(-coef_c * eta))
    return j_rxn
