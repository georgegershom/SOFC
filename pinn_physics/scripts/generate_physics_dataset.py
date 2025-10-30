from __future__ import annotations
import json
import os
from typing import Dict, Any
import numpy as np

from physics.constants import DefaultConfig
from physics.sampling import sample_interior, sample_boundary_edges, sample_internal_interface


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_physics_spec(config: DefaultConfig) -> Dict[str, Any]:
    c = config.constants
    d = config.domain
    m = config.materials
    k = config.kinetics
    h = config.convection

    spec: Dict[str, Any] = {
        "fields": [
            {"name": "T", "description": "temperature (K)"},
            {"name": "u_x", "description": "x displacement (m)"},
            {"name": "u_y", "description": "y displacement (m)"},
            {"name": "phi_e", "description": "electronic potential (V)"},
            {"name": "phi_i", "description": "ionic potential (V)"},
        ],
        "pdes": [
            {
                "name": "charge_conservation_electronic",
                "form": "div( sigma_e * grad(phi_e) ) = 0",
                "unknowns": ["phi_e"],
                "parameters": {"sigma_e": m.electronic_conductivity_sigma_e},
            },
            {
                "name": "charge_conservation_ionic",
                "form": "div( kappa_i * grad(phi_i) ) = 0",
                "unknowns": ["phi_i"],
                "parameters": {"kappa_i": m.ionic_conductivity_kappa_i},
            },
            {
                "name": "energy_conservation",
                "form": "rho*cp*dT/dt = div(k*grad(T)) + Q_joule + Q_rxn",
                "unknowns": ["T"],
                "parameters": {
                    "k": m.thermal_conductivity_k,
                    "rho": m.density_rho,
                    "cp": m.heat_capacity_cp,
                },
                "sources": {
                    "Q_joule": "sigma_e*|grad(phi_e)|^2 + kappa_i*|grad(phi_i)|^2",
                    "Q_rxn": "eta * j_rxn (optional, at interfaces)",
                },
            },
            {
                "name": "linear_momentum_equilibrium",
                "form": "div( sigma(u, T) ) = 0",
                "unknowns": ["u_x", "u_y"],
                "parameters": {
                    "E": m.youngs_modulus_E,
                    "nu": m.poissons_ratio_nu,
                    "alpha_T": m.thermal_expansion_alpha_T,
                    "T_ref": m.reference_temperature_T_ref,
                },
                "constitutive": "Hooke's law (plane stress default)",
            },
        ],
        "constitutive_laws": [
            {"name": "Fourier", "law": "q = -k * grad(T)"},
            {"name": "Hooke", "law": "sigma = C : (eps - alpha_T*(T-T_ref) I)"},
            {
                "name": "Butler-Volmer",
                "law": "j = j0[exp(alpha_a F eta/RT) - exp(-alpha_c F eta/RT)]",
                "parameters": {
                    "j0": k.exchange_current_density_j0,
                    "alpha_a": k.alpha_anodic,
                    "alpha_c": k.alpha_cathodic,
                },
            },
        ],
        "boundary_conditions": [
            {
                "field": "T",
                "type": "Dirichlet",
                "location": "left edge",
                "equation": "T = T_left",
                "values": {"T_left": m.reference_temperature_T_ref + 12.0},
            },
            {
                "field": "T",
                "type": "Convective",
                "location": "right edge",
                "equation": "-k * grad(T)·n = h (T - T_inf)",
                "values": {"h": h.h_coefficient, "T_inf": h.ambient_temperature_T_inf},
            },
            {
                "field": "T",
                "type": "Insulated",
                "location": "top/bottom",
                "equation": "grad(T)·n = 0",
                "values": {},
            },
            {
                "field": "u",
                "type": "Dirichlet",
                "location": "left edge",
                "equation": "u = 0",
                "values": {},
            },
            {
                "field": "u",
                "type": "Traction-free",
                "location": "top/right/bottom",
                "equation": "sigma·n = 0",
                "values": {},
            },
            {
                "field": "phi_e",
                "type": "Dirichlet",
                "location": "left edge",
                "equation": "phi_e = V_app",
                "values": {"V_app": 1.0},
            },
            {
                "field": "phi_e",
                "type": "Dirichlet",
                "location": "right edge",
                "equation": "phi_e = 0",
                "values": {},
            },
            {
                "field": "phi_i",
                "type": "Neumann (no-flux)",
                "location": "top/bottom",
                "equation": "-kappa_i grad(phi_i)·n = 0",
                "values": {},
            },
            {
                "field": "phi_e & phi_i",
                "type": "Butler-Volmer interface",
                "location": "internal line x=Lx/2",
                "equation": "j = j0[exp(alpha_a F eta/RT) - exp(-alpha_c F eta/RT)]",
                "coupling": [
                    "-sigma_e grad(phi_e)·n = j",
                    "+kappa_i grad(phi_i)·n = j",
                ],
                "values": {"U_eq": k.open_circuit_potential_U_eq},
            },
        ],
        "domain": {
            "Lx": d.length_x,
            "Ly": d.length_y,
            "time": {"enabled": d.has_time, "t_max": d.time_max},
        },
        "constants": {"R": c.gas_constant_R, "F": c.faraday_F},
    }
    return spec


def main() -> None:
    config = DefaultConfig()

    out_dir = os.path.join("/workspace", "pinn_physics", "datasets", "low_fidelity_physics_v1")
    ensure_dir(out_dir)

    # 1) Sampling
    interior = sample_interior(config.domain, n_points=20000, seed=42)
    edges = sample_boundary_edges(config.domain, n_per_edge=2000, seed=123)
    interface = sample_internal_interface(config.domain, x_location=config.domain.length_x * 0.5, n_points=3000, seed=7)

    # 2) Boundary value assignments (constants per spec)
    T_left = config.materials.reference_temperature_T_ref + 12.0
    V_app = 1.0

    # Pack boundary sets by field
    thermal = {
        "dirichlet_coords": edges["left"]["coords"],
        "dirichlet_T": np.full((edges["left"]["coords"].shape[0], 1), T_left),
        "convective_coords": edges["right"]["coords"],
        "convective_normal": edges["right"]["normal"],
        "convective_h": np.full((edges["right"]["coords"].shape[0], 1), config.convection.h_coefficient),
        "convective_Tinf": np.full((edges["right"]["coords"].shape[0], 1), config.convection.ambient_temperature_T_inf),
        "insulated_coords": np.vstack([edges["top"]["coords"], edges["bottom"]["coords"]]),
        "insulated_normal": np.vstack([edges["top"]["normal"], edges["bottom"]["normal"]]),
    }

    mechanical = {
        "dirichlet_coords": edges["left"]["coords"],
        "dirichlet_u": np.zeros((edges["left"]["coords"].shape[0], 2), dtype=np.float64),
        "tractionfree_coords": np.vstack([edges["top"]["coords"], edges["right"]["coords"], edges["bottom"]["coords"]]),
        "tractionfree_normal": np.vstack([edges["top"]["normal"], edges["right"]["normal"], edges["bottom"]["normal"]]),
    }

    electronic = {
        "dirichlet_left_coords": edges["left"]["coords"],
        "dirichlet_left_phi": np.full((edges["left"]["coords"].shape[0], 1), V_app),
        "dirichlet_right_coords": edges["right"]["coords"],
        "dirichlet_right_phi": np.zeros((edges["right"]["coords"].shape[0], 1), dtype=np.float64),
    }

    ionic = {
        "noflux_coords": np.vstack([edges["top"]["coords"], edges["bottom"]["coords"]]),
        "noflux_normal": np.vstack([edges["top"]["normal"], edges["bottom"]["normal"]]),
    }

    interface_set = {
        "coords": interface["coords"],
        "normal": interface["normal"],
        "U_eq": np.full((interface["coords"].shape[0], 1), config.kinetics.open_circuit_potential_U_eq),
        "j0": np.full((interface["coords"].shape[0], 1), config.kinetics.exchange_current_density_j0),
        "alpha_a": np.full((interface["coords"].shape[0], 1), config.kinetics.alpha_anodic),
        "alpha_c": np.full((interface["coords"].shape[0], 1), config.kinetics.alpha_cathodic),
    }

    # 3) Save arrays
    npz_path = os.path.join(out_dir, "dataset_v1.npz")
    np.savez_compressed(
        npz_path,
        interior=interior,
        thermal_dirichlet_coords=thermal["dirichlet_coords"],
        thermal_dirichlet_T=thermal["dirichlet_T"],
        thermal_convective_coords=thermal["convective_coords"],
        thermal_convective_normal=thermal["convective_normal"],
        thermal_convective_h=thermal["convective_h"],
        thermal_convective_Tinf=thermal["convective_Tinf"],
        thermal_insulated_coords=thermal["insulated_coords"],
        thermal_insulated_normal=thermal["insulated_normal"],
        mech_dirichlet_coords=mechanical["dirichlet_coords"],
        mech_dirichlet_u=mechanical["dirichlet_u"],
        mech_tractionfree_coords=mechanical["tractionfree_coords"],
        mech_tractionfree_normal=mechanical["tractionfree_normal"],
        elec_dirichlet_left_coords=electronic["dirichlet_left_coords"],
        elec_dirichlet_left_phi=electronic["dirichlet_left_phi"],
        elec_dirichlet_right_coords=electronic["dirichlet_right_coords"],
        elec_dirichlet_right_phi=electronic["dirichlet_right_phi"],
        ion_noflux_coords=ionic["noflux_coords"],
        ion_noflux_normal=ionic["noflux_normal"],
        interface_coords=interface_set["coords"],
        interface_normal=interface_set["normal"],
        interface_U_eq=interface_set["U_eq"],
        interface_j0=interface_set["j0"],
        interface_alpha_a=interface_set["alpha_a"],
        interface_alpha_c=interface_set["alpha_c"],
    )

    # 4) Save physics spec JSON
    spec = build_physics_spec(config)
    with open(os.path.join(out_dir, "physics_spec.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)

    # 5) Save params JSON for quick reference
    params = {
        "constants": {
            "R": config.constants.gas_constant_R,
            "F": config.constants.faraday_F,
        },
        "domain": {
            "Lx": config.domain.length_x,
            "Ly": config.domain.length_y,
            "has_time": config.domain.has_time,
            "time_max": config.domain.time_max,
        },
        "materials": {
            "sigma_e": config.materials.electronic_conductivity_sigma_e,
            "kappa_i": config.materials.ionic_conductivity_kappa_i,
            "k": config.materials.thermal_conductivity_k,
            "rho": config.materials.density_rho,
            "cp": config.materials.heat_capacity_cp,
            "E": config.materials.youngs_modulus_E,
            "nu": config.materials.poissons_ratio_nu,
            "alpha_T": config.materials.thermal_expansion_alpha_T,
            "T_ref": config.materials.reference_temperature_T_ref,
        },
        "kinetics": {
            "j0": config.kinetics.exchange_current_density_j0,
            "alpha_a": config.kinetics.alpha_anodic,
            "alpha_c": config.kinetics.alpha_cathodic,
            "U_eq": config.kinetics.open_circuit_potential_U_eq,
        },
        "convection": {
            "h": config.convection.h_coefficient,
            "T_inf": config.convection.ambient_temperature_T_inf,
        },
    }
    with open(os.path.join(out_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    print(f"Saved dataset to: {npz_path}")


if __name__ == "__main__":
    main()
