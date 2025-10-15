from __future__ import annotations
import os
import json
from typing import Dict, Any, Tuple
import numpy as np
from tqdm import tqdm

from .config import GeneratorConfig
from .sampling import sample_inputs
from .surrogates.thermo_chem_elec import (
    generate_current_density_field,
    generate_temperature_field,
    generate_species_fields,
    generate_overpotentials,
    generate_lf_temperature_1d,
)
from .surrogates.mechanics import compute_mechanical_fields, estimate_time_to_failure_hours
from .fidelity import downsample_average
from .io_utils import ensure_dir, save_metadata_json, write_inputs_csv, save_npz


def _coords_1d(n: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n, dtype=np.float64)


def _coords_3d(shape: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
    nx, ny, nz = shape
    return {"x": _coords_1d(nx), "y": _coords_1d(ny), "z": _coords_1d(nz)}


def generate_all(cfg: GeneratorConfig) -> str:
    out_base = cfg.out_dir
    ensure_dir(out_base)

    rng = np.random.default_rng(cfg.seed)

    meta: Dict[str, Any] = {
        "version": 1,
        "seed": cfg.seed,
        "sizes": {
            "lf": cfg.sizes.lf,
            "mf": cfg.sizes.mf,
            "hf": cfg.sizes.hf,
        },
        "grids": {
            "lf": {"nx": cfg.grids.lf_shape[0]},
            "mf": {"shape": cfg.grids.mf_shape},
            "hf": {"shape": cfg.grids.hf_shape},
        },
        "variables": {
            "T": "Temperature [K]",
            "i": "Current density [A/m^2]",
            "p_H2": "H2 partial pressure [Pa]",
            "p_H2O": "H2O partial pressure [Pa]",
            "p_O2": "O2 partial pressure [Pa]",
            "eta_act": "Activation overpotential [V]",
            "eta_ohm": "Ohmic overpotential [V]",
            "eta_conc": "Concentration overpotential [V]",
            "sigma_vm": "Von Mises stress [Pa]",
            "epsilon_eq": "Equivalent strain [-]",
            "damage": "Damage indicator [0-1]",
            "time_to_failure_hours": "Estimated hours to failure [h]",
        },
    }
    save_metadata_json(os.path.join(out_base, "metadata.json"), meta)

    # Generate LF inputs and outputs
    lf_inputs = sample_inputs(cfg.sizes.lf, cfg.seed + 1)
    write_inputs_csv(os.path.join(out_base, "lf", "inputs.csv"), [s.to_dict() for s in lf_inputs])

    # For LF, we will compute a single HF field instance per sample then downsample to LF 1D T, and use scalars for others
    for s in tqdm(lf_inputs, desc="LF", leave=False):
        seed = cfg.seed + 1000 + s.sample_id
        # small grid for fast derivation of LF 1D
        hf_shape_small = (max(32, cfg.grids.hf_shape[0]//2), max(32, cfg.grids.hf_shape[1]//2), max(8, cfg.grids.hf_shape[2]//2))
        i_field = generate_current_density_field(hf_shape_small, s.i_avg, s.Uf, seed)
        T = generate_temperature_field(hf_shape_small, s.T_in, i_field)

        T1d = generate_lf_temperature_1d(cfg.grids.lf_shape[0], T)
        i_scalar = float(s.i_avg)

        # simple overpotentials as scalars from spatial averages
        over = generate_overpotentials(hf_shape_small, T, i_field)
        eta_act = float(np.mean(over["eta_act"]))
        eta_ohm = float(np.mean(over["eta_ohm"]))
        eta_conc = float(np.mean(over["eta_conc"]))

        mech = compute_mechanical_fields(T, i_field, s.E, s.nu, s.alpha_CTE, s.operating_hours)
        sigma_mean = float(np.mean(mech["sigma_vm"]))
        eps_mean = float(np.mean(mech["epsilon_eq"]))
        dmg_mean = float(np.mean(mech["damage"]))
        ttf = estimate_time_to_failure_hours(mech["sigma_vm"], T)

        arrays = {
            "x": _coords_1d(cfg.grids.lf_shape[0]),
            "T_1d": T1d.astype(np.float32),
            "i_scalar": np.float32(i_scalar),
            "eta_act_scalar": np.float32(eta_act),
            "eta_ohm_scalar": np.float32(eta_ohm),
            "eta_conc_scalar": np.float32(eta_conc),
            "sigma_vm_mean": np.float32(sigma_mean),
            "epsilon_eq_mean": np.float32(eps_mean),
            "damage_mean": np.float32(dmg_mean),
            "time_to_failure_hours": np.float32(ttf),
        }
        save_npz(os.path.join(out_base, "lf", "outputs", f"sample_{s.sample_id:06d}.npz"), arrays)

    # Generate MF samples by sub-sampling LF inputs
    mf_count = cfg.sizes.mf
    mf_inputs = lf_inputs[:mf_count]
    write_inputs_csv(os.path.join(out_base, "mf", "inputs.csv"), [s.to_dict() for s in mf_inputs])
    mf_shape = cfg.grids.mf_shape

    for s in tqdm(mf_inputs, desc="MF", leave=False):
        seed = cfg.seed + 2000 + s.sample_id
        i_field_hf = generate_current_density_field(cfg.grids.hf_shape, s.i_avg, s.Uf, seed)
        T_hf = generate_temperature_field(cfg.grids.hf_shape, s.T_in, i_field_hf)
        T = downsample_average(T_hf, mf_shape)
        i_field = downsample_average(i_field_hf, mf_shape)
        species = generate_species_fields(mf_shape, i_field, s.pressure, s.y_H2_in, s.y_H2O_in, s.y_O2_in, s.Uf)
        over = generate_overpotentials(mf_shape, T, i_field)
        mech = compute_mechanical_fields(T, i_field, s.E, s.nu, s.alpha_CTE, s.operating_hours)
        ttf = estimate_time_to_failure_hours(mech["sigma_vm"], T)

        arrays = {
            **_coords_3d(mf_shape),
            "T": T.astype(np.float32),
            "i": i_field.astype(np.float32),
            "p_H2": species["p_H2"].astype(np.float32),
            "p_H2O": species["p_H2O"].astype(np.float32),
            "p_O2": species["p_O2"].astype(np.float32),
            "eta_act": over["eta_act"].astype(np.float32),
            "eta_ohm": over["eta_ohm"].astype(np.float32),
            "eta_conc": over["eta_conc"].astype(np.float32),
            "sigma_vm": mech["sigma_vm"].astype(np.float32),
            "epsilon_eq": mech["epsilon_eq"].astype(np.float32),
            "damage": mech["damage"].astype(np.float32),
            "time_to_failure_hours": np.float32(ttf),
        }
        save_npz(os.path.join(out_base, "mf", "outputs", f"sample_{s.sample_id:06d}.npz"), arrays)

    # Generate HF samples as a subset of MF
    hf_count = cfg.sizes.hf
    hf_inputs = mf_inputs[:hf_count]
    write_inputs_csv(os.path.join(out_base, "hf", "inputs.csv"), [s.to_dict() for s in hf_inputs])

    for s in tqdm(hf_inputs, desc="HF", leave=False):
        seed = cfg.seed + 3000 + s.sample_id
        i_field = generate_current_density_field(cfg.grids.hf_shape, s.i_avg, s.Uf, seed)
        T = generate_temperature_field(cfg.grids.hf_shape, s.T_in, i_field)
        species = generate_species_fields(cfg.grids.hf_shape, i_field, s.pressure, s.y_H2_in, s.y_H2O_in, s.y_O2_in, s.Uf)
        over = generate_overpotentials(cfg.grids.hf_shape, T, i_field)
        mech = compute_mechanical_fields(T, i_field, s.E, s.nu, s.alpha_CTE, s.operating_hours)
        ttf = estimate_time_to_failure_hours(mech["sigma_vm"], T)

        arrays = {
            **_coords_3d(cfg.grids.hf_shape),
            "T": T.astype(np.float32),
            "i": i_field.astype(np.float32),
            "p_H2": species["p_H2"].astype(np.float32),
            "p_H2O": species["p_H2O"].astype(np.float32),
            "p_O2": species["p_O2"].astype(np.float32),
            "eta_act": over["eta_act"].astype(np.float32),
            "eta_ohm": over["eta_ohm"].astype(np.float32),
            "eta_conc": over["eta_conc"].astype(np.float32),
            "sigma_vm": mech["sigma_vm"].astype(np.float32),
            "epsilon_eq": mech["epsilon_eq"].astype(np.float32),
            "damage": mech["damage"].astype(np.float32),
            "time_to_failure_hours": np.float32(ttf),
        }
        save_npz(os.path.join(out_base, "hf", "outputs", f"sample_{s.sample_id:06d}.npz"), arrays)

    return out_base
