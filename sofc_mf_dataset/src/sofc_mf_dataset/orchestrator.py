from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any
import numpy as np
from tqdm import tqdm

from .config import GeneratorConfig
from .sampling import sample_inputs, InputSample
from .physics.thermo import synthesize_temperature_fields, synthesize_current_fields
from .physics.electrochem import species_fields, overpotentials
from .physics.mechanics import stress_strain_damage
from .io.writer import save_sample_npz, ensure_dir
from .io.index import append_index_csv


def _meta_from_input(sample: InputSample) -> Dict[str, Any]:
    meta = {name: val for name, val in zip(InputSample.names(), sample.as_vector())}
    return meta


def _arrays_for_lf(sample: InputSample, cfg: GeneratorConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    T = synthesize_temperature_fields(sample, cfg, rng)
    I = synthesize_current_fields(sample, cfg, T["mf_2d"], T["hf_3d"], rng)

    return {
        # Thermo
        "thermo.T.lf_1d": T["lf_1d"],
        "thermo.i.lf_scalar": I["lf_scalar"],
    }


def _arrays_for_mf(sample: InputSample, cfg: GeneratorConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    T = synthesize_temperature_fields(sample, cfg, rng)
    I = synthesize_current_fields(sample, cfg, T["mf_2d"], T["hf_3d"], rng)
    S = species_fields(sample, cfg, T["mf_2d"].shape, T["hf_3d"].shape, rng)
    O = overpotentials(sample, cfg, I["mf_2d"], I["hf_3d"], T["mf_2d"], T["hf_3d"], S)

    M = stress_strain_damage(sample, cfg, T["mf_2d"], T["hf_3d"], O, rng)

    arrays: Dict[str, np.ndarray] = {}
    # Thermo-Chemo-Electrical
    arrays.update({
        "thermo.T.mf_2d": T["mf_2d"],
        "thermo.T.hf_3d": T["hf_3d"],
        "thermo.i.mf_2d": I["mf_2d"],
        "thermo.i.hf_3d": I["hf_3d"],
        "species.h2.mf_2d": S["h2_2d"],
        "species.h2o.mf_2d": S["h2o_2d"],
        "species.o2.mf_2d": S["o2_2d"],
        "species.h2.hf_3d": S["h2_3d"],
        "species.h2o.hf_3d": S["h2o_3d"],
        "species.o2.hf_3d": S["o2_3d"],
        "overpot.eta_act.mf_2d": O["eta_act_2d"],
        "overpot.eta_ohm.mf_2d": O["eta_ohm_2d"],
        "overpot.eta_conc.mf_2d": O["eta_conc_2d"],
        "overpot.eta_act.hf_3d": O["eta_act_3d"],
        "overpot.eta_ohm.hf_3d": O["eta_ohm_3d"],
        "overpot.eta_conc.hf_3d": O["eta_conc_3d"],
    })

    # Mechanical
    arrays.update({
        "mech.sigma_vm.mf_2d": M["sigma_vm_2d"],
        "mech.sigma_vm.hf_3d": M["sigma_vm_3d"],
        "mech.strain.mf_2d": M["strain_2d"],
        "mech.strain.hf_3d": M["strain_3d"],
        "mech.damage.mf_2d": M["damage_2d"],
        "mech.damage_bin.mf_2d": M["damage_2d_binary"],
        "mech.damage.hf_3d": M["damage_3d"],
        "mech.damage_bin.hf_3d": M["damage_3d_binary"],
        "mech.delamination_mid_G": M["delamination_mid_G"],
        "mech.delamination_mid_prob": M["delamination_mid_prob"],
        "mech.ni_coarsening.hf_3d": M["ni_coarsening_3d"],
        "mech.sigma_vm.avg": M["sigma_vm_avg"],
        "mech.strain.avg": M["strain_avg"],
        "life.time_to_failure_hours": M["time_to_failure_hours"],
    })

    # Also include LF scalars for convenience
    arrays.update({
        "thermo.T.lf_1d": synthesize_temperature_fields(sample, cfg, rng)["lf_1d"],
        "thermo.i.lf_scalar": np.array([sample.average_current_Apcm2], dtype=np.float32),
    })

    return arrays


def generate_dataset(cfg: GeneratorConfig) -> Path:
    ensure_dir(cfg.out_dir)

    # Plan subsets
    assert cfg.num_mf <= cfg.num_lf and cfg.num_hf <= cfg.num_mf

    rng = np.random.default_rng(cfg.seed)

    # Sample LF inputs
    lf_inputs = sample_inputs(cfg.num_lf, seed=rng.integers(0, 2**32 - 1))

    # Select MF/HF subsets
    mf_indices = rng.choice(cfg.num_lf, size=cfg.num_mf, replace=False)
    hf_indices = rng.choice(mf_indices, size=cfg.num_hf, replace=False)

    index_rows = []

    # Generate LF
    for idx, sample in enumerate(tqdm(lf_inputs, desc="LF")):
        arrays = _arrays_for_lf(sample, cfg, rng)
        meta = _meta_from_input(sample)
        meta.update({
            "has_mf": bool(idx in mf_indices),
            "has_hf": bool(idx in hf_indices),
            "fidelity": "LF",
        })
        path = save_sample_npz(cfg.out_dir / "lf", idx, arrays, meta)
        index_rows.append({"sample_id": idx, "phase": "LF", "npz": str(path), **meta})

    # Generate MF/HF for the selected subset
    for idx in tqdm(mf_indices.tolist(), desc="MF/HF"):
        sample = lf_inputs[idx]
        arrays = _arrays_for_mf(sample, cfg, rng)
        meta = _meta_from_input(sample)
        has_hf = bool(idx in hf_indices)
        meta.update({
            "has_mf": True,
            "has_hf": has_hf,
            "fidelity": "MFHF",
        })
        path = save_sample_npz(cfg.out_dir / "mf_hf", idx, arrays, meta)
        index_rows.append({"sample_id": idx, "phase": "MFHF", "npz": str(path), **meta})

    append_index_csv(cfg.out_dir / "index.csv", index_rows)

    return cfg.out_dir
