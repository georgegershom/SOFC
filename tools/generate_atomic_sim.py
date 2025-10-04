#!/usr/bin/env python3
"""
Synthetic Atomic-Scale Simulation Dataset Generator

This script fabricates plausible DFT- and MD-like numerical outputs for use in
surrogate modeling, phase-field parameterization, or crystal plasticity inputs.

Outputs are placed under the provided output directory (default: data/atomic_sim):

- dft/
  - formation_energies.csv
  - grain_boundary_energies.csv
  - surface_energies.csv
  - activation_barriers.csv
  - neb_curves/
    - neb_curve_<process>_<material>_<path_id>.csv

- md/
  - grain_boundary_sliding/
    - gbs_summary.csv
    - run_<id>.csv (stress-strain curves per run)
  - dislocation_mobility/
    - mobility_summary.csv
    - run_<id>.csv (time series of position/velocity)

- manifest.json: file inventory and high-level schemas
- metadata.json: generator metadata (seed, version, timestamp)

The values are NOT results of real simulations; they are statistically fabricated
but aim for reasonable magnitudes and trends to support downstream model development.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

GENERATOR_VERSION = "1.0.0"


@dataclass
class GeneratorConfig:
    output_dir: str
    seed: int
    num_neb_paths_per_process: int = 8
    num_gbs_runs: int = 24
    num_dislocation_runs: int = 24


def ensure_directory_exists(directory_path: str) -> None:
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def write_csv(file_path: str, header: List[str], rows: List[List[object]]) -> None:
    ensure_directory_exists(os.path.dirname(file_path))
    with open(file_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_json(file_path: str, payload: Dict) -> None:
    ensure_directory_exists(os.path.dirname(file_path))
    with open(file_path, "w") as handle:
        json.dump(payload, handle, indent=2)


class SyntheticAtomicDataGenerator:
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.random = random.Random(config.seed)
        self.manifest: Dict[str, Dict] = {"files": {}, "schemas": {}}

        self.materials = [
            # Representative structural metals
            {"name": "Fe", "lattice": "bcc", "baseline_Ev": 1.9, "baseline_gamma_surf": 2.45},
            {"name": "Ni", "lattice": "fcc", "baseline_Ev": 1.6, "baseline_gamma_surf": 2.35},
            {"name": "Al", "lattice": "fcc", "baseline_Ev": 0.75, "baseline_gamma_surf": 1.10},
            {"name": "Ti", "lattice": "hcp", "baseline_Ev": 1.8, "baseline_gamma_surf": 1.95},
            {"name": "Cu", "lattice": "fcc", "baseline_Ev": 1.3, "baseline_gamma_surf": 1.80},
        ]

        self.gb_catalog = [
            # Coincidence site lattice Σ values and example planes
            {"sigma": 5, "plane": "(210)"},
            {"sigma": 11, "plane": "(113)"},
            {"sigma": 13, "plane": "(510)"},
            {"sigma": 17, "plane": "(410)"},
        ]

        self.surfaces = ["(100)", "(110)", "(111)", "(112)"]

        self.diffusion_processes = [
            {"name": "vacancy_migration", "mean_barrier": 0.85, "std": 0.15},
            {"name": "solute_drag", "mean_barrier": 0.35, "std": 0.10},
            {"name": "interstitial_migration", "mean_barrier": 0.55, "std": 0.12},
        ]

    # --------------------------- Public API ---------------------------
    def run(self) -> None:
        dft_dir = os.path.join(self.config.output_dir, "dft")
        md_dir = os.path.join(self.config.output_dir, "md")
        ensure_directory_exists(dft_dir)
        ensure_directory_exists(md_dir)

        self._generate_dft_data(dft_dir)
        self._generate_md_data(md_dir)

        self._write_metadata_and_manifest()

    # --------------------------- DFT data -----------------------------
    def _generate_dft_data(self, dft_dir: str) -> None:
        formation_rows: List[List[object]] = []
        gb_energy_rows: List[List[object]] = []
        surface_rows: List[List[object]] = []
        barrier_rows: List[List[object]] = []

        # Defect formation energies
        defect_types = ["vacancy", "interstitial", "substitutional"]
        environments = ["bulk", "grain_boundary", "surface"]

        for material in self.materials:
            for defect in defect_types:
                for env in environments:
                    base_ev = material["baseline_Ev"]
                    env_factor = {"bulk": 1.0, "grain_boundary": 0.85, "surface": 0.70}[env]
                    defect_shift = {"vacancy": 1.0, "interstitial": 0.8, "substitutional": 1.2}[defect]
                    noise = self.random.gauss(0.0, 0.08)
                    energy_eV = max(0.05, base_ev * env_factor * defect_shift + noise)
                    uncertainty_eV = abs(self.random.gauss(0.04, 0.02))

                    formation_rows.append([
                        material["name"],
                        material["lattice"],
                        defect,
                        env,
                        round(energy_eV, 4),
                        round(uncertainty_eV, 4),
                        300,  # nominal T for static DFT in K
                        "synthetic"
                    ])

        # Grain boundary energies (J/m^2)
        for material in self.materials:
            for gb in self.gb_catalog:
                # Simple trend: higher Σ often higher energy; hcp slightly lower GB energy on average
                sigma_factor = 0.15 + 0.015 * math.log(gb["sigma"])  # gentle increase with Σ
                lattice_adjust = {"bcc": 1.00, "fcc": 0.95, "hcp": 0.90}[material["lattice"]]
                noise = self.random.gauss(0.0, 0.05)
                gamma_J_m2 = max(0.2, lattice_adjust * (0.6 + sigma_factor) + noise)
                gb_energy_rows.append([
                    material["name"],
                    material["lattice"],
                    gb["sigma"],
                    gb["plane"],
                    round(gamma_J_m2, 4),
                    "synthetic"
                ])

        # Surface energies (J/m^2)
        for material in self.materials:
            for surf in self.surfaces:
                miller_factor = {
                    "(111)": 1.00,
                    "(100)": 1.10,
                    "(110)": 1.05,
                    "(112)": 1.15,
                }.get(surf, 1.05)
                noise = self.random.gauss(0.0, 0.05)
                gamma = max(0.4, material["baseline_gamma_surf"] * miller_factor + noise)
                surface_rows.append([
                    material["name"],
                    material["lattice"],
                    surf,
                    round(gamma, 4),
                    "synthetic"
                ])

        # Activation barrier table and NEB curves
        neb_dir = os.path.join(dft_dir, "neb_curves")
        ensure_directory_exists(neb_dir)

        neb_curve_files: List[str] = []
        for material in self.materials:
            for proc in self.diffusion_processes:
                for path_idx in range(self.config.num_neb_paths_per_process):
                    barrier = max(
                        0.02,
                        self.random.gauss(proc["mean_barrier"], proc["std"]) *
                        {
                            "bcc": 1.05,
                            "fcc": 0.95,
                            "hcp": 1.00,
                        }[material["lattice"]]
                    )
                    prefactor_THz = max(0.1, self.random.gauss(5.0, 1.5))

                    path_id = f"{proc['name']}_{material['name']}_{path_idx:02d}"
                    barrier_rows.append([
                        material["name"],
                        material["lattice"],
                        proc["name"],
                        path_id,
                        round(barrier, 4),
                        round(prefactor_THz, 3),
                        "synthetic"
                    ])

                    # Write NEB curve for this barrier
                    curve_path = os.path.join(neb_dir, f"neb_curve_{path_id}.csv")
                    neb_curve_files.append(os.path.relpath(curve_path, self.config.output_dir))

                    reaction_coords = [i / 20.0 for i in range(21)]  # 21 images, s∈[0,1]
                    rows = [["reaction_coordinate", "energy_eV"]]
                    for s in reaction_coords:
                        # Symmetric barrier shape with small roughness
                        energy = barrier * (4.0 * s * (1.0 - s))
                        energy += self.random.gauss(0.0, 0.01)
                        rows.append([round(s, 4), round(max(0.0, energy), 5)])

                    # Manually write CSV with header included
                    ensure_directory_exists(os.path.dirname(curve_path))
                    with open(curve_path, "w", newline="") as fh:
                        writer = csv.writer(fh)
                        writer.writerows(rows)

        # Persist DFT tables
        formation_path = os.path.join(dft_dir, "formation_energies.csv")
        gb_energy_path = os.path.join(dft_dir, "grain_boundary_energies.csv")
        surface_path = os.path.join(dft_dir, "surface_energies.csv")
        barriers_path = os.path.join(dft_dir, "activation_barriers.csv")

        write_csv(
            formation_path,
            [
                "material", "lattice", "defect_type", "environment",
                "formation_energy_eV", "uncertainty_eV", "temperature_K", "method"
            ],
            formation_rows,
        )
        write_csv(
            gb_energy_path,
            ["material", "lattice", "sigma", "plane", "gamma_J_per_m2", "method"],
            gb_energy_rows,
        )
        write_csv(
            surface_path,
            ["material", "lattice", "surface_hkl", "gamma_J_per_m2", "method"],
            surface_rows,
        )
        write_csv(
            barriers_path,
            ["material", "lattice", "process", "path_id", "barrier_eV", "prefactor_THz", "method"],
            barrier_rows,
        )

        # Update manifest
        self._add_to_manifest(
            os.path.relpath(formation_path, self.config.output_dir),
            schema={
                "material": "str",
                "lattice": "str",
                "defect_type": "str",
                "environment": "str",
                "formation_energy_eV": "float",
                "uncertainty_eV": "float",
                "temperature_K": "int",
                "method": "str",
            },
        )
        self._add_to_manifest(
            os.path.relpath(gb_energy_path, self.config.output_dir),
            schema={
                "material": "str",
                "lattice": "str",
                "sigma": "int",
                "plane": "str",
                "gamma_J_per_m2": "float",
                "method": "str",
            },
        )
        self._add_to_manifest(
            os.path.relpath(surface_path, self.config.output_dir),
            schema={
                "material": "str",
                "lattice": "str",
                "surface_hkl": "str",
                "gamma_J_per_m2": "float",
                "method": "str",
            },
        )
        self._add_to_manifest(
            os.path.relpath(barriers_path, self.config.output_dir),
            schema={
                "material": "str",
                "lattice": "str",
                "process": "str",
                "path_id": "str",
                "barrier_eV": "float",
                "prefactor_THz": "float",
                "method": "str",
            },
        )
        for rel_curve_path in neb_curve_files:
            self._add_to_manifest(rel_curve_path, schema={"reaction_coordinate": "float", "energy_eV": "float"})

    # --------------------------- MD data ------------------------------
    def _generate_md_data(self, md_dir: str) -> None:
        self._generate_md_gb_sliding(os.path.join(md_dir, "grain_boundary_sliding"))
        self._generate_md_dislocation_mobility(os.path.join(md_dir, "dislocation_mobility"))

    def _generate_md_gb_sliding(self, out_dir: str) -> None:
        ensure_directory_exists(out_dir)
        rows_summary: List[List[object]] = []

        # Choose combinations
        temperatures_K = [600, 800, 1000]  # intermediate to high temperatures
        strain_rates_s = [1e6, 1e7]  # synthetic values for illustration

        run_counter = 0
        for material in self.materials:
            for gb in self.gb_catalog:
                for T in temperatures_K:
                    for strain_rate in strain_rates_s:
                        run_counter += 1
                        run_id = f"gbs_{material['name']}_S{gb['sigma']}_{T}K_{int(strain_rate):d}s-1_{run_counter:03d}"
                        curve_path = os.path.join(out_dir, f"run_{run_id}.csv")

                        # Parametrize peak stress as decreasing with temperature and Σ (loosely)
                        base_peak = {"bcc": 2.2, "fcc": 1.8, "hcp": 2.0}[material["lattice"]]
                        sigma_factor = 1.0 - 0.02 * math.log(gb["sigma"])  # slightly weaker for higher Σ
                        temp_factor = max(0.3, 1.4 - (T - 600) / 1000.0)
                        rate_factor = 1.0 + 0.08 * math.log10(max(1.0, strain_rate))
                        tau_peak = max(0.2, base_peak * sigma_factor * temp_factor * (0.6 + 0.4 * self.random.random()))
                        tau_peak *= 1.0 + 0.03 * (math.log10(strain_rate) - 6.0)  # modest rate strengthening

                        gamma0 = 0.01 + 0.01 * self.random.random()  # characteristic strain for hardening
                        soft_rate = 6.0 + 2.0 * self.random.random()  # softening rate constant

                        # Generate stress-strain curve
                        header = [
                            "shear_strain",
                            "shear_stress_GPa",
                            "temperature_K",
                            "strain_rate_s^-1",
                            "gb_sigma",
                            "gb_plane",
                            "material",
                            "run_id",
                        ]
                        rows: List[List[object]] = []

                        num_points = 121
                        for i in range(num_points):
                            shear_strain = i * 0.002  # up to ~0.24
                            # Simple elasto-plastic-like response with peak then softening
                            tau = tau_peak * (1.0 - math.exp(-shear_strain / max(1e-6, gamma0)))
                            tau *= math.exp(-shear_strain / soft_rate)
                            tau += self.random.gauss(0.0, 0.03)
                            tau = max(0.0, tau)

                            rows.append([
                                round(shear_strain, 5),
                                round(tau, 5),
                                T,
                                float(strain_rate),
                                gb["sigma"],
                                gb["plane"],
                                material["name"],
                                run_id,
                            ])

                        # Write curve file
                        ensure_directory_exists(os.path.dirname(curve_path))
                        with open(curve_path, "w", newline="") as fh:
                            writer = csv.writer(fh)
                            writer.writerow(header)
                            writer.writerows(rows)

                        # Summaries
                        shear_stresses = [r[1] for r in rows]
                        peak_tau = max(shear_stresses)
                        yield_like = next((r[1] for r in rows if r[0] >= gamma0), peak_tau * 0.7)

                        rows_summary.append([
                            material["name"],
                            gb["sigma"],
                            gb["plane"],
                            T,
                            float(strain_rate),
                            round(peak_tau, 5),
                            round(yield_like, 5),
                            run_id,
                        ])

        summary_path = os.path.join(out_dir, "gbs_summary.csv")
        write_csv(
            summary_path,
            [
                "material",
                "gb_sigma",
                "gb_plane",
                "temperature_K",
                "strain_rate_s^-1",
                "peak_shear_stress_GPa",
                "yield_like_shear_GPa",
                "run_id",
            ],
            rows_summary,
        )

        # Manifest entries
        self._add_to_manifest(
            os.path.relpath(summary_path, self.config.output_dir),
            schema={
                "material": "str",
                "gb_sigma": "int",
                "gb_plane": "str",
                "temperature_K": "int",
                "strain_rate_s^-1": "float",
                "peak_shear_stress_GPa": "float",
                "yield_like_shear_GPa": "float",
                "run_id": "str",
            },
        )
        # Add all run files to manifest
        for fname in os.listdir(out_dir):
            if fname.startswith("run_") and fname.endswith(".csv"):
                self._add_to_manifest(
                    os.path.relpath(os.path.join(out_dir, fname), self.config.output_dir),
                    schema={
                        "shear_strain": "float",
                        "shear_stress_GPa": "float",
                        "temperature_K": "int",
                        "strain_rate_s^-1": "float",
                        "gb_sigma": "int",
                        "gb_plane": "str",
                        "material": "str",
                        "run_id": "str",
                    },
                )

    def _generate_md_dislocation_mobility(self, out_dir: str) -> None:
        ensure_directory_exists(out_dir)
        rows_summary: List[List[object]] = []

        characters = ["edge", "screw", "mixed"]
        temperatures_K = [500, 700, 900]
        solute_levels_at_pct = [0.0, 0.5, 1.0]  # atomic percent
        tau_values_GPa = [0.05, 0.1, 0.2, 0.4]

        run_uuid_ns = uuid.uuid4().hex[:8]
        run_index = 0

        for material in self.materials:
            for character in characters:
                for T in temperatures_K:
                    for solute in solute_levels_at_pct:
                        for tau in tau_values_GPa:
                            run_index += 1
                            run_id = f"dl_{material['name']}_{character}_{T}K_{solute:.1f}at_{tau:.2f}GPa_{run_uuid_ns}_{run_index:03d}"
                            file_path = os.path.join(out_dir, f"run_{run_id}.csv")

                            # Mobility model: v = M(T,solute,character,material) * tau + noise
                            lattice_factor = {"bcc": 0.8, "fcc": 1.0, "hcp": 0.9}[material["lattice"]]
                            char_factor = {"edge": 1.0, "screw": 0.6, "mixed": 0.8}[character]
                            temp_factor = 0.3 + 0.0009 * T  # increases with T
                            solute_drag_factor = 1.0 / (1.0 + 1.5 * solute)  # slows with solute

                            mobility = 4.0 * lattice_factor * char_factor * temp_factor * solute_drag_factor
                            mobility *= 1.0 + self.random.gauss(0.0, 0.05)
                            mobility = max(0.02, mobility)  # nm/ps per GPa (synthetic units)

                            # Time series
                            header = [
                                "time_ps",
                                "position_nm",
                                "velocity_nm_ps",
                                "temperature_K",
                                "tau_GPa",
                                "solute_at_pct",
                                "material",
                                "character",
                                "run_id",
                            ]

                            num_steps = 501
                            dt = 0.1  # ps
                            position_nm = 0.0
                            rows: List[List[object]] = []

                            velocities: List[float] = []
                            for step in range(num_steps):
                                time_ps = step * dt
                                velocity = mobility * tau
                                # Add thermal noise and small 1/f-like drift
                                velocity += self.random.gauss(0.0, 0.02 * (1.0 + 0.5 * (900 - T) / 400.0))
                                velocity = max(0.0, velocity)
                                position_nm += velocity * dt

                                rows.append([
                                    round(time_ps, 5),
                                    round(position_nm, 6),
                                    round(velocity, 6),
                                    T,
                                    round(tau, 3),
                                    round(solute, 3),
                                    material["name"],
                                    character,
                                    run_id,
                                ])
                                velocities.append(velocity)

                            # Persist run
                            ensure_directory_exists(os.path.dirname(file_path))
                            with open(file_path, "w", newline="") as fh:
                                writer = csv.writer(fh)
                                writer.writerow(header)
                                writer.writerows(rows)

                            mean_velocity = sum(velocities) / len(velocities)
                            rows_summary.append([
                                material["name"],
                                character,
                                T,
                                round(tau, 3),
                                round(float(solute), 3),
                                round(mean_velocity, 6),
                                round(mobility, 6),
                                run_id,
                            ])

        summary_path = os.path.join(out_dir, "mobility_summary.csv")
        write_csv(
            summary_path,
            [
                "material",
                "character",
                "temperature_K",
                "tau_GPa",
                "solute_at_pct",
                "mean_velocity_nm_ps",
                "mobility_nm_ps_per_GPa",
                "run_id",
            ],
            rows_summary,
        )

        self._add_to_manifest(
            os.path.relpath(summary_path, self.config.output_dir),
            schema={
                "material": "str",
                "character": "str",
                "temperature_K": "int",
                "tau_GPa": "float",
                "solute_at_pct": "float",
                "mean_velocity_nm_ps": "float",
                "mobility_nm_ps_per_GPa": "float",
                "run_id": "str",
            },
        )
        for fname in os.listdir(out_dir):
            if fname.startswith("run_") and fname.endswith(".csv"):
                self._add_to_manifest(
                    os.path.relpath(os.path.join(out_dir, fname), self.config.output_dir),
                    schema={
                        "time_ps": "float",
                        "position_nm": "float",
                        "velocity_nm_ps": "float",
                        "temperature_K": "int",
                        "tau_GPa": "float",
                        "solute_at_pct": "float",
                        "material": "str",
                        "character": "str",
                        "run_id": "str",
                    },
                )

    # ------------------------ Metadata/Manifest -----------------------
    def _add_to_manifest(self, rel_path: str, schema: Dict[str, str]) -> None:
        self.manifest["files"][rel_path] = {"schema_ref": rel_path}
        self.manifest["schemas"][rel_path] = schema

    def _write_metadata_and_manifest(self) -> None:
        metadata = {
            "generator_version": GENERATOR_VERSION,
            "seed": self.config.seed,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "note": "All values are synthetic and statistically fabricated for development use only.",
        }
        write_json(os.path.join(self.config.output_dir, "metadata.json"), metadata)
        write_json(os.path.join(self.config.output_dir, "manifest.json"), self.manifest)


def parse_args(argv: List[str]) -> GeneratorConfig:
    parser = argparse.ArgumentParser(description="Generate synthetic DFT/MD dataset")
    parser.add_argument(
        "--output-dir",
        default=os.path.join("data", "atomic_sim"),
        help="Output directory for generated dataset (default: data/atomic_sim)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for reproducibility (default: 12345)",
    )
    parser.add_argument(
        "--neb-paths",
        type=int,
        default=8,
        help="Number of NEB paths per process/material (default: 8)",
    )
    parser.add_argument(
        "--gbs-runs",
        type=int,
        default=24,
        help="Target count for GBS runs (approximate, depends on cartesian product)",
    )
    parser.add_argument(
        "--dl-runs",
        type=int,
        default=24,
        help="Target count for dislocation runs (approximate, depends on cartesian product)",
    )

    args = parser.parse_args(argv)

    # The exact run counts are determined by internal cartesian products; the provided
    # args allow scaling but are not strict guarantees. We keep parameters for possible
    # future use (e.g., sub-sampling) to respect user's scaling intent.
    return GeneratorConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        num_neb_paths_per_process=args.neb_paths,
        num_gbs_runs=args.gbs_runs,
        num_dislocation_runs=args.dl_runs,
    )


def main(argv: List[str]) -> int:
    config = parse_args(argv)
    ensure_directory_exists(config.output_dir)
    generator = SyntheticAtomicDataGenerator(config)
    generator.run()
    print(f"Synthetic dataset generated at: {os.path.abspath(config.output_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
