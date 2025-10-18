from __future__ import annotations

import argparse
import os
from typing import List, Dict, Tuple

from .models import get_microstructure_for_mix, monte_carlo_summary
from .utils import ensure_dir, write_json, DatasetMeta
from .exporters import export_comsol_csv, export_abaqus_inp, export_ansys_apdl


DEFAULT_MIXES = ["C", "R5S", "R10S", "R15S", "R20S", "R10L"]


def build_temperature_grid(tmin: float, tmax: float, step: float) -> List[float]:
    assert tmax >= tmin and step > 0
    n = int(round((tmax - tmin) / step)) + 1
    temps = [tmin + i * step for i in range(n)]
    # ensure exact endpoints
    temps[0] = tmin
    temps[-1] = tmax
    return temps


def generate_dataset(
    output_dir: str,
    mixes: List[str],
    tmin: float,
    tmax: float,
    step: float,
    calib_samples: int,
    valid_samples: int,
    calib_seed: int,
    valid_seed: int,
) -> None:
    ensure_dir(output_dir)

    temperature_grid = build_temperature_grid(tmin, tmax, step)
    units = {
        "T": "C",
        "rho": "kg/m^3",
        "k": "W/m-K",
        "cp": "J/kg-K",
        "E": "Pa",
        "nu": "-",
        "alpha_th": "1/K",
        "k_perm": "m^2",
        "phi": "-",
        "alpha_biot": "-",
        "fc": "Pa",
    }

    for data_type, num_samples, seed in (
        ("Calibration", calib_samples, calib_seed),
        ("Validation", valid_samples, valid_seed),
    ):
        data_type_dir = os.path.join(output_dir, data_type)
        ensure_dir(data_type_dir)
        for mix_id in mixes:
            mix_dir = os.path.join(data_type_dir, mix_id)
            ensure_dir(mix_dir)

            micro = get_microstructure_for_mix(mix_id)
            summary = monte_carlo_summary(micro, temperature_grid, num_samples, seed)

            meta = DatasetMeta(
                mix_id=mix_id,
                data_type=data_type,
                temperature_min_c=tmin,
                temperature_max_c=tmax,
                temperature_step_c=step,
                units=units,
            )
            write_json(os.path.join(mix_dir, "metadata.json"), meta.to_dict())

            # Exporters
            export_comsol_csv(
                os.path.join(mix_dir, f"{mix_id}_{data_type}_COMSOL.csv"),
                mix_id,
                data_type,
                temperature_grid,
                summary,
            )
            export_abaqus_inp(
                os.path.join(mix_dir, f"{mix_id}_{data_type}_ABAQUS.inp"),
                mix_id,
                data_type,
                temperature_grid,
                summary,
            )
            export_ansys_apdl(
                os.path.join(mix_dir, f"{mix_id}_{data_type}_ANSYS.apdl"),
                mix_id,
                data_type,
                temperature_grid,
                summary,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate thermo-mechanical-transport dataset for high-performance rubberized concrete "
            "with temperature dependence, stochastic bounds, and FEA exports."
        )
    )
    parser.add_argument("--output", default="outputs/dataset_v1", help="Output directory root")
    parser.add_argument("--mixes", nargs="*", default=DEFAULT_MIXES, help="Mix IDs to generate")
    parser.add_argument("--tmin", type=float, default=20.0)
    parser.add_argument("--tmax", type=float, default=800.0)
    parser.add_argument("--step", type=float, default=20.0)
    parser.add_argument("--calib-samples", type=int, default=200)
    parser.add_argument("--valid-samples", type=int, default=200)
    parser.add_argument("--calib-seed", type=int, default=1337)
    parser.add_argument("--valid-seed", type=int, default=4242)
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output,
        mixes=[m.upper() for m in args.mixes],
        tmin=args.tmin,
        tmax=args.tmax,
        step=args.step,
        calib_samples=args.calib_samples,
        valid_samples=args.valid_samples,
        calib_seed=args.calib_seed,
        valid_seed=args.valid_seed,
    )


if __name__ == "__main__":
    main()
