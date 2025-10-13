import argparse
from pathlib import Path
from typing import Tuple

from .sampling import latin_hypercube_samples, PARAM_RANGES
from .physics import generate_grid, generate_fields
from .io import write_dataset1_hdf5
from .experimental import synthesize_experimental_dataset
from .realtime import synthesize_realtime_dataset


def parse_grid(arg: str) -> Tuple[int, int, int]:
    if "x" not in arg:
        raise argparse.ArgumentTypeError("Grid must be formatted like 32x32x8")
    parts = arg.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Grid must be formatted like 32x32x8")
    nx, ny, nz = map(int, parts)
    if min(nx, ny, nz) < 4:
        raise argparse.ArgumentTypeError("Grid dimensions must be >= 4")
    return nx, ny, nz


def main():
    parser = argparse.ArgumentParser(description="SOFC dataset generator (adaptive-scale PI-DT)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate datasets")
    g.add_argument("--dataset", choices=["all", "1", "2", "3"], default="all")
    g.add_argument("--out", type=Path, default=Path("data/sofc_datasets"))
    g.add_argument("--n-samples", type=int, default=16)
    g.add_argument("--grid", type=parse_grid, default="32x32x8")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--duration-hours", type=float, default=24.0, help="Duration for datasets 2/3")
    g.add_argument("--sample-rate-hz", type=float, default=1.0, help="Sample rate for datasets 2/3")
    g.add_argument("--eis-period-hours", type=float, default=6.0, help="Interval between EIS measurements in Dataset 2")

    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "generate":
        if args.dataset in ("all", "1"):
            d1dir = out_dir / "dataset1_hf"
            d1dir.mkdir(parents=True, exist_ok=True)
            nx, ny, nz = args.grid if isinstance(args.grid, tuple) else parse_grid(args.grid)
            x, y, z = generate_grid(nx, ny, nz)
            samples = latin_hypercube_samples(args.n_samples, seed=args.seed)

            write_dataset1_hdf5(
                out_path=d1dir / "dataset1.h5",
                grid=(x, y, z),
                samples=samples,
                field_fn=generate_fields,
                param_ranges=PARAM_RANGES,
            )

        if args.dataset in ("all", "2"):
            d2dir = out_dir / "dataset2_exp"
            d2dir.mkdir(parents=True, exist_ok=True)
            synthesize_experimental_dataset(
                out_dir=d2dir,
                duration_hours=args.duration_hours,
                sample_rate_hz=args.sample_rate_hz,
                eis_period_hours=args.eis_period_hours,
                seed=args.seed,
            )

        if args.dataset in ("all", "3"):
            d3dir = out_dir / "dataset3_rt"
            d3dir.mkdir(parents=True, exist_ok=True)
            synthesize_realtime_dataset(
                out_dir=d3dir,
                duration_hours=args.duration_hours,
                sample_rate_hz=args.sample_rate_hz,
                seed=args.seed,
            )


if __name__ == "__main__":
    main()
