import argparse
from pathlib import Path
from .generate import generate_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic SOFC warp-stress dataset (Dataset 1).",
    )
    p.add_argument("out", type=Path, help="Output HDF5 file path")
    p.add_argument("--n", type=int, default=32, help="Number of DOE samples")
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    p.add_argument(
        "--grid", type=int, nargs=2, default=[64, 64], help="Grid size (nx ny) for surfaces"
    )
    p.add_argument(
        "--thickness", type=int, default=16, help="Number of through-thickness voxels"
    )
    p.add_argument(
        "--zip", action="store_true", help="Zip the dataset after generation"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        output_path=args.out,
        num_samples=args.n,
        seed=args.seed,
        grid_shape=(args.grid[0], args.grid[1]),
        num_z=args.thickness,
        zip_after=args.zip,
    )


if __name__ == "__main__":
    main()
