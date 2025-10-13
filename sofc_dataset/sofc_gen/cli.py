import argparse
from .generate_all import generate_all

def main():
    parser = argparse.ArgumentParser(description="Generate SOFC synthetic datasets")
    parser.add_argument('--out', type=str, default='data/out', help='Output directory')
    parser.add_argument('--n-sims', type=int, default=10, help='Number of Dataset1 simulation cases')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grid', type=int, nargs=3, default=[32, 32, 8], help='Grid size (nx ny nz)')
    parser.add_argument('--archive', action='store_true', help='Zip archive after generation')
    args = parser.parse_args()
    generate_all(out_dir=args.out, n_sims=args.n_sims, seed=args.seed, grid=tuple(args.grid), archive=args.archive)
