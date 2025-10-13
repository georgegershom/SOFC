import argparse
from pathlib import Path
from .config import DatasetConfig
from .generate_dataset import generate


def main():
    parser = argparse.ArgumentParser(description='Generate SOFC PI-DT synthetic dataset')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--profiles', type=int, default=5)
    parser.add_argument('--eis', type=int, default=40)
    parser.add_argument('--micro', type=int, default=3)
    parser.add_argument('--vox', type=int, default=64)
    parser.add_argument('--ir_frames', type=int, default=50)
    parser.add_argument('--ir_size', type=int, default=128)
    parser.add_argument('--tc', type=int, default=6)
    parser.add_argument('--sg', type=int, default=4)
    parser.add_argument('--dic_frames', type=int, default=40)
    parser.add_argument('--aging', type=int, default=500)
    parser.add_argument('--no_cad', action='store_true')
    args = parser.parse_args()

    cfg = DatasetConfig(
        output_dir=Path(args.out),
        seed=args.seed,
        num_microstructures=args.micro,
        micro_voxels=args.vox,
        include_cad=not args.no_cad,
        num_operating_profiles=args.profiles,
        num_eis_points=args.eis,
        num_thermocouples=args.tc,
        ir_frames=args.ir_frames,
        ir_image_size=args.ir_size,
        num_strain_gauges=args.sg,
        dic_frames=args.dic_frames,
        aging_hours=args.aging,
    )
    generate(cfg)

if __name__ == '__main__':
    main()
