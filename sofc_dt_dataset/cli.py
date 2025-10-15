from __future__ import annotations
import argparse
import os
import json
import shutil
from .config import GeneratorConfig, GridConfig, DatasetSizes
from .generate import generate_all


def parse_shape(s: str):
    parts = s.lower().split("x")
    if len(parts) == 1:
        return (int(parts[0]),)
    return tuple(int(p) for p in parts)


def main():
    p = argparse.ArgumentParser(description="Generate Multi-Fidelity SOFC target dataset")
    p.add_argument("--out", dest="out_dir", default="datasets/sofc_mf_dataset_v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lf", dest="lf", type=int, default=200)
    p.add_argument("--mf", dest="mf", type=int, default=50)
    p.add_argument("--hf", dest="hf", type=int, default=10)
    p.add_argument("--lf-shape", dest="lf_shape", default="64")
    p.add_argument("--mf-shape", dest="mf_shape", default="32x32x8")
    p.add_argument("--hf-shape", dest="hf_shape", default="64x64x16")
    p.add_argument("--zip", dest="zip_it", action="store_true")

    args = p.parse_args()

    grids = GridConfig(
        hf_shape=parse_shape(args.hf_shape),
        mf_shape=parse_shape(args.mf_shape),
        lf_shape=parse_shape(args.lf_shape),
    )
    sizes = DatasetSizes(lf=args.lf, mf=args.mf, hf=args.hf)

    cfg = GeneratorConfig(out_dir=args.out_dir, seed=args.seed, grids=grids, sizes=sizes)
    out_dir = generate_all(cfg)

    if args.zip_it:
        base = os.path.abspath(out_dir)
        zip_path = base + ".zip"
        if os.path.exists(zip_path):
            os.remove(zip_path)
        shutil.make_archive(base, "zip", base)
        print(f"Zipped dataset at: {zip_path}")
    else:
        print(f"Dataset written to: {out_dir}")


if __name__ == "__main__":
    main()
