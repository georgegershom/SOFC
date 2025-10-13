from __future__ import annotations

import argparse
import os
from typing import Dict
import numpy as np

from .sampling import LatinHypercubeSampler, default_parameter_space
from .physics import SyntheticPhysicsGenerator, Geometry, PhysicsConfig
from .experimental import ExperimentalDataGenerator, ExperimentalConfig
from .realtime import RealTimeDataGenerator, RealTimeConfig
from .io_utils import ensure_dir, save_npz, save_json, Manifest, write_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SOFC dataset generator")
    p.add_argument("--n-samples", type=int, default=10, help="Number of Dataset1 samples")
    p.add_argument("--out", type=str, default="data", help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--nx", type=int, default=32)
    p.add_argument("--ny", type=int, default=32)
    p.add_argument("--nz", type=int, default=8)
    p.add_argument("--duration-exp", type=int, default=3600)
    p.add_argument("--duration-rt", type=int, default=900)
    return p.parse_args()


def generate(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)

    base_dir = args.out
    d1_dir = os.path.join(base_dir, "dataset1_sim")
    d2_dir = os.path.join(base_dir, "dataset2_exp")
    d3_dir = os.path.join(base_dir, "dataset3_rt")
    ensure_dir(d1_dir)
    ensure_dir(d2_dir)
    ensure_dir(d3_dir)

    # Dataset 1: Synthetic multi-physics fields
    sampler = LatinHypercubeSampler(default_parameter_space(), random_seed=args.seed)
    inputs = sampler.sample(args.n_samples)
    geom = Geometry(nx=args.nx, ny=args.ny, nz=args.nz)
    phys_gen = SyntheticPhysicsGenerator(geom=geom, cfg=PhysicsConfig(), seed=args.seed)

    manifest1_items = []
    for i in range(args.n_samples):
        ip = {k: float(v[i]) for k, v in inputs.items()}
        fields = phys_gen.generate_fields(ip)
        out_f = os.path.join(d1_dir, f"sample_{i:05d}.npz")
        save_npz(out_f, {"inputs": ip, **fields})
        manifest1_items.append({"file": os.path.basename(out_f), "inputs": ip, "voltage": fields["voltage"]})

    write_manifest(
        os.path.join(d1_dir, "manifest.json"),
        Manifest(
            dataset_name="Dataset1_SyntheticPhysics",
            version="0.1.0",
            description="Synthetic multi-physics fields for SOFC",
            num_items=len(manifest1_items),
            schema={"file": "npz", "fields": ["temperature", "current_density", "stress", "strain", "displacement", "species", "creep_damage", "fracture_metrics"]},
        ),
    )
    save_json(os.path.join(d1_dir, "index.json"), {"items": manifest1_items})

    # Dataset 2: Experimental-like time series and sensor data
    exp_cfg = ExperimentalConfig(duration_s=args.duration_exp)
    exp_gen = ExperimentalDataGenerator(cfg=exp_cfg, seed=args.seed + 1)
    exp_data = exp_gen.generate({k: float(v[0]) for k, v in inputs.items()})
    save_npz(os.path.join(d2_dir, "experiment_00000.npz"), exp_data)
    write_manifest(
        os.path.join(d2_dir, "manifest.json"),
        Manifest(
            dataset_name="Dataset2_ExperimentalLike",
            version="0.1.0",
            description="Synthetic experimental validation dataset",
            num_items=1,
            schema={"file": "npz", "contents": ["time_series", "EIS", "thermal_image", "thermocouples", "strain_gauges", "AE_events"]},
        ),
    )

    # Dataset 3: Real-time stream with periodic EIS/thermal marks
    rt_cfg = RealTimeConfig(duration_s=args.duration_rt)
    rt_gen = RealTimeDataGenerator(cfg=rt_cfg, seed=args.seed + 2)
    rt_data = rt_gen.generate({k: float(v[0]) for k, v in inputs.items()})
    save_npz(os.path.join(d3_dir, "realtime_00000.npz"), rt_data)
    write_manifest(
        os.path.join(d3_dir, "manifest.json"),
        Manifest(
            dataset_name="Dataset3_RealTime",
            version="0.1.0",
            description="Synthetic real-time operational dataset",
            num_items=1,
            schema={"file": "npz", "contents": ["time_series", "capture_marks"]},
        ),
    )


def main():
    args = parse_args()
    generate(args)


if __name__ == "__main__":
    main()
