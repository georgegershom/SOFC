import os
from .dataset1_sim import generate_dataset1
from .dataset2_exp import generate_dataset2
from .dataset3_stream import generate_dataset3
from .utils import ensure_dir, write_manifest, make_archive
from typing import Tuple

def generate_all(out_dir: str, n_sims: int, seed: int, grid: Tuple[int, int, int], archive: bool=False):
    ensure_dir(out_dir)
    d1_dir = os.path.join(out_dir, 'dataset1')
    d2_dir = os.path.join(out_dir, 'dataset2')
    d3_dir = os.path.join(out_dir, 'dataset3')
    ensure_dir(d1_dir); ensure_dir(d2_dir); ensure_dir(d3_dir)

    meta1 = generate_dataset1(d1_dir, n_sims=n_sims, seed=seed, grid=grid)
    meta2 = generate_dataset2(d2_dir, seed=seed)
    meta3 = generate_dataset3(d3_dir, seed=seed)

    manifest = {
        'dataset1': meta1,
        'dataset2': meta2,
        'dataset3': meta3,
    }
    write_manifest(os.path.join(out_dir, 'manifest.json'), manifest)
    if archive:
        return make_archive(out_dir)
    return manifest
