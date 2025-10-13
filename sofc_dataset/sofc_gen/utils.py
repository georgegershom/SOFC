import os
import json
import numpy as np
import shutil
from typing import Dict

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def latin_hypercube(n: int, d: int, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    cut = np.linspace(0, 1, n + 1)
    u = rng.random((n, d))
    a = cut[:n]
    b = cut[1:n+1]
    rdpoints = u*(b - a)[:, None] + a[:, None]
    H = np.zeros_like(rdpoints)
    for j in range(d):
        order = rng.permutation(n)
        H[:, j] = rdpoints[order, 0]
    return H

def write_manifest(path: str, manifest: Dict):
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)

def make_archive(out_dir: str) -> str:
    base = os.path.abspath(out_dir)
    archive = shutil.make_archive(base, 'zip', base)
    return archive
