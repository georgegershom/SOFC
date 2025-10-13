import numpy as np
from pathlib import Path
from ..utils import write_json


def generate_post_mortem(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed + 5)
    pm_dir = output_dir / 'post_mortem'
    pm_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic SEM-like grayscale images showing cracks/pores via Perlin-like noise
    def synth_image(size=256, cracks=5):
        img = rng.normal(scale=10, size=(size, size)).astype(np.float32)
        for _ in range(cracks):
            x0 = rng.integers(0, size)
            y0 = rng.integers(0, size)
            length = rng.integers(size//3, size)
            theta = rng.uniform(0, np.pi)
            xs = (x0 + np.arange(length)*np.cos(theta)).astype(int)
            ys = (y0 + np.arange(length)*np.sin(theta)).astype(int)
            xs = np.clip(xs, 0, size-1)
            ys = np.clip(ys, 0, size-1)
            img[ys, xs] -= 50
        img -= img.min()
        img /= (img.max() + 1e-8)
        img *= 255
        return img.astype(np.uint8)

    for i in range(6):
        np.save(pm_dir / f'sem_{i:02d}.npy', synth_image(256, cracks=rng.integers(3,8)))
    write_json(pm_dir / 'metadata.json', {'images': [f'sem_{i:02d}.npy' for i in range(6)], 'modality': 'synthetic_SEM'})
