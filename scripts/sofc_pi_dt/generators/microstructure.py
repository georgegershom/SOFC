import numpy as np
from pathlib import Path
from ..utils import write_json


def generate_microstructure(output_dir: Path, seed: int, count: int, voxels: int) -> None:
    rng = np.random.default_rng(seed)
    meta = []
    micro_dir = output_dir / 'microstructure'
    micro_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        # Simple synthetic 3-phase microstructure via thresholded Gaussian fields
        field = rng.normal(size=(voxels, voxels, voxels)).astype(np.float32)
        thresholds = np.quantile(field, [0.33, 0.66])
        phases = np.digitize(field, thresholds)  # 0,1,2 for anode/electrolyte/cathode
        np.save(micro_dir / f'volume_{i:03d}.npy', phases)
        meta.append({
            'file': f'volume_{i:03d}.npy',
            'voxels': voxels,
            'phases': ['anode', 'electrolyte', 'cathode']
        })
    write_json(micro_dir / 'metadata.json', {'count': count, 'items': meta})
