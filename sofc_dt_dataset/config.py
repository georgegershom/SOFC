from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GridConfig:
    # Shapes are tuples of (nx, ny, nz) for 3D; LF uses (nx,) for 1D
    hf_shape: Tuple[int, int, int] = (64, 64, 16)
    mf_shape: Tuple[int, int, int] = (32, 32, 8)
    lf_shape: Tuple[int] = (64,)


@dataclass
class DatasetSizes:
    lf: int = 500
    mf: int = 100
    hf: int = 20


@dataclass
class GeneratorConfig:
    out_dir: str = "datasets/sofc_mf_dataset_v1"
    seed: int = 42
    grids: GridConfig = field(default_factory=GridConfig)
    sizes: DatasetSizes = field(default_factory=DatasetSizes)
