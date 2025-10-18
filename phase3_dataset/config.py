from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Phase3Config:
    output_root: str = "/workspace/data/phase3"
    rng_seed: int = 1337
    temperatures_c: List[int] = (20, 200, 400, 600, 800)
    specimens: List[str] = ("control", "rubber")
    replicates_per_condition: int = 5
    microct_volume_shape: tuple = (128, 128, 128)


CONFIG = Phase3Config()