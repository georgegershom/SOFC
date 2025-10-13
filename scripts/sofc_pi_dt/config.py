from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetConfig:
    output_dir: Path
    seed: int = 42
    num_microstructures: int = 3
    micro_voxels: int = 64  # per side for 3D volume
    include_cad: bool = True
    num_operating_profiles: int = 5
    num_eis_points: int = 40
    num_thermocouples: int = 6
    ir_frames: int = 50
    ir_image_size: int = 128
    num_strain_gauges: int = 4
    dic_frames: int = 40
    aging_hours: int = 500
