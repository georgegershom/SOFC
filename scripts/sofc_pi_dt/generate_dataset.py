from pathlib import Path
from .config import DatasetConfig
from .generators.microstructure import generate_microstructure
from .generators.geometry import generate_geometry
from .generators.operations import generate_operational_profiles
from .generators.electrochem import generate_electrochem
from .generators.thermal_structural import generate_thermal_structural
from .generators.degradation import generate_degradation
from .generators.post_mortem import generate_post_mortem


def generate(config: DatasetConfig) -> None:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    generate_microstructure(out, config.seed, config.num_microstructures, config.micro_voxels)
    if config.include_cad:
        generate_geometry(out)
    generate_operational_profiles(out, config.seed, config.num_operating_profiles)
    generate_electrochem(out, config.seed, config.num_operating_profiles, config.num_eis_points)
    generate_thermal_structural(out, config.seed, config.num_thermocouples, config.ir_frames, config.ir_image_size, config.num_strain_gauges, config.dic_frames)
    generate_degradation(out, config.seed, config.aging_hours)
    generate_post_mortem(out, config.seed)
