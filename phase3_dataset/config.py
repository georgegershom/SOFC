from pathlib import Path

# Output root directory
OUTPUT_ROOT = Path("phase3_microchem_output")

# Dataset definition
TEMPERATURE_STEPS_C = [200, 400, 600, 800]
REPLICATES_PER_CONDITION = 3
SEM_FOVS_PER_REPLICATE = 5

# Voxel scaling for synthetic Micro-CT volumes
VOLUME_SHAPE = (96, 96, 96)  # (Z, Y, X)
VOXEL_SIZE_UM = 5.0  # micrometers per voxel

# Mix designs (high-performance rubberized concrete variants)
MIXES = [
    {
        "Mix_ID": "HPRC-0",
        "description": "High-Performance Concrete, 0% rubber",
        "rubber_vol_frac": 0.00,  # volume fraction of rubber particles in fresh mix
        "silica_fume_frac": 0.10,  # reduces portlandite due to pozzolanic reaction
        "aggregate_vol_frac": 0.35,
    },
    {
        "Mix_ID": "HPRC-10",
        "description": "High-Performance Rubberized Concrete, 10% rubber",
        "rubber_vol_frac": 0.10,
        "silica_fume_frac": 0.10,
        "aggregate_vol_frac": 0.35,
    },
    {
        "Mix_ID": "HPRC-20",
        "description": "High-Performance Rubberized Concrete, 20% rubber",
        "rubber_vol_frac": 0.20,
        "silica_fume_frac": 0.10,
        "aggregate_vol_frac": 0.35,
    },
]

# Sample ID format helper (used by generator too)
SAMPLE_ID_FORMAT = "C28-{mix}-{temp}C-Furnace-Rep{rep}"
