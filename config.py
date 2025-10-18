"""
Configuration file for rubberized concrete experimental dataset generation
"""

import numpy as np

# Mix Design Parameters
MIX_DESIGNS = {
    'control': {
        'w_c_ratio': 0.45,
        'cement_content': 400,  # kg/m³
        'rubber_content': 0,    # % by volume
        'rubber_size': 'N/A',
        'admixtures': 'None'
    },
    'rubber_5_fine': {
        'w_c_ratio': 0.45,
        'cement_content': 400,
        'rubber_content': 5,    # % by volume
        'rubber_size': '0.5-2mm',
        'admixtures': 'SP + AEA'
    },
    'rubber_10_fine': {
        'w_c_ratio': 0.45,
        'cement_content': 400,
        'rubber_content': 10,
        'rubber_size': '0.5-2mm',
        'admixtures': 'SP + AEA'
    },
    'rubber_15_fine': {
        'w_c_ratio': 0.45,
        'cement_content': 400,
        'rubber_content': 15,
        'rubber_size': '0.5-2mm',
        'admixtures': 'SP + AEA'
    },
    'rubber_10_coarse': {
        'w_c_ratio': 0.45,
        'cement_content': 400,
        'rubber_content': 10,
        'rubber_size': '2-5mm',
        'admixtures': 'SP + AEA'
    },
    'rubber_15_coarse': {
        'w_c_ratio': 0.45,
        'cement_content': 400,
        'rubber_content': 15,
        'rubber_size': '2-5mm',
        'admixtures': 'SP + AEA'
    }
}

# Test Conditions
CURING_AGES = [7, 28, 56]  # days
TEMPERATURE_LEVELS = [23, 200, 400, 600, 800]  # °C
COOLING_REGIMES = ['furnace_cooling', 'water_quenching']
SPECIMENS_PER_CONDITION = 3

# Test Parameters
HEATING_RATE = 8  # °C/min
SOAK_TIME = 60  # minutes
SPECIMEN_DIMENSIONS = {
    'compressive': (150, 150, 150),  # mm
    'tensile': (150, 300, 150),      # mm
    'flexural': (100, 100, 400),     # mm
    'modulus': (150, 150, 300)       # mm
}

# Material Properties (Base values for control mix)
BASE_PROPERTIES = {
    'compressive_strength_28d': 45.0,  # MPa
    'tensile_strength_28d': 3.5,       # MPa
    'flexural_strength_28d': 5.2,      # MPa
    'modulus_elasticity_28d': 30000,   # MPa
    'density': 2400,                   # kg/m³
    'upv_ambient': 4500,               # m/s
    'thermal_expansion_coeff': 10e-6,  # /°C
    'poisson_ratio': 0.18
}

# Statistical Parameters
COEFFICIENT_OF_VARIATION = {
    'compressive_strength': 0.08,
    'tensile_strength': 0.12,
    'flexural_strength': 0.10,
    'modulus_elasticity': 0.06,
    'density': 0.02,
    'upv': 0.05
}

# Temperature Effects (Reduction factors)
TEMP_REDUCTION_FACTORS = {
    'compressive_strength': {
        23: 1.00,
        200: 0.85,
        400: 0.65,
        600: 0.35,
        800: 0.15
    },
    'tensile_strength': {
        23: 1.00,
        200: 0.80,
        400: 0.55,
        600: 0.25,
        800: 0.10
    },
    'modulus_elasticity': {
        23: 1.00,
        200: 0.75,
        400: 0.50,
        600: 0.20,
        800: 0.05
    }
}

# Rubber Content Effects
RUBBER_EFFECTS = {
    'compressive_strength': -0.15,  # % reduction per 1% rubber
    'tensile_strength': -0.08,
    'modulus_elasticity': -0.20,
    'thermal_conductivity': -0.25,
    'thermal_expansion': 0.30,  # % increase per 1% rubber
    'fire_resistance': 0.40     # % improvement per 1% rubber
}