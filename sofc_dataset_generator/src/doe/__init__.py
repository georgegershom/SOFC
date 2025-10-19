"""
Design of Experiments (DOE) Module for SOFC Dataset Generation

This module implements various DOE strategies for generating manufacturing parameter
variations that will be used in FEA simulations to create the synthetic dataset.
"""

from .doe_generator import DOEGenerator, ParameterSpace
from .sofc_parameters import SOFCParameters, ManufacturingParameters
from .sampling_strategies import LatinHypercubeSampling, SobolSampling, RandomSampling

__all__ = [
    'DOEGenerator', 
    'ParameterSpace', 
    'SOFCParameters', 
    'ManufacturingParameters',
    'LatinHypercubeSampling',
    'SobolSampling', 
    'RandomSampling'
]