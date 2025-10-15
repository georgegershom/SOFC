"""
Finite Element Analysis (FEA) Simulation Framework

This module provides the core FEA simulation capabilities for SOFC thermo-mechanical
analysis, including mesh generation, material property assignment, and stress analysis.
"""

from .mesh_generator import SOFCMeshGenerator, MeshParameters
from .fea_solver import FEASolver, SimulationResults
# from .thermo_mechanical import ThermoMechanicalAnalysis
from .stress_analysis import StressAnalyzer, StressField
from .warp_analysis import WarpAnalyzer, WarpField

__all__ = [
    'SOFCMeshGenerator',
    'MeshParameters', 
    'FEASolver',
    'SimulationResults',
    'StressAnalyzer',
    'StressField',
    'WarpAnalyzer',
    'WarpField'
]