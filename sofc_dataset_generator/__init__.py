"""
SOFC Digital Twin Dataset Generator

A comprehensive system for generating multi-fidelity datasets for
Adaptive-Scale Physics-Informed Digital Twin for SOFC Thermo-Structural Integrity Monitoring.
"""

from .core.dataset_generator import SOFCDatasetGenerator
from .physics_simulators.electrochemical import ElectrochemicalSimulator
from .physics_simulators.thermal import ThermalSimulator
from .physics_simulators.structural import StructuralSimulator
from .data_generators.high_fidelity import HighFidelityDataGenerator
from .data_generators.experimental import ExperimentalDataGenerator
from .data_generators.monitoring import MonitoringDataGenerator
from .degradation_models.crack_propagation import CrackPropagationModel
from .degradation_models.material_aging import MaterialAgingModel

__version__ = "1.0.0"
__author__ = "SOFC Research Team"

__all__ = [
    "SOFCDatasetGenerator",
    "ElectrochemicalSimulator",
    "ThermalSimulator", 
    "StructuralSimulator",
    "HighFidelityDataGenerator",
    "ExperimentalDataGenerator",
    "MonitoringDataGenerator",
    "CrackPropagationModel",
    "MaterialAgingModel"
]