"""
Thermo-Mechanical Modeling Dataset Generator
=============================================
Development and Validation of a Thermo-Mechanical Model for Fire-Resistant 
Structural Elements Utilizing High-Performance Rubberized Concrete

This package generates comprehensive numerical modeling datasets for finite element analysis.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .core import ThermoMechanicalDataset
from .thermal_properties import ThermalPropertyGenerator
from .mechanical_properties import MechanicalPropertyGenerator
from .transport_properties import TransportPropertyGenerator
from .fea_exporters import ABAQUSExporter, ANSYSExporter, COMSOLExporter

__all__ = [
    'ThermoMechanicalDataset',
    'ThermalPropertyGenerator',
    'MechanicalPropertyGenerator', 
    'TransportPropertyGenerator',
    'ABAQUSExporter',
    'ANSYSExporter',
    'COMSOLExporter'
]