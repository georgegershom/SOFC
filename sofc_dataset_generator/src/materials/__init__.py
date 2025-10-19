"""
SOFC Material Properties Module

This module defines material properties for SOFC components including:
- 8YSZ Electrolyte
- Ni-YSZ Anode
- LSM-YSZ Cathode
- Crofer 22 APU Interconnect

All properties include temperature dependence for accurate FEA simulation.
"""

from .sofc_materials import SOFCMaterials, MaterialProperty
from .creep_models import CreepModel, NortonBaileyCreep

__all__ = ['SOFCMaterials', 'MaterialProperty', 'CreepModel', 'NortonBaileyCreep']