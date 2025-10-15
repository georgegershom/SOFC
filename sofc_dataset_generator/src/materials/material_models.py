"""
Material Property Models for SOFC Components

This module implements temperature-dependent material property models
for anode, electrolyte, and cathode materials used in SOFC manufacturing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json


@dataclass
class MaterialProperties:
    """Container for material properties at a given temperature."""
    # Mechanical properties
    elastic_modulus: float  # Pa
    poisson_ratio: float
    yield_strength: float  # Pa
    ultimate_strength: float  # Pa
    
    # Thermal properties
    thermal_expansion: float  # 1/K
    thermal_conductivity: float  # W/m·K
    specific_heat: float  # J/kg·K
    density: float  # kg/m³
    
    # Creep properties
    creep_coefficient: float
    creep_exponent: float
    activation_energy: float  # J/mol
    
    # Sintering properties
    shrinkage_rate: float
    densification_rate: float


class MaterialModel(ABC):
    """Abstract base class for material property models."""
    
    @abstractmethod
    def get_properties(self, temperature: float, **kwargs) -> MaterialProperties:
        """Get material properties at given temperature."""
        pass
    
    @abstractmethod
    def get_property_range(self, property_name: str, temp_range: Tuple[float, float]) -> Tuple[float, float]:
        """Get property range over temperature range."""
        pass


class SOFCMaterialModel:
    """Combined material model for all SOFC layers."""
    
    def __init__(self):
        """Initialize SOFC material model."""
        # Reference temperature
        self.ref_temp = 298.15  # K (25°C)
        
        # Base properties for each layer
        self.base_properties = {
            'anode': {
                'elastic_modulus': 200e9,  # Pa
                'poisson_ratio': 0.3,
                'thermal_expansion': 12e-6,  # 1/K
                'thermal_conductivity': 5.0,  # W/m·K
                'specific_heat': 500.0,  # J/kg·K
                'density': 6000.0,  # kg/m³
            },
            'electrolyte': {
                'elastic_modulus': 220e9,  # Pa
                'poisson_ratio': 0.25,
                'thermal_expansion': 10e-6,  # 1/K
                'thermal_conductivity': 2.5,  # W/m·K
                'specific_heat': 450.0,  # J/kg·K
                'density': 6100.0,  # kg/m³
            },
            'cathode': {
                'elastic_modulus': 150e9,  # Pa
                'poisson_ratio': 0.28,
                'thermal_expansion': 11e-6,  # 1/K
                'thermal_conductivity': 3.0,  # W/m·K
                'specific_heat': 480.0,  # J/kg·K
                'density': 5800.0,  # kg/m³
            }
        }
    
    def get_properties_at_temperature(self, temperature: float, 
                                    doe_parameters: Dict[str, Any]) -> Dict[str, MaterialProperties]:
        """Get properties for all layers at given temperature."""
        temp_k = temperature + 273.15  # Convert to Kelvin
        
        properties = {}
        
        # Anode properties
        anode_porosity = doe_parameters.get('material.anode.porosity', 0.35)
        anode_ni_content = doe_parameters.get('material.anode.ni_content', 0.5)
        properties['anode'] = self._get_anode_properties(temp_k, anode_porosity, anode_ni_content)
        
        # Electrolyte properties
        electrolyte_grain_size = doe_parameters.get('material.electrolyte.grain_size', 1.0)
        electrolyte_density_fraction = doe_parameters.get('material.electrolyte.density_fraction', 0.95)
        properties['electrolyte'] = self._get_electrolyte_properties(temp_k, electrolyte_grain_size, electrolyte_density_fraction)
        
        # Cathode properties
        cathode_porosity = doe_parameters.get('material.cathode.porosity', 0.4)
        cathode_lsm_content = doe_parameters.get('material.cathode.lsm_content', 0.5)
        properties['cathode'] = self._get_cathode_properties(temp_k, cathode_porosity, cathode_lsm_content)
        
        return properties
    
    def _get_anode_properties(self, temp_k: float, porosity: float, ni_content: float) -> MaterialProperties:
        """Get anode (Ni-YSZ) properties."""
        base = self.base_properties['anode']
        
        # Temperature-dependent elastic modulus
        temp_factor = 1 - 0.0003 * (temp_k - self.ref_temp)
        elastic_modulus = base['elastic_modulus'] * temp_factor
        
        # Porosity effect (Gibson-Ashby model)
        porosity_factor = (1 - porosity) ** 2
        elastic_modulus *= porosity_factor
        
        # Ni content effect
        ni_factor = 0.7 + 0.6 * ni_content
        elastic_modulus *= ni_factor
        
        # Other properties
        poisson_ratio = base['poisson_ratio'] * (1 + 0.0001 * (temp_k - self.ref_temp))
        thermal_expansion = base['thermal_expansion'] * (1 + 0.0002 * (temp_k - self.ref_temp))
        thermal_conductivity = base['thermal_conductivity'] * (1 - porosity) ** 1.5 * (0.5 + 1.5 * ni_content)
        density = base['density'] * (1 - porosity)
        
        # Creep properties
        creep_coefficient = 1e-20 * np.exp(-300000 / (8.314 * temp_k))
        creep_exponent = 3.0
        activation_energy = 300000.0
        
        # Sintering properties
        shrinkage_rate = self._calculate_shrinkage_rate(temp_k, porosity, 'anode')
        densification_rate = self._calculate_densification_rate(temp_k, porosity, 'anode')
        
        # Strength properties
        yield_strength = 100e6 * (1 - porosity) ** 1.5 * (1 - 0.0005 * (temp_k - self.ref_temp))
        ultimate_strength = yield_strength * 1.5
        
        return MaterialProperties(
            elastic_modulus=elastic_modulus,
            poisson_ratio=poisson_ratio,
            yield_strength=yield_strength,
            ultimate_strength=ultimate_strength,
            thermal_expansion=thermal_expansion,
            thermal_conductivity=thermal_conductivity,
            specific_heat=base['specific_heat'],
            density=density,
            creep_coefficient=creep_coefficient,
            creep_exponent=creep_exponent,
            activation_energy=activation_energy,
            shrinkage_rate=shrinkage_rate,
            densification_rate=densification_rate
        )
    
    def _get_electrolyte_properties(self, temp_k: float, grain_size: float, density_fraction: float) -> MaterialProperties:
        """Get electrolyte (YSZ) properties."""
        base = self.base_properties['electrolyte']
        
        # Temperature-dependent elastic modulus
        temp_factor = 1 - 0.0002 * (temp_k - self.ref_temp)
        elastic_modulus = base['elastic_modulus'] * temp_factor
        
        # Density effect
        porosity = 1 - density_fraction
        density_factor = density_fraction ** 2.5
        elastic_modulus *= density_factor
        
        # Grain size effect
        grain_factor = 1 + 0.1 / np.sqrt(grain_size)
        elastic_modulus *= grain_factor
        
        # Other properties
        poisson_ratio = base['poisson_ratio']
        thermal_expansion = base['thermal_expansion'] * (1 + 0.0001 * (temp_k - self.ref_temp))
        thermal_conductivity = base['thermal_conductivity'] * density_fraction ** 1.8 * (1 - 0.1 * (1 / grain_size))
        density = base['density'] * density_fraction
        
        # Creep properties (limited for YSZ)
        creep_coefficient = 1e-25 * np.exp(-500000 / (8.314 * temp_k))
        creep_exponent = 1.0
        activation_energy = 500000.0
        
        # Sintering properties
        shrinkage_rate = self._calculate_shrinkage_rate(temp_k, porosity, 'electrolyte', grain_size=grain_size)
        densification_rate = self._calculate_densification_rate(temp_k, porosity, 'electrolyte', grain_size=grain_size)
        
        # Strength properties
        yield_strength = 200e6 * density_fraction ** 2 * grain_factor
        ultimate_strength = yield_strength * 1.2
        
        return MaterialProperties(
            elastic_modulus=elastic_modulus,
            poisson_ratio=poisson_ratio,
            yield_strength=yield_strength,
            ultimate_strength=ultimate_strength,
            thermal_expansion=thermal_expansion,
            thermal_conductivity=thermal_conductivity,
            specific_heat=base['specific_heat'],
            density=density,
            creep_coefficient=creep_coefficient,
            creep_exponent=creep_exponent,
            activation_energy=activation_energy,
            shrinkage_rate=shrinkage_rate,
            densification_rate=densification_rate
        )
    
    def _get_cathode_properties(self, temp_k: float, porosity: float, lsm_content: float) -> MaterialProperties:
        """Get cathode (LSM-YSZ) properties."""
        base = self.base_properties['cathode']
        
        # Temperature-dependent elastic modulus
        temp_factor = 1 - 0.0004 * (temp_k - self.ref_temp)
        elastic_modulus = base['elastic_modulus'] * temp_factor
        
        # Porosity effect
        porosity_factor = (1 - porosity) ** 2.2
        elastic_modulus *= porosity_factor
        
        # LSM content effect
        lsm_factor = 0.8 + 0.4 * lsm_content
        elastic_modulus *= lsm_factor
        
        # Other properties
        poisson_ratio = base['poisson_ratio']
        thermal_expansion = base['thermal_expansion'] * (0.9 + 0.2 * lsm_content)
        thermal_conductivity = base['thermal_conductivity'] * (1 - porosity) ** 1.6 * (0.7 + 0.6 * lsm_content)
        density = base['density'] * (1 - porosity)
        
        # Creep properties
        creep_coefficient = 1e-18 * np.exp(-280000 / (8.314 * temp_k))
        creep_exponent = 2.5
        activation_energy = 280000.0
        
        # Sintering properties
        shrinkage_rate = self._calculate_shrinkage_rate(temp_k, porosity, 'cathode', lsm_content=lsm_content)
        densification_rate = self._calculate_densification_rate(temp_k, porosity, 'cathode', lsm_content=lsm_content)
        
        # Strength properties
        yield_strength = 80e6 * (1 - porosity) ** 1.8 * lsm_factor
        ultimate_strength = yield_strength * 1.3
        
        return MaterialProperties(
            elastic_modulus=elastic_modulus,
            poisson_ratio=poisson_ratio,
            yield_strength=yield_strength,
            ultimate_strength=ultimate_strength,
            thermal_expansion=thermal_expansion,
            thermal_conductivity=thermal_conductivity,
            specific_heat=base['specific_heat'],
            density=density,
            creep_coefficient=creep_coefficient,
            creep_exponent=creep_exponent,
            activation_energy=activation_energy,
            shrinkage_rate=shrinkage_rate,
            densification_rate=densification_rate
        )
    
    def _calculate_shrinkage_rate(self, temp_k: float, porosity: float, layer_type: str, **kwargs) -> float:
        """Calculate shrinkage rate during sintering."""
        if layer_type == 'anode':
            if temp_k < 1273.15:  # Below 1000°C
                return 0.0
            return 1e-6 * porosity * np.exp(-400000 / (8.314 * temp_k))
        
        elif layer_type == 'electrolyte':
            if temp_k < 1373.15:  # Below 1100°C
                return 0.0
            grain_size = kwargs.get('grain_size', 1.0)
            return 1e-7 * porosity * (1 / grain_size) * np.exp(-450000 / (8.314 * temp_k))
        
        elif layer_type == 'cathode':
            if temp_k < 1173.15:  # Below 900°C
                return 0.0
            lsm_content = kwargs.get('lsm_content', 0.5)
            return 1e-6 * porosity * (0.5 + lsm_content) * np.exp(-320000 / (8.314 * temp_k))
        
        return 0.0
    
    def _calculate_densification_rate(self, temp_k: float, porosity: float, layer_type: str, **kwargs) -> float:
        """Calculate densification rate during sintering."""
        if layer_type == 'anode':
            if temp_k < 1273.15:  # Below 1000°C
                return 0.0
            return 1e-5 * porosity ** 2 * np.exp(-350000 / (8.314 * temp_k))
        
        elif layer_type == 'electrolyte':
            if temp_k < 1373.15:  # Below 1100°C
                return 0.0
            grain_size = kwargs.get('grain_size', 1.0)
            return 1e-6 * porosity ** 1.5 * (1 / grain_size) * np.exp(-400000 / (8.314 * temp_k))
        
        elif layer_type == 'cathode':
            if temp_k < 1173.15:  # Below 900°C
                return 0.0
            lsm_content = kwargs.get('lsm_content', 0.5)
            return 1e-5 * porosity ** 1.8 * (0.5 + lsm_content) * np.exp(-300000 / (8.314 * temp_k))
        
        return 0.0
    
    def get_effective_properties(self, temperature: float, doe_parameters: Dict[str, Any]) -> MaterialProperties:
        """Get effective properties for the composite SOFC structure."""
        layer_props = self.get_properties_at_temperature(temperature, doe_parameters)
        
        # Get layer thicknesses
        anode_thickness = doe_parameters.get('geometry.anode_thickness', 500e-6)  # m
        electrolyte_thickness = doe_parameters.get('geometry.electrolyte_thickness', 15e-6)  # m
        cathode_thickness = doe_parameters.get('geometry.cathode_thickness', 40e-6)  # m
        
        total_thickness = anode_thickness + electrolyte_thickness + cathode_thickness
        
        # Volume fractions
        anode_fraction = anode_thickness / total_thickness
        electrolyte_fraction = electrolyte_thickness / total_thickness
        cathode_fraction = cathode_thickness / total_thickness
        
        # Rule of mixtures for effective properties
        effective_modulus = (
            anode_fraction * layer_props['anode'].elastic_modulus +
            electrolyte_fraction * layer_props['electrolyte'].elastic_modulus +
            cathode_fraction * layer_props['cathode'].elastic_modulus
        )
        
        effective_expansion = (
            anode_fraction * layer_props['anode'].thermal_expansion +
            electrolyte_fraction * layer_props['electrolyte'].thermal_expansion +
            cathode_fraction * layer_props['cathode'].thermal_expansion
        )
        
        effective_density = (
            anode_fraction * layer_props['anode'].density +
            electrolyte_fraction * layer_props['electrolyte'].density +
            cathode_fraction * layer_props['cathode'].density
        )
        
        # Use electrolyte properties as baseline for other properties
        return MaterialProperties(
            elastic_modulus=effective_modulus,
            poisson_ratio=layer_props['electrolyte'].poisson_ratio,
            yield_strength=min(p.yield_strength for p in layer_props.values()),
            ultimate_strength=min(p.ultimate_strength for p in layer_props.values()),
            thermal_expansion=effective_expansion,
            thermal_conductivity=layer_props['electrolyte'].thermal_conductivity,
            specific_heat=layer_props['electrolyte'].specific_heat,
            density=effective_density,
            creep_coefficient=max(p.creep_coefficient for p in layer_props.values()),
            creep_exponent=layer_props['electrolyte'].creep_exponent,
            activation_energy=layer_props['electrolyte'].activation_energy,
            shrinkage_rate=max(p.shrinkage_rate for p in layer_props.values()),
            densification_rate=max(p.densification_rate for p in layer_props.values())
        )


def create_sofc_material_model() -> SOFCMaterialModel:
    """Factory function to create SOFC material model."""
    return SOFCMaterialModel()


if __name__ == "__main__":
    # Test material models
    material_model = create_sofc_material_model()
    
    # Test parameters
    doe_params = {
        'material.anode.porosity': 0.35,
        'material.anode.ni_content': 0.5,
        'material.electrolyte.grain_size': 1.0,
        'material.electrolyte.density_fraction': 0.95,
        'material.cathode.porosity': 0.4,
        'material.cathode.lsm_content': 0.5,
        'geometry.anode_thickness': 500e-6,
        'geometry.electrolyte_thickness': 15e-6,
        'geometry.cathode_thickness': 40e-6
    }
    
    # Test at different temperatures
    temperatures = [25, 1000, 1400]
    
    for temp in temperatures:
        print(f"\nTemperature: {temp}°C")
        props = material_model.get_effective_properties(temp, doe_params)
        print(f"  Elastic Modulus: {props.elastic_modulus/1e9:.1f} GPa")
        print(f"  Thermal Expansion: {props.thermal_expansion*1e6:.1f} ppm/K")
        print(f"  Density: {props.density:.0f} kg/m³")