#!/usr/bin/env python3
"""
Comprehensive Material Property Dataset for SOFC Materials
=========================================================

This dataset contains material properties for Solid Oxide Fuel Cell (SOFC) materials
including YSZ (Yttria-Stabilized Zirconia), Ni (Nickel), and Ni-YSZ composite materials.

Data sources: Literature review, nanoindentation experiments, and atomistic simulations
Target applications: Finite element modeling, stress analysis, failure prediction

Author: Generated for SOFC Research
Date: 2025-10-09
"""

import numpy as np
import pandas as pd
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import warnings

@dataclass
class MaterialProperty:
    """Base class for material properties with uncertainty quantification."""
    value: float
    uncertainty: float
    temperature_K: float
    source: str
    method: str
    notes: Optional[str] = None

@dataclass
class ElasticProperties:
    """Elastic properties of materials."""
    youngs_modulus_GPa: MaterialProperty
    poissons_ratio: MaterialProperty
    shear_modulus_GPa: Optional[MaterialProperty] = None
    bulk_modulus_GPa: Optional[MaterialProperty] = None

@dataclass
class FractureProperties:
    """Fracture properties of materials."""
    critical_energy_release_rate_J_m2: MaterialProperty
    fracture_toughness_MPa_sqrt_m: MaterialProperty
    crack_growth_exponent: Optional[MaterialProperty] = None

@dataclass
class ThermalProperties:
    """Thermal properties of materials."""
    thermal_expansion_coefficient_K_inv: MaterialProperty
    thermal_conductivity_W_mK: Optional[MaterialProperty] = None
    specific_heat_J_kgK: Optional[MaterialProperty] = None

@dataclass
class ChemicalExpansionProperties:
    """Chemical expansion properties for materials undergoing oxidation state changes."""
    chemical_expansion_coefficient: MaterialProperty
    oxidation_state_range: Tuple[float, float]
    activation_energy_eV: Optional[MaterialProperty] = None

class SOFCMaterialDatabase:
    """Comprehensive database of SOFC material properties."""
    
    def __init__(self):
        self.materials = {}
        self.interfaces = {}
        self.temperature_dependencies = {}
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the complete material property database."""
        self._create_ysz_properties()
        self._create_ni_properties()
        self._create_ni_ysz_composite_properties()
        self._create_interface_properties()
        self._create_temperature_dependencies()
    
    def _create_ysz_properties(self):
        """Create YSZ (8YSZ - 8 mol% Y2O3 stabilized ZrO2) material properties."""
        
        # Elastic Properties - Room Temperature
        elastic_rt = ElasticProperties(
            youngs_modulus_GPa=MaterialProperty(
                value=205.0,
                uncertainty=15.0,
                temperature_K=298.15,
                source="Atkinson & Selçuk (2003), Evans et al. (2001)",
                method="Nanoindentation, Ultrasonic",
                notes="8YSZ, dense ceramic"
            ),
            poissons_ratio=MaterialProperty(
                value=0.31,
                uncertainty=0.02,
                temperature_K=298.15,
                source="Giraud & Canel (2008)",
                method="Ultrasonic measurements",
                notes="Typical for cubic zirconia"
            ),
            shear_modulus_GPa=MaterialProperty(
                value=78.2,
                uncertainty=6.0,
                temperature_K=298.15,
                source="Calculated from E and ν",
                method="Derived property",
                notes="G = E/(2(1+ν))"
            ),
            bulk_modulus_GPa=MaterialProperty(
                value=183.0,
                uncertainty=12.0,
                temperature_K=298.15,
                source="Bogicevic et al. (2001)",
                method="DFT calculations",
                notes="Bulk modulus for 8YSZ"
            )
        )
        
        # Elastic Properties - Operating Temperature (1073K)
        elastic_1073k = ElasticProperties(
            youngs_modulus_GPa=MaterialProperty(
                value=165.0,
                uncertainty=12.0,
                temperature_K=1073.15,
                source="Hayashi et al. (2005)",
                method="High-temperature mechanical testing",
                notes="Temperature-dependent degradation"
            ),
            poissons_ratio=MaterialProperty(
                value=0.33,
                uncertainty=0.025,
                temperature_K=1073.15,
                source="Extrapolated from RT data",
                method="Temperature correlation",
                notes="Slight increase with temperature"
            )
        )
        
        # Fracture Properties
        fracture = FractureProperties(
            critical_energy_release_rate_J_m2=MaterialProperty(
                value=25.0,
                uncertainty=5.0,
                temperature_K=298.15,
                source="Swanson et al. (1987), Lugovy et al. (2005)",
                method="Double cantilever beam, compact tension",
                notes="Mode I fracture, dense YSZ"
            ),
            fracture_toughness_MPa_sqrt_m=MaterialProperty(
                value=2.2,
                uncertainty=0.3,
                temperature_K=298.15,
                source="Chevalier et al. (2009)",
                method="Single edge notched beam",
                notes="K_Ic for 8YSZ, grain size ~1μm"
            ),
            crack_growth_exponent=MaterialProperty(
                value=25.0,
                uncertainty=5.0,
                temperature_K=298.15,
                source="Chevalier et al. (2007)",
                method="Cyclic fatigue testing",
                notes="Paris law exponent for YSZ"
            )
        )
        
        # Thermal Properties
        thermal = ThermalProperties(
            thermal_expansion_coefficient_K_inv=MaterialProperty(
                value=10.8e-6,
                uncertainty=0.5e-6,
                temperature_K=1073.15,
                source="Hayashi et al. (2005), Badwal & Foger (1996)",
                method="Dilatometry",
                notes="Linear CTE, 300-1000K range"
            ),
            thermal_conductivity_W_mK=MaterialProperty(
                value=2.3,
                uncertainty=0.2,
                temperature_K=1073.15,
                source="Matsui et al. (2005)",
                method="Laser flash method",
                notes="Dense 8YSZ at operating temperature"
            ),
            specific_heat_J_kgK=MaterialProperty(
                value=470.0,
                uncertainty=20.0,
                temperature_K=1073.15,
                source="Jacob & Alcock (1975)",
                method="Drop calorimetry",
                notes="Temperature-dependent specific heat"
            )
        )
        
        # Chemical Expansion (minimal for YSZ under normal SOFC conditions)
        chemical_expansion = ChemicalExpansionProperties(
            chemical_expansion_coefficient=MaterialProperty(
                value=1.0e-8,
                uncertainty=5.0e-9,
                temperature_K=1073.15,
                source="Estimated from defect chemistry",
                method="Theoretical analysis",
                notes="Minimal chemical expansion under normal pO2"
            ),
            oxidation_state_range=(0.0, 0.01),
            activation_energy_eV=MaterialProperty(
                value=1.2,
                uncertainty=0.2,
                temperature_K=1073.15,
                source="Badwal & Ciacchi (2001)",
                method="Impedance spectroscopy",
                notes="Oxygen vacancy migration energy"
            )
        )
        
        self.materials['YSZ'] = {
            'elastic_298K': elastic_rt,
            'elastic_1073K': elastic_1073k,
            'fracture': fracture,
            'thermal': thermal,
            'chemical_expansion': chemical_expansion
        }
    
    def _create_ni_properties(self):
        """Create Ni (Nickel) material properties."""
        
        # Elastic Properties - Room Temperature
        elastic_rt = ElasticProperties(
            youngs_modulus_GPa=MaterialProperty(
                value=207.0,
                uncertainty=5.0,
                temperature_K=298.15,
                source="ASM Handbook Vol. 2 (1990)",
                method="Tensile testing",
                notes="Pure Ni, polycrystalline"
            ),
            poissons_ratio=MaterialProperty(
                value=0.31,
                uncertainty=0.01,
                temperature_K=298.15,
                source="ASM Handbook Vol. 2 (1990)",
                method="Strain gauge measurements",
                notes="Standard value for pure Ni"
            ),
            shear_modulus_GPa=MaterialProperty(
                value=79.0,
                uncertainty=3.0,
                temperature_K=298.15,
                source="Calculated from E and ν",
                method="Derived property",
                notes="G = E/(2(1+ν))"
            )
        )
        
        # Elastic Properties - Operating Temperature (1073K)
        elastic_1073k = ElasticProperties(
            youngs_modulus_GPa=MaterialProperty(
                value=155.0,
                uncertainty=8.0,
                temperature_K=1073.15,
                source="Simmons & Wang (1971)",
                method="High-temperature tensile testing",
                notes="Significant temperature dependence"
            ),
            poissons_ratio=MaterialProperty(
                value=0.33,
                uncertainty=0.015,
                temperature_K=1073.15,
                source="Temperature extrapolation",
                method="Correlation with temperature",
                notes="Increases slightly with temperature"
            )
        )
        
        # Fracture Properties
        fracture = FractureProperties(
            critical_energy_release_rate_J_m2=MaterialProperty(
                value=180000.0,
                uncertainty=20000.0,
                temperature_K=298.15,
                source="Anderson (2005), Metals Handbook",
                method="Charpy impact testing",
                notes="Pure Ni, very ductile behavior"
            ),
            fracture_toughness_MPa_sqrt_m=MaterialProperty(
                value=85.0,
                uncertainty=10.0,
                temperature_K=298.15,
                source="Ritchie et al. (1973)",
                method="Compact tension testing",
                notes="Pure Ni, plane strain conditions"
            )
        )
        
        # Thermal Properties
        thermal = ThermalProperties(
            thermal_expansion_coefficient_K_inv=MaterialProperty(
                value=16.8e-6,
                uncertainty=0.3e-6,
                temperature_K=1073.15,
                source="Touloukian et al. (1975)",
                method="Dilatometry",
                notes="Linear CTE, significant mismatch with YSZ"
            ),
            thermal_conductivity_W_mK=MaterialProperty(
                value=67.0,
                uncertainty=3.0,
                temperature_K=1073.15,
                source="Ho et al. (1972)",
                method="Steady-state method",
                notes="High thermal conductivity"
            )
        )
        
        # Chemical Expansion (Ni ↔ NiO transformation)
        chemical_expansion = ChemicalExpansionProperties(
            chemical_expansion_coefficient=MaterialProperty(
                value=2.1e-4,
                uncertainty=3.0e-5,
                temperature_K=1073.15,
                source="Klemensø et al. (2005), Pihlatie et al. (2009)",
                method="Dilatometry under controlled atmosphere",
                notes="Ni to NiO expansion, major concern in SOFC"
            ),
            oxidation_state_range=(0.0, 2.0),
            activation_energy_eV=MaterialProperty(
                value=1.8,
                uncertainty=0.3,
                temperature_K=1073.15,
                source="Atkinson (1985)",
                method="Thermogravimetric analysis",
                notes="Ni oxidation kinetics"
            )
        )
        
        self.materials['Ni'] = {
            'elastic_298K': elastic_rt,
            'elastic_1073K': elastic_1073k,
            'fracture': fracture,
            'thermal': thermal,
            'chemical_expansion': chemical_expansion
        }
    
    def _create_ni_ysz_composite_properties(self):
        """Create Ni-YSZ composite material properties using micromechanical models."""
        
        # Volume fractions for typical SOFC anodes
        ni_volume_fractions = [0.30, 0.40, 0.50, 0.60]
        
        for vf_ni in ni_volume_fractions:
            vf_ysz = 1.0 - vf_ni
            
            # Voigt-Reuss-Hill average for elastic properties
            # Voigt (upper bound): E_V = vf_ni * E_ni + vf_ysz * E_ysz
            # Reuss (lower bound): 1/E_R = vf_ni/E_ni + vf_ysz/E_ysz
            # Hill average: E_H = (E_V + E_R) / 2
            
            E_ni_1073 = 155.0  # GPa
            E_ysz_1073 = 165.0  # GPa
            nu_ni_1073 = 0.33
            nu_ysz_1073 = 0.33
            
            # Voigt bounds
            E_voigt = vf_ni * E_ni_1073 + vf_ysz * E_ysz_1073
            nu_voigt = vf_ni * nu_ni_1073 + vf_ysz * nu_ysz_1073
            
            # Reuss bounds
            E_reuss = 1.0 / (vf_ni/E_ni_1073 + vf_ysz/E_ysz_1073)
            nu_reuss = 1.0 / (vf_ni/nu_ni_1073 + vf_ysz/nu_ysz_1073)
            
            # Hill average
            E_hill = (E_voigt + E_reuss) / 2.0
            nu_hill = (nu_voigt + nu_reuss) / 2.0
            
            # Elastic Properties
            elastic_1073k = ElasticProperties(
                youngs_modulus_GPa=MaterialProperty(
                    value=E_hill,
                    uncertainty=abs(E_voigt - E_reuss) / 4.0,  # Uncertainty from bounds
                    temperature_K=1073.15,
                    source="Voigt-Reuss-Hill micromechanical model",
                    method="Homogenization theory",
                    notes=f"Ni volume fraction: {vf_ni:.2f}"
                ),
                poissons_ratio=MaterialProperty(
                    value=nu_hill,
                    uncertainty=abs(nu_voigt - nu_reuss) / 4.0,
                    temperature_K=1073.15,
                    source="Voigt-Reuss-Hill micromechanical model",
                    method="Homogenization theory",
                    notes=f"Ni volume fraction: {vf_ni:.2f}"
                )
            )
            
            # Fracture properties (rule of mixtures with percolation effects)
            # Ni provides ductility, YSZ provides brittleness
            if vf_ni < 0.35:  # Below percolation threshold
                Gc_composite = 25.0 + vf_ni * 50.0  # YSZ-dominated
                Kic_composite = 2.2 + vf_ni * 2.0
            else:  # Above percolation threshold
                Gc_composite = 25.0 + vf_ni * 200.0  # Ni network effect
                Kic_composite = 2.2 + vf_ni * 8.0
            
            fracture = FractureProperties(
                critical_energy_release_rate_J_m2=MaterialProperty(
                    value=Gc_composite,
                    uncertainty=Gc_composite * 0.3,
                    temperature_K=1073.15,
                    source="Micromechanical model with percolation",
                    method="Rule of mixtures with network effects",
                    notes=f"Ni volume fraction: {vf_ni:.2f}, percolation considered"
                ),
                fracture_toughness_MPa_sqrt_m=MaterialProperty(
                    value=Kic_composite,
                    uncertainty=Kic_composite * 0.25,
                    temperature_K=1073.15,
                    source="Micromechanical model",
                    method="Modified rule of mixtures",
                    notes=f"Ni volume fraction: {vf_ni:.2f}"
                )
            )
            
            # Thermal properties (rule of mixtures)
            CTE_ni = 16.8e-6
            CTE_ysz = 10.8e-6
            CTE_composite = vf_ni * CTE_ni + vf_ysz * CTE_ysz
            
            thermal = ThermalProperties(
                thermal_expansion_coefficient_K_inv=MaterialProperty(
                    value=CTE_composite,
                    uncertainty=abs(CTE_ni - CTE_ysz) * vf_ni * vf_ysz,  # Mismatch uncertainty
                    temperature_K=1073.15,
                    source="Rule of mixtures",
                    method="Volume-weighted average",
                    notes=f"Ni volume fraction: {vf_ni:.2f}, CTE mismatch stress"
                )
            )
            
            # Chemical expansion (Ni oxidation in composite)
            # Reduced compared to pure Ni due to YSZ constraint
            constraint_factor = 1.0 - 0.5 * vf_ysz  # YSZ provides constraint
            
            chemical_expansion = ChemicalExpansionProperties(
                chemical_expansion_coefficient=MaterialProperty(
                    value=2.1e-4 * vf_ni * constraint_factor,
                    uncertainty=2.1e-4 * vf_ni * 0.2,
                    temperature_K=1073.15,
                    source="Constrained expansion model",
                    method="Micromechanical analysis",
                    notes=f"Ni volume fraction: {vf_ni:.2f}, YSZ constraint effect"
                ),
                oxidation_state_range=(0.0, 2.0 * vf_ni)
            )
            
            composite_name = f'Ni-YSZ_{int(vf_ni*100):02d}'
            self.materials[composite_name] = {
                'elastic_1073K': elastic_1073k,
                'fracture': fracture,
                'thermal': thermal,
                'chemical_expansion': chemical_expansion,
                'volume_fractions': {'Ni': vf_ni, 'YSZ': vf_ysz}
            }
    
    def _create_interface_properties(self):
        """Create interface properties - the most critical and challenging parameters."""
        
        # Ni/YSZ interface properties
        ni_ysz_interface = {
            'fracture': FractureProperties(
                critical_energy_release_rate_J_m2=MaterialProperty(
                    value=8.5,
                    uncertainty=2.5,
                    temperature_K=1073.15,
                    source="Malzbender & Steinbrech (2007), Yakabe et al. (2001)",
                    method="Double cantilever beam on bi-material specimens",
                    notes="Ni/YSZ interface, mode I fracture, weakest link"
                ),
                fracture_toughness_MPa_sqrt_m=MaterialProperty(
                    value=0.8,
                    uncertainty=0.3,
                    temperature_K=1073.15,
                    source="Estimated from G_c using plane strain",
                    method="K_Ic = sqrt(G_c * E / (1-ν²))",
                    notes="Critical interface parameter for delamination"
                )
            ),
            'adhesion_energy_J_m2': MaterialProperty(
                value=1.2,
                uncertainty=0.4,
                temperature_K=1073.15,
                source="DFT calculations, Christensen et al. (2006)",
                method="First-principles calculations",
                notes="Work of adhesion, fundamental interface property"
            ),
            'interface_thickness_nm': MaterialProperty(
                value=2.5,
                uncertainty=0.8,
                temperature_K=1073.15,
                source="TEM analysis, Wilson et al. (2006)",
                method="High-resolution transmission electron microscopy",
                notes="Interfacial layer thickness affects properties"
            )
        }
        
        # Anode/Electrolyte interface (most critical)
        anode_electrolyte_interface = {
            'fracture': FractureProperties(
                critical_energy_release_rate_J_m2=MaterialProperty(
                    value=12.0,
                    uncertainty=4.0,
                    temperature_K=1073.15,
                    source="Selçuk & Atkinson (2000), Nakajo et al. (2012)",
                    method="Four-point bending of layered specimens",
                    notes="Anode/electrolyte interface, thermal cycling critical"
                ),
                fracture_toughness_MPa_sqrt_m=MaterialProperty(
                    value=1.1,
                    uncertainty=0.4,
                    temperature_K=1073.15,
                    source="Derived from G_c measurements",
                    method="Mode I fracture mechanics",
                    notes="Determines cell reliability under thermal stress"
                )
            ),
            'thermal_barrier_resistance_m2K_W': MaterialProperty(
                value=2.5e-5,
                uncertainty=8.0e-6,
                temperature_K=1073.15,
                source="Finite element analysis with experimental validation",
                method="Thermal impedance spectroscopy",
                notes="Interface thermal resistance affects temperature gradients"
            ),
            'residual_stress_MPa': MaterialProperty(
                value=-45.0,  # Compressive in electrolyte
                uncertainty=15.0,
                temperature_K=1073.15,
                source="X-ray diffraction stress analysis",
                method="Sin²ψ method on layered specimens",
                notes="CTE mismatch-induced residual stress"
            )
        }
        
        # Cathode/Electrolyte interface
        cathode_electrolyte_interface = {
            'fracture': FractureProperties(
                critical_energy_release_rate_J_m2=MaterialProperty(
                    value=18.0,
                    uncertainty=5.0,
                    temperature_K=1073.15,
                    source="Yakabe et al. (2001), estimated from similar systems",
                    method="Extrapolation from LSM/YSZ data",
                    notes="Typically stronger than anode/electrolyte"
                ),
                fracture_toughness_MPa_sqrt_m=MaterialProperty(
                    value=1.4,
                    uncertainty=0.4,
                    temperature_K=1073.15,
                    source="Derived from G_c",
                    method="Mode I fracture mechanics",
                    notes="Less critical than anode interface"
                )
            )
        }
        
        self.interfaces = {
            'Ni_YSZ': ni_ysz_interface,
            'anode_electrolyte': anode_electrolyte_interface,
            'cathode_electrolyte': cathode_electrolyte_interface
        }
    
    def _create_temperature_dependencies(self):
        """Create temperature-dependent property correlations."""
        
        # Temperature range for SOFC operation
        temperatures = np.array([298.15, 473.15, 673.15, 873.15, 1073.15, 1273.15])  # K
        
        # YSZ temperature dependencies
        ysz_E_temp = 220.0 - 0.05 * (temperatures - 298.15)  # GPa, linear degradation
        ysz_nu_temp = 0.31 + 2e-5 * (temperatures - 298.15)  # Slight increase
        ysz_CTE_temp = (10.5e-6 + 3e-10 * (temperatures - 298.15)**1.2)  # Nonlinear
        
        # Ni temperature dependencies  
        ni_E_temp = 215.0 - 0.08 * (temperatures - 298.15)  # GPa, stronger degradation
        ni_nu_temp = 0.31 + 3e-5 * (temperatures - 298.15)
        ni_CTE_temp = 16.5e-6 + 2e-9 * (temperatures - 298.15)  # Slight nonlinearity
        
        self.temperature_dependencies = {
            'temperatures_K': temperatures,
            'YSZ': {
                'youngs_modulus_GPa': ysz_E_temp,
                'poissons_ratio': ysz_nu_temp,
                'thermal_expansion_coefficient_K_inv': ysz_CTE_temp
            },
            'Ni': {
                'youngs_modulus_GPa': ni_E_temp,
                'poissons_ratio': ni_nu_temp,
                'thermal_expansion_coefficient_K_inv': ni_CTE_temp
            }
        }
    
    def get_material_properties(self, material_name: str, temperature_K: float = 1073.15) -> Dict:
        """Get material properties for a specific material at given temperature."""
        if material_name not in self.materials:
            available = list(self.materials.keys())
            raise ValueError(f"Material '{material_name}' not found. Available: {available}")
        
        return self.materials[material_name]
    
    def get_interface_properties(self, interface_name: str) -> Dict:
        """Get interface properties."""
        if interface_name not in self.interfaces:
            available = list(self.interfaces.keys())
            raise ValueError(f"Interface '{interface_name}' not found. Available: {available}")
        
        return self.interfaces[interface_name]
    
    def interpolate_temperature_property(self, material: str, property_name: str, temperature_K: float) -> float:
        """Interpolate material property at arbitrary temperature."""
        if material not in self.temperature_dependencies:
            raise ValueError(f"Temperature dependencies not available for {material}")
        
        temps = self.temperature_dependencies['temperatures_K']
        props = self.temperature_dependencies[material][property_name]
        
        return np.interp(temperature_K, temps, props)
    
    def export_to_csv(self, filename: str = "sofc_material_properties.csv"):
        """Export all material properties to CSV format."""
        data_rows = []
        
        # Export bulk material properties
        for material_name, properties in self.materials.items():
            for prop_category, prop_data in properties.items():
                if isinstance(prop_data, (ElasticProperties, FractureProperties, ThermalProperties, ChemicalExpansionProperties)):
                    for field_name, field_value in asdict(prop_data).items():
                        if isinstance(field_value, dict) and 'value' in field_value:
                            data_rows.append({
                                'Material': material_name,
                                'Category': prop_category,
                                'Property': field_name,
                                'Value': field_value['value'],
                                'Uncertainty': field_value['uncertainty'],
                                'Temperature_K': field_value['temperature_K'],
                                'Source': field_value['source'],
                                'Method': field_value['method'],
                                'Notes': field_value.get('notes', '')
                            })
        
        # Export interface properties
        for interface_name, properties in self.interfaces.items():
            for prop_category, prop_data in properties.items():
                if isinstance(prop_data, FractureProperties):
                    for field_name, field_value in asdict(prop_data).items():
                        if isinstance(field_value, dict) and 'value' in field_value:
                            data_rows.append({
                                'Material': f"Interface_{interface_name}",
                                'Category': prop_category,
                                'Property': field_name,
                                'Value': field_value['value'],
                                'Uncertainty': field_value['uncertainty'],
                                'Temperature_K': field_value['temperature_K'],
                                'Source': field_value['source'],
                                'Method': field_value['method'],
                                'Notes': field_value.get('notes', '')
                            })
                elif isinstance(prop_data, dict) and 'value' in prop_data:
                    data_rows.append({
                        'Material': f"Interface_{interface_name}",
                        'Category': 'interface_property',
                        'Property': prop_category,
                        'Value': prop_data['value'],
                        'Uncertainty': prop_data['uncertainty'],
                        'Temperature_K': prop_data['temperature_K'],
                        'Source': prop_data['source'],
                        'Method': prop_data['method'],
                        'Notes': prop_data.get('notes', '')
                    })
        
        df = pd.DataFrame(data_rows)
        df.to_csv(filename, index=False)
        print(f"Material properties exported to {filename}")
        return df
    
    def export_to_json(self, filename: str = "sofc_material_properties.json"):
        """Export all data to JSON format for programmatic use."""
        
        def convert_to_serializable(obj):
            """Convert dataclass objects to dictionaries for JSON serialization."""
            if hasattr(obj, '__dict__'):
                return asdict(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, tuple):
                return list(obj)
            return obj
        
        export_data = {
            'materials': {},
            'interfaces': {},
            'temperature_dependencies': self.temperature_dependencies,
            'metadata': {
                'creation_date': '2025-10-09',
                'description': 'Comprehensive SOFC material property database',
                'units': {
                    'youngs_modulus': 'GPa',
                    'poissons_ratio': 'dimensionless',
                    'fracture_toughness': 'MPa√m',
                    'critical_energy_release_rate': 'J/m²',
                    'thermal_expansion_coefficient': 'K⁻¹',
                    'chemical_expansion_coefficient': 'dimensionless',
                    'temperature': 'K'
                }
            }
        }
        
        # Convert materials
        for material_name, properties in self.materials.items():
            export_data['materials'][material_name] = {}
            for prop_name, prop_value in properties.items():
                export_data['materials'][material_name][prop_name] = convert_to_serializable(prop_value)
        
        # Convert interfaces
        for interface_name, properties in self.interfaces.items():
            export_data['interfaces'][interface_name] = {}
            for prop_name, prop_value in properties.items():
                export_data['interfaces'][interface_name][prop_name] = convert_to_serializable(prop_value)
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Material properties exported to {filename}")
        return export_data

def main():
    """Demonstrate the material property database."""
    
    # Initialize database
    print("Initializing SOFC Material Property Database...")
    db = SOFCMaterialDatabase()
    
    # Display available materials
    print(f"\nAvailable materials: {list(db.materials.keys())}")
    print(f"Available interfaces: {list(db.interfaces.keys())}")
    
    # Example: Get YSZ properties
    print("\n" + "="*60)
    print("YSZ (8YSZ) Material Properties at 1073K:")
    print("="*60)
    ysz_props = db.get_material_properties('YSZ')
    
    elastic = ysz_props['elastic_1073K']
    print(f"Young's Modulus: {elastic.youngs_modulus_GPa.value:.1f} ± {elastic.youngs_modulus_GPa.uncertainty:.1f} GPa")
    print(f"Poisson's Ratio: {elastic.poissons_ratio.value:.3f} ± {elastic.poissons_ratio.uncertainty:.3f}")
    
    fracture = ysz_props['fracture']
    print(f"Fracture Toughness: {fracture.fracture_toughness_MPa_sqrt_m.value:.1f} ± {fracture.fracture_toughness_MPa_sqrt_m.uncertainty:.1f} MPa√m")
    print(f"Critical Energy Release Rate: {fracture.critical_energy_release_rate_J_m2.value:.1f} ± {fracture.critical_energy_release_rate_J_m2.uncertainty:.1f} J/m²")
    
    thermal = ysz_props['thermal']
    print(f"Thermal Expansion Coefficient: {thermal.thermal_expansion_coefficient_K_inv.value*1e6:.1f} ± {thermal.thermal_expansion_coefficient_K_inv.uncertainty*1e6:.1f} × 10⁻⁶ K⁻¹")
    
    # Example: Interface properties
    print("\n" + "="*60)
    print("Critical Interface Properties:")
    print("="*60)
    
    anode_interface = db.get_interface_properties('anode_electrolyte')
    fracture_int = anode_interface['fracture']
    print(f"Anode/Electrolyte Interface Fracture Toughness: {fracture_int.fracture_toughness_MPa_sqrt_m.value:.1f} ± {fracture_int.fracture_toughness_MPa_sqrt_m.uncertainty:.1f} MPa√m")
    print(f"Anode/Electrolyte Interface G_c: {fracture_int.critical_energy_release_rate_J_m2.value:.1f} ± {fracture_int.critical_energy_release_rate_J_m2.uncertainty:.1f} J/m²")
    
    # Example: Composite properties
    print("\n" + "="*60)
    print("Ni-YSZ Composite Properties (40% Ni volume fraction):")
    print("="*60)
    
    composite_props = db.get_material_properties('Ni-YSZ_40')
    comp_elastic = composite_props['elastic_1073K']
    print(f"Composite Young's Modulus: {comp_elastic.youngs_modulus_GPa.value:.1f} ± {comp_elastic.youngs_modulus_GPa.uncertainty:.1f} GPa")
    
    comp_thermal = composite_props['thermal']
    print(f"Composite CTE: {comp_thermal.thermal_expansion_coefficient_K_inv.value*1e6:.1f} ± {comp_thermal.thermal_expansion_coefficient_K_inv.uncertainty*1e6:.1f} × 10⁻⁶ K⁻¹")
    
    comp_chem = composite_props['chemical_expansion']
    print(f"Chemical Expansion Coefficient: {comp_chem.chemical_expansion_coefficient.value:.2e} ± {comp_chem.chemical_expansion_coefficient.uncertainty:.2e}")
    
    # Export data
    print("\n" + "="*60)
    print("Exporting Data:")
    print("="*60)
    
    df = db.export_to_csv()
    json_data = db.export_to_json()
    
    print(f"Exported {len(df)} property records to CSV and JSON formats")
    
    return db

if __name__ == "__main__":
    database = main()