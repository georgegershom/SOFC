"""
SOFC Material Properties Dataset Generator
=========================================

This module generates comprehensive material properties for Solid Oxide Fuel Cell (SOFC) components
including anode (Ni-YSZ), electrolyte (8YSZ), cathode (LSM), and interconnects.

Properties include:
- Thermal expansion coefficients
- Mechanical properties (Young's modulus, Poisson's ratio, density)
- Creep parameters (Norton-Bailey model)
- Plasticity parameters (Johnson-Cook model)
- Porosity data
- Thermal properties (conductivity, specific heat)
- Electrochemical properties
"""

import numpy as np
import pandas as pd
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

@dataclass
class ThermalProperties:
    """Thermal properties for SOFC materials"""
    thermal_expansion_coefficient: float  # 1/K
    thermal_conductivity: float  # W/m·K
    specific_heat_capacity: float  # J/kg·K
    temperature_range: tuple  # (min_temp, max_temp) in K

@dataclass
class MechanicalProperties:
    """Mechanical properties for SOFC materials"""
    youngs_modulus: float  # GPa
    poissons_ratio: float  # dimensionless
    density: float  # kg/m³
    ultimate_tensile_strength: float  # MPa
    yield_strength: float  # MPa

@dataclass
class CreepParameters:
    """Norton-Bailey creep model parameters: ε̇ = B * σⁿ * exp(-Q/RT)"""
    B: float  # Pre-exponential factor (1/Pa^n·s)
    n: float  # Stress exponent (dimensionless)
    Q: float  # Activation energy (J/mol)
    temperature_range: tuple  # (min_temp, max_temp) in K

@dataclass
class PlasticityParameters:
    """Johnson-Cook plasticity model parameters"""
    A: float  # Initial yield stress (MPa)
    B: float  # Hardening constant (MPa)
    n: float  # Hardening exponent
    C: float  # Strain rate constant
    m: float  # Thermal softening exponent
    T_melt: float  # Melting temperature (K)
    T_room: float  # Room temperature (K)

@dataclass
class PorosityData:
    """Porosity characteristics"""
    porosity: float  # Volume fraction (0-1)
    pore_size_mean: float  # μm
    pore_size_std: float  # μm
    tortuosity: float  # dimensionless

@dataclass
class ElectrochemicalProperties:
    """Electrochemical properties"""
    ionic_conductivity: float  # S/m
    electronic_conductivity: float  # S/m
    activation_overpotential: float  # V
    exchange_current_density: float  # A/m²
    activation_energy_ionic: float  # J/mol
    activation_energy_electronic: float  # J/mol

@dataclass
class SOFCMaterial:
    """Complete material properties for SOFC component"""
    name: str
    composition: str
    thermal: ThermalProperties
    mechanical: MechanicalProperties
    creep: Optional[CreepParameters]
    plasticity: Optional[PlasticityParameters]
    porosity: PorosityData
    electrochemical: ElectrochemicalProperties

class SOFCDatasetGenerator:
    """Generator for SOFC material properties dataset"""
    
    def __init__(self):
        self.materials = {}
        self._generate_all_materials()
    
    def _generate_all_materials(self):
        """Generate properties for all SOFC materials"""
        self.materials = {
            'ni_ysz_anode': self._generate_ni_ysz_anode(),
            '8ysz_electrolyte': self._generate_8ysz_electrolyte(),
            'lsm_cathode': self._generate_lsm_cathode(),
            'crofer22_interconnect': self._generate_crofer22_interconnect(),
            'lscf_cathode': self._generate_lscf_cathode(),
            'cgd_electrolyte': self._generate_cgd_electrolyte()
        }
    
    def _generate_ni_ysz_anode(self) -> SOFCMaterial:
        """Generate Ni-YSZ anode properties"""
        return SOFCMaterial(
            name="Ni-YSZ Anode",
            composition="60% Ni + 40% 8YSZ",
            thermal=ThermalProperties(
                thermal_expansion_coefficient=12.5e-6,  # 1/K
                thermal_conductivity=6.2,  # W/m·K at 1073K
                specific_heat_capacity=450,  # J/kg·K
                temperature_range=(673, 1273)
            ),
            mechanical=MechanicalProperties(
                youngs_modulus=45.0,  # GPa
                poissons_ratio=0.31,
                density=6800,  # kg/m³
                ultimate_tensile_strength=180,  # MPa
                yield_strength=120  # MPa
            ),
            creep=CreepParameters(
                B=2.5e-15,  # 1/Pa^n·s
                n=2.8,
                Q=320000,  # J/mol
                temperature_range=(973, 1273)
            ),
            plasticity=PlasticityParameters(
                A=120,  # MPa
                B=85,   # MPa
                n=0.42,
                C=0.025,
                m=0.8,
                T_melt=1728,  # K
                T_room=298    # K
            ),
            porosity=PorosityData(
                porosity=0.35,  # 35%
                pore_size_mean=1.2,  # μm
                pore_size_std=0.8,
                tortuosity=3.2
            ),
            electrochemical=ElectrochemicalProperties(
                ionic_conductivity=0.02,  # S/m at 1073K
                electronic_conductivity=15000,  # S/m
                activation_overpotential=0.65,  # V
                exchange_current_density=5500,  # A/m²
                activation_energy_ionic=95000,  # J/mol
                activation_energy_electronic=25000  # J/mol
            )
        )
    
    def _generate_8ysz_electrolyte(self) -> SOFCMaterial:
        """Generate 8YSZ electrolyte properties"""
        return SOFCMaterial(
            name="8YSZ Electrolyte",
            composition="ZrO₂ + 8 mol% Y₂O₃",
            thermal=ThermalProperties(
                thermal_expansion_coefficient=10.8e-6,  # 1/K
                thermal_conductivity=2.16,  # W/m·K at 1073K
                specific_heat_capacity=470,  # J/kg·K
                temperature_range=(673, 1273)
            ),
            mechanical=MechanicalProperties(
                youngs_modulus=200.0,  # GPa
                poissons_ratio=0.31,
                density=6100,  # kg/m³
                ultimate_tensile_strength=250,  # MPa
                yield_strength=180  # MPa
            ),
            creep=CreepParameters(
                B=8.2e-18,  # 1/Pa^n·s
                n=1.0,
                Q=520000,  # J/mol
                temperature_range=(1073, 1273)
            ),
            plasticity=None,  # Ceramic - brittle behavior
            porosity=PorosityData(
                porosity=0.05,  # 5% - dense electrolyte
                pore_size_mean=0.1,  # μm
                pore_size_std=0.05,
                tortuosity=1.8
            ),
            electrochemical=ElectrochemicalProperties(
                ionic_conductivity=0.13,  # S/m at 1073K
                electronic_conductivity=1e-8,  # S/m (negligible)
                activation_overpotential=0.0,  # V (pure ionic conductor)
                exchange_current_density=0,  # A/m² (no electrochemical reaction)
                activation_energy_ionic=87000,  # J/mol
                activation_energy_electronic=0  # J/mol
            )
        )
    
    def _generate_lsm_cathode(self) -> SOFCMaterial:
        """Generate LSM cathode properties"""
        return SOFCMaterial(
            name="LSM Cathode",
            composition="La₀.₈Sr₀.₂MnO₃",
            thermal=ThermalProperties(
                thermal_expansion_coefficient=11.2e-6,  # 1/K
                thermal_conductivity=3.4,  # W/m·K at 1073K
                specific_heat_capacity=420,  # J/kg·K
                temperature_range=(673, 1273)
            ),
            mechanical=MechanicalProperties(
                youngs_modulus=120.0,  # GPa
                poissons_ratio=0.28,
                density=6500,  # kg/m³
                ultimate_tensile_strength=95,  # MPa
                yield_strength=65  # MPa
            ),
            creep=CreepParameters(
                B=1.8e-16,  # 1/Pa^n·s
                n=1.2,
                Q=380000,  # J/mol
                temperature_range=(973, 1273)
            ),
            plasticity=None,  # Ceramic - brittle behavior
            porosity=PorosityData(
                porosity=0.30,  # 30%
                pore_size_mean=0.8,  # μm
                pore_size_std=0.5,
                tortuosity=2.8
            ),
            electrochemical=ElectrochemicalProperties(
                ionic_conductivity=0.008,  # S/m at 1073K
                electronic_conductivity=280,  # S/m
                activation_overpotential=0.15,  # V
                exchange_current_density=1200,  # A/m²
                activation_energy_ionic=150000,  # J/mol
                activation_energy_electronic=45000  # J/mol
            )
        )
    
    def _generate_crofer22_interconnect(self) -> SOFCMaterial:
        """Generate Crofer22 APU interconnect properties"""
        return SOFCMaterial(
            name="Crofer22 APU Interconnect",
            composition="Fe-22Cr-0.5Mn-0.3Ti-0.07La",
            thermal=ThermalProperties(
                thermal_expansion_coefficient=11.8e-6,  # 1/K
                thermal_conductivity=25.0,  # W/m·K at 1073K
                specific_heat_capacity=460,  # J/kg·K
                temperature_range=(673, 1273)
            ),
            mechanical=MechanicalProperties(
                youngs_modulus=220.0,  # GPa
                poissons_ratio=0.30,
                density=7800,  # kg/m³
                ultimate_tensile_strength=550,  # MPa
                yield_strength=380  # MPa
            ),
            creep=CreepParameters(
                B=5.2e-20,  # 1/Pa^n·s
                n=4.5,
                Q=420000,  # J/mol
                temperature_range=(973, 1273)
            ),
            plasticity=PlasticityParameters(
                A=380,  # MPa
                B=275,  # MPa
                n=0.36,
                C=0.018,
                m=0.9,
                T_melt=1773,  # K
                T_room=298   # K
            ),
            porosity=PorosityData(
                porosity=0.02,  # 2% - dense metallic
                pore_size_mean=0.05,  # μm
                pore_size_std=0.02,
                tortuosity=1.2
            ),
            electrochemical=ElectrochemicalProperties(
                ionic_conductivity=1e-10,  # S/m (negligible)
                electronic_conductivity=850000,  # S/m
                activation_overpotential=0.0,  # V
                exchange_current_density=0,  # A/m²
                activation_energy_ionic=0,  # J/mol
                activation_energy_electronic=15000  # J/mol
            )
        )
    
    def _generate_lscf_cathode(self) -> SOFCMaterial:
        """Generate LSCF cathode properties (alternative to LSM)"""
        return SOFCMaterial(
            name="LSCF Cathode",
            composition="La₀.₆Sr₀.₄Co₀.₂Fe₀.₈O₃",
            thermal=ThermalProperties(
                thermal_expansion_coefficient=15.8e-6,  # 1/K
                thermal_conductivity=2.8,  # W/m·K at 1073K
                specific_heat_capacity=440,  # J/kg·K
                temperature_range=(673, 1273)
            ),
            mechanical=MechanicalProperties(
                youngs_modulus=95.0,  # GPa
                poissons_ratio=0.26,
                density=6800,  # kg/m³
                ultimate_tensile_strength=75,  # MPa
                yield_strength=50  # MPa
            ),
            creep=CreepParameters(
                B=3.2e-15,  # 1/Pa^n·s
                n=1.8,
                Q=350000,  # J/mol
                temperature_range=(973, 1273)
            ),
            plasticity=None,  # Ceramic - brittle behavior
            porosity=PorosityData(
                porosity=0.32,  # 32%
                pore_size_mean=0.6,  # μm
                pore_size_std=0.4,
                tortuosity=2.5
            ),
            electrochemical=ElectrochemicalProperties(
                ionic_conductivity=0.15,  # S/m at 1073K
                electronic_conductivity=450,  # S/m
                activation_overpotential=0.08,  # V
                exchange_current_density=8500,  # A/m²
                activation_energy_ionic=95000,  # J/mol
                activation_energy_electronic=35000  # J/mol
            )
        )
    
    def _generate_cgd_electrolyte(self) -> SOFCMaterial:
        """Generate CGO (Ceria-Gadolinia) electrolyte properties"""
        return SOFCMaterial(
            name="CGO Electrolyte",
            composition="Ce₀.₈Gd₀.₂O₁.₉",
            thermal=ThermalProperties(
                thermal_expansion_coefficient=12.8e-6,  # 1/K
                thermal_conductivity=2.8,  # W/m·K at 1073K
                specific_heat_capacity=380,  # J/kg·K
                temperature_range=(673, 1073)
            ),
            mechanical=MechanicalProperties(
                youngs_modulus=180.0,  # GPa
                poissons_ratio=0.32,
                density=7200,  # kg/m³
                ultimate_tensile_strength=220,  # MPa
                yield_strength=160  # MPa
            ),
            creep=CreepParameters(
                B=1.5e-17,  # 1/Pa^n·s
                n=1.1,
                Q=480000,  # J/mol
                temperature_range=(873, 1073)
            ),
            plasticity=None,  # Ceramic - brittle behavior
            porosity=PorosityData(
                porosity=0.04,  # 4% - dense electrolyte
                pore_size_mean=0.08,  # μm
                pore_size_std=0.04,
                tortuosity=1.6
            ),
            electrochemical=ElectrochemicalProperties(
                ionic_conductivity=0.08,  # S/m at 873K
                electronic_conductivity=5e-6,  # S/m (small electronic contribution)
                activation_overpotential=0.0,  # V
                exchange_current_density=0,  # A/m²
                activation_energy_ionic=65000,  # J/mol
                activation_energy_electronic=120000  # J/mol
            )
        )
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert materials data to pandas DataFrame"""
        data = []
        for material_id, material in self.materials.items():
            row = {
                'Material_ID': material_id,
                'Name': material.name,
                'Composition': material.composition,
                
                # Thermal properties
                'TEC_1/K': material.thermal.thermal_expansion_coefficient,
                'Thermal_Conductivity_W/mK': material.thermal.thermal_conductivity,
                'Specific_Heat_J/kgK': material.thermal.specific_heat_capacity,
                'Temp_Range_K': f"{material.thermal.temperature_range[0]}-{material.thermal.temperature_range[1]}",
                
                # Mechanical properties
                'Youngs_Modulus_GPa': material.mechanical.youngs_modulus,
                'Poissons_Ratio': material.mechanical.poissons_ratio,
                'Density_kg/m3': material.mechanical.density,
                'UTS_MPa': material.mechanical.ultimate_tensile_strength,
                'Yield_Strength_MPa': material.mechanical.yield_strength,
                
                # Creep parameters
                'Creep_B': material.creep.B if material.creep else None,
                'Creep_n': material.creep.n if material.creep else None,
                'Creep_Q_J/mol': material.creep.Q if material.creep else None,
                
                # Plasticity parameters
                'JC_A_MPa': material.plasticity.A if material.plasticity else None,
                'JC_B_MPa': material.plasticity.B if material.plasticity else None,
                'JC_n': material.plasticity.n if material.plasticity else None,
                'JC_C': material.plasticity.C if material.plasticity else None,
                'JC_m': material.plasticity.m if material.plasticity else None,
                
                # Porosity
                'Porosity': material.porosity.porosity,
                'Pore_Size_Mean_um': material.porosity.pore_size_mean,
                'Tortuosity': material.porosity.tortuosity,
                
                # Electrochemical
                'Ionic_Conductivity_S/m': material.electrochemical.ionic_conductivity,
                'Electronic_Conductivity_S/m': material.electrochemical.electronic_conductivity,
                'Activation_Overpotential_V': material.electrochemical.activation_overpotential,
                'Exchange_Current_Density_A/m2': material.electrochemical.exchange_current_density,
                'Ea_ionic_J/mol': material.electrochemical.activation_energy_ionic,
                'Ea_electronic_J/mol': material.electrochemical.activation_energy_electronic
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def to_json(self, filename: str = None) -> str:
        """Export materials data to JSON format"""
        json_data = {}
        for material_id, material in self.materials.items():
            json_data[material_id] = asdict(material)
        
        json_str = json.dumps(json_data, indent=2, default=str)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def get_material(self, material_id: str) -> SOFCMaterial:
        """Get specific material by ID"""
        return self.materials.get(material_id)
    
    def list_materials(self) -> List[str]:
        """List all available material IDs"""
        return list(self.materials.keys())

# Example usage and data generation
if __name__ == "__main__":
    # Generate the dataset
    generator = SOFCDatasetGenerator()
    
    # Export to different formats
    df = generator.to_dataframe()
    df.to_csv('/workspace/sofc_materials_dataset.csv', index=False)
    
    # Export to JSON
    generator.to_json('/workspace/sofc_materials_dataset.json')
    
    # Print summary
    print("SOFC Material Properties Dataset Generated")
    print("=" * 50)
    print(f"Number of materials: {len(generator.materials)}")
    print("\nMaterials included:")
    for material_id in generator.list_materials():
        material = generator.get_material(material_id)
        print(f"- {material.name} ({material_id})")
    
    print(f"\nDataset saved to:")
    print(f"- CSV: /workspace/sofc_materials_dataset.csv")
    print(f"- JSON: /workspace/sofc_materials_dataset.json")
    
    # Display first few rows
    print("\nSample data (first 3 columns):")
    print(df[['Material_ID', 'Name', 'Composition']].to_string(index=False))