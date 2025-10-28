"""
SOFC Material Properties Database
==================================
Comprehensive dataset of thermo-physical, mechanical, and electrochemical 
properties for Solid Oxide Fuel Cell components.

Author: Generated Dataset
Date: 2025
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class ThermoPhysicalProperties:
    """Thermo-physical properties of SOFC materials"""
    thermal_expansion_coefficient: float  # 1/K
    thermal_conductivity: float  # W/(m·K)
    specific_heat_capacity: float  # J/(kg·K)
    density: float  # kg/m3
    porosity: float  # dimensionless
    temperature: float  # K


@dataclass
class MechanicalProperties:
    """Mechanical properties including elasticity and creep"""
    youngs_modulus: float  # Pa
    poissons_ratio: float  # dimensionless
    # Norton-Bailey creep parameters: strain_rate = B * sigma^n * exp(-Q/RT)
    creep_B: Optional[float] = None  # 1/(Pa^n·s)
    creep_n: Optional[float] = None  # dimensionless
    creep_Q: Optional[float] = None  # J/mol


@dataclass
class JohnsonCookPlasticity:
    """Johnson-Cook plasticity model parameters"""
    A: float  # Initial yield stress (Pa)
    B: float  # Hardening modulus (Pa)
    n: float  # Hardening exponent
    C: float  # Strain rate sensitivity
    m: float  # Thermal softening exponent


@dataclass
class ElectrochemicalProperties:
    """Electrochemical properties for SOFC materials"""
    electronic_conductivity: float  # S/m
    ionic_conductivity: float  # S/m
    exchange_current_density: Optional[float] = None  # A/m2
    activation_overpotential_coeff: Optional[float] = None  # dimensionless
    activation_energy: Optional[float] = None  # J/mol


class SOFCMaterialDatabase:
    """Database containing all SOFC material properties"""
    
    def __init__(self):
        self.materials = self._initialize_database()
    
    def _initialize_database(self) -> Dict:
        """Initialize the complete material database"""
        
        # ===========================
        # ANODE: Ni-YSZ
        # ===========================
        ni_ysz_thermo = ThermoPhysicalProperties(
            thermal_expansion_coefficient=12.5e-6,
            thermal_conductivity=6.2,
            specific_heat_capacity=450.0,
            density=5800.0,
            porosity=0.35,
            temperature=1073.0
        )
        
        ni_ysz_mech = MechanicalProperties(
            youngs_modulus=45e9,
            poissons_ratio=0.28,
            creep_B=2.8e-13,
            creep_n=2.1,
            creep_Q=320e3
        )
        
        ni_ysz_plasticity = JohnsonCookPlasticity(
            A=180e6,
            B=420e6,
            n=0.42,
            C=0.015,
            m=1.1
        )
        
        ni_ysz_electrochem = ElectrochemicalProperties(
            electronic_conductivity=15000.0,
            ionic_conductivity=0.8,
            exchange_current_density=4500.0,
            activation_overpotential_coeff=0.5
        )
        
        # ===========================
        # ELECTROLYTE: 8YSZ
        # ===========================
        ysz_thermo = ThermoPhysicalProperties(
            thermal_expansion_coefficient=10.5e-6,
            thermal_conductivity=2.7,
            specific_heat_capacity=470.0,
            density=5900.0,
            porosity=0.02,
            temperature=1073.0
        )
        
        ysz_mech = MechanicalProperties(
            youngs_modulus=200e9,
            poissons_ratio=0.31,
            creep_B=1.5e-15,
            creep_n=1.0,
            creep_Q=520e3
        )
        
        ysz_electrochem = ElectrochemicalProperties(
            electronic_conductivity=0.001,
            ionic_conductivity=3.2,
            activation_energy=80e3
        )
        
        # ===========================
        # ELECTROLYTE: CGO
        # ===========================
        cgo_thermo = ThermoPhysicalProperties(
            thermal_expansion_coefficient=12.5e-6,
            thermal_conductivity=3.5,
            specific_heat_capacity=450.0,
            density=7200.0,
            porosity=0.03,
            temperature=873.0
        )
        
        cgo_mech = MechanicalProperties(
            youngs_modulus=180e9,
            poissons_ratio=0.30
        )
        
        cgo_electrochem = ElectrochemicalProperties(
            electronic_conductivity=0.05,
            ionic_conductivity=8.5,
            activation_energy=65e3
        )
        
        # ===========================
        # CATHODE: LSM
        # ===========================
        lsm_thermo = ThermoPhysicalProperties(
            thermal_expansion_coefficient=11.8e-6,
            thermal_conductivity=3.5,
            specific_heat_capacity=520.0,
            density=5200.0,
            porosity=0.30,
            temperature=1073.0
        )
        
        lsm_mech = MechanicalProperties(
            youngs_modulus=55e9,
            poissons_ratio=0.25
        )
        
        lsm_electrochem = ElectrochemicalProperties(
            electronic_conductivity=25000.0,
            ionic_conductivity=0.05,
            exchange_current_density=2800.0,
            activation_overpotential_coeff=0.5
        )
        
        # ===========================
        # CATHODE: LSM-YSZ Composite
        # ===========================
        lsm_ysz_thermo = ThermoPhysicalProperties(
            thermal_expansion_coefficient=11.2e-6,
            thermal_conductivity=4.0,
            specific_heat_capacity=485.0,
            density=5400.0,
            porosity=0.35,
            temperature=1073.0
        )
        
        lsm_ysz_mech = MechanicalProperties(
            youngs_modulus=50e9,
            poissons_ratio=0.27
        )
        
        lsm_ysz_electrochem = ElectrochemicalProperties(
            electronic_conductivity=18000.0,
            ionic_conductivity=1.2,
            exchange_current_density=3500.0,
            activation_overpotential_coeff=0.5
        )
        
        # ===========================
        # CATHODE: LSCF (Intermediate Temperature)
        # ===========================
        lscf_thermo = ThermoPhysicalProperties(
            thermal_expansion_coefficient=14.5e-6,
            thermal_conductivity=2.8,
            specific_heat_capacity=510.0,
            density=5500.0,
            porosity=0.35,
            temperature=873.0
        )
        
        lscf_mech = MechanicalProperties(
            youngs_modulus=60e9,
            poissons_ratio=0.26
        )
        
        lscf_electrochem = ElectrochemicalProperties(
            electronic_conductivity=35000.0,
            ionic_conductivity=2.5,
            exchange_current_density=5200.0,
            activation_overpotential_coeff=0.5
        )
        
        # ===========================
        # INTERCONNECT: Crofer 22 APU
        # ===========================
        crofer_thermo = ThermoPhysicalProperties(
            thermal_expansion_coefficient=12.0e-6,
            thermal_conductivity=22.0,
            specific_heat_capacity=600.0,
            density=7600.0,
            porosity=0.0,
            temperature=1073.0
        )
        
        crofer_mech = MechanicalProperties(
            youngs_modulus=170e9,
            poissons_ratio=0.30,
            creep_B=8.5e-12,
            creep_n=4.2,
            creep_Q=280e3
        )
        
        crofer_electrochem = ElectrochemicalProperties(
            electronic_conductivity=1.2e6,
            ionic_conductivity=0.0
        )
        
        # Assemble database
        database = {
            'anode': {
                'Ni-YSZ': {
                    'thermo_physical': ni_ysz_thermo,
                    'mechanical': ni_ysz_mech,
                    'plasticity': ni_ysz_plasticity,
                    'electrochemical': ni_ysz_electrochem
                }
            },
            'electrolyte': {
                '8YSZ': {
                    'thermo_physical': ysz_thermo,
                    'mechanical': ysz_mech,
                    'electrochemical': ysz_electrochem
                },
                'CGO': {
                    'thermo_physical': cgo_thermo,
                    'mechanical': cgo_mech,
                    'electrochemical': cgo_electrochem
                }
            },
            'cathode': {
                'LSM': {
                    'thermo_physical': lsm_thermo,
                    'mechanical': lsm_mech,
                    'electrochemical': lsm_electrochem
                },
                'LSM-YSZ': {
                    'thermo_physical': lsm_ysz_thermo,
                    'mechanical': lsm_ysz_mech,
                    'electrochemical': lsm_ysz_electrochem
                },
                'LSCF': {
                    'thermo_physical': lscf_thermo,
                    'mechanical': lscf_mech,
                    'electrochemical': lscf_electrochem
                }
            },
            'interconnect': {
                'Crofer_22_APU': {
                    'thermo_physical': crofer_thermo,
                    'mechanical': crofer_mech,
                    'electrochemical': crofer_electrochem
                }
            }
        }
        
        return database
    
    def get_material_properties(self, component: str, material: str) -> Dict:
        """
        Retrieve all properties for a specific material
        
        Args:
            component: 'anode', 'electrolyte', 'cathode', or 'interconnect'
            material: Material name (e.g., 'Ni-YSZ', '8YSZ', 'LSM')
        
        Returns:
            Dictionary containing all material properties
        """
        try:
            return self.materials[component.lower()][material]
        except KeyError:
            raise ValueError(f"Material {material} not found in {component} database")
    
    def get_thermal_expansion_coefficient(self, component: str, material: str) -> float:
        """Get TEC in 1/K"""
        props = self.get_material_properties(component, material)
        return props['thermo_physical'].thermal_expansion_coefficient
    
    def get_youngs_modulus(self, component: str, material: str) -> float:
        """Get Young's modulus in Pa"""
        props = self.get_material_properties(component, material)
        return props['mechanical'].youngs_modulus
    
    def get_ionic_conductivity(self, component: str, material: str, 
                               temperature: Optional[float] = None) -> float:
        """
        Get ionic conductivity in S/m
        
        If temperature is provided and activation energy is available,
        uses Arrhenius equation to calculate conductivity at that temperature.
        """
        props = self.get_material_properties(component, material)
        sigma_0 = props['electrochemical'].ionic_conductivity
        
        if temperature is not None and props['electrochemical'].activation_energy is not None:
            T_ref = props['thermo_physical'].temperature
            Ea = props['electrochemical'].activation_energy
            R = 8.314  # J/(mol·K)
            
            # Arrhenius equation: sigma(T) = sigma_0 * exp(-Ea/R * (1/T - 1/T_ref))
            sigma = sigma_0 * np.exp(-Ea / R * (1/temperature - 1/T_ref))
            return sigma
        
        return sigma_0
    
    def calculate_creep_rate(self, component: str, material: str, 
                            stress: float, temperature: float) -> float:
        """
        Calculate creep strain rate using Norton-Bailey law
        
        strain_rate = B * sigma^n * exp(-Q / (R*T))
        
        Args:
            component: Component type
            material: Material name
            stress: Applied stress (Pa)
            temperature: Temperature (K)
        
        Returns:
            Creep strain rate (1/s)
        """
        props = self.get_material_properties(component, material)
        mech = props['mechanical']
        
        if mech.creep_B is None:
            raise ValueError(f"Creep parameters not available for {material}")
        
        R = 8.314  # J/(mol·K)
        strain_rate = (mech.creep_B * 
                      np.power(stress, mech.creep_n) * 
                      np.exp(-mech.creep_Q / (R * temperature)))
        
        return strain_rate
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export entire database to pandas DataFrame for easy viewing"""
        data = []
        
        for component, materials in self.materials.items():
            for material_name, properties in materials.items():
                
                # Thermo-physical
                tp = properties['thermo_physical']
                base_data = {
                    'Component': component,
                    'Material': material_name,
                    'TEC (1/K)': tp.thermal_expansion_coefficient,
                    'Thermal Conductivity (W/m·K)': tp.thermal_conductivity,
                    'Specific Heat (J/kg·K)': tp.specific_heat_capacity,
                    'Density (kg/m3)': tp.density,
                    'Porosity': tp.porosity,
                    'Reference Temp (K)': tp.temperature
                }
                
                # Mechanical
                mech = properties['mechanical']
                base_data.update({
                    'Youngs Modulus (GPa)': mech.youngs_modulus / 1e9,
                    'Poissons Ratio': mech.poissons_ratio,
                    'Creep B': mech.creep_B,
                    'Creep n': mech.creep_n,
                    'Creep Q (kJ/mol)': mech.creep_Q / 1e3 if mech.creep_Q else None
                })
                
                # Electrochemical
                ec = properties['electrochemical']
                base_data.update({
                    'Electronic Cond. (S/m)': ec.electronic_conductivity,
                    'Ionic Cond. (S/m)': ec.ionic_conductivity,
                    'Exchange Current Density (A/m2)': ec.exchange_current_density,
                    'Activation Coeff.': ec.activation_overpotential_coeff
                })
                
                data.append(base_data)
        
        return pd.DataFrame(data)
    
    def print_summary(self):
        """Print a summary of available materials"""
        print("=" * 70)
        print("SOFC MATERIAL PROPERTIES DATABASE")
        print("=" * 70)
        
        for component, materials in self.materials.items():
            print(f"\n{component.upper()}:")
            for material_name in materials.keys():
                print(f"  - {material_name}")
        
        print("\n" + "=" * 70)


# Example usage
if __name__ == "__main__":
    # Initialize database
    db = SOFCMaterialDatabase()
    
    # Print summary
    db.print_summary()
    
    # Example: Get Ni-YSZ properties
    print("\n\nExample: Ni-YSZ Anode Properties")
    print("-" * 50)
    ni_ysz = db.get_material_properties('anode', 'Ni-YSZ')
    print(f"TEC: {ni_ysz['thermo_physical'].thermal_expansion_coefficient*1e6:.2f} × 10⁻⁶ 1/K")
    print(f"Young's Modulus: {ni_ysz['mechanical'].youngs_modulus/1e9:.1f} GPa")
    print(f"Porosity: {ni_ysz['thermo_physical'].porosity:.2%}")
    print(f"Electronic Conductivity: {ni_ysz['electrochemical'].electronic_conductivity:.0f} S/m")
    
    # Example: Calculate creep rate
    print("\n\nExample: Creep Calculation for Ni-YSZ at 1073K, 50 MPa")
    print("-" * 50)
    creep_rate = db.calculate_creep_rate('anode', 'Ni-YSZ', 
                                         stress=50e6, temperature=1073)
    print(f"Creep strain rate: {creep_rate:.3e} 1/s")
    
    # Example: Temperature-dependent conductivity
    print("\n\nExample: YSZ Ionic Conductivity vs Temperature")
    print("-" * 50)
    temperatures = [873, 973, 1073, 1173]
    for T in temperatures:
        sigma = db.get_ionic_conductivity('electrolyte', '8YSZ', temperature=T)
        print(f"T = {T}K: σ_ion = {sigma:.3f} S/m")
    
    # Export to DataFrame
    print("\n\nExporting to DataFrame...")
    df = db.export_to_dataframe()
    print(df.to_string())
    
    # Save to CSV
    df.to_csv('sofc_materials_summary.csv', index=False)
    print("\nData exported to 'sofc_materials_summary.csv'")
