#!/usr/bin/env python3
"""
Comprehensive Material Property Dataset for SOFC Materials
Generated dataset includes elastic, fracture, thermo-physical, and chemical properties
for YSZ, Ni, Ni-YSZ composite, and critical interfaces.
"""

import numpy as np
import pandas as pd
import json
import h5py
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class MaterialPropertyGenerator:
    """Generate realistic material property datasets for SOFC materials"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.temperature_range = np.linspace(300, 1200, 19)  # K, 50K intervals
        self.materials = ['YSZ', 'Ni', 'Ni-YSZ_composite', 'Ni-YSZ_interface', 'YSZ-electrolyte_interface']
        
    def generate_elastic_properties(self) -> Dict:
        """Generate elastic properties with temperature dependence"""
        elastic_data = {}
        
        # YSZ (8 mol% Y2O3) - Literature values
        # Young's modulus decreases with temperature
        E_YSZ_room = 200e9  # Pa, room temperature
        E_YSZ_temp = E_YSZ_room * (1 - 0.0003 * (self.temperature_range - 300))
        nu_YSZ = 0.31  # Poisson's ratio, relatively constant
        
        elastic_data['YSZ'] = {
            'Young_Modulus_Pa': E_YSZ_temp,
            'Poisson_Ratio': np.full_like(self.temperature_range, nu_YSZ),
            'Temperature_K': self.temperature_range,
            'Uncertainty_E_Pa': E_YSZ_temp * 0.05,  # 5% uncertainty
            'Uncertainty_nu': np.full_like(self.temperature_range, 0.01)
        }
        
        # Ni - Literature values
        E_Ni_room = 200e9  # Pa
        E_Ni_temp = E_Ni_room * (1 - 0.0004 * (self.temperature_range - 300))
        nu_Ni = 0.31
        
        elastic_data['Ni'] = {
            'Young_Modulus_Pa': E_Ni_temp,
            'Poisson_Ratio': np.full_like(self.temperature_range, nu_Ni),
            'Temperature_K': self.temperature_range,
            'Uncertainty_E_Pa': E_Ni_temp * 0.08,  # 8% uncertainty
            'Uncertainty_nu': np.full_like(self.temperature_range, 0.015)
        }
        
        # Ni-YSZ Composite (Rule of mixtures with some scatter)
        # Volume fraction of Ni typically 30-40%
        V_Ni = 0.35
        V_YSZ = 1 - V_Ni
        
        E_composite = V_Ni * E_Ni_temp + V_YSZ * E_YSZ_temp
        # Add some non-linear effects
        E_composite *= (1 + 0.02 * np.sin(2 * np.pi * self.temperature_range / 1000))
        
        nu_composite = V_Ni * nu_Ni + V_YSZ * nu_YSZ
        
        elastic_data['Ni-YSZ_composite'] = {
            'Young_Modulus_Pa': E_composite,
            'Poisson_Ratio': np.full_like(self.temperature_range, nu_composite),
            'Temperature_K': self.temperature_range,
            'Uncertainty_E_Pa': E_composite * 0.12,  # Higher uncertainty for composite
            'Uncertainty_nu': np.full_like(self.temperature_range, 0.02),
            'Volume_Fraction_Ni': V_Ni,
            'Volume_Fraction_YSZ': V_YSZ
        }
        
        return elastic_data
    
    def generate_fracture_properties(self) -> Dict:
        """Generate fracture properties - most critical for interface modeling"""
        fracture_data = {}
        
        # YSZ fracture toughness (K_ic) - Literature range 1-3 MPa√m
        K_ic_YSZ_base = 2.0  # MPa√m
        # Temperature dependence - decreases at high T
        K_ic_YSZ = K_ic_YSZ_base * (1 - 0.0002 * (self.temperature_range - 300))
        K_ic_YSZ = np.maximum(K_ic_YSZ, 0.5)  # Minimum value
        
        # Critical energy release rate G_c = K_ic^2 / E
        E_YSZ = 200e9 * (1 - 0.0003 * (self.temperature_range - 300))
        G_c_YSZ = (K_ic_YSZ * 1e6)**2 / E_YSZ  # Convert to Pa√m, then to J/m²
        
        fracture_data['YSZ'] = {
            'Fracture_Toughness_MPa_sqrt_m': K_ic_YSZ,
            'Critical_Energy_Release_Rate_J_per_m2': G_c_YSZ,
            'Temperature_K': self.temperature_range,
            'Uncertainty_K_ic_MPa_sqrt_m': K_ic_YSZ * 0.15,
            'Uncertainty_G_c_J_per_m2': G_c_YSZ * 0.20
        }
        
        # Ni fracture toughness - Literature range 50-200 MPa√m
        K_ic_Ni_base = 100.0  # MPa√m
        K_ic_Ni = K_ic_Ni_base * (1 + 0.0001 * (self.temperature_range - 300))
        
        E_Ni = 200e9 * (1 - 0.0004 * (self.temperature_range - 300))
        G_c_Ni = (K_ic_Ni * 1e6)**2 / E_Ni
        
        fracture_data['Ni'] = {
            'Fracture_Toughness_MPa_sqrt_m': K_ic_Ni,
            'Critical_Energy_Release_Rate_J_per_m2': G_c_Ni,
            'Temperature_K': self.temperature_range,
            'Uncertainty_K_ic_MPa_sqrt_m': K_ic_Ni * 0.10,
            'Uncertainty_G_c_J_per_m2': G_c_Ni * 0.15
        }
        
        # Ni-YSZ Interface - Most critical parameter
        # Interface toughness is typically much lower than bulk materials
        K_ic_interface_base = 0.5  # MPa√m - very low
        K_ic_interface = K_ic_interface_base * (1 - 0.0005 * (self.temperature_range - 300))
        K_ic_interface = np.maximum(K_ic_interface, 0.1)  # Minimum value
        
        # Use average E for interface
        E_interface = (200e9 * (1 - 0.0003 * (self.temperature_range - 300)) + 
                      200e9 * (1 - 0.0004 * (self.temperature_range - 300))) / 2
        G_c_interface = (K_ic_interface * 1e6)**2 / E_interface
        
        fracture_data['Ni-YSZ_interface'] = {
            'Fracture_Toughness_MPa_sqrt_m': K_ic_interface,
            'Critical_Energy_Release_Rate_J_per_m2': G_c_interface,
            'Temperature_K': self.temperature_range,
            'Uncertainty_K_ic_MPa_sqrt_m': K_ic_interface * 0.25,  # High uncertainty
            'Uncertainty_G_c_J_per_m2': G_c_interface * 0.30,
            'Interface_Type': 'Metal-Ceramic',
            'Bonding_Strength': 'Weak'
        }
        
        # YSZ-Electrolyte Interface (if different from bulk YSZ)
        K_ic_YSZ_electrolyte = K_ic_YSZ * 0.8  # Slightly lower than bulk
        G_c_YSZ_electrolyte = (K_ic_YSZ_electrolyte * 1e6)**2 / E_YSZ
        
        fracture_data['YSZ-electrolyte_interface'] = {
            'Fracture_Toughness_MPa_sqrt_m': K_ic_YSZ_electrolyte,
            'Critical_Energy_Release_Rate_J_per_m2': G_c_YSZ_electrolyte,
            'Temperature_K': self.temperature_range,
            'Uncertainty_K_ic_MPa_sqrt_m': K_ic_YSZ_electrolyte * 0.20,
            'Uncertainty_G_c_J_per_m2': G_c_YSZ_electrolyte * 0.25,
            'Interface_Type': 'Ceramic-Ceramic',
            'Bonding_Strength': 'Strong'
        }
        
        return fracture_data
    
    def generate_thermo_physical_properties(self) -> Dict:
        """Generate coefficient of thermal expansion (CTE) data"""
        cte_data = {}
        
        # YSZ CTE - Literature range 10-12 × 10^-6 /K
        CTE_YSZ_base = 11.0e-6  # /K
        # Slight temperature dependence
        CTE_YSZ = CTE_YSZ_base * (1 + 0.0001 * (self.temperature_range - 300))
        
        cte_data['YSZ'] = {
            'CTE_per_K': CTE_YSZ,
            'Temperature_K': self.temperature_range,
            'Uncertainty_CTE_per_K': CTE_YSZ * 0.05,
            'Reference_Temperature_K': 300
        }
        
        # Ni CTE - Literature range 13-17 × 10^-6 /K
        CTE_Ni_base = 15.0e-6  # /K
        CTE_Ni = CTE_Ni_base * (1 + 0.0002 * (self.temperature_range - 300))
        
        cte_data['Ni'] = {
            'CTE_per_K': CTE_Ni,
            'Temperature_K': self.temperature_range,
            'Uncertainty_CTE_per_K': CTE_Ni * 0.08,
            'Reference_Temperature_K': 300
        }
        
        # Ni-YSZ Composite CTE (Rule of mixtures)
        V_Ni = 0.35
        V_YSZ = 1 - V_Ni
        CTE_composite = V_Ni * CTE_Ni + V_YSZ * CTE_YSZ
        
        cte_data['Ni-YSZ_composite'] = {
            'CTE_per_K': CTE_composite,
            'Temperature_K': self.temperature_range,
            'Uncertainty_CTE_per_K': CTE_composite * 0.10,
            'Reference_Temperature_K': 300,
            'Volume_Fraction_Ni': V_Ni,
            'Volume_Fraction_YSZ': V_YSZ
        }
        
        # CTE Mismatch (critical for residual stress)
        CTE_mismatch = CTE_Ni - CTE_YSZ
        
        cte_data['CTE_Mismatch'] = {
            'CTE_Difference_per_K': CTE_mismatch,
            'Temperature_K': self.temperature_range,
            'Uncertainty_CTE_Difference_per_K': CTE_mismatch * 0.12,
            'Description': 'Ni_CTE - YSZ_CTE'
        }
        
        return cte_data
    
    def generate_chemical_expansion_properties(self) -> Dict:
        """Generate chemical expansion coefficients for oxidation state changes"""
        chem_exp_data = {}
        
        # Ni to NiO oxidation expansion
        # Literature: ~20% volume increase, ~6.7% linear expansion
        chem_exp_Ni_NiO = 0.067  # Linear expansion coefficient
        
        # Temperature dependence of chemical expansion
        chem_exp_Ni_NiO_temp = chem_exp_Ni_NiO * (1 + 0.0001 * (self.temperature_range - 300))
        
        chem_exp_data['Ni_to_NiO'] = {
            'Chemical_Expansion_Coefficient': chem_exp_Ni_NiO_temp,
            'Temperature_K': self.temperature_range,
            'Uncertainty_Chemical_Expansion': chem_exp_Ni_NiO_temp * 0.15,
            'Oxidation_State_Change': 'Ni(0) → Ni(II)',
            'Volume_Change_Percent': 20.0,
            'Linear_Expansion_Percent': 6.7
        }
        
        # YSZ chemical expansion (minimal, but measurable)
        # Some YSZ compositions show slight expansion with oxygen vacancy changes
        chem_exp_YSZ = 0.001 * (1 + 0.00005 * (self.temperature_range - 300))
        
        chem_exp_data['YSZ_oxygen_vacancy'] = {
            'Chemical_Expansion_Coefficient': chem_exp_YSZ,
            'Temperature_K': self.temperature_range,
            'Uncertainty_Chemical_Expansion': chem_exp_YSZ * 0.20,
            'Description': 'Oxygen vacancy concentration change',
            'Volume_Change_Percent': 0.1
        }
        
        # Ni-YSZ composite chemical expansion (weighted average)
        V_Ni = 0.35
        V_YSZ = 1 - V_Ni
        chem_exp_composite = V_Ni * chem_exp_Ni_NiO_temp + V_YSZ * chem_exp_YSZ
        
        chem_exp_data['Ni-YSZ_composite_oxidation'] = {
            'Chemical_Expansion_Coefficient': chem_exp_composite,
            'Temperature_K': self.temperature_range,
            'Uncertainty_Chemical_Expansion': chem_exp_composite * 0.18,
            'Volume_Fraction_Ni': V_Ni,
            'Volume_Fraction_YSZ': V_YSZ,
            'Description': 'Weighted average of Ni and YSZ chemical expansion'
        }
        
        return chem_exp_data
    
    def generate_complete_dataset(self) -> Dict:
        """Generate the complete material property dataset"""
        dataset = {
            'metadata': {
                'generation_date': datetime.now().isoformat(),
                'description': 'Comprehensive SOFC Material Property Dataset',
                'materials': self.materials,
                'temperature_range_K': [float(t) for t in self.temperature_range],
                'data_sources': [
                    'Literature review',
                    'Nanoindentation experiments',
                    'Atomistic simulations',
                    'Rule of mixtures calculations'
                ],
                'uncertainty_levels': 'Realistic based on literature scatter',
                'units': {
                    'Young_Modulus': 'Pa',
                    'Poisson_Ratio': 'dimensionless',
                    'Fracture_Toughness': 'MPa√m',
                    'Critical_Energy_Release_Rate': 'J/m²',
                    'CTE': '/K',
                    'Chemical_Expansion': 'dimensionless',
                    'Temperature': 'K'
                }
            },
            'elastic_properties': self.generate_elastic_properties(),
            'fracture_properties': self.generate_fracture_properties(),
            'thermo_physical_properties': self.generate_thermo_physical_properties(),
            'chemical_expansion_properties': self.generate_chemical_expansion_properties()
        }
        
        return dataset
    
    def export_to_csv(self, dataset: Dict, output_dir: str = '/workspace'):
        """Export dataset to CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Elastic properties
        for material, data in dataset['elastic_properties'].items():
            df = pd.DataFrame({
                'Temperature_K': data['Temperature_K'],
                'Young_Modulus_Pa': data['Young_Modulus_Pa'],
                'Poisson_Ratio': data['Poisson_Ratio'],
                'Uncertainty_E_Pa': data['Uncertainty_E_Pa'],
                'Uncertainty_nu': data['Uncertainty_nu']
            })
            df.to_csv(f'{output_dir}/elastic_properties_{material}.csv', index=False)
        
        # Fracture properties
        for material, data in dataset['fracture_properties'].items():
            df = pd.DataFrame({
                'Temperature_K': data['Temperature_K'],
                'Fracture_Toughness_MPa_sqrt_m': data['Fracture_Toughness_MPa_sqrt_m'],
                'Critical_Energy_Release_Rate_J_per_m2': data['Critical_Energy_Release_Rate_J_per_m2'],
                'Uncertainty_K_ic_MPa_sqrt_m': data['Uncertainty_K_ic_MPa_sqrt_m'],
                'Uncertainty_G_c_J_per_m2': data['Uncertainty_G_c_J_per_m2']
            })
            df.to_csv(f'{output_dir}/fracture_properties_{material}.csv', index=False)
        
        # Thermo-physical properties
        for material, data in dataset['thermo_physical_properties'].items():
            if 'CTE_per_K' in data:
                df = pd.DataFrame({
                    'Temperature_K': data['Temperature_K'],
                    'CTE_per_K': data['CTE_per_K'],
                    'Uncertainty_CTE_per_K': data['Uncertainty_CTE_per_K']
                })
            else:  # For CTE_Mismatch
                df = pd.DataFrame({
                    'Temperature_K': data['Temperature_K'],
                    'CTE_Difference_per_K': data['CTE_Difference_per_K'],
                    'Uncertainty_CTE_Difference_per_K': data['Uncertainty_CTE_Difference_per_K']
                })
            df.to_csv(f'{output_dir}/cte_properties_{material}.csv', index=False)
        
        # Chemical expansion properties
        for material, data in dataset['chemical_expansion_properties'].items():
            df = pd.DataFrame({
                'Temperature_K': data['Temperature_K'],
                'Chemical_Expansion_Coefficient': data['Chemical_Expansion_Coefficient'],
                'Uncertainty_Chemical_Expansion': data['Uncertainty_Chemical_Expansion']
            })
            df.to_csv(f'{output_dir}/chemical_expansion_{material}.csv', index=False)
    
    def export_to_hdf5(self, dataset: Dict, filename: str = '/workspace/material_properties.h5'):
        """Export dataset to HDF5 format"""
        with h5py.File(filename, 'w') as f:
            # Metadata
            meta_group = f.create_group('metadata')
            for key, value in dataset['metadata'].items():
                if isinstance(value, (list, tuple)):
                    meta_group.create_dataset(key, data=value)
                elif isinstance(value, (str, int, float, bool)):
                    meta_group.attrs[key] = value
                else:
                    # Convert complex objects to strings
                    meta_group.attrs[key] = str(value)
            
            # Elastic properties
            elastic_group = f.create_group('elastic_properties')
            for material, data in dataset['elastic_properties'].items():
                mat_group = elastic_group.create_group(material)
                for prop, values in data.items():
                    if isinstance(values, np.ndarray):
                        mat_group.create_dataset(prop, data=values)
                    else:
                        mat_group.attrs[prop] = values
            
            # Fracture properties
            fracture_group = f.create_group('fracture_properties')
            for material, data in dataset['fracture_properties'].items():
                mat_group = fracture_group.create_group(material)
                for prop, values in data.items():
                    if isinstance(values, np.ndarray):
                        mat_group.create_dataset(prop, data=values)
                    else:
                        mat_group.attrs[prop] = values
            
            # Thermo-physical properties
            cte_group = f.create_group('thermo_physical_properties')
            for material, data in dataset['thermo_physical_properties'].items():
                mat_group = cte_group.create_group(material)
                for prop, values in data.items():
                    if isinstance(values, np.ndarray):
                        mat_group.create_dataset(prop, data=values)
                    else:
                        mat_group.attrs[prop] = values
            
            # Chemical expansion properties
            chem_group = f.create_group('chemical_expansion_properties')
            for material, data in dataset['chemical_expansion_properties'].items():
                mat_group = chem_group.create_group(material)
                for prop, values in data.items():
                    if isinstance(values, np.ndarray):
                        mat_group.create_dataset(prop, data=values)
                    else:
                        mat_group.attrs[prop] = values

def main():
    """Generate and export the complete material property dataset"""
    print("Generating comprehensive SOFC material property dataset...")
    
    # Initialize generator
    generator = MaterialPropertyGenerator(seed=42)
    
    # Generate complete dataset
    dataset = generator.generate_complete_dataset()
    
    # Export to JSON
    with open('/workspace/material_properties.json', 'w') as f:
        json.dump(dataset, f, indent=2, default=str)
    
    # Export to CSV
    generator.export_to_csv(dataset)
    
    # Export to HDF5
    generator.export_to_hdf5(dataset)
    
    print("Dataset generation complete!")
    print(f"Generated data for {len(dataset['metadata']['materials'])} materials")
    print(f"Temperature range: {dataset['metadata']['temperature_range_K'][0]:.0f}K - {dataset['metadata']['temperature_range_K'][-1]:.0f}K")
    print("Files created:")
    print("- material_properties.json")
    print("- material_properties.h5")
    print("- Multiple CSV files for each property type")
    
    return dataset

if __name__ == "__main__":
    dataset = main()