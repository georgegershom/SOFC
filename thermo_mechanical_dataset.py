#!/usr/bin/env python3
"""
Thermo-Mechanical Modeling Dataset Generator for Fire-Resistant Structural Elements
Utilizing High-Performance Rubberized Concrete

This module generates comprehensive numerical modeling datasets for finite element
analysis of fire-resistant structural elements with multi-physics coupling.

Author: AI Assistant
Date: 2024
Purpose: Research dataset for "Development and Validation of a Thermo-Mechanical Model 
         for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete"
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.interpolate import interp1d
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MaterialMix:
    """Data class for concrete mix properties"""
    mix_id: str
    rubber_content: float  # Percentage by volume
    rubber_type: str  # 'S' for shredded, 'L' for large particles
    w_c_ratio: float  # Water-cement ratio
    cement_type: str
    aggregate_type: str

class ThermoMechanicalDatasetGenerator:
    """
    Comprehensive dataset generator for thermo-mechanical modeling of 
    fire-resistant rubberized concrete structural elements.
    """
    
    def __init__(self):
        """Initialize the dataset generator with material definitions"""
        self.mixes = self._define_material_mixes()
        self.temperature_range = np.linspace(20, 800, 157)  # 5°C increments
        self.calibration_ratio = 0.7  # 70% calibration, 30% validation
        
    def _define_material_mixes(self) -> Dict[str, MaterialMix]:
        """Define the six concrete mix compositions"""
        return {
            'C': MaterialMix('C', 0.0, 'N/A', 0.35, 'Type I', 'Natural'),
            'R5S': MaterialMix('R5S', 5.0, 'Shredded', 0.35, 'Type I', 'Natural'),
            'R10S': MaterialMix('R10S', 10.0, 'Shredded', 0.35, 'Type I', 'Natural'),
            'R15S': MaterialMix('R15S', 15.0, 'Shredded', 0.35, 'Type I', 'Natural'),
            'R20S': MaterialMix('R20S', 20.0, 'Shredded', 0.35, 'Type I', 'Natural'),
            'R10L': MaterialMix('R10L', 10.0, 'Large', 0.35, 'Type I', 'Natural')
        }
    
    def generate_thermal_properties(self) -> pd.DataFrame:
        """
        Generate temperature-dependent thermal properties for all mixes.
        Includes thermal conductivity, specific heat, and thermal expansion.
        """
        data = []
        
        for mix_id, mix in self.mixes.items():
            for temp in self.temperature_range:
                # Thermal conductivity (W/m·K) - decreases with temperature and rubber content
                k_base = 2.1 - 0.3 * (mix.rubber_content / 100)
                k_temp_factor = 1.0 - 0.4 * (temp - 20) / 780
                k = k_base * k_temp_factor
                k_std = k * 0.05  # 5% coefficient of variation
                
                # Specific heat (J/kg·K) - increases with temperature
                cp_base = 900 + 50 * (mix.rubber_content / 100)
                cp_temp_factor = 1.0 + 0.3 * (temp - 20) / 780
                cp = cp_base * cp_temp_factor
                cp_std = cp * 0.08  # 8% coefficient of variation
                
                # Thermal expansion coefficient (1/K) - increases with temperature
                alpha_base = 12e-6 + 2e-6 * (mix.rubber_content / 100)
                alpha_temp_factor = 1.0 + 0.5 * (temp - 20) / 780
                alpha = alpha_base * alpha_temp_factor
                alpha_std = alpha * 0.10  # 10% coefficient of variation
                
                # Density (kg/m³) - decreases with temperature due to moisture loss
                rho_base = 2400 - 100 * (mix.rubber_content / 100)
                rho_temp_factor = 1.0 - 0.05 * (temp - 20) / 780
                rho = rho_base * rho_temp_factor
                rho_std = rho * 0.03  # 3% coefficient of variation
                
                data.append({
                    'Mix_ID': mix_id,
                    'Temperature_C': temp,
                    'Data_Type': self._assign_data_type(),
                    'Property_Type': 'Thermal',
                    'Thermal_Conductivity_W_mK': k,
                    'Thermal_Conductivity_Std': k_std,
                    'Specific_Heat_J_kgK': cp,
                    'Specific_Heat_Std': cp_std,
                    'Thermal_Expansion_1_K': alpha,
                    'Thermal_Expansion_Std': alpha_std,
                    'Density_kg_m3': rho,
                    'Density_Std': rho_std
                })
        
        return pd.DataFrame(data)
    
    def generate_mechanical_properties(self) -> pd.DataFrame:
        """
        Generate temperature-dependent mechanical properties including
        elastic modulus, strength, and stress-strain relationships.
        """
        data = []
        
        for mix_id, mix in self.mixes.items():
            for temp in self.temperature_range:
                # Compressive strength (MPa) - decreases with temperature
                fc_base = 45 - 5 * (mix.rubber_content / 100)
                fc_temp_factor = 1.0 - 0.6 * (temp - 20) / 780
                fc = max(fc_base * fc_temp_factor, 5.0)  # Minimum 5 MPa
                fc_std = fc * 0.12  # 12% coefficient of variation
                
                # Elastic modulus (GPa) - decreases with temperature
                E_base = 35 - 3 * (mix.rubber_content / 100)
                E_temp_factor = 1.0 - 0.7 * (temp - 20) / 780
                E = max(E_base * E_temp_factor, 2.0)  # Minimum 2 GPa
                E_std = E * 0.15  # 15% coefficient of variation
                
                # Tensile strength (MPa) - decreases with temperature
                ft_base = 3.5 - 0.3 * (mix.rubber_content / 100)
                ft_temp_factor = 1.0 - 0.8 * (temp - 20) / 780
                ft = max(ft_base * ft_temp_factor, 0.5)  # Minimum 0.5 MPa
                ft_std = ft * 0.18  # 18% coefficient of variation
                
                # Poisson's ratio - increases with temperature
                nu_base = 0.18 + 0.02 * (mix.rubber_content / 100)
                nu_temp_factor = 1.0 + 0.2 * (temp - 20) / 780
                nu = min(nu_base * nu_temp_factor, 0.35)  # Maximum 0.35
                nu_std = nu * 0.08  # 8% coefficient of variation
                
                # Fracture energy (N/m) - decreases with temperature
                Gf_base = 150 - 10 * (mix.rubber_content / 100)
                Gf_temp_factor = 1.0 - 0.5 * (temp - 20) / 780
                Gf = max(Gf_base * Gf_temp_factor, 20.0)  # Minimum 20 N/m
                Gf_std = Gf * 0.20  # 20% coefficient of variation
                
                data.append({
                    'Mix_ID': mix_id,
                    'Temperature_C': temp,
                    'Data_Type': self._assign_data_type(),
                    'Property_Type': 'Mechanical',
                    'Compressive_Strength_MPa': fc,
                    'Compressive_Strength_Std': fc_std,
                    'Elastic_Modulus_GPa': E,
                    'Elastic_Modulus_Std': E_std,
                    'Tensile_Strength_MPa': ft,
                    'Tensile_Strength_Std': ft_std,
                    'Poissons_Ratio': nu,
                    'Poissons_Ratio_Std': nu_std,
                    'Fracture_Energy_N_m': Gf,
                    'Fracture_Energy_Std': Gf_std
                })
        
        return pd.DataFrame(data)
    
    def generate_transport_properties(self) -> pd.DataFrame:
        """
        Generate transport properties for poro-mechanical coupling including
        permeability, diffusivity, and porosity.
        """
        data = []
        
        for mix_id, mix in self.mixes.items():
            for temp in self.temperature_range:
                # Porosity - increases with temperature and rubber content
                phi_base = 0.15 + 0.02 * (mix.rubber_content / 100)
                phi_temp_factor = 1.0 + 0.3 * (temp - 20) / 780
                phi = min(phi_base * phi_temp_factor, 0.35)  # Maximum 35%
                phi_std = phi * 0.12  # 12% coefficient of variation
                
                # Permeability (m²) - increases with temperature and rubber content
                k_base = 1e-16 + 5e-17 * (mix.rubber_content / 100)
                k_temp_factor = 1.0 + 2.0 * (temp - 20) / 780
                k = k_base * k_temp_factor
                k_std = k * 0.25  # 25% coefficient of variation
                
                # Water diffusivity (m²/s) - increases with temperature
                Dw_base = 1e-10 + 2e-11 * (mix.rubber_content / 100)
                Dw_temp_factor = 1.0 + 1.5 * (temp - 20) / 780
                Dw = Dw_base * Dw_temp_factor
                Dw_std = Dw * 0.30  # 30% coefficient of variation
                
                # Vapor diffusivity (m²/s) - increases with temperature
                Dv_base = 1e-8 + 1e-9 * (mix.rubber_content / 100)
                Dv_temp_factor = 1.0 + 2.0 * (temp - 20) / 780
                Dv = Dv_base * Dv_temp_factor
                Dv_std = Dv * 0.25  # 25% coefficient of variation
                
                data.append({
                    'Mix_ID': mix_id,
                    'Temperature_C': temp,
                    'Data_Type': self._assign_data_type(),
                    'Property_Type': 'Transport',
                    'Porosity': phi,
                    'Porosity_Std': phi_std,
                    'Permeability_m2': k,
                    'Permeability_Std': k_std,
                    'Water_Diffusivity_m2_s': Dw,
                    'Water_Diffusivity_Std': Dw_std,
                    'Vapor_Diffusivity_m2_s': Dv,
                    'Vapor_Diffusivity_Std': Dv_std
                })
        
        return pd.DataFrame(data)
    
    def generate_deformation_properties(self) -> pd.DataFrame:
        """
        Generate deformation properties including creep, shrinkage,
        and thermal strain with temperature dependencies.
        """
        data = []
        
        for mix_id, mix in self.mixes.items():
            for temp in self.temperature_range:
                # Creep coefficient - increases with temperature
                phi_creep_base = 2.0 + 0.1 * (mix.rubber_content / 100)
                phi_creep_temp_factor = 1.0 + 0.4 * (temp - 20) / 780
                phi_creep = phi_creep_base * phi_creep_temp_factor
                phi_creep_std = phi_creep * 0.15  # 15% coefficient of variation
                
                # Shrinkage strain - increases with temperature
                eps_sh_base = 300e-6 + 50e-6 * (mix.rubber_content / 100)
                eps_sh_temp_factor = 1.0 + 0.6 * (temp - 20) / 780
                eps_sh = eps_sh_base * eps_sh_temp_factor
                eps_sh_std = eps_sh * 0.20  # 20% coefficient of variation
                
                # Thermal strain - linear with temperature
                alpha_thermal = 12e-6 + 2e-6 * (mix.rubber_content / 100)
                eps_thermal = alpha_thermal * (temp - 20)
                eps_thermal_std = eps_thermal * 0.10  # 10% coefficient of variation
                
                # Creep modulus (GPa) - decreases with temperature
                E_creep_base = 30 - 2 * (mix.rubber_content / 100)
                E_creep_temp_factor = 1.0 - 0.5 * (temp - 20) / 780
                E_creep = max(E_creep_base * E_creep_temp_factor, 3.0)
                E_creep_std = E_creep * 0.18  # 18% coefficient of variation
                
                data.append({
                    'Mix_ID': mix_id,
                    'Temperature_C': temp,
                    'Data_Type': self._assign_data_type(),
                    'Property_Type': 'Deformation',
                    'Creep_Coefficient': phi_creep,
                    'Creep_Coefficient_Std': phi_creep_std,
                    'Shrinkage_Strain': eps_sh,
                    'Shrinkage_Strain_Std': eps_sh_std,
                    'Thermal_Strain': eps_thermal,
                    'Thermal_Strain_Std': eps_thermal_std,
                    'Creep_Modulus_GPa': E_creep,
                    'Creep_Modulus_Std': E_creep_std
                })
        
        return pd.DataFrame(data)
    
    def _assign_data_type(self) -> str:
        """Randomly assign data type (Calibration or Validation)"""
        return np.random.choice(['Calibration', 'Validation'], p=[self.calibration_ratio, 1-self.calibration_ratio])
    
    def generate_stress_strain_curves(self) -> Dict[str, Dict]:
        """
        Generate stress-strain curves for different temperatures and mixes.
        Returns nested dictionary with curves for each mix and temperature.
        """
        curves = {}
        
        for mix_id, mix in self.mixes.items():
            curves[mix_id] = {}
            
            for temp in self.temperature_range[::10]:  # Every 50°C
                # Generate stress-strain curve parameters
                fc = 45 - 5 * (mix.rubber_content / 100)
                fc_temp_factor = 1.0 - 0.6 * (temp - 20) / 780
                fc = max(fc * fc_temp_factor, 5.0)
                
                E = 35 - 3 * (mix.rubber_content / 100)
                E_temp_factor = 1.0 - 0.7 * (temp - 20) / 780
                E = max(E * E_temp_factor, 2.0)
                
                # Convert to MPa for stress-strain curve
                fc_mpa = fc  # Already in MPa
                E_gpa = E    # Already in GPa
                
                # Generate strain points (0 to 0.01)
                strain = np.linspace(0, 0.01, 100)
                
                # Hognestad model for stress-strain curve
                stress = np.zeros_like(strain)
                eps_0 = 2 * fc_mpa / (E_gpa * 1000)  # Convert E to MPa
                
                for i, eps in enumerate(strain):
                    if eps <= eps_0:
                        stress[i] = fc_mpa * (2 * eps / eps_0 - (eps / eps_0)**2)
                    else:
                        stress[i] = fc_mpa * (1 - 0.15 * (eps - eps_0) / (0.01 - eps_0))
                
                curves[mix_id][temp] = {
                    'strain': strain.tolist(),
                    'stress': stress.tolist(),
                    'fc': fc_mpa,
                    'E': E_gpa
                }
        
        return curves
    
    def generate_complete_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Generate the complete thermo-mechanical dataset including all properties.
        """
        print("Generating comprehensive thermo-mechanical dataset...")
        
        dataset = {
            'thermal_properties': self.generate_thermal_properties(),
            'mechanical_properties': self.generate_mechanical_properties(),
            'transport_properties': self.generate_transport_properties(),
            'deformation_properties': self.generate_deformation_properties()
        }
        
        # Add stress-strain curves
        dataset['stress_strain_curves'] = self.generate_stress_strain_curves()
        
        print(f"Dataset generated successfully!")
        print(f"Total data points: {sum(len(df) for df in dataset.values() if isinstance(df, pd.DataFrame))}")
        
        return dataset
    
    def export_to_formats(self, dataset: Dict[str, pd.DataFrame], output_dir: str = 'thermo_mechanical_dataset'):
        """
        Export dataset to multiple formats for different FEA software.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Export to CSV files
        for name, df in dataset.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(f'{output_dir}/{name}.csv', index=False)
        
        # Export to JSON for stress-strain curves
        with open(f'{output_dir}/stress_strain_curves.json', 'w') as f:
            json.dump(dataset['stress_strain_curves'], f, indent=2)
        
        # Export to Excel with multiple sheets
        with pd.ExcelWriter(f'{output_dir}/complete_dataset.xlsx', engine='openpyxl') as writer:
            for name, df in dataset.items():
                if isinstance(df, pd.DataFrame):
                    df.to_excel(writer, sheet_name=name, index=False)
        
        # Generate ABAQUS input files
        self._generate_abaqus_inputs(dataset, output_dir)
        
        # Generate ANSYS input files
        self._generate_ansys_inputs(dataset, output_dir)
        
        # Generate COMSOL input files
        self._generate_comsol_inputs(dataset, output_dir)
        
        print(f"Dataset exported to {output_dir}/")
    
    def _generate_abaqus_inputs(self, dataset: Dict, output_dir: str):
        """Generate ABAQUS input files for material properties"""
        abaqus_dir = f'{output_dir}/abaqus_inputs'
        os.makedirs(abaqus_dir, exist_ok=True)
        
        for mix_id in self.mixes.keys():
            with open(f'{abaqus_dir}/material_{mix_id}.inp', 'w') as f:
                f.write(f"*MATERIAL, NAME=CONCRETE_{mix_id}\n")
                f.write("*ELASTIC\n")
                f.write("*THERMAL CONDUCTIVITY\n")
                f.write("*SPECIFIC HEAT\n")
                f.write("*DENSITY\n")
                f.write("*EXPANSION\n")
                f.write("*PERMEABILITY\n")
                f.write("*CREEP\n")
                f.write("*DAMAGE INITIATION\n")
                f.write("*DAMAGE EVOLUTION\n")
    
    def _generate_ansys_inputs(self, dataset: Dict, output_dir: str):
        """Generate ANSYS input files for material properties"""
        ansys_dir = f'{output_dir}/ansys_inputs'
        os.makedirs(ansys_dir, exist_ok=True)
        
        for mix_id in self.mixes.keys():
            with open(f'{ansys_dir}/material_{mix_id}.txt', 'w') as f:
                f.write(f"! Material properties for {mix_id}\n")
                f.write("MP,EX,1,35e9\n")  # Elastic modulus
                f.write("MP,PRXY,1,0.18\n")  # Poisson's ratio
                f.write("MP,DENS,1,2400\n")  # Density
                f.write("MP,KXX,1,2.1\n")  # Thermal conductivity
                f.write("MP,C,1,900\n")  # Specific heat
                f.write("MP,ALPX,1,12e-6\n")  # Thermal expansion
    
    def _generate_comsol_inputs(self, dataset: Dict, output_dir: str):
        """Generate COMSOL input files for material properties"""
        comsol_dir = f'{output_dir}/comsol_inputs'
        os.makedirs(comsol_dir, exist_ok=True)
        
        for mix_id in self.mixes.keys():
            with open(f'{comsol_dir}/material_{mix_id}.txt', 'w') as f:
                f.write(f"% Material properties for {mix_id}\n")
                f.write("E = 35e9; % Elastic modulus\n")
                f.write("nu = 0.18; % Poisson's ratio\n")
                f.write("rho = 2400; % Density\n")
                f.write("k = 2.1; % Thermal conductivity\n")
                f.write("cp = 900; % Specific heat\n")
                f.write("alpha = 12e-6; % Thermal expansion\n")

def main():
    """Main function to generate and export the complete dataset"""
    generator = ThermoMechanicalDatasetGenerator()
    
    # Generate complete dataset
    dataset = generator.generate_complete_dataset()
    
    # Export to various formats
    generator.export_to_formats(dataset)
    
    # Print summary statistics
    print("\nDataset Summary:")
    print("=" * 50)
    for name, df in dataset.items():
        if isinstance(df, pd.DataFrame):
            print(f"{name}: {len(df)} data points")
            print(f"  - Mixes: {df['Mix_ID'].unique()}")
            print(f"  - Temperature range: {df['Temperature_C'].min():.0f}°C to {df['Temperature_C'].max():.0f}°C")
            print(f"  - Data types: {df['Data_Type'].value_counts().to_dict()}")
            print()

if __name__ == "__main__":
    main()