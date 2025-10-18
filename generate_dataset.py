#!/usr/bin/env python3
"""
Comprehensive Dataset Generator for Fire-Resistant Rubberized Concrete
Thermo-Mechanical Modeling and Validation

This script generates realistic experimental datasets for:
1. Model Input Data (Material Properties)
2. Model Validation Data (Experimental Measurements)

Author: AI Assistant
Date: 2025-10-18
Topic: Development and Validation of a Thermo-Mechanical Model for 
       Fire-Resistant Structural Elements Utilizing High-Performance 
       Rubberized Concrete
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
from scipy import interpolate
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

class RubberizedConcreteDataGenerator:
    def __init__(self, rubber_content_range=[0, 20], seed=42):
        """
        Initialize the dataset generator
        
        Parameters:
        rubber_content_range: [min, max] rubber content by volume (%)
        seed: random seed for reproducibility
        """
        np.random.seed(seed)
        self.rubber_content_range = rubber_content_range
        self.temperature_range = np.linspace(20, 1000, 100)  # 20°C to 1000°C
        
        # Define rubber content levels for testing
        self.rubber_contents = [0, 5, 10, 15, 20]  # % by volume
        
        # Create output directory
        self.output_dir = "rubberized_concrete_dataset"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize metadata
        self.metadata = {
            "dataset_name": "Fire-Resistant Rubberized Concrete Dataset",
            "generation_date": datetime.now().isoformat(),
            "temperature_range": {"min": 20, "max": 1000, "unit": "°C"},
            "rubber_content_range": {"min": rubber_content_range[0], "max": rubber_content_range[1], "unit": "% by volume"},
            "description": "Comprehensive dataset for thermo-mechanical modeling of fire-resistant rubberized concrete",
            "data_types": []
        }

    def generate_thermal_properties(self):
        """Generate temperature-dependent thermal properties"""
        print("Generating thermal properties dataset...")
        
        thermal_data = {}
        
        for rubber_content in self.rubber_contents:
            # Thermal Conductivity (W/m·K)
            # Base concrete: ~1.5-2.0 W/m·K at room temp, decreases with temperature
            # Rubber reduces thermal conductivity
            k_base = 1.8 - 0.3 * (rubber_content / 20)  # Reduction due to rubber
            thermal_conductivity = k_base * (1 - 0.0008 * (self.temperature_range - 20)) * \
                                 np.exp(-0.0002 * (self.temperature_range - 20))
            thermal_conductivity = np.maximum(thermal_conductivity, 0.1)  # Minimum value
            
            # Add realistic noise
            thermal_conductivity += np.random.normal(0, 0.02, len(thermal_conductivity))
            
            # Specific Heat (J/kg·K)
            # Increases with temperature, rubber slightly increases specific heat
            cp_base = 900 + 50 * (rubber_content / 20)
            specific_heat = cp_base + 0.5 * (self.temperature_range - 20) + \
                          100 * np.exp(-((self.temperature_range - 100) / 50)**2)  # Peak around 100°C (moisture)
            specific_heat += np.random.normal(0, 10, len(specific_heat))
            
            # Density (kg/m³)
            # Decreases with rubber content and temperature
            density_base = 2400 - 200 * (rubber_content / 20)  # Rubber is lighter
            density = density_base * (1 - 0.0001 * (self.temperature_range - 20))  # Thermal expansion
            density += np.random.normal(0, 5, len(density))
            
            thermal_data[f"rubber_{rubber_content}pct"] = {
                "rubber_content": rubber_content,
                "temperature": self.temperature_range.tolist(),
                "thermal_conductivity": thermal_conductivity.tolist(),
                "specific_heat": specific_heat.tolist(),
                "density": density.tolist()
            }
        
        # Save thermal properties
        with open(f"{self.output_dir}/thermal_properties.json", "w") as f:
            json.dump(thermal_data, f, indent=2)
        
        # Create CSV format
        thermal_df_list = []
        for rubber_content in self.rubber_contents:
            key = f"rubber_{rubber_content}pct"
            df_temp = pd.DataFrame({
                'rubber_content_pct': rubber_content,
                'temperature_C': thermal_data[key]['temperature'],
                'thermal_conductivity_W_m_K': thermal_data[key]['thermal_conductivity'],
                'specific_heat_J_kg_K': thermal_data[key]['specific_heat'],
                'density_kg_m3': thermal_data[key]['density']
            })
            thermal_df_list.append(df_temp)
        
        thermal_df = pd.concat(thermal_df_list, ignore_index=True)
        thermal_df.to_csv(f"{self.output_dir}/thermal_properties.csv", index=False)
        
        self.metadata["data_types"].append("thermal_properties")
        return thermal_data

    def generate_mechanical_properties(self):
        """Generate temperature-dependent mechanical properties"""
        print("Generating mechanical properties dataset...")
        
        mechanical_data = {}
        
        for rubber_content in self.rubber_contents:
            # Compressive Strength (MPa)
            # Decreases with temperature and rubber content
            fc_base = 35 - 8 * (rubber_content / 20)  # Rubber reduces strength
            compressive_strength = fc_base * np.exp(-0.001 * (self.temperature_range - 20)) * \
                                 (1 - 0.8 * np.maximum(0, (self.temperature_range - 300) / 700))
            compressive_strength = np.maximum(compressive_strength, 1.0)
            compressive_strength += np.random.normal(0, 0.5, len(compressive_strength))
            
            # Tensile Strength (MPa)
            # Typically 8-12% of compressive strength, rubber can improve ductility
            tensile_factor = 0.10 + 0.02 * (rubber_content / 20)
            tensile_strength = compressive_strength * tensile_factor
            tensile_strength += np.random.normal(0, 0.1, len(tensile_strength))
            
            # Elastic Modulus (GPa)
            # Decreases with temperature and rubber content
            E_base = 30 - 8 * (rubber_content / 20)
            elastic_modulus = E_base * np.exp(-0.0015 * (self.temperature_range - 20)) * \
                            (1 - 0.9 * np.maximum(0, (self.temperature_range - 400) / 600))
            elastic_modulus = np.maximum(elastic_modulus, 0.5)
            elastic_modulus += np.random.normal(0, 0.3, len(elastic_modulus))
            
            # Poisson's Ratio
            # Slightly increases with temperature and rubber content
            poisson_base = 0.18 + 0.05 * (rubber_content / 20)
            poisson_ratio = poisson_base + 0.00005 * (self.temperature_range - 20)
            poisson_ratio = np.clip(poisson_ratio, 0.15, 0.35)
            poisson_ratio += np.random.normal(0, 0.005, len(poisson_ratio))
            
            mechanical_data[f"rubber_{rubber_content}pct"] = {
                "rubber_content": rubber_content,
                "temperature": self.temperature_range.tolist(),
                "compressive_strength": compressive_strength.tolist(),
                "tensile_strength": tensile_strength.tolist(),
                "elastic_modulus": elastic_modulus.tolist(),
                "poisson_ratio": poisson_ratio.tolist()
            }
        
        # Save mechanical properties
        with open(f"{self.output_dir}/mechanical_properties.json", "w") as f:
            json.dump(mechanical_data, f, indent=2)
        
        # Create CSV format
        mechanical_df_list = []
        for rubber_content in self.rubber_contents:
            key = f"rubber_{rubber_content}pct"
            df_temp = pd.DataFrame({
                'rubber_content_pct': rubber_content,
                'temperature_C': mechanical_data[key]['temperature'],
                'compressive_strength_MPa': mechanical_data[key]['compressive_strength'],
                'tensile_strength_MPa': mechanical_data[key]['tensile_strength'],
                'elastic_modulus_GPa': mechanical_data[key]['elastic_modulus'],
                'poisson_ratio': mechanical_data[key]['poisson_ratio']
            })
            mechanical_df_list.append(df_temp)
        
        mechanical_df = pd.concat(mechanical_df_list, ignore_index=True)
        mechanical_df.to_csv(f"{self.output_dir}/mechanical_properties.csv", index=False)
        
        self.metadata["data_types"].append("mechanical_properties")
        return mechanical_data

    def generate_deformation_properties(self):
        """Generate deformation properties including thermal expansion and transient strain"""
        print("Generating deformation properties dataset...")
        
        deformation_data = {}
        
        for rubber_content in self.rubber_contents:
            # Coefficient of Thermal Expansion (1/K)
            # Rubber typically has higher CTE than concrete
            alpha_concrete = 10e-6  # 1/K
            alpha_rubber = 150e-6   # 1/K
            alpha_composite = alpha_concrete + (alpha_rubber - alpha_concrete) * (rubber_content / 100)
            
            # CTE varies slightly with temperature
            cte = alpha_composite * (1 + 0.0001 * (self.temperature_range - 20))
            cte += np.random.normal(0, 0.5e-6, len(cte))
            
            # Transient Thermal Strain (dimensionless)
            # This is the additional strain due to temperature rate effects
            # Typically occurs during first heating
            transient_strain = np.zeros_like(self.temperature_range)
            
            # Transient strain peaks around 100-200°C due to moisture migration
            for i, T in enumerate(self.temperature_range):
                if T > 50:
                    transient_strain[i] = 0.0005 * np.exp(-((T - 150) / 80)**2) * \
                                        (1 - 0.3 * rubber_content / 20)  # Rubber reduces transient effects
            
            transient_strain += np.random.normal(0, 0.00005, len(transient_strain))
            transient_strain = np.maximum(transient_strain, 0)
            
            # Creep Strain (time-dependent, at constant stress and temperature)
            # Generate creep data for different stress levels and temperatures
            stress_levels = [0.2, 0.4, 0.6]  # Fraction of compressive strength
            time_points = np.logspace(0, 4, 50)  # 1 to 10000 hours
            
            creep_data = {}
            for stress_level in stress_levels:
                creep_data[f"stress_{stress_level}"] = {}
                for temp_idx, temp in enumerate([100, 200, 300, 400, 500]):
                    if temp <= 1000:
                        # Creep increases with temperature and stress
                        creep_factor = stress_level * (1 + 0.002 * temp) * (1 + 0.1 * rubber_content / 20)
                        creep_strain_time = creep_factor * 0.001 * (time_points**0.3) / 1000
                        creep_strain_time += np.random.normal(0, creep_strain_time * 0.05)
                        
                        creep_data[f"stress_{stress_level}"][f"temp_{temp}C"] = {
                            "time_hours": time_points.tolist(),
                            "creep_strain": creep_strain_time.tolist()
                        }
            
            deformation_data[f"rubber_{rubber_content}pct"] = {
                "rubber_content": rubber_content,
                "temperature": self.temperature_range.tolist(),
                "thermal_expansion_coefficient": cte.tolist(),
                "transient_thermal_strain": transient_strain.tolist(),
                "creep_data": creep_data
            }
        
        # Save deformation properties
        with open(f"{self.output_dir}/deformation_properties.json", "w") as f:
            json.dump(deformation_data, f, indent=2)
        
        # Create CSV for thermal expansion and transient strain
        deformation_df_list = []
        for rubber_content in self.rubber_contents:
            key = f"rubber_{rubber_content}pct"
            df_temp = pd.DataFrame({
                'rubber_content_pct': rubber_content,
                'temperature_C': deformation_data[key]['temperature'],
                'thermal_expansion_coeff_per_K': deformation_data[key]['thermal_expansion_coefficient'],
                'transient_thermal_strain': deformation_data[key]['transient_thermal_strain']
            })
            deformation_df_list.append(df_temp)
        
        deformation_df = pd.concat(deformation_df_list, ignore_index=True)
        deformation_df.to_csv(f"{self.output_dir}/deformation_properties.csv", index=False)
        
        self.metadata["data_types"].append("deformation_properties")
        return deformation_data

    def generate_poromechanical_properties(self):
        """Generate poro-mechanical properties (permeability and porosity)"""
        print("Generating poro-mechanical properties dataset...")
        
        poro_data = {}
        
        for rubber_content in self.rubber_contents:
            # Initial Porosity
            # Rubber particles can increase porosity
            porosity_base = 0.12 + 0.03 * (rubber_content / 20)
            
            # Porosity evolution with temperature
            # Increases due to thermal damage and moisture loss
            porosity_temp = np.zeros_like(self.temperature_range)
            for i, T in enumerate(self.temperature_range):
                if T <= 100:
                    porosity_temp[i] = porosity_base
                elif T <= 300:
                    # Gradual increase due to moisture loss
                    porosity_temp[i] = porosity_base + 0.02 * (T - 100) / 200
                else:
                    # Accelerated increase due to thermal damage
                    porosity_temp[i] = porosity_base + 0.02 + 0.05 * (T - 300) / 700
            
            porosity_temp += np.random.normal(0, 0.005, len(porosity_temp))
            porosity_temp = np.clip(porosity_temp, 0.05, 0.4)
            
            # Permeability (m²)
            # Follows Kozeny-Carman relation with porosity
            # k = k0 * (φ/φ0)³ * ((1-φ0)/(1-φ))²
            k0 = 1e-18  # Base permeability (m²)
            phi0 = porosity_base
            
            permeability = k0 * (porosity_temp / phi0)**3 * \
                          ((1 - phi0) / (1 - porosity_temp))**2
            
            # Add temperature-dependent effects
            # Permeability increases more at high temperatures due to microcracking
            temp_factor = 1 + 0.001 * np.maximum(0, self.temperature_range - 200)
            permeability *= temp_factor
            
            permeability += np.random.normal(0, permeability * 0.1)
            permeability = np.maximum(permeability, 1e-20)
            
            # Damage parameter (0 = no damage, 1 = complete damage)
            damage = np.zeros_like(self.temperature_range)
            for i, T in enumerate(self.temperature_range):
                if T > 300:
                    damage[i] = min(0.8, (T - 300) / 700)
            
            # Rubber content affects damage evolution
            damage *= (1 - 0.2 * rubber_content / 20)  # Rubber provides some protection
            damage += np.random.normal(0, 0.02, len(damage))
            damage = np.clip(damage, 0, 1)
            
            poro_data[f"rubber_{rubber_content}pct"] = {
                "rubber_content": rubber_content,
                "temperature": self.temperature_range.tolist(),
                "porosity": porosity_temp.tolist(),
                "permeability": permeability.tolist(),
                "damage_parameter": damage.tolist()
            }
        
        # Save poro-mechanical properties
        with open(f"{self.output_dir}/poromechanical_properties.json", "w") as f:
            json.dump(poro_data, f, indent=2)
        
        # Create CSV format
        poro_df_list = []
        for rubber_content in self.rubber_contents:
            key = f"rubber_{rubber_content}pct"
            df_temp = pd.DataFrame({
                'rubber_content_pct': rubber_content,
                'temperature_C': poro_data[key]['temperature'],
                'porosity': poro_data[key]['porosity'],
                'permeability_m2': poro_data[key]['permeability'],
                'damage_parameter': poro_data[key]['damage_parameter']
            })
            poro_df_list.append(df_temp)
        
        poro_df = pd.concat(poro_df_list, ignore_index=True)
        poro_df.to_csv(f"{self.output_dir}/poromechanical_properties.csv", index=False)
        
        self.metadata["data_types"].append("poromechanical_properties")
        return poro_data

if __name__ == "__main__":
    # Initialize generator
    generator = RubberizedConcreteDataGenerator()
    
    # Generate all material property datasets
    print("=== GENERATING MATERIAL PROPERTIES DATASETS ===")
    thermal_data = generator.generate_thermal_properties()
    mechanical_data = generator.generate_mechanical_properties()
    deformation_data = generator.generate_deformation_properties()
    poro_data = generator.generate_poromechanical_properties()
    
    print(f"\nMaterial properties datasets generated successfully!")
    print(f"Output directory: {generator.output_dir}")