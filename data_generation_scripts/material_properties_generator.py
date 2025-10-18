#!/usr/bin/env python3
"""
Material Properties Data Generator for Fire-Resistant Rubberized Concrete
Generates realistic synthetic data for constituent materials
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
import json
from datetime import datetime
import os

class MaterialPropertiesGenerator:
    def __init__(self, seed=42):
        """Initialize the generator with random seed for reproducibility"""
        np.random.seed(seed)
        self.data = {}
        
    def generate_cement_properties(self, n_samples=100):
        """Generate cement properties data"""
        print("Generating cement properties...")
        
        # Chemical composition (XRF data) - based on typical OPC CEM I 42.5R
        cement_data = {
            'SiO2': np.random.normal(20.5, 1.2, n_samples),
            'Al2O3': np.random.normal(5.2, 0.8, n_samples),
            'Fe2O3': np.random.normal(3.1, 0.5, n_samples),
            'CaO': np.random.normal(64.8, 2.1, n_samples),
            'MgO': np.random.normal(1.8, 0.3, n_samples),
            'SO3': np.random.normal(2.9, 0.4, n_samples),
            'K2O': np.random.normal(0.8, 0.2, n_samples),
            'Na2O': np.random.normal(0.3, 0.1, n_samples),
            'TiO2': np.random.normal(0.3, 0.05, n_samples),
            'P2O5': np.random.normal(0.1, 0.02, n_samples),
            'MnO': np.random.normal(0.05, 0.01, n_samples),
            'SrO': np.random.normal(0.02, 0.005, n_samples),
            'LOI': np.random.normal(1.2, 0.3, n_samples),
            'Free_CaO': np.random.normal(0.8, 0.2, n_samples)
        }
        
        # Bogue composition
        bogue_data = {
            'C3S': np.random.normal(58.5, 3.2, n_samples),
            'C2S': np.random.normal(18.2, 2.1, n_samples),
            'C3A': np.random.normal(8.1, 1.5, n_samples),
            'C4AF': np.random.normal(9.4, 1.2, n_samples)
        }
        
        # Physical properties
        physical_data = {
            'Specific_Gravity': np.random.normal(3.15, 0.05, n_samples),
            'Blaine_Fineness': np.random.normal(380, 25, n_samples),
            'Initial_Setting_Time': np.random.normal(165, 15, n_samples),
            'Final_Setting_Time': np.random.normal(245, 20, n_samples),
            'Soundness_Expansion': np.random.normal(1.2, 0.3, n_samples)
        }
        
        # Thermal properties (temperature-dependent)
        temperatures = np.linspace(20, 800, 100)
        thermal_data = {
            'Temperature': temperatures,
            'Thermal_Conductivity': self._generate_thermal_conductivity(temperatures),
            'Specific_Heat': self._generate_specific_heat(temperatures),
            'Thermal_Expansion': self._generate_thermal_expansion(temperatures)
        }
        
        self.data['cement'] = {
            'chemical_composition': cement_data,
            'bogue_composition': bogue_data,
            'physical_properties': physical_data,
            'thermal_properties': thermal_data
        }
        
        return self.data['cement']
    
    def generate_aggregate_properties(self, n_samples=100):
        """Generate aggregate properties data"""
        print("Generating aggregate properties...")
        
        # Coarse aggregates
        coarse_data = {
            'Specific_Gravity_SSD': np.random.normal(2.68, 0.05, n_samples),
            'Specific_Gravity_Apparent': np.random.normal(2.72, 0.04, n_samples),
            'Water_Absorption': np.random.normal(0.8, 0.2, n_samples),
            'Bulk_Density_Loose': np.random.normal(1450, 50, n_samples),
            'Bulk_Density_Compacted': np.random.normal(1650, 45, n_samples),
            'Angularity_Number': np.random.normal(8.5, 1.2, n_samples),
            'Flakiness_Index': np.random.normal(12.3, 2.1, n_samples)
        }
        
        # Fine aggregates
        fine_data = {
            'Specific_Gravity_SSD': np.random.normal(2.65, 0.04, n_samples),
            'Specific_Gravity_Apparent': np.random.normal(2.69, 0.03, n_samples),
            'Water_Absorption': np.random.normal(1.2, 0.3, n_samples),
            'Bulk_Density_Loose': np.random.normal(1550, 40, n_samples),
            'Bulk_Density_Compacted': np.random.normal(1750, 35, n_samples),
            'Fineness_Modulus': np.random.normal(2.8, 0.3, n_samples)
        }
        
        # Sieve analysis data
        sieve_sizes = [0.075, 0.15, 0.3, 0.6, 1.18, 2.36, 4.75, 9.5, 12.5, 19, 25, 37.5]
        coarse_sieve = self._generate_sieve_analysis(sieve_sizes, 'coarse')
        fine_sieve = self._generate_sieve_analysis(sieve_sizes, 'fine')
        
        # Thermal properties
        temperatures = np.linspace(20, 800, 100)
        thermal_data = {
            'Temperature': temperatures,
            'Thermal_Conductivity': self._generate_thermal_conductivity(temperatures, material='aggregate'),
            'Specific_Heat': self._generate_specific_heat(temperatures, material='aggregate'),
            'Thermal_Expansion': self._generate_thermal_expansion(temperatures, material='aggregate')
        }
        
        self.data['aggregates'] = {
            'coarse': coarse_data,
            'fine': fine_data,
            'coarse_sieve_analysis': coarse_sieve,
            'fine_sieve_analysis': fine_sieve,
            'thermal_properties': thermal_data
        }
        
        return self.data['aggregates']
    
    def generate_rubber_properties(self, n_samples=100):
        """Generate crumb rubber properties data"""
        print("Generating rubber properties...")
        
        # Physical properties
        physical_data = {
            'Specific_Gravity_Apparent': np.random.normal(1.15, 0.05, n_samples),
            'Specific_Gravity_Bulk': np.random.normal(0.65, 0.08, n_samples),
            'Water_Absorption': np.random.normal(2.5, 0.8, n_samples),
            'Bulk_Density_Loose': np.random.normal(450, 50, n_samples),
            'Bulk_Density_Compacted': np.random.normal(520, 45, n_samples),
            'Shore_A_Hardness': np.random.normal(65, 8, n_samples),
            'Mohs_Hardness': np.random.normal(2.5, 0.3, n_samples)
        }
        
        # Particle size distribution
        size_ranges = ['0.075-1', '1-2', '2-4', '4-8', '8-12']
        size_distribution = {}
        for size in size_ranges:
            size_distribution[f'Size_{size}_mm'] = np.random.normal(20, 5, n_samples)
        
        # Chemical composition
        chemical_data = {
            'Carbon_Content': np.random.normal(78.5, 2.1, n_samples),
            'Hydrogen_Content': np.random.normal(6.8, 0.5, n_samples),
            'Nitrogen_Content': np.random.normal(0.3, 0.1, n_samples),
            'Sulfur_Content': np.random.normal(1.8, 0.4, n_samples),
            'Oxygen_Content': np.random.normal(12.6, 1.8, n_samples),
            'Ash_Content': np.random.normal(4.2, 0.8, n_samples),
            'Volatile_Matter': np.random.normal(15.3, 2.1, n_samples)
        }
        
        # Thermal analysis (TGA)
        temperatures = np.linspace(25, 800, 100)
        tga_data = {
            'Temperature': temperatures,
            'Weight_Loss': self._generate_tga_curve(temperatures),
            'Weight_Loss_Rate': self._generate_tga_derivative(temperatures)
        }
        
        # Fire resistance properties
        fire_data = {
            'Ignition_Temperature': np.random.normal(320, 25, n_samples),
            'Heat_Release_Rate_Peak': np.random.normal(450, 80, n_samples),
            'Smoke_Production_Rate': np.random.normal(0.8, 0.2, n_samples),
            'Char_Yield': np.random.normal(15.2, 3.1, n_samples)
        }
        
        # Morphological properties
        morphological_data = {
            'Surface_Area_BET': np.random.normal(2.8, 0.5, n_samples),
            'Aspect_Ratio': np.random.normal(1.8, 0.3, n_samples),
            'Sphericity': np.random.normal(0.75, 0.08, n_samples),
            'Surface_Roughness': np.random.normal(2.1, 0.4, n_samples)
        }
        
        self.data['rubber'] = {
            'physical_properties': physical_data,
            'size_distribution': size_distribution,
            'chemical_composition': chemical_data,
            'thermal_analysis': tga_data,
            'fire_resistance': fire_data,
            'morphological': morphological_data
        }
        
        return self.data['rubber']
    
    def generate_water_properties(self, n_samples=100):
        """Generate water quality data"""
        print("Generating water properties...")
        
        water_data = {
            'pH': np.random.normal(7.2, 0.3, n_samples),
            'TDS': np.random.normal(180, 25, n_samples),
            'Chloride_Content': np.random.normal(25, 8, n_samples),
            'Sulfate_Content': np.random.normal(45, 12, n_samples),
            'Organic_Matter': np.random.normal(2.1, 0.8, n_samples),
            'Temperature': np.random.normal(20, 1, n_samples)
        }
        
        self.data['water'] = water_data
        return water_data
    
    def generate_admixture_properties(self, n_samples=100):
        """Generate chemical admixture data"""
        print("Generating admixture properties...")
        
        admixture_data = {
            'Superplasticizer_Dosage': np.random.normal(1.2, 0.3, n_samples),
            'Superplasticizer_Solid_Content': np.random.normal(35, 3, n_samples),
            'Superplasticizer_pH': np.random.normal(6.8, 0.4, n_samples),
            'Air_Entraining_Dosage': np.random.normal(0.05, 0.01, n_samples),
            'Air_Content_Achieved': np.random.normal(4.5, 0.8, n_samples),
            'PP_Fiber_Length': np.random.normal(12, 2, n_samples),
            'PP_Fiber_Diameter': np.random.normal(0.018, 0.003, n_samples),
            'PP_Fiber_Dosage': np.random.normal(0.9, 0.2, n_samples),
            'Steel_Fiber_Length': np.random.normal(30, 5, n_samples),
            'Steel_Fiber_Diameter': np.random.normal(0.5, 0.05, n_samples),
            'Steel_Fiber_Dosage': np.random.normal(1.5, 0.3, n_samples)
        }
        
        self.data['admixtures'] = admixture_data
        return admixture_data
    
    def _generate_thermal_conductivity(self, temperatures, material='cement'):
        """Generate temperature-dependent thermal conductivity"""
        if material == 'cement':
            # Typical cement thermal conductivity decreases with temperature
            base_conductivity = 1.2
            temp_factor = 1 - 0.3 * (temperatures - 20) / 780
        else:  # aggregate
            # Aggregate thermal conductivity is more stable
            base_conductivity = 2.8
            temp_factor = 1 - 0.15 * (temperatures - 20) / 780
        
        return base_conductivity * temp_factor + np.random.normal(0, 0.05, len(temperatures))
    
    def _generate_specific_heat(self, temperatures, material='cement'):
        """Generate temperature-dependent specific heat"""
        if material == 'cement':
            # Cement specific heat increases with temperature
            base_heat = 0.9
            temp_factor = 1 + 0.4 * (temperatures - 20) / 780
        else:  # aggregate
            # Aggregate specific heat is more stable
            base_heat = 0.8
            temp_factor = 1 + 0.2 * (temperatures - 20) / 780
        
        return base_heat * temp_factor + np.random.normal(0, 0.02, len(temperatures))
    
    def _generate_thermal_expansion(self, temperatures, material='cement'):
        """Generate temperature-dependent thermal expansion coefficient"""
        if material == 'cement':
            # Cement thermal expansion increases with temperature
            base_expansion = 10e-6
            temp_factor = 1 + 0.5 * (temperatures - 20) / 780
        else:  # aggregate
            # Aggregate thermal expansion is more linear
            base_expansion = 8e-6
            temp_factor = 1 + 0.3 * (temperatures - 20) / 780
        
        return base_expansion * temp_factor + np.random.normal(0, 0.5e-6, len(temperatures))
    
    def _generate_sieve_analysis(self, sieve_sizes, aggregate_type):
        """Generate sieve analysis data"""
        if aggregate_type == 'coarse':
            # Coarse aggregate grading
            passing_percentages = [0, 0, 0, 0, 0, 0, 5, 25, 45, 75, 95, 100]
        else:  # fine
            # Fine aggregate grading
            passing_percentages = [0, 5, 15, 35, 55, 75, 95, 100, 100, 100, 100, 100]
        
        # Add some random variation
        passing_percentages = [max(0, min(100, p + np.random.normal(0, 3))) for p in passing_percentages]
        
        return dict(zip(sieve_sizes, passing_percentages))
    
    def _generate_tga_curve(self, temperatures):
        """Generate TGA weight loss curve for rubber"""
        # Typical rubber decomposition pattern
        weight_loss = np.zeros_like(temperatures)
        
        # Initial weight loss (moisture, volatiles)
        weight_loss[temperatures < 150] = 2 * (temperatures[temperatures < 150] - 25) / 125
        
        # Main decomposition (300-500°C)
        mask = (temperatures >= 300) & (temperatures < 500)
        weight_loss[mask] = 2 + 60 * (temperatures[mask] - 300) / 200
        
        # Final decomposition (500-800°C)
        mask = temperatures >= 500
        weight_loss[mask] = 62 + 25 * (temperatures[mask] - 500) / 300
        
        # Add random noise
        weight_loss += np.random.normal(0, 0.5, len(temperatures))
        
        return np.clip(weight_loss, 0, 100)
    
    def _generate_tga_derivative(self, temperatures):
        """Generate TGA derivative curve"""
        # This would be the derivative of the TGA curve
        # Simplified version for demonstration
        derivative = np.zeros_like(temperatures)
        
        # Peak decomposition rate around 400°C
        peak_temp = 400
        width = 50
        derivative = 0.3 * np.exp(-0.5 * ((temperatures - peak_temp) / width) ** 2)
        
        return derivative
    
    def save_data(self, filename='material_properties_data.json'):
        """Save generated data to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_data = convert_numpy(self.data)
        
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"Data saved to {filename}")
    
    def generate_summary_report(self):
        """Generate a summary report of the data"""
        report = {
            'generation_date': datetime.now().isoformat(),
            'total_samples': 100,
            'materials_characterized': list(self.data.keys()),
            'data_points_per_material': {}
        }
        
        for material, data in self.data.items():
            if isinstance(data, dict):
                total_points = sum(len(v) if isinstance(v, (list, np.ndarray)) else 1 for v in data.values())
                report['data_points_per_material'][material] = total_points
        
        return report

def main():
    """Main function to generate all material properties data"""
    print("Starting Material Properties Data Generation...")
    print("=" * 50)
    
    # Initialize generator
    generator = MaterialPropertiesGenerator(seed=42)
    
    # Generate all material properties
    generator.generate_cement_properties()
    generator.generate_aggregate_properties()
    generator.generate_rubber_properties()
    generator.generate_water_properties()
    generator.generate_admixture_properties()
    
    # Save data
    generator.save_data('material_properties_data.json')
    
    # Generate summary report
    report = generator.generate_summary_report()
    print("\nSummary Report:")
    print(json.dumps(report, indent=2))
    
    print("\nData generation completed successfully!")

if __name__ == "__main__":
    main()