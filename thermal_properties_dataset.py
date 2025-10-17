#!/usr/bin/env python3
"""
Thermal Properties Dataset Generator for Fire-Resistant Rubberized Concrete
Pillar 2: High-Temperature Experimental Investigation - Thermal Properties

This module generates comprehensive thermal property data including:
- TGA/DSC analysis (20°C to 800°C)
- Thermal conductivity and specific heat at multiple temperatures
- Coefficient of thermal expansion (CTE)
- In-situ mass loss during heating
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import json
from datetime import datetime
import os

class ThermalPropertiesGenerator:
    def __init__(self):
        self.temperature_range = np.linspace(20, 800, 781)  # 1°C increments
        self.mix_types = {
            'Control': {'rubber_content': 0.0, 'w_c_ratio': 0.45, 'cement_type': 'OPC'},
            'R5': {'rubber_content': 0.05, 'w_c_ratio': 0.45, 'cement_type': 'OPC'},
            'R10': {'rubber_content': 0.10, 'w_c_ratio': 0.45, 'cement_type': 'OPC'},
            'R15': {'rubber_content': 0.15, 'w_c_ratio': 0.45, 'cement_type': 'OPC'},
            'R20': {'rubber_content': 0.20, 'w_c_ratio': 0.45, 'cement_type': 'OPC'},
            'Raw_Rubber': {'rubber_content': 1.0, 'w_c_ratio': 0.0, 'cement_type': 'None'}
        }
        
        # Key decomposition temperatures (in °C)
        self.decomposition_temps = {
            'rubber_onset': 300,
            'rubber_peak': 400,
            'portlandite': 450,
            'calcite_onset': 600,
            'calcite_peak': 750
        }
    
    def generate_tga_data(self, mix_type):
        """Generate TGA (Thermal Gravimetric Analysis) data"""
        temp = self.temperature_range
        mix_props = self.mix_types[mix_type]
        rubber_content = mix_props['rubber_content']
        
        # Base mass loss curve for cement paste
        base_mass_loss = np.zeros_like(temp)
        
        # Water loss (20-150°C)
        water_loss = 0.15 * (1 - rubber_content) * np.exp(-((temp - 100) / 30) ** 2)
        base_mass_loss += water_loss
        
        # Rubber decomposition (300-500°C)
        if rubber_content > 0:
            rubber_loss = rubber_content * 0.95 * (1 / (1 + np.exp(-(temp - 350) / 20)))
            base_mass_loss += rubber_loss
        
        # Portlandite decomposition (400-500°C)
        portlandite_loss = 0.08 * (1 - rubber_content) * (1 / (1 + np.exp(-(temp - 450) / 15)))
        base_mass_loss += portlandite_loss
        
        # Carbonate decomposition (600-800°C)
        carbonate_loss = 0.12 * (1 - rubber_content) * (1 / (1 + np.exp(-(temp - 700) / 30)))
        base_mass_loss += carbonate_loss
        
        # Add noise to simulate experimental conditions
        noise = np.random.normal(0, 0.002, len(temp))
        mass_loss = np.clip(base_mass_loss + noise, 0, 1)
        
        # Calculate derivative (DTG)
        dtg = np.gradient(mass_loss, temp)
        dtg = savgol_filter(dtg, 21, 3)  # Smooth the derivative
        
        return {
            'temperature': temp,
            'mass_loss_percent': mass_loss * 100,
            'dtg': dtg * 100,
            'residual_mass_percent': (1 - mass_loss) * 100
        }
    
    def generate_dsc_data(self, mix_type):
        """Generate DSC (Differential Scanning Calorimetry) data"""
        temp = self.temperature_range
        mix_props = self.mix_types[mix_type]
        rubber_content = mix_props['rubber_content']
        
        heat_flow = np.zeros_like(temp)
        
        # Endothermic peaks for decomposition
        if rubber_content > 0:
            # Rubber decomposition endotherm
            rubber_peak = -rubber_content * 2.5 * np.exp(-((temp - 400) / 30) ** 2)
            heat_flow += rubber_peak
        
        # Portlandite decomposition endotherm
        portlandite_peak = -0.3 * (1 - rubber_content) * np.exp(-((temp - 450) / 20) ** 2)
        heat_flow += portlandite_peak
        
        # Carbonate decomposition endotherm
        carbonate_peak = -0.4 * (1 - rubber_content) * np.exp(-((temp - 700) / 40) ** 2)
        heat_flow += carbonate_peak
        
        # Add baseline drift and noise
        baseline = 0.01 * (temp - 20) / 100  # Slight upward drift
        noise = np.random.normal(0, 0.05, len(temp))
        heat_flow += baseline + noise
        
        return {
            'temperature': temp,
            'heat_flow_mW_mg': heat_flow
        }
    
    def generate_thermal_conductivity(self, mix_type):
        """Generate thermal conductivity data at multiple temperatures"""
        test_temps = [25, 100, 200, 400, 600]
        mix_props = self.mix_types[mix_type]
        rubber_content = mix_props['rubber_content']
        
        # Base thermal conductivity decreases with temperature
        base_k = 2.5 - 0.5 * rubber_content  # W/m·K at 25°C
        
        k_values = []
        for T in test_temps:
            # Thermal conductivity decreases with temperature
            k = base_k * (1 - 0.3 * (T - 25) / 775)
            
            # Rubber reduces thermal conductivity more at higher temperatures
            rubber_effect = -0.8 * rubber_content * (T / 800) ** 2
            k += rubber_effect
            
            # Add experimental uncertainty
            k += np.random.normal(0, 0.05)
            k_values.append(max(0.1, k))  # Ensure positive values
        
        return {
            'temperature': test_temps,
            'thermal_conductivity': k_values,
            'units': 'W/m·K'
        }
    
    def generate_specific_heat(self, mix_type):
        """Generate specific heat data at multiple temperatures"""
        test_temps = [25, 100, 200, 400, 600]
        mix_props = self.mix_types[mix_type]
        rubber_content = mix_props['rubber_content']
        
        # Base specific heat increases with temperature
        base_cp = 0.9 + 0.1 * rubber_content  # kJ/kg·K at 25°C
        
        cp_values = []
        for T in test_temps:
            # Specific heat increases with temperature
            cp = base_cp * (1 + 0.2 * (T - 25) / 775)
            
            # Rubber increases specific heat capacity
            rubber_effect = 0.3 * rubber_content * (1 + 0.1 * T / 100)
            cp += rubber_effect
            
            # Add experimental uncertainty
            cp += np.random.normal(0, 0.02)
            cp_values.append(max(0.5, cp))  # Ensure reasonable values
        
        return {
            'temperature': test_temps,
            'specific_heat': cp_values,
            'units': 'kJ/kg·K'
        }
    
    def generate_cte_data(self, mix_type):
        """Generate coefficient of thermal expansion data"""
        temp = np.linspace(20, 600, 581)  # 1°C/min heating rate
        mix_props = self.mix_types[mix_type]
        rubber_content = mix_props['rubber_content']
        
        # Base CTE for cement paste
        base_cte = 12e-6  # /°C
        
        # CTE increases with temperature and rubber content
        cte = base_cte * (1 + 0.5 * (temp - 20) / 580)
        cte += 5e-6 * rubber_content * (1 + 0.3 * (temp - 20) / 580)
        
        # Add some nonlinearity at high temperatures
        cte += 2e-6 * np.exp((temp - 400) / 100) * (temp > 400)
        
        # Add experimental noise
        cte += np.random.normal(0, 0.5e-6, len(temp))
        
        return {
            'temperature': temp,
            'cte': cte,
            'units': '/°C'
        }
    
    def generate_mass_loss_during_heating(self, mix_type):
        """Generate in-situ mass loss data during heating"""
        temp = np.linspace(20, 800, 781)
        mix_props = self.mix_types[mix_type]
        rubber_content = mix_props['rubber_content']
        
        # Initial mass (normalized to 1.0)
        initial_mass = 1.0
        mass_loss = np.zeros_like(temp)
        
        # Water loss (20-150°C)
        water_loss = 0.15 * (1 - rubber_content) * (1 / (1 + np.exp(-(temp - 100) / 20)))
        mass_loss += water_loss
        
        # Rubber decomposition (300-500°C)
        if rubber_content > 0:
            rubber_loss = rubber_content * 0.95 * (1 / (1 + np.exp(-(temp - 350) / 25)))
            mass_loss += rubber_loss
        
        # Portlandite decomposition (400-500°C)
        portlandite_loss = 0.08 * (1 - rubber_content) * (1 / (1 + np.exp(-(temp - 450) / 15)))
        mass_loss += portlandite_loss
        
        # Carbonate decomposition (600-800°C)
        carbonate_loss = 0.12 * (1 - rubber_content) * (1 / (1 + np.exp(-(temp - 700) / 30)))
        mass_loss += carbonate_loss
        
        # Add experimental noise
        noise = np.random.normal(0, 0.001, len(temp))
        mass_loss += noise
        mass_loss = np.clip(mass_loss, 0, 1)
        
        current_mass = initial_mass - mass_loss
        
        return {
            'temperature': temp,
            'mass_loss_percent': mass_loss * 100,
            'current_mass_percent': current_mass * 100,
            'mass_loss_rate': np.gradient(mass_loss, temp) * 100
        }
    
    def generate_all_thermal_data(self):
        """Generate complete thermal properties dataset for all mix types"""
        all_data = {}
        
        for mix_type in self.mix_types.keys():
            print(f"Generating thermal data for {mix_type}...")
            
            all_data[mix_type] = {
                'mix_properties': self.mix_types[mix_type],
                'tga': self.generate_tga_data(mix_type),
                'dsc': self.generate_dsc_data(mix_type),
                'thermal_conductivity': self.generate_thermal_conductivity(mix_type),
                'specific_heat': self.generate_specific_heat(mix_type),
                'cte': self.generate_cte_data(mix_type),
                'mass_loss_heating': self.generate_mass_loss_during_heating(mix_type)
            }
        
        return all_data
    
    def save_data(self, data, output_dir='/workspace/thermal_data'):
        """Save all thermal data to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON for easy access
        with open(f'{output_dir}/thermal_properties_complete.json', 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save individual CSV files for each mix type
        for mix_type, mix_data in data.items():
            mix_dir = f'{output_dir}/{mix_type}'
            os.makedirs(mix_dir, exist_ok=True)
            
            # TGA data
            tga_df = pd.DataFrame(mix_data['tga'])
            tga_df.to_csv(f'{mix_dir}/tga_data.csv', index=False)
            
            # DSC data
            dsc_df = pd.DataFrame(mix_data['dsc'])
            dsc_df.to_csv(f'{mix_dir}/dsc_data.csv', index=False)
            
            # Thermal conductivity
            k_df = pd.DataFrame(mix_data['thermal_conductivity'])
            k_df.to_csv(f'{mix_dir}/thermal_conductivity.csv', index=False)
            
            # Specific heat
            cp_df = pd.DataFrame(mix_data['specific_heat'])
            cp_df.to_csv(f'{mix_dir}/specific_heat.csv', index=False)
            
            # CTE data
            cte_df = pd.DataFrame(mix_data['cte'])
            cte_df.to_csv(f'{mix_dir}/cte_data.csv', index=False)
            
            # Mass loss during heating
            mass_df = pd.DataFrame(mix_data['mass_loss_heating'])
            mass_df.to_csv(f'{mix_dir}/mass_loss_heating.csv', index=False)
        
        print(f"Thermal data saved to {output_dir}")
    
    def create_visualizations(self, data, output_dir='/workspace/thermal_data/plots'):
        """Create comprehensive visualizations of thermal data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # TGA comparison plot
        plt.figure(figsize=(12, 8))
        for mix_type, mix_data in data.items():
            if mix_type != 'Raw_Rubber':
                plt.plot(mix_data['tga']['temperature'], 
                        mix_data['tga']['mass_loss_percent'], 
                        label=f'{mix_type} (R={self.mix_types[mix_type]["rubber_content"]*100:.0f}%)',
                        linewidth=2)
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Mass Loss (%)')
        plt.title('Thermal Gravimetric Analysis - Mass Loss vs Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/tga_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # DSC comparison plot
        plt.figure(figsize=(12, 8))
        for mix_type, mix_data in data.items():
            if mix_type != 'Raw_Rubber':
                plt.plot(mix_data['dsc']['temperature'], 
                        mix_data['dsc']['heat_flow_mW_mg'], 
                        label=f'{mix_type} (R={self.mix_types[mix_type]["rubber_content"]*100:.0f}%)',
                        linewidth=2)
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Heat Flow (mW/mg)')
        plt.title('Differential Scanning Calorimetry - Heat Flow vs Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/dsc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Thermal conductivity vs temperature
        plt.figure(figsize=(10, 6))
        for mix_type, mix_data in data.items():
            if mix_type != 'Raw_Rubber':
                plt.plot(mix_data['thermal_conductivity']['temperature'], 
                        mix_data['thermal_conductivity']['thermal_conductivity'], 
                        'o-', label=f'{mix_type} (R={self.mix_types[mix_type]["rubber_content"]*100:.0f}%)',
                        markersize=8, linewidth=2)
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Thermal Conductivity (W/m·K)')
        plt.title('Thermal Conductivity vs Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/thermal_conductivity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # CTE vs temperature
        plt.figure(figsize=(10, 6))
        for mix_type, mix_data in data.items():
            if mix_type != 'Raw_Rubber':
                plt.plot(mix_data['cte']['temperature'], 
                        mix_data['cte']['cte'] * 1e6, 
                        label=f'{mix_type} (R={self.mix_types[mix_type]["rubber_content"]*100:.0f}%)',
                        linewidth=2)
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Coefficient of Thermal Expansion (×10⁻⁶ /°C)')
        plt.title('Coefficient of Thermal Expansion vs Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/cte_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to generate and save thermal properties dataset"""
    print("Generating High-Temperature Thermal Properties Dataset...")
    print("=" * 60)
    
    generator = ThermalPropertiesGenerator()
    
    # Generate all thermal data
    thermal_data = generator.generate_all_thermal_data()
    
    # Save data
    generator.save_data(thermal_data)
    
    # Create visualizations
    generator.create_visualizations(thermal_data)
    
    print("\nThermal Properties Dataset Generation Complete!")
    print("Generated data includes:")
    print("- TGA/DSC analysis (20°C to 800°C)")
    print("- Thermal conductivity at multiple temperatures")
    print("- Specific heat capacity at multiple temperatures")
    print("- Coefficient of thermal expansion")
    print("- In-situ mass loss during heating")
    print(f"- Data saved to /workspace/thermal_data/")
    print(f"- Plots saved to /workspace/thermal_data/plots/")

if __name__ == "__main__":
    main()