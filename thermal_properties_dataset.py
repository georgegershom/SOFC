#!/usr/bin/env python3
"""
Thermal Properties Dataset Generator for Fire-Resistant Rubberized Concrete
Generates comprehensive thermal property data including TGA/DSC, thermal conductivity,
specific heat, CTE, and mass loss measurements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import json
from datetime import datetime
import os

class ThermalPropertiesGenerator:
    def __init__(self):
        self.mix_designs = {
            'Control': {'cement': 100, 'water': 40, 'aggregate': 180, 'rubber': 0},
            'Low_Rubber': {'cement': 100, 'water': 40, 'aggregate': 160, 'rubber': 20},
            'Medium_Rubber': {'cement': 100, 'water': 40, 'aggregate': 140, 'rubber': 40},
            'High_Rubber': {'cement': 100, 'water': 40, 'aggregate': 120, 'rubber': 60}
        }
        
        # Material decomposition temperatures (°C)
        self.decomposition_temps = {
            'rubber': {'start': 300, 'peak': 400, 'end': 500},
            'portlandite': {'start': 400, 'peak': 450, 'end': 500},
            'carbonates': {'start': 600, 'peak': 700, 'end': 800},
            'cement_paste': {'start': 200, 'peak': 350, 'end': 600}
        }
    
    def generate_tga_dsc_data(self, mix_name, rubber_content):
        """Generate TGA and DSC data for concrete mix"""
        temp_range = np.linspace(20, 800, 1000)
        
        # Base mass loss curve for cement paste
        cement_mass_loss = self._generate_cement_mass_loss(temp_range)
        
        # Rubber decomposition curve
        rubber_mass_loss = self._generate_rubber_mass_loss(temp_range, rubber_content)
        
        # Combined mass loss
        total_mass_loss = cement_mass_loss + rubber_mass_loss
        
        # Generate DSC heat flow data
        heat_flow = self._generate_heat_flow(temp_range, rubber_content)
        
        # Add realistic noise
        noise_factor = 0.02
        total_mass_loss += np.random.normal(0, noise_factor, len(temp_range))
        heat_flow += np.random.normal(0, 0.1, len(temp_range))
        
        return {
            'temperature': temp_range,
            'mass_loss_percent': total_mass_loss,
            'heat_flow_mW_mg': heat_flow,
            'mix_name': mix_name,
            'rubber_content': rubber_content
        }
    
    def _generate_cement_mass_loss(self, temp):
        """Generate cement paste mass loss curve"""
        mass_loss = np.zeros_like(temp)
        
        # Free water loss (20-100°C)
        mask1 = (temp >= 20) & (temp <= 100)
        mass_loss[mask1] = 3 * (temp[mask1] - 20) / 80
        
        # Bound water loss (100-200°C)
        mask2 = (temp >= 100) & (temp <= 200)
        mass_loss[mask2] = 3 + 2 * (temp[mask2] - 100) / 100
        
        # Portlandite decomposition (400-500°C)
        mask3 = (temp >= 400) & (temp <= 500)
        mass_loss[mask3] = 5 + 3 * (temp[mask3] - 400) / 100
        
        # Carbonate decomposition (600-800°C)
        mask4 = (temp >= 600) & (temp <= 800)
        mass_loss[mask4] = 8 + 4 * (temp[mask4] - 600) / 200
        
        return mass_loss
    
    def _generate_rubber_mass_loss(self, temp, rubber_content):
        """Generate rubber mass loss curve"""
        mass_loss = np.zeros_like(temp)
        
        # Rubber decomposition (300-500°C)
        mask = (temp >= 300) & (temp <= 500)
        rubber_loss = rubber_content * 0.8 * (temp[mask] - 300) / 200
        mass_loss[mask] = rubber_loss
        
        return mass_loss
    
    def _generate_heat_flow(self, temp, rubber_content):
        """Generate DSC heat flow data"""
        heat_flow = np.zeros_like(temp)
        
        # Endothermic peaks for water loss
        mask1 = (temp >= 80) & (temp <= 120)
        heat_flow[mask1] = -2 * np.exp(-((temp[mask1] - 100) / 20) ** 2)
        
        # Exothermic peak for rubber oxidation
        if rubber_content > 0:
            mask2 = (temp >= 350) & (temp <= 450)
            heat_flow[mask2] = rubber_content * 0.1 * np.exp(-((temp[mask2] - 400) / 30) ** 2)
        
        # Endothermic peak for portlandite decomposition
        mask3 = (temp >= 420) & (temp <= 480)
        heat_flow[mask3] = -1.5 * np.exp(-((temp[mask3] - 450) / 20) ** 2)
        
        return heat_flow
    
    def generate_thermal_conductivity_data(self, mix_name, rubber_content):
        """Generate thermal conductivity vs temperature data"""
        temperatures = [25, 100, 200, 400, 600]
        
        # Base thermal conductivity for concrete
        k_base = 1.5  # W/m·K at 25°C
        
        # Rubber reduces thermal conductivity
        rubber_factor = 1 - (rubber_content / 100) * 0.3
        
        # Temperature dependence (decreases with temperature)
        k_values = []
        for T in temperatures:
            k = k_base * rubber_factor * (1 - 0.0005 * (T - 25))
            k_values.append(k)
        
        # Add measurement uncertainty
        k_values = [k + np.random.normal(0, 0.05) for k in k_values]
        
        return {
            'temperature': temperatures,
            'thermal_conductivity': k_values,
            'mix_name': mix_name,
            'rubber_content': rubber_content
        }
    
    def generate_specific_heat_data(self, mix_name, rubber_content):
        """Generate specific heat vs temperature data"""
        temperatures = [25, 100, 200, 400, 600]
        
        # Base specific heat for concrete
        cp_base = 900  # J/kg·K at 25°C
        
        # Rubber increases specific heat slightly
        rubber_factor = 1 + (rubber_content / 100) * 0.1
        
        # Temperature dependence (increases with temperature)
        cp_values = []
        for T in temperatures:
            cp = cp_base * rubber_factor * (1 + 0.0003 * (T - 25))
            cp_values.append(cp)
        
        # Add measurement uncertainty
        cp_values = [cp + np.random.normal(0, 20) for cp in cp_values]
        
        return {
            'temperature': temperatures,
            'specific_heat': cp_values,
            'mix_name': mix_name,
            'rubber_content': rubber_content
        }
    
    def generate_cte_data(self, mix_name, rubber_content):
        """Generate coefficient of thermal expansion data"""
        temperatures = np.linspace(25, 600, 100)
        
        # Base CTE for concrete
        cte_base = 12e-6  # 1/K
        
        # Rubber increases CTE
        rubber_factor = 1 + (rubber_content / 100) * 0.5
        
        # Temperature dependence (increases with temperature)
        cte_values = cte_base * rubber_factor * (1 + 0.0002 * (temperatures - 25))
        
        # Add measurement uncertainty
        cte_values += np.random.normal(0, 0.5e-6, len(temperatures))
        
        return {
            'temperature': temperatures,
            'cte': cte_values,
            'mix_name': mix_name,
            'rubber_content': rubber_content
        }
    
    def generate_mass_loss_during_heating(self, mix_name, rubber_content, heating_rate=5):
        """Generate in-situ mass loss during heating test"""
        time_points = np.linspace(0, 120, 1200)  # 2 hours at 5°C/min
        temperatures = time_points * heating_rate
        
        # Initial mass
        initial_mass = 1000  # grams
        
        # Mass loss rate depends on temperature and rubber content
        mass_loss_rates = []
        current_mass = initial_mass
        
        for i, T in enumerate(temperatures):
            if T <= 100:
                # Free water loss
                loss_rate = 0.1
            elif T <= 200:
                # Bound water loss
                loss_rate = 0.05
            elif T <= 300:
                # Cement paste decomposition
                loss_rate = 0.02
            elif T <= 500 and rubber_content > 0:
                # Rubber decomposition
                loss_rate = 0.3 * (rubber_content / 100)
            elif T <= 600:
                # Portlandite decomposition
                loss_rate = 0.1
            else:
                # Carbonate decomposition
                loss_rate = 0.05
            
            current_mass -= loss_rate
            mass_loss_rates.append(current_mass)
        
        # Add realistic noise
        mass_loss_rates = np.array(mass_loss_rates)
        mass_loss_rates += np.random.normal(0, 2, len(mass_loss_rates))
        
        return {
            'time': time_points,
            'temperature': temperatures,
            'mass': mass_loss_rates,
            'mass_loss_percent': (initial_mass - mass_loss_rates) / initial_mass * 100,
            'mix_name': mix_name,
            'rubber_content': rubber_content,
            'heating_rate': heating_rate
        }
    
    def generate_all_thermal_data(self):
        """Generate complete thermal properties dataset"""
        all_data = {}
        
        for mix_name, composition in self.mix_designs.items():
            rubber_content = composition['rubber']
            
            print(f"Generating thermal data for {mix_name} (Rubber: {rubber_content}%)")
            
            mix_data = {
                'tga_dsc': self.generate_tga_dsc_data(mix_name, rubber_content),
                'thermal_conductivity': self.generate_thermal_conductivity_data(mix_name, rubber_content),
                'specific_heat': self.generate_specific_heat_data(mix_name, rubber_content),
                'cte': self.generate_cte_data(mix_name, rubber_content),
                'mass_loss_heating': self.generate_mass_loss_during_heating(mix_name, rubber_content)
            }
            
            all_data[mix_name] = mix_data
        
        return all_data
    
    def save_thermal_data(self, data, output_dir='/workspace/thermal_data'):
        """Save thermal data to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON for easy access
        with open(f'{output_dir}/thermal_properties_complete.json', 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save individual CSV files for each property
        for mix_name, mix_data in data.items():
            mix_dir = f'{output_dir}/{mix_name}'
            os.makedirs(mix_dir, exist_ok=True)
            
            # TGA/DSC data
            tga_df = pd.DataFrame(mix_data['tga_dsc'])
            tga_df.to_csv(f'{mix_dir}/tga_dsc_data.csv', index=False)
            
            # Thermal conductivity
            k_df = pd.DataFrame(mix_data['thermal_conductivity'])
            k_df.to_csv(f'{mix_dir}/thermal_conductivity.csv', index=False)
            
            # Specific heat
            cp_df = pd.DataFrame(mix_data['specific_heat'])
            cp_df.to_csv(f'{mix_dir}/specific_heat.csv', index=False)
            
            # CTE
            cte_df = pd.DataFrame(mix_data['cte'])
            cte_df.to_csv(f'{mix_dir}/cte.csv', index=False)
            
            # Mass loss during heating
            mass_df = pd.DataFrame(mix_data['mass_loss_heating'])
            mass_df.to_csv(f'{mix_dir}/mass_loss_heating.csv', index=False)
        
        print(f"Thermal data saved to {output_dir}")
    
    def create_thermal_plots(self, data, output_dir='/workspace/thermal_data/plots'):
        """Create visualization plots for thermal data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # TGA curves
        plt.figure(figsize=(12, 8))
        for mix_name, mix_data in data.items():
            tga = mix_data['tga_dsc']
            plt.plot(tga['temperature'], tga['mass_loss_percent'], 
                    label=f'{mix_name} (Rubber: {tga["rubber_content"]}%)', linewidth=2)
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Mass Loss (%)')
        plt.title('Thermogravimetric Analysis (TGA) - Mass Loss vs Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/tga_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # DSC curves
        plt.figure(figsize=(12, 8))
        for mix_name, mix_data in data.items():
            dsc = mix_data['tga_dsc']
            plt.plot(dsc['temperature'], dsc['heat_flow_mW_mg'], 
                    label=f'{mix_name} (Rubber: {dsc["rubber_content"]}%)', linewidth=2)
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Heat Flow (mW/mg)')
        plt.title('Differential Scanning Calorimetry (DSC) - Heat Flow vs Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/dsc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Thermal conductivity
        plt.figure(figsize=(10, 6))
        for mix_name, mix_data in data.items():
            k_data = mix_data['thermal_conductivity']
            plt.plot(k_data['temperature'], k_data['thermal_conductivity'], 
                    'o-', label=f'{mix_name} (Rubber: {k_data["rubber_content"]}%)', linewidth=2, markersize=6)
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Thermal Conductivity (W/m·K)')
        plt.title('Thermal Conductivity vs Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/thermal_conductivity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Mass loss during heating
        plt.figure(figsize=(12, 8))
        for mix_name, mix_data in data.items():
            mass_data = mix_data['mass_loss_heating']
            plt.plot(mass_data['temperature'], mass_data['mass_loss_percent'], 
                    label=f'{mix_name} (Rubber: {mass_data["rubber_content"]}%)', linewidth=2)
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Mass Loss (%)')
        plt.title('In-situ Mass Loss During Heating')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/mass_loss_heating.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Thermal plots saved to {output_dir}")

def main():
    """Main function to generate thermal properties dataset"""
    print("Generating High-Temperature Thermal Properties Dataset")
    print("=" * 60)
    
    generator = ThermalPropertiesGenerator()
    
    # Generate all thermal data
    thermal_data = generator.generate_all_thermal_data()
    
    # Save data
    generator.save_thermal_data(thermal_data)
    
    # Create plots
    generator.create_thermal_plots(thermal_data)
    
    print("\nThermal Properties Dataset Generation Complete!")
    print("Generated data for 4 concrete mixes:")
    for mix_name in generator.mix_designs.keys():
        print(f"  - {mix_name}")
    
    print("\nData includes:")
    print("  - TGA/DSC curves (20-800°C)")
    print("  - Thermal conductivity (25-600°C)")
    print("  - Specific heat (25-600°C)")
    print("  - Coefficient of thermal expansion (25-600°C)")
    print("  - In-situ mass loss during heating")

if __name__ == "__main__":
    main()