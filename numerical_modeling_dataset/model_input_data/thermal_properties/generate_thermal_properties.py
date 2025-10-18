#!/usr/bin/env python3
"""
Generate Temperature-Dependent Thermal Properties for Rubberized Concrete
Based on literature data and experimental correlations for fire-resistant concrete
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import json
from datetime import datetime

class ThermalPropertiesGenerator:
    def __init__(self, rubber_content_percent=0):
        """
        Initialize thermal properties generator
        rubber_content_percent: 0-30% typical range for rubberized concrete
        """
        self.rubber_content = rubber_content_percent
        self.temperature_range = np.arange(20, 1201, 10)  # 20°C to 1200°C
        
    def thermal_conductivity(self, T):
        """
        Generate thermal conductivity k(T) [W/m·K]
        Based on Eurocode 2 and modified for rubber content
        """
        # Base concrete thermal conductivity (Eurocode 2)
        if T <= 20:
            k_base = 1.36
        elif T <= 100:
            k_base = 1.36 - 0.136 * (T - 20) / 80
        elif T <= 200:
            k_base = 1.224 - 0.124 * (T - 100) / 100
        elif T <= 400:
            k_base = 1.1 - 0.2 * (T - 200) / 200
        elif T <= 600:
            k_base = 0.9 - 0.15 * (T - 400) / 200
        elif T <= 800:
            k_base = 0.75 - 0.1 * (T - 600) / 200
        elif T <= 1000:
            k_base = 0.65 - 0.1 * (T - 800) / 200
        else:
            k_base = 0.55
            
        # Rubber modification factor (rubber has lower k ~ 0.16 W/m·K)
        rubber_factor = 1 - (self.rubber_content / 100) * 0.35
        
        # Add some realistic noise
        noise = np.random.normal(0, 0.02)
        
        return max(0.1, k_base * rubber_factor + noise)
    
    def specific_heat(self, T):
        """
        Generate specific heat capacity cp(T) [J/kg·K]
        Including peak at 100-200°C due to moisture evaporation
        """
        # Base concrete specific heat
        if T <= 100:
            cp_base = 900
        elif T <= 200:
            # Peak due to moisture evaporation
            cp_base = 900 + 1500 * np.exp(-((T - 115) / 15) ** 2)
        elif T <= 400:
            cp_base = 1000 + (T - 200)
        elif T <= 600:
            cp_base = 1200
        elif T <= 800:
            cp_base = 1200 - 100 * (T - 600) / 200
        else:
            cp_base = 1100
            
        # Rubber modification (rubber has higher cp ~ 2000 J/kg·K)
        rubber_factor = 1 + (self.rubber_content / 100) * 0.25
        
        # Add noise
        noise = np.random.normal(0, 20)
        
        return max(800, cp_base * rubber_factor + noise)
    
    def density(self, T):
        """
        Generate density ρ(T) [kg/m³]
        Accounting for moisture loss and decomposition
        """
        # Initial density at room temperature
        rho_0 = 2400 - self.rubber_content * 8  # Rubber reduces density
        
        # Mass loss factors
        if T <= 100:
            mass_loss = 0
        elif T <= 200:
            # Moisture evaporation (3-5% mass loss)
            mass_loss = 0.04 * (T - 100) / 100
        elif T <= 400:
            # Dehydration of cement paste
            mass_loss = 0.04 + 0.03 * (T - 200) / 200
        elif T <= 600:
            # Decomposition of Ca(OH)2
            mass_loss = 0.07 + 0.02 * (T - 400) / 200
        elif T <= 800:
            # Rubber decomposition (if present)
            rubber_loss = (self.rubber_content / 100) * 0.15 * (T - 600) / 200
            mass_loss = 0.09 + rubber_loss
        else:
            # CaCO3 decomposition
            mass_loss = 0.09 + (self.rubber_content / 100) * 0.15 + 0.03 * min((T - 800) / 200, 1)
            
        # Add noise
        noise = np.random.normal(0, 10)
        
        return max(1800, rho_0 * (1 - mass_loss) + noise)
    
    def thermal_diffusivity(self, T):
        """
        Calculate thermal diffusivity α(T) [m²/s]
        α = k / (ρ * cp)
        """
        k = self.thermal_conductivity(T)
        rho = self.density(T)
        cp = self.specific_heat(T)
        
        return k / (rho * cp)
    
    def generate_dataset(self, specimen_variations=5):
        """
        Generate complete dataset with variations for multiple specimens
        """
        datasets = {}
        
        for specimen in range(1, specimen_variations + 1):
            data = {
                'temperature_C': [],
                'thermal_conductivity_W_mK': [],
                'specific_heat_J_kgK': [],
                'density_kg_m3': [],
                'thermal_diffusivity_m2_s': [],
                'volumetric_heat_capacity_J_m3K': []
            }
            
            # Add some specimen-to-specimen variation
            self.rubber_content += np.random.normal(0, 1)
            
            for T in self.temperature_range:
                k = self.thermal_conductivity(T)
                cp = self.specific_heat(T)
                rho = self.density(T)
                alpha = k / (rho * cp)
                vol_heat_cap = rho * cp
                
                data['temperature_C'].append(T)
                data['thermal_conductivity_W_mK'].append(k)
                data['specific_heat_J_kgK'].append(cp)
                data['density_kg_m3'].append(rho)
                data['thermal_diffusivity_m2_s'].append(alpha)
                data['volumetric_heat_capacity_J_m3K'].append(vol_heat_cap)
            
            datasets[f'specimen_{specimen}'] = pd.DataFrame(data)
            
            # Reset rubber content
            self.rubber_content -= np.random.normal(0, 1)
        
        return datasets
    
    def generate_hot_disk_data(self, test_temperatures=[20, 100, 200, 400, 600]):
        """
        Generate simulated Hot Disk measurement data
        """
        hot_disk_data = {
            'test_temperature_C': [],
            'thermal_conductivity_W_mK': [],
            'thermal_diffusivity_mm2_s': [],
            'volumetric_heat_capacity_MJ_m3K': [],
            'measurement_uncertainty_percent': [],
            'probe_type': [],
            'measurement_time_s': [],
            'heating_power_W': []
        }
        
        for T in test_temperatures:
            k = self.thermal_conductivity(T)
            alpha = self.thermal_diffusivity(T)
            rho = self.density(T)
            cp = self.specific_heat(T)
            
            # Add measurement uncertainty
            uncertainty = 3 + np.random.uniform(-0.5, 0.5)  # 2.5-3.5%
            
            hot_disk_data['test_temperature_C'].append(T)
            hot_disk_data['thermal_conductivity_W_mK'].append(k * (1 + np.random.normal(0, 0.01)))
            hot_disk_data['thermal_diffusivity_mm2_s'].append(alpha * 1e6 * (1 + np.random.normal(0, 0.01)))
            hot_disk_data['volumetric_heat_capacity_MJ_m3K'].append(rho * cp / 1e6 * (1 + np.random.normal(0, 0.01)))
            hot_disk_data['measurement_uncertainty_percent'].append(uncertainty)
            hot_disk_data['probe_type'].append('Kapton 5501')
            hot_disk_data['measurement_time_s'].append(np.random.uniform(20, 40))
            hot_disk_data['heating_power_W'].append(np.random.uniform(0.5, 2.0))
        
        return pd.DataFrame(hot_disk_data)

def main():
    # Generate datasets for different rubber contents
    rubber_contents = [0, 5, 10, 15, 20, 25, 30]  # Percentage
    
    all_data = {}
    
    for rubber_pct in rubber_contents:
        print(f"Generating thermal properties for {rubber_pct}% rubber content...")
        
        generator = ThermalPropertiesGenerator(rubber_content_percent=rubber_pct)
        
        # Generate main dataset
        datasets = generator.generate_dataset(specimen_variations=3)
        
        # Generate Hot Disk data
        hot_disk = generator.generate_hot_disk_data()
        
        # Store all data
        all_data[f'rubber_{rubber_pct}pct'] = {
            'specimens': datasets,
            'hot_disk': hot_disk
        }
        
        # Save individual CSV files
        for specimen_name, df in datasets.items():
            filename = f'thermal_properties_rubber_{rubber_pct}pct_{specimen_name}.csv'
            df.to_csv(filename, index=False)
            print(f"  Saved: {filename}")
        
        # Save Hot Disk data
        hot_disk_filename = f'hot_disk_measurements_rubber_{rubber_pct}pct.csv'
        hot_disk.to_csv(hot_disk_filename, index=False)
        print(f"  Saved: {hot_disk_filename}")
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for rubber_pct in rubber_contents:
        specimen_1 = all_data[f'rubber_{rubber_pct}pct']['specimens']['specimen_1']
        
        # Thermal conductivity
        axes[0, 0].plot(specimen_1['temperature_C'], 
                       specimen_1['thermal_conductivity_W_mK'],
                       label=f'{rubber_pct}% rubber', alpha=0.7)
        
        # Specific heat
        axes[0, 1].plot(specimen_1['temperature_C'],
                       specimen_1['specific_heat_J_kgK'],
                       label=f'{rubber_pct}% rubber', alpha=0.7)
        
        # Density
        axes[1, 0].plot(specimen_1['temperature_C'],
                       specimen_1['density_kg_m3'],
                       label=f'{rubber_pct}% rubber', alpha=0.7)
        
        # Thermal diffusivity
        axes[1, 1].plot(specimen_1['temperature_C'],
                       specimen_1['thermal_diffusivity_m2_s'] * 1e6,
                       label=f'{rubber_pct}% rubber', alpha=0.7)
    
    # Format plots
    axes[0, 0].set_xlabel('Temperature (°C)')
    axes[0, 0].set_ylabel('Thermal Conductivity (W/m·K)')
    axes[0, 0].set_title('Temperature-Dependent Thermal Conductivity')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Temperature (°C)')
    axes[0, 1].set_ylabel('Specific Heat (J/kg·K)')
    axes[0, 1].set_title('Temperature-Dependent Specific Heat Capacity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Temperature (°C)')
    axes[1, 0].set_ylabel('Density (kg/m³)')
    axes[1, 0].set_title('Temperature-Dependent Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Temperature (°C)')
    axes[1, 1].set_ylabel('Thermal Diffusivity (mm²/s)')
    axes[1, 1].set_title('Temperature-Dependent Thermal Diffusivity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Thermal Properties of Rubberized Concrete at Elevated Temperatures', fontsize=14)
    plt.tight_layout()
    plt.savefig('thermal_properties_overview.png', dpi=300)
    plt.close()
    
    # Generate metadata
    metadata = {
        'generated_date': datetime.now().isoformat(),
        'temperature_range_C': [20, 1200],
        'rubber_content_range_percent': [0, 30],
        'test_method': 'Hot Disk Thermal Constants Analyzer',
        'standard': 'ISO 22007-2',
        'specimen_dimensions_mm': {
            'diameter': 100,
            'thickness': 50
        },
        'conditioning': 'Specimens dried at 105°C for 24 hours before testing',
        'data_source': 'Synthetic data based on Eurocode 2 and literature correlations'
    }
    
    with open('thermal_properties_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nThermal properties dataset generation complete!")
    print(f"Generated data for {len(rubber_contents)} rubber content variations")
    print(f"Total number of data files: {len(rubber_contents) * 4}")

if __name__ == "__main__":
    main()