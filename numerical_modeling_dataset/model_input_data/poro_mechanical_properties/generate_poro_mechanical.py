#!/usr/bin/env python3
"""
Generate Poro-Mechanical Properties for Rubberized Concrete
Includes permeability, porosity, pore pressure, and moisture transport properties
Critical for modeling spalling and moisture-driven phenomena
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import erf
import json
from datetime import datetime

class PoroMechanicalGenerator:
    def __init__(self, rubber_content_percent=0, w_c_ratio=0.4):
        """
        Initialize poro-mechanical properties generator
        rubber_content_percent: 0-30% typical range
        w_c_ratio: water-cement ratio (0.3-0.5 typical for HPC)
        """
        self.rubber_content = rubber_content_percent
        self.w_c = w_c_ratio
        self.temperature_range = np.arange(20, 801, 20)
        
        # Initial properties at room temperature
        self.porosity_0 = 0.08 + 0.15 * w_c_ratio + rubber_content_percent * 0.002  # Rubber slightly increases porosity
        self.permeability_0 = 1e-18 * (1 + rubber_content_percent * 0.1)  # m^2
        self.saturation_0 = 0.75  # Initial saturation degree
        
    def porosity(self, T, damage=0):
        """
        Generate porosity φ(T, D) [-]
        Accounts for dehydration, microcracking, and damage
        """
        # Temperature-induced changes
        if T <= 100:
            temp_factor = 1.0
        elif T <= 200:
            # Evaporation increases porosity
            temp_factor = 1.0 + 0.1 * (T - 100) / 100
        elif T <= 400:
            # Dehydration of cement paste
            temp_factor = 1.1 + 0.15 * (T - 200) / 200
        elif T <= 600:
            # Ca(OH)2 decomposition
            temp_factor = 1.25 + 0.2 * (T - 400) / 200
        else:
            # Further decomposition
            temp_factor = 1.45 + 0.15 * (T - 600) / 200
        
        # Damage-induced porosity increase
        damage_factor = 1 + 2 * damage  # Damage significantly increases porosity
        
        # Rubber degradation effect (above 200°C)
        if T > 200:
            rubber_degradation = (self.rubber_content / 100) * 0.05 * min((T - 200) / 400, 1)
        else:
            rubber_degradation = 0
        
        porosity = self.porosity_0 * temp_factor * damage_factor + rubber_degradation
        
        # Add noise
        noise = np.random.normal(0, 0.005)
        
        return min(0.5, porosity + noise)  # Cap at 50% porosity
    
    def permeability(self, T, damage=0, porosity_val=None):
        """
        Generate intrinsic permeability k(T, D, φ) [m^2]
        Strongly coupled with porosity and damage
        """
        if porosity_val is None:
            porosity_val = self.porosity(T, damage)
        
        # Kozeny-Carman relation: k ~ φ³/(1-φ)²
        if porosity_val < 0.99:
            kozeny_factor = (porosity_val ** 3) / ((1 - porosity_val) ** 2)
            kozeny_factor = kozeny_factor / (self.porosity_0 ** 3 / (1 - self.porosity_0) ** 2)
        else:
            kozeny_factor = 1e6  # Very high permeability for extreme porosity
        
        # Temperature effect on permeability
        if T <= 100:
            temp_factor = 1.0
        elif T <= 200:
            # Microcracking begins
            temp_factor = 1.0 + 10 * (T - 100) / 100
        elif T <= 400:
            # Significant microcracking
            temp_factor = 11.0 + 89 * (T - 200) / 200
        elif T <= 600:
            # Severe degradation
            temp_factor = 100 + 900 * (T - 400) / 200
        else:
            temp_factor = 1000 + 4000 * min((T - 600) / 200, 1)
        
        # Damage effect (exponential increase)
        damage_factor = np.exp(5 * damage)
        
        # Rubber effect (creates additional pathways when degraded)
        if T > 250:
            rubber_factor = 1 + (self.rubber_content / 100) * 10 * min((T - 250) / 350, 1)
        else:
            rubber_factor = 1.0
        
        permeability = self.permeability_0 * kozeny_factor * temp_factor * damage_factor * rubber_factor
        
        # Add noise
        noise = np.random.normal(0, permeability * 0.1)
        
        return max(1e-20, min(1e-10, permeability + noise))  # Bounded permeability
    
    def saturation_degree(self, T, time_hours=0):
        """
        Generate saturation degree S(T, t) [-]
        Accounts for evaporation and moisture transport
        """
        # Temperature-driven evaporation
        if T <= 100:
            evap_rate = 0.0
        elif T <= 200:
            evap_rate = 0.05 * (T - 100) / 100  # per hour
        elif T <= 400:
            evap_rate = 0.05 + 0.15 * (T - 200) / 200
        else:
            evap_rate = 0.2
        
        # Time-dependent drying
        saturation = self.saturation_0 * np.exp(-evap_rate * time_hours)
        
        # Minimum saturation (bound water)
        min_saturation = 0.1 * np.exp(-T / 500)
        
        # Add noise
        noise = np.random.normal(0, 0.02)
        
        return max(min_saturation, min(1.0, saturation + noise))
    
    def pore_pressure(self, T, saturation=None):
        """
        Generate pore pressure p(T) [MPa]
        Critical for spalling prediction
        """
        if saturation is None:
            saturation = self.saturation_degree(T, time_hours=0.5)
        
        # Clapeyron equation for vapor pressure
        if T <= 100:
            p_vapor = 0.101325 * np.exp(17.27 * (T - 20) / (T + 237.3))  # MPa
        else:
            # Above 100°C, pressure builds rapidly
            p_vapor = 0.101325 * np.exp((T - 100) / 50)
        
        # Saturation effect
        p_pore = p_vapor * saturation
        
        # Pressure buildup factor (depends on permeability)
        k = self.permeability(T)
        if k < 1e-16:
            buildup_factor = 5.0  # Low permeability causes pressure buildup
        elif k < 1e-14:
            buildup_factor = 2.0
        else:
            buildup_factor = 1.0
        
        p_pore *= buildup_factor
        
        # Rubber content effect (rubber can accommodate some pressure)
        rubber_relief = 1 - (self.rubber_content / 100) * 0.2
        p_pore *= rubber_relief
        
        # Add noise
        noise = np.random.normal(0, p_pore * 0.1)
        
        return max(0, p_pore + noise)
    
    def moisture_diffusivity(self, T, saturation=None):
        """
        Generate moisture diffusivity D_m(T, S) [m^2/s]
        """
        if saturation is None:
            saturation = self.saturation_degree(T)
        
        # Base diffusivity at room temperature
        D_0 = 1e-9  # m^2/s
        
        # Temperature dependence (Arrhenius-type)
        temp_factor = np.exp(-2800 * (1 / (T + 273) - 1 / 293))
        
        # Saturation dependence
        sat_factor = saturation ** 2
        
        # Porosity effect
        porosity_val = self.porosity(T)
        porosity_factor = porosity_val / self.porosity_0
        
        diffusivity = D_0 * temp_factor * sat_factor * porosity_factor
        
        # Add noise
        noise = np.random.normal(0, diffusivity * 0.05)
        
        return max(1e-12, diffusivity + noise)
    
    def damage_evolution(self, T, stress_ratio=0.3, time_hours=1):
        """
        Generate damage parameter D(T, σ, t) [-]
        0 = undamaged, 1 = fully damaged
        """
        # Temperature-induced damage
        if T <= 200:
            temp_damage = 0.0
        elif T <= 400:
            temp_damage = 0.1 * (T - 200) / 200
        elif T <= 600:
            temp_damage = 0.1 + 0.3 * (T - 400) / 200
        else:
            temp_damage = 0.4 + 0.4 * min((T - 600) / 200, 1)
        
        # Stress-induced damage
        stress_damage = stress_ratio ** 2 * 0.3
        
        # Time evolution (logarithmic)
        time_factor = min(1.0, np.log10(1 + time_hours) / 2)
        
        # Combined damage
        damage = (temp_damage + stress_damage) * time_factor
        
        # Rubber mitigation (reduces damage slightly)
        rubber_mitigation = 1 - (self.rubber_content / 100) * 0.1
        damage *= rubber_mitigation
        
        # Add noise
        noise = np.random.normal(0, 0.02)
        
        return max(0, min(1, damage + noise))
    
    def generate_dataset(self, damage_levels=[0, 0.1, 0.3, 0.5]):
        """
        Generate complete poro-mechanical dataset
        """
        datasets = {}
        
        for damage in damage_levels:
            data = {
                'temperature_C': [],
                'porosity': [],
                'permeability_m2': [],
                'saturation_degree': [],
                'pore_pressure_MPa': [],
                'moisture_diffusivity_m2_s': [],
                'damage_parameter': [],
                'rubber_content_%': []
            }
            
            for T in self.temperature_range:
                phi = self.porosity(T, damage)
                k = self.permeability(T, damage, phi)
                S = self.saturation_degree(T, time_hours=1)
                p = self.pore_pressure(T, S)
                D_m = self.moisture_diffusivity(T, S)
                
                data['temperature_C'].append(T)
                data['porosity'].append(phi)
                data['permeability_m2'].append(k)
                data['saturation_degree'].append(S)
                data['pore_pressure_MPa'].append(p)
                data['moisture_diffusivity_m2_s'].append(D_m)
                data['damage_parameter'].append(damage)
                data['rubber_content_%'].append(self.rubber_content)
            
            datasets[f'damage_{damage}'] = pd.DataFrame(data)
        
        return datasets
    
    def generate_permeability_test_data(self):
        """
        Generate simulated permeability test data
        Using gas permeability and water permeability methods
        """
        test_data = []
        
        test_temps = [20, 105, 200, 300, 400, 500, 600]
        
        for T in test_temps:
            for test_num in range(1, 4):
                # Gas permeability test
                k_gas = self.permeability(T, damage=0.05 * T / 100)
                
                # Water permeability (typically lower than gas)
                k_water = k_gas * 0.1
                
                record = {
                    'test_id': f'PERM_{T}C_{test_num}',
                    'temperature_C': T,
                    'conditioning': 'Dried at 105°C for 24h' if T > 20 else 'As-cast',
                    'gas_permeability_m2': k_gas * (1 + np.random.normal(0, 0.05)),
                    'water_permeability_m2': k_water * (1 + np.random.normal(0, 0.08)),
                    'test_method_gas': 'Cembureau method',
                    'test_method_water': 'Constant head',
                    'pressure_gradient_MPa_m': np.random.uniform(0.1, 0.5),
                    'flow_rate_ml_min': np.random.uniform(0.1, 10),
                    'specimen_thickness_mm': 50,
                    'specimen_diameter_mm': 100,
                    'rubber_content_%': self.rubber_content
                }
                
                test_data.append(record)
        
        return pd.DataFrame(test_data)
    
    def generate_moisture_transport_data(self, exposure_conditions=['sealed', 'RH50', 'RH95']):
        """
        Generate moisture transport test data
        """
        transport_data = []
        
        for condition in exposure_conditions:
            for T in [20, 50, 80, 105, 150, 200]:
                # Set boundary conditions
                if condition == 'sealed':
                    RH_surface = 100
                    moisture_loss_rate = 0
                elif condition == 'RH50':
                    RH_surface = 50
                    moisture_loss_rate = 0.001 * np.exp(T / 100)  # kg/m²·h
                else:  # RH95
                    RH_surface = 95
                    moisture_loss_rate = 0.0001 * np.exp(T / 150)
                
                # Generate moisture profiles at different depths
                depths_mm = [0, 10, 20, 30, 40, 50]
                for depth in depths_mm:
                    # Moisture content decreases with depth from surface
                    moisture_factor = np.exp(-depth / 20)
                    S = self.saturation_degree(T, time_hours=2) * moisture_factor
                    
                    record = {
                        'temperature_C': T,
                        'exposure_condition': condition,
                        'surface_RH_%': RH_surface,
                        'depth_mm': depth,
                        'saturation_degree': S,
                        'moisture_content_kg_m3': S * 200,  # Assuming 200 kg/m³ max water content
                        'moisture_loss_rate_kg_m2h': moisture_loss_rate * (1 - depth / 100),
                        'test_duration_hours': 24,
                        'rubber_content_%': self.rubber_content
                    }
                    
                    transport_data.append(record)
        
        return pd.DataFrame(transport_data)

def main():
    # Generate data for different configurations
    rubber_contents = [0, 5, 10, 15, 20, 25, 30]
    wc_ratios = [0.35, 0.40, 0.45]
    
    all_data = {}
    
    for wc in wc_ratios:
        for rubber_pct in rubber_contents:
            print(f"Generating poro-mechanical properties for w/c={wc}, {rubber_pct}% rubber...")
            
            generator = PoroMechanicalGenerator(
                rubber_content_percent=rubber_pct,
                w_c_ratio=wc
            )
            
            # Generate main dataset
            datasets = generator.generate_dataset()
            
            # Generate permeability test data
            perm_data = generator.generate_permeability_test_data()
            
            # Generate moisture transport data
            transport_data = generator.generate_moisture_transport_data()
            
            # Store data
            key = f'wc_{int(wc*100)}_rubber_{rubber_pct}pct'
            all_data[key] = {
                'datasets': datasets,
                'permeability': perm_data,
                'transport': transport_data
            }
            
            # Save CSV files
            for damage_name, df in datasets.items():
                filename = f'poro_mechanical_{key}_{damage_name}.csv'
                df.to_csv(filename, index=False)
            
            perm_filename = f'permeability_tests_{key}.csv'
            perm_data.to_csv(perm_filename, index=False)
            
            transport_filename = f'moisture_transport_{key}.csv'
            transport_data.to_csv(transport_filename, index=False)
    
    print(f"\nSaved {len(rubber_contents) * len(wc_ratios) * 6} poro-mechanical property files")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot for w/c = 0.40
    wc = 0.40
    
    # Porosity evolution with rubber content
    for rubber_pct in rubber_contents:
        key = f'wc_{int(wc*100)}_rubber_{rubber_pct}pct'
        if key in all_data:
            data_d0 = all_data[key]['datasets']['damage_0']
            axes[0, 0].plot(data_d0['temperature_C'],
                          data_d0['porosity'],
                          label=f'{rubber_pct}% rubber', alpha=0.7)
    
    # Permeability with temperature (15% rubber, different damage)
    key = f'wc_{int(wc*100)}_rubber_15pct'
    if key in all_data:
        for damage_name, df in all_data[key]['datasets'].items():
            damage_val = float(damage_name.split('_')[1])
            axes[0, 1].semilogy(df['temperature_C'],
                               df['permeability_m2'],
                               label=f'D = {damage_val}', marker='o', markersize=4)
    
    # Pore pressure evolution
    for rubber_pct in [0, 10, 20, 30]:
        key = f'wc_{int(wc*100)}_rubber_{rubber_pct}pct'
        if key in all_data:
            data_d0 = all_data[key]['datasets']['damage_0']
            axes[0, 2].plot(data_d0['temperature_C'],
                          data_d0['pore_pressure_MPa'],
                          label=f'{rubber_pct}% rubber', linewidth=2)
    
    # Saturation degree
    key = f'wc_{int(wc*100)}_rubber_15pct'
    if key in all_data:
        data_d0 = all_data[key]['datasets']['damage_0']
        axes[1, 0].plot(data_d0['temperature_C'],
                       data_d0['saturation_degree'],
                       'b-', linewidth=2, label='15% rubber')
        axes[1, 0].axhline(y=0.75, color='r', linestyle='--', label='Initial saturation')
    
    # Moisture diffusivity
    for rubber_pct in [0, 15, 30]:
        key = f'wc_{int(wc*100)}_rubber_{rubber_pct}pct'
        if key in all_data:
            data_d0 = all_data[key]['datasets']['damage_0']
            axes[1, 1].semilogy(data_d0['temperature_C'],
                               data_d0['moisture_diffusivity_m2_s'],
                               label=f'{rubber_pct}% rubber', marker='s', markersize=4)
    
    # Damage evolution comparison
    temperatures = [200, 400, 600]
    x_pos = np.arange(len(temperatures))
    width = 0.2
    
    for i, rubber_pct in enumerate([0, 15, 30]):
        key = f'wc_{int(wc*100)}_rubber_{rubber_pct}pct'
        if key in all_data:
            damage_values = []
            for T in temperatures:
                # Simulate damage after 2 hours at 30% stress
                gen = PoroMechanicalGenerator(rubber_pct, wc)
                D = gen.damage_evolution(T, stress_ratio=0.3, time_hours=2)
                damage_values.append(D)
            
            axes[1, 2].bar(x_pos + i * width, damage_values, width,
                         label=f'{rubber_pct}% rubber')
    
    axes[1, 2].set_xticks(x_pos + width)
    axes[1, 2].set_xticklabels([f'{T}°C' for T in temperatures])
    
    # Format all plots
    axes[0, 0].set_xlabel('Temperature (°C)')
    axes[0, 0].set_ylabel('Porosity (-)')
    axes[0, 0].set_title('Temperature-Dependent Porosity (w/c=0.40)')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Temperature (°C)')
    axes[0, 1].set_ylabel('Permeability (m²)')
    axes[0, 1].set_title('Permeability vs Damage (15% Rubber)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_xlabel('Temperature (°C)')
    axes[0, 2].set_ylabel('Pore Pressure (MPa)')
    axes[0, 2].set_title('Pore Pressure Buildup')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Temperature (°C)')
    axes[1, 0].set_ylabel('Saturation Degree (-)')
    axes[1, 0].set_title('Moisture Loss During Heating')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    axes[1, 1].set_xlabel('Temperature (°C)')
    axes[1, 1].set_ylabel('Moisture Diffusivity (m²/s)')
    axes[1, 1].set_title('Moisture Transport Properties')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].set_xlabel('Temperature')
    axes[1, 2].set_ylabel('Damage Parameter (-)')
    axes[1, 2].set_title('Damage Evolution (2 hours, 30% stress)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim([0, 1])
    
    plt.suptitle('Poro-Mechanical Properties of Rubberized Concrete', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('poro_mechanical_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate metadata
    metadata = {
        'generated_date': datetime.now().isoformat(),
        'temperature_range_C': [20, 800],
        'rubber_content_range_percent': [0, 30],
        'water_cement_ratios': wc_ratios,
        'properties': {
            'porosity': {
                'range': [0.08, 0.5],
                'units': 'fraction',
                'method': 'Mercury intrusion porosimetry'
            },
            'permeability': {
                'range': [1e-20, 1e-10],
                'units': 'm^2',
                'methods': ['Gas permeability (Cembureau)', 'Water permeability']
            },
            'pore_pressure': {
                'range': [0, 10],
                'units': 'MPa',
                'critical_value': '2-4 MPa for spalling'
            },
            'saturation': {
                'range': [0, 1],
                'units': 'fraction',
                'initial': 0.75
            }
        },
        'damage_model': 'Mazars damage model with temperature coupling',
        'moisture_transport': 'Fick\'s law with temperature-dependent diffusivity',
        'data_source': 'Synthetic data based on literature and theoretical models'
    }
    
    with open('poro_mechanical_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Poro-mechanical properties dataset generation complete!")

if __name__ == "__main__":
    main()