#!/usr/bin/env python3
"""
Generate Temperature-Dependent Mechanical Properties for Rubberized Concrete
Includes compressive strength, tensile strength, elastic modulus, and Poisson's ratio
Based on in-situ testing under elevated temperatures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline
import json
from datetime import datetime

class MechanicalPropertiesGenerator:
    def __init__(self, rubber_content_percent=0, fc_20=40):
        """
        Initialize mechanical properties generator
        rubber_content_percent: 0-30% typical range
        fc_20: Compressive strength at 20°C in MPa (default 40 MPa for high-performance concrete)
        """
        self.rubber_content = rubber_content_percent
        self.fc_20 = fc_20 * (1 - rubber_content_percent * 0.005)  # Rubber slightly reduces strength
        self.temperature_range = np.arange(20, 1001, 20)  # 20°C to 1000°C
        
    def compressive_strength(self, T):
        """
        Generate compressive strength fc(T) [MPa]
        Based on Eurocode 2 and modified for rubber content
        """
        # Reduction factors from Eurocode 2 (siliceous aggregates)
        if T <= 20:
            kc = 1.0
        elif T <= 100:
            kc = 1.0
        elif T <= 200:
            kc = 0.95 - 0.05 * (T - 100) / 100
        elif T <= 300:
            kc = 0.90 - 0.05 * (T - 200) / 100
        elif T <= 400:
            kc = 0.85 - 0.10 * (T - 300) / 100
        elif T <= 500:
            kc = 0.75 - 0.15 * (T - 400) / 100
        elif T <= 600:
            kc = 0.60 - 0.15 * (T - 500) / 100
        elif T <= 700:
            kc = 0.45 - 0.15 * (T - 600) / 100
        elif T <= 800:
            kc = 0.30 - 0.15 * (T - 700) / 100
        elif T <= 900:
            kc = 0.15 - 0.07 * (T - 800) / 100
        else:
            kc = 0.08 - 0.04 * min((T - 900) / 100, 1)
            
        # Rubber modification - rubber provides some insulation benefit
        rubber_factor = 1 + (self.rubber_content / 100) * 0.05 * np.exp(-T/400)
        
        # Add realistic scatter
        scatter = np.random.normal(0, 0.02 * self.fc_20)
        
        return max(0.5, self.fc_20 * kc * rubber_factor + scatter)
    
    def tensile_strength(self, T):
        """
        Generate tensile strength ft(T) [MPa]
        Typically 8-12% of compressive strength
        """
        fc = self.compressive_strength(T)
        
        # Base ratio decreases with temperature
        if T <= 100:
            ratio = 0.10
        elif T <= 300:
            ratio = 0.10 - 0.02 * (T - 100) / 200
        elif T <= 600:
            ratio = 0.08 - 0.03 * (T - 300) / 300
        else:
            ratio = 0.05 - 0.02 * min((T - 600) / 400, 1)
            
        # Rubber improves tensile behavior slightly
        rubber_factor = 1 + (self.rubber_content / 100) * 0.15
        
        # Add scatter
        scatter = np.random.normal(0, 0.1)
        
        return max(0.1, fc * ratio * rubber_factor + scatter)
    
    def elastic_modulus(self, T):
        """
        Generate elastic modulus E(T) [GPa]
        """
        # Base elastic modulus at room temperature
        E_20 = 3320 * np.sqrt(self.fc_20) + 6900  # ACI formula in MPa
        E_20 = E_20 / 1000  # Convert to GPa
        
        # Temperature reduction factors
        if T <= 20:
            kE = 1.0
        elif T <= 100:
            kE = 1.0 - 0.05 * (T - 20) / 80
        elif T <= 200:
            kE = 0.95 - 0.10 * (T - 100) / 100
        elif T <= 300:
            kE = 0.85 - 0.15 * (T - 200) / 100
        elif T <= 400:
            kE = 0.70 - 0.15 * (T - 300) / 100
        elif T <= 500:
            kE = 0.55 - 0.15 * (T - 400) / 100
        elif T <= 600:
            kE = 0.40 - 0.15 * (T - 500) / 100
        elif T <= 700:
            kE = 0.25 - 0.10 * (T - 600) / 100
        elif T <= 800:
            kE = 0.15 - 0.07 * (T - 700) / 100
        else:
            kE = 0.08 - 0.05 * min((T - 800) / 200, 1)
            
        # Rubber reduces stiffness
        rubber_factor = 1 - (self.rubber_content / 100) * 0.25
        
        # Add scatter
        scatter = np.random.normal(0, 0.5)
        
        return max(0.5, E_20 * kE * rubber_factor + scatter)
    
    def poissons_ratio(self, T):
        """
        Generate Poisson's ratio ν(T) [-]
        Generally increases with temperature due to microcracking
        """
        # Base Poisson's ratio
        if T <= 100:
            nu_base = 0.20
        elif T <= 400:
            nu_base = 0.20 + 0.05 * (T - 100) / 300
        elif T <= 700:
            nu_base = 0.25 + 0.05 * (T - 400) / 300
        else:
            nu_base = min(0.30 + 0.05 * (T - 700) / 300, 0.35)
            
        # Rubber increases Poisson's ratio
        rubber_factor = 1 + (self.rubber_content / 100) * 0.15
        
        # Add small scatter
        scatter = np.random.normal(0, 0.01)
        
        nu = nu_base * rubber_factor + scatter
        
        # Ensure physically realistic bounds
        return max(0.1, min(0.45, nu))
    
    def stress_strain_parameters(self, T):
        """
        Generate stress-strain curve parameters
        Returns: peak strain, ultimate strain, stress-strain curve type
        """
        # Peak strain increases with temperature
        if T <= 100:
            eps_peak = 0.0025
        elif T <= 400:
            eps_peak = 0.0025 + 0.0035 * (T - 100) / 300
        elif T <= 700:
            eps_peak = 0.006 + 0.009 * (T - 400) / 300
        else:
            eps_peak = 0.015 + 0.01 * min((T - 700) / 300, 1)
            
        # Ultimate strain
        eps_ult = eps_peak * (3 + 0.5 * T / 100)
        
        # Rubber increases ductility
        rubber_factor = 1 + (self.rubber_content / 100) * 0.3
        
        return {
            'peak_strain': eps_peak * rubber_factor,
            'ultimate_strain': eps_ult * rubber_factor,
            'curve_type': 'parabolic-rectangular' if T < 400 else 'linear-descending'
        }
    
    def generate_dataset(self, specimen_variations=5, loading_rates=[0.5, 2.0, 10.0]):
        """
        Generate complete dataset with variations for multiple specimens and loading rates
        """
        datasets = {}
        
        for rate in loading_rates:
            rate_data = {}
            
            for specimen in range(1, specimen_variations + 1):
                data = {
                    'temperature_C': [],
                    'compressive_strength_MPa': [],
                    'tensile_strength_MPa': [],
                    'elastic_modulus_GPa': [],
                    'poissons_ratio': [],
                    'peak_strain': [],
                    'ultimate_strain': [],
                    'loading_rate_MPa_s': [],
                    'specimen_id': []
                }
                
                # Add loading rate effect (higher rates = slightly higher strength)
                rate_factor = 1 + np.log10(rate) * 0.05
                
                for T in self.temperature_range:
                    fc = self.compressive_strength(T) * rate_factor
                    ft = self.tensile_strength(T) * rate_factor
                    E = self.elastic_modulus(T)
                    nu = self.poissons_ratio(T)
                    strain_params = self.stress_strain_parameters(T)
                    
                    data['temperature_C'].append(T)
                    data['compressive_strength_MPa'].append(fc)
                    data['tensile_strength_MPa'].append(ft)
                    data['elastic_modulus_GPa'].append(E)
                    data['poissons_ratio'].append(nu)
                    data['peak_strain'].append(strain_params['peak_strain'])
                    data['ultimate_strain'].append(strain_params['ultimate_strain'])
                    data['loading_rate_MPa_s'].append(rate)
                    data['specimen_id'].append(f'RC{self.rubber_content:02d}_{specimen:03d}')
                
                rate_data[f'specimen_{specimen}'] = pd.DataFrame(data)
            
            datasets[f'loading_rate_{rate}MPa_s'] = rate_data
        
        return datasets
    
    def generate_insitu_test_data(self, test_temps=[20, 200, 400, 600, 800]):
        """
        Generate simulated in-situ test data with detailed measurements
        """
        test_data = []
        
        for T in test_temps:
            # Multiple tests at each temperature
            for test_num in range(1, 4):
                fc = self.compressive_strength(T)
                ft = self.tensile_strength(T)
                E = self.elastic_modulus(T)
                nu = self.poissons_ratio(T)
                strain_params = self.stress_strain_parameters(T)
                
                test_record = {
                    'test_id': f'INSITU_{T}C_{test_num:02d}',
                    'temperature_C': T,
                    'heating_rate_C_min': np.random.uniform(2, 5),
                    'stabilization_time_min': np.random.uniform(30, 60),
                    'specimen_size_mm': '150x300 cylinder',
                    'compressive_strength_MPa': fc + np.random.normal(0, fc * 0.03),
                    'strength_COV_%': np.random.uniform(3, 7),
                    'tensile_strength_MPa': ft + np.random.normal(0, ft * 0.05),
                    'elastic_modulus_GPa': E + np.random.normal(0, E * 0.04),
                    'poissons_ratio': nu + np.random.normal(0, 0.01),
                    'peak_strain': strain_params['peak_strain'] + np.random.normal(0, 0.0001),
                    'failure_mode': self._get_failure_mode(T),
                    'test_duration_min': np.random.uniform(15, 45),
                    'rubber_content_%': self.rubber_content
                }
                
                test_data.append(test_record)
        
        return pd.DataFrame(test_data)
    
    def _get_failure_mode(self, T):
        """Determine failure mode based on temperature"""
        if T < 300:
            return 'Crushing'
        elif T < 500:
            return 'Crushing with minor spalling'
        elif T < 700:
            return 'Explosive spalling'
        else:
            return 'Gradual disintegration'
    
    def generate_stress_strain_curves(self, temperatures=[20, 200, 400, 600]):
        """
        Generate complete stress-strain curves for specific temperatures
        """
        curves = {}
        
        for T in temperatures:
            fc = self.compressive_strength(T)
            E = self.elastic_modulus(T) * 1000  # Convert to MPa
            strain_params = self.stress_strain_parameters(T)
            
            # Generate strain points
            strains = np.linspace(0, strain_params['ultimate_strain'], 200)
            stresses = []
            
            for eps in strains:
                if eps <= strain_params['peak_strain']:
                    # Ascending branch (parabolic)
                    n = E * strain_params['peak_strain'] / fc
                    stress = fc * (n * (eps / strain_params['peak_strain']) - 
                                  (eps / strain_params['peak_strain'])**2) / (n - 1)
                else:
                    # Descending branch
                    if strain_params['curve_type'] == 'parabolic-rectangular':
                        stress = fc
                    else:
                        # Linear descending
                        stress = fc * (1 - 0.85 * (eps - strain_params['peak_strain']) / 
                                     (strain_params['ultimate_strain'] - strain_params['peak_strain']))
                
                stresses.append(max(0, stress))
            
            curves[f'T_{T}C'] = pd.DataFrame({
                'strain': strains,
                'stress_MPa': stresses,
                'temperature_C': T
            })
        
        return curves

def main():
    # Generate datasets for different rubber contents and strength classes
    rubber_contents = [0, 5, 10, 15, 20, 25, 30]
    strength_classes = [30, 40, 50, 60]  # MPa at 20°C
    
    all_data = {}
    
    for rubber_pct in rubber_contents:
        for fc_class in strength_classes:
            print(f"Generating mechanical properties for {rubber_pct}% rubber, fc={fc_class} MPa...")
            
            generator = MechanicalPropertiesGenerator(
                rubber_content_percent=rubber_pct,
                fc_20=fc_class
            )
            
            # Generate main dataset
            datasets = generator.generate_dataset(specimen_variations=3)
            
            # Generate in-situ test data
            insitu_data = generator.generate_insitu_test_data()
            
            # Generate stress-strain curves
            ss_curves = generator.generate_stress_strain_curves()
            
            # Store all data
            key = f'rubber_{rubber_pct}pct_fc{fc_class}'
            all_data[key] = {
                'datasets': datasets,
                'insitu': insitu_data,
                'stress_strain': ss_curves
            }
            
            # Save CSV files
            for rate_name, rate_data in datasets.items():
                for specimen_name, df in rate_data.items():
                    filename = f'mechanical_props_{key}_{rate_name}_{specimen_name}.csv'
                    df.to_csv(filename, index=False)
            
            # Save in-situ data
            insitu_filename = f'insitu_tests_{key}.csv'
            insitu_data.to_csv(insitu_filename, index=False)
            
            # Save stress-strain curves
            for temp_name, curve_df in ss_curves.items():
                ss_filename = f'stress_strain_{key}_{temp_name}.csv'
                curve_df.to_csv(ss_filename, index=False)
    
    print(f"\nSaved {len(rubber_contents) * len(strength_classes) * 15} mechanical property files")
    
    # Create comprehensive summary plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot for fc=40 MPa class
    for rubber_pct in rubber_contents:
        key = f'rubber_{rubber_pct}pct_fc40'
        if key in all_data:
            specimen_data = all_data[key]['datasets']['loading_rate_2.0MPa_s']['specimen_1']
            
            # Compressive strength
            axes[0, 0].plot(specimen_data['temperature_C'],
                          specimen_data['compressive_strength_MPa'],
                          label=f'{rubber_pct}% rubber', marker='o', markersize=3)
            
            # Tensile strength
            axes[0, 1].plot(specimen_data['temperature_C'],
                          specimen_data['tensile_strength_MPa'],
                          label=f'{rubber_pct}% rubber', marker='s', markersize=3)
            
            # Elastic modulus
            axes[0, 2].plot(specimen_data['temperature_C'],
                          specimen_data['elastic_modulus_GPa'],
                          label=f'{rubber_pct}% rubber', marker='^', markersize=3)
            
            # Poisson's ratio
            axes[1, 0].plot(specimen_data['temperature_C'],
                          specimen_data['poissons_ratio'],
                          label=f'{rubber_pct}% rubber', marker='d', markersize=3)
            
            # Peak strain
            axes[1, 1].plot(specimen_data['temperature_C'],
                          specimen_data['peak_strain'] * 1000,  # Convert to millistrain
                          label=f'{rubber_pct}% rubber', marker='v', markersize=3)
    
    # Stress-strain curves at different temperatures
    ax_ss = axes[1, 2]
    key = 'rubber_15pct_fc40'  # Example for 15% rubber
    if key in all_data:
        for temp_name, curve_df in all_data[key]['stress_strain'].items():
            ax_ss.plot(curve_df['strain'] * 1000, curve_df['stress_MPa'],
                      label=f"T = {curve_df['temperature_C'].iloc[0]}°C", linewidth=2)
    
    # Format all plots
    axes[0, 0].set_xlabel('Temperature (°C)')
    axes[0, 0].set_ylabel('Compressive Strength (MPa)')
    axes[0, 0].set_title('Temperature-Dependent Compressive Strength')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Temperature (°C)')
    axes[0, 1].set_ylabel('Tensile Strength (MPa)')
    axes[0, 1].set_title('Temperature-Dependent Tensile Strength')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_xlabel('Temperature (°C)')
    axes[0, 2].set_ylabel('Elastic Modulus (GPa)')
    axes[0, 2].set_title('Temperature-Dependent Elastic Modulus')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Temperature (°C)')
    axes[1, 0].set_ylabel("Poisson's Ratio (-)")
    axes[1, 0].set_title("Temperature-Dependent Poisson's Ratio")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Temperature (°C)')
    axes[1, 1].set_ylabel('Peak Strain (millistrain)')
    axes[1, 1].set_title('Temperature-Dependent Peak Strain')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    ax_ss.set_xlabel('Strain (millistrain)')
    ax_ss.set_ylabel('Stress (MPa)')
    ax_ss.set_title('Stress-Strain Curves (15% Rubber, fc=40 MPa)')
    ax_ss.legend()
    ax_ss.grid(True, alpha=0.3)
    
    plt.suptitle('Mechanical Properties of Rubberized Concrete at Elevated Temperatures', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('mechanical_properties_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate metadata
    metadata = {
        'generated_date': datetime.now().isoformat(),
        'temperature_range_C': [20, 1000],
        'rubber_content_range_percent': [0, 30],
        'strength_classes_MPa': strength_classes,
        'loading_rates_MPa_s': [0.5, 2.0, 10.0],
        'test_standards': [
            'ASTM C39 - Compressive Strength',
            'ASTM C469 - Elastic Modulus',
            'ASTM C496 - Tensile Strength',
            'RILEM TC 200-HTC - High Temperature Testing'
        ],
        'specimen_geometry': {
            'type': 'Cylinder',
            'diameter_mm': 150,
            'height_mm': 300
        },
        'heating_protocol': {
            'rate_C_min': 3,
            'stabilization_time_min': 45
        }
    }
    
    with open('mechanical_properties_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Mechanical properties dataset generation complete!")

if __name__ == "__main__":
    main()