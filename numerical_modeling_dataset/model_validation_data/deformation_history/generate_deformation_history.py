#!/usr/bin/env python3
"""
Generate Deformation/Strain History Validation Data
Critical data for validating thermo-mechanical coupled behavior
Includes strain measurements under combined thermal and mechanical loading
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import json
from datetime import datetime

class DeformationHistoryGenerator:
    def __init__(self, specimen_type='cylinder', rubber_content_percent=0):
        """
        Initialize deformation history generator
        specimen_type: 'cylinder', 'prism', 'cube'
        rubber_content_percent: 0-30% typical range
        """
        self.specimen_type = specimen_type
        self.rubber_content = rubber_content_percent
        
        # Specimen dimensions (mm)
        if specimen_type == 'cylinder':
            self.diameter = 150
            self.height = 300
        elif specimen_type == 'prism':
            self.width = 100
            self.depth = 100
            self.height = 400
        else:  # cube
            self.width = 150
            self.depth = 150
            self.height = 150
        
        # Material properties at room temperature
        self.fc_20 = 40 * (1 - rubber_content_percent * 0.005)  # MPa
        self.E_20 = 30 * (1 - rubber_content_percent * 0.02)  # GPa
    
    def temperature_history(self, time_min, heating_rate=5):
        """
        Generate temperature history for the specimen
        """
        if time_min <= 0:
            return 20
        
        # Linear heating up to target temperature
        T = 20 + heating_rate * time_min
        
        # Cap at maximum test temperature
        return min(T, 800)
    
    def mechanical_strain(self, stress_MPa, temperature):
        """
        Calculate mechanical strain from applied stress
        """
        # Temperature-dependent elastic modulus
        if temperature <= 100:
            E_T = self.E_20
        elif temperature <= 400:
            E_T = self.E_20 * (1 - 0.5 * (temperature - 100) / 300)
        elif temperature <= 700:
            E_T = self.E_20 * (0.5 - 0.35 * (temperature - 400) / 300)
        else:
            E_T = self.E_20 * 0.15
        
        # Elastic strain
        elastic_strain = stress_MPa / (E_T * 1000)  # Convert GPa to MPa
        
        # Add plastic strain at high stress/temperature
        stress_ratio = stress_MPa / self.fc_20
        if stress_ratio > 0.4 and temperature > 200:
            plastic_factor = (stress_ratio - 0.4) * (temperature / 500)
            plastic_strain = elastic_strain * plastic_factor * 2
        else:
            plastic_strain = 0
        
        return elastic_strain + plastic_strain
    
    def thermal_strain(self, temperature):
        """
        Calculate free thermal expansion strain
        """
        if temperature <= 20:
            return 0
        
        # CTE varies with temperature (×10^-6/°C)
        if temperature <= 100:
            cte_avg = 10
        elif temperature <= 400:
            cte_avg = 12
        elif temperature <= 700:
            cte_avg = 14
        else:
            cte_avg = 12
        
        # Rubber increases CTE
        cte_avg *= (1 + self.rubber_content / 100 * 0.3)
        
        return cte_avg * (temperature - 20) * 1e-6
    
    def transient_strain(self, temperature, stress_MPa, time_hours):
        """
        Calculate transient thermal strain (LITS)
        """
        if temperature <= 100 or stress_MPa <= 0:
            return 0
        
        # Transient strain coefficient
        k_tr = 2.5e-6  # per MPa per °C
        
        # Temperature factor
        temp_factor = min((temperature - 100) / 500, 1.0)
        
        # Stress factor
        stress_factor = stress_MPa / self.fc_20
        
        # Time factor (logarithmic)
        time_factor = np.log10(1 + time_hours * 60)
        
        # Rubber increases transient strain
        rubber_factor = 1 + self.rubber_content / 100 * 0.4
        
        return k_tr * temperature * stress_MPa * temp_factor * time_factor * rubber_factor
    
    def creep_strain(self, temperature, stress_MPa, time_hours):
        """
        Calculate creep strain
        """
        if stress_MPa <= 0:
            return 0
        
        # Basic creep coefficient
        phi_0 = 2.0  # Creep coefficient at 20°C
        
        # Temperature amplification
        if temperature <= 100:
            temp_amp = 1.0
        elif temperature <= 400:
            temp_amp = 1.0 + 3.0 * (temperature - 100) / 300
        else:
            temp_amp = 4.0 + 6.0 * min((temperature - 400) / 300, 1.0)
        
        # Stress dependency
        stress_ratio = stress_MPa / self.fc_20
        
        # Time function
        time_function = (time_hours / (time_hours + 1)) ** 0.3
        
        # Elastic strain for normalization
        elastic_strain = stress_MPa / (self.E_20 * 1000)
        
        return phi_0 * temp_amp * stress_ratio * time_function * elastic_strain
    
    def total_strain(self, time_min, stress_MPa, heating_rate=5):
        """
        Calculate total strain at given time
        """
        temperature = self.temperature_history(time_min, heating_rate)
        time_hours = time_min / 60
        
        # Strain components
        mech_strain = self.mechanical_strain(stress_MPa, temperature)
        therm_strain = self.thermal_strain(temperature)
        trans_strain = self.transient_strain(temperature, stress_MPa, time_hours)
        creep_str = self.creep_strain(temperature, stress_MPa, time_hours)
        
        # Total strain
        total = mech_strain + therm_strain + trans_strain + creep_str
        
        return {
            'total': total,
            'mechanical': mech_strain,
            'thermal': therm_strain,
            'transient': trans_strain,
            'creep': creep_str,
            'temperature': temperature
        }
    
    def generate_strain_history(self, load_level=0.3, heating_rate=5, duration_min=120):
        """
        Generate complete strain history under thermo-mechanical loading
        """
        # Applied stress
        stress_MPa = load_level * self.fc_20
        
        # Time points (higher resolution at beginning)
        time_points = np.concatenate([
            np.arange(0, 10, 0.5),
            np.arange(10, 30, 1),
            np.arange(30, duration_min + 1, 2)
        ])
        
        data = {
            'time_min': [],
            'temperature_C': [],
            'applied_stress_MPa': [],
            'total_strain': [],
            'mechanical_strain': [],
            'thermal_strain': [],
            'transient_strain': [],
            'creep_strain': [],
            'strain_rate_per_min': []
        }
        
        prev_total_strain = 0
        prev_time = 0
        
        for t in time_points:
            strain_data = self.total_strain(t, stress_MPa, heating_rate)
            
            # Calculate strain rate
            if t > 0:
                strain_rate = (strain_data['total'] - prev_total_strain) / (t - prev_time)
            else:
                strain_rate = 0
            
            # Add noise to simulate measurement uncertainty
            noise_factor = 1 + np.random.normal(0, 0.02)
            
            data['time_min'].append(t)
            data['temperature_C'].append(strain_data['temperature'])
            data['applied_stress_MPa'].append(stress_MPa)
            data['total_strain'].append(strain_data['total'] * noise_factor)
            data['mechanical_strain'].append(strain_data['mechanical'] * noise_factor)
            data['thermal_strain'].append(strain_data['thermal'] * noise_factor)
            data['transient_strain'].append(strain_data['transient'] * noise_factor)
            data['creep_strain'].append(strain_data['creep'] * noise_factor)
            data['strain_rate_per_min'].append(strain_rate * 1e6)  # Convert to microstrain/min
            
            prev_total_strain = strain_data['total']
            prev_time = t
        
        return pd.DataFrame(data)
    
    def generate_multi_load_test(self, load_levels=[0.1, 0.2, 0.3, 0.4, 0.5]):
        """
        Generate strain data for multiple load levels
        """
        multi_load_data = {}
        
        for load in load_levels:
            print(f"    Load level: {load*100:.0f}% fc")
            strain_history = self.generate_strain_history(
                load_level=load,
                heating_rate=5,
                duration_min=120
            )
            multi_load_data[f'load_{int(load*100)}pct'] = strain_history
        
        return multi_load_data
    
    def generate_lvdt_data(self, load_level=0.3, duration_min=120):
        """
        Generate LVDT displacement measurements
        """
        strain_history = self.generate_strain_history(load_level, duration_min=duration_min)
        
        # Multiple LVDT positions
        lvdt_data = {
            'time_min': strain_history['time_min'].values,
            'temperature_C': strain_history['temperature_C'].values
        }
        
        # Calculate displacements for different gauge lengths
        gauge_lengths = {
            'LVDT_1': 100,  # mm
            'LVDT_2': 150,
            'LVDT_3': 200,
            'LVDT_axial_full': self.height if hasattr(self, 'height') else 300
        }
        
        for lvdt_name, gauge_length in gauge_lengths.items():
            # Convert strain to displacement
            displacement = strain_history['total_strain'].values * gauge_length
            
            # Add LVDT-specific noise and drift
            drift = np.cumsum(np.random.normal(0, 0.001, len(displacement)))
            noise = np.random.normal(0, 0.01, len(displacement))
            
            lvdt_data[f'{lvdt_name}_mm'] = displacement + drift + noise
            lvdt_data[f'{lvdt_name}_gauge_mm'] = gauge_length
        
        # Add lateral LVDTs (for Poisson's ratio measurement)
        if self.specimen_type == 'cylinder':
            lateral_gauge = self.diameter
        else:
            lateral_gauge = self.width
        
        # Lateral strain (negative due to Poisson effect)
        poisson_ratio = 0.2 + strain_history['temperature_C'].values / 2000  # Increases with temperature
        lateral_strain = -strain_history['total_strain'].values * poisson_ratio
        lateral_displacement = lateral_strain * lateral_gauge
        
        lvdt_data['LVDT_lateral_mm'] = lateral_displacement + np.random.normal(0, 0.01, len(lateral_displacement))
        lvdt_data['LVDT_lateral_gauge_mm'] = lateral_gauge
        
        return pd.DataFrame(lvdt_data)
    
    def generate_strain_gauge_data(self, load_level=0.3, duration_min=120):
        """
        Generate high-temperature strain gauge measurements
        """
        strain_history = self.generate_strain_history(load_level, duration_min=duration_min)
        
        sg_data = {
            'time_min': strain_history['time_min'].values,
            'temperature_C': strain_history['temperature_C'].values
        }
        
        # Multiple strain gauges at different positions
        positions = {
            'SG_1_mid': 0.5,  # Mid-height
            'SG_2_quarter': 0.25,  # Quarter height
            'SG_3_three_quarter': 0.75,  # Three-quarter height
            'SG_4_lateral': -0.2  # Lateral gauge (negative for compression)
        }
        
        for gauge_name, position_factor in positions.items():
            if 'lateral' in gauge_name:
                # Lateral strain
                poisson_ratio = 0.2 + strain_history['temperature_C'].values / 2000
                gauge_strain = -strain_history['total_strain'].values * poisson_ratio
            else:
                # Axial strain with position variation
                gauge_strain = strain_history['total_strain'].values * (0.9 + 0.2 * position_factor)
            
            # Temperature compensation error
            temp_error = (strain_history['temperature_C'].values - 20) * 1e-6 * np.random.uniform(-1, 1)
            
            # Gauge factor drift with temperature
            gauge_factor_drift = 1 + (strain_history['temperature_C'].values - 20) / 1000 * 0.02
            
            # Add measurement effects
            measured_strain = gauge_strain * gauge_factor_drift + temp_error
            
            # Add noise
            noise = np.random.normal(0, 5e-6, len(measured_strain))
            
            sg_data[f'{gauge_name}_strain'] = measured_strain + noise
            
            # Temperature limit check (most gauges fail above 500°C)
            temp_limit = 500 if 'high_temp' not in gauge_name else 800
            failure_idx = np.where(strain_history['temperature_C'].values > temp_limit)[0]
            if len(failure_idx) > 0:
                sg_data[f'{gauge_name}_strain'][failure_idx[0]:] = np.nan
        
        return pd.DataFrame(sg_data)
    
    def generate_cyclic_loading_data(self, max_load=0.4, num_cycles=5, temp_per_cycle=100):
        """
        Generate strain data under cyclic thermal-mechanical loading
        """
        cyclic_data = {
            'time_min': [],
            'cycle_number': [],
            'temperature_C': [],
            'applied_stress_MPa': [],
            'total_strain': [],
            'residual_strain': []
        }
        
        time_counter = 0
        residual_strain = 0
        
        for cycle in range(1, num_cycles + 1):
            target_temp = min(cycle * temp_per_cycle, 600)
            
            # Heating phase with load
            heating_time = target_temp / 5  # 5°C/min heating rate
            for t in np.linspace(0, heating_time, 20):
                temp = 20 + t * 5
                stress = max_load * self.fc_20 * (t / heating_time)  # Gradual loading
                
                strain_data = self.total_strain(time_counter + t, stress, heating_rate=5)
                
                cyclic_data['time_min'].append(time_counter + t)
                cyclic_data['cycle_number'].append(cycle)
                cyclic_data['temperature_C'].append(temp)
                cyclic_data['applied_stress_MPa'].append(stress)
                cyclic_data['total_strain'].append(strain_data['total'] + residual_strain)
                cyclic_data['residual_strain'].append(residual_strain)
            
            # Hold at temperature
            hold_time = 30
            for t in np.linspace(0, hold_time, 10):
                strain_data = self.total_strain(
                    time_counter + heating_time + t,
                    max_load * self.fc_20,
                    heating_rate=0
                )
                
                cyclic_data['time_min'].append(time_counter + heating_time + t)
                cyclic_data['cycle_number'].append(cycle)
                cyclic_data['temperature_C'].append(target_temp)
                cyclic_data['applied_stress_MPa'].append(max_load * self.fc_20)
                cyclic_data['total_strain'].append(strain_data['total'] + residual_strain)
                cyclic_data['residual_strain'].append(residual_strain)
            
            # Unloading
            unload_time = 5
            for t in np.linspace(0, unload_time, 5):
                stress = max_load * self.fc_20 * (1 - t / unload_time)
                strain_data = self.total_strain(
                    time_counter + heating_time + hold_time + t,
                    stress,
                    heating_rate=0
                )
                
                cyclic_data['time_min'].append(time_counter + heating_time + hold_time + t)
                cyclic_data['cycle_number'].append(cycle)
                cyclic_data['temperature_C'].append(target_temp)
                cyclic_data['applied_stress_MPa'].append(stress)
                cyclic_data['total_strain'].append(strain_data['total'] + residual_strain)
                cyclic_data['residual_strain'].append(residual_strain)
            
            # Update residual strain (accumulation of plastic deformation)
            residual_strain += 0.0002 * cycle  # Incremental plastic strain
            
            # Cooling (simplified - keeping some residual temperature)
            time_counter += heating_time + hold_time + unload_time + 30
        
        return pd.DataFrame(cyclic_data)

def main():
    # Test configurations
    specimen_types = ['cylinder', 'prism', 'cube']
    rubber_contents = [0, 10, 20, 30]
    
    all_data = {}
    
    for specimen in specimen_types:
        for rubber_pct in rubber_contents:
            print(f"Generating deformation history for {specimen}, {rubber_pct}% rubber...")
            
            generator = DeformationHistoryGenerator(
                specimen_type=specimen,
                rubber_content_percent=rubber_pct
            )
            
            # Generate multi-load test data
            multi_load = generator.generate_multi_load_test()
            
            # Generate LVDT data
            lvdt_data = generator.generate_lvdt_data(load_level=0.3, duration_min=150)
            
            # Generate strain gauge data
            sg_data = generator.generate_strain_gauge_data(load_level=0.3, duration_min=150)
            
            # Generate cyclic loading data
            cyclic_data = generator.generate_cyclic_loading_data()
            
            # Store data
            key = f'{specimen}_{rubber_pct}pct'
            all_data[key] = {
                'multi_load': multi_load,
                'lvdt': lvdt_data,
                'strain_gauge': sg_data,
                'cyclic': cyclic_data
            }
            
            # Save CSV files
            for load_name, df in multi_load.items():
                filename = f'strain_history_{specimen}_{rubber_pct}pct_{load_name}.csv'
                df.to_csv(filename, index=False)
            
            lvdt_data.to_csv(f'lvdt_measurements_{specimen}_{rubber_pct}pct.csv', index=False)
            sg_data.to_csv(f'strain_gauge_{specimen}_{rubber_pct}pct.csv', index=False)
            cyclic_data.to_csv(f'cyclic_loading_{specimen}_{rubber_pct}pct.csv', index=False)
    
    print(f"\nSaved {len(specimen_types) * len(rubber_contents) * 8} deformation history files")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Strain evolution for different load levels (cylinder, 15% rubber)
    key = 'cylinder_15pct'
    if key in all_data:
        for load_name, df in all_data[key]['multi_load'].items():
            load_pct = int(load_name.split('_')[1].replace('pct', ''))
            axes[0, 0].plot(df['time_min'], df['total_strain'] * 1000,
                          label=f'{load_pct}% fc', alpha=0.7)
        
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('Total Strain (millistrain)')
        axes[0, 0].set_title('Strain Evolution - Different Load Levels (Cylinder, 15% Rubber)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Strain components breakdown (30% load)
    key = 'cylinder_15pct'
    if key in all_data:
        df = all_data[key]['multi_load']['load_30pct']
        
        axes[0, 1].plot(df['time_min'], df['mechanical_strain'] * 1000, label='Mechanical')
        axes[0, 1].plot(df['time_min'], df['thermal_strain'] * 1000, label='Thermal')
        axes[0, 1].plot(df['time_min'], df['transient_strain'] * 1000, label='Transient')
        axes[0, 1].plot(df['time_min'], df['creep_strain'] * 1000, label='Creep')
        axes[0, 1].plot(df['time_min'], df['total_strain'] * 1000,
                       'k-', linewidth=2, label='Total')
        
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_ylabel('Strain (millistrain)')
        axes[0, 1].set_title('Strain Components (30% Load, Cylinder, 15% Rubber)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Temperature vs Strain for different rubber contents
    load_level = 'load_30pct'
    for rubber_pct in [0, 15, 30]:
        key = f'cylinder_{rubber_pct}pct'
        if key in all_data and load_level in all_data[key]['multi_load']:
            df = all_data[key]['multi_load'][load_level]
            axes[0, 2].plot(df['temperature_C'], df['total_strain'] * 1000,
                          label=f'{rubber_pct}% rubber', linewidth=2)
    
    axes[0, 2].set_xlabel('Temperature (°C)')
    axes[0, 2].set_ylabel('Total Strain (millistrain)')
    axes[0, 2].set_title('Temperature-Strain Relationship (30% Load)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: LVDT measurements
    key = 'cylinder_15pct'
    if key in all_data:
        lvdt = all_data[key]['lvdt']
        
        axes[1, 0].plot(lvdt['time_min'], lvdt['LVDT_1_mm'], label='LVDT 1 (100mm)')
        axes[1, 0].plot(lvdt['time_min'], lvdt['LVDT_2_mm'], label='LVDT 2 (150mm)')
        axes[1, 0].plot(lvdt['time_min'], lvdt['LVDT_axial_full_mm'], label='Full height')
        
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].set_ylabel('Displacement (mm)')
        axes[1, 0].set_title('LVDT Displacement Measurements (Cylinder, 15% Rubber)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Cyclic loading
    key = 'cylinder_15pct'
    if key in all_data:
        cyclic = all_data[key]['cyclic']
        
        axes[1, 1].plot(cyclic['time_min'], cyclic['total_strain'] * 1000, 'b-', label='Total strain')
        axes[1, 1].plot(cyclic['time_min'], cyclic['residual_strain'] * 1000,
                       'r--', label='Residual strain')
        
        # Add cycle markers
        for cycle in cyclic['cycle_number'].unique():
            cycle_data = cyclic[cyclic['cycle_number'] == cycle]
            if not cycle_data.empty:
                axes[1, 1].axvline(x=cycle_data['time_min'].iloc[0],
                                 color='gray', linestyle=':', alpha=0.3)
        
        axes[1, 1].set_xlabel('Time (minutes)')
        axes[1, 1].set_ylabel('Strain (millistrain)')
        axes[1, 1].set_title('Cyclic Thermal-Mechanical Loading (Cylinder, 15% Rubber)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Strain rate evolution
    key = 'cylinder_15pct'
    if key in all_data:
        df = all_data[key]['multi_load']['load_30pct']
        
        # Secondary y-axis for temperature
        ax2 = axes[1, 2].twinx()
        
        axes[1, 2].plot(df['time_min'], df['strain_rate_per_min'],
                       'b-', label='Strain rate')
        ax2.plot(df['time_min'], df['temperature_C'],
                'r--', label='Temperature')
        
        axes[1, 2].set_xlabel('Time (minutes)')
        axes[1, 2].set_ylabel('Strain Rate (μstrain/min)', color='b')
        ax2.set_ylabel('Temperature (°C)', color='r')
        axes[1, 2].set_title('Strain Rate Evolution (30% Load, Cylinder, 15% Rubber)')
        axes[1, 2].tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Deformation History Validation Data', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('deformation_history_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate metadata
    metadata = {
        'generated_date': datetime.now().isoformat(),
        'specimen_types': specimen_types,
        'rubber_content_range_percent': [0, 30],
        'test_conditions': {
            'heating_rate_C_min': 5,
            'load_levels': [0.1, 0.2, 0.3, 0.4, 0.5],
            'max_temperature_C': 800,
            'loading_type': 'Constant compressive stress'
        },
        'measurement_systems': {
            'LVDT': {
                'type': 'High-temperature LVDT',
                'range_mm': '±25',
                'accuracy': '±0.01 mm',
                'max_temp_C': 1000,
                'gauge_lengths_mm': [100, 150, 200, 'full_height']
            },
            'strain_gauges': {
                'type': 'High-temperature resistance strain gauge',
                'gauge_factor': 2.1,
                'max_temp_C': 500,
                'accuracy': '±5 microstrain',
                'temperature_compensation': 'Self-compensating'
            },
            'data_acquisition': {
                'system': 'HBM QuantumX',
                'sampling_rate_Hz': 10,
                'resolution_bits': 24
            }
        },
        'strain_components': {
            'mechanical': 'Instantaneous elastic and plastic strain',
            'thermal': 'Free thermal expansion',
            'transient': 'Load-induced thermal strain (LITS)',
            'creep': 'Time-dependent deformation',
            'total': 'Sum of all components'
        },
        'data_usage': 'Model validation - Independent from calibration data'
    }
    
    with open('deformation_history_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Deformation history dataset generation complete!")

if __name__ == "__main__":
    main()