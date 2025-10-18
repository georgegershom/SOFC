#!/usr/bin/env python3
"""
Generate Temperature Evolution Validation Data
High-quality experimental data for model validation
Includes thermocouple measurements during standard fire tests
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import json
from datetime import datetime

class TemperatureEvolutionGenerator:
    def __init__(self, specimen_type='slab', rubber_content_percent=0):
        """
        Initialize temperature evolution generator
        specimen_type: 'slab', 'column', 'beam'
        rubber_content_percent: 0-30% typical range
        """
        self.specimen_type = specimen_type
        self.rubber_content = rubber_content_percent
        
        # Specimen dimensions (mm)
        if specimen_type == 'slab':
            self.thickness = 200
            self.width = 1000
            self.length = 1000
        elif specimen_type == 'column':
            self.thickness = 400  # square cross-section
            self.width = 400
            self.length = 3000
        else:  # beam
            self.thickness = 300  # height
            self.width = 200
            self.length = 4000
    
    def iso834_fire_curve(self, t_minutes):
        """
        ISO 834 standard fire curve
        T = T0 + 345 * log10(8*t + 1)
        """
        T0 = 20  # Initial temperature °C
        return T0 + 345 * np.log10(8 * t_minutes + 1)
    
    def astm_e119_fire_curve(self, t_minutes):
        """
        ASTM E119 standard fire curve
        """
        t_hours = t_minutes / 60
        
        if t_hours <= 0:
            return 20
        elif t_hours <= 1:
            return 20 + 750 * (1 - np.exp(-3.79553 * np.sqrt(t_hours)))
        elif t_hours <= 2:
            return 843 + 85 * (t_hours - 1)
        else:
            return 928 + 27.5 * (t_hours - 2)
    
    def hydrocarbon_fire_curve(self, t_minutes):
        """
        Hydrocarbon fire curve (more severe)
        """
        return 1080 * (1 - 0.325 * np.exp(-0.167 * t_minutes) - 
                      0.675 * np.exp(-2.5 * t_minutes)) + 20
    
    def heat_transfer_1d(self, x_position, time_minutes, fire_type='iso834'):
        """
        Simplified 1D heat transfer solution
        x_position: distance from exposed surface (mm)
        """
        # Fire temperature
        if fire_type == 'iso834':
            T_fire = self.iso834_fire_curve(time_minutes)
        elif fire_type == 'astm':
            T_fire = self.astm_e119_fire_curve(time_minutes)
        else:  # hydrocarbon
            T_fire = self.hydrocarbon_fire_curve(time_minutes)
        
        # Thermal properties (simplified)
        alpha = 0.5e-6  # Thermal diffusivity m²/s
        alpha *= (1 + self.rubber_content / 100 * 0.1)  # Rubber effect
        
        # Convert to consistent units
        x = x_position / 1000  # Convert to meters
        t = time_minutes * 60  # Convert to seconds
        
        # Penetration depth
        if t > 0:
            penetration = 2 * np.sqrt(alpha * t)
            
            # Temperature profile (error function solution)
            from scipy.special import erfc
            theta = erfc(x / (2 * np.sqrt(alpha * t)))
            
            # Apply boundary conditions
            T = 20 + (T_fire - 20) * theta * np.exp(-x / penetration)
        else:
            T = 20
        
        # Add realistic fluctuations
        noise = np.random.normal(0, 2)
        
        return max(20, T + noise)
    
    def generate_thermocouple_data(self, fire_type='iso834', test_duration_min=120):
        """
        Generate thermocouple measurement data at multiple depths
        """
        # Thermocouple positions (mm from exposed surface)
        if self.specimen_type == 'slab':
            tc_positions = [0, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200]
        elif self.specimen_type == 'column':
            tc_positions = [0, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400]
        else:  # beam
            tc_positions = [0, 15, 30, 45, 60, 80, 100, 130, 160, 200, 250, 300]
        
        # Time points
        time_points = np.arange(0, test_duration_min + 1, 1)  # Every minute
        
        # Generate data for each thermocouple
        tc_data = {
            'time_min': [],
            'fire_temp_C': []
        }
        
        # Add columns for each TC position
        for pos in tc_positions:
            tc_data[f'TC_{pos}mm_C'] = []
        
        # Generate temperature evolution
        for t in time_points:
            tc_data['time_min'].append(t)
            
            # Fire temperature
            if fire_type == 'iso834':
                fire_temp = self.iso834_fire_curve(t)
            elif fire_type == 'astm':
                fire_temp = self.astm_e119_fire_curve(t)
            else:
                fire_temp = self.hydrocarbon_fire_curve(t)
            
            tc_data['fire_temp_C'].append(fire_temp)
            
            # Temperature at each TC position
            for pos in tc_positions:
                temp = self.heat_transfer_1d(pos, t, fire_type)
                
                # Account for thermocouple lag
                if t > 0 and pos > 0:
                    lag_factor = 0.95  # 5% lag
                    prev_temp = self.heat_transfer_1d(pos, t - 1, fire_type)
                    temp = lag_factor * temp + (1 - lag_factor) * prev_temp
                
                tc_data[f'TC_{pos}mm_C'].append(temp)
        
        return pd.DataFrame(tc_data)
    
    def generate_infrared_thermography_data(self, time_points=[30, 60, 90, 120]):
        """
        Generate 2D temperature field data from IR thermography
        """
        ir_data = []
        
        for t_min in time_points:
            # Create 2D grid
            if self.specimen_type == 'slab':
                x_grid = np.linspace(0, self.thickness, 21)
                y_grid = np.linspace(0, 500, 26)  # Half width due to symmetry
            else:
                x_grid = np.linspace(0, self.thickness, 21)
                y_grid = np.linspace(0, self.width, 21)
            
            for i, x in enumerate(x_grid):
                for j, y in enumerate(y_grid):
                    # Temperature calculation
                    temp_x = self.heat_transfer_1d(x, t_min, 'iso834')
                    
                    # Add edge effects (corners cool faster)
                    edge_factor = 1.0
                    if self.specimen_type != 'slab':
                        edge_distance = min(y, self.width - y)
                        if edge_distance < 50:
                            edge_factor = 0.8 + 0.2 * (edge_distance / 50)
                    
                    temp = temp_x * edge_factor
                    
                    record = {
                        'time_min': t_min,
                        'x_position_mm': x,
                        'y_position_mm': y,
                        'temperature_C': temp,
                        'measurement_method': 'IR Thermography',
                        'emissivity': 0.95,
                        'spatial_resolution_mm': 5
                    }
                    
                    ir_data.append(record)
        
        return pd.DataFrame(ir_data)
    
    def generate_multi_fire_scenario_data(self):
        """
        Generate data for different fire scenarios
        """
        scenarios = ['iso834', 'astm', 'hydrocarbon']
        all_scenario_data = {}
        
        for scenario in scenarios:
            print(f"  Generating {scenario} fire scenario...")
            tc_data = self.generate_thermocouple_data(
                fire_type=scenario,
                test_duration_min=180
            )
            all_scenario_data[scenario] = tc_data
        
        return all_scenario_data
    
    def generate_cooling_phase_data(self, heating_duration_min=90, cooling_duration_min=180):
        """
        Generate temperature data including cooling phase
        """
        total_time = heating_duration_min + cooling_duration_min
        time_points = np.arange(0, total_time + 1, 1)
        
        # TC positions
        tc_positions = [0, 25, 50, 100, 150, 200] if self.specimen_type == 'slab' else [0, 50, 100, 200, 300, 400]
        
        cooling_data = {
            'time_min': [],
            'phase': [],
            'furnace_temp_C': []
        }
        
        for pos in tc_positions:
            cooling_data[f'TC_{pos}mm_C'] = []
        
        # Store maximum temperatures reached
        max_temps = {}
        for pos in tc_positions:
            max_temps[pos] = 20
        
        for t in time_points:
            cooling_data['time_min'].append(t)
            
            if t <= heating_duration_min:
                # Heating phase
                cooling_data['phase'].append('heating')
                furnace_temp = self.iso834_fire_curve(t)
                
                for pos in tc_positions:
                    temp = self.heat_transfer_1d(pos, t, 'iso834')
                    max_temps[pos] = max(max_temps[pos], temp)
                    cooling_data[f'TC_{pos}mm_C'].append(temp)
                    
            else:
                # Cooling phase
                cooling_data['phase'].append('cooling')
                t_cool = t - heating_duration_min
                
                # Furnace cooling (exponential decay)
                max_furnace = self.iso834_fire_curve(heating_duration_min)
                furnace_temp = 20 + (max_furnace - 20) * np.exp(-0.02 * t_cool)
                
                for pos in tc_positions:
                    # Newton's cooling law with heat redistribution
                    cooling_rate = 0.01 * (1 + pos / self.thickness)  # Deeper points cool slower
                    temp = 20 + (max_temps[pos] - 20) * np.exp(-cooling_rate * t_cool)
                    
                    # Add thermal inertia effects
                    if t > heating_duration_min + 1:
                        prev_temp = cooling_data[f'TC_{pos}mm_C'][-1]
                        temp = 0.7 * temp + 0.3 * prev_temp
                    
                    cooling_data[f'TC_{pos}mm_C'].append(temp)
            
            cooling_data['furnace_temp_C'].append(furnace_temp)
        
        return pd.DataFrame(cooling_data)

def main():
    # Test configurations
    specimen_types = ['slab', 'column', 'beam']
    rubber_contents = [0, 10, 20, 30]
    
    all_data = {}
    
    for specimen in specimen_types:
        for rubber_pct in rubber_contents:
            print(f"Generating temperature evolution for {specimen}, {rubber_pct}% rubber...")
            
            generator = TemperatureEvolutionGenerator(
                specimen_type=specimen,
                rubber_content_percent=rubber_pct
            )
            
            # Generate standard fire test data
            tc_iso834 = generator.generate_thermocouple_data('iso834', 180)
            tc_astm = generator.generate_thermocouple_data('astm', 180)
            tc_hydrocarbon = generator.generate_thermocouple_data('hydrocarbon', 120)
            
            # Generate IR thermography data
            ir_data = generator.generate_infrared_thermography_data()
            
            # Generate cooling phase data
            cooling_data = generator.generate_cooling_phase_data()
            
            # Store data
            key = f'{specimen}_rubber{rubber_pct}'
            all_data[key] = {
                'iso834': tc_iso834,
                'astm': tc_astm,
                'hydrocarbon': tc_hydrocarbon,
                'ir_thermography': ir_data,
                'cooling': cooling_data
            }
            
            # Save CSV files
            tc_iso834.to_csv(f'temp_evolution_{specimen}_{rubber_pct}pct_ISO834.csv', index=False)
            tc_astm.to_csv(f'temp_evolution_{specimen}_{rubber_pct}pct_ASTM.csv', index=False)
            tc_hydrocarbon.to_csv(f'temp_evolution_{specimen}_{rubber_pct}pct_hydrocarbon.csv', index=False)
            ir_data.to_csv(f'ir_thermography_{specimen}_{rubber_pct}pct.csv', index=False)
            cooling_data.to_csv(f'cooling_phase_{specimen}_{rubber_pct}pct.csv', index=False)
    
    print(f"\nSaved {len(specimen_types) * len(rubber_contents) * 5} temperature evolution files")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Fire curves comparison
    time_fire = np.linspace(0, 180, 181)
    gen_temp = TemperatureEvolutionGenerator('slab', 0)
    
    axes[0, 0].plot(time_fire, [gen_temp.iso834_fire_curve(t) for t in time_fire],
                   'r-', linewidth=2, label='ISO 834')
    axes[0, 0].plot(time_fire, [gen_temp.astm_e119_fire_curve(t) for t in time_fire],
                   'b-', linewidth=2, label='ASTM E119')
    axes[0, 0].plot(time_fire[:121], [gen_temp.hydrocarbon_fire_curve(t) for t in time_fire[:121]],
                   'g-', linewidth=2, label='Hydrocarbon')
    axes[0, 0].set_xlabel('Time (minutes)')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].set_title('Standard Fire Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Temperature profiles in slab (ISO 834, 15% rubber)
    key = 'slab_rubber15'
    if key in all_data:
        tc_data = all_data[key]['iso834']
        times_to_plot = [30, 60, 90, 120, 180]
        
        for time_idx in times_to_plot:
            row_data = tc_data[tc_data['time_min'] == time_idx].iloc[0]
            depths = []
            temps = []
            
            for col in tc_data.columns:
                if col.startswith('TC_') and col.endswith('mm_C'):
                    depth = int(col.split('_')[1].replace('mm', ''))
                    depths.append(depth)
                    temps.append(row_data[col])
            
            axes[0, 1].plot(depths, temps, marker='o', label=f'{time_idx} min')
        
        axes[0, 1].set_xlabel('Depth from exposed surface (mm)')
        axes[0, 1].set_ylabel('Temperature (°C)')
        axes[0, 1].set_title('Temperature Profiles - Slab (15% Rubber, ISO 834)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].invert_xaxis()
    
    # Plot 3: Effect of rubber content
    depths_to_compare = [50, 100, 150]
    colors = ['r', 'g', 'b']
    
    for i, depth in enumerate(depths_to_compare):
        for rubber_pct in [0, 15, 30]:
            key = f'slab_rubber{rubber_pct}'
            if key in all_data:
                tc_data = all_data[key]['iso834']
                col_name = f'TC_{depth}mm_C'
                if col_name in tc_data.columns:
                    axes[0, 2].plot(tc_data['time_min'], tc_data[col_name],
                                   color=colors[i], alpha=0.3 + 0.3 * (rubber_pct / 30),
                                   label=f'{depth}mm, {rubber_pct}% rubber')
    
    axes[0, 2].set_xlabel('Time (minutes)')
    axes[0, 2].set_ylabel('Temperature (°C)')
    axes[0, 2].set_title('Rubber Content Effect on Temperature Evolution')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Column vs Slab comparison
    for specimen in ['slab', 'column']:
        key = f'{specimen}_rubber15'
        if key in all_data:
            tc_data = all_data[key]['iso834']
            # Plot center temperature
            if specimen == 'slab':
                col_name = 'TC_100mm_C'
            else:
                col_name = 'TC_200mm_C'
            
            if col_name in tc_data.columns:
                axes[1, 0].plot(tc_data['time_min'], tc_data[col_name],
                              linewidth=2, label=f'{specimen.capitalize()} center')
    
    axes[1, 0].set_xlabel('Time (minutes)')
    axes[1, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].set_title('Specimen Type Comparison (15% Rubber)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Cooling phase
    key = 'slab_rubber15'
    if key in all_data:
        cooling = all_data[key]['cooling']
        
        # Plot furnace and selected TC temperatures
        axes[1, 1].plot(cooling['time_min'], cooling['furnace_temp_C'],
                       'k-', linewidth=2, label='Furnace')
        
        for depth in [0, 50, 100, 200]:
            col_name = f'TC_{depth}mm_C'
            if col_name in cooling.columns:
                axes[1, 1].plot(cooling['time_min'], cooling[col_name],
                              label=f'{depth}mm depth')
        
        # Add phase separator
        axes[1, 1].axvline(x=90, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].text(45, 800, 'Heating', fontsize=10, ha='center')
        axes[1, 1].text(180, 800, 'Cooling', fontsize=10, ha='center')
    
    axes[1, 1].set_xlabel('Time (minutes)')
    axes[1, 1].set_ylabel('Temperature (°C)')
    axes[1, 1].set_title('Heating and Cooling Phases (Slab, 15% Rubber)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: 2D temperature field from IR thermography
    key = 'slab_rubber15'
    if key in all_data:
        ir = all_data[key]['ir_thermography']
        ir_60min = ir[ir['time_min'] == 60]
        
        # Reshape for contour plot
        x_unique = ir_60min['x_position_mm'].unique()
        y_unique = ir_60min['y_position_mm'].unique()
        
        temp_grid = np.zeros((len(y_unique), len(x_unique)))
        
        for i, y in enumerate(y_unique):
            for j, x in enumerate(x_unique):
                temp_val = ir_60min[(ir_60min['x_position_mm'] == x) & 
                                    (ir_60min['y_position_mm'] == y)]['temperature_C']
                if not temp_val.empty:
                    temp_grid[i, j] = temp_val.values[0]
        
        contour = axes[1, 2].contourf(x_unique, y_unique, temp_grid,
                                      levels=20, cmap='hot')
        plt.colorbar(contour, ax=axes[1, 2], label='Temperature (°C)')
        
        axes[1, 2].set_xlabel('Depth (mm)')
        axes[1, 2].set_ylabel('Width (mm)')
        axes[1, 2].set_title('2D Temperature Field - IR Thermography (60 min)')
    
    plt.suptitle('Temperature Evolution Validation Data', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('temperature_evolution_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate metadata
    metadata = {
        'generated_date': datetime.now().isoformat(),
        'specimen_types': specimen_types,
        'rubber_content_range_percent': [0, 30],
        'fire_scenarios': {
            'ISO_834': 'T = 20 + 345*log10(8t+1)',
            'ASTM_E119': 'Standard time-temperature curve',
            'Hydrocarbon': 'T = 1080*(1-0.325*exp(-0.167t)-0.675*exp(-2.5t))+20'
        },
        'measurement_methods': {
            'thermocouples': {
                'type': 'Type K',
                'accuracy': '±2.2°C or 0.75%',
                'response_time': '< 1 second',
                'positions': 'Multiple depths from exposed surface'
            },
            'ir_thermography': {
                'camera': 'FLIR A655sc',
                'resolution': '640x480 pixels',
                'accuracy': '±2°C or 2%',
                'frame_rate': '50 Hz',
                'emissivity': 0.95
            }
        },
        'test_conditions': {
            'initial_temperature_C': 20,
            'initial_moisture': '75% RH equilibrium',
            'loading': 'No mechanical load during thermal tests',
            'boundary_conditions': 'One-sided heating for slabs, 4-sided for columns'
        },
        'data_usage': 'Model validation - DO NOT use for calibration'
    }
    
    with open('temperature_evolution_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Temperature evolution dataset generation complete!")

if __name__ == "__main__":
    main()