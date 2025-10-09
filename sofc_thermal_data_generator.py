#!/usr/bin/env python3
"""
SOFC Thermal History Data Generator
Generates comprehensive thermal data for Solid Oxide Fuel Cell operations
including sintering, thermal cycling, and steady-state operation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

class SOFCThermalDataGenerator:
    def __init__(self):
        """Initialize the SOFC thermal data generator with realistic parameters."""
        # SOFC dimensions (mm)
        self.cell_length = 100  # mm
        self.cell_width = 100   # mm
        self.cell_thickness = 0.5  # mm
        
        # Spatial resolution
        self.spatial_points_x = 50
        self.spatial_points_y = 50
        
        # Material properties and operating conditions
        self.materials = {
            'anode': {'max_temp': 1000, 'thermal_conductivity': 2.5},
            'cathode': {'max_temp': 1000, 'thermal_conductivity': 2.0},
            'electrolyte': {'max_temp': 1000, 'thermal_conductivity': 2.8},
            'interconnect': {'max_temp': 900, 'thermal_conductivity': 25.0}
        }
        
        # Create coordinate grids
        self.x = np.linspace(0, self.cell_length, self.spatial_points_x)
        self.y = np.linspace(0, self.cell_width, self.spatial_points_y)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
    def generate_sintering_data(self):
        """Generate thermal data for sintering and co-firing processes."""
        print("Generating sintering & co-firing thermal data...")
        
        # Sintering profile parameters
        sintering_stages = {
            'heating_1': {'start_temp': 25, 'end_temp': 600, 'duration': 4, 'rate': 2.4},
            'dwell_1': {'temp': 600, 'duration': 2},
            'heating_2': {'start_temp': 600, 'end_temp': 1200, 'duration': 6, 'rate': 1.67},
            'sintering_dwell': {'temp': 1200, 'duration': 4},
            'heating_3': {'start_temp': 1200, 'end_temp': 1450, 'duration': 2.5, 'rate': 1.67},
            'co_firing_dwell': {'temp': 1450, 'duration': 3},
            'cooling_1': {'start_temp': 1450, 'end_temp': 800, 'duration': 8, 'rate': -1.35},
            'cooling_2': {'start_temp': 800, 'end_temp': 25, 'duration': 12, 'rate': -1.08}
        }
        
        # Generate time series
        total_time = sum([stage['duration'] for stage in sintering_stages.values()])
        time_points = int(total_time * 60)  # 1 minute resolution
        time_array = np.linspace(0, total_time, time_points)
        
        # Generate temperature profile
        temp_profile = []
        current_time = 0
        
        for stage_name, params in sintering_stages.items():
            stage_duration = params['duration']
            stage_points = int(stage_duration * 60)
            
            if 'dwell' in stage_name:
                # Constant temperature with small fluctuations
                base_temp = params['temp']
                fluctuation = np.random.normal(0, 2, stage_points)
                stage_temps = base_temp + fluctuation
            else:
                # Linear heating/cooling with realistic variations
                start_temp = params['start_temp']
                end_temp = params['end_temp']
                linear_temps = np.linspace(start_temp, end_temp, stage_points)
                # Add realistic thermal lag and overshoot
                noise = np.random.normal(0, 3, stage_points)
                stage_temps = linear_temps + noise
            
            temp_profile.extend(stage_temps)
            current_time += stage_duration
        
        # Generate spatial temperature distribution
        spatial_data = []
        for i, temp in enumerate(temp_profile):
            # Create realistic spatial gradients
            center_temp = temp
            
            # Add radial temperature gradient (hotter in center)
            center_x, center_y = self.cell_length/2, self.cell_width/2
            distance_from_center = np.sqrt((self.X - center_x)**2 + (self.Y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            
            # Temperature drops towards edges
            temp_gradient = 15 * (distance_from_center / max_distance)
            spatial_temp = center_temp - temp_gradient
            
            # Add random hot spots and variations
            hot_spots = np.random.normal(0, 5, (self.spatial_points_y, self.spatial_points_x))
            spatial_temp += gaussian_filter(hot_spots, sigma=2)
            
            spatial_data.append({
                'time_hours': time_array[i],
                'center_temp': center_temp,
                'spatial_temp': spatial_temp.copy(),
                'min_temp': np.min(spatial_temp),
                'max_temp': np.max(spatial_temp),
                'temp_gradient': np.max(spatial_temp) - np.min(spatial_temp)
            })
        
        # Create DataFrame
        sintering_df = pd.DataFrame([
            {
                'time_hours': data['time_hours'],
                'center_temperature_C': data['center_temp'],
                'min_temperature_C': data['min_temp'],
                'max_temperature_C': data['max_temp'],
                'temperature_gradient_C': data['temp_gradient'],
                'process_stage': self._get_process_stage(data['time_hours'], sintering_stages)
            }
            for data in spatial_data
        ])
        
        return sintering_df, spatial_data, sintering_stages
    
    def generate_thermal_cycling_data(self, num_cycles=10):
        """Generate thermal cycling data for start-up and shut-down operations."""
        print(f"Generating thermal cycling data for {num_cycles} cycles...")
        
        cycling_data = []
        spatial_cycling_data = []
        
        for cycle in range(num_cycles):
            print(f"  Generating cycle {cycle + 1}/{num_cycles}")
            
            # Start-up phase (30 minutes)
            startup_time = np.linspace(0, 0.5, 30)  # 30 points over 0.5 hours
            startup_temps = self._generate_startup_profile(startup_time)
            
            # Steady operation (4 hours)
            steady_time = np.linspace(0.5, 4.5, 240)  # 240 points over 4 hours
            steady_temps = self._generate_steady_operation_profile(steady_time)
            
            # Shutdown phase (45 minutes)
            shutdown_time = np.linspace(4.5, 5.25, 45)  # 45 points over 0.75 hours
            shutdown_temps = self._generate_shutdown_profile(shutdown_time)
            
            # Combine phases
            cycle_time = np.concatenate([startup_time, steady_time, shutdown_time])
            cycle_temps = np.concatenate([startup_temps, steady_temps, shutdown_temps])
            
            # Add cycle offset
            cycle_time_offset = cycle_time + (cycle * 8)  # 8 hours between cycles
            
            # Generate spatial data for this cycle
            for i, (time_val, temp) in enumerate(zip(cycle_time_offset, cycle_temps)):
                spatial_temp = self._generate_spatial_distribution(temp, time_val)
                
                phase = self._determine_cycle_phase(time_val % 8)
                
                cycling_data.append({
                    'cycle_number': cycle + 1,
                    'time_hours': time_val,
                    'phase': phase,
                    'center_temperature_C': temp,
                    'min_temperature_C': np.min(spatial_temp),
                    'max_temperature_C': np.max(spatial_temp),
                    'temperature_gradient_C': np.max(spatial_temp) - np.min(spatial_temp),
                    'thermal_stress_indicator': self._calculate_thermal_stress(spatial_temp)
                })
                
                if i % 10 == 0:  # Store spatial data every 10th point to save memory
                    spatial_cycling_data.append({
                        'cycle': cycle + 1,
                        'time_hours': time_val,
                        'phase': phase,
                        'spatial_temp': spatial_temp.copy()
                    })
        
        cycling_df = pd.DataFrame(cycling_data)
        return cycling_df, spatial_cycling_data
    
    def generate_steady_state_data(self, duration_hours=24):
        """Generate steady-state operation data with realistic temperature gradients."""
        print(f"Generating steady-state operation data for {duration_hours} hours...")
        
        time_points = duration_hours * 60  # 1 minute resolution
        time_array = np.linspace(0, duration_hours, time_points)
        
        steady_data = []
        spatial_steady_data = []
        
        # Base operating temperature
        base_temp = 800  # °C
        
        for i, time_val in enumerate(time_array):
            # Add realistic variations
            daily_variation = 10 * np.sin(2 * np.pi * time_val / 24)  # Daily cycle
            load_variation = 5 * np.sin(2 * np.pi * time_val / 4)     # Load cycle
            random_noise = np.random.normal(0, 2)
            
            center_temp = base_temp + daily_variation + load_variation + random_noise
            
            # Generate spatial distribution
            spatial_temp = self._generate_spatial_distribution(center_temp, time_val, steady_state=True)
            
            steady_data.append({
                'time_hours': time_val,
                'center_temperature_C': center_temp,
                'min_temperature_C': np.min(spatial_temp),
                'max_temperature_C': np.max(spatial_temp),
                'temperature_gradient_C': np.max(spatial_temp) - np.min(spatial_temp),
                'current_density_A_cm2': 0.5 + 0.1 * np.sin(2 * np.pi * time_val / 4),
                'fuel_utilization': 0.85 + 0.05 * np.sin(2 * np.pi * time_val / 6),
                'thermal_stress_indicator': self._calculate_thermal_stress(spatial_temp)
            })
            
            # Store spatial data every hour
            if i % 60 == 0:
                spatial_steady_data.append({
                    'time_hours': time_val,
                    'spatial_temp': spatial_temp.copy()
                })
        
        steady_df = pd.DataFrame(steady_data)
        return steady_df, spatial_steady_data
    
    def _generate_startup_profile(self, time_array):
        """Generate realistic start-up temperature profile."""
        # Exponential approach to operating temperature
        target_temp = 800
        initial_temp = 25
        time_constant = 0.15  # hours
        
        temps = initial_temp + (target_temp - initial_temp) * (1 - np.exp(-time_array / time_constant))
        
        # Add realistic overshoot and settling
        overshoot = 20 * np.exp(-(time_array - 0.2)**2 / 0.01)
        temps += overshoot
        
        # Add noise
        noise = np.random.normal(0, 3, len(temps))
        return temps + noise
    
    def _generate_steady_operation_profile(self, time_array):
        """Generate steady operation temperature profile."""
        base_temp = 800
        # Small variations during steady operation
        variations = 5 * np.sin(2 * np.pi * (time_array - 0.5) / 2)
        noise = np.random.normal(0, 2, len(time_array))
        return base_temp + variations + noise
    
    def _generate_shutdown_profile(self, time_array):
        """Generate realistic shutdown temperature profile."""
        initial_temp = 800
        final_temp = 25
        
        # Exponential decay
        time_offset = time_array - time_array[0]
        time_constant = 0.2  # hours
        
        temps = final_temp + (initial_temp - final_temp) * np.exp(-time_offset / time_constant)
        
        # Add noise
        noise = np.random.normal(0, 2, len(temps))
        return temps + noise
    
    def _generate_spatial_distribution(self, center_temp, time_val, steady_state=False):
        """Generate realistic spatial temperature distribution."""
        # Base temperature field
        spatial_temp = np.full((self.spatial_points_y, self.spatial_points_x), center_temp)
        
        # Add fuel inlet/outlet gradient (left to right)
        fuel_gradient = np.linspace(-10, 15, self.spatial_points_x)
        spatial_temp += fuel_gradient[np.newaxis, :]
        
        # Add air inlet/outlet gradient (bottom to top)
        air_gradient = np.linspace(-5, 10, self.spatial_points_y)
        spatial_temp += air_gradient[:, np.newaxis]
        
        # Add hot spots near current collectors
        hot_spot_1 = 8 * np.exp(-((self.X - 25)**2 + (self.Y - 25)**2) / 200)
        hot_spot_2 = 6 * np.exp(-((self.X - 75)**2 + (self.Y - 75)**2) / 300)
        spatial_temp += hot_spot_1 + hot_spot_2
        
        # Add time-dependent variations
        if steady_state:
            time_variation = 3 * np.sin(2 * np.pi * time_val / 4)
        else:
            time_variation = 2 * np.sin(2 * np.pi * time_val)
        
        spatial_temp += time_variation
        
        # Add random variations
        random_field = np.random.normal(0, 2, (self.spatial_points_y, self.spatial_points_x))
        spatial_temp += gaussian_filter(random_field, sigma=1)
        
        return spatial_temp
    
    def _calculate_thermal_stress(self, spatial_temp):
        """Calculate thermal stress indicator based on temperature gradients."""
        # Calculate gradients
        grad_x = np.gradient(spatial_temp, axis=1)
        grad_y = np.gradient(spatial_temp, axis=0)
        
        # Thermal stress is proportional to temperature gradients
        thermal_stress = np.sqrt(grad_x**2 + grad_y**2)
        
        # Return maximum thermal stress as indicator
        return np.max(thermal_stress)
    
    def _get_process_stage(self, time_hours, stages):
        """Determine the current process stage based on time."""
        current_time = 0
        for stage_name, params in stages.items():
            if current_time <= time_hours <= current_time + params['duration']:
                return stage_name
            current_time += params['duration']
        return 'unknown'
    
    def _determine_cycle_phase(self, cycle_time):
        """Determine the phase within a thermal cycle."""
        if cycle_time <= 0.5:
            return 'startup'
        elif cycle_time <= 4.5:
            return 'steady_operation'
        elif cycle_time <= 5.25:
            return 'shutdown'
        else:
            return 'idle'
    
    def save_data(self, sintering_df, cycling_df, steady_df, spatial_data):
        """Save all generated data to files."""
        print("Saving thermal data to files...")
        
        # Create output directory
        os.makedirs('sofc_thermal_data', exist_ok=True)
        
        # Save CSV files
        sintering_df.to_csv('sofc_thermal_data/sintering_thermal_data.csv', index=False)
        cycling_df.to_csv('sofc_thermal_data/thermal_cycling_data.csv', index=False)
        steady_df.to_csv('sofc_thermal_data/steady_state_thermal_data.csv', index=False)
        
        # Save spatial data as numpy arrays
        np.savez_compressed('sofc_thermal_data/spatial_thermal_data.npz', **spatial_data)
        
        # Save metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'cell_dimensions': {
                'length_mm': self.cell_length,
                'width_mm': self.cell_width,
                'thickness_mm': self.cell_thickness
            },
            'spatial_resolution': {
                'points_x': self.spatial_points_x,
                'points_y': self.spatial_points_y
            },
            'data_description': {
                'sintering_data': 'Temperature profiles during sintering and co-firing process',
                'cycling_data': 'Thermal cycling data for start-up and shut-down operations',
                'steady_state_data': 'Long-term steady-state operation temperature data',
                'spatial_data': 'Spatial temperature distributions across cell surface'
            }
        }
        
        with open('sofc_thermal_data/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Data saved successfully!")
        print(f"  - Sintering data: {len(sintering_df)} time points")
        print(f"  - Cycling data: {len(cycling_df)} time points")
        print(f"  - Steady-state data: {len(steady_df)} time points")
        print(f"  - Spatial data arrays saved to compressed file")

def main():
    """Main function to generate all SOFC thermal data."""
    print("SOFC Thermal History Data Generator")
    print("=" * 50)
    
    generator = SOFCThermalDataGenerator()
    
    # Generate all thermal data
    sintering_df, sintering_spatial, sintering_stages = generator.generate_sintering_data()
    cycling_df, cycling_spatial = generator.generate_thermal_cycling_data(num_cycles=10)
    steady_df, steady_spatial = generator.generate_steady_state_data(duration_hours=48)
    
    # Combine spatial data
    spatial_data = {
        'sintering_spatial': sintering_spatial,
        'cycling_spatial': cycling_spatial,
        'steady_spatial': steady_spatial,
        'x_coordinates': generator.x,
        'y_coordinates': generator.y
    }
    
    # Save all data
    generator.save_data(sintering_df, cycling_df, steady_df, spatial_data)
    
    print("\nData generation complete!")
    print("Files generated:")
    print("  - sofc_thermal_data/sintering_thermal_data.csv")
    print("  - sofc_thermal_data/thermal_cycling_data.csv") 
    print("  - sofc_thermal_data/steady_state_thermal_data.csv")
    print("  - sofc_thermal_data/spatial_thermal_data.npz")
    print("  - sofc_thermal_data/metadata.json")

if __name__ == "__main__":
    main()