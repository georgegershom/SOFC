#!/usr/bin/env python3
"""
SOFC Thermal History Data Generator
Generates comprehensive thermal data for SOFC analysis including:
- Sintering & Co-firing temperature profiles
- Thermal cycling data (start-up/shut-down)
- Steady-state operation temperature gradients
- Residual stress calculations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SOFCThermalDataGenerator:
    def __init__(self):
        # SOFC geometry parameters (typical planar SOFC)
        self.cell_length = 100e-3  # 100 mm
        self.cell_width = 100e-3   # 100 mm
        self.anode_thickness = 500e-6  # 500 μm
        self.electrolyte_thickness = 10e-6  # 10 μm
        self.cathode_thickness = 50e-6  # 50 μm
        
        # Material properties
        self.materials = {
            'anode': {
                'thermal_conductivity': 2.0,  # W/m·K
                'thermal_expansion': 12.5e-6,  # 1/K
                'density': 6000,  # kg/m³
                'specific_heat': 500  # J/kg·K
            },
            'electrolyte': {
                'thermal_conductivity': 2.5,  # W/m·K
                'thermal_expansion': 10.8e-6,  # 1/K
                'density': 6000,  # kg/m³
                'specific_heat': 500  # J/kg·K
            },
            'cathode': {
                'thermal_conductivity': 1.8,  # W/m·K
                'thermal_expansion': 12.0e-6,  # 1/K
                'density': 6000,  # kg/m³
                'specific_heat': 500  # J/kg·K
            }
        }
        
        # Spatial grid
        self.x = np.linspace(0, self.cell_length, 50)
        self.y = np.linspace(0, self.cell_width, 50)
        self.z_anode = np.linspace(0, self.anode_thickness, 10)
        self.z_electrolyte = np.linspace(self.anode_thickness, 
                                       self.anode_thickness + self.electrolyte_thickness, 5)
        self.z_cathode = np.linspace(self.anode_thickness + self.electrolyte_thickness,
                                   self.anode_thickness + self.electrolyte_thickness + self.cathode_thickness, 10)
        
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
    def generate_sintering_data(self):
        """Generate sintering and co-firing temperature profiles"""
        print("Generating sintering and co-firing data...")
        
        # Sintering temperature profile (typical for SOFC materials)
        sintering_temp = 1400  # °C
        co_firing_temp = 1200  # °C
        
        # Time profile for sintering process
        time_sintering = np.linspace(0, 8*3600, 1000)  # 8 hours in seconds
        
        # Temperature ramp profile
        def sintering_profile(t):
            if t < 2*3600:  # Ramp up
                return 25 + (sintering_temp - 25) * (t / (2*3600))
            elif t < 6*3600:  # Hold at sintering temperature
                return sintering_temp
            else:  # Cool down
                return sintering_temp - (sintering_temp - 25) * ((t - 6*3600) / (2*3600))
        
        # Spatial temperature distribution during sintering
        # Non-uniform heating due to furnace characteristics
        def spatial_temp_distribution(x, y, t, base_temp):
            # Add spatial variations due to furnace non-uniformity
            center_x, center_y = self.cell_length/2, self.cell_width/2
            distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            
            # Temperature variation: cooler at edges
            spatial_factor = 1 - 0.1 * (distance_from_center / max_distance)
            
            # Add some random variation
            noise = np.random.normal(0, 2, x.shape)
            
            return base_temp * spatial_factor + noise
        
        # Generate data for each time step
        sintering_data = []
        for i, t in enumerate(time_sintering):
            base_temp = sintering_profile(t)
            
            # Generate spatial temperature distribution
            temp_field = spatial_temp_distribution(self.X, self.Y, t, base_temp)
            
            # Calculate temperature gradients
            grad_x = np.gradient(temp_field, axis=1)
            grad_y = np.gradient(temp_field, axis=0)
            
            # Store data
            for j in range(self.X.shape[0]):
                for k in range(self.X.shape[1]):
                    sintering_data.append({
                        'time': t,
                        'x_position': self.X[j, k],
                        'y_position': self.Y[j, k],
                        'temperature': temp_field[j, k],
                        'gradient_x': grad_x[j, k],
                        'gradient_y': grad_y[j, k],
                        'process': 'sintering'
                    })
        
        return pd.DataFrame(sintering_data)
    
    def generate_thermal_cycling_data(self):
        """Generate thermal cycling data for start-up and shut-down"""
        print("Generating thermal cycling data...")
        
        # Define cycling parameters
        num_cycles = 50
        cycle_duration = 2*3600  # 2 hours per cycle
        start_temp = 25  # °C
        operating_temp = 800  # °C
        
        cycling_data = []
        
        for cycle in range(num_cycles):
            # Start-up phase (0 to 30 minutes)
            start_up_time = np.linspace(0, 30*60, 100)
            start_up_temp = start_temp + (operating_temp - start_temp) * (start_up_time / (30*60))
            
            # Hold phase (30 minutes to 1.5 hours)
            hold_time = np.linspace(30*60, 90*60, 100)
            hold_temp = np.full_like(hold_time, operating_temp)
            
            # Shut-down phase (1.5 to 2 hours)
            shut_down_time = np.linspace(90*60, 120*60, 100)
            shut_down_temp = operating_temp - (operating_temp - start_temp) * ((shut_down_time - 90*60) / (30*60))
            
            # Combine cycle phases
            cycle_time = np.concatenate([start_up_time, hold_time, shut_down_time])
            cycle_temp = np.concatenate([start_up_temp, hold_temp, shut_down_temp])
            
            # Add spatial variations
            for i, (t, temp) in enumerate(zip(cycle_time, cycle_temp)):
                # Add thermal gradients across the cell
                temp_gradient = 5 * np.sin(2*np.pi * self.X / self.cell_length) * np.cos(2*np.pi * self.Y / self.cell_width)
                spatial_temp = temp + temp_gradient
                
                # Add some random variation
                noise = np.random.normal(0, 1, self.X.shape)
                spatial_temp += noise
                
                # Calculate thermal stress indicators
                max_temp = np.max(spatial_temp)
                min_temp = np.min(spatial_temp)
                temp_range = max_temp - min_temp
                
                # Store data
                for j in range(self.X.shape[0]):
                    for k in range(self.X.shape[1]):
                        cycling_data.append({
                            'cycle_number': cycle + 1,
                            'time': t + cycle * cycle_duration,
                            'x_position': self.X[j, k],
                            'y_position': self.Y[j, k],
                            'temperature': spatial_temp[j, k],
                            'max_temp': max_temp,
                            'min_temp': min_temp,
                            'temp_range': temp_range,
                            'phase': 'start_up' if t < 30*60 else 'hold' if t < 90*60 else 'shut_down'
                        })
        
        return pd.DataFrame(cycling_data)
    
    def generate_steady_state_data(self):
        """Generate steady-state operation temperature gradients"""
        print("Generating steady-state operation data...")
        
        # Steady-state operating conditions
        operating_temp = 800  # °C
        current_density = 0.5  # A/cm²
        
        # Generate temperature field with realistic gradients
        def steady_state_temp_field(x, y, z):
            # Base temperature
            base_temp = operating_temp
            
            # Current density effect (higher current = higher temperature)
            current_effect = 20 * (x / self.cell_length) * (y / self.cell_width)
            
            # Edge cooling effect
            edge_cooling = -10 * np.minimum(
                np.minimum(x, self.cell_length - x),
                np.minimum(y, self.cell_width - y)
            ) / min(self.cell_length, self.cell_width)
            
            # Z-direction gradient (through thickness)
            z_gradient = -5 * (z / (self.anode_thickness + self.electrolyte_thickness + self.cathode_thickness))
            
            return base_temp + current_effect + edge_cooling + z_gradient
        
        # Generate 3D temperature field
        steady_state_data = []
        
        # Anode layer
        for i, z in enumerate(self.z_anode):
            temp_field = steady_state_temp_field(self.X, self.Y, z)
            for j in range(self.X.shape[0]):
                for k in range(self.X.shape[1]):
                    steady_state_data.append({
                        'x_position': self.X[j, k],
                        'y_position': self.Y[j, k],
                        'z_position': z,
                        'temperature': temp_field[j, k],
                        'layer': 'anode',
                        'current_density': current_density
                    })
        
        # Electrolyte layer
        for i, z in enumerate(self.z_electrolyte):
            temp_field = steady_state_temp_field(self.X, self.Y, z)
            for j in range(self.X.shape[0]):
                for k in range(self.X.shape[1]):
                    steady_state_data.append({
                        'x_position': self.X[j, k],
                        'y_position': self.Y[j, k],
                        'z_position': z,
                        'temperature': temp_field[j, k],
                        'layer': 'electrolyte',
                        'current_density': current_density
                    })
        
        # Cathode layer
        for i, z in enumerate(self.z_cathode):
            temp_field = steady_state_temp_field(self.X, self.Y, z)
            for j in range(self.X.shape[0]):
                for k in range(self.X.shape[1]):
                    steady_state_data.append({
                        'x_position': self.X[j, k],
                        'y_position': self.Y[j, k],
                        'z_position': z,
                        'temperature': temp_field[j, k],
                        'layer': 'cathode',
                        'current_density': current_density
                    })
        
        return pd.DataFrame(steady_state_data)
    
    def calculate_residual_stresses(self, sintering_data):
        """Calculate residual stresses from thermal history"""
        print("Calculating residual stresses...")
        
        stress_data = []
        
        # Calculate thermal expansion and resulting stresses
        for _, row in sintering_data.iterrows():
            temp = row['temperature']
            x, y = row['x_position'], row['y_position']
            
            # Calculate thermal expansion for each layer
            for layer, props in self.materials.items():
                # Thermal strain
                thermal_strain = props['thermal_expansion'] * (temp - 25)  # Reference temp 25°C
                
                # Calculate stress (simplified)
                # Assuming plane stress condition
                E = 200e9  # Young's modulus (Pa)
                nu = 0.3   # Poisson's ratio
                
                # Thermal stress
                thermal_stress = E * thermal_strain / (1 - nu)
                
                stress_data.append({
                    'x_position': x,
                    'y_position': y,
                    'temperature': temp,
                    'layer': layer,
                    'thermal_strain': thermal_strain,
                    'thermal_stress': thermal_stress,
                    'time': row['time']
                })
        
        return pd.DataFrame(stress_data)
    
    def generate_all_data(self):
        """Generate all thermal history data"""
        print("Generating comprehensive SOFC thermal history data...")
        
        # Generate all datasets
        sintering_data = self.generate_sintering_data()
        cycling_data = self.generate_thermal_cycling_data()
        steady_state_data = self.generate_steady_state_data()
        stress_data = self.calculate_residual_stresses(sintering_data)
        
        # Save data
        sintering_data.to_csv('sofc_sintering_data.csv', index=False)
        cycling_data.to_csv('sofc_thermal_cycling_data.csv', index=False)
        steady_state_data.to_csv('sofc_steady_state_data.csv', index=False)
        stress_data.to_csv('sofc_residual_stress_data.csv', index=False)
        
        # Generate summary statistics
        summary = {
            'sintering_data': {
                'total_records': len(sintering_data),
                'temperature_range': [sintering_data['temperature'].min(), sintering_data['temperature'].max()],
                'time_range': [sintering_data['time'].min(), sintering_data['time'].max()]
            },
            'cycling_data': {
                'total_records': len(cycling_data),
                'num_cycles': cycling_data['cycle_number'].max(),
                'temperature_range': [cycling_data['temperature'].min(), cycling_data['temperature'].max()]
            },
            'steady_state_data': {
                'total_records': len(steady_state_data),
                'temperature_range': [steady_state_data['temperature'].min(), steady_state_data['temperature'].max()],
                'layers': steady_state_data['layer'].unique().tolist()
            },
            'stress_data': {
                'total_records': len(stress_data),
                'stress_range': [stress_data['thermal_stress'].min(), stress_data['thermal_stress'].max()],
                'strain_range': [stress_data['thermal_strain'].min(), stress_data['thermal_strain'].max()]
            }
        }
        
        with open('sofc_thermal_data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Data generation complete!")
        print(f"Sintering data: {len(sintering_data):,} records")
        print(f"Cycling data: {len(cycling_data):,} records")
        print(f"Steady-state data: {len(steady_state_data):,} records")
        print(f"Stress data: {len(stress_data):,} records")
        
        return sintering_data, cycling_data, steady_state_data, stress_data

if __name__ == "__main__":
    generator = SOFCThermalDataGenerator()
    sintering_data, cycling_data, steady_state_data, stress_data = generator.generate_all_data()