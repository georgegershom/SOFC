#!/usr/bin/env python3
"""
Generate Spalling Pattern and Failure Data
Critical validation data for explosive spalling and failure prediction
Includes time-to-spalling, spalling depth, failure modes, and damage patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import weibull_min
from scipy.spatial.distance import cdist
import json
from datetime import datetime

class SpallingFailureGenerator:
    def __init__(self, specimen_type='slab', rubber_content_percent=0, moisture_content=0.75):
        """
        Initialize spalling and failure data generator
        specimen_type: 'slab', 'column', 'wall'
        rubber_content_percent: 0-30% - rubber reduces spalling risk
        moisture_content: 0-1 saturation degree - higher moisture increases spalling
        """
        self.specimen_type = specimen_type
        self.rubber_content = rubber_content_percent
        self.moisture_content = moisture_content
        
        # Specimen dimensions (mm)
        if specimen_type == 'slab':
            self.width = 600
            self.length = 600
            self.thickness = 200
        elif specimen_type == 'column':
            self.width = 400
            self.length = 400
            self.thickness = 400
            self.height = 3000
        else:  # wall
            self.width = 2000
            self.length = 100  # wall thickness
            self.thickness = 3000  # wall height
        
        # Spalling risk factors
        self.base_spalling_risk = self._calculate_spalling_risk()
    
    def _calculate_spalling_risk(self):
        """
        Calculate base spalling risk based on material properties
        """
        # Moisture effect (dominant factor)
        moisture_factor = self.moisture_content ** 2
        
        # Rubber effect (reduces spalling)
        rubber_factor = 1 - (self.rubber_content / 100) * 0.6
        
        # Permeability effect (assumed based on rubber content)
        permeability_factor = 1 - (self.rubber_content / 100) * 0.3
        
        # Combined risk (0-1)
        risk = moisture_factor * rubber_factor * permeability_factor
        
        return min(1.0, max(0.0, risk))
    
    def time_to_spalling(self, fire_intensity='standard', load_level=0.3):
        """
        Calculate time to first spalling event (minutes)
        Returns None if no spalling occurs
        """
        if self.base_spalling_risk < 0.2:
            # Very low risk - no spalling
            return None
        
        # Base time depends on fire intensity
        if fire_intensity == 'standard':
            base_time = 15  # minutes
        elif fire_intensity == 'hydrocarbon':
            base_time = 5
        else:  # slow
            base_time = 30
        
        # Moisture effect (higher moisture = earlier spalling)
        moisture_factor = 2 - self.moisture_content
        
        # Load effect (higher load = earlier spalling)
        load_factor = 1 - load_level * 0.3
        
        # Rubber effect (delays spalling)
        rubber_factor = 1 + (self.rubber_content / 100) * 0.5
        
        # Random variation (Weibull distribution)
        shape = 2.5  # Shape parameter
        scale = base_time * moisture_factor * load_factor * rubber_factor
        
        time = weibull_min.rvs(shape, scale=scale)
        
        # Add some noise
        time += np.random.normal(0, 2)
        
        return max(1, time)  # Minimum 1 minute
    
    def spalling_depth(self, time_minutes, location='center'):
        """
        Calculate spalling depth at given time and location
        """
        if self.base_spalling_risk < 0.2:
            return 0
        
        # Maximum spalling depth
        max_depth = 50 * self.base_spalling_risk  # mm
        
        # Time evolution (logarithmic growth)
        if time_minutes > 0:
            time_factor = np.log10(1 + time_minutes / 10)
        else:
            time_factor = 0
        
        # Location effect
        if location == 'corner':
            location_factor = 1.3  # Corners spall more
        elif location == 'edge':
            location_factor = 1.15
        else:  # center
            location_factor = 1.0
        
        # Calculate depth
        depth = max_depth * time_factor * location_factor
        
        # Add variability
        depth *= np.random.uniform(0.8, 1.2)
        
        return min(depth, self.thickness * 0.3)  # Max 30% of thickness
    
    def generate_spalling_events(self, test_duration_min=120):
        """
        Generate sequence of spalling events during fire test
        """
        events = []
        
        # First spalling
        t_first = self.time_to_spalling()
        
        if t_first is None or t_first > test_duration_min:
            # No spalling occurs
            return pd.DataFrame(events)
        
        current_time = t_first
        event_number = 1
        cumulative_depth = 0
        
        while current_time < test_duration_min and cumulative_depth < self.thickness * 0.4:
            # Spalling location (random)
            x_pos = np.random.uniform(50, self.width - 50)
            y_pos = np.random.uniform(50, self.length - 50) if self.specimen_type != 'column' else 0
            z_pos = np.random.uniform(100, 1000) if self.specimen_type == 'column' else 0
            
            # Determine location type
            edge_dist = min(x_pos, self.width - x_pos, y_pos, self.length - y_pos)
            if edge_dist < 100:
                location_type = 'corner' if edge_dist < 50 else 'edge'
            else:
                location_type = 'center'
            
            # Spalling characteristics
            depth = self.spalling_depth(current_time, location_type)
            area = np.random.uniform(100, 500) * self.base_spalling_risk  # cm²
            
            # Sound characteristics (for acoustic monitoring validation)
            sound_level = 80 + depth * 0.5 + np.random.normal(0, 5)  # dB
            frequency = 2000 + depth * 20 + np.random.normal(0, 200)  # Hz
            
            event = {
                'event_number': event_number,
                'time_min': current_time,
                'x_position_mm': x_pos,
                'y_position_mm': y_pos,
                'z_position_mm': z_pos,
                'location_type': location_type,
                'spalling_depth_mm': depth,
                'spalled_area_cm2': area,
                'cumulative_depth_mm': cumulative_depth + depth,
                'sound_level_dB': sound_level,
                'peak_frequency_Hz': frequency,
                'specimen_type': self.specimen_type,
                'rubber_content_%': self.rubber_content
            }
            
            events.append(event)
            
            # Update for next event
            cumulative_depth += depth
            event_number += 1
            
            # Time to next spalling (increases with rubber content)
            time_interval = np.random.exponential(10 + self.rubber_content)
            current_time += time_interval
        
        return pd.DataFrame(events)
    
    def generate_failure_mode_data(self, load_level=0.3, fire_duration_min=120):
        """
        Determine failure mode and time
        """
        # Possible failure modes
        modes = {
            'spalling': self.base_spalling_risk,
            'crushing': load_level,
            'buckling': 0.2 if self.specimen_type == 'column' else 0.05,
            'thermal_bowing': 0.3 if self.specimen_type == 'wall' else 0.1,
            'shear': 0.1,
            'no_failure': 0.3 + self.rubber_content / 100
        }
        
        # Normalize probabilities
        total_prob = sum(modes.values())
        for mode in modes:
            modes[mode] /= total_prob
        
        # Select failure mode based on probabilities
        mode_selected = np.random.choice(list(modes.keys()), p=list(modes.values()))
        
        # Time to failure
        if mode_selected == 'no_failure':
            time_to_failure = None
            residual_capacity = 0.3 + self.rubber_content / 100 * 0.2
        else:
            # Base time depends on mode and conditions
            if mode_selected == 'spalling':
                base_time = 30 / self.base_spalling_risk if self.base_spalling_risk > 0 else 180
            elif mode_selected == 'crushing':
                base_time = 60 / load_level
            elif mode_selected == 'buckling':
                base_time = 45
            elif mode_selected == 'thermal_bowing':
                base_time = 90
            else:  # shear
                base_time = 75
            
            # Rubber extends failure time
            rubber_factor = 1 + self.rubber_content / 100 * 0.4
            
            time_to_failure = base_time * rubber_factor * np.random.uniform(0.8, 1.2)
            time_to_failure = min(time_to_failure, fire_duration_min)
            
            # Residual capacity at failure
            residual_capacity = np.random.uniform(0.1, 0.3)
        
        failure_data = {
            'failure_mode': mode_selected,
            'time_to_failure_min': time_to_failure,
            'load_level': load_level,
            'fire_duration_min': fire_duration_min,
            'residual_capacity': residual_capacity,
            'failure_temperature_C': 20 + 345 * np.log10(8 * time_to_failure / 60 + 1) if time_to_failure else None,
            'rubber_content_%': self.rubber_content,
            'moisture_content': self.moisture_content
        }
        
        return failure_data
    
    def generate_damage_pattern(self, time_min=60, grid_size=20):
        """
        Generate 2D damage pattern on exposed surface
        """
        # Create grid
        x_grid = np.linspace(0, self.width, grid_size)
        y_grid = np.linspace(0, self.length, grid_size)
        
        damage_grid = np.zeros((grid_size, grid_size))
        
        # Generate spalling events up to current time
        events = self.generate_spalling_events(time_min)
        
        if not events.empty:
            # Create damage field from spalling events
            for _, event in events.iterrows():
                if event['time_min'] <= time_min:
                    # Event center
                    center_x = event['x_position_mm']
                    center_y = event['y_position_mm']
                    
                    # Damage radius (from area)
                    radius = np.sqrt(event['spalled_area_cm2'] * 100 / np.pi)
                    
                    # Apply damage to grid
                    for i, x in enumerate(x_grid):
                        for j, y in enumerate(y_grid):
                            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                            if dist < radius:
                                # Damage intensity decreases from center
                                intensity = (1 - dist / radius) * event['spalling_depth_mm'] / 50
                                damage_grid[j, i] = min(1.0, damage_grid[j, i] + intensity)
        
        # Add thermal damage gradient
        thermal_damage = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                # Edge effects (edges heat more)
                edge_dist = min(x_grid[i], self.width - x_grid[i],
                              y_grid[j], self.length - y_grid[j])
                edge_factor = 1 - edge_dist / (min(self.width, self.length) / 2)
                
                thermal_damage[j, i] = 0.1 * (time_min / 120) * (1 + edge_factor * 0.3)
        
        # Combine spalling and thermal damage
        total_damage = np.minimum(damage_grid + thermal_damage, 1.0)
        
        return x_grid, y_grid, total_damage
    
    def generate_crack_pattern(self, time_min=60, num_cracks=None):
        """
        Generate surface crack pattern data
        """
        if num_cracks is None:
            # Number of cracks increases with time and decreases with rubber
            num_cracks = int(time_min / 10 * (1 - self.rubber_content / 100 * 0.5))
            num_cracks = max(1, num_cracks + np.random.poisson(2))
        
        cracks = []
        
        for i in range(num_cracks):
            # Crack starting point
            x_start = np.random.uniform(0, self.width)
            y_start = np.random.uniform(0, self.length)
            
            # Crack direction (tend to be perpendicular to edges)
            if x_start < 100 or x_start > self.width - 100:
                angle = np.random.normal(0, 30)  # Vertical tendency
            else:
                angle = np.random.uniform(0, 360)
            
            # Crack length (increases with time)
            length = np.random.uniform(50, 200) * (1 + time_min / 60)
            
            # Crack width (increases with temperature/time)
            width = np.random.uniform(0.1, 2.0) * (1 + time_min / 120)
            
            # Crack end point
            x_end = x_start + length * np.cos(np.radians(angle))
            y_end = y_start + length * np.sin(np.radians(angle))
            
            # Clip to boundaries
            x_end = np.clip(x_end, 0, self.width)
            y_end = np.clip(y_end, 0, self.length)
            
            crack = {
                'crack_id': i + 1,
                'time_min': time_min,
                'x_start_mm': x_start,
                'y_start_mm': y_start,
                'x_end_mm': x_end,
                'y_end_mm': y_end,
                'length_mm': np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2),
                'width_mm': width,
                'angle_deg': angle,
                'crack_type': 'thermal' if time_min < 30 else 'thermo-mechanical'
            }
            
            cracks.append(crack)
        
        return pd.DataFrame(cracks)

def main():
    # Test configurations
    specimen_types = ['slab', 'column', 'wall']
    rubber_contents = [0, 10, 20, 30]
    moisture_contents = [0.5, 0.75, 0.95]  # Different saturation levels
    
    all_data = {}
    
    for specimen in specimen_types:
        for rubber_pct in rubber_contents:
            for moisture in moisture_contents:
                print(f"Generating spalling/failure data for {specimen}, {rubber_pct}% rubber, {moisture*100:.0f}% moisture...")
                
                generator = SpallingFailureGenerator(
                    specimen_type=specimen,
                    rubber_content_percent=rubber_pct,
                    moisture_content=moisture
                )
                
                # Generate spalling events
                spalling_events = generator.generate_spalling_events(test_duration_min=180)
                
                # Generate failure data for different load levels
                failure_data_list = []
                for load in [0.2, 0.3, 0.4]:
                    failure_data = generator.generate_failure_mode_data(load_level=load, fire_duration_min=180)
                    failure_data['test_id'] = f'{specimen}_{rubber_pct}pct_{int(moisture*100)}m_{int(load*100)}L'
                    failure_data_list.append(failure_data)
                
                failure_df = pd.DataFrame(failure_data_list)
                
                # Generate crack patterns at different times
                crack_data_list = []
                for t in [30, 60, 90, 120]:
                    cracks = generator.generate_crack_pattern(time_min=t)
                    crack_data_list.append(cracks)
                
                if crack_data_list:
                    crack_df = pd.concat(crack_data_list, ignore_index=True)
                else:
                    crack_df = pd.DataFrame()
                
                # Store data
                key = f'{specimen}_{rubber_pct}pct_{int(moisture*100)}m'
                all_data[key] = {
                    'spalling_events': spalling_events,
                    'failure_modes': failure_df,
                    'crack_patterns': crack_df
                }
                
                # Save CSV files
                if not spalling_events.empty:
                    spalling_events.to_csv(f'spalling_events_{key}.csv', index=False)
                failure_df.to_csv(f'failure_modes_{key}.csv', index=False)
                if not crack_df.empty:
                    crack_df.to_csv(f'crack_patterns_{key}.csv', index=False)
    
    print(f"\nSaved {len(specimen_types) * len(rubber_contents) * len(moisture_contents) * 3} spalling/failure files")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Time to spalling vs moisture content (slab, different rubber contents)
    moisture_range = np.linspace(0.3, 1.0, 20)
    for rubber_pct in [0, 15, 30]:
        times_to_spalling = []
        for m in moisture_range:
            gen = SpallingFailureGenerator('slab', rubber_pct, m)
            t = gen.time_to_spalling()
            times_to_spalling.append(t if t else 200)  # Use 200 min for no spalling
        
        axes[0, 0].plot(moisture_range * 100, times_to_spalling,
                       label=f'{rubber_pct}% rubber', linewidth=2)
    
    axes[0, 0].set_xlabel('Moisture Content (%)')
    axes[0, 0].set_ylabel('Time to First Spalling (min)')
    axes[0, 0].set_title('Moisture Effect on Spalling Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 200])
    
    # Plot 2: Spalling depth evolution (slab, 75% moisture)
    key = 'slab_10pct_75m'
    if key in all_data and not all_data[key]['spalling_events'].empty:
        events = all_data[key]['spalling_events']
        axes[0, 1].scatter(events['time_min'], events['spalling_depth_mm'],
                         s=events['spalled_area_cm2'], alpha=0.6)
        axes[0, 1].plot(events['time_min'], events['cumulative_depth_mm'],
                       'r-', linewidth=2, label='Cumulative depth')
        
        axes[0, 1].set_xlabel('Time (min)')
        axes[0, 1].set_ylabel('Spalling Depth (mm)')
        axes[0, 1].set_title('Spalling Evolution (Slab, 10% Rubber, 75% Moisture)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Failure modes distribution
    failure_modes_count = {}
    for key in all_data:
        if 'slab' in key and '75m' in key:  # Focus on slab with 75% moisture
            failures = all_data[key]['failure_modes']
            for mode in failures['failure_mode']:
                if mode not in failure_modes_count:
                    failure_modes_count[mode] = 0
                failure_modes_count[mode] += 1
    
    if failure_modes_count:
        modes = list(failure_modes_count.keys())
        counts = list(failure_modes_count.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(modes)))
        
        axes[0, 2].pie(counts, labels=modes, colors=colors, autopct='%1.1f%%')
        axes[0, 2].set_title('Failure Mode Distribution (Slab, 75% Moisture)')
    
    # Plot 4: Damage pattern heatmap
    gen = SpallingFailureGenerator('slab', 15, 0.75)
    x_grid, y_grid, damage = gen.generate_damage_pattern(time_min=90)
    
    im = axes[1, 0].imshow(damage, cmap='hot', extent=[0, gen.width, 0, gen.length],
                          origin='lower', vmin=0, vmax=1)
    axes[1, 0].set_xlabel('Width (mm)')
    axes[1, 0].set_ylabel('Length (mm)')
    axes[1, 0].set_title('Damage Pattern (Slab, 15% Rubber, t=90 min)')
    plt.colorbar(im, ax=axes[1, 0], label='Damage Level')
    
    # Plot 5: Time to failure vs load level
    load_levels = np.linspace(0.1, 0.6, 10)
    for rubber_pct in [0, 15, 30]:
        times_to_failure = []
        for load in load_levels:
            gen = SpallingFailureGenerator('column', rubber_pct, 0.75)
            failure_data = gen.generate_failure_mode_data(load_level=load)
            ttf = failure_data['time_to_failure_min']
            times_to_failure.append(ttf if ttf else 200)
        
        axes[1, 1].plot(load_levels * 100, times_to_failure,
                       label=f'{rubber_pct}% rubber', linewidth=2)
    
    axes[1, 1].set_xlabel('Load Level (% fc)')
    axes[1, 1].set_ylabel('Time to Failure (min)')
    axes[1, 1].set_title('Load Effect on Failure Time (Column, 75% Moisture)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Crack pattern visualization
    gen = SpallingFailureGenerator('slab', 10, 0.75)
    cracks_60 = gen.generate_crack_pattern(time_min=60)
    
    axes[1, 2].set_xlim([0, gen.width])
    axes[1, 2].set_ylim([0, gen.length])
    
    for _, crack in cracks_60.iterrows():
        axes[1, 2].plot([crack['x_start_mm'], crack['x_end_mm']],
                       [crack['y_start_mm'], crack['y_end_mm']],
                       'r-', linewidth=crack['width_mm'], alpha=0.7)
    
    # Add specimen boundary
    rect = patches.Rectangle((0, 0), gen.width, gen.length,
                            linewidth=2, edgecolor='k', facecolor='none')
    axes[1, 2].add_patch(rect)
    
    axes[1, 2].set_xlabel('Width (mm)')
    axes[1, 2].set_ylabel('Length (mm)')
    axes[1, 2].set_title('Surface Crack Pattern (Slab, 10% Rubber, t=60 min)')
    axes[1, 2].set_aspect('equal')
    
    plt.suptitle('Spalling Patterns and Failure Validation Data', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('spalling_failure_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate metadata
    metadata = {
        'generated_date': datetime.now().isoformat(),
        'specimen_types': specimen_types,
        'rubber_content_range_percent': [0, 30],
        'moisture_content_range': [0.5, 0.95],
        'spalling_characteristics': {
            'depth_range_mm': [5, 50],
            'area_range_cm2': [50, 500],
            'time_to_first_spalling_min': [5, 120],
            'acoustic_emission': {
                'sound_level_dB': [70, 100],
                'frequency_Hz': [1000, 5000]
            }
        },
        'failure_modes': [
            'spalling',
            'crushing',
            'buckling',
            'thermal_bowing',
            'shear',
            'no_failure'
        ],
        'measurement_techniques': {
            'visual_inspection': 'High-speed camera (1000 fps)',
            'acoustic_monitoring': 'Piezoelectric sensors',
            'laser_scanning': '3D surface profiling',
            'thermal_imaging': 'IR camera for hot spots',
            'digital_image_correlation': 'Full-field strain measurement'
        },
        'critical_parameters': {
            'pore_pressure_threshold_MPa': 4,
            'tensile_strength_threshold_MPa': 3,
            'moisture_clog_depth_mm': 20,
            'temperature_gradient_C_mm': 10
        },
        'rubber_benefits': {
            'spalling_reduction': 'Up to 60% reduction in spalling risk',
            'failure_time_extension': 'Up to 40% increase in time to failure',
            'crack_reduction': 'Up to 50% reduction in crack density',
            'improved_permeability': 'Enhanced moisture escape paths'
        },
        'data_usage': 'Model validation for spalling and failure prediction'
    }
    
    with open('spalling_failure_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Spalling pattern and failure dataset generation complete!")

if __name__ == "__main__":
    main()