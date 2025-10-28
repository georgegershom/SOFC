"""
Generate synthetic synchrotron X-ray tomography data for SOFC creep analysis.
This simulates 3D microstructure and 4D time-evolution data.
"""

import numpy as np
import h5py
from scipy import ndimage
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

class SynchrotronTomographyGenerator:
    """Generate synthetic synchrotron tomography data for SOFC materials."""
    
    def __init__(self, volume_size=(256, 256, 256), voxel_size=0.65e-6):
        """
        Initialize the generator.
        
        Parameters:
        -----------
        volume_size : tuple
            3D volume dimensions in voxels (z, y, x)
        voxel_size : float
            Physical size of each voxel in meters (typically sub-micron)
        """
        self.volume_size = volume_size
        self.voxel_size = voxel_size  # meters
        self.material_phases = {
            'matrix': 1,           # Base metal matrix
            'grain_boundary': 2,   # Grain boundary regions
            'pore': 0,            # Initial porosity
            'oxide': 3,           # Oxide phases
            'cavity': 4           # Creep cavities
        }
        
    def generate_grain_structure(self, num_grains=100):
        """
        Generate realistic grain structure using Voronoi tessellation.
        
        Parameters:
        -----------
        num_grains : int
            Number of grains in the volume
            
        Returns:
        --------
        volume : ndarray
            3D array with grain structure
        grain_map : dict
            Mapping of grain IDs to properties
        """
        print(f"Generating grain structure with {num_grains} grains...")
        
        # Create random seed points for grains
        seed_points = np.random.rand(num_grains, 3)
        seed_points[:, 0] *= self.volume_size[0]
        seed_points[:, 1] *= self.volume_size[1]
        seed_points[:, 2] *= self.volume_size[2]
        
        # Initialize volume
        volume = np.ones(self.volume_size, dtype=np.uint16) * self.material_phases['matrix']
        
        # Create coordinate grid
        z, y, x = np.mgrid[0:self.volume_size[0], 
                           0:self.volume_size[1], 
                           0:self.volume_size[2]]
        
        # Assign voxels to nearest grain seed (simplified Voronoi)
        grain_ids = np.zeros(self.volume_size, dtype=np.uint16)
        
        for i, seed in enumerate(seed_points):
            # Calculate distance from each voxel to this seed
            dist = np.sqrt((z - seed[0])**2 + (y - seed[1])**2 + (x - seed[2])**2)
            
            # Update grain assignment if this seed is closer
            if i == 0:
                min_dist = dist
                grain_ids[:] = i + 1
            else:
                mask = dist < min_dist
                grain_ids[mask] = i + 1
                min_dist = np.minimum(min_dist, dist)
        
        # Generate grain properties
        grain_map = {}
        for i in range(1, num_grains + 1):
            grain_map[i] = {
                'volume': np.sum(grain_ids == i),
                'orientation': np.random.rand(3) * 360,  # Euler angles
                'phase': 'ferrite' if np.random.rand() > 0.1 else 'carbide',
                'elastic_modulus': 200 + np.random.randn() * 10,  # GPa
                'initial_stress': np.random.randn(3) * 50  # MPa
            }
        
        # Add grain boundaries
        print("Adding grain boundaries...")
        boundaries = self._detect_grain_boundaries(grain_ids)
        volume[boundaries] = self.material_phases['grain_boundary']
        
        return volume, grain_ids, grain_map
    
    def _detect_grain_boundaries(self, grain_ids, thickness=2):
        """
        Detect and mark grain boundaries.
        
        Parameters:
        -----------
        grain_ids : ndarray
            3D array with grain IDs
        thickness : int
            Boundary thickness in voxels
            
        Returns:
        --------
        boundaries : ndarray
            Boolean array marking boundary voxels
        """
        # Find edges using gradient
        grad_z = np.abs(np.diff(grain_ids, axis=0, prepend=grain_ids[0:1]))
        grad_y = np.abs(np.diff(grain_ids, axis=1, prepend=grain_ids[:,0:1]))
        grad_x = np.abs(np.diff(grain_ids, axis=2, prepend=grain_ids[:,:,0:1]))
        
        boundaries = (grad_z > 0) | (grad_y > 0) | (grad_x > 0)
        
        # Dilate boundaries to desired thickness
        if thickness > 1:
            boundaries = ndimage.binary_dilation(boundaries, iterations=thickness-1)
        
        return boundaries
    
    def add_initial_defects(self, volume, porosity=0.02, num_inclusions=20):
        """
        Add realistic initial defects to the microstructure.
        
        Parameters:
        -----------
        volume : ndarray
            3D volume array
        porosity : float
            Volume fraction of initial porosity
        num_inclusions : int
            Number of oxide inclusions
            
        Returns:
        --------
        volume : ndarray
            Updated volume with defects
        defect_map : dict
            Information about added defects
        """
        print(f"Adding initial defects (porosity={porosity:.1%})...")
        
        defect_map = {'pores': [], 'inclusions': []}
        
        # Add randomly distributed small pores
        num_pores = int(porosity * volume.size / 27)  # Assuming average pore is 3x3x3 voxels
        
        for _ in range(num_pores):
            # Random pore location
            z = np.random.randint(3, self.volume_size[0] - 3)
            y = np.random.randint(3, self.volume_size[1] - 3)
            x = np.random.randint(3, self.volume_size[2] - 3)
            
            # Random pore size (1-5 voxel radius)
            radius = np.random.randint(1, 4)
            
            # Create spherical pore
            zz, yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
            mask = zz*zz + yy*yy + xx*xx <= radius*radius
            
            # Add pore to volume
            z_slice = slice(max(0, z-radius), min(self.volume_size[0], z+radius+1))
            y_slice = slice(max(0, y-radius), min(self.volume_size[1], y+radius+1))
            x_slice = slice(max(0, x-radius), min(self.volume_size[2], x+radius+1))
            
            volume[z_slice, y_slice, x_slice][mask] = self.material_phases['pore']
            
            defect_map['pores'].append({
                'center': [z, y, x],
                'radius': radius,
                'volume': np.sum(mask)
            })
        
        # Add oxide inclusions
        for _ in range(num_inclusions):
            z = np.random.randint(5, self.volume_size[0] - 5)
            y = np.random.randint(5, self.volume_size[1] - 5)
            x = np.random.randint(5, self.volume_size[2] - 5)
            
            # Irregular shape using random walk
            size = np.random.randint(3, 8)
            inclusion_mask = np.zeros((size*2+1, size*2+1, size*2+1), dtype=bool)
            inclusion_mask[size, size, size] = True
            
            for _ in range(size * 10):
                # Random walk to create irregular shape
                dz, dy, dx = np.random.randint(-1, 2, 3)
                new_pos = np.array([size + dz, size + dy, size + dx])
                if np.all(new_pos >= 0) and np.all(new_pos < size*2+1):
                    inclusion_mask[tuple(new_pos)] = True
            
            # Add to volume
            z_slice = slice(max(0, z-size), min(self.volume_size[0], z+size+1))
            y_slice = slice(max(0, y-size), min(self.volume_size[1], y+size+1))
            x_slice = slice(max(0, x-size), min(self.volume_size[2], x+size+1))
            
            volume[z_slice, y_slice, x_slice][inclusion_mask] = self.material_phases['oxide']
            
            defect_map['inclusions'].append({
                'center': [z, y, x],
                'size': size,
                'volume': np.sum(inclusion_mask)
            })
        
        return volume, defect_map
    
    def simulate_creep_evolution(self, initial_volume, time_steps=10, 
                                temperature=700, stress=100, time_hours=1000):
        """
        Simulate time evolution of microstructure under creep conditions.
        
        Parameters:
        -----------
        initial_volume : ndarray
            Initial 3D microstructure
        time_steps : int
            Number of time points to simulate
        temperature : float
            Temperature in Celsius
        stress : float
            Applied stress in MPa
        time_hours : float
            Total simulation time in hours
            
        Returns:
        --------
        time_series : list
            List of 3D volumes at each time step
        evolution_data : dict
            Creep evolution metrics
        """
        print(f"Simulating creep evolution: T={temperature}°C, σ={stress}MPa, t={time_hours}h")
        
        time_series = [initial_volume.copy()]
        evolution_data = {
            'time_hours': np.linspace(0, time_hours, time_steps),
            'cavity_volume': [0],
            'crack_length': [0],
            'strain': [0],
            'damage': [0]
        }
        
        current_volume = initial_volume.copy()
        
        for step in range(1, time_steps):
            print(f"  Time step {step}/{time_steps}...")
            
            # Calculate creep rate (simplified Norton law)
            A = 1e-10 * np.exp(-temperature/100)  # Simplified pre-exponential
            n = 5  # Stress exponent
            creep_rate = A * (stress ** n) * (step / time_steps)
            
            # Cavity nucleation (preferentially at grain boundaries)
            gb_mask = current_volume == self.material_phases['grain_boundary']
            nucleation_prob = creep_rate * 0.01  # Simplified probability
            
            # Random cavity nucleation
            nucleation_mask = gb_mask & (np.random.rand(*current_volume.shape) < nucleation_prob)
            current_volume[nucleation_mask] = self.material_phases['cavity']
            
            # Cavity growth (dilation of existing cavities)
            cavity_mask = current_volume == self.material_phases['cavity']
            if np.any(cavity_mask):
                # Grow cavities slightly
                grown_cavities = ndimage.binary_dilation(cavity_mask, iterations=1)
                growth_mask = grown_cavities & (current_volume != self.material_phases['pore'])
                
                # Only grow into matrix or grain boundary regions
                valid_growth = growth_mask & (
                    (current_volume == self.material_phases['matrix']) |
                    (current_volume == self.material_phases['grain_boundary'])
                )
                current_volume[valid_growth] = self.material_phases['cavity']
            
            # Crack propagation (connect nearby cavities)
            if step > time_steps // 2:  # Cracks appear in later stages
                cavity_mask = current_volume == self.material_phases['cavity']
                # Find connected components
                labeled, num_features = ndimage.label(cavity_mask)
                
                # Connect close cavities to simulate crack coalescence
                for i in range(1, min(num_features, 10)):
                    component_mask = labeled == i
                    dilated = ndimage.binary_dilation(component_mask, iterations=3)
                    
                    for j in range(i+1, min(num_features, 10)):
                        other_mask = labeled == j
                        if np.any(dilated & other_mask):
                            # Connect these two components
                            self._connect_regions(current_volume, component_mask, other_mask)
            
            # Calculate metrics
            cavity_volume = np.sum(current_volume == self.material_phases['cavity'])
            evolution_data['cavity_volume'].append(cavity_volume)
            
            # Estimate crack length (simplified)
            if cavity_volume > 0:
                crack_skeleton = ndimage.binary_erosion(
                    current_volume == self.material_phases['cavity'],
                    iterations=1
                )
                crack_length = np.sum(crack_skeleton) * self.voxel_size * 1e6  # Convert to μm
            else:
                crack_length = 0
            evolution_data['crack_length'].append(crack_length)
            
            # Calculate strain (simplified)
            strain = creep_rate * evolution_data['time_hours'][step]
            evolution_data['strain'].append(strain)
            
            # Calculate damage parameter
            damage = cavity_volume / current_volume.size
            evolution_data['damage'].append(damage)
            
            time_series.append(current_volume.copy())
        
        return time_series, evolution_data
    
    def _connect_regions(self, volume, mask1, mask2):
        """Connect two regions with a simple path."""
        # Find closest points between regions
        coords1 = np.array(np.where(mask1)).T
        coords2 = np.array(np.where(mask2)).T
        
        if len(coords1) > 0 and len(coords2) > 0:
            # Find minimum distance pair
            min_dist = float('inf')
            best_pair = None
            
            for c1 in coords1[::10]:  # Sample for efficiency
                for c2 in coords2[::10]:
                    dist = np.linalg.norm(c1 - c2)
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (c1, c2)
            
            if best_pair and min_dist < 20:  # Only connect if reasonably close
                # Create simple linear connection
                p1, p2 = best_pair
                num_points = int(min_dist) + 1
                for i in range(num_points):
                    t = i / max(num_points - 1, 1)
                    point = p1 * (1 - t) + p2 * t
                    point = point.astype(int)
                    
                    if np.all(point >= 0) and np.all(point < volume.shape):
                        volume[tuple(point)] = self.material_phases['cavity']
    
    def save_tomography_data(self, volume, metadata, filename):
        """
        Save tomography data in HDF5 format.
        
        Parameters:
        -----------
        volume : ndarray
            3D volume data
        metadata : dict
            Associated metadata
        filename : str
            Output filename
        """
        print(f"Saving data to {filename}...")
        
        with h5py.File(filename, 'w') as f:
            # Save volume data
            dset = f.create_dataset('volume', data=volume, compression='gzip')
            
            # Add attributes
            dset.attrs['voxel_size'] = self.voxel_size
            dset.attrs['units'] = 'meters'
            dset.attrs['dimensions'] = 'ZYX'
            
            # Save metadata as attributes
            for key, value in metadata.items():
                if isinstance(value, dict):
                    # Convert dict to JSON string with numpy converter
                    dset.attrs[key] = json.dumps(value, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x))
                elif isinstance(value, (list, np.ndarray)):
                    # Save arrays as datasets
                    f.create_dataset(f'metadata/{key}', data=value)
                else:
                    dset.attrs[key] = value
        
        print(f"  Saved {volume.shape} volume ({volume.nbytes / 1e6:.1f} MB)")


def main():
    """Generate complete synthetic tomography dataset."""
    
    # Initialize generator
    gen = SynchrotronTomographyGenerator(
        volume_size=(256, 256, 256),  # Reasonable size for demo
        voxel_size=0.65e-6  # 0.65 micron resolution
    )
    
    # Generate initial microstructure
    print("\n=== Generating Initial Microstructure ===")
    initial_volume, grain_ids, grain_map = gen.generate_grain_structure(num_grains=150)
    initial_volume, defect_map = gen.add_initial_defects(initial_volume, porosity=0.02)
    
    # Save initial state
    initial_metadata = {
        'timestamp': datetime.now().isoformat(),
        'material': 'Cr-Fe-Ni Alloy (Simulated)',
        'temperature': 25,  # Room temperature for initial scan
        'stress': 0,
        'time_hours': 0,
        'grain_count': len(grain_map),
        'defect_statistics': defect_map
    }
    
    os.makedirs('synchrotron_data/tomography/initial', exist_ok=True)
    gen.save_tomography_data(
        initial_volume,
        initial_metadata,
        'synchrotron_data/tomography/initial/initial_microstructure.h5'
    )
    
    # Also save grain map
    with h5py.File('synchrotron_data/tomography/initial/grain_map.h5', 'w') as f:
        f.create_dataset('grain_ids', data=grain_ids, compression='gzip')
        # Convert grain properties with proper serialization
        grain_props_serializable = {}
        for k, v in grain_map.items():
            grain_props_serializable[str(k)] = {}
            for key, val in v.items():
                if isinstance(val, np.ndarray):
                    grain_props_serializable[str(k)][key] = val.tolist()
                elif isinstance(val, (np.integer, np.int64)):
                    grain_props_serializable[str(k)][key] = int(val)
                elif isinstance(val, (np.floating, np.float64)):
                    grain_props_serializable[str(k)][key] = float(val)
                else:
                    grain_props_serializable[str(k)][key] = val
        f.attrs['grain_properties'] = json.dumps(grain_props_serializable)
    
    # Generate time series data for different conditions
    print("\n=== Generating Time Series Data ===")
    
    test_conditions = [
        {'temperature': 600, 'stress': 150, 'time_hours': 500, 'time_steps': 8},
        {'temperature': 700, 'stress': 100, 'time_hours': 1000, 'time_steps': 10},
        {'temperature': 800, 'stress': 75, 'time_hours': 2000, 'time_steps': 12}
    ]
    
    os.makedirs('synchrotron_data/tomography/time_series', exist_ok=True)
    
    for i, condition in enumerate(test_conditions):
        print(f"\nCondition {i+1}: T={condition['temperature']}°C, σ={condition['stress']}MPa")
        
        time_series, evolution_data = gen.simulate_creep_evolution(
            initial_volume,
            time_steps=condition['time_steps'],
            temperature=condition['temperature'],
            stress=condition['stress'],
            time_hours=condition['time_hours']
        )
        
        # Save time series
        series_filename = (f"synchrotron_data/tomography/time_series/"
                          f"creep_T{condition['temperature']}_S{condition['stress']}.h5")
        
        with h5py.File(series_filename, 'w') as f:
            # Save all time steps
            for t, volume in enumerate(time_series):
                dset = f.create_dataset(f'time_{t:03d}', data=volume, compression='gzip')
                dset.attrs['time_hours'] = evolution_data['time_hours'][t]
                dset.attrs['temperature'] = condition['temperature']
                dset.attrs['stress'] = condition['stress']
                dset.attrs['cavity_volume'] = evolution_data['cavity_volume'][t]
                dset.attrs['strain'] = evolution_data['strain'][t]
                dset.attrs['damage'] = evolution_data['damage'][t]
            
            # Save evolution metrics
            f.create_dataset('evolution/time_hours', data=evolution_data['time_hours'])
            f.create_dataset('evolution/cavity_volume', data=evolution_data['cavity_volume'])
            f.create_dataset('evolution/crack_length', data=evolution_data['crack_length'])
            f.create_dataset('evolution/strain', data=evolution_data['strain'])
            f.create_dataset('evolution/damage', data=evolution_data['damage'])
            
            # Add test metadata
            f.attrs['test_conditions'] = json.dumps(condition)
            f.attrs['material'] = 'Cr-Fe-Ni Alloy (Simulated)'
            f.attrs['voxel_size'] = gen.voxel_size
    
    print("\n=== Tomography Data Generation Complete ===")
    print("Data saved in synchrotron_data/tomography/")


if __name__ == '__main__':
    main()