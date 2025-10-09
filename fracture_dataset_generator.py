#!/usr/bin/env python3
"""
Ground Truth Fracture Dataset Generator for SOFC PINN Training
==============================================================

This module generates synthetic "ground truth" fracture data for training Physics-Informed 
Neural Networks (PINNs) and validating fracture prediction models in Solid Oxide Fuel Cells.

The dataset includes:
1. In-situ crack evolution data (3D tomographic time-series)
2. Ex-situ post-mortem analysis data (SEM/FIB-like images)
3. Macroscopic performance degradation data correlated with delamination

Author: AI Assistant
Date: 2025-10-09
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import h5py
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class FractureDatasetGenerator:
    """
    Generates synthetic fracture datasets for SOFC analysis.
    """
    
    def __init__(self, seed=42):
        """Initialize the dataset generator with reproducible random seed."""
        np.random.seed(seed)
        self.seed = seed
        
        # Physical parameters based on YSZ electrolyte properties
        self.material_props = {
            'youngs_modulus': 170e9,  # Pa at 800°C
            'poisson_ratio': 0.23,
            'thermal_expansion': 10.5e-6,  # K^-1
            'fracture_toughness': 3.0e6,  # Pa*m^0.5
            'characteristic_length': 150e-6,  # electrolyte thickness (m)
            'operating_temp': 800 + 273.15,  # K
            'stress_threshold': 165e6,  # Pa (flexural strength)
        }
        
        # Dataset parameters
        self.dataset_params = {
            'grid_size': (64, 64, 32),  # 3D voxel grid (reduced for faster generation)
            'time_steps': 25,  # temporal resolution (reduced)
            'voxel_size': 2.34e-6,  # meters per voxel (150μm / 64)
            'time_step': 3600,  # seconds (1 hour intervals)
            'num_samples': 10,  # number of different crack scenarios (reduced for demo)
        }
        
    def generate_phase_field_evolution(self, crack_id=0):
        """
        Generate 3D phase-field evolution data representing crack propagation.
        
        Args:
            crack_id: Unique identifier for this crack scenario
            
        Returns:
            dict: Contains 4D phase field data (x, y, z, t) and metadata
        """
        nx, ny, nz = self.dataset_params['grid_size']
        nt = self.dataset_params['time_steps']
        
        # Initialize phase field (0 = intact material, 1 = fully cracked)
        phase_field = np.zeros((nx, ny, nz, nt))
        
        # Create initial crack nucleation sites based on stress concentrations
        nucleation_sites = self._generate_nucleation_sites(crack_id)
        
        # Simulate crack evolution using modified Allen-Cahn equation
        for t in range(nt):
            if t == 0:
                # Initialize with small perturbations at nucleation sites
                for site in nucleation_sites:
                    x, y, z = site['position']
                    radius = site['initial_radius']
                    self._add_initial_crack(phase_field[:,:,:,t], x, y, z, radius)
            else:
                # Evolve phase field based on fracture mechanics
                phase_field[:,:,:,t] = self._evolve_phase_field(
                    phase_field[:,:,:,t-1], t, crack_id
                )
        
        # Add realistic noise and artifacts
        phase_field = self._add_tomography_artifacts(phase_field)
        
        return {
            'phase_field': phase_field,
            'crack_id': crack_id,
            'nucleation_sites': nucleation_sites,
            'material_properties': self.material_props,
            'grid_parameters': {
                'dimensions': self.dataset_params['grid_size'],
                'voxel_size': self.dataset_params['voxel_size'],
                'time_step': self.dataset_params['time_step'],
            },
            'physical_time': np.arange(nt) * self.dataset_params['time_step'],
        }
    
    def _generate_nucleation_sites(self, crack_id):
        """Generate realistic crack nucleation sites based on stress analysis."""
        np.random.seed(self.seed + crack_id)
        
        # Number of nucleation sites (typically 1-5 for electrolyte)
        n_sites = np.random.randint(1, 4)
        sites = []
        
        nx, ny, nz = self.dataset_params['grid_size']
        
        for i in range(n_sites):
            # Prefer edge and interface locations for nucleation
            if np.random.random() < 0.7:  # Edge nucleation
                if np.random.random() < 0.5:
                    x = np.random.choice([5, nx-5])  # Near x edges
                    y = np.random.randint(10, ny-10)
                else:
                    x = np.random.randint(10, nx-10)
                    y = np.random.choice([5, ny-5])  # Near y edges
                z = np.random.randint(5, nz-5)
            else:  # Bulk nucleation (defects, inclusions)
                x = np.random.randint(20, nx-20)
                y = np.random.randint(20, ny-20)
                z = np.random.randint(10, nz-10)
            
            # Initial crack size (sub-voxel to few voxels)
            initial_radius = np.random.uniform(0.5, 2.0)
            
            # Stress intensity factor (drives growth rate)
            stress_intensity = np.random.uniform(0.8, 1.5) * self.material_props['fracture_toughness']
            
            sites.append({
                'position': (x, y, z),
                'initial_radius': initial_radius,
                'stress_intensity': stress_intensity,
                'nucleation_time': np.random.randint(0, 5),  # Time step when nucleation occurs
            })
        
        return sites
    
    def _add_initial_crack(self, phase_field, x, y, z, radius):
        """Add initial crack at nucleation site."""
        nx, ny, nz = phase_field.shape
        
        # Create spherical crack region
        for i in range(max(0, int(x-radius-1)), min(nx, int(x+radius+2))):
            for j in range(max(0, int(y-radius-1)), min(ny, int(y+radius+2))):
                for k in range(max(0, int(z-radius-1)), min(nz, int(z+radius+2))):
                    dist = np.sqrt((i-x)**2 + (j-y)**2 + (k-z)**2)
                    if dist <= radius:
                        # Smooth transition using tanh function
                        phase_field[i,j,k] = 0.5 * (1 + np.tanh((radius - dist) / 0.3))
    
    def _evolve_phase_field(self, prev_phase, time_step, crack_id):
        """Evolve phase field using fracture mechanics principles."""
        # Parameters for phase field evolution
        gamma = 2.7e-3  # Surface energy density (J/m²)
        gc = 2.0 * gamma / self.material_props['characteristic_length']
        
        # Mobility parameter (controls crack speed)
        mobility = 1e-6 * (1 + 0.1 * np.sin(0.1 * time_step))  # Add temporal variation
        
        # Compute gradients
        grad_x = np.gradient(prev_phase, axis=0)
        grad_y = np.gradient(prev_phase, axis=1)
        grad_z = np.gradient(prev_phase, axis=2)
        grad_mag_sq = grad_x**2 + grad_y**2 + grad_z**2
        
        # Laplacian
        laplacian = ndimage.laplace(prev_phase)
        
        # Stress-driven term (simplified)
        stress_field = self._compute_stress_field(prev_phase, time_step)
        driving_force = stress_field * (1 - prev_phase)**2
        
        # Phase field evolution equation (Allen-Cahn type)
        dt = 0.1  # Normalized time step
        l_c = self.material_props['characteristic_length'] / self.dataset_params['voxel_size']
        
        dpdt = mobility * (
            gc * l_c * laplacian - 
            gc / l_c * prev_phase * (1 - prev_phase) * (1 - 2*prev_phase) +
            driving_force
        )
        
        # Update phase field
        new_phase = prev_phase + dt * dpdt
        
        # Ensure bounds [0, 1]
        new_phase = np.clip(new_phase, 0, 1)
        
        # Add stochastic fluctuations
        noise = np.random.normal(0, 0.01, new_phase.shape)
        new_phase += noise * (new_phase > 0.1) * (new_phase < 0.9)
        new_phase = np.clip(new_phase, 0, 1)
        
        return new_phase
    
    def _compute_stress_field(self, phase_field, time_step):
        """Compute simplified stress field for crack driving force."""
        # Thermal stress component
        thermal_stress = self.material_props['youngs_modulus'] * \
                        self.material_props['thermal_expansion'] * 50  # 50K temperature variation
        
        # Mechanical stress concentrations near crack tips
        stress_field = np.ones_like(phase_field) * thermal_stress
        
        # Enhance stress near crack tips (phase_field ≈ 0.5)
        crack_tip_mask = (phase_field > 0.1) & (phase_field < 0.9)
        stress_field[crack_tip_mask] *= (2.0 + np.sin(0.05 * time_step))
        
        # Add spatial stress variations
        nx, ny, nz = phase_field.shape
        x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
        
        # Edge stress concentrations
        edge_factor = 1.0 + 0.5 * np.exp(-np.minimum(
            np.minimum(x, nx-x), np.minimum(y, ny-y)
        ) / 10.0)
        
        stress_field *= edge_factor
        
        return stress_field / self.material_props['stress_threshold']  # Normalize
    
    def _add_tomography_artifacts(self, phase_field):
        """Add realistic synchrotron X-ray tomography artifacts."""
        # Ring artifacts (common in tomography)
        for t in range(phase_field.shape[3]):
            for z in range(phase_field.shape[2]):
                # Add subtle ring patterns
                center_x, center_y = phase_field.shape[0]//2, phase_field.shape[1]//2
                x, y = np.meshgrid(np.arange(phase_field.shape[0]), 
                                  np.arange(phase_field.shape[1]), indexing='ij')
                r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                ring_artifact = 0.02 * np.sin(0.3 * r) * np.exp(-r/50)
                phase_field[:,:,z,t] += ring_artifact
        
        # Beam hardening effects
        phase_field *= (1 + 0.05 * np.random.random(phase_field.shape))
        
        # Poisson noise (ensure positive values)
        poisson_data = np.abs(phase_field) * 100 + 1e-6  # Add small offset to avoid zero
        phase_field += np.random.poisson(poisson_data) / 100 - poisson_data / 100
        
        # Ensure bounds
        phase_field = np.clip(phase_field, 0, 1)
        
        return phase_field
    
    def generate_sem_postmortem_data(self, phase_field_data, num_images=20):
        """
        Generate synthetic SEM/FIB post-mortem analysis images.
        
        Args:
            phase_field_data: Final state phase field data
            num_images: Number of SEM images to generate
            
        Returns:
            dict: SEM images and analysis data
        """
        final_phase = phase_field_data['phase_field'][:,:,:,-1]
        
        sem_data = {
            'images': [],
            'crack_measurements': [],
            'microstructure_data': [],
            'imaging_parameters': {
                'accelerating_voltage': '15 kV',
                'working_distance': '8.5 mm',
                'magnification_range': '500x - 50000x',
                'resolution': '2048x2048 pixels',
                'pixel_size_nm': 50,  # nanometers per pixel
            }
        }
        
        for img_id in range(num_images):
            # Select random cross-section through the 3D volume
            section_type = np.random.choice(['xy', 'xz', 'yz'])
            
            if section_type == 'xy':
                z_slice = np.random.randint(10, final_phase.shape[2]-10)
                section = final_phase[:, :, z_slice]
                normal = 'z'
            elif section_type == 'xz':
                y_slice = np.random.randint(10, final_phase.shape[1]-10)
                section = final_phase[:, y_slice, :]
                normal = 'y'
            else:  # yz
                x_slice = np.random.randint(10, final_phase.shape[0]-10)
                section = final_phase[x_slice, :, :]
                normal = 'x'
            
            # Generate SEM-like image
            sem_image = self._generate_sem_image(section, img_id)
            
            # Analyze crack features
            crack_analysis = self._analyze_crack_features(section, sem_image)
            
            # Generate microstructure data
            microstructure = self._generate_microstructure_data(section)
            
            sem_data['images'].append({
                'image_id': img_id,
                'image_data': sem_image,
                'section_type': section_type,
                'section_position': locals()[f'{normal.lower()}_slice'] if section_type != 'xy' else z_slice,
                'magnification': np.random.choice([1000, 2000, 5000, 10000, 20000]),
                'contrast': np.random.uniform(0.7, 1.3),
                'brightness': np.random.uniform(0.8, 1.2),
            })
            
            sem_data['crack_measurements'].append(crack_analysis)
            sem_data['microstructure_data'].append(microstructure)
        
        return sem_data
    
    def _generate_sem_image(self, phase_section, img_id):
        """Generate realistic SEM image from phase field section."""
        # Upsample for higher resolution SEM image
        target_size = (512, 512)
        from scipy.ndimage import zoom
        
        zoom_factors = (target_size[0] / phase_section.shape[0], 
                       target_size[1] / phase_section.shape[1])
        upsampled = zoom(phase_section, zoom_factors, order=1)
        
        # Convert phase field to SEM contrast
        # Cracks appear dark, material appears bright
        sem_image = 1.0 - upsampled
        
        # Add material contrast variations
        grain_structure = self._generate_grain_structure(target_size)
        sem_image *= (0.8 + 0.2 * grain_structure)
        
        # Add SEM-specific artifacts
        # Charging effects
        charging = 0.1 * np.random.random(target_size)
        sem_image += charging
        
        # Edge enhancement (typical in SEM)
        from scipy import ndimage
        edges = ndimage.sobel(sem_image)
        sem_image += 0.1 * edges
        
        # Noise (shot noise, thermal noise)
        noise = np.random.normal(0, 0.05, target_size)
        sem_image += noise
        
        # Normalize to [0, 1]
        sem_image = np.clip(sem_image, 0, 1)
        
        return sem_image
    
    def _generate_grain_structure(self, image_size):
        """Generate realistic polycrystalline grain structure."""
        # Voronoi-like grain structure
        n_grains = np.random.randint(50, 150)
        
        # Random grain centers
        centers = np.random.random((n_grains, 2)) * np.array(image_size)
        
        # Create coordinate grid
        x, y = np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1]), indexing='ij')
        coords = np.stack([x.ravel(), y.ravel()], axis=1)
        
        # Assign each pixel to nearest grain center
        distances = cdist(coords, centers)
        grain_ids = np.argmin(distances, axis=1)
        grain_map = grain_ids.reshape(image_size)
        
        # Assign random contrast to each grain
        grain_contrasts = np.random.uniform(0.7, 1.3, n_grains)
        grain_structure = grain_contrasts[grain_map]
        
        # Smooth grain boundaries
        grain_structure = ndimage.gaussian_filter(grain_structure, sigma=1.0)
        
        return grain_structure
    
    def _analyze_crack_features(self, phase_section, sem_image):
        """Analyze crack features from SEM image."""
        # Convert to physical units (micrometers)
        voxel_size_um = self.dataset_params['voxel_size'] * 1e6
        
        # Threshold to identify cracks
        crack_mask = phase_section > 0.5
        
        # Measure crack properties
        from scipy import ndimage
        from skimage import measure
        
        # Label connected crack regions
        labeled_cracks, n_cracks = ndimage.label(crack_mask)
        
        crack_features = []
        for i in range(1, n_cracks + 1):
            crack_region = (labeled_cracks == i)
            
            # Measure geometric properties
            props = measure.regionprops(crack_region.astype(int))[0]
            
            features = {
                'crack_id': i,
                'area_um2': props.area * voxel_size_um**2,
                'perimeter_um': props.perimeter * voxel_size_um,
                'major_axis_um': props.major_axis_length * voxel_size_um,
                'minor_axis_um': props.minor_axis_length * voxel_size_um,
                'eccentricity': props.eccentricity,
                'orientation_deg': np.degrees(props.orientation),
                'centroid': props.centroid,
                'bbox': props.bbox,
            }
            
            # Crack opening displacement (COD)
            features['mean_cod_um'] = np.mean(phase_section[crack_region]) * voxel_size_um
            features['max_cod_um'] = np.max(phase_section[crack_region]) * voxel_size_um
            
            crack_features.append(features)
        
        return {
            'num_cracks': n_cracks,
            'total_crack_area_um2': sum(f['area_um2'] for f in crack_features),
            'crack_density_per_mm2': n_cracks / (np.prod(phase_section.shape) * voxel_size_um**2 * 1e-6),
            'individual_cracks': crack_features,
        }
    
    def _generate_microstructure_data(self, phase_section):
        """Generate microstructure characterization data."""
        voxel_size_um = self.dataset_params['voxel_size'] * 1e6
        
        # Porosity analysis
        total_area = np.prod(phase_section.shape) * voxel_size_um**2
        crack_area = np.sum(phase_section > 0.1) * voxel_size_um**2
        porosity = crack_area / total_area
        
        # Grain size analysis (simulated)
        avg_grain_size = np.random.uniform(2.0, 8.0)  # micrometers
        grain_size_std = avg_grain_size * 0.3
        
        # Interface properties
        interface_roughness = np.random.uniform(0.1, 0.5)  # micrometers RMS
        
        return {
            'porosity_percent': porosity * 100,
            'average_grain_size_um': avg_grain_size,
            'grain_size_std_um': grain_size_std,
            'interface_roughness_um_rms': interface_roughness,
            'phase_composition': {
                'YSZ_volume_fraction': 1.0 - porosity,
                'void_volume_fraction': porosity,
            }
        }
    
    def generate_performance_degradation_data(self, phase_field_data, operating_hours=5000):
        """
        Generate macroscopic performance degradation data correlated with delamination.
        
        Args:
            phase_field_data: Time-series phase field data
            operating_hours: Total operating time in hours
            
        Returns:
            dict: Performance degradation measurements
        """
        time_steps = phase_field_data['physical_time'] / 3600  # Convert to hours
        phase_evolution = phase_field_data['phase_field']
        
        # Calculate delamination area over time
        voxel_volume = self.dataset_params['voxel_size']**3
        delamination_area = []
        
        for t in range(phase_evolution.shape[3]):
            # Count cracked voxels (phase > 0.5)
            cracked_voxels = np.sum(phase_evolution[:,:,:,t] > 0.5)
            # Convert to area (assuming cracks are primarily planar)
            area_m2 = cracked_voxels * voxel_volume**(2/3)
            delamination_area.append(area_m2)
        
        delamination_area = np.array(delamination_area)
        
        # Generate correlated performance data
        performance_data = self._generate_electrochemical_performance(
            time_steps, delamination_area
        )
        
        # Add mechanical degradation metrics
        mechanical_data = self._generate_mechanical_degradation(
            time_steps, delamination_area
        )
        
        # Combine all degradation data
        degradation_data = {
            'time_hours': time_steps,
            'delamination_area_m2': delamination_area,
            'electrochemical_performance': performance_data,
            'mechanical_properties': mechanical_data,
            'operating_conditions': {
                'temperature_C': 800,
                'current_density_A_cm2': 0.5,
                'fuel_utilization': 0.85,
                'air_utilization': 0.25,
            },
            'correlations': self._calculate_degradation_correlations(
                delamination_area, performance_data, mechanical_data
            )
        }
        
        return degradation_data
    
    def _generate_electrochemical_performance(self, time_hours, delamination_area):
        """Generate electrochemical performance degradation."""
        # Initial performance values
        initial_voltage = 0.75  # V at 0.5 A/cm²
        initial_asr = 0.15  # Ω·cm² (area-specific resistance)
        initial_power_density = 375  # mW/cm²
        
        # Degradation mechanisms
        # 1. Ohmic resistance increase due to delamination
        relative_delamination = delamination_area / delamination_area[-1] if delamination_area[-1] > 0 else 0
        ohmic_degradation = 1 + 2.0 * relative_delamination  # 200% increase at full delamination
        
        # 2. Time-dependent degradation (independent of delamination)
        time_degradation = 1 + 0.0001 * time_hours  # 1% per 100 hours baseline
        
        # 3. Thermal cycling effects (periodic)
        thermal_cycles = time_hours / 24  # Daily cycles
        cycling_degradation = 1 + 0.05 * np.sin(2 * np.pi * thermal_cycles / 30) * (time_hours / 1000)
        
        # Calculate performance metrics
        asr = initial_asr * ohmic_degradation * time_degradation * cycling_degradation
        
        # Voltage at constant current (0.5 A/cm²)
        current_density = 0.5
        voltage = initial_voltage - current_density * (asr - initial_asr)
        
        # Power density
        power_density = voltage * current_density * 1000  # mW/cm²
        
        # Add measurement noise
        voltage += np.random.normal(0, 0.005, len(voltage))
        asr += np.random.normal(0, 0.002, len(asr))
        power_density += np.random.normal(0, 5, len(power_density))
        
        return {
            'voltage_V': voltage,
            'area_specific_resistance_ohm_cm2': asr,
            'power_density_mW_cm2': power_density,
            'degradation_rate_mV_per_1000h': np.gradient(voltage * 1000, time_hours) * 1000,
            'efficiency_percent': voltage / 1.25 * 100,  # Theoretical max ~1.25V
        }
    
    def _generate_mechanical_degradation(self, time_hours, delamination_area):
        """Generate mechanical property degradation."""
        # Initial mechanical properties
        initial_stiffness = self.material_props['youngs_modulus']
        initial_strength = self.material_props['stress_threshold']
        
        # Degradation due to microcracking
        relative_damage = delamination_area / np.max(delamination_area) if np.max(delamination_area) > 0 else 0
        
        # Stiffness reduction (more sensitive to microcracking)
        stiffness_reduction = 0.3 * relative_damage  # Up to 30% reduction
        effective_stiffness = initial_stiffness * (1 - stiffness_reduction)
        
        # Strength reduction (catastrophic with major cracks)
        strength_reduction = 0.6 * relative_damage**2  # Nonlinear reduction
        effective_strength = initial_strength * (1 - strength_reduction)
        
        # Fracture toughness evolution
        initial_toughness = self.material_props['fracture_toughness']
        toughness_reduction = 0.4 * relative_damage
        effective_toughness = initial_toughness * (1 - toughness_reduction)
        
        # Add measurement uncertainty
        effective_stiffness += np.random.normal(0, 0.05 * initial_stiffness, len(effective_stiffness))
        effective_strength += np.random.normal(0, 0.1 * initial_strength, len(effective_strength))
        effective_toughness += np.random.normal(0, 0.1 * initial_toughness, len(effective_toughness))
        
        return {
            'effective_youngs_modulus_GPa': effective_stiffness / 1e9,
            'effective_strength_MPa': effective_strength / 1e6,
            'effective_fracture_toughness_MPa_m05': effective_toughness / 1e6,
            'damage_parameter': relative_damage,
            'stiffness_retention_percent': (effective_stiffness / initial_stiffness) * 100,
        }
    
    def _calculate_degradation_correlations(self, delamination_area, performance_data, mechanical_data):
        """Calculate correlation coefficients between degradation metrics."""
        correlations = {}
        
        # Correlation between delamination and voltage
        correlations['delamination_voltage'] = np.corrcoef(
            delamination_area, performance_data['voltage_V']
        )[0, 1]
        
        # Correlation between delamination and ASR
        correlations['delamination_asr'] = np.corrcoef(
            delamination_area, performance_data['area_specific_resistance_ohm_cm2']
        )[0, 1]
        
        # Correlation between mechanical and electrical properties
        correlations['stiffness_voltage'] = np.corrcoef(
            mechanical_data['effective_youngs_modulus_GPa'], performance_data['voltage_V']
        )[0, 1]
        
        return correlations
    
    def save_dataset(self, output_dir='fracture_dataset'):
        """
        Generate and save complete fracture dataset.
        
        Args:
            output_dir: Directory to save dataset files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Generating fracture dataset with {self.dataset_params['num_samples']} samples...")
        
        # Generate multiple crack scenarios
        for sample_id in range(self.dataset_params['num_samples']):
            print(f"Generating sample {sample_id + 1}/{self.dataset_params['num_samples']}")
            
            # 1. Generate in-situ crack evolution data
            phase_field_data = self.generate_phase_field_evolution(sample_id)
            
            # 2. Generate ex-situ SEM data
            sem_data = self.generate_sem_postmortem_data(phase_field_data)
            
            # 3. Generate performance degradation data
            performance_data = self.generate_performance_degradation_data(phase_field_data)
            
            # Save individual sample
            sample_dir = output_path / f'sample_{sample_id:03d}'
            sample_dir.mkdir(exist_ok=True)
            
            # Save as HDF5 for efficient storage of large arrays
            with h5py.File(sample_dir / 'phase_field_data.h5', 'w') as f:
                f.create_dataset('phase_field', data=phase_field_data['phase_field'], 
                               compression='gzip')
                f.create_dataset('physical_time', data=phase_field_data['physical_time'])
                f.attrs['crack_id'] = phase_field_data['crack_id']
                
                # Save nucleation sites
                nucleation_group = f.create_group('nucleation_sites')
                for i, site in enumerate(phase_field_data['nucleation_sites']):
                    site_group = nucleation_group.create_group(f'site_{i}')
                    site_group.create_dataset('position', data=site['position'])
                    site_group.attrs['initial_radius'] = site['initial_radius']
                    site_group.attrs['stress_intensity'] = site['stress_intensity']
                    site_group.attrs['nucleation_time'] = site['nucleation_time']
            
            # Save SEM data
            with h5py.File(sample_dir / 'sem_data.h5', 'w') as f:
                images_group = f.create_group('images')
                for i, img_data in enumerate(sem_data['images']):
                    img_group = images_group.create_group(f'image_{i}')
                    img_group.create_dataset('image_data', data=img_data['image_data'], 
                                           compression='gzip')
                    for key, value in img_data.items():
                        if key != 'image_data':
                            # Convert strings to bytes for HDF5 compatibility
                            if isinstance(value, str):
                                img_group.attrs[key] = value.encode('utf-8')
                            else:
                                img_group.attrs[key] = value
            
            # Save metadata as JSON
            metadata = {
                'sample_id': sample_id,
                'material_properties': phase_field_data['material_properties'],
                'grid_parameters': phase_field_data['grid_parameters'],
                'sem_parameters': sem_data['imaging_parameters'],
                'crack_measurements': sem_data['crack_measurements'],
                'microstructure_data': sem_data['microstructure_data'],
            }
            
            with open(sample_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Save performance data as JSON
            with open(sample_dir / 'performance_data.json', 'w') as f:
                json.dump(performance_data, f, indent=2, default=str)
        
        # Create dataset summary
        self._create_dataset_summary(output_path)
        
        print(f"Dataset generation complete! Saved to {output_path}")
    
    def _create_dataset_summary(self, output_path):
        """Create summary documentation for the dataset."""
        summary = {
            'dataset_info': {
                'name': 'SOFC Ground Truth Fracture Dataset',
                'version': '1.0',
                'creation_date': '2025-10-09',
                'num_samples': self.dataset_params['num_samples'],
                'description': 'Synthetic ground truth fracture data for PINN training and validation',
            },
            'data_types': {
                'in_situ_crack_evolution': {
                    'format': 'HDF5',
                    'dimensions': '4D (x, y, z, t)',
                    'resolution': self.dataset_params['grid_size'],
                    'temporal_resolution': f"{self.dataset_params['time_step']} seconds",
                    'physical_size': f"{self.dataset_params['voxel_size']*1e6:.2f} μm/voxel",
                },
                'ex_situ_sem_images': {
                    'format': 'HDF5',
                    'resolution': '512x512 pixels',
                    'pixel_size': '50 nm/pixel',
                    'num_images_per_sample': 20,
                },
                'performance_degradation': {
                    'format': 'JSON',
                    'metrics': ['voltage', 'ASR', 'power_density', 'mechanical_properties'],
                    'correlations': 'Included with delamination area',
                }
            },
            'physical_parameters': self.material_props,
            'simulation_parameters': self.dataset_params,
        }
        
        with open(output_path / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create README
        readme_content = """# SOFC Ground Truth Fracture Dataset

This dataset contains synthetic "ground truth" fracture data for training Physics-Informed Neural Networks (PINNs) and validating fracture prediction models in Solid Oxide Fuel Cells.

## Dataset Structure

```
fracture_dataset/
├── dataset_summary.json          # Dataset metadata and parameters
├── sample_000/                   # First sample
│   ├── phase_field_data.h5      # 4D crack evolution data
│   ├── sem_data.h5              # SEM/FIB post-mortem images
│   ├── metadata.json            # Sample-specific metadata
│   └── performance_data.json    # Degradation measurements
├── sample_001/                   # Second sample
│   └── ...
└── sample_099/                   # Last sample
    └── ...
```

## Data Types

### 1. In-situ Crack Evolution Data
- **File**: `phase_field_data.h5`
- **Format**: 4D array (x, y, z, t) representing crack phase field
- **Values**: 0 = intact material, 1 = fully cracked
- **Resolution**: 128×128×64 voxels, 1.17 μm/voxel
- **Temporal**: 50 time steps, 1 hour intervals

### 2. Ex-situ Post-mortem Analysis
- **File**: `sem_data.h5`
- **Format**: 2D SEM-like images (512×512 pixels)
- **Resolution**: 50 nm/pixel
- **Content**: Cross-sections through final crack state
- **Analysis**: Crack measurements, microstructure data

### 3. Macroscopic Performance Degradation
- **File**: `performance_data.json`
- **Metrics**: Voltage, ASR, power density, mechanical properties
- **Correlations**: Linked to delamination area evolution
- **Duration**: 5000 hours of simulated operation

## Usage

```python
import h5py
import json

# Load phase field data
with h5py.File('sample_000/phase_field_data.h5', 'r') as f:
    phase_field = f['phase_field'][:]
    time_array = f['physical_time'][:]

# Load performance data
with open('sample_000/performance_data.json', 'r') as f:
    performance = json.load(f)
```

## Physical Basis

The dataset is based on realistic material properties for 8YSZ electrolyte:
- Young's modulus: 170 GPa (at 800°C)
- Thermal expansion: 10.5×10⁻⁶ K⁻¹
- Fracture toughness: 3.0 MPa√m
- Operating temperature: 800°C

## Applications

- PINN training for fracture prediction
- Validation of phase-field models
- Correlation analysis between microstructure and performance
- Development of degradation prediction algorithms

## Citation

If you use this dataset, please cite:
"Synthetic Ground Truth Fracture Dataset for SOFC PINN Training and Validation"
Generated: 2025-10-09
"""
        
        with open(output_path / 'README.md', 'w') as f:
            f.write(readme_content)


def main():
    """Main function to generate the complete fracture dataset."""
    generator = FractureDatasetGenerator(seed=42)
    
    # Generate and save the complete dataset
    generator.save_dataset('fracture_dataset')
    
    # Create visualization script
    create_visualization_script()
    
    print("\nDataset generation complete!")
    print("Run 'python visualize_dataset.py' to explore the generated data.")


def create_visualization_script():
    """Create a script to visualize the generated dataset."""
    viz_script = '''#!/usr/bin/env python3
"""
Visualization script for the SOFC fracture dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
from pathlib import Path

def visualize_sample(sample_id=0):
    """Visualize data from a specific sample."""
    sample_dir = Path(f'fracture_dataset/sample_{sample_id:03d}')
    
    # Load phase field data
    with h5py.File(sample_dir / 'phase_field_data.h5', 'r') as f:
        phase_field = f['phase_field'][:]
        time_array = f['physical_time'][:]
    
    # Load performance data
    with open(sample_dir / 'performance_data.json', 'r') as f:
        performance = json.load(f)
    
    # Load SEM data
    with h5py.File(sample_dir / 'sem_data.h5', 'r') as f:
        first_image = f['images/image_0/image_data'][:]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'SOFC Fracture Dataset - Sample {sample_id}', fontsize=16)
    
    # Phase field evolution
    times_to_show = [0, len(time_array)//4, len(time_array)//2, -1]
    for i, t_idx in enumerate(times_to_show[:3]):
        ax = axes[0, i]
        # Show middle z-slice
        z_mid = phase_field.shape[2] // 2
        im = ax.imshow(phase_field[:, :, z_mid, t_idx], cmap='hot', vmin=0, vmax=1)
        ax.set_title(f't = {time_array[t_idx]/3600:.1f} hours')
        ax.set_xlabel('X (voxels)')
        ax.set_ylabel('Y (voxels)')
        plt.colorbar(im, ax=ax, label='Phase field')
    
    # SEM image
    axes[0, 2].remove()
    ax_sem = fig.add_subplot(2, 3, 3)
    ax_sem.imshow(first_image, cmap='gray')
    ax_sem.set_title('SEM Post-mortem Image')
    ax_sem.set_xlabel('X (pixels)')
    ax_sem.set_ylabel('Y (pixels)')
    
    # Performance degradation
    time_hours = np.array(performance['time_hours'])
    
    ax = axes[1, 0]
    ax.plot(time_hours, performance['electrochemical_performance']['voltage_V'])
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('Voltage Degradation')
    ax.grid(True)
    
    ax = axes[1, 1]
    ax.plot(time_hours, performance['electrochemical_performance']['area_specific_resistance_ohm_cm2'])
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('ASR (Ω·cm²)')
    ax.set_title('Resistance Increase')
    ax.grid(True)
    
    ax = axes[1, 2]
    ax.plot(time_hours, np.array(performance['delamination_area_m2']) * 1e6)  # Convert to mm²
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Delamination Area (mm²)')
    ax.set_title('Crack Area Growth')
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_dataset_statistics():
    """Plot statistics across all samples in the dataset."""
    dataset_dir = Path('fracture_dataset')
    
    # Collect statistics from all samples
    final_crack_areas = []
    final_voltages = []
    num_nucleation_sites = []
    
    for sample_dir in sorted(dataset_dir.glob('sample_*')):
        # Load performance data
        with open(sample_dir / 'performance_data.json', 'r') as f:
            performance = json.load(f)
        
        final_crack_areas.append(performance['delamination_area_m2'][-1] * 1e6)  # mm²
        final_voltages.append(performance['electrochemical_performance']['voltage_V'][-1])
        
        # Load nucleation site count
        with open(sample_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Count nucleation sites from crack measurements
        crack_data = metadata['crack_measurements'][0]  # First SEM image
        num_nucleation_sites.append(crack_data['num_cracks'])
    
    # Create statistics plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Dataset Statistics (All Samples)', fontsize=16)
    
    # Final crack area distribution
    axes[0, 0].hist(final_crack_areas, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Final Crack Area (mm²)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Final Crack Areas')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Final voltage distribution
    axes[0, 1].hist(final_voltages, bins=20, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].set_xlabel('Final Voltage (V)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Final Voltages')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Correlation between crack area and voltage
    axes[1, 0].scatter(final_crack_areas, final_voltages, alpha=0.6)
    axes[1, 0].set_xlabel('Final Crack Area (mm²)')
    axes[1, 0].set_ylabel('Final Voltage (V)')
    axes[1, 0].set_title('Crack Area vs. Voltage Correlation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Calculate and display correlation
    correlation = np.corrcoef(final_crack_areas, final_voltages)[0, 1]
    axes[1, 0].text(0.05, 0.95, f'R = {correlation:.3f}', transform=axes[1, 0].transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Number of nucleation sites
    axes[1, 1].hist(num_nucleation_sites, bins=range(1, max(num_nucleation_sites)+2), 
                   alpha=0.7, edgecolor='black', color='green')
    axes[1, 1].set_xlabel('Number of Nucleation Sites')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Nucleation Sites')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\\nDataset Summary Statistics:")
    print(f"Number of samples: {len(final_crack_areas)}")
    print(f"Final crack area: {np.mean(final_crack_areas):.3f} ± {np.std(final_crack_areas):.3f} mm²")
    print(f"Final voltage: {np.mean(final_voltages):.3f} ± {np.std(final_voltages):.3f} V")
    print(f"Crack-voltage correlation: {correlation:.3f}")
    print(f"Average nucleation sites: {np.mean(num_nucleation_sites):.1f}")

if __name__ == '__main__':
    print("SOFC Fracture Dataset Visualization")
    print("1. Visualizing sample 0...")
    visualize_sample(0)
    
    print("\\n2. Generating dataset statistics...")
    plot_dataset_statistics()
'''
    
    with open('visualize_dataset.py', 'w') as f:
        f.write(viz_script)


if __name__ == '__main__':
    main()