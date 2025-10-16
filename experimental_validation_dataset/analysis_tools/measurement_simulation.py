#!/usr/bin/env python3
"""
Measurement Simulation Tool for SOFC Experimental Validation Dataset

This module provides tools for simulating realistic experimental measurements
including warp field measurements and residual stress measurements with
appropriate noise, uncertainties, and measurement artifacts.

Author: Research Team
Date: 2024-01-15
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import ndimage
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from pathlib import Path

@dataclass
class MeasurementConfig:
    """Configuration for measurement simulation"""
    technique: str
    lateral_resolution: float  # μm
    vertical_resolution: float  # μm or nm
    measurement_area: Tuple[float, float]  # (width, height) in mm
    noise_level: float  # RMS noise level
    systematic_error: float  # Systematic measurement error
    uncertainty: float  # Measurement uncertainty

class WarpMeasurementSimulator:
    """Simulates warp field measurements using various techniques"""
    
    def __init__(self, config: MeasurementConfig):
        self.config = config
        self.techniques = {
            'laser_confocal': self._simulate_laser_confocal,
            'white_light_interferometry': self._simulate_white_light_interferometry,
            'structured_light': self._simulate_structured_light
        }
    
    def simulate_measurement(self, true_warp_field: np.ndarray, 
                           sample_coords: Tuple[np.ndarray, np.ndarray]) -> Dict:
        """
        Simulate warp measurement based on technique
        
        Args:
            true_warp_field: True warp field (Z values) in μm
            sample_coords: (X, Y) coordinate arrays in μm
            
        Returns:
            Dictionary containing simulated measurement data
        """
        if self.config.technique not in self.techniques:
            raise ValueError(f"Unknown technique: {self.config.technique}")
        
        return self.techniques[self.config.technique](true_warp_field, sample_coords)
    
    def _simulate_laser_confocal(self, true_warp: np.ndarray, 
                                coords: Tuple[np.ndarray, np.ndarray]) -> Dict:
        """Simulate laser scanning confocal microscopy measurement"""
        X, Y = coords
        
        # Apply lateral resolution limitation
        lateral_res_um = self.config.lateral_resolution
        vertical_res_um = self.config.vertical_resolution
        
        # Downsample to measurement resolution
        downsample_factor = max(1, int(lateral_res_um))
        if downsample_factor > 1:
            measured_warp = true_warp[::downsample_factor, ::downsample_factor]
            measured_X = X[::downsample_factor, ::downsample_factor]
            measured_Y = Y[::downsample_factor, ::downsample_factor]
        else:
            measured_warp = true_warp.copy()
            measured_X = X.copy()
            measured_Y = Y.copy()
        
        # Add measurement noise
        noise = np.random.normal(0, self.config.noise_level, measured_warp.shape)
        measured_warp += noise
        
        # Add systematic error (e.g., calibration drift)
        systematic_error = np.random.normal(0, self.config.systematic_error, measured_warp.shape)
        measured_warp += systematic_error
        
        # Apply vertical resolution limitation
        measured_warp = np.round(measured_warp / vertical_res_um) * vertical_res_um
        
        return {
            'technique': 'laser_scanning_confocal_microscopy',
            'coordinates': {
                'X': measured_X.tolist(),
                'Y': measured_Y.tolist()
            },
            'warp_field': measured_warp.tolist(),
            'resolution': {
                'lateral': lateral_res_um,
                'vertical': vertical_res_um
            },
            'uncertainty': self.config.uncertainty,
            'noise_level': self.config.noise_level,
            'systematic_error': self.config.systematic_error
        }
    
    def _simulate_white_light_interferometry(self, true_warp: np.ndarray,
                                           coords: Tuple[np.ndarray, np.ndarray]) -> Dict:
        """Simulate white light interferometry measurement"""
        X, Y = coords
        
        # Higher resolution but smaller field of view
        lateral_res_um = self.config.lateral_resolution
        vertical_res_nm = self.config.vertical_resolution
        
        # Select measurement region (typically smaller than full sample)
        region_size = min(4.0, min(X.shape) * lateral_res_um / 1000)  # mm
        center_x, center_y = X.shape[1]//2, X.shape[0]//2
        region_pixels = int(region_size * 1000 / lateral_res_um)
        
        start_x = max(0, center_x - region_pixels//2)
        end_x = min(X.shape[1], start_x + region_pixels)
        start_y = max(0, center_y - region_pixels//2)
        end_y = min(X.shape[0], start_y + region_pixels)
        
        measured_warp = true_warp[start_y:end_y, start_x:end_x]
        measured_X = X[start_y:end_y, start_x:end_x]
        measured_Y = Y[start_y:end_y, start_x:end_x]
        
        # Add high-precision noise
        noise = np.random.normal(0, self.config.noise_level, measured_warp.shape)
        measured_warp += noise
        
        # Add systematic error
        systematic_error = np.random.normal(0, self.config.systematic_error, measured_warp.shape)
        measured_warp += systematic_error
        
        # Apply vertical resolution limitation (in nm)
        measured_warp = np.round(measured_warp * 1000 / vertical_res_nm) * vertical_res_nm / 1000
        
        return {
            'technique': 'white_light_interferometry',
            'coordinates': {
                'X': measured_X.tolist(),
                'Y': measured_Y.tolist()
            },
            'warp_field': measured_warp.tolist(),
            'resolution': {
                'lateral': lateral_res_um,
                'vertical': vertical_res_nm
            },
            'uncertainty': self.config.uncertainty,
            'noise_level': self.config.noise_level,
            'systematic_error': self.config.systematic_error,
            'measurement_region': {
                'center': [float(center_x), float(center_y)],
                'size_mm': region_size
            }
        }
    
    def _simulate_structured_light(self, true_warp: np.ndarray,
                                 coords: Tuple[np.ndarray, np.ndarray]) -> Dict:
        """Simulate structured light 3D scanning measurement"""
        X, Y = coords
        
        # Lower resolution but full surface coverage
        lateral_res_um = self.config.lateral_resolution
        vertical_res_um = self.config.vertical_resolution
        
        # Downsample to measurement resolution
        downsample_factor = max(1, int(lateral_res_um / 10))  # 10x coarser than confocal
        if downsample_factor > 1:
            measured_warp = true_warp[::downsample_factor, ::downsample_factor]
            measured_X = X[::downsample_factor, ::downsample_factor]
            measured_Y = Y[::downsample_factor, ::downsample_factor]
        else:
            measured_warp = true_warp.copy()
            measured_X = X.copy()
            measured_Y = Y.copy()
        
        # Add measurement noise (higher than confocal due to lower resolution)
        noise = np.random.normal(0, self.config.noise_level * 2, measured_warp.shape)
        measured_warp += noise
        
        # Add systematic error
        systematic_error = np.random.normal(0, self.config.systematic_error, measured_warp.shape)
        measured_warp += systematic_error
        
        # Apply vertical resolution limitation
        measured_warp = np.round(measured_warp / vertical_res_um) * vertical_res_um
        
        # Simulate missing data points (holes in scan)
        missing_data_mask = np.random.random(measured_warp.shape) < 0.05  # 5% missing data
        measured_warp[missing_data_mask] = np.nan
        
        return {
            'technique': 'structured_light_3d_scanning',
            'coordinates': {
                'X': measured_X.tolist(),
                'Y': measured_Y.tolist()
            },
            'warp_field': measured_warp.tolist(),
            'resolution': {
                'lateral': lateral_res_um,
                'vertical': vertical_res_um
            },
            'uncertainty': self.config.uncertainty,
            'noise_level': self.config.noise_level,
            'systematic_error': self.config.systematic_error,
            'missing_data_percentage': 5.0
        }

class StressMeasurementSimulator:
    """Simulates residual stress measurements using various techniques"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.techniques = {
            'curvature_method': self._simulate_curvature_method,
            'layer_removal': self._simulate_layer_removal,
            'xrd': self._simulate_xrd,
            'raman': self._simulate_raman
        }
    
    def simulate_measurement(self, true_stress_field: np.ndarray,
                           measurement_points: List[Tuple[float, float]]) -> Dict:
        """
        Simulate stress measurement based on technique
        
        Args:
            true_stress_field: True stress field in MPa
            measurement_points: List of (x, y) coordinates for point measurements
            
        Returns:
            Dictionary containing simulated measurement data
        """
        technique = self.config['technique']
        if technique not in self.techniques:
            raise ValueError(f"Unknown technique: {technique}")
        
        return self.techniques[technique](true_stress_field, measurement_points)
    
    def _simulate_curvature_method(self, true_stress: np.ndarray,
                                  points: List[Tuple[float, float]]) -> Dict:
        """Simulate curvature-based stress measurement"""
        # Curvature method gives through-thickness average stress
        # Simulate by taking average of stress field
        avg_stress = np.mean(true_stress)
        
        # Add measurement uncertainty
        uncertainty = self.config.get('uncertainty', 0.10)  # 10% uncertainty
        noise = np.random.normal(0, avg_stress * uncertainty)
        measured_stress = avg_stress + noise
        
        return {
            'technique': 'curvature_based_inverse_method',
            'stress_value': float(measured_stress),
            'stress_type': 'through_thickness_average',
            'uncertainty': uncertainty,
            'measurement_area': 'full_sample',
            'units': 'MPa'
        }
    
    def _simulate_layer_removal(self, true_stress: np.ndarray,
                               points: List[Tuple[float, float]]) -> Dict:
        """Simulate layer removal stress measurement"""
        # Layer removal gives through-thickness stress gradient
        # Simulate by creating stress profile through thickness
        thickness_layers = 10  # 10 layers through thickness
        stress_profile = []
        
        for i in range(thickness_layers):
            # Simulate stress at each layer (decreasing from surface)
            layer_stress = np.mean(true_stress) * (1 - i / thickness_layers)
            uncertainty = self.config.get('uncertainty', 0.05)  # 5% uncertainty
            noise = np.random.normal(0, layer_stress * uncertainty)
            stress_profile.append(layer_stress + noise)
        
        return {
            'technique': 'layer_removal_warp_measurement',
            'stress_profile': stress_profile,
            'stress_type': 'through_thickness_gradient',
            'layer_thickness': 50.0,  # μm per layer
            'uncertainty': self.config.get('uncertainty', 0.05),
            'measurement_type': 'destructive',
            'units': 'MPa'
        }
    
    def _simulate_xrd(self, true_stress: np.ndarray,
                     points: List[Tuple[float, float]]) -> Dict:
        """Simulate XRD stress measurement"""
        # XRD gives point-wise stress measurements
        measured_stresses = []
        
        for x, y in points:
            # Interpolate stress at measurement point
            stress_value = self._interpolate_stress_at_point(true_stress, x, y)
            
            # Add measurement uncertainty
            uncertainty = self.config.get('uncertainty', 0.15)  # 15% uncertainty
            noise = np.random.normal(0, stress_value * uncertainty)
            measured_stress = stress_value + noise
            
            measured_stresses.append({
                'x': x,
                'y': y,
                'stress': float(measured_stress),
                'uncertainty': uncertainty
            })
        
        return {
            'technique': 'x_ray_diffraction',
            'measurements': measured_stresses,
            'stress_type': 'point_wise',
            'spatial_resolution': 100.0,  # μm
            'units': 'MPa'
        }
    
    def _simulate_raman(self, true_stress: np.ndarray,
                       points: List[Tuple[float, float]]) -> Dict:
        """Simulate Raman spectroscopy stress measurement"""
        # Raman gives local stress measurements with high spatial resolution
        measured_stresses = []
        
        for x, y in points:
            # Interpolate stress at measurement point
            stress_value = self._interpolate_stress_at_point(true_stress, x, y)
            
            # Add measurement uncertainty (higher than XRD)
            uncertainty = self.config.get('uncertainty', 0.20)  # 20% uncertainty
            noise = np.random.normal(0, stress_value * uncertainty)
            measured_stress = stress_value + noise
            
            measured_stresses.append({
                'x': x,
                'y': y,
                'stress': float(measured_stress),
                'uncertainty': uncertainty
            })
        
        return {
            'technique': 'raman_spectroscopy',
            'measurements': measured_stresses,
            'stress_type': 'local_measurements',
            'spatial_resolution': 1.0,  # μm
            'units': 'MPa'
        }
    
    def _interpolate_stress_at_point(self, stress_field: np.ndarray,
                                   x: float, y: float) -> float:
        """Interpolate stress value at specific point"""
        # Simple bilinear interpolation
        # This is a simplified version - in practice would use proper interpolation
        h, w = stress_field.shape
        x_idx = int(x * w / 100)  # Assuming 100mm sample
        y_idx = int(y * h / 100)
        
        # Clamp to valid indices
        x_idx = max(0, min(w-1, x_idx))
        y_idx = max(0, min(h-1, y_idx))
        
        return float(stress_field[y_idx, x_idx])

def generate_sample_data(sample_id: str, fabrication_params: Dict) -> Dict:
    """
    Generate synthetic sample data based on fabrication parameters
    
    Args:
        sample_id: Unique sample identifier
        fabrication_params: Fabrication parameters from fabrication plan
        
    Returns:
        Dictionary containing complete sample data
    """
    # Generate sample geometry
    sample_size = 100.0  # mm
    resolution = 0.1  # mm per pixel
    n_pixels = int(sample_size / resolution)
    
    # Create coordinate arrays
    x = np.linspace(0, sample_size, n_pixels)
    y = np.linspace(0, sample_size, n_pixels)
    X, Y = np.meshgrid(x, y)
    
    # Generate synthetic warp field based on fabrication parameters
    warp_field = generate_synthetic_warp_field(X, Y, fabrication_params)
    
    # Generate synthetic stress field
    stress_field = generate_synthetic_stress_field(X, Y, fabrication_params)
    
    # Simulate measurements
    sample_data = {
        'sample_id': sample_id,
        'fabrication_params': fabrication_params,
        'geometry': {
            'length': sample_size,
            'width': sample_size,
            'thickness': fabrication_params.get('total_thickness', 0.5)
        },
        'measurements': {}
    }
    
    # Simulate warp measurements
    warp_configs = [
        MeasurementConfig('laser_confocal', 1.0, 0.1, (100, 100), 0.5, 0.1, 0.1),
        MeasurementConfig('white_light_interferometry', 0.5, 0.001, (4, 4), 0.1, 0.05, 0.05),
        MeasurementConfig('structured_light', 10.0, 5.0, (100, 100), 2.0, 0.5, 1.0)
    ]
    
    for config in warp_configs:
        simulator = WarpMeasurementSimulator(config)
        measurement_data = simulator.simulate_measurement(warp_field, (X, Y))
        sample_data['measurements'][config.technique] = measurement_data
    
    # Simulate stress measurements
    stress_configs = [
        {'technique': 'curvature_method', 'uncertainty': 0.10},
        {'technique': 'layer_removal', 'uncertainty': 0.05},
        {'technique': 'xrd', 'uncertainty': 0.15},
        {'technique': 'raman', 'uncertainty': 0.20}
    ]
    
    # Define measurement points
    measurement_points = [
        (25, 25), (50, 25), (75, 25),
        (25, 50), (50, 50), (75, 50),
        (25, 75), (50, 75), (75, 75)
    ]
    
    for config in stress_configs:
        simulator = StressMeasurementSimulator(config)
        measurement_data = simulator.simulate_measurement(stress_field, measurement_points)
        sample_data['measurements'][config['technique']] = measurement_data
    
    return sample_data

def generate_synthetic_warp_field(X: np.ndarray, Y: np.ndarray, 
                                params: Dict) -> np.ndarray:
    """Generate synthetic warp field based on fabrication parameters"""
    # Base warp pattern (saddle shape typical of SOFC warping)
    center_x, center_y = X.shape[1]//2, X.shape[0]//2
    
    # Distance from center
    R = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Warp magnitude based on fabrication parameters
    thickness_ratio = params.get('thickness_ratio', 0.3)
    sintering_temp = params.get('sintering_temperature', 1300)
    cooling_rate = params.get('cooling_rate', 5)
    
    # Calculate warp magnitude (in μm)
    base_warp = 100 * thickness_ratio * (sintering_temp - 1200) / 200 * (cooling_rate / 5)
    
    # Create saddle-shaped warp pattern
    warp_field = base_warp * (R / 50)**2 * np.sin(2 * np.arctan2(Y - center_y, X - center_x))
    
    # Add some random variation
    noise = np.random.normal(0, base_warp * 0.1, warp_field.shape)
    warp_field += noise
    
    return warp_field

def generate_synthetic_stress_field(X: np.ndarray, Y: np.ndarray,
                                  params: Dict) -> np.ndarray:
    """Generate synthetic stress field based on fabrication parameters"""
    # Base stress pattern (higher at edges, lower in center)
    center_x, center_y = X.shape[1]//2, X.shape[0]//2
    
    # Distance from center
    R = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Stress magnitude based on fabrication parameters
    thickness_ratio = params.get('thickness_ratio', 0.3)
    sintering_temp = params.get('sintering_temperature', 1300)
    cooling_rate = params.get('cooling_rate', 5)
    
    # Calculate base stress (in MPa)
    base_stress = 50 + 100 * thickness_ratio * (sintering_temp - 1200) / 200 * (cooling_rate / 5)
    
    # Create stress pattern (higher at edges)
    stress_field = base_stress * (1 + R / 50)
    
    # Add some random variation
    noise = np.random.normal(0, base_stress * 0.1, stress_field.shape)
    stress_field += noise
    
    return stress_field

def main():
    """Main function to generate sample data"""
    # Load fabrication plan
    with open('/workspace/experimental_validation_dataset/metadata/fabrication_plan.json', 'r') as f:
        fabrication_plan = json.load(f)
    
    # Generate samples for each group
    sample_id = 1
    for group in fabrication_plan['fabrication_plan']['sample_design']['sample_groups']:
        group_id = group['group_id']
        num_samples = group['samples']
        
        print(f"Generating {num_samples} samples for group: {group_id}")
        
        for i in range(num_samples):
            sample_name = f"sample_{sample_id:03d}"
            
            # Create fabrication parameters for this sample
            fabrication_params = {
                'group_id': group_id,
                'sample_number': i + 1,
                'thickness_ratio': np.random.uniform(
                    group['parameters']['thickness_ratio_range'][0],
                    group['parameters']['thickness_ratio_range'][1]
                ) if 'thickness_ratio_range' in group['parameters'] else 0.3,
                'sintering_temperature': np.random.uniform(
                    group['parameters']['sintering_temperature'] - 10,
                    group['parameters']['sintering_temperature'] + 10
                ) if 'sintering_temperature' in group['parameters'] else 1300,
                'cooling_rate': np.random.uniform(
                    group['parameters']['cooling_rate'] - 1,
                    group['parameters']['cooling_rate'] + 1
                ) if 'cooling_rate' in group['parameters'] else 5,
                'atmosphere': group['parameters'].get('atmosphere', 'Air'),
                'total_thickness': 0.5 + np.random.uniform(-0.1, 0.1)
            }
            
            # Generate sample data
            sample_data = generate_sample_data(sample_name, fabrication_params)
            
            # Save sample data
            sample_dir = f"/workspace/experimental_validation_dataset/fabricated_samples/{sample_name}"
            os.makedirs(sample_dir, exist_ok=True)
            
            with open(f"{sample_dir}/geometry.json", 'w') as f:
                json.dump(sample_data['geometry'], f, indent=2)
            
            with open(f"{sample_dir}/fabrication_params.json", 'w') as f:
                json.dump(sample_data['fabrication_params'], f, indent=2)
            
            # Save measurements
            os.makedirs(f"{sample_dir}/warp_measurements", exist_ok=True)
            os.makedirs(f"{sample_dir}/stress_measurements", exist_ok=True)
            
            for technique, data in sample_data['measurements'].items():
                if technique in ['laser_confocal', 'white_light_interferometry', 'structured_light']:
                    filename = f"{sample_dir}/warp_measurements/{technique}.json"
                else:
                    filename = f"{sample_dir}/stress_measurements/{technique}.json"
                
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
            
            sample_id += 1
    
    print(f"Generated {sample_id - 1} samples total")

if __name__ == "__main__":
    main()