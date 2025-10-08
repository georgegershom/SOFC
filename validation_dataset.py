#!/usr/bin/env python3
"""
FEM Model Validation & Analysis Dataset Generator
================================================

This script generates a comprehensive dataset for validating FEM models
and performing residual analysis on solid oxide fuel cell (SOFC) electrolytes.

Categories covered:
1. Residual Stress State (Experimental)
2. Crack Initiation & Propagation
3. Collocation Point Data (Simulation Output)
4. Multi-Scale Data (Macro, Meso, Micro)
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from scipy import stats
import h5py

class ValidationDatasetGenerator:
    """Generate synthetic validation dataset for FEM model validation"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.dataset_metadata = {
            "generation_date": datetime.now().isoformat(),
            "description": "Synthetic dataset for FEM model validation and residual analysis",
            "material": "YSZ (Yttria-Stabilized Zirconia) SOFC Electrolyte",
            "scale_levels": ["macro", "meso", "micro"]
        }
    
    def generate_macro_scale_data(self) -> Dict[str, Any]:
        """Generate macro-scale experimental data"""
        print("Generating macro-scale experimental data...")
        
        # Bulk material properties
        bulk_properties = {
            "elastic_modulus": {
                "value": 200e9,  # Pa
                "uncertainty": 5e9,
                "measurement_method": "Ultrasonic pulse-echo"
            },
            "poisson_ratio": {
                "value": 0.31,
                "uncertainty": 0.02,
                "measurement_method": "Strain gauge"
            },
            "coefficient_thermal_expansion": {
                "value": 10.5e-6,  # K^-1
                "uncertainty": 0.5e-6,
                "measurement_method": "Dilatometry"
            },
            "density": {
                "value": 6050,  # kg/m^3
                "uncertainty": 50,
                "measurement_method": "Archimedes principle"
            }
        }
        
        # Surface residual stress measurements (XRD)
        xrd_measurements = self._generate_xrd_data()
        
        # Raman spectroscopy data
        raman_data = self._generate_raman_data()
        
        # Cell dimensions and sintering profile
        cell_geometry = {
            "diameter": 25e-3,  # m
            "thickness": 200e-6,  # m
            "active_area": 4.91e-4,  # m^2
            "porosity": 0.05  # 5% porosity
        }
        
        sintering_profile = self._generate_sintering_profile()
        
        return {
            "bulk_properties": bulk_properties,
            "xrd_measurements": xrd_measurements,
            "raman_data": raman_data,
            "cell_geometry": cell_geometry,
            "sintering_profile": sintering_profile
        }
    
    def _generate_xrd_data(self) -> Dict[str, Any]:
        """Generate X-ray diffraction residual stress data"""
        # Surface stress measurements at different locations
        positions = np.linspace(0, 25e-3, 20)  # 20 points across diameter
        stress_values = []
        
        for pos in positions:
            # Simulate stress variation across surface
            base_stress = -150e6  # Compressive stress in Pa
            variation = 50e6 * np.sin(2 * np.pi * pos / 25e-3)
            stress = base_stress + variation + np.random.normal(0, 10e6)
            stress_values.append(stress)
        
        return {
            "positions": positions.tolist(),
            "stress_values": stress_values,
            "measurement_conditions": {
                "x_ray_energy": "8.04 keV (Cu Kα)",
                "penetration_depth": "5-10 μm",
                "beam_size": "1 mm",
                "measurement_angle": "45°"
            },
            "uncertainty": "±15 MPa"
        }
    
    def _generate_raman_data(self) -> Dict[str, Any]:
        """Generate Raman spectroscopy data"""
        # Raman shift data for different stress states
        raman_shifts = {
            "unstressed": 470.5,  # cm^-1
            "compressed": 472.3,  # cm^-1
            "tensile": 468.7     # cm^-1
        }
        
        # Stress calibration curve
        stress_calibration = {
            "slope": -0.8,  # cm^-1/MPa
            "intercept": 470.5,
            "r_squared": 0.95
        }
        
        return {
            "raman_shifts": raman_shifts,
            "stress_calibration": stress_calibration,
            "measurement_conditions": {
                "laser_wavelength": "532 nm",
                "power": "5 mW",
                "spot_size": "2 μm",
                "integration_time": "10 s"
            }
        }
    
    def _generate_sintering_profile(self) -> Dict[str, Any]:
        """Generate sintering temperature profile"""
        time_points = np.linspace(0, 3600, 100)  # 1 hour sintering
        temperature = 1400 + 200 * np.exp(-time_points/1800)  # Cooling profile
        
        return {
            "time": time_points.tolist(),
            "temperature": temperature.tolist(),
            "heating_rate": "5 K/min",
            "peak_temperature": 1600,
            "hold_time": 1200  # seconds
        }
    
    def generate_meso_scale_data(self) -> Dict[str, Any]:
        """Generate meso-scale microstructure data"""
        print("Generating meso-scale microstructure data...")
        
        # SEM/CT scan data simulation
        microstructure = self._generate_microstructure_data()
        
        # Grain size distribution
        grain_sizes = self._generate_grain_size_distribution()
        
        # Porosity analysis
        porosity_data = self._generate_porosity_data()
        
        # Representative Volume Element (RVE) data
        rve_data = self._generate_rve_data()
        
        return {
            "microstructure": microstructure,
            "grain_sizes": grain_sizes,
            "porosity": porosity_data,
            "rve_data": rve_data
        }
    
    def _generate_microstructure_data(self) -> Dict[str, Any]:
        """Generate synthetic microstructure data"""
        # Simulate grain boundaries and pores
        n_grains = 500
        grain_centers = np.random.uniform(0, 100, (n_grains, 2))  # 100x100 μm area
        grain_sizes = np.random.lognormal(2, 0.5, n_grains)  # log-normal distribution
        
        # Generate pore locations
        n_pores = 50
        pore_centers = np.random.uniform(0, 100, (n_pores, 2))
        pore_sizes = np.random.uniform(0.5, 3, n_pores)  # μm
        
        return {
            "grain_centers": grain_centers.tolist(),
            "grain_sizes": grain_sizes.tolist(),
            "pore_centers": pore_centers.tolist(),
            "pore_sizes": pore_sizes.tolist(),
            "domain_size": [100, 100],  # μm
            "total_grains": n_grains,
            "total_pores": n_pores
        }
    
    def _generate_grain_size_distribution(self) -> Dict[str, Any]:
        """Generate grain size distribution data"""
        # Log-normal distribution parameters
        mu, sigma = 2.0, 0.5
        grain_sizes = np.random.lognormal(mu, sigma, 1000)
        
        return {
            "mean_size": np.mean(grain_sizes),
            "std_size": np.std(grain_sizes),
            "median_size": np.median(grain_sizes),
            "distribution": grain_sizes.tolist(),
            "statistics": {
                "mean": np.mean(grain_sizes),
                "std": np.std(grain_sizes),
                "min": np.min(grain_sizes),
                "max": np.max(grain_sizes),
                "percentiles": {
                    "25th": np.percentile(grain_sizes, 25),
                    "75th": np.percentile(grain_sizes, 75),
                    "95th": np.percentile(grain_sizes, 95)
                }
            }
        }
    
    def _generate_porosity_data(self) -> Dict[str, Any]:
        """Generate porosity analysis data"""
        # Porosity distribution across sample
        x_coords = np.linspace(0, 25, 50)  # mm
        y_coords = np.linspace(0, 25, 50)  # mm
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Simulate porosity variation
        porosity = 0.05 + 0.02 * np.sin(2 * np.pi * X / 10) * np.cos(2 * np.pi * Y / 10)
        porosity += np.random.normal(0, 0.005, porosity.shape)
        
        return {
            "coordinates": {"x": X.tolist(), "y": Y.tolist()},
            "porosity_field": porosity.tolist(),
            "average_porosity": np.mean(porosity),
            "porosity_std": np.std(porosity),
            "max_porosity": np.max(porosity),
            "min_porosity": np.min(porosity)
        }
    
    def _generate_rve_data(self) -> Dict[str, Any]:
        """Generate Representative Volume Element data"""
        # RVE dimensions
        rve_size = [50, 50, 10]  # μm
        
        # Stress concentration factors at pores
        stress_concentration_factors = {
            "spherical_pores": 2.0,
            "elliptical_pores": 2.5,
            "crack_like_pores": 5.0
        }
        
        # Grain boundary stress concentrations
        gb_stress_factors = np.random.uniform(1.2, 2.0, 100)
        
        return {
            "rve_dimensions": rve_size,
            "stress_concentration_factors": stress_concentration_factors,
            "gb_stress_factors": gb_stress_factors.tolist(),
            "element_count": 10000,
            "node_count": 12000
        }
    
    def generate_crack_data(self) -> Dict[str, Any]:
        """Generate crack initiation and propagation data"""
        print("Generating crack initiation and propagation data...")
        
        # Crack locations from SEM analysis
        crack_locations = self._generate_crack_locations()
        
        # Critical loads and temperatures
        critical_conditions = self._generate_critical_conditions()
        
        # Crack propagation data
        crack_propagation = self._generate_crack_propagation()
        
        return {
            "crack_locations": crack_locations,
            "critical_conditions": critical_conditions,
            "crack_propagation": crack_propagation
        }
    
    def _generate_crack_locations(self) -> Dict[str, Any]:
        """Generate crack location data from SEM analysis"""
        # Simulate crack locations in microstructure
        n_cracks = 25
        crack_positions = np.random.uniform(0, 100, (n_cracks, 2))  # μm
        crack_lengths = np.random.exponential(5, n_cracks)  # μm
        crack_orientations = np.random.uniform(0, 2*np.pi, n_cracks)
        
        # Crack types
        crack_types = np.random.choice(['intergranular', 'transgranular', 'mixed'], n_cracks)
        
        return {
            "positions": crack_positions.tolist(),
            "lengths": crack_lengths.tolist(),
            "orientations": crack_orientations.tolist(),
            "types": crack_types.tolist(),
            "total_cracks": n_cracks
        }
    
    def _generate_critical_conditions(self) -> Dict[str, Any]:
        """Generate critical load and temperature data"""
        return {
            "critical_temperature": {
                "value": 1200,  # K
                "uncertainty": 50,
                "measurement_method": "Thermal cycling"
            },
            "critical_stress": {
                "value": 200e6,  # Pa
                "uncertainty": 20e6,
                "measurement_method": "Biaxial flexure"
            },
            "critical_strain": {
                "value": 0.001,
                "uncertainty": 0.0001,
                "measurement_method": "Strain gauge"
            },
            "fatigue_life": {
                "cycles_to_failure": 10000,
                "stress_amplitude": 100e6,  # Pa
                "frequency": 1  # Hz
            }
        }
    
    def _generate_crack_propagation(self) -> Dict[str, Any]:
        """Generate crack propagation data"""
        # Crack growth rate data
        stress_intensity_range = np.linspace(1, 10, 20)  # MPa√m
        crack_growth_rate = 1e-12 * (stress_intensity_range ** 3)  # m/cycle
        
        return {
            "stress_intensity_range": stress_intensity_range.tolist(),
            "crack_growth_rate": crack_growth_rate.tolist(),
            "paris_law_params": {
                "C": 1e-12,
                "m": 3.0
            }
        }
    
    def generate_fem_simulation_data(self) -> Dict[str, Any]:
        """Generate FEM simulation output data"""
        print("Generating FEM simulation output data...")
        
        # Full-field simulation data
        full_field_data = self._generate_full_field_data()
        
        # Collocation point data
        collocation_data = self._generate_collocation_data()
        
        return {
            "full_field": full_field_data,
            "collocation_points": collocation_data
        }
    
    def _generate_full_field_data(self) -> Dict[str, Any]:
        """Generate full-field FEM simulation data"""
        # Create a 3D mesh
        nx, ny, nz = 50, 50, 10
        x = np.linspace(0, 25e-3, nx)
        y = np.linspace(0, 25e-3, ny)
        z = np.linspace(0, 200e-6, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Generate temperature field
        T = 800 + 200 * np.exp(-Z / 100e-6) + 50 * np.sin(2 * np.pi * X / 25e-3)
        
        # Generate stress field
        sigma_xx = -100e6 + 50e6 * np.cos(2 * np.pi * X / 25e-3)
        sigma_yy = -80e6 + 30e6 * np.sin(2 * np.pi * Y / 25e-3)
        sigma_zz = -50e6 + 20e6 * np.random.normal(0, 1, X.shape)
        
        # Generate displacement field
        ux = 1e-6 * X * np.sin(2 * np.pi * Y / 25e-3)
        uy = 1e-6 * Y * np.cos(2 * np.pi * X / 25e-3)
        uz = 5e-6 * Z * np.exp(-Z / 100e-6)
        
        # Generate strain field
        epsilon_xx = 0.001 * np.cos(2 * np.pi * X / 25e-3)
        epsilon_yy = 0.0008 * np.sin(2 * np.pi * Y / 25e-3)
        epsilon_zz = 0.0005 * np.exp(-Z / 100e-6)
        
        return {
            "coordinates": {
                "x": X.tolist(),
                "y": Y.tolist(),
                "z": Z.tolist()
            },
            "temperature": T.tolist(),
            "stress": {
                "xx": sigma_xx.tolist(),
                "yy": sigma_yy.tolist(),
                "zz": sigma_zz.tolist()
            },
            "displacement": {
                "x": ux.tolist(),
                "y": uy.tolist(),
                "z": uz.tolist()
            },
            "strain": {
                "xx": epsilon_xx.tolist(),
                "yy": epsilon_yy.tolist(),
                "zz": epsilon_zz.tolist()
            },
            "mesh_info": {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "total_nodes": nx * ny * nz,
                "total_elements": (nx-1) * (ny-1) * (nz-1)
            }
        }
    
    def _generate_collocation_data(self) -> Dict[str, Any]:
        """Generate collocation point data"""
        # Select strategic points for collocation
        n_points = 200
        
        # Points near pores
        pore_points = np.random.uniform(0, 25e-3, (50, 2))
        
        # Points at grain boundaries
        gb_points = np.random.uniform(0, 25e-3, (50, 2))
        
        # Points at free surfaces
        surface_points = np.random.uniform(0, 25e-3, (50, 2))
        
        # Points in bulk
        bulk_points = np.random.uniform(0, 25e-3, (50, 2))
        
        # Combine all points
        all_points = np.vstack([pore_points, gb_points, surface_points, bulk_points])
        
        # Generate data at collocation points
        collocation_data = []
        for i, (x, y) in enumerate(all_points):
            z = np.random.uniform(0, 200e-6)
            
            # Simulate measurement-like data
            T = 800 + 200 * np.exp(-z / 100e-6) + np.random.normal(0, 10)
            sigma_xx = -100e6 + 50e6 * np.cos(2 * np.pi * x / 25e-3) + np.random.normal(0, 5e6)
            sigma_yy = -80e6 + 30e6 * np.sin(2 * np.pi * y / 25e-3) + np.random.normal(0, 5e6)
            
            point_data = {
                "point_id": i,
                "coordinates": [x, y, z],
                "temperature": T,
                "stress": [sigma_xx, sigma_yy, -50e6],
                "displacement": [1e-6 * x, 1e-6 * y, 5e-6 * z],
                "strain": [0.001, 0.0008, 0.0005],
                "point_type": ["pore", "grain_boundary", "surface", "bulk"][i // 50]
            }
            collocation_data.append(point_data)
        
        return {
            "points": collocation_data,
            "total_points": n_points,
            "point_types": {
                "pore": 50,
                "grain_boundary": 50,
                "surface": 50,
                "bulk": 50
            }
        }
    
    def generate_micro_scale_data(self) -> Dict[str, Any]:
        """Generate micro-scale grain boundary data"""
        print("Generating micro-scale grain boundary data...")
        
        # Grain boundary properties
        gb_properties = self._generate_gb_properties()
        
        # Crystallographic orientation data (EBSD)
        ebsd_data = self._generate_ebsd_data()
        
        # Local stress concentrations
        local_stress = self._generate_local_stress_data()
        
        return {
            "grain_boundary_properties": gb_properties,
            "ebsd_data": ebsd_data,
            "local_stress": local_stress
        }
    
    def _generate_gb_properties(self) -> Dict[str, Any]:
        """Generate grain boundary properties"""
        return {
            "energy": {
                "high_angle": 1.0,  # J/m^2
                "low_angle": 0.3,   # J/m^2
                "twin_boundary": 0.1  # J/m^2
            },
            "diffusivity": {
                "oxygen": 1e-12,  # m^2/s
                "yttrium": 1e-15   # m^2/s
            },
            "mechanical_properties": {
                "strength": 500e6,  # Pa
                "toughness": 2.0,   # MPa√m
                "slip_resistance": 100e6  # Pa
            }
        }
    
    def _generate_ebsd_data(self) -> Dict[str, Any]:
        """Generate EBSD crystallographic data"""
        # Generate Euler angles for grains
        n_grains = 1000
        phi1 = np.random.uniform(0, 2*np.pi, n_grains)
        Phi = np.random.uniform(0, np.pi, n_grains)
        phi2 = np.random.uniform(0, 2*np.pi, n_grains)
        
        # Misorientation angles
        misorientation = np.random.exponential(10, n_grains)  # degrees
        
        return {
            "euler_angles": {
                "phi1": phi1.tolist(),
                "Phi": Phi.tolist(),
                "phi2": phi2.tolist()
            },
            "misorientation_angles": misorientation.tolist(),
            "grain_count": n_grains,
            "texture_components": {
                "cube": 0.15,
                "goss": 0.10,
                "brass": 0.08,
                "random": 0.67
            }
        }
    
    def _generate_local_stress_data(self) -> Dict[str, Any]:
        """Generate local stress concentration data"""
        # Stress concentrations at grain boundaries
        gb_stress_factors = np.random.lognormal(0.5, 0.3, 100)
        
        # Stress gradients
        stress_gradients = np.random.normal(1e9, 2e8, 100)  # Pa/m
        
        return {
            "stress_concentration_factors": gb_stress_factors.tolist(),
            "stress_gradients": stress_gradients.tolist(),
            "critical_stress_intensity": 2.0,  # MPa√m
            "crack_tip_stress": 500e6  # Pa
        }
    
    def generate_complete_dataset(self) -> Dict[str, Any]:
        """Generate the complete validation dataset"""
        print("Generating complete validation dataset...")
        
        dataset = {
            "metadata": self.dataset_metadata,
            "macro_scale": self.generate_macro_scale_data(),
            "meso_scale": self.generate_meso_scale_data(),
            "crack_data": self.generate_crack_data(),
            "fem_simulation": self.generate_fem_simulation_data(),
            "micro_scale": self.generate_micro_scale_data()
        }
        
        return dataset
    
    def save_dataset(self, dataset: Dict[str, Any], filename: str = "validation_dataset.json"):
        """Save dataset to JSON file"""
        print(f"Saving dataset to {filename}...")
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved successfully!")
    
    def save_hdf5_dataset(self, dataset: Dict[str, Any], filename: str = "validation_dataset.h5"):
        """Save dataset to HDF5 file for efficient storage"""
        print(f"Saving dataset to {filename}...")
        with h5py.File(filename, 'w') as f:
            # Create groups for different scales
            for scale, data in dataset.items():
                if scale == "metadata":
                    continue
                group = f.create_group(scale)
                self._save_dict_to_hdf5(data, group)
        
        print(f"HDF5 dataset saved successfully!")
    
    def _save_dict_to_hdf5(self, data: Dict, group: h5py.Group):
        """Recursively save dictionary to HDF5 group"""
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._save_dict_to_hdf5(value, subgroup)
            elif isinstance(value, list):
                group.create_dataset(key, data=np.array(value))
            else:
                group.attrs[key] = value

def main():
    """Main function to generate and save the validation dataset"""
    print("FEM Model Validation Dataset Generator")
    print("=" * 50)
    
    # Initialize generator
    generator = ValidationDatasetGenerator(seed=42)
    
    # Generate complete dataset
    dataset = generator.generate_complete_dataset()
    
    # Save in multiple formats
    generator.save_dataset(dataset, "validation_dataset.json")
    generator.save_hdf5_dataset(dataset, "validation_dataset.h5")
    
    # Print summary
    print("\nDataset Summary:")
    print(f"- Macro-scale data: {len(dataset['macro_scale'])} categories")
    print(f"- Meso-scale data: {len(dataset['meso_scale'])} categories")
    print(f"- Crack data: {len(dataset['crack_data'])} categories")
    print(f"- FEM simulation data: {len(dataset['fem_simulation'])} categories")
    print(f"- Micro-scale data: {len(dataset['micro_scale'])} categories")
    
    print("\nFiles generated:")
    print("- validation_dataset.json (human-readable)")
    print("- validation_dataset.h5 (efficient binary format)")
    
    return dataset

if __name__ == "__main__":
    dataset = main()