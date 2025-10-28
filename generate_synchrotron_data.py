#!/usr/bin/env python3
"""
Synthetic Synchrotron X-ray Data Generator for SOFC Creep Studies
==================================================================

This script generates synthetic 4D (3D + Time) synchrotron X-ray tomography
and diffraction data simulating in-operando creep deformation in SOFC materials.

Author: Synthetic Data Generator
Date: 2025-10-04
"""

import numpy as np
import h5py
import json
import os
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance_matrix
from datetime import datetime
from pathlib import Path

class SyntheticSynchrotronDataGenerator:
    """
    Generates synthetic synchrotron X-ray experimental data for SOFC research.
    """
    
    def __init__(self, output_dir="synchrotron_data", random_seed=42):
        """
        Initialize the data generator.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save generated data
        random_seed : int
            Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        np.random.seed(random_seed)
        
        # Data storage
        self.tomography_dir = self.output_dir / "tomography"
        self.diffraction_dir = self.output_dir / "diffraction"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.tomography_dir, self.diffraction_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Material properties (Ferritic Stainless Steel - typical SOFC interconnect)
        self.material = {
            "name": "Ferritic Stainless Steel (Crofer 22 APU)",
            "composition": {
                "Fe": 75.0,  # wt%
                "Cr": 22.0,
                "Mn": 0.5,
                "Ti": 0.08,
                "La": 0.04,
                "Si": 0.02,
                "C": 0.01
            },
            "phases": {
                "ferrite_alpha": {"fraction": 0.98, "lattice_parameter": 2.866},  # Angstroms
                "chromia_Cr2O3": {"fraction": 0.02, "lattice_parameter": 4.959}
            },
            "grain_size_um": 25.0,  # micrometers
            "elastic_modulus_GPa": 220.0,
            "poissons_ratio": 0.29
        }
        
        # Experimental parameters
        self.experiment = {
            "temperature_C": 700.0,
            "applied_stress_MPa": 50.0,
            "test_duration_hours": 100.0,
            "num_time_steps": 11,  # Initial + 10 time steps
            "scan_interval_hours": 10.0,
            "beam_energy_keV": 25.0,
            "voxel_size_um": 0.65,  # Resolution
            "image_dimensions": [512, 512, 512]  # 3D volume
        }
        
        # Sample geometry
        self.sample = {
            "geometry": "cylindrical",
            "diameter_mm": 3.0,
            "height_mm": 5.0,
            "gauge_length_mm": 3.0
        }
        
        # Creep parameters (for synthetic evolution)
        self.creep_params = {
            "primary_creep_rate": 1e-7,  # 1/s
            "secondary_creep_rate": 5e-8,  # 1/s
            "cavity_nucleation_rate": 0.05,  # per time step
            "cavity_growth_rate": 0.15,  # expansion factor
            "crack_propagation_rate": 0.08,
            "grain_boundary_sliding": 0.02
        }
    
    def generate_initial_microstructure(self):
        """
        Generate initial 3D microstructure with grains, grain boundaries, and initial porosity.
        
        Returns:
        --------
        microstructure : ndarray
            3D array representing the material microstructure
        grain_map : ndarray
            3D array with grain IDs
        """
        dims = self.experiment["image_dimensions"]
        
        # Generate Voronoi-like grain structure
        num_grains = int(np.prod(dims) / (self.material["grain_size_um"] / 
                                           self.experiment["voxel_size_um"])**3)
        num_grains = max(50, min(num_grains, 200))  # Reasonable range
        
        # Create grain centers
        grain_centers = np.random.rand(num_grains, 3) * np.array(dims)
        
        # Create mesh grid
        z, y, x = np.mgrid[0:dims[0], 0:dims[1], 0:dims[2]]
        coords = np.stack([z.ravel(), y.ravel(), x.ravel()], axis=1)
        
        # Assign each voxel to nearest grain center
        print("Generating grain structure...")
        grain_map = np.zeros(dims, dtype=np.int32)
        chunk_size = 100000
        for i in range(0, len(coords), chunk_size):
            chunk = coords[i:i+chunk_size]
            distances = distance_matrix(chunk, grain_centers)
            grain_ids = np.argmin(distances, axis=1)
            indices = np.unravel_index(range(i, min(i+chunk_size, len(coords))), dims)
            grain_map[indices] = grain_ids
        
        # Create grain boundaries (where grain ID changes)
        grain_boundaries = self._detect_grain_boundaries(grain_map)
        
        # Initialize microstructure (density map: 0=void, 1=material)
        microstructure = np.ones(dims, dtype=np.float32)
        
        # Add initial porosity (2-5%)
        initial_porosity = np.random.rand(*dims) < 0.03
        microstructure[initial_porosity] = 0.0
        
        # Add some initial defects at grain boundaries
        gb_defects = grain_boundaries & (np.random.rand(*dims) < 0.05)
        microstructure[gb_defects] = 0.3  # Partial void
        
        # Smooth slightly for realism
        microstructure = gaussian_filter(microstructure, sigma=0.5)
        
        return microstructure, grain_map, grain_boundaries
    
    def _detect_grain_boundaries(self, grain_map):
        """Detect grain boundaries where grain ID changes."""
        boundaries = np.zeros_like(grain_map, dtype=bool)
        
        # Check neighbors in all 3 directions
        boundaries[:-1,:,:] |= grain_map[:-1,:,:] != grain_map[1:,:,:]
        boundaries[:,:-1,:] |= grain_map[:,:-1,:] != grain_map[:,1:,:]
        boundaries[:,:,:-1] |= grain_map[:,:,:-1] != grain_map[:,:,1:]
        
        return boundaries
    
    def evolve_microstructure(self, microstructure, grain_boundaries, time_step):
        """
        Evolve microstructure to simulate creep damage over time.
        
        Parameters:
        -----------
        microstructure : ndarray
            Current microstructure state
        grain_boundaries : ndarray
            Boolean array of grain boundary locations
        time_step : int
            Current time step (0 = initial)
        
        Returns:
        --------
        evolved_microstructure : ndarray
            Updated microstructure with creep damage
        """
        evolved = microstructure.copy()
        
        if time_step == 0:
            return evolved
        
        # 1. Cavity nucleation at grain boundaries
        cavity_candidates = grain_boundaries & (microstructure > 0.7)
        nucleation_prob = self.creep_params["cavity_nucleation_rate"] * time_step
        new_cavities = cavity_candidates & (np.random.rand(*microstructure.shape) < nucleation_prob)
        evolved[new_cavities] = 0.2
        
        # 2. Cavity growth (existing cavities expand)
        existing_cavities = microstructure < 0.5
        growth_rate = self.creep_params["cavity_growth_rate"]
        
        # Dilate cavities
        from scipy.ndimage import binary_dilation
        if np.any(existing_cavities):
            structure = np.ones((3,3,3))
            iterations = max(1, int(time_step * growth_rate))
            grown_cavities = binary_dilation(existing_cavities, structure=structure, 
                                            iterations=iterations)
            evolved[grown_cavities] = 0.0
        
        # 3. Crack propagation along grain boundaries
        crack_seeds = existing_cavities & grain_boundaries
        if np.any(crack_seeds):
            crack_prop = binary_dilation(crack_seeds, iterations=int(time_step * 
                                        self.creep_params["crack_propagation_rate"] * 2))
            crack_path = crack_prop & grain_boundaries
            evolved[crack_path] = 0.0
        
        # 4. Grain boundary sliding (slight blur along boundaries)
        gb_region = binary_dilation(grain_boundaries, iterations=2)
        sliding_effect = gaussian_filter(evolved * gb_region.astype(float), 
                                        sigma=time_step * self.creep_params["grain_boundary_sliding"])
        evolved[gb_region] = 0.7 * evolved[gb_region] + 0.3 * sliding_effect[gb_region]
        
        # 5. Add some realistic noise
        noise = np.random.normal(0, 0.02, evolved.shape)
        evolved = np.clip(evolved + noise, 0, 1)
        
        return evolved
    
    def generate_tomography_time_series(self):
        """
        Generate complete 4D tomography dataset (3D + time).
        """
        print("\n" + "="*70)
        print("GENERATING 4D TOMOGRAPHY DATA")
        print("="*70)
        
        # Generate initial microstructure
        print("\n[1/3] Generating initial microstructure...")
        initial_micro, grain_map, grain_boundaries = self.generate_initial_microstructure()
        
        # Save grain map
        grain_map_file = self.tomography_dir / "grain_map.h5"
        with h5py.File(grain_map_file, 'w') as f:
            f.create_dataset('grain_ids', data=grain_map, compression='gzip')
            f.attrs['num_grains'] = len(np.unique(grain_map))
            f.attrs['average_grain_size_um'] = self.material['grain_size_um']
        print(f"   Saved grain map to {grain_map_file}")
        
        # Generate time series
        print("\n[2/3] Generating time-lapse evolution...")
        num_steps = self.experiment["num_time_steps"]
        
        tomography_file = self.tomography_dir / "tomography_4D.h5"
        
        with h5py.File(tomography_file, 'w') as f:
            # Create datasets
            tomo_dataset = f.create_dataset(
                'tomography',
                shape=(num_steps, *self.experiment["image_dimensions"]),
                dtype=np.float32,
                compression='gzip',
                chunks=(1, 128, 128, 128)
            )
            
            time_dataset = f.create_dataset('time_hours', shape=(num_steps,), dtype=np.float32)
            
            # Generate each time step
            for t in range(num_steps):
                time_hours = t * self.experiment["scan_interval_hours"]
                print(f"   Time step {t}/{num_steps-1}: t = {time_hours:.1f} hours")
                
                # Evolve microstructure
                evolved_micro = self.evolve_microstructure(initial_micro, grain_boundaries, t)
                
                # Simulate X-ray attenuation (Beer-Lambert law)
                # Higher density = higher attenuation
                attenuation = evolved_micro * np.random.normal(1.0, 0.05, evolved_micro.shape)
                attenuation = np.clip(attenuation, 0, 1)
                
                # Add Poisson noise (photon counting statistics)
                attenuation = np.random.poisson(attenuation * 1000) / 1000.0
                
                # Store
                tomo_dataset[t] = attenuation.astype(np.float32)
                time_dataset[t] = time_hours
            
            # Add metadata
            f.attrs['temperature_C'] = self.experiment['temperature_C']
            f.attrs['applied_stress_MPa'] = self.experiment['applied_stress_MPa']
            f.attrs['voxel_size_um'] = self.experiment['voxel_size_um']
            f.attrs['beam_energy_keV'] = self.experiment['beam_energy_keV']
            f.attrs['dimensions'] = self.experiment['image_dimensions']
            f.attrs['num_time_steps'] = num_steps
            f.attrs['scan_interval_hours'] = self.experiment['scan_interval_hours']
        
        print(f"\n   ✓ Saved 4D tomography to {tomography_file}")
        print(f"   Data shape: {(num_steps, *self.experiment['image_dimensions'])}")
        
        # Analyze and save metrics
        print("\n[3/3] Computing microstructural metrics...")
        metrics = self._compute_tomography_metrics(tomography_file)
        
        metrics_file = self.tomography_dir / "tomography_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"   Saved metrics to {metrics_file}")
        
        return tomography_file, metrics_file
    
    def _compute_tomography_metrics(self, tomography_file):
        """Compute key metrics from tomography data."""
        metrics = {
            "time_hours": [],
            "porosity_percent": [],
            "cavity_count": [],
            "crack_volume_mm3": [],
            "mean_grain_boundary_integrity": []
        }
        
        with h5py.File(tomography_file, 'r') as f:
            tomo_data = f['tomography']
            time_data = f['time_hours']
            voxel_size = f.attrs['voxel_size_um']
            
            for t in range(len(time_data)):
                volume = tomo_data[t]
                
                # Porosity
                porosity = np.sum(volume < 0.3) / volume.size * 100
                
                # Cavity count (connected components)
                from scipy.ndimage import label
                binary_voids = volume < 0.3
                labeled, num_cavities = label(binary_voids)
                
                # Crack volume
                crack_voxels = np.sum(volume < 0.1)
                crack_volume_mm3 = crack_voxels * (voxel_size * 1e-3)**3
                
                # Grain boundary integrity (average value at GB)
                gb_integrity = np.mean(volume[volume < 0.9])
                
                metrics["time_hours"].append(float(time_data[t]))
                metrics["porosity_percent"].append(float(porosity))
                metrics["cavity_count"].append(int(num_cavities))
                metrics["crack_volume_mm3"].append(float(crack_volume_mm3))
                metrics["mean_grain_boundary_integrity"].append(float(gb_integrity))
        
        return metrics
    
    def generate_xrd_data(self):
        """
        Generate synthetic X-ray diffraction data including:
        - Phase identification patterns
        - Residual stress/strain maps
        """
        print("\n" + "="*70)
        print("GENERATING X-RAY DIFFRACTION (XRD) DATA")
        print("="*70)
        
        # 1. Generate diffraction patterns
        print("\n[1/3] Generating diffraction patterns...")
        pattern_file = self._generate_diffraction_patterns()
        
        # 2. Generate strain/stress maps
        print("\n[2/3] Generating strain/stress maps...")
        strain_file = self._generate_strain_maps()
        
        # 3. Generate phase maps
        print("\n[3/3] Generating phase distribution maps...")
        phase_file = self._generate_phase_maps()
        
        return pattern_file, strain_file, phase_file
    
    def _generate_diffraction_patterns(self):
        """Generate synthetic XRD patterns for phase identification."""
        
        # 2-theta angles (degrees)
        two_theta = np.linspace(20, 80, 3000)
        
        patterns = {}
        
        # Ferrite (α-Fe) peaks (BCC)
        ferrite_peaks = [44.7, 65.0, 82.3]  # Main peaks
        ferrite_pattern = np.zeros_like(two_theta)
        for peak in ferrite_peaks:
            ferrite_pattern += 1000 * np.exp(-((two_theta - peak) / 0.3)**2)
        
        # Chromia (Cr2O3) peaks
        chromia_peaks = [33.6, 36.2, 50.2, 54.8, 63.4]
        chromia_pattern = np.zeros_like(two_theta)
        for peak in chromia_peaks:
            chromia_pattern += 200 * np.exp(-((two_theta - peak) / 0.25)**2)
        
        # Combined pattern with noise
        combined_pattern = (ferrite_pattern * 0.98 + chromia_pattern * 0.02 + 
                          np.random.normal(50, 10, len(two_theta)))
        
        patterns["two_theta_deg"] = two_theta.tolist()
        patterns["intensity_counts"] = np.maximum(combined_pattern, 0).tolist()
        patterns["phases_detected"] = {
            "ferrite_alpha_Fe": {
                "fraction": 0.98,
                "peaks_deg": ferrite_peaks,
                "crystal_system": "BCC",
                "lattice_parameter_angstrom": 2.866
            },
            "chromia_Cr2O3": {
                "fraction": 0.02,
                "peaks_deg": chromia_peaks,
                "crystal_system": "Rhombohedral",
                "lattice_parameter_angstrom": 4.959
            }
        }
        
        pattern_file = self.diffraction_dir / "xrd_patterns.json"
        with open(pattern_file, 'w') as f:
            json.dump(patterns, f, indent=2)
        
        print(f"   Saved XRD patterns to {pattern_file}")
        return pattern_file
    
    def _generate_strain_maps(self):
        """Generate 3D strain and stress distribution maps."""
        
        # Simplified 2D maps for each time step (can extend to 3D)
        map_size = (256, 256)
        num_steps = self.experiment["num_time_steps"]
        
        strain_file = self.diffraction_dir / "strain_stress_maps.h5"
        
        with h5py.File(strain_file, 'w') as f:
            # Create datasets
            strain_dataset = f.create_dataset(
                'elastic_strain',
                shape=(num_steps, *map_size),
                dtype=np.float32,
                compression='gzip'
            )
            
            stress_dataset = f.create_dataset(
                'residual_stress_MPa',
                shape=(num_steps, *map_size),
                dtype=np.float32,
                compression='gzip'
            )
            
            time_dataset = f.create_dataset('time_hours', shape=(num_steps,), dtype=np.float32)
            
            for t in range(num_steps):
                time_hours = t * self.experiment["scan_interval_hours"]
                
                # Create realistic strain distribution
                # Higher strain accumulates over time near defects
                x, y = np.meshgrid(np.linspace(0, 1, map_size[1]), 
                                  np.linspace(0, 1, map_size[0]))
                
                # Base uniform strain from applied stress
                base_strain = 0.001 * (1 + t * 0.1)
                
                # Add localized strain concentrations
                num_hotspots = 5 + t
                strain_map = np.ones(map_size) * base_strain
                
                for _ in range(num_hotspots):
                    cx, cy = np.random.rand(2)
                    hotspot = np.exp(-((x - cx)**2 + (y - cy)**2) / (0.05 + t * 0.01))
                    strain_map += hotspot * 0.005 * (1 + t * 0.2)
                
                # Add noise
                strain_map += np.random.normal(0, 0.0002, map_size)
                strain_map = np.clip(strain_map, 0, 0.05)
                
                # Calculate stress from strain (Hooke's law)
                E = self.material["elastic_modulus_GPa"] * 1000  # Convert to MPa
                stress_map = strain_map * E
                
                strain_dataset[t] = strain_map.astype(np.float32)
                stress_dataset[t] = stress_map.astype(np.float32)
                time_dataset[t] = time_hours
            
            # Add metadata
            f.attrs['temperature_C'] = self.experiment['temperature_C']
            f.attrs['applied_stress_MPa'] = self.experiment['applied_stress_MPa']
            f.attrs['elastic_modulus_GPa'] = self.material['elastic_modulus_GPa']
            f.attrs['map_dimensions'] = map_size
        
        print(f"   Saved strain/stress maps to {strain_file}")
        return strain_file
    
    def _generate_phase_maps(self):
        """Generate spatial distribution of phases."""
        
        map_size = self.experiment["image_dimensions"]
        
        # Create phase distribution (mostly ferrite with chromia at surface/GB)
        phase_map = np.ones(map_size, dtype=np.int8)  # 1 = ferrite
        
        # Add chromia layer (surface oxide)
        phase_map[:5, :, :] = 2  # Top surface
        phase_map[-5:, :, :] = 2  # Bottom surface
        
        # Random chromia particles at grain boundaries
        chromia_particles = np.random.rand(*map_size) < 0.02
        phase_map[chromia_particles] = 2
        
        phase_file = self.diffraction_dir / "phase_map.h5"
        
        with h5py.File(phase_file, 'w') as f:
            f.create_dataset('phase_ids', data=phase_map, compression='gzip')
            f.attrs['phase_legend'] = json.dumps({
                "1": "Ferrite (α-Fe)",
                "2": "Chromia (Cr2O3)"
            })
            f.attrs['ferrite_fraction'] = float(np.sum(phase_map == 1) / phase_map.size)
            f.attrs['chromia_fraction'] = float(np.sum(phase_map == 2) / phase_map.size)
        
        print(f"   Saved phase map to {phase_file}")
        return phase_file
    
    def generate_metadata(self):
        """
        Generate comprehensive metadata files.
        """
        print("\n" + "="*70)
        print("GENERATING METADATA")
        print("="*70)
        
        # 1. Experimental parameters
        exp_params = {
            "experiment_id": f"SOFC_CREEP_{datetime.now().strftime('%Y%m%d')}",
            "facility": "European Synchrotron Radiation Facility (ESRF)",
            "beamline": "ID19 - Imaging Beamline",
            "date": datetime.now().isoformat(),
            "operator": "Synthetic Data Generator",
            
            "test_conditions": {
                "temperature_C": self.experiment["temperature_C"],
                "temperature_stability_C": 2.0,
                "applied_stress_MPa": self.experiment["applied_stress_MPa"],
                "stress_type": "uniaxial tension",
                "atmosphere": "Air",
                "test_duration_hours": self.experiment["test_duration_hours"],
                "scan_interval_hours": self.experiment["scan_interval_hours"]
            },
            
            "imaging_parameters": {
                "technique": "X-ray computed tomography",
                "beam_energy_keV": self.experiment["beam_energy_keV"],
                "voxel_size_um": self.experiment["voxel_size_um"],
                "field_of_view_mm": [
                    dim * self.experiment["voxel_size_um"] / 1000 
                    for dim in self.experiment["image_dimensions"]
                ],
                "image_dimensions_pixels": self.experiment["image_dimensions"],
                "num_projections": 2000,
                "exposure_time_ms": 50,
                "detector": "PCO edge 5.5 (2560 x 2160 pixels)"
            },
            
            "diffraction_parameters": {
                "technique": "Energy-dispersive X-ray diffraction",
                "beam_energy_keV": self.experiment["beam_energy_keV"],
                "two_theta_range_deg": [20, 80],
                "step_size_deg": 0.02,
                "detector": "Pilatus 2M"
            }
        }
        
        exp_file = self.metadata_dir / "experimental_parameters.json"
        with open(exp_file, 'w') as f:
            json.dump(exp_params, f, indent=2)
        print(f"\n[1/3] Saved experimental parameters to {exp_file}")
        
        # 2. Material specifications
        material_spec = {
            "material_name": self.material["name"],
            "material_class": "Ferritic Stainless Steel",
            "application": "SOFC Metallic Interconnect",
            
            "composition_wt_percent": self.material["composition"],
            
            "heat_treatment": {
                "annealing_temperature_C": 1050,
                "annealing_time_hours": 2,
                "cooling_rate_C_per_min": 50,
                "atmosphere": "Argon"
            },
            
            "microstructure": {
                "grain_size_um": self.material["grain_size_um"],
                "grain_size_distribution": "log-normal",
                "grain_morphology": "equiaxed",
                "texture": "random"
            },
            
            "phases": self.material["phases"],
            
            "mechanical_properties": {
                "elastic_modulus_GPa": self.material["elastic_modulus_GPa"],
                "poissons_ratio": self.material["poissons_ratio"],
                "yield_strength_MPa": 280,
                "tensile_strength_MPa": 450,
                "elongation_percent": 25
            },
            
            "thermal_properties": {
                "thermal_expansion_coefficient_per_K": 12.0e-6,
                "thermal_conductivity_W_m_K": 25.0,
                "specific_heat_J_kg_K": 460
            }
        }
        
        material_file = self.metadata_dir / "material_specifications.json"
        with open(material_file, 'w') as f:
            json.dump(material_spec, f, indent=2)
        print(f"[2/3] Saved material specifications to {material_file}")
        
        # 3. Sample geometry
        sample_spec = {
            "sample_id": "SOFC-IC-001",
            "geometry": self.sample["geometry"],
            "dimensions_mm": {
                "diameter": self.sample["diameter_mm"],
                "height": self.sample["height_mm"],
                "gauge_length": self.sample["gauge_length_mm"]
            },
            "surface_finish": "polished (1 μm diamond)",
            "mass_mg": 350.5,
            "preparation_method": "Wire electrical discharge machining (EDM)",
            "pre_test_inspection": {
                "visual": "No visible defects",
                "dimensional_tolerance_um": 10,
                "surface_roughness_Ra_um": 0.2
            }
        }
        
        sample_file = self.metadata_dir / "sample_geometry.json"
        with open(sample_file, 'w') as f:
            json.dump(sample_spec, f, indent=2)
        print(f"[3/3] Saved sample geometry to {sample_file}")
        
        return exp_file, material_file, sample_file
    
    def generate_all(self):
        """
        Generate complete synthetic dataset.
        """
        print("\n")
        print("╔" + "="*68 + "╗")
        print("║" + " "*10 + "SYNTHETIC SYNCHROTRON X-RAY DATA GENERATOR" + " "*16 + "║")
        print("║" + " "*20 + "FOR SOFC CREEP STUDIES" + " "*25 + "║")
        print("╚" + "="*68 + "╝")
        
        # Generate all data
        tomo_file, metrics_file = self.generate_tomography_time_series()
        pattern_file, strain_file, phase_file = self.generate_xrd_data()
        exp_file, material_file, sample_file = self.generate_metadata()
        
        # Create summary
        summary = {
            "generation_date": datetime.now().isoformat(),
            "output_directory": str(self.output_dir),
            "files_generated": {
                "tomography": {
                    "4D_data": str(tomo_file.relative_to(self.output_dir)),
                    "grain_map": str((self.tomography_dir / "grain_map.h5").relative_to(self.output_dir)),
                    "metrics": str(metrics_file.relative_to(self.output_dir))
                },
                "diffraction": {
                    "xrd_patterns": str(pattern_file.relative_to(self.output_dir)),
                    "strain_stress_maps": str(strain_file.relative_to(self.output_dir)),
                    "phase_map": str(phase_file.relative_to(self.output_dir))
                },
                "metadata": {
                    "experimental_parameters": str(exp_file.relative_to(self.output_dir)),
                    "material_specifications": str(material_file.relative_to(self.output_dir)),
                    "sample_geometry": str(sample_file.relative_to(self.output_dir))
                }
            },
            "data_statistics": {
                "total_time_steps": self.experiment["num_time_steps"],
                "test_duration_hours": self.experiment["test_duration_hours"],
                "total_3D_volumes": self.experiment["num_time_steps"],
                "voxel_dimensions": self.experiment["image_dimensions"],
                "spatial_resolution_um": self.experiment["voxel_size_um"],
                "total_data_size_GB": self._estimate_data_size()
            }
        }
        
        summary_file = self.output_dir / "dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*70)
        print("✓ DATA GENERATION COMPLETE!")
        print("="*70)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Summary file: {summary_file}")
        print(f"\nEstimated total size: {summary['data_statistics']['total_data_size_GB']:.2f} GB")
        
        return summary_file
    
    def _estimate_data_size(self):
        """Estimate total dataset size in GB."""
        # Tomography: 4 bytes per voxel * dimensions * time steps
        tomo_size = (4 * np.prod(self.experiment["image_dimensions"]) * 
                    self.experiment["num_time_steps"])
        
        # Grain map
        grain_size = 4 * np.prod(self.experiment["image_dimensions"])
        
        # Strain maps: 2 datasets * 4 bytes * map size * time steps
        strain_size = 2 * 4 * 256 * 256 * self.experiment["num_time_steps"]
        
        # Phase map
        phase_size = np.prod(self.experiment["image_dimensions"])
        
        total_bytes = tomo_size + grain_size + strain_size + phase_size
        return total_bytes / 1e9


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate synthetic synchrotron X-ray data for SOFC creep studies"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="synchrotron_data",
        help="Output directory for generated data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=700.0,
        help="Test temperature in Celsius"
    )
    parser.add_argument(
        "--stress",
        type=float,
        default=50.0,
        help="Applied stress in MPa"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=100.0,
        help="Test duration in hours"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = SyntheticSynchrotronDataGenerator(
        output_dir=args.output_dir,
        random_seed=args.seed
    )
    
    # Update parameters if provided
    generator.experiment["temperature_C"] = args.temperature
    generator.experiment["applied_stress_MPa"] = args.stress
    generator.experiment["test_duration_hours"] = args.duration
    
    # Generate all data
    summary_file = generator.generate_all()
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Explore the data using: python visualize_data.py")
    print("2. Analyze metrics: python analyze_metrics.py")
    print("3. Read the README.md for detailed documentation")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
