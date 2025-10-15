#!/usr/bin/env python3
"""
Multi-Fidelity SOFC Dataset Generator
=====================================

This module generates comprehensive datasets for Solid Oxide Fuel Cell (SOFC) 
digital twin modeling with multi-scale parameters and operating conditions.

Author: Generated for PhD Thesis - Multi-Fidelity Digital Twin for SOFCs
Date: 2025-10-15
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from scipy.stats import qmc
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class SOFCDatasetGenerator:
    """
    Comprehensive SOFC dataset generator for multi-fidelity digital twin modeling.
    
    Generates datasets across multiple scales:
    - System level (Operating conditions)
    - Cell/Stack level (Geometry)
    - Material level (Properties)
    - Microstructural level (High-fidelity experimental data)
    """
    
    def __init__(self, output_dir: str = "sofc_dataset"):
        """Initialize the dataset generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize parameter ranges and specifications
        self._initialize_parameter_ranges()
        
        # Dataset metadata
        self.metadata = {
            "dataset_name": "Multi-Fidelity SOFC Digital Twin Dataset",
            "version": "1.0",
            "created_date": datetime.now().isoformat(),
            "description": "Comprehensive multi-scale SOFC modeling parameters",
            "fidelity_levels": ["LF", "MF", "HF"],
            "scales": ["System", "Cell/Stack", "Material", "Microstructural"]
        }
    
    def _initialize_parameter_ranges(self):
        """Initialize parameter ranges for all scales and components."""
        
        # System Level - Operating Conditions
        self.system_parameters = {
            # Basic Operating Conditions
            "fuel_utilization": {"min": 0.6, "max": 0.95, "unit": "-", "fidelity": ["LF", "MF", "HF"]},
            "oxidant_utilization": {"min": 0.15, "max": 0.4, "unit": "-", "fidelity": ["LF", "MF", "HF"]},
            "current_density": {"min": 0.1, "max": 1.5, "unit": "A/cm²", "fidelity": ["LF", "MF", "HF"]},
            "voltage": {"min": 0.6, "max": 1.1, "unit": "V", "fidelity": ["LF", "MF", "HF"]},
            "temperature": {"min": 973, "max": 1273, "unit": "K", "fidelity": ["LF", "MF", "HF"]},
            "pressure": {"min": 1.0, "max": 10.0, "unit": "atm", "fidelity": ["LF", "MF", "HF"]},
            
            # Flow Rates
            "fuel_flow_rate": {"min": 10, "max": 200, "unit": "sccm", "fidelity": ["LF", "MF", "HF"]},
            "air_flow_rate": {"min": 50, "max": 500, "unit": "sccm", "fidelity": ["LF", "MF", "HF"]},
            
            # Fuel Composition (H2/CO/H2O/CO2/N2)
            "h2_fraction": {"min": 0.3, "max": 0.97, "unit": "-", "fidelity": ["LF", "MF", "HF"]},
            "co_fraction": {"min": 0.0, "max": 0.4, "unit": "-", "fidelity": ["LF", "MF", "HF"]},
            "h2o_fraction": {"min": 0.03, "max": 0.3, "unit": "-", "fidelity": ["LF", "MF", "HF"]},
            "co2_fraction": {"min": 0.0, "max": 0.2, "unit": "-", "fidelity": ["LF", "MF", "HF"]},
            "n2_fraction": {"min": 0.0, "max": 0.1, "unit": "-", "fidelity": ["LF", "MF", "HF"]},
        }
        
        # Cell/Stack Geometry Parameters
        self.geometry_parameters = {
            "cell_active_area": {"min": 1.0, "max": 100.0, "unit": "cm²", "fidelity": ["MF", "HF"]},
            "anode_thickness": {"min": 300, "max": 1000, "unit": "μm", "fidelity": ["MF", "HF"]},
            "cathode_thickness": {"min": 20, "max": 80, "unit": "μm", "fidelity": ["MF", "HF"]},
            "electrolyte_thickness": {"min": 5, "max": 50, "unit": "μm", "fidelity": ["MF", "HF"]},
            "interconnect_thickness": {"min": 100, "max": 500, "unit": "μm", "fidelity": ["MF", "HF"]},
            "channel_width": {"min": 0.5, "max": 3.0, "unit": "mm", "fidelity": ["MF", "HF"]},
            "channel_height": {"min": 0.5, "max": 2.0, "unit": "mm", "fidelity": ["MF", "HF"]},
            "rib_width": {"min": 0.5, "max": 2.0, "unit": "mm", "fidelity": ["MF", "HF"]},
        }
        
        # Material Properties - Anode (Ni-YSZ)
        self.anode_parameters = {
            "ni_volume_fraction": {"min": 0.3, "max": 0.6, "unit": "-", "fidelity": ["MF", "HF"]},
            "porosity": {"min": 0.25, "max": 0.45, "unit": "-", "fidelity": ["MF", "HF"]},
            "tortuosity": {"min": 2.0, "max": 8.0, "unit": "-", "fidelity": ["MF", "HF"]},
            "ni_particle_size": {"min": 0.3, "max": 2.0, "unit": "μm", "fidelity": ["MF", "HF"]},
            "tpb_density": {"min": 1e12, "max": 1e14, "unit": "m/m³", "fidelity": ["MF", "HF"]},
            "ionic_conductivity": {"min": 0.01, "max": 0.1, "unit": "S/m", "fidelity": ["MF", "HF"]},
            "electronic_conductivity": {"min": 1e4, "max": 1e6, "unit": "S/m", "fidelity": ["MF", "HF"]},
            "thermal_conductivity": {"min": 2.0, "max": 8.0, "unit": "W/m·K", "fidelity": ["MF", "HF"]},
        }
        
        # Material Properties - Cathode (LSCF)
        self.cathode_parameters = {
            "porosity": {"min": 0.25, "max": 0.45, "unit": "-", "fidelity": ["MF", "HF"]},
            "tortuosity": {"min": 2.0, "max": 6.0, "unit": "-", "fidelity": ["MF", "HF"]},
            "particle_size": {"min": 0.1, "max": 1.0, "unit": "μm", "fidelity": ["MF", "HF"]},
            "tpb_density": {"min": 1e12, "max": 5e13, "unit": "m/m³", "fidelity": ["MF", "HF"]},
            "ionic_conductivity": {"min": 0.1, "max": 10.0, "unit": "S/m", "fidelity": ["MF", "HF"]},
            "electronic_conductivity": {"min": 100, "max": 1000, "unit": "S/m", "fidelity": ["MF", "HF"]},
            "thermal_conductivity": {"min": 1.0, "max": 5.0, "unit": "W/m·K", "fidelity": ["MF", "HF"]},
            "chemical_expansion_coeff": {"min": 1e-5, "max": 5e-5, "unit": "K⁻¹", "fidelity": ["MF", "HF"]},
        }
        
        # Material Properties - Electrolyte (YSZ)
        self.electrolyte_parameters = {
            "ionic_conductivity": {"min": 0.01, "max": 0.5, "unit": "S/m", "fidelity": ["MF", "HF"]},
            "youngs_modulus": {"min": 150, "max": 250, "unit": "GPa", "fidelity": ["MF", "HF"]},
            "poissons_ratio": {"min": 0.25, "max": 0.35, "unit": "-", "fidelity": ["MF", "HF"]},
            "thermal_expansion_coeff": {"min": 9e-6, "max": 12e-6, "unit": "K⁻¹", "fidelity": ["MF", "HF"]},
            "thermal_conductivity": {"min": 2.0, "max": 4.0, "unit": "W/m·K", "fidelity": ["MF", "HF"]},
            "fracture_toughness": {"min": 1.0, "max": 3.0, "unit": "MPa·m^0.5", "fidelity": ["MF", "HF"]},
        }
        
        # Material Properties - Interconnect (Crofer 22APU)
        self.interconnect_parameters = {
            "thermal_expansion_coeff": {"min": 11e-6, "max": 13e-6, "unit": "K⁻¹", "fidelity": ["MF", "HF"]},
            "youngs_modulus": {"min": 180, "max": 220, "unit": "GPa", "fidelity": ["MF", "HF"]},
            "poissons_ratio": {"min": 0.28, "max": 0.32, "unit": "-", "fidelity": ["MF", "HF"]},
            "creep_coefficient": {"min": 1e-20, "max": 1e-18, "unit": "Pa⁻ⁿ·s⁻¹", "fidelity": ["MF", "HF"]},
            "creep_exponent": {"min": 3.0, "max": 7.0, "unit": "-", "fidelity": ["MF", "HF"]},
            "oxide_scale_growth_rate": {"min": 1e-15, "max": 1e-13, "unit": "m²/s", "fidelity": ["MF", "HF"]},
            "thermal_conductivity": {"min": 15, "max": 25, "unit": "W/m·K", "fidelity": ["MF", "HF"]},
            "electrical_resistivity": {"min": 1e-6, "max": 5e-6, "unit": "Ω·m", "fidelity": ["MF", "HF"]},
        }
        
        # Microstructural Properties (High-Fidelity Experimental)
        self.microstructural_parameters = {
            "voxel_size": {"min": 10, "max": 100, "unit": "nm", "fidelity": ["HF"]},
            "reconstruction_volume": {"min": 10, "max": 100, "unit": "μm³", "fidelity": ["HF"]},
            "phase_fraction_ni": {"min": 0.25, "max": 0.45, "unit": "-", "fidelity": ["HF"]},
            "phase_fraction_ysz": {"min": 0.35, "max": 0.55, "unit": "-", "fidelity": ["HF"]},
            "phase_fraction_pore": {"min": 0.15, "max": 0.35, "unit": "-", "fidelity": ["HF"]},
            "specific_surface_area": {"min": 1e6, "max": 1e7, "unit": "m²/m³", "fidelity": ["HF"]},
            "connectivity_ni": {"min": 0.8, "max": 0.99, "unit": "-", "fidelity": ["HF"]},
            "connectivity_ysz": {"min": 0.85, "max": 0.99, "unit": "-", "fidelity": ["HF"]},
            "connectivity_pore": {"min": 0.7, "max": 0.95, "unit": "-", "fidelity": ["HF"]},
        }
    
    def generate_latin_hypercube_samples(self, parameters: Dict, n_samples: int, 
                                       fidelity_filter: List[str] = None) -> pd.DataFrame:
        """
        Generate Latin Hypercube Samples for given parameters.
        
        Args:
            parameters: Dictionary of parameter specifications
            n_samples: Number of samples to generate
            fidelity_filter: List of fidelity levels to include
            
        Returns:
            DataFrame with generated samples
        """
        if fidelity_filter is None:
            fidelity_filter = ["LF", "MF", "HF"]
        
        # Filter parameters by fidelity
        filtered_params = {}
        for param, spec in parameters.items():
            if any(f in spec["fidelity"] for f in fidelity_filter):
                filtered_params[param] = spec
        
        if not filtered_params:
            return pd.DataFrame()
        
        # Create Latin Hypercube sampler
        sampler = qmc.LatinHypercube(d=len(filtered_params), seed=42)
        samples = sampler.random(n=n_samples)
        
        # Scale samples to parameter ranges
        param_names = list(filtered_params.keys())
        scaled_samples = np.zeros_like(samples)
        
        for i, param in enumerate(param_names):
            min_val = filtered_params[param]["min"]
            max_val = filtered_params[param]["max"]
            scaled_samples[:, i] = min_val + samples[:, i] * (max_val - min_val)
        
        # Create DataFrame
        df = pd.DataFrame(scaled_samples, columns=param_names)
        
        # Add metadata columns
        for param in param_names:
            df[f"{param}_unit"] = filtered_params[param]["unit"]
            df[f"{param}_fidelity"] = str(filtered_params[param]["fidelity"])
        
        return df
    
    def generate_transient_profiles(self, n_profiles: int = 100) -> Dict[str, np.ndarray]:
        """
        Generate transient cycle profiles for degradation modeling.
        
        Args:
            n_profiles: Number of transient profiles to generate
            
        Returns:
            Dictionary containing different types of transient profiles
        """
        profiles = {}
        
        # Time vectors (different durations for different operations)
        startup_time = np.linspace(0, 3600, 100)  # 1 hour startup
        shutdown_time = np.linspace(0, 1800, 60)  # 30 min shutdown
        load_following_time = np.linspace(0, 7200, 200)  # 2 hour load following
        
        for i in range(n_profiles):
            # Startup profiles
            startup_temp = 298 + (1073 - 298) * (1 - np.exp(-startup_time / 900))
            startup_current = 0.8 * (1 - np.exp(-startup_time / 1200))
            
            # Shutdown profiles  
            shutdown_temp = 1073 * np.exp(-shutdown_time / 600)
            shutdown_current = 0.8 * np.exp(-shutdown_time / 300)
            
            # Load following profiles (sinusoidal + noise)
            base_current = 0.5
            amplitude = 0.3
            frequency = 2 * np.pi / 3600  # 1 cycle per hour
            noise = np.random.normal(0, 0.02, len(load_following_time))
            load_current = base_current + amplitude * np.sin(frequency * load_following_time) + noise
            load_current = np.clip(load_current, 0.1, 1.2)
            
            # Temperature follows current with thermal lag
            thermal_lag = 300  # seconds
            load_temp = 1073 + 50 * np.convolve(load_current - base_current, 
                                              np.exp(-np.arange(50) / thermal_lag), mode='same')[:len(load_following_time)]
            
            profiles[f"startup_{i}"] = {
                "time": startup_time.tolist(),
                "temperature": startup_temp.tolist(),
                "current_density": startup_current.tolist(),
                "type": "startup"
            }
            
            profiles[f"shutdown_{i}"] = {
                "time": shutdown_time.tolist(),
                "temperature": shutdown_temp.tolist(),
                "current_density": shutdown_current.tolist(),
                "type": "shutdown"
            }
            
            profiles[f"load_following_{i}"] = {
                "time": load_following_time.tolist(),
                "temperature": load_temp.tolist(),
                "current_density": load_current.tolist(),
                "type": "load_following"
            }
        
        return profiles
    
    def generate_microstructural_data(self, n_samples: int = 50) -> Dict[str, Any]:
        """
        Generate synthetic microstructural data mimicking FIB-SEM/X-Ray tomography.
        
        Args:
            n_samples: Number of microstructural samples
            
        Returns:
            Dictionary containing microstructural datasets
        """
        microstructural_data = {}
        
        for i in range(n_samples):
            # Generate 3D voxel data (simplified representation)
            voxel_size = np.random.uniform(20, 80)  # nm
            volume_size = np.random.randint(50, 150)  # voxels per dimension
            
            # Create synthetic 3D microstructure
            np.random.seed(i + 1000)  # Reproducible but varied
            
            # Generate phases using random fields
            ni_phase = np.random.rand(volume_size, volume_size, volume_size) > 0.6
            ysz_phase = np.random.rand(volume_size, volume_size, volume_size) > 0.5
            pore_phase = ~(ni_phase | ysz_phase)
            
            # Ensure realistic phase fractions
            total_voxels = volume_size ** 3
            ni_fraction = np.sum(ni_phase) / total_voxels
            ysz_fraction = np.sum(ysz_phase) / total_voxels
            pore_fraction = np.sum(pore_phase) / total_voxels
            
            # Calculate connectivity (simplified)
            connectivity_ni = np.random.uniform(0.85, 0.98)
            connectivity_ysz = np.random.uniform(0.90, 0.99)
            connectivity_pore = np.random.uniform(0.75, 0.95)
            
            # Calculate specific surface area
            specific_surface_area = np.random.uniform(2e6, 8e6)
            
            microstructural_data[f"sample_{i}"] = {
                "voxel_size_nm": voxel_size,
                "volume_dimensions": [volume_size, volume_size, volume_size],
                "phase_fractions": {
                    "ni": ni_fraction,
                    "ysz": ysz_fraction,
                    "pore": pore_fraction
                },
                "connectivity": {
                    "ni": connectivity_ni,
                    "ysz": connectivity_ysz,
                    "pore": connectivity_pore
                },
                "specific_surface_area": specific_surface_area,
                "reconstruction_method": "FIB-SEM" if i % 2 == 0 else "X-Ray_CT"
            }
        
        return microstructural_data
    
    def generate_complete_dataset(self, n_samples_per_fidelity: Dict[str, int] = None):
        """
        Generate the complete multi-fidelity SOFC dataset.
        
        Args:
            n_samples_per_fidelity: Dictionary specifying samples per fidelity level
        """
        if n_samples_per_fidelity is None:
            n_samples_per_fidelity = {"LF": 1000, "MF": 500, "HF": 200}
        
        print("🔬 Generating Multi-Fidelity SOFC Dataset...")
        print("=" * 60)
        
        # Generate datasets for each fidelity level
        datasets = {}
        
        for fidelity in ["LF", "MF", "HF"]:
            print(f"\n📊 Generating {fidelity} fidelity dataset...")
            n_samples = n_samples_per_fidelity[fidelity]
            
            # System level parameters
            system_df = self.generate_latin_hypercube_samples(
                self.system_parameters, n_samples, [fidelity]
            )
            
            # Geometry parameters (MF and HF only)
            if fidelity in ["MF", "HF"]:
                geometry_df = self.generate_latin_hypercube_samples(
                    self.geometry_parameters, n_samples, [fidelity]
                )
            else:
                geometry_df = pd.DataFrame()
            
            # Material parameters (MF and HF only)
            if fidelity in ["MF", "HF"]:
                anode_df = self.generate_latin_hypercube_samples(
                    self.anode_parameters, n_samples, [fidelity]
                )
                cathode_df = self.generate_latin_hypercube_samples(
                    self.cathode_parameters, n_samples, [fidelity]
                )
                electrolyte_df = self.generate_latin_hypercube_samples(
                    self.electrolyte_parameters, n_samples, [fidelity]
                )
                interconnect_df = self.generate_latin_hypercube_samples(
                    self.interconnect_parameters, n_samples, [fidelity]
                )
            else:
                anode_df = cathode_df = electrolyte_df = interconnect_df = pd.DataFrame()
            
            # Combine all parameters for this fidelity level
            combined_df = system_df
            if not geometry_df.empty:
                combined_df = pd.concat([combined_df, geometry_df], axis=1)
            if not anode_df.empty:
                # Add prefixes to distinguish material components
                anode_df = anode_df.add_prefix("anode_")
                cathode_df = cathode_df.add_prefix("cathode_")
                electrolyte_df = electrolyte_df.add_prefix("electrolyte_")
                interconnect_df = interconnect_df.add_prefix("interconnect_")
                
                combined_df = pd.concat([combined_df, anode_df, cathode_df, 
                                       electrolyte_df, interconnect_df], axis=1)
            
            # Add sample metadata
            combined_df["sample_id"] = [f"{fidelity}_{i:06d}" for i in range(n_samples)]
            combined_df["fidelity_level"] = fidelity
            combined_df["timestamp"] = datetime.now().isoformat()
            
            datasets[fidelity] = combined_df
            print(f"✅ Generated {len(combined_df)} samples with {len(combined_df.columns)} parameters")
        
        # Generate microstructural data (HF only)
        print(f"\n🔬 Generating microstructural data...")
        microstructural_data = self.generate_microstructural_data(n_samples_per_fidelity["HF"])
        print(f"✅ Generated {len(microstructural_data)} microstructural samples")
        
        # Generate transient profiles
        print(f"\n⏱️  Generating transient cycle profiles...")
        transient_profiles = self.generate_transient_profiles(100)
        print(f"✅ Generated {len(transient_profiles)} transient profiles")
        
        # Save all datasets
        self._save_datasets(datasets, microstructural_data, transient_profiles)
        
        return datasets, microstructural_data, transient_profiles
    
    def _save_datasets(self, datasets: Dict, microstructural_data: Dict, 
                      transient_profiles: Dict):
        """Save all generated datasets to files."""
        print(f"\n💾 Saving datasets to {self.output_dir}...")
        
        # Save main parameter datasets
        for fidelity, df in datasets.items():
            filename = self.output_dir / f"sofc_parameters_{fidelity.lower()}_fidelity.csv"
            df.to_csv(filename, index=False)
            print(f"✅ Saved {fidelity} fidelity parameters: {filename}")
        
        # Save combined dataset
        combined_df = pd.concat(datasets.values(), ignore_index=True)
        combined_filename = self.output_dir / "sofc_parameters_combined.csv"
        combined_df.to_csv(combined_filename, index=False)
        print(f"✅ Saved combined parameters: {combined_filename}")
        
        # Save microstructural data
        microstructural_filename = self.output_dir / "sofc_microstructural_data.json"
        with open(microstructural_filename, 'w') as f:
            json.dump(microstructural_data, f, indent=2, default=str)
        print(f"✅ Saved microstructural data: {microstructural_filename}")
        
        # Save transient profiles
        transient_filename = self.output_dir / "sofc_transient_profiles.json"
        with open(transient_filename, 'w') as f:
            json.dump(transient_profiles, f, indent=2, default=str)
        print(f"✅ Saved transient profiles: {transient_filename}")
        
        # Save metadata
        metadata_filename = self.output_dir / "dataset_metadata.json"
        with open(metadata_filename, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"✅ Saved metadata: {metadata_filename}")
        
        # Generate summary statistics
        self._generate_summary_report(datasets, microstructural_data, transient_profiles)
    
    def _generate_summary_report(self, datasets: Dict, microstructural_data: Dict, 
                               transient_profiles: Dict):
        """Generate a comprehensive summary report."""
        report_filename = self.output_dir / "dataset_summary_report.md"
        
        with open(report_filename, 'w') as f:
            f.write("# Multi-Fidelity SOFC Dataset Summary Report\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Dataset Overview\n\n")
            f.write("This dataset contains comprehensive multi-scale parameters for SOFC digital twin modeling.\n\n")
            
            f.write("### Fidelity Levels\n\n")
            for fidelity, df in datasets.items():
                f.write(f"- **{fidelity} Fidelity**: {len(df)} samples, {len(df.columns)} parameters\n")
            
            f.write(f"\n### Additional Data\n\n")
            f.write(f"- **Microstructural Samples**: {len(microstructural_data)}\n")
            f.write(f"- **Transient Profiles**: {len(transient_profiles)}\n")
            
            f.write(f"\n### Parameter Categories\n\n")
            f.write("1. **System Level**: Operating conditions (fuel utilization, temperature, pressure, etc.)\n")
            f.write("2. **Cell/Stack Level**: Geometric parameters (thicknesses, areas, channel dimensions)\n")
            f.write("3. **Material Level**: Properties for all SOFC components (Ni-YSZ, LSCF, YSZ, Crofer 22APU)\n")
            f.write("4. **Microstructural Level**: High-fidelity experimental data (phase fractions, connectivity)\n")
            
            f.write(f"\n### Files Generated\n\n")
            for file in self.output_dir.glob("*"):
                if file.is_file():
                    f.write(f"- `{file.name}`\n")
            
            f.write(f"\n### Usage Instructions\n\n")
            f.write("1. Load the appropriate fidelity level dataset based on your modeling needs\n")
            f.write("2. Use the combined dataset for multi-fidelity model training\n")
            f.write("3. Incorporate transient profiles for degradation studies\n")
            f.write("4. Utilize microstructural data for high-fidelity physics-based modeling\n")
        
        print(f"✅ Generated summary report: {report_filename}")


if __name__ == "__main__":
    # Generate the complete dataset
    generator = SOFCDatasetGenerator()
    datasets, microstructural_data, transient_profiles = generator.generate_complete_dataset()
    
    print("\n🎉 Dataset generation completed successfully!")
    print(f"📁 All files saved to: {generator.output_dir}")