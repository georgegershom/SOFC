#!/usr/bin/env python3
"""
Multi-Fidelity Digital Twin for SOFCs: Dataset Generator
Multi-Scale Modeling and Deep Learning for Predicting Thermo-Mechanical Degradation

This module generates comprehensive datasets for SOFC digital twin modeling across
multiple fidelity levels (Low, Medium, High) and scales (System, Cell/Stack, Microstructural).
"""

import numpy as np
import pandas as pd
import h5py
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import qmc
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SOFCDatasetGenerator:
    """
    Comprehensive dataset generator for SOFC multi-fidelity digital twin modeling.
    """
    
    def __init__(self, random_seed=42):
        """Initialize the dataset generator with random seed for reproducibility."""
        np.random.seed(random_seed)
        self.random_seed = random_seed
        
        # Define fidelity levels
        self.fidelity_levels = ['LF', 'MF', 'HF']  # Low, Medium, High Fidelity
        
        # Define scales
        self.scales = ['System', 'Cell_Stack', 'Microstructural']
        
        # Initialize parameter ranges based on literature and experimental data
        self._define_parameter_ranges()
        
    def _define_parameter_ranges(self):
        """Define realistic parameter ranges based on SOFC literature and experimental data."""
        
        # System-level Operating Conditions
        self.system_params = {
            'fuel_utilization': (0.6, 0.9),  # Uf
            'oxidant_utilization': (0.15, 0.25),
            'current_density': (0.1, 1.0),  # A/cm²
            'voltage': (0.6, 1.0),  # V
            'temperature': (973, 1073),  # K (700-800°C)
            'pressure': (1.0, 1.2),  # atm
            'fuel_flow_rate': (50, 200),  # sccm
            'air_flow_rate': (200, 800),  # sccm
            'fuel_composition_h2': (0.8, 1.0),  # H2 mole fraction
            'fuel_composition_h2o': (0.0, 0.2),  # H2O mole fraction
            'air_composition_o2': (0.18, 0.21),  # O2 mole fraction
            'air_composition_n2': (0.79, 0.82),  # N2 mole fraction
        }
        
        # Transient cycle parameters
        self.transient_params = {
            'startup_ramp_rate': (1, 10),  # K/min
            'shutdown_ramp_rate': (1, 15),  # K/min
            'load_following_ramp_rate': (0.1, 1.0),  # A/cm²/min
            'thermal_cycling_frequency': (0.1, 2.0),  # cycles/hour
            'load_cycling_frequency': (0.5, 5.0),  # cycles/hour
        }
        
        # Cell/Stack Geometry Parameters
        self.geometry_params = {
            'cell_active_area': (25, 100),  # cm²
            'anode_thickness': (200, 800),  # μm
            'cathode_thickness': (20, 100),  # μm
            'electrolyte_thickness': (5, 20),  # μm
            'interconnect_thickness': (100, 300),  # μm
            'channel_width': (0.5, 2.0),  # mm
            'channel_height': (0.5, 2.0),  # mm
            'rib_width': (0.5, 2.0),  # mm
            'number_of_cells': (1, 50),  # cells in stack
        }
        
        # Anode (Ni-YSZ) Material Properties
        self.anode_params = {
            'porosity': (0.2, 0.4),
            'tortuosity': (2.0, 6.0),
            'ni_particle_size': (0.5, 3.0),  # μm
            'ysz_particle_size': (0.3, 1.0),  # μm
            'tpb_density': (1e12, 1e15),  # m⁻²
            'ionic_conductivity': (0.01, 0.1),  # S/m
            'electronic_conductivity': (1000, 10000),  # S/m
            'youngs_modulus': (50, 200),  # GPa
            'poisson_ratio': (0.25, 0.35),
            'thermal_expansion_coefficient': (10e-6, 15e-6),  # K⁻¹
            'creep_exponent': (1.5, 3.0),
            'creep_activation_energy': (100, 200),  # kJ/mol
        }
        
        # Cathode (LSCF) Material Properties
        self.cathode_params = {
            'porosity': (0.15, 0.35),
            'tortuosity': (2.5, 7.0),
            'lscf_particle_size': (0.2, 2.0),  # μm
            'gdc_particle_size': (0.1, 0.8),  # μm
            'tpb_density': (1e12, 1e15),  # m⁻²
            'ionic_conductivity': (0.1, 1.0),  # S/m
            'electronic_conductivity': (100, 1000),  # S/m
            'youngs_modulus': (80, 150),  # GPa
            'poisson_ratio': (0.25, 0.35),
            'thermal_expansion_coefficient': (12e-6, 20e-6),  # K⁻¹
            'chemical_expansion_coefficient': (0.5e-4, 2.0e-4),  # K⁻¹
            'creep_exponent': (1.8, 3.5),
            'creep_activation_energy': (120, 250),  # kJ/mol
        }
        
        # Electrolyte (YSZ) Material Properties
        self.electrolyte_params = {
            'ionic_conductivity': (0.01, 0.1),  # S/m
            'youngs_modulus': (200, 300),  # GPa
            'poisson_ratio': (0.3, 0.4),
            'thermal_expansion_coefficient': (10e-6, 12e-6),  # K⁻¹
            'fracture_toughness': (1.0, 3.0),  # MPa·m^0.5
            'weibull_modulus': (5, 15),
            'weibull_scale_parameter': (100, 500),  # MPa
        }
        
        # Interconnect (Crofer 22APU) Material Properties
        self.interconnect_params = {
            'thermal_expansion_coefficient': (11e-6, 13e-6),  # K⁻¹
            'youngs_modulus': (200, 250),  # GPa
            'poisson_ratio': (0.3, 0.35),
            'creep_exponent': (2.0, 4.0),
            'creep_activation_energy': (150, 300),  # kJ/mol
            'oxide_scale_growth_rate': (1e-15, 1e-12),  # m²/s
            'electrical_resistivity': (1e-6, 1e-5),  # Ω·m
        }
        
        # Microstructural Properties (from FIB-SEM/X-Ray Tomography)
        self.microstructural_params = {
            'phase_fraction_ni': (0.25, 0.45),
            'phase_fraction_ysz': (0.35, 0.55),
            'phase_fraction_pore': (0.15, 0.35),
            'specific_surface_area': (1e4, 1e6),  # m²/m³
            'connectivity_ni': (0.6, 0.9),
            'connectivity_ysz': (0.7, 0.95),
            'pore_size_distribution_mean': (0.1, 2.0),  # μm
            'pore_size_distribution_std': (0.05, 1.0),  # μm
            'particle_size_distribution_mean': (0.5, 2.5),  # μm
            'particle_size_distribution_std': (0.2, 1.0),  # μm
            'tortuosity_factor': (1.5, 4.0),
        }
        
        # Degradation parameters
        self.degradation_params = {
            'ni_agglomeration_rate': (1e-6, 1e-4),  # s⁻¹
            'ni_oxidation_rate': (1e-8, 1e-6),  # s⁻¹
            'cathode_poisoning_rate': (1e-7, 1e-5),  # s⁻¹
            'electrolyte_cracking_rate': (1e-9, 1e-7),  # s⁻¹
            'interconnect_corrosion_rate': (1e-8, 1e-6),  # s⁻¹
            'thermal_stress_accumulation': (0.1, 10.0),  # MPa
            'redox_cycling_damage': (0.01, 1.0),  # dimensionless
        }

    def generate_lhs_samples(self, n_samples, param_ranges, fidelity_level='MF'):
        """
        Generate Latin Hypercube Samples for given parameter ranges.
        
        Args:
            n_samples (int): Number of samples to generate
            param_ranges (dict): Dictionary of parameter ranges
            fidelity_level (str): Fidelity level ('LF', 'MF', 'HF')
            
        Returns:
            dict: Dictionary of sampled parameters
        """
        # Adjust sample size based on fidelity level
        fidelity_multiplier = {'LF': 0.5, 'MF': 1.0, 'HF': 2.0}
        adjusted_samples = int(n_samples * fidelity_multiplier[fidelity_level])
        
        # Create Latin Hypercube Sampler
        sampler = qmc.LatinHypercube(d=len(param_ranges), seed=self.random_seed)
        samples = sampler.random(n=adjusted_samples)
        
        # Scale samples to parameter ranges
        param_names = list(param_ranges.keys())
        param_values = {}
        
        for i, param_name in enumerate(param_names):
            min_val, max_val = param_ranges[param_name]
            param_values[param_name] = samples[:, i] * (max_val - min_val) + min_val
        
        # Ensure all arrays have the same length
        target_length = len(param_values[param_names[0]])
        for param_name in param_names:
            if len(param_values[param_name]) != target_length:
                param_values[param_name] = param_values[param_name][:target_length]
            
        return param_values

    def generate_operating_conditions(self, n_samples=1000, fidelity_level='MF'):
        """Generate system-level operating conditions."""
        print(f"Generating {n_samples} operating condition samples for {fidelity_level}...")
        
        # Combine system and transient parameters
        all_params = {**self.system_params, **self.transient_params}
        
        # Generate samples
        samples = self.generate_lhs_samples(n_samples, all_params, fidelity_level)
        
        # Add derived parameters
        samples['power_density'] = samples['current_density'] * samples['voltage']  # W/cm²
        actual_n_samples = len(samples['fuel_utilization'])
        samples['fuel_utilization_actual'] = samples['fuel_utilization'] * (1 + np.random.normal(0, 0.05, actual_n_samples))
        samples['efficiency'] = samples['voltage'] / 1.25  # Approximate efficiency
        
        # Add fidelity level
        samples['fidelity_level'] = [fidelity_level] * n_samples
        
        return samples

    def generate_geometry_parameters(self, n_samples=1000, fidelity_level='MF'):
        """Generate cell/stack geometry parameters."""
        print(f"Generating {n_samples} geometry parameter samples for {fidelity_level}...")
        
        samples = self.generate_lhs_samples(n_samples, self.geometry_params, fidelity_level)
        
        # Add derived geometric parameters
        samples['aspect_ratio'] = samples['cell_active_area'] / (samples['anode_thickness'] * 1e-4)
        samples['total_thickness'] = (samples['anode_thickness'] + 
                                   samples['cathode_thickness'] + 
                                   samples['electrolyte_thickness'] + 
                                   samples['interconnect_thickness'])
        samples['volume_fraction_anode'] = samples['anode_thickness'] / samples['total_thickness']
        samples['volume_fraction_cathode'] = samples['cathode_thickness'] / samples['total_thickness']
        samples['volume_fraction_electrolyte'] = samples['electrolyte_thickness'] / samples['total_thickness']
        
        samples['fidelity_level'] = [fidelity_level] * n_samples
        
        return samples

    def generate_material_properties(self, n_samples=1000, fidelity_level='MF'):
        """Generate material properties for all components."""
        print(f"Generating {n_samples} material property samples for {fidelity_level}...")
        
        # Generate samples for each component
        anode_samples = self.generate_lhs_samples(n_samples, self.anode_params, fidelity_level)
        cathode_samples = self.generate_lhs_samples(n_samples, self.cathode_params, fidelity_level)
        electrolyte_samples = self.generate_lhs_samples(n_samples, self.electrolyte_params, fidelity_level)
        interconnect_samples = self.generate_lhs_samples(n_samples, self.interconnect_params, fidelity_level)
        
        # Combine all material properties
        all_samples = {}
        
        # Add component prefixes
        for key, values in anode_samples.items():
            all_samples[f'anode_{key}'] = values
            
        for key, values in cathode_samples.items():
            all_samples[f'cathode_{key}'] = values
            
        for key, values in electrolyte_samples.items():
            all_samples[f'electrolyte_{key}'] = values
            
        for key, values in interconnect_samples.items():
            all_samples[f'interconnect_{key}'] = values
        
        # Add derived material properties
        all_samples['anode_effective_conductivity'] = (
            1 / (1/all_samples['anode_ionic_conductivity'] + 1/all_samples['anode_electronic_conductivity'])
        )
        all_samples['cathode_effective_conductivity'] = (
            1 / (1/all_samples['cathode_ionic_conductivity'] + 1/all_samples['cathode_electronic_conductivity'])
        )
        
        all_samples['fidelity_level'] = [fidelity_level] * n_samples
        
        return all_samples

    def generate_microstructural_properties(self, n_samples=1000, fidelity_level='HF'):
        """Generate microstructural properties from FIB-SEM/X-Ray data."""
        print(f"Generating {n_samples} microstructural property samples for {fidelity_level}...")
        
        samples = self.generate_lhs_samples(n_samples, self.microstructural_params, fidelity_level)
        
        # Add derived microstructural parameters
        samples['total_porosity'] = samples['phase_fraction_pore']
        samples['solid_phase_fraction'] = 1 - samples['phase_fraction_pore']
        samples['ni_ysz_ratio'] = samples['phase_fraction_ni'] / samples['phase_fraction_ysz']
        
        # Generate 3D voxel data simulation (simplified)
        voxel_size = 100  # 100x100x100 voxels
        samples['voxel_data_shape'] = [voxel_size] * n_samples
        samples['voxel_data_dtype'] = ['float32'] * n_samples
        
        samples['fidelity_level'] = [fidelity_level] * n_samples
        
        return samples

    def generate_degradation_parameters(self, n_samples=1000, fidelity_level='MF'):
        """Generate degradation parameters for thermo-mechanical analysis."""
        print(f"Generating {n_samples} degradation parameter samples for {fidelity_level}...")
        
        samples = self.generate_lhs_samples(n_samples, self.degradation_params, fidelity_level)
        
        # Add derived degradation metrics
        samples['total_degradation_rate'] = (
            samples['ni_agglomeration_rate'] + 
            samples['ni_oxidation_rate'] + 
            samples['cathode_poisoning_rate'] + 
            samples['electrolyte_cracking_rate'] + 
            samples['interconnect_corrosion_rate']
        )
        
        samples['thermal_mechanical_damage'] = (
            samples['thermal_stress_accumulation'] * 
            samples['redox_cycling_damage']
        )
        
        samples['fidelity_level'] = [fidelity_level] * n_samples
        
        return samples

    def generate_comprehensive_dataset(self, n_samples_per_fidelity=1000):
        """Generate comprehensive multi-fidelity dataset."""
        print("Generating comprehensive multi-fidelity SOFC dataset...")
        
        all_datasets = {}
        
        for fidelity in self.fidelity_levels:
            print(f"\n=== Generating {fidelity} Fidelity Dataset ===")
            
            # Use consistent sample size for all fidelity levels
            actual_samples = n_samples_per_fidelity
            
            # Generate all parameter categories
            operating_conditions = self.generate_operating_conditions(actual_samples, fidelity)
            geometry_params = self.generate_geometry_parameters(actual_samples, fidelity)
            material_props = self.generate_material_properties(actual_samples, fidelity)
            microstructural_props = self.generate_microstructural_properties(actual_samples, fidelity)
            degradation_params = self.generate_degradation_parameters(actual_samples, fidelity)
            
            # Find the minimum length across all datasets
            min_length = min(
                len(operating_conditions[list(operating_conditions.keys())[0]]),
                len(geometry_params[list(geometry_params.keys())[0]]),
                len(material_props[list(material_props.keys())[0]]),
                len(microstructural_props[list(microstructural_props.keys())[0]]),
                len(degradation_params[list(degradation_params.keys())[0]])
            )
            
            # Truncate all datasets to the same length
            combined_data = {}
            for data_dict in [operating_conditions, geometry_params, material_props, 
                            microstructural_props, degradation_params]:
                for key, values in data_dict.items():
                    if isinstance(values, (list, np.ndarray)):
                        combined_data[key] = values[:min_length]
                    else:
                        combined_data[key] = values
            
            all_datasets[fidelity] = combined_data
        
        return all_datasets

    def save_dataset(self, datasets, output_dir='sofc_dataset'):
        """Save dataset in multiple formats."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving dataset to {output_dir}/...")
        
        # Save as CSV files
        for fidelity, data in datasets.items():
            # Debug: Check array lengths
            lengths = {key: len(values) if hasattr(values, '__len__') else 1 for key, values in data.items()}
            print(f"Debug - {fidelity} array lengths: {dict(list(lengths.items())[:5])}...")
            
            # Ensure all arrays have the same length
            max_length = max(lengths.values())
            for key, values in data.items():
                if hasattr(values, '__len__') and len(values) != max_length:
                    if isinstance(values, list):
                        data[key] = values[:max_length]
                    elif isinstance(values, np.ndarray):
                        data[key] = values[:max_length]
                    elif isinstance(values, str):
                        # For string values, repeat to match length
                        data[key] = [values] * max_length
            
            df = pd.DataFrame(data)
            csv_path = f"{output_dir}/sofc_dataset_{fidelity.lower()}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved {csv_path}")
        
        # Save as HDF5
        h5_path = f"{output_dir}/sofc_dataset.h5"
        with h5py.File(h5_path, 'w') as f:
            for fidelity, data in datasets.items():
                group = f.create_group(fidelity)
                for key, values in data.items():
                    if isinstance(values, list) and all(isinstance(v, str) for v in values):
                        # String data
                        group.create_dataset(key, data=[v.encode('utf-8') for v in values])
                    else:
                        # Numeric data
                        group.create_dataset(key, data=values)
        print(f"Saved {h5_path}")
        
        # Save metadata
        metadata = {
            'description': 'Multi-Fidelity Digital Twin for SOFCs Dataset',
            'fidelity_levels': self.fidelity_levels,
            'scales': self.scales,
            'total_samples': sum(len(data[list(data.keys())[0]]) for data in datasets.values()),
            'parameter_categories': {
                'system_parameters': list(self.system_params.keys()),
                'transient_parameters': list(self.transient_params.keys()),
                'geometry_parameters': list(self.geometry_params.keys()),
                'anode_parameters': list(self.anode_params.keys()),
                'cathode_parameters': list(self.cathode_params.keys()),
                'electrolyte_parameters': list(self.electrolyte_params.keys()),
                'interconnect_parameters': list(self.interconnect_params.keys()),
                'microstructural_parameters': list(self.microstructural_params.keys()),
                'degradation_parameters': list(self.degradation_params.keys()),
            },
            'generation_info': {
                'random_seed': self.random_seed,
                'sampling_method': 'Latin Hypercube Sampling',
                'generation_date': pd.Timestamp.now().isoformat(),
            }
        }
        
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved {output_dir}/metadata.json")
        
        return output_dir

    def create_visualizations(self, datasets, output_dir='sofc_dataset'):
        """Create comprehensive visualizations of the dataset."""
        print("Creating dataset visualizations...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create visualization directory
        viz_dir = f"{output_dir}/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Parameter distribution plots
        self._plot_parameter_distributions(datasets, viz_dir)
        
        # 2. Fidelity level comparison plots
        self._plot_fidelity_comparisons(datasets, viz_dir)
        
        # 3. Correlation heatmaps
        self._plot_correlation_heatmaps(datasets, viz_dir)
        
        # 4. Multi-dimensional parameter space visualization
        self._plot_parameter_space(datasets, viz_dir)
        
        print(f"Visualizations saved to {viz_dir}/")

    def _plot_parameter_distributions(self, datasets, viz_dir):
        """Plot parameter distributions for each fidelity level."""
        for fidelity, data in datasets.items():
            df = pd.DataFrame(data)
            
            # Select key parameters for visualization
            key_params = ['temperature', 'current_density', 'voltage', 'fuel_utilization',
                         'anode_porosity', 'cathode_porosity', 'electrolyte_thickness']
            
            # Filter to existing parameters
            existing_params = [p for p in key_params if p in df.columns]
            
            if existing_params:
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                axes = axes.flatten()
                
                for i, param in enumerate(existing_params[:8]):
                    if i < len(axes):
                        axes[i].hist(df[param], bins=50, alpha=0.7, edgecolor='black')
                        axes[i].set_title(f'{param} - {fidelity}')
                        axes[i].set_xlabel(param)
                        axes[i].set_ylabel('Frequency')
                
                # Hide unused subplots
                for i in range(len(existing_params), len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(f"{viz_dir}/parameter_distributions_{fidelity.lower()}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()

    def _plot_fidelity_comparisons(self, datasets, viz_dir):
        """Plot comparisons across fidelity levels."""
        # Compare key parameters across fidelity levels
        key_params = ['temperature', 'current_density', 'voltage', 'fuel_utilization']
        
        for param in key_params:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for fidelity, data in datasets.items():
                if param in data:
                    ax.hist(data[param], bins=50, alpha=0.6, label=fidelity, density=True)
            
            ax.set_title(f'{param} Distribution Across Fidelity Levels')
            ax.set_xlabel(param)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/fidelity_comparison_{param}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_correlation_heatmaps(self, datasets, viz_dir):
        """Plot correlation heatmaps for each fidelity level."""
        for fidelity, data in datasets.items():
            df = pd.DataFrame(data)
            
            # Select numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 1:
                # Calculate correlation matrix
                corr_matrix = df[numeric_cols].corr()
                
                # Create heatmap
                plt.figure(figsize=(15, 12))
                sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0,
                           square=True, cbar_kws={'shrink': 0.8})
                plt.title(f'Parameter Correlation Matrix - {fidelity}')
                plt.tight_layout()
                plt.savefig(f"{viz_dir}/correlation_heatmap_{fidelity.lower()}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()

    def _plot_parameter_space(self, datasets, viz_dir):
        """Plot multi-dimensional parameter space using PCA."""
        from sklearn.decomposition import PCA
        
        # Combine all fidelity data
        all_data = []
        fidelity_labels = []
        
        for fidelity, data in datasets.items():
            df = pd.DataFrame(data)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                all_data.append(df[numeric_cols].values)
                fidelity_labels.extend([fidelity] * len(df))
        
        if all_data:
            # Combine all data
            X = np.vstack(all_data)
            
            # Apply PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            # Plot
            plt.figure(figsize=(12, 8))
            colors = ['red', 'green', 'blue']
            for i, fidelity in enumerate(self.fidelity_levels):
                mask = np.array(fidelity_labels) == fidelity
                if np.any(mask):
                    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                              c=colors[i], label=fidelity, alpha=0.6, s=20)
            
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('Multi-Dimensional Parameter Space (PCA)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/parameter_space_pca.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()

def main():
    """Main function to generate the complete SOFC dataset."""
    print("=== Multi-Fidelity SOFC Dataset Generator ===")
    print("Generating comprehensive dataset for SOFC digital twin modeling...")
    
    # Initialize generator
    generator = SOFCDatasetGenerator(random_seed=42)
    
    # Generate comprehensive dataset
    datasets = generator.generate_comprehensive_dataset(n_samples_per_fidelity=1000)
    
    # Save dataset
    output_dir = generator.save_dataset(datasets)
    
    # Create visualizations
    generator.create_visualizations(datasets, output_dir)
    
    print(f"\n=== Dataset Generation Complete ===")
    print(f"Dataset saved to: {output_dir}/")
    print(f"Total samples generated: {sum(len(data[list(data.keys())[0]]) for data in datasets.values())}")
    print(f"Fidelity levels: {', '.join(datasets.keys())}")
    
    return datasets, output_dir

if __name__ == "__main__":
    datasets, output_dir = main()