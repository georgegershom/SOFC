#!/usr/bin/env python3
"""
Simplified Multi-Fidelity SOFC Dataset Generator
"""

import numpy as np
import pandas as pd
import h5py
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import qmc
import os
import warnings
warnings.filterwarnings('ignore')

class SimpleSOFCDatasetGenerator:
    """Simplified SOFC dataset generator with consistent sample sizes."""
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.fidelity_levels = ['LF', 'MF', 'HF']
        
        # Define parameter ranges
        self._define_parameter_ranges()
        
    def _define_parameter_ranges(self):
        """Define parameter ranges for SOFC modeling."""
        
        # System Operating Conditions
        self.system_params = {
            'fuel_utilization': (0.6, 0.9),
            'oxidant_utilization': (0.15, 0.25),
            'current_density': (0.1, 1.0),  # A/cm²
            'voltage': (0.6, 1.0),  # V
            'temperature': (973, 1073),  # K
            'pressure': (1.0, 1.2),  # atm
            'fuel_flow_rate': (50, 200),  # sccm
            'air_flow_rate': (200, 800),  # sccm
        }
        
        # Transient Parameters
        self.transient_params = {
            'startup_ramp_rate': (1, 10),  # K/min
            'shutdown_ramp_rate': (1, 15),  # K/min
            'load_following_ramp_rate': (0.1, 1.0),  # A/cm²/min
            'thermal_cycling_frequency': (0.1, 2.0),  # cycles/hour
        }
        
        # Geometry Parameters
        self.geometry_params = {
            'cell_active_area': (25, 100),  # cm²
            'anode_thickness': (200, 800),  # μm
            'cathode_thickness': (20, 100),  # μm
            'electrolyte_thickness': (5, 20),  # μm
            'interconnect_thickness': (100, 300),  # μm
            'channel_width': (0.5, 2.0),  # mm
            'channel_height': (0.5, 2.0),  # mm
        }
        
        # Anode Material Properties
        self.anode_params = {
            'anode_porosity': (0.2, 0.4),
            'anode_tortuosity': (2.0, 6.0),
            'anode_ni_particle_size': (0.5, 3.0),  # μm
            'anode_ysz_particle_size': (0.3, 1.0),  # μm
            'anode_tpb_density': (1e12, 1e15),  # m⁻²
            'anode_ionic_conductivity': (0.01, 0.1),  # S/m
            'anode_electronic_conductivity': (1000, 10000),  # S/m
            'anode_youngs_modulus': (50, 200),  # GPa
            'anode_poisson_ratio': (0.25, 0.35),
            'anode_thermal_expansion_coefficient': (10e-6, 15e-6),  # K⁻¹
        }
        
        # Cathode Material Properties
        self.cathode_params = {
            'cathode_porosity': (0.15, 0.35),
            'cathode_tortuosity': (2.5, 7.0),
            'cathode_lscf_particle_size': (0.2, 2.0),  # μm
            'cathode_gdc_particle_size': (0.1, 0.8),  # μm
            'cathode_tpb_density': (1e12, 1e15),  # m⁻²
            'cathode_ionic_conductivity': (0.1, 1.0),  # S/m
            'cathode_electronic_conductivity': (100, 1000),  # S/m
            'cathode_youngs_modulus': (80, 150),  # GPa
            'cathode_poisson_ratio': (0.25, 0.35),
            'cathode_thermal_expansion_coefficient': (12e-6, 20e-6),  # K⁻¹
            'cathode_chemical_expansion_coefficient': (0.5e-4, 2.0e-4),  # K⁻¹
        }
        
        # Electrolyte Material Properties
        self.electrolyte_params = {
            'electrolyte_ionic_conductivity': (0.01, 0.1),  # S/m
            'electrolyte_youngs_modulus': (200, 300),  # GPa
            'electrolyte_poisson_ratio': (0.3, 0.4),
            'electrolyte_thermal_expansion_coefficient': (10e-6, 12e-6),  # K⁻¹
            'electrolyte_fracture_toughness': (1.0, 3.0),  # MPa·m^0.5
        }
        
        # Interconnect Material Properties
        self.interconnect_params = {
            'interconnect_thermal_expansion_coefficient': (11e-6, 13e-6),  # K⁻¹
            'interconnect_youngs_modulus': (200, 250),  # GPa
            'interconnect_poisson_ratio': (0.3, 0.35),
            'interconnect_oxide_scale_growth_rate': (1e-15, 1e-12),  # m²/s
            'interconnect_electrical_resistivity': (1e-6, 1e-5),  # Ω·m
        }
        
        # Microstructural Properties
        self.microstructural_params = {
            'phase_fraction_ni': (0.25, 0.45),
            'phase_fraction_ysz': (0.35, 0.55),
            'phase_fraction_pore': (0.15, 0.35),
            'specific_surface_area': (1e4, 1e6),  # m²/m³
            'connectivity_ni': (0.6, 0.9),
            'connectivity_ysz': (0.7, 0.95),
            'pore_size_distribution_mean': (0.1, 2.0),  # μm
            'particle_size_distribution_mean': (0.5, 2.5),  # μm
        }
        
        # Degradation Parameters
        self.degradation_params = {
            'ni_agglomeration_rate': (1e-6, 1e-4),  # s⁻¹
            'ni_oxidation_rate': (1e-8, 1e-6),  # s⁻¹
            'cathode_poisoning_rate': (1e-7, 1e-5),  # s⁻¹
            'electrolyte_cracking_rate': (1e-9, 1e-7),  # s⁻¹
            'thermal_stress_accumulation': (0.1, 10.0),  # MPa
            'redox_cycling_damage': (0.01, 1.0),  # dimensionless
        }

    def generate_lhs_samples(self, n_samples, param_ranges):
        """Generate Latin Hypercube Samples."""
        sampler = qmc.LatinHypercube(d=len(param_ranges), seed=self.random_seed)
        samples = sampler.random(n=n_samples)
        
        param_values = {}
        param_names = list(param_ranges.keys())
        
        for i, param_name in enumerate(param_names):
            min_val, max_val = param_ranges[param_name]
            param_values[param_name] = samples[:, i] * (max_val - min_val) + min_val
            
        return param_values

    def generate_dataset(self, n_samples=1000, fidelity_level='MF'):
        """Generate complete dataset for given fidelity level."""
        print(f"Generating {n_samples} samples for {fidelity_level} fidelity...")
        
        # Combine all parameter ranges
        all_params = {}
        all_params.update(self.system_params)
        all_params.update(self.transient_params)
        all_params.update(self.geometry_params)
        all_params.update(self.anode_params)
        all_params.update(self.cathode_params)
        all_params.update(self.electrolyte_params)
        all_params.update(self.interconnect_params)
        all_params.update(self.microstructural_params)
        all_params.update(self.degradation_params)
        
        # Generate samples
        samples = self.generate_lhs_samples(n_samples, all_params)
        
        # Add derived parameters
        samples['power_density'] = samples['current_density'] * samples['voltage']
        samples['efficiency'] = samples['voltage'] / 1.25
        samples['total_thickness'] = (samples['anode_thickness'] + 
                                   samples['cathode_thickness'] + 
                                   samples['electrolyte_thickness'] + 
                                   samples['interconnect_thickness'])
        
        # Add fidelity level
        samples['fidelity_level'] = [fidelity_level] * n_samples
        
        return samples

    def generate_multi_fidelity_dataset(self, n_samples_per_fidelity=1000):
        """Generate multi-fidelity dataset."""
        print("Generating multi-fidelity SOFC dataset...")
        
        all_datasets = {}
        
        for fidelity in self.fidelity_levels:
            print(f"\n=== Generating {fidelity} Fidelity Dataset ===")
            all_datasets[fidelity] = self.generate_dataset(n_samples_per_fidelity, fidelity)
        
        return all_datasets

    def save_dataset(self, datasets, output_dir='sofc_dataset'):
        """Save dataset in multiple formats."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving dataset to {output_dir}/...")
        
        # Save as CSV files
        for fidelity, data in datasets.items():
            df = pd.DataFrame(data)
            csv_path = f"{output_dir}/sofc_dataset_{fidelity.lower()}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved {csv_path} with {len(df)} samples and {len(df.columns)} parameters")
        
        # Save as HDF5
        h5_path = f"{output_dir}/sofc_dataset.h5"
        with h5py.File(h5_path, 'w') as f:
            for fidelity, data in datasets.items():
                group = f.create_group(fidelity)
                for key, values in data.items():
                    if isinstance(values, list) and all(isinstance(v, str) for v in values):
                        group.create_dataset(key, data=[v.encode('utf-8') for v in values])
                    else:
                        group.create_dataset(key, data=values)
        print(f"Saved {h5_path}")
        
        # Save metadata
        metadata = {
            'description': 'Multi-Fidelity Digital Twin for SOFCs Dataset',
            'fidelity_levels': self.fidelity_levels,
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
        """Create visualizations of the dataset."""
        print("Creating dataset visualizations...")
        
        viz_dir = f"{output_dir}/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Set up plotting
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Parameter distribution plots
        self._plot_parameter_distributions(datasets, viz_dir)
        
        # 2. Fidelity comparison plots
        self._plot_fidelity_comparisons(datasets, viz_dir)
        
        # 3. Correlation heatmap
        self._plot_correlation_heatmap(datasets, viz_dir)
        
        print(f"Visualizations saved to {viz_dir}/")

    def _plot_parameter_distributions(self, datasets, viz_dir):
        """Plot parameter distributions."""
        for fidelity, data in datasets.items():
            df = pd.DataFrame(data)
            
            # Select key parameters
            key_params = ['temperature', 'current_density', 'voltage', 'fuel_utilization',
                         'anode_porosity', 'cathode_porosity', 'electrolyte_thickness']
            
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
        """Plot fidelity level comparisons."""
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

    def _plot_correlation_heatmap(self, datasets, viz_dir):
        """Plot correlation heatmap."""
        for fidelity, data in datasets.items():
            df = pd.DataFrame(data)
            
            # Select numeric columns
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

def main():
    """Main function to generate the complete SOFC dataset."""
    print("=== Multi-Fidelity SOFC Dataset Generator ===")
    print("Generating comprehensive dataset for SOFC digital twin modeling...")
    
    # Initialize generator
    generator = SimpleSOFCDatasetGenerator(random_seed=42)
    
    # Generate multi-fidelity dataset
    datasets = generator.generate_multi_fidelity_dataset(n_samples_per_fidelity=1000)
    
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