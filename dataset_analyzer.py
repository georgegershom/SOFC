#!/usr/bin/env python3
"""
SOFC Dataset Analysis and Visualization Tool
============================================

This module provides comprehensive analysis and visualization capabilities
for the multi-fidelity SOFC dataset.

Author: Generated for PhD Thesis - Multi-Fidelity Digital Twin for SOFCs
Date: 2025-10-15
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class SOFCDatasetAnalyzer:
    """
    Comprehensive analyzer for SOFC multi-fidelity datasets.
    """
    
    def __init__(self, dataset_dir: str = "sofc_dataset"):
        """Initialize the analyzer with dataset directory."""
        self.dataset_dir = Path(dataset_dir)
        self.datasets = {}
        self.microstructural_data = {}
        self.transient_profiles = {}
        self.metadata = {}
        
        self._load_datasets()
    
    def _load_datasets(self):
        """Load all dataset files."""
        try:
            # Load parameter datasets
            for fidelity in ['lf', 'mf', 'hf']:
                file_path = self.dataset_dir / f"sofc_parameters_{fidelity}_fidelity.csv"
                if file_path.exists():
                    self.datasets[fidelity.upper()] = pd.read_csv(file_path)
            
            # Load combined dataset
            combined_path = self.dataset_dir / "sofc_parameters_combined.csv"
            if combined_path.exists():
                self.datasets['combined'] = pd.read_csv(combined_path)
            
            # Load microstructural data
            micro_path = self.dataset_dir / "sofc_microstructural_data.json"
            if micro_path.exists():
                with open(micro_path, 'r') as f:
                    self.microstructural_data = json.load(f)
            
            # Load transient profiles
            transient_path = self.dataset_dir / "sofc_transient_profiles.json"
            if transient_path.exists():
                with open(transient_path, 'r') as f:
                    self.transient_profiles = json.load(f)
            
            # Load metadata
            metadata_path = self.dataset_dir / "dataset_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            print("✅ Successfully loaded all dataset files")
            
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
    
    def generate_dataset_overview(self):
        """Generate comprehensive dataset overview."""
        print("\n" + "="*80)
        print("📊 SOFC MULTI-FIDELITY DATASET OVERVIEW")
        print("="*80)
        
        # Basic statistics
        print(f"\n📈 Dataset Statistics:")
        print(f"{'Fidelity Level':<15} {'Samples':<10} {'Parameters':<12} {'Memory (MB)':<12}")
        print("-" * 50)
        
        total_samples = 0
        for fidelity, df in self.datasets.items():
            if fidelity != 'combined':
                memory_mb = df.memory_usage(deep=True).sum() / 1024**2
                print(f"{fidelity:<15} {len(df):<10} {len(df.columns):<12} {memory_mb:.2f}")
                total_samples += len(df)
        
        print(f"{'TOTAL':<15} {total_samples:<10} {'-':<12} {'-'}")
        
        # Additional data
        print(f"\n🔬 Additional Data:")
        print(f"- Microstructural samples: {len(self.microstructural_data)}")
        print(f"- Transient profiles: {len(self.transient_profiles)}")
        
        # Parameter categories
        if 'HF' in self.datasets:
            df = self.datasets['HF']
            system_params = [col for col in df.columns if not any(prefix in col for prefix in ['anode_', 'cathode_', 'electrolyte_', 'interconnect_', 'cell_', 'channel_', 'rib_'])]
            geometry_params = [col for col in df.columns if any(prefix in col for prefix in ['cell_', 'channel_', 'rib_']) and 'thickness' in col]
            material_params = [col for col in df.columns if any(prefix in col for prefix in ['anode_', 'cathode_', 'electrolyte_', 'interconnect_'])]
            
            print(f"\n🏗️ Parameter Categories (HF Dataset):")
            print(f"- System level: {len(system_params)} parameters")
            print(f"- Geometry level: {len(geometry_params)} parameters") 
            print(f"- Material level: {len(material_params)} parameters")
    
    def plot_parameter_distributions(self, fidelity: str = 'HF', save_plots: bool = True):
        """Plot parameter distributions for a given fidelity level."""
        if fidelity not in self.datasets:
            print(f"❌ Fidelity level {fidelity} not found in datasets")
            return
        
        df = self.datasets[fidelity]
        
        # Filter numeric columns (exclude metadata columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if not col.endswith('_unit') and not col.endswith('_fidelity')]
        
        # System parameters
        system_params = [col for col in numeric_cols if not any(prefix in col for prefix in 
                        ['anode_', 'cathode_', 'electrolyte_', 'interconnect_', 'cell_', 'channel_', 'rib_'])]
        
        if system_params:
            n_cols = min(4, len(system_params))
            n_rows = (len(system_params) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, param in enumerate(system_params[:12]):  # Limit to first 12
                if i < len(axes):
                    axes[i].hist(df[param], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    axes[i].set_title(f'{param}', fontsize=10)
                    axes[i].set_xlabel('Value')
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(len(system_params), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.suptitle(f'System Parameter Distributions ({fidelity} Fidelity)', fontsize=14, y=1.02)
            
            if save_plots:
                plt.savefig(self.dataset_dir / f'system_parameters_{fidelity.lower()}.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Material parameters (if available)
        if fidelity in ['MF', 'HF']:
            material_params = [col for col in numeric_cols if any(prefix in col for prefix in 
                             ['anode_', 'cathode_', 'electrolyte_', 'interconnect_'])]
            
            if material_params:
                # Group by component
                components = ['anode', 'cathode', 'electrolyte', 'interconnect']
                
                for component in components:
                    comp_params = [col for col in material_params if col.startswith(f'{component}_')]
                    
                    if comp_params:
                        n_cols = min(3, len(comp_params))
                        n_rows = (len(comp_params) + n_cols - 1) // n_cols
                        
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
                        if n_rows == 1:
                            axes = [axes] if n_cols == 1 else axes
                        else:
                            axes = axes.flatten()
                        
                        for i, param in enumerate(comp_params):
                            if i < len(axes):
                                axes[i].hist(df[param], bins=25, alpha=0.7, color='lightcoral', edgecolor='black')
                                axes[i].set_title(f'{param.replace(f"{component}_", "")}', fontsize=10)
                                axes[i].set_xlabel('Value')
                                axes[i].set_ylabel('Frequency')
                                axes[i].grid(True, alpha=0.3)
                        
                        # Hide empty subplots
                        for i in range(len(comp_params), len(axes)):
                            axes[i].set_visible(False)
                        
                        plt.tight_layout()
                        plt.suptitle(f'{component.title()} Parameter Distributions ({fidelity} Fidelity)', 
                                   fontsize=14, y=1.02)
                        
                        if save_plots:
                            plt.savefig(self.dataset_dir / f'{component}_parameters_{fidelity.lower()}.png', 
                                      dpi=300, bbox_inches='tight')
                        plt.show()
    
    def plot_correlation_matrix(self, fidelity: str = 'HF', save_plot: bool = True):
        """Plot correlation matrix for key parameters."""
        if fidelity not in self.datasets:
            print(f"❌ Fidelity level {fidelity} not found in datasets")
            return
        
        df = self.datasets[fidelity]
        
        # Select key system parameters for correlation analysis
        key_params = [
            'fuel_utilization', 'oxidant_utilization', 'current_density', 
            'voltage', 'temperature', 'pressure'
        ]
        
        # Add geometry parameters if available
        if fidelity in ['MF', 'HF']:
            key_params.extend([
                'cell_active_area', 'anode_thickness', 'cathode_thickness', 'electrolyte_thickness'
            ])
        
        # Filter available parameters
        available_params = [param for param in key_params if param in df.columns]
        
        if len(available_params) < 2:
            print("❌ Not enough parameters for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = df[available_params].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        
        plt.title(f'Parameter Correlation Matrix ({fidelity} Fidelity)', fontsize=14, pad=20)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(self.dataset_dir / f'correlation_matrix_{fidelity.lower()}.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_transient_profiles(self, n_samples: int = 5, save_plot: bool = True):
        """Plot sample transient profiles."""
        if not self.transient_profiles:
            print("❌ No transient profiles data available")
            return
        
        # Group profiles by type
        profile_types = {}
        for key, profile in self.transient_profiles.items():
            profile_type = profile['type']
            if profile_type not in profile_types:
                profile_types[profile_type] = []
            profile_types[profile_type].append((key, profile))
        
        # Plot each type
        for profile_type, profiles in profile_types.items():
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Sample a few profiles
            sample_profiles = profiles[:n_samples]
            
            for key, profile in sample_profiles:
                time = np.array(profile['time'], dtype=float)
                temp = np.array(profile['temperature'], dtype=float)
                current = np.array(profile['current_density'], dtype=float)
                
                ax1.plot(time/3600, temp, alpha=0.7, label=f'{key}')
                ax2.plot(time/3600, current, alpha=0.7, label=f'{key}')
            
            ax1.set_ylabel('Temperature (K)')
            ax1.set_title(f'{profile_type.replace("_", " ").title()} Profiles - Temperature')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            ax2.set_xlabel('Time (hours)')
            ax2.set_ylabel('Current Density (A/cm²)')
            ax2.set_title(f'{profile_type.replace("_", " ").title()} Profiles - Current Density')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            if save_plot:
                plt.savefig(self.dataset_dir / f'transient_profiles_{profile_type}.png', 
                           dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_microstructural_analysis(self, save_plot: bool = True):
        """Analyze and plot microstructural data."""
        if not self.microstructural_data:
            print("❌ No microstructural data available")
            return
        
        # Extract data for analysis
        voxel_sizes = []
        ni_fractions = []
        ysz_fractions = []
        pore_fractions = []
        ni_connectivity = []
        ysz_connectivity = []
        pore_connectivity = []
        specific_surface_areas = []
        
        for sample_data in self.microstructural_data.values():
            voxel_sizes.append(sample_data['voxel_size_nm'])
            ni_fractions.append(sample_data['phase_fractions']['ni'])
            ysz_fractions.append(sample_data['phase_fractions']['ysz'])
            pore_fractions.append(sample_data['phase_fractions']['pore'])
            ni_connectivity.append(sample_data['connectivity']['ni'])
            ysz_connectivity.append(sample_data['connectivity']['ysz'])
            pore_connectivity.append(sample_data['connectivity']['pore'])
            specific_surface_areas.append(sample_data['specific_surface_area'])
        
        # Create comprehensive microstructural analysis plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Phase fractions
        axes[0,0].hist(ni_fractions, bins=20, alpha=0.7, color='silver', label='Ni', edgecolor='black')
        axes[0,0].hist(ysz_fractions, bins=20, alpha=0.7, color='gold', label='YSZ', edgecolor='black')
        axes[0,0].hist(pore_fractions, bins=20, alpha=0.7, color='lightblue', label='Pore', edgecolor='black')
        axes[0,0].set_title('Phase Fraction Distributions')
        axes[0,0].set_xlabel('Volume Fraction')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Connectivity
        axes[0,1].scatter(ni_fractions, ni_connectivity, alpha=0.6, color='silver', label='Ni')
        axes[0,1].scatter(ysz_fractions, ysz_connectivity, alpha=0.6, color='gold', label='YSZ')
        axes[0,1].scatter(pore_fractions, pore_connectivity, alpha=0.6, color='lightblue', label='Pore')
        axes[0,1].set_title('Phase Fraction vs Connectivity')
        axes[0,1].set_xlabel('Volume Fraction')
        axes[0,1].set_ylabel('Connectivity')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Specific surface area vs porosity
        axes[0,2].scatter(pore_fractions, specific_surface_areas, alpha=0.6, color='red')
        axes[0,2].set_title('Porosity vs Specific Surface Area')
        axes[0,2].set_xlabel('Pore Fraction')
        axes[0,2].set_ylabel('Specific Surface Area (m²/m³)')
        axes[0,2].grid(True, alpha=0.3)
        
        # Voxel size distribution
        axes[1,0].hist(voxel_sizes, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1,0].set_title('Voxel Size Distribution')
        axes[1,0].set_xlabel('Voxel Size (nm)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].grid(True, alpha=0.3)
        
        # Ternary-like plot for phase fractions
        axes[1,1].scatter(ni_fractions, ysz_fractions, c=pore_fractions, cmap='viridis', alpha=0.6)
        axes[1,1].set_title('Phase Fraction Relationships')
        axes[1,1].set_xlabel('Ni Fraction')
        axes[1,1].set_ylabel('YSZ Fraction')
        cbar = plt.colorbar(axes[1,1].collections[0], ax=axes[1,1])
        cbar.set_label('Pore Fraction')
        axes[1,1].grid(True, alpha=0.3)
        
        # Connectivity comparison
        connectivity_data = [ni_connectivity, ysz_connectivity, pore_connectivity]
        axes[1,2].boxplot(connectivity_data, labels=['Ni', 'YSZ', 'Pore'])
        axes[1,2].set_title('Connectivity Distributions by Phase')
        axes[1,2].set_ylabel('Connectivity')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Microstructural Analysis (High-Fidelity Data)', fontsize=16, y=1.02)
        
        if save_plot:
            plt.savefig(self.dataset_dir / 'microstructural_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report."""
        print("\n🔍 Generating comprehensive analysis report...")
        
        # Generate all visualizations
        self.generate_dataset_overview()
        
        for fidelity in ['LF', 'MF', 'HF']:
            if fidelity in self.datasets:
                print(f"\n📈 Analyzing {fidelity} fidelity dataset...")
                self.plot_parameter_distributions(fidelity, save_plots=True)
                self.plot_correlation_matrix(fidelity, save_plot=True)
        
        self.plot_transient_profiles(save_plot=True)
        self.plot_microstructural_analysis(save_plot=True)
        
        print(f"\n✅ Analysis complete! All plots saved to {self.dataset_dir}")


if __name__ == "__main__":
    # Run comprehensive analysis
    analyzer = SOFCDatasetAnalyzer()
    analyzer.generate_comprehensive_report()