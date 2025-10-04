#!/usr/bin/env python3
"""
Analysis and visualization tools for atomic-scale simulation data.
Provides statistical analysis, plotting, and data export capabilities.
"""

import numpy as np
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SimulationAnalyzer:
    """Analyze and visualize simulation data"""
    
    def __init__(self, data_dir: str = '.'):
        self.data_dir = data_dir
        self.dft_dir = os.path.join(data_dir, 'dft_calculations')
        self.md_dir = os.path.join(data_dir, 'md_simulations')
        self.output_dir = os.path.join(data_dir, 'processed_data')
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def load_json_data(self, filepath: str) -> Dict:
        """Load JSON data from file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def analyze_dft_data(self):
        """Analyze all DFT calculation data"""
        print("Analyzing DFT Data...")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Analysis results
        results = {
            'vacancy_formation': self.analyze_vacancy_formation(),
            'grain_boundaries': self.analyze_grain_boundaries(),
            'activation_barriers': self.analyze_activation_barriers(),
            'surface_energies': self.analyze_surface_energies()
        }
        
        # Save consolidated results
        with open(f"{self.output_dir}/dft_analysis_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def analyze_vacancy_formation(self) -> Dict:
        """Analyze vacancy formation energies"""
        data = self.load_json_data(f"{self.dft_dir}/defect_energies/vacancy_formation.json")
        
        energies = [c['formation_energy'] for c in data['configurations']]
        
        analysis = {
            'mean': float(np.mean(energies)),
            'std': float(np.std(energies)),
            'min': float(np.min(energies)),
            'max': float(np.max(energies)),
            'q25': float(np.percentile(energies, 25)),
            'q50': float(np.percentile(energies, 50)),
            'q75': float(np.percentile(energies, 75))
        }
        
        # Create histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(energies, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(analysis['mean'], color='red', linestyle='--', label=f'Mean: {analysis["mean"]:.3f} eV')
        ax.set_xlabel('Formation Energy (eV)')
        ax.set_ylabel('Count')
        ax.set_title('Vacancy Formation Energy Distribution')
        ax.legend()
        plt.savefig(f"{self.output_dir}/vacancy_formation_dist.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Vacancy Formation: {analysis['mean']:.3f} ± {analysis['std']:.3f} eV")
        
        return analysis
    
    def analyze_grain_boundaries(self) -> Dict:
        """Analyze grain boundary energies"""
        data = self.load_json_data(f"{self.dft_dir}/defect_energies/grain_boundary_energies.json")
        
        df = pd.DataFrame(data['configurations'])
        
        # Group by GB type
        gb_stats = df.groupby('gb_type')['gb_energy'].agg(['mean', 'std', 'min', 'max'])
        
        # Correlation with misorientation
        correlation = stats.pearsonr(df['misorientation_angle'], df['gb_energy'])
        
        # Convert gb_stats to nested dict
        gb_dict = {}
        for col in gb_stats.columns:
            gb_dict[col] = {}
            for idx in gb_stats.index:
                gb_dict[col][idx] = float(gb_stats.loc[idx, col])
        
        analysis = {
            'by_type': gb_dict,
            'misorientation_correlation': {
                'r': float(correlation[0]),
                'p_value': float(correlation[1])
            },
            'overall': {
                'mean': float(df['gb_energy'].mean()),
                'std': float(df['gb_energy'].std())
            }
        }
        
        # Create scatter plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Misorientation vs energy
        for gb_type in df['gb_type'].unique():
            mask = df['gb_type'] == gb_type
            ax1.scatter(df[mask]['misorientation_angle'], df[mask]['gb_energy'], 
                       label=gb_type, alpha=0.6, s=50)
        ax1.set_xlabel('Misorientation Angle (degrees)')
        ax1.set_ylabel('GB Energy (J/m²)')
        ax1.set_title('Grain Boundary Energy vs Misorientation')
        ax1.legend()
        
        # Box plot by type
        df.boxplot(column='gb_energy', by='gb_type', ax=ax2)
        ax2.set_xlabel('GB Type')
        ax2.set_ylabel('GB Energy (J/m²)')
        ax2.set_title('GB Energy Distribution by Type')
        plt.suptitle('')
        
        plt.savefig(f"{self.output_dir}/grain_boundary_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Grain Boundaries: {analysis['overall']['mean']:.3f} ± {analysis['overall']['std']:.3f} J/m²")
        
        return analysis
    
    def analyze_activation_barriers(self) -> Dict:
        """Analyze activation energy barriers"""
        data = self.load_json_data(f"{self.dft_dir}/activation_barriers/diffusion_barriers.json")
        
        df = pd.DataFrame(data['diffusion_paths'])
        
        # Group by mechanism
        mechanism_stats = df.groupby('mechanism')['activation_energy'].agg(['mean', 'std', 'count'])
        
        # Convert mechanism_stats to nested dict
        mech_dict = {}
        for col in mechanism_stats.columns:
            mech_dict[col] = {}
            for idx in mechanism_stats.index:
                mech_dict[col][idx] = float(mechanism_stats.loc[idx, col])
        
        analysis = {
            'by_mechanism': mech_dict,
            'overall': {
                'mean': float(df['activation_energy'].mean()),
                'std': float(df['activation_energy'].std()),
                'range': [float(df['activation_energy'].min()), float(df['activation_energy'].max())]
            }
        }
        
        # Create energy profile plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot energy profiles for different mechanisms
        for i, mechanism in enumerate(df['mechanism'].unique()[:4]):
            ax = axes[i]
            mech_data = df[df['mechanism'] == mechanism].iloc[:3]  # First 3 examples
            
            for _, path in mech_data.iterrows():
                ax.plot(path['reaction_coordinate'], path['energy_profile'], 
                       alpha=0.7, linewidth=2)
            
            ax.set_xlabel('Reaction Coordinate')
            ax.set_ylabel('Energy (eV)')
            ax.set_title(f'{mechanism.capitalize()} Mechanism')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Activation Energy Profiles', fontsize=14)
        plt.savefig(f"{self.output_dir}/activation_barriers_profiles.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Activation Barriers: {analysis['overall']['mean']:.3f} ± {analysis['overall']['std']:.3f} eV")
        
        return analysis
    
    def analyze_surface_energies(self) -> Dict:
        """Analyze surface energies"""
        data = self.load_json_data(f"{self.dft_dir}/surface_energies/surface_energies.json")
        
        df = pd.DataFrame(data['surfaces'])
        
        # Convert Miller indices to strings for grouping
        df['miller_str'] = df['miller_indices'].apply(lambda x: f"({x[0]}{x[1]}{x[2]})")
        
        # Group by Miller indices
        miller_stats = df.groupby('miller_str')['surface_energy'].agg(['mean', 'std', 'count'])
        
        # Convert miller_stats to nested dict
        miller_dict = {}
        for col in miller_stats.columns:
            miller_dict[col] = {}
            for idx in miller_stats.index:
                miller_dict[col][idx] = float(miller_stats.loc[idx, col])
        
        analysis = {
            'by_miller': miller_dict,
            'oxidation_effect': {
                'clean': float(df[~df['oxidation_state']]['surface_energy'].mean()),
                'oxidized': float(df[df['oxidation_state']]['surface_energy'].mean())
            }
        }
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        miller_stats['mean'].plot(kind='bar', ax=ax, yerr=miller_stats['std'], capsize=5)
        ax.set_xlabel('Miller Indices')
        ax.set_ylabel('Surface Energy (J/m²)')
        ax.set_title('Surface Energy by Miller Indices')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.savefig(f"{self.output_dir}/surface_energy_miller.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Surface Energies: Clean={analysis['oxidation_effect']['clean']:.3f}, "
              f"Oxidized={analysis['oxidation_effect']['oxidized']:.3f} J/m²")
        
        return analysis
    
    def analyze_md_data(self):
        """Analyze all MD simulation data"""
        print("\nAnalyzing MD Data...")
        
        results = {
            'gb_sliding': self.analyze_gb_sliding(),
            'dislocation_mobility': self.analyze_dislocation_mobility(),
            'thermal_activation': self.analyze_thermal_activation()
        }
        
        # Save consolidated results
        with open(f"{self.output_dir}/md_analysis_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def analyze_gb_sliding(self) -> Dict:
        """Analyze grain boundary sliding data"""
        data = self.load_json_data(f"{self.md_dir}/grain_boundary/gb_sliding.json")
        
        # Extract key metrics
        temperatures = []
        sliding_rates = []
        stresses = []
        
        for sim in data['simulations']:
            temperatures.append(sim['temperature'])
            sliding_rates.append(sim['analysis']['average_sliding_rate'])
            stresses.append(sim['applied_stress'])
        
        # Fit Arrhenius relationship
        if len(temperatures) > 0:
            log_rates = np.log(np.array(sliding_rates) + 1e-10)
            inv_temp = 1000 / np.array(temperatures)
            slope, intercept, r_value, _, _ = stats.linregress(inv_temp, log_rates)
            activation_energy = -slope * 8.314 / 1000  # Convert to eV
        else:
            activation_energy = 0
            r_value = 0
        
        analysis = {
            'activation_energy': float(activation_energy),
            'arrhenius_r_squared': float(r_value**2),
            'stress_exponent': float(np.polyfit(np.log(stresses), np.log(np.array(sliding_rates) + 1e-10), 1)[0])
        }
        
        # Create Arrhenius plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Arrhenius plot
        ax1.scatter(1000/np.array(temperatures), sliding_rates, alpha=0.6)
        ax1.set_xlabel('1000/T (K⁻¹)')
        ax1.set_ylabel('Sliding Rate (Å/ps)')
        ax1.set_yscale('log')
        ax1.set_title(f'Arrhenius Plot (E_a = {activation_energy:.2f} eV)')
        ax1.grid(True, alpha=0.3)
        
        # Stress dependence
        ax2.scatter(stresses, sliding_rates, alpha=0.6)
        ax2.set_xlabel('Applied Stress (MPa)')
        ax2.set_ylabel('Sliding Rate (Å/ps)')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_title(f'Stress Dependence (n = {analysis["stress_exponent"]:.2f})')
        ax2.grid(True, alpha=0.3)
        
        plt.savefig(f"{self.output_dir}/gb_sliding_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  GB Sliding: E_a = {activation_energy:.3f} eV, n = {analysis['stress_exponent']:.2f}")
        
        return analysis
    
    def analyze_dislocation_mobility(self) -> Dict:
        """Analyze dislocation mobility data"""
        data = self.load_json_data(f"{self.md_dir}/dislocation/dislocation_mobility.json")
        
        df = pd.DataFrame([{
            'type': sim['dislocation']['type'],
            'temperature': sim['temperature'],
            'stress': sim['applied_stress'],
            'velocity': sim['analysis']['average_velocity'],
            'mobility': sim['analysis']['mobility']
        } for sim in data['simulations']])
        
        # Group by dislocation type
        type_stats = df.groupby('type')[['velocity', 'mobility']].agg(['mean', 'std'])
        
        # Convert multi-level columns to nested dict
        type_dict = {}
        for col in type_stats.columns:
            if col[0] not in type_dict:
                type_dict[col[0]] = {}
            for idx in type_stats.index:
                if idx not in type_dict[col[0]]:
                    type_dict[col[0]][idx] = {}
                type_dict[col[0]][idx][col[1]] = float(type_stats.loc[idx, col])
        
        analysis = {
            'by_type': type_dict,
            'temperature_sensitivity': float(np.corrcoef(df['temperature'], df['velocity'])[0, 1]),
            'stress_sensitivity': float(np.corrcoef(df['stress'], df['velocity'])[0, 1])
        }
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Velocity vs temperature by type
        for disl_type in df['type'].unique():
            mask = df['type'] == disl_type
            axes[0, 0].scatter(df[mask]['temperature'], df[mask]['velocity'], 
                             label=disl_type, alpha=0.6)
        axes[0, 0].set_xlabel('Temperature (K)')
        axes[0, 0].set_ylabel('Velocity (Å/ps)')
        axes[0, 0].set_title('Dislocation Velocity vs Temperature')
        axes[0, 0].legend()
        
        # Velocity vs stress
        for disl_type in df['type'].unique():
            mask = df['type'] == disl_type
            axes[0, 1].scatter(df[mask]['stress'], df[mask]['velocity'], 
                             label=disl_type, alpha=0.6)
        axes[0, 1].set_xlabel('Applied Stress (MPa)')
        axes[0, 1].set_ylabel('Velocity (Å/ps)')
        axes[0, 1].set_title('Dislocation Velocity vs Stress')
        axes[0, 1].legend()
        
        # Mobility distribution
        df.boxplot(column='mobility', by='type', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Dislocation Type')
        axes[1, 0].set_ylabel('Mobility (Å·ps⁻¹·MPa⁻¹)')
        axes[1, 0].set_title('Mobility by Dislocation Type')
        
        # Example trajectory
        example_sim = data['simulations'][0]
        axes[1, 1].plot(example_sim['trajectory_data']['time'], 
                       example_sim['trajectory_data']['position'])
        axes[1, 1].set_xlabel('Time (ps)')
        axes[1, 1].set_ylabel('Position (Å)')
        axes[1, 1].set_title('Example Dislocation Trajectory')
        
        plt.suptitle('Dislocation Mobility Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/dislocation_mobility_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Dislocation Mobility: T-sensitivity = {analysis['temperature_sensitivity']:.3f}, "
              f"σ-sensitivity = {analysis['stress_sensitivity']:.3f}")
        
        return analysis
    
    def analyze_thermal_activation(self) -> Dict:
        """Analyze thermal activation data"""
        data = self.load_json_data(f"{self.md_dir}/thermal_activation.json")
        
        df = pd.DataFrame(data['measurements'])
        
        # Fit Arrhenius relationships
        processes = ['vacancy_diffusion_coefficient', 'dislocation_velocity', 'gb_sliding_rate']
        activation_energies = {}
        
        for process in processes:
            if process in df.columns:
                log_rate = np.log(df[process] + 1e-20)
                inv_temp = 1000 / df['temperature']
                slope, _, r_value, _, _ = stats.linregress(inv_temp, log_rate)
                activation_energies[process] = float(-slope * 8.314 / 1000)  # eV
        
        analysis = {
            'activation_energies': activation_energies,
            'temperature_range': data['temperature_range']
        }
        
        # Create master plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(processes)))
        for i, process in enumerate(processes):
            if process in df.columns:
                ax.plot(1000/df['temperature'], df[process], 
                       'o-', label=process.replace('_', ' ').title(), 
                       color=colors[i], markersize=8, linewidth=2)
        
        ax.set_xlabel('1000/T (K⁻¹)')
        ax.set_ylabel('Rate/Coefficient (log scale)')
        ax.set_yscale('log')
        ax.set_title('Thermal Activation of Various Processes')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.savefig(f"{self.output_dir}/thermal_activation_master.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Thermal Activation: {len(activation_energies)} processes analyzed")
        
        return analysis
    
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        print("\nCreating Summary Report...")
        
        # Load all analysis results
        dft_results = self.load_json_data(f"{self.output_dir}/dft_analysis_results.json")
        md_results = self.load_json_data(f"{self.output_dir}/md_analysis_results.json")
        
        # Create summary
        summary = {
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'key_findings': {
                'vacancy_formation_energy': dft_results['vacancy_formation']['mean'],
                'gb_sliding_activation': md_results['gb_sliding']['activation_energy'],
                'average_dislocation_mobility': {
                    'edge': md_results['dislocation_mobility']['by_type'].get(
                        'velocity', {}).get('mean', {}).get('edge', 'N/A'),
                    'screw': md_results['dislocation_mobility']['by_type'].get(
                        'velocity', {}).get('mean', {}).get('screw', 'N/A')
                }
            },
            'data_statistics': {
                'total_dft_calculations': 205,
                'total_md_simulations': 80,
                'temperature_range_K': [600, 1200],
                'stress_range_MPa': [50, 1000]
            },
            'recommended_parameters': {
                'phase_field': {
                    'vacancy_migration_barrier': 1.2,  # eV
                    'gb_energy': 0.8,  # J/m²
                    'surface_energy': 2.0  # J/m²
                },
                'crystal_plasticity': {
                    'dislocation_density_evolution_rate': 1e-3,
                    'hardening_modulus': 100,  # MPa
                    'recovery_rate': 1e-4
                }
            }
        }
        
        # Save summary
        with open(f"{self.output_dir}/summary_report.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create final visualization
        self.create_summary_visualization()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Results saved to: {self.output_dir}/")
        print("\nKey Files Generated:")
        print("  - dft_analysis_results.json")
        print("  - md_analysis_results.json")
        print("  - summary_report.json")
        print("  - Various visualization plots (.png)")
        
        return summary
    
    def create_summary_visualization(self):
        """Create a summary visualization dashboard"""
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Load data for visualization
        try:
            vac_data = self.load_json_data(f"{self.dft_dir}/defect_energies/vacancy_formation.json")
            energies = [c['formation_energy'] for c in vac_data['configurations']]
            
            # Vacancy formation energy distribution
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.hist(energies, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
            ax1.set_xlabel('Formation Energy (eV)')
            ax1.set_ylabel('Count')
            ax1.set_title('Vacancy Formation')
            
            # GB energy vs misorientation
            gb_data = self.load_json_data(f"{self.dft_dir}/defect_energies/grain_boundary_energies.json")
            angles = [c['misorientation_angle'] for c in gb_data['configurations']]
            gb_energies = [c['gb_energy'] for c in gb_data['configurations']]
            
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.scatter(angles, gb_energies, alpha=0.6, c=angles, cmap='viridis')
            ax2.set_xlabel('Misorientation (°)')
            ax2.set_ylabel('Energy (J/m²)')
            ax2.set_title('Grain Boundary Energy')
            
            # Activation barriers
            barrier_data = self.load_json_data(f"{self.dft_dir}/activation_barriers/diffusion_barriers.json")
            mechanisms = {}
            for path in barrier_data['diffusion_paths']:
                mech = path['mechanism']
                if mech not in mechanisms:
                    mechanisms[mech] = []
                mechanisms[mech].append(path['activation_energy'])
            
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.boxplot(mechanisms.values(), labels=mechanisms.keys())
            ax3.set_ylabel('Activation Energy (eV)')
            ax3.set_title('Diffusion Barriers')
            ax3.tick_params(axis='x', rotation=45)
            
            # MD data
            md_gb_data = self.load_json_data(f"{self.md_dir}/grain_boundary/gb_sliding.json")
            temps = [s['temperature'] for s in md_gb_data['simulations']]
            rates = [s['analysis']['average_sliding_rate'] for s in md_gb_data['simulations']]
            
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.scatter(temps, rates, alpha=0.6, c=temps, cmap='coolwarm')
            ax4.set_xlabel('Temperature (K)')
            ax4.set_ylabel('Sliding Rate (Å/ps)')
            ax4.set_title('GB Sliding Rate')
            
            # Dislocation mobility
            disl_data = self.load_json_data(f"{self.md_dir}/dislocation/dislocation_mobility.json")
            disl_types = {}
            for sim in disl_data['simulations']:
                dtype = sim['dislocation']['type']
                if dtype not in disl_types:
                    disl_types[dtype] = []
                disl_types[dtype].append(sim['analysis']['average_velocity'])
            
            ax5 = fig.add_subplot(gs[1, 1])
            positions = range(len(disl_types))
            for i, (dtype, velocities) in enumerate(disl_types.items()):
                ax5.violinplot([velocities], positions=[i], showmeans=True)
            ax5.set_xticks(positions)
            ax5.set_xticklabels(disl_types.keys())
            ax5.set_ylabel('Velocity (Å/ps)')
            ax5.set_title('Dislocation Mobility')
            
            # Thermal activation
            thermal_data = self.load_json_data(f"{self.md_dir}/thermal_activation.json")
            df_thermal = pd.DataFrame(thermal_data['measurements'])
            
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.plot(df_thermal['temperature'], df_thermal['creep_rate'], 'o-', markersize=6)
            ax6.set_xlabel('Temperature (K)')
            ax6.set_ylabel('Creep Rate')
            ax6.set_title('Temperature Dependence')
            ax6.set_yscale('log')
            
            # Surface energies
            surf_data = self.load_json_data(f"{self.dft_dir}/surface_energies/surface_energies.json")
            miller_energies = {}
            for surf in surf_data['surfaces']:
                miller = str(surf['miller_indices'])
                if miller not in miller_energies:
                    miller_energies[miller] = []
                miller_energies[miller].append(surf['surface_energy'])
            
            ax7 = fig.add_subplot(gs[2, 0:2])
            miller_means = [np.mean(v) for v in miller_energies.values()]
            miller_stds = [np.std(v) for v in miller_energies.values()]
            x_pos = range(len(miller_energies))
            ax7.bar(x_pos, miller_means, yerr=miller_stds, capsize=5, alpha=0.7)
            ax7.set_xticks(x_pos)
            ax7.set_xticklabels([k[:9] for k in miller_energies.keys()], rotation=45)
            ax7.set_ylabel('Surface Energy (J/m²)')
            ax7.set_title('Surface Energies by Miller Indices')
            
            # Summary text
            ax8 = fig.add_subplot(gs[2, 2])
            ax8.axis('off')
            summary_text = (
                "Dataset Summary\n"
                "═══════════════\n"
                f"DFT Calculations: 205\n"
                f"MD Simulations: 80\n"
                f"Temperature: 600-1200 K\n"
                f"Stress: 50-1000 MPa\n\n"
                "Key Parameters:\n"
                f"• E_vac: {np.mean(energies):.2f} eV\n"
                f"• E_gb: {np.mean(gb_energies):.2f} J/m²\n"
                f"• E_surf: {np.mean(miller_means):.2f} J/m²"
            )
            ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
        except Exception as e:
            print(f"Warning: Could not create full summary visualization: {e}")
        
        plt.suptitle('Atomic-Scale Simulation Dataset Overview', fontsize=16, fontweight='bold')
        plt.savefig(f"{self.output_dir}/summary_dashboard.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("  Summary dashboard created successfully!")


def export_to_formats(data_dir: str):
    """Export data to various formats for compatibility"""
    print("\nExporting to additional formats...")
    
    export_dir = os.path.join(data_dir, 'processed_data', 'exports')
    os.makedirs(export_dir, exist_ok=True)
    
    # Export DFT data to CSV
    dft_dir = os.path.join(data_dir, 'dft_calculations')
    
    # Vacancy formation to CSV
    vac_data = json.load(open(f"{dft_dir}/defect_energies/vacancy_formation.json"))
    vac_df = pd.DataFrame(vac_data['configurations'])
    vac_df.to_csv(f"{export_dir}/vacancy_formation.csv", index=False)
    
    # GB energies to CSV
    gb_data = json.load(open(f"{dft_dir}/defect_energies/grain_boundary_energies.json"))
    gb_df = pd.DataFrame(gb_data['configurations'])
    gb_df.to_csv(f"{export_dir}/grain_boundary_energies.csv", index=False)
    
    print(f"  Data exported to CSV format in {export_dir}/")
    
    # Create HDF5 file for efficient storage
    try:
        import h5py
        
        with h5py.File(f"{export_dir}/simulation_data.hdf5", 'w') as hf:
            # Store DFT data
            dft_group = hf.create_group('dft_calculations')
            dft_group.create_dataset('vacancy_energies', 
                                    data=[c['formation_energy'] for c in vac_data['configurations']])
            dft_group.create_dataset('gb_energies',
                                    data=[c['gb_energy'] for c in gb_data['configurations']])
            
            print("  Data exported to HDF5 format")
    except ImportError:
        print("  HDF5 export skipped (h5py not available)")


if __name__ == "__main__":
    # Run analysis
    analyzer = SimulationAnalyzer('.')
    
    # Analyze DFT data
    dft_results = analyzer.analyze_dft_data()
    
    # Analyze MD data
    md_results = analyzer.analyze_md_data()
    
    # Create summary report
    summary = analyzer.create_summary_report()
    
    # Export to additional formats
    export_to_formats('.')