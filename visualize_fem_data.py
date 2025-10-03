#!/usr/bin/env python3
"""
FEM Dataset Visualization Tool
Creates plots and visualizations of the generated FEM simulation data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FEMDataVisualizer:
    def __init__(self, data_dir="/workspace/fem_dataset"):
        """Initialize the visualizer with data directory"""
        self.data_dir = data_dir
        self.load_data()
        
    def load_data(self):
        """Load all the generated data"""
        print("📊 Loading FEM dataset...")
        
        # Load CSV files
        self.nodes = pd.read_csv(f"{self.data_dir}/nodes.csv")
        self.elements = pd.read_csv(f"{self.data_dir}/elements.csv")
        
        # Load JSON files
        json_files = [
            'boundary_conditions', 'damage_evolution', 'failure_predictions',
            'field_distributions', 'material_models', 'mesh_quality',
            'strain_fields', 'stress_distributions', 'thermal_profiles'
        ]
        
        for file in json_files:
            with open(f"{self.data_dir}/{file}.json", 'r') as f:
                setattr(self, file, json.load(f))
        
        print("✅ Data loaded successfully!")
    
    def plot_mesh_overview(self):
        """Plot mesh overview and quality metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Mesh Overview and Quality Metrics', fontsize=16, fontweight='bold')
        
        # 3D scatter plot of nodes
        ax1 = axes[0, 0]
        scatter = ax1.scatter(self.nodes['x'], self.nodes['y'], 
                             c=self.nodes['z'], cmap='viridis', alpha=0.6, s=1)
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        ax1.set_title('Node Distribution (colored by Z)')
        plt.colorbar(scatter, ax=ax1, label='Z coordinate')
        
        # Element size distribution
        ax2 = axes[0, 1]
        ax2.hist(self.elements['element_size'], bins=50, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Element Size')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Element Size Distribution')
        ax2.axvline(self.mesh_quality['avg_element_size'], color='red', 
                   linestyle='--', label=f"Mean: {self.mesh_quality['avg_element_size']:.3f}")
        ax2.legend()
        
        # Element types
        ax3 = axes[1, 0]
        element_counts = self.elements['element_type'].value_counts()
        ax3.pie(element_counts.values, labels=element_counts.index, autopct='%1.1f%%')
        ax3.set_title('Element Type Distribution')
        
        # Interface refinement levels
        ax4 = axes[1, 1]
        refinement_counts = self.elements['interface_refinement'].value_counts().sort_index()
        bars = ax4.bar(refinement_counts.index, refinement_counts.values, 
                      alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Refinement Level')
        ax4.set_ylabel('Number of Elements')
        ax4.set_title('Interface Refinement Distribution')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/mesh_overview.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_boundary_conditions(self):
        """Plot boundary condition time histories"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Boundary Conditions Over Time', fontsize=16, fontweight='bold')
        
        time = np.array(self.boundary_conditions['temperature']['values']['time'])
        
        # Temperature BC
        ax1 = axes[0, 0]
        temp = np.array(self.boundary_conditions['temperature']['values']['temperature'])
        ax1.plot(time/60, temp, 'r-', linewidth=2, label='Temperature')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Temperature Boundary Condition')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Displacement BC
        ax2 = axes[0, 1]
        ux = np.array(self.boundary_conditions['displacement']['values']['ux'])
        uy = np.array(self.boundary_conditions['displacement']['values']['uy'])
        ax2.plot(time/60, ux*1000, 'b-', linewidth=2, label='Ux')
        ax2.plot(time/60, uy*1000, 'g-', linewidth=2, label='Uy')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Displacement (mm)')
        ax2.set_title('Displacement Boundary Conditions')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Voltage BC
        ax3 = axes[1, 0]
        voltage = np.array(self.boundary_conditions['voltage']['values']['voltage'])
        ax3.plot(time/60, voltage, 'm-', linewidth=2, label='Voltage')
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Voltage (V)')
        ax3.set_title('Voltage Boundary Condition')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Heat flux BC
        ax4 = axes[1, 1]
        flux = np.array(self.boundary_conditions['heat_flux']['values']['flux'])
        ax4.plot(time/60, flux, 'orange', linewidth=2, label='Heat Flux')
        ax4.set_xlabel('Time (minutes)')
        ax4.set_ylabel('Heat Flux (W/m²)')
        ax4.set_title('Heat Flux Boundary Condition')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/boundary_conditions.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_thermal_profiles(self):
        """Plot thermal loading profiles"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Thermal Loading Profiles', fontsize=16, fontweight='bold')
        
        # Plot first 4 thermal profiles
        profile_keys = list(self.thermal_profiles.keys())[:4]
        
        for i, key in enumerate(profile_keys):
            ax = axes[i//2, i%2]
            profile = self.thermal_profiles[key]
            time = np.array(profile['time']) / 60  # Convert to minutes
            temp = np.array(profile['temperature'])
            
            ax.plot(time, temp, linewidth=2, label=f"Profile {i+1}")
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Temperature (°C)')
            ax.set_title(f'Thermal Profile {i+1}\n'
                        f'Heating: {profile["heating_rate"]:.1f}°C/min, '
                        f'Cooling: {profile["cooling_rate"]:.1f}°C/min')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/thermal_profiles.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_stress_evolution(self):
        """Plot stress evolution over time"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Stress Evolution Over Time', fontsize=16, fontweight='bold')
        
        time = np.array(self.stress_distributions['von_mises']['time']) / 60
        
        # Von Mises stress statistics
        ax1 = axes[0, 0]
        von_mises = np.array(self.stress_distributions['von_mises']['values'])
        
        # Calculate statistics over elements at each time step
        mean_stress = np.mean(von_mises, axis=0)
        max_stress = np.max(von_mises, axis=0)
        min_stress = np.min(von_mises, axis=0)
        std_stress = np.std(von_mises, axis=0)
        
        ax1.plot(time, mean_stress, 'b-', linewidth=2, label='Mean')
        ax1.fill_between(time, mean_stress - std_stress, mean_stress + std_stress, 
                        alpha=0.3, label='±1 Std Dev')
        ax1.plot(time, max_stress, 'r--', linewidth=1, label='Maximum')
        ax1.plot(time, min_stress, 'g--', linewidth=1, label='Minimum')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Von Mises Stress (MPa)')
        ax1.set_title('Von Mises Stress Statistics')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Principal stress evolution
        ax2 = axes[0, 1]
        sigma1 = np.array(self.stress_distributions['principal']['sigma_1'])
        sigma2 = np.array(self.stress_distributions['principal']['sigma_2'])
        sigma3 = np.array(self.stress_distributions['principal']['sigma_3'])
        
        ax2.plot(time, np.mean(sigma1, axis=0), 'r-', linewidth=2, label='σ₁')
        ax2.plot(time, np.mean(sigma2, axis=0), 'g-', linewidth=2, label='σ₂')
        ax2.plot(time, np.mean(sigma3, axis=0), 'b-', linewidth=2, label='σ₃')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Principal Stress (MPa)')
        ax2.set_title('Principal Stress Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Interfacial shear stress
        ax3 = axes[1, 0]
        shear_stress = np.array(self.stress_distributions['interfacial_shear']['values'])
        mean_shear = np.mean(shear_stress, axis=0)
        max_shear = np.max(shear_stress, axis=0)
        critical_shear = self.stress_distributions['interfacial_shear']['critical_value']
        
        ax3.plot(time, mean_shear, 'purple', linewidth=2, label='Mean Shear')
        ax3.plot(time, max_shear, 'orange', linewidth=2, label='Max Shear')
        ax3.axhline(critical_shear, color='red', linestyle='--', 
                   label=f'Critical ({critical_shear} MPa)')
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Interfacial Shear Stress (MPa)')
        ax3.set_title('Interfacial Shear Stress')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Stress distribution histogram at final time
        ax4 = axes[1, 1]
        final_stress = von_mises[:, -1]
        ax4.hist(final_stress, bins=50, alpha=0.7, edgecolor='black', density=True)
        ax4.axvline(np.mean(final_stress), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(final_stress):.1f} MPa')
        ax4.set_xlabel('Von Mises Stress (MPa)')
        ax4.set_ylabel('Probability Density')
        ax4.set_title('Final Stress Distribution')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/stress_evolution.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_damage_evolution(self):
        """Plot damage evolution and statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Damage Evolution Analysis', fontsize=16, fontweight='bold')
        
        time = np.array(self.damage_evolution['time']) / 60
        damage = np.array(self.damage_evolution['damage_variable'])
        
        # Damage statistics over time
        ax1 = axes[0, 0]
        mean_damage = np.mean(damage, axis=0)
        max_damage = np.max(damage, axis=0)
        damaged_fraction = np.sum(damage > 0, axis=0) / damage.shape[0]
        
        ax1.plot(time, mean_damage, 'r-', linewidth=2, label='Mean Damage')
        ax1.plot(time, max_damage, 'b-', linewidth=2, label='Max Damage')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Damage Variable')
        ax1.set_title('Damage Evolution Statistics')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Fraction of damaged elements
        ax2 = axes[0, 1]
        ax2.plot(time, damaged_fraction * 100, 'g-', linewidth=2)
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Damaged Elements (%)')
        ax2.set_title('Percentage of Damaged Elements')
        ax2.grid(True, alpha=0.3)
        
        # Final damage distribution
        ax3 = axes[1, 0]
        final_damage = damage[:, -1]
        damaged_elements = final_damage[final_damage > 0]
        
        if len(damaged_elements) > 0:
            ax3.hist(damaged_elements, bins=30, alpha=0.7, edgecolor='black')
            ax3.axvline(np.mean(damaged_elements), color='red', linestyle='--',
                       label=f'Mean: {np.mean(damaged_elements):.3f}')
        ax3.set_xlabel('Damage Variable')
        ax3.set_ylabel('Number of Elements')
        ax3.set_title('Final Damage Distribution (Damaged Elements Only)')
        ax3.grid(True, alpha=0.3)
        if len(damaged_elements) > 0:
            ax3.legend()
        
        # Damage initiation timeline
        ax4 = axes[1, 1]
        initiation_times = []
        for elem_idx in range(damage.shape[0]):
            damage_history = damage[elem_idx, :]
            first_damage = np.where(damage_history > 0)[0]
            if len(first_damage) > 0:
                initiation_times.append(time[first_damage[0]])
        
        if len(initiation_times) > 0:
            ax4.hist(initiation_times, bins=20, alpha=0.7, edgecolor='black')
            ax4.axvline(np.mean(initiation_times), color='red', linestyle='--',
                       label=f'Mean: {np.mean(initiation_times):.1f} min')
        ax4.set_xlabel('Initiation Time (minutes)')
        ax4.set_ylabel('Number of Elements')
        ax4.set_title('Damage Initiation Timeline')
        ax4.grid(True, alpha=0.3)
        if len(initiation_times) > 0:
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/damage_evolution.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_field_distributions(self):
        """Plot temperature and voltage field distributions"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Field Distributions', fontsize=16, fontweight='bold')
        
        # Get coordinates and field data
        x = np.array(self.field_distributions['temperature']['coordinates']['x'])
        y = np.array(self.field_distributions['temperature']['coordinates']['y'])
        z = np.array(self.field_distributions['temperature']['coordinates']['z'])
        
        temp_data = np.array(self.field_distributions['temperature']['values'])
        voltage_data = np.array(self.field_distributions['voltage']['values'])
        time = np.array(self.field_distributions['temperature']['time'])
        
        # Temperature field at different times
        time_indices = [0, len(time)//2, -1]
        time_labels = ['Initial', 'Mid-time', 'Final']
        
        for i, (t_idx, label) in enumerate(zip(time_indices, time_labels)):
            # Temperature
            ax_temp = axes[0, i]
            scatter = ax_temp.scatter(x, y, c=temp_data[:, t_idx], 
                                    cmap='coolwarm', s=2, alpha=0.7)
            ax_temp.set_xlabel('X coordinate')
            ax_temp.set_ylabel('Y coordinate')
            ax_temp.set_title(f'Temperature Field - {label}\n'
                            f'Time: {time[t_idx]/60:.1f} min')
            plt.colorbar(scatter, ax=ax_temp, label='Temperature (°C)')
            
            # Voltage
            ax_volt = axes[1, i]
            scatter = ax_volt.scatter(x, y, c=voltage_data[:, t_idx], 
                                    cmap='plasma', s=2, alpha=0.7)
            ax_volt.set_xlabel('X coordinate')
            ax_volt.set_ylabel('Y coordinate')
            ax_volt.set_title(f'Voltage Field - {label}\n'
                            f'Time: {time[t_idx]/60:.1f} min')
            plt.colorbar(scatter, ax=ax_volt, label='Voltage (V)')
        
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/field_distributions.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_failure_analysis(self):
        """Plot failure predictions and analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Failure Analysis', fontsize=16, fontweight='bold')
        
        time = np.array(self.failure_predictions['delamination']['time']) / 60
        
        # Delamination risk evolution
        ax1 = axes[0, 0]
        delam_risk = np.array(self.failure_predictions['delamination']['risk_factor'])
        mean_risk = np.mean(delam_risk, axis=0)
        max_risk = np.max(delam_risk, axis=0)
        
        ax1.plot(time, mean_risk, 'b-', linewidth=2, label='Mean Risk')
        ax1.plot(time, max_risk, 'r-', linewidth=2, label='Max Risk')
        ax1.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Critical (1.0)')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Delamination Risk Factor')
        ax1.set_title('Delamination Risk Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Delamination initiation
        ax2 = axes[0, 1]
        delam_initiated = np.array(self.failure_predictions['delamination']['initiated'])
        initiated_fraction = np.sum(delam_initiated, axis=0) / delam_initiated.shape[0]
        
        ax2.plot(time, initiated_fraction * 100, 'g-', linewidth=2)
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Delaminated Interfaces (%)')
        ax2.set_title('Delamination Progression')
        ax2.grid(True, alpha=0.3)
        
        # Crack initiation analysis
        ax3 = axes[1, 0]
        crack_initiated = np.array(self.failure_predictions['crack_initiation']['initiated'])
        crack_fraction = np.sum(crack_initiated, axis=0) / crack_initiated.shape[0]
        
        ax3.plot(time, crack_fraction * 100, 'purple', linewidth=2)
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Cracked Elements (%)')
        ax3.set_title('Crack Initiation Progression')
        ax3.grid(True, alpha=0.3)
        
        # Stress amplitude distribution for crack-prone elements
        ax4 = axes[1, 1]
        stress_amplitude = np.array(self.failure_predictions['crack_initiation']['stress_amplitude'])
        threshold = self.failure_predictions['crack_initiation']['parameters']['threshold_stress']
        
        ax4.hist(stress_amplitude / 1e6, bins=30, alpha=0.7, edgecolor='black')
        ax4.axvline(threshold / 1e6, color='red', linestyle='--', 
                   label=f'Threshold: {threshold/1e6:.1f} MPa')
        ax4.set_xlabel('Stress Amplitude (MPa)')
        ax4.set_ylabel('Number of Elements')
        ax4.set_title('Stress Amplitude Distribution\n(Crack-Prone Elements)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/failure_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_plots(self):
        """Create all summary plots"""
        print("🎨 Creating visualization plots...")
        
        self.plot_mesh_overview()
        self.plot_boundary_conditions()
        self.plot_thermal_profiles()
        self.plot_stress_evolution()
        self.plot_damage_evolution()
        self.plot_field_distributions()
        self.plot_failure_analysis()
        
        print("✅ All plots created and saved!")
    
    def print_dataset_summary(self):
        """Print a comprehensive summary of the dataset"""
        print("\n" + "="*80)
        print("📊 FEM SIMULATION DATASET SUMMARY")
        print("="*80)
        
        print(f"\n🔧 MESH DATA:")
        print(f"   • Total nodes: {len(self.nodes):,}")
        print(f"   • Total elements: {len(self.elements):,}")
        print(f"   • Element types: {', '.join(self.elements['element_type'].unique())}")
        print(f"   • Element size range: {self.mesh_quality['min_element_size']:.2e} - {self.mesh_quality['max_element_size']:.2e}")
        
        print(f"\n🌡️ BOUNDARY CONDITIONS:")
        print(f"   • Temperature BC nodes: {len(self.boundary_conditions['temperature']['node_ids'])}")
        print(f"   • Displacement BC nodes: {len(self.boundary_conditions['displacement']['node_ids'])}")
        print(f"   • Voltage BC nodes: {len(self.boundary_conditions['voltage']['node_ids'])}")
        print(f"   • Simulation time: {self.boundary_conditions['time_parameters']['total_time']} seconds")
        
        print(f"\n🔬 MATERIALS:")
        for name, props in self.material_models.items():
            print(f"   • {props['name']} (ID: {props['id']})")
        
        print(f"\n🔥 THERMAL PROFILES:")
        print(f"   • Number of profiles: {len(self.thermal_profiles)}")
        
        print(f"\n💪 STRESS ANALYSIS:")
        print(f"   • Von Mises stress range: {self.stress_distributions['von_mises']['min_value']:.1f} - {self.stress_distributions['von_mises']['max_value']:.1f} MPa")
        print(f"   • Interface elements: {len(self.stress_distributions['interfacial_shear']['element_ids'])}")
        
        print(f"\n💥 DAMAGE ANALYSIS:")
        stats = self.damage_evolution['damage_statistics']
        print(f"   • Elements with damage: {stats['damaged_elements_count']}")
        print(f"   • Failed elements: {stats['failed_elements_count']}")
        print(f"   • Maximum damage: {stats['max_damage']:.3f}")
        
        print(f"\n🌡️⚡ FIELD DISTRIBUTIONS:")
        temp_stats = self.field_distributions['temperature']['statistics']
        volt_stats = self.field_distributions['voltage']['statistics']
        print(f"   • Temperature range: {temp_stats['min_temp']:.1f} - {temp_stats['max_temp']:.1f} °C")
        print(f"   • Voltage range: {volt_stats['min_voltage']:.2f} - {volt_stats['max_voltage']:.2f} V")
        
        print(f"\n🔍 FAILURE PREDICTIONS:")
        delam_stats = self.failure_predictions['delamination']['statistics']
        crack_stats = self.failure_predictions['crack_initiation']['statistics']
        print(f"   • Delamination initiated: {delam_stats['initiated_count']} interfaces")
        print(f"   • Crack initiation: {crack_stats['initiated_count']} elements")
        
        print("\n" + "="*80)

def main():
    """Main function to create visualizations"""
    visualizer = FEMDataVisualizer()
    visualizer.print_dataset_summary()
    visualizer.create_summary_plots()
    
    print(f"\n🎉 Visualization complete! Check the plots in /workspace/fem_dataset/")

if __name__ == "__main__":
    main()