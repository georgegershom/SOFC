#!/usr/bin/env python3
"""
Dataset Visualization and Analysis Tool
Creates comprehensive plots and analysis of the generated datasets

Author: AI Assistant
Date: 2025-10-18
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

class DatasetVisualizer:
    def __init__(self, data_dir="rubberized_concrete_dataset"):
        self.data_dir = data_dir
        self.output_dir = os.path.join(data_dir, "visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Load all datasets
        self.load_datasets()

    def load_datasets(self):
        """Load all generated datasets"""
        print("Loading datasets...")
        
        # Load JSON files
        json_files = [
            "thermal_properties.json",
            "mechanical_properties.json", 
            "deformation_properties.json",
            "poromechanical_properties.json",
            "temperature_evolution_validation.json",
            "deformation_strain_validation.json",
            "spalling_failure_validation.json"
        ]
        
        self.datasets = {}
        for file in json_files:
            filepath = os.path.join(self.data_dir, file)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    key = file.replace('.json', '')
                    self.datasets[key] = json.load(f)
        
        # Load CSV files
        csv_files = [
            "thermal_properties.csv",
            "mechanical_properties.csv",
            "deformation_properties.csv", 
            "poromechanical_properties.csv",
            "temperature_evolution_validation.csv",
            "deformation_strain_validation.csv",
            "spalling_failure_summary.csv"
        ]
        
        self.dataframes = {}
        for file in csv_files:
            filepath = os.path.join(self.data_dir, file)
            if os.path.exists(filepath):
                key = file.replace('.csv', '')
                self.dataframes[key] = pd.read_csv(filepath)

    def plot_thermal_properties(self):
        """Plot thermal properties vs temperature"""
        print("Creating thermal properties plots...")
        
        if 'thermal_properties' not in self.datasets:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temperature-Dependent Thermal Properties of Rubberized Concrete', fontsize=16, fontweight='bold')
        
        rubber_contents = [0, 5, 10, 15, 20]
        colors = plt.cm.viridis(np.linspace(0, 1, len(rubber_contents)))
        
        data = self.datasets['thermal_properties']
        
        for i, rubber_content in enumerate(rubber_contents):
            key = f"rubber_{rubber_content}pct"
            if key in data:
                temp = np.array(data[key]['temperature'])
                k = np.array(data[key]['thermal_conductivity'])
                cp = np.array(data[key]['specific_heat'])
                rho = np.array(data[key]['density'])
                
                # Thermal conductivity
                axes[0,0].plot(temp, k, label=f'{rubber_content}% rubber', color=colors[i], linewidth=2)
                
                # Specific heat
                axes[0,1].plot(temp, cp, label=f'{rubber_content}% rubber', color=colors[i], linewidth=2)
                
                # Density
                axes[1,0].plot(temp, rho, label=f'{rubber_content}% rubber', color=colors[i], linewidth=2)
                
                # Thermal diffusivity
                alpha = k / (rho * cp) * 1e6  # Convert to mm²/s
                axes[1,1].plot(temp, alpha, label=f'{rubber_content}% rubber', color=colors[i], linewidth=2)
        
        # Formatting
        axes[0,0].set_xlabel('Temperature (°C)')
        axes[0,0].set_ylabel('Thermal Conductivity (W/m·K)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].set_xlabel('Temperature (°C)')
        axes[0,1].set_ylabel('Specific Heat (J/kg·K)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        axes[1,0].set_xlabel('Temperature (°C)')
        axes[1,0].set_ylabel('Density (kg/m³)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].set_xlabel('Temperature (°C)')
        axes[1,1].set_ylabel('Thermal Diffusivity (mm²/s)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'thermal_properties.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_mechanical_properties(self):
        """Plot mechanical properties vs temperature"""
        print("Creating mechanical properties plots...")
        
        if 'mechanical_properties' not in self.datasets:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temperature-Dependent Mechanical Properties of Rubberized Concrete', fontsize=16, fontweight='bold')
        
        rubber_contents = [0, 5, 10, 15, 20]
        colors = plt.cm.plasma(np.linspace(0, 1, len(rubber_contents)))
        
        data = self.datasets['mechanical_properties']
        
        for i, rubber_content in enumerate(rubber_contents):
            key = f"rubber_{rubber_content}pct"
            if key in data:
                temp = np.array(data[key]['temperature'])
                fc = np.array(data[key]['compressive_strength'])
                ft = np.array(data[key]['tensile_strength'])
                E = np.array(data[key]['elastic_modulus'])
                nu = np.array(data[key]['poisson_ratio'])
                
                # Compressive strength
                axes[0,0].plot(temp, fc, label=f'{rubber_content}% rubber', color=colors[i], linewidth=2)
                
                # Tensile strength
                axes[0,1].plot(temp, ft, label=f'{rubber_content}% rubber', color=colors[i], linewidth=2)
                
                # Elastic modulus
                axes[1,0].plot(temp, E, label=f'{rubber_content}% rubber', color=colors[i], linewidth=2)
                
                # Poisson's ratio
                axes[1,1].plot(temp, nu, label=f'{rubber_content}% rubber', color=colors[i], linewidth=2)
        
        # Formatting
        axes[0,0].set_xlabel('Temperature (°C)')
        axes[0,0].set_ylabel('Compressive Strength (MPa)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].set_xlabel('Temperature (°C)')
        axes[0,1].set_ylabel('Tensile Strength (MPa)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        axes[1,0].set_xlabel('Temperature (°C)')
        axes[1,0].set_ylabel('Elastic Modulus (GPa)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].set_xlabel('Temperature (°C)')
        axes[1,1].set_ylabel("Poisson's Ratio")
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mechanical_properties.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_temperature_evolution(self):
        """Plot temperature evolution validation data"""
        print("Creating temperature evolution plots...")
        
        if 'temperature_evolution_validation' not in self.datasets:
            return
        
        # Plot for small cube specimen with ISO834 fire curve
        data = self.datasets['temperature_evolution_validation']
        
        if 'small_cube' in data:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Temperature Evolution in Small Cube Specimens (ISO834 Fire)', fontsize=16, fontweight='bold')
            
            rubber_contents = [0, 10, 20]
            
            for i, rubber_content in enumerate(rubber_contents):
                key = f"rubber_{rubber_content}pct"
                if key in data['small_cube'] and 'ISO834' in data['small_cube'][key]:
                    test_data = data['small_cube'][key]['ISO834']
                    time_hours = np.array(test_data['time_hours'])
                    tc_data = test_data['thermocouple_data']
                    
                    # Plot furnace temperature
                    axes[0,i].plot(time_hours, tc_data['furnace_temperature'], 'r-', linewidth=3, label='Furnace')
                    
                    # Plot thermocouple temperatures
                    for tc_name in ['surface', '25mm', 'center']:
                        if tc_name in tc_data:
                            axes[0,i].plot(time_hours, tc_data[tc_name]['temperature'], 
                                         label=f'{tc_name} ({tc_data[tc_name]["depth_mm"]}mm)', linewidth=2)
                    
                    axes[0,i].set_title(f'{rubber_content}% Rubber Content')
                    axes[0,i].set_xlabel('Time (hours)')
                    axes[0,i].set_ylabel('Temperature (°C)')
                    axes[0,i].legend()
                    axes[0,i].grid(True, alpha=0.3)
                    
                    # Temperature difference plot
                    if 'surface' in tc_data and 'center' in tc_data:
                        temp_diff = np.array(tc_data['surface']['temperature']) - np.array(tc_data['center']['temperature'])
                        axes[1,i].plot(time_hours, temp_diff, 'b-', linewidth=2)
                        axes[1,i].set_title(f'Surface-Center Temperature Difference')
                        axes[1,i].set_xlabel('Time (hours)')
                        axes[1,i].set_ylabel('Temperature Difference (°C)')
                        axes[1,i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'temperature_evolution.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def plot_strain_evolution(self):
        """Plot strain evolution validation data"""
        print("Creating strain evolution plots...")
        
        if 'deformation_strain_validation' not in self.datasets:
            return
        
        data = self.datasets['deformation_strain_validation']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Strain Evolution Under Thermal-Mechanical Loading', fontsize=16, fontweight='bold')
        
        # Plot for different rubber contents under medium load
        rubber_contents = [0, 10, 20]
        colors = ['red', 'blue', 'green']
        
        for i, rubber_content in enumerate(rubber_contents):
            key = f"rubber_{rubber_content}pct"
            if key in data and 'medium_load' in data[key]:
                test_data = data[key]['medium_load']
                time_hours = np.array(test_data['time_hours'])
                temperature = np.array(test_data['temperature_C'])
                total_strain = np.array(test_data['total_strain'])
                thermal_strain = np.array(test_data['thermal_strain'])
                mechanical_strain = np.array(test_data['mechanical_strain'])
                creep_strain = np.array(test_data['creep_strain'])
                
                # Total strain vs time
                axes[0,0].plot(time_hours, total_strain * 1000, label=f'{rubber_content}% rubber', 
                             color=colors[i], linewidth=2)
                
                # Strain components for 20% rubber
                if rubber_content == 20:
                    axes[0,1].plot(time_hours, thermal_strain * 1000, label='Thermal', linewidth=2)
                    axes[0,1].plot(time_hours, mechanical_strain * 1000, label='Mechanical', linewidth=2)
                    axes[0,1].plot(time_hours, creep_strain * 1000, label='Creep', linewidth=2)
                    axes[0,1].plot(time_hours, total_strain * 1000, label='Total', linewidth=2, linestyle='--')
                
                # Strain vs temperature
                axes[1,0].plot(temperature, total_strain * 1000, label=f'{rubber_content}% rubber',
                             color=colors[i], linewidth=2)
                
                # Displacement
                displacement = np.array(test_data['displacement_mm'])
                axes[1,1].plot(time_hours, displacement, label=f'{rubber_content}% rubber',
                             color=colors[i], linewidth=2)
        
        # Formatting
        axes[0,0].set_xlabel('Time (hours)')
        axes[0,0].set_ylabel('Total Strain (×10⁻³)')
        axes[0,0].set_title('Total Strain Evolution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].set_xlabel('Time (hours)')
        axes[0,1].set_ylabel('Strain (×10⁻³)')
        axes[0,1].set_title('Strain Components (20% Rubber)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        axes[1,0].set_xlabel('Temperature (°C)')
        axes[1,0].set_ylabel('Total Strain (×10⁻³)')
        axes[1,0].set_title('Strain vs Temperature')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].set_xlabel('Time (hours)')
        axes[1,1].set_ylabel('Displacement (mm)')
        axes[1,1].set_title('Displacement (100mm gauge)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'strain_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_spalling_analysis(self):
        """Plot spalling and failure analysis"""
        print("Creating spalling analysis plots...")
        
        if 'spalling_failure_validation' not in self.datasets:
            return
        
        data = self.datasets['spalling_failure_validation']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Spalling Behavior and Failure Analysis', fontsize=16, fontweight='bold')
        
        # Spalling occurrence matrix
        rubber_contents = [0, 5, 10, 15, 20]
        conditions = ['standard_fire', 'rapid_heating', 'high_load', 'high_moisture']
        
        spalling_matrix = np.zeros((len(conditions), len(rubber_contents)))
        max_depth_matrix = np.zeros((len(conditions), len(rubber_contents)))
        
        for i, condition in enumerate(conditions):
            for j, rubber_content in enumerate(rubber_contents):
                key = f"rubber_{rubber_content}pct"
                if key in data and condition in data[key]:
                    test_data = data[key][condition]
                    spalling_matrix[i, j] = 1 if test_data['spalling_occurred'] else 0
                    max_depth_matrix[i, j] = test_data['max_spalling_depth_mm']
        
        # Spalling occurrence heatmap
        im1 = axes[0,0].imshow(spalling_matrix, cmap='RdYlBu_r', aspect='auto')
        axes[0,0].set_xticks(range(len(rubber_contents)))
        axes[0,0].set_xticklabels([f'{rc}%' for rc in rubber_contents])
        axes[0,0].set_yticks(range(len(conditions)))
        axes[0,0].set_yticklabels([c.replace('_', ' ').title() for c in conditions])
        axes[0,0].set_title('Spalling Occurrence')
        axes[0,0].set_xlabel('Rubber Content')
        
        # Add text annotations
        for i in range(len(conditions)):
            for j in range(len(rubber_contents)):
                text = 'Yes' if spalling_matrix[i, j] else 'No'
                axes[0,0].text(j, i, text, ha="center", va="center", color="white", fontweight='bold')
        
        # Maximum spalling depth heatmap
        im2 = axes[0,1].imshow(max_depth_matrix, cmap='Reds', aspect='auto')
        axes[0,1].set_xticks(range(len(rubber_contents)))
        axes[0,1].set_xticklabels([f'{rc}%' for rc in rubber_contents])
        axes[0,1].set_yticks(range(len(conditions)))
        axes[0,1].set_yticklabels([c.replace('_', ' ').title() for c in conditions])
        axes[0,1].set_title('Maximum Spalling Depth (mm)')
        axes[0,1].set_xlabel('Rubber Content')
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=axes[0,1])
        cbar2.set_label('Depth (mm)')
        
        # Spalling depth evolution for high moisture condition
        colors = plt.cm.viridis(np.linspace(0, 1, len(rubber_contents)))
        for i, rubber_content in enumerate(rubber_contents):
            key = f"rubber_{rubber_content}pct"
            if key in data and 'high_moisture' in data[key]:
                test_data = data[key]['high_moisture']
                time_min = np.array(test_data['time_minutes'])
                depth = np.array(test_data['spalling_depth_mm'])
                axes[1,0].plot(time_min, depth, label=f'{rubber_content}% rubber', 
                             color=colors[i], linewidth=2)
        
        axes[1,0].set_xlabel('Time (minutes)')
        axes[1,0].set_ylabel('Spalling Depth (mm)')
        axes[1,0].set_title('Spalling Depth Evolution (High Moisture)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Failure time analysis
        failure_times = []
        rubber_labels = []
        
        for rubber_content in rubber_contents:
            key = f"rubber_{rubber_content}pct"
            if key in data:
                times = []
                for condition in conditions:
                    if condition in data[key]:
                        failure_time = data[key][condition]['failure_time_min']
                        if failure_time is not None:
                            times.append(failure_time)
                
                if times:
                    failure_times.append(times)
                    rubber_labels.append(f'{rubber_content}%')
        
        if failure_times:
            axes[1,1].boxplot(failure_times, labels=rubber_labels)
            axes[1,1].set_xlabel('Rubber Content')
            axes[1,1].set_ylabel('Failure Time (minutes)')
            axes[1,1].set_title('Failure Time Distribution')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'spalling_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_summary_dashboard(self):
        """Create a comprehensive summary dashboard"""
        print("Creating summary dashboard...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Rubberized Concrete Fire-Resistance Dataset - Summary Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Material properties summary
        if 'thermal_properties' in self.dataframes:
            ax1 = fig.add_subplot(gs[0, :2])
            df = self.dataframes['thermal_properties']
            for rubber in [0, 10, 20]:
                data = df[df['rubber_content_pct'] == rubber]
                ax1.plot(data['temperature_C'], data['thermal_conductivity_W_m_K'], 
                        label=f'{rubber}% rubber', linewidth=2)
            ax1.set_xlabel('Temperature (°C)')
            ax1.set_ylabel('Thermal Conductivity (W/m·K)')
            ax1.set_title('Thermal Conductivity vs Temperature')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        if 'mechanical_properties' in self.dataframes:
            ax2 = fig.add_subplot(gs[0, 2:])
            df = self.dataframes['mechanical_properties']
            for rubber in [0, 10, 20]:
                data = df[df['rubber_content_pct'] == rubber]
                ax2.plot(data['temperature_C'], data['compressive_strength_MPa'], 
                        label=f'{rubber}% rubber', linewidth=2)
            ax2.set_xlabel('Temperature (°C)')
            ax2.set_ylabel('Compressive Strength (MPa)')
            ax2.set_title('Compressive Strength vs Temperature')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Validation data summary
        if 'temperature_evolution_validation' in self.dataframes:
            ax3 = fig.add_subplot(gs[1, :2])
            df = self.dataframes['temperature_evolution_validation']
            # Plot center temperature for different rubber contents
            for rubber in [0, 10, 20]:
                data = df[(df['rubber_content_pct'] == rubber) & 
                         (df['thermocouple_location'] == 'center') &
                         (df['fire_curve'] == 'ISO834') &
                         (df['specimen_type'] == 'small_cube')]
                if not data.empty:
                    ax3.plot(data['time_hours'], data['temperature_C'], 
                            label=f'{rubber}% rubber', linewidth=2)
            ax3.set_xlabel('Time (hours)')
            ax3.set_ylabel('Temperature (°C)')
            ax3.set_title('Center Temperature Evolution (ISO834)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        if 'deformation_strain_validation' in self.dataframes:
            ax4 = fig.add_subplot(gs[1, 2:])
            df = self.dataframes['deformation_strain_validation']
            for rubber in [0, 10, 20]:
                data = df[(df['rubber_content_pct'] == rubber) & 
                         (df['loading_scenario'] == 'medium_load')]
                if not data.empty:
                    ax4.plot(data['time_hours'], data['total_strain'] * 1000, 
                            label=f'{rubber}% rubber', linewidth=2)
            ax4.set_xlabel('Time (hours)')
            ax4.set_ylabel('Total Strain (×10⁻³)')
            ax4.set_title('Strain Evolution (Medium Load)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Spalling analysis
        if 'spalling_failure_summary' in self.dataframes:
            ax5 = fig.add_subplot(gs[2, :2])
            df = self.dataframes['spalling_failure_summary']
            
            # Spalling occurrence by rubber content
            spall_data = df.groupby('rubber_content_pct')['spalling_occurred'].mean()
            ax5.bar(spall_data.index, spall_data.values, alpha=0.7, color='coral')
            ax5.set_xlabel('Rubber Content (%)')
            ax5.set_ylabel('Spalling Probability')
            ax5.set_title('Spalling Occurrence by Rubber Content')
            ax5.grid(True, alpha=0.3)
            
            ax6 = fig.add_subplot(gs[2, 2:])
            # Maximum spalling depth by rubber content
            depth_data = df.groupby('rubber_content_pct')['max_spalling_depth_mm'].mean()
            ax6.bar(depth_data.index, depth_data.values, alpha=0.7, color='lightblue')
            ax6.set_xlabel('Rubber Content (%)')
            ax6.set_ylabel('Average Max Spalling Depth (mm)')
            ax6.set_title('Average Maximum Spalling Depth')
            ax6.grid(True, alpha=0.3)
        
        # Dataset statistics
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        # Create statistics table
        stats_text = "DATASET STATISTICS\n\n"
        
        if hasattr(self, 'dataframes'):
            for name, df in self.dataframes.items():
                if not df.empty:
                    stats_text += f"{name.replace('_', ' ').title()}:\n"
                    stats_text += f"  • {len(df):,} data points\n"
                    if 'rubber_content_pct' in df.columns:
                        rubber_range = f"{df['rubber_content_pct'].min()}-{df['rubber_content_pct'].max()}%"
                        stats_text += f"  • Rubber content range: {rubber_range}\n"
                    if 'temperature_C' in df.columns:
                        temp_range = f"{df['temperature_C'].min():.0f}-{df['temperature_C'].max():.0f}°C"
                        stats_text += f"  • Temperature range: {temp_range}\n"
                    stats_text += "\n"
        
        ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.savefig(os.path.join(self.output_dir, 'summary_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("Generating all visualization plots...")
        
        self.plot_thermal_properties()
        self.plot_mechanical_properties()
        self.plot_temperature_evolution()
        self.plot_strain_evolution()
        self.plot_spalling_analysis()
        self.create_summary_dashboard()
        
        print(f"All plots saved to: {self.output_dir}")

if __name__ == "__main__":
    visualizer = DatasetVisualizer()
    visualizer.generate_all_plots()
    print("Visualization complete!")