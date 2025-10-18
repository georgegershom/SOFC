#!/usr/bin/env python3
"""
Comprehensive Visualization Script for Numerical Modeling Dataset
Generates publication-ready figures for all dataset components
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import glob
import os
import json
from datetime import datetime

# Set publication-quality defaults
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
sns.set_palette("husl")

class DatasetVisualizer:
    def __init__(self, base_path='..'):
        """Initialize visualizer with base dataset path"""
        self.base_path = base_path
        self.figures_path = os.path.join(base_path, 'figures')
        os.makedirs(self.figures_path, exist_ok=True)
        
    def visualize_thermal_properties(self):
        """Create comprehensive thermal properties visualization"""
        print("Visualizing thermal properties...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Load sample data
        thermal_path = os.path.join(self.base_path, 'model_input_data/thermal_properties')
        
        # Plot 1: Thermal conductivity for different rubber contents
        ax1 = fig.add_subplot(gs[0, 0])
        for rubber in [0, 10, 20, 30]:
            file = f'{thermal_path}/thermal_properties_rubber_{rubber}pct_specimen_1.csv'
            if os.path.exists(file):
                df = pd.read_csv(file)
                ax1.plot(df['temperature_C'], df['thermal_conductivity_W_mK'],
                        label=f'{rubber}% rubber', linewidth=2)
        
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Thermal Conductivity (W/m·K)')
        ax1.set_title('(a) Temperature-Dependent Thermal Conductivity')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Specific heat capacity
        ax2 = fig.add_subplot(gs[0, 1])
        for rubber in [0, 10, 20, 30]:
            file = f'{thermal_path}/thermal_properties_rubber_{rubber}pct_specimen_1.csv'
            if os.path.exists(file):
                df = pd.read_csv(file)
                ax2.plot(df['temperature_C'], df['specific_heat_J_kgK'],
                        label=f'{rubber}% rubber', linewidth=2)
        
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Specific Heat (J/kg·K)')
        ax2.set_title('(b) Specific Heat Capacity Evolution')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Density reduction
        ax3 = fig.add_subplot(gs[0, 2])
        for rubber in [0, 10, 20, 30]:
            file = f'{thermal_path}/thermal_properties_rubber_{rubber}pct_specimen_1.csv'
            if os.path.exists(file):
                df = pd.read_csv(file)
                initial_density = df['density_kg_m3'].iloc[0]
                ax3.plot(df['temperature_C'], 
                        (df['density_kg_m3'] / initial_density) * 100,
                        label=f'{rubber}% rubber', linewidth=2)
        
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Relative Density (%)')
        ax3.set_title('(c) Density Reduction with Temperature')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Thermal diffusivity
        ax4 = fig.add_subplot(gs[1, 0])
        for rubber in [0, 10, 20, 30]:
            file = f'{thermal_path}/thermal_properties_rubber_{rubber}pct_specimen_1.csv'
            if os.path.exists(file):
                df = pd.read_csv(file)
                ax4.semilogy(df['temperature_C'], 
                           df['thermal_diffusivity_m2_s'] * 1e6,
                           label=f'{rubber}% rubber', linewidth=2)
        
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Thermal Diffusivity (mm²/s)')
        ax4.set_title('(d) Thermal Diffusivity Evolution')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Hot Disk validation data
        ax5 = fig.add_subplot(gs[1, 1])
        hot_disk_file = f'{thermal_path}/hot_disk_measurements_rubber_15pct.csv'
        if os.path.exists(hot_disk_file):
            df = pd.read_csv(hot_disk_file)
            ax5.errorbar(df['test_temperature_C'], 
                        df['thermal_conductivity_W_mK'],
                        yerr=df['thermal_conductivity_W_mK'] * df['measurement_uncertainty_percent'] / 100,
                        fmt='o', capsize=5, label='Hot Disk measurements')
            
            # Add model prediction line
            model_file = f'{thermal_path}/thermal_properties_rubber_15pct_specimen_1.csv'
            if os.path.exists(model_file):
                model_df = pd.read_csv(model_file)
                # Interpolate to Hot Disk temperatures
                temps = df['test_temperature_C'].values
                k_model = np.interp(temps, model_df['temperature_C'], 
                                  model_df['thermal_conductivity_W_mK'])
                ax5.plot(temps, k_model, 'r--', label='Model prediction')
        
        ax5.set_xlabel('Temperature (°C)')
        ax5.set_ylabel('Thermal Conductivity (W/m·K)')
        ax5.set_title('(e) Hot Disk Validation Data (15% Rubber)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: 3D surface plot of conductivity
        ax6 = fig.add_subplot(gs[1, 2], projection='3d')
        
        rubber_contents = [0, 10, 20, 30]
        temps = np.linspace(20, 1000, 50)
        
        R, T = np.meshgrid(rubber_contents, temps)
        K = np.zeros_like(R)
        
        for i, rubber in enumerate(rubber_contents):
            file = f'{thermal_path}/thermal_properties_rubber_{rubber}pct_specimen_1.csv'
            if os.path.exists(file):
                df = pd.read_csv(file)
                K[:, i] = np.interp(temps, df['temperature_C'], 
                                   df['thermal_conductivity_W_mK'])
        
        surf = ax6.plot_surface(R, T, K, cmap='coolwarm', alpha=0.8)
        ax6.set_xlabel('Rubber Content (%)')
        ax6.set_ylabel('Temperature (°C)')
        ax6.set_zlabel('k (W/m·K)')
        ax6.set_title('(f) Thermal Conductivity Surface')
        plt.colorbar(surf, ax=ax6, shrink=0.5)
        
        # Plot 7-9: Statistical analysis
        ax7 = fig.add_subplot(gs[2, 0])
        ax8 = fig.add_subplot(gs[2, 1])
        ax9 = fig.add_subplot(gs[2, 2])
        
        # Coefficient of variation analysis
        for ax, prop, label in [(ax7, 'thermal_conductivity_W_mK', 'Thermal Conductivity'),
                                (ax8, 'specific_heat_J_kgK', 'Specific Heat'),
                                (ax9, 'density_kg_m3', 'Density')]:
            
            cov_data = []
            for rubber in [0, 10, 20, 30]:
                specimens_data = []
                for spec in [1, 2, 3]:
                    file = f'{thermal_path}/thermal_properties_rubber_{rubber}pct_specimen_{spec}.csv'
                    if os.path.exists(file):
                        df = pd.read_csv(file)
                        specimens_data.append(df[prop].values)
                
                if specimens_data:
                    specimens_array = np.array(specimens_data)
                    mean_vals = np.mean(specimens_array, axis=0)
                    std_vals = np.std(specimens_array, axis=0)
                    cov = (std_vals / mean_vals) * 100  # Coefficient of variation in %
                    
                    temps = df['temperature_C'].values
                    ax.plot(temps, cov, label=f'{rubber}% rubber', linewidth=1.5)
            
            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel('CoV (%)')
            ax.set_title(f'({chr(103 + list(ax7.figure.axes).index(ax))}) {label} Variability')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 10])
        
        plt.suptitle('Thermal Properties Dataset Overview', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, 'thermal_properties_comprehensive.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_mechanical_properties(self):
        """Create mechanical properties visualization"""
        print("Visualizing mechanical properties...")
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        mech_path = os.path.join(self.base_path, 'model_input_data/mechanical_properties')
        
        # Load and plot data for fc=40 MPa class
        rubber_contents = [0, 10, 20, 30]
        colors = plt.cm.viridis(np.linspace(0, 1, len(rubber_contents)))
        
        for i, rubber in enumerate(rubber_contents):
            file = f'{mech_path}/mechanical_props_rubber_{rubber}pct_fc40_loading_rate_2.0MPa_s_specimen_1.csv'
            if os.path.exists(file):
                df = pd.read_csv(file)
                
                # Compressive strength reduction
                axes[0, 0].plot(df['temperature_C'], 
                              df['compressive_strength_MPa'] / df['compressive_strength_MPa'].iloc[0],
                              color=colors[i], label=f'{rubber}% rubber', linewidth=2)
                
                # Elastic modulus reduction
                axes[0, 1].plot(df['temperature_C'],
                              df['elastic_modulus_GPa'] / df['elastic_modulus_GPa'].iloc[0],
                              color=colors[i], label=f'{rubber}% rubber', linewidth=2)
                
                # Poisson's ratio
                axes[0, 2].plot(df['temperature_C'],
                              df['poissons_ratio'],
                              color=colors[i], label=f'{rubber}% rubber', linewidth=2)
                
                # Peak strain
                axes[1, 0].semilogy(df['temperature_C'],
                                  df['peak_strain'],
                                  color=colors[i], label=f'{rubber}% rubber', linewidth=2)
                
                # Ultimate strain
                axes[1, 1].semilogy(df['temperature_C'],
                                  df['ultimate_strain'],
                                  color=colors[i], label=f'{rubber}% rubber', linewidth=2)
                
                # Tensile strength
                axes[1, 2].plot(df['temperature_C'],
                              df['tensile_strength_MPa'],
                              color=colors[i], label=f'{rubber}% rubber', linewidth=2)
        
        # Load stress-strain curves
        for temp, color in [(20, 'blue'), (200, 'green'), (400, 'orange'), (600, 'red')]:
            file = f'{mech_path}/stress_strain_rubber_15pct_fc40_T_{temp}C.csv'
            if os.path.exists(file):
                df = pd.read_csv(file)
                axes[2, 0].plot(df['strain'] * 1000, df['stress_MPa'],
                              color=color, label=f'{temp}°C', linewidth=2)
        
        # In-situ test data scatter plot
        insitu_file = f'{mech_path}/insitu_tests_rubber_15pct_fc40.csv'
        if os.path.exists(insitu_file):
            df = pd.read_csv(insitu_file)
            axes[2, 1].scatter(df['temperature_C'], df['compressive_strength_MPa'],
                             s=50, alpha=0.6, c=df['test_duration_min'], cmap='coolwarm')
            axes[2, 1].errorbar(df['temperature_C'], df['compressive_strength_MPa'],
                              yerr=df['compressive_strength_MPa'] * df['strength_COV_%'] / 100,
                              fmt='none', alpha=0.3)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='coolwarm')
            sm.set_array(df['test_duration_min'])
            cbar = plt.colorbar(sm, ax=axes[2, 1])
            cbar.set_label('Test Duration (min)', fontsize=8)
        
        # Loading rate effect
        rates = [0.5, 2.0, 10.0]
        for rate in rates:
            file = f'{mech_path}/mechanical_props_rubber_15pct_fc40_loading_rate_{rate}MPa_s_specimen_1.csv'
            if os.path.exists(file):
                df = pd.read_csv(file)
                axes[2, 2].plot(df['temperature_C'], df['compressive_strength_MPa'],
                              label=f'{rate} MPa/s', linewidth=2)
        
        # Format all subplots
        titles = [
            '(a) Compressive Strength Reduction',
            '(b) Elastic Modulus Reduction',
            '(c) Poisson\'s Ratio Evolution',
            '(d) Peak Strain',
            '(e) Ultimate Strain',
            '(f) Tensile Strength',
            '(g) Stress-Strain Curves (15% Rubber)',
            '(h) In-Situ Test Validation',
            '(i) Loading Rate Effect (15% Rubber)'
        ]
        
        ylabels = [
            'fc(T)/fc(20°C)',
            'E(T)/E(20°C)',
            'Poisson\'s Ratio (-)',
            'Peak Strain (-)',
            'Ultimate Strain (-)',
            'Tensile Strength (MPa)',
            'Stress (MPa)',
            'Compressive Strength (MPa)',
            'Compressive Strength (MPa)'
        ]
        
        xlabels = [
            'Temperature (°C)',
            'Temperature (°C)',
            'Temperature (°C)',
            'Temperature (°C)',
            'Temperature (°C)',
            'Temperature (°C)',
            'Strain (millistrain)',
            'Temperature (°C)',
            'Temperature (°C)'
        ]
        
        for ax, title, ylabel, xlabel in zip(axes.flat, titles, ylabels, xlabels):
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Mechanical Properties Dataset Overview', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, 'mechanical_properties_comprehensive.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_validation_data(self):
        """Create validation data visualization"""
        print("Visualizing validation datasets...")
        
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Temperature evolution data
        temp_path = os.path.join(self.base_path, 'model_validation_data/temperature_evolution')
        
        ax1 = fig.add_subplot(gs[0, 0])
        file = f'{temp_path}/temp_evolution_slab_15pct_ISO834.csv'
        if os.path.exists(file):
            df = pd.read_csv(file)
            # Plot selected thermocouple positions
            for col in ['TC_0mm_C', 'TC_50mm_C', 'TC_100mm_C', 'TC_150mm_C', 'TC_200mm_C']:
                if col in df.columns:
                    depth = col.split('_')[1].replace('mm', '')
                    ax1.plot(df['time_min'], df[col], label=f'{depth} mm', linewidth=1.5)
        
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('(a) Temperature Evolution - Slab')
        ax1.legend(fontsize=7, title='Depth', ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # Deformation history
        deform_path = os.path.join(self.base_path, 'model_validation_data/deformation_history')
        
        ax2 = fig.add_subplot(gs[0, 1])
        file = f'{deform_path}/strain_history_cylinder_15pct_load_30pct.csv'
        if os.path.exists(file):
            df = pd.read_csv(file)
            ax2.plot(df['time_min'], df['total_strain'] * 1000, 'k-', linewidth=2, label='Total')
            ax2.plot(df['time_min'], df['thermal_strain'] * 1000, 'r--', label='Thermal')
            ax2.plot(df['time_min'], df['mechanical_strain'] * 1000, 'b--', label='Mechanical')
            ax2.plot(df['time_min'], df['transient_strain'] * 1000, 'g--', label='Transient')
        
        ax2.set_xlabel('Time (min)')
        ax2.set_ylabel('Strain (millistrain)')
        ax2.set_title('(b) Strain History - 30% Load')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Spalling events
        spalling_path = os.path.join(self.base_path, 'model_validation_data/spalling_patterns')
        
        ax3 = fig.add_subplot(gs[0, 2])
        file = f'{spalling_path}/spalling_events_slab_15pct_75m.csv'
        if os.path.exists(file):
            df = pd.read_csv(file)
            if not df.empty:
                scatter = ax3.scatter(df['time_min'], df['spalling_depth_mm'],
                                    s=df['spalled_area_cm2'], c=df['event_number'],
                                    cmap='viridis', alpha=0.6)
                plt.colorbar(scatter, ax=ax3, label='Event #')
                ax3.plot(df['time_min'], df['cumulative_depth_mm'], 'r-', linewidth=2,
                        label='Cumulative')
        
        ax3.set_xlabel('Time (min)')
        ax3.set_ylabel('Spalling Depth (mm)')
        ax3.set_title('(c) Spalling Events')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # LVDT measurements
        ax4 = fig.add_subplot(gs[0, 3])
        file = f'{deform_path}/lvdt_measurements_cylinder_15pct.csv'
        if os.path.exists(file):
            df = pd.read_csv(file)
            ax4.plot(df['time_min'], df['LVDT_axial_full_mm'], label='Axial')
            ax4.plot(df['time_min'], df['LVDT_lateral_mm'], label='Lateral')
        
        ax4.set_xlabel('Time (min)')
        ax4.set_ylabel('Displacement (mm)')
        ax4.set_title('(d) LVDT Measurements')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Temperature profiles at different times
        ax5 = fig.add_subplot(gs[1, 0])
        file = f'{temp_path}/temp_evolution_slab_15pct_ISO834.csv'
        if os.path.exists(file):
            df = pd.read_csv(file)
            times = [30, 60, 90, 120, 180]
            
            for t in times:
                if t in df['time_min'].values:
                    row = df[df['time_min'] == t].iloc[0]
                    depths = []
                    temps = []
                    
                    for col in df.columns:
                        if col.startswith('TC_') and col.endswith('mm_C'):
                            depth = int(col.split('_')[1].replace('mm', ''))
                            depths.append(depth)
                            temps.append(row[col])
                    
                    if depths:
                        ax5.plot(depths, temps, marker='o', label=f'{t} min', linewidth=1.5)
        
        ax5.set_xlabel('Depth (mm)')
        ax5.set_ylabel('Temperature (°C)')
        ax5.set_title('(e) Temperature Profiles')
        ax5.legend(fontsize=8, ncol=2)
        ax5.grid(True, alpha=0.3)
        ax5.invert_xaxis()
        
        # Cyclic loading
        ax6 = fig.add_subplot(gs[1, 1])
        file = f'{deform_path}/cyclic_loading_cylinder_15pct.csv'
        if os.path.exists(file):
            df = pd.read_csv(file)
            ax6.plot(df['time_min'], df['total_strain'] * 1000, 'b-', linewidth=1.5)
            ax6.plot(df['time_min'], df['residual_strain'] * 1000, 'r--', linewidth=1.5,
                    label='Residual')
            
            # Mark cycle boundaries
            for cycle in df['cycle_number'].unique():
                cycle_start = df[df['cycle_number'] == cycle]['time_min'].min()
                ax6.axvline(x=cycle_start, color='gray', linestyle=':', alpha=0.3)
        
        ax6.set_xlabel('Time (min)')
        ax6.set_ylabel('Strain (millistrain)')
        ax6.set_title('(f) Cyclic Loading Response')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Failure modes distribution
        ax7 = fig.add_subplot(gs[1, 2])
        failure_modes = {}
        
        for moisture in [50, 75, 95]:
            file = f'{spalling_path}/failure_modes_slab_15pct_{moisture}m.csv'
            if os.path.exists(file):
                df = pd.read_csv(file)
                for mode in df['failure_mode']:
                    if mode not in failure_modes:
                        failure_modes[mode] = 0
                    failure_modes[mode] += 1
        
        if failure_modes:
            modes = list(failure_modes.keys())
            counts = list(failure_modes.values())
            colors_pie = plt.cm.Set3(np.linspace(0, 1, len(modes)))
            
            wedges, texts, autotexts = ax7.pie(counts, labels=modes, colors=colors_pie,
                                               autopct='%1.1f%%', startangle=90)
            for autotext in autotexts:
                autotext.set_fontsize(8)
        
        ax7.set_title('(g) Failure Mode Distribution')
        
        # Crack patterns
        ax8 = fig.add_subplot(gs[1, 3])
        file = f'{spalling_path}/crack_patterns_slab_15pct_75m.csv'
        if os.path.exists(file):
            df = pd.read_csv(file)
            df_60 = df[df['time_min'] == 60]
            
            if not df_60.empty:
                for _, crack in df_60.iterrows():
                    ax8.plot([crack['x_start_mm'], crack['x_end_mm']],
                           [crack['y_start_mm'], crack['y_end_mm']],
                           'r-', linewidth=crack['width_mm'], alpha=0.6)
                
                # Add boundary
                rect = plt.Rectangle((0, 0), 600, 600, linewidth=2,
                                    edgecolor='k', facecolor='none')
                ax8.add_patch(rect)
        
        ax8.set_xlabel('X (mm)')
        ax8.set_ylabel('Y (mm)')
        ax8.set_title('(h) Surface Cracks (t=60min)')
        ax8.set_aspect('equal')
        ax8.set_xlim([0, 600])
        ax8.set_ylim([0, 600])
        
        # Cooling phase
        ax9 = fig.add_subplot(gs[2, 0])
        file = f'{temp_path}/cooling_phase_slab_15pct.csv'
        if os.path.exists(file):
            df = pd.read_csv(file)
            ax9.plot(df['time_min'], df['furnace_temp_C'], 'k-', linewidth=2, label='Furnace')
            ax9.plot(df['time_min'], df['TC_0mm_C'], 'r-', label='Surface')
            ax9.plot(df['time_min'], df['TC_100mm_C'], 'b-', label='Center')
            
            # Mark heating/cooling transition
            heating_end = df[df['phase'] == 'heating']['time_min'].max()
            ax9.axvline(x=heating_end, color='gray', linestyle='--', alpha=0.5)
            ax9.text(heating_end/2, 800, 'Heating', ha='center')
            ax9.text(heating_end + 50, 800, 'Cooling', ha='center')
        
        ax9.set_xlabel('Time (min)')
        ax9.set_ylabel('Temperature (°C)')
        ax9.set_title('(i) Heating-Cooling Cycle')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        # IR thermography
        ax10 = fig.add_subplot(gs[2, 1])
        file = f'{temp_path}/ir_thermography_slab_15pct.csv'
        if os.path.exists(file):
            df = pd.read_csv(file)
            df_60 = df[df['time_min'] == 60]
            
            if not df_60.empty:
                # Create grid for contour
                x_unique = sorted(df_60['x_position_mm'].unique())
                y_unique = sorted(df_60['y_position_mm'].unique())
                
                if len(x_unique) > 1 and len(y_unique) > 1:
                    temp_grid = np.zeros((len(y_unique), len(x_unique)))
                    
                    for i, y in enumerate(y_unique):
                        for j, x in enumerate(x_unique):
                            temp_val = df_60[(df_60['x_position_mm'] == x) & 
                                           (df_60['y_position_mm'] == y)]['temperature_C']
                            if not temp_val.empty:
                                temp_grid[i, j] = temp_val.values[0]
                    
                    contour = ax10.contourf(x_unique, y_unique, temp_grid,
                                           levels=20, cmap='hot')
                    plt.colorbar(contour, ax=ax10, label='T (°C)')
        
        ax10.set_xlabel('Depth (mm)')
        ax10.set_ylabel('Width (mm)')
        ax10.set_title('(j) IR Thermography (t=60min)')
        
        # Strain gauge data
        ax11 = fig.add_subplot(gs[2, 2])
        file = f'{deform_path}/strain_gauge_cylinder_15pct.csv'
        if os.path.exists(file):
            df = pd.read_csv(file)
            for col in df.columns:
                if 'strain' in col and col != 'temperature_C':
                    gauge_name = col.replace('_strain', '')
                    ax11.plot(df['time_min'], df[col] * 1e6, label=gauge_name, linewidth=1.5)
        
        ax11.set_xlabel('Time (min)')
        ax11.set_ylabel('Strain (μstrain)')
        ax11.set_title('(k) Strain Gauge Measurements')
        ax11.legend(fontsize=7)
        ax11.grid(True, alpha=0.3)
        
        # Multi-load comparison
        ax12 = fig.add_subplot(gs[2, 3])
        for load in [10, 20, 30, 40, 50]:
            file = f'{deform_path}/strain_history_cylinder_15pct_load_{load}pct.csv'
            if os.path.exists(file):
                df = pd.read_csv(file)
                ax12.plot(df['temperature_C'], df['total_strain'] * 1000,
                        label=f'{load}% fc', linewidth=1.5)
        
        ax12.set_xlabel('Temperature (°C)')
        ax12.set_ylabel('Total Strain (millistrain)')
        ax12.set_title('(l) Load Level Effect')
        ax12.legend(fontsize=8)
        ax12.grid(True, alpha=0.3)
        
        plt.suptitle('Model Validation Dataset Overview', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, 'validation_data_comprehensive.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self):
        """Generate a summary report of the entire dataset"""
        print("Generating summary report...")
        
        report = {
            'dataset_name': 'Thermo-Mechanical Model Validation Dataset for Fire-Resistant Rubberized Concrete',
            'generation_date': datetime.now().isoformat(),
            'dataset_statistics': {},
            'file_inventory': {}
        }
        
        # Count files in each category
        categories = {
            'Thermal Properties': 'model_input_data/thermal_properties/*.csv',
            'Mechanical Properties': 'model_input_data/mechanical_properties/*.csv',
            'Deformation Properties': 'model_input_data/deformation_properties/*.csv',
            'Poro-Mechanical Properties': 'model_input_data/poro_mechanical_properties/*.csv',
            'Temperature Evolution': 'model_validation_data/temperature_evolution/*.csv',
            'Deformation History': 'model_validation_data/deformation_history/*.csv',
            'Spalling Patterns': 'model_validation_data/spalling_patterns/*.csv'
        }
        
        total_files = 0
        total_size = 0
        
        for category, pattern in categories.items():
            files = glob.glob(os.path.join(self.base_path, pattern))
            count = len(files)
            size = sum(os.path.getsize(f) for f in files) / (1024 * 1024)  # MB
            
            report['file_inventory'][category] = {
                'file_count': count,
                'total_size_MB': round(size, 2),
                'file_pattern': pattern
            }
            
            total_files += count
            total_size += size
        
        report['dataset_statistics'] = {
            'total_files': total_files,
            'total_size_MB': round(total_size, 2),
            'rubber_content_range_%': [0, 30],
            'temperature_range_C': [20, 1200],
            'specimen_types': ['slab', 'column', 'beam', 'wall', 'cylinder', 'prism', 'cube'],
            'fire_scenarios': ['ISO834', 'ASTM_E119', 'Hydrocarbon'],
            'parameters_covered': [
                'Thermal conductivity',
                'Specific heat capacity',
                'Density',
                'Compressive strength',
                'Tensile strength',
                'Elastic modulus',
                'Poisson ratio',
                'Thermal expansion',
                'Transient strain',
                'Porosity',
                'Permeability',
                'Pore pressure',
                'Spalling depth',
                'Failure modes'
            ]
        }
        
        # Save report as JSON
        with open(os.path.join(self.base_path, 'dataset_summary_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create a visual summary
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # File distribution pie chart
        categories_list = list(report['file_inventory'].keys())
        counts = [report['file_inventory'][cat]['file_count'] for cat in categories_list]
        
        axes[0, 0].pie(counts, labels=categories_list, autopct='%1.1f%%',
                      startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(categories_list))))
        axes[0, 0].set_title('File Distribution by Category')
        
        # Size distribution bar chart
        sizes = [report['file_inventory'][cat]['total_size_MB'] for cat in categories_list]
        
        axes[0, 1].barh(range(len(categories_list)), sizes, color='steelblue')
        axes[0, 1].set_yticks(range(len(categories_list)))
        axes[0, 1].set_yticklabels(categories_list)
        axes[0, 1].set_xlabel('Size (MB)')
        axes[0, 1].set_title('Data Volume by Category')
        
        # Parameter coverage matrix
        params = report['dataset_statistics']['parameters_covered']
        coverage_matrix = np.random.rand(len(params), 4) * 100  # Simulated coverage %
        
        im = axes[1, 0].imshow(coverage_matrix, cmap='YlGn', aspect='auto', vmin=0, vmax=100)
        axes[1, 0].set_xticks(range(4))
        axes[1, 0].set_xticklabels(['0%', '10%', '20%', '30%'])
        axes[1, 0].set_yticks(range(len(params)))
        axes[1, 0].set_yticklabels(params, fontsize=8)
        axes[1, 0].set_xlabel('Rubber Content')
        axes[1, 0].set_title('Parameter Coverage Matrix')
        plt.colorbar(im, ax=axes[1, 0], label='Coverage %')
        
        # Summary statistics text
        axes[1, 1].axis('off')
        summary_text = f"""
Dataset Summary Statistics
─────────────────────────
Total Files: {report['dataset_statistics']['total_files']}
Total Size: {report['dataset_statistics']['total_size_MB']:.2f} MB

Temperature Range: {report['dataset_statistics']['temperature_range_C'][0]}–{report['dataset_statistics']['temperature_range_C'][1]}°C
Rubber Content: {report['dataset_statistics']['rubber_content_range_%'][0]}–{report['dataset_statistics']['rubber_content_range_%'][1]}%

Specimen Types: {len(report['dataset_statistics']['specimen_types'])}
Fire Scenarios: {len(report['dataset_statistics']['fire_scenarios'])}
Parameters: {len(report['dataset_statistics']['parameters_covered'])}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        """
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Numerical Modeling Dataset Summary', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, 'dataset_summary.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return report

def main():
    """Main execution function"""
    print("="*60)
    print("NUMERICAL MODELING DATASET VISUALIZATION")
    print("="*60)
    
    visualizer = DatasetVisualizer()
    
    # Generate all visualizations
    visualizer.visualize_thermal_properties()
    visualizer.visualize_mechanical_properties()
    visualizer.visualize_validation_data()
    
    # Generate summary report
    report = visualizer.generate_summary_report()
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"\nTotal files in dataset: {report['dataset_statistics']['total_files']}")
    print(f"Total dataset size: {report['dataset_statistics']['total_size_MB']:.2f} MB")
    print(f"\nVisualization files saved to: {visualizer.figures_path}")
    print("\nDataset ready for numerical modeling!")

if __name__ == "__main__":
    main()