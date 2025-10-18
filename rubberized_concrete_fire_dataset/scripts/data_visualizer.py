#!/usr/bin/env python3
"""
Advanced Visualization Suite for Rubberized Concrete Fire Resistance Dataset
Generates publication-quality figures for all experimental data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, interpolate
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality parameters
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['figure.dpi'] = 150
sns.set_palette("husl")

class DataVisualizer:
    def __init__(self):
        self.base_path = "raw_data"
        self.output_path = "visualizations"
        self.colors = {
            0: '#2E86AB', 5: '#A23B72', 10: '#F18F01', 
            15: '#C73E1D', 20: '#6B0504'
        }
        
    def plot_strength_development(self):
        """Plot strength development over time for all mixes"""
        df = pd.read_csv(f"{self.base_path}/ambient_tests/ambient_mechanical_properties.csv")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Group by mix and age, calculate mean and std
        grouped = df.groupby(['Mix_ID', 'Age_days']).agg({
            'Compressive_Strength_MPa': ['mean', 'std'],
            'Splitting_Tensile_MPa': ['mean', 'std'],
            'Modulus_Elasticity_GPa': ['mean', 'std'],
            'UPV_m_s': ['mean', 'std']
        }).reset_index()
        
        properties = [
            ('Compressive_Strength_MPa', 'Compressive Strength (MPa)', axes[0, 0]),
            ('Splitting_Tensile_MPa', 'Splitting Tensile Strength (MPa)', axes[0, 1]),
            ('Modulus_Elasticity_GPa', 'Elastic Modulus (GPa)', axes[1, 0]),
            ('UPV_m_s', 'UPV (m/s)', axes[1, 1])
        ]
        
        for prop, label, ax in properties:
            for mix_id in df['Mix_ID'].unique():
                if 'RC' in mix_id and '_' not in mix_id:  # Only base mixes
                    mix_data = grouped[grouped['Mix_ID'] == mix_id]
                    rubber_content = int(mix_id.replace('RC', ''))
                    
                    ages = mix_data['Age_days']
                    means = mix_data[(prop, 'mean')]
                    stds = mix_data[(prop, 'std')]
                    
                    ax.errorbar(ages, means, yerr=stds, 
                              label=f'{rubber_content}% Rubber',
                              marker='o', capsize=5, capthick=1,
                              color=self.colors.get(rubber_content, 'gray'))
            
            ax.set_xlabel('Age (days)')
            ax.set_ylabel(label)
            ax.legend(loc='best', frameon=True, fancybox=True)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 60)
        
        plt.suptitle('Strength Development of Rubberized Concrete at Ambient Temperature', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/strength_development.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_temperature_effects(self):
        """Plot residual properties vs temperature for different rubber contents"""
        df = pd.read_csv(f"{self.base_path}/high_temp_tests/residual_properties.csv")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Filter for furnace-cooled samples (primary data)
        df_fc = df[df['Cooling_Method'] == 'Furnace_Cooled']
        
        properties = [
            ('Residual_Compressive_MPa', 'Residual Compressive Strength (MPa)', axes[0, 0]),
            ('Residual_Tensile_MPa', 'Residual Tensile Strength (MPa)', axes[0, 1]),
            ('Residual_Modulus_GPa', 'Residual Elastic Modulus (GPa)', axes[0, 2]),
            ('Mass_Loss_%', 'Mass Loss (%)', axes[1, 0]),
            ('Crack_Density_cracks_m', 'Crack Density (cracks/m)', axes[1, 1]),
            ('Max_Crack_Width_mm', 'Max Crack Width (mm)', axes[1, 2])
        ]
        
        for prop, label, ax in properties:
            for mix_id in ['RC0', 'RC5', 'RC10', 'RC15', 'RC20']:
                mix_data = df_fc[df_fc['Mix_ID'] == mix_id]
                rubber_content = int(mix_id.replace('RC', ''))
                
                # Group by temperature and calculate statistics
                temp_grouped = mix_data.groupby('Target_Temperature_C')[prop].agg(['mean', 'std']).reset_index()
                
                ax.errorbar(temp_grouped['Target_Temperature_C'], 
                          temp_grouped['mean'],
                          yerr=temp_grouped['std'],
                          label=f'{rubber_content}% Rubber',
                          marker='o', capsize=3,
                          color=self.colors[rubber_content])
                
                # Add smooth interpolation
                if len(temp_grouped) > 3:
                    temps_smooth = np.linspace(23, 800, 100)
                    interp = interpolate.interp1d(temp_grouped['Target_Temperature_C'], 
                                                 temp_grouped['mean'], 
                                                 kind='cubic', fill_value='extrapolate')
                    ax.plot(temps_smooth, interp(temps_smooth), 
                           color=self.colors[rubber_content], 
                           alpha=0.3, linewidth=2)
            
            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel(label)
            ax.legend(loc='best', frameon=True, fancybox=True, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 850)
            
            # Add critical temperature markers
            ax.axvline(x=400, color='orange', linestyle='--', alpha=0.5, linewidth=0.5)
            ax.axvline(x=600, color='red', linestyle='--', alpha=0.5, linewidth=0.5)
        
        plt.suptitle('Temperature-Dependent Properties of Rubberized Concrete (Furnace Cooled)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/temperature_effects.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_cooling_method_comparison(self):
        """Compare effects of cooling methods on residual properties"""
        df = pd.read_csv(f"{self.base_path}/high_temp_tests/residual_properties.csv")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Focus on RC10 for clear comparison
        df_rc10 = df[df['Mix_ID'] == 'RC10']
        
        properties = [
            ('Residual_Compressive_MPa', 'Residual Compressive Strength (MPa)', axes[0, 0]),
            ('Residual_Tensile_MPa', 'Residual Tensile Strength (MPa)', axes[0, 1]),
            ('Crack_Density_cracks_m', 'Crack Density (cracks/m)', axes[1, 0]),
            ('Max_Crack_Width_mm', 'Maximum Crack Width (mm)', axes[1, 1])
        ]
        
        for prop, label, ax in properties:
            for cooling in ['Furnace_Cooled', 'Water_Quenched']:
                cooling_data = df_rc10[df_rc10['Cooling_Method'] == cooling]
                temp_grouped = cooling_data.groupby('Target_Temperature_C')[prop].agg(['mean', 'std']).reset_index()
                
                style = '-' if cooling == 'Furnace_Cooled' else '--'
                color = 'blue' if cooling == 'Furnace_Cooled' else 'red'
                label_text = 'Furnace Cooled' if cooling == 'Furnace_Cooled' else 'Water Quenched'
                
                ax.errorbar(temp_grouped['Target_Temperature_C'], 
                          temp_grouped['mean'],
                          yerr=temp_grouped['std'],
                          label=label_text,
                          marker='o', linestyle=style,
                          color=color, capsize=5)
            
            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel(label)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Add shaded regions for temperature ranges
            ax.axvspan(0, 200, alpha=0.1, color='green', label='Low Risk')
            ax.axvspan(200, 400, alpha=0.1, color='yellow')
            ax.axvspan(400, 600, alpha=0.1, color='orange')
            ax.axvspan(600, 800, alpha=0.1, color='red', label='High Risk')
        
        plt.suptitle('Effect of Cooling Method on RC10 Properties', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/cooling_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_in_situ_vs_residual(self):
        """Compare in-situ (hot) properties with residual (cold) properties"""
        residual_df = pd.read_csv(f"{self.base_path}/high_temp_tests/residual_properties.csv")
        in_situ_df = pd.read_csv(f"{self.base_path}/in_situ_tests/in_situ_properties.csv")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Focus on RC10 furnace-cooled for residual
        residual_rc10 = residual_df[(residual_df['Mix_ID'] == 'RC10') & 
                                   (residual_df['Cooling_Method'] == 'Furnace_Cooled')]
        in_situ_rc10 = in_situ_df[in_situ_df['Mix_ID'] == 'RC10']
        
        # Compressive strength comparison
        ax = axes[0]
        
        # Residual strength
        res_grouped = residual_rc10.groupby('Target_Temperature_C')['Residual_Compressive_MPa'].agg(['mean', 'std']).reset_index()
        ax.errorbar(res_grouped['Target_Temperature_C'], res_grouped['mean'], 
                   yerr=res_grouped['std'], label='Residual (Post-cooling)',
                   marker='s', color='blue', capsize=5)
        
        # In-situ strength
        situ_grouped = in_situ_rc10.groupby('Test_Temperature_C')['Hot_Compressive_MPa'].agg(['mean', 'std']).reset_index()
        ax.errorbar(situ_grouped['Test_Temperature_C'], situ_grouped['mean'],
                   yerr=situ_grouped['std'], label='In-situ (Hot)',
                   marker='o', color='red', capsize=5)
        
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Compressive Strength (MPa)')
        ax.set_title('Compressive Strength: In-situ vs Residual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Modulus comparison
        ax = axes[1]
        
        # Residual modulus
        res_mod = residual_rc10.groupby('Target_Temperature_C')['Residual_Modulus_GPa'].agg(['mean', 'std']).reset_index()
        ax.errorbar(res_mod['Target_Temperature_C'], res_mod['mean'],
                   yerr=res_mod['std'], label='Residual (Post-cooling)',
                   marker='s', color='blue', capsize=5)
        
        # In-situ modulus
        situ_mod = in_situ_rc10.groupby('Test_Temperature_C')['Hot_Modulus_GPa'].agg(['mean', 'std']).reset_index()
        ax.errorbar(situ_mod['Test_Temperature_C'], situ_mod['mean'],
                   yerr=situ_mod['std'], label='In-situ (Hot)',
                   marker='o', color='red', capsize=5)
        
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Elastic Modulus (GPa)')
        ax.set_title('Elastic Modulus: In-situ vs Residual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('In-situ vs Residual Properties (RC10)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/in_situ_vs_residual.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_thermal_strain_behavior(self):
        """Plot thermal strain and LITS behavior"""
        df = pd.read_csv(f"{self.base_path}/in_situ_tests/in_situ_properties.csv")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Thermal strain vs temperature for different rubber contents
        ax = axes[0, 0]
        for mix_id in ['RC0', 'RC5', 'RC10', 'RC15', 'RC20']:
            mix_data = df[df['Mix_ID'] == mix_id]
            rubber_content = int(mix_id.replace('RC', ''))
            
            temp_grouped = mix_data.groupby('Test_Temperature_C')['Thermal_Strain_x10-6'].mean().reset_index()
            ax.plot(temp_grouped['Test_Temperature_C'], temp_grouped['Thermal_Strain_x10-6'],
                   label=f'{rubber_content}% Rubber', marker='o',
                   color=self.colors[rubber_content])
        
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Thermal Strain (×10⁻⁶)')
        ax.set_title('Free Thermal Strain')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # LITS vs temperature
        ax = axes[0, 1]
        for mix_id in ['RC0', 'RC5', 'RC10', 'RC15', 'RC20']:
            mix_data = df[df['Mix_ID'] == mix_id]
            rubber_content = int(mix_id.replace('RC', ''))
            
            temp_grouped = mix_data.groupby('Test_Temperature_C')['LITS_x10-6'].mean().reset_index()
            ax.plot(temp_grouped['Test_Temperature_C'], temp_grouped['LITS_x10-6'],
                   label=f'{rubber_content}% Rubber', marker='s',
                   color=self.colors[rubber_content])
        
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('LITS (×10⁻⁶)')
        ax.set_title('Load-Induced Thermal Strain (40% load)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # CTE vs temperature
        ax = axes[1, 0]
        for mix_id in ['RC0', 'RC5', 'RC10', 'RC15', 'RC20']:
            mix_data = df[df['Mix_ID'] == mix_id]
            rubber_content = int(mix_id.replace('RC', ''))
            
            temp_grouped = mix_data.groupby('Test_Temperature_C')['CTE_x10-6_per_C'].mean().reset_index()
            ax.plot(temp_grouped['Test_Temperature_C'], temp_grouped['CTE_x10-6_per_C'],
                   label=f'{rubber_content}% Rubber', marker='^',
                   color=self.colors[rubber_content])
        
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('CTE (×10⁻⁶/°C)')
        ax.set_title('Coefficient of Thermal Expansion')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Total strain comparison
        ax = axes[1, 1]
        rc10_data = df[df['Mix_ID'] == 'RC10']
        temp_grouped = rc10_data.groupby('Test_Temperature_C').mean().reset_index()
        
        width = 30
        x = temp_grouped['Test_Temperature_C']
        
        ax.bar(x - width/2, temp_grouped['Thermal_Strain_x10-6'], 
               width, label='Thermal Strain', color='blue', alpha=0.7)
        ax.bar(x + width/2, temp_grouped['LITS_x10-6'], 
               width, label='LITS', color='red', alpha=0.7)
        
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Strain (×10⁻⁶)')
        ax.set_title('Strain Components for RC10')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Thermal Deformation Behavior of Rubberized Concrete', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/thermal_strain.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_spalling_analysis(self):
        """Analyze spalling behavior and pore pressure development"""
        df = pd.read_csv(f"{self.base_path}/spalling_data/spalling_pore_pressure.csv")
        
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 3, figure=fig)
        
        # Spalling probability vs temperature
        ax1 = fig.add_subplot(gs[0, :2])
        spalling_prob = df.groupby(['Mix_ID', 'Temperature_C'])['Spalling_Occurred'].mean().reset_index()
        
        for mix_id in ['RC0', 'RC10', 'RC10_PP']:
            mix_data = spalling_prob[spalling_prob['Mix_ID'] == mix_id]
            label = mix_id.replace('RC', '').replace('_', ' + ')
            label = f"{label}% Rubber" if 'PP' not in label else label + ' Fibers'
            ax1.plot(mix_data['Temperature_C'], mix_data['Spalling_Occurred'] * 100,
                    marker='o', label=label, linewidth=2)
        
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Spalling Probability (%)')
        ax1.set_title('Spalling Risk Assessment')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Pore pressure distribution
        ax2 = fig.add_subplot(gs[0, 2])
        rc10_600 = df[(df['Mix_ID'] == 'RC10') & (df['Temperature_C'] == 600)]
        depths = rc10_600.groupby('Depth_mm')['Peak_Pore_Pressure_MPa'].mean()
        
        ax2.barh(depths.index, depths.values, color='orange', alpha=0.7)
        ax2.set_ylabel('Depth (mm)')
        ax2.set_xlabel('Peak Pore Pressure (MPa)')
        ax2.set_title('Pore Pressure at 600°C (RC10)')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Spalling depth vs temperature
        ax3 = fig.add_subplot(gs[1, 0])
        spalled = df[df['Spalling_Occurred'] == True]
        spalling_depth = spalled.groupby(['Mix_ID', 'Temperature_C'])['Spalling_Depth_mm'].mean().reset_index()
        
        for mix_id in ['RC0', 'RC5', 'RC10']:
            mix_data = spalling_depth[spalling_depth['Mix_ID'] == mix_id]
            if len(mix_data) > 0:
                rubber = int(mix_id.replace('RC', ''))
                ax3.scatter(mix_data['Temperature_C'], mix_data['Spalling_Depth_mm'],
                          label=f'{rubber}% Rubber', s=100, alpha=0.7,
                          color=self.colors.get(rubber, 'gray'))
        
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Spalling Depth (mm)')
        ax3.set_title('Spalling Severity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Time to peak pore pressure
        ax4 = fig.add_subplot(gs[1, 1])
        time_to_peak = df.groupby(['Depth_mm', 'Temperature_C'])['Time_to_Peak_min'].mean().unstack()
        
        im = ax4.imshow(time_to_peak.values, aspect='auto', cmap='coolwarm',
                       extent=[time_to_peak.columns.min(), time_to_peak.columns.max(),
                              time_to_peak.index.max(), time_to_peak.index.min()])
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Depth (mm)')
        ax4.set_title('Time to Peak Pore Pressure (min)')
        plt.colorbar(im, ax=ax4)
        
        # PP fiber effectiveness
        ax5 = fig.add_subplot(gs[1, 2])
        pp_comparison = df[df['Mix_ID'].isin(['RC10', 'RC10_PP'])]
        pp_pressure = pp_comparison.groupby(['Mix_ID', 'Temperature_C'])['Peak_Pore_Pressure_MPa'].mean().reset_index()
        
        rc10_pp = pp_pressure[pp_pressure['Mix_ID'] == 'RC10']
        rc10_pp_fibers = pp_pressure[pp_pressure['Mix_ID'] == 'RC10_PP']
        
        x = rc10_pp['Temperature_C']
        width = 35
        
        ax5.bar(x - width/2, rc10_pp['Peak_Pore_Pressure_MPa'].values,
               width, label='RC10', color='orange', alpha=0.7)
        if len(rc10_pp_fibers) > 0:
            ax5.bar(x + width/2, rc10_pp_fibers['Peak_Pore_Pressure_MPa'].values,
                   width, label='RC10 + PP', color='green', alpha=0.7)
        
        ax5.set_xlabel('Temperature (°C)')
        ax5.set_ylabel('Peak Pore Pressure (MPa)')
        ax5.set_title('Effect of PP Fibers on Pore Pressure')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Spalling type distribution
        ax6 = fig.add_subplot(gs[2, :])
        spalling_types = df[df['Spalling_Occurred'] == True].groupby(['Temperature_C', 'Spalling_Type']).size().unstack(fill_value=0)
        
        if not spalling_types.empty:
            spalling_types.plot(kind='bar', stacked=True, ax=ax6, 
                               color=['red', 'orange', 'yellow', 'gray'])
            ax6.set_xlabel('Temperature (°C)')
            ax6.set_ylabel('Number of Specimens')
            ax6.set_title('Spalling Type Distribution')
            ax6.legend(title='Spalling Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax6.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Spalling Behavior and Pore Pressure Analysis', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/spalling_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_stress_strain_curves(self):
        """Plot stress-strain curves at different temperatures"""
        df = pd.read_csv(f"{self.base_path}/high_temp_tests/stress_strain_curves.csv")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        temperatures = [23, 200, 400, 600, 800]
        
        for idx, temp in enumerate(temperatures):
            ax = axes[idx]
            temp_data = df[df['Temperature_C'] == temp]
            
            for mix_id in ['RC0', 'RC5', 'RC10', 'RC15', 'RC20']:
                mix_data = temp_data[temp_data['Mix_ID'] == mix_id]
                if len(mix_data) > 0:
                    rubber = int(mix_id.replace('RC', ''))
                    
                    # Sort by strain for proper plotting
                    mix_data = mix_data.sort_values('Strain')
                    
                    ax.plot(mix_data['Strain'] * 1000, mix_data['Stress_MPa'],
                           label=f'{rubber}% Rubber', linewidth=2,
                           color=self.colors[rubber])
                    
                    # Mark peak point
                    peak_idx = mix_data['Stress_MPa'].idxmax()
                    if pd.notna(peak_idx):
                        peak_strain = mix_data.loc[peak_idx, 'Strain'] * 1000
                        peak_stress = mix_data.loc[peak_idx, 'Stress_MPa']
                        ax.scatter(peak_strain, peak_stress, s=50, 
                                 color=self.colors[rubber], zorder=5)
            
            ax.set_xlabel('Strain (×10⁻³)')
            ax.set_ylabel('Stress (MPa)')
            ax.set_title(f'Temperature: {temp}°C')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 40)
        
        # Summary plot showing peak stress evolution
        ax = axes[5]
        peak_data = df.groupby(['Mix_ID', 'Temperature_C'])['Peak_Stress_MPa'].first().reset_index()
        
        for mix_id in ['RC0', 'RC10', 'RC20']:
            mix_data = peak_data[peak_data['Mix_ID'] == mix_id]
            rubber = int(mix_id.replace('RC', ''))
            ax.plot(mix_data['Temperature_C'], mix_data['Peak_Stress_MPa'],
                   marker='o', label=f'{rubber}% Rubber', linewidth=2,
                   color=self.colors[rubber])
        
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Peak Stress (MPa)')
        ax.set_title('Peak Stress vs Temperature')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Stress-Strain Behavior at Elevated Temperatures', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/stress_strain_curves.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("Generating comprehensive visualizations...")
        
        print("1. Plotting strength development...")
        self.plot_strength_development()
        
        print("2. Plotting temperature effects...")
        self.plot_temperature_effects()
        
        print("3. Plotting cooling method comparison...")
        self.plot_cooling_method_comparison()
        
        print("4. Plotting in-situ vs residual properties...")
        self.plot_in_situ_vs_residual()
        
        print("5. Plotting thermal strain behavior...")
        self.plot_thermal_strain_behavior()
        
        print("6. Plotting spalling analysis...")
        self.plot_spalling_analysis()
        
        print("7. Plotting stress-strain curves...")
        self.plot_stress_strain_curves()
        
        print("\nAll visualizations generated successfully!")

if __name__ == "__main__":
    visualizer = DataVisualizer()
    visualizer.generate_all_plots()