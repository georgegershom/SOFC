"""
Dataset Visualization and Analysis Script
For Multi-Fidelity SOFC Degradation Dataset

This script creates comprehensive visualizations of the generated dataset
including statistical distributions, spatial fields, and time-series data.
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class DatasetVisualizer:
    """Visualize and analyze the multi-fidelity SOFC dataset"""
    
    def __init__(self, dataset_path="sofc_multifidelity_dataset"):
        self.dataset_path = Path(dataset_path)
        self.viz_path = self.dataset_path / 'visualizations'
        self.viz_path.mkdir(exist_ok=True)
        
    def visualize_phase1_LF(self):
        """Visualize Low-Fidelity dataset"""
        print("\n" + "="*70)
        print("Visualizing Phase 1: Low-Fidelity Dataset")
        print("="*70)
        
        # Load data
        df = pd.read_csv(self.dataset_path / 'phase1_LF' / 'phase1_LF_complete.csv')
        
        # 1. Input distributions
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Phase 1: Input Variable Distributions', fontsize=16, fontweight='bold')
        
        inputs = [
            ('fuel_utilization', 'Fuel Utilization', '-'),
            ('operating_temperature_K', 'Operating Temperature', 'K'),
            ('current_density_A_cm2', 'Current Density', 'A/cm²'),
            ('pressure_atm', 'Pressure', 'atm'),
            ('air_stoichiometry', 'Air Stoichiometry', '-'),
            ('cycles', 'Thermal Cycles', '-'),
            ('thermal_cycling_rate_K_min', 'Cycling Rate', 'K/min'),
            ('cell_thickness_mm', 'Cell Thickness', 'mm'),
            ('active_area_cm2', 'Active Area', 'cm²')
        ]
        
        for idx, (var, label, unit) in enumerate(inputs):
            ax = axes.flatten()[idx]
            ax.hist(df[var], bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel(f'{label} ({unit})' if unit != '-' else label)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_path / 'phase1_input_distributions.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: phase1_input_distributions.png")
        plt.close()
        
        # 2. Output distributions
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Phase 1: Output Variable Distributions', fontsize=16, fontweight='bold')
        
        outputs = [
            ('voltage_V', 'Voltage', 'V'),
            ('power_density_W_cm2', 'Power Density', 'W/cm²'),
            ('avg_temperature_K', 'Average Temperature', 'K'),
            ('avg_von_mises_stress_MPa', 'Von Mises Stress', 'MPa'),
            ('Ni_particle_size_nm', 'Ni Particle Size', 'nm'),
            ('crack_probability', 'Crack Probability', '-'),
            ('delamination_risk', 'Delamination Risk', '-'),
            ('time_to_failure_hours', 'Time to Failure', 'hours'),
            ('voltage_degradation_percent', 'Voltage Degradation', '%')
        ]
        
        for idx, (var, label, unit) in enumerate(outputs):
            ax = axes.flatten()[idx]
            data = df[var]
            if var == 'time_to_failure_hours':
                data = np.log10(data)
                ax.set_xlabel(f'log₁₀({label}) ({unit})' if unit != '-' else f'log₁₀({label})')
            else:
                ax.set_xlabel(f'{label} ({unit})' if unit != '-' else label)
            ax.hist(data, bins=50, alpha=0.7, edgecolor='black', color='coral')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_path / 'phase1_output_distributions.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: phase1_output_distributions.png")
        plt.close()
        
        # 3. Correlation matrix
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Select key variables for correlation
        corr_vars = [
            'operating_temperature_K', 'current_density_A_cm2', 'fuel_utilization',
            'cycles', 'voltage_V', 'power_density_W_cm2', 
            'avg_von_mises_stress_MPa', 'Ni_particle_size_nm',
            'crack_probability', 'delamination_risk'
        ]
        
        corr_matrix = df[corr_vars].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Phase 1: Variable Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.viz_path / 'phase1_correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: phase1_correlation_matrix.png")
        plt.close()
        
        # 4. Key relationships
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Phase 1: Key Physical Relationships', fontsize=16, fontweight='bold')
        
        # Temperature vs Stress
        ax = axes[0, 0]
        scatter = ax.scatter(df['operating_temperature_K'], df['avg_von_mises_stress_MPa'],
                            c=df['current_density_A_cm2'], cmap='viridis', alpha=0.5, s=10)
        ax.set_xlabel('Operating Temperature (K)')
        ax.set_ylabel('Von Mises Stress (MPa)')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Current Density (A/cm²)')
        ax.grid(True, alpha=0.3)
        
        # Current vs Voltage
        ax = axes[0, 1]
        scatter = ax.scatter(df['current_density_A_cm2'], df['voltage_V'],
                            c=df['operating_temperature_K'], cmap='plasma', alpha=0.5, s=10)
        ax.set_xlabel('Current Density (A/cm²)')
        ax.set_ylabel('Voltage (V)')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Temperature (K)')
        ax.grid(True, alpha=0.3)
        
        # Cycles vs Degradation
        ax = axes[1, 0]
        scatter = ax.scatter(df['cycles'], df['Ni_particle_size_nm'],
                            c=df['operating_temperature_K'], cmap='hot', alpha=0.5, s=10)
        ax.set_xlabel('Thermal Cycles')
        ax.set_ylabel('Ni Particle Size (nm)')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Temperature (K)')
        ax.grid(True, alpha=0.3)
        
        # Stress vs Time to Failure
        ax = axes[1, 1]
        scatter = ax.scatter(df['avg_von_mises_stress_MPa'], 
                            np.log10(df['time_to_failure_hours']),
                            c=df['thermal_cycling_rate_K_min'], cmap='coolwarm', alpha=0.5, s=10)
        ax.set_xlabel('Von Mises Stress (MPa)')
        ax.set_ylabel('log₁₀(Time to Failure) (hours)')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cycling Rate (K/min)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_path / 'phase1_physical_relationships.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: phase1_physical_relationships.png")
        plt.close()
        
    def visualize_phase2_MF(self):
        """Visualize Mid-Fidelity dataset"""
        print("\n" + "="*70)
        print("Visualizing Phase 2: Mid-Fidelity Dataset")
        print("="*70)
        
        # Load global data
        df = pd.read_csv(self.dataset_path / 'phase2_MF' / 'phase2_MF_global.csv')
        
        # Load spatial data
        with h5py.File(self.dataset_path / 'phase2_MF' / 'phase2_MF_complete.h5', 'r') as f:
            # Check available spatial fields
            spatial_fields = list(f['spatial_fields'].keys())
            print(f"Available spatial fields: {spatial_fields}")
            
            if 'temperature' in f['spatial_fields']:
                # Visualize first few samples
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                fig.suptitle('Phase 2: Temperature Field Examples', fontsize=16, fontweight='bold')
                
                for i in range(min(6, len(f['spatial_fields/temperature'].keys()))):
                    ax = axes.flatten()[i]
                    T_field = f[f'spatial_fields/temperature/sample_{i}'][:]
                    
                    im = ax.imshow(T_field, cmap='hot', aspect='auto', origin='lower')
                    ax.set_title(f'Sample {i}')
                    ax.set_xlabel('x (grid points)')
                    ax.set_ylabel('y (grid points)')
                    plt.colorbar(im, ax=ax, label='Temperature (K)')
                
                plt.tight_layout()
                plt.savefig(self.viz_path / 'phase2_temperature_fields.png', dpi=300, bbox_inches='tight')
                print("✓ Saved: phase2_temperature_fields.png")
                plt.close()
            
            if 'stress' in f['spatial_fields']:
                # Visualize stress fields
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                fig.suptitle('Phase 2: Stress Field Examples', fontsize=16, fontweight='bold')
                
                for i in range(min(6, len(f['spatial_fields/stress'].keys()))):
                    ax = axes.flatten()[i]
                    sigma_field = f[f'spatial_fields/stress/sample_{i}'][:]
                    
                    im = ax.imshow(sigma_field, cmap='RdYlBu_r', aspect='auto', origin='lower')
                    ax.set_title(f'Sample {i}')
                    ax.set_xlabel('x (grid points)')
                    ax.set_ylabel('y (grid points)')
                    plt.colorbar(im, ax=ax, label='Stress (MPa)')
                
                plt.tight_layout()
                plt.savefig(self.viz_path / 'phase2_stress_fields.png', dpi=300, bbox_inches='tight')
                print("✓ Saved: phase2_stress_fields.png")
                plt.close()
        
        # Spatial statistics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Phase 2: Spatial Statistics', fontsize=16, fontweight='bold')
        
        # Temperature statistics
        ax = axes[0, 0]
        ax.scatter(df['operating_temperature_K'], df['max_temperature_K'], 
                  alpha=0.5, s=10, label='Max')
        ax.scatter(df['operating_temperature_K'], df['min_temperature_K'], 
                  alpha=0.5, s=10, label='Min')
        ax.set_xlabel('Operating Temperature (K)')
        ax.set_ylabel('Measured Temperature (K)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Temperature gradient
        ax = axes[0, 1]
        ax.hist(df['temperature_std_K'], bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Temperature Std Dev (K)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Current density distribution
        ax = axes[1, 0]
        ax.scatter(df['current_density_A_cm2'], df['max_current_density_A_cm2'],
                  alpha=0.5, s=10)
        ax.set_xlabel('Average Current Density (A/cm²)')
        ax.set_ylabel('Max Current Density (A/cm²)')
        ax.grid(True, alpha=0.3)
        
        # Stress statistics
        ax = axes[1, 1]
        ax.scatter(df['avg_stress_MPa'], df['max_stress_MPa'],
                  c=df['operating_temperature_K'], cmap='hot', alpha=0.5, s=10)
        ax.set_xlabel('Average Stress (MPa)')
        ax.set_ylabel('Max Stress (MPa)')
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Temperature (K)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_path / 'phase2_spatial_statistics.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: phase2_spatial_statistics.png")
        plt.close()
        
    def visualize_phase3_HF(self):
        """Visualize High-Fidelity dataset"""
        print("\n" + "="*70)
        print("Visualizing Phase 3: High-Fidelity Dataset")
        print("="*70)
        
        # Load data
        df = pd.read_csv(self.dataset_path / 'phase3_HF' / 'phase3_HF_complete.csv')
        
        # 1. Damage indicators
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Phase 3: Damage Indicators', fontsize=16, fontweight='bold')
        
        # Crack initiation
        ax = axes[0, 0]
        ax.hist(df['crack_initiation_indicator'], bins=50, alpha=0.7, edgecolor='black', color='red')
        ax.axvline(x=0.8, color='black', linestyle='--', label='Initiation threshold')
        ax.set_xlabel('Crack Initiation Indicator')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Crack length
        ax = axes[0, 1]
        ax.scatter(df['cycles'], df['crack_length_um'],
                  c=df['operating_temperature_K'], cmap='hot', alpha=0.6, s=20)
        ax.set_xlabel('Thermal Cycles')
        ax.set_ylabel('Crack Length (μm)')
        ax.set_yscale('log')
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Temperature (K)')
        ax.grid(True, alpha=0.3)
        
        # Crack propagation rate
        ax = axes[0, 2]
        ax.hist(np.log10(df['crack_propagation_rate_nm_cycle'] + 1e-10), 
               bins=50, alpha=0.7, edgecolor='black', color='orange')
        ax.set_xlabel('log₁₀(Crack Propagation Rate) (nm/cycle)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Delamination indicator
        ax = axes[1, 0]
        ax.hist(df['delamination_indicator'], bins=50, alpha=0.7, edgecolor='black', color='purple')
        ax.axvline(x=1.0, color='black', linestyle='--', label='Failure threshold')
        ax.set_xlabel('Delamination Indicator')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Ni coarsening
        ax = axes[1, 1]
        ax.scatter(df['Ni_particle_size_initial_nm'], df['Ni_particle_size_current_nm'],
                  c=df['cycles'], cmap='viridis', alpha=0.6, s=20)
        ax.plot([400, 600], [400, 600], 'k--', label='No change')
        ax.set_xlabel('Initial Ni Particle Size (nm)')
        ax.set_ylabel('Current Ni Particle Size (nm)')
        ax.legend()
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Cycles')
        ax.grid(True, alpha=0.3)
        
        # TPB loss
        ax = axes[1, 2]
        ax.scatter(df['cycles'], df['TPB_loss_percent'],
                  c=df['operating_temperature_K'], cmap='plasma', alpha=0.6, s=20)
        ax.set_xlabel('Thermal Cycles')
        ax.set_ylabel('TPB Loss (%)')
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Temperature (K)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_path / 'phase3_damage_indicators.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: phase3_damage_indicators.png")
        plt.close()
        
        # 2. Life prediction
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Phase 3: Life Prediction Analysis', fontsize=16, fontweight='bold')
        
        # Time to failure vs stress
        ax = axes[0, 0]
        scatter = ax.scatter(df['max_stress_MPa'], np.log10(df['time_to_failure_hours']),
                            c=df['thermal_cycling_amplitude_K'], cmap='coolwarm', alpha=0.6, s=20)
        ax.set_xlabel('Max Stress (MPa)')
        ax.set_ylabel('log₁₀(Time to Failure) (hours)')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cycling Amplitude (K)')
        ax.grid(True, alpha=0.3)
        
        # Remaining life
        ax = axes[0, 1]
        remaining_life = df['remaining_life_percent'].dropna()
        if len(remaining_life) > 0:
            ax.hist(remaining_life, bins=50, alpha=0.7, edgecolor='black', color='green')
        ax.set_xlabel('Remaining Life (%)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Performance degradation
        ax = axes[1, 0]
        ax.scatter(df['cycles'], df['voltage_degradation_mV'],
                  c=df['operating_temperature_K'], cmap='hot', alpha=0.6, s=20)
        ax.set_xlabel('Thermal Cycles')
        ax.set_ylabel('Voltage Degradation (mV)')
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Temperature (K)')
        ax.grid(True, alpha=0.3)
        
        # Binary damage state
        ax = axes[1, 1]
        damage_counts = pd.DataFrame({
            'Crack': df['crack_present'].sum(),
            'Delamination': df['delamination_present'].sum(),
            'Both': ((df['crack_present'] == 1) & (df['delamination_present'] == 1)).sum(),
            'None': ((df['crack_present'] == 0) & (df['delamination_present'] == 0)).sum()
        }, index=[0]).T
        
        ax.bar(damage_counts.index, damage_counts[0], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Damage State Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.viz_path / 'phase3_life_prediction.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: phase3_life_prediction.png")
        plt.close()
        
        # 3. Spatial fields (if available)
        try:
            with h5py.File(self.dataset_path / 'phase3_HF' / 'phase3_HF_complete.h5', 'r') as f:
                if 'spatial_fields_2D_slices' in f:
                    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
                    fig.suptitle('Phase 3: High-Fidelity Spatial Fields', fontsize=16, fontweight='bold')
                    
                    for i in range(min(3, len(f['spatial_fields_2D_slices'].keys()) - 3)):
                        sample_grp = f[f'spatial_fields_2D_slices/sample_{i}']
                        
                        # Temperature
                        ax = axes[i, 0]
                        T_field = sample_grp['temperature_K'][:]
                        im = ax.imshow(T_field, cmap='hot', aspect='auto', origin='lower')
                        ax.set_title(f'Temperature - Sample {i}')
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        plt.colorbar(im, ax=ax, label='T (K)')
                        
                        # Stress
                        ax = axes[i, 1]
                        sigma_field = sample_grp['stress_MPa'][:]
                        im = ax.imshow(sigma_field, cmap='RdYlBu_r', aspect='auto', origin='lower')
                        ax.set_title(f'Stress - Sample {i}')
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        plt.colorbar(im, ax=ax, label='σ (MPa)')
                        
                        # Damage
                        ax = axes[i, 2]
                        damage_field = sample_grp['damage_indicator'][:]
                        im = ax.imshow(damage_field, cmap='Reds', aspect='auto', origin='lower', vmin=0, vmax=1)
                        ax.set_title(f'Damage - Sample {i}')
                        ax.set_xlabel('x')
                        ax.set_ylabel('y')
                        plt.colorbar(im, ax=ax, label='Damage')
                    
                    plt.tight_layout()
                    plt.savefig(self.viz_path / 'phase3_spatial_fields.png', dpi=300, bbox_inches='tight')
                    print("✓ Saved: phase3_spatial_fields.png")
                    plt.close()
        except Exception as e:
            print(f"⚠ Could not visualize spatial fields: {e}")
    
    def visualize_phase4_experimental(self):
        """Visualize Experimental dataset"""
        print("\n" + "="*70)
        print("Visualizing Phase 4: Experimental Dataset")
        print("="*70)
        
        # Load data
        df_summary = pd.read_csv(self.dataset_path / 'phase4_experimental' / 'experimental_summary.csv')
        df_iv = pd.read_csv(self.dataset_path / 'phase4_experimental' / 'IV_curves_timeseries.csv')
        df_eis = pd.read_csv(self.dataset_path / 'phase4_experimental' / 'EIS_measurements.csv')
        df_micro = pd.read_csv(self.dataset_path / 'phase4_experimental' / 'microstructural_characterization.csv')
        
        # 1. Experimental test matrix
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Phase 4: Experimental Test Matrix', fontsize=16, fontweight='bold')
        
        # Test conditions
        ax = axes[0, 0]
        scatter = ax.scatter(df_summary['operating_temperature_K'], 
                            df_summary['current_density_A_cm2'],
                            s=df_summary['test_duration_hours']/10,
                            c=df_summary['fuel_utilization'],
                            cmap='viridis', alpha=0.7, edgecolor='black', linewidth=1)
        ax.set_xlabel('Operating Temperature (K)')
        ax.set_ylabel('Current Density (A/cm²)')
        ax.set_title('Test Conditions (size = duration)')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Fuel Utilization')
        ax.grid(True, alpha=0.3)
        
        # Test duration distribution
        ax = axes[0, 1]
        ax.bar(range(len(df_summary)), df_summary['test_duration_hours'], 
              alpha=0.7, edgecolor='black')
        ax.set_xlabel('Cell ID')
        ax.set_ylabel('Test Duration (hours)')
        ax.set_title('Test Duration by Cell')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Degradation rate
        ax = axes[1, 0]
        ax.bar(range(len(df_summary)), df_summary['voltage_degradation_rate_mV_kh'],
              alpha=0.7, edgecolor='black', color='red')
        ax.set_xlabel('Cell ID')
        ax.set_ylabel('Degradation Rate (mV/kh)')
        ax.set_title('Voltage Degradation Rate')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Microstructural degradation
        ax = axes[1, 1]
        width = 0.35
        x = np.arange(len(df_summary))
        ax.bar(x - width/2, df_summary['TPB_loss_percent'], width, 
              label='TPB Loss', alpha=0.7, edgecolor='black')
        ax.bar(x + width/2, df_summary['cracks_observed']*100, width,
              label='Cracks (×100)', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Cell ID')
        ax.set_ylabel('Degradation Metric')
        ax.set_title('Microstructural Degradation')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.viz_path / 'phase4_experimental_overview.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: phase4_experimental_overview.png")
        plt.close()
        
        # 2. I-V curves over time
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Phase 4: I-V Curves Evolution', fontsize=16, fontweight='bold')
        
        cell_ids = df_iv['cell_id'].unique()[:6]
        
        for idx, cell_id in enumerate(cell_ids):
            ax = axes.flatten()[idx]
            cell_data = df_iv[df_iv['cell_id'] == cell_id]
            
            time_points = sorted(cell_data['time_hours'].unique())
            colors = plt.cm.viridis(np.linspace(0, 1, len(time_points)))
            
            for t, color in zip(time_points, colors):
                data_t = cell_data[cell_data['time_hours'] == t]
                ax.plot(data_t['current_density_A_cm2'], data_t['voltage_V'],
                       'o-', color=color, alpha=0.7, label=f't={t}h', markersize=3)
            
            ax.set_xlabel('Current Density (A/cm²)')
            ax.set_ylabel('Voltage (V)')
            ax.set_title(f'{cell_id}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_path / 'phase4_IV_curves.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: phase4_IV_curves.png")
        plt.close()
        
        # 3. EIS Nyquist plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Phase 4: EIS Nyquist Plots', fontsize=16, fontweight='bold')
        
        for idx, cell_id in enumerate(cell_ids):
            ax = axes.flatten()[idx]
            cell_eis = df_eis[df_eis['cell_id'] == cell_id]
            
            time_points = sorted(cell_eis['time_hours'].unique())
            colors = plt.cm.plasma(np.linspace(0, 1, len(time_points)))
            
            for t, color in zip(time_points, colors):
                data_t = cell_eis[cell_eis['time_hours'] == t]
                ax.plot(data_t['Z_real_ohm'], -data_t['Z_imag_ohm'],
                       'o-', color=color, alpha=0.7, label=f't={t}h', markersize=2)
            
            ax.set_xlabel('Z\' (Ω)')
            ax.set_ylabel('-Z\'\' (Ω)')
            ax.set_title(f'{cell_id}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(self.viz_path / 'phase4_EIS_nyquist.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: phase4_EIS_nyquist.png")
        plt.close()
        
        # 4. Microstructural evolution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Phase 4: Microstructural Evolution', fontsize=16, fontweight='bold')
        
        # Ni particle size evolution
        ax = axes[0, 0]
        for cell_id in df_micro['cell_id'].unique()[:5]:
            cell_micro = df_micro[df_micro['cell_id'] == cell_id].sort_values('time_hours')
            ax.plot(cell_micro['time_hours'], cell_micro['Ni_particle_size_nm'],
                   'o-', alpha=0.7, label=cell_id, markersize=6)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Ni Particle Size (nm)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # TPB density evolution
        ax = axes[0, 1]
        for cell_id in df_micro['cell_id'].unique()[:5]:
            cell_micro = df_micro[df_micro['cell_id'] == cell_id].sort_values('time_hours')
            ax.plot(cell_micro['time_hours'], cell_micro['TPB_density_um_um3'],
                   'o-', alpha=0.7, label=cell_id, markersize=6)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('TPB Density (μm/μm³)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Crack observation
        ax = axes[1, 0]
        eol_data = df_micro[df_micro['stage'] == 'EOL']
        cracked = eol_data[eol_data['cracks_observed'] == 1]
        no_crack = eol_data[eol_data['cracks_observed'] == 0]
        
        ax.scatter(no_crack['time_hours'], no_crack['Ni_particle_size_nm'],
                  marker='o', s=80, alpha=0.6, label='No cracks', edgecolor='black')
        ax.scatter(cracked['time_hours'], cracked['Ni_particle_size_nm'],
                  marker='x', s=80, alpha=0.6, label='Cracked', color='red', linewidth=2)
        ax.set_xlabel('Total Test Time (hours)')
        ax.set_ylabel('Final Ni Particle Size (nm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Porosity change
        ax = axes[1, 1]
        for cell_id in df_micro['cell_id'].unique()[:5]:
            cell_micro = df_micro[df_micro['cell_id'] == cell_id].sort_values('time_hours')
            ax.plot(cell_micro['time_hours'], cell_micro['porosity_fraction']*100,
                   'o-', alpha=0.7, label=cell_id, markersize=6)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Porosity (%)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_path / 'phase4_microstructural_evolution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: phase4_microstructural_evolution.png")
        plt.close()
        
    def create_summary_report(self):
        """Create a comprehensive summary visualization"""
        print("\n" + "="*70)
        print("Creating Multi-Fidelity Summary Report")
        print("="*70)
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Multi-Fidelity SOFC Dataset: Comprehensive Summary', 
                    fontsize=18, fontweight='bold')
        
        # Load all datasets
        df_lf = pd.read_csv(self.dataset_path / 'phase1_LF' / 'phase1_LF_complete.csv')
        df_mf = pd.read_csv(self.dataset_path / 'phase2_MF' / 'phase2_MF_global.csv')
        df_hf = pd.read_csv(self.dataset_path / 'phase3_HF' / 'phase3_HF_complete.csv')
        df_exp = pd.read_csv(self.dataset_path / 'phase4_experimental' / 'experimental_summary.csv')
        
        # Dataset sizes
        ax = fig.add_subplot(gs[0, 0])
        phases = ['Phase 1\n(LF)', 'Phase 2\n(MF)', 'Phase 3\n(HF)', 'Phase 4\n(Exp)']
        sizes = [len(df_lf), len(df_mf), len(df_hf), len(df_exp)]
        colors_bar = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        bars = ax.bar(phases, sizes, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Number of Samples', fontweight='bold')
        ax.set_title('Dataset Sizes', fontweight='bold')
        ax.set_yscale('log')
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{size}', ha='center', va='bottom', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Operating conditions coverage
        ax = fig.add_subplot(gs[0, 1])
        ax.scatter(df_lf['operating_temperature_K'], df_lf['current_density_A_cm2'],
                  alpha=0.3, s=5, label='LF', color='blue')
        ax.scatter(df_mf['operating_temperature_K'], df_mf['current_density_A_cm2'],
                  alpha=0.5, s=10, label='MF', color='green')
        ax.scatter(df_hf['operating_temperature_K'], df_hf['current_density_A_cm2'],
                  alpha=0.7, s=30, label='HF', color='orange', edgecolor='black', linewidth=0.5)
        ax.scatter(df_exp['operating_temperature_K'], df_exp['current_density_A_cm2'],
                  alpha=1.0, s=100, label='Exp', color='red', marker='*', edgecolor='black', linewidth=1)
        ax.set_xlabel('Operating Temperature (K)', fontweight='bold')
        ax.set_ylabel('Current Density (A/cm²)', fontweight='bold')
        ax.set_title('Operating Condition Coverage', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Stress distribution across fidelities
        ax = fig.add_subplot(gs[0, 2])
        data_stress = [
            df_lf['avg_von_mises_stress_MPa'],
            df_mf['avg_stress_MPa'],
            df_hf['von_mises_stress_MPa'],
        ]
        bp = ax.boxplot(data_stress, labels=['LF', 'MF', 'HF'], patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_bar[:3]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Von Mises Stress (MPa)', fontweight='bold')
        ax.set_title('Stress Distribution by Fidelity', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Degradation metrics
        ax = fig.add_subplot(gs[0, 3])
        metrics = ['Ni Size\n(nm)', 'Crack\nProb', 'Delam\nRisk']
        lf_metrics = [df_lf['Ni_particle_size_nm'].mean()-500, 
                     df_lf['crack_probability'].mean()*100,
                     df_lf['delamination_risk'].mean()*100]
        hf_metrics = [df_hf['Ni_particle_size_current_nm'].mean()-500,
                     df_hf['crack_initiation_indicator'].mean()*100,
                     df_hf['delamination_indicator'].mean()*100]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width/2, lf_metrics, width, label='LF', alpha=0.7, 
              edgecolor='black', color='blue')
        ax.bar(x + width/2, hf_metrics, width, label='HF', alpha=0.7,
              edgecolor='black', color='orange')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel('Degradation Metric', fontweight='bold')
        ax.set_title('Average Degradation Indicators', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Temperature vs cycles (multi-fidelity)
        ax = fig.add_subplot(gs[1, :2])
        scatter1 = ax.scatter(df_lf['cycles'], df_lf['operating_temperature_K'],
                             c=df_lf['avg_von_mises_stress_MPa'], cmap='hot',
                             alpha=0.3, s=5, label='LF')
        scatter2 = ax.scatter(df_mf['cycles'], df_mf['operating_temperature_K'],
                             c=df_mf['avg_stress_MPa'], cmap='hot',
                             alpha=0.5, s=10, label='MF')
        scatter3 = ax.scatter(df_hf['cycles'], df_hf['operating_temperature_K'],
                             c=df_hf['von_mises_stress_MPa'], cmap='hot',
                             alpha=0.8, s=30, edgecolor='black', linewidth=0.5, label='HF')
        ax.set_xlabel('Thermal Cycles', fontweight='bold')
        ax.set_ylabel('Operating Temperature (K)', fontweight='bold')
        ax.set_title('Operating Envelope (color = stress MPa)', fontweight='bold')
        ax.legend()
        cbar = plt.colorbar(scatter3, ax=ax)
        cbar.set_label('Stress (MPa)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Time to failure
        ax = fig.add_subplot(gs[1, 2:])
        ax.scatter(df_lf['avg_von_mises_stress_MPa'], 
                  np.log10(df_lf['time_to_failure_hours']),
                  alpha=0.3, s=5, label='LF', color='blue')
        ax.scatter(df_hf['von_mises_stress_MPa'],
                  np.log10(df_hf['time_to_failure_hours']),
                  alpha=0.7, s=30, label='HF', color='orange', edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Von Mises Stress (MPa)', fontweight='bold')
        ax.set_ylabel('log₁₀(Time to Failure) (hours)', fontweight='bold')
        ax.set_title('Life Prediction: Multi-Fidelity Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Experimental validation
        ax = fig.add_subplot(gs[2, 0])
        ax.bar(range(len(df_exp)), df_exp['voltage_degradation_rate_mV_kh'],
              alpha=0.7, edgecolor='black', color='red')
        ax.set_xlabel('Cell Number', fontweight='bold')
        ax.set_ylabel('Degradation Rate\n(mV/kh)', fontweight='bold')
        ax.set_title('Experimental Degradation', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Damage distribution (HF)
        ax = fig.add_subplot(gs[2, 1])
        damage_data = pd.DataFrame({
            'No Damage': ((df_hf['crack_present']==0) & (df_hf['delamination_present']==0)).sum(),
            'Crack Only': ((df_hf['crack_present']==1) & (df_hf['delamination_present']==0)).sum(),
            'Delam Only': ((df_hf['crack_present']==0) & (df_hf['delamination_present']==1)).sum(),
            'Both': ((df_hf['crack_present']==1) & (df_hf['delamination_present']==1)).sum()
        }, index=[0])
        
        damage_data.T.plot(kind='bar', ax=ax, legend=False, color=['green', 'orange', 'red', 'darkred'],
                          alpha=0.7, edgecolor='black')
        ax.set_ylabel('Number of Samples', fontweight='bold')
        ax.set_title('HF Damage Distribution', fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Fidelity comparison table
        ax = fig.add_subplot(gs[2, 2:])
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [
            ['Fidelity', 'Samples', 'Spatial', 'Damage', 'Compute'],
            ['Phase 1 (LF)', f'{len(df_lf):,}', '1D/Lumped', 'Indicators', 'Fast'],
            ['Phase 2 (MF)', f'{len(df_mf):,}', '2D Grid', 'Probabilities', 'Medium'],
            ['Phase 3 (HF)', f'{len(df_hf):,}', '3D Fine', 'Explicit', 'Slow'],
            ['Phase 4 (Exp)', f'{len(df_exp)}', 'Multi-scale', 'Observed', 'Real']
        ]
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(5):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code rows
        row_colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        for i, color in enumerate(row_colors, start=1):
            for j in range(5):
                table[(i, j)].set_facecolor(color)
                table[(i, j)].set_alpha(0.3)
        
        ax.set_title('Dataset Characteristics', fontweight='bold', pad=20)
        
        plt.savefig(self.viz_path / 'multifidelity_summary.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: multifidelity_summary.png")
        plt.close()
        
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "="*80)
        print(" "*20 + "DATASET VISUALIZATION SUITE")
        print("="*80)
        
        try:
            self.visualize_phase1_LF()
            self.visualize_phase2_MF()
            self.visualize_phase3_HF()
            self.visualize_phase4_experimental()
            self.create_summary_report()
            
            print("\n" + "="*80)
            print(" "*25 + "VISUALIZATION COMPLETE!")
            print("="*80)
            print(f"\n📊 All visualizations saved to: {self.viz_path}")
            print("\n✅ Dataset ready for analysis and model training!\n")
            
        except Exception as e:
            print(f"\n❌ Error during visualization: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    visualizer = DatasetVisualizer()
    visualizer.generate_all_visualizations()
