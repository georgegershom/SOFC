#!/usr/bin/env python3
"""
Comprehensive analysis and visualization tool for material properties datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, interpolate
import json
import os
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_datasets():
    """Load all generated datasets"""
    
    datasets = {}
    
    # Mechanical properties
    if os.path.exists('../mechanical/mechanical_properties.csv'):
        datasets['mechanical'] = pd.read_csv('../mechanical/mechanical_properties.csv')
        print("✅ Loaded mechanical properties dataset")
    
    # Creep properties
    if os.path.exists('../creep/creep_curves_full.csv'):
        datasets['creep'] = pd.read_csv('../creep/creep_curves_full.csv')
        datasets['creep_summary'] = pd.read_csv('../creep/creep_summary.csv')
        print("✅ Loaded creep properties dataset")
    
    # Thermophysical properties
    if os.path.exists('../thermophysical/thermophysical_properties.csv'):
        datasets['thermophysical'] = pd.read_csv('../thermophysical/thermophysical_properties.csv')
        print("✅ Loaded thermophysical properties dataset")
    
    # Electrochemical properties
    if os.path.exists('../electrochemical/electrochemical_conductivity.csv'):
        datasets['electrochemical'] = pd.read_csv('../electrochemical/electrochemical_conductivity.csv')
        datasets['corrosion'] = pd.read_csv('../electrochemical/electrochemical_corrosion.csv')
        print("✅ Loaded electrochemical properties dataset")
    
    return datasets

def generate_correlation_analysis(datasets):
    """Generate correlation analysis between different properties"""
    
    if 'mechanical' not in datasets or 'thermophysical' not in datasets:
        return None
    
    # Merge datasets on temperature
    mech = datasets['mechanical']
    thermo = datasets['thermophysical']
    
    # Find common temperatures
    common_temps = set(mech['Temperature_C']).intersection(set(thermo['Temperature_C']))
    
    if not common_temps:
        return None
    
    # Create merged dataset
    merged_data = []
    for temp in sorted(common_temps):
        mech_row = mech[mech['Temperature_C'] == temp].iloc[0]
        thermo_row = thermo[thermo['Temperature_C'] == temp].iloc[0]
        
        merged_data.append({
            'Temperature_C': temp,
            'Youngs_Modulus_GPa': mech_row['Youngs_Modulus_GPa'],
            'Tensile_Strength_MPa': mech_row['Tensile_Strength_MPa'],
            'Poissons_Ratio': mech_row['Poissons_Ratio'],
            'CTE_ppm_per_K': thermo_row['CTE_ppm_per_K'],
            'Thermal_Conductivity_W_per_mK': thermo_row['Thermal_Conductivity_W_per_mK'],
            'Specific_Heat_J_per_kgK': thermo_row['Specific_Heat_J_per_kgK']
        })
    
    correlation_df = pd.DataFrame(merged_data)
    
    # Calculate correlation matrix
    corr_matrix = correlation_df.drop('Temperature_C', axis=1).corr()
    
    # Create correlation plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Matrix: Mechanical vs Thermophysical Properties', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../docs/correlation_matrix.png', dpi=300)
    plt.close()
    
    return correlation_df, corr_matrix

def generate_property_models(datasets):
    """Generate empirical models for property prediction"""
    
    models = {}
    
    if 'mechanical' in datasets:
        mech = datasets['mechanical']
        
        # Fit Young's Modulus vs Temperature
        T = mech['Temperature_C'].values
        E = mech['Youngs_Modulus_GPa'].values
        
        # Polynomial fit
        E_coeffs = np.polyfit(T, E, 3)
        E_model = np.poly1d(E_coeffs)
        
        models['youngs_modulus'] = {
            'type': 'polynomial',
            'order': 3,
            'coefficients': E_coeffs.tolist(),
            'r_squared': 1 - np.sum((E - E_model(T))**2) / np.sum((E - E.mean())**2),
            'equation': f"E(T) = {E_coeffs[0]:.2e}*T³ + {E_coeffs[1]:.2e}*T² + {E_coeffs[2]:.2e}*T + {E_coeffs[3]:.2f}"
        }
        
        # Fit Tensile Strength vs Temperature (exponential for high T)
        TS = mech['Tensile_Strength_MPa'].values
        
        # Piecewise model
        T_crit = 600  # Critical temperature
        mask_low = T < T_crit
        mask_high = T >= T_crit
        
        # Low temperature: linear
        if mask_low.any():
            low_coeffs = np.polyfit(T[mask_low], TS[mask_low], 1)
            
        # High temperature: exponential decay
        if mask_high.any():
            log_TS = np.log(TS[mask_high])
            high_coeffs = np.polyfit(T[mask_high], log_TS, 1)
        
        models['tensile_strength'] = {
            'type': 'piecewise',
            'critical_temp': T_crit,
            'low_temp_coeffs': low_coeffs.tolist() if mask_low.any() else None,
            'high_temp_coeffs': high_coeffs.tolist() if mask_high.any() else None,
            'description': f"Linear below {T_crit}°C, exponential decay above"
        }
    
    if 'thermophysical' in datasets:
        thermo = datasets['thermophysical']
        
        # Fit CTE vs Temperature
        T = thermo['Temperature_C'].values
        CTE = thermo['CTE_ppm_per_K'].values
        
        # Logarithmic fit
        CTE_coeffs = np.polyfit(np.log(T + 1), CTE, 1)
        
        models['cte'] = {
            'type': 'logarithmic',
            'coefficients': CTE_coeffs.tolist(),
            'equation': f"CTE(T) = {CTE_coeffs[0]:.2f} * ln(T+1) + {CTE_coeffs[1]:.2f}"
        }
    
    return models

def generate_summary_report(datasets):
    """Generate a comprehensive summary report"""
    
    report = {
        'generation_date': datetime.now().isoformat(),
        'dataset_summary': {}
    }
    
    # Mechanical properties summary
    if 'mechanical' in datasets:
        mech = datasets['mechanical']
        report['dataset_summary']['mechanical'] = {
            'total_points': len(mech),
            'temperature_range_C': [float(mech['Temperature_C'].min()), 
                                   float(mech['Temperature_C'].max())],
            'youngs_modulus_range_GPa': [float(mech['Youngs_Modulus_GPa'].min()), 
                                        float(mech['Youngs_Modulus_GPa'].max())],
            'tensile_strength_range_MPa': [float(mech['Tensile_Strength_MPa'].min()), 
                                          float(mech['Tensile_Strength_MPa'].max())],
            'properties_measured': list(mech.columns)
        }
    
    # Creep properties summary
    if 'creep' in datasets:
        creep = datasets['creep']
        report['dataset_summary']['creep'] = {
            'total_points': len(creep),
            'unique_tests': creep['Test_ID'].nunique(),
            'temperature_range_C': [float(creep['Temperature_C'].min()), 
                                   float(creep['Temperature_C'].max())],
            'stress_range_MPa': [float(creep['Stress_MPa'].min()), 
                               float(creep['Stress_MPa'].max())],
            'max_test_duration_hours': float(creep['Time_Hours'].max()),
            'failed_specimens': int(creep.groupby('Test_ID')['Specimen_Failed'].first().sum())
        }
    
    # Thermophysical properties summary
    if 'thermophysical' in datasets:
        thermo = datasets['thermophysical']
        report['dataset_summary']['thermophysical'] = {
            'total_points': len(thermo),
            'temperature_range_C': [float(thermo['Temperature_C'].min()), 
                                   float(thermo['Temperature_C'].max())],
            'cte_range_ppm_per_K': [float(thermo['CTE_ppm_per_K'].min()), 
                                   float(thermo['CTE_ppm_per_K'].max())],
            'thermal_conductivity_range_W_per_mK': [float(thermo['Thermal_Conductivity_W_per_mK'].min()), 
                                                   float(thermo['Thermal_Conductivity_W_per_mK'].max())],
            'max_linear_expansion_percent': float(thermo['Linear_Expansion_Percent'].max())
        }
    
    # Electrochemical properties summary
    if 'electrochemical' in datasets:
        electro = datasets['electrochemical']
        report['dataset_summary']['electrochemical'] = {
            'total_points': len(electro),
            'temperature_range_C': [float(electro['Temperature_C'].min()), 
                                   float(electro['Temperature_C'].max())],
            'oxygen_pressure_environments': electro['Environment'].unique().tolist(),
            'conductivity_type': ['Electronic', 'Ionic', 'Mixed'],
            'electronic_conductivity_range_S_per_cm': [float(electro['Electronic_Conductivity_S_per_cm'].min()), 
                                                      float(electro['Electronic_Conductivity_S_per_cm'].max())]
        }
    
    # Corrosion data summary
    if 'corrosion' in datasets:
        corr = datasets['corrosion']
        report['dataset_summary']['corrosion'] = {
            'total_tests': len(corr),
            'electrolytes': corr['Electrolyte'].unique().tolist(),
            'temperature_range_C': [float(corr['Temperature_C'].min()), 
                                   float(corr['Temperature_C'].max())],
            'corrosion_rate_range_mm_per_year': [float(corr['Corrosion_Rate_mm_per_year'].min()), 
                                                float(corr['Corrosion_Rate_mm_per_year'].max())]
        }
    
    return report

def create_master_visualization(datasets):
    """Create a comprehensive master visualization"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Mechanical properties
    if 'mechanical' in datasets:
        mech = datasets['mechanical']
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(mech['Temperature_C'], mech['Youngs_Modulus_GPa'], 'b-', linewidth=2)
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel("Young's Modulus (GPa)")
        ax1.set_title('Elastic Modulus')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(mech['Temperature_C'], mech['Tensile_Strength_MPa'], 'r-', linewidth=2)
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Tensile Strength (MPa)')
        ax2.set_title('Strength')
        ax2.grid(True, alpha=0.3)
    
    # Creep properties
    if 'creep' in datasets:
        creep = datasets['creep']
        
        ax3 = fig.add_subplot(gs[0, 2])
        # Select one condition for display
        sample = creep[(creep['Temperature_C'] == 800) & (creep['Stress_MPa'] == 200)]
        if not sample.empty:
            ax3.loglog(sample['Time_Hours'], sample['Creep_Strain_Percent'], 'g-', linewidth=2)
            ax3.set_xlabel('Time (hours)')
            ax3.set_ylabel('Creep Strain (%)')
            ax3.set_title('Creep @ 800°C, 200MPa')
            ax3.grid(True, alpha=0.3, which='both')
    
    # Thermophysical properties
    if 'thermophysical' in datasets:
        thermo = datasets['thermophysical']
        
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.plot(thermo['Temperature_C'], thermo['CTE_ppm_per_K'], 'c-', linewidth=2)
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('CTE (ppm/K)')
        ax4.set_title('Thermal Expansion')
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.plot(thermo['Temperature_C'], thermo['Thermal_Conductivity_W_per_mK'], 'm-', linewidth=2)
        ax5.set_xlabel('Temperature (°C)')
        ax5.set_ylabel('k (W/m·K)')
        ax5.set_title('Thermal Conductivity')
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.plot(thermo['Temperature_C'], thermo['Specific_Heat_J_per_kgK'], 'orange', linewidth=2)
        ax6.set_xlabel('Temperature (°C)')
        ax6.set_ylabel('Cp (J/kg·K)')
        ax6.set_title('Specific Heat')
        ax6.grid(True, alpha=0.3)
    
    # Electrochemical properties
    if 'electrochemical' in datasets:
        electro = datasets['electrochemical']
        
        ax7 = fig.add_subplot(gs[1, 2])
        # Select Air environment
        air_data = electro[electro['Environment'] == 'Air']
        if not air_data.empty:
            T_inv = 1000 / air_data['Temperature_K']
            ax7.semilogy(T_inv, air_data['Electronic_Conductivity_S_per_cm'], 'ko-', 
                        label='Electronic', markersize=3)
            ax7.semilogy(T_inv, air_data['Ionic_Conductivity_S_per_cm'], 'ro-', 
                        label='Ionic', markersize=3)
            ax7.set_xlabel('1000/T (K⁻¹)')
            ax7.set_ylabel('Conductivity (S/cm)')
            ax7.set_title('Conductivity in Air')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
    
    # Corrosion properties
    if 'corrosion' in datasets:
        corr = datasets['corrosion']
        
        ax8 = fig.add_subplot(gs[1, 3])
        # Plot for seawater
        seawater = corr[corr['Electrolyte'] == 'Simulated_Seawater']
        if not seawater.empty:
            ax8.semilogy(seawater['Temperature_C'], 
                        seawater['Corrosion_Current_Density_A_per_cm2'], 'bs-', linewidth=2)
            ax8.set_xlabel('Temperature (°C)')
            ax8.set_ylabel('i_corr (A/cm²)')
            ax8.set_title('Corrosion in Seawater')
            ax8.grid(True, alpha=0.3)
    
    # Combined property map (if correlation data available)
    if 'mechanical' in datasets and 'thermophysical' in datasets:
        ax9 = fig.add_subplot(gs[2, :2])
        
        # Create a 2D property map
        mech = datasets['mechanical']
        thermo = datasets['thermophysical']
        
        # Find common temperatures and create normalized property map
        common_temps = sorted(set(mech['Temperature_C']).intersection(set(thermo['Temperature_C'])))
        
        if common_temps:
            properties = ['E_norm', 'TS_norm', 'CTE_norm', 'k_norm']
            data_matrix = np.zeros((len(properties), len(common_temps)))
            
            for i, temp in enumerate(common_temps):
                mech_row = mech[mech['Temperature_C'] == temp].iloc[0]
                thermo_row = thermo[thermo['Temperature_C'] == temp].iloc[0]
                
                # Normalize to 0-1
                data_matrix[0, i] = (mech_row['Youngs_Modulus_GPa'] - mech['Youngs_Modulus_GPa'].min()) / \
                                   (mech['Youngs_Modulus_GPa'].max() - mech['Youngs_Modulus_GPa'].min())
                data_matrix[1, i] = (mech_row['Tensile_Strength_MPa'] - mech['Tensile_Strength_MPa'].min()) / \
                                   (mech['Tensile_Strength_MPa'].max() - mech['Tensile_Strength_MPa'].min())
                data_matrix[2, i] = (thermo_row['CTE_ppm_per_K'] - thermo['CTE_ppm_per_K'].min()) / \
                                   (thermo['CTE_ppm_per_K'].max() - thermo['CTE_ppm_per_K'].min())
                data_matrix[3, i] = (thermo_row['Thermal_Conductivity_W_per_mK'] - thermo['Thermal_Conductivity_W_per_mK'].min()) / \
                                   (thermo['Thermal_Conductivity_W_per_mK'].max() - thermo['Thermal_Conductivity_W_per_mK'].min())
            
            im = ax9.imshow(data_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
            ax9.set_yticks(range(len(properties)))
            ax9.set_yticklabels(['E (norm)', 'TS (norm)', 'CTE (norm)', 'k (norm)'])
            ax9.set_xticks(range(0, len(common_temps), max(1, len(common_temps)//10)))
            ax9.set_xticklabels([f'{common_temps[i]}°C' for i in range(0, len(common_temps), max(1, len(common_temps)//10))], 
                               rotation=45)
            ax9.set_title('Normalized Property Map vs Temperature')
            plt.colorbar(im, ax=ax9, label='Normalized Value')
    
    # Data statistics summary
    ax10 = fig.add_subplot(gs[2, 2:])
    ax10.axis('off')
    
    summary_text = "📊 Dataset Summary\n" + "="*40 + "\n"
    
    if 'mechanical' in datasets:
        summary_text += f"Mechanical: {len(datasets['mechanical'])} points\n"
    if 'creep' in datasets:
        summary_text += f"Creep: {len(datasets['creep']):,} points, {datasets['creep']['Test_ID'].nunique()} tests\n"
    if 'thermophysical' in datasets:
        summary_text += f"Thermophysical: {len(datasets['thermophysical'])} points\n"
    if 'electrochemical' in datasets:
        summary_text += f"Electrochemical: {len(datasets['electrochemical'])} points\n"
    if 'corrosion' in datasets:
        summary_text += f"Corrosion: {len(datasets['corrosion'])} tests\n"
    
    summary_text += "\n" + "="*40 + "\n"
    summary_text += "✅ All datasets ready for multi-physics modeling\n"
    summary_text += "✅ Temperature-dependent properties captured\n"
    summary_text += "✅ Multiple test conditions included\n"
    summary_text += "✅ Synthetic data based on literature values"
    
    ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Material Properties Dataset Overview - Multi-Physics Model Calibration', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../docs/master_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Created master visualization")

def main():
    """Main analysis function"""
    
    print("🔍 Analyzing Material Properties Datasets")
    print("="*60)
    
    # Load all datasets
    datasets = load_datasets()
    
    if not datasets:
        print("❌ No datasets found. Please run generate_all_datasets.py first.")
        return
    
    # Create docs directory
    os.makedirs('../docs', exist_ok=True)
    
    # Generate correlation analysis
    print("\n📊 Generating correlation analysis...")
    correlation_data, corr_matrix = generate_correlation_analysis(datasets)
    if correlation_data is not None:
        correlation_data.to_csv('../docs/correlation_data.csv', index=False)
        print("✅ Saved correlation data")
    
    # Generate property models
    print("\n🔧 Generating empirical models...")
    models = generate_property_models(datasets)
    if models:
        with open('../docs/empirical_models.json', 'w') as f:
            json.dump(models, f, indent=2)
        print("✅ Saved empirical models")
    
    # Generate summary report
    print("\n📄 Generating summary report...")
    report = generate_summary_report(datasets)
    with open('../docs/summary_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("✅ Saved summary report")
    
    # Create master visualization
    print("\n🎨 Creating master visualization...")
    create_master_visualization(datasets)
    
    print("\n" + "="*60)
    print("✅ Analysis complete!")
    print("📁 Check the 'docs' folder for analysis results:")
    print("   • correlation_matrix.png - Property correlations")
    print("   • master_visualization.png - Overview of all properties")
    print("   • summary_report.json - Detailed statistics")
    print("   • empirical_models.json - Fitted models for properties")
    print("="*60)

if __name__ == "__main__":
    main()