#!/usr/bin/env python3
"""
Generate Electrochemical Properties Dataset
Simulates ionic and electronic conductivity data for multi-physics models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import json

# Set random seed for reproducibility
np.random.seed(42)

def arrhenius_conductivity(T, sigma_0, E_a, R=8.314):
    """
    Arrhenius model for temperature-dependent conductivity
    σ = σ_0 * exp(-E_a / (R*T))
    
    T: Temperature in Kelvin
    sigma_0: Pre-exponential factor
    E_a: Activation energy (J/mol)
    R: Gas constant (J/mol·K)
    """
    return sigma_0 * np.exp(-E_a / (R * T))

def generate_electrochemical_properties():
    """
    Generate temperature and environment-dependent electrochemical properties
    Relevant for oxide layer formation, corrosion, and ionic transport
    """
    
    # Temperature range (Celsius)
    temperatures = np.arange(200, 1201, 25)  # 200°C to 1200°C
    
    # Oxygen partial pressures (atm) - different test environments
    oxygen_pressures = [1e-20, 1e-15, 1e-10, 1e-5, 0.001, 0.01, 0.21, 1.0]
    pressure_labels = ['Ultra_Low', 'Very_Low', 'Low', 'Reduced', 
                      'Mild_Reducing', 'Moderate', 'Air', 'Pure_O2']
    
    all_data = []
    
    for p_O2, p_label in zip(oxygen_pressures, pressure_labels):
        # Electronic Conductivity (S/cm) - typically decreases with oxidation
        electronic_cond = []
        
        # Ionic Conductivity (S/cm) - oxide scale, increases with temperature
        ionic_cond = []
        
        # Mixed Ionic-Electronic Conductivity regions
        mixed_cond = []
        
        # Oxide thickness growth parameter
        oxide_thickness = []
        
        for T in temperatures:
            T_kelvin = T + 273.15
            
            # Electronic conductivity - metallic behavior modified by oxidation
            if p_O2 < 1e-10:
                # Metallic behavior dominates
                sigma_e = 1e4 * (300 / T_kelvin) ** 1.5  # Metallic conductivity decreases with T
            else:
                # Oxidation reduces electronic conductivity
                oxidation_factor = 1 / (1 + 100 * p_O2)
                sigma_e = 1e4 * (300 / T_kelvin) ** 1.5 * oxidation_factor
            
            # Add scatter
            sigma_e *= (1 + np.random.normal(0, 0.05))
            electronic_cond.append(max(1e-2, sigma_e))
            
            # Ionic conductivity - through oxide scale
            # Activation energy depends on oxide type
            E_a_ionic = 150000 - 20000 * np.log10(p_O2 + 1e-25)  # J/mol
            sigma_i_0 = 1e6 * p_O2 ** 0.25  # Pre-exponential factor
            
            sigma_i = arrhenius_conductivity(T_kelvin, sigma_i_0, E_a_ionic)
            
            # Limit and add scatter
            sigma_i *= (1 + np.random.normal(0, 0.1))
            sigma_i = max(1e-12, min(1e2, sigma_i))
            ionic_cond.append(sigma_i)
            
            # Mixed conductivity (relevant for SOFC/SOEC applications)
            sigma_mixed = np.sqrt(sigma_e * sigma_i)
            mixed_cond.append(sigma_mixed)
            
            # Oxide thickness (μm) - parabolic growth law
            if T > 600:
                k_p = 1e-6 * np.exp(-120000 / (8.314 * T_kelvin)) * p_O2 ** 0.5
                thickness = np.sqrt(k_p * 1000) * 1e6  # Convert to μm for 1000 hours
            else:
                thickness = 0.1 * (T / 600) ** 2 * p_O2 ** 0.25
            
            thickness *= (1 + np.random.normal(0, 0.1))
            oxide_thickness.append(max(0.01, thickness))
        
        # Calculate additional derived properties
        for i, T in enumerate(temperatures):
            T_kelvin = T + 273.15
            
            # Transference numbers
            t_ion = ionic_cond[i] / (ionic_cond[i] + electronic_cond[i])
            t_elec = electronic_cond[i] / (ionic_cond[i] + electronic_cond[i])
            
            # Activation energies (calculated from local slopes)
            if i > 0:
                dT = (temperatures[i] - temperatures[i-1])
                T_avg = (temperatures[i] + temperatures[i-1]) / 2 + 273.15
                
                if electronic_cond[i] > 0 and electronic_cond[i-1] > 0:
                    E_a_elec = -8.314 * T_avg ** 2 * np.log(electronic_cond[i] / electronic_cond[i-1]) / dT
                else:
                    E_a_elec = np.nan
                    
                if ionic_cond[i] > 0 and ionic_cond[i-1] > 0:
                    E_a_ion = -8.314 * T_avg ** 2 * np.log(ionic_cond[i] / ionic_cond[i-1]) / dT
                else:
                    E_a_ion = np.nan
            else:
                E_a_elec = np.nan
                E_a_ion = np.nan
            
            # Defect concentrations (simplified model)
            vacancy_conc = 1e20 * np.exp(-50000 / (8.314 * T_kelvin)) * p_O2 ** (-0.25)
            interstitial_conc = 1e18 * np.exp(-80000 / (8.314 * T_kelvin)) * p_O2 ** 0.5
            
            all_data.append({
                'Temperature_C': T,
                'Temperature_K': T_kelvin,
                'Oxygen_Pressure_atm': p_O2,
                'Environment': p_label,
                'Electronic_Conductivity_S_per_cm': np.round(electronic_cond[i], 6),
                'Ionic_Conductivity_S_per_cm': np.round(ionic_cond[i], 12),
                'Mixed_Conductivity_S_per_cm': np.round(mixed_cond[i], 9),
                'Ionic_Transference_Number': np.round(t_ion, 6),
                'Electronic_Transference_Number': np.round(t_elec, 6),
                'Oxide_Thickness_um_1000h': np.round(oxide_thickness[i], 3),
                'Activation_Energy_Electronic_kJ_per_mol': np.round(E_a_elec / 1000, 2) if not np.isnan(E_a_elec) else None,
                'Activation_Energy_Ionic_kJ_per_mol': np.round(E_a_ion / 1000, 2) if not np.isnan(E_a_ion) else None,
                'Oxygen_Vacancy_Concentration_per_cm3': np.round(vacancy_conc, 2),
                'Interstitial_Concentration_per_cm3': np.round(interstitial_conc, 2),
                'Measurement_Method': 'Four_Point_Probe',
                'Sample_Type': 'Sintered_Pellet',
                'Material': 'Superalloy_A_Oxidized'
            })
    
    df = pd.DataFrame(all_data)
    
    # Add corrosion current density data (electrochemical corrosion)
    corrosion_data = []
    
    # Different electrolyte conditions
    electrolytes = [
        {'name': '3.5%_NaCl', 'pH': 7, 'conductivity': 53.0},  # mS/cm
        {'name': '0.1M_H2SO4', 'pH': 1, 'conductivity': 39.1},
        {'name': '0.1M_NaOH', 'pH': 13, 'conductivity': 17.7},
        {'name': 'Simulated_Seawater', 'pH': 8.2, 'conductivity': 50.0}
    ]
    
    test_temps = [25, 40, 60, 80]  # °C
    
    for electrolyte in electrolytes:
        for temp in test_temps:
            # Tafel parameters
            E_corr = -0.45 + np.random.normal(0, 0.02)  # V vs SCE
            
            # Corrosion current density (A/cm²)
            i_corr_base = 1e-6 * 10 ** (electrolyte['pH'] / 5)
            i_corr = i_corr_base * np.exp((temp - 25) * 0.03)
            i_corr *= (1 + np.random.normal(0, 0.1))
            
            # Tafel slopes
            beta_a = 60 + np.random.normal(0, 5)  # mV/decade
            beta_c = -120 + np.random.normal(0, 10)  # mV/decade
            
            # Polarization resistance
            R_p = 2.303 * 8.314 * (temp + 273.15) / (96485 * i_corr * (1/abs(beta_a) + 1/abs(beta_c)))
            
            corrosion_data.append({
                'Temperature_C': temp,
                'Electrolyte': electrolyte['name'],
                'pH': electrolyte['pH'],
                'Electrolyte_Conductivity_mS_per_cm': electrolyte['conductivity'],
                'Corrosion_Potential_V_vs_SCE': np.round(E_corr, 3),
                'Corrosion_Current_Density_A_per_cm2': np.round(i_corr, 10),
                'Corrosion_Rate_mm_per_year': np.round(i_corr * 3.27, 6),  # Conversion factor for Fe
                'Tafel_Slope_Anodic_mV_per_decade': np.round(beta_a, 1),
                'Tafel_Slope_Cathodic_mV_per_decade': np.round(beta_c, 1),
                'Polarization_Resistance_ohm_cm2': np.round(R_p, 2),
                'Test_Method': 'Potentiodynamic_Polarization',
                'Reference_Electrode': 'SCE',
                'Counter_Electrode': 'Platinum',
                'Scan_Rate_mV_per_s': 0.5
            })
    
    df_corrosion = pd.DataFrame(corrosion_data)
    
    # Metadata
    metadata = {
        'conductivity_measurements': {
            'method': 'Four-point probe with guard electrodes',
            'sample_preparation': 'Sintered pellets, 10mm diameter, 2mm thickness',
            'contact_material': 'Platinum paste',
            'frequency_range_Hz': '0.1 - 1e6',
            'voltage_amplitude_mV': 10
        },
        'oxidation_conditions': {
            'pre_oxidation': '1000°C for 24 hours in respective atmosphere',
            'oxide_phases': ['Cr2O3', 'Al2O3', 'NiO', 'Spinel phases'],
            'characterization': 'XRD, SEM-EDS, XPS'
        },
        'electrochemical_corrosion': {
            'working_electrode_area_cm2': 1.0,
            'deaeration': 'N2 purge for 30 minutes',
            'stabilization_time_min': 60,
            'ir_compensation': 'Applied'
        },
        'relevant_applications': [
            'Solid Oxide Fuel Cells (SOFC)',
            'High-temperature corrosion',
            'Thermal barrier coatings',
            'Electrochemical sensors'
        ],
        'generation_date': pd.Timestamp.now().isoformat()
    }
    
    return df, df_corrosion, metadata

def plot_conductivity_arrhenius(df, save_path):
    """Create Arrhenius plots for conductivity data"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Arrhenius Plots of Conductivity', fontsize=14, fontweight='bold')
    
    # Get unique environments
    environments = df['Environment'].unique()[:4]  # Plot first 4 for clarity
    
    # Electronic conductivity
    ax = axes[0]
    for env in environments:
        data = df[df['Environment'] == env]
        T_inv = 1000 / data['Temperature_K']  # 1000/T for better scale
        sigma_e = data['Electronic_Conductivity_S_per_cm']
        
        # Only plot non-zero values
        mask = sigma_e > 0
        if mask.any():
            ax.semilogy(T_inv[mask], sigma_e[mask], 'o-', label=env, linewidth=2, markersize=4)
    
    ax.set_xlabel('1000/T (K⁻¹)', fontsize=12)
    ax.set_ylabel('Electronic Conductivity (S/cm)', fontsize=12)
    ax.set_title('Electronic Conductivity')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Ionic conductivity
    ax = axes[1]
    for env in environments:
        data = df[df['Environment'] == env]
        T_inv = 1000 / data['Temperature_K']
        sigma_i = data['Ionic_Conductivity_S_per_cm']
        
        # Only plot non-zero values
        mask = sigma_i > 0
        if mask.any():
            ax.semilogy(T_inv[mask], sigma_i[mask], 's-', label=env, linewidth=2, markersize=4)
    
    ax.set_xlabel('1000/T (K⁻¹)', fontsize=12)
    ax.set_ylabel('Ionic Conductivity (S/cm)', fontsize=12)
    ax.set_title('Ionic Conductivity')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_corrosion_data(df_corrosion, save_path):
    """Create visualization of corrosion data"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Electrochemical Corrosion Properties', fontsize=14, fontweight='bold')
    
    # Corrosion current vs temperature
    ax = axes[0, 0]
    for electrolyte in df_corrosion['Electrolyte'].unique():
        data = df_corrosion[df_corrosion['Electrolyte'] == electrolyte]
        ax.semilogy(data['Temperature_C'], data['Corrosion_Current_Density_A_per_cm2'], 
                   'o-', label=electrolyte, linewidth=2, markersize=6)
    
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('i_corr (A/cm²)', fontsize=11)
    ax.set_title('Corrosion Current Density')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Corrosion rate
    ax = axes[0, 1]
    for electrolyte in df_corrosion['Electrolyte'].unique():
        data = df_corrosion[df_corrosion['Electrolyte'] == electrolyte]
        ax.plot(data['Temperature_C'], data['Corrosion_Rate_mm_per_year'], 
               's-', label=electrolyte, linewidth=2, markersize=6)
    
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Corrosion Rate (mm/year)', fontsize=11)
    ax.set_title('Corrosion Rate')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Corrosion potential vs pH
    ax = axes[1, 0]
    data_25C = df_corrosion[df_corrosion['Temperature_C'] == 25]
    ax.scatter(data_25C['pH'], data_25C['Corrosion_Potential_V_vs_SCE'], 
              c=data_25C['Electrolyte_Conductivity_mS_per_cm'], cmap='viridis', s=100)
    ax.set_xlabel('pH', fontsize=11)
    ax.set_ylabel('E_corr (V vs SCE)', fontsize=11)
    ax.set_title('Corrosion Potential vs pH (25°C)')
    cbar = plt.colorbar(ax.scatter(data_25C['pH'], data_25C['Corrosion_Potential_V_vs_SCE'],
                                   c=data_25C['Electrolyte_Conductivity_mS_per_cm'], cmap='viridis', s=100),
                       ax=ax)
    cbar.set_label('Conductivity (mS/cm)', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Polarization resistance
    ax = axes[1, 1]
    for electrolyte in df_corrosion['Electrolyte'].unique():
        data = df_corrosion[df_corrosion['Electrolyte'] == electrolyte]
        ax.semilogy(data['Temperature_C'], data['Polarization_Resistance_ohm_cm2'], 
                   '^-', label=electrolyte, linewidth=2, markersize=6)
    
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('R_p (Ω·cm²)', fontsize=11)
    ax.set_title('Polarization Resistance')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_transference_numbers(df, save_path):
    """Plot ionic and electronic transference numbers"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Select one environment for clarity
    data = df[df['Environment'] == 'Air']
    
    ax.plot(data['Temperature_C'], data['Ionic_Transference_Number'], 
           'b-', label='Ionic', linewidth=2.5)
    ax.plot(data['Temperature_C'], data['Electronic_Transference_Number'], 
           'r-', label='Electronic', linewidth=2.5)
    
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Transference Number', fontsize=12)
    ax.set_title('Ionic vs Electronic Transport in Air', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Add shaded regions
    ax.fill_between(data['Temperature_C'], 0, data['Ionic_Transference_Number'], 
                   alpha=0.2, color='blue')
    ax.fill_between(data['Temperature_C'], data['Ionic_Transference_Number'], 1, 
                   alpha=0.2, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Generate data
    print("🔄 Generating electrochemical properties data...")
    df, df_corrosion, metadata = generate_electrochemical_properties()
    
    # Save conductivity data to CSV
    csv_path = '../electrochemical/electrochemical_conductivity.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved conductivity data to {csv_path}")
    
    # Save corrosion data to CSV
    corrosion_path = '../electrochemical/electrochemical_corrosion.csv'
    df_corrosion.to_csv(corrosion_path, index=False)
    print(f"✅ Saved corrosion data to {corrosion_path}")
    
    # Save metadata to JSON
    json_path = '../electrochemical/electrochemical_metadata.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata to {json_path}")
    
    # Create visualizations
    arrhenius_path = '../electrochemical/conductivity_arrhenius.png'
    plot_conductivity_arrhenius(df, arrhenius_path)
    print(f"✅ Saved Arrhenius plot to {arrhenius_path}")
    
    corrosion_plot_path = '../electrochemical/corrosion_properties.png'
    plot_corrosion_data(df_corrosion, corrosion_plot_path)
    print(f"✅ Saved corrosion plots to {corrosion_plot_path}")
    
    transference_path = '../electrochemical/transference_numbers.png'
    plot_transference_numbers(df, transference_path)
    print(f"✅ Saved transference number plot to {transference_path}")
    
    # Display statistics
    print("\n📊 Conductivity Data Statistics:")
    print(f"Total data points: {len(df):,}")
    print(f"Temperature range: {df['Temperature_C'].min()}-{df['Temperature_C'].max()}°C")
    print(f"O₂ pressure range: {df['Oxygen_Pressure_atm'].min():.0e} - {df['Oxygen_Pressure_atm'].max():.0f} atm")
    print(f"Electronic conductivity range: {df['Electronic_Conductivity_S_per_cm'].min():.2e} - {df['Electronic_Conductivity_S_per_cm'].max():.2e} S/cm")
    print(f"Ionic conductivity range: {df['Ionic_Conductivity_S_per_cm'].min():.2e} - {df['Ionic_Conductivity_S_per_cm'].max():.2e} S/cm")
    
    print("\n📊 Corrosion Data Statistics:")
    print(f"Total corrosion tests: {len(df_corrosion)}")
    print(f"Electrolytes tested: {df_corrosion['Electrolyte'].nunique()}")
    print(f"Corrosion rate range: {df_corrosion['Corrosion_Rate_mm_per_year'].min():.4f} - {df_corrosion['Corrosion_Rate_mm_per_year'].max():.4f} mm/year")