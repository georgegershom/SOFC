#!/usr/bin/env python3
"""
YSZ Material Properties Visualization Script

This script generates plots for all temperature-dependent material properties
of Yttria-Stabilized Zirconia (YSZ) for SOFC electrolyte applications.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_data(filepath='ysz_material_properties_dataset.csv'):
    """Load the YSZ material properties dataset."""
    return pd.read_csv(filepath)

def create_property_plots(df):
    """Create comprehensive plots for all material properties."""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 24))
    
    # Define colors for consistency
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # 1. Young's Modulus
    ax1 = plt.subplot(4, 3, 1)
    plt.plot(df['Temperature_C'], df['Youngs_Modulus_GPa'], 'o-', color=colors[0], linewidth=2, markersize=4)
    plt.xlabel('Temperature (°C)')
    plt.ylabel("Young's Modulus (GPa)")
    plt.title("Young's Modulus vs Temperature")
    plt.grid(True, alpha=0.3)
    
    # 2. Poisson's Ratio
    ax2 = plt.subplot(4, 3, 2)
    plt.plot(df['Temperature_C'], df['Poissons_Ratio'], 'o-', color=colors[1], linewidth=2, markersize=4)
    plt.xlabel('Temperature (°C)')
    plt.ylabel("Poisson's Ratio")
    plt.title("Poisson's Ratio vs Temperature")
    plt.grid(True, alpha=0.3)
    
    # 3. Coefficient of Thermal Expansion
    ax3 = plt.subplot(4, 3, 3)
    plt.plot(df['Temperature_C'], df['CTE_1e6_per_K'], 'o-', color=colors[2], linewidth=2, markersize=4)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('CTE (×10⁻⁶/K)')
    plt.title('Coefficient of Thermal Expansion vs Temperature')
    plt.grid(True, alpha=0.3)
    
    # 4. Density
    ax4 = plt.subplot(4, 3, 4)
    plt.plot(df['Temperature_C'], df['Density_g_cm3'], 'o-', color=colors[3], linewidth=2, markersize=4)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Density (g/cm³)')
    plt.title('Density vs Temperature')
    plt.grid(True, alpha=0.3)
    
    # 5. Thermal Conductivity
    ax5 = plt.subplot(4, 3, 5)
    plt.plot(df['Temperature_C'], df['Thermal_Conductivity_W_mK'], 'o-', color=colors[4], linewidth=2, markersize=4)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Thermal Conductivity (W/m·K)')
    plt.title('Thermal Conductivity vs Temperature')
    plt.grid(True, alpha=0.3)
    
    # 6. Fracture Toughness
    ax6 = plt.subplot(4, 3, 6)
    plt.plot(df['Temperature_C'], df['Fracture_Toughness_MPa_sqrt_m'], 'o-', color=colors[5], linewidth=2, markersize=4)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Fracture Toughness (MPa√m)')
    plt.title('Fracture Toughness vs Temperature')
    plt.grid(True, alpha=0.3)
    
    # 7. Weibull Modulus
    ax7 = plt.subplot(4, 3, 7)
    plt.plot(df['Temperature_C'], df['Weibull_Modulus'], 'o-', color=colors[6], linewidth=2, markersize=4)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Weibull Modulus')
    plt.title('Weibull Modulus vs Temperature')
    plt.grid(True, alpha=0.3)
    
    # 8. Characteristic Strength
    ax8 = plt.subplot(4, 3, 8)
    plt.plot(df['Temperature_C'], df['Characteristic_Strength_MPa'], 'o-', color=colors[7], linewidth=2, markersize=4)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Characteristic Strength (MPa)')
    plt.title('Characteristic Strength vs Temperature')
    plt.grid(True, alpha=0.3)
    
    # 9. Creep Parameter A (log scale)
    ax9 = plt.subplot(4, 3, 9)
    plt.semilogy(df['Temperature_C'], df['Creep_A_1_Pa_s'], 'o-', color=colors[0], linewidth=2, markersize=4)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Creep Parameter A (Pa⁻¹s⁻¹)')
    plt.title('Creep Parameter A vs Temperature (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    # 10. Combined Mechanical Properties
    ax10 = plt.subplot(4, 3, 10)
    ax10_twin = ax10.twinx()
    
    line1 = ax10.plot(df['Temperature_C'], df['Youngs_Modulus_GPa'], 'o-', color=colors[0], 
                     linewidth=2, markersize=4, label="Young's Modulus")
    ax10.set_xlabel('Temperature (°C)')
    ax10.set_ylabel("Young's Modulus (GPa)", color=colors[0])
    ax10.tick_params(axis='y', labelcolor=colors[0])
    
    line2 = ax10_twin.plot(df['Temperature_C'], df['Fracture_Toughness_MPa_sqrt_m'], 's-', color=colors[5], 
                          linewidth=2, markersize=4, label='Fracture Toughness')
    ax10_twin.set_ylabel('Fracture Toughness (MPa√m)', color=colors[5])
    ax10_twin.tick_params(axis='y', labelcolor=colors[5])
    
    plt.title('Mechanical Properties vs Temperature')
    ax10.grid(True, alpha=0.3)
    
    # 11. Thermal Properties Combined
    ax11 = plt.subplot(4, 3, 11)
    ax11_twin = ax11.twinx()
    
    line1 = ax11.plot(df['Temperature_C'], df['CTE_1e6_per_K'], 'o-', color=colors[2], 
                     linewidth=2, markersize=4, label='CTE')
    ax11.set_xlabel('Temperature (°C)')
    ax11.set_ylabel('CTE (×10⁻⁶/K)', color=colors[2])
    ax11.tick_params(axis='y', labelcolor=colors[2])
    
    line2 = ax11_twin.plot(df['Temperature_C'], df['Thermal_Conductivity_W_mK'], 's-', color=colors[4], 
                          linewidth=2, markersize=4, label='Thermal Conductivity')
    ax11_twin.set_ylabel('Thermal Conductivity (W/m·K)', color=colors[4])
    ax11_twin.tick_params(axis='y', labelcolor=colors[4])
    
    plt.title('Thermal Properties vs Temperature')
    ax11.grid(True, alpha=0.3)
    
    # 12. Arrhenius Plot for Creep
    ax12 = plt.subplot(4, 3, 12)
    inv_temp = 1000 / df['Temperature_K']  # 1000/T for better scaling
    plt.semilogy(inv_temp, df['Creep_A_1_Pa_s'], 'o-', color=colors[1], linewidth=2, markersize=4)
    plt.xlabel('1000/T (K⁻¹)')
    plt.ylabel('Creep Parameter A (Pa⁻¹s⁻¹)')
    plt.title('Arrhenius Plot for Creep Parameter A')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    return fig

def create_summary_table(df):
    """Create a summary table of key properties at important temperatures."""
    
    # Select key temperatures
    key_temps = [25, 500, 800, 1000, 1200, 1500]
    summary_data = []
    
    for temp in key_temps:
        # Find closest temperature in dataset
        idx = (df['Temperature_C'] - temp).abs().idxmin()
        row_data = df.iloc[idx]
        
        summary_data.append({
            'Temperature (°C)': int(row_data['Temperature_C']),
            'E (GPa)': f"{row_data['Youngs_Modulus_GPa']:.1f}",
            'ν': f"{row_data['Poissons_Ratio']:.3f}",
            'CTE (×10⁻⁶/K)': f"{row_data['CTE_1e6_per_K']:.1f}",
            'k (W/m·K)': f"{row_data['Thermal_Conductivity_W_mK']:.2f}",
            'K_IC (MPa√m)': f"{row_data['Fracture_Toughness_MPa_sqrt_m']:.1f}",
            'σ₀ (MPa)': f"{row_data['Characteristic_Strength_MPa']:.0f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def main():
    """Main function to generate all plots and summaries."""
    
    # Load data
    print("Loading YSZ material properties dataset...")
    df = load_data()
    
    # Create plots
    print("Generating property plots...")
    fig = create_property_plots(df)
    
    # Save plots
    plt.savefig('ysz_material_properties_plots.png', dpi=300, bbox_inches='tight')
    plt.savefig('ysz_material_properties_plots.pdf', bbox_inches='tight')
    print("Plots saved as 'ysz_material_properties_plots.png' and '.pdf'")
    
    # Create and save summary table
    print("Generating summary table...")
    summary_df = create_summary_table(df)
    summary_df.to_csv('ysz_properties_summary.csv', index=False)
    print("Summary table saved as 'ysz_properties_summary.csv'")
    
    # Print summary to console
    print("\nYSZ Material Properties Summary:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Temperature range: {df['Temperature_C'].min():.0f}°C to {df['Temperature_C'].max():.0f}°C")
    print(f"Number of data points: {len(df)}")
    print(f"Properties included: {len(df.columns) - 2}")  # Excluding temperature columns
    
    plt.show()

if __name__ == "__main__":
    main()