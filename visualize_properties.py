"""
SOFC Material Properties Visualization
=======================================
Create plots and comparisons of material properties
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sofc_material_properties import SOFCMaterialDatabase

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')

def plot_tec_comparison(db):
    """Compare thermal expansion coefficients across materials"""
    materials = {
        'Ni-YSZ (Anode)': db.get_thermal_expansion_coefficient('anode', 'Ni-YSZ'),
        '8YSZ (Electrolyte)': db.get_thermal_expansion_coefficient('electrolyte', '8YSZ'),
        'CGO (Electrolyte)': db.get_thermal_expansion_coefficient('electrolyte', 'CGO'),
        'LSM (Cathode)': db.get_thermal_expansion_coefficient('cathode', 'LSM'),
        'LSM-YSZ (Cathode)': db.get_thermal_expansion_coefficient('cathode', 'LSM-YSZ'),
        'LSCF (Cathode)': db.get_thermal_expansion_coefficient('cathode', 'LSCF'),
        'Crofer (Interconnect)': db.get_thermal_expansion_coefficient('interconnect', 'Crofer_22_APU')
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(materials.keys())
    values = [v * 1e6 for v in materials.values()]  # Convert to 10^-6 /K
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#f0932b', '#eb4d4b', '#95afc0']
    
    bars = ax.barh(names, values, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_xlabel('Thermal Expansion Coefficient (× 10⁻⁶ K⁻¹)', fontsize=12, fontweight='bold')
    ax.set_title('Thermal Expansion Coefficients of SOFC Materials', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.2, i, f'{val:.1f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('tec_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: tec_comparison.png")
    plt.close()


def plot_conductivity_vs_temperature(db):
    """Plot ionic conductivity vs temperature for electrolytes"""
    temperatures = np.linspace(773, 1273, 50)
    
    # Calculate conductivities
    sigma_ysz = [db.get_ionic_conductivity('electrolyte', '8YSZ', T) for T in temperatures]
    sigma_cgo = [db.get_ionic_conductivity('electrolyte', 'CGO', T) for T in temperatures]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(temperatures - 273.15, sigma_ysz, 'o-', linewidth=2.5, 
            markersize=5, label='8YSZ', color='#3498db')
    ax.plot(temperatures - 273.15, sigma_cgo, 's-', linewidth=2.5, 
            markersize=5, label='CGO (Gd-doped Ceria)', color='#e74c3c')
    
    ax.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ionic Conductivity (S/m)', fontsize=12, fontweight='bold')
    ax.set_title('Temperature-Dependent Ionic Conductivity\n(Arrhenius Behavior)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Add annotations
    ax.annotate('Intermediate\nTemperature\nSOFC', xy=(650, 5), 
                fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.annotate('High\nTemperature\nSOFC', xy=(850, 5), 
                fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('conductivity_vs_temperature.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: conductivity_vs_temperature.png")
    plt.close()


def plot_mechanical_properties(db):
    """Compare Young's modulus and Poisson's ratio"""
    materials_data = {
        'Ni-YSZ': ('anode', 'Ni-YSZ'),
        '8YSZ': ('electrolyte', '8YSZ'),
        'CGO': ('electrolyte', 'CGO'),
        'LSM': ('cathode', 'LSM'),
        'LSM-YSZ': ('cathode', 'LSM-YSZ'),
        'LSCF': ('cathode', 'LSCF'),
        'Crofer 22': ('interconnect', 'Crofer_22_APU')
    }
    
    youngs_moduli = []
    poisson_ratios = []
    names = []
    
    for name, (comp, mat) in materials_data.items():
        props = db.get_material_properties(comp, mat)
        youngs_moduli.append(props['mechanical'].youngs_modulus / 1e9)  # Convert to GPa
        poisson_ratios.append(props['mechanical'].poissons_ratio)
        names.append(name)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    
    # Young's Modulus
    bars1 = ax1.bar(names, youngs_moduli, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel("Young's Modulus (GPa)", fontsize=12, fontweight='bold')
    ax1.set_title("Elastic Stiffness", fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, youngs_moduli):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Poisson's Ratio
    bars2 = ax2.bar(names, poisson_ratios, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel("Poisson's Ratio", fontsize=12, fontweight='bold')
    ax2.set_title("Lateral Strain Response", fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim([0, 0.4])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, poisson_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('mechanical_properties.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: mechanical_properties.png")
    plt.close()


def plot_creep_rates(db):
    """Plot creep rates for materials with creep parameters"""
    stresses = np.linspace(10e6, 100e6, 50)  # 10-100 MPa
    temperature = 1073  # K
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ni-YSZ
    creep_ni_ysz = [db.calculate_creep_rate('anode', 'Ni-YSZ', s, temperature) 
                    for s in stresses]
    
    # 8YSZ
    creep_ysz = [db.calculate_creep_rate('electrolyte', '8YSZ', s, temperature) 
                 for s in stresses]
    
    # Crofer
    creep_crofer = [db.calculate_creep_rate('interconnect', 'Crofer_22_APU', s, temperature) 
                    for s in stresses]
    
    ax.loglog(stresses/1e6, creep_ni_ysz, 'o-', linewidth=2.5, 
              label='Ni-YSZ (Anode)', markersize=5, color='#e74c3c')
    ax.loglog(stresses/1e6, creep_ysz, 's-', linewidth=2.5, 
              label='8YSZ (Electrolyte)', markersize=5, color='#3498db')
    ax.loglog(stresses/1e6, creep_crofer, '^-', linewidth=2.5, 
              label='Crofer 22 APU (Interconnect)', markersize=5, color='#2ecc71')
    
    ax.set_xlabel('Applied Stress (MPa)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Creep Strain Rate (s⁻¹)', fontsize=12, fontweight='bold')
    ax.set_title(f'Creep Behavior at T = {temperature}K ({temperature-273}°C)\nNorton-Bailey Power Law', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, frameon=True, shadow=True)
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('creep_rates.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: creep_rates.png")
    plt.close()


def plot_porosity_comparison(db):
    """Compare porosity across different components"""
    materials = {
        'Ni-YSZ\n(Anode)': ('anode', 'Ni-YSZ'),
        '8YSZ\n(Electrolyte)': ('electrolyte', '8YSZ'),
        'CGO\n(Electrolyte)': ('electrolyte', 'CGO'),
        'LSM\n(Cathode)': ('cathode', 'LSM'),
        'LSM-YSZ\n(Cathode)': ('cathode', 'LSM-YSZ'),
        'LSCF\n(Cathode)': ('cathode', 'LSCF')
    }
    
    porosities = []
    names = []
    colors_list = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#f0932b', '#eb4d4b']
    
    for name, (comp, mat) in materials.items():
        props = db.get_material_properties(comp, mat)
        porosity = props['thermo_physical'].porosity * 100  # Convert to percentage
        porosities.append(porosity)
        names.append(name)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, porosities, color=colors_list, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Porosity (%)', fontsize=12, fontweight='bold')
    ax.set_title('Porosity Requirements in SOFC Components', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 45])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels and functional notes
    for bar, val in zip(bars, porosities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add annotations
    ax.axhline(y=30, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.5, 31, 'Min. for gas diffusion', fontsize=9, color='red', style='italic')
    
    ax.axhline(y=5, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.5, 6, 'Gas-tight requirement', fontsize=9, color='blue', style='italic')
    
    plt.tight_layout()
    plt.savefig('porosity_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: porosity_comparison.png")
    plt.close()


def plot_electrochemical_properties(db):
    """Compare electrochemical properties"""
    materials = {
        'Ni-YSZ\n(Anode)': ('anode', 'Ni-YSZ'),
        'LSM\n(Cathode)': ('cathode', 'LSM'),
        'LSM-YSZ\n(Cathode)': ('cathode', 'LSM-YSZ'),
        'LSCF\n(Cathode)': ('cathode', 'LSCF')
    }
    
    names = []
    i0_values = []
    sigma_e_values = []
    sigma_i_values = []
    
    for name, (comp, mat) in materials.items():
        props = db.get_material_properties(comp, mat)
        ec = props['electrochemical']
        names.append(name)
        i0_values.append(ec.exchange_current_density if ec.exchange_current_density else 0)
        sigma_e_values.append(ec.electronic_conductivity)
        sigma_i_values.append(ec.ionic_conductivity)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Exchange current density
    colors = ['#e74c3c', '#f39c12', '#f1c40f', '#e67e22']
    bars1 = ax1.bar(names, i0_values, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Exchange Current Density (A/m²)', fontsize=12, fontweight='bold')
    ax1.set_title('Electrochemical Activity', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars1, i0_values):
        if val > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Conductivities
    x = np.arange(len(names))
    width = 0.35
    
    bars2a = ax2.bar(x - width/2, sigma_e_values, width, label='Electronic', 
                     color='#3498db', edgecolor='black', linewidth=1.2)
    bars2b = ax2.bar(x + width/2, sigma_i_values, width, label='Ionic', 
                     color='#2ecc71', edgecolor='black', linewidth=1.2)
    
    ax2.set_ylabel('Conductivity (S/m)', fontsize=12, fontweight='bold')
    ax2.set_title('Electronic vs Ionic Conductivity', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.legend(fontsize=11, frameon=True, shadow=True)
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('electrochemical_properties.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: electrochemical_properties.png")
    plt.close()


def create_summary_table():
    """Create a nicely formatted summary table"""
    db = SOFCMaterialDatabase()
    df = db.export_to_dataframe()
    
    # Select key columns
    summary_df = df[['Component', 'Material', 'TEC (1/K)', 'Youngs Modulus (GPa)', 
                     'Porosity', 'Electronic Cond. (S/m)', 'Ionic Cond. (S/m)']].copy()
    
    # Format for display
    summary_df['TEC (× 10⁻⁶ /K)'] = (summary_df['TEC (1/K)'] * 1e6).round(1)
    summary_df['E (GPa)'] = summary_df['Youngs Modulus (GPa)'].round(0)
    summary_df['Porosity (%)'] = (summary_df['Porosity'] * 100).round(0)
    summary_df['σ_e (S/m)'] = summary_df['Electronic Cond. (S/m)'].apply(lambda x: f'{x:.2e}')
    summary_df['σ_i (S/m)'] = summary_df['Ionic Cond. (S/m)'].apply(lambda x: f'{x:.2e}')
    
    final_df = summary_df[['Component', 'Material', 'TEC (× 10⁻⁶ /K)', 'E (GPa)', 
                           'Porosity (%)', 'σ_e (S/m)', 'σ_i (S/m)']]
    
    print("\n" + "="*90)
    print("SOFC MATERIAL PROPERTIES SUMMARY TABLE")
    print("="*90)
    print(final_df.to_string(index=False))
    print("="*90)
    
    return final_df


def main():
    """Generate all visualizations"""
    print("\n" + "="*70)
    print("GENERATING SOFC MATERIAL PROPERTY VISUALIZATIONS")
    print("="*70 + "\n")
    
    # Initialize database
    db = SOFCMaterialDatabase()
    
    # Create all plots
    print("Creating visualizations...")
    plot_tec_comparison(db)
    plot_conductivity_vs_temperature(db)
    plot_mechanical_properties(db)
    plot_creep_rates(db)
    plot_porosity_comparison(db)
    plot_electrochemical_properties(db)
    
    # Create summary table
    create_summary_table()
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  • tec_comparison.png")
    print("  • conductivity_vs_temperature.png")
    print("  • mechanical_properties.png")
    print("  • creep_rates.png")
    print("  • porosity_comparison.png")
    print("  • electrochemical_properties.png")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
