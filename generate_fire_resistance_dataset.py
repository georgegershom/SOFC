import os
import json
import numpy as np
import pandas as pd

# Use a non-interactive backend for headless environments before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def generate_ambient_dataset(random_seed: int = 42) -> pd.DataFrame:
    """
    Generate ambient condition dataset for 6 mixes at 3 curing ages with 3 specimens per condition.
    Returns a DataFrame with compressive/tensile strengths, modulus, density, and UPV.
    """
    np.random.seed(random_seed)

    mix_ids = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']
    ages = [7, 28, 56]
    specimens_per_condition = 3

    ambient_data = []

    for mix in mix_ids:
        # Base properties decrease with rubber content/size according to index
        base_fc_28 = 65 - (mix_ids.index(mix) * 8)  # MPa
        base_ft_28 = base_fc_28 * 0.10  # ~10% of compressive
        base_E_28 = 35000 - (mix_ids.index(mix) * 3000)  # MPa
        base_UPV = 4500 - (mix_ids.index(mix) * 150)  # m/s

        for age in ages:
            # Age factors for strength and stiffness gain
            if age == 7:
                age_factor_fc, age_factor_ft, age_factor_E = 0.75, 0.75, 0.85
            elif age == 28:
                age_factor_fc, age_factor_ft, age_factor_E = 1.0, 1.0, 1.0
            else:  # 56 days
                age_factor_fc, age_factor_ft, age_factor_E = 1.05, 1.03, 1.02

            for spec_num in range(1, specimens_per_condition + 1):
                specimen_id = f"{mix}-{age}-A-{spec_num}"

                # Introduce realistic variability
                fc = (base_fc_28 * age_factor_fc) * np.random.normal(1.0, 0.05)
                ft = (base_ft_28 * age_factor_ft) * np.random.normal(1.0, 0.06)
                E = (base_E_28 * age_factor_E) * np.random.normal(1.0, 0.04)
                UPV = base_UPV * np.random.normal(1.0, 0.02)
                density = 2400 - (mix_ids.index(mix) * 40)  # kg/m3

                ambient_data.append({
                    'Specimen_ID': specimen_id,
                    'Mix_ID': mix,
                    'Curing_Age_days': age,
                    'Test_Type': 'Ambient',
                    'Compressive_Strength_MPa': max(0.0, fc),
                    'Tensile_Strength_MPa': max(0.0, ft),
                    'Modulus_of_Elasticity_MPa': max(0.0, E),
                    'Dry_Density_kgm3': density,
                    'UPV_mps': UPV
                })

    df_ambient = pd.DataFrame(ambient_data)
    return df_ambient


def generate_residual_dataset(df_ambient: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate residual property dataset across peak temperatures, cooling methods, and heating rates.
    Includes spalling indicators and UPV correlation with damage.
    """
    np.random.seed(random_seed + 1)

    mix_ids = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']
    peak_temps = [23, 200, 400, 600, 800]
    cooling_methods = ['Furnace', 'Quench']
    heating_rates = ['5_C_per_min']  # standard heating for all mixes

    # Add rapid heating for selected mixes to study spalling
    spalling_mixes = ['C', 'R15S', 'R20S']
    specimens_per_condition = 3

    residual_data = []

    for mix in mix_ids:
        ambient_fc = df_ambient[(df_ambient['Mix_ID'] == mix) &
                                (df_ambient['Curing_Age_days'] == 28) &
                                (df_ambient['Test_Type'] == 'Ambient')]['Compressive_Strength_MPa'].mean()

        for temp in peak_temps:
            for cooling in cooling_methods:
                hr_list = list(heating_rates)
                if mix in spalling_mixes:
                    hr_list = ['5_C_per_min', '10_C_per_min']

                for hr in hr_list:
                    for spec_num in range(1, specimens_per_condition + 1):
                        specimen_id = f"{mix}-28-R-{temp}-{cooling}-{hr}-{spec_num}"

                        # Mass Loss: increases with temperature, more for rubber mixes
                        base_mass_loss = 0.0
                        if temp >= 105:
                            base_mass_loss = (temp / 1000.0) * 8.0  # dehydration baseline
                            if mix != 'C':
                                base_mass_loss += (mix_ids.index(mix) * 0.5) * (temp / 800.0)  # rubber burnout

                        # Strength retention as function of T and mix
                        if temp <= 200:
                            strength_retention = 1.0 + np.random.normal(0.0, 0.05)
                        elif temp <= 400:
                            if mix == 'C':
                                strength_retention = 0.75 - (temp - 200.0) * 0.002
                            else:
                                strength_retention = 0.65 - (temp - 200.0) * 0.0015
                        elif temp <= 600:
                            if mix == 'C':
                                strength_retention = 0.40 - (temp - 400.0) * 0.002
                            else:
                                strength_retention = 0.30 - (temp - 400.0) * 0.0015
                        else:  # 800 C
                            strength_retention = 0.10 + np.random.normal(0.0, 0.03)

                        # Thermal shock due to quenching
                        if cooling == 'Quench' and temp > 105:
                            strength_retention *= 0.85

                        # Spalling risk for control under rapid heating
                        if hr == '10_C_per_min' and mix == 'C' and temp >= 400:
                            strength_retention *= 0.70
                            base_mass_loss *= 1.30

                        # UPV correlates with damage
                        UPV_retention = max(0.0, strength_retention) ** 0.5

                        # Fabricate data with noise
                        fc_residual = (ambient_fc * max(0.0, strength_retention)) * np.random.normal(1.0, 0.08)
                        mass_loss_pct = max(0.0, base_mass_loss) * np.random.normal(1.0, 0.10)
                        UPV_residual = (4500.0 * UPV_retention) * np.random.normal(1.0, 0.05)

                        # Spalling flag
                        spalling_occurred = False
                        spalling_depth_mm = 0.0
                        if mix == 'C' and temp >= 400 and hr == '10_C_per_min':
                            spalling_occurred = True
                            spalling_depth_mm = float(np.random.uniform(5.0, 25.0))
                        elif mix in ['R15S', 'R20S'] and temp >= 400 and hr == '10_C_per_min':
                            if np.random.random() > 0.7:  # 30% chance
                                spalling_occurred = True
                                spalling_depth_mm = float(np.random.uniform(2.0, 10.0))

                        residual_data.append({
                            'Specimen_ID': specimen_id,
                            'Mix_ID': mix,
                            'Peak_Temperature_C': temp,
                            'Heating_Rate': hr,
                            'Cooling_Method': cooling,
                            'Test_Type': 'Residual',
                            'Mass_Loss_pct': max(0.0, mass_loss_pct),
                            'UPV_mps': max(500.0, UPV_residual),
                            'Residual_Compressive_Strength_MPa': max(0.0, fc_residual),
                            'Spalling_Occurred': spalling_occurred,
                            'Spalling_Depth_mm': spalling_depth_mm,
                            'Visual_Cracking_Rating': np.random.choice(['None', 'Minor', 'Moderate', 'Severe'],
                                                                      p=[0.3, 0.4, 0.2, 0.1])
                        })

    df_residual = pd.DataFrame(residual_data)
    return df_residual


def generate_in_situ_dataset(df_ambient: pd.DataFrame, random_seed: int = 42):
    """
    Generate in-situ dataset for key mixes and temperatures, including synthetic
    stress-strain curves stored in a dict keyed by specimen_id.
    Returns (df_in_situ, stress_strain_curves_dict)
    """
    np.random.seed(random_seed + 2)

    key_mixes = ['C', 'R10S', 'R20S']
    in_situ_temps = [23, 200, 400, 600]
    specimens_per_condition = 2

    in_situ_data = []
    stress_strain_curves = {}

    for mix in key_mixes:
        ambient_fc = df_ambient[(df_ambient['Mix_ID'] == mix) &
                                (df_ambient['Curing_Age_days'] == 28) &
                                (df_ambient['Test_Type'] == 'Ambient')]['Compressive_Strength_MPa'].mean()
        ambient_E = df_ambient[(df_ambient['Mix_ID'] == mix) &
                               (df_ambient['Curing_Age_days'] == 28) &
                               (df_ambient['Test_Type'] == 'Ambient')]['Modulus_of_Elasticity_MPa'].mean()

        for temp in in_situ_temps:
            for spec_num in range(1, specimens_per_condition + 1):
                specimen_id = f"{mix}-28-IS-{temp}-{spec_num}"

                # In-situ retention factors
                if temp <= 200:
                    strength_retention = 1.0
                    E_retention = 0.9
                elif temp <= 400:
                    strength_retention = 0.8 - (temp - 200.0) * 0.001
                    E_retention = 0.6
                else:  # 600 C
                    strength_retention = 0.4
                    E_retention = 0.2

                # Rubber mixes show more ductility
                mix_ids = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']
                peak_strain_factor = 1.0 if mix == 'C' else 1.0 + (mix_ids.index(mix) * 0.2)

                fc_in_situ = ambient_fc * strength_retention * np.random.normal(1.0, 0.07)
                E_in_situ = ambient_E * E_retention * np.random.normal(1.0, 0.06)

                # Generate stress-strain curve
                strain_points = np.linspace(0.0, 0.025, 50)
                peak_strain = 0.002 * peak_strain_factor * (1.0 + 0.005 * temp)
                stress_points = []
                for strain in strain_points:
                    if strain <= peak_strain:
                        # Parabolic ascending branch
                        ratio = strain / max(1e-9, peak_strain)
                        stress = fc_in_situ * (2.0 * ratio - ratio ** 2)
                    else:
                        # Linear descending branch; more gradual for rubber mixes
                        descent_factor = 0.30 if mix == 'C' else 0.15
                        stress = fc_in_situ * max(0.0, 1.0 - descent_factor * (strain - peak_strain) / max(1e-9, peak_strain))
                    stress_points.append(stress * np.random.normal(1.0, 0.02))

                stress_strain_curves[specimen_id] = {
                    'strain': strain_points.tolist(),
                    'stress': stress_points,
                }

                in_situ_data.append({
                    'Specimen_ID': specimen_id,
                    'Mix_ID': mix,
                    'Test_Temperature_C': temp,
                    'Test_Type': 'In-Situ',
                    'InSitu_Compressive_Strength_MPa': max(0.0, fc_in_situ),
                    'InSitu_Modulus_of_Elasticity_MPa': max(0.0, E_in_situ),
                    'Peak_Strain': peak_strain,
                    'Poissons_Ratio': max(0.1, 0.2 - (temp * 0.0002)),
                })

    df_in_situ = pd.DataFrame(in_situ_data)
    return df_in_situ, stress_strain_curves


def generate_pore_pressure_dataset(random_seed: int = 42):
    """
    Generate pore pressure synthetic time series and a summary DataFrame of peak pressures.
    Returns (pore_pressure_time_series_list, df_peak_pressures)
    """
    np.random.seed(random_seed + 3)

    pore_pressure_data = []
    times = np.linspace(0, 120, 121)  # minutes

    for mix in ['C', 'R20S']:
        for depth in [10, 25, 40]:  # mm from exposed surface
            for run in [1, 2]:
                pressures = []
                for t in times:
                    temp_at_t = min(800.0, t * 10.0)
                    if mix == 'C':
                        # Control: high pore pressure build-up, sharp peak around ~250 C (~25 min)
                        base_pressure = 0.5 * np.exp(-((t - 25.0) / 15.0) ** 2) * (depth / 40.0)
                        if temp_at_t > 300.0:
                            base_pressure *= 0.5
                    else:
                        # Rubber concrete: lower, broader peak due to melting
                        base_pressure = 0.3 * np.exp(-((t - 30.0) / 25.0) ** 2) * (depth / 40.0)

                    pressure = base_pressure + np.random.normal(0.0, 0.02)
                    pressures.append(max(0.0, pressure))

                pore_pressure_data.append({
                    'Mix_ID': mix,
                    'Depth_mm': depth,
                    'Run': run,
                    'Time_min': times.tolist(),
                    'Pore_Pressure_MPa': pressures,
                })

    # Summary of peak pressures
    peak_pressures = []
    for pp in pore_pressure_data:
        pp_array = np.array(pp['Pore_Pressure_MPa'])
        t_array = np.array(pp['Time_min'])
        peak_idx = int(np.argmax(pp_array))
        peak_pressures.append({
            'Mix_ID': pp['Mix_ID'],
            'Depth_mm': pp['Depth_mm'],
            'Run': pp['Run'],
            'Peak_Pressure_MPa': float(np.max(pp_array)),
            'Time_of_Peak_min': float(t_array[peak_idx]),
        })

    df_pore_pressure = pd.DataFrame(peak_pressures)
    return pore_pressure_data, df_pore_pressure


def save_datasets(df_ambient: pd.DataFrame,
                   df_residual: pd.DataFrame,
                   df_in_situ: pd.DataFrame,
                   stress_strain_curves: dict,
                   df_pore_pressure: pd.DataFrame,
                   output_dir: str = '.') -> None:
    """
    Save datasets to CSV/JSON in the specified directory.
    """
    ambient_path = os.path.join(output_dir, 'ambient_properties.csv')
    residual_path = os.path.join(output_dir, 'residual_properties_high_temp.csv')
    in_situ_path = os.path.join(output_dir, 'in_situ_properties.csv')
    pore_pressure_path = os.path.join(output_dir, 'pore_pressure_summary.csv')
    ss_json_path = os.path.join(output_dir, 'stress_strain_curves.json')

    df_ambient.to_csv(ambient_path, index=False)
    df_residual.to_csv(residual_path, index=False)
    df_in_situ.to_csv(in_situ_path, index=False)
    df_pore_pressure.to_csv(pore_pressure_path, index=False)

    ss_curves_serializable = {k: {'strain': v['strain'], 'stress': v['stress']} for k, v in stress_strain_curves.items()}
    with open(ss_json_path, 'w') as f:
        json.dump(ss_curves_serializable, f, indent=2)


def plot_in_situ_curves(stress_strain_curves: dict, output_dir: str = '.') -> None:
    plt.figure(figsize=(10, 6))
    for mix in ['C', 'R10S', 'R20S']:
        spec_id = f"{mix}-28-IS-400-1"
        if spec_id in stress_strain_curves:
            data = stress_strain_curves[spec_id]
            plt.plot(np.array(data['strain']) * 100.0, data['stress'], label=f'{mix}, 400°C', linewidth=2)

    plt.xlabel('Strain (%)')
    plt.ylabel('Stress (MPa)')
    plt.title('In-Situ Compressive Stress-Strain Behavior at High Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'in_situ_stress_strain_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_pore_pressure(pore_pressure_data: list, output_dir: str = '.') -> None:
    plt.figure(figsize=(12, 6))
    for mix in ['C', 'R20S']:
        for depth in [10, 25, 40]:
            # Get the first run for plotting
            data = next(pp for pp in pore_pressure_data if pp['Mix_ID'] == mix and pp['Depth_mm'] == depth and pp['Run'] == 1)
            plt.plot(data['Time_min'], data['Pore_Pressure_MPa'], label=f'{mix}, {depth}mm', linewidth=2)

    plt.xlabel('Time (min)')
    plt.ylabel('Pore Pressure (MPa)')
    plt.title('Pore Pressure Evolution During Heating\n(10°C/min heating rate)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pore_pressure_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_master_summary(df_residual: pd.DataFrame, output_dir: str = '.') -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Residual strength vs temperature (Furnace cooled, standard heating)
    for mix in ['C', 'R10S', 'R20S']:
        mix_data = df_residual[(df_residual['Mix_ID'] == mix) &
                               (df_residual['Cooling_Method'] == 'Furnace') &
                               (df_residual['Heating_Rate'] == '5_C_per_min')]
        strength_by_temp = mix_data.groupby('Peak_Temperature_C')['Residual_Compressive_Strength_MPa'].mean()
        axes[0, 0].plot(strength_by_temp.index, strength_by_temp.values, 'o-', label=mix, linewidth=2, markersize=6)

    axes[0, 0].set_xlabel('Peak Temperature (°C)')
    axes[0, 0].set_ylabel('Residual Compressive Strength (MPa)')
    axes[0, 0].set_title('A. Strength Degradation with Temperature\n(Furnace Cooled)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Effect of cooling method at 400 C
    temp = 400
    for mix in ['C', 'R20S']:
        for cooling in ['Furnace', 'Quench']:
            cool_data = df_residual[(df_residual['Mix_ID'] == mix) &
                                    (df_residual['Peak_Temperature_C'] == temp) &
                                    (df_residual['Heating_Rate'] == '5_C_per_min') &
                                    (df_residual['Cooling_Method'] == cooling)]
            strength = cool_data['Residual_Compressive_Strength_MPa'].mean()
            axes[0, 1].bar(f"{mix}\n{cooling}", strength, alpha=0.7,
                           color='red' if cooling == 'Quench' else 'blue')

    axes[0, 1].set_ylabel('Residual Compressive Strength (MPa)')
    axes[0, 1].set_title(f'B. Effect of Cooling Method at {temp}°C\n(Thermal Shock Damage)')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Mass loss correlation
    for mix in ['C', 'R10S', 'R20S']:
        mix_data = df_residual[(df_residual['Mix_ID'] == mix) &
                               (df_residual['Cooling_Method'] == 'Furnace') &
                               (df_residual['Heating_Rate'] == '5_C_per_min')]
        axes[1, 0].scatter(mix_data['Mass_Loss_pct'], mix_data['Residual_Compressive_Strength_MPa'],
                           label=mix, alpha=0.6, s=50)

    axes[1, 0].set_xlabel('Mass Loss (%)')
    axes[1, 0].set_ylabel('Residual Compressive Strength (MPa)')
    axes[1, 0].set_title('C. Strength vs. Mass Loss Correlation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Spalling risk summary (rapid heating)
    spalling_summary = df_residual[df_residual['Heating_Rate'] == '10_C_per_min']
    if not spalling_summary.empty:
        spalling_rates = spalling_summary.groupby(['Mix_ID', 'Peak_Temperature_C'])['Spalling_Occurred'].mean().reset_index()
        pivot_spalling = spalling_rates.pivot(index='Peak_Temperature_C', columns='Mix_ID', values='Spalling_Occurred')
        pivot_spalling.plot(kind='bar', ax=axes[1, 1], width=0.8)
        axes[1, 1].legend(title='Mix ID')
    else:
        axes[1, 1].text(0.5, 0.5, 'No rapid heating data', ha='center', va='center')

    axes[1, 1].set_xlabel('Peak Temperature (°C)')
    axes[1, 1].set_ylabel('Probability of Spalling')
    axes[1, 1].set_title('D. Spalling Risk: Rapid Heating (10°C/min)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_high_temperature_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main() -> None:
    # Create output in current working directory
    out_dir = '.'

    # Part A: Ambient data
    df_ambient = generate_ambient_dataset()
    print('AMBIENT CONDITION DATASET (First 10 rows)')
    print(df_ambient.head(10).round(2))

    # Part B: Residual properties after high-temperature exposure
    df_residual = generate_residual_dataset(df_ambient)
    print('\nRESIDUAL PROPERTIES DATASET (First 15 rows)')
    print(df_residual.head(15).round(2))

    # Part C: In-situ high-temperature tests
    df_in_situ, stress_strain_curves = generate_in_situ_dataset(df_ambient)
    print('\nIN-SITU PROPERTIES DATASET (First 10 rows)')
    print(df_in_situ.head(10).round(2))

    # Part D: Advanced measurements — pore pressure (time series with summary)
    pore_pressure_time_series, df_pore_pressure = generate_pore_pressure_dataset()
    print('\nPEAK PORE PRESSURE SUMMARY (First 10 rows)')
    print(df_pore_pressure.head(10).round(3))

    # Save all datasets
    save_datasets(df_ambient, df_residual, df_in_situ, stress_strain_curves, df_pore_pressure, out_dir)

    # Figures
    plot_in_situ_curves(stress_strain_curves, out_dir)
    plot_pore_pressure(pore_pressure_time_series, out_dir)
    plot_master_summary(df_residual, out_dir)

    # Final summary
    print('\n' + '=' * 70)
    print('DATASET GENERATION COMPLETE')
    print('=' * 70)
    print(f'Ambient tests: {len(df_ambient)} specimens')
    print(f'Residual high-temperature tests: {len(df_residual)} specimens')
    print(f'In-situ high-temperature tests: {len(df_in_situ)} specimens')
    print(f'Pore pressure experiments: {len(df_pore_pressure)} configurations')
    print('\nAll datasets and figures have been generated and saved:')
    print(' - ambient_properties.csv')
    print(' - residual_properties_high_temp.csv')
    print(' - in_situ_properties.csv')
    print(' - stress_strain_curves.json')
    print(' - pore_pressure_summary.csv')
    print(' - in_situ_stress_strain_curves.png')
    print(' - pore_pressure_evolution.png')
    print(' - comprehensive_high_temperature_analysis.png')


if __name__ == '__main__':
    main()
