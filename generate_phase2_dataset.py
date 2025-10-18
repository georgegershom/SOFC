import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns


def main() -> None:
    np.random.seed(42)  # Reproducibility
    sns.set(style="whitegrid")

    # -----------------------------
    # Part A: Ambient Condition Tests
    # -----------------------------
    mix_ids = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']
    ages = [7, 28, 56]
    specimens_per_condition = 3

    ambient_data = []

    for mix in mix_ids:
        # Base properties decrease with rubber content
        base_fc_28 = 65 - (mix_ids.index(mix) * 8)  # Control=65 MPa, -8 MPa per step
        base_ft_28 = base_fc_28 * 0.10  # Tensile ~10% of compressive
        base_E_28 = 35000 - (mix_ids.index(mix) * 3000)  # MPa
        base_UPV = 4500 - (mix_ids.index(mix) * 150)  # m/s

        for age in ages:
            # Age factor: strength gain from 7->28->56 days
            if age == 7:
                age_factor_fc, age_factor_ft, age_factor_E = 0.75, 0.75, 0.85
            elif age == 28:
                age_factor_fc, age_factor_ft, age_factor_E = 1.00, 1.00, 1.00
            else:  # 56 days
                age_factor_fc, age_factor_ft, age_factor_E = 1.05, 1.03, 1.02

            for spec_num in range(1, specimens_per_condition + 1):
                specimen_id = f"{mix}-{age}-A-{spec_num}"
                # Introduce realistic variability (5-6% COV for strength)
                fc = (base_fc_28 * age_factor_fc) * np.random.normal(1.0, 0.05)
                ft = (base_ft_28 * age_factor_ft) * np.random.normal(1.0, 0.06)
                E = (base_E_28 * age_factor_E) * np.random.normal(1.0, 0.04)
                UPV = base_UPV * np.random.normal(1.0, 0.02)
                density = 2400 - (mix_ids.index(mix) * 40)  # kg/m³

                ambient_data.append({
                    'Specimen_ID': specimen_id,
                    'Mix_ID': mix,
                    'Curing_Age_days': age,
                    'Test_Type': 'Ambient',
                    'Compressive_Strength_MPa': max(0.0, fc),
                    'Tensile_Strength_MPa': max(0.0, ft),
                    'Modulus_of_Elasticity_MPa': max(0.0, E),
                    'Dry_Density_kgm3': density,
                    'UPV_mps': max(0.0, UPV)
                })

    df_ambient = pd.DataFrame(ambient_data)
    print("AMBIENT CONDITION DATASET (First 10 rows)")
    print(df_ambient.head(10).round(2))

    # ---------------------------------------------
    # Part B: High-Temperature Residual Properties
    # ---------------------------------------------
    peak_temps = [23, 200, 400, 600, 800]
    cooling_methods = ['Furnace', 'Quench']
    heating_rates = ['5_C_per_min']  # Default; rapid added for spalling study
    spalling_mixes = ['C', 'R15S', 'R20S']

    specimens_per_condition = 3
    residual_data = []

    for mix in mix_ids:
        # Base ambient 28-day strength for degradation reference
        ambient_fc = df_ambient[(df_ambient['Mix_ID'] == mix) &
                                (df_ambient['Curing_Age_days'] == 28) &
                                (df_ambient['Test_Type'] == 'Ambient')]['Compressive_Strength_MPa'].mean()

        for temp in peak_temps:
            for cooling in cooling_methods:
                hr_list = heating_rates.copy()
                if mix in spalling_mixes:
                    hr_list = ['5_C_per_min', '10_C_per_min']

                for hr in hr_list:
                    for spec_num in range(1, specimens_per_condition + 1):
                        specimen_id = f"{mix}-28-R-{temp}-{cooling}-{hr}-{spec_num}"

                        # --- Mass loss (dehydration + rubber burn-off) ---
                        base_mass_loss = 0.0
                        if temp >= 105:
                            base_mass_loss = (temp / 1000.0) * 8.0  # Dehydration-driven
                            if mix != 'C':
                                base_mass_loss += (mix_ids.index(mix) * 0.5) * (temp / 800.0)  # Rubber

                        # --- Strength retention (temperature- and mix-dependent) ---
                        if temp <= 200:
                            strength_retention = 1.0 + np.random.normal(0.0, 0.05)
                        elif temp <= 400:
                            if mix == 'C':
                                strength_retention = 0.75 - (temp - 200) * 0.002
                            else:
                                strength_retention = 0.65 - (temp - 200) * 0.0015
                        elif temp <= 600:
                            if mix == 'C':
                                strength_retention = 0.40 - (temp - 400) * 0.002
                            else:
                                strength_retention = 0.30 - (temp - 400) * 0.0015
                        else:  # 800°C
                            strength_retention = 0.10 + np.random.normal(0.0, 0.03)

                        # Cooling shock: quenching penalty
                        if cooling == 'Quench' and temp > 105:
                            strength_retention *= 0.85

                        # Rapid heating spalling penalty for plain concrete
                        if hr == '10_C_per_min' and mix == 'C' and temp >= 400:
                            strength_retention *= 0.70
                            base_mass_loss *= 1.30

                        # UPV retention correlates with damage (sublinear)
                        UPV_retention = max(0.0, strength_retention) ** 0.5

                        # --- Fabricate data points with noise ---
                        fc_residual = (ambient_fc * strength_retention) * np.random.normal(1.0, 0.08)
                        mass_loss_pct = base_mass_loss * np.random.normal(1.0, 0.10)
                        UPV_residual = (4500.0 * UPV_retention) * np.random.normal(1.0, 0.05)

                        # Spalling flags
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
                            'Visual_Cracking_Rating': np.random.choice(
                                ['None', 'Minor', 'Moderate', 'Severe'], p=[0.3, 0.4, 0.2, 0.1]
                            )
                        })

    df_residual = pd.DataFrame(residual_data)
    print("\nRESIDUAL PROPERTIES DATASET (First 15 rows)")
    print(df_residual.head(15).round(2))

    # -----------------------------
    # Part C: In-Situ Tests
    # -----------------------------
    key_mixes = ['C', 'R10S', 'R20S']  # Control, medium rubber, high rubber
    in_situ_temps = [23, 200, 400, 600]
    specimens_per_condition = 2

    in_situ_data = []
    stress_strain_curves: dict[str, dict[str, list[float]]] = {}

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

                # In-situ retention (distinct from residual)
                if temp <= 200:
                    strength_retention = 1.0
                    E_retention = 0.9
                elif temp <= 400:
                    strength_retention = 0.8 - (temp - 200) * 0.001
                    E_retention = 0.6
                else:  # 600°C
                    strength_retention = 0.4
                    E_retention = 0.2

                peak_strain_factor = 1.0
                if mix != 'C':
                    peak_strain_factor = 1.0 + (mix_ids.index(mix) * 0.2)

                fc_in_situ = ambient_fc * strength_retention * np.random.normal(1.0, 0.07)
                E_in_situ = ambient_E * E_retention * np.random.normal(1.0, 0.06)

                # Synthetic stress-strain curve
                strain_points = np.linspace(0.0, 0.025, 50)
                peak_strain = 0.002 * peak_strain_factor * (1.0 + 0.005 * temp)
                stress_points = []

                for strain in strain_points:
                    if strain <= peak_strain:
                        # Parabolic ascending branch
                        x = strain / peak_strain if peak_strain > 0 else 0.0
                        stress = fc_in_situ * (2.0 * x - x**2)
                    else:
                        # Linear descending branch (more gradual for rubber mixes)
                        descent_factor = 0.30 if mix == 'C' else 0.15
                        stress = fc_in_situ * max(0.0, 1.0 - descent_factor * (strain - peak_strain) / max(peak_strain, 1e-9))
                    stress_points.append(stress * np.random.normal(1.0, 0.02))

                stress_strain_curves[specimen_id] = {'strain': strain_points.tolist(), 'stress': stress_points}

                in_situ_data.append({
                    'Specimen_ID': specimen_id,
                    'Mix_ID': mix,
                    'Test_Temperature_C': temp,
                    'Test_Type': 'In-Situ',
                    'InSitu_Compressive_Strength_MPa': max(0.0, fc_in_situ),
                    'InSitu_Modulus_of_Elasticity_MPa': max(0.0, E_in_situ),
                    'Peak_Strain': peak_strain,
                    'Poissons_Ratio': max(0.1, 0.2 - (temp * 0.0002))
                })

    df_in_situ = pd.DataFrame(in_situ_data)
    print("\nIN-SITU PROPERTIES DATASET")
    print(df_in_situ.round(2))

    # Plot sample stress-strain curves
    plt.figure(figsize=(10, 6))
    for mix in key_mixes:
        spec_id = f"{mix}-28-IS-400-1"
        if spec_id in stress_strain_curves:
            plt.plot(np.array(stress_strain_curves[spec_id]['strain']) * 100.0,
                     stress_strain_curves[spec_id]['stress'],
                     label=f'{mix}, 400°C', linewidth=2)

    plt.xlabel('Strain (%)')
    plt.ylabel('Stress (MPa)')
    plt.title('In-Situ Compressive Stress-Strain Behavior at High Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('in_situ_stress_strain_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --------------------------------------------------
    # Part D: Advanced Measurements - Pore Pressure
    # --------------------------------------------------
    pore_pressure_data = []
    times = np.linspace(0.0, 120.0, 121)  # minutes

    for mix in ['C', 'R20S']:
        for depth in [10, 25, 40]:  # mm
            for run in [1, 2]:
                pressures = []
                for t in times:
                    temp_at_t = min(800.0, t * 10.0)
                    if mix == 'C':
                        base_pressure = 0.5 * np.exp(-((t - 25.0) / 15.0) ** 2) * (depth / 40.0)  # Gaussian peak
                        if temp_at_t > 300.0:
                            base_pressure *= 0.5  # Pressure release after dehydration
                    else:
                        base_pressure = 0.3 * np.exp(-((t - 30.0) / 25.0) ** 2) * (depth / 40.0)

                    pressure = base_pressure + np.random.normal(0.0, 0.02)
                    pressures.append(max(0.0, pressure))

                pore_pressure_data.append({
                    'Mix_ID': mix,
                    'Depth_mm': depth,
                    'Run': run,
                    'Time_min': times.tolist(),
                    'Pore_Pressure_MPa': pressures
                })

    # Peak pressure summary
    peak_pressures = []
    for pp in pore_pressure_data:
        peak_index = int(np.argmax(pp['Pore_Pressure_MPa']))
        peak_pressures.append({
            'Mix_ID': pp['Mix_ID'],
            'Depth_mm': pp['Depth_mm'],
            'Run': pp['Run'],
            'Peak_Pressure_MPa': float(np.max(pp['Pore_Pressure_MPa'])),
            'Time_of_Peak_min': float(pp['Time_min'][peak_index])
        })

    df_pore_pressure = pd.DataFrame(peak_pressures)
    print("\nPEAK PORE PRESSURE SUMMARY")
    print(df_pore_pressure.round(3))

    # Pore pressure evolution plot (first run per series)
    plt.figure(figsize=(12, 6))
    for mix in ['C', 'R20S']:
        for depth in [10, 25, 40]:
            data = next(pp for pp in pore_pressure_data if pp['Mix_ID'] == mix and pp['Depth_mm'] == depth and pp['Run'] == 1)
            plt.plot(data['Time_min'], data['Pore_Pressure_MPa'], label=f'{mix}, {depth}mm', linewidth=2)

    plt.xlabel('Time (min)')
    plt.ylabel('Pore Pressure (MPa)')
    plt.title('Pore Pressure Evolution During Heating\n(10°C/min heating rate)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pore_pressure_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # -----------------------------
    # Exports
    # -----------------------------
    df_ambient.to_csv('ambient_properties.csv', index=False)
    df_residual.to_csv('residual_properties_high_temp.csv', index=False)
    df_in_situ.to_csv('in_situ_properties.csv', index=False)
    df_pore_pressure.to_csv('pore_pressure_summary.csv', index=False)

    # Save stress-strain curves JSON
    with open('stress_strain_curves.json', 'w') as f:
        json.dump(stress_strain_curves, f, indent=2)

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"Ambient tests: {len(df_ambient)} specimens")
    print(f"Residual high-temperature tests: {len(df_residual)} specimens")
    print(f"In-situ high-temperature tests: {len(df_in_situ)} specimens")
    print(f"Pore pressure experiments: {len(df_pore_pressure)} configurations")

    # -----------------------------
    # Master summary figure
    # -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Residual strength vs temperature (Furnace cooled, 5C/min)
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

    # Plot 2: Effect of cooling method at 400C
    temp_plot = 400
    labels = []
    values = []
    colors = []
    for mix in ['C', 'R20S']:
        for cooling in ['Furnace', 'Quench']:
            cool_data = df_residual[(df_residual['Mix_ID'] == mix) &
                                    (df_residual['Peak_Temperature_C'] == temp_plot) &
                                    (df_residual['Heating_Rate'] == '5_C_per_min') &
                                    (df_residual['Cooling_Method'] == cooling)]
            strength = cool_data['Residual_Compressive_Strength_MPa'].mean()
            labels.append(f"{mix}\n{cooling}")
            values.append(strength)
            colors.append('red' if cooling == 'Quench' else 'blue')
    axes[0, 1].bar(labels, values, alpha=0.7, color=colors)
    axes[0, 1].set_ylabel('Residual Compressive Strength (MPa)')
    axes[0, 1].set_title(f'B. Effect of Cooling Method at {temp_plot}°C\n(Thermal Shock Damage)')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Mass loss correlation
    for mix in ['C', 'R10S', 'R20S']:
        mix_data = df_residual[(df_residual['Mix_ID'] == mix) &
                               (df_residual['Cooling_Method'] == 'Furnace') &
                               (df_residual['Heating_Rate'] == '5_C_per_min')]
        axes[1, 0].scatter(mix_data['Mass_Loss_pct'], mix_data['Residual_Compressive_Strength_MPa'], label=mix, alpha=0.6, s=50)

    axes[1, 0].set_xlabel('Mass Loss (%)')
    axes[1, 0].set_ylabel('Residual Compressive Strength (MPa)')
    axes[1, 0].set_title('C. Strength vs. Mass Loss Correlation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Spalling risk summary (10C/min)
    spalling_summary = df_residual[df_residual['Heating_Rate'] == '10_C_per_min']
    if not spalling_summary.empty:
        spalling_rates = spalling_summary.groupby(['Mix_ID', 'Peak_Temperature_C'])['Spalling_Occurred'].mean().reset_index()
        pivot_spalling = spalling_rates.pivot(index='Peak_Temperature_C', columns='Mix_ID', values='Spalling_Occurred')
        pivot_spalling.plot(kind='bar', ax=axes[1, 1], width=0.8)
        axes[1, 1].set_xlabel('Peak Temperature (°C)')
        axes[1, 1].set_ylabel('Probability of Spalling')
        axes[1, 1].set_title('D. Spalling Risk: Rapid Heating (10°C/min)')
        axes[1, 1].legend(title='Mix ID')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No rapid-heating configurations', ha='center', va='center')
        axes[1, 1].set_axis_off()

    plt.tight_layout()
    plt.savefig('comprehensive_high_temperature_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nAll datasets and figures have been generated and saved.")
    print("This comprehensive dataset is now ready for thermo-mechanical model development and validation.")


if __name__ == "__main__":
    main()
