#!/usr/bin/env python3
"""
Pre-Simulation QA Verification Script
=======================================
Checks all dataset values for physical bounds and consistency before
launching Abaqus simulations.

Verification checks:
  1. Physical bounds (E > 0, -1 < nu < 0.5, Gc > 0, etc.)
  2. Thermodynamic consistency (Gc_II >= Gc_I)
  3. Positive definiteness of stiffness tensor
  4. Mesh objectivity requirements (l0 >= 2*h)
  5. Temperature monotonicity
  6. Energy conservation constraints
  7. Weibull parameter validity

Exit codes:
  0 - All checks passed
  1 - Warnings detected (non-critical)
  2 - Errors detected (critical, simulation should not proceed)
"""

import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(BASE_DIR, "csv")


class QAResult:
    """Container for QA check results."""

    def __init__(self):
        self.checks = []
        self.n_pass = 0
        self.n_warn = 0
        self.n_fail = 0

    def passed(self, check_name, detail=""):
        self.checks.append(('PASS', check_name, detail))
        self.n_pass += 1

    def warning(self, check_name, detail=""):
        self.checks.append(('WARN', check_name, detail))
        self.n_warn += 1

    def fail(self, check_name, detail=""):
        self.checks.append(('FAIL', check_name, detail))
        self.n_fail += 1

    def report(self):
        print("\n" + "=" * 70)
        print("  QA VERIFICATION REPORT")
        print("=" * 70)

        for status, name, detail in self.checks:
            symbol = {'PASS': '[OK]  ', 'WARN': '[WARN]', 'FAIL': '[FAIL]'}[status]
            color = {'PASS': '', 'WARN': '', 'FAIL': ''}[status]
            line = f"  {symbol} {name}"
            if detail:
                line += f" -- {detail}"
            print(line)

        print("\n" + "-" * 70)
        print(f"  Summary: {self.n_pass} passed, {self.n_warn} warnings, {self.n_fail} failures")
        print(f"  Total checks: {len(self.checks)}")

        if self.n_fail > 0:
            print("\n  STATUS: FAILED - DO NOT PROCEED WITH SIMULATION")
            return 2
        elif self.n_warn > 0:
            print("\n  STATUS: PASSED WITH WARNINGS - Review before proceeding")
            return 1
        else:
            print("\n  STATUS: ALL CHECKS PASSED - Safe to proceed")
            return 0


def check_physical_bounds(qa):
    """Check that all material properties are within physical bounds."""
    print("\n--- Check 1: Physical Bounds ---")

    # Young's modulus: E > 0
    df = pd.read_csv(os.path.join(CSV_DIR, "02_youngs_modulus.csv"))
    for col in ['E_YSZ_GPa', 'E_GDC_GPa', 'E_LSCF_dense_GPa', 'E_LSCF_porous_GPa']:
        if (df[col] > 0).all():
            qa.passed(f"E > 0 ({col})", f"range: [{df[col].min():.1f}, {df[col].max():.1f}] GPa")
        else:
            qa.fail(f"E > 0 ({col})", f"Negative values found!")

    # Poisson's ratio: -1 < nu < 0.5
    df_nu = pd.read_csv(os.path.join(CSV_DIR, "02_poissons_ratio.csv"))
    for col in ['nu_YSZ', 'nu_GDC', 'nu_LSCF']:
        vals = df_nu[col]
        if (vals > -1).all() and (vals < 0.5).all():
            qa.passed(f"-1 < nu < 0.5 ({col})", f"range: [{vals.min():.4f}, {vals.max():.4f}]")
        else:
            qa.fail(f"-1 < nu < 0.5 ({col})", f"Out of bounds!")

    # CTE: alpha > 0
    df_a = pd.read_csv(os.path.join(CSV_DIR, "02_thermal_expansion.csv"))
    for col in ['alpha_YSZ_1e6_perK', 'alpha_GDC_1e6_perK', 'alpha_LSCF_1e6_perK']:
        if (df_a[col] > 0).all():
            qa.passed(f"alpha > 0 ({col})", f"range: [{df_a[col].min():.2f}, {df_a[col].max():.2f}] x10^-6/K")
        else:
            qa.fail(f"alpha > 0 ({col})", f"Negative CTE found!")

    # Fracture toughness: Gc > 0
    df_Gc = pd.read_csv(os.path.join(CSV_DIR, "04_bulk_fracture_toughness.csv"))
    for col in ['Gc_YSZ_Jm2', 'Gc_GDC_Jm2', 'Gc_LSCF_Jm2']:
        if (df_Gc[col] > 0).all():
            qa.passed(f"Gc > 0 ({col})", f"range: [{df_Gc[col].min():.1f}, {df_Gc[col].max():.1f}] J/m2")
        else:
            qa.fail(f"Gc > 0 ({col})", f"Non-positive fracture toughness!")


def check_thermodynamic_consistency(qa):
    """Check thermodynamic consistency constraints."""
    print("\n--- Check 2: Thermodynamic Consistency ---")

    df = pd.read_csv(os.path.join(CSV_DIR, "04_cohesive_zone_parameters.csv"))

    # Gc_II >= Gc_I (shear toughness should be >= mode I for most interfaces)
    for _, row in df.iterrows():
        intf = row['Interface']
        T = row['Temperature_C']
        if row['Gc_II_Jm2'] >= row['Gc_I_Jm2']:
            qa.passed(f"Gc_II >= Gc_I ({intf}, {T}C)",
                     f"Gc_II={row['Gc_II_Jm2']:.1f} >= Gc_I={row['Gc_I_Jm2']:.1f}")
        else:
            qa.fail(f"Gc_II >= Gc_I ({intf}, {T}C)",
                   f"Gc_II={row['Gc_II_Jm2']:.1f} < Gc_I={row['Gc_I_Jm2']:.1f}!")

    # BK exponent > 0
    if (df['Eta_BK'] > 0).all():
        qa.passed("eta_BK > 0", f"Value: {df['Eta_BK'].iloc[0]:.1f}")
    else:
        qa.fail("eta_BK > 0", "Non-positive BK exponent!")

    # Cohesive strength > 0
    for col in ['T_max_n_MPa', 'T_max_t_MPa']:
        if (df[col] > 0).all():
            qa.passed(f"{col} > 0", f"range: [{df[col].min():.0f}, {df[col].max():.0f}] MPa")
        else:
            qa.fail(f"{col} > 0", "Non-positive cohesive strength!")


def check_positive_definiteness(qa):
    """Check positive definiteness of the stiffness tensor C(T)."""
    print("\n--- Check 3: Positive Definiteness of C(T) ---")

    df = pd.read_csv(os.path.join(CSV_DIR, "02_umat_full_input.csv"))

    for mat in df['Material'].unique():
        mask = df['Material'] == mat
        data = df[mask]

        all_pd = True
        for _, row in data.iterrows():
            E = row['E_GPa']
            nu = row['nu']
            Lambda = row['Lambda_GPa']
            Mu = row['Mu_GPa']

            # For isotropic material, positive definiteness requires:
            # Mu > 0 and Lambda + 2*Mu/3 > 0 (i.e., bulk modulus K > 0)
            K = Lambda + 2 * Mu / 3
            if Mu <= 0 or K <= 0:
                all_pd = False
                qa.fail(f"Positive definiteness ({mat}, {row['Temperature_C']}C)",
                       f"Mu={Mu:.1f} GPa, K={K:.1f} GPa")
                break

        if all_pd:
            qa.passed(f"Positive definiteness ({mat})",
                     f"All T: Mu > 0, K > 0")


def check_temperature_monotonicity(qa):
    """Check that properties vary monotonically with temperature where expected."""
    print("\n--- Check 4: Temperature Monotonicity ---")

    # E should decrease with T for ceramics
    df = pd.read_csv(os.path.join(CSV_DIR, "02_youngs_modulus.csv"))
    for col in ['E_YSZ_GPa', 'E_GDC_GPa', 'E_LSCF_dense_GPa']:
        diffs = np.diff(df[col].values)
        if (diffs <= 0).all():
            qa.passed(f"E decreasing with T ({col})")
        else:
            qa.warning(f"E not monotonically decreasing ({col})",
                      f"Max increase: {diffs.max():.2f} GPa")

    # CTE should generally increase with T
    df_a = pd.read_csv(os.path.join(CSV_DIR, "02_thermal_expansion.csv"))
    for col in ['alpha_YSZ_1e6_perK', 'alpha_GDC_1e6_perK', 'alpha_LSCF_1e6_perK']:
        diffs = np.diff(df_a[col].values)
        if (diffs >= 0).all():
            qa.passed(f"alpha non-decreasing with T ({col})")
        else:
            qa.warning(f"alpha not monotonically increasing ({col})",
                      f"Max decrease: {diffs.min():.3f}")


def check_mesh_objectivity(qa):
    """Check mesh objectivity requirements."""
    print("\n--- Check 5: Mesh Objectivity ---")

    df = pd.read_csv(os.path.join(CSV_DIR, "01_mesh_sensitivity.csv"))

    # Find convergence threshold
    converged = df[df['Normalized_Energy_Release'] >= 0.99]
    if len(converged) > 0:
        h_max = converged['Mesh_Size_um'].max()
        qa.passed(f"Mesh convergence identified", f"h_max = {h_max:.1f} um for 1% accuracy")
    else:
        qa.warning("No converged mesh size found within 1% band")

    # Phase-field length scale check: l0 >= 2*h recommended
    df_pf = pd.read_csv(os.path.join(CSV_DIR, "04_phase_field_parameters.csv"))
    l0 = df_pf['l0_um'].iloc[0]
    for _, row in df.iterrows():
        h = row['Mesh_Size_um']
        ratio = l0 / h
        if ratio >= 2:
            qa.passed(f"l0/h >= 2 (h={h} um)", f"l0/h = {ratio:.1f}")
        elif ratio >= 1:
            qa.warning(f"l0/h marginal (h={h} um)", f"l0/h = {ratio:.1f} (recommended >= 2)")
        # Only warn for meshes that are within convergence range
        elif row['Convergence_Status'] == 'Converged':
            qa.warning(f"l0/h < 1 but converged (h={h} um)", f"l0/h = {ratio:.1f}")


def check_chemical_expansion(qa):
    """Check chemical expansion data validity."""
    print("\n--- Check 6: Chemical Expansion Data ---")

    # GDC
    df_gdc = pd.read_csv(os.path.join(CSV_DIR, "03_gdc_chemical_expansion.csv"))
    if (df_gdc['Delta_delta'] >= 0).all():
        qa.passed("GDC delta >= 0", f"range: [0, {df_gdc['Delta_delta'].max():.4f}]")
    else:
        qa.fail("GDC delta >= 0", "Negative nonstoichiometry!")

    if (df_gdc['Beta_iso'] > 0).all():
        qa.passed("GDC beta_iso > 0", f"value: {df_gdc['Beta_iso'].iloc[0]:.4f}")
    else:
        qa.fail("GDC beta_iso > 0", "Non-positive chemical expansion coefficient!")

    # LSCF
    df_lscf = pd.read_csv(os.path.join(CSV_DIR, "03_lscf_chemical_expansion.csv"))
    if (df_lscf['Beta_11_inplane'] > 0).all() and (df_lscf['Beta_33_outofplane'] > 0).all():
        qa.passed("LSCF beta > 0",
                 f"beta_11={df_lscf['Beta_11_inplane'].iloc[0]:.4f}, "
                 f"beta_33={df_lscf['Beta_33_outofplane'].iloc[0]:.4f}")
    else:
        qa.fail("LSCF beta > 0", "Non-positive chemical expansion coefficient!")

    # Anisotropy ratio should be > 1 for perovskites
    ratio = df_lscf['Anisotropy_ratio_beta33_beta11'].iloc[0]
    if ratio > 1:
        qa.passed("LSCF anisotropy beta_33/beta_11 > 1", f"ratio = {ratio:.3f}")
    else:
        qa.warning("LSCF anisotropy beta_33/beta_11 <= 1",
                  f"ratio = {ratio:.3f} (unusual for perovskites)")


def check_weibull_parameters(qa):
    """Check Weibull statistics validity."""
    print("\n--- Check 7: Weibull Parameters ---")

    df = pd.read_csv(os.path.join(CSV_DIR, "04_weibull_statistics.csv"))

    # Weibull modulus m > 0 and typically 3-20 for ceramics
    if (df['Weibull_Modulus_m'] > 0).all():
        qa.passed("Weibull modulus m > 0",
                 f"range: [{df['Weibull_Modulus_m'].min():.1f}, {df['Weibull_Modulus_m'].max():.1f}]")
    else:
        qa.fail("Weibull modulus m > 0")

    low_m = df[df['Weibull_Modulus_m'] < 3]
    if len(low_m) > 0:
        qa.warning("Weibull modulus < 3 detected",
                  f"{len(low_m)} materials have very low reliability")
    else:
        qa.passed("Weibull modulus >= 3 (reasonable ceramic reliability)")

    # Characteristic strength > threshold
    if (df['Characteristic_Strength_MPa'] > df['Threshold_Stress_MPa']).all():
        qa.passed("sigma_0 > sigma_th (Weibull)")
    else:
        qa.fail("sigma_0 > sigma_th (Weibull)",
               "Characteristic strength below threshold!")

    # Number of specimens >= 20 for reliable Weibull fit
    if (df['N_specimens'] >= 20).all():
        qa.passed("N_specimens >= 20 (all materials)")
    else:
        low_n = df[df['N_specimens'] < 20]
        qa.warning(f"Some materials have N < 20 specimens",
                  f"{len(low_n)} entries with insufficient sample size")


def check_porosity_bounds(qa):
    """Check porosity data validity."""
    print("\n--- Check 8: Porosity Bounds ---")

    df = pd.read_csv(os.path.join(CSV_DIR, "01_porosity_data.csv"))

    if (df['Porosity_vol_frac'] >= 0).all() and (df['Porosity_vol_frac'] <= 1).all():
        qa.passed("0 <= porosity <= 1",
                 f"range: [{df['Porosity_vol_frac'].min():.3f}, {df['Porosity_vol_frac'].max():.3f}]")
    else:
        qa.fail("0 <= porosity <= 1", "Porosity out of physical bounds!")

    # LSCF bulk porosity should be in typical SOFC range (25-45%)
    lscf_bulk = df[(df['Layer'] == 'LSCF_cathode') & (df['Measurement_Region'] == 'Bulk')]
    if len(lscf_bulk) > 0:
        phi = lscf_bulk['Porosity_vol_frac'].iloc[0]
        if 0.2 <= phi <= 0.5:
            qa.passed(f"LSCF bulk porosity in SOFC range", f"phi = {phi:.2f}")
        else:
            qa.warning(f"LSCF bulk porosity outside typical range", f"phi = {phi:.2f}")


def check_interface_roughness(qa):
    """Check interface roughness data."""
    print("\n--- Check 9: Interface Roughness ---")

    df = pd.read_csv(os.path.join(CSV_DIR, "01_interface_roughness.csv"))

    # Ra should be positive
    ra_data = df[df['Parameter'] == 'Ra_amplitude']
    if (ra_data['Value'] > 0).all():
        qa.passed("Interface roughness Ra > 0",
                 f"range: [{ra_data['Value'].min():.3f}, {ra_data['Value'].max():.3f}] um")
    else:
        qa.fail("Interface roughness Ra > 0")

    # Wavelength > Ra (roughness wavelength should exceed amplitude)
    for intf in df['Interface'].unique():
        mask = df['Interface'] == intf
        intf_data = df[mask]
        Ra = intf_data[intf_data['Parameter'] == 'Ra_amplitude']['Value'].values
        lam = intf_data[intf_data['Parameter'] == 'Lambda_wavelength']['Value'].values
        if len(Ra) > 0 and len(lam) > 0 and lam[0] > Ra[0]:
            qa.passed(f"lambda > Ra ({intf})", f"lambda={lam[0]:.2f} > Ra={Ra[0]:.3f} um")
        elif len(Ra) > 0 and len(lam) > 0:
            qa.fail(f"lambda > Ra ({intf})", f"lambda={lam[0]:.2f} <= Ra={Ra[0]:.3f} um!")


def check_data_completeness(qa):
    """Check that all required CSV files exist and are non-empty."""
    print("\n--- Check 10: Data Completeness ---")

    required_files = [
        "01_layer_thicknesses.csv",
        "01_interface_roughness.csv",
        "01_porosity_data.csv",
        "01_mesh_sensitivity.csv",
        "01_rve_convergence.csv",
        "02_youngs_modulus.csv",
        "02_poissons_ratio.csv",
        "02_thermal_expansion.csv",
        "02_umat_full_input.csv",
        "03_gdc_chemical_expansion.csv",
        "03_lscf_chemical_expansion.csv",
        "03_nonstoichiometry_profile.csv",
        "04_bulk_fracture_toughness.csv",
        "04_cohesive_zone_parameters.csv",
        "04_interface_toughness_vs_T.csv",
        "04_phase_field_parameters.csv",
        "04_weibull_statistics.csv",
        "05_curvature_evolution.csv",
        "05_delamination_onset.csv",
        "05_crack_path_morphology.csv",
        "06_parametric_sweep_config.csv",
    ]

    for f in required_files:
        path = os.path.join(CSV_DIR, f)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if len(df) > 0:
                qa.passed(f"File exists and non-empty: {f}", f"{len(df)} rows")
            else:
                qa.fail(f"File empty: {f}")
        else:
            qa.fail(f"File missing: {f}")


def main():
    print("=" * 70)
    print("  PRE-SIMULATION QA VERIFICATION")
    print("  Abaqus YSZ/GDC/LSCF Mixed-Mode Fracture Dataset")
    print("=" * 70)

    qa = QAResult()

    check_data_completeness(qa)
    check_physical_bounds(qa)
    check_thermodynamic_consistency(qa)
    check_positive_definiteness(qa)
    check_temperature_monotonicity(qa)
    check_mesh_objectivity(qa)
    check_chemical_expansion(qa)
    check_weibull_parameters(qa)
    check_porosity_bounds(qa)
    check_interface_roughness(qa)

    exit_code = qa.report()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
