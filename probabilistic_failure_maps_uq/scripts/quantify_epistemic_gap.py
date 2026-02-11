#!/usr/bin/env python3
"""
Quantify epistemic uncertainty gap from synthetic stochastic input files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    csv_dir = base_dir / "csv"

    # Load the surrogate datasets
    df_indep = pd.read_csv(csv_dir / "stochastic_inputs_rho0.00.csv")
    df_corr = pd.read_csv(csv_dir / "stochastic_inputs_rho0.50.csv")
    df_ht = pd.read_csv(csv_dir / "stochastic_inputs_HT_uncorrelated.csv")

    # Calculate "System" Resistance (sum of Gc for both interfaces)
    # This represents the total energy barrier to full delamination.
    r_indep = df_indep["Gc_YSZ_GDC"] + df_indep["Gc_GDC_LSCF"]
    r_corr = df_corr["Gc_YSZ_GDC"] + df_corr["Gc_GDC_LSCF"]
    r_ht = df_ht["Gc_YSZ_GDC"] + df_ht["Gc_GDC_LSCF"]

    # Calculate failure probability for a hypothetical driving force J_applied = 3.0 J/m^2
    j_app = 3.0
    pf_indep = np.mean(r_indep < j_app)
    pf_corr = np.mean(r_corr < j_app)
    pf_ht = np.mean(r_ht < j_app)

    print("--- IMPACT OF MISSING DATA ---")
    print(f"Scenario 1 (Rho=0.0): P_fail = {pf_indep:.4f}")
    print(f"Scenario 2 (Rho=0.5): P_fail = {pf_corr:.4f}")
    print(f"Scenario 3 (High-T):  P_fail = {pf_ht:.4f}")
    print("-" * 30)
    print(f"Epistemic Uncertainty Gap (Rho): {abs(pf_indep - pf_corr) * 100:.2f}%")

    summary = pd.DataFrame(
        [
            {"metric": "J_applied_J_m2", "value": j_app},
            {"metric": "P_fail_rho0_00", "value": pf_indep},
            {"metric": "P_fail_rho0_50", "value": pf_corr},
            {"metric": "P_fail_HT_uncorrelated", "value": pf_ht},
            {"metric": "Epistemic_gap_percent_rho", "value": abs(pf_indep - pf_corr) * 100},
        ]
    )
    summary.to_csv(csv_dir / "epistemic_gap_summary_from_script.csv", index=False)


if __name__ == "__main__":
    main()
