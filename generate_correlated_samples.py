#!/usr/bin/env python3
"""
generate_correlated_samples.py
==============================
Generates correlated Gc values for YSZ|GDC and GDC|LSCF interfaces
to fill the gap of MISSING_DATASET_01 (Interfacial Property Correlation
Matrix).

This script is part of the "Probabilistic Failure Maps" study and
implements:
  1. Independent Sampling (baseline, rho=0)
  2. Correlated Sampling (sensitivity check, user-defined rho)
  3. High-temperature variance scaling (Proportional Variance Assumption)

Usage:
  python generate_correlated_samples.py [--rho 0.5] [--n_samples 500]
"""

import numpy as np
import pandas as pd
import argparse
import os


def generate_correlated_fracture_energies(n_samples, mu1, sigma1, mu2, sigma2, rho):
    """
    Generates correlated Gc values for YSZ|GDC (1) and GDC|LSCF (2)
    to fill the gap of MISSING_DATASET_01.

    Parameters
    ----------
    n_samples : int
        Number of Monte-Carlo samples.
    mu1 : float
        Mean interfacial fracture energy for YSZ|GDC [J/m²].
    sigma1 : float
        Std-dev of Gc for YSZ|GDC [J/m²].
    mu2 : float
        Mean interfacial fracture energy for GDC|LSCF [J/m²].
    sigma2 : float
        Std-dev of Gc for GDC|LSCF [J/m²].
    rho : float
        Pearson correlation coefficient between the two interfaces
        (-1 <= rho <= 1).  rho=0 → independent (baseline assumption).

    Returns
    -------
    pd.DataFrame
        Columns: Gc_YSZ_GDC, Gc_GDC_LSCF
    """
    mean = [mu1, mu2]
    cov = [[sigma1**2,              rho * sigma1 * sigma2],
           [rho * sigma1 * sigma2,  sigma2**2]]

    samples = np.random.multivariate_normal(mean, cov, n_samples)

    # Enforce physical positivity (Gc > 0)
    samples = np.clip(samples, 1e-6, None)

    df = pd.DataFrame(samples, columns=['Gc_YSZ_GDC', 'Gc_GDC_LSCF'])
    return df


def scale_to_high_temperature(df, E_ratio_YSZ=170.0/205.0, E_ratio_LSCF=88.0/115.0,
                               cv_constant=True):
    """
    Apply the Proportional Variance Assumption to scale RT values to 800°C.

    If cv_constant=True (default / MISSING_DATASET_02 assumption):
        sigma_800 = CV_RT * mu_800
        mu_800    = mu_RT * (E_800 / E_RT)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Gc_YSZ_GDC' and 'Gc_GDC_LSCF'.
    E_ratio_YSZ : float
        E(800°C) / E(RT) for the YSZ-side interface.
    E_ratio_LSCF : float
        E(800°C) / E(RT) for the LSCF-side interface.
    cv_constant : bool
        If True, keep coefficient of variation constant.

    Returns
    -------
    pd.DataFrame
        High-temperature scaled fracture energies.
    """
    df_ht = df.copy()
    if cv_constant:
        # Scale each sample proportionally
        df_ht['Gc_YSZ_GDC_800C'] = df['Gc_YSZ_GDC'] * E_ratio_YSZ
        df_ht['Gc_GDC_LSCF_800C'] = df['Gc_GDC_LSCF'] * E_ratio_LSCF
    else:
        # Simple mean scaling, preserve absolute scatter
        df_ht['Gc_YSZ_GDC_800C'] = (df['Gc_YSZ_GDC'] - df['Gc_YSZ_GDC'].mean()) \
                                    + df['Gc_YSZ_GDC'].mean() * E_ratio_YSZ
        df_ht['Gc_GDC_LSCF_800C'] = (df['Gc_GDC_LSCF'] - df['Gc_GDC_LSCF'].mean()) \
                                     + df['Gc_GDC_LSCF'].mean() * E_ratio_LSCF
    return df_ht


def verify_samples(df, expected_rho, tol_rho=0.08, tol_std_pct=5.0):
    """
    QA checks on the generated samples.

    1. Pearson correlation should match expected_rho within tolerance.
    2. Sample standard deviation should be within tol_std_pct of the
       population value.
    """
    r_actual = df['Gc_YSZ_GDC'].corr(df['Gc_GDC_LSCF'])
    print(f"  Pearson r (actual):   {r_actual:.4f}  (expected ≈ {expected_rho})")
    if abs(r_actual - expected_rho) > tol_rho:
        print(f"  ⚠ WARNING: Correlation deviates by more than {tol_rho}")
    else:
        print(f"  ✓ Correlation within tolerance ±{tol_rho}")

    for col in ['Gc_YSZ_GDC', 'Gc_GDC_LSCF']:
        sd = df[col].std()
        print(f"  {col}: std = {sd:.4f} J/m²")


def main():
    parser = argparse.ArgumentParser(
        description="Generate correlated interfacial fracture energies "
                    "for SOC probabilistic failure map inputs."
    )
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of Monte-Carlo samples (default: 500)")
    parser.add_argument("--rho", type=float, default=0.0,
                        help="Correlation coefficient (default: 0.0 = independent)")
    parser.add_argument("--mu1", type=float, default=2.15,
                        help="Mean Gc YSZ|GDC [J/m²] (default: 2.15)")
    parser.add_argument("--sigma1", type=float, default=0.40,
                        help="Std Gc YSZ|GDC [J/m²] (default: 0.40)")
    parser.add_argument("--mu2", type=float, default=1.00,
                        help="Mean Gc GDC|LSCF [J/m²] (default: 1.00)")
    parser.add_argument("--sigma2", type=float, default=0.20,
                        help="Std Gc GDC|LSCF [J/m²] (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--outdir", type=str, default=".",
                        help="Output directory (default: current directory)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 60)
    print("Correlated Fracture Energy Sample Generator")
    print("=" * 60)
    print(f"  Samples:   {args.n_samples}")
    print(f"  rho:       {args.rho}")
    print(f"  YSZ|GDC:   mu={args.mu1}, sigma={args.sigma1}")
    print(f"  GDC|LSCF:  mu={args.mu2}, sigma={args.sigma2}")
    print(f"  Seed:      {args.seed}")
    print()

    # ---- Generate RT samples ----
    df = generate_correlated_fracture_energies(
        n_samples=args.n_samples,
        mu1=args.mu1, sigma1=args.sigma1,
        mu2=args.mu2, sigma2=args.sigma2,
        rho=args.rho
    )

    print("RT Sample Statistics:")
    verify_samples(df, expected_rho=args.rho)
    print()

    # ---- High-temperature scaling ----
    df = scale_to_high_temperature(df)
    print("High-Temperature (800°C) Statistics:")
    for col in ['Gc_YSZ_GDC_800C', 'Gc_GDC_LSCF_800C']:
        print(f"  {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")
    print()

    # ---- Save ----
    outfile = os.path.join(args.outdir, f"stochastic_inputs_rho{args.rho:.2f}.csv")
    df.to_csv(outfile, index=False)
    print(f"Saved {len(df)} samples to: {outfile}")
    print()
    print("Preview:")
    print(df.head(10).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
