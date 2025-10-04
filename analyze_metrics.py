#!/usr/bin/env python3
"""
Analysis Tools for Synthetic Synchrotron X-ray Data
====================================================

This script provides quantitative analysis of creep deformation metrics
extracted from the synthetic synchrotron data.
"""

import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit


class CreepDataAnalyzer:
    """
    Quantitative analysis of creep deformation data.
    """
    
    def __init__(self, data_dir="synchrotron_data"):
        """Initialize analyzer with data directory."""
        self.data_dir = Path(data_dir)
        self.tomography_dir = self.data_dir / "tomography"
        self.diffraction_dir = self.data_dir / "diffraction"
        self.metadata_dir = self.data_dir / "metadata"
        
        # Load metrics
        metrics_file = self.tomography_dir / "tomography_metrics.json"
        with open(metrics_file, 'r') as f:
            self.metrics = json.load(f)
        
        # Load experimental parameters
        exp_file = self.metadata_dir / "experimental_parameters.json"
        with open(exp_file, 'r') as f:
            self.exp_params = json.load(f)
    
    def fit_primary_creep(self):
        """
        Fit primary creep model: ε = ε₀ + A*t^n
        where n < 1 for primary creep
        """
        time = np.array(self.metrics['time_hours']) * 3600  # Convert to seconds
        strain = np.array(self.metrics['porosity_percent']) / 100  # Normalized
        
        # Primary creep model
        def primary_creep(t, eps0, A, n):
            return eps0 + A * t**n
        
        # Fit to first half of data (primary creep region)
        mid = len(time) // 2
        try:
            popt, pcov = curve_fit(primary_creep, time[:mid], strain[:mid], 
                                  p0=[0.03, 1e-6, 0.3])
            
            eps0, A, n = popt
            perr = np.sqrt(np.diag(pcov))
            
            results = {
                "model": "Primary Creep: ε = ε₀ + A*t^n",
                "parameters": {
                    "eps0": float(eps0),
                    "A": float(A),
                    "n": float(n)
                },
                "standard_errors": {
                    "eps0_err": float(perr[0]),
                    "A_err": float(perr[1]),
                    "n_err": float(perr[2])
                },
                "fit_quality": {
                    "r_squared": float(self._calculate_r_squared(
                        strain[:mid], primary_creep(time[:mid], *popt)))
                }
            }
            
            print("\n" + "="*70)
            print("PRIMARY CREEP ANALYSIS")
            print("="*70)
            print(f"Model: {results['model']}")
            print(f"  ε₀ = {eps0:.6f} ± {perr[0]:.6f}")
            print(f"  A  = {A:.6e} ± {perr[1]:.6e}")
            print(f"  n  = {n:.4f} ± {perr[2]:.4f}")
            print(f"  R² = {results['fit_quality']['r_squared']:.4f}")
            
            return results
        
        except Exception as e:
            print(f"Warning: Could not fit primary creep model: {e}")
            return None
    
    def fit_secondary_creep(self):
        """
        Fit secondary creep model: ε = ε₀ + ε̇_s*t
        where ε̇_s is the steady-state creep rate
        """
        time = np.array(self.metrics['time_hours']) * 3600  # Seconds
        strain = np.array(self.metrics['porosity_percent']) / 100
        
        # Linear fit to second half (steady-state region)
        mid = len(time) // 2
        
        # Linear regression
        coeffs = np.polyfit(time[mid:], strain[mid:], 1)
        strain_rate = coeffs[0]
        intercept = coeffs[1]
        
        # Calculate R²
        strain_fit = np.polyval(coeffs, time[mid:])
        r_squared = self._calculate_r_squared(strain[mid:], strain_fit)
        
        results = {
            "model": "Secondary Creep: ε = ε₀ + ε̇_s*t",
            "parameters": {
                "eps0": float(intercept),
                "strain_rate_per_s": float(strain_rate),
                "strain_rate_per_hour": float(strain_rate * 3600)
            },
            "fit_quality": {
                "r_squared": float(r_squared)
            }
        }
        
        print("\n" + "="*70)
        print("SECONDARY CREEP ANALYSIS")
        print("="*70)
        print(f"Model: {results['model']}")
        print(f"  ε₀  = {intercept:.6f}")
        print(f"  ε̇_s = {strain_rate:.6e} s⁻¹")
        print(f"      = {strain_rate * 3600:.6e} h⁻¹")
        print(f"  R²  = {r_squared:.4f}")
        
        return results
    
    def analyze_cavity_nucleation(self):
        """
        Analyze cavity nucleation kinetics.
        """
        time = np.array(self.metrics['time_hours'])
        cavity_count = np.array(self.metrics['cavity_count'])
        
        # Calculate nucleation rate (cavities per hour)
        nucleation_rate = np.gradient(cavity_count, time)
        
        results = {
            "total_cavities_final": int(cavity_count[-1]),
            "total_cavities_initial": int(cavity_count[0]),
            "net_cavities_nucleated": int(cavity_count[-1] - cavity_count[0]),
            "average_nucleation_rate_per_hour": float(np.mean(nucleation_rate[1:])),
            "peak_nucleation_rate_per_hour": float(np.max(nucleation_rate[1:])),
            "time_of_peak_nucleation_hours": float(time[np.argmax(nucleation_rate[1:]) + 1])
        }
        
        print("\n" + "="*70)
        print("CAVITY NUCLEATION ANALYSIS")
        print("="*70)
        print(f"Initial cavity count: {results['total_cavities_initial']}")
        print(f"Final cavity count: {results['total_cavities_final']}")
        print(f"Net nucleated: {results['net_cavities_nucleated']}")
        print(f"Average rate: {results['average_nucleation_rate_per_hour']:.2f} cavities/hour")
        print(f"Peak rate: {results['peak_nucleation_rate_per_hour']:.2f} cavities/hour")
        print(f"  at t = {results['time_of_peak_nucleation_hours']:.1f} hours")
        
        return results
    
    def analyze_crack_propagation(self):
        """
        Analyze crack propagation kinetics.
        """
        time = np.array(self.metrics['time_hours'])
        crack_volume = np.array(self.metrics['crack_volume_mm3'])
        
        # Calculate crack growth rate
        growth_rate = np.gradient(crack_volume, time)
        
        # Fit exponential growth if applicable
        try:
            def exp_growth(t, V0, k):
                return V0 * np.exp(k * t)
            
            popt, _ = curve_fit(exp_growth, time[1:], crack_volume[1:], 
                              p0=[crack_volume[1], 0.01])
            
            V0, k = popt
            doubling_time = np.log(2) / k if k > 0 else np.inf
            
            results = {
                "initial_crack_volume_mm3": float(crack_volume[0]),
                "final_crack_volume_mm3": float(crack_volume[-1]),
                "total_crack_growth_mm3": float(crack_volume[-1] - crack_volume[0]),
                "average_growth_rate_mm3_per_hour": float(np.mean(growth_rate[1:])),
                "exponential_fit": {
                    "V0": float(V0),
                    "growth_constant_k": float(k),
                    "doubling_time_hours": float(doubling_time) if np.isfinite(doubling_time) else "N/A"
                }
            }
        except:
            results = {
                "initial_crack_volume_mm3": float(crack_volume[0]),
                "final_crack_volume_mm3": float(crack_volume[-1]),
                "total_crack_growth_mm3": float(crack_volume[-1] - crack_volume[0]),
                "average_growth_rate_mm3_per_hour": float(np.mean(growth_rate[1:])),
                "exponential_fit": None
            }
        
        print("\n" + "="*70)
        print("CRACK PROPAGATION ANALYSIS")
        print("="*70)
        print(f"Initial crack volume: {results['initial_crack_volume_mm3']:.6f} mm³")
        print(f"Final crack volume: {results['final_crack_volume_mm3']:.6f} mm³")
        print(f"Total growth: {results['total_crack_growth_mm3']:.6f} mm³")
        print(f"Average rate: {results['average_growth_rate_mm3_per_hour']:.6e} mm³/hour")
        
        if results.get('exponential_fit'):
            exp_fit = results['exponential_fit']
            print(f"\nExponential fit: V(t) = V₀*exp(k*t)")
            print(f"  V₀ = {exp_fit['V0']:.6e} mm³")
            print(f"  k  = {exp_fit['growth_constant_k']:.4f} h⁻¹")
            if isinstance(exp_fit['doubling_time_hours'], float):
                print(f"  Doubling time = {exp_fit['doubling_time_hours']:.1f} hours")
        
        return results
    
    def analyze_strain_distribution(self):
        """
        Analyze strain/stress distributions from XRD data.
        """
        strain_file = self.diffraction_dir / "strain_stress_maps.h5"
        
        with h5py.File(strain_file, 'r') as f:
            strain_data = f['elastic_strain'][:]
            stress_data = f['residual_stress_MPa'][:]
            time = f['time_hours'][:]
        
        results = {
            "time_evolution": []
        }
        
        for t_idx, t in enumerate(time):
            strain = strain_data[t_idx]
            stress = stress_data[t_idx]
            
            time_point = {
                "time_hours": float(t),
                "strain": {
                    "mean": float(np.mean(strain)),
                    "std": float(np.std(strain)),
                    "min": float(np.min(strain)),
                    "max": float(np.max(strain)),
                    "percentile_95": float(np.percentile(strain, 95))
                },
                "stress_MPa": {
                    "mean": float(np.mean(stress)),
                    "std": float(np.std(stress)),
                    "min": float(np.min(stress)),
                    "max": float(np.max(stress)),
                    "percentile_95": float(np.percentile(stress, 95))
                }
            }
            
            results["time_evolution"].append(time_point)
        
        print("\n" + "="*70)
        print("STRAIN/STRESS DISTRIBUTION ANALYSIS")
        print("="*70)
        print(f"Initial state (t = {time[0]:.1f} h):")
        print(f"  Mean strain: {results['time_evolution'][0]['strain']['mean']:.6f}")
        print(f"  Mean stress: {results['time_evolution'][0]['stress_MPa']['mean']:.2f} MPa")
        print(f"\nFinal state (t = {time[-1]:.1f} h):")
        print(f"  Mean strain: {results['time_evolution'][-1]['strain']['mean']:.6f}")
        print(f"  Mean stress: {results['time_evolution'][-1]['stress_MPa']['mean']:.2f} MPa")
        print(f"\nStrain accumulation: {results['time_evolution'][-1]['strain']['mean'] - results['time_evolution'][0]['strain']['mean']:.6f}")
        
        return results
    
    def _calculate_r_squared(self, y_true, y_pred):
        """Calculate R² coefficient of determination."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def generate_full_report(self, output_file="analysis_report.json"):
        """
        Generate comprehensive analysis report.
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE CREEP ANALYSIS REPORT")
        print("="*70)
        
        report = {
            "analysis_timestamp": np.datetime64('now').astype(str),
            "experimental_conditions": self.exp_params['test_conditions'],
            "primary_creep": self.fit_primary_creep(),
            "secondary_creep": self.fit_secondary_creep(),
            "cavity_nucleation": self.analyze_cavity_nucleation(),
            "crack_propagation": self.analyze_crack_propagation(),
            "strain_distribution": self.analyze_strain_distribution()
        }
        
        # Save report
        output_path = self.data_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n" + "="*70)
        print(f"✓ Analysis report saved to: {output_path}")
        print("="*70 + "\n")
        
        return report


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze synthetic synchrotron creep data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="synchrotron_data",
        help="Directory containing generated data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis_report.json",
        help="Output file for analysis report"
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = CreepDataAnalyzer(data_dir=args.data_dir)
    
    # Generate full report
    report = analyzer.generate_full_report(output_file=args.output)
    
    print("\nAnalysis complete. Key findings:")
    print("-" * 70)
    
    if report['primary_creep']:
        print(f"Primary creep exponent (n): {report['primary_creep']['parameters']['n']:.4f}")
    
    if report['secondary_creep']:
        print(f"Secondary creep rate: {report['secondary_creep']['parameters']['strain_rate_per_hour']:.6e} h⁻¹")
    
    print(f"Total cavities nucleated: {report['cavity_nucleation']['net_cavities_nucleated']}")
    print(f"Total crack growth: {report['crack_propagation']['total_crack_growth_mm3']:.6f} mm³")
    print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
