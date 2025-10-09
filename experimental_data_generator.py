#!/usr/bin/env python3
"""
Experimental Data Generator for SOFC Material Properties
=======================================================

This module generates synthetic experimental data based on realistic measurement
techniques and includes proper uncertainty quantification, measurement artifacts,
and statistical distributions that would be observed in real experiments.

Author: Generated for SOFC Research
Date: 2025-10-09
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

@dataclass
class ExperimentalConditions:
    """Experimental conditions for material property measurements."""
    temperature_K: float
    atmosphere: str  # 'air', 'N2', 'H2', 'Ar', etc.
    pressure_Pa: float
    humidity_percent: float
    loading_rate_per_sec: Optional[float] = None
    sample_geometry: Optional[str] = None
    measurement_duration_hours: Optional[float] = None

class NanoindentationDataGenerator:
    """Generate realistic nanoindentation data for elastic property measurement."""
    
    def __init__(self, material_name: str, true_properties: Dict):
        self.material_name = material_name
        self.true_E = true_properties['youngs_modulus_GPa']
        self.true_nu = true_properties['poissons_ratio']
        self.true_H = self.estimate_hardness()  # GPa
        
    def estimate_hardness(self) -> float:
        """Estimate hardness from Young's modulus using empirical correlations."""
        if 'YSZ' in self.material_name:
            return self.true_E / 8.0  # Typical for ceramics
        elif 'Ni' in self.material_name:
            return self.true_E / 15.0  # Typical for metals
        else:
            return self.true_E / 12.0  # Composite estimate
    
    def generate_load_displacement_curves(self, n_curves: int = 50, max_depth_nm: float = 2000) -> pd.DataFrame:
        """Generate realistic load-displacement curves from nanoindentation."""
        
        data_points = []
        
        for curve_id in range(n_curves):
            # Experimental variations
            surface_roughness = np.random.normal(0, 5)  # nm
            tip_bluntness = np.random.normal(0, 2)  # nm
            drift_rate = np.random.normal(0, 0.1)  # nm/s
            
            # Depth array
            depths = np.linspace(0, max_depth_nm, 100)
            
            # Oliver-Pharr model with realistic artifacts
            loads = []
            for i, h in enumerate(depths):
                # Ideal load (Berkovich indenter)
                h_eff = max(h - surface_roughness - tip_bluntness, 0.1)
                P_ideal = (4 * self.true_E * np.tan(np.radians(70.3)) * h_eff**2) / (3 * (1 - self.true_nu**2))
                
                # Add experimental artifacts
                thermal_drift = drift_rate * i * 0.1
                machine_compliance = P_ideal * 1e-6  # nm/mN
                pile_up_effect = 0.95 if h > 500 else 1.0  # Pile-up reduces apparent depth
                
                P_measured = P_ideal * pile_up_effect + np.random.normal(0, P_ideal * 0.02)
                h_measured = h + thermal_drift + machine_compliance + np.random.normal(0, 2)
                
                loads.append(max(P_measured, 0))
                
                data_points.append({
                    'curve_id': curve_id,
                    'depth_nm': h_measured,
                    'load_mN': P_measured / 1000,  # Convert to mN
                    'contact_depth_nm': h_measured * 0.75,  # Approximate contact depth
                    'surface_roughness_nm': surface_roughness,
                    'measurement_time_s': i * 0.5
                })
        
        return pd.DataFrame(data_points)
    
    def analyze_nanoindentation_data(self, df: pd.DataFrame) -> Dict:
        """Analyze nanoindentation data to extract elastic properties."""
        
        results = []
        
        for curve_id in df['curve_id'].unique():
            curve_data = df[df['curve_id'] == curve_id].copy()
            
            # Find maximum load and corresponding depth
            max_load_idx = curve_data['load_mN'].idxmax()
            max_load = curve_data.loc[max_load_idx, 'load_mN']
            max_depth = curve_data.loc[max_load_idx, 'depth_nm']
            
            # Unloading curve analysis (Oliver-Pharr method)
            unloading_data = curve_data[curve_data.index >= max_load_idx].copy()
            
            if len(unloading_data) > 10:
                # Fit power law to unloading curve: P = α(h - hf)^m
                try:
                    def power_law(h, alpha, hf, m):
                        return alpha * np.power(np.maximum(h - hf, 0.001), m)
                    
                    popt, _ = curve_fit(power_law, unloading_data['depth_nm'], 
                                      unloading_data['load_mN'], 
                                      p0=[1, max_depth*0.8, 1.5],
                                      bounds=([0, 0, 1.0], [np.inf, max_depth, 3.0]))
                    
                    alpha, hf, m = popt
                    
                    # Calculate contact stiffness S = dP/dh at max load
                    S = alpha * m * (max_depth - hf)**(m-1)  # mN/nm
                    
                    # Calculate contact area (Berkovich indenter)
                    hc = max_depth - 0.75 * max_load / S  # Contact depth
                    Ac = 24.5 * hc**2  # nm² for perfect Berkovich
                    
                    # Calculate reduced modulus
                    Er = (np.sqrt(np.pi) * S) / (2 * np.sqrt(Ac)) * 1000  # GPa (conversion from mN/nm to GPa)
                    
                    # Calculate Young's modulus (assuming diamond tip: Ei=1141 GPa, νi=0.07)
                    Ei, nu_i = 1141, 0.07
                    E_sample = 1 / ((1 - self.true_nu**2)/Er - (1 - nu_i**2)/Ei)
                    
                    # Calculate hardness
                    H = max_load * 1000 / Ac  # GPa (conversion from mN to μN)
                    
                    results.append({
                        'curve_id': curve_id,
                        'youngs_modulus_GPa': E_sample,
                        'hardness_GPa': H,
                        'contact_stiffness_mN_nm': S,
                        'contact_depth_nm': hc,
                        'contact_area_nm2': Ac,
                        'max_load_mN': max_load,
                        'max_depth_nm': max_depth,
                        'residual_depth_nm': hf,
                        'power_law_exponent': m
                    })
                    
                except Exception as e:
                    # Failed curve fit - mark as invalid
                    results.append({
                        'curve_id': curve_id,
                        'youngs_modulus_GPa': np.nan,
                        'hardness_GPa': np.nan,
                        'error': str(e)
                    })
        
        results_df = pd.DataFrame(results)
        
        # Statistical analysis
        valid_results = results_df.dropna(subset=['youngs_modulus_GPa']) if 'youngs_modulus_GPa' in results_df.columns else results_df.dropna()
        
        statistics = {
            'n_valid_measurements': len(valid_results),
            'n_total_measurements': len(results_df),
            'success_rate': len(valid_results) / len(results_df) if len(results_df) > 0 else 0,
        }
        
        # Add Young's modulus statistics if available
        if 'youngs_modulus_GPa' in valid_results.columns and len(valid_results) > 0:
            statistics['youngs_modulus'] = {
                'mean_GPa': valid_results['youngs_modulus_GPa'].mean(),
                'std_GPa': valid_results['youngs_modulus_GPa'].std(),
                'median_GPa': valid_results['youngs_modulus_GPa'].median(),
                'cv_percent': valid_results['youngs_modulus_GPa'].std() / valid_results['youngs_modulus_GPa'].mean() * 100 if valid_results['youngs_modulus_GPa'].mean() != 0 else 0
            }
        else:
            statistics['youngs_modulus'] = {
                'mean_GPa': 0,
                'std_GPa': 0,
                'median_GPa': 0,
                'cv_percent': 0
            }
        
        # Add hardness statistics if available
        if 'hardness_GPa' in valid_results.columns and len(valid_results) > 0:
            statistics['hardness'] = {
                'mean_GPa': valid_results['hardness_GPa'].mean(),
                'std_GPa': valid_results['hardness_GPa'].std(),
                'median_GPa': valid_results['hardness_GPa'].median(),
                'cv_percent': valid_results['hardness_GPa'].std() / valid_results['hardness_GPa'].mean() * 100 if valid_results['hardness_GPa'].mean() != 0 else 0
            }
        else:
            statistics['hardness'] = {
                'mean_GPa': 0,
                'std_GPa': 0,
                'median_GPa': 0,
                'cv_percent': 0
            }
        
        return {
            'raw_results': results_df,
            'statistics': statistics,
            'experimental_conditions': {
                'material': self.material_name,
                'true_E_GPa': self.true_E,
                'true_nu': self.true_nu,
                'measurement_method': 'nanoindentation_oliver_pharr'
            }
        }

class FractureTestingDataGenerator:
    """Generate realistic fracture testing data."""
    
    def __init__(self, material_name: str, true_properties: Dict):
        self.material_name = material_name
        self.true_Kic = true_properties['fracture_toughness_MPa_sqrt_m']
        self.true_Gc = true_properties['critical_energy_release_rate_J_m2']
        
    def generate_compact_tension_data(self, n_specimens: int = 15) -> pd.DataFrame:
        """Generate compact tension (CT) test data."""
        
        specimens = []
        
        for specimen_id in range(n_specimens):
            # Specimen geometry (CT specimen)
            W = 25.0 + np.random.normal(0, 0.1)  # Width, mm
            B = 12.5 + np.random.normal(0, 0.05)  # Thickness, mm
            a0 = W * 0.5 + np.random.normal(0, 0.2)  # Initial crack length, mm
            
            # Pre-cracking (fatigue)
            fatigue_cycles = np.random.randint(50000, 150000)
            da_fatigue = np.random.normal(2.0, 0.3)  # mm
            a_precrack = a0 + da_fatigue
            
            # Load-displacement curve generation
            displacements = np.linspace(0, 2.0, 200)  # mm
            loads = []
            crack_lengths = []
            
            current_a = a_precrack
            
            for i, delta in enumerate(displacements):
                # Compliance function for CT specimen
                alpha = current_a / W
                f_alpha = ((2 + alpha) * (0.886 + 4.64*alpha - 13.32*alpha**2 + 14.72*alpha**3 - 5.6*alpha**4)) / ((1 - alpha)**1.5)
                
                # Elastic compliance
                C_elastic = f_alpha**2 / (B * W * self.true_Kic**2 / self.true_Gc)
                
                # Load from compliance
                if delta < 0.1:  # Linear elastic region
                    P = delta / C_elastic
                else:  # Crack growth region
                    # R-curve behavior (rising crack resistance)
                    da_stable = (delta - 0.1) * 0.5  # Stable crack growth
                    current_a = min(a_precrack + da_stable, W * 0.8)
                    
                    # Update compliance with new crack length
                    alpha = current_a / W
                    f_alpha = ((2 + alpha) * (0.886 + 4.64*alpha - 13.32*alpha**2 + 14.72*alpha**3 - 5.6*alpha**4)) / ((1 - alpha)**1.5)
                    C_elastic = f_alpha**2 / (B * W * self.true_Kic**2 / self.true_Gc)
                    
                    P = delta / C_elastic
                
                # Add experimental noise
                P_measured = P * (1 + np.random.normal(0, 0.02))
                loads.append(max(P_measured, 0))
                crack_lengths.append(current_a)
                
                # Check for instability
                K_applied = (P * f_alpha) / (B * np.sqrt(W))
                if K_applied > self.true_Kic * 1.1:  # Unstable fracture
                    break
            
            # Find critical values
            max_load_idx = np.argmax(loads)
            P_max = loads[max_load_idx]
            a_critical = crack_lengths[max_load_idx]
            
            # Calculate fracture toughness
            alpha_c = a_critical / W
            f_alpha_c = ((2 + alpha_c) * (0.886 + 4.64*alpha_c - 13.32*alpha_c**2 + 14.72*alpha_c**3 - 5.6*alpha_c**4)) / ((1 - alpha_c)**1.5)
            Kic_measured = (P_max * f_alpha_c) / (B * np.sqrt(W))
            
            # Calculate J-integral (energy method)
            area_under_curve = np.trapz(loads[:max_load_idx+1], displacements[:max_load_idx+1])
            J_critical = area_under_curve / (B * (W - a_critical))
            
            specimens.append({
                'specimen_id': specimen_id,
                'width_mm': W,
                'thickness_mm': B,
                'initial_crack_length_mm': a0,
                'precrack_length_mm': a_precrack,
                'critical_crack_length_mm': a_critical,
                'max_load_N': P_max,
                'fracture_toughness_MPa_sqrt_m': Kic_measured,
                'j_integral_kJ_m2': J_critical,
                'fatigue_cycles': fatigue_cycles,
                'crack_growth_mm': a_critical - a_precrack,
                'load_displacement_data': list(zip(displacements[:max_load_idx+1], loads[:max_load_idx+1]))
            })
        
        return pd.DataFrame(specimens)
    
    def generate_interface_fracture_data(self, interface_type: str, n_specimens: int = 12) -> pd.DataFrame:
        """Generate interface fracture test data (double cantilever beam)."""
        
        specimens = []
        
        for specimen_id in range(n_specimens):
            # Bi-material specimen geometry
            L = 100.0 + np.random.normal(0, 1.0)  # Length, mm
            h1 = 3.0 + np.random.normal(0, 0.1)   # Thickness material 1, mm
            h2 = 3.0 + np.random.normal(0, 0.1)   # Thickness material 2, mm
            b = 10.0 + np.random.normal(0, 0.1)   # Width, mm
            a0 = 20.0 + np.random.normal(0, 0.5)  # Initial crack length, mm
            
            # Material properties (simplified)
            E1, E2 = 200.0, 180.0  # GPa
            
            # Load-displacement data generation
            displacements = np.linspace(0, 5.0, 100)  # mm
            loads = []
            crack_lengths = []
            
            current_a = a0
            
            for delta in displacements:
                # Compliance for DCB specimen
                C = (8 * current_a**3) / (3 * E1 * b * h1**3) + (8 * current_a**3) / (3 * E2 * b * h2**3)
                
                # Load from compliance
                P = delta / C
                
                # Interface crack growth
                G_applied = (P**2 * C) / (2 * b)
                
                if G_applied > self.true_Gc:
                    da = (G_applied - self.true_Gc) * 0.001  # Crack extension
                    current_a += da
                
                # Add measurement noise
                P_measured = P * (1 + np.random.normal(0, 0.03))
                loads.append(max(P_measured, 0))
                crack_lengths.append(current_a)
                
                # Check for complete failure
                if current_a > L * 0.8:
                    break
            
            # Calculate interface toughness
            max_stable_load = max(loads) * 0.95  # Before instability
            max_stable_idx = next(i for i, p in enumerate(loads) if p >= max_stable_load)
            
            C_critical = (8 * crack_lengths[max_stable_idx]**3) / (3 * E1 * b * h1**3) + (8 * crack_lengths[max_stable_idx]**3) / (3 * E2 * b * h2**3)
            Gc_measured = (loads[max_stable_idx]**2 * C_critical) / (2 * b)
            
            specimens.append({
                'specimen_id': specimen_id,
                'interface_type': interface_type,
                'length_mm': L,
                'thickness_1_mm': h1,
                'thickness_2_mm': h2,
                'width_mm': b,
                'initial_crack_length_mm': a0,
                'critical_crack_length_mm': crack_lengths[max_stable_idx],
                'critical_load_N': loads[max_stable_idx],
                'interface_toughness_J_m2': Gc_measured,
                'crack_extension_mm': crack_lengths[max_stable_idx] - a0,
                'load_displacement_data': list(zip(displacements[:len(loads)], loads))
            })
        
        return pd.DataFrame(specimens)

class ThermalAnalysisDataGenerator:
    """Generate thermal analysis data (dilatometry, DSC, TGA)."""
    
    def __init__(self, material_name: str, true_properties: Dict):
        self.material_name = material_name
        self.true_CTE = true_properties['thermal_expansion_coefficient_K_inv']
        
    def generate_dilatometry_data(self, T_range: Tuple[float, float] = (298, 1273), 
                                heating_rate_K_min: float = 5.0) -> pd.DataFrame:
        """Generate dilatometry (thermal expansion) data."""
        
        T_start, T_end = T_range
        n_points = int((T_end - T_start) / heating_rate_K_min * 60 / 10)  # 10 second intervals
        
        temperatures = np.linspace(T_start, T_end, n_points)
        
        data_points = []
        L0 = 10.0  # Initial length, mm
        
        for i, T in enumerate(temperatures):
            # Theoretical thermal expansion
            dT = T - T_start
            
            # Non-linear CTE (realistic behavior)
            CTE_T = self.true_CTE * (1 + 2e-5 * dT + 1e-8 * dT**2)
            
            # Integrated thermal strain
            thermal_strain = self.true_CTE * dT + 1e-5 * dT**2 + 1e-9 * dT**3
            
            # Length change
            dL = L0 * thermal_strain
            L_current = L0 + dL
            
            # Add experimental artifacts
            baseline_drift = 0.001 * i / n_points  # mm
            thermal_lag = np.sin(2 * np.pi * i / 50) * 0.0001  # mm, periodic noise
            measurement_noise = np.random.normal(0, 0.0002)  # mm
            
            L_measured = L_current + baseline_drift + thermal_lag + measurement_noise
            
            # Calculate instantaneous CTE
            if i > 0:
                dL_dT = (L_measured - data_points[-1]['length_mm']) / (T - temperatures[i-1])
                CTE_instantaneous = dL_dT / L0
            else:
                CTE_instantaneous = self.true_CTE
            
            data_points.append({
                'temperature_K': T,
                'temperature_C': T - 273.15,
                'length_mm': L_measured,
                'length_change_mm': L_measured - L0,
                'thermal_strain': (L_measured - L0) / L0,
                'instantaneous_CTE_K_inv': CTE_instantaneous,
                'time_minutes': i * heating_rate_K_min / 60,
                'heating_rate_K_min': heating_rate_K_min
            })
        
        df = pd.DataFrame(data_points)
        
        # Calculate average CTE over different temperature ranges
        df['average_CTE_300_600K'] = df[df['temperature_K'] <= 600]['thermal_strain'].iloc[-1] / (600 - 300) * L0
        df['average_CTE_600_1000K'] = (df[df['temperature_K'] <= 1000]['thermal_strain'].iloc[-1] - 
                                      df[df['temperature_K'] <= 600]['thermal_strain'].iloc[-1]) / (1000 - 600) * L0
        
        return df
    
    def generate_chemical_expansion_data(self, pO2_range: Tuple[float, float] = (-20, -10), 
                                       temperature_K: float = 1073) -> pd.DataFrame:
        """Generate chemical expansion data as function of oxygen partial pressure."""
        
        log_pO2_values = np.linspace(pO2_range[0], pO2_range[1], 50)
        pO2_values = 10**log_pO2_values  # Pa
        
        data_points = []
        L0 = 10.0  # Initial length, mm
        
        for i, (log_pO2, pO2) in enumerate(zip(log_pO2_values, pO2_values)):
            
            # Chemical expansion model (depends on material)
            # Equilibrium constant and defect chemistry
            K_eq = np.exp(-2.0 + 20000 / (8.314 * temperature_K))  # Simplified
            oxidation_fraction = K_eq * pO2**0.5 / (1 + K_eq * pO2**0.5)
            
            if 'Ni' in self.material_name and 'YSZ' not in self.material_name:
                # Pure Ni oxidation: Ni + 1/2 O2 → NiO
                # Volume expansion: NiO has ~70% larger molar volume than Ni
                chemical_strain = oxidation_fraction * 0.21  # Linear expansion coefficient
                
            elif 'YSZ' in self.material_name and 'Ni' not in self.material_name:
                # Minimal chemical expansion for YSZ
                chemical_strain = 1e-6 * np.log10(max(pO2 / 1e-15, 1e-10))  # Very small effect
                
            else:  # Composite
                # Weighted average with constraint effects
                ni_fraction = 0.4  # Assume 40% Ni
                ni_strain = 0.21 * oxidation_fraction
                constraint_factor = 1 - 0.6 * (1 - ni_fraction)  # YSZ constrains Ni
                chemical_strain = ni_fraction * ni_strain * constraint_factor
            
            # Add experimental noise
            noise_std = abs(chemical_strain * 0.05) if chemical_strain != 0 else 1e-8
            measurement_noise = np.random.normal(0, noise_std)
            chemical_strain_measured = chemical_strain + measurement_noise
            
            # Length change
            dL_chemical = L0 * chemical_strain_measured
            L_measured = L0 + dL_chemical
            
            data_points.append({
                'log_pO2_Pa': log_pO2,
                'pO2_Pa': pO2,
                'temperature_K': temperature_K,
                'length_mm': L_measured,
                'chemical_strain': chemical_strain_measured,
                'chemical_expansion_coefficient': chemical_strain_measured / np.log10(pO2 / 1e-15) if pO2 > 1e-15 else 0,
                'oxidation_fraction': oxidation_fraction if 'Ni' in self.material_name else 0
            })
        
        return pd.DataFrame(data_points)

def generate_comprehensive_experimental_dataset(material_database) -> Dict:
    """Generate comprehensive experimental dataset for all materials."""
    
    experimental_data = {}
    
    # Materials to test
    materials_to_test = ['YSZ', 'Ni', 'Ni-YSZ_40', 'Ni-YSZ_50']
    
    for material_name in materials_to_test:
        print(f"Generating experimental data for {material_name}...")
        
        material_props = material_database.get_material_properties(material_name)
        
        # Extract properties for data generation
        elastic_props = material_props['elastic_1073K']
        fracture_props = material_props['fracture']
        thermal_props = material_props['thermal']
        chemical_props = material_props['chemical_expansion']
        
        true_properties = {
            'youngs_modulus_GPa': elastic_props.youngs_modulus_GPa.value,
            'poissons_ratio': elastic_props.poissons_ratio.value,
            'fracture_toughness_MPa_sqrt_m': fracture_props.fracture_toughness_MPa_sqrt_m.value,
            'critical_energy_release_rate_J_m2': fracture_props.critical_energy_release_rate_J_m2.value,
            'thermal_expansion_coefficient_K_inv': thermal_props.thermal_expansion_coefficient_K_inv.value
        }
        
        # Generate nanoindentation data
        nano_gen = NanoindentationDataGenerator(material_name, true_properties)
        nano_raw_data = nano_gen.generate_load_displacement_curves(n_curves=30)
        nano_analysis = nano_gen.analyze_nanoindentation_data(nano_raw_data)
        
        # Generate fracture data
        fracture_gen = FractureTestingDataGenerator(material_name, true_properties)
        ct_data = fracture_gen.generate_compact_tension_data(n_specimens=12)
        
        # Generate thermal data
        thermal_gen = ThermalAnalysisDataGenerator(material_name, true_properties)
        dilatometry_data = thermal_gen.generate_dilatometry_data()
        chemical_expansion_data = thermal_gen.generate_chemical_expansion_data()
        
        experimental_data[material_name] = {
            'nanoindentation': {
                'raw_data': nano_raw_data,
                'analysis_results': nano_analysis
            },
            'fracture_testing': {
                'compact_tension': ct_data
            },
            'thermal_analysis': {
                'dilatometry': dilatometry_data,
                'chemical_expansion': chemical_expansion_data
            },
            'true_properties': true_properties
        }
    
    # Generate interface data
    print("Generating interface fracture data...")
    interface_materials = ['anode_electrolyte', 'Ni_YSZ']
    
    for interface_name in interface_materials:
        interface_props = material_database.get_interface_properties(interface_name)
        
        true_interface_props = {
            'fracture_toughness_MPa_sqrt_m': interface_props['fracture'].fracture_toughness_MPa_sqrt_m.value,
            'critical_energy_release_rate_J_m2': interface_props['fracture'].critical_energy_release_rate_J_m2.value
        }
        
        fracture_gen = FractureTestingDataGenerator(f"Interface_{interface_name}", true_interface_props)
        interface_fracture_data = fracture_gen.generate_interface_fracture_data(interface_name, n_specimens=10)
        
        experimental_data[f"Interface_{interface_name}"] = {
            'interface_fracture': interface_fracture_data,
            'true_properties': true_interface_props
        }
    
    return experimental_data

def export_experimental_data(experimental_data: Dict, base_filename: str = "experimental_data"):
    """Export experimental data to multiple formats."""
    
    # Export to CSV files
    for material_name, data in experimental_data.items():
        safe_name = material_name.replace('/', '_').replace(' ', '_')
        
        for test_type, test_data in data.items():
            if test_type == 'true_properties':
                continue
                
            if isinstance(test_data, dict):
                for sub_test, sub_data in test_data.items():
                    if isinstance(sub_data, pd.DataFrame):
                        filename = f"{base_filename}_{safe_name}_{test_type}_{sub_test}.csv"
                        sub_data.to_csv(filename, index=False)
                        print(f"Exported {filename}")
            elif isinstance(test_data, pd.DataFrame):
                filename = f"{base_filename}_{safe_name}_{test_type}.csv"
                test_data.to_csv(filename, index=False)
                print(f"Exported {filename}")
    
    # Export summary statistics
    summary_stats = {}
    
    for material_name, data in experimental_data.items():
        if 'nanoindentation' in data:
            nano_stats = data['nanoindentation']['analysis_results']['statistics']
            summary_stats[material_name] = {
                'measured_E_GPa': nano_stats['youngs_modulus']['mean_GPa'],
                'measured_E_std_GPa': nano_stats['youngs_modulus']['std_GPa'],
                'true_E_GPa': data['true_properties']['youngs_modulus_GPa'],
                'E_error_percent': abs(nano_stats['youngs_modulus']['mean_GPa'] - 
                                     data['true_properties']['youngs_modulus_GPa']) / 
                                     data['true_properties']['youngs_modulus_GPa'] * 100
            }
    
    summary_df = pd.DataFrame(summary_stats).T
    summary_df.to_csv(f"{base_filename}_summary_statistics.csv")
    print(f"Exported {base_filename}_summary_statistics.csv")
    
    return summary_stats

if __name__ == "__main__":
    # This will be run when the main database is executed
    print("Experimental Data Generator Module Loaded")
    print("Use generate_comprehensive_experimental_dataset() to create synthetic experimental data")