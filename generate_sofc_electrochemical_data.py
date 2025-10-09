#!/usr/bin/env python3
"""
SOFC Electrochemical Loading Data Generator
Generates synthetic but realistic data for:
1. IV curves (current-voltage characteristics)
2. Electrochemical Impedance Spectroscopy (EIS)
3. Overpotentials (anode, cathode, ohmic)
4. Oxygen chemical potential gradients
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime

# Physical constants
F = 96485  # Faraday constant (C/mol)
R = 8.314  # Gas constant (J/mol·K)
T_OPERATING = 1073.15  # Operating temperature (800°C in Kelvin)

class SOFCElectrochemicalDataGenerator:
    """Generate realistic SOFC electrochemical data"""
    
    def __init__(self, temp_celsius=800):
        self.temp_k = temp_celsius + 273.15
        self.temp_c = temp_celsius
        
        # SOFC material and operating parameters
        self.E_ocv = 1.05  # Open circuit voltage at 800°C (V)
        self.R_ohmic = 0.15  # Ohmic resistance (Ω·cm²)
        self.i0_anode = 5000  # Anode exchange current density (A/m²)
        self.i0_cathode = 1000  # Cathode exchange current density (A/m²)
        self.alpha_anode = 0.5  # Anode charge transfer coefficient
        self.alpha_cathode = 0.5  # Cathode charge transfer coefficient
        self.L_diffusion = 50e-6  # Diffusion length (m)
        self.D_eff = 1e-5  # Effective diffusion coefficient (m²/s)
        
    def generate_iv_curve(self, i_max=15000, n_points=100):
        """
        Generate IV curve data
        Returns: DataFrame with current density and voltage
        """
        # Current density range (A/m²)
        current_density = np.linspace(0, i_max, n_points)
        
        # Calculate voltage components
        voltage = np.zeros(n_points)
        eta_anode = np.zeros(n_points)
        eta_cathode = np.zeros(n_points)
        eta_ohmic = np.zeros(n_points)
        eta_conc = np.zeros(n_points)
        
        for i in range(n_points):
            i_cd = current_density[i]
            
            # Activation overpotentials (Butler-Volmer)
            if i_cd > 0:
                eta_anode[i] = (R * self.temp_k / (self.alpha_anode * F)) * \
                              np.log(i_cd / self.i0_anode + 1)
                eta_cathode[i] = (R * self.temp_k / (self.alpha_cathode * F)) * \
                                np.log(i_cd / self.i0_cathode + 1)
            
            # Ohmic overpotential
            eta_ohmic[i] = i_cd * self.R_ohmic / 10000  # Convert to V
            
            # Concentration overpotential (simplified)
            i_lim = 20000  # Limiting current density (A/m²)
            if i_cd < i_lim * 0.95:
                eta_conc[i] = (R * self.temp_k / (4 * F)) * \
                             np.log(1 / (1 - i_cd / i_lim))
            else:
                eta_conc[i] = eta_conc[i-1] * 1.5  # Rapid increase near limit
            
            # Total voltage
            voltage[i] = self.E_ocv - eta_anode[i] - eta_cathode[i] - \
                        eta_ohmic[i] - eta_conc[i]
        
        # Convert to common units (current density in A/cm²)
        df = pd.DataFrame({
            'Current_Density_A_cm2': current_density / 10000,
            'Voltage_V': voltage,
            'Overpotential_Anode_V': eta_anode,
            'Overpotential_Cathode_V': eta_cathode,
            'Overpotential_Ohmic_V': eta_ohmic,
            'Overpotential_Concentration_V': eta_conc,
            'Power_Density_W_cm2': (current_density / 10000) * voltage,
            'Temperature_C': self.temp_c
        })
        
        return df
    
    def generate_eis_data(self, current_density_list=[0.0, 0.3, 0.5, 0.8, 1.0]):
        """
        Generate Electrochemical Impedance Spectroscopy data
        Returns: DataFrame with frequency, real and imaginary impedance
        """
        # Frequency range (Hz)
        freq = np.logspace(-2, 5, 100)  # 0.01 Hz to 100 kHz
        omega = 2 * np.pi * freq
        
        eis_data = []
        
        for i_cd in current_density_list:
            # Equivalent circuit parameters (vary with current)
            R_s = 0.10 + 0.02 * i_cd  # Series resistance (Ω·cm²)
            R_ct_anode = 0.15 - 0.05 * i_cd  # Anode charge transfer (Ω·cm²)
            R_ct_cathode = 0.25 - 0.08 * i_cd  # Cathode charge transfer (Ω·cm²)
            C_dl_anode = 0.02  # Anode double layer capacitance (F/cm²)
            C_dl_cathode = 0.015  # Cathode double layer capacitance (F/cm²)
            
            # Warburg impedance parameters
            sigma_w = 0.1 * (1 + i_cd)  # Warburg coefficient
            
            for f, w in zip(freq, omega):
                # Impedance components
                Z_Rs = R_s  # Series resistance
                
                # Anode RC element
                Z_anode_real = R_ct_anode / (1 + (w * R_ct_anode * C_dl_anode)**2)
                Z_anode_imag = -w * R_ct_anode**2 * C_dl_anode / \
                              (1 + (w * R_ct_anode * C_dl_anode)**2)
                
                # Cathode RC element
                Z_cathode_real = R_ct_cathode / (1 + (w * R_ct_cathode * C_dl_cathode)**2)
                Z_cathode_imag = -w * R_ct_cathode**2 * C_dl_cathode / \
                                (1 + (w * R_ct_cathode * C_dl_cathode)**2)
                
                # Warburg impedance (diffusion)
                Z_w_real = sigma_w / np.sqrt(w)
                Z_w_imag = -sigma_w / np.sqrt(w)
                
                # Total impedance
                Z_real = Z_Rs + Z_anode_real + Z_cathode_real + Z_w_real
                Z_imag = Z_anode_imag + Z_cathode_imag + Z_w_imag
                Z_magnitude = np.sqrt(Z_real**2 + Z_imag**2)
                Z_phase = np.arctan2(Z_imag, Z_real) * 180 / np.pi
                
                eis_data.append({
                    'Current_Density_A_cm2': i_cd,
                    'Frequency_Hz': f,
                    'Z_Real_Ohm_cm2': Z_real,
                    'Z_Imag_Ohm_cm2': Z_imag,
                    'Z_Magnitude_Ohm_cm2': Z_magnitude,
                    'Z_Phase_deg': Z_phase,
                    'Temperature_C': self.temp_c
                })
        
        return pd.DataFrame(eis_data)
    
    def generate_overpotential_stress_data(self, n_samples=50):
        """
        Generate overpotential data with associated stress from Ni oxidation
        Focus on anode overpotentials that can lead to Ni to NiO conversion
        """
        # Current density range
        current_density = np.linspace(0, 1.5, n_samples)  # A/cm²
        
        data = []
        
        for i_cd in current_density:
            # Calculate overpotentials
            if i_cd > 0:
                eta_anode = (R * self.temp_k / (self.alpha_anode * F)) * \
                           np.log(i_cd * 10000 / self.i0_anode + 1)
            else:
                eta_anode = 0
            
            # Oxygen partial pressure at anode (related to oxidation risk)
            # Higher overpotentials mean higher local oxygen activity
            P_O2_anode = 1e-20 * np.exp(eta_anode * 4 * F / (R * self.temp_k))
            
            # Critical oxygen partial pressure for NiO formation at 800°C
            P_O2_critical = 1e-15  # Pa (approximate)
            
            # Oxidation risk factor
            oxidation_risk = P_O2_anode / P_O2_critical if P_O2_anode > 0 else 0
            
            # Volume change stress from Ni to NiO conversion
            # NiO has ~68% larger volume than Ni
            volume_expansion = 0.68
            E_YSZ = 170e9  # Young's modulus of YSZ at 800°C (Pa)
            nu_YSZ = 0.23  # Poisson's ratio
            
            # Stress induced by volumetric expansion (simplified)
            # Assumes constrained expansion in electrolyte
            if oxidation_risk > 1.0:
                fraction_oxidized = min(0.3, (oxidation_risk - 1.0) * 0.05)
                epsilon_vol = fraction_oxidized * volume_expansion / 3
                stress_induced = E_YSZ / (1 - 2*nu_YSZ) * epsilon_vol / 1e6  # MPa
            else:
                fraction_oxidized = 0
                stress_induced = 0
            
            # Oxygen chemical potential gradient
            # Gradient across electrolyte drives ionic current
            delta_mu_O2 = -4 * F * i_cd * 10000 * self.R_ohmic  # J/mol
            
            # Gradient in oxygen partial pressure
            ln_ratio_PO2 = delta_mu_O2 / (R * self.temp_k)
            
            data.append({
                'Current_Density_A_cm2': i_cd,
                'Voltage_V': self.E_ocv - eta_anode - i_cd * self.R_ohmic / 10000,
                'Overpotential_Anode_V': eta_anode,
                'O2_Partial_Pressure_Anode_Pa': P_O2_anode,
                'O2_Partial_Pressure_Cathode_Pa': 21000,  # Air cathode
                'O2_Chemical_Potential_Gradient_J_mol': delta_mu_O2,
                'ln_PO2_Ratio_Cathode_Anode': ln_ratio_PO2,
                'Oxidation_Risk_Factor': oxidation_risk,
                'Ni_Fraction_Oxidized': fraction_oxidized,
                'Stress_Induced_MPa': stress_induced,
                'Temperature_C': self.temp_c
            })
        
        return pd.DataFrame(data)
    
    def generate_multi_temperature_data(self, temp_range=[700, 750, 800, 850]):
        """Generate IV curves at multiple temperatures"""
        all_data = []
        
        for temp in temp_range:
            generator = SOFCElectrochemicalDataGenerator(temp_celsius=temp)
            df = generator.generate_iv_curve(n_points=50)
            all_data.append(df)
        
        return pd.concat(all_data, ignore_index=True)
    
    def generate_time_series_degradation(self, duration_hours=1000, 
                                        current_density=0.5):
        """
        Generate time-series data showing performance degradation
        and increasing overpotentials over time
        """
        time_points = np.linspace(0, duration_hours, 100)
        
        # Degradation rates (per hour)
        k_anode = 5e-5  # Anode degradation rate
        k_cathode = 8e-5  # Cathode degradation rate
        k_ohmic = 3e-5  # Ohmic resistance increase rate
        
        data = []
        
        for t in time_points:
            # Increased resistances due to degradation
            R_ohmic_t = self.R_ohmic * (1 + k_ohmic * t)
            i0_anode_t = self.i0_anode * (1 - k_anode * t)
            i0_cathode_t = self.i0_cathode * (1 - k_cathode * t)
            
            # Overpotentials with degradation
            eta_anode = (R * self.temp_k / (self.alpha_anode * F)) * \
                       np.log(current_density * 10000 / i0_anode_t + 1)
            eta_cathode = (R * self.temp_k / (self.alpha_cathode * F)) * \
                         np.log(current_density * 10000 / i0_cathode_t + 1)
            eta_ohmic = current_density * R_ohmic_t / 10000
            
            voltage = self.E_ocv - eta_anode - eta_cathode - eta_ohmic
            
            data.append({
                'Time_hours': t,
                'Current_Density_A_cm2': current_density,
                'Voltage_V': voltage,
                'Overpotential_Anode_V': eta_anode,
                'Overpotential_Cathode_V': eta_cathode,
                'Overpotential_Ohmic_V': eta_ohmic,
                'R_Ohmic_Ohm_cm2': R_ohmic_t,
                'Power_Density_W_cm2': voltage * current_density,
                'Degradation_Rate_mV_per_kh': (self.E_ocv - voltage - 
                    (self.E_ocv - self.R_ohmic * current_density / 10000)) / t * 1000 if t > 0 else 0,
                'Temperature_C': self.temp_c
            })
        
        return pd.DataFrame(data)


def generate_complete_dataset():
    """Generate complete SOFC electrochemical loading dataset"""
    
    print("Generating SOFC Electrochemical Loading Dataset...")
    print("="*60)
    
    # Initialize generator
    generator = SOFCElectrochemicalDataGenerator(temp_celsius=800)
    
    # 1. Generate IV curve data
    print("\n1. Generating IV curve data...")
    iv_data = generator.generate_iv_curve(i_max=15000, n_points=100)
    print(f"   Generated {len(iv_data)} data points")
    iv_data.to_csv('sofc_iv_curve_800C.csv', index=False)
    
    # 2. Generate EIS data at multiple current densities
    print("\n2. Generating EIS data...")
    eis_data = generator.generate_eis_data(
        current_density_list=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    print(f"   Generated {len(eis_data)} data points")
    eis_data.to_csv('sofc_eis_data.csv', index=False)
    
    # 3. Generate overpotential and stress data
    print("\n3. Generating overpotential-stress coupling data...")
    overpotential_stress = generator.generate_overpotential_stress_data(n_samples=100)
    print(f"   Generated {len(overpotential_stress)} data points")
    overpotential_stress.to_csv('sofc_overpotential_stress_data.csv', index=False)
    
    # 4. Generate multi-temperature data
    print("\n4. Generating multi-temperature IV curves...")
    multi_temp_data = generator.generate_multi_temperature_data(
        temp_range=[650, 700, 750, 800, 850]
    )
    print(f"   Generated {len(multi_temp_data)} data points")
    multi_temp_data.to_csv('sofc_multi_temperature_iv_curves.csv', index=False)
    
    # 5. Generate time-series degradation data
    print("\n5. Generating degradation time-series data...")
    degradation_data = generator.generate_time_series_degradation(
        duration_hours=5000, 
        current_density=0.5
    )
    print(f"   Generated {len(degradation_data)} data points")
    degradation_data.to_csv('sofc_degradation_time_series.csv', index=False)
    
    # 6. Generate summary statistics and metadata
    print("\n6. Generating dataset metadata...")
    metadata = {
        'dataset_name': 'SOFC Electrochemical Loading Data',
        'generation_date': datetime.now().isoformat(),
        'temperature_range_C': [650, 850],
        'current_density_range_A_cm2': [0, 1.5],
        'description': {
            'iv_curve': 'Current-voltage characteristics showing operating voltage and current density',
            'eis': 'Electrochemical Impedance Spectroscopy data at multiple operating conditions',
            'overpotential_stress': 'Overpotentials (anode, cathode, ohmic) with induced stress from Ni oxidation',
            'multi_temperature': 'IV curves at various operating temperatures',
            'degradation': 'Time-series data showing performance degradation over operational lifetime'
        },
        'key_phenomena': {
            'oxygen_chemical_potential': 'Gradient across electrolyte drives ionic current',
            'anode_oxidation': 'High overpotentials can lead to local Ni to NiO oxidation',
            'volume_change_stress': 'NiO formation causes 68% volume expansion, inducing stress',
            'stress_coupling': 'Electrochemical loading couples to mechanical stress through oxidation'
        },
        'material_parameters': {
            'E_OCV_at_800C': '1.05 V',
            'R_ohmic': '0.15 Ω·cm²',
            'anode_exchange_current_density': '5000 A/m²',
            'cathode_exchange_current_density': '1000 A/m²'
        },
        'files_generated': [
            'sofc_iv_curve_800C.csv',
            'sofc_eis_data.csv',
            'sofc_overpotential_stress_data.csv',
            'sofc_multi_temperature_iv_curves.csv',
            'sofc_degradation_time_series.csv'
        ]
    }
    
    with open('sofc_dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 7. Generate summary statistics
    summary_stats = {
        'iv_curve_statistics': {
            'max_current_density_A_cm2': float(iv_data['Current_Density_A_cm2'].max()),
            'voltage_at_peak_power_V': float(iv_data.loc[iv_data['Power_Density_W_cm2'].idxmax(), 'Voltage_V']),
            'peak_power_density_W_cm2': float(iv_data['Power_Density_W_cm2'].max()),
            'max_anode_overpotential_V': float(iv_data['Overpotential_Anode_V'].max()),
            'max_cathode_overpotential_V': float(iv_data['Overpotential_Cathode_V'].max())
        },
        'stress_coupling_statistics': {
            'max_stress_induced_MPa': float(overpotential_stress['Stress_Induced_MPa'].max()),
            'max_oxidation_risk_factor': float(overpotential_stress['Oxidation_Risk_Factor'].max()),
            'max_O2_chemical_potential_gradient_J_mol': float(
                abs(overpotential_stress['O2_Chemical_Potential_Gradient_J_mol']).max()
            )
        },
        'degradation_statistics': {
            'voltage_degradation_mV_per_kh': float(degradation_data['Degradation_Rate_mV_per_kh'].iloc[-1]),
            'total_voltage_loss_mV': float((degradation_data['Voltage_V'].iloc[0] - 
                                           degradation_data['Voltage_V'].iloc[-1]) * 1000)
        }
    }
    
    with open('sofc_dataset_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print("\n" + "="*60)
    print("Dataset generation complete!")
    print("\nFiles created:")
    for file in metadata['files_generated']:
        print(f"  - {file}")
    print("  - sofc_dataset_metadata.json")
    print("  - sofc_dataset_summary.json")
    print("\nKey Results:")
    print(f"  Peak Power Density: {summary_stats['iv_curve_statistics']['peak_power_density_W_cm2']:.3f} W/cm²")
    print(f"  Max Induced Stress: {summary_stats['stress_coupling_statistics']['max_stress_induced_MPa']:.2f} MPa")
    print(f"  Degradation Rate: {summary_stats['degradation_statistics']['voltage_degradation_mV_per_kh']:.2f} mV/kh")
    
    return metadata, summary_stats


if __name__ == "__main__":
    metadata, summary = generate_complete_dataset()
