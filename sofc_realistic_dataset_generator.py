#!/usr/bin/env python3
"""
Realistic SOFC Electrochemical Loading Dataset Generator

This module generates comprehensive and realistic electrochemical datasets for 
Solid Oxide Fuel Cells (SOFCs) with proper parameter scaling and physical constraints.

Author: Generated for SOFC Research
Date: 2025-10-09
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Physical constants
F = 96485.33212  # Faraday constant (C/mol)
R = 8.314462618  # Universal gas constant (J/mol/K)

@dataclass
class SOFCOperatingConditions:
    """Operating conditions for SOFC electrochemical testing"""
    temperature: float  # K
    pressure_anode: float  # Pa
    pressure_cathode: float  # Pa
    fuel_composition: Dict[str, float]  # mole fractions
    air_composition: Dict[str, float]  # mole fractions

@dataclass
class SOFCGeometry:
    """SOFC cell geometry parameters"""
    electrolyte_thickness: float  # m
    anode_thickness: float  # m
    cathode_thickness: float  # m
    active_area: float  # m²

class RealisticSOFCGenerator:
    """Generator for realistic SOFC electrochemical loading datasets"""
    
    def __init__(self, 
                 operating_conditions: SOFCOperatingConditions,
                 geometry: SOFCGeometry):
        self.conditions = operating_conditions
        self.geometry = geometry
        
    def calculate_nernst_potential(self) -> float:
        """Calculate theoretical Nernst potential"""
        T = self.conditions.temperature
        
        # Partial pressures
        p_H2 = self.conditions.pressure_anode * self.conditions.fuel_composition.get('H2', 0.97)
        p_H2O = self.conditions.pressure_anode * self.conditions.fuel_composition.get('H2O', 0.03)
        p_O2 = self.conditions.pressure_cathode * self.conditions.air_composition.get('O2', 0.21)
        
        # Nernst equation
        E_nernst = 1.253 - 2.4516e-4 * T + (R * T / (2 * F)) * np.log(p_H2 * np.sqrt(p_O2) / p_H2O)
        
        return E_nernst
    
    def calculate_ohmic_resistance(self) -> float:
        """Calculate area-specific resistance (ASR) for electrolyte"""
        T = self.conditions.temperature
        
        # YSZ conductivity with Arrhenius dependence
        # σ = σ₀ * exp(-Ea/RT) where σ₀ = 3.34e4 S/m, Ea = 80 kJ/mol
        sigma_ysz = 3.34e4 * np.exp(-80000 / (R * T))  # S/m
        
        # Area-specific resistance
        ASR_ohmic = self.geometry.electrolyte_thickness / sigma_ysz  # Ω⋅m²
        
        return ASR_ohmic
    
    def calculate_anode_overpotential(self, current_density: float) -> Tuple[float, float]:
        """Calculate anode overpotential and Ni oxidation risk"""
        T = self.conditions.temperature
        
        # Realistic anode kinetics parameters
        i0_anode = 3000  # A/m² - exchange current density at 800°C
        alpha_anode = 0.5  # charge transfer coefficient
        
        if current_density > 0.1:
            # Butler-Volmer equation (simplified for high current densities)
            eta_anode = (R * T / (alpha_anode * F)) * np.log(current_density / i0_anode)
        else:
            eta_anode = 0
        
        # Ni oxidation risk assessment
        # Based on local oxygen partial pressure increase with current density
        p_O2_base = 1e-20  # Base oxygen partial pressure at anode
        p_O2_local = p_O2_base * np.exp(current_density / 8000)  # Gradual increase
        
        # Ni/NiO equilibrium potential (vs. reference)
        E_NiNiO = -0.234 + (R * T / (2 * F)) * np.log(p_O2_local / 1e5)  # vs. 1 bar O2
        
        # Oxidation risk (positive values indicate risk)
        oxidation_risk = max(0, eta_anode + E_NiNiO + 0.1)  # Offset for realistic threshold
        
        return eta_anode, oxidation_risk
    
    def calculate_cathode_overpotential(self, current_density: float) -> float:
        """Calculate cathode overpotential"""
        T = self.conditions.temperature
        
        # Realistic cathode kinetics parameters
        i0_cathode = 1500  # A/m² - exchange current density at 800°C
        alpha_cathode = 0.5  # charge transfer coefficient
        
        if current_density > 0.1:
            # Butler-Volmer equation
            eta_cathode = (R * T / (alpha_cathode * 2 * F)) * np.log(current_density / i0_cathode)
        else:
            eta_cathode = 0
            
        return eta_cathode
    
    def calculate_concentration_overpotential(self, current_density: float) -> Tuple[float, float]:
        """Calculate concentration overpotentials"""
        # Limiting current densities
        i_limit_anode = 12000  # A/m²
        i_limit_cathode = 15000  # A/m²
        
        # Concentration overpotentials (simplified model)
        if current_density < i_limit_anode * 0.9:
            eta_conc_anode = (R * self.conditions.temperature / (2 * F)) * \
                           (current_density / i_limit_anode)**2
        else:
            eta_conc_anode = 0.2  # Cap at reasonable value
            
        if current_density < i_limit_cathode * 0.9:
            eta_conc_cathode = (R * self.conditions.temperature / (4 * F)) * \
                             (current_density / i_limit_cathode)**2
        else:
            eta_conc_cathode = 0.15  # Cap at reasonable value
        
        return eta_conc_anode, eta_conc_cathode
    
    def generate_iv_curve(self, current_range: np.ndarray) -> Dict:
        """Generate realistic IV curve data"""
        E_nernst = self.calculate_nernst_potential()
        ASR_ohmic = self.calculate_ohmic_resistance()
        
        voltages = []
        overpotentials = {
            'anode_activation': [],
            'cathode_activation': [],
            'anode_concentration': [],
            'cathode_concentration': [],
            'ohmic': [],
            'ni_oxidation_risk': []
        }
        
        for i in current_range:
            # Calculate overpotentials
            eta_a_act, ni_ox_risk = self.calculate_anode_overpotential(i)
            eta_c_act = self.calculate_cathode_overpotential(i)
            eta_a_conc, eta_c_conc = self.calculate_concentration_overpotential(i)
            
            # Ohmic overpotential
            eta_ohmic = i * ASR_ohmic  # V
            
            # Total voltage
            V_cell = E_nernst - eta_a_act - eta_c_act - eta_a_conc - eta_c_conc - eta_ohmic
            
            # Ensure voltage doesn't go below reasonable minimum
            V_cell = max(V_cell, 0.1)
            
            voltages.append(V_cell)
            overpotentials['anode_activation'].append(eta_a_act)
            overpotentials['cathode_activation'].append(eta_c_act)
            overpotentials['anode_concentration'].append(eta_a_conc)
            overpotentials['cathode_concentration'].append(eta_c_conc)
            overpotentials['ohmic'].append(eta_ohmic)
            overpotentials['ni_oxidation_risk'].append(ni_ox_risk)
        
        return {
            'current_density': current_range.tolist(),
            'voltage': voltages,
            'nernst_potential': E_nernst,
            'overpotentials': overpotentials,
            'power_density': (np.array(voltages) * current_range).tolist(),
            'asr_ohmic': ASR_ohmic
        }
    
    def generate_eis_data(self, frequencies: np.ndarray, current_density: float = 5000) -> Dict:
        """Generate realistic EIS data"""
        T = self.conditions.temperature
        
        # Equivalent circuit parameters (realistic values for SOFC at 800°C)
        R_ohmic = self.calculate_ohmic_resistance()  # Ω⋅m²
        
        # Anode arc parameters
        R_anode = 0.08e-4  # Ω⋅m²
        C_anode = 0.05  # F⋅m⁻²
        n_anode = 0.85  # CPE exponent
        
        # Cathode arc parameters  
        R_cathode = 0.12e-4  # Ω⋅m²
        C_cathode = 0.15  # F⋅m⁻²
        n_cathode = 0.75  # CPE exponent
        
        # Gas diffusion (Warburg) parameters
        sigma_w = 0.015e-4  # Ω⋅s⁻⁰·⁵⋅m²
        
        omega = 2 * np.pi * frequencies
        
        # Calculate impedance components
        Z_ohmic = R_ohmic
        
        # Anode impedance (R-CPE parallel combination)
        Z_cpe_anode = 1 / (C_anode * (1j * omega)**n_anode)
        Z_anode = (R_anode * Z_cpe_anode) / (R_anode + Z_cpe_anode)
        
        # Cathode impedance (R-CPE parallel combination)
        Z_cpe_cathode = 1 / (C_cathode * (1j * omega)**n_cathode)
        Z_cathode = (R_cathode * Z_cpe_cathode) / (R_cathode + Z_cpe_cathode)
        
        # Warburg impedance
        Z_warburg = sigma_w * (1 - 1j) / np.sqrt(omega)
        
        # Total impedance
        Z_total = Z_ohmic + Z_anode + Z_cathode + Z_warburg
        
        return {
            'frequency': frequencies.tolist(),
            'impedance_real': np.real(Z_total).tolist(),
            'impedance_imag': np.imag(Z_total).tolist(),
            'impedance_magnitude': np.abs(Z_total).tolist(),
            'phase_angle': np.angle(Z_total, deg=True).tolist(),
            'circuit_parameters': {
                'R_ohmic': R_ohmic,
                'R_anode': R_anode,
                'R_cathode': R_cathode,
                'C_anode': C_anode,
                'C_cathode': C_cathode,
                'n_anode': n_anode,
                'n_cathode': n_cathode,
                'sigma_warburg': sigma_w
            }
        }
    
    def calculate_oxygen_chemical_potential_gradient(self, current_density: float) -> Dict:
        """Calculate oxygen chemical potential gradient across electrolyte"""
        T = self.conditions.temperature
        
        # Oxygen partial pressures
        p_O2_cathode = self.conditions.pressure_cathode * \
                      self.conditions.air_composition.get('O2', 0.21)
        
        # Anode side - equilibrium with fuel
        p_H2 = self.conditions.pressure_anode * \
               self.conditions.fuel_composition.get('H2', 0.97)
        p_H2O = self.conditions.pressure_anode * \
                self.conditions.fuel_composition.get('H2O', 0.03)
        
        # Water-gas shift equilibrium
        K_eq = np.exp(-(241800 - 44.3 * T) / (R * T))
        p_O2_anode = (p_H2O / (p_H2 * np.sqrt(K_eq)))**2
        
        # Effect of current density on local oxygen partial pressures
        p_O2_anode_local = p_O2_anode * np.exp(current_density / 10000)
        
        # Chemical potentials
        mu_O2_cathode = R * T * np.log(p_O2_cathode / 1e5)
        mu_O2_anode = R * T * np.log(p_O2_anode_local / 1e5)
        
        # Gradient across electrolyte
        gradient = (mu_O2_cathode - mu_O2_anode) / self.geometry.electrolyte_thickness
        
        # Position-dependent profile
        positions = np.linspace(0, self.geometry.electrolyte_thickness, 50)
        mu_profile = mu_O2_anode + (mu_O2_cathode - mu_O2_anode) * \
                    positions / self.geometry.electrolyte_thickness
        
        return {
            'mu_O2_anode': mu_O2_anode,
            'mu_O2_cathode': mu_O2_cathode,
            'gradient': gradient,
            'positions': positions.tolist(),
            'mu_profile': mu_profile.tolist(),
            'p_O2_anode': p_O2_anode_local,
            'p_O2_cathode': p_O2_cathode,
            'driving_force': mu_O2_cathode - mu_O2_anode
        }
    
    def calculate_ni_oxidation_stress(self, current_density: float, ni_oxidation_risk: float) -> Dict:
        """Calculate volume change and stress from Ni to NiO conversion"""
        
        # Ni volume fraction in anode
        ni_volume_fraction = 0.35
        
        # Volume expansion upon oxidation (NiO has ~1.7x volume of Ni)
        volume_expansion_factor = 0.7
        
        # Oxidation fraction based on oxidation risk
        if ni_oxidation_risk > 0.01:
            oxidation_fraction = min(1.0, (ni_oxidation_risk - 0.01) / 0.1)
        else:
            oxidation_fraction = 0
        
        # Volumetric strain
        volumetric_strain = oxidation_fraction * volume_expansion_factor * ni_volume_fraction
        
        # Stress calculation (elastic model)
        E_anode = 55e9  # Pa - Young's modulus of Ni-YSZ
        nu_anode = 0.29  # Poisson's ratio
        
        # Hydrostatic stress from constrained expansion
        sigma_hydrostatic = E_anode * volumetric_strain / (3 * (1 - 2 * nu_anode))
        
        # Von Mises equivalent stress
        sigma_von_mises = sigma_hydrostatic
        
        # Stress transmitted to electrolyte (simplified)
        stress_transmission = 0.25  # 25% of anode stress transmitted
        sigma_electrolyte = sigma_von_mises * stress_transmission
        
        # Risk assessment
        if ni_oxidation_risk > 0.05:
            risk_level = 'High'
        elif ni_oxidation_risk > 0.02:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'oxidation_fraction': oxidation_fraction,
            'volumetric_strain': volumetric_strain,
            'hydrostatic_stress': sigma_hydrostatic,
            'von_mises_stress': sigma_von_mises,
            'electrolyte_stress': sigma_electrolyte,
            'oxidation_risk_level': risk_level,
            'ni_volume_fraction': ni_volume_fraction,
            'volume_expansion_factor': volume_expansion_factor
        }
    
    def generate_complete_dataset(self, 
                                  current_range: Optional[np.ndarray] = None,
                                  frequency_range: Optional[np.ndarray] = None) -> Dict:
        """Generate complete realistic electrochemical dataset"""
        
        if current_range is None:
            current_range = np.linspace(0, 10000, 101)  # A/m²
        
        if frequency_range is None:
            frequency_range = np.logspace(-2, 6, 50)  # 0.01 Hz to 1 MHz
        
        print("Generating Realistic SOFC Electrochemical Loading Dataset...")
        print(f"Operating Temperature: {self.conditions.temperature:.1f} K ({self.conditions.temperature-273.15:.1f} °C)")
        print(f"Current Density Range: {current_range[0]:.0f} - {current_range[-1]:.0f} A/m²")
        print(f"Electrolyte Thickness: {self.geometry.electrolyte_thickness*1e6:.0f} μm")
        
        # Generate IV curve data
        print("Generating IV curve data...")
        iv_data = self.generate_iv_curve(current_range)
        
        # Generate EIS data at multiple current densities
        print("Generating EIS data...")
        eis_data = {}
        test_currents = [0, 2000, 5000, 8000]  # A/m²
        for i_test in test_currents:
            eis_data[f'current_{i_test}'] = self.generate_eis_data(frequency_range, i_test)
        
        # Generate detailed analysis at operating points
        print("Calculating chemical gradients and stress analysis...")
        operating_points = [1000, 3000, 5000, 7000]  # A/m²
        detailed_analysis = {}
        
        for i_op in operating_points:
            eta_a_act, ni_ox_risk = self.calculate_anode_overpotential(i_op)
            chemical_potential = self.calculate_oxygen_chemical_potential_gradient(i_op)
            stress_analysis = self.calculate_ni_oxidation_stress(i_op, ni_ox_risk)
            
            detailed_analysis[f'current_{i_op}'] = {
                'current_density': i_op,
                'anode_overpotential': eta_a_act,
                'ni_oxidation_risk': ni_ox_risk,
                'chemical_potential_gradient': chemical_potential,
                'stress_analysis': stress_analysis
            }
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(iv_data, detailed_analysis)
        
        # Compile complete dataset
        complete_dataset = {
            'metadata': {
                'description': 'Realistic SOFC Electrochemical Loading Dataset',
                'generated_date': '2025-10-09',
                'temperature_C': self.conditions.temperature - 273.15,
                'electrolyte_thickness_um': self.geometry.electrolyte_thickness * 1e6,
                'active_area_cm2': self.geometry.active_area * 1e4,
                'operating_conditions': asdict(self.conditions),
                'geometry': asdict(self.geometry)
            },
            'iv_curve': iv_data,
            'eis_data': eis_data,
            'detailed_analysis': detailed_analysis,
            'summary_statistics': summary_stats
        }
        
        return complete_dataset
    
    def _calculate_summary_statistics(self, iv_data: Dict, detailed_analysis: Dict) -> Dict:
        """Calculate summary statistics"""
        
        # Find maximum power point
        power_density = np.array(iv_data['power_density'])
        max_power_idx = np.argmax(power_density)
        
        max_power_density = power_density[max_power_idx]
        max_power_current = iv_data['current_density'][max_power_idx]
        max_power_voltage = iv_data['voltage'][max_power_idx]
        
        # Overpotential statistics
        ohmic_losses = np.array(iv_data['overpotentials']['ohmic'])
        activation_losses = np.array(iv_data['overpotentials']['anode_activation']) + \
                          np.array(iv_data['overpotentials']['cathode_activation'])
        
        # Ni oxidation risk assessment
        oxidation_risks = [detailed_analysis[key]['ni_oxidation_risk'] 
                          for key in detailed_analysis.keys()]
        max_oxidation_risk = max(oxidation_risks)
        
        return {
            'max_power_density': float(max_power_density),
            'max_power_current_density': float(max_power_current),
            'max_power_voltage': float(max_power_voltage),
            'nernst_potential': iv_data['nernst_potential'],
            'asr_ohmic': iv_data['asr_ohmic'],
            'max_ohmic_loss': float(np.max(ohmic_losses)),
            'max_activation_loss': float(np.max(activation_losses)),
            'max_ni_oxidation_risk': max_oxidation_risk,
            'operating_temperature_C': self.conditions.temperature - 273.15,
            'electrolyte_thickness_um': self.geometry.electrolyte_thickness * 1e6
        }

def create_realistic_sofc_parameters():
    """Create realistic SOFC parameters"""
    
    # Operating conditions (800°C)
    operating_conditions = SOFCOperatingConditions(
        temperature=1073.15,  # 800°C
        pressure_anode=101325,  # 1 atm
        pressure_cathode=101325,  # 1 atm
        fuel_composition={'H2': 0.97, 'H2O': 0.03},
        air_composition={'O2': 0.21, 'N2': 0.79}
    )
    
    # Cell geometry
    geometry = SOFCGeometry(
        electrolyte_thickness=150e-6,  # 150 μm
        anode_thickness=300e-6,  # 300 μm
        cathode_thickness=50e-6,  # 50 μm
        active_area=0.01  # 10 cm²
    )
    
    return operating_conditions, geometry

def export_realistic_dataset(dataset: Dict, output_dir: str = "sofc_realistic_data"):
    """Export realistic dataset to multiple formats"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nExporting dataset to {output_path}...")
    
    # Export complete dataset as JSON
    with open(output_path / "sofc_realistic_electrochemical_dataset.json", 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Export IV curve data as CSV
    iv_df = pd.DataFrame({
        'Current_Density_A_per_m2': dataset['iv_curve']['current_density'],
        'Voltage_V': dataset['iv_curve']['voltage'],
        'Power_Density_W_per_m2': dataset['iv_curve']['power_density'],
        'Anode_Overpotential_V': dataset['iv_curve']['overpotentials']['anode_activation'],
        'Cathode_Overpotential_V': dataset['iv_curve']['overpotentials']['cathode_activation'],
        'Ohmic_Overpotential_V': dataset['iv_curve']['overpotentials']['ohmic'],
        'Ni_Oxidation_Risk_V': dataset['iv_curve']['overpotentials']['ni_oxidation_risk'],
        'Anode_Conc_Overpotential_V': dataset['iv_curve']['overpotentials']['anode_concentration'],
        'Cathode_Conc_Overpotential_V': dataset['iv_curve']['overpotentials']['cathode_concentration']
    })
    iv_df.to_csv(output_path / "iv_curve_realistic.csv", index=False)
    
    # Export EIS data
    for current_key, eis_data in dataset['eis_data'].items():
        eis_df = pd.DataFrame({
            'Frequency_Hz': eis_data['frequency'],
            'Impedance_Real_Ohm_m2': eis_data['impedance_real'],
            'Impedance_Imag_Ohm_m2': eis_data['impedance_imag'],
            'Impedance_Magnitude_Ohm_m2': eis_data['impedance_magnitude'],
            'Phase_Angle_deg': eis_data['phase_angle']
        })
        eis_df.to_csv(output_path / f"eis_realistic_{current_key}.csv", index=False)
    
    # Export detailed analysis
    detailed_data = []
    for current_key, analysis in dataset['detailed_analysis'].items():
        row = {
            'Current_Density_A_per_m2': analysis['current_density'],
            'Anode_Overpotential_V': analysis['anode_overpotential'],
            'Ni_Oxidation_Risk_V': analysis['ni_oxidation_risk'],
            'O2_Chemical_Potential_Gradient_J_per_mol_per_m': analysis['chemical_potential_gradient']['gradient'],
            'Oxidation_Fraction': analysis['stress_analysis']['oxidation_fraction'],
            'Volumetric_Strain': analysis['stress_analysis']['volumetric_strain'],
            'Von_Mises_Stress_Pa': analysis['stress_analysis']['von_mises_stress'],
            'Electrolyte_Stress_Pa': analysis['stress_analysis']['electrolyte_stress'],
            'Risk_Level': analysis['stress_analysis']['oxidation_risk_level']
        }
        detailed_data.append(row)
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(output_path / "detailed_realistic_analysis.csv", index=False)
    
    print(f"Realistic dataset exported successfully!")
    print(f"Files created:")
    for file in output_path.glob("*.csv"):
        print(f"  - {file.name}")
    print(f"  - sofc_realistic_electrochemical_dataset.json")

def plot_realistic_dataset(dataset: Dict, output_dir: str = "sofc_realistic_data"):
    """Generate plots for realistic dataset"""
    
    output_path = Path(output_dir)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Realistic SOFC Electrochemical Loading Dataset', fontsize=16, fontweight='bold')
    
    # IV Curve and Power
    ax = axes[0, 0]
    current = np.array(dataset['iv_curve']['current_density']) / 1000  # A/cm²
    voltage = dataset['iv_curve']['voltage']
    power = np.array(dataset['iv_curve']['power_density']) / 10000  # W/cm²
    
    ax.plot(current, voltage, 'b-', linewidth=2, label='Voltage')
    ax2 = ax.twinx()
    ax2.plot(current, power, 'r-', linewidth=2, label='Power Density')
    ax.set_xlabel('Current Density (A/cm²)')
    ax.set_ylabel('Voltage (V)', color='b')
    ax2.set_ylabel('Power Density (W/cm²)', color='r')
    ax.set_title('IV and Power Curves')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.5)
    
    # Overpotential Breakdown
    ax = axes[0, 1]
    overpotentials = dataset['iv_curve']['overpotentials']
    ax.plot(current, overpotentials['anode_activation'], 'g-', label='Anode Activation', linewidth=2)
    ax.plot(current, overpotentials['cathode_activation'], 'orange', label='Cathode Activation', linewidth=2)
    ax.plot(current, overpotentials['ohmic'], 'r-', label='Ohmic', linewidth=2)
    ax.plot(current, overpotentials['ni_oxidation_risk'], 'm--', label='Ni Oxidation Risk', linewidth=2)
    ax.set_xlabel('Current Density (A/cm²)')
    ax.set_ylabel('Overpotential (V)')
    ax.set_title('Overpotential Breakdown')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # EIS Nyquist Plot
    ax = axes[0, 2]
    colors = ['blue', 'green', 'orange', 'red']
    for i, (current_key, eis_data) in enumerate(dataset['eis_data'].items()):
        real_z = np.array(eis_data['impedance_real']) * 10000  # Ω⋅cm²
        imag_z = -np.array(eis_data['impedance_imag']) * 10000  # Ω⋅cm²
        current_val = current_key.split('_')[1]
        ax.plot(real_z, imag_z, 'o-', color=colors[i], markersize=3, 
                label=f'{int(current_val)/1000:.1f} A/cm²')
    ax.set_xlabel('Real Impedance (Ω⋅cm²)')
    ax.set_ylabel('-Imaginary Impedance (Ω⋅cm²)')
    ax.set_title('EIS Nyquist Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Chemical Potential Gradient
    ax = axes[1, 0]
    for current_key, analysis in dataset['detailed_analysis'].items():
        current_val = int(current_key.split('_')[1])
        positions = np.array(analysis['chemical_potential_gradient']['positions']) * 1e6  # μm
        mu_profile = analysis['chemical_potential_gradient']['mu_profile']
        ax.plot(positions, np.array(mu_profile)/1000, label=f'{current_val/1000:.1f} A/cm²')
    ax.set_xlabel('Position in Electrolyte (μm)')
    ax.set_ylabel('O₂ Chemical Potential (kJ/mol)')
    ax.set_title('O₂ Chemical Potential Gradient')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ni Oxidation and Stress
    ax = axes[1, 1]
    currents_detail = []
    oxidation_fractions = []
    stresses = []
    
    for current_key, analysis in dataset['detailed_analysis'].items():
        current_val = int(current_key.split('_')[1])
        currents_detail.append(current_val / 1000)  # A/cm²
        oxidation_fractions.append(analysis['stress_analysis']['oxidation_fraction'] * 100)
        stresses.append(analysis['stress_analysis']['electrolyte_stress'] / 1e6)  # MPa
    
    ax.plot(currents_detail, oxidation_fractions, 'ro-', linewidth=2, markersize=6, label='Ni Oxidation (%)')
    ax2 = ax.twinx()
    ax2.plot(currents_detail, stresses, 'bs-', linewidth=2, markersize=6, label='Electrolyte Stress (MPa)')
    ax.set_xlabel('Current Density (A/cm²)')
    ax.set_ylabel('Ni Oxidation Fraction (%)', color='r')
    ax2.set_ylabel('Electrolyte Stress (MPa)', color='b')
    ax.set_title('Ni Oxidation and Induced Stress')
    ax.grid(True, alpha=0.3)
    
    # Summary Statistics
    ax = axes[1, 2]
    ax.axis('off')
    stats = dataset['summary_statistics']
    stats_text = f"""
    SOFC Performance Summary
    
    Operating Temperature: {stats['operating_temperature_C']:.0f} °C
    Electrolyte Thickness: {stats['electrolyte_thickness_um']:.0f} μm
    
    Nernst Potential: {stats['nernst_potential']:.3f} V
    ASR (Ohmic): {stats['asr_ohmic']*1e4:.2f} Ω⋅cm²
    
    Max Power Density: {stats['max_power_density']/10000:.2f} W/cm²
    @ Current Density: {stats['max_power_current_density']/1000:.2f} A/cm²
    @ Voltage: {stats['max_power_voltage']:.3f} V
    
    Max Ohmic Loss: {stats['max_ohmic_loss']:.3f} V
    Max Activation Loss: {stats['max_activation_loss']:.3f} V
    Max Ni Oxidation Risk: {stats['max_ni_oxidation_risk']:.3f} V
    """
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path / "sofc_realistic_dataset_overview.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Overview plot saved to {output_path / 'sofc_realistic_dataset_overview.png'}")

def main():
    """Main function to generate realistic SOFC dataset"""
    
    print("="*70)
    print("REALISTIC SOFC ELECTROCHEMICAL LOADING DATASET GENERATOR")
    print("="*70)
    
    # Create parameters
    operating_conditions, geometry = create_realistic_sofc_parameters()
    
    # Initialize generator
    generator = RealisticSOFCGenerator(operating_conditions, geometry)
    
    # Generate dataset
    dataset = generator.generate_complete_dataset()
    
    # Export dataset
    export_realistic_dataset(dataset)
    
    # Generate plots
    plot_realistic_dataset(dataset)
    
    print("\n" + "="*70)
    print("REALISTIC DATASET GENERATION COMPLETE!")
    print("="*70)
    
    stats = dataset['summary_statistics']
    print(f"Key Performance Metrics:")
    print(f"  - Maximum Power Density: {stats['max_power_density']/10000:.2f} W/cm²")
    print(f"  - ASR (Ohmic): {stats['asr_ohmic']*1e4:.2f} Ω⋅cm²")
    print(f"  - Maximum Ni Oxidation Risk: {stats['max_ni_oxidation_risk']:.3f} V")
    print(f"  - Operating Temperature: {stats['operating_temperature_C']:.0f} °C")
    print(f"  - Electrolyte Thickness: {stats['electrolyte_thickness_um']:.0f} μm")

if __name__ == "__main__":
    main()