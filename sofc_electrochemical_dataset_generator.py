#!/usr/bin/env python3
"""
SOFC Electrochemical Loading Dataset Generator

This module generates comprehensive electrochemical datasets for Solid Oxide Fuel Cells (SOFCs)
including IV curves, Electrochemical Impedance Spectroscopy (EIS) data, overpotentials,
and oxygen chemical potential gradients.

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
k_B = 1.380649e-23  # Boltzmann constant (J/K)

@dataclass
class SOFCOperatingConditions:
    """Operating conditions for SOFC electrochemical testing"""
    temperature: float  # K
    pressure_anode: float  # Pa
    pressure_cathode: float  # Pa
    fuel_composition: Dict[str, float]  # mole fractions
    air_composition: Dict[str, float]  # mole fractions
    fuel_utilization: float  # fraction
    air_utilization: float  # fraction

@dataclass
class SOFCGeometry:
    """SOFC cell geometry parameters"""
    electrolyte_thickness: float  # m
    anode_thickness: float  # m
    cathode_thickness: float  # m
    active_area: float  # m²
    electrolyte_conductivity: float  # S/m at reference conditions
    anode_conductivity: float  # S/m
    cathode_conductivity: float  # S/m

@dataclass
class MaterialProperties:
    """Material properties for SOFC components"""
    # YSZ Electrolyte properties
    ysz_activation_energy: float  # J/mol
    ysz_pre_exponential: float  # S/m
    
    # Ni-YSZ Anode properties
    ni_volume_fraction: float
    ni_oxidation_threshold: float  # V vs reference
    ni_volume_expansion: float  # fractional expansion upon oxidation
    
    # LSM Cathode properties
    lsm_exchange_current: float  # A/m²
    lsm_activation_energy: float  # J/mol

class SOFCElectrochemicalGenerator:
    """Generator for SOFC electrochemical loading datasets"""
    
    def __init__(self, 
                 operating_conditions: SOFCOperatingConditions,
                 geometry: SOFCGeometry,
                 material_props: MaterialProperties):
        self.conditions = operating_conditions
        self.geometry = geometry
        self.materials = material_props
        
        # Initialize data storage
        self.iv_data = {}
        self.eis_data = {}
        self.overpotential_data = {}
        self.chemical_potential_data = {}
        self.stress_data = {}
        
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
    
    def calculate_electrolyte_resistance(self, current_density: float) -> float:
        """Calculate electrolyte ohmic resistance with temperature dependence"""
        T = self.conditions.temperature
        
        # YSZ conductivity with Arrhenius dependence - more realistic values
        sigma_ysz = 3.34e4 * np.exp(-80000 / (R * T))  # S/m
        
        # Ohmic resistance
        R_ohmic = self.geometry.electrolyte_thickness / (sigma_ysz * self.geometry.active_area)
        
        return R_ohmic
    
    def calculate_anode_overpotential(self, current_density: float) -> Tuple[float, float]:
        """Calculate anode overpotential and Ni oxidation risk"""
        T = self.conditions.temperature
        
        # Butler-Volmer kinetics for anode - more realistic parameters
        i0_anode = 8000 * np.exp(-100000 / (R * T))  # Exchange current density (A/m²)
        
        if current_density > 0:
            eta_anode = (R * T / (2 * F)) * np.arcsinh(current_density / (2 * i0_anode))
        else:
            eta_anode = 0
        
        # Ni oxidation risk assessment
        # Local oxygen partial pressure increases with current density
        p_O2_local = 1e-20 * np.exp(current_density / 5000)  # More gradual increase
        
        # Ni/NiO equilibrium potential
        E_NiNiO = -0.234 + (R * T / (2 * F)) * np.log(p_O2_local)
        
        # Oxidation risk (positive values indicate risk)
        oxidation_risk = eta_anode - E_NiNiO
        
        return eta_anode, max(0, oxidation_risk)
    
    def calculate_cathode_overpotential(self, current_density: float) -> float:
        """Calculate cathode overpotential"""
        T = self.conditions.temperature
        
        # LSM cathode kinetics - more realistic parameters
        i0_cathode = 5000 * np.exp(-110000 / (R * T))  # Exchange current density (A/m²)
        
        if current_density > 0:
            eta_cathode = (R * T / (4 * F)) * np.arcsinh(current_density / (2 * i0_cathode))
        else:
            eta_cathode = 0
            
        return eta_cathode
    
    def calculate_concentration_overpotential(self, current_density: float) -> Tuple[float, float]:
        """Calculate concentration overpotentials at anode and cathode"""
        # Simplified concentration overpotential model
        i_limit_anode = 15000  # A/m² - limiting current density at anode
        i_limit_cathode = 20000  # A/m² - limiting current density at cathode
        
        if current_density < i_limit_anode * 0.95:
            eta_conc_anode = (R * self.conditions.temperature / (2 * F)) * np.log(1 / (1 - current_density / i_limit_anode))
        else:
            eta_conc_anode = 0.5  # Cap at reasonable value
            
        if current_density < i_limit_cathode * 0.95:
            eta_conc_cathode = (R * self.conditions.temperature / (4 * F)) * np.log(1 / (1 - current_density / i_limit_cathode))
        else:
            eta_conc_cathode = 0.3  # Cap at reasonable value
        
        return eta_conc_anode, eta_conc_cathode
    
    def generate_iv_curve(self, current_range: np.ndarray) -> Dict:
        """Generate IV curve data with overpotential breakdown"""
        E_nernst = self.calculate_nernst_potential()
        
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
            R_ohmic = self.calculate_electrolyte_resistance(i)
            eta_ohmic = i * R_ohmic / self.geometry.active_area
            
            # Total voltage
            V_cell = E_nernst - eta_a_act - eta_c_act - eta_a_conc - eta_c_conc - eta_ohmic
            
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
            'power_density': (np.array(voltages) * current_range).tolist()
        }
    
    def generate_eis_data(self, frequencies: np.ndarray, current_density: float = 5000) -> Dict:
        """Generate EIS data with realistic impedance characteristics"""
        T = self.conditions.temperature
        
        # Circuit elements for equivalent circuit model
        # R_ohm + (R_anode//CPE_anode) + (R_cathode//CPE_cathode) + Warburg
        
        R_ohmic = self.calculate_electrolyte_resistance(current_density)
        
        # Anode impedance parameters
        R_anode = 0.15e-4  # Ω⋅m²
        CPE_anode_T = 0.02  # F⋅s^(n-1)⋅m^-2
        CPE_anode_n = 0.85
        
        # Cathode impedance parameters  
        R_cathode = 0.08e-4  # Ω⋅m²
        CPE_cathode_T = 0.15  # F⋅s^(n-1)⋅m^-2
        CPE_cathode_n = 0.75
        
        # Warburg impedance parameters
        sigma_w = 0.02e-4  # Ω⋅s^(-0.5)⋅m²
        
        omega = 2 * np.pi * frequencies
        
        # Calculate impedance components
        Z_ohmic = R_ohmic / self.geometry.active_area
        
        # CPE impedance: Z_CPE = 1/(T*(jω)^n)
        Z_anode_cpe = 1 / (CPE_anode_T * (1j * omega)**CPE_anode_n)
        Z_anode = R_anode / (1 + R_anode * CPE_anode_T * (1j * omega)**CPE_anode_n)
        
        Z_cathode_cpe = 1 / (CPE_cathode_T * (1j * omega)**CPE_cathode_n)
        Z_cathode = R_cathode / (1 + R_cathode * CPE_cathode_T * (1j * omega)**CPE_cathode_n)
        
        # Warburg impedance: Z_w = σ_w * (1-j) / sqrt(ω)
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
                'R_ohmic': R_ohmic / self.geometry.active_area,
                'R_anode': R_anode,
                'R_cathode': R_cathode,
                'CPE_anode_T': CPE_anode_T,
                'CPE_anode_n': CPE_anode_n,
                'CPE_cathode_T': CPE_cathode_T,
                'CPE_cathode_n': CPE_cathode_n,
                'sigma_warburg': sigma_w
            }
        }
    
    def calculate_oxygen_chemical_potential_gradient(self, current_density: float) -> Dict:
        """Calculate oxygen chemical potential gradient across electrolyte"""
        T = self.conditions.temperature
        
        # Oxygen partial pressures at electrodes
        p_O2_cathode = self.conditions.pressure_cathode * self.conditions.air_composition.get('O2', 0.21)
        
        # Anode side - calculate from fuel composition and current density
        p_H2 = self.conditions.pressure_anode * self.conditions.fuel_composition.get('H2', 0.97)
        p_H2O = self.conditions.pressure_anode * self.conditions.fuel_composition.get('H2O', 0.03)
        
        # Equilibrium p_O2 at anode from water-gas shift
        K_eq = np.exp(-(241800 - 44.3 * T) / (R * T))  # Water formation equilibrium
        p_O2_anode = (p_H2O / (p_H2 * K_eq))**2
        
        # Chemical potential of oxygen
        mu_O2_cathode = R * T * np.log(p_O2_cathode / 1e5)  # Reference to 1 bar
        mu_O2_anode = R * T * np.log(p_O2_anode / 1e5)
        
        # Gradient across electrolyte
        gradient = (mu_O2_cathode - mu_O2_anode) / self.geometry.electrolyte_thickness
        
        # Position-dependent chemical potential (linear assumption)
        positions = np.linspace(0, self.geometry.electrolyte_thickness, 50)
        mu_profile = mu_O2_anode + (mu_O2_cathode - mu_O2_anode) * positions / self.geometry.electrolyte_thickness
        
        return {
            'mu_O2_anode': mu_O2_anode,
            'mu_O2_cathode': mu_O2_cathode,
            'gradient': gradient,
            'positions': positions.tolist(),
            'mu_profile': mu_profile.tolist(),
            'p_O2_anode': p_O2_anode,
            'p_O2_cathode': p_O2_cathode,
            'driving_force': mu_O2_cathode - mu_O2_anode
        }
    
    def calculate_ni_oxidation_stress(self, current_density: float, ni_oxidation_risk: float) -> Dict:
        """Calculate volume change and stress from Ni to NiO conversion"""
        
        # Volume expansion upon Ni oxidation (NiO has ~1.7x volume of Ni)
        volume_expansion = self.materials.ni_volume_expansion
        
        # Fraction of Ni oxidized (simplified model based on oxidation risk)
        if ni_oxidation_risk > 0:
            oxidation_fraction = min(1.0, ni_oxidation_risk / 0.1)  # Saturates at 10% overpotential
        else:
            oxidation_fraction = 0
        
        # Volumetric strain from oxidation
        volumetric_strain = oxidation_fraction * volume_expansion * self.materials.ni_volume_fraction
        
        # Stress calculation (simplified elastic model)
        # Assume constrained expansion in anode
        E_anode = 55e9  # Young's modulus of Ni-YSZ (Pa)
        nu_anode = 0.29  # Poisson's ratio
        
        # Hydrostatic stress from volume expansion
        sigma_hydrostatic = E_anode * volumetric_strain / (3 * (1 - 2 * nu_anode))
        
        # Equivalent von Mises stress (assuming isotropic expansion)
        sigma_von_mises = sigma_hydrostatic
        
        # Stress transmitted to electrolyte (interface stress)
        # Simplified model - fraction of anode stress transmitted
        stress_transmission_factor = 0.3
        sigma_electrolyte = sigma_von_mises * stress_transmission_factor
        
        return {
            'oxidation_fraction': oxidation_fraction,
            'volumetric_strain': volumetric_strain,
            'hydrostatic_stress': sigma_hydrostatic,
            'von_mises_stress': sigma_von_mises,
            'electrolyte_stress': sigma_electrolyte,
            'oxidation_risk_level': 'High' if ni_oxidation_risk > 0.05 else 'Medium' if ni_oxidation_risk > 0.01 else 'Low'
        }
    
    def generate_complete_dataset(self, 
                                  current_range: Optional[np.ndarray] = None,
                                  frequency_range: Optional[np.ndarray] = None) -> Dict:
        """Generate complete electrochemical dataset"""
        
        if current_range is None:
            current_range = np.linspace(0, 10000, 101)  # A/m²
        
        if frequency_range is None:
            frequency_range = np.logspace(-2, 6, 50)  # 0.01 Hz to 1 MHz
        
        print("Generating SOFC Electrochemical Loading Dataset...")
        print(f"Operating Temperature: {self.conditions.temperature:.1f} K ({self.conditions.temperature-273.15:.1f} °C)")
        print(f"Current Density Range: {current_range[0]:.0f} - {current_range[-1]:.0f} A/m²")
        print(f"Frequency Range: {frequency_range[0]:.2e} - {frequency_range[-1]:.2e} Hz")
        
        # Generate IV curve data
        print("Generating IV curve data...")
        iv_data = self.generate_iv_curve(current_range)
        
        # Generate EIS data at multiple current densities
        print("Generating EIS data...")
        eis_data = {}
        test_currents = [0, 2000, 5000, 8000]  # A/m²
        for i_test in test_currents:
            eis_data[f'current_{i_test}'] = self.generate_eis_data(frequency_range, i_test)
        
        # Generate detailed analysis at specific operating points
        print("Calculating overpotentials and chemical gradients...")
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
        
        # Compile complete dataset
        complete_dataset = {
            'metadata': {
                'description': 'SOFC Electrochemical Loading Dataset',
                'generated_date': '2025-10-09',
                'operating_conditions': asdict(self.conditions),
                'geometry': asdict(self.geometry),
                'material_properties': asdict(self.materials)
            },
            'iv_curve': iv_data,
            'eis_data': eis_data,
            'detailed_analysis': detailed_analysis,
            'summary_statistics': self._calculate_summary_statistics(iv_data, detailed_analysis)
        }
        
        return complete_dataset
    
    def _calculate_summary_statistics(self, iv_data: Dict, detailed_analysis: Dict) -> Dict:
        """Calculate summary statistics for the dataset"""
        
        # IV curve statistics - handle potential NaN values
        power_density = np.array(iv_data['power_density'])
        valid_power = power_density[~np.isnan(power_density) & (power_density > 0)]
        
        if len(valid_power) > 0:
            max_power_idx = np.argmax(valid_power)
            # Find the index in the original array
            valid_indices = np.where(~np.isnan(power_density) & (power_density > 0))[0]
            original_idx = valid_indices[max_power_idx]
            
            max_power_density = iv_data['power_density'][original_idx]
            max_power_current = iv_data['current_density'][original_idx]
            max_power_voltage = iv_data['voltage'][original_idx]
        else:
            max_power_density = 0
            max_power_current = 0
            max_power_voltage = 0
        
        # Overpotential statistics
        ohmic_losses = np.array(iv_data['overpotentials']['ohmic'])
        activation_losses = np.array(iv_data['overpotentials']['anode_activation']) + \
                          np.array(iv_data['overpotentials']['cathode_activation'])
        
        # Ni oxidation risk assessment
        oxidation_risks = [detailed_analysis[key]['ni_oxidation_risk'] for key in detailed_analysis.keys()]
        max_oxidation_risk = max(oxidation_risks)
        
        return {
            'max_power_density': max_power_density,
            'max_power_current_density': max_power_current,
            'max_power_voltage': max_power_voltage,
            'nernst_potential': iv_data['nernst_potential'],
            'max_ohmic_loss': float(np.max(ohmic_losses)),
            'max_activation_loss': float(np.max(activation_losses)),
            'max_ni_oxidation_risk': max_oxidation_risk,
            'operating_temperature_C': self.conditions.temperature - 273.15,
            'electrolyte_thickness_um': self.geometry.electrolyte_thickness * 1e6
        }

def create_default_sofc_parameters():
    """Create default SOFC parameters for dataset generation"""
    
    # Operating conditions (800°C operation)
    operating_conditions = SOFCOperatingConditions(
        temperature=1073.15,  # 800°C in Kelvin
        pressure_anode=101325,  # 1 atm
        pressure_cathode=101325,  # 1 atm
        fuel_composition={'H2': 0.97, 'H2O': 0.03},
        air_composition={'O2': 0.21, 'N2': 0.79},
        fuel_utilization=0.85,
        air_utilization=0.25
    )
    
    # Cell geometry
    geometry = SOFCGeometry(
        electrolyte_thickness=150e-6,  # 150 μm
        anode_thickness=300e-6,  # 300 μm
        cathode_thickness=50e-6,  # 50 μm
        active_area=0.01,  # 10 cm²
        electrolyte_conductivity=0.1,  # S/m at reference
        anode_conductivity=1000,  # S/m
        cathode_conductivity=100  # S/m
    )
    
    # Material properties
    material_props = MaterialProperties(
        ysz_activation_energy=80000,  # J/mol
        ysz_pre_exponential=3.34e4,  # S/m
        ni_volume_fraction=0.35,
        ni_oxidation_threshold=0.05,  # V
        ni_volume_expansion=0.7,  # 70% volume increase
        lsm_exchange_current=1000,  # A/m²
        lsm_activation_energy=120000  # J/mol
    )
    
    return operating_conditions, geometry, material_props

def export_dataset(dataset: Dict, output_dir: str = "sofc_data"):
    """Export dataset to multiple formats"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nExporting dataset to {output_path}...")
    
    # Export complete dataset as JSON
    with open(output_path / "sofc_electrochemical_dataset.json", 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Export IV curve data as CSV
    iv_df = pd.DataFrame({
        'Current_Density_A_per_m2': dataset['iv_curve']['current_density'],
        'Voltage_V': dataset['iv_curve']['voltage'],
        'Power_Density_W_per_m2': dataset['iv_curve']['power_density'],
        'Anode_Overpotential_V': dataset['iv_curve']['overpotentials']['anode_activation'],
        'Cathode_Overpotential_V': dataset['iv_curve']['overpotentials']['cathode_activation'],
        'Ohmic_Overpotential_V': dataset['iv_curve']['overpotentials']['ohmic'],
        'Ni_Oxidation_Risk_V': dataset['iv_curve']['overpotentials']['ni_oxidation_risk']
    })
    iv_df.to_csv(output_path / "iv_curve_data.csv", index=False)
    
    # Export EIS data for each current density
    for current_key, eis_data in dataset['eis_data'].items():
        eis_df = pd.DataFrame({
            'Frequency_Hz': eis_data['frequency'],
            'Impedance_Real_Ohm_m2': eis_data['impedance_real'],
            'Impedance_Imag_Ohm_m2': eis_data['impedance_imag'],
            'Impedance_Magnitude_Ohm_m2': eis_data['impedance_magnitude'],
            'Phase_Angle_deg': eis_data['phase_angle']
        })
        eis_df.to_csv(output_path / f"eis_data_{current_key}.csv", index=False)
    
    # Export detailed analysis
    detailed_df_data = []
    for current_key, analysis in dataset['detailed_analysis'].items():
        row = {
            'Current_Density_A_per_m2': analysis['current_density'],
            'Anode_Overpotential_V': analysis['anode_overpotential'],
            'Ni_Oxidation_Risk_V': analysis['ni_oxidation_risk'],
            'O2_Chemical_Potential_Gradient_J_per_mol_per_m': analysis['chemical_potential_gradient']['gradient'],
            'Oxidation_Fraction': analysis['stress_analysis']['oxidation_fraction'],
            'Volumetric_Strain': analysis['stress_analysis']['volumetric_strain'],
            'Von_Mises_Stress_Pa': analysis['stress_analysis']['von_mises_stress'],
            'Electrolyte_Stress_Pa': analysis['stress_analysis']['electrolyte_stress']
        }
        detailed_df_data.append(row)
    
    detailed_df = pd.DataFrame(detailed_df_data)
    detailed_df.to_csv(output_path / "detailed_electrochemical_analysis.csv", index=False)
    
    print(f"Dataset exported successfully to {output_path}")
    print(f"Files created:")
    print(f"  - sofc_electrochemical_dataset.json (complete dataset)")
    print(f"  - iv_curve_data.csv (IV curve and overpotentials)")
    print(f"  - eis_data_*.csv (EIS data at different currents)")
    print(f"  - detailed_electrochemical_analysis.csv (chemical gradients and stress)")

def plot_dataset_overview(dataset: Dict, output_dir: str = "sofc_data"):
    """Generate overview plots of the dataset"""
    
    output_path = Path(output_dir)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SOFC Electrochemical Loading Dataset Overview', fontsize=16, fontweight='bold')
    
    # IV Curve
    ax = axes[0, 0]
    current = np.array(dataset['iv_curve']['current_density']) / 1000  # Convert to A/cm²
    voltage = dataset['iv_curve']['voltage']
    power = np.array(dataset['iv_curve']['power_density']) / 10000  # Convert to W/cm²
    
    ax.plot(current, voltage, 'b-', linewidth=2, label='Voltage')
    ax2 = ax.twinx()
    ax2.plot(current, power, 'r-', linewidth=2, label='Power Density')
    ax.set_xlabel('Current Density (A/cm²)')
    ax.set_ylabel('Voltage (V)', color='b')
    ax2.set_ylabel('Power Density (W/cm²)', color='r')
    ax.set_title('IV Curve')
    ax.grid(True, alpha=0.3)
    
    # Overpotentials
    ax = axes[0, 1]
    overpotentials = dataset['iv_curve']['overpotentials']
    ax.plot(current, overpotentials['anode_activation'], label='Anode Activation')
    ax.plot(current, overpotentials['cathode_activation'], label='Cathode Activation')
    ax.plot(current, overpotentials['ohmic'], label='Ohmic')
    ax.plot(current, overpotentials['ni_oxidation_risk'], label='Ni Oxidation Risk')
    ax.set_xlabel('Current Density (A/cm²)')
    ax.set_ylabel('Overpotential (V)')
    ax.set_title('Overpotential Breakdown')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # EIS Nyquist Plot
    ax = axes[0, 2]
    colors = ['blue', 'green', 'orange', 'red']
    for i, (current_key, eis_data) in enumerate(dataset['eis_data'].items()):
        real_z = np.array(eis_data['impedance_real']) * 10000  # Convert to Ω⋅cm²
        imag_z = -np.array(eis_data['impedance_imag']) * 10000  # Convert to Ω⋅cm²
        current_val = current_key.split('_')[1]
        ax.plot(real_z, imag_z, 'o-', color=colors[i], markersize=3, 
                label=f'{current_val} A/m²')
    ax.set_xlabel('Real Impedance (Ω⋅cm²)')
    ax.set_ylabel('-Imaginary Impedance (Ω⋅cm²)')
    ax.set_title('EIS Nyquist Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Chemical Potential Gradient
    ax = axes[1, 0]
    for current_key, analysis in dataset['detailed_analysis'].items():
        current_val = int(current_key.split('_')[1])
        positions = np.array(analysis['chemical_potential_gradient']['positions']) * 1e6  # Convert to μm
        mu_profile = analysis['chemical_potential_gradient']['mu_profile']
        ax.plot(positions, mu_profile, label=f'{current_val/1000:.1f} A/cm²')
    ax.set_xlabel('Position in Electrolyte (μm)')
    ax.set_ylabel('O₂ Chemical Potential (J/mol)')
    ax.set_title('O₂ Chemical Potential Gradient')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ni Oxidation and Stress
    ax = axes[1, 1]
    currents = []
    oxidation_fractions = []
    stresses = []
    
    for current_key, analysis in dataset['detailed_analysis'].items():
        current_val = int(current_key.split('_')[1])
        currents.append(current_val / 1000)  # Convert to A/cm²
        oxidation_fractions.append(analysis['stress_analysis']['oxidation_fraction'] * 100)
        stresses.append(analysis['stress_analysis']['electrolyte_stress'] / 1e6)  # Convert to MPa
    
    ax.plot(currents, oxidation_fractions, 'ro-', label='Ni Oxidation (%)')
    ax2 = ax.twinx()
    ax2.plot(currents, stresses, 'bs-', label='Electrolyte Stress (MPa)')
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
    Max Power Density: {stats['max_power_density']/10000:.2f} W/cm²
    @ Current Density: {stats['max_power_current_density']/1000:.2f} A/cm²
    @ Voltage: {stats['max_power_voltage']:.3f} V
    
    Max Ohmic Loss: {stats['max_ohmic_loss']:.3f} V
    Max Activation Loss: {stats['max_activation_loss']:.3f} V
    Max Ni Oxidation Risk: {stats['max_ni_oxidation_risk']:.4f} V
    """
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path / "sofc_dataset_overview.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Overview plot saved to {output_path / 'sofc_dataset_overview.png'}")

def main():
    """Main function to generate SOFC electrochemical dataset"""
    
    print("="*60)
    print("SOFC Electrochemical Loading Dataset Generator")
    print("="*60)
    
    # Create SOFC parameters
    operating_conditions, geometry, material_props = create_default_sofc_parameters()
    
    # Initialize generator
    generator = SOFCElectrochemicalGenerator(operating_conditions, geometry, material_props)
    
    # Define analysis ranges
    current_range = np.linspace(0, 10000, 101)  # 0 to 10 A/cm² (in A/m²)
    frequency_range = np.logspace(-2, 6, 50)  # 0.01 Hz to 1 MHz
    
    # Generate complete dataset
    dataset = generator.generate_complete_dataset(current_range, frequency_range)
    
    # Export dataset
    export_dataset(dataset)
    
    # Generate plots
    plot_dataset_overview(dataset)
    
    print("\n" + "="*60)
    print("Dataset Generation Complete!")
    print("="*60)
    print(f"Total data points generated:")
    print(f"  - IV curve: {len(dataset['iv_curve']['current_density'])} points")
    print(f"  - EIS data: {len(dataset['eis_data'])} current levels × {len(frequency_range)} frequencies")
    print(f"  - Detailed analysis: {len(dataset['detailed_analysis'])} operating points")
    print(f"\nKey findings:")
    stats = dataset['summary_statistics']
    print(f"  - Maximum power density: {stats['max_power_density']/10000:.2f} W/cm²")
    print(f"  - Maximum Ni oxidation risk: {stats['max_ni_oxidation_risk']:.4f} V")
    print(f"  - Operating temperature: {stats['operating_temperature_C']:.0f} °C")

if __name__ == "__main__":
    main()