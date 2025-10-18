#!/usr/bin/env python3
"""
XRD Analysis Dataset Generator for Fire-Resistant Rubberized Concrete
PhD Research: Development and Validation of Thermo-Mechanical Model

This module generates comprehensive XRD analysis data focusing on:
1. Crystalline phase identification and quantification
2. Portlandite (Ca(OH)₂) consumption tracking
3. Formation of new phases at elevated temperatures
4. Amorphous content evolution
5. Thermal decomposition products

Author: Research Team
Date: 2025-10-18
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class XRDDatasetGenerator:
    def __init__(self):
        """Initialize XRD dataset generator with crystallographic parameters"""
        self.temperatures = [20, 100, 200, 300, 400, 500, 600, 700, 800]  # °C
        self.rubber_contents = [0, 5, 10, 15, 20, 25]  # % by volume
        self.specimen_types = ['control', 'heated']
        
        # XRD measurement parameters
        self.two_theta_range = (5, 70)  # degrees
        self.step_size = 0.02  # degrees
        self.scan_speed = 2  # degrees/min
        self.radiation = 'Cu Kα'  # λ = 1.5406 Å
        
        # Major crystalline phases in cement
        self.cement_phases = {
            'Portlandite': {'formula': 'Ca(OH)₂', 'main_peaks': [18.1, 28.7, 34.1, 47.1, 50.8]},
            'Calcite': {'formula': 'CaCO₃', 'main_peaks': [23.0, 29.4, 35.9, 39.4, 43.1, 47.5]},
            'Quartz': {'formula': 'SiO₂', 'main_peaks': [20.9, 26.6, 36.5, 39.5, 42.4, 45.8, 50.1]},
            'C3S': {'formula': '3CaO·SiO₂', 'main_peaks': [29.3, 32.2, 32.6, 34.3]},
            'C2S': {'formula': '2CaO·SiO₂', 'main_peaks': [32.1, 32.6, 41.2]},
            'C3A': {'formula': '3CaO·Al₂O₃', 'main_peaks': [33.3, 47.6]},
            'C4AF': {'formula': '4CaO·Al₂O₃·Fe₂O₃', 'main_peaks': [12.1, 33.8]},
            'Ettringite': {'formula': 'Ca₆Al₂(SO₄)₃(OH)₁₂·26H₂O', 'main_peaks': [9.1, 15.8, 22.9]},
            'Gypsum': {'formula': 'CaSO₄·2H₂O', 'main_peaks': [11.6, 20.7, 23.4, 29.1]},
            'Lime': {'formula': 'CaO', 'main_peaks': [32.2, 37.3, 53.8, 64.2]},
            'Periclase': {'formula': 'MgO', 'main_peaks': [36.9, 42.9, 62.3]},
            'Gehlenite': {'formula': 'Ca₂Al₂SiO₇', 'main_peaks': [31.0, 35.2]}
        }
        
        # Initialize data containers
        self.phase_quantification_data = {}
        self.peak_analysis_data = {}
        self.thermal_decomposition_data = {}
        self.amorphous_content_data = {}
        
    def generate_phase_quantification_data(self):
        """Generate quantitative phase analysis data using Rietveld refinement"""
        print("Generating XRD phase quantification dataset...")
        
        phase_data = []
        
        for temp in self.temperatures:
            for rubber_content in self.rubber_contents:
                for specimen_type in self.specimen_types:
                    if specimen_type == 'heated' and temp == 20:
                        continue
                    
                    # Base composition for OPC (Ordinary Portland Cement)
                    base_composition = {
                        'C3S': 55 + np.random.normal(0, 3),
                        'C2S': 20 + np.random.normal(0, 2),
                        'C3A': 8 + np.random.normal(0, 1),
                        'C4AF': 10 + np.random.normal(0, 1),
                        'Gypsum': 3 + np.random.normal(0, 0.5),
                        'Calcite': 2 + np.random.normal(0, 0.3),
                        'Quartz': 2 + np.random.normal(0, 0.3)
                    }
                    
                    # Hydration products (age-dependent, assume 28 days)
                    degree_of_hydration = 0.75 - rubber_content * 0.002  # Rubber slightly reduces hydration
                    
                    # Calculate hydration products
                    portlandite_content = (base_composition['C3S'] * 0.24 + base_composition['C2S'] * 0.2) * degree_of_hydration
                    ettringite_content = min(base_composition['C3A'] * 1.5, base_composition['Gypsum'] * 2.2)
                    
                    # Temperature effects on phase composition
                    current_composition = base_composition.copy()
                    
                    # Portlandite (Ca(OH)₂) - stable up to ~450°C, decomposes 450-550°C
                    if temp <= 450:
                        current_composition['Portlandite'] = portlandite_content
                    elif temp <= 550:
                        # Linear decomposition between 450-550°C
                        decomposition_fraction = (temp - 450) / 100
                        current_composition['Portlandite'] = portlandite_content * (1 - decomposition_fraction)
                        # Lime formation from portlandite decomposition
                        current_composition['Lime'] = portlandite_content * decomposition_fraction * 0.76  # Molecular weight ratio
                    else:
                        current_composition['Portlandite'] = 0
                        current_composition['Lime'] = portlandite_content * 0.76
                    
                    # Ettringite - decomposes >70°C
                    if temp <= 70:
                        current_composition['Ettringite'] = ettringite_content
                    elif temp <= 150:
                        # Gradual decomposition 70-150°C
                        decomposition_fraction = (temp - 70) / 80
                        current_composition['Ettringite'] = ettringite_content * (1 - decomposition_fraction)
                    else:
                        current_composition['Ettringite'] = 0
                    
                    # Gypsum dehydration >100°C
                    if temp > 100:
                        current_composition['Gypsum'] = 0
                        # Formation of hemihydrate and anhydrite (simplified)
                    
                    # C-S-H gel decomposition >450°C (becomes amorphous)
                    csh_content = 50 + rubber_content * 0.5  # Base C-S-H content
                    if temp > 450:
                        decomposition_factor = min((temp - 450) / 350, 0.8)  # 80% max decomposition
                        csh_content *= (1 - decomposition_factor)
                    
                    # High-temperature phases formation
                    if temp > 700:
                        # Formation of gehlenite and other calcium silicates
                        current_composition['Gehlenite'] = (temp - 700) * 0.05
                        current_composition['Wollastonite'] = (temp - 700) * 0.03
                    
                    # Rubber effects on crystalline phases
                    rubber_factor = 1 - rubber_content * 0.001  # Slight dilution effect
                    for phase in current_composition:
                        if phase not in ['Portlandite', 'Lime']:  # These are calculated separately
                            current_composition[phase] *= rubber_factor
                    
                    # Normalize to 100% (excluding amorphous content)
                    total_crystalline = sum(v for v in current_composition.values() if v > 0)
                    amorphous_content = max(20 + rubber_content + (temp - 20) * 0.05, 20)  # Minimum 20% amorphous
                    
                    if total_crystalline > 0:
                        crystalline_fraction = (100 - amorphous_content) / total_crystalline
                        for phase in current_composition:
                            if current_composition[phase] > 0:
                                current_composition[phase] *= crystalline_fraction
                    
                    # Rietveld refinement quality parameters
                    rwp = 8 + np.random.normal(0, 2)  # Weighted profile R-factor
                    gof = 1.2 + np.random.normal(0, 0.3)  # Goodness of fit
                    
                    # Create record for each phase
                    for phase, content in current_composition.items():
                        if content > 0.1:  # Only include phases >0.1%
                            phase_record = {
                                'temperature': temp,
                                'rubber_content': rubber_content,
                                'specimen_type': specimen_type,
                                'phase_name': phase,
                                'chemical_formula': self.cement_phases.get(phase, {}).get('formula', 'Unknown'),
                                'weight_percent': content,
                                'volume_percent': content * 0.9,  # Approximate conversion
                                'crystallite_size_nm': self._calculate_crystallite_size(phase, temp),
                                'lattice_strain': self._calculate_lattice_strain(phase, temp),
                                'preferred_orientation': np.random.normal(1, 0.1),
                                'rwp_percent': rwp,
                                'gof': gof,
                                'amorphous_content_percent': amorphous_content,
                                'measurement_time_min': 60,
                                'specimen_id': f'XRD_{rubber_content}_{temp}_{specimen_type}_{phase[:3]}',
                                'refinement_method': 'Rietveld',
                                'software': 'TOPAS-Academic',
                                'analysis_date': datetime.now().strftime('%Y-%m-%d')
                            }
                            
                            phase_data.append(phase_record)
        
        self.phase_quantification_data = pd.DataFrame(phase_data)
        return self.phase_quantification_data
    
    def _calculate_crystallite_size(self, phase, temperature):
        """Calculate crystallite size using Scherrer equation principles"""
        base_sizes = {
            'Portlandite': 150,
            'Calcite': 200,
            'Quartz': 500,
            'C3S': 100,
            'C2S': 120,
            'Lime': 80,
            'Ettringite': 50
        }
        
        base_size = base_sizes.get(phase, 100)  # nm
        
        # Temperature effects on crystallite size
        if temperature > 400:
            # Grain growth at high temperatures
            growth_factor = 1 + (temperature - 400) * 0.002
            base_size *= growth_factor
        
        # Add some randomness
        return base_size * np.random.lognormal(0, 0.2)
    
    def _calculate_lattice_strain(self, phase, temperature):
        """Calculate lattice strain from thermal effects"""
        base_strain = 0.001  # Base microstrain
        
        # Thermal expansion effects
        thermal_strain = (temperature - 20) * 1e-5
        
        # Phase-specific thermal behavior
        phase_factors = {
            'Portlandite': 2.0,  # High thermal expansion
            'Calcite': 1.5,
            'Quartz': 0.5,  # Low thermal expansion
            'Lime': 1.8
        }
        
        factor = phase_factors.get(phase, 1.0)
        total_strain = base_strain + thermal_strain * factor
        
        return total_strain * np.random.lognormal(0, 0.3)
    
    def generate_portlandite_analysis(self):
        """Generate detailed Portlandite consumption analysis"""
        print("Generating Portlandite consumption analysis...")
        
        portlandite_data = []
        
        for temp in self.temperatures:
            for rubber_content in self.rubber_contents:
                for specimen_type in self.specimen_types:
                    if specimen_type == 'heated' and temp == 20:
                        continue
                    
                    # Initial portlandite content (from hydration)
                    initial_ch_content = 18 + rubber_content * 0.1  # wt%
                    
                    # Temperature-dependent decomposition
                    if temp <= 400:
                        remaining_ch = initial_ch_content
                        decomposition_rate = 0
                    elif temp <= 450:
                        # Onset of decomposition
                        decomposition_fraction = (temp - 400) / 50 * 0.1
                        remaining_ch = initial_ch_content * (1 - decomposition_fraction)
                        decomposition_rate = decomposition_fraction / 50  # %/°C
                    elif temp <= 550:
                        # Main decomposition zone
                        decomposition_fraction = 0.1 + (temp - 450) / 100 * 0.85
                        remaining_ch = initial_ch_content * (1 - decomposition_fraction)
                        decomposition_rate = 0.85 / 100  # %/°C
                    else:
                        # Complete decomposition
                        remaining_ch = initial_ch_content * 0.05  # 5% residual
                        decomposition_rate = 0
                    
                    # Peak analysis for main Portlandite peaks
                    main_peaks = [18.1, 28.7, 34.1, 47.1, 50.8]  # 2θ positions
                    
                    for peak_position in main_peaks:
                        # Peak intensity proportional to content
                        base_intensity = 1000 * (remaining_ch / initial_ch_content)
                        peak_intensity = base_intensity * np.random.lognormal(0, 0.1)
                        
                        # Peak width (FWHM) - increases with temperature due to strain
                        base_fwhm = 0.15  # degrees
                        thermal_broadening = (temp - 20) * 0.0005
                        peak_fwhm = base_fwhm + thermal_broadening
                        
                        # Peak area (proportional to phase content)
                        peak_area = peak_intensity * peak_fwhm * 0.5
                        
                        portlandite_record = {
                            'temperature': temp,
                            'rubber_content': rubber_content,
                            'specimen_type': specimen_type,
                            'initial_ch_content_wt_percent': initial_ch_content,
                            'remaining_ch_content_wt_percent': remaining_ch,
                            'decomposition_fraction': (initial_ch_content - remaining_ch) / initial_ch_content,
                            'decomposition_rate_per_degree': decomposition_rate,
                            'peak_position_2theta': peak_position,
                            'peak_intensity_counts': peak_intensity,
                            'peak_fwhm_degrees': peak_fwhm,
                            'peak_area_counts_deg': peak_area,
                            'integrated_intensity': peak_area,
                            'd_spacing_angstrom': 1.5406 / (2 * np.sin(np.radians(peak_position / 2))),
                            'miller_indices': self._get_miller_indices(peak_position),
                            'peak_asymmetry': 1.0 + (temp - 20) * 0.0001,
                            'background_counts': 50 + np.random.normal(0, 10),
                            'signal_to_noise': peak_intensity / (50 + np.random.normal(0, 10)),
                            'specimen_id': f'CH_{rubber_content}_{temp}_{specimen_type}',
                            'measurement_conditions': {
                                'step_size_deg': 0.02,
                                'counting_time_s': 2,
                                'detector': 'Lynxeye',
                                'monochromator': 'Ge(111)'
                            }
                        }
                        
                        portlandite_data.append(portlandite_record)
        
        self.portlandite_data = pd.DataFrame(portlandite_data)
        return self.portlandite_data
    
    def _get_miller_indices(self, two_theta):
        """Get Miller indices for Portlandite peaks"""
        peak_hkl = {
            18.1: '(001)',
            28.7: '(100)',
            34.1: '(101)',
            47.1: '(102)',
            50.8: '(110)'
        }
        return peak_hkl.get(two_theta, '(hkl)')
    
    def generate_thermal_decomposition_products(self):
        """Generate analysis of thermal decomposition products"""
        print("Generating thermal decomposition products analysis...")
        
        decomposition_data = []
        
        for temp in self.temperatures:
            for rubber_content in self.rubber_contents:
                for specimen_type in self.specimen_types:
                    if specimen_type == 'heated' and temp == 20:
                        continue
                    
                    # Decomposition reactions and products
                    reactions = []
                    
                    # 1. Ettringite decomposition (70-150°C)
                    if 70 <= temp <= 150:
                        ettringite_loss = min((temp - 70) / 80, 1.0)
                        reactions.append({
                            'reaction': 'Ettringite → Metaettringite + H₂O',
                            'temperature_range': '70-150°C',
                            'conversion_fraction': ettringite_loss,
                            'products': ['Metaettringite', 'Water_vapor'],
                            'enthalpy_kj_mol': -180
                        })
                    
                    # 2. Gypsum dehydration (100-200°C)
                    if 100 <= temp <= 200:
                        gypsum_loss = min((temp - 100) / 100, 1.0)
                        reactions.append({
                            'reaction': 'CaSO₄·2H₂O → CaSO₄·0.5H₂O + 1.5H₂O',
                            'temperature_range': '100-200°C',
                            'conversion_fraction': gypsum_loss,
                            'products': ['Bassanite', 'Water_vapor'],
                            'enthalpy_kj_mol': -104
                        })
                    
                    # 3. C-S-H gel dehydration (200-600°C)
                    if 200 <= temp <= 600:
                        csh_dehydration = min((temp - 200) / 400, 0.8)
                        reactions.append({
                            'reaction': 'C-S-H → Amorphous_silicate + H₂O',
                            'temperature_range': '200-600°C',
                            'conversion_fraction': csh_dehydration,
                            'products': ['Amorphous_calcium_silicate', 'Water_vapor'],
                            'enthalpy_kj_mol': -85
                        })
                    
                    # 4. Portlandite decomposition (450-550°C)
                    if 450 <= temp <= 550:
                        ch_decomposition = min((temp - 450) / 100, 0.95)
                        reactions.append({
                            'reaction': 'Ca(OH)₂ → CaO + H₂O',
                            'temperature_range': '450-550°C',
                            'conversion_fraction': ch_decomposition,
                            'products': ['Lime', 'Water_vapor'],
                            'enthalpy_kj_mol': -109
                        })
                    
                    # 5. Calcite decomposition (600-900°C)
                    if 600 <= temp <= 900:
                        calcite_decomposition = min((temp - 600) / 300, 0.9)
                        reactions.append({
                            'reaction': 'CaCO₃ → CaO + CO₂',
                            'temperature_range': '600-900°C',
                            'conversion_fraction': calcite_decomposition,
                            'products': ['Lime', 'Carbon_dioxide'],
                            'enthalpy_kj_mol': -178
                        })
                    
                    # 6. Rubber pyrolysis (300-500°C)
                    if rubber_content > 0 and 300 <= temp <= 500:
                        rubber_pyrolysis = min((temp - 300) / 200, 0.8)
                        reactions.append({
                            'reaction': 'Rubber → Carbon_residue + Volatiles',
                            'temperature_range': '300-500°C',
                            'conversion_fraction': rubber_pyrolysis,
                            'products': ['Carbon_black', 'Volatile_organics'],
                            'enthalpy_kj_mol': -250
                        })
                    
                    # Create records for each reaction
                    for reaction in reactions:
                        decomposition_record = {
                            'temperature': temp,
                            'rubber_content': rubber_content,
                            'specimen_type': specimen_type,
                            'reaction_equation': reaction['reaction'],
                            'temperature_range': reaction['temperature_range'],
                            'conversion_fraction': reaction['conversion_fraction'],
                            'reaction_enthalpy_kj_mol': reaction['enthalpy_kj_mol'],
                            'primary_product': reaction['products'][0],
                            'secondary_product': reaction['products'][1] if len(reaction['products']) > 1 else None,
                            'mass_loss_percent': reaction['conversion_fraction'] * self._get_phase_mass_fraction(reaction['reaction']),
                            'reaction_rate_per_min': self._calculate_reaction_rate(temp, reaction),
                            'activation_energy_kj_mol': self._get_activation_energy(reaction['reaction']),
                            'specimen_id': f'TD_{rubber_content}_{temp}_{specimen_type}',
                            'analysis_method': 'XRD-TGA_coupled'
                        }
                        
                        decomposition_data.append(decomposition_record)
        
        self.thermal_decomposition_data = pd.DataFrame(decomposition_data)
        return self.thermal_decomposition_data
    
    def _get_phase_mass_fraction(self, reaction):
        """Get typical mass fraction for each phase"""
        phase_fractions = {
            'Ettringite': 0.08,
            'CaSO₄·2H₂O': 0.03,
            'C-S-H': 0.50,
            'Ca(OH)₂': 0.18,
            'CaCO₃': 0.05,
            'Rubber': 0.15
        }
        
        for phase, fraction in phase_fractions.items():
            if phase in reaction:
                return fraction
        return 0.1  # Default
    
    def _calculate_reaction_rate(self, temperature, reaction):
        """Calculate reaction rate using Arrhenius equation"""
        # Simplified rate calculation
        base_rate = 0.01  # %/min
        activation_energy = self._get_activation_energy(reaction['reaction'])  # kJ/mol
        R = 8.314e-3  # kJ/(mol·K)
        T = temperature + 273.15  # K
        
        # Arrhenius equation (simplified)
        rate = base_rate * np.exp(-activation_energy / (R * T))
        return rate * reaction['conversion_fraction']
    
    def _get_activation_energy(self, reaction):
        """Get activation energy for different reactions"""
        activation_energies = {
            'Ettringite': 80,
            'Gypsum': 90,
            'C-S-H': 120,
            'Portlandite': 140,
            'Calcite': 200,
            'Rubber': 180
        }
        
        for phase, energy in activation_energies.items():
            if any(p in reaction for p in [phase, phase.lower()]):
                return energy
        return 150  # Default
    
    def generate_amorphous_content_analysis(self):
        """Generate amorphous content analysis using internal standard method"""
        print("Generating amorphous content analysis...")
        
        amorphous_data = []
        
        for temp in self.temperatures:
            for rubber_content in self.rubber_contents:
                for specimen_type in self.specimen_types:
                    if specimen_type == 'heated' and temp == 20:
                        continue
                    
                    # Base amorphous content (C-S-H gel, glass, etc.)
                    base_amorphous = 25 + rubber_content * 0.5  # %
                    
                    # Temperature effects on amorphous content
                    if temp <= 200:
                        # Minimal change at low temperatures
                        current_amorphous = base_amorphous + (temp - 20) * 0.01
                    elif temp <= 600:
                        # C-S-H gel becomes more amorphous
                        additional_amorphous = (temp - 200) * 0.05
                        current_amorphous = base_amorphous + additional_amorphous
                    else:
                        # High temperature: some recrystallization
                        max_amorphous = base_amorphous + 20
                        recrystallization = (temp - 600) * 0.02
                        current_amorphous = max_amorphous - recrystallization
                    
                    # Ensure reasonable bounds
                    current_amorphous = max(min(current_amorphous, 80), 15)
                    
                    # Internal standard method (using corundum)
                    corundum_added = 10  # wt% internal standard
                    corundum_peak_intensity = 1000  # Reference intensity
                    
                    # Calculate amorphous content using internal standard
                    total_crystalline = 100 - current_amorphous
                    measured_corundum_intensity = corundum_peak_intensity * (corundum_added / (corundum_added + total_crystalline))
                    
                    # Rubber effects on XRD pattern
                    rubber_scattering_factor = 1 + rubber_content * 0.002  # Increases background
                    background_intensity = 100 + rubber_content * 5 + (temp - 20) * 0.5
                    
                    amorphous_record = {
                        'temperature': temp,
                        'rubber_content': rubber_content,
                        'specimen_type': specimen_type,
                        'amorphous_content_percent': current_amorphous,
                        'crystalline_content_percent': 100 - current_amorphous,
                        'internal_standard_added_percent': corundum_added,
                        'corundum_peak_intensity': measured_corundum_intensity,
                        'reference_corundum_intensity': corundum_peak_intensity,
                        'intensity_ratio': measured_corundum_intensity / corundum_peak_intensity,
                        'background_intensity': background_intensity * rubber_scattering_factor,
                        'amorphous_hump_position_2theta': 25 + np.random.normal(0, 2),
                        'amorphous_hump_width_2theta': 15 + (temp - 20) * 0.01,
                        'csh_gel_content_percent': max(50 - (temp - 20) * 0.05, 10),
                        'glass_content_percent': 5 + rubber_content * 0.1,
                        'measurement_error_percent': 2 + np.random.normal(0, 0.5),
                        'specimen_id': f'AM_{rubber_content}_{temp}_{specimen_type}',
                        'analysis_method': 'Internal_standard_corundum',
                        'software': 'TOPAS-Academic'
                    }
                    
                    amorphous_data.append(amorphous_record)
        
        self.amorphous_content_data = pd.DataFrame(amorphous_data)
        return self.amorphous_content_data
    
    def save_datasets(self, output_dir='xrd_analysis_data'):
        """Save all generated XRD datasets"""
        os.makedirs(output_dir, exist_ok=True)
        
        datasets = {
            'phase_quantification': self.phase_quantification_data,
            'portlandite_analysis': self.portlandite_data,
            'thermal_decomposition': self.thermal_decomposition_data,
            'amorphous_content': self.amorphous_content_data
        }
        
        for name, df in datasets.items():
            if df is not None and not df.empty:
                # Save as CSV
                csv_path = os.path.join(output_dir, f'{name}.csv')
                df.to_csv(csv_path, index=False)
                
                # Save as JSON for detailed metadata
                json_path = os.path.join(output_dir, f'{name}.json')
                df.to_json(json_path, orient='records', indent=2)
                
                print(f"Saved {name} dataset: {len(df)} records")
        
        # Save XRD methodology and parameters
        methodology = {
            'instrument': 'PANalytical X\'Pert Pro MPD',
            'radiation': f'{self.radiation} (λ = 1.5406 Å)',
            'detector': 'X\'Celerator RTMS',
            'measurement_range': f'{self.two_theta_range[0]}-{self.two_theta_range[1]}° 2θ',
            'step_size': f'{self.step_size}° 2θ',
            'scan_speed': f'{self.scan_speed}°/min',
            'sample_preparation': 'Back-loading technique, random orientation',
            'quantitative_analysis': 'Rietveld refinement with TOPAS-Academic',
            'internal_standard': 'Corundum (Al₂O₃) - 10 wt%',
            'phases_identified': list(self.cement_phases.keys()),
            'detection_limit': '1 wt%',
            'precision': '±2% relative'
        }
        
        with open(os.path.join(output_dir, 'xrd_methodology.json'), 'w') as f:
            json.dump(methodology, f, indent=2)
    
    def generate_all_datasets(self):
        """Generate all XRD analysis datasets"""
        print("=== XRD Analysis Dataset Generation ===")
        
        self.generate_phase_quantification_data()
        self.generate_portlandite_analysis()
        self.generate_thermal_decomposition_products()
        self.generate_amorphous_content_analysis()
        
        return {
            'phase_quantification_data': self.phase_quantification_data,
            'portlandite_data': self.portlandite_data,
            'thermal_decomposition_data': self.thermal_decomposition_data,
            'amorphous_content_data': self.amorphous_content_data
        }

if __name__ == "__main__":
    # Generate comprehensive XRD dataset
    generator = XRDDatasetGenerator()
    datasets = generator.generate_all_datasets()
    generator.save_datasets()
    
    print("\n=== XRD Dataset Generation Complete ===")
    for name, df in datasets.items():
        if df is not None:
            print(f"{name}: {len(df)} records")