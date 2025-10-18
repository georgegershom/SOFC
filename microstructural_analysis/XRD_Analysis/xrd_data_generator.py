"""
XRD (X-Ray Diffraction) Analysis Data Generator for Rubberized Concrete
PhD Research: Crystalline Phase Evolution under Thermal Loading
Focus: Portlandite consumption, CSH decomposition, new phase formation
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

class XRDDataGenerator:
    """
    Generate comprehensive XRD data showing crystalline phase transformations
    in rubberized concrete under elevated temperatures
    """
    
    def __init__(self):
        self.temperatures = [20, 200, 400, 600, 800]  # °C
        self.rubber_contents = [0, 5, 10, 15, 20]  # % by volume
        
        # Key crystalline phases in cement paste
        self.phases = {
            'Portlandite': {'formula': 'Ca(OH)2', 'peaks': [18.0, 34.1, 47.1, 50.8]},
            'Calcite': {'formula': 'CaCO3', 'peaks': [29.4, 39.4, 43.2, 47.5, 48.5]},
            'CSH_gel': {'formula': 'C-S-H', 'peaks': [29.3, 32.0, 50.1]},  # Broad peaks
            'Ettringite': {'formula': 'Ca6Al2(SO4)3(OH)12·26H2O', 'peaks': [9.1, 15.8, 22.9]},
            'Quartz': {'formula': 'SiO2', 'peaks': [26.6, 20.9, 50.1, 60.0]},
            'Alite': {'formula': 'C3S', 'peaks': [32.2, 34.3, 41.2, 51.7]},
            'Belite': {'formula': 'C2S', 'peaks': [32.1, 32.6, 34.4, 41.3]},
            'Lime': {'formula': 'CaO', 'peaks': [32.2, 37.4, 53.9, 64.2]},  # Forms at high temp
            'Anhydrite': {'formula': 'CaSO4', 'peaks': [25.4, 31.3, 38.6, 40.8]},
            'Gehlenite': {'formula': 'Ca2Al2SiO7', 'peaks': [31.3, 36.8, 52.2]},  # High temp phase
        }
        
    def generate_phase_quantification(self):
        """
        Generate quantitative phase analysis using Rietveld refinement simulation
        This is PhD-level analysis showing phase transformations
        """
        data = []
        
        for temp in self.temperatures:
            for rubber in self.rubber_contents:
                for replicate in range(5):  # Multiple samples per condition
                    
                    phase_contents = {}
                    
                    # Portlandite (Ca(OH)2) - decreases with temperature
                    if temp <= 400:
                        portlandite = 20 - rubber * 0.5 - (temp - 20) * 0.01 + np.random.normal(0, 1)
                    elif temp <= 500:
                        portlandite = 5 - rubber * 0.2 + np.random.normal(0, 0.5)
                    else:
                        portlandite = 0  # Fully decomposed
                    phase_contents['Portlandite_wt%'] = max(0, portlandite)
                    
                    # C-S-H gel - main binding phase
                    if temp <= 200:
                        csh = 50 - rubber * 1.0 + np.random.normal(0, 2)
                    elif temp <= 400:
                        csh = 40 - rubber * 0.8 - (temp - 200) * 0.05 + np.random.normal(0, 2)
                    elif temp <= 600:
                        csh = 20 - rubber * 0.5 - (temp - 400) * 0.08 + np.random.normal(0, 1.5)
                    else:
                        csh = 5 - rubber * 0.1 + np.random.normal(0, 0.5)
                    phase_contents['CSH_wt%'] = max(0, csh)
                    
                    # Calcite (CaCO3) - from carbonation
                    calcite = 3 + rubber * 0.1 + (temp / 100) * 0.5 + np.random.normal(0, 0.5)
                    phase_contents['Calcite_wt%'] = min(15, max(0, calcite))
                    
                    # Ettringite - decomposes at low temperature
                    if temp <= 70:
                        ettringite = 2 - rubber * 0.05 + np.random.normal(0, 0.2)
                    else:
                        ettringite = 0
                    phase_contents['Ettringite_wt%'] = max(0, ettringite)
                    
                    # Quartz (from aggregates)
                    quartz = 15 + np.random.normal(0, 1)
                    phase_contents['Quartz_wt%'] = max(5, quartz)
                    
                    # Unhydrated cement phases
                    alite = 3 - (temp / 200) * 0.5 + np.random.normal(0, 0.3)
                    belite = 5 - (temp / 300) * 0.5 + np.random.normal(0, 0.4)
                    phase_contents['Alite_wt%'] = max(0, alite)
                    phase_contents['Belite_wt%'] = max(0, belite)
                    
                    # High temperature phases
                    if temp >= 500:
                        lime = (temp - 500) * 0.02 + np.random.normal(0, 0.3)
                        phase_contents['Lime_wt%'] = max(0, lime)
                    else:
                        phase_contents['Lime_wt%'] = 0
                    
                    if temp >= 600:
                        anhydrite = (temp - 600) * 0.01 + np.random.normal(0, 0.2)
                        gehlenite = (temp - 600) * 0.005 + np.random.normal(0, 0.1)
                        phase_contents['Anhydrite_wt%'] = max(0, anhydrite)
                        phase_contents['Gehlenite_wt%'] = max(0, gehlenite)
                    else:
                        phase_contents['Anhydrite_wt%'] = 0
                        phase_contents['Gehlenite_wt%'] = 0
                    
                    # Amorphous content (increases with rubber and temperature)
                    amorphous = 10 + rubber * 1.5 + (temp / 50) + np.random.normal(0, 2)
                    phase_contents['Amorphous_wt%'] = min(40, max(5, amorphous))
                    
                    # Calculate crystallinity index
                    total_crystalline = 100 - phase_contents['Amorphous_wt%']
                    crystallinity_index = total_crystalline / 100
                    
                    # Lattice parameters (showing thermal expansion/contraction)
                    if 'Portlandite_wt%' in phase_contents and phase_contents['Portlandite_wt%'] > 0:
                        # Portlandite lattice parameters
                        a_portlandite = 3.593 + (temp - 20) * 0.00001 + np.random.normal(0, 0.001)
                        c_portlandite = 4.911 + (temp - 20) * 0.00002 + np.random.normal(0, 0.001)
                    else:
                        a_portlandite = None
                        c_portlandite = None
                    
                    # Peak broadening (indicates crystal size and strain)
                    fwhm_avg = 0.15 + (temp / 2000) + rubber * 0.01 + np.random.normal(0, 0.01)
                    
                    # Crystallite size (Scherrer equation)
                    crystallite_size = max(10, 50 - temp * 0.03 - rubber * 0.5 + np.random.normal(0, 3))
                    
                    # Microstrain
                    microstrain = 0.001 + temp * 0.000002 + rubber * 0.00001 + np.random.normal(0, 0.0001)
                    
                    record = {
                        'Temperature_C': temp,
                        'Rubber_Content_%': rubber,
                        'Sample_ID': f'XRD_{temp}_{rubber}_{replicate+1}',
                        'Replicate': replicate + 1,
                        'Crystallinity_Index': crystallinity_index,
                        'FWHM_Avg_deg': fwhm_avg,
                        'Crystallite_Size_nm': crystallite_size,
                        'Microstrain': microstrain,
                        'a_Portlandite_A': a_portlandite,
                        'c_Portlandite_A': c_portlandite,
                        'Timestamp': datetime.now().isoformat()
                    }
                    
                    # Add phase contents to record
                    record.update(phase_contents)
                    
                    # Normalize phases to 100%
                    total = sum([v for k, v in phase_contents.items() if 'wt%' in k])
                    for key in phase_contents:
                        if 'wt%' in key:
                            record[key] = (phase_contents[key] / total) * 100
                    
                    data.append(record)
        
        return pd.DataFrame(data)
    
    def generate_diffraction_patterns(self):
        """
        Generate simulated XRD diffraction patterns
        Shows peak positions, intensities, and broadening
        """
        patterns = []
        
        two_theta = np.linspace(5, 70, 3251)  # 2θ range from 5° to 70°
        
        for temp in self.temperatures:
            for rubber in self.rubber_contents:
                
                # Initialize pattern
                intensity = np.zeros_like(two_theta)
                
                # Add background
                background = 100 + 20 * np.exp(-two_theta/30) + np.random.normal(0, 2, len(two_theta))
                intensity += background
                
                # Add peaks for each phase based on temperature
                for phase, info in self.phases.items():
                    
                    # Determine phase intensity based on temperature
                    if phase == 'Portlandite':
                        if temp <= 400:
                            phase_intensity = 1000 - temp * 1.5 - rubber * 20
                        else:
                            phase_intensity = 100 - temp * 0.1
                    
                    elif phase == 'CSH_gel':
                        if temp <= 600:
                            phase_intensity = 500 - temp * 0.5 - rubber * 10
                        else:
                            phase_intensity = 50
                    
                    elif phase == 'Calcite':
                        phase_intensity = 200 + temp * 0.1
                    
                    elif phase == 'Ettringite':
                        phase_intensity = 150 if temp <= 70 else 0
                    
                    elif phase == 'Quartz':
                        phase_intensity = 800  # Stable
                    
                    elif phase == 'Lime':
                        phase_intensity = (temp - 500) * 2 if temp > 500 else 0
                    
                    elif phase == 'Anhydrite':
                        phase_intensity = (temp - 600) * 1.5 if temp > 600 else 0
                    
                    else:
                        phase_intensity = 100
                    
                    if phase_intensity > 0:
                        # Add peaks
                        for peak_pos in info['peaks']:
                            # Peak broadening increases with temperature
                            width = 0.15 + temp * 0.0001 + rubber * 0.001
                            
                            # Add Gaussian peak
                            peak = phase_intensity * np.exp(-((two_theta - peak_pos) / width)**2)
                            intensity += peak
                
                # Add noise
                intensity += np.random.normal(0, 5, len(intensity))
                
                # Store pattern
                pattern_data = {
                    'Temperature_C': temp,
                    'Rubber_Content_%': rubber,
                    'Pattern_ID': f'XRD_Pattern_{temp}_{rubber}',
                    '2Theta': two_theta.tolist(),
                    'Intensity': intensity.tolist(),
                    'Max_Intensity': float(np.max(intensity)),
                    'Background_Level': float(np.mean(background)),
                    'Signal_to_Noise': float(np.max(intensity) / np.std(intensity[:100])),
                    'Timestamp': datetime.now().isoformat()
                }
                
                patterns.append(pattern_data)
        
        return patterns
    
    def generate_peak_analysis(self):
        """
        Detailed peak analysis for phase identification
        """
        data = []
        
        for temp in self.temperatures:
            for rubber in self.rubber_contents:
                
                # Analyze major peaks
                peaks_found = []
                
                # Portlandite peaks
                if temp <= 500:
                    for peak in [18.0, 34.1, 47.1]:
                        intensity = max(0, 1000 - temp * 1.5 - rubber * 20 + np.random.normal(0, 50))
                        if intensity > 50:
                            peaks_found.append({
                                'Peak_2Theta': peak + np.random.normal(0, 0.05),
                                'Peak_Intensity': intensity,
                                'Peak_Phase': 'Portlandite',
                                'd_spacing_A': 4.93 / np.sin(np.radians(peak/2)),
                                'FWHM': 0.15 + temp * 0.0001,
                                'Crystallite_Size_nm': 45 - temp * 0.02
                            })
                
                # C-S-H broad peaks
                if temp <= 600:
                    for peak in [29.3, 50.1]:
                        intensity = max(0, 300 - temp * 0.3 - rubber * 10 + np.random.normal(0, 30))
                        if intensity > 30:
                            peaks_found.append({
                                'Peak_2Theta': peak + np.random.normal(0, 0.1),
                                'Peak_Intensity': intensity,
                                'Peak_Phase': 'C-S-H',
                                'd_spacing_A': 3.04 / np.sin(np.radians(peak/2)),
                                'FWHM': 0.5 + temp * 0.0002,  # Broader peaks
                                'Crystallite_Size_nm': 10 - temp * 0.005
                            })
                
                # Quartz peaks (stable)
                for peak in [26.6, 20.9]:
                    intensity = 800 + np.random.normal(0, 40)
                    peaks_found.append({
                        'Peak_2Theta': peak + np.random.normal(0, 0.02),
                        'Peak_Intensity': intensity,
                        'Peak_Phase': 'Quartz',
                        'd_spacing_A': 3.34 / np.sin(np.radians(peak/2)),
                        'FWHM': 0.12,
                        'Crystallite_Size_nm': 100
                    })
                
                # High temperature phases
                if temp >= 500:
                    # Lime peaks
                    for peak in [37.4, 53.9]:
                        intensity = (temp - 500) * 2 + np.random.normal(0, 20)
                        if intensity > 20:
                            peaks_found.append({
                                'Peak_2Theta': peak + np.random.normal(0, 0.05),
                                'Peak_Intensity': intensity,
                                'Peak_Phase': 'Lime',
                                'd_spacing_A': 2.40 / np.sin(np.radians(peak/2)),
                                'FWHM': 0.18,
                                'Crystallite_Size_nm': 35
                            })
                
                # Create records for each peak
                for i, peak_info in enumerate(peaks_found):
                    record = {
                        'Temperature_C': temp,
                        'Rubber_Content_%': rubber,
                        'Peak_Number': i + 1,
                        'Sample_ID': f'XRD_Peak_{temp}_{rubber}_{i+1}',
                        **peak_info,
                        'Timestamp': datetime.now().isoformat()
                    }
                    data.append(record)
        
        return pd.DataFrame(data)
    
    def generate_texture_analysis(self):
        """
        Preferred orientation and texture analysis
        Important for understanding mechanical properties
        """
        data = []
        
        for temp in self.temperatures:
            for rubber in self.rubber_contents:
                
                # Texture coefficient for main phases
                tc_portlandite = 1.0 + (temp - 20) * 0.0001 + np.random.normal(0, 0.05)
                tc_csh = 1.0 - rubber * 0.01 + np.random.normal(0, 0.03)
                tc_quartz = 1.0 + np.random.normal(0, 0.02)  # Should remain constant
                
                # March-Dollase parameter (1 = random, <1 = preferred orientation)
                march_dollase = 1.0 - rubber * 0.005 - (temp - 20) * 0.00005 + np.random.normal(0, 0.02)
                
                # Pole figure intensity variation
                pole_figure_max = 1.2 + rubber * 0.01 + (temp - 20) * 0.0001
                pole_figure_min = 0.8 - rubber * 0.01 - (temp - 20) * 0.0001
                
                data.append({
                    'Temperature_C': temp,
                    'Rubber_Content_%': rubber,
                    'Texture_Coefficient_Portlandite': tc_portlandite if temp <= 500 else None,
                    'Texture_Coefficient_CSH': tc_csh if temp <= 600 else None,
                    'Texture_Coefficient_Quartz': tc_quartz,
                    'March_Dollase_Parameter': max(0.5, min(1.0, march_dollase)),
                    'Pole_Figure_Max_Intensity': pole_figure_max,
                    'Pole_Figure_Min_Intensity': max(0.1, pole_figure_min),
                    'Anisotropy_Degree': (pole_figure_max - pole_figure_min) / pole_figure_max,
                    'Timestamp': datetime.now().isoformat()
                })
        
        return pd.DataFrame(data)
    
    def save_all_datasets(self):
        """Save all XRD analysis datasets"""
        print("\nGenerating XRD Analysis Datasets...")
        
        # Generate datasets
        phase_data = self.generate_phase_quantification()
        patterns = self.generate_diffraction_patterns()
        peak_data = self.generate_peak_analysis()
        texture_data = self.generate_texture_analysis()
        
        # Save phase quantification
        phase_data.to_csv('XRD_phase_quantification.csv', index=False)
        
        # Save diffraction patterns (as JSON due to array data)
        with open('XRD_diffraction_patterns.json', 'w') as f:
            json.dump(patterns, f)
        
        # Save peak analysis
        peak_data.to_csv('XRD_peak_analysis.csv', index=False)
        
        # Save texture analysis
        texture_data.to_csv('XRD_texture_analysis.csv', index=False)
        
        # Generate summary
        summary = {
            'Dataset': 'XRD Analysis for Rubberized Concrete',
            'Generated': datetime.now().isoformat(),
            'Total_Phase_Analyses': len(phase_data),
            'Total_Diffraction_Patterns': len(patterns),
            'Total_Peaks_Analyzed': len(peak_data),
            'Total_Texture_Analyses': len(texture_data),
            'Temperature_Range_C': f"{min(self.temperatures)}-{max(self.temperatures)}",
            'Rubber_Contents_%': self.rubber_contents,
            'Key_Findings': {
                'Portlandite_Decomposition_Temp': '400-500°C',
                'CSH_Major_Decomposition': '600-800°C',
                'New_High_Temp_Phases': ['Lime (>500°C)', 'Anhydrite (>600°C)', 'Gehlenite (>600°C)'],
                'Max_Amorphous_Content_%': float(phase_data['Amorphous_wt%'].max()),
                'Crystallinity_Range': f"{phase_data['Crystallinity_Index'].min():.2f}-{phase_data['Crystallinity_Index'].max():.2f}"
            }
        }
        
        with open('XRD_analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Generated {len(phase_data)} phase quantification analyses")
        print(f"✓ Generated {len(patterns)} diffraction patterns")
        print(f"✓ Generated {len(peak_data)} peak analyses")
        print(f"✓ Generated {len(texture_data)} texture analyses")
        
        return phase_data, patterns, peak_data, texture_data

if __name__ == "__main__":
    generator = XRDDataGenerator()
    datasets = generator.save_all_datasets()
    print("\n✅ XRD Analysis Dataset Generation Complete!")