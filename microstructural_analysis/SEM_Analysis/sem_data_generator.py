"""
SEM Analysis Data Generator for Rubberized Concrete under Thermal Loading
PhD Research: Thermo-Mechanical Model for Fire-Resistant Structural Elements
Focus: ITZ characterization, microcracking evolution, rubber degradation morphology
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats, ndimage
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

class SEMDataGenerator:
    """
    Generate realistic SEM analysis data for rubberized concrete specimens
    including ITZ measurements, crack analysis, and rubber degradation assessment
    """
    
    def __init__(self):
        self.temperatures = [20, 200, 400, 600, 800]  # °C
        self.rubber_contents = [0, 5, 10, 15, 20]  # % by volume
        self.magnifications = [500, 1000, 2500, 5000, 10000]  # X
        self.specimen_ages = [28, 56, 90]  # days
        
    def generate_itz_measurements(self):
        """
        Generate ITZ (Interfacial Transition Zone) measurements
        Critical for understanding rubber-cement and aggregate-cement bonding
        """
        data = []
        
        for temp in self.temperatures:
            for rubber in self.rubber_contents:
                for age in self.specimen_ages:
                    # ITZ thickness increases with temperature and rubber content
                    base_itz_rubber = 15 + rubber * 2.5  # μm
                    base_itz_aggregate = 10 + rubber * 0.5  # μm
                    
                    # Temperature effect on ITZ
                    temp_factor = 1 + (temp - 20) / 1000
                    
                    # Age effect (hydration reduces ITZ initially)
                    age_factor = 1 - 0.1 * np.log(age/28)
                    
                    # Generate multiple measurements per condition
                    for i in range(30):  # 30 measurements per condition
                        # Add realistic variation
                        noise_rubber = np.random.normal(0, 2)
                        noise_aggregate = np.random.normal(0, 1.5)
                        
                        itz_rubber_cement = base_itz_rubber * temp_factor * age_factor + noise_rubber
                        itz_aggregate_cement = base_itz_aggregate * temp_factor * age_factor + noise_aggregate
                        
                        # Porosity in ITZ (critical for permeability)
                        porosity_rubber_itz = 0.25 + rubber/100 + temp/2000 + np.random.normal(0, 0.02)
                        porosity_aggregate_itz = 0.20 + rubber/200 + temp/3000 + np.random.normal(0, 0.015)
                        
                        # Ca/Si ratio in ITZ (indicates C-S-H quality)
                        ca_si_rubber = 1.7 - rubber/100 - temp/2000 + np.random.normal(0, 0.1)
                        ca_si_aggregate = 1.8 - rubber/150 - temp/2500 + np.random.normal(0, 0.08)
                        
                        # Microhardness (Vickers hardness)
                        hardness_rubber_itz = 40 - rubber*0.8 - temp*0.02 + np.random.normal(0, 3)
                        hardness_aggregate_itz = 50 - rubber*0.3 - temp*0.025 + np.random.normal(0, 2.5)
                        
                        data.append({
                            'Temperature_C': temp,
                            'Rubber_Content_%': rubber,
                            'Age_days': age,
                            'Measurement_ID': f'SEM_{temp}_{rubber}_{age}_{i+1}',
                            'ITZ_Rubber_Cement_um': max(5, itz_rubber_cement),
                            'ITZ_Aggregate_Cement_um': max(3, itz_aggregate_cement),
                            'Porosity_Rubber_ITZ': min(0.6, max(0.1, porosity_rubber_itz)),
                            'Porosity_Aggregate_ITZ': min(0.5, max(0.08, porosity_aggregate_itz)),
                            'Ca_Si_Ratio_Rubber_ITZ': max(0.8, ca_si_rubber),
                            'Ca_Si_Ratio_Aggregate_ITZ': max(1.0, ca_si_aggregate),
                            'Microhardness_Rubber_ITZ_HV': max(10, hardness_rubber_itz),
                            'Microhardness_Aggregate_ITZ_HV': max(15, hardness_aggregate_itz),
                            'Timestamp': datetime.now().isoformat()
                        })
        
        return pd.DataFrame(data)
    
    def generate_microcrack_analysis(self):
        """
        Generate microcrack analysis data from SEM imaging
        Critical for understanding failure mechanisms
        """
        data = []
        
        for temp in self.temperatures:
            for rubber in self.rubber_contents:
                for mag in [1000, 2500, 5000]:  # Typical magnifications for crack analysis
                    
                    # Base crack density (cracks/mm²)
                    if temp <= 200:
                        base_crack_density = 0.5 + rubber * 0.02
                    elif temp <= 400:
                        base_crack_density = 2.0 + rubber * 0.1
                    elif temp <= 600:
                        base_crack_density = 8.0 + rubber * 0.3
                    else:
                        base_crack_density = 20.0 + rubber * 0.5
                    
                    # Generate multiple fields of view
                    for fov in range(20):  # 20 fields of view per condition
                        
                        # Crack characteristics
                        crack_density = base_crack_density * np.random.lognormal(0, 0.3)
                        
                        # Average crack width (μm)
                        avg_crack_width = (0.5 + temp/500 + rubber/50) * np.random.lognormal(0, 0.2)
                        
                        # Maximum crack width
                        max_crack_width = avg_crack_width * np.random.uniform(2, 5)
                        
                        # Crack length distribution parameters
                        avg_crack_length = (10 + temp/20 + rubber) * np.random.lognormal(0, 0.25)
                        
                        # Crack orientation (0° = horizontal, 90° = vertical)
                        # Thermal cracks tend to be more random at higher temps
                        if temp > 400:
                            orientation_std = 45  # More random
                        else:
                            orientation_std = 20  # More aligned
                        
                        dominant_orientation = np.random.uniform(0, 180)
                        
                        # Crack connectivity (0-1, important for permeability)
                        connectivity = min(1, 0.1 + temp/1000 + rubber/50 + np.random.normal(0, 0.05))
                        
                        # Fractal dimension (complexity of crack pattern)
                        fractal_dim = 1.2 + temp/2000 + rubber/100 + np.random.normal(0, 0.05)
                        
                        # Crack types classification
                        if temp <= 200:
                            crack_type = np.random.choice(['Shrinkage', 'Interface', 'Mechanical'], 
                                                         p=[0.5, 0.3, 0.2])
                        elif temp <= 400:
                            crack_type = np.random.choice(['Thermal', 'Interface', 'Dehydration'], 
                                                         p=[0.4, 0.3, 0.3])
                        else:
                            crack_type = np.random.choice(['Thermal', 'Decomposition', 'Spalling'], 
                                                         p=[0.5, 0.3, 0.2])
                        
                        data.append({
                            'Temperature_C': temp,
                            'Rubber_Content_%': rubber,
                            'Magnification_X': mag,
                            'Field_of_View': fov + 1,
                            'Image_ID': f'SEM_CRACK_{temp}_{rubber}_{mag}_{fov+1}',
                            'Crack_Density_per_mm2': max(0, crack_density),
                            'Avg_Crack_Width_um': max(0.1, avg_crack_width),
                            'Max_Crack_Width_um': max(0.2, max_crack_width),
                            'Avg_Crack_Length_um': max(1, avg_crack_length),
                            'Dominant_Orientation_deg': dominant_orientation,
                            'Orientation_StdDev_deg': orientation_std,
                            'Crack_Connectivity': max(0, min(1, connectivity)),
                            'Fractal_Dimension': min(2, max(1, fractal_dim)),
                            'Primary_Crack_Type': crack_type,
                            'Area_Fraction_Cracks_%': min(30, crack_density * avg_crack_width / 100),
                            'Timestamp': datetime.now().isoformat()
                        })
        
        return pd.DataFrame(data)
    
    def generate_rubber_degradation_morphology(self):
        """
        Generate rubber particle degradation analysis
        PhD-level analysis of rubber transformation under heating
        """
        data = []
        
        for temp in self.temperatures:
            for rubber in self.rubber_contents[1:]:  # Skip 0% rubber
                for particle_size in ['Fine(0-1mm)', 'Medium(1-2mm)', 'Coarse(2-4mm)']:
                    
                    # Analyze multiple particles
                    for particle_id in range(25):  # 25 particles per condition
                        
                        # Initial particle characteristics
                        if particle_size == 'Fine(0-1mm)':
                            initial_size = np.random.uniform(0.2, 1.0)
                        elif particle_size == 'Medium(1-2mm)':
                            initial_size = np.random.uniform(1.0, 2.0)
                        else:
                            initial_size = np.random.uniform(2.0, 4.0)
                        
                        # Degradation characteristics
                        if temp <= 200:
                            # Minimal degradation
                            size_change = np.random.uniform(-5, 2)  # % change
                            pore_formation = np.random.uniform(0, 5)  # % area
                            surface_roughness = np.random.uniform(1.1, 1.3)  # Ra increase factor
                            degradation_state = 'Intact'
                            
                        elif temp <= 400:
                            # Onset of pyrolysis
                            size_change = np.random.uniform(-15, -5)
                            pore_formation = np.random.uniform(5, 20)
                            surface_roughness = np.random.uniform(1.5, 2.5)
                            degradation_state = 'Partial_Pyrolysis'
                            
                        elif temp <= 600:
                            # Significant pyrolysis
                            size_change = np.random.uniform(-40, -20)
                            pore_formation = np.random.uniform(20, 50)
                            surface_roughness = np.random.uniform(2.5, 4.0)
                            degradation_state = 'Advanced_Pyrolysis'
                            
                        else:
                            # Complete degradation
                            size_change = np.random.uniform(-70, -45)
                            pore_formation = np.random.uniform(50, 80)
                            surface_roughness = np.random.uniform(4.0, 6.0)
                            degradation_state = 'Carbonized'
                        
                        # Gap formation at rubber-paste interface
                        gap_width = max(0, (temp - 200) * 0.02 + np.random.normal(0, 2))
                        
                        # Chemical composition changes (EDS data simulation)
                        carbon_content = 85 + temp * 0.01 + np.random.normal(0, 2)  # wt%
                        oxygen_content = 10 - temp * 0.008 + np.random.normal(0, 1)  # wt%
                        sulfur_content = 2.5 - temp * 0.002 + np.random.normal(0, 0.3)  # wt%
                        zinc_content = 1.5 - temp * 0.001 + np.random.normal(0, 0.2)  # wt% (from vulcanization)
                        
                        # Void characteristics within degraded rubber
                        void_diameter_avg = 0.5 + temp * 0.01 + np.random.normal(0, 0.1)  # μm
                        void_density = temp * 0.5 + np.random.normal(0, 10)  # voids/mm²
                        
                        # Adhesion quality score (1-10)
                        adhesion_score = max(1, 10 - temp * 0.01 - np.random.normal(0, 0.5))
                        
                        data.append({
                            'Temperature_C': temp,
                            'Rubber_Content_%': rubber,
                            'Particle_Size_Class': particle_size,
                            'Particle_ID': particle_id + 1,
                            'Initial_Size_mm': initial_size,
                            'Size_Change_%': size_change,
                            'Final_Size_mm': initial_size * (1 + size_change/100),
                            'Pore_Formation_%': pore_formation,
                            'Surface_Roughness_Factor': surface_roughness,
                            'Degradation_State': degradation_state,
                            'Interface_Gap_um': max(0, gap_width),
                            'Carbon_Content_wt%': min(95, max(70, carbon_content)),
                            'Oxygen_Content_wt%': max(1, oxygen_content),
                            'Sulfur_Content_wt%': max(0.1, sulfur_content),
                            'Zinc_Content_wt%': max(0.1, zinc_content),
                            'Internal_Void_Diameter_um': max(0.1, void_diameter_avg),
                            'Internal_Void_Density_per_mm2': max(0, void_density),
                            'Adhesion_Quality_Score': adhesion_score,
                            'Timestamp': datetime.now().isoformat()
                        })
        
        return pd.DataFrame(data)
    
    def generate_paste_morphology_analysis(self):
        """
        Analyze cement paste morphology changes with temperature
        """
        data = []
        
        for temp in self.temperatures:
            for rubber in self.rubber_contents:
                # Multiple analysis regions
                for region_id in range(15):
                    
                    # C-S-H gel characteristics
                    if temp <= 200:
                        csh_morphology = 'Fibrillar'
                        csh_density = 0.85 - rubber * 0.01 + np.random.normal(0, 0.02)
                    elif temp <= 400:
                        csh_morphology = 'Foil-like'
                        csh_density = 0.75 - rubber * 0.015 + np.random.normal(0, 0.03)
                    elif temp <= 600:
                        csh_morphology = 'Degraded'
                        csh_density = 0.50 - rubber * 0.02 + np.random.normal(0, 0.04)
                    else:
                        csh_morphology = 'Decomposed'
                        csh_density = 0.20 - rubber * 0.01 + np.random.normal(0, 0.05)
                    
                    # Portlandite crystals
                    ch_crystal_size = max(0.5, 5 - temp * 0.005 + np.random.normal(0, 0.5))  # μm
                    ch_content = max(0, 25 - temp * 0.03 - rubber * 0.2 + np.random.normal(0, 2))  # area %
                    
                    # Ettringite needles
                    if temp <= 70:
                        ettringite_present = True
                        ettringite_length = np.random.uniform(2, 10)  # μm
                    else:
                        ettringite_present = False
                        ettringite_length = 0
                    
                    # Capillary pores
                    capillary_porosity = 0.15 + temp * 0.0002 + rubber * 0.01 + np.random.normal(0, 0.02)
                    avg_pore_diameter = 0.05 + temp * 0.0001 + rubber * 0.002 + np.random.normal(0, 0.01)  # μm
                    
                    # Microcracks in paste
                    paste_crack_density = max(0, temp * 0.01 - 1 + rubber * 0.05 + np.random.normal(0, 0.5))
                    
                    data.append({
                        'Temperature_C': temp,
                        'Rubber_Content_%': rubber,
                        'Region_ID': region_id + 1,
                        'CSH_Morphology': csh_morphology,
                        'CSH_Relative_Density': max(0.1, min(1, csh_density)),
                        'Portlandite_Crystal_Size_um': ch_crystal_size,
                        'Portlandite_Area_%': ch_content,
                        'Ettringite_Present': ettringite_present,
                        'Ettringite_Length_um': ettringite_length,
                        'Capillary_Porosity': min(0.5, max(0.05, capillary_porosity)),
                        'Avg_Pore_Diameter_um': avg_pore_diameter,
                        'Paste_Crack_Density_per_mm': paste_crack_density,
                        'Timestamp': datetime.now().isoformat()
                    })
        
        return pd.DataFrame(data)
    
    def generate_eds_elemental_mapping(self):
        """
        Generate EDS (Energy Dispersive Spectroscopy) elemental mapping data
        """
        data = []
        
        elements = ['Ca', 'Si', 'Al', 'Fe', 'Mg', 'S', 'K', 'Na', 'C', 'O']
        
        for temp in self.temperatures:
            for rubber in self.rubber_contents:
                # Multiple mapping areas
                for map_id in range(10):
                    
                    elem_data = {
                        'Temperature_C': temp,
                        'Rubber_Content_%': rubber,
                        'Map_ID': f'EDS_{temp}_{rubber}_{map_id+1}',
                        'Analysis_Area_um2': 10000  # 100x100 μm
                    }
                    
                    # Element concentrations change with temperature
                    if temp <= 400:
                        elem_data['Ca_wt%'] = 25 - rubber * 0.5 + np.random.normal(0, 1)
                        elem_data['Si_wt%'] = 15 - rubber * 0.3 + np.random.normal(0, 0.8)
                        elem_data['O_wt%'] = 45 - rubber * 0.2 + np.random.normal(0, 1.5)
                        elem_data['C_wt%'] = 2 + rubber * 0.8 + np.random.normal(0, 0.3)
                    else:
                        elem_data['Ca_wt%'] = 30 - rubber * 0.4 + np.random.normal(0, 1.2)
                        elem_data['Si_wt%'] = 18 - rubber * 0.2 + np.random.normal(0, 0.9)
                        elem_data['O_wt%'] = 40 - rubber * 0.3 + np.random.normal(0, 1.8)
                        elem_data['C_wt%'] = 1 + rubber * 0.6 + np.random.normal(0, 0.2)
                    
                    elem_data['Al_wt%'] = 3 + np.random.normal(0, 0.2)
                    elem_data['Fe_wt%'] = 2 + np.random.normal(0, 0.15)
                    elem_data['Mg_wt%'] = 1.5 + np.random.normal(0, 0.1)
                    elem_data['S_wt%'] = 1 + rubber * 0.05 + np.random.normal(0, 0.1)
                    elem_data['K_wt%'] = 0.8 + np.random.normal(0, 0.05)
                    elem_data['Na_wt%'] = 0.5 + np.random.normal(0, 0.03)
                    
                    # Normalize to 100%
                    total = sum([elem_data[f'{e}_wt%'] for e in elements])
                    for e in elements:
                        elem_data[f'{e}_wt%'] = max(0, (elem_data[f'{e}_wt%'] / total) * 100)
                    
                    elem_data['Timestamp'] = datetime.now().isoformat()
                    data.append(elem_data)
        
        return pd.DataFrame(data)
    
    def save_all_datasets(self):
        """Save all generated datasets"""
        print("Generating SEM Analysis Datasets...")
        
        # Generate all datasets
        itz_data = self.generate_itz_measurements()
        crack_data = self.generate_microcrack_analysis()
        rubber_data = self.generate_rubber_degradation_morphology()
        paste_data = self.generate_paste_morphology_analysis()
        eds_data = self.generate_eds_elemental_mapping()
        
        # Save to CSV files
        itz_data.to_csv('ITZ_measurements.csv', index=False)
        crack_data.to_csv('microcrack_analysis.csv', index=False)
        rubber_data.to_csv('rubber_degradation_morphology.csv', index=False)
        paste_data.to_csv('paste_morphology.csv', index=False)
        eds_data.to_csv('EDS_elemental_mapping.csv', index=False)
        
        # Generate summary statistics
        summary = {
            'Dataset': 'SEM Analysis for Rubberized Concrete',
            'Generated': datetime.now().isoformat(),
            'Total_ITZ_Measurements': len(itz_data),
            'Total_Crack_Analyses': len(crack_data),
            'Total_Rubber_Particles_Analyzed': len(rubber_data),
            'Total_Paste_Regions': len(paste_data),
            'Total_EDS_Maps': len(eds_data),
            'Temperature_Range_C': f"{min(self.temperatures)}-{max(self.temperatures)}",
            'Rubber_Contents_%': self.rubber_contents,
            'Analysis_Parameters': {
                'ITZ_Thickness_Range_um': f"{itz_data['ITZ_Rubber_Cement_um'].min():.1f}-{itz_data['ITZ_Rubber_Cement_um'].max():.1f}",
                'Max_Crack_Density_per_mm2': f"{crack_data['Crack_Density_per_mm2'].max():.1f}",
                'Rubber_Degradation_States': rubber_data['Degradation_State'].unique().tolist()
            }
        }
        
        with open('SEM_analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Generated {len(itz_data)} ITZ measurements")
        print(f"✓ Generated {len(crack_data)} microcrack analyses")
        print(f"✓ Generated {len(rubber_data)} rubber degradation analyses")
        print(f"✓ Generated {len(paste_data)} paste morphology analyses")
        print(f"✓ Generated {len(eds_data)} EDS elemental maps")
        
        return itz_data, crack_data, rubber_data, paste_data, eds_data

if __name__ == "__main__":
    generator = SEMDataGenerator()
    datasets = generator.save_all_datasets()
    print("\n✅ SEM Analysis Dataset Generation Complete!")