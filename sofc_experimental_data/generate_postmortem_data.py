"""
Post-Mortem Analysis Data Generator for SOFC
Generates SEM image analysis, EDS line scans, and nanoindentation data
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from scipy import ndimage, signal

class PostMortemDataGenerator:
    def __init__(self, base_path="post_mortem"):
        self.base_path = base_path
        np.random.seed(44)
        
        # Material properties for nanoindentation
        self.material_properties = {
            'YSZ': {
                'E_mean': 184.7,  # GPa
                'E_std': 8.5,
                'H_mean': 13.5,   # GPa (Hardness)
                'H_std': 1.2,
                'creep_compliance': 0.008,  # 1/GPa
                'composition': {'Zr': 82, 'Y': 8, 'O': 10}
            },
            'Ni-YSZ': {
                'E_mean': 109.8,  # GPa
                'E_std': 12.3,
                'H_mean': 6.8,
                'H_std': 1.5,
                'creep_compliance': 0.015,
                'composition': {'Ni': 40, 'Zr': 49.2, 'Y': 4.8, 'O': 6}
            },
            'Ni': {
                'E_mean': 200.0,
                'E_std': 10.0,
                'H_mean': 3.5,
                'H_std': 0.8,
                'creep_compliance': 0.025,
                'composition': {'Ni': 100}
            },
            'GDC': {
                'E_mean': 175.0,
                'E_std': 9.0,
                'H_mean': 11.2,
                'H_std': 1.0,
                'creep_compliance': 0.010,
                'composition': {'Ce': 72, 'Gd': 18, 'O': 10}
            },
            'LSM': {
                'E_mean': 120.0,
                'E_std': 15.0,
                'H_mean': 8.5,
                'H_std': 1.8,
                'creep_compliance': 0.012,
                'composition': {'La': 30, 'Sr': 10, 'Mn': 40, 'O': 20}
            }
        }
    
    def generate_sem_crack_analysis(self, experiment_type):
        """Generate SEM-based crack density quantification data"""
        print(f"  Generating SEM crack analysis for {experiment_type}...")
        
        # Define analysis regions
        regions = [
            {'name': 'Anode_Support', 'area_mm2': 1.0, 'material': 'Ni-YSZ'},
            {'name': 'Anode_Electrolyte_Interface', 'area_mm2': 0.5, 'material': 'Interface'},
            {'name': 'Electrolyte', 'area_mm2': 0.3, 'material': 'YSZ'},
            {'name': 'Electrolyte_Cathode_Interface', 'area_mm2': 0.5, 'material': 'Interface'},
            {'name': 'Cathode', 'area_mm2': 0.8, 'material': 'GDC-LSM'}
        ]
        
        data_collection = []
        
        # Base crack density depends on experiment type
        if experiment_type == 'sintering':
            base_crack_density = 2.5  # cracks/mm²
            interface_factor = 2.0
        elif experiment_type == 'thermal_cycling':
            base_crack_density = 8.5  # More cracks due to cycling
            interface_factor = 3.5
        else:  # startup_shutdown
            base_crack_density = 5.0
            interface_factor = 2.8
        
        for region in regions:
            # Higher crack density at interfaces
            if 'Interface' in region['name']:
                crack_density = base_crack_density * interface_factor
            else:
                crack_density = base_crack_density
            
            # Add statistical variation
            measured_density = np.random.poisson(crack_density * region['area_mm2']) / region['area_mm2']
            
            # Generate individual crack measurements
            n_cracks = int(measured_density * region['area_mm2'])
            
            crack_lengths = np.random.exponential(20, n_cracks)  # μm
            crack_widths = crack_lengths * 0.05 + np.random.normal(0, 0.5, n_cracks)
            crack_widths[crack_widths < 0.1] = 0.1
            
            # Crack orientation (0° = horizontal, 90° = vertical)
            if 'Interface' in region['name']:
                # Interface cracks tend to be horizontal
                crack_angles = np.random.normal(0, 15, n_cracks)
            else:
                # Random orientations in bulk
                crack_angles = np.random.uniform(-90, 90, n_cracks)
            
            # Crack types
            crack_types = np.random.choice(['transgranular', 'intergranular', 'interface'],
                                         n_cracks, p=[0.3, 0.5, 0.2])
            
            data_collection.append({
                'region': region['name'],
                'material': region['material'],
                'area_analyzed_mm2': region['area_mm2'],
                'total_cracks': n_cracks,
                'crack_density_per_mm2': measured_density,
                'mean_crack_length_um': np.mean(crack_lengths) if n_cracks > 0 else 0,
                'std_crack_length_um': np.std(crack_lengths) if n_cracks > 0 else 0,
                'max_crack_length_um': np.max(crack_lengths) if n_cracks > 0 else 0,
                'mean_crack_width_um': np.mean(crack_widths) if n_cracks > 0 else 0,
                'predominant_angle_deg': np.median(crack_angles) if n_cracks > 0 else 0,
                'transgranular_fraction': np.sum(crack_types == 'transgranular') / max(n_cracks, 1),
                'intergranular_fraction': np.sum(crack_types == 'intergranular') / max(n_cracks, 1),
                'interface_fraction': np.sum(crack_types == 'interface') / max(n_cracks, 1)
            })
            
            # Save detailed crack data
            if n_cracks > 0:
                crack_details = pd.DataFrame({
                    'crack_id': range(n_cracks),
                    'region': region['name'],
                    'length_um': crack_lengths,
                    'width_um': crack_widths,
                    'angle_deg': crack_angles,
                    'type': crack_types
                })
                crack_details.to_csv(
                    os.path.join(self.base_path, experiment_type,
                                f'crack_details_{region["name"]}.csv'),
                    index=False
                )
        
        df = pd.DataFrame(data_collection)
        df.to_csv(os.path.join(self.base_path, experiment_type,
                               'sem_crack_analysis.csv'), index=False)
        
        # Generate SEM image metadata
        sem_metadata = {
            'instrument': 'FEI Quanta 250 FEG',
            'voltage_kV': 15,
            'working_distance_mm': 10,
            'magnification': '5000x',
            'detector': 'SE2',
            'image_resolution': '4096x3536',
            'pixel_size_nm': 25,
            'analysis_software': 'ImageJ 1.53',
            'crack_detection_method': 'Thresholding + Manual verification',
            'images': [f'SEM_{region["name"]}_{i:03d}.tif' 
                      for region in regions for i in range(5)]
        }
        
        with open(os.path.join(self.base_path, experiment_type,
                               'sem_metadata.json'), 'w') as f:
            json.dump(sem_metadata, f, indent=2)
        
        return df
    
    def generate_eds_line_scans(self, experiment_type):
        """Generate EDS line scan data for elemental composition"""
        print(f"  Generating EDS line scan data for {experiment_type}...")
        
        # Line scan across SOFC layers
        scan_length_um = 600
        n_points = 300
        positions = np.linspace(0, scan_length_um, n_points)
        
        # Layer boundaries (micrometers)
        layers = [
            {'start': 0, 'end': 500, 'material': 'Ni-YSZ'},
            {'start': 500, 'end': 520, 'material': 'Ni-YSZ'},  # Functional layer
            {'start': 520, 'end': 530, 'material': 'YSZ'},
            {'start': 530, 'end': 550, 'material': 'GDC'},
            {'start': 550, 'end': 600, 'material': 'LSM'}
        ]
        
        # Initialize element concentrations
        elements = ['Ni', 'Zr', 'Y', 'Ce', 'Gd', 'La', 'Sr', 'Mn', 'O']
        concentrations = {elem: np.zeros(n_points) for elem in elements}
        
        for layer in layers:
            mask = (positions >= layer['start']) & (positions < layer['end'])
            material = layer['material']
            
            if material == 'Ni-YSZ':
                # Gradient in Ni concentration (higher near anode support)
                ni_gradient = 1 - (positions - layer['start']) / (layer['end'] - layer['start']) * 0.3
                concentrations['Ni'][mask] = 40 * ni_gradient[mask] + np.random.normal(0, 2, np.sum(mask))
                concentrations['Zr'][mask] = 49.2 - 10 * ni_gradient[mask] + np.random.normal(0, 2, np.sum(mask))
                concentrations['Y'][mask] = 4.8 + np.random.normal(0, 0.5, np.sum(mask))
                concentrations['O'][mask] = 6 + np.random.normal(0, 1, np.sum(mask))
                
            elif material == 'YSZ':
                concentrations['Zr'][mask] = 82 + np.random.normal(0, 2, np.sum(mask))
                concentrations['Y'][mask] = 8 + np.random.normal(0, 0.5, np.sum(mask))
                concentrations['O'][mask] = 10 + np.random.normal(0, 1, np.sum(mask))
                
            elif material == 'GDC':
                concentrations['Ce'][mask] = 72 + np.random.normal(0, 2, np.sum(mask))
                concentrations['Gd'][mask] = 18 + np.random.normal(0, 1, np.sum(mask))
                concentrations['O'][mask] = 10 + np.random.normal(0, 1, np.sum(mask))
                
            elif material == 'LSM':
                concentrations['La'][mask] = 30 + np.random.normal(0, 1.5, np.sum(mask))
                concentrations['Sr'][mask] = 10 + np.random.normal(0, 0.8, np.sum(mask))
                concentrations['Mn'][mask] = 40 + np.random.normal(0, 2, np.sum(mask))
                concentrations['O'][mask] = 20 + np.random.normal(0, 1, np.sum(mask))
        
        # Add interdiffusion at interfaces
        from scipy.ndimage import gaussian_filter1d
        for elem in elements:
            concentrations[elem] = gaussian_filter1d(concentrations[elem], sigma=3)
            concentrations[elem][concentrations[elem] < 0] = 0
        
        # Normalize to 100% at each point
        total = np.zeros(n_points)
        for elem in elements:
            total += concentrations[elem]
        
        for elem in elements:
            concentrations[elem] = concentrations[elem] / total * 100
        
        # Create DataFrame
        eds_data = pd.DataFrame({'position_um': positions})
        for elem in elements:
            eds_data[f'{elem}_at_percent'] = concentrations[elem]
        
        # Add experiment-specific modifications
        if experiment_type == 'thermal_cycling':
            # Ni migration/coarsening
            eds_data.loc[eds_data['position_um'] < 500, 'Ni_at_percent'] *= \
                np.random.uniform(0.85, 0.95, np.sum(eds_data['position_um'] < 500))
        elif experiment_type == 'startup_shutdown':
            # Cr poisoning at cathode (if present)
            cr_contamination = np.zeros(n_points)
            cr_contamination[positions > 550] = np.random.exponential(0.5, np.sum(positions > 550))
            eds_data['Cr_at_percent'] = cr_contamination
        
        eds_data.to_csv(os.path.join(self.base_path, experiment_type,
                                     'eds_line_scan.csv'), index=False)
        
        # Generate EDS mapping data (2D compositional maps)
        map_size = (100, 100)  # pixels
        eds_maps = {}
        
        for elem in ['Ni', 'Zr', 'Y', 'O']:
            # Create 2D concentration map with realistic features
            base_map = np.random.normal(30, 5, map_size)
            
            # Add grain boundary effects (lower concentration)
            n_grains = 20
            for _ in range(n_grains):
                x, y = np.random.randint(0, 100, 2)
                radius = np.random.randint(5, 15)
                Y, X = np.ogrid[:100, :100]
                mask = (X - x)**2 + (Y - y)**2 <= radius**2
                base_map[mask] *= np.random.uniform(0.8, 1.2)
            
            # Smooth the map
            base_map = gaussian_filter1d(gaussian_filter1d(base_map, sigma=2, axis=0), sigma=2, axis=1)
            base_map[base_map < 0] = 0
            
            eds_maps[elem] = base_map.tolist()
        
        with open(os.path.join(self.base_path, experiment_type,
                               'eds_maps.json'), 'w') as f:
            json.dump(eds_maps, f)
        
        # EDS metadata
        eds_metadata = {
            'instrument': 'Oxford Instruments X-Max 80',
            'voltage_kV': 20,
            'beam_current_nA': 1.5,
            'acquisition_time_s': 60,
            'dead_time_percent': 25,
            'elements_analyzed': elements,
            'quantification_method': 'Standardless ZAF',
            'software': 'AZtec 3.3'
        }
        
        with open(os.path.join(self.base_path, experiment_type,
                               'eds_metadata.json'), 'w') as f:
            json.dump(eds_metadata, f, indent=2)
        
        return eds_data
    
    def generate_nanoindentation_data(self, experiment_type):
        """Generate nanoindentation data (Young's modulus, hardness, creep)"""
        print(f"  Generating nanoindentation data for {experiment_type}...")
        
        # Grid of indentation points
        grid_size = (10, 10)  # 10x10 grid
        spacing_um = 50  # 50 μm spacing
        
        data_collection = []
        
        # Generate data for each material region
        for material_name, props in self.material_properties.items():
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    x_pos = i * spacing_um
                    y_pos = j * spacing_um
                    
                    # Young's modulus with spatial variation
                    E_base = props['E_mean']
                    E_variation = props['E_std']
                    
                    # Add degradation effects based on experiment type
                    if experiment_type == 'thermal_cycling':
                        # Microcracking reduces modulus
                        degradation_factor = np.random.uniform(0.85, 0.98)
                    elif experiment_type == 'startup_shutdown':
                        degradation_factor = np.random.uniform(0.90, 0.99)
                    else:
                        degradation_factor = 1.0
                    
                    E_measured = (E_base + np.random.normal(0, E_variation)) * degradation_factor
                    
                    # Hardness
                    H_measured = props['H_mean'] + np.random.normal(0, props['H_std'])
                    H_measured *= degradation_factor
                    
                    # Load-displacement curve parameters
                    max_load_mN = 10
                    max_depth_nm = np.sqrt(max_load_mN * 1000 / (H_measured * np.pi)) * 1000
                    
                    # Creep compliance (time-dependent deformation)
                    creep_compliance = props['creep_compliance'] * (2 - degradation_factor)
                    creep_depth_nm = max_depth_nm * creep_compliance * np.log(1 + 30)  # 30s hold
                    
                    # Contact area
                    contact_area_nm2 = max_load_mN * 1e6 / H_measured
                    
                    # Reduced modulus
                    E_r = E_measured / (1 - 0.3**2)  # Assuming Poisson's ratio = 0.3
                    
                    data_collection.append({
                        'material': material_name,
                        'x_position_um': x_pos,
                        'y_position_um': y_pos,
                        'youngs_modulus_GPa': E_measured,
                        'hardness_GPa': H_measured,
                        'reduced_modulus_GPa': E_r,
                        'max_load_mN': max_load_mN,
                        'max_depth_nm': max_depth_nm,
                        'contact_area_nm2': contact_area_nm2,
                        'creep_compliance_1_GPa': creep_compliance,
                        'creep_depth_nm': creep_depth_nm,
                        'plastic_work_pJ': max_load_mN * max_depth_nm * 0.7,  # 70% plastic
                        'elastic_work_pJ': max_load_mN * max_depth_nm * 0.3,  # 30% elastic
                        'oliver_pharr_beta': 1.034,  # Berkovich tip
                        'measurement_temp_C': 25,
                        'drift_rate_nm_s': np.random.uniform(0.01, 0.05)
                    })
        
        df = pd.DataFrame(data_collection)
        df.to_csv(os.path.join(self.base_path, experiment_type,
                               'nanoindentation_grid.csv'), index=False)
        
        # Generate load-displacement curves for selected indents
        selected_indents = df.sample(n=10)
        load_curves = []
        
        for _, indent in selected_indents.iterrows():
            # Generate load-displacement curve
            n_points = 200
            
            # Loading segment
            load_depths = np.linspace(0, indent['max_depth_nm'], n_points//2)
            loads = (load_depths / indent['max_depth_nm'])**2 * indent['max_load_mN']
            
            # Unloading segment (follows power law)
            unload_depths = np.linspace(indent['max_depth_nm'], 
                                       indent['max_depth_nm'] * 0.2, n_points//2)
            unload_loads = indent['max_load_mN'] * \
                          ((unload_depths - indent['max_depth_nm'] * 0.2) / 
                           (indent['max_depth_nm'] * 0.8))**1.5
            
            curve_data = {
                'material': indent['material'],
                'x_position_um': indent['x_position_um'],
                'y_position_um': indent['y_position_um'],
                'loading_depth_nm': load_depths.tolist(),
                'loading_force_mN': loads.tolist(),
                'unloading_depth_nm': unload_depths.tolist(),
                'unloading_force_mN': unload_loads.tolist()
            }
            load_curves.append(curve_data)
        
        with open(os.path.join(self.base_path, experiment_type,
                               'load_displacement_curves.json'), 'w') as f:
            json.dump(load_curves, f, indent=2)
        
        # Generate modulus maps
        for material in self.material_properties.keys():
            material_data = df[df['material'] == material]
            if len(material_data) > 0:
                # Create 2D interpolated map
                from scipy.interpolate import griddata
                
                xi = np.linspace(0, grid_size[0] * spacing_um, 50)
                yi = np.linspace(0, grid_size[1] * spacing_um, 50)
                xi, yi = np.meshgrid(xi, yi)
                
                points = material_data[['x_position_um', 'y_position_um']].values
                values = material_data['youngs_modulus_GPa'].values
                
                if len(points) > 3:  # Need at least 4 points for cubic interpolation
                    zi = griddata(points, values, (xi, yi), method='cubic')
                    
                    map_data = {
                        'material': material,
                        'x_coords_um': xi[0, :].tolist(),
                        'y_coords_um': yi[:, 0].tolist(),
                        'modulus_map_GPa': np.nan_to_num(zi, nan=np.mean(values)).tolist()
                    }
                    
                    with open(os.path.join(self.base_path, experiment_type,
                                          f'modulus_map_{material}.json'), 'w') as f:
                        json.dump(map_data, f)
        
        # Nanoindentation metadata
        nano_metadata = {
            'instrument': 'Hysitron TI 950 TriboIndenter',
            'tip_type': 'Berkovich diamond',
            'tip_radius_nm': 150,
            'approach_velocity_nm_s': 10,
            'loading_rate_mN_s': 0.5,
            'hold_time_s': 30,
            'unloading_rate_mN_s': 0.5,
            'thermal_drift_correction': True,
            'frame_stiffness_N_m': 1e6,
            'poisson_ratio_sample': 0.3,
            'poisson_ratio_indenter': 0.07,
            'youngs_modulus_indenter_GPa': 1140
        }
        
        with open(os.path.join(self.base_path, experiment_type,
                               'nanoindentation_metadata.json'), 'w') as f:
            json.dump(nano_metadata, f, indent=2)
        
        return df
    
    def generate_porosity_analysis(self, experiment_type):
        """Generate porosity and microstructure analysis data"""
        print(f"  Generating porosity analysis data for {experiment_type}...")
        
        materials = ['Ni-YSZ_anode', 'YSZ_electrolyte', 'GDC_cathode']
        
        data_collection = []
        
        for material in materials:
            # Base porosity values
            if 'anode' in material:
                base_porosity = 0.35  # 35% porosity for anode
                pore_size_mean = 2.5  # μm
            elif 'electrolyte' in material:
                base_porosity = 0.02  # 2% porosity for dense electrolyte
                pore_size_mean = 0.1
            else:  # cathode
                base_porosity = 0.30  # 30% porosity for cathode
                pore_size_mean = 1.8
            
            # Modify based on experiment type
            if experiment_type == 'sintering':
                porosity = base_porosity * np.random.uniform(0.9, 1.0)
            elif experiment_type == 'thermal_cycling':
                # Coarsening increases pore size, may change porosity
                porosity = base_porosity * np.random.uniform(0.95, 1.05)
                pore_size_mean *= np.random.uniform(1.1, 1.3)
            else:
                porosity = base_porosity * np.random.uniform(0.98, 1.02)
            
            # Generate pore size distribution (log-normal)
            n_pores = 500
            pore_sizes = np.random.lognormal(np.log(pore_size_mean), 0.5, n_pores)
            
            # Triple phase boundary length (important for electrodes)
            if 'electrolyte' not in material:
                tpb_length = 5e12 * (1 - np.exp(-porosity/0.2))  # m/m³
            else:
                tpb_length = 0
            
            data_collection.append({
                'material': material,
                'porosity_fraction': porosity,
                'porosity_percent': porosity * 100,
                'mean_pore_size_um': np.mean(pore_sizes),
                'median_pore_size_um': np.median(pore_sizes),
                'std_pore_size_um': np.std(pore_sizes),
                'd10_pore_size_um': np.percentile(pore_sizes, 10),
                'd50_pore_size_um': np.percentile(pore_sizes, 50),
                'd90_pore_size_um': np.percentile(pore_sizes, 90),
                'total_pore_count': n_pores,
                'pore_connectivity': np.random.uniform(0.7, 0.95),
                'tortuosity': 1 / porosity**0.5 if porosity > 0 else np.inf,
                'tpb_length_m_per_m3': tpb_length,
                'surface_area_m2_per_m3': 6 / pore_size_mean * 1e6 * porosity,
                'permeability_m2': porosity**3 * (pore_size_mean * 1e-6)**2 / 180
            })
            
            # Save pore size distribution
            pore_dist = pd.DataFrame({
                'pore_diameter_um': pore_sizes,
                'material': material
            })
            pore_dist.to_csv(os.path.join(self.base_path, experiment_type,
                                         f'pore_distribution_{material}.csv'),
                            index=False)
        
        df = pd.DataFrame(data_collection)
        df.to_csv(os.path.join(self.base_path, experiment_type,
                               'porosity_analysis.csv'), index=False)
        
        return df
    
    def run_all(self):
        """Generate all post-mortem analysis datasets"""
        for exp_type in ['sintering', 'thermal_cycling', 'startup_shutdown']:
            print(f"\nGenerating post-mortem data for {exp_type}...")
            self.generate_sem_crack_analysis(exp_type)
            self.generate_eds_line_scans(exp_type)
            self.generate_nanoindentation_data(exp_type)
            self.generate_porosity_analysis(exp_type)
        
        print("\nPost-mortem data generation complete!")

if __name__ == "__main__":
    generator = PostMortemDataGenerator()
    generator.run_all()