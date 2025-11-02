#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOFC????????????
????"????????DIC-FEM??"???????????????????

?????????
???2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
import h5py
import json
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SOFCDatasetGenerator:
    """SOFC????????????"""
    
    def __init__(self, output_dir="./sofc_dataset"):
        """
        ?????????
        
        ??:
            output_dir: ??????
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ?????
        self.dirs = {
            'sem': self.output_dir / 'SEM_images',
            'muct_initial': self.output_dir / 'muCT_initial',
            'muct_sintered': self.output_dir / 'muCT_sintered',
            'eds': self.output_dir / 'EDS_maps',
            'material_props': self.output_dir / 'material_properties',
            'fem_geometry': self.output_dir / 'FEM_geometry',
            'constitutive': self.output_dir / 'constitutive_models'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # ??????????????
        self.material_properties = self._initialize_material_properties()
        
        # ??????
        self.constitutive_models = self._initialize_constitutive_models()
    
    def _initialize_material_properties(self):
        """?????????"""
        return {
            '8YSZ_Electrolyte': {
                'Young_Modulus_GPa': {'RT': 200, '800C': 170, 'equation': 'linear'},
                'Poisson_Ratio': {'RT': 0.23, '800C': 0.23, 'equation': 'constant'},
                'CTE_1e6_perK': {'RT': 10.0, '800C': 10.5, 'equation': 'linear'},
                'Thermal_Conductivity_W_mK': {'RT': 2.1, '800C': 2.3, 'equation': 'linear'},
                'Density_kg_m3': 5900,
                'Fracture_Strength_MPa': 165,
                'Fracture_Toughness_MPa_sqrtm': 3.0,
                'Creep_Parameters': {
                    'B': 8.5e-12,  # s^-1 MPa^-n
                    'n': 1.8,
                    'Q': 385000,  # J/mol
                }
            },
            'Ni_YSZ_Anode': {
                'Young_Modulus_GPa': {'RT': 55, '800C': 29, 'equation': 'exponential'},
                'Poisson_Ratio': {'RT': 0.29, '800C': 0.29, 'equation': 'constant'},
                'CTE_1e6_perK': {'RT': 12.5, '800C': 13.3, 'equation': 'linear'},
                'Thermal_Conductivity_W_mK': {'RT': 6.0, '800C': 4.5, 'equation': 'linear'},
                'Density_kg_m3': 6400,
                'Porosity': 0.25,  # ?????
                'Ni_content_vol': 0.4
            },
            'LSM_Cathode': {
                'Young_Modulus_GPa': {'RT': 45, '800C': 40, 'equation': 'linear'},
                'Poisson_Ratio': {'RT': 0.25, '800C': 0.25, 'equation': 'constant'},
                'CTE_1e6_perK': {'RT': 11.5, '800C': 12.0, 'equation': 'linear'},
                'Thermal_Conductivity_W_mK': {'RT': 3.5, '800C': 2.8, 'equation': 'linear'},
                'Density_kg_m3': 5800,
                'Porosity': 0.30
            },
            'Crofer22APU_Interconnect': {
                'Young_Modulus_GPa': {'RT': 160, '800C': 140, 'equation': 'linear'},
                'Poisson_Ratio': {'RT': 0.30, '800C': 0.30, 'equation': 'constant'},
                'CTE_1e6_perK': {'RT': 11.5, '800C': 11.9, 'equation': 'linear'},
                'Density_kg_m3': 7700,
                'Yield_Strength_MPa': {'RT': 450, '800C': 180},
                'Creep_Parameters': {
                    'A': 1.5e-20,  # s^-1 MPa^-n
                    'n': 5.2,
                    'Q': 280000,  # J/mol
                }
            }
        }
    
    def _initialize_constitutive_models(self):
        """?????????"""
        return {
            'Linear_Elastic': {
                'model_type': 'elastic',
                'parameters': {
                    'E_function': 'E(T) = E0 - alpha_E * (T - T0)',
                    'E0_GPa': 200,
                    'alpha_E_GPa_perK': 0.0375,
                    'T0_K': 298,
                    'nu': 0.23
                }
            },
            'Viscoelastic_Norton_Bailey': {
                'model_type': 'viscoelastic',
                'parameters': {
                    'creep_law': 'epsilon_dot = B * sigma^n * exp(-Q/RT)',
                    'B': 8.5e-12,  # s^-1 MPa^-n
                    'n': 1.8,
                    'Q_J_per_mol': 385000,
                    'R_J_per_molK': 8.314,
                    'elastic_properties': {
                        'E0_GPa': 200,
                        'alpha_E_GPa_perK': 0.0375,
                        'nu': 0.23
                    }
                }
            },
            'Generalized_Maxwell': {
                'model_type': 'viscoelastic_multiple_relaxation',
                'parameters': {
                    'relaxation_times_s': [1e2, 1e4, 1e6, 1e8],
                    'relaxation_moduli_GPa': [5, 10, 15, 20],
                    'equilibrium_modulus_GPa': 170,
                    'Poisson_ratio': 0.23
                }
            }
        }
    
    def generate_sem_image(self, layer_type='cross_section', state='initial', 
                          resolution=(2048, 2048), pixel_size_nm=50):
        """
        ????SEM??
        
        ??:
            layer_type: 'cross_section' ? 'interface'
            state: 'initial' ? 'sintered'
            resolution: ????? (height, width)
            pixel_size_nm: ????????
        
        ??:
            SEM????????
        """
        height, width = resolution
        
        if layer_type == 'cross_section':
            # ?????????????
            image = np.zeros((height, width), dtype=np.float32)
            
            # ?????????
            layer_thicknesses = {
                'anode': int(300e-6 / (pixel_size_nm * 1e-9)),  # 300??
                'electrolyte': int(150e-6 / (pixel_size_nm * 1e-9)),  # 150??
                'cathode': int(50e-6 / (pixel_size_nm * 1e-9)),  # 50??
            }
            
            y_pos = 0
            layers = []
            
            # ???
            anode_bottom = y_pos + layer_thicknesses['anode']
            # ???????
            roughness = np.random.normal(0, 5, width)  # 5?????
            interface_y = anode_bottom + roughness
            interface_y = np.clip(interface_y, y_pos, height-1).astype(int)
            
            # ?????????
            porosity_mask = np.random.random((height, width)) < 0.02  # 2%??
            image[0:anode_bottom] = 120  # ?????
            image[0:anode_bottom][porosity_mask[0:anode_bottom]] = 50  # ????
            
            y_pos = anode_bottom
            layers.append(('anode', 0, anode_bottom))
            
            # ????
            electrolyte_bottom = min(y_pos + layer_thicknesses['electrolyte'], height)
            if y_pos < height:
                image[y_pos:electrolyte_bottom] = 200  # ?????
            # ????????????
            if state == 'sintered':
                # ?????????
                num_cracks = np.random.randint(5, 15)
                for _ in range(num_cracks):
                    valid_y_max = min(electrolyte_bottom, height)
                    if y_pos < valid_y_max:
                        crack_y = np.random.randint(y_pos, valid_y_max)
                        crack_length = np.random.randint(10, min(50, width))
                        crack_start = np.random.randint(0, max(1, width - crack_length))
                        crack_end = min(crack_start + crack_length, width)
                        if 0 <= crack_y < height and crack_start < width:
                            image[crack_y, crack_start:crack_end] = 50
            
            y_pos = electrolyte_bottom
            layers.append(('electrolyte', anode_bottom, electrolyte_bottom))
            
            # ???
            cathode_bottom = min(y_pos + layer_thicknesses['cathode'], height)
            image[y_pos:cathode_bottom] = 180
            layers.append(('cathode', y_pos, cathode_bottom))
            
            # ?????????SEM???
            image = image + np.random.normal(0, 10, image.shape)
            image = np.clip(image, 0, 255).astype(np.uint8)
            
            # ?????????SEM??????
            image = gaussian_filter(image.astype(float), sigma=1.0).astype(np.uint8)
            
        elif layer_type == 'interface':
            # ??????????
            image = np.ones((height, width), dtype=np.float32) * 150
            
            # ????????
            interface_y = height // 2
            
            # ?????????
            image[0:interface_y] = 200
            # ????????
            image[interface_y:] = 120
            
            # ???????
            roughness_profile = np.random.normal(0, 3, width)
            for x in range(width):
                y_offset = int(roughness_profile[x])
                y_pos = np.clip(interface_y + y_offset, 0, height-1)
                # ??????
                transition_width = 5
                for dy in range(-transition_width, transition_width+1):
                    y = y_pos + dy
                    if 0 <= y < height:
                        blend = 1 - abs(dy) / transition_width
                        image[y, x] = 200 * blend + 120 * (1 - blend)
            
            # ?????????????
            if state == 'sintered':
                num_defects = np.random.randint(3, 8)
                for _ in range(num_defects):
                    defect_x = np.random.randint(0, width)
                    defect_y = np.random.randint(interface_y - 20, interface_y + 20)
                    defect_size = np.random.randint(10, 30)
                    defect = np.zeros((defect_size, defect_size))
                    rr, cc = np.ogrid[:defect_size, :defect_size]
                    mask = (rr - defect_size//2)**2 + (cc - defect_size//2)**2 <= (defect_size//2)**2
                    defect[mask] = 50  # ??????
                    
                    y_start = max(0, defect_y - defect_size//2)
                    y_end = min(height, defect_y + defect_size//2)
                    x_start = max(0, defect_x - defect_size//2)
                    x_end = min(width, defect_x + defect_size//2)
                    image[y_start:y_end, x_start:x_end] = np.minimum(
                        image[y_start:y_end, x_start:x_end],
                        defect[:y_end-y_start, :x_end-x_start]
                    )
            
            image = image + np.random.normal(0, 8, image.shape)
            image = np.clip(image, 0, 255).astype(np.uint8)
            image = gaussian_filter(image.astype(float), sigma=0.5).astype(np.uint8)
        
        metadata = {
            'layer_type': layer_type,
            'state': state,
            'resolution': resolution,
            'pixel_size_nm': pixel_size_nm,
            'timestamp': datetime.now().isoformat(),
            'generator': 'SOFC_Dataset_Generator'
        }
        
        return image, metadata
    
    def generate_muct_scan(self, state='initial', volume_size=(512, 512, 512), 
                           voxel_size_um=2.0):
        """
        ?????CT????
        
        ??:
            state: 'initial' ? 'sintered'
            volume_size: ???? (x, y, z)
            voxel_size_um: ????????
        
        ??:
            3D????????
        """
        nx, ny, nz = volume_size
        volume = np.zeros(volume_size, dtype=np.uint16)
        
        # ?????????
        layer_thicknesses = {
            'anode': 300,    # ??
            'electrolyte': 150,
            'cathode': 50,
        }
        
        # ?????
        layer_voxels = {k: int(v / voxel_size_um) for k, v in layer_thicknesses.items()}
        
        z_pos = 0
        
        # ???
        anode_thickness = layer_voxels['anode']
        anode_volume = np.ones((nx, ny, anode_thickness), dtype=np.uint16) * 3000  # ???
        
        # ????????
        if state == 'initial':
            # ????
            porosity = np.random.random((nx, ny, anode_thickness)) < 0.25
            anode_volume[porosity] = 0  # ??????
            
            # ????????
            num_particles = 5000
            particle_radius_voxels = 5  # 5????
            for _ in range(num_particles):
                px = np.random.randint(particle_radius_voxels, nx - particle_radius_voxels)
                py = np.random.randint(particle_radius_voxels, ny - particle_radius_voxels)
                pz = np.random.randint(z_pos + particle_radius_voxels, 
                                      z_pos + anode_thickness - particle_radius_voxels)
                
                # ??????
                # ??anode_volume????????mask
                xx, yy, zz_local = np.ogrid[:nx, :ny, :anode_thickness]
                zz_global = zz_local + z_pos  # ?????z??
                mask = ((xx - px)**2 + (yy - py)**2 + 
                       (zz_global - pz)**2) <= particle_radius_voxels**2
                
                # ??????????
                mask = mask & (zz_global >= z_pos) & (zz_global < z_pos + anode_thickness)
                anode_volume[mask] = 3500  # ??????
        
        volume[:, :, z_pos:z_pos+anode_thickness] = anode_volume
        z_pos += anode_thickness
        
        # ????
        electrolyte_thickness = layer_voxels['electrolyte']
        electrolyte_volume = np.ones((nx, ny, electrolyte_thickness), dtype=np.uint16) * 5000
        
        # ??????????????
        if state == 'initial':
            # ?????
            defects = np.random.random((nx, ny, electrolyte_thickness)) < 0.005
            electrolyte_volume[defects] = 4000  # ?????
        
        # ???????????
        if state == 'sintered':
            num_cracks = np.random.randint(3, 8)
            for _ in range(num_cracks):
                crack_plane = np.random.choice(['xy', 'xz', 'yz'])
                if crack_plane == 'xy':
                    z_crack_local = np.random.randint(0, electrolyte_thickness)  # ????
                    crack_length = np.random.randint(50, min(200, nx, ny))
                    x_start = np.random.randint(0, max(1, nx - crack_length))
                    y_start = np.random.randint(0, max(1, ny - crack_length))
                    x_end = min(x_start + crack_length, nx)
                    y_end = min(y_start + crack_length, ny)
                    if 0 <= z_crack_local < electrolyte_thickness:
                        electrolyte_volume[x_start:x_end, y_start:y_end, z_crack_local] = 0
                # ???????????
        
        volume[:, :, z_pos:z_pos+electrolyte_thickness] = electrolyte_volume
        z_pos += electrolyte_thickness
        
        # ???
        cathode_thickness = min(layer_voxels['cathode'], nz - z_pos)
        cathode_volume = np.ones((nx, ny, cathode_thickness), dtype=np.uint16) * 4500
        if state == 'initial':
            porosity = np.random.random((nx, ny, cathode_thickness)) < 0.30
            cathode_volume[porosity] = 0
        
        if z_pos + cathode_thickness <= nz:
            volume[:, :, z_pos:z_pos+cathode_thickness] = cathode_volume
        
        # ?????????????- ????
        if state == 'sintered':
            # ?????
            xx, yy = np.meshgrid(np.linspace(0, 2*np.pi, nx),
                               np.linspace(0, 2*np.pi, ny))
            warp_amplitude = 20  # ??
            z_warp = (np.sin(xx) * np.cos(yy) * warp_amplitude).astype(int)
            
            # ??????????
            # ????????????????????????
            # ??????????????????????
            warped_volume = volume.copy()
            # ??????????????????
            density_variation = np.random.normal(1.0, 0.05, volume.shape)
            warped_volume = (warped_volume.astype(float) * density_variation).astype(np.uint16)
            volume = warped_volume
        
        # ???????CT???
        volume = volume.astype(float)
        volume = volume + np.random.normal(0, 50, volume.shape)
        volume = np.clip(volume, 0, 65535).astype(np.uint16)
        
        metadata = {
            'state': state,
            'volume_size': volume_size,
            'voxel_size_um': voxel_size_um,
            'layer_thicknesses_um': layer_thicknesses,
            'timestamp': datetime.now().isoformat(),
            'generator': 'SOFC_Dataset_Generator'
        }
        
        return volume, metadata
    
    def generate_eds_map(self, elements=['Ni', 'Zr', 'Y', 'La', 'Sr', 'Mn']):
        """
        ??EDS?????
        
        ??:
            elements: ????
        
        ??:
            ???????????
        """
        resolution = (1024, 1024)
        maps = {}
        
        # ?????
        anode_thickness = 300  # ?????????
        electrolyte_thickness = 150
        cathode_thickness = 50
        
        # ??????
        height = anode_thickness + electrolyte_thickness + cathode_thickness
        
        for element in elements:
            map_array = np.zeros((height, resolution[1]), dtype=np.float32)
            
            # ?????????
            if element == 'Ni':
                # Ni?????
                map_array[0:anode_thickness, :] = np.random.uniform(0.3, 0.5, 
                    (anode_thickness, resolution[1]))
                # ????????
                diffusion = np.exp(-np.linspace(0, 3, electrolyte_thickness))
                map_array[anode_thickness:anode_thickness+electrolyte_thickness, :] = \
                    0.05 * diffusion[:, np.newaxis]
            
            elif element in ['Zr', 'Y']:
                # Zr?Y????????
                map_array[:, :] = 0.2  # ??
                # ???????
                map_array[anode_thickness:anode_thickness+electrolyte_thickness, :] = 0.6
            
            elif element in ['La', 'Sr', 'Mn']:
                # LSM???????
                cathode_start = anode_thickness + electrolyte_thickness
                map_array[cathode_start:cathode_start+cathode_thickness, :] = \
                    np.random.uniform(0.4, 0.6, (cathode_thickness, resolution[1]))
            
            # ???????????????
            map_array = gaussian_filter(map_array, sigma=5)
            map_array = map_array + np.random.normal(0, 0.02, map_array.shape)
            map_array = np.clip(map_array, 0, 1)
            
            maps[element] = map_array
        
        metadata = {
            'elements': elements,
            'resolution': resolution,
            'timestamp': datetime.now().isoformat()
        }
        
        return maps, metadata
    
    def generate_fem_geometry(self, cell_size_mm=(100, 100), mesh_density='medium'):
        """
        ??FEM???????
        
        ??:
            cell_size_mm: ???? (?, ?) ??
            mesh_density: 'coarse', 'medium', 'fine'
        
        ??:
            ?????????
        """
        mesh_densities = {
            'coarse': {'elements_per_mm': 2, 'layers_per_thickness': 3},
            'medium': {'elements_per_mm': 5, 'layers_per_thickness': 8},
            'fine': {'elements_per_mm': 10, 'layers_per_thickness': 16}
        }
        
        density = mesh_densities[mesh_density]
        
        # ???????
        layer_thicknesses = {
            'anode': 300,
            'electrolyte': 150,
            'cathode': 50,
        }
        
        # ????
        cell_length_mm, cell_width_mm = cell_size_mm
        
        geometry = {
            'cell_dimensions_mm': {
                'length': cell_length_mm,
                'width': cell_width_mm
            },
            'layer_thicknesses_um': layer_thicknesses,
            'mesh_parameters': {
                'density': mesh_density,
                'elements_per_mm': density['elements_per_mm'],
                'total_elements_x': int(cell_length_mm * density['elements_per_mm']),
                'total_elements_y': int(cell_width_mm * density['elements_per_mm']),
                'elements_per_layer': {
                    'anode': density['layers_per_thickness'],
                    'electrolyte': density['layers_per_thickness'],
                    'cathode': density['layers_per_thickness'] // 2
                }
            },
            'geometric_features': {
                'chamfer_radius_mm': 0.5,
                'via_holes': {
                    'present': True,
                    'diameter_mm': 1.0,
                    'locations': []  # ????????
                }
            }
        }
        
        # ??????
        total_elements = (geometry['mesh_parameters']['total_elements_x'] *
                         geometry['mesh_parameters']['total_elements_y'] *
                         sum(geometry['mesh_parameters']['elements_per_layer'].values()))
        
        geometry['mesh_parameters']['total_elements'] = total_elements
        
        return geometry
    
    def save_dataset(self):
        """???????"""
        print("????SOFC?????????...")
        
        # 1. ??????
        print("\n1. ???????????...")
        with open(self.dirs['material_props'] / 'material_properties.json', 'w') as f:
            json.dump(self.material_properties, f, indent=2)
        
        with open(self.dirs['constitutive'] / 'constitutive_models.json', 'w') as f:
            json.dump(self.constitutive_models, f, indent=2)
        
        # 2. ?????SEM??
        print("\n2. ??SEM??...")
        sem_types = [
            ('cross_section', 'initial'),
            ('cross_section', 'sintered'),
            ('interface', 'initial'),
            ('interface', 'sintered')
        ]
        
        for layer_type, state in sem_types:
            image, metadata = self.generate_sem_image(layer_type, state)
            filename = f"SEM_{layer_type}_{state}.png"
            plt.imsave(self.dirs['sem'] / filename, image, cmap='gray')
            
            # ?????
            with open(self.dirs['sem'] / f"{filename}.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # 3. ??????CT??
        print("\n3. ???CT????...")
        for state in ['initial', 'sintered']:
            volume, metadata = self.generate_muct_scan(state, volume_size=(256, 256, 256))
            
            # ???HDF5??
            filename = f"muCT_{state}.h5"
            with h5py.File(self.dirs['muct_initial' if state == 'initial' else 'muct_sintered'] / filename, 'w') as f:
                f.create_dataset('volume', data=volume, compression='gzip')
                # ???HDF5???????
                for key, value in metadata.items():
                    if isinstance(value, str):
                        f.attrs[key] = value
                    elif isinstance(value, (int, float)):
                        f.attrs[key] = value
                    elif isinstance(value, (list, tuple)):
                        f.attrs[key] = np.array(value)
                # ?????metadata?JSON???
                f.attrs['metadata_json'] = json.dumps(metadata)
            
            # ???????????
            slice_indices = [64, 128, 192]
            for i, slice_idx in enumerate(slice_indices):
                slice_img = volume[:, :, slice_idx]
                plt.imsave(
                    (self.dirs['muct_initial' if state == 'initial' else 'muct_sintered'] / 
                     f"muCT_{state}_slice_{i}.png"),
                    slice_img, cmap='gray'
                )
        
        # 4. ?????EDS??
        print("\n4. ??EDS?????...")
        eds_maps, eds_metadata = self.generate_eds_map()
        for element, map_array in eds_maps.items():
            plt.imsave(
                self.dirs['eds'] / f"EDS_{element}.png",
                map_array, cmap='viridis'
            )
        
        with open(self.dirs['eds'] / 'EDS_metadata.json', 'w') as f:
            json.dump(eds_metadata, f, indent=2)
        
        # 5. ?????FEM??
        print("\n5. ??FEM????...")
        for density in ['coarse', 'medium', 'fine']:
            geometry = self.generate_fem_geometry(mesh_density=density)
            filename = f"FEM_geometry_{density}.json"
            with open(self.dirs['fem_geometry'] / filename, 'w') as f:
                json.dump(geometry, f, indent=2)
        
        # 6. ?????????
        print("\n6. ???????...")
        dataset_index = {
            'dataset_name': 'SOFC_Microstructure_Informed_Dataset',
            'description': '??????????DIC-FEM????????',
            'generation_date': datetime.now().isoformat(),
            'research_topic': 'A Closed-Loop, Microstructure-Informed DIC-FEM Framework for the Real-Time Mitigation of Sintering Stresses in Solid Oxide Fuel Cells through Targeted Creep Activation',
            'contents': {
                'material_properties': str(self.dirs['material_props'].relative_to(self.output_dir)),
                'constitutive_models': str(self.dirs['constitutive'].relative_to(self.output_dir)),
                'sem_images': str(self.dirs['sem'].relative_to(self.output_dir)),
                'muct_initial': str(self.dirs['muct_initial'].relative_to(self.output_dir)),
                'muct_sintered': str(self.dirs['muct_sintered'].relative_to(self.output_dir)),
                'eds_maps': str(self.dirs['eds'].relative_to(self.output_dir)),
                'fem_geometry': str(self.dirs['fem_geometry'].relative_to(self.output_dir))
            },
            'file_counts': {
                'sem_images': len(list(self.dirs['sem'].glob('*.png'))),
                'muct_files': len(list(self.dirs['muct_initial'].glob('*.h5')) +
                                  list(self.dirs['muct_sintered'].glob('*.h5'))),
                'eds_maps': len(list(self.dirs['eds'].glob('*.png'))),
                'fem_geometries': len(list(self.dirs['fem_geometry'].glob('*.json')))
            }
        }
        
        with open(self.output_dir / 'dataset_index.json', 'w') as f:
            json.dump(dataset_index, f, indent=2)
        
        print(f"\n? ????????")
        print(f"????: {self.output_dir.absolute()}")
        print(f"\n?????:")
        print(f"  - SEM??: {dataset_index['file_counts']['sem_images']} ?")
        print(f"  - ?CT??: {dataset_index['file_counts']['muct_files']} ?")
        print(f"  - EDS??: {dataset_index['file_counts']['eds_maps']} ?")
        print(f"  - FEM??: {dataset_index['file_counts']['fem_geometries']} ?")
        
        return dataset_index


def main():
    """???"""
    generator = SOFCDatasetGenerator(output_dir="./sofc_dataset")
    dataset_index = generator.save_dataset()
    
    print("\n" + "="*60)
    print("???????????????? sofc_dataset/ ??")
    print("="*60)


if __name__ == "__main__":
    main()
