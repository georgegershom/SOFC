#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
???????????????
??????????? (SOFC) ?FEM??

?????????
1. ????????????????????????
2. ??????????Norton????
3. ????????????????????????
"""

import numpy as np
import pandas as pd
import json
import h5py
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ????????????
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SOFCMaterialDatasetGenerator:
    """SOFC????????"""
    
    def __init__(self):
        self.R = 8.314  # ???? (J/mol?K)
        self.materials = {
            'anode': 'Ni-YSZ',
            'electrolyte': '8YSZ',
            'cathode': 'LSM-YSZ',
            'interconnect': 'Crofer22APU'
        }
        
    def generate_green_state_properties(self):
        """??????????"""
        print("??????????...")
        
        data = {}
        
        for layer, material in self.materials.items():
            if layer == 'interconnect':
                continue  # ???????????
                
            # ???? (kg/m?) - ?????
            green_density = {
                'anode': 2800,
                'electrolyte': 3200,
                'cathode': 2700
            }[layer]
            
            # ?????? (kg/m?)
            sintered_density = {
                'anode': 6100,
                'electrolyte': 5900,
                'cathode': 5800
            }[layer]
            
            # ?????
            initial_porosity = 1 - green_density / sintered_density
            
            # ????? (wt%)
            binder_content = {
                'anode': 8.5,
                'electrolyte': 6.0,
                'cathode': 7.5
            }[layer]
            
            # ????? (wt%)
            porogen_content = {
                'anode': 12.0,
                'electrolyte': 5.0,
                'cathode': 15.0
            }[layer]
            
            data[layer] = {
                'material': material,
                'green_density_kg_m3': green_density,
                'sintered_density_kg_m3': sintered_density,
                'initial_porosity': initial_porosity,
                'relative_density': green_density / sintered_density,
                'binder_content_wt_percent': binder_content,
                'porogen_content_wt_percent': porogen_content,
                'solid_fraction': 1 - initial_porosity
            }
        
        return data
    
    def generate_sintering_kinetics(self, T_range=None, t_range=None):
        """?????????????????
        
        ??:
            T_range: ???? (K)??? [300, 1673]
            t_range: ???? (?)??? [0, 7200]
        """
        print("?????????...")
        
        if T_range is None:
            T_range = np.array([300, 1673])  # ???1400?C
        
        if t_range is None:
            t_range = np.array([0, 7200])  # 2??
        
        # ??????????
        T = np.linspace(T_range[0], T_range[1], 200)
        t = np.linspace(t_range[0], t_range[1], 100)
        
        kinetics_data = {}
        
        for layer, material in self.materials.items():
            if layer == 'interconnect':
                continue
            
            # ????????????SOFC???
            sintering_params = {
                'anode': {
                    'T_sinter_start': 800,  # ?????? (?C)
                    'T_sinter_peak': 1300,   # ?????? (?C)
                    'T_sinter_end': 1400,    # ?????? (?C)
                    'max_shrinkage': 0.45,   # ????? (dL/L0)
                    'activation_energy': 350, # ??? (kJ/mol)
                },
                'electrolyte': {
                    'T_sinter_start': 1000,
                    'T_sinter_peak': 1400,
                    'T_sinter_end': 1500,
                    'max_shrinkage': 0.38,
                    'activation_energy': 450,
                },
                'cathode': {
                    'T_sinter_start': 850,
                    'T_sinter_peak': 1200,
                    'T_sinter_end': 1300,
                    'max_shrinkage': 0.42,
                    'activation_energy': 380,
                }
            }
            
            params = sintering_params[layer]
            
            # ?????????????
            shrinkage_time = []
            for time in t:
                # ????????
                T_K = params['T_sinter_peak'] + 273.15
                rate = np.exp(-params['activation_energy'] * 1000 / (self.R * T_K))
                shrinkage = params['max_shrinkage'] * (1 - np.exp(-rate * time / 3600))
                shrinkage_time.append(shrinkage)
            
            # ??????????
            shrinkage_temp = []
            for temp in T:
                T_C = temp - 273.15
                if T_C < params['T_sinter_start']:
                    shrinkage = 0
                elif T_C < params['T_sinter_peak']:
                    # ????
                    fraction = (T_C - params['T_sinter_start']) / (params['T_sinter_peak'] - params['T_sinter_start'])
                    shrinkage = params['max_shrinkage'] * 0.5 * (1 - np.cos(np.pi * fraction))
                elif T_C < params['T_sinter_end']:
                    # ????
                    fraction = (T_C - params['T_sinter_peak']) / (params['T_sinter_end'] - params['T_sinter_peak'])
                    shrinkage = params['max_shrinkage'] * (1 - 0.1 * fraction)
                else:
                    shrinkage = params['max_shrinkage'] * 0.9
                
                shrinkage_temp.append(shrinkage)
            
            kinetics_data[layer] = {
                'material': material,
                'temperature_K': T.tolist(),
                'shrinkage_vs_temperature': shrinkage_temp,
                'time_s': t.tolist(),
                'shrinkage_vs_time': shrinkage_time,
                'sintering_params': params
            }
            
            # ??????????????-????
            if layer == 'anode':
                # ??-???????
                bilayer_shrinkage = []
                anode_shrink = np.array(shrinkage_temp)
                # ????????????????????
                # ?????????????
                bilayer_shrinkage = (0.6 * anode_shrink + 0.4 * np.array(shrinkage_temp)).tolist()
                
                kinetics_data['anode_electrolyte_bilayer'] = {
                    'material_pair': 'Ni-YSZ / 8YSZ',
                    'temperature_K': T.tolist(),
                    'shrinkage_vs_temperature': bilayer_shrinkage,
                    'stress_mismatch_MPa': (np.array(shrinkage_temp) - anode_shrink).tolist()
                }
        
        return kinetics_data
    
    def generate_cte_data(self, T_range=None):
        """??????? (CTE) ??"""
        print("?????????...")
        
        if T_range is None:
            T_range = np.linspace(300, 1673, 100)  # ???1400?C
        
        cte_data = {}
        
        for layer, material in self.materials.items():
            # CTE?????????
            cte_params = {
                'anode': {
                    'CTE_25C': 12.5e-6,  # K??
                    'CTE_800C': 13.3e-6,
                    'temperature_dependence': 0.0008e-6  # ????
                },
                'electrolyte': {
                    'CTE_25C': 10.0e-6,
                    'CTE_800C': 10.5e-6,
                    'temperature_dependence': 0.0006e-6
                },
                'cathode': {
                    'CTE_25C': 11.5e-6,
                    'CTE_800C': 12.0e-6,
                    'temperature_dependence': 0.0006e-6
                },
                'interconnect': {
                    'CTE_25C': 11.5e-6,
                    'CTE_800C': 11.9e-6,
                    'temperature_dependence': 0.0005e-6
                }
            }
            
            params = cte_params[layer]
            
            # ??CTE????????????
            CTE = []
            thermal_strain = []
            ref_temp = 298.15  # ???? (K)
            
            for T in T_range:
                T_C = T - 273.15
                # ????
                cte_value = params['CTE_25C'] + params['temperature_dependence'] * T_C
                CTE.append(cte_value)
                
                # ??????????????
                strain = cte_value * (T - ref_temp)
                thermal_strain.append(strain)
            
            cte_data[layer] = {
                'material': material,
                'temperature_K': T_range.tolist(),
                'CTE_K_inv': CTE,
                'thermal_strain': thermal_strain,
                'CTE_25C': params['CTE_25C'],
                'CTE_800C': params['CTE_800C']
            }
        
        return cte_data
    
    def generate_creep_constitutive_data(self, T_list=None, stress_range=None):
        """???????????
        
        ??Norton????_dot = A * ?^n * exp(-Q/RT)
        
        ??:
            T_list: ???? (K)??? [1073, 1273, 1473]
            stress_range: ???? (MPa)??? [1, 100]
        """
        print("????????...")
        
        if T_list is None:
            T_list = np.array([1073, 1273, 1473])  # 800?C, 1000?C, 1200?C
        
        if stress_range is None:
            stress_range = np.logspace(0, 2, 20)  # 1-100 MPa?????
        
        creep_data = {}
        
        # Norton??????????????
        creep_params = {
            'anode': {
                'A': 8.5e-12,  # ????? (s?? MPa??)
                'n': 1.8,      # ????
                'Q': 385000,   # ??? (J/mol)
                'state': 'green'  # ????
            },
            'anode_sintered': {
                'A': 2.1e-15,
                'n': 2.2,
                'Q': 420000,
                'state': 'sintered'
            },
            'electrolyte': {
                'A': 3.5e-15,
                'n': 1.5,
                'Q': 450000,
                'state': 'green'
            },
            'electrolyte_sintered': {
                'A': 8.5e-18,
                'n': 1.8,
                'Q': 480000,
                'state': 'sintered'
            },
            'cathode': {
                'A': 6.2e-13,
                'n': 1.9,
                'Q': 380000,
                'state': 'green'
            },
            'cathode_sintered': {
                'A': 1.5e-16,
                'n': 2.1,
                'Q': 400000,
                'state': 'sintered'
            }
        }
        
        for key, params in creep_params.items():
            # ?????
            layer = key.split('_')[0]
            state = params['state']
            
            A = params['A']
            n = params['n']
            Q = params['Q']
            
            # ?????????
            creep_strain_rate = []
            creep_data_points = []
            
            for T in T_list:
                for sigma in stress_range:
                    # Norton??
                    epsilon_dot = A * (sigma ** n) * np.exp(-Q / (self.R * T))
                    
                    creep_strain_rate.append(epsilon_dot)
                    creep_data_points.append({
                        'temperature_K': T,
                        'temperature_C': T - 273.15,
                        'stress_MPa': sigma,
                        'strain_rate_s_inv': epsilon_dot
                    })
            
            # ??????
            if layer not in creep_data:
                creep_data[layer] = {}
            
            creep_data[layer][state] = {
                'material': self.materials.get(layer, layer),
                'state': state,
                'creep_parameters': {
                    'A_pre_exponential': A,
                    'n_stress_exponent': n,
                    'Q_activation_energy_J_mol': Q,
                    'Q_activation_energy_kJ_mol': Q / 1000
                },
                'creep_data': creep_data_points,
                'temperature_range_K': T_list.tolist(),
                'stress_range_MPa': stress_range.tolist()
            }
            
            # ??????????
            # ?????????log(?_dot) vs log(?)?????n
            T_fit = T_list[1]  # ??????
            log_sigma = np.log10(stress_range)
            log_eps_dot = np.log10([A * (s ** n) * np.exp(-Q / (self.R * T_fit)) for s in stress_range])
            
            creep_data[layer][state]['fit_validation'] = {
                'temperature_K': T_fit,
                'log_stress': log_sigma.tolist(),
                'log_strain_rate': log_eps_dot.tolist(),
                'fitted_n': n,  # ???????????
                'theoretical_n': n
            }
        
        return creep_data
    
    def generate_elastic_properties(self, T_range=None, density_range=None):
        """??????????????????
        
        ??:
            T_range: ???? (K)
            density_range: ?????? (0-1)
        """
        print("????????...")
        
        if T_range is None:
            T_range = np.linspace(300, 1673, 50)
        
        if density_range is None:
            density_range = np.linspace(0.45, 0.98, 30)  # ????????????
        
        elastic_data = {}
        
        for layer, material in self.materials.items():
            if layer == 'interconnect':
                # ??????????????
                continue
            
            # ??????
            elastic_params = {
                'anode': {
                    'E_25C_GPa': 55,
                    'E_800C_GPa': 29,
                    'nu': 0.29,
                    'E_full_density_GPa': 180,  # ??????????
                    'density_exponent': 2.5  # ??????
                },
                'electrolyte': {
                    'E_25C_GPa': 200,
                    'E_800C_GPa': 170,
                    'nu': 0.23,
                    'E_full_density_GPa': 210,
                    'density_exponent': 2.8
                },
                'cathode': {
                    'E_25C_GPa': 45,
                    'E_800C_GPa': 40,
                    'nu': 0.25,
                    'E_full_density_GPa': 150,
                    'density_exponent': 2.6
                }
            }
            
            params = elastic_params[layer]
            
            # ?????????
            E_vs_T = []
            for T in T_range:
                T_C = T - 273.15
                # ????
                E = params['E_25C_GPa'] - (params['E_25C_GPa'] - params['E_800C_GPa']) * (T_C / 775)
                E_vs_T.append(max(E, params['E_800C_GPa'] * 0.5))  # ?????
            
            # ?????????????????
            E_vs_rho = []
            for rho in density_range:
                # Gibson-Ashby???E/E0 = (?/?0)^n
                E = params['E_full_density_GPa'] * (rho ** params['density_exponent'])
                E_vs_rho.append(E)
            
            # ?????????????????????
            nu_vs_T = [params['nu'] * (1 - 0.05 * (T - 300) / 1373) for T in T_range]
            nu_vs_rho = [params['nu'] * (0.9 + 0.1 * rho) for rho in density_range]
            
            elastic_data[layer] = {
                'material': material,
                'temperature_K': T_range.tolist(),
                'youngs_modulus_vs_temperature_GPa': E_vs_T,
                'poissons_ratio_vs_temperature': nu_vs_T,
                'relative_density': density_range.tolist(),
                'youngs_modulus_vs_density_GPa': E_vs_rho,
                'poissons_ratio_vs_density': nu_vs_rho,
                'elastic_params': {
                    'E_25C_GPa': params['E_25C_GPa'],
                    'E_800C_GPa': params['E_800C_GPa'],
                    'nu_constant': params['nu'],
                    'E_full_density_GPa': params['E_full_density_GPa']
                }
            }
        
        return elastic_data
    
    def generate_complete_dataset(self):
        """????????"""
        print("=" * 60)
        print("????SOFC????????????")
        print("=" * 60)
        
        dataset = {
            'metadata': {
                'title': 'SOFC Material Property & Constitutive Model Dataset',
                'description': '??FEM???????????????',
                'research_topic': 'A Closed-Loop, Microstructure-Informed DIC-FEM Framework for the Real-Time Mitigation of Sintering Stresses in Solid Oxide Fuel Cells through Targeted Creep Activation',
                'materials': self.materials,
                'generation_date': pd.Timestamp.now().isoformat()
            },
            'green_state_properties': self.generate_green_state_properties(),
            'sintering_kinetics': self.generate_sintering_kinetics(),
            'cte_data': self.generate_cte_data(),
            'creep_constitutive_data': self.generate_creep_constitutive_data(),
            'elastic_properties': self.generate_elastic_properties()
        }
        
        print("\n????????")
        return dataset
    
    def save_dataset(self, dataset, output_dir='./material_dataset'):
        """??????????"""
        print(f"\n????????: {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. ???JSON??
        json_path = output_path / 'material_dataset.json'
        print(f"??JSON??: {json_path}")
        
        # ??numpy???????JSON???
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        dataset_serializable = convert_to_serializable(dataset)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_serializable, f, indent=2, ensure_ascii=False)
        
        # 2. ???HDF5??????????
        hdf5_path = output_path / 'material_dataset.h5'
        print(f"??HDF5??: {hdf5_path}")
        
        with h5py.File(hdf5_path, 'w') as f:
            # ?????
            meta_group = f.create_group('metadata')
            for key, value in dataset['metadata'].items():
                if isinstance(value, str):
                    meta_group.attrs[key] = value
                elif isinstance(value, dict):
                    for k, v in value.items():
                        meta_group.attrs[f'{key}_{k}'] = v
            
            # ???????
            for category, data in dataset.items():
                if category == 'metadata':
                    continue
                
                category_group = f.create_group(category)
                
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, dict):
                            item_group = category_group.create_group(key)
                            self._save_dict_to_hdf5(value, item_group)
                        else:
                            category_group.create_dataset(key, data=np.array(value))
        
        # 3. ???CSV?????????
        csv_dir = output_path / 'csv_files'
        csv_dir.mkdir(exist_ok=True)
        print(f"??CSV?????: {csv_dir}")
        
        # ??????
        green_df = pd.DataFrame(dataset['green_state_properties']).T
        green_df.to_csv(csv_dir / 'green_state_properties.csv')
        
        # ?????
        for layer, data in dataset['sintering_kinetics'].items():
            if 'temperature_K' in data:
                df = pd.DataFrame({
                    'temperature_K': data['temperature_K'],
                    'shrinkage': data['shrinkage_vs_temperature']
                })
                df.to_csv(csv_dir / f'sintering_kinetics_{layer}.csv', index=False)
        
        # CTE??
        for layer, data in dataset['cte_data'].items():
            df = pd.DataFrame({
                'temperature_K': data['temperature_K'],
                'CTE_K_inv': data['CTE_K_inv'],
                'thermal_strain': data['thermal_strain']
            })
            df.to_csv(csv_dir / f'cte_{layer}.csv', index=False)
        
        # ????
        for layer, layer_data in dataset['creep_constitutive_data'].items():
            for state, state_data in layer_data.items():
                creep_df = pd.DataFrame(state_data['creep_data'])
                creep_df.to_csv(csv_dir / f'creep_{layer}_{state}.csv', index=False)
        
        # ????
        for layer, data in dataset['elastic_properties'].items():
            # ????
            df_T = pd.DataFrame({
                'temperature_K': data['temperature_K'],
                'youngs_modulus_GPa': data['youngs_modulus_vs_temperature_GPa'],
                'poissons_ratio': data['poissons_ratio_vs_temperature']
            })
            df_T.to_csv(csv_dir / f'elastic_properties_T_{layer}.csv', index=False)
            
            # ????
            df_rho = pd.DataFrame({
                'relative_density': data['relative_density'],
                'youngs_modulus_GPa': data['youngs_modulus_vs_density_GPa'],
                'poissons_ratio': data['poissons_ratio_vs_density']
            })
            df_rho.to_csv(csv_dir / f'elastic_properties_rho_{layer}.csv', index=False)
        
        print(f"\n????????: {output_path.absolute()}")
        return output_path
    
    def _save_dict_to_hdf5(self, data_dict, group):
        """???????HDF5?"""
        for key, value in data_dict.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._save_dict_to_hdf5(value, subgroup)
            elif isinstance(value, list):
                # ?????numpy??
                try:
                    group.create_dataset(key, data=np.array(value))
                except:
                    # ???????????
                    group.attrs[key] = str(value)
            elif isinstance(value, (str, int, float)):
                group.attrs[key] = value
            else:
                try:
                    group.create_dataset(key, data=np.array(value))
                except:
                    group.attrs[key] = str(value)


def main():
    """???"""
    generator = SOFCMaterialDatasetGenerator()
    
    # ???????
    dataset = generator.generate_complete_dataset()
    
    # ?????
    output_dir = generator.save_dataset(dataset)
    
    print("\n" + "=" * 60)
    print("???????????")
    print("=" * 60)
    print(f"\n????: {output_dir.absolute()}")
    print("\n???????:")
    print("  - material_dataset.json (??JSON??)")
    print("  - material_dataset.h5 (HDF5????????)")
    print("  - csv_files/ (CSV???????Excel?????)")
    
    return dataset


if __name__ == '__main__':
    dataset = main()
