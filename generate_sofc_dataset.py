#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOFC???????????????
???A Closed-Loop, Microstructure-Informed DIC-FEM Framework for the Real-Time Mitigation 
of Sintering Stresses in Solid Oxide Fuel Cells through Targeted Creep Activation

?????????
1. ???????????
2. ???????
3. ?????(CTE)
4. ?????????
5. ????????????????
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

class SOFCDatasetGenerator:
    """SOFC??????????"""
    
    def __init__(self):
        """??????"""
        self.materials = {
            'anode': 'Ni-YSZ',
            'electrolyte': '8YSZ',
            'cathode': 'LSM-YSZ',
            'interconnect': 'Crofer22APU'
        }
        self.R = 8.314  # ???? (J/mol/K)
        
    def generate_thermophysical_properties(self):
        """???????????????"""
        print("?????????...")
        
        data = {}
        
        # ?? (Ni-YSZ)
        data['anode'] = {
            'material': 'Ni-YSZ',
            'green_state': {
                'density_green': 3800.0,  # kg/m?
                'density_sintered': 5900.0,  # kg/m?
                'porosity_initial': 0.55,  # ?????
                'porosity_final': 0.30,  # ?????
                'binder_content': 0.08,  # ????? (????)
                'porogen_content': 0.02,  # ????? (????)
                'volatile_content': 0.10,  # ????? (????)
            },
            'particle_size': {
                'Ni_powder_mean': 1.5e-6,  # m
                'YSZ_powder_mean': 0.5e-6,  # m
                'powder_distribution': 'log_normal'
            }
        }
        
        # ??? (8YSZ)
        data['electrolyte'] = {
            'material': '8YSZ',
            'green_state': {
                'density_green': 2800.0,  # kg/m?
                'density_sintered': 5900.0,  # kg/m?
                'porosity_initial': 0.52,  # ?????
                'porosity_final': 0.03,  # ?????????
                'binder_content': 0.06,  # ?????
                'porogen_content': 0.00,  # ???????????
                'volatile_content': 0.08,  # ?????
            },
            'particle_size': {
                'YSZ_powder_mean': 0.3e-6,  # m????
                'powder_distribution': 'log_normal'
            }
        }
        
        # ?? (LSM-YSZ)
        data['cathode'] = {
            'material': 'LSM-YSZ',
            'green_state': {
                'density_green': 3200.0,  # kg/m?
                'density_sintered': 5800.0,  # kg/m?
                'porosity_initial': 0.58,  # ???????????
                'porosity_final': 0.35,  # ?????????????
                'binder_content': 0.07,  # ?????
                'porogen_content': 0.05,  # ?????
                'volatile_content': 0.09,  # ?????
            },
            'particle_size': {
                'LSM_powder_mean': 0.8e-6,  # m
                'YSZ_powder_mean': 0.4e-6,  # m
                'powder_distribution': 'log_normal'
            }
        }
        
        return data
    
    def generate_sintering_kinetics(self):
        """?????????????????"""
        print("?????????...")
        
        # ????????1400?C
        temperatures = np.linspace(25, 1400, 276)  # 5?C??
        
        data = {}
        
        # ???????
        for layer_name in ['anode', 'electrolyte', 'cathode']:
            if layer_name == 'anode':
                # Ni-YSZ????1100-1300?C?????
                T_start = 1100
                T_final = 1300
                max_shrinkage = -0.18  # 18%??
            elif layer_name == 'electrolyte':
                # 8YSZ?????1200-1400?C?????
                T_start = 1200
                T_final = 1400
                max_shrinkage = -0.22  # 22%??
            elif layer_name == 'cathode':
                # LSM-YSZ????1100-1250?C?????
                T_start = 1100
                T_final = 1250
                max_shrinkage = -0.15  # 15%??
            
            # ?????? (dL/L0 vs T)
            shrinkage = np.zeros_like(temperatures)
            for i, T in enumerate(temperatures):
                if T < T_start:
                    shrinkage[i] = 0.0
                elif T >= T_start and T <= T_final:
                    # S???
                    x = (T - T_start) / (T_final - T_start)
                    shrinkage[i] = max_shrinkage * (x**3 * (10 - 15*x + 6*x**2))
                else:
                    shrinkage[i] = max_shrinkage
            
            # ???????????????
            time_points = np.logspace(0, 4, 100)  # 1??10000??????
            isothermal_T = [1000, 1100, 1200, 1300]  # ?????
            
            time_data = {}
            for T_iso in isothermal_T:
                if T_iso >= T_start:
                    # ????????dL/L0 = -A * (t/tau)^n
                    if T_iso <= T_final:
                        tau = 3600 * (1 + (T_final - T_iso) / 100)  # ????
                        A = max_shrinkage * (T_iso - T_start) / (T_final - T_start)
                        n = 0.4  # ????
                    else:
                        tau = 1800
                        A = max_shrinkage
                        n = 0.3
                    
                    shrinkage_t = -A * (time_points / tau)**n
                    time_data[f'T_{T_iso:.0f}C'] = {
                        'time_seconds': time_points.tolist(),
                        'shrinkage_dL_L0': shrinkage_t.tolist(),
                        'temperature_C': T_iso
                    }
            
            data[f'{layer_name}_single'] = {
                'temperature_C': temperatures.tolist(),
                'shrinkage_dL_L0': shrinkage.tolist(),
                'isothermal_kinetics': time_data
            }
        
        # ?????????
        bilayer_pairs = [
            ('anode', 'electrolyte'),
            ('electrolyte', 'cathode')
        ]
        
        for layer1, layer2 in bilayer_pairs:
            pair_name = f'{layer1}_{layer2}'
            
            # ?????????????
            T1 = np.array(data[f'{layer1}_single']['temperature_C'])
            shrink1 = np.array(data[f'{layer1}_single']['shrinkage_dL_L0'])
            
            T2 = np.array(data[f'{layer2}_single']['temperature_C'])
            shrink2 = np.array(data[f'{layer2}_single']['shrinkage_dL_L0'])
            
            # ????????
            T_common = np.linspace(25, 1400, 276)
            shrink1_interp = np.interp(T_common, T1, shrink1)
            shrink2_interp = np.interp(T_common, T2, shrink2)
            
            # ???????????????
            shrink_bilayer = 0.6 * shrink1_interp + 0.4 * shrink2_interp  # ????
            
            # ?????????????
            mismatch_strain = shrink1_interp - shrink2_interp
            
            data[f'{pair_name}_bilayer'] = {
                'temperature_C': T_common.tolist(),
                'shrinkage_dL_L0': shrink_bilayer.tolist(),
                'mismatch_strain': mismatch_strain.tolist(),
                'layer1_shrinkage': shrink1_interp.tolist(),
                'layer2_shrinkage': shrink2_interp.tolist()
            }
        
        return data
    
    def generate_cte_data(self):
        """???????(CTE)??"""
        print("?????????...")
        
        temperatures = np.linspace(25, 1400, 276)  # ??????
        
        data = {}
        
        # ?? (Ni-YSZ)
        # CTE????????12.5-13.3 ? 10^-6 K^-1
        cte_anode = 12.5e-6 + (temperatures - 25) / (1400 - 25) * 0.8e-6
        
        # ??? (8YSZ)
        # CTE????????10.0-10.5 ? 10^-6 K^-1
        cte_electrolyte = 10.0e-6 + (temperatures - 25) / (1400 - 25) * 0.5e-6
        
        # ?? (LSM-YSZ)
        # CTE????????11.5-12.0 ? 10^-6 K^-1
        cte_cathode = 11.5e-6 + (temperatures - 25) / (1400 - 25) * 0.5e-6
        
        # ??? (Crofer 22 APU)
        # CTE????????11.5-11.9 ? 10^-6 K^-1
        cte_interconnect = 11.5e-6 + (temperatures - 25) / (1400 - 25) * 0.4e-6
        
        # ??????????????
        thermal_strain_anode = np.cumsum(cte_anode) * (temperatures[1] - temperatures[0])
        thermal_strain_electrolyte = np.cumsum(cte_electrolyte) * (temperatures[1] - temperatures[0])
        thermal_strain_cathode = np.cumsum(cte_cathode) * (temperatures[1] - temperatures[0])
        thermal_strain_interconnect = np.cumsum(cte_interconnect) * (temperatures[1] - temperatures[0])
        
        data['anode'] = {
            'temperature_C': temperatures.tolist(),
            'cte_per_K': cte_anode.tolist(),
            'thermal_strain': thermal_strain_anode.tolist()
        }
        
        data['electrolyte'] = {
            'temperature_C': temperatures.tolist(),
            'cte_per_K': cte_electrolyte.tolist(),
            'thermal_strain': thermal_strain_electrolyte.tolist()
        }
        
        data['cathode'] = {
            'temperature_C': temperatures.tolist(),
            'cte_per_K': cte_cathode.tolist(),
            'thermal_strain': thermal_strain_cathode.tolist()
        }
        
        data['interconnect'] = {
            'temperature_C': temperatures.tolist(),
            'cte_per_K': cte_interconnect.tolist(),
            'thermal_strain': thermal_strain_interconnect.tolist()
        }
        
        return data
    
    def generate_creep_constitutive_data(self):
        """???????????"""
        print("????????...")
        
        # ??????
        test_temperatures = [800, 1000, 1200]  # ?C
        test_stresses = np.logspace(5, 7, 20)  # 0.1-10 MPa (Pa)
        test_times = np.logspace(0, 4, 100)  # 1??10000?
        
        data = {}
        
        # ?????????Norton??: ?_dot = A * ?^n * exp(-Q/RT)?
        creep_params = {
            'anode': {
                'A': 8.5e-12,  # s^-1 MPa^-n
                'n': 1.8,  # ????
                'Q': 385000,  # ??? (J/mol)
                'state': ['green', 'sintering']  # ???????
            },
            'electrolyte': {
                'A': 1.2e-15,  # s^-1 MPa^-n????????
                'n': 2.2,  # ????
                'Q': 420000,  # ??? (J/mol)
                'state': ['green', 'sintering']
            },
            'cathode': {
                'A': 2.5e-11,  # s^-1 MPa^-n
                'n': 1.6,  # ????
                'Q': 360000,  # ??? (J/mol)
                'state': ['green', 'sintering']
            }
        }
        
        for material_name, params in creep_params.items():
            material_data = {
                'creep_parameters': {
                    'A': params['A'],
                    'n': params['n'],
                    'Q': params['Q'],
                    'units': {
                        'A': 's^-1 MPa^-n',
                        'n': 'dimensionless',
                        'Q': 'J/mol'
                    }
                },
                'test_data': {}
            }
            
            # ????????
            for state in params['state']:
                state_data = {}
                
                # ????????????????????
                if state == 'green':
                    A_adj = params['A'] * 10  # ?????????
                    Q_adj = params['Q'] * 0.9  # ?????
                else:
                    A_adj = params['A']
                    Q_adj = params['Q']
                
                for T_C in test_temperatures:
                    T_K = T_C + 273.15
                    
                    # ?????????????
                    creep_rates = []
                    for sigma in test_stresses:
                        sigma_MPa = sigma / 1e6
                        # Norton??
                        creep_rate = A_adj * (sigma_MPa ** params['n']) * np.exp(-Q_adj / (self.R * T_K))
                        creep_rates.append(creep_rate)
                    
                    # ??????????????
                    creep_strains = []
                    for sigma in test_stresses:
                        sigma_MPa = sigma / 1e6
                        creep_rate = A_adj * (sigma_MPa ** params['n']) * np.exp(-Q_adj / (self.R * T_K))
                        # ??????
                        creep_strain = creep_rate * test_times
                        creep_strains.append(creep_strain.tolist())
                    
                    state_data[f'T_{T_C:.0f}C'] = {
                        'temperature_C': T_C,
                        'temperature_K': T_K,
                        'stress_Pa': test_stresses.tolist(),
                        'stress_MPa': (test_stresses / 1e6).tolist(),
                        'creep_rate_per_s': creep_rates,
                        'time_seconds': test_times.tolist(),
                        'creep_strain_at_stresses': creep_strains
                    }
                
                material_data['test_data'][state] = state_data
            
            data[material_name] = material_data
        
        return data
    
    def generate_elastic_properties(self):
        """????????????????????"""
        print("????????...")
        
        temperatures = np.linspace(25, 1400, 276)
        relative_densities = np.linspace(0.45, 0.99, 55)  # ????????????
        
        data = {}
        
        # ??????????
        elastic_params = {
            'anode': {
                'E0': 55e9,  # ????????? (Pa)
                'E_T_coeff': -0.03e9,  # ???? (Pa/K)
                'nu': 0.29,  # ???
                'density_dependence': 'exponential'  # ??????
            },
            'electrolyte': {
                'E0': 200e9,  # ????????? (Pa)
                'E_T_coeff': -0.0375e9,  # ???? (Pa/K)
                'nu': 0.23,  # ???
                'density_dependence': 'power_law'
            },
            'cathode': {
                'E0': 45e9,  # ????????? (Pa)
                'E_T_coeff': -0.008e9,  # ???? (Pa/K)
                'nu': 0.25,  # ???
                'density_dependence': 'exponential'
            }
        }
        
        for material_name, params in elastic_params.items():
            material_data = {
                'temperature_dependent': {},
                'density_dependent': {}
            }
            
            # ??????????????
            E_T = params['E0'] + params['E_T_coeff'] * (temperatures - 25)
            E_T = np.maximum(E_T, 0.1 * params['E0'])  # ?????
            
            material_data['temperature_dependent'] = {
                'temperature_C': temperatures.tolist(),
                'youngs_modulus_Pa': E_T.tolist(),
                'youngs_modulus_GPa': (E_T / 1e9).tolist(),
                'poissons_ratio': [params['nu']] * len(temperatures)
            }
            
            # ???????????????
            if params['density_dependence'] == 'exponential':
                # E/E0 = (rho/rho0)^m?m???2-3
                E_rho = params['E0'] * (relative_densities ** 2.5)
            elif params['density_dependence'] == 'power_law':
                # ????
                E_rho = params['E0'] * (relative_densities ** 3.0)
            
            # ???????????????
            nu_rho = params['nu'] * (0.7 + 0.3 * relative_densities)
            
            material_data['density_dependent'] = {
                'relative_density': relative_densities.tolist(),
                'porosity': (1 - relative_densities).tolist(),
                'youngs_modulus_Pa': E_rho.tolist(),
                'youngs_modulus_GPa': (E_rho / 1e9).tolist(),
                'poissons_ratio': nu_rho.tolist(),
                'temperature_C': 25.0  # ??
            }
            
            # ?????????????2D????
            E_combined = np.zeros((len(temperatures), len(relative_densities)))
            for i, T in enumerate(temperatures):
                E_T_val = params['E0'] + params['E_T_coeff'] * (T - 25)
                E_T_val = max(E_T_val, 0.1 * params['E0'])
                
                if params['density_dependence'] == 'exponential':
                    E_rho_vals = E_T_val * (relative_densities ** 2.5)
                else:
                    E_rho_vals = E_T_val * (relative_densities ** 3.0)
                
                E_combined[i, :] = E_rho_vals
            
            material_data['combined_temperature_density'] = {
                'temperature_C': temperatures.tolist(),
                'relative_density': relative_densities.tolist(),
                'youngs_modulus_Pa': E_combined.tolist(),
                'youngs_modulus_GPa': (E_combined / 1e9).tolist()
            }
            
            data[material_name] = material_data
        
        return data
    
    def save_datasets(self, output_dir='sofc_dataset'):
        """???????"""
        print(f"\n????????: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # ??????
        thermophysical = self.generate_thermophysical_properties()
        sintering = self.generate_sintering_kinetics()
        cte = self.generate_cte_data()
        creep = self.generate_creep_constitutive_data()
        elastic = self.generate_elastic_properties()
        
        # ??????
        complete_dataset = {
            'metadata': {
                'generation_date': datetime.now().isoformat(),
                'description': 'SOFC Material Property & Constitutive Model Dataset',
                'application': 'A Closed-Loop, Microstructure-Informed DIC-FEM Framework for the Real-Time Mitigation of Sintering Stresses in Solid Oxide Fuel Cells through Targeted Creep Activation',
                'materials': self.materials,
                'units': {
                    'temperature': 'Celsius',
                    'stress': 'Pa',
                    'strain': 'dimensionless',
                    'time': 'seconds',
                    'density': 'kg/m?'
                }
            },
            'thermophysical_properties': thermophysical,
            'sintering_kinetics': sintering,
            'coefficient_of_thermal_expansion': cte,
            'creep_constitutive_data': creep,
            'elastic_properties': elastic
        }
        
        # ???JSON
        json_path = os.path.join(output_dir, 'sofc_material_dataset.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(complete_dataset, f, indent=2, ensure_ascii=False)
        print(f"? JSON??????: {json_path}")
        
        # ???CSV??????????????
        csv_dir = os.path.join(output_dir, 'csv_files')
        os.makedirs(csv_dir, exist_ok=True)
        
        # 1. ?????
        for layer, props in thermophysical.items():
            df = pd.DataFrame([props['green_state']])
            df['layer'] = layer
            df['material'] = props['material']
            csv_path = os.path.join(csv_dir, f'thermophysical_{layer}.csv')
            df.to_csv(csv_path, index=False)
            print(f"? CSV???: {csv_path}")
        
        # 2. ???????
        for key, data in sintering.items():
            df = pd.DataFrame({
                'temperature_C': data['temperature_C'],
                'shrinkage_dL_L0': data['shrinkage_dL_L0']
            })
            if 'mismatch_strain' in data:
                df['mismatch_strain'] = data['mismatch_strain']
                df['layer1_shrinkage'] = data['layer1_shrinkage']
                df['layer2_shrinkage'] = data['layer2_shrinkage']
            csv_path = os.path.join(csv_dir, f'sintering_{key}.csv')
            df.to_csv(csv_path, index=False)
            print(f"? CSV???: {csv_path}")
        
        # 3. CTE??
        for layer, data in cte.items():
            df = pd.DataFrame({
                'temperature_C': data['temperature_C'],
                'cte_per_K': data['cte_per_K'],
                'thermal_strain': data['thermal_strain']
            })
            csv_path = os.path.join(csv_dir, f'cte_{layer}.csv')
            df.to_csv(csv_path, index=False)
            print(f"? CSV???: {csv_path}")
        
        # 4. ??????????????
        creep_summary = []
        for material, data in creep.items():
            params = data['creep_parameters']
            creep_summary.append({
                'material': material,
                'A': params['A'],
                'n': params['n'],
                'Q': params['Q'],
                'units_A': params['units']['A'],
                'units_n': params['units']['n'],
                'units_Q': params['units']['Q']
            })
        df_creep = pd.DataFrame(creep_summary)
        csv_path = os.path.join(csv_dir, 'creep_parameters.csv')
        df_creep.to_csv(csv_path, index=False)
        print(f"? CSV???: {csv_path}")
        
        # 5. ??????
        for material, data in elastic.items():
            # ????
            df_temp = pd.DataFrame(data['temperature_dependent'])
            csv_path = os.path.join(csv_dir, f'elastic_temperature_{material}.csv')
            df_temp.to_csv(csv_path, index=False)
            print(f"? CSV???: {csv_path}")
            
            # ????
            df_rho = pd.DataFrame(data['density_dependent'])
            csv_path = os.path.join(csv_dir, f'elastic_density_{material}.csv')
            df_rho.to_csv(csv_path, index=False)
            print(f"? CSV???: {csv_path}")
        
        # ??README??
        readme_content = f"""# SOFC????????????

## ????
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## ?????
???????????????(SOFC)?????????????????????????(FEM)???????(DIC)???

## ?????

### 1. ????? (thermophysical_properties)
- ??????????
- ?????????
- ??????

### 2. ????? (sintering_kinetics)
- ???????? (dL/L0 vs T)
- ?????????
- ??????
- ???????

### 3. ????? (coefficient_of_thermal_expansion)
- ?????CTE?
- ???????

### 4. ?????? (creep_constitutive_data)
- Norton???? (A, n, Q)
- ??????????????
- ????????????

### 5. ???? (elastic_properties)
- ?????????????
- ???????????
- ???????????????

## ????

### JSON??
- `sofc_material_dataset.json`: ??????JSON???

### CSV??
??CSV????? `csv_files/` ????
- `thermophysical_*.csv`: ?????
- `sintering_*.csv`: ???????
- `cte_*.csv`: ???????
- `creep_parameters.csv`: ??????
- `elastic_temperature_*.csv`: ?????????
- `elastic_density_*.csv`: ?????????

## ????

### Python??
```python
import json
import pandas as pd

# ??JSON???
with open('sofc_material_dataset.json', 'r') as f:
    dataset = json.load(f)

# ??CSV??
sintering_data = pd.read_csv('csv_files/sintering_anode_single.csv')
cte_data = pd.read_csv('csv_files/cte_electrolyte.csv')
```

## ????
- ??????????SOFC??????
- ???????800-1200?C????
- ???????????????????
- ?????????????????

## ??
?????A Closed-Loop, Microstructure-Informed DIC-FEM Framework for the Real-Time Mitigation of Sintering Stresses in Solid Oxide Fuel Cells through Targeted Creep Activation
"""
        
        readme_path = os.path.join(output_dir, 'README.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"? README???: {readme_path}")
        
        print(f"\n????????")
        print(f"????: {len(thermophysical)} ?????????")
        print(f"          {len(sintering)} ????????")
        print(f"          {len(cte)} ????CTE??")
        print(f"          {len(creep)} ??????????")
        print(f"          {len(elastic)} ??????????")
        
        return complete_dataset


def main():
    """???"""
    print("=" * 80)
    print("SOFC???????????????")
    print("=" * 80)
    
    generator = SOFCDatasetGenerator()
    dataset = generator.save_datasets()
    
    print("\n" + "=" * 80)
    print("????????")
    print("=" * 80)


if __name__ == '__main__':
    main()
