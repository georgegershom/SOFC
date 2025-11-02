#!/usr/bin/env python3
"""
SOFC Co-Sintering Material Property & Constitutive Model Dataset Generator
???????????????????????????

??????: A Closed-Loop, Microstructure-Informed DIC-FEM Framework 
for the Real-Time Mitigation of Sintering Stresses in Solid Oxide Fuel Cells 
through Targeted Creep Activation
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from scipy.interpolate import interp1d

class SOFCMaterialDatasetGenerator:
    """SOFC????????"""
    
    def __init__(self, output_dir="sofc_dataset"):
        self.output_dir = output_dir
        self.materials = {
            'anode': 'NiO-YSZ',  # ???-????????
            'electrolyte': '8YSZ',  # 8%????????
            'cathode': 'LSM-YSZ',  # ????????-YSZ????
            'interconnect': 'Crofer22APU'  # ????
        }
        
        # ???? (?C)
        self.temp_range = np.linspace(25, 1400, 100)
        # ?????? (?C)
        self.sintering_temps = np.array([800, 900, 1000, 1100, 1200, 1300, 1400])
        # ???? (MPa)
        self.stress_range = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        # ???? (??)
        self.time_range = np.linspace(0, 10, 200)
        
        self._create_output_directories()
    
    def _create_output_directories(self):
        """????????"""
        dirs = [
            self.output_dir,
            f"{self.output_dir}/thermal_physical",
            f"{self.output_dir}/sintering_kinetics",
            f"{self.output_dir}/creep_data",
            f"{self.output_dir}/elastic_properties",
            f"{self.output_dir}/constitutive_parameters",
            f"{self.output_dir}/visualization",
            f"{self.output_dir}/raw_experimental"
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def generate_green_state_properties(self):
        """??????????"""
        print("????????????...")
        
        green_props = {}
        for layer, material in self.materials.items():
            if layer == 'interconnect':
                continue  # ????????????
                
            green_props[layer] = {
                'material': material,
                'density_green': self._get_green_density(layer),  # g/cm?
                'porosity_green': self._get_green_porosity(layer),  # %
                'relative_density': self._get_relative_density(layer),  # %
                'binder_content': self._get_binder_content(layer),  # wt%
                'porogen_content': self._get_porogen_content(layer),  # wt%
                'particle_size_d50': self._get_particle_size(layer),  # ?m
                'specific_surface_area': self._get_ssa(layer),  # m?/g
                'green_strength': self._get_green_strength(layer),  # MPa
            }
        
        # ???JSON?CSV
        with open(f"{self.output_dir}/thermal_physical/green_state_properties.json", 'w') as f:
            json.dump(green_props, f, indent=2)
        
        df = pd.DataFrame(green_props).T
        df.to_csv(f"{self.output_dir}/thermal_physical/green_state_properties.csv")
        
        return green_props
    
    def generate_sintering_kinetics(self):
        """????????? - ????"""
        print("???????????...")
        
        kinetics_data = {}
        
        for layer, material in self.materials.items():
            if layer == 'interconnect':
                continue
            
            # ????????
            data = self._generate_dilatometry_curve(layer)
            kinetics_data[layer] = data
            
            # ??????
            df = pd.DataFrame(data)
            df.to_csv(f"{self.output_dir}/sintering_kinetics/{layer}_dilatometry.csv", index=False)
        
        # ????????
        bilayer_pairs = [
            ('anode', 'electrolyte'),
            ('electrolyte', 'cathode')
        ]
        
        for layer1, layer2 in bilayer_pairs:
            bilayer_data = self._generate_bilayer_dilatometry(layer1, layer2)
            kinetics_data[f'{layer1}_{layer2}'] = bilayer_data
            
            df = pd.DataFrame(bilayer_data)
            df.to_csv(f"{self.output_dir}/sintering_kinetics/{layer1}_{layer2}_bilayer.csv", index=False)
        
        return kinetics_data
    
    def generate_cte_data(self):
        """???????(CTE)??"""
        print("????CTE??...")
        
        cte_data = {}
        
        for layer, material in self.materials.items():
            temps = self.temp_range
            cte = self._calculate_cte(layer, temps)
            thermal_strain = self._calculate_thermal_strain(layer, temps)
            
            data = {
                'temperature_C': temps,
                'CTE_ppm_per_K': cte,
                'thermal_strain': thermal_strain,
                'instantaneous_CTE': np.gradient(thermal_strain, temps) * 1e6
            }
            
            cte_data[layer] = data
            
            df = pd.DataFrame(data)
            df.to_csv(f"{self.output_dir}/thermal_physical/{layer}_CTE.csv", index=False)
        
        return cte_data
    
    def generate_creep_test_data(self):
        """????????"""
        print("??????????...")
        
        creep_data = {}
        
        for layer, material in self.materials.items():
            if layer == 'interconnect':
                continue
            
            layer_creep = {}
            
            for temp in self.sintering_temps:
                if temp < 800:
                    continue
                    
                for stress in self.stress_range:
                    test_key = f'T{temp}C_S{stress}MPa'
                    
                    # ??????-????
                    creep_strain, creep_rate = self._generate_creep_curve(
                        layer, temp, stress, self.time_range
                    )
                    
                    layer_creep[test_key] = {
                        'temperature_C': float(temp),
                        'stress_MPa': float(stress),
                        'time_hours': self.time_range.tolist(),
                        'creep_strain': creep_strain.tolist(),
                        'creep_rate_per_hour': creep_rate.tolist(),
                        'steady_state_rate': float(self._get_steady_state_rate(creep_rate))
                    }
            
            creep_data[layer] = layer_creep
            
            # ?????????
            with open(f"{self.output_dir}/creep_data/{layer}_creep_tests.json", 'w') as f:
                json.dump(layer_creep, f, indent=2)
        
        return creep_data
    
    def generate_norton_parameters(self):
        """??Norton??????"""
        print("????Norton??????...")
        
        norton_params = {}
        
        for layer, material in self.materials.items():
            if layer == 'interconnect':
                continue
            
            # Norton??: ?_dot = A * ?^n * exp(-Q/RT)
            params = self._fit_norton_parameters(layer)
            
            norton_params[layer] = {
                'material': material,
                'A_pre_exponential': params['A'],  # 1/(MPa^n * s)
                'n_stress_exponent': params['n'],  # ???
                'Q_activation_energy': params['Q'],  # kJ/mol
                'R_gas_constant': 8.314,  # J/(mol*K)
                'valid_temp_range_C': [800, 1400],
                'valid_stress_range_MPa': [0.1, 10.0],
                'confidence_interval_95': {
                    'A': [params['A'] * 0.8, params['A'] * 1.2],
                    'n': [params['n'] - 0.2, params['n'] + 0.2],
                    'Q': [params['Q'] - 20, params['Q'] + 20]
                }
            }
        
        # ??Norton??
        with open(f"{self.output_dir}/constitutive_parameters/norton_law_parameters.json", 'w') as f:
            json.dump(norton_params, f, indent=2)
        
        df = pd.DataFrame(norton_params).T
        df.to_csv(f"{self.output_dir}/constitutive_parameters/norton_law_parameters.csv")
        
        return norton_params
    
    def generate_elastic_properties(self):
        """????????"""
        print("??????????...")
        
        elastic_data = {}
        
        for layer, material in self.materials.items():
            # ?????? (???????)
            relative_densities = np.linspace(0.50, 0.98, 20)
            
            layer_elastic = []
            
            for rho_rel in relative_densities:
                for temp in self.temp_range[::10]:  # ??10??
                    E = self._calculate_youngs_modulus(layer, temp, rho_rel)
                    nu = self._calculate_poisson_ratio(layer, temp, rho_rel)
                    G = E / (2 * (1 + nu))  # ????
                    K = E / (3 * (1 - 2 * nu))  # ????
                    
                    layer_elastic.append({
                        'temperature_C': temp,
                        'relative_density': rho_rel,
                        'porosity': 1 - rho_rel,
                        'youngs_modulus_GPa': E,
                        'poisson_ratio': nu,
                        'shear_modulus_GPa': G,
                        'bulk_modulus_GPa': K
                    })
            
            elastic_data[layer] = layer_elastic
            
            df = pd.DataFrame(layer_elastic)
            df.to_csv(f"{self.output_dir}/elastic_properties/{layer}_elastic_properties.csv", index=False)
        
        return elastic_data
    
    def generate_density_evolution(self):
        """??????????????"""
        print("??????????...")
        
        density_data = {}
        
        for layer, material in self.materials.items():
            if layer == 'interconnect':
                continue
            
            # ??????: ??+??
            heating_rate = 5  # ?C/min
            hold_temp = 1400  # ?C
            hold_time = 2  # hours
            
            # ??-??-????
            time_temp_density = self._simulate_density_evolution(
                layer, heating_rate, hold_temp, hold_time
            )
            
            density_data[layer] = time_temp_density
            
            df = pd.DataFrame(time_temp_density)
            df.to_csv(f"{self.output_dir}/sintering_kinetics/{layer}_density_evolution.csv", index=False)
        
        return density_data
    
    # ========== ?????? ==========
    
    def _get_green_density(self, layer):
        """??????"""
        densities = {
            'anode': 3.2,  # NiO-YSZ
            'electrolyte': 3.0,  # 8YSZ
            'cathode': 3.5,  # LSM-YSZ
        }
        return densities.get(layer, 3.0)
    
    def _get_green_porosity(self, layer):
        """???????"""
        porosities = {
            'anode': 45.0,  # ????????????
            'electrolyte': 35.0,
            'cathode': 40.0,
        }
        return porosities.get(layer, 40.0)
    
    def _get_relative_density(self, layer):
        """??????"""
        return 100 - self._get_green_porosity(layer)
    
    def _get_binder_content(self, layer):
        """???????"""
        return np.random.uniform(2.0, 5.0)
    
    def _get_porogen_content(self, layer):
        """?????????"""
        porogen = {
            'anode': 15.0,  # ????????
            'electrolyte': 0.0,  # ???????
            'cathode': 10.0,
        }
        return porogen.get(layer, 5.0)
    
    def _get_particle_size(self, layer):
        """??????"""
        sizes = {
            'anode': 0.8,
            'electrolyte': 0.5,
            'cathode': 1.0,
        }
        return sizes.get(layer, 0.7)
    
    def _get_ssa(self, layer):
        """??????"""
        ssa = {
            'anode': 8.5,
            'electrolyte': 12.0,
            'cathode': 6.0,
        }
        return ssa.get(layer, 8.0)
    
    def _get_green_strength(self, layer):
        """??????"""
        return np.random.uniform(1.5, 3.5)
    
    def _generate_dilatometry_curve(self, layer):
        """????????"""
        temps = self.temp_range
        
        # ??????
        T_onset = self._get_sintering_onset_temp(layer)
        T_end = self._get_sintering_end_temp(layer)
        max_shrinkage = self._get_max_shrinkage(layer)
        
        # S?????
        shrinkage = max_shrinkage / (1 + np.exp(-0.02 * (temps - (T_onset + T_end) / 2)))
        
        # ???????
        thermal_expansion = self._calculate_thermal_strain(layer, temps)
        
        # ??? = ??? - ????
        total_strain = thermal_expansion - shrinkage / 100
        
        # ????
        shrinkage_rate = np.gradient(shrinkage, temps)
        
        return {
            'temperature_C': temps.tolist(),
            'linear_shrinkage_percent': shrinkage.tolist(),
            'total_strain': total_strain.tolist(),
            'shrinkage_rate_per_C': shrinkage_rate.tolist(),
            'relative_density': (0.55 + 0.43 * shrinkage / max_shrinkage).tolist()
        }
    
    def _generate_bilayer_dilatometry(self, layer1, layer2):
        """????????"""
        temps = self.temp_range
        
        # ??????
        data1 = self._generate_dilatometry_curve(layer1)
        data2 = self._generate_dilatometry_curve(layer2)
        
        # ?????????????(????)
        thickness_ratio = 0.5  # ????????
        
        bilayer_shrinkage = (
            np.array(data1['linear_shrinkage_percent']) * thickness_ratio +
            np.array(data2['linear_shrinkage_percent']) * (1 - thickness_ratio)
        )
        
        # ????(??CTE???)
        cte1 = self._calculate_cte(layer1, temps)
        cte2 = self._calculate_cte(layer2, temps)
        cte_mismatch = cte1 - cte2
        
        # ???????
        E_avg = 100  # GPa
        stress_MPa = E_avg * 1000 * cte_mismatch * (temps - 25) / 1e6
        
        return {
            'temperature_C': temps.tolist(),
            'bilayer_shrinkage_percent': bilayer_shrinkage.tolist(),
            'cte_mismatch_ppm_per_K': cte_mismatch.tolist(),
            'estimated_stress_MPa': stress_MPa.tolist(),
            'layer1_shrinkage': data1['linear_shrinkage_percent'],
            'layer2_shrinkage': data2['linear_shrinkage_percent']
        }
    
    def _get_sintering_onset_temp(self, layer):
        """??????"""
        temps = {
            'anode': 900,
            'electrolyte': 1100,
            'cathode': 950,
        }
        return temps.get(layer, 1000)
    
    def _get_sintering_end_temp(self, layer):
        """??????"""
        temps = {
            'anode': 1300,
            'electrolyte': 1400,
            'cathode': 1350,
        }
        return temps.get(layer, 1350)
    
    def _get_max_shrinkage(self, layer):
        """?????"""
        shrinkage = {
            'anode': 18.0,  # %
            'electrolyte': 22.0,
            'cathode': 20.0,
        }
        return shrinkage.get(layer, 20.0)
    
    def _calculate_cte(self, layer, temps):
        """???????"""
        # CTE?????????
        base_cte = {
            'anode': 12.5,  # ppm/K at RT
            'electrolyte': 10.5,
            'cathode': 11.8,
            'interconnect': 12.0,
        }
        
        cte_base = base_cte.get(layer, 11.0)
        # ?????: CTE(T) = CTE0 * (1 + ?*T)
        alpha = 0.0001
        cte = cte_base * (1 + alpha * (temps - 25))
        
        return cte
    
    def _calculate_thermal_strain(self, layer, temps):
        """?????"""
        cte = self._calculate_cte(layer, temps)
        # ??CTE????
        thermal_strain = np.cumsum(cte * np.gradient(temps)) / 1e6
        return thermal_strain
    
    def _generate_creep_curve(self, layer, temp, stress, time):
        """??????"""
        # Norton????
        params = self._fit_norton_parameters(layer)
        
        A = params['A']
        n = params['n']
        Q = params['Q']
        R = 8.314  # J/(mol*K)
        T_K = temp + 273.15
        
        # ????: ?_dot = A * ?^n * exp(-Q/RT)
        creep_rate_const = A * (stress ** n) * np.exp(-Q * 1000 / (R * T_K))
        
        # ???????
        # ??? (transient)
        t1 = 0.5  # hours
        primary_strain = 0.002 * (1 - np.exp(-time / t1))
        
        # ???? (steady-state)
        secondary_strain = creep_rate_const * time * 3600  # ?????
        
        # ???? (tertiary) - ?????????
        if stress > 5.0 and time[-1] > 5:
            t3 = time[-1] / 2
            tertiary_strain = 0.001 * np.exp((time - t3) / 2) * (time > t3)
        else:
            tertiary_strain = 0
        
        total_strain = primary_strain + secondary_strain + tertiary_strain
        
        # ????
        creep_rate = np.gradient(total_strain, time)
        
        return total_strain, creep_rate
    
    def _get_steady_state_rate(self, creep_rate):
        """????????"""
        # ???50%??????
        start_idx = len(creep_rate) // 4
        end_idx = 3 * len(creep_rate) // 4
        return float(np.mean(creep_rate[start_idx:end_idx]))
    
    def _fit_norton_parameters(self, layer):
        """??Norton??"""
        # ????????
        norton_params = {
            'anode': {
                'A': 1.5e-5,  # 1/(MPa^n * s)
                'n': 2.2,
                'Q': 450,  # kJ/mol
            },
            'electrolyte': {
                'A': 8.0e-7,
                'n': 1.8,
                'Q': 520,
            },
            'cathode': {
                'A': 2.5e-5,
                'n': 2.5,
                'Q': 420,
            },
        }
        return norton_params.get(layer, {'A': 1e-5, 'n': 2.0, 'Q': 450})
    
    def _calculate_youngs_modulus(self, layer, temp, relative_density):
        """??????"""
        # ????????
        E0_RT = {
            'anode': 180,  # GPa at room temp
            'electrolyte': 210,
            'cathode': 150,
            'interconnect': 200,
        }
        
        E0 = E0_RT.get(layer, 180)
        
        # ?????: E(T) = E0 * (1 - ?*T)
        beta = 0.0003
        E_T = E0 * (1 - beta * (temp - 25))
        
        # ??????: E(?) = E0 * exp(b*(?-1))
        # ?????: E(?) = E0 * ?^m
        m = 3.5
        E = E_T * (relative_density ** m)
        
        return max(E, 1.0)  # ???1 GPa
    
    def _calculate_poisson_ratio(self, layer, temp, relative_density):
        """?????"""
        # ???????
        nu0 = {
            'anode': 0.28,
            'electrolyte': 0.32,
            'cathode': 0.30,
            'interconnect': 0.29,
        }
        
        nu_base = nu0.get(layer, 0.30)
        
        # ????????
        nu = nu_base * (0.8 + 0.2 * relative_density)
        
        return min(nu, 0.49)  # ??0.49
    
    def _simulate_density_evolution(self, layer, heating_rate, hold_temp, hold_time):
        """??????"""
        # ????-????
        t_heat = hold_temp / heating_rate  # minutes
        t_hold = hold_time * 60  # minutes
        
        time_heat = np.linspace(0, t_heat, int(t_heat * 2))
        time_hold = np.linspace(t_heat, t_heat + t_hold, int(t_hold / 5))
        
        temp_heat = time_heat * heating_rate
        temp_hold = np.full_like(time_hold, hold_temp)
        
        time_total = np.concatenate([time_heat, time_hold])
        temp_total = np.concatenate([temp_heat, temp_hold])
        
        # ??????
        rho0 = self._get_relative_density(layer) / 100
        rho_final = 0.96
        
        T_onset = self._get_sintering_onset_temp(layer)
        
        density = np.zeros_like(time_total)
        
        for i, (t, T) in enumerate(zip(time_total, temp_total)):
            if T < T_onset:
                density[i] = rho0
            else:
                # ?????
                k = 0.01 * np.exp(-300000 / (8.314 * (T + 273.15)))
                dt = time_total[i] - time_total[i-1] if i > 0 else 0
                drho = k * (rho_final - density[i-1] if i > 0 else rho0) * dt
                density[i] = (density[i-1] if i > 0 else rho0) + drho
        
        return {
            'time_minutes': time_total.tolist(),
            'temperature_C': temp_total.tolist(),
            'relative_density': density.tolist(),
            'densification_rate': np.gradient(density, time_total).tolist()
        }
    
    def generate_complete_dataset(self):
        """???????"""
        print("\n" + "="*60)
        print("????SOFC??????????")
        print("="*60 + "\n")
        
        metadata = {
            'dataset_name': 'SOFC Co-Sintering Material Property & Constitutive Model Dataset',
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'research_topic': 'A Closed-Loop, Microstructure-Informed DIC-FEM Framework for Real-Time Mitigation of Sintering Stresses',
            'materials': self.materials,
            'temperature_range_C': [float(self.temp_range[0]), float(self.temp_range[-1])],
            'description': 'Comprehensive material property dataset for FEM simulation of SOFC co-sintering process'
        }
        
        # ???????
        green_props = self.generate_green_state_properties()
        sintering_kinetics = self.generate_sintering_kinetics()
        cte_data = self.generate_cte_data()
        creep_data = self.generate_creep_test_data()
        norton_params = self.generate_norton_parameters()
        elastic_data = self.generate_elastic_properties()
        density_evolution = self.generate_density_evolution()
        
        # ?????
        metadata['data_files'] = {
            'green_state': 'thermal_physical/green_state_properties.json',
            'sintering_kinetics': 'sintering_kinetics/*.csv',
            'cte_data': 'thermal_physical/*_CTE.csv',
            'creep_tests': 'creep_data/*_creep_tests.json',
            'norton_parameters': 'constitutive_parameters/norton_law_parameters.json',
            'elastic_properties': 'elastic_properties/*_elastic_properties.csv',
            'density_evolution': 'sintering_kinetics/*_density_evolution.csv'
        }
        
        with open(f"{self.output_dir}/dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n" + "="*60)
        print("???????!")
        print(f"????: {self.output_dir}")
        print("="*60 + "\n")
        
        return metadata


if __name__ == "__main__":
    generator = SOFCMaterialDatasetGenerator()
    metadata = generator.generate_complete_dataset()
    
    print("\n???????:")
    print("-" * 60)
    for category, files in metadata['data_files'].items():
        print(f"  {category}: {files}")
    print("-" * 60)
