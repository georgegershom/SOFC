#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOFC????????
???????????
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class SOFCSinteringDataset:
    """SOFC???????????"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        ??????
        
        Args:
            base_dir: ?????????????????????
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.raw_data_dir = self.base_dir / "raw_data"
        self.processed_dir = self.base_dir / "processed_data"
        
        # ????????
        with open(self.processed_dir / "constitutive_models.json", 'r', encoding='utf-8') as f:
            self.constitutive_models = json.load(f)
    
    def get_green_state_properties(self, material: Optional[str] = None) -> pd.DataFrame:
        """
        ????????
        
        Args:
            material: ?????None??????
        
        Returns:
            ???????DataFrame
        """
        df = pd.read_csv(self.raw_data_dir / "green_state_properties.csv")
        
        if material:
            df = df[df['Material'] == material]
        
        return df
    
    def get_sintering_kinetics(self, layer: str) -> pd.DataFrame:
        """
        ?????????
        
        Args:
            layer: 'anode', 'electrolyte', ? 'cathode'
        
        Returns:
            ?????DataFrame
        """
        file_map = {
            'anode': 'anode_dilatometry.csv',
            'electrolyte': 'electrolyte_dilatometry.csv',
            'cathode': 'cathode_dilatometry.csv'
        }
        
        if layer not in file_map:
            raise ValueError(f"Layer???: {list(file_map.keys())}")
        
        return pd.read_csv(self.raw_data_dir / "sintering_kinetics" / file_map[layer])
    
    def get_bilayer_data(self) -> pd.DataFrame:
        """????????"""
        return pd.read_csv(self.raw_data_dir / "sintering_kinetics" / "bilayer_anode_electrolyte.csv")
    
    def get_cte_data(self, material: Optional[str] = None) -> pd.DataFrame:
        """
        ?????????
        
        Args:
            material: ????
        
        Returns:
            CTE??DataFrame
        """
        df = pd.read_csv(self.raw_data_dir / "cte_data" / "thermal_expansion_coefficients.csv")
        
        if material:
            df = df[df['Material'] == material]
        
        return df
    
    def get_creep_data(self, layer: str, temperature_C: int) -> pd.DataFrame:
        """
        ????????
        
        Args:
            layer: 'anode', 'electrolyte', ? 'cathode'
            temperature_C: ????
        
        Returns:
            ????DataFrame
        """
        filename = f"{layer}_creep_{temperature_C}C.csv"
        filepath = self.raw_data_dir / "creep_tests" / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"?????: {filename}")
        
        return pd.read_csv(filepath)
    
    def get_elastic_properties(self, material: Optional[str] = None, 
                               temperature_C: Optional[int] = None,
                               relative_density: Optional[float] = None) -> pd.DataFrame:
        """
        ????????
        
        Args:
            material: ????
            temperature_C: ??
            relative_density: ????
        
        Returns:
            ????DataFrame
        """
        df = pd.read_csv(self.raw_data_dir / "elastic_properties" / "youngs_modulus_temperature.csv")
        
        if material:
            df = df[df['Material'] == material]
        
        if temperature_C is not None:
            df = df[df['Temperature_C'] == temperature_C]
        
        if relative_density is not None:
            # ????????
            df = df.iloc[(df['Relative_Density'] - relative_density).abs().argsort()[:1]]
        
        return df
    
    def get_norton_parameters(self, layer: str, temperature_C: float) -> Dict:
        """
        ??Norton??????
        
        Args:
            layer: 'anode', 'electrolyte', ? 'cathode'
            temperature_C: ??
        
        Returns:
            ??A, n, Q???
        """
        if layer not in self.constitutive_models:
            raise ValueError(f"Layer???: {list(self.constitutive_models.keys())}")
        
        creep_params = self.constitutive_models[layer]['creep_parameters']
        
        # ????????
        for temp_range in creep_params['temperature_ranges']:
            range_str = temp_range['range_C']
            t_min, t_max = map(int, range_str.split('-'))
            
            if t_min <= temperature_C <= t_max:
                return {
                    'A': temp_range['A_prefactor'],
                    'n': temp_range['n_stress_exponent'],
                    'Q': temp_range['Q_activation_energy_kJ_mol'],
                    'temperature_range': range_str,
                    'density_range': temp_range['relative_density_range']
                }
        
        raise ValueError(f"?? {temperature_C}?C ??????")
    
    def calculate_creep_rate(self, layer: str, temperature_C: float, 
                            stress_MPa: float, relative_density: float) -> float:
        """
        ???????
        
        Args:
            layer: ???
            temperature_C: ??
            stress_MPa: ??
            relative_density: ????
        
        Returns:
            ??? (s^-1)
        """
        params = self.get_norton_parameters(layer, temperature_C)
        
        # Norton??
        A = params['A']
        n = params['n']
        Q = params['Q']
        
        # Arrhenius??
        R = 8.314  # J/(mol?K)
        T_K = temperature_C + 273.15
        arrhenius = np.exp(-Q * 1000 / (R * T_K))
        
        # ????
        density_corr = self.constitutive_models[layer]['creep_parameters']['density_correction_factor']
        b = density_corr['b_parameter']
        rho_0 = density_corr['rho_0']
        density_factor = np.exp(b * (relative_density - rho_0))
        
        # ?????
        strain_rate = A * (stress_MPa ** n) * arrhenius * density_factor
        
        return strain_rate
    
    def calculate_youngs_modulus(self, layer: str, temperature_C: float, 
                                 relative_density: float) -> float:
        """
        ??????
        
        Args:
            layer: ???
            temperature_C: ??
            relative_density: ????
        
        Returns:
            ???? (GPa)
        """
        elastic_props = self.constitutive_models[layer]['elastic_properties']
        
        E_ref = elastic_props['reference_modulus_GPa']
        T_ref = elastic_props['reference_temperature_C']
        rho_ref = elastic_props['reference_density']
        alpha = elastic_props['temperature_coefficient']
        m = elastic_props['density_exponent']
        
        # E(T, ?) = E_ref * (1 + ?(T - T_ref)) * (?/?_ref)^m
        E = E_ref * (1 + alpha * (temperature_C - T_ref)) * ((relative_density / rho_ref) ** m)
        
        return max(E, 0.1)  # ????
    
    def get_summary_statistics(self) -> Dict:
        """?????????"""
        stats = {
            "materials": {
                "anode": "Ni-YSZ",
                "electrolyte": "8YSZ",
                "cathode": "LSM-YSZ"
            },
            "temperature_ranges": {},
            "stress_ranges": {},
            "density_ranges": {}
        }
        
        for layer in ['anode', 'electrolyte', 'cathode']:
            # ????
            temp_ranges = [tr['range_C'] for tr in 
                          self.constitutive_models[layer]['creep_parameters']['temperature_ranges']]
            stats['temperature_ranges'][layer] = temp_ranges
            
            # ???????????
            creep_files = list((self.raw_data_dir / "creep_tests").glob(f"{layer}_creep_*.csv"))
            if creep_files:
                df = pd.read_csv(creep_files[0])
                stats['stress_ranges'][layer] = f"{df['Stress_MPa'].min()}-{df['Stress_MPa'].max()} MPa"
            
            # ????
            green_props = self.get_green_state_properties(self.constitutive_models[layer]['material'])
            if not green_props.empty:
                init_porosity = green_props['Initial_Porosity_fraction'].values[0]
                init_density = 1 - init_porosity
                stats['density_ranges'][layer] = f"{init_density:.2f}-0.98"
        
        return stats
    
    def __repr__(self):
        """?????"""
        return (f"SOFCSinteringDataset(\n"
                f"  ????: {self.base_dir}\n"
                f"  ??: ??(Ni-YSZ), ???(8YSZ), ??(LSM-YSZ)\n"
                f"  ????: ????, ?????, CTE, ??, ????\n"
                f")")


def example_usage():
    """????"""
    print("\n" + "="*70)
    print("  SOFC?????????")
    print("="*70 + "\n")
    
    # ???????
    dataset = SOFCSinteringDataset()
    print(dataset)
    
    # ??1: ??????
    print("\n??1: ??????")
    print("-" * 70)
    anode_green = dataset.get_green_state_properties('Ni-YSZ')
    print(anode_green.to_string(index=False))
    
    # ??2: ???????
    print("\n\n??2: ????????? (?5?)")
    print("-" * 70)
    electrolyte_sintering = dataset.get_sintering_kinetics('electrolyte')
    print(electrolyte_sintering.head().to_string(index=False))
    
    # ??3: ??Norton??
    print("\n\n??3: ??1000?C Norton????")
    print("-" * 70)
    norton_params = dataset.get_norton_parameters('anode', 1000)
    for key, value in norton_params.items():
        print(f"  {key}: {value}")
    
    # ??4: ???????
    print("\n\n??4: ???????")
    print("-" * 70)
    strain_rate = dataset.calculate_creep_rate(
        layer='anode',
        temperature_C=1000,
        stress_MPa=2.0,
        relative_density=0.70
    )
    print(f"  ??: T=1000?C, ?=2.0 MPa, ?=0.70")
    print(f"  ???: ?? = {strain_rate:.3e} s??")
    
    # ??5: ??????
    print("\n\n??5: ??????")
    print("-" * 70)
    youngs_mod = dataset.calculate_youngs_modulus(
        layer='electrolyte',
        temperature_C=1200,
        relative_density=0.90
    )
    print(f"  ??: T=1200?C, ?=0.90")
    print(f"  ????: E = {youngs_mod:.1f} GPa")
    
    # ??6: ?????
    print("\n\n??6: ???????")
    print("-" * 70)
    summary = dataset.get_summary_statistics()
    for key, value in summary.items():
        print(f"\n{key}:")
        for subkey, subvalue in value.items():
            print(f"  {subkey}: {subvalue}")
    
    print("\n" + "="*70)
    print("  ?????")
    print("="*70 + "\n")


if __name__ == "__main__":
    example_usage()
