#!/usr/bin/env python3
"""
Material Property & Constitutive Model Dataset Loader
??SOFC?????????????????
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

class MaterialDataLoader:
    """????????"""
    
    def __init__(self, dataset_root=None):
        """
        ????????
        
        Parameters:
        -----------
        dataset_root : str or Path
            ????????????????????????
        """
        if dataset_root is None:
            dataset_root = Path(__file__).parent.parent
        self.dataset_root = Path(dataset_root)
        self.thermal_dir = self.dataset_root / "thermal_properties"
        self.creep_dir = self.dataset_root / "creep_data"
        self.elastic_dir = self.dataset_root / "elastic_properties"
    
    def load_green_state_properties(self):
        """??????????"""
        file_path = self.thermal_dir / "green_state_properties.csv"
        return pd.read_csv(file_path)
    
    def load_dilatometry_data(self, material="8YSZ"):
        """
        ????????
        
        Parameters:
        -----------
        material : str
            ???? ("8YSZ", "NiYSZ", "bilayer_anode_electrolyte")
        """
        if material == "8YSZ":
            file_path = self.thermal_dir / "dilatometry_single_layer_8YSZ.csv"
        elif material == "NiYSZ":
            file_path = self.thermal_dir / "dilatometry_single_layer_NiYSZ.csv"
        elif material == "bilayer_anode_electrolyte":
            file_path = self.thermal_dir / "dilatometry_bilayer_anode_electrolyte.csv"
        else:
            raise ValueError(f"????: {material}")
        
        return pd.read_csv(file_path)
    
    def load_CTE_data(self, material=None):
        """
        ?????????
        
        Parameters:
        -----------
        material : str or None
            ???? ("8YSZ", "Ni-YSZ", "LSM-YSZ")?None??????
        """
        file_path = self.thermal_dir / "CTE_data.csv"
        df = pd.read_csv(file_path)
        
        if material is not None:
            df = df[df['Material'] == material]
        
        return df
    
    def load_creep_test_data(self, material="8YSZ"):
        """
        ????????
        
        Parameters:
        -----------
        material : str
            ???? ("8YSZ", "NiYSZ")
        """
        if material == "8YSZ":
            file_path = self.creep_dir / "creep_test_data_8YSZ.csv"
        elif material == "NiYSZ":
            file_path = self.creep_dir / "creep_test_data_NiYSZ.csv"
        else:
            raise ValueError(f"????: {material}")
        
        return pd.read_csv(file_path)
    
    def load_norton_parameters(self, material=None, state=None):
        """
        ??Norton??????
        
        Parameters:
        -----------
        material : str or None
            ?????None??????
        state : str or None
            ?? ("Green", "Sintering")?None??????
        """
        file_path = self.creep_dir / "norton_creep_parameters.csv"
        df = pd.read_csv(file_path)
        
        if material is not None:
            df = df[df['Material'] == material]
        if state is not None:
            df = df[df['State'] == state]
        
        return df
    
    def load_youngs_modulus(self, material=None, relative_density=None):
        """
        ????????
        
        Parameters:
        -----------
        material : str or None
            ?????None??????
        relative_density : float or None
            ?????None??????
        """
        file_path = self.elastic_dir / "youngs_modulus_temperature.csv"
        df = pd.read_csv(file_path)
        
        if material is not None:
            df = df[df['Material'] == material]
        if relative_density is not None:
            df = df[df['Relative_Density'] == relative_density]
        
        return df
    
    def load_poissons_ratio(self, material=None, relative_density=None):
        """
        ???????
        
        Parameters:
        -----------
        material : str or None
            ?????None??????
        relative_density : float or None
            ?????None??????
        """
        file_path = self.elastic_dir / "poissons_ratio_data.csv"
        df = pd.read_csv(file_path)
        
        if material is not None:
            df = df[df['Material'] == material]
        if relative_density is not None:
            df = df[df['Relative_Density'] == relative_density]
        
        return df
    
    def get_creep_strain_rate(self, material, state, temperature, stress):
        """
        ??Norton?????????
        
        Parameters:
        -----------
        material : str
            ????
        state : str
            ?? ("Green", "Sintering")
        temperature : float
            ?? (???)
        stress : float
            ?? (MPa)
        
        Returns:
        --------
        strain_rate : float
            ????? (s^-1)
        """
        params = self.load_norton_parameters(material=material, state=state)
        
        if len(params) == 0:
            raise ValueError(f"????? {material} ? {state} ??????")
        
        param = params.iloc[0]
        A = param['A_pre_exp_s_minus1_MPa_minus_n']
        n = param['n_stress_exponent']
        Q = param['Q_activation_energy_kJ_mol'] * 1000  # ???J/mol
        R = 8.314  # ???? J/(mol?K)
        T = temperature + 273.15  # ??????
        
        strain_rate = A * (stress ** n) * np.exp(-Q / (R * T))
        
        return strain_rate
    
    def interpolate_property(self, df, property_name, x_name, x_value, y_name=None, y_value=None):
        """
        ????????
        
        Parameters:
        -----------
        df : DataFrame
            ???
        property_name : str
            ????????
        x_name : str
            x????
        x_value : float
            x??
        y_name : str or None
            y???????2D???
        y_value : float or None
            y?????2D???
        
        Returns:
        --------
        interpolated_value : float
            ???????
        """
        from scipy.interpolate import interp1d, griddata
        
        if y_name is None:
            # 1D??
            sorted_df = df.sort_values(x_name)
            f = interp1d(sorted_df[x_name], sorted_df[property_name], 
                        kind='linear', bounds_error=False, fill_value='extrapolate')
            return float(f(x_value))
        else:
            # 2D??
            points = df[[x_name, y_name]].values
            values = df[property_name].values
            return float(griddata(points, values, (x_value, y_value), method='linear'))


def main():
    """????"""
    loader = MaterialDataLoader()
    
    print("=== ???????????? ===\n")
    
    # ????????
    print("1. ??????:")
    green_props = loader.load_green_state_properties()
    print(green_props.to_string())
    print()
    
    # ??Norton??
    print("2. Norton??????:")
    creep_params = loader.load_norton_parameters()
    print(creep_params.to_string())
    print()
    
    # ???????
    print("3. ??8YSZ?Green?????????:")
    strain_rate = loader.get_creep_strain_rate("8YSZ", "Green", 1200, 10)
    print(f"   ??: 1200?C, ??: 10 MPa")
    print(f"   ???: {strain_rate:.2e} s^-1")
    print()
    
    # ??CTE??
    print("4. 8YSZ??????:")
    cte_data = loader.load_CTE_data("8YSZ")
    print(cte_data[['Temperature_C', 'CTE_10minus6_per_K']].to_string())
    print()
    
    # ??????
    print("5. 8YSZ???? (????, ???):")
    youngs = loader.load_youngs_modulus("8YSZ", relative_density=0.99)
    print(youngs[['Temperature_C', 'Youngs_Modulus_GPa']].to_string())


if __name__ == "__main__":
    main()
