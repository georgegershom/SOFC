"""
Model Calibration Helper Script
Provides functions to prepare data for numerical model calibration
and validation for fire-resistant rubberized concrete
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from data_loader import RubberizedConcreteDataLoader
from typing import Dict, Tuple, Callable


class ModelCalibrationHelper:
    """
    Helper class for model calibration and parameter fitting
    """
    
    def __init__(self, data_loader: RubberizedConcreteDataLoader):
        """
        Initialize calibration helper
        
        Args:
            data_loader: Instance of data loader with all data
        """
        self.loader = data_loader
        
    @staticmethod
    def eurocode_thermal_conductivity(T: np.ndarray, k0: float, k1: float, k2: float) -> np.ndarray:
        """
        Eurocode-style thermal conductivity model
        k(T) = k0 + k1*T + k2*T^2
        
        Args:
            T: Temperature array (°C)
            k0, k1, k2: Model parameters
            
        Returns:
            Thermal conductivity array
        """
        return k0 + k1 * T + k2 * T**2
    
    @staticmethod
    def exponential_decay_strength(T: np.ndarray, f0: float, T1: float, alpha: float) -> np.ndarray:
        """
        Exponential decay model for strength degradation
        f(T) = f0 * exp(-alpha * (T - T1))
        
        Args:
            T: Temperature array (°C)
            f0: Reference strength
            T1: Reference temperature
            alpha: Decay coefficient
            
        Returns:
            Strength array
        """
        return f0 * np.exp(-alpha * np.maximum(T - T1, 0))
    
    @staticmethod
    def power_law_porosity(T: np.ndarray, p0: float, Tref: float, n: float) -> np.ndarray:
        """
        Power law model for porosity evolution
        p(T) = p0 * (1 + (T/Tref)^n)
        
        Args:
            T: Temperature array (°C)
            p0: Initial porosity
            Tref: Reference temperature
            n: Exponent
            
        Returns:
            Porosity array
        """
        return p0 * (1 + (T/Tref)**n)
    
    def fit_thermal_conductivity_model(self, rubber_content: int = 0) -> Dict:
        """
        Fit thermal conductivity model to data
        
        Args:
            rubber_content: Rubber percentage (0, 10, 20, 30)
            
        Returns:
            Dictionary with fitted parameters and statistics
        """
        # Get data
        cond_data = self.loader.thermal_data['conductivity']
        data = cond_data[cond_data['Rubber_Content_Percent'] == rubber_content]
        
        T = data['Temperature_C'].values
        k = data['Thermal_Conductivity_W_mK'].values
        
        # Fit model
        try:
            popt, pcov = curve_fit(
                self.eurocode_thermal_conductivity,
                T, k,
                p0=[1.5, -0.001, 0.0],
                maxfev=10000
            )
            
            # Calculate R²
            k_pred = self.eurocode_thermal_conductivity(T, *popt)
            ss_res = np.sum((k - k_pred)**2)
            ss_tot = np.sum((k - np.mean(k))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {
                'parameters': {'k0': popt[0], 'k1': popt[1], 'k2': popt[2]},
                'covariance': pcov,
                'r_squared': r_squared,
                'rmse': np.sqrt(np.mean((k - k_pred)**2)),
                'model_function': lambda T: self.eurocode_thermal_conductivity(T, *popt)
            }
        except Exception as e:
            print(f"Error fitting thermal conductivity: {e}")
            return None
    
    def fit_strength_degradation_model(self, rubber_content: int = 0) -> Dict:
        """
        Fit strength degradation model to compressive strength data
        
        Args:
            rubber_content: Rubber percentage
            
        Returns:
            Dictionary with fitted parameters
        """
        # Get data
        strength_data = self.loader.mechanical_data['compressive_strength']
        data = strength_data[strength_data['Rubber_Content_Percent'] == rubber_content]
        
        T = data['Temperature_C'].values
        f_c = data['Compressive_Strength_MPa'].values
        
        # Use piecewise linear model for better fit
        # Split into regions
        T_peak = 200  # Peak temperature
        
        # Before peak (slight increase)
        mask1 = T <= T_peak
        if np.sum(mask1) > 2:
            p1 = np.polyfit(T[mask1], f_c[mask1], 1)
        else:
            p1 = [0, f_c[0]]
        
        # After peak (decay)
        mask2 = T > T_peak
        if np.sum(mask2) > 3:
            try:
                popt, _ = curve_fit(
                    self.exponential_decay_strength,
                    T[mask2], f_c[mask2],
                    p0=[f_c[mask2][0], T_peak, 0.002],
                    maxfev=10000
                )
            except:
                popt = [f_c[mask2][0], T_peak, 0.002]
        else:
            popt = [f_c[0], T_peak, 0.002]
        
        return {
            'pre_peak': {'slope': p1[0], 'intercept': p1[1]},
            'post_peak': {'f0': popt[0], 'T1': popt[1], 'alpha': popt[2]},
            'T_peak': T_peak
        }
    
    def create_material_input_file(
        self,
        rubber_content: int,
        output_format: str = 'ABAQUS'
    ) -> str:
        """
        Create material input file for FE software
        
        Args:
            rubber_content: Rubber percentage
            output_format: 'ABAQUS', 'ANSYS', or 'COMSOL'
            
        Returns:
            String with formatted material properties
        """
        # Get data
        cond = self.loader.thermal_data['conductivity']
        spec_heat = self.loader.thermal_data['specific_heat']
        density = self.loader.thermal_data['density']
        strength = self.loader.mechanical_data['compressive_strength']
        modulus = self.loader.mechanical_data['elastic_modulus']
        
        # Filter for rubber content
        cond = cond[cond['Rubber_Content_Percent'] == rubber_content]
        spec_heat = spec_heat[spec_heat['Rubber_Content_Percent'] == rubber_content]
        density = density[density['Rubber_Content_Percent'] == rubber_content]
        strength = strength[strength['Rubber_Content_Percent'] == rubber_content]
        modulus = modulus[modulus['Rubber_Content_Percent'] == rubber_content]
        
        if output_format == 'ABAQUS':
            return self._create_abaqus_input(
                rubber_content, cond, spec_heat, density, strength, modulus
            )
        elif output_format == 'ANSYS':
            return self._create_ansys_input(
                rubber_content, cond, spec_heat, density, strength, modulus
            )
        else:
            return "Format not yet implemented"
    
    def _create_abaqus_input(self, rc, cond, spec_heat, density, strength, modulus):
        """Create ABAQUS material input"""
        
        output = f"*MATERIAL, NAME=RubberConcrete_{rc}pct\n"
        output += "*DENSITY\n"
        for _, row in density.iterrows():
            output += f"{row['Density_kg_m3']}, {row['Temperature_C']}\n"
        
        output += "*ELASTIC, TYPE=ISOTROPIC\n"
        for _, row in modulus.iterrows():
            poisson = 0.18 + 0.0001 * row['Temperature_C']  # Simplified
            output += f"{row['Elastic_Modulus_GPa']*1000}, {poisson}, {row['Temperature_C']}\n"
        
        output += "*CONDUCTIVITY\n"
        for _, row in cond.iterrows():
            output += f"{row['Thermal_Conductivity_W_mK']}, {row['Temperature_C']}\n"
        
        output += "*SPECIFIC HEAT\n"
        for _, row in spec_heat.iterrows():
            output += f"{row['Specific_Heat_J_kgK']}, {row['Temperature_C']}\n"
        
        output += "*CONCRETE\n"
        for _, row in strength.iterrows():
            output += f"{row['Compressive_Strength_MPa']}, {row['Temperature_C']}\n"
        
        return output
    
    def _create_ansys_input(self, rc, cond, spec_heat, density, strength, modulus):
        """Create ANSYS material input"""
        
        output = f"! Material Properties for Rubberized Concrete {rc}%\n"
        output += f"MP,DENS,1,{density['Density_kg_m3'].iloc[0]}\n"
        output += f"MP,EX,1,{modulus['Elastic_Modulus_GPa'].iloc[0]*1e9}\n"
        output += f"MP,KXX,1,{cond['Thermal_Conductivity_W_mK'].iloc[0]}\n"
        output += f"MP,C,1,{spec_heat['Specific_Heat_J_kgK'].iloc[0]}\n"
        output += "\n! Temperature-dependent properties\n"
        output += "MPTEMP,1,20,100,200,300,400,500,600,700,800,900,1000\n"
        
        # Add temperature-dependent properties
        output += "! Conductivity\n"
        output += "MPDATA,KXX,1,1,"
        temps = [20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        for t in temps:
            k = cond[cond['Temperature_C'] == t]['Thermal_Conductivity_W_mK'].values
            if len(k) > 0:
                output += f"{k[0]},"
        output = output.rstrip(',') + "\n"
        
        return output
    
    def export_validation_dataset(
        self,
        fire_curve: str,
        rubber_content: int,
        output_file: str
    ):
        """
        Export validation dataset for external tools
        
        Args:
            fire_curve: Fire curve type
            rubber_content: Rubber percentage
            output_file: Output file path
        """
        validation = self.loader.get_validation_data_for_model(fire_curve, rubber_content)
        
        # Combine temperature and strain data
        temp_data = validation['temperature']
        strain_data = validation['strain']
        
        # Save to CSV
        with open(output_file, 'w') as f:
            f.write(f"# Validation Data: {fire_curve}, {rubber_content}% Rubber\n")
            f.write("# Temperature Evolution Data\n")
            temp_data.to_csv(f, index=False)
            f.write("\n# Strain Evolution Data\n")
            strain_data.to_csv(f, index=False)
        
        print(f"Validation dataset exported to: {output_file}")


def main():
    """Example usage"""
    
    # Load data
    loader = RubberizedConcreteDataLoader()
    loader.load_all_data()
    
    # Create calibration helper
    helper = ModelCalibrationHelper(loader)
    
    # Fit thermal conductivity model
    print("\nFitting thermal conductivity model for 0% rubber...")
    result = helper.fit_thermal_conductivity_model(rubber_content=0)
    if result:
        print(f"  Parameters: {result['parameters']}")
        print(f"  R²: {result['r_squared']:.4f}")
        print(f"  RMSE: {result['rmse']:.4f} W/mK")
    
    # Create ABAQUS input
    print("\nCreating ABAQUS material input for 20% rubber...")
    abaqus_input = helper.create_material_input_file(20, 'ABAQUS')
    print("  First 500 characters:")
    print(abaqus_input[:500])
    
    # Export validation dataset
    print("\nExporting validation dataset...")
    helper.export_validation_dataset('ISO834', 10, '../validation_ISO834_10pct.csv')


if __name__ == "__main__":
    main()
