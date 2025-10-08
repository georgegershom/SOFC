"""
Material Properties Dataset Loader and Interpolator for YSZ
For FEM Thermomechanical Simulation

This module provides functions to load, interpolate, and visualize
temperature-dependent material properties for Yttria-Stabilized Zirconia (YSZ).
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path


class YSZMaterialProperties:
    """
    Class to handle YSZ material properties with temperature interpolation.
    """
    
    def __init__(self, data_dir="."):
        """
        Initialize the material properties loader.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the CSV files
        """
        self.data_dir = Path(data_dir)
        self.props_df = None
        self.weibull_df = None
        self.creep_params_df = None
        self.interpolators = {}
        
        self._load_data()
        self._create_interpolators()
    
    def _load_data(self):
        """Load all material property CSV files."""
        # Main temperature-dependent properties
        self.props_df = pd.read_csv(
            self.data_dir / "ysz_material_properties.csv"
        )
        
        # Weibull statistical parameters
        self.weibull_df = pd.read_csv(
            self.data_dir / "weibull_parameters.csv"
        )
        
        # Creep model parameters
        self.creep_params_df = pd.read_csv(
            self.data_dir / "creep_model_parameters.csv"
        )
        
        print("✓ Loaded material properties dataset")
        print(f"  Temperature range: {self.props_df['Temperature_C'].min()}°C to "
              f"{self.props_df['Temperature_C'].max()}°C")
        print(f"  Number of data points: {len(self.props_df)}")
    
    def _create_interpolators(self):
        """Create interpolation functions for all temperature-dependent properties."""
        temp = self.props_df['Temperature_C'].values
        
        # Create interpolators for main properties
        property_columns = [
            'Youngs_Modulus_GPa',
            'Poissons_Ratio',
            'CTE_1e-6_K',
            'Density_kg_m3',
            'Thermal_Conductivity_W_mK',
            'Fracture_Toughness_MPa_m0.5',
            'Creep_Exponent_n',
            'Creep_Activation_Energy_kJ_mol'
        ]
        
        for col in property_columns:
            values = self.props_df[col].values
            # Use cubic interpolation for smooth curves
            self.interpolators[col] = interp1d(
                temp, values, kind='cubic', 
                fill_value='extrapolate', bounds_error=False
            )
        
        # Weibull parameters
        temp_weibull = self.weibull_df['Temperature_C'].values
        for col in ['Weibull_Modulus_m', 'Characteristic_Strength_MPa']:
            values = self.weibull_df[col].values
            self.interpolators[col] = interp1d(
                temp_weibull, values, kind='cubic',
                fill_value='extrapolate', bounds_error=False
            )
        
        print("✓ Created interpolation functions")
    
    def get_property(self, property_name, temperature_c):
        """
        Get interpolated property value at specified temperature.
        
        Parameters:
        -----------
        property_name : str
            Name of the property (column name from CSV)
        temperature_c : float or array-like
            Temperature in Celsius
            
        Returns:
        --------
        float or array
            Interpolated property value(s)
        """
        if property_name not in self.interpolators:
            raise ValueError(f"Property '{property_name}' not found. "
                           f"Available: {list(self.interpolators.keys())}")
        
        return self.interpolators[property_name](temperature_c)
    
    def get_all_properties(self, temperature_c):
        """
        Get all material properties at specified temperature.
        
        Parameters:
        -----------
        temperature_c : float
            Temperature in Celsius
            
        Returns:
        --------
        dict
            Dictionary of all material properties
        """
        props = {}
        for name, interpolator in self.interpolators.items():
            props[name] = float(interpolator(temperature_c))
        
        return props
    
    def get_creep_rate(self, stress_mpa, temperature_c, grain_size_um=1.0):
        """
        Calculate creep strain rate using power-law creep model.
        
        ε̇ = A * σ^n * d^-m * exp(-Q/RT)
        
        Parameters:
        -----------
        stress_mpa : float
            Applied stress in MPa
        temperature_c : float
            Temperature in Celsius
        grain_size_um : float
            Grain size in micrometers
            
        Returns:
        --------
        float
            Creep strain rate in 1/s
        """
        # Get parameters
        params = self.creep_params_df.set_index('Parameter')
        A = params.loc['A_creep', 'Value']
        n = self.get_property('Creep_Exponent_n', temperature_c)
        Q = params.loc['Q_activation', 'Value'] * 1000  # Convert to J/mol
        R = params.loc['R_gas_constant', 'Value']
        m = params.loc['m_grain_size_exponent', 'Value']
        
        # Convert to absolute temperature
        T_kelvin = temperature_c + 273.15
        
        # Convert stress to Pa
        stress_pa = stress_mpa * 1e6
        
        # Calculate creep rate
        creep_rate = (A * (stress_pa ** n) * (grain_size_um ** -m) * 
                     np.exp(-Q / (R * T_kelvin)))
        
        return creep_rate
    
    def plot_properties(self, save_path=None):
        """
        Create visualization of all temperature-dependent properties.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
        """
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('YSZ Material Properties vs Temperature', 
                     fontsize=16, fontweight='bold')
        
        temp = self.props_df['Temperature_C'].values
        temp_smooth = np.linspace(temp.min(), temp.max(), 200)
        
        # Plot 1: Young's Modulus
        ax = axes[0, 0]
        E_smooth = self.get_property('Youngs_Modulus_GPa', temp_smooth)
        ax.plot(temp, self.props_df['Youngs_Modulus_GPa'], 'o', label='Data')
        ax.plot(temp_smooth, E_smooth, '-', label='Interpolated')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Young\'s Modulus (GPa)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 2: Poisson's Ratio
        ax = axes[0, 1]
        nu_smooth = self.get_property('Poissons_Ratio', temp_smooth)
        ax.plot(temp, self.props_df['Poissons_Ratio'], 'o')
        ax.plot(temp_smooth, nu_smooth, '-')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Poisson\'s Ratio')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: CTE
        ax = axes[0, 2]
        cte_smooth = self.get_property('CTE_1e-6_K', temp_smooth)
        ax.plot(temp, self.props_df['CTE_1e-6_K'], 'o')
        ax.plot(temp_smooth, cte_smooth, '-')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('CTE (10⁻⁶/K)')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Density
        ax = axes[1, 0]
        rho_smooth = self.get_property('Density_kg_m3', temp_smooth)
        ax.plot(temp, self.props_df['Density_kg_m3'], 'o')
        ax.plot(temp_smooth, rho_smooth, '-')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Density (kg/m³)')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Thermal Conductivity
        ax = axes[1, 1]
        k_smooth = self.get_property('Thermal_Conductivity_W_mK', temp_smooth)
        ax.plot(temp, self.props_df['Thermal_Conductivity_W_mK'], 'o')
        ax.plot(temp_smooth, k_smooth, '-')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Thermal Conductivity (W/m·K)')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Fracture Toughness
        ax = axes[1, 2]
        KIC_smooth = self.get_property('Fracture_Toughness_MPa_m0.5', temp_smooth)
        ax.plot(temp, self.props_df['Fracture_Toughness_MPa_m0.5'], 'o')
        ax.plot(temp_smooth, KIC_smooth, '-')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Fracture Toughness (MPa·m⁰·⁵)')
        ax.grid(True, alpha=0.3)
        
        # Plot 7: Weibull Modulus
        ax = axes[2, 0]
        temp_weibull = self.weibull_df['Temperature_C'].values
        m_smooth = self.get_property('Weibull_Modulus_m', temp_smooth)
        ax.plot(temp_weibull, self.weibull_df['Weibull_Modulus_m'], 'o')
        ax.plot(temp_smooth, m_smooth, '-')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Weibull Modulus')
        ax.grid(True, alpha=0.3)
        
        # Plot 8: Characteristic Strength
        ax = axes[2, 1]
        sigma_smooth = self.get_property('Characteristic_Strength_MPa', temp_smooth)
        ax.plot(temp_weibull, self.weibull_df['Characteristic_Strength_MPa'], 'o')
        ax.plot(temp_smooth, sigma_smooth, '-')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Characteristic Strength (MPa)')
        ax.grid(True, alpha=0.3)
        
        # Plot 9: Creep Exponent
        ax = axes[2, 2]
        n_smooth = self.get_property('Creep_Exponent_n', temp_smooth)
        ax.plot(temp, self.props_df['Creep_Exponent_n'], 'o')
        ax.plot(temp_smooth, n_smooth, '-')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Creep Exponent n')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
        
        return fig
    
    def export_for_fem(self, output_file, temperature_points=None):
        """
        Export material properties in FEM-friendly format.
        
        Parameters:
        -----------
        output_file : str
            Output file path
        temperature_points : array-like, optional
            Specific temperature points to export. If None, uses original data points.
        """
        if temperature_points is None:
            temperature_points = self.props_df['Temperature_C'].values
        
        # Create export dataframe
        export_data = {'Temperature_C': temperature_points}
        
        for prop_name in self.interpolators.keys():
            if prop_name not in ['Weibull_Modulus_m', 'Characteristic_Strength_MPa']:
                export_data[prop_name] = self.get_property(prop_name, temperature_points)
        
        export_df = pd.DataFrame(export_data)
        export_df.to_csv(output_file, index=False)
        print(f"✓ Exported FEM-ready data to {output_file}")
        
        return export_df
    
    def print_summary(self, temperature_c=25):
        """
        Print a summary of material properties at specified temperature.
        
        Parameters:
        -----------
        temperature_c : float
            Temperature in Celsius (default: 25°C)
        """
        print(f"\n{'='*70}")
        print(f"YSZ MATERIAL PROPERTIES AT {temperature_c}°C")
        print(f"{'='*70}")
        
        props = self.get_all_properties(temperature_c)
        
        print(f"\nMechanical Properties:")
        print(f"  Young's Modulus (E):        {props['Youngs_Modulus_GPa']:.2f} GPa")
        print(f"  Poisson's Ratio (ν):        {props['Poissons_Ratio']:.3f}")
        print(f"  Fracture Toughness (K_IC):  {props['Fracture_Toughness_MPa_m0.5']:.2f} MPa·m⁰·⁵")
        
        print(f"\nThermal Properties:")
        print(f"  CTE (α):                    {props['CTE_1e-6_K']:.2f} × 10⁻⁶ K⁻¹")
        print(f"  Thermal Conductivity (k):   {props['Thermal_Conductivity_W_mK']:.2f} W/m·K")
        print(f"  Density (ρ):                {props['Density_kg_m3']:.1f} kg/m³")
        
        print(f"\nStatistical Strength:")
        print(f"  Weibull Modulus (m):        {props['Weibull_Modulus_m']:.2f}")
        print(f"  Characteristic Strength:    {props['Characteristic_Strength_MPa']:.1f} MPa")
        
        print(f"\nCreep Parameters:")
        print(f"  Stress Exponent (n):        {props['Creep_Exponent_n']:.2f}")
        print(f"  Activation Energy (Q):      {props['Creep_Activation_Energy_kJ_mol']:.1f} kJ/mol")
        
        print(f"{'='*70}\n")


def example_usage():
    """Demonstrate usage of the material properties loader."""
    
    # Initialize the properties loader
    ysz = YSZMaterialProperties()
    
    # Print summary at room temperature
    ysz.print_summary(temperature_c=25)
    
    # Print summary at operating temperature
    ysz.print_summary(temperature_c=800)
    
    # Get specific property at a temperature
    E_600C = ysz.get_property('Youngs_Modulus_GPa', 600)
    print(f"Young's Modulus at 600°C: {E_600C:.2f} GPa\n")
    
    # Calculate creep rate
    creep_rate = ysz.get_creep_rate(
        stress_mpa=50, 
        temperature_c=1000, 
        grain_size_um=1.5
    )
    print(f"Creep rate at 1000°C, 50 MPa: {creep_rate:.3e} s⁻¹\n")
    
    # Plot all properties
    ysz.plot_properties(save_path='ysz_properties_plot.png')
    
    # Export for FEM
    ysz.export_for_fem('ysz_fem_input.csv')


if __name__ == "__main__":
    example_usage()