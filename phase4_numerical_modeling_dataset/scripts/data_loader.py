"""
Data Loader for Phase 4 Numerical Modeling Dataset
Loads and processes experimental data for thermo-mechanical modeling
of fire-resistant rubberized concrete
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

class RubberizedConcreteDataLoader:
    """
    Comprehensive data loader for rubberized concrete fire testing data
    """
    
    def __init__(self, base_path: str = "../"):
        """
        Initialize data loader
        
        Args:
            base_path: Path to the dataset root directory
        """
        self.base_path = Path(base_path)
        self.model_input_path = self.base_path / "model_input_data"
        self.validation_path = self.base_path / "model_validation_data"
        
        # Data containers
        self.thermal_data = {}
        self.mechanical_data = {}
        self.deformation_data = {}
        self.poro_mechanical_data = {}
        self.validation_data = {}
        
    def load_all_data(self) -> Dict:
        """
        Load all datasets
        
        Returns:
            Dictionary containing all loaded data
        """
        print("Loading Phase 4 Numerical Modeling Dataset...")
        
        # Load model input data
        self.load_thermal_properties()
        self.load_mechanical_properties()
        self.load_deformation_properties()
        self.load_poro_mechanical_properties()
        
        # Load validation data
        self.load_validation_data()
        
        print("Dataset loading complete!")
        
        return {
            'thermal': self.thermal_data,
            'mechanical': self.mechanical_data,
            'deformation': self.deformation_data,
            'poro_mechanical': self.poro_mechanical_data,
            'validation': self.validation_data
        }
    
    def load_thermal_properties(self):
        """Load thermal properties data"""
        print("  Loading thermal properties...")
        
        thermal_path = self.model_input_path / "thermal_properties"
        
        self.thermal_data['conductivity'] = pd.read_csv(
            thermal_path / "thermal_conductivity.csv"
        )
        self.thermal_data['specific_heat'] = pd.read_csv(
            thermal_path / "specific_heat_capacity.csv"
        )
        self.thermal_data['density'] = pd.read_csv(
            thermal_path / "density.csv"
        )
        
        print(f"    Loaded {len(self.thermal_data)} thermal property datasets")
        
    def load_mechanical_properties(self):
        """Load mechanical properties data"""
        print("  Loading mechanical properties...")
        
        mech_path = self.model_input_path / "mechanical_properties"
        
        self.mechanical_data['compressive_strength'] = pd.read_csv(
            mech_path / "compressive_strength.csv"
        )
        self.mechanical_data['tensile_strength'] = pd.read_csv(
            mech_path / "tensile_strength.csv"
        )
        self.mechanical_data['elastic_modulus'] = pd.read_csv(
            mech_path / "elastic_modulus.csv"
        )
        self.mechanical_data['poissons_ratio'] = pd.read_csv(
            mech_path / "poissons_ratio.csv"
        )
        
        print(f"    Loaded {len(self.mechanical_data)} mechanical property datasets")
        
    def load_deformation_properties(self):
        """Load deformation properties data"""
        print("  Loading deformation properties...")
        
        deform_path = self.model_input_path / "deformation_properties"
        
        self.deformation_data['cte'] = pd.read_csv(
            deform_path / "coefficient_thermal_expansion.csv"
        )
        self.deformation_data['transient_thermal_strain'] = pd.read_csv(
            deform_path / "transient_thermal_strain.csv"
        )
        
        print(f"    Loaded {len(self.deformation_data)} deformation property datasets")
        
    def load_poro_mechanical_properties(self):
        """Load poro-mechanical properties data"""
        print("  Loading poro-mechanical properties...")
        
        poro_path = self.model_input_path / "poro_mechanical_properties"
        
        self.poro_mechanical_data['permeability'] = pd.read_csv(
            poro_path / "permeability.csv"
        )
        self.poro_mechanical_data['porosity'] = pd.read_csv(
            poro_path / "porosity.csv"
        )
        
        print(f"    Loaded {len(self.poro_mechanical_data)} poro-mechanical property datasets")
        
    def load_validation_data(self):
        """Load validation data"""
        print("  Loading validation data...")
        
        # Temperature profiles
        temp_path = self.validation_path / "temperature_profiles"
        self.validation_data['iso834_temps'] = pd.read_csv(
            temp_path / "ISO834_fire_test_thermocouple_data.csv"
        )
        self.validation_data['astm_e119_temps'] = pd.read_csv(
            temp_path / "ASTM_E119_fire_test_thermocouple_data.csv"
        )
        self.validation_data['hydrocarbon_temps'] = pd.read_csv(
            temp_path / "hydrocarbon_fire_test_thermocouple_data.csv"
        )
        
        # Strain histories
        strain_path = self.validation_path / "strain_histories"
        self.validation_data['axial_strain'] = pd.read_csv(
            strain_path / "axial_strain_under_load_ISO834.csv"
        )
        self.validation_data['radial_deformation'] = pd.read_csv(
            strain_path / "radial_deformation_under_thermal_load.csv"
        )
        
        # Spalling data
        spall_path = self.validation_path / "spalling_data"
        self.validation_data['spalling_iso834'] = pd.read_csv(
            spall_path / "spalling_observations_ISO834.csv"
        )
        self.validation_data['spalling_astm'] = pd.read_csv(
            spall_path / "spalling_observations_ASTM_E119.csv"
        )
        self.validation_data['spalling_hydrocarbon'] = pd.read_csv(
            spall_path / "spalling_observations_hydrocarbon.csv"
        )
        
        print(f"    Loaded {len(self.validation_data)} validation datasets")
    
    def get_material_properties_at_temperature(
        self, 
        rubber_content: float, 
        temperature: float
    ) -> Dict:
        """
        Get interpolated material properties at specific temperature and rubber content
        
        Args:
            rubber_content: Rubber replacement percentage (0, 10, 20, 30)
            temperature: Temperature in Celsius
            
        Returns:
            Dictionary of material properties
        """
        properties = {}
        
        # Filter data for specific rubber content
        thermal_cond = self.thermal_data['conductivity']
        thermal_cond_filtered = thermal_cond[
            thermal_cond['Rubber_Content_Percent'] == rubber_content
        ]
        
        # Interpolate thermal conductivity
        properties['thermal_conductivity'] = np.interp(
            temperature,
            thermal_cond_filtered['Temperature_C'].values,
            thermal_cond_filtered['Thermal_Conductivity_W_mK'].values
        )
        
        # Similar interpolation for other properties
        # Add more properties as needed
        
        return properties
    
    def get_validation_data_for_model(
        self,
        fire_curve: str = 'ISO834',
        rubber_content: float = 0
    ) -> Dict:
        """
        Get validation data for a specific fire curve and rubber content
        
        Args:
            fire_curve: 'ISO834', 'ASTM_E119', or 'Hydrocarbon'
            rubber_content: Rubber replacement percentage
            
        Returns:
            Dictionary with temperature and strain validation data
        """
        validation = {}
        
        # Get temperature data
        if fire_curve == 'ISO834':
            temp_data = self.validation_data['iso834_temps']
        elif fire_curve == 'ASTM_E119':
            temp_data = self.validation_data['astm_e119_temps']
        else:
            temp_data = self.validation_data['hydrocarbon_temps']
            
        validation['temperature'] = temp_data[
            temp_data['Rubber_Content_Percent'] == rubber_content
        ]
        
        # Get strain data
        strain_data = self.validation_data['axial_strain']
        validation['strain'] = strain_data[
            strain_data['Rubber_Content_Percent'] == rubber_content
        ]
        
        return validation


def example_usage():
    """Example of how to use the data loader"""
    
    # Initialize loader
    loader = RubberizedConcreteDataLoader()
    
    # Load all data
    all_data = loader.load_all_data()
    
    # Get properties at specific conditions
    props = loader.get_material_properties_at_temperature(
        rubber_content=20,
        temperature=500
    )
    print(f"\nMaterial properties at 500°C with 20% rubber:")
    print(f"  Thermal conductivity: {props['thermal_conductivity']:.4f} W/mK")
    
    # Get validation data
    validation = loader.get_validation_data_for_model(
        fire_curve='ISO834',
        rubber_content=10
    )
    print(f"\nValidation data loaded for ISO834 with 10% rubber")
    print(f"  Temperature data points: {len(validation['temperature'])}")
    print(f"  Strain data points: {len(validation['strain'])}")


if __name__ == "__main__":
    example_usage()
