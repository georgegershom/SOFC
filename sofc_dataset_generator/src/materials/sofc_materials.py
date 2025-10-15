"""
SOFC Material Properties Definition

Based on the research article data and literature values for realistic SOFC materials.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Callable, Tuple
import json


@dataclass
class MaterialProperty:
    """Container for temperature-dependent material properties"""
    name: str
    unit: str
    room_temp_value: float
    high_temp_value: float
    temperature_function: Callable[[float], float]
    
    def evaluate(self, temperature: float) -> float:
        """Evaluate property at given temperature"""
        return self.temperature_function(temperature)


class SOFCMaterials:
    """SOFC material properties with temperature dependence"""
    
    def __init__(self):
        self.materials = self._initialize_materials()
    
    def _initialize_materials(self) -> Dict[str, Dict[str, MaterialProperty]]:
        """Initialize all SOFC material properties"""
        
        materials = {}
        
        # 8YSZ Electrolyte Properties
        materials['8YSZ'] = {
            'youngs_modulus': MaterialProperty(
                name='Young\'s Modulus',
                unit='GPa',
                room_temp_value=200.0,
                high_temp_value=170.0,
                temperature_function=lambda T: 200.0 - 0.0375 * (T - 25.0)
            ),
            'poisson_ratio': MaterialProperty(
                name='Poisson\'s Ratio',
                unit='-',
                room_temp_value=0.23,
                high_temp_value=0.23,
                temperature_function=lambda T: 0.23
            ),
            'cte': MaterialProperty(
                name='Coefficient of Thermal Expansion',
                unit='1/K',
                room_temp_value=10.0e-6,
                high_temp_value=10.5e-6,
                temperature_function=lambda T: (10.0 + 0.5 * (T - 25.0) / 775.0) * 1e-6
            ),
            'thermal_conductivity': MaterialProperty(
                name='Thermal Conductivity',
                unit='W/m·K',
                room_temp_value=2.1,
                high_temp_value=2.3,
                temperature_function=lambda T: 2.1 + 0.2 * (T - 25.0) / 775.0
            ),
            'specific_heat': MaterialProperty(
                name='Specific Heat',
                unit='J/kg·K',
                room_temp_value=450.0,
                high_temp_value=600.0,
                temperature_function=lambda T: 450.0 + 150.0 * (T - 25.0) / 775.0
            ),
            'density': MaterialProperty(
                name='Density',
                unit='kg/m³',
                room_temp_value=5900.0,
                high_temp_value=5900.0,
                temperature_function=lambda T: 5900.0
            ),
            'flexural_strength': MaterialProperty(
                name='Flexural Strength',
                unit='MPa',
                room_temp_value=165.0,
                high_temp_value=150.0,
                temperature_function=lambda T: 165.0 - 15.0 * (T - 25.0) / 775.0
            )
        }
        
        # Ni-YSZ Anode Properties
        materials['NiYSZ'] = {
            'youngs_modulus': MaterialProperty(
                name='Young\'s Modulus',
                unit='GPa',
                room_temp_value=55.0,
                high_temp_value=29.0,
                temperature_function=lambda T: 55.0 * np.exp(-0.0012 * (T - 25.0))
            ),
            'poisson_ratio': MaterialProperty(
                name='Poisson\'s Ratio',
                unit='-',
                room_temp_value=0.29,
                high_temp_value=0.29,
                temperature_function=lambda T: 0.29
            ),
            'cte': MaterialProperty(
                name='Coefficient of Thermal Expansion',
                unit='1/K',
                room_temp_value=12.5e-6,
                high_temp_value=13.3e-6,
                temperature_function=lambda T: (12.5 + 0.8 * (T - 25.0) / 775.0) * 1e-6
            ),
            'thermal_conductivity': MaterialProperty(
                name='Thermal Conductivity',
                unit='W/m·K',
                room_temp_value=6.0,
                high_temp_value=4.5,
                temperature_function=lambda T: 6.0 - 1.5 * (T - 25.0) / 775.0
            ),
            'specific_heat': MaterialProperty(
                name='Specific Heat',
                unit='J/kg·K',
                room_temp_value=480.0,
                high_temp_value=520.0,
                temperature_function=lambda T: 480.0 + 40.0 * (T - 25.0) / 775.0
            ),
            'density': MaterialProperty(
                name='Density',
                unit='kg/m³',
                room_temp_value=6500.0,
                high_temp_value=6500.0,
                temperature_function=lambda T: 6500.0
            )
        }
        
        # LSM-YSZ Cathode Properties
        materials['LSM'] = {
            'youngs_modulus': MaterialProperty(
                name='Young\'s Modulus',
                unit='GPa',
                room_temp_value=45.0,
                high_temp_value=40.0,
                temperature_function=lambda T: 45.0 - 5.0 * (T - 25.0) / 775.0
            ),
            'poisson_ratio': MaterialProperty(
                name='Poisson\'s Ratio',
                unit='-',
                room_temp_value=0.25,
                high_temp_value=0.25,
                temperature_function=lambda T: 0.25
            ),
            'cte': MaterialProperty(
                name='Coefficient of Thermal Expansion',
                unit='1/K',
                room_temp_value=11.5e-6,
                high_temp_value=12.0e-6,
                temperature_function=lambda T: (11.5 + 0.5 * (T - 25.0) / 775.0) * 1e-6
            ),
            'thermal_conductivity': MaterialProperty(
                name='Thermal Conductivity',
                unit='W/m·K',
                room_temp_value=3.5,
                high_temp_value=2.8,
                temperature_function=lambda T: 3.5 - 0.7 * (T - 25.0) / 775.0
            ),
            'specific_heat': MaterialProperty(
                name='Specific Heat',
                unit='J/kg·K',
                room_temp_value=500.0,
                high_temp_value=550.0,
                temperature_function=lambda T: 500.0 + 50.0 * (T - 25.0) / 775.0
            ),
            'density': MaterialProperty(
                name='Density',
                unit='kg/m³',
                room_temp_value=6200.0,
                high_temp_value=6200.0,
                temperature_function=lambda T: 6200.0
            )
        }
        
        # Crofer 22 APU Interconnect Properties
        materials['Crofer22APU'] = {
            'youngs_modulus': MaterialProperty(
                name='Young\'s Modulus',
                unit='GPa',
                room_temp_value=160.0,
                high_temp_value=140.0,
                temperature_function=lambda T: 160.0 - 20.0 * (T - 25.0) / 775.0
            ),
            'poisson_ratio': MaterialProperty(
                name='Poisson\'s Ratio',
                unit='-',
                room_temp_value=0.30,
                high_temp_value=0.30,
                temperature_function=lambda T: 0.30
            ),
            'cte': MaterialProperty(
                name='Coefficient of Thermal Expansion',
                unit='1/K',
                room_temp_value=11.5e-6,
                high_temp_value=11.9e-6,
                temperature_function=lambda T: (11.5 + 0.4 * (T - 25.0) / 775.0) * 1e-6
            ),
            'thermal_conductivity': MaterialProperty(
                name='Thermal Conductivity',
                unit='W/m·K',
                room_temp_value=25.0,
                high_temp_value=22.0,
                temperature_function=lambda T: 25.0 - 3.0 * (T - 25.0) / 775.0
            ),
            'specific_heat': MaterialProperty(
                name='Specific Heat',
                unit='J/kg·K',
                room_temp_value=500.0,
                high_temp_value=600.0,
                temperature_function=lambda T: 500.0 + 100.0 * (T - 25.0) / 775.0
            ),
            'density': MaterialProperty(
                name='Density',
                unit='kg/m³',
                room_temp_value=7800.0,
                high_temp_value=7800.0,
                temperature_function=lambda T: 7800.0
            ),
            'yield_strength': MaterialProperty(
                name='Yield Strength',
                unit='MPa',
                room_temp_value=250.0,
                high_temp_value=150.0,
                temperature_function=lambda T: 250.0 - 100.0 * (T - 25.0) / 775.0
            )
        }
        
        return materials
    
    def get_property(self, material: str, property_name: str, temperature: float) -> float:
        """Get material property at specific temperature"""
        if material not in self.materials:
            raise ValueError(f"Material {material} not found")
        if property_name not in self.materials[material]:
            raise ValueError(f"Property {property_name} not found for material {material}")
        
        return self.materials[material][property_name].evaluate(temperature)
    
    def get_all_properties(self, material: str, temperature: float) -> Dict[str, float]:
        """Get all properties for a material at specific temperature"""
        if material not in self.materials:
            raise ValueError(f"Material {material} not found")
        
        return {prop_name: prop.evaluate(temperature) 
                for prop_name, prop in self.materials[material].items()}
    
    def get_elastic_properties(self, material: str, temperature: float) -> Tuple[float, float]:
        """Get elastic properties (E, nu) for FEA"""
        E = self.get_property(material, 'youngs_modulus', temperature)
        nu = self.get_property(material, 'poisson_ratio', temperature)
        return E, nu
    
    def get_thermal_properties(self, material: str, temperature: float) -> Tuple[float, float, float]:
        """Get thermal properties (k, cp, rho) for thermal analysis"""
        k = self.get_property(material, 'thermal_conductivity', temperature)
        cp = self.get_property(material, 'specific_heat', temperature)
        rho = self.get_property(material, 'density', temperature)
        return k, cp, rho
    
    def get_thermal_expansion(self, material: str, temperature: float) -> float:
        """Get coefficient of thermal expansion"""
        return self.get_property(material, 'cte', temperature)
    
    def export_material_data(self, filename: str, temperature_range: Tuple[float, float, int] = (25, 800, 100)):
        """Export material data to JSON file for external use"""
        T_min, T_max, n_points = temperature_range
        temperatures = np.linspace(T_min, T_max, n_points)
        
        export_data = {}
        for material_name, properties in self.materials.items():
            export_data[material_name] = {}
            for prop_name, prop in properties.items():
                export_data[material_name][prop_name] = {
                    'unit': prop.unit,
                    'values': [prop.evaluate(T) for T in temperatures],
                    'temperatures': temperatures.tolist()
                }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Material data exported to {filename}")


if __name__ == "__main__":
    # Example usage
    materials = SOFCMaterials()
    
    # Test at room temperature
    print("8YSZ at 25°C:")
    props = materials.get_all_properties('8YSZ', 25.0)
    for name, value in props.items():
        print(f"  {name}: {value:.3f}")
    
    # Test at operating temperature
    print("\n8YSZ at 800°C:")
    props = materials.get_all_properties('8YSZ', 800.0)
    for name, value in props.items():
        print(f"  {name}: {value:.3f}")
    
    # Export data
    materials.export_material_data('sofc_materials.json')