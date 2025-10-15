"""
Multi-Scale SOFC Parameter Definitions and Ranges
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

@dataclass
class ParameterRange:
    """Define parameter range with units and fidelity level"""
    name: str
    min_val: float
    max_val: float
    nominal: float
    unit: str
    fidelity: List[str]  # ['LF', 'MF', 'HF']
    description: str = ""

class SOFCParameters:
    """Complete parameter space for SOFC multi-fidelity modeling"""
    
    def __init__(self):
        self.parameters = self._initialize_parameters()
    
    def _initialize_parameters(self) -> Dict[str, Dict[str, ParameterRange]]:
        """Initialize all parameter ranges across scales"""
        
        params = {
            # System Level - Operating Conditions
            'system': {
                'fuel_utilization': ParameterRange(
                    'Fuel Utilization', 0.4, 0.9, 0.7, '-',
                    ['LF', 'MF', 'HF'],
                    'Fraction of fuel consumed'
                ),
                'oxidant_utilization': ParameterRange(
                    'Oxidant Utilization', 0.1, 0.4, 0.25, '-',
                    ['LF', 'MF', 'HF'],
                    'Fraction of oxidant consumed'
                ),
                'current_density': ParameterRange(
                    'Current Density', 0.0, 1.0, 0.5, 'A/cm²',
                    ['LF', 'MF', 'HF'],
                    'Operating current density'
                ),
                'operating_voltage': ParameterRange(
                    'Operating Voltage', 0.6, 0.9, 0.75, 'V',
                    ['LF', 'MF', 'HF'],
                    'Cell operating voltage'
                ),
                'temperature': ParameterRange(
                    'Operating Temperature', 923, 1123, 1023, 'K',
                    ['LF', 'MF', 'HF'],
                    'Cell operating temperature'
                ),
                'pressure': ParameterRange(
                    'Operating Pressure', 1.0, 5.0, 1.0, 'bar',
                    ['LF', 'MF', 'HF'],
                    'System pressure'
                ),
                'fuel_flow_rate': ParameterRange(
                    'Fuel Flow Rate', 0.5, 5.0, 2.0, 'L/min',
                    ['LF', 'MF', 'HF'],
                    'Fuel inlet flow rate'
                ),
                'air_flow_rate': ParameterRange(
                    'Air Flow Rate', 5.0, 50.0, 20.0, 'L/min',
                    ['LF', 'MF', 'HF'],
                    'Air inlet flow rate'
                ),
                'steam_carbon_ratio': ParameterRange(
                    'Steam/Carbon Ratio', 1.5, 3.0, 2.0, '-',
                    ['MF', 'HF'],
                    'For reforming reactions'
                ),
            },
            
            # Transient Operating Profiles
            'transient': {
                'startup_rate': ParameterRange(
                    'Startup Heating Rate', 1.0, 10.0, 5.0, 'K/min',
                    ['MF', 'HF'],
                    'Temperature ramp rate during startup'
                ),
                'shutdown_rate': ParameterRange(
                    'Shutdown Cooling Rate', 1.0, 10.0, 5.0, 'K/min',
                    ['MF', 'HF'],
                    'Temperature ramp rate during shutdown'
                ),
                'load_ramp_rate': ParameterRange(
                    'Load Change Rate', 0.01, 0.1, 0.05, 'A/cm²/min',
                    ['MF', 'HF'],
                    'Current density change rate'
                ),
                'thermal_cycles': ParameterRange(
                    'Number of Thermal Cycles', 0, 1000, 100, 'cycles',
                    ['LF', 'MF', 'HF'],
                    'Total thermal cycles experienced'
                ),
                'redox_cycles': ParameterRange(
                    'Number of Redox Cycles', 0, 100, 10, 'cycles',
                    ['MF', 'HF'],
                    'Total redox cycles experienced'
                ),
            },
            
            # Cell/Stack Geometry
            'geometry': {
                'active_area': ParameterRange(
                    'Cell Active Area', 10, 200, 100, 'cm²',
                    ['LF', 'MF', 'HF'],
                    'Active electrochemical area'
                ),
                'anode_thickness': ParameterRange(
                    'Anode Thickness', 200, 1000, 500, 'μm',
                    ['MF', 'HF'],
                    'Anode support layer thickness'
                ),
                'cathode_thickness': ParameterRange(
                    'Cathode Thickness', 20, 100, 50, 'μm',
                    ['MF', 'HF'],
                    'Cathode layer thickness'
                ),
                'electrolyte_thickness': ParameterRange(
                    'Electrolyte Thickness', 5, 20, 10, 'μm',
                    ['MF', 'HF'],
                    'Electrolyte layer thickness'
                ),
                'interconnect_thickness': ParameterRange(
                    'Interconnect Thickness', 2000, 4000, 3000, 'μm',
                    ['MF', 'HF'],
                    'Metallic interconnect thickness'
                ),
                'channel_width': ParameterRange(
                    'Flow Channel Width', 1, 5, 2, 'mm',
                    ['MF', 'HF'],
                    'Flow channel width'
                ),
                'channel_height': ParameterRange(
                    'Flow Channel Height', 0.5, 2, 1, 'mm',
                    ['MF', 'HF'],
                    'Flow channel height'
                ),
                'rib_width': ParameterRange(
                    'Rib Width', 1, 5, 2, 'mm',
                    ['MF', 'HF'],
                    'Interconnect rib width'
                ),
            },
            
            # Material Properties - Anode (Ni-YSZ)
            'anode_material': {
                'anode_porosity': ParameterRange(
                    'Anode Porosity', 0.25, 0.45, 0.35, '-',
                    ['MF', 'HF'],
                    'Volume fraction of pores'
                ),
                'anode_tortuosity': ParameterRange(
                    'Anode Tortuosity', 2.0, 6.0, 3.5, '-',
                    ['MF', 'HF'],
                    'Pore tortuosity factor'
                ),
                'ni_particle_size': ParameterRange(
                    'Ni Particle Size', 0.5, 3.0, 1.5, 'μm',
                    ['HF'],
                    'Initial Ni particle diameter'
                ),
                'tpb_density': ParameterRange(
                    'TPB Density', 1e6, 1e8, 1e7, 'μm/μm³',
                    ['HF'],
                    'Triple phase boundary density'
                ),
                'anode_ionic_conductivity': ParameterRange(
                    'Anode Ionic Conductivity', 0.1, 10, 2.0, 'S/cm',
                    ['MF', 'HF'],
                    'YSZ phase ionic conductivity at T'
                ),
                'anode_electronic_conductivity': ParameterRange(
                    'Anode Electronic Conductivity', 100, 10000, 1000, 'S/cm',
                    ['MF', 'HF'],
                    'Ni phase electronic conductivity'
                ),
                'ni_volume_fraction': ParameterRange(
                    'Ni Volume Fraction', 0.3, 0.5, 0.4, '-',
                    ['MF', 'HF'],
                    'Nickel phase volume fraction'
                ),
                'ysz_volume_fraction': ParameterRange(
                    'YSZ Volume Fraction', 0.25, 0.45, 0.35, '-',
                    ['MF', 'HF'],
                    'YSZ phase volume fraction'
                ),
            },
            
            # Material Properties - Cathode (LSCF/LSM-YSZ)
            'cathode_material': {
                'cathode_porosity': ParameterRange(
                    'Cathode Porosity', 0.30, 0.50, 0.40, '-',
                    ['MF', 'HF'],
                    'Volume fraction of pores'
                ),
                'cathode_tortuosity': ParameterRange(
                    'Cathode Tortuosity', 2.0, 6.0, 3.5, '-',
                    ['MF', 'HF'],
                    'Pore tortuosity factor'
                ),
                'cathode_particle_size': ParameterRange(
                    'Cathode Particle Size', 0.2, 2.0, 0.8, 'μm',
                    ['HF'],
                    'LSCF/LSM particle diameter'
                ),
                'cathode_ionic_conductivity': ParameterRange(
                    'Cathode Ionic Conductivity', 0.001, 0.1, 0.01, 'S/cm',
                    ['MF', 'HF'],
                    'Mixed ionic-electronic conductor'
                ),
                'cathode_electronic_conductivity': ParameterRange(
                    'Cathode Electronic Conductivity', 10, 1000, 100, 'S/cm',
                    ['MF', 'HF'],
                    'Electronic conductivity'
                ),
                'cathode_exchange_current': ParameterRange(
                    'Exchange Current Density', 0.01, 1.0, 0.1, 'A/cm²',
                    ['MF', 'HF'],
                    'Oxygen reduction reaction kinetics'
                ),
                'chemical_expansion_coeff': ParameterRange(
                    'Chemical Expansion Coefficient', 1e-3, 1e-2, 5e-3, '-',
                    ['HF'],
                    'Stoichiometry-induced expansion'
                ),
            },
            
            # Material Properties - Electrolyte (YSZ)
            'electrolyte_material': {
                'electrolyte_ionic_conductivity': ParameterRange(
                    'Electrolyte Ionic Conductivity', 0.01, 0.1, 0.05, 'S/cm',
                    ['LF', 'MF', 'HF'],
                    'YSZ ionic conductivity at T'
                ),
                'youngs_modulus': ParameterRange(
                    'Young\'s Modulus', 180, 220, 200, 'GPa',
                    ['MF', 'HF'],
                    'Elastic modulus of YSZ'
                ),
                'poissons_ratio': ParameterRange(
                    'Poisson\'s Ratio', 0.28, 0.32, 0.30, '-',
                    ['MF', 'HF'],
                    'Poisson\'s ratio of YSZ'
                ),
                'thermal_expansion_ysz': ParameterRange(
                    'YSZ CTE', 9.5e-6, 11.5e-6, 10.5e-6, '1/K',
                    ['MF', 'HF'],
                    'Coefficient of thermal expansion'
                ),
                'fracture_toughness': ParameterRange(
                    'Fracture Toughness', 1.5, 3.0, 2.2, 'MPa·m^0.5',
                    ['HF'],
                    'Critical stress intensity factor'
                ),
                'weibull_modulus': ParameterRange(
                    'Weibull Modulus', 5, 20, 10, '-',
                    ['HF'],
                    'Statistical strength distribution'
                ),
            },
            
            # Material Properties - Interconnect (Crofer 22 APU)
            'interconnect_material': {
                'thermal_expansion_ic': ParameterRange(
                    'Interconnect CTE', 10e-6, 13e-6, 11.8e-6, '1/K',
                    ['MF', 'HF'],
                    'Coefficient of thermal expansion'
                ),
                'oxide_growth_rate': ParameterRange(
                    'Oxide Scale Growth Rate', 1e-14, 1e-12, 1e-13, 'cm²/s',
                    ['HF'],
                    'Parabolic oxidation rate constant'
                ),
                'creep_activation_energy': ParameterRange(
                    'Creep Activation Energy', 250, 350, 300, 'kJ/mol',
                    ['HF'],
                    'Activation energy for creep'
                ),
                'creep_stress_exponent': ParameterRange(
                    'Creep Stress Exponent', 3, 7, 5, '-',
                    ['HF'],
                    'Power law creep exponent'
                ),
                'yield_strength': ParameterRange(
                    'Yield Strength', 200, 400, 300, 'MPa',
                    ['MF', 'HF'],
                    'Yield strength at temperature'
                ),
                'electrical_resistivity': ParameterRange(
                    'Electrical Resistivity', 0.8e-4, 1.5e-4, 1.0e-4, 'Ω·cm',
                    ['MF', 'HF'],
                    'Bulk electrical resistivity'
                ),
            },
            
            # Microstructural Properties (from imaging)
            'microstructure': {
                'phase_fraction_pore': ParameterRange(
                    'Pore Phase Fraction', 0.20, 0.50, 0.35, '-',
                    ['HF'],
                    'Volume fraction from 3D reconstruction'
                ),
                'phase_fraction_ni': ParameterRange(
                    'Ni Phase Fraction', 0.25, 0.45, 0.35, '-',
                    ['HF'],
                    'Ni volume fraction from imaging'
                ),
                'phase_fraction_ysz': ParameterRange(
                    'YSZ Phase Fraction', 0.25, 0.45, 0.30, '-',
                    ['HF'],
                    'YSZ volume fraction from imaging'
                ),
                'specific_surface_area': ParameterRange(
                    'Specific Surface Area', 1e6, 1e7, 3e6, '1/m',
                    ['HF'],
                    'Surface area per unit volume'
                ),
                'mean_pore_radius': ParameterRange(
                    'Mean Pore Radius', 0.1, 2.0, 0.5, 'μm',
                    ['HF'],
                    'Average pore radius'
                ),
                'connectivity_ni': ParameterRange(
                    'Ni Phase Connectivity', 0.85, 0.99, 0.95, '-',
                    ['HF'],
                    'Percolation probability'
                ),
                'connectivity_ysz': ParameterRange(
                    'YSZ Phase Connectivity', 0.85, 0.99, 0.95, '-',
                    ['HF'],
                    'Percolation probability'
                ),
                'connectivity_pore': ParameterRange(
                    'Pore Phase Connectivity', 0.90, 0.99, 0.97, '-',
                    ['HF'],
                    'Percolation probability'
                ),
            },
            
            # Degradation Parameters
            'degradation': {
                'ni_coarsening_rate': ParameterRange(
                    'Ni Coarsening Rate Constant', 1e-30, 1e-28, 1e-29, 'm³/s',
                    ['HF'],
                    'Ostwald ripening rate'
                ),
                'chromium_poisoning_rate': ParameterRange(
                    'Cr Poisoning Rate', 0, 1e-8, 1e-9, 'mol/m²/s',
                    ['HF'],
                    'Chromium deposition rate'
                ),
                'carbon_deposition_rate': ParameterRange(
                    'Carbon Deposition Rate', 0, 1e-7, 1e-8, 'mol/m²/s',
                    ['MF', 'HF'],
                    'Carbon formation rate'
                ),
                'sulfur_poisoning_level': ParameterRange(
                    'Sulfur Content', 0, 10, 1, 'ppm',
                    ['MF', 'HF'],
                    'Fuel sulfur contamination'
                ),
                'thermal_shock_resistance': ParameterRange(
                    'Thermal Shock Parameter', 100, 500, 200, 'K',
                    ['MF', 'HF'],
                    'Critical temperature difference'
                ),
            }
        }
        
        return params
    
    def get_parameter_vector(self, fidelity: str = 'HF') -> Dict[str, float]:
        """Get all parameters for a given fidelity level"""
        vector = {}
        for category, params in self.parameters.items():
            for param_name, param in params.items():
                if fidelity in param.fidelity:
                    vector[f"{category}.{param_name}"] = param.nominal
        return vector
    
    def get_parameter_bounds(self, fidelity: str = 'HF') -> Tuple[List[float], List[float]]:
        """Get min/max bounds for all parameters at given fidelity"""
        lower_bounds = []
        upper_bounds = []
        
        for category, params in self.parameters.items():
            for param_name, param in params.items():
                if fidelity in param.fidelity:
                    lower_bounds.append(param.min_val)
                    upper_bounds.append(param.max_val)
        
        return lower_bounds, upper_bounds
    
    def get_parameter_names(self, fidelity: str = 'HF') -> List[str]:
        """Get list of parameter names for given fidelity"""
        names = []
        for category, params in self.parameters.items():
            for param_name, param in params.items():
                if fidelity in param.fidelity:
                    names.append(f"{category}.{param_name}")
        return names
    
    def get_parameter_info(self, fidelity: str = 'HF') -> List[Dict[str, Any]]:
        """Get detailed information for all parameters"""
        info = []
        for category, params in self.parameters.items():
            for param_name, param in params.items():
                if fidelity in param.fidelity:
                    info.append({
                        'category': category,
                        'name': param_name,
                        'full_name': param.name,
                        'min': param.min_val,
                        'max': param.max_val,
                        'nominal': param.nominal,
                        'unit': param.unit,
                        'description': param.description,
                        'fidelity': param.fidelity
                    })
        return info