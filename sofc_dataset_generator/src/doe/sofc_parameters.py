"""
SOFC Manufacturing Parameters Definition

Defines the parameter space for SOFC manufacturing variations that will be used
in the Design of Experiments to generate diverse synthetic datasets.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


class ParameterType(Enum):
    """Types of parameters for DOE"""
    GEOMETRIC = "geometric"
    MATERIAL = "material"
    PROCESS = "process"
    ENVIRONMENTAL = "environmental"


@dataclass
class ParameterRange:
    """Defines a parameter range with distribution type"""
    name: str
    min_value: float
    max_value: float
    distribution: str = "uniform"  # uniform, normal, lognormal
    mean: Optional[float] = None
    std: Optional[float] = None
    unit: str = ""
    description: str = ""
    
    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """Sample values from the parameter range"""
        if random_state is not None:
            np.random.seed(random_state)
        
        if self.distribution == "uniform":
            return np.random.uniform(self.min_value, self.max_value, n_samples)
        elif self.distribution == "normal":
            if self.mean is None:
                self.mean = (self.min_value + self.max_value) / 2
            if self.std is None:
                self.std = (self.max_value - self.min_value) / 6
            samples = np.random.normal(self.mean, self.std, n_samples)
            # Clip to bounds
            return np.clip(samples, self.min_value, self.max_value)
        elif self.distribution == "lognormal":
            if self.mean is None:
                self.mean = np.log((self.min_value + self.max_value) / 2)
            if self.std is None:
                self.std = 0.1
            samples = np.random.lognormal(self.mean, self.std, n_samples)
            # Clip to bounds
            return np.clip(samples, self.min_value, self.max_value)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


@dataclass
class SOFCParameters:
    """SOFC manufacturing parameters for DOE"""
    
    # Geometric parameters
    cell_length: ParameterRange = field(default_factory=lambda: ParameterRange(
        name="cell_length",
        min_value=80.0,
        max_value=120.0,
        distribution="normal",
        mean=100.0,
        std=5.0,
        unit="mm",
        description="SOFC cell length"
    ))
    
    cell_width: ParameterRange = field(default_factory=lambda: ParameterRange(
        name="cell_width",
        min_value=80.0,
        max_value=120.0,
        distribution="normal",
        mean=100.0,
        std=5.0,
        unit="mm",
        description="SOFC cell width"
    ))
    
    electrolyte_thickness: ParameterRange = field(default_factory=lambda: ParameterRange(
        name="electrolyte_thickness",
        min_value=100.0,
        max_value=200.0,
        distribution="normal",
        mean=150.0,
        std=15.0,
        unit="μm",
        description="8YSZ electrolyte thickness"
    ))
    
    anode_thickness: ParameterRange = field(default_factory=lambda: ParameterRange(
        name="anode_thickness",
        min_value=200.0,
        max_value=400.0,
        distribution="normal",
        mean=300.0,
        std=30.0,
        unit="μm",
        description="Ni-YSZ anode thickness"
    ))
    
    cathode_thickness: ParameterRange = field(default_factory=lambda: ParameterRange(
        name="cathode_thickness",
        min_value=30.0,
        max_value=70.0,
        distribution="normal",
        mean=50.0,
        std=8.0,
        unit="μm",
        description="LSM-YSZ cathode thickness"
    ))
    
    interconnect_thickness: ParameterRange = field(default_factory=lambda: ParameterRange(
        name="interconnect_thickness",
        min_value=1.5,
        max_value=2.5,
        distribution="normal",
        mean=2.0,
        std=0.15,
        unit="mm",
        description="Crofer 22 APU interconnect thickness"
    ))
    
    # Material property variations (as percentage of nominal)
    electrolyte_youngs_modulus_variation: ParameterRange = field(default_factory=lambda: ParameterRange(
        name="electrolyte_youngs_modulus_variation",
        min_value=-10.0,
        max_value=10.0,
        distribution="normal",
        mean=0.0,
        std=3.0,
        unit="%",
        description="8YSZ Young's modulus variation from nominal"
    ))
    
    electrolyte_cte_variation: ParameterRange = field(default_factory=lambda: ParameterRange(
        name="electrolyte_cte_variation",
        min_value=-5.0,
        max_value=5.0,
        distribution="normal",
        mean=0.0,
        std=1.5,
        unit="%",
        description="8YSZ CTE variation from nominal"
    ))
    
    anode_youngs_modulus_variation: ParameterRange = field(default_factory=lambda: ParameterRange(
        name="anode_youngs_modulus_variation",
        min_value=-15.0,
        max_value=15.0,
        distribution="normal",
        mean=0.0,
        std=5.0,
        unit="%",
        description="Ni-YSZ Young's modulus variation from nominal"
    ))
    
    anode_cte_variation: ParameterRange = field(default_factory=lambda: ParameterRange(
        name="anode_cte_variation",
        min_value=-8.0,
        max_value=8.0,
        distribution="normal",
        mean=0.0,
        std=2.5,
        unit="%",
        description="Ni-YSZ CTE variation from nominal"
    ))
    
    cathode_youngs_modulus_variation: ParameterRange = field(default_factory=lambda: ParameterRange(
        name="cathode_youngs_modulus_variation",
        min_value=-12.0,
        max_value=12.0,
        distribution="normal",
        mean=0.0,
        std=4.0,
        unit="%",
        description="LSM Young's modulus variation from nominal"
    ))
    
    cathode_cte_variation: ParameterRange = field(default_factory=lambda: ParameterRange(
        name="cathode_cte_variation",
        min_value=-6.0,
        max_value=6.0,
        distribution="normal",
        mean=0.0,
        std=2.0,
        unit="%",
        description="LSM CTE variation from nominal"
    ))
    
    # Process parameters
    sintering_temperature: ParameterRange = field(default_factory=lambda: ParameterRange(
        name="sintering_temperature",
        min_value=1300.0,
        max_value=1400.0,
        distribution="normal",
        mean=1350.0,
        std=15.0,
        unit="°C",
        description="Co-sintering temperature"
    ))
    
    cooling_rate: ParameterRange = field(default_factory=lambda: ParameterRange(
        name="cooling_rate",
        min_value=1.0,
        max_value=5.0,
        distribution="lognormal",
        mean=np.log(2.0),
        std=0.3,
        unit="°C/min",
        description="Cooling rate from sintering temperature"
    ))
    
    assembly_pressure: ParameterRange = field(default_factory=lambda: ParameterRange(
        name="assembly_pressure",
        min_value=0.1,
        max_value=0.4,
        distribution="normal",
        mean=0.25,
        std=0.05,
        unit="MPa",
        description="Stack assembly pressure"
    ))
    
    # Environmental parameters
    operating_temperature: ParameterRange = field(default_factory=lambda: ParameterRange(
        name="operating_temperature",
        min_value=750.0,
        max_value=850.0,
        distribution="normal",
        mean=800.0,
        std=15.0,
        unit="°C",
        description="SOFC operating temperature"
    ))
    
    thermal_gradient: ParameterRange = field(default_factory=lambda: ParameterRange(
        name="thermal_gradient",
        min_value=10.0,
        max_value=100.0,
        distribution="lognormal",
        mean=np.log(50.0),
        std=0.4,
        unit="°C",
        description="Maximum thermal gradient across cell"
    ))
    
    # Creep parameters (for viscoelastic model)
    creep_activation_energy_variation: ParameterRange = field(default_factory=lambda: ParameterRange(
        name="creep_activation_energy_variation",
        min_value=-10.0,
        max_value=10.0,
        distribution="normal",
        mean=0.0,
        std=3.0,
        unit="%",
        description="8YSZ creep activation energy variation from nominal"
    ))
    
    creep_stress_exponent_variation: ParameterRange = field(default_factory=lambda: ParameterRange(
        name="creep_stress_exponent_variation",
        min_value=-15.0,
        max_value=15.0,
        distribution="normal",
        mean=0.0,
        std=5.0,
        unit="%",
        description="8YSZ creep stress exponent variation from nominal"
    ))
    
    def get_all_parameters(self) -> List[ParameterRange]:
        """Get all parameter ranges as a list"""
        return [
            self.cell_length,
            self.cell_width,
            self.electrolyte_thickness,
            self.anode_thickness,
            self.cathode_thickness,
            self.interconnect_thickness,
            self.electrolyte_youngs_modulus_variation,
            self.electrolyte_cte_variation,
            self.anode_youngs_modulus_variation,
            self.anode_cte_variation,
            self.cathode_youngs_modulus_variation,
            self.cathode_cte_variation,
            self.sintering_temperature,
            self.cooling_rate,
            self.assembly_pressure,
            self.operating_temperature,
            self.thermal_gradient,
            self.creep_activation_energy_variation,
            self.creep_stress_exponent_variation
        ]
    
    def get_parameter_names(self) -> List[str]:
        """Get all parameter names"""
        return [param.name for param in self.get_all_parameters()]
    
    def get_geometric_parameters(self) -> List[ParameterRange]:
        """Get geometric parameters only"""
        return [
            self.cell_length,
            self.cell_width,
            self.electrolyte_thickness,
            self.anode_thickness,
            self.cathode_thickness,
            self.interconnect_thickness
        ]
    
    def get_material_parameters(self) -> List[ParameterRange]:
        """Get material property variation parameters"""
        return [
            self.electrolyte_youngs_modulus_variation,
            self.electrolyte_cte_variation,
            self.anode_youngs_modulus_variation,
            self.anode_cte_variation,
            self.cathode_youngs_modulus_variation,
            self.cathode_cte_variation,
            self.creep_activation_energy_variation,
            self.creep_stress_exponent_variation
        ]
    
    def get_process_parameters(self) -> List[ParameterRange]:
        """Get manufacturing process parameters"""
        return [
            self.sintering_temperature,
            self.cooling_rate,
            self.assembly_pressure
        ]
    
    def get_environmental_parameters(self) -> List[ParameterRange]:
        """Get environmental/operational parameters"""
        return [
            self.operating_temperature,
            self.thermal_gradient
        ]


@dataclass
class ManufacturingParameters:
    """Container for a single set of manufacturing parameters"""
    values: Dict[str, float]
    parameter_ranges: SOFCParameters
    
    def __post_init__(self):
        """Validate parameter values against ranges"""
        for param_name, value in self.values.items():
            param_range = getattr(self.parameter_ranges, param_name)
            if not (param_range.min_value <= value <= param_range.max_value):
                print(f"Warning: {param_name} = {value} is outside range "
                      f"[{param_range.min_value}, {param_range.max_value}]")
    
    def get_geometric_values(self) -> Dict[str, float]:
        """Get geometric parameter values"""
        geometric_params = self.parameter_ranges.get_geometric_parameters()
        return {param.name: self.values[param.name] for param in geometric_params}
    
    def get_material_values(self) -> Dict[str, float]:
        """Get material parameter values"""
        material_params = self.parameter_ranges.get_material_parameters()
        return {param.name: self.values[param.name] for param in material_params}
    
    def get_process_values(self) -> Dict[str, float]:
        """Get process parameter values"""
        process_params = self.parameter_ranges.get_process_parameters()
        return {param.name: self.values[param.name] for param in process_params}
    
    def get_environmental_values(self) -> Dict[str, float]:
        """Get environmental parameter values"""
        env_params = self.parameter_ranges.get_environmental_parameters()
        return {param.name: self.values[param.name] for param in env_params}
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return self.values.copy()
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array in parameter order"""
        param_names = self.parameter_ranges.get_parameter_names()
        return np.array([self.values[name] for name in param_names])


if __name__ == "__main__":
    # Example usage
    sofc_params = SOFCParameters()
    
    print("SOFC Manufacturing Parameters:")
    print("=" * 50)
    
    for param in sofc_params.get_all_parameters():
        print(f"{param.name:35s}: {param.min_value:8.2f} - {param.max_value:8.2f} {param.unit:8s} ({param.distribution})")
    
    print(f"\nTotal parameters: {len(sofc_params.get_all_parameters())}")
    print(f"Geometric parameters: {len(sofc_params.get_geometric_parameters())}")
    print(f"Material parameters: {len(sofc_params.get_material_parameters())}")
    print(f"Process parameters: {len(sofc_params.get_process_parameters())}")
    print(f"Environmental parameters: {len(sofc_params.get_environmental_parameters())}")