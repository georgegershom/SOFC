from dataclasses import dataclass

KELVIN_OFFSET = 273.15
ABSOLUTE_ZERO_C = -273.15

@dataclass(frozen=True)
class Units:
    temperature_c: str = "°C"
    temperature_k: str = "K"
    density: str = "kg/m^3"
    conductivity: str = "W/m/K"
    specific_heat: str = "J/kg/K"
    elastic_modulus: str = "Pa"
    poisson_ratio: str = "-"
    thermal_expansion: str = "1/K"
    permeability: str = "m^2"
    biot: str = "-"
    compressive_strength: str = "Pa"

UNITS = Units()
