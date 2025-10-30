from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalConstants:
    gas_constant_R: float = 8.314462618  # J/(mol*K)
    faraday_F: float = 96485.33212       # C/mol


@dataclass(frozen=True)
class DomainConfig:
    length_x: float = 1.0  # m
    length_y: float = 0.5  # m
    has_time: bool = True
    time_max: float = 10.0  # s


@dataclass(frozen=True)
class MaterialProperties:
    # Electrical
    electronic_conductivity_sigma_e: float = 1.0   # S/m
    ionic_conductivity_kappa_i: float = 0.8        # S/m

    # Thermal
    thermal_conductivity_k: float = 5.0            # W/(m*K)
    density_rho: float = 2000.0                    # kg/m^3
    heat_capacity_cp: float = 800.0                # J/(kg*K)

    # Mechanical (plane stress default)
    youngs_modulus_E: float = 5.0e9                # Pa
    poissons_ratio_nu: float = 0.30
    thermal_expansion_alpha_T: float = 1.5e-5      # 1/K
    reference_temperature_T_ref: float = 298.15    # K


@dataclass(frozen=True)
class ElectrochemKinetics:
    exchange_current_density_j0: float = 1.0       # A/m^2
    alpha_anodic: float = 0.5
    alpha_cathodic: float = 0.5
    open_circuit_potential_U_eq: float = 0.2       # V (simplified constant)


@dataclass(frozen=True)
class ConvectiveBoundary:
    h_coefficient: float = 10.0  # W/(m^2*K)
    ambient_temperature_T_inf: float = 298.15  # K


@dataclass(frozen=True)
class DefaultConfig:
    constants: PhysicalConstants = PhysicalConstants()
    domain: DomainConfig = DomainConfig()
    materials: MaterialProperties = MaterialProperties()
    kinetics: ElectrochemKinetics = ElectrochemKinetics()
    convection: ConvectiveBoundary = ConvectiveBoundary()
