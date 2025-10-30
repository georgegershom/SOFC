from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass(frozen=True)
class ContinuousParam:
    name: str
    low: float
    high: float
    units: str
    description: str
    log_scale: bool = False


@dataclass(frozen=True)
class CategoricalParam:
    name: str
    choices: List[str]
    description: str


Param = Union[ContinuousParam, CategoricalParam]


def get_parameter_space() -> Dict[str, Param]:
    """Define the SOFC single-cell/stack parameter space.

    Ranges are broad and physically plausible; this is for synthetic data generation
    and does not replace detailed engineering specification.
    """
    params: Dict[str, Param] = {}

    # Operating conditions
    params["voltage_V"] = ContinuousParam(
        name="voltage_V", low=0.6, high=1.1, units="V", description="Cell voltage"
    )
    params["current_density_A_per_cm2"] = ContinuousParam(
        name="current_density_A_per_cm2",
        low=0.1,
        high=2.0,
        units="A/cm^2",
        description="Imposed current density (average)",
    )
    params["fuel_flow_rate_slpm"] = ContinuousParam(
        name="fuel_flow_rate_slpm",
        low=0.1,
        high=5.0,
        units="SLPM",
        description="Fuel (H2) flow rate at STP",
    )
    params["air_flow_rate_slpm"] = ContinuousParam(
        name="air_flow_rate_slpm",
        low=0.5,
        high=20.0,
        units="SLPM",
        description="Air flow rate at STP",
    )
    params["inlet_temp_fuel_C"] = ContinuousParam(
        name="inlet_temp_fuel_C",
        low=600.0,
        high=900.0,
        units="C",
        description="Fuel inlet temperature",
    )
    params["inlet_temp_air_C"] = ContinuousParam(
        name="inlet_temp_air_C",
        low=600.0,
        high=900.0,
        units="C",
        description="Air inlet temperature",
    )

    # Geometric parameters
    params["active_area_cm2"] = ContinuousParam(
        name="active_area_cm2",
        low=1.0,
        high=100.0,
        units="cm^2",
        description="Electrochemically active area",
    )
    params["flow_channel_design"] = CategoricalParam(
        name="flow_channel_design",
        choices=["straight", "serpentine", "interdigitated"],
        description="Flow field topology",
    )
    params["flow_channel_pitch_mm"] = ContinuousParam(
        name="flow_channel_pitch_mm",
        low=0.5,
        high=2.0,
        units="mm",
        description="Characteristic flow channel pitch",
    )

    # Layer thickness (microns)
    params["thickness_anode_um"] = ContinuousParam(
        name="thickness_anode_um", low=200.0, high=1000.0, units="um", description="Anode thickness"
    )
    params["thickness_electrolyte_um"] = ContinuousParam(
        name="thickness_electrolyte_um", low=5.0, high=50.0, units="um", description="Electrolyte thickness"
    )
    params["thickness_cathode_um"] = ContinuousParam(
        name="thickness_cathode_um", low=20.0, high=100.0, units="um", description="Cathode thickness"
    )
    params["thickness_interconnect_um"] = ContinuousParam(
        name="thickness_interconnect_um", low=200.0, high=1000.0, units="um", description="Interconnect thickness"
    )
    params["thickness_sealant_um"] = ContinuousParam(
        name="thickness_sealant_um", low=50.0, high=500.0, units="um", description="Sealant thickness"
    )

    # Material properties per layer
    # Porosity (electrolyte near-dense)
    params["anode_porosity"] = ContinuousParam("anode_porosity", 0.2, 0.5, "-", "Anode porosity")
    params["electrolyte_porosity"] = ContinuousParam(
        "electrolyte_porosity", 0.0, 0.1, "-", "Electrolyte porosity (nearly dense)"
    )
    params["cathode_porosity"] = ContinuousParam("cathode_porosity", 0.2, 0.5, "-", "Cathode porosity")
    params["interconnect_porosity"] = ContinuousParam(
        "interconnect_porosity", 0.0, 0.05, "-", "Interconnect porosity"
    )
    params["sealant_porosity"] = ContinuousParam("sealant_porosity", 0.0, 0.1, "-", "Sealant porosity")

    # Permeability (m^2) - log scale
    params["anode_permeability_m2"] = ContinuousParam(
        "anode_permeability_m2", 1e-15, 1e-12, "m^2", "Anode permeability", log_scale=True
    )
    params["cathode_permeability_m2"] = ContinuousParam(
        "cathode_permeability_m2", 1e-15, 1e-12, "m^2", "Cathode permeability", log_scale=True
    )
    params["electrolyte_permeability_m2"] = ContinuousParam(
        "electrolyte_permeability_m2", 1e-20, 1e-16, "m^2", "Electrolyte permeability", log_scale=True
    )

    # Conductivities (S/m)
    params["anode_electronic_sigma_S_m"] = ContinuousParam(
        "anode_electronic_sigma_S_m", 1e4, 1e6, "S/m", "Anode electronic conductivity", log_scale=True
    )
    params["cathode_electronic_sigma_S_m"] = ContinuousParam(
        "cathode_electronic_sigma_S_m", 1e4, 1e6, "S/m", "Cathode electronic conductivity", log_scale=True
    )
    params["interconnect_electronic_sigma_S_m"] = ContinuousParam(
        "interconnect_electronic_sigma_S_m", 1e5, 1e7, "S/m", "Interconnect electronic conductivity", log_scale=True
    )
    params["electrolyte_ionic_sigma_S_m"] = ContinuousParam(
        "electrolyte_ionic_sigma_S_m", 0.5, 20.0, "S/m", "Electrolyte ionic conductivity"
    )
    params["anode_ionic_sigma_S_m"] = ContinuousParam(
        "anode_ionic_sigma_S_m", 10.0, 1000.0, "S/m", "Anode ionic conductivity", log_scale=True
    )
    params["cathode_ionic_sigma_S_m"] = ContinuousParam(
        "cathode_ionic_sigma_S_m", 10.0, 1000.0, "S/m", "Cathode ionic conductivity", log_scale=True
    )

    # Elastic properties
    params["anode_E_GPa"] = ContinuousParam("anode_E_GPa", 50.0, 200.0, "GPa", "Anode Young's modulus")
    params["electrolyte_E_GPa"] = ContinuousParam("electrolyte_E_GPa", 150.0, 300.0, "GPa", "Electrolyte Young's modulus")
    params["cathode_E_GPa"] = ContinuousParam("cathode_E_GPa", 50.0, 200.0, "GPa", "Cathode Young's modulus")
    params["interconnect_E_GPa"] = ContinuousParam(
        "interconnect_E_GPa", 100.0, 250.0, "GPa", "Interconnect Young's modulus"
    )
    params["sealant_E_GPa"] = ContinuousParam("sealant_E_GPa", 5.0, 50.0, "GPa", "Sealant Young's modulus")

    # Thermal expansion (1e-6 / K)
    params["anode_cte_1e6_per_K"] = ContinuousParam("anode_cte_1e6_per_K", 8.0, 14.0, "1e-6/K", "Anode CTE")
    params["electrolyte_cte_1e6_per_K"] = ContinuousParam(
        "electrolyte_cte_1e6_per_K", 8.0, 12.0, "1e-6/K", "Electrolyte CTE"
    )
    params["cathode_cte_1e6_per_K"] = ContinuousParam("cathode_cte_1e6_per_K", 10.0, 16.0, "1e-6/K", "Cathode CTE")
    params["interconnect_cte_1e6_per_K"] = ContinuousParam(
        "interconnect_cte_1e6_per_K", 11.0, 17.0, "1e-6/K", "Interconnect CTE"
    )
    params["sealant_cte_1e6_per_K"] = ContinuousParam("sealant_cte_1e6_per_K", 7.0, 12.0, "1e-6/K", "Sealant CTE")

    return params
