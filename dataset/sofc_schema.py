"""
SOFC multi-scale parameter schema with ranges, units, types, and fidelity associations.
This module centralizes the parameter dictionary used by the dataset generator.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Any


@dataclass
class Parameter:
    key: str
    name: str
    category: str
    unit: Optional[str]
    description: str
    ptype: str  # 'continuous' | 'integer' | 'categorical'
    min: Optional[float] = None
    max: Optional[float] = None
    scale: str = "linear"  # 'linear' | 'log'
    levels: Optional[List[Union[float, str]]] = None  # for factorial or categorical
    fidelity: List[str] = None  # subset of {LF, MF, HF}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Helper for concise construction

def P(
    key: str,
    name: str,
    category: str,
    unit: Optional[str],
    description: str,
    ptype: str,
    min: Optional[float] = None,
    max: Optional[float] = None,
    scale: str = "linear",
    levels: Optional[List[Union[float, str]]] = None,
    fidelity: Optional[List[str]] = None,
) -> Parameter:
    return Parameter(
        key=key,
        name=name,
        category=category,
        unit=unit,
        description=description,
        ptype=ptype,
        min=min,
        max=max,
        scale=scale,
        levels=levels,
        fidelity=fidelity or ["LF", "MF", "HF"],
    )


def get_schema() -> Dict[str, Any]:
    """Returns a schema dictionary with parameters grouped by category.

    Notes on ranges (typical SOFC context, planar, anode-supported):
    - Temperature in degC (650-850)
    - Pressure in bar (1-5)
    - Current density in A/cm^2 (0.1-1.5)
    - Voltage in V (0.6-0.9 open-circuit lower under load)
    - Flow rates in sccm for lab-scale single cells
    - Thickness in micrometers (µm)
    - Channel dimensions in millimeters (mm)
    - Conductivities in S/cm (effective, temperature-dependent in reality)
    - CTE in 1/K (x1e-6 range typical)
    """

    params: List[Parameter] = []

    # System - Operating Conditions (LF, MF, HF)
    params += [
        P(
            key="system.fuel_utilization",
            name="Fuel Utilization",
            category="System/Operating",
            unit="fraction",
            description="Fraction of fuel consumed (anode side).",
            ptype="continuous",
            min=0.60,
            max=0.90,
            fidelity=["LF", "MF", "HF"],
        ),
        P(
            key="system.oxidant_utilization",
            name="Oxidant Utilization",
            category="System/Operating",
            unit="fraction",
            description="Fraction of oxygen consumed (cathode side).",
            ptype="continuous",
            min=0.15,
            max=0.50,
            fidelity=["LF", "MF", "HF"],
        ),
        P(
            key="system.current_density",
            name="Current Density",
            category="System/Operating",
            unit="A/cm^2",
            description="Applied current density.",
            ptype="continuous",
            min=0.10,
            max=1.50,
            fidelity=["LF", "MF", "HF"],
        ),
        P(
            key="system.cell_voltage",
            name="Cell Voltage",
            category="System/Operating",
            unit="V",
            description="Cell operating voltage.",
            ptype="continuous",
            min=0.60,
            max=0.90,
            fidelity=["LF", "MF", "HF"],
        ),
        P(
            key="system.temperature",
            name="Temperature",
            category="System/Operating",
            unit="degC",
            description="Operating temperature at active area.",
            ptype="continuous",
            min=650.0,
            max=850.0,
            fidelity=["LF", "MF", "HF"],
        ),
        P(
            key="system.pressure",
            name="Pressure",
            category="System/Operating",
            unit="bar",
            description="Operating total pressure.",
            ptype="continuous",
            min=1.0,
            max=5.0,
            fidelity=["LF", "MF", "HF"],
        ),
        P(
            key="system.fuel_flow_rate",
            name="Fuel Flow Rate",
            category="System/Operating",
            unit="sccm",
            description="Anode fuel inlet volumetric flow rate.",
            ptype="continuous",
            min=50.0,
            max=500.0,
            fidelity=["LF", "MF", "HF"],
        ),
        P(
            key="system.air_flow_rate",
            name="Air Flow Rate",
            category="System/Operating",
            unit="sccm",
            description="Cathode air inlet volumetric flow rate.",
            ptype="continuous",
            min=200.0,
            max=2000.0,
            fidelity=["LF", "MF", "HF"],
        ),
        P(
            key="system.fuel_type",
            name="Fuel Type",
            category="System/Operating",
            unit=None,
            description="Fuel species mix (simplified).",
            ptype="categorical",
            levels=["H2", "H2-CO", "CH4-SR"],
            fidelity=["LF", "MF", "HF"],
        ),
        P(
            key="system.steam_to_carbon",
            name="Steam-to-Carbon Ratio",
            category="System/Operating",
            unit="mol/mol",
            description="Steam-to-carbon ratio for hydrocarbon fuels (ignored for pure H2).",
            ptype="continuous",
            min=1.0,
            max=3.0,
            fidelity=["LF", "MF", "HF"],
        ),
    ]

    # Transient Cycles (important for degradation) - apply to all fidelities as input descriptors
    params += [
        P(
            key="transient.startup_ramp_rate",
            name="Startup Temperature Ramp",
            category="System/Transient",
            unit="K/min",
            description="Temperature ramp rate during startup.",
            ptype="continuous",
            min=1.0,
            max=5.0,
            fidelity=["LF", "MF", "HF"],
        ),
        P(
            key="transient.shutdown_ramp_rate",
            name="Shutdown Temperature Ramp",
            category="System/Transient",
            unit="K/min",
            description="Temperature ramp rate during shutdown.",
            ptype="continuous",
            min=1.0,
            max=5.0,
            fidelity=["LF", "MF", "HF"],
        ),
        P(
            key="transient.load_ramp_rate",
            name="Load Ramp Rate",
            category="System/Transient",
            unit="A/cm^2/min",
            description="Rate of load change during ramps.",
            ptype="continuous",
            min=0.01,
            max=0.20,
            fidelity=["LF", "MF", "HF"],
        ),
        P(
            key="transient.cycles_per_1000h",
            name="Cycles per 1000h",
            category="System/Transient",
            unit="count/1000h",
            description="Number of start/stop cycles per 1000h.",
            ptype="integer",
            min=0,
            max=50,
            fidelity=["LF", "MF", "HF"],
        ),
        P(
            key="transient.dwell_time_max_load",
            name="Dwell at Max Load",
            category="System/Transient",
            unit="min",
            description="Dwell duration at maximum load during cycles.",
            ptype="integer",
            min=10,
            max=300,
            fidelity=["LF", "MF", "HF"],
        ),
    ]

    # Cell/Stack Geometry (MF, HF)
    params += [
        P(
            key="geometry.active_area",
            name="Active Area",
            category="Cell/Geometry",
            unit="cm^2",
            description="Electrochemically active area.",
            ptype="continuous",
            min=1.0,
            max=100.0,
            fidelity=["MF", "HF"],
        ),
        P(
            key="geometry.anode_thickness",
            name="Anode Thickness",
            category="Cell/Geometry",
            unit="um",
            description="Ni-YSZ anode total thickness (support + functional).",
            ptype="continuous",
            min=200.0,
            max=1000.0,
            fidelity=["MF", "HF"],
        ),
        P(
            key="geometry.cathode_thickness",
            name="Cathode Thickness",
            category="Cell/Geometry",
            unit="um",
            description="LSCF-based cathode thickness.",
            ptype="continuous",
            min=20.0,
            max=100.0,
            fidelity=["MF", "HF"],
        ),
        P(
            key="geometry.electrolyte_thickness",
            name="Electrolyte Thickness",
            category="Cell/Geometry",
            unit="um",
            description="YSZ electrolyte thickness.",
            ptype="continuous",
            min=5.0,
            max=20.0,
            fidelity=["MF", "HF"],
        ),
        P(
            key="geometry.interconnect_thickness",
            name="Interconnect Thickness",
            category="Cell/Geometry",
            unit="um",
            description="Crofer interconnect plate thickness.",
            ptype="continuous",
            min=500.0,
            max=2000.0,
            fidelity=["MF", "HF"],
        ),
        P(
            key="geometry.channel_width",
            name="Channel Width",
            category="Cell/Geometry",
            unit="mm",
            description="Flow field channel width.",
            ptype="continuous",
            min=0.5,
            max=2.0,
            fidelity=["MF", "HF"],
        ),
        P(
            key="geometry.channel_height",
            name="Channel Height",
            category="Cell/Geometry",
            unit="mm",
            description="Flow field channel height.",
            ptype="continuous",
            min=0.5,
            max=2.0,
            fidelity=["MF", "HF"],
        ),
    ]

    # Material Properties (MF, HF)
    # Anode (Ni-YSZ)
    params += [
        P(
            key="anode.porosity",
            name="Anode Porosity",
            category="Material/Anode",
            unit="fraction",
            description="Bulk porosity of anode composite.",
            ptype="continuous",
            min=0.25,
            max=0.45,
            fidelity=["MF", "HF"],
        ),
        P(
            key="anode.tortuosity",
            name="Anode Tortuosity",
            category="Material/Anode",
            unit="-",
            description="Effective tortuosity for gas transport.",
            ptype="continuous",
            min=2.0,
            max=5.0,
            fidelity=["MF", "HF"],
        ),
        P(
            key="anode.ni_particle_size",
            name="Ni Particle Size",
            category="Material/Anode",
            unit="um",
            description="Mean Ni particle diameter.",
            ptype="continuous",
            min=0.3,
            max=5.0,
            scale="log",
            fidelity=["MF", "HF"],
        ),
        P(
            key="anode.tpb_density",
            name="TPB Density",
            category="Material/Anode",
            unit="um^-2",
            description="Triple-phase boundary density (surface per volume proxy).",
            ptype="continuous",
            min=2.0,
            max=10.0,
            fidelity=["MF", "HF"],
        ),
        P(
            key="anode.ionic_conductivity",
            name="Anode Ionic Conductivity",
            category="Material/Anode",
            unit="S/cm",
            description="Effective ionic conductivity of YSZ in anode.",
            ptype="continuous",
            min=0.01,
            max=1.0,
            scale="log",
            fidelity=["MF", "HF"],
        ),
        P(
            key="anode.electronic_conductivity",
            name="Anode Electronic Conductivity",
            category="Material/Anode",
            unit="S/cm",
            description="Effective electronic conductivity via Ni network.",
            ptype="continuous",
            min=100.0,
            max=1000.0,
            scale="log",
            fidelity=["MF", "HF"],
        ),
    ]

    # Cathode (LSCF)
    params += [
        P(
            key="cathode.porosity",
            name="Cathode Porosity",
            category="Material/Cathode",
            unit="fraction",
            description="Bulk porosity of cathode composite.",
            ptype="continuous",
            min=0.20,
            max=0.40,
            fidelity=["MF", "HF"],
        ),
        P(
            key="cathode.tortuosity",
            name="Cathode Tortuosity",
            category="Material/Cathode",
            unit="-",
            description="Effective tortuosity for gas transport.",
            ptype="continuous",
            min=2.0,
            max=5.0,
            fidelity=["MF", "HF"],
        ),
        P(
            key="cathode.particle_size",
            name="Cathode Particle Size",
            category="Material/Cathode",
            unit="um",
            description="Mean LSCF particle diameter.",
            ptype="continuous",
            min=0.2,
            max=3.0,
            scale="log",
            fidelity=["MF", "HF"],
        ),
        P(
            key="cathode.ionic_conductivity",
            name="Cathode Ionic Conductivity",
            category="Material/Cathode",
            unit="S/cm",
            description="Effective ionic conductivity (if composite with GDC).",
            ptype="continuous",
            min=0.001,
            max=0.10,
            scale="log",
            fidelity=["MF", "HF"],
        ),
        P(
            key="cathode.electronic_conductivity",
            name="Cathode Electronic Conductivity",
            category="Material/Cathode",
            unit="S/cm",
            description="Effective electronic conductivity via LSCF network.",
            ptype="continuous",
            min=10.0,
            max=500.0,
            scale="log",
            fidelity=["MF", "HF"],
        ),
        P(
            key="cathode.chem_expansion_coeff",
            name="Chemical Expansion Coeff",
            category="Material/Cathode",
            unit="1/K",
            description="Chemical expansion coefficient due to stoichiometry changes.",
            ptype="continuous",
            min=5e-6,
            max=30e-6,
            scale="linear",
            fidelity=["MF", "HF"],
        ),
    ]

    # Electrolyte (YSZ)
    params += [
        P(
            key="electrolyte.ionic_conductivity",
            name="Electrolyte Ionic Conductivity",
            category="Material/Electrolyte",
            unit="S/cm",
            description="YSZ ionic conductivity.",
            ptype="continuous",
            min=0.01,
            max=1.0,
            scale="log",
            fidelity=["MF", "HF"],
        ),
        P(
            key="electrolyte.youngs_modulus",
            name="Young's Modulus (YSZ)",
            category="Material/Electrolyte",
            unit="GPa",
            description="Elastic modulus of YSZ.",
            ptype="continuous",
            min=150.0,
            max=220.0,
            fidelity=["MF", "HF"],
        ),
        P(
            key="electrolyte.poissons_ratio",
            name="Poisson's Ratio (YSZ)",
            category="Material/Electrolyte",
            unit="-",
            description="Poisson's ratio of YSZ.",
            ptype="continuous",
            min=0.20,
            max=0.30,
            fidelity=["MF", "HF"],
        ),
        P(
            key="electrolyte.cte",
            name="CTE (YSZ)",
            category="Material/Electrolyte",
            unit="1/K",
            description="Thermal expansion coefficient of YSZ.",
            ptype="continuous",
            min=9e-6,
            max=11e-6,
            fidelity=["MF", "HF"],
        ),
    ]

    # Interconnect (Crofer 22 APU)
    params += [
        P(
            key="interconnect.cte",
            name="CTE (Crofer)",
            category="Material/Interconnect",
            unit="1/K",
            description="Thermal expansion coefficient of Crofer 22 APU.",
            ptype="continuous",
            min=12e-6,
            max=14e-6,
            fidelity=["MF", "HF"],
        ),
        P(
            key="interconnect.creep_n",
            name="Creep Norton Exponent",
            category="Material/Interconnect",
            unit="-",
            description="Norton creep exponent n.",
            ptype="continuous",
            min=3.0,
            max=10.0,
            fidelity=["MF", "HF"],
        ),
        P(
            key="interconnect.creep_A",
            name="Creep A Coefficient",
            category="Material/Interconnect",
            unit="(1/MPa^n)/s",
            description="Norton law prefactor A (simplified).",
            ptype="continuous",
            min=1e-20,
            max=1e-15,
            scale="log",
            fidelity=["MF", "HF"],
        ),
        P(
            key="interconnect.oxide_growth_rate",
            name="Oxide Scale Growth Rate",
            category="Material/Interconnect",
            unit="nm/h",
            description="Chromia scale growth rate.",
            ptype="continuous",
            min=0.1,
            max=10.0,
            scale="log",
            fidelity=["MF", "HF"],
        ),
    ]

    # Microstructural properties (HF Experimental / Synthetic)
    # These are produced/derived in HF via synthetic voxel generation and analysis
    params += [
        P(
            key="micro.voxel_size",
            name="Voxel Size",
            category="Microstructural",
            unit="um",
            description="Voxel edge length (analysis resolution).",
            ptype="continuous",
            min=0.1,
            max=0.5,
            fidelity=["HF"],
        ),
        P(
            key="micro.anode_phase_fractions",
            name="Anode Phase Fractions",
            category="Microstructural",
            unit=None,
            description="Fractions for Ni/YSZ/Pore (derived).",
            ptype="categorical",
            levels=["auto-derived"],
            fidelity=["HF"],
        ),
        P(
            key="micro.cathode_phase_fractions",
            name="Cathode Phase Fractions",
            category="Microstructural",
            unit=None,
            description="Fractions for LSCF/GDC/Pore (derived).",
            ptype="categorical",
            levels=["auto-derived"],
            fidelity=["HF"],
        ),
        P(
            key="micro.specific_surface_area",
            name="Specific Surface Area",
            category="Microstructural",
            unit="1/um",
            description="Interface area per unit volume (approx).",
            ptype="continuous",
            min=0.0,
            max=1e3,
            fidelity=["HF"],
        ),
        P(
            key="micro.connectivity",
            name="Connectivity Metrics",
            category="Microstructural",
            unit=None,
            description="Largest cluster fractions per solid phase (derived).",
            ptype="categorical",
            levels=["auto-derived"],
            fidelity=["HF"],
        ),
    ]

    # Build the grouped schema
    by_fidelity: Dict[str, List[Dict[str, Any]]] = {"LF": [], "MF": [], "HF": []}
    for p in params:
        for f in p.fidelity:
            by_fidelity[f].append(p.to_dict())

    schema = {
        "title": "SOFC Multi-Fidelity Input Schema",
        "version": 1,
        "fidelities": ["LF", "MF", "HF"],
        "parameters": [p.to_dict() for p in params],
        "by_fidelity": by_fidelity,
        "notes": {
            "units": "See each parameter. Thickness in um, channels in mm, flow in sccm.",
            "distributions": "Unless otherwise stated, parameters are sampled uniformly within bounds. Log scale indicates uniform in log-space.",
            "microstructure": "HF microstructures are synthetic voxel volumes with uint8 labels for phases.",
        },
    }

    return schema


if __name__ == "__main__":
    import json, sys
    s = get_schema()
    json.dump(s, sys.stdout, indent=2)
    print()
