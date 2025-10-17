#!/usr/bin/env python3
"""
Dataset generator for Retrofit Intervention & Lifecycle Data (Decision Engine).

Generates synthetic-yet-plausible CSV datasets for:
- retrofit_measures.csv: Library of measures and linked performance, costs, lifespans
- lci_materials.csv: Embodied carbon (A1-A3, A1-A5, B, C) and sources
- maintenance.csv: Scheduled maintenance costs/intervals for systems
- operational_factors.csv: Operational carbon intensity factors per energy carrier
- degradation_curves.csv: Annual degradation rates for system performance
- grid_carbon_intensity_hourly.csv: Hourly kgCO2e/kWh for a synthetic grid region
- measure_bill_of_materials.csv: Links measures to LCI materials with quantities

All numbers are fabricated for demonstration purposes.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple


RANDOM_SEED = 42
random.seed(RANDOM_SEED)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _round(value: float, ndigits: int = 3) -> float:
    return float(round(value, ndigits))


# ----------------------------- Retrofit Measures -----------------------------

@dataclass
class RetrofitMeasure:
    measure_id: str
    category: str
    name: str
    description: str
    performance_metric: str
    baseline_value: float
    improved_value: float
    unit: str
    capex_usd: float
    opex_delta_usd_per_year: float
    lifespan_years: int
    system_id: str


def generate_retrofit_measures() -> List[RetrofitMeasure]:
    measures: List[RetrofitMeasure] = []

    # Envelope measures
    envelope_base = [
        (
            "roof_polyiso_4in",
            "Envelope",
            "Roof insulation upgrade (polyiso, +4in)",
            "Add 4 in polyisocyanurate insulation to roof to reduce U-value",
            "U-value",
            0.35,
            0.20,
            "W/m2K",
            22000,
            -500,
            35,
            "roof_insulation",
        ),
        (
            "roof_polyiso_6in",
            "Envelope",
            "Roof insulation upgrade (polyiso, +6in)",
            "Add 6 in polyisocyanurate insulation to roof to reduce U-value further",
            "U-value",
            0.35,
            0.15,
            "W/m2K",
            28000,
            -700,
            40,
            "roof_insulation",
        ),
        (
            "wall_eps_3in",
            "Envelope",
            "Wall insulation upgrade (EPS, +3in)",
            "Add 3 in EPS external insulation to reduce wall U-value",
            "U-value",
            0.55,
            0.35,
            "W/m2K",
            45000,
            -1200,
            50,
            "wall_insulation",
        ),
        (
            "window_u12",
            "Envelope",
            "High-performance window replacement (U=1.2)",
            "Replace existing windows with U=1.2 W/m2K double/triple glazing",
            "U-value",
            2.7,
            1.2,
            "W/m2K",
            180000,
            -4000,
            30,
            "windows",
        ),
        (
            "airtightness_upgrade",
            "Envelope",
            "Airtightness improvement (50%)",
            "Envelope infiltration reduction via sealing and detailing improvements",
            "ACH50",
            10.0,
            5.0,
            "1/h",
            25000,
            -800,
            25,
            "airtightness",
        ),
    ]

    # HVAC systems
    hvac_base = [
        (
            "boiler_95",
            "HVAC",
            "Condensing boiler replacement (95%)",
            "Replace legacy boiler with 95% efficient condensing model",
            "Thermal efficiency",
            0.80,
            0.95,
            "fraction",
            85000,
            -3500,
            20,
            "boiler",
        ),
        (
            "ashp_seasonal_3.5",
            "HVAC",
            "Air-source heat pump (SCOP 3.5)",
            "Install variable-speed ASHP to provide heating/cooling",
            "SCOP",
            2.4,
            3.5,
            "ratio",
            240000,
            -12000,
            18,
            "heat_pump",
        ),
        (
            "vrf_system_cop4",
            "HVAC",
            "VRF system (COP 4.0)",
            "Install VRF with heat recovery fan coils",
            "COP",
            2.8,
            4.0,
            "ratio",
            320000,
            -18000,
            18,
            "vrf",
        ),
        (
            "heat_recovery_vent",
            "HVAC",
            "Heat recovery ventilation (75%)",
            "Add centralized HRV to reduce ventilation heat losses",
            "Sensible recovery",
            0.0,
            0.75,
            "fraction",
            90000,
            -6000,
            20,
            "hrv",
        ),
    ]

    # Renewables and DERs
    der_base = [
        (
            "pv_10kw",
            "DER",
            "Solar PV array (10 kW)",
            "Install rooftop solar PV array rated at 10 kW",
            "Capacity",
            0.0,
            10.0,
            "kW",
            20000,
            -50,
            30,
            "pv",
        ),
        (
            "pv_50kw",
            "DER",
            "Solar PV array (50 kW)",
            "Install rooftop solar PV array rated at 50 kW",
            "Capacity",
            0.0,
            50.0,
            "kW",
            90000,
            -250,
            30,
            "pv",
        ),
        (
            "battery_100kwh",
            "DER",
            "Li-ion battery (100 kWh)",
            "Add lithium-ion battery storage with 100 kWh usable capacity",
            "Usable capacity",
            0.0,
            100.0,
            "kWh",
            70000,
            -400,
            12,
            "battery",
        ),
        (
            "solar_thermal_dhw",
            "DER",
            "Solar thermal DHW (40% solar fraction)",
            "Install solar thermal collectors for domestic hot water",
            "DHW solar fraction",
            0.0,
            0.40,
            "fraction",
            55000,
            -1000,
            25,
            "solar_thermal",
        ),
    ]

    for row in envelope_base + hvac_base + der_base:
        measures.append(RetrofitMeasure(*row))

    return measures


def write_retrofit_measures_csv(path: str, measures: List[RetrofitMeasure]) -> None:
    fieldnames = [
        "measure_id",
        "category",
        "name",
        "description",
        "performance_metric",
        "baseline_value",
        "improved_value",
        "unit",
        "capex_usd",
        "opex_delta_usd_per_year",
        "lifespan_years",
        "system_id",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in measures:
            writer.writerow(dataclasses.asdict(m))


# ----------------------------- LCI Materials ------------------------------

@dataclass
class LciMaterial:
    material_id: str
    name: str
    category: str
    gwp_a1_a3_kgco2e_per_unit: float
    gwp_a1_a5_kgco2e_per_unit: float
    gwp_b_total_kgco2e_per_unit: float
    gwp_c_total_kgco2e_per_unit: float
    functional_unit: str
    source: str


def generate_lci_materials() -> List[LciMaterial]:
    # Fabricated values inspired by ranges in ICE/EPD datasets
    base = [
        ("polyiso", "Polyisocyanurate insulation", "Insulation", 3.5, 4.1, 0.2, 0.1, "kg", "EPD-synthetic"),
        ("eps", "Expanded polystyrene (EPS)", "Insulation", 2.8, 3.3, 0.15, 0.08, "kg", "EPD-synthetic"),
        ("glass_unit", "Double-glazed window unit", "Fenestration", 7.0, 8.2, 0.4, 0.2, "kg", "ICE-synthetic"),
        ("al_frame", "Aluminum window frame", "Fenestration", 9.5, 10.8, 0.6, 0.3, "kg", "ICE-synthetic"),
        ("steel", "Structural steel", "Structure", 1.8, 2.2, 0.1, 0.05, "kg", "Ecoinvent-synthetic"),
        ("concrete_c30", "Concrete C30/37", "Structure", 0.12, 0.16, 0.01, 0.02, "kg", "Ecoinvent-synthetic"),
        ("li_ion_battery", "Lithium-ion battery pack", "Electrochemical", 80.0, 90.0, 3.0, 5.0, "kWh", "EPD-synthetic"),
        ("pv_module", "Crystalline silicon PV module", "PV", 600.0, 650.0, 15.0, 20.0, "kWp", "EPD-synthetic"),
        ("copper_wire", "Copper wiring", "Electrical", 4.0, 4.6, 0.1, 0.1, "kg", "ICE-synthetic"),
    ]
    materials = [LciMaterial(*row) for row in base]
    return materials


def write_lci_materials_csv(path: str, materials: List[LciMaterial]) -> None:
    fieldnames = [
        "material_id",
        "name",
        "category",
        "gwp_a1_a3_kgco2e_per_unit",
        "gwp_a1_a5_kgco2e_per_unit",
        "gwp_b_total_kgco2e_per_unit",
        "gwp_c_total_kgco2e_per_unit",
        "functional_unit",
        "source",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in materials:
            writer.writerow(dataclasses.asdict(m))


# ----------------------------- Measure Bill of Materials ----------------------

@dataclass
class MeasureMaterial:
    measure_id: str
    material_id: str
    quantity: float
    unit: str
    basis_notes: str


def generate_measure_bill_of_materials() -> List[MeasureMaterial]:
    # Simple fabricated quantities linking measures to materials
    bom: List[MeasureMaterial] = []

    # Envelope
    bom.append(MeasureMaterial("roof_polyiso_4in", "polyiso", 5000, "kg", "Assumes 2500 m2 roof, 2 kg/m2 per inch"))
    bom.append(MeasureMaterial("roof_polyiso_6in", "polyiso", 7500, "kg", "Assumes 2500 m2 roof, 2 kg/m2 per inch"))
    bom.append(MeasureMaterial("wall_eps_3in", "eps", 9000, "kg", "Assumes 3000 m2 walls, 1 kg/m2 per inch"))
    bom.append(MeasureMaterial("window_u12", "glass_unit", 12000, "kg", "Window area 1200 m2, 10 kg/m2"))
    bom.append(MeasureMaterial("window_u12", "al_frame", 3000, "kg", "Aluminum frames and spacers"))

    # Airtightness uses sealants not listed; approximate with EPS proxy
    bom.append(MeasureMaterial("airtightness_upgrade", "eps", 300, "kg", "Sealants/foams proxy"))

    # HVAC
    bom.append(MeasureMaterial("boiler_95", "steel", 3000, "kg", "Boiler body and piping"))
    bom.append(MeasureMaterial("boiler_95", "copper_wire", 200, "kg", "Electrical and controls"))

    bom.append(MeasureMaterial("ashp_seasonal_3.5", "steel", 2500, "kg", "Outdoor units and piping"))
    bom.append(MeasureMaterial("ashp_seasonal_3.5", "copper_wire", 400, "kg", "Wiring and motors"))

    bom.append(MeasureMaterial("vrf_system_cop4", "steel", 4000, "kg", "Condensers and fan coils"))
    bom.append(MeasureMaterial("vrf_system_cop4", "copper_wire", 600, "kg", "Wiring and refrigerant lines"))

    bom.append(MeasureMaterial("heat_recovery_vent", "steel", 1200, "kg", "Casing and ducts"))
    bom.append(MeasureMaterial("heat_recovery_vent", "copper_wire", 150, "kg", "Wiring and fans"))

    # DERs
    bom.append(MeasureMaterial("pv_10kw", "pv_module", 10, "kWp", "10 kWp array"))
    bom.append(MeasureMaterial("pv_10kw", "copper_wire", 120, "kg", "Balance of system"))

    bom.append(MeasureMaterial("pv_50kw", "pv_module", 50, "kWp", "50 kWp array"))
    bom.append(MeasureMaterial("pv_50kw", "copper_wire", 400, "kg", "Balance of system"))

    bom.append(MeasureMaterial("battery_100kwh", "li_ion_battery", 100, "kWh", "Pack energy capacity"))
    bom.append(MeasureMaterial("battery_100kwh", "steel", 500, "kg", "Racks and enclosures"))

    bom.append(MeasureMaterial("solar_thermal_dhw", "steel", 800, "kg", "Collectors, tanks, piping"))
    bom.append(MeasureMaterial("solar_thermal_dhw", "copper_wire", 80, "kg", "Pumps and controls"))

    return bom


def write_measure_bom_csv(path: str, items: List[MeasureMaterial]) -> None:
    fieldnames = ["measure_id", "material_id", "quantity", "unit", "basis_notes"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for it in items:
            writer.writerow(dataclasses.asdict(it))


# ----------------------------- Maintenance ------------------------------

@dataclass
class MaintenanceItem:
    system_id: str
    description: str
    interval_years: int
    cost_usd: float


def generate_maintenance_items() -> List[MaintenanceItem]:
    base = [
        ("roof_insulation", "Roof inspection and minor repairs", 5, 1500),
        ("wall_insulation", "Façade inspection and sealant refresh", 7, 2500),
        ("windows", "Gasket and seal replacement", 10, 10000),
        ("airtightness", "Blower-door test and resealing", 8, 3000),
        ("boiler", "Annual service and tune-up", 1, 1200),
        ("heat_pump", "Filter and refrigerant check", 1, 900),
        ("vrf", "VRF system inspection and cleaning", 1, 1300),
        ("hrv", "HRV core cleaning and filter change", 1, 700),
        ("pv", "Inverter inspection and cleaning", 1, 600),
        ("battery", "BMS calibration and health check", 1, 800),
        ("solar_thermal", "Glycol top-up and pump check", 2, 1000),
    ]
    return [MaintenanceItem(*row) for row in base]


def write_maintenance_csv(path: str, items: List[MaintenanceItem]) -> None:
    fieldnames = ["system_id", "description", "interval_years", "cost_usd"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for it in items:
            writer.writerow(dataclasses.asdict(it))


# ----------------------------- Operational Factors ------------------------------

@dataclass
class OperationalFactor:
    energy_carrier: str
    unit: str
    kgco2e_per_unit: float
    source: str


def generate_operational_factors() -> List[OperationalFactor]:
    # Typical illustrative intensity values (fabricated, region-neutral)
    base = [
        ("electricity", "kWh", 0.35, "Grid mix synthetic"),
        ("natural_gas", "kWh", 0.20, "Combustion factor synthetic"),
        ("district_heat", "kWh", 0.18, "District mix synthetic"),
        ("fuel_oil", "kWh", 0.27, "Combustion factor synthetic"),
    ]
    return [OperationalFactor(*row) for row in base]


def write_operational_factors_csv(path: str, items: List[OperationalFactor]) -> None:
    fieldnames = ["energy_carrier", "unit", "kgco2e_per_unit", "source"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for it in items:
            writer.writerow(dataclasses.asdict(it))


# ----------------------------- Degradation Curves ------------------------------

@dataclass
class DegradationCurve:
    system_id: str
    parameter: str
    annual_degradation_fraction: float
    notes: str


def generate_degradation_curves() -> List[DegradationCurve]:
    base = [
        ("pv", "capacity_factor", 0.005, "1/2% per year PV module degradation"),
        ("battery", "usable_capacity", 0.03, "3% per year due to cycling"),
        ("boiler", "efficiency", 0.004, "Scaling and wear"),
        ("heat_pump", "cop", 0.006, "Refrigerant and compressor wear"),
        ("vrf", "cop", 0.005, "Fan coil dirt accumulation"),
        ("hrv", "sensible_recovery", 0.005, "Filter fouling if not maintained"),
        ("windows", "u_value", 0.002, "Sealant aging"),
        ("roof_insulation", "r_value", 0.001, "Settling over decades"),
        ("wall_insulation", "r_value", 0.001, "Moisture effects over time"),
    ]
    return [DegradationCurve(*row) for row in base]


def write_degradation_curves_csv(path: str, items: List[DegradationCurve]) -> None:
    fieldnames = ["system_id", "parameter", "annual_degradation_fraction", "notes"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for it in items:
            writer.writerow(dataclasses.asdict(it))


# ----------------------------- Grid Carbon Intensity (Hourly) ------------------

def generate_hourly_grid_intensity(start: datetime, hours: int) -> List[Tuple[datetime, float]]:
    values: List[Tuple[datetime, float]] = []
    # Build a daily sinusoidal pattern with seasonal modulation
    for i in range(hours):
        t = start + timedelta(hours=i)
        hour = t.hour
        day_of_year = int(t.strftime("%j"))

        # Daily: peak around 19:00, trough at 3:00
        daily = 0.5 + 0.5 * math.sin((hour - 3) / 24.0 * 2.0 * math.pi)
        # Seasonal: higher in winter
        seasonal = 0.7 + 0.3 * math.sin((day_of_year - 30) / 365.0 * 2.0 * math.pi)
        base = 0.20 + 0.25 * seasonal + 0.15 * daily
        noise = random.uniform(-0.02, 0.02)
        intensity = max(0.05, _round(base + noise, 3))
        values.append((t, intensity))
    return values


def write_grid_intensity_csv(path: str, series: List[Tuple[datetime, float]]) -> None:
    fieldnames = ["timestamp", "kgco2e_per_kwh"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ts, val in series:
            writer.writerow({"timestamp": ts.isoformat(), "kgco2e_per_kwh": val})


# ----------------------------- Orchestration ------------------------------

def main() -> None:
    retrofit_measures = generate_retrofit_measures()
    lci_materials = generate_lci_materials()
    measure_bom = generate_measure_bill_of_materials()
    maintenance_items = generate_maintenance_items()
    operational_factors = generate_operational_factors()
    degradation_curves = generate_degradation_curves()
    hourly_intensity = generate_hourly_grid_intensity(
        start=datetime(2024, 1, 1, 0, 0, 0), hours=8760
    )

    write_retrofit_measures_csv(os.path.join(DATA_DIR, "retrofit_measures.csv"), retrofit_measures)
    write_lci_materials_csv(os.path.join(DATA_DIR, "lci_materials.csv"), lci_materials)
    write_measure_bom_csv(os.path.join(DATA_DIR, "measure_bill_of_materials.csv"), measure_bom)
    write_maintenance_csv(os.path.join(DATA_DIR, "maintenance.csv"), maintenance_items)
    write_operational_factors_csv(os.path.join(DATA_DIR, "operational_factors.csv"), operational_factors)
    write_degradation_curves_csv(os.path.join(DATA_DIR, "degradation_curves.csv"), degradation_curves)
    write_grid_intensity_csv(os.path.join(DATA_DIR, "grid_carbon_intensity_hourly.csv"), hourly_intensity)

    print("Generated dataset in:", DATA_DIR)


if __name__ == "__main__":
    main()
