#!/usr/bin/env python3
import csv
import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List

DATASET_VERSION = "1.0.0"
SEED = 42
random.seed(SEED)

ROOT = "/workspace/data/retrofit_decision_engine"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, fieldnames: List[str], rows: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


# ---------------------- LCI: Materials ----------------------

def generate_materials_lci() -> List[Dict]:
    # Values are plausible, fabricated, and normalized to per-kg basis
    # GWP values represent cradle-to-gate A1-A3, with estimates for A1-A5, B, C
    materials = [
        {
            "material_id": "MAT_POLYISO",
            "name": "Polyisocyanurate insulation",
            "category": "insulation",
            "density_kg_per_m3": 32,
            "gwp_a1a3_kgco2e_per_kg": 3.2,
            "gwp_a1a5_kgco2e_per_kg": 3.6,
            "gwp_b_kgco2e_per_kg": 0.2,
            "gwp_c_kgco2e_per_kg": 0.1,
            "source": "Fabricated typical range inspired by ICE/Ecoinvent/EPDs"
        },
        {
            "material_id": "MAT_MINERAL_WOOL",
            "name": "Mineral wool insulation",
            "category": "insulation",
            "density_kg_per_m3": 40,
            "gwp_a1a3_kgco2e_per_kg": 1.6,
            "gwp_a1a5_kgco2e_per_kg": 2.0,
            "gwp_b_kgco2e_per_kg": 0.15,
            "gwp_c_kgco2e_per_kg": 0.05,
            "source": "Fabricated typical range inspired by ICE/Ecoinvent/EPDs"
        },
        {
            "material_id": "MAT_XPS",
            "name": "Extruded polystyrene (XPS)",
            "category": "insulation",
            "density_kg_per_m3": 35,
            "gwp_a1a3_kgco2e_per_kg": 6.5,
            "gwp_a1a5_kgco2e_per_kg": 7.2,
            "gwp_b_kgco2e_per_kg": 0.25,
            "gwp_c_kgco2e_per_kg": 0.1,
            "source": "Fabricated typical range inspired by ICE/Ecoinvent/EPDs"
        },
        {
            "material_id": "MAT_EPS",
            "name": "Expanded polystyrene (EPS)",
            "category": "insulation",
            "density_kg_per_m3": 20,
            "gwp_a1a3_kgco2e_per_kg": 2.5,
            "gwp_a1a5_kgco2e_per_kg": 2.9,
            "gwp_b_kgco2e_per_kg": 0.15,
            "gwp_c_kgco2e_per_kg": 0.08,
            "source": "Fabricated typical range inspired by ICE/Ecoinvent/EPDs"
        },
        {
            "material_id": "MAT_SPF",
            "name": "Spray polyurethane foam (SPF)",
            "category": "insulation",
            "density_kg_per_m3": 40,
            "gwp_a1a3_kgco2e_per_kg": 5.8,
            "gwp_a1a5_kgco2e_per_kg": 6.4,
            "gwp_b_kgco2e_per_kg": 0.25,
            "gwp_c_kgco2e_per_kg": 0.1,
            "source": "Fabricated typical range inspired by ICE/Ecoinvent/EPDs"
        },
        {
            "material_id": "MAT_GYPSUM",
            "name": "Gypsum board",
            "category": "interior",
            "density_kg_per_m3": 800,
            "gwp_a1a3_kgco2e_per_kg": 0.35,
            "gwp_a1a5_kgco2e_per_kg": 0.45,
            "gwp_b_kgco2e_per_kg": 0.02,
            "gwp_c_kgco2e_per_kg": 0.02,
            "source": "Fabricated typical range inspired by ICE/Ecoinvent/EPDs"
        },
        {
            "material_id": "MAT_CONCRETE_C30",
            "name": "Concrete C30/37",
            "category": "concrete",
            "density_kg_per_m3": 2400,
            "gwp_a1a3_kgco2e_per_kg": 0.12,
            "gwp_a1a5_kgco2e_per_kg": 0.17,
            "gwp_b_kgco2e_per_kg": 0.01,
            "gwp_c_kgco2e_per_kg": 0.02,
            "source": "Fabricated typical range inspired by ICE/Ecoinvent/EPDs"
        },
        {
            "material_id": "MAT_STEEL_STRUCT",
            "name": "Structural steel",
            "category": "metals",
            "density_kg_per_m3": 7850,
            "gwp_a1a3_kgco2e_per_kg": 2.0,
            "gwp_a1a5_kgco2e_per_kg": 2.3,
            "gwp_b_kgco2e_per_kg": 0.02,
            "gwp_c_kgco2e_per_kg": 0.05,
            "source": "Fabricated typical range inspired by ICE/Ecoinvent/EPDs"
        },
        {
            "material_id": "MAT_ALUMINUM",
            "name": "Aluminum extrusion",
            "category": "metals",
            "density_kg_per_m3": 2700,
            "gwp_a1a3_kgco2e_per_kg": 8.5,
            "gwp_a1a5_kgco2e_per_kg": 9.2,
            "gwp_b_kgco2e_per_kg": 0.05,
            "gwp_c_kgco2e_per_kg": 0.08,
            "source": "Fabricated typical range inspired by ICE/Ecoinvent/EPDs"
        },
        {
            "material_id": "MAT_GLASS",
            "name": "Float glass",
            "category": "fenestration",
            "density_kg_per_m3": 2500,
            "gwp_a1a3_kgco2e_per_kg": 1.2,
            "gwp_a1a5_kgco2e_per_kg": 1.5,
            "gwp_b_kgco2e_per_kg": 0.03,
            "gwp_c_kgco2e_per_kg": 0.05,
            "source": "Fabricated typical range inspired by ICE/Ecoinvent/EPDs"
        },
        {
            "material_id": "MAT_PVC",
            "name": "Polyvinyl chloride (PVC)",
            "category": "plastics",
            "density_kg_per_m3": 1380,
            "gwp_a1a3_kgco2e_per_kg": 2.7,
            "gwp_a1a5_kgco2e_per_kg": 3.0,
            "gwp_b_kgco2e_per_kg": 0.03,
            "gwp_c_kgco2e_per_kg": 0.06,
            "source": "Fabricated typical range inspired by ICE/Ecoinvent/EPDs"
        },
        {
            "material_id": "MAT_COPPER",
            "name": "Copper",
            "category": "metals",
            "density_kg_per_m3": 8900,
            "gwp_a1a3_kgco2e_per_kg": 4.0,
            "gwp_a1a5_kgco2e_per_kg": 4.5,
            "gwp_b_kgco2e_per_kg": 0.02,
            "gwp_c_kgco2e_per_kg": 0.05,
            "source": "Fabricated typical range inspired by ICE/Ecoinvent/EPDs"
        },
        {
            "material_id": "MAT_PEX",
            "name": "PEX piping",
            "category": "plastics",
            "density_kg_per_m3": 940,
            "gwp_a1a3_kgco2e_per_kg": 2.2,
            "gwp_a1a5_kgco2e_per_kg": 2.5,
            "gwp_b_kgco2e_per_kg": 0.02,
            "gwp_c_kgco2e_per_kg": 0.05,
            "source": "Fabricated typical range inspired by ICE/Ecoinvent/EPDs"
        },
        {
            "material_id": "MAT_R410A",
            "name": "R410A (refrigerant)",
            "category": "refrigerant",
            "density_kg_per_m3": "",
            "gwp_a1a3_kgco2e_per_kg": 10,
            "gwp_a1a5_kgco2e_per_kg": 12,
            "gwp_b_kgco2e_per_kg": 0,
            "gwp_c_kgco2e_per_kg": 0,
            "source": "Fabricated LCI; leakage impact tracked separately"
        },
        {
            "material_id": "MAT_R32",
            "name": "R32 (refrigerant)",
            "category": "refrigerant",
            "density_kg_per_m3": "",
            "gwp_a1a3_kgco2e_per_kg": 8,
            "gwp_a1a5_kgco2e_per_kg": 9,
            "gwp_b_kgco2e_per_kg": 0,
            "gwp_c_kgco2e_per_kg": 0,
            "source": "Fabricated LCI; leakage impact tracked separately"
        },
        {
            "material_id": "MAT_SOFTWOOD",
            "name": "Softwood lumber",
            "category": "wood",
            "density_kg_per_m3": 500,
            "gwp_a1a3_kgco2e_per_kg": 0.2,
            "gwp_a1a5_kgco2e_per_kg": 0.25,
            "gwp_b_kgco2e_per_kg": 0.01,
            "gwp_c_kgco2e_per_kg": 0.02,
            "source": "Fabricated typical range inspired by ICE/Ecoinvent/EPDs"
        },
        {
            "material_id": "MAT_ASPHALT",
            "name": "Asphalt roofing",
            "category": "membrane",
            "density_kg_per_m3": 1100,
            "gwp_a1a3_kgco2e_per_kg": 0.7,
            "gwp_a1a5_kgco2e_per_kg": 0.9,
            "gwp_b_kgco2e_per_kg": 0.03,
            "gwp_c_kgco2e_per_kg": 0.05,
            "source": "Fabricated typical range inspired by ICE/Ecoinvent/EPDs"
        },
        {
            "material_id": "MAT_EPDM",
            "name": "EPDM roofing membrane",
            "category": "membrane",
            "density_kg_per_m3": 1200,
            "gwp_a1a3_kgco2e_per_kg": 3.0,
            "gwp_a1a5_kgco2e_per_kg": 3.3,
            "gwp_b_kgco2e_per_kg": 0.05,
            "gwp_c_kgco2e_per_kg": 0.08,
            "source": "Fabricated typical range inspired by ICE/Ecoinvent/EPDs"
        },
        {
            "material_id": "MAT_TPO",
            "name": "TPO roofing membrane",
            "category": "membrane",
            "density_kg_per_m3": 900,
            "gwp_a1a3_kgco2e_per_kg": 2.0,
            "gwp_a1a5_kgco2e_per_kg": 2.3,
            "gwp_b_kgco2e_per_kg": 0.05,
            "gwp_c_kgco2e_per_kg": 0.08,
            "source": "Fabricated typical range inspired by ICE/Ecoinvent/EPDs"
        },
    ]
    return materials


# ---------------------- LCI: Systems ----------------------

def generate_systems_lci() -> List[Dict]:
    systems = [
        {
            "system_id": "SYS_GAS_BOILER_COND",
            "name": "Condensing gas boiler",
            "category": "HVAC",
            "unit": "per_kw",
            "gwp_a1a3_kgco2e_per_unit": 25.0,
            "gwp_a1a5_kgco2e_per_unit": 28.0,
            "lifetime_years_default": 20,
            "source": "Fabricated typical range inspired by EPDs"
        },
        {
            "system_id": "SYS_AIR_TO_WATER_HP",
            "name": "Air-to-water heat pump",
            "category": "HVAC",
            "unit": "per_kw",
            "gwp_a1a3_kgco2e_per_unit": 60.0,
            "gwp_a1a5_kgco2e_per_unit": 66.0,
            "lifetime_years_default": 15,
            "source": "Fabricated typical range inspired by EPDs"
        },
        {
            "system_id": "SYS_AIR_TO_AIR_HP",
            "name": "Air-to-air heat pump (RTU/mini-split)",
            "category": "HVAC",
            "unit": "per_kw",
            "gwp_a1a3_kgco2e_per_unit": 52.0,
            "gwp_a1a5_kgco2e_per_unit": 58.0,
            "lifetime_years_default": 15,
            "source": "Fabricated typical range inspired by EPDs"
        },
        {
            "system_id": "SYS_WATER_COOLED_CHILLER",
            "name": "Water-cooled chiller",
            "category": "HVAC",
            "unit": "per_kw",
            "gwp_a1a3_kgco2e_per_unit": 80.0,
            "gwp_a1a5_kgco2e_per_unit": 90.0,
            "lifetime_years_default": 25,
            "source": "Fabricated typical range inspired by EPDs"
        },
        {
            "system_id": "SYS_ERV",
            "name": "Energy recovery ventilator",
            "category": "HVAC",
            "unit": "per_cfm",
            "gwp_a1a3_kgco2e_per_unit": 0.18,
            "gwp_a1a5_kgco2e_per_unit": 0.22,
            "lifetime_years_default": 20,
            "source": "Fabricated typical range inspired by EPDs"
        },
        {
            "system_id": "SYS_LED_FIXTURE",
            "name": "LED luminaire",
            "category": "Electrical",
            "unit": "per_fixture",
            "gwp_a1a3_kgco2e_per_unit": 12.0,
            "gwp_a1a5_kgco2e_per_unit": 14.0,
            "lifetime_years_default": 12,
            "source": "Fabricated typical range inspired by EPDs"
        },
        {
            "system_id": "SYS_PV_MODULE",
            "name": "PV modules + BOS",
            "category": "Electrical",
            "unit": "per_kw",
            "gwp_a1a3_kgco2e_per_unit": 600.0,
            "gwp_a1a5_kgco2e_per_unit": 660.0,
            "lifetime_years_default": 25,
            "source": "Fabricated typical range inspired by EPDs"
        },
        {
            "system_id": "SYS_PV_INVERTER",
            "name": "PV string inverter",
            "category": "Electrical",
            "unit": "per_kw",
            "gwp_a1a3_kgco2e_per_unit": 80.0,
            "gwp_a1a5_kgco2e_per_unit": 90.0,
            "lifetime_years_default": 12,
            "source": "Fabricated typical range inspired by EPDs"
        },
        {
            "system_id": "SYS_LI_ION_BATTERY",
            "name": "Li-ion battery pack",
            "category": "Electrical",
            "unit": "per_kwh_cap",
            "gwp_a1a3_kgco2e_per_unit": 90.0,
            "gwp_a1a5_kgco2e_per_unit": 100.0,
            "lifetime_years_default": 10,
            "source": "Fabricated typical range inspired by EPDs"
        }
    ]
    return systems


# ---------------------- Measures ----------------------

def generate_measures() -> List[Dict]:
    measures: List[Dict] = []

    def add_measure(mid: str, name: str, category: str, target: str, unit: str, unit_desc: str,
                    capex_per_unit_usd: float, fixed_cost_usd: float, lifespan_years: int,
                    perf: Dict, notes: str = ""):
        base = {
            "measure_id": mid,
            "name": name,
            "category": category,
            "target": target,
            "unit": unit,
            "unit_desc": unit_desc,
            "capex_per_unit_usd": round(capex_per_unit_usd, 2),
            "fixed_cost_usd": round(fixed_cost_usd, 2),
            "lifespan_years": lifespan_years,
            "currency": "USD",
            "opex_annual_delta_usd": round(perf.get("opex_annual_delta_usd", 0.0), 2),
            "perf_u_value_old": perf.get("u_old", ""),
            "perf_u_value_new": perf.get("u_new", ""),
            "perf_r_value_old": perf.get("r_old", ""),
            "perf_r_value_new": perf.get("r_new", ""),
            "perf_hvac_eff_old": perf.get("eff_old", ""),
            "perf_hvac_eff_new": perf.get("eff_new", ""),
            "perf_lpd_old_w_per_m2": perf.get("lpd_old", ""),
            "perf_lpd_new_w_per_m2": perf.get("lpd_new", ""),
            "perf_infiltration_ach50_old": perf.get("ach50_old", ""),
            "perf_infiltration_ach50_new": perf.get("ach50_new", ""),
            "pv_nameplate_kw": perf.get("pv_kw", ""),
            "battery_capacity_kwh": perf.get("bat_kwh", ""),
            "control_savings_pct_hvac": perf.get("ctrl_hvac_pct", ""),
            "control_savings_pct_ltg": perf.get("ctrl_ltg_pct", ""),
            "notes": notes,
        }
        measures.append(base)

    # Envelope - Roof insulation (thickness variants per m2)
    for inches, r_per_inch, material in [
        (2, 6.0, "POLYISO"), (4, 6.0, "POLYISO"), (6, 6.0, "POLYISO"),
        (4, 4.2, "XPS"), (6, 4.2, "XPS"), (6, 3.8, "MINERAL_WOOL")
    ]:
        r_add = inches * r_per_inch * 0.1761  # convert (h·ft²·°F/BTU) to m²K/W approx
        u_old = 0.5  # W/m²K baseline
        r_old = 1 / u_old
        r_new = r_old + r_add
        u_new = 1 / r_new
        capex = 18 * inches  # USD/m² rough installed cost scaling with thickness
        add_measure(
            mid=f"ME_EN_ROOF_R{int(r_add*10)}_{material}",
            name=f"Add {inches} inches {material.replace('_',' ').title()} roof insulation",
            category="Envelope",
            target="Roof",
            unit="m2",
            unit_desc="Per square meter of roof area",
            capex_per_unit_usd=capex,
            fixed_cost_usd=0,
            lifespan_years=40,
            perf={"u_old": round(u_old, 3), "u_new": round(u_new, 3), "r_old": round(r_old, 2), "r_new": round(r_new, 2)},
            notes="Assumes overlay above deck; includes membrane touch-ups"
        )

    # Envelope - Air sealing per m2 of envelope area (proxy)
    add_measure(
        mid="ME_EN_AIR_SEALING",
        name="Comprehensive air sealing",
        category="Envelope",
        target="WholeEnvelope",
        unit="m2",
        unit_desc="Per square meter of envelope area (proxy)",
        capex_per_unit_usd=8.0,
        fixed_cost_usd=1500.0,
        lifespan_years=20,
        perf={"ach50_old": 8.0, "ach50_new": 4.0},
        notes="Includes sealing of penetrations and weatherstripping"
    )

    # Envelope - Windows upgrades per m2
    for u_new, desc in [(1.8, "Double-pane low-E"), (1.2, "Triple-pane low-E")]:
        add_measure(
            mid=f"ME_EN_WINDOW_U{int(u_new*10)}",
            name=f"Replace windows with {desc}",
            category="Envelope",
            target="Windows",
            unit="m2",
            unit_desc="Per square meter of window area",
            capex_per_unit_usd=350.0 if u_new == 1.8 else 500.0,
            fixed_cost_usd=2000.0,
            lifespan_years=30,
            perf={"u_old": 3.2, "u_new": u_new},
            notes="Includes frames and glazing; U-value in W/m²K"
        )

    # Envelope - Exterior shading per m2 glazing
    add_measure(
        mid="ME_EN_SHADING",
        name="Install exterior shading devices",
        category="Envelope",
        target="Shading",
        unit="m2",
        unit_desc="Per square meter of glazing",
        capex_per_unit_usd=120.0,
        fixed_cost_usd=1000.0,
        lifespan_years=25,
        perf={"opex_annual_delta_usd": 0.0},
        notes="Reduces solar gains; improves comfort"
    )

    # HVAC - Boiler to condensing boiler per kW
    add_measure(
        mid="ME_HV_BOILER_COND",
        name="Replace boiler with condensing boiler",
        category="HVAC",
        target="Boiler",
        unit="kw",
        unit_desc="Per kW of boiler capacity",
        capex_per_unit_usd=180.0,
        fixed_cost_usd=5000.0,
        lifespan_years=20,
        perf={"eff_old": 0.82, "eff_new": 0.96},
        notes="High-efficiency condensing with O2 trim"
    )

    # HVAC - Electrify: air-to-water heat pump per kW
    add_measure(
        mid="ME_HV_ATW_HP",
        name="Replace boiler with air-to-water heat pump",
        category="HVAC",
        target="BoilerToHP",
        unit="kw",
        unit_desc="Per kW of heating capacity",
        capex_per_unit_usd=650.0,
        fixed_cost_usd=12000.0,
        lifespan_years=15,
        perf={"eff_old": 0.82, "eff_new": 3.2},
        notes="COP at 7°C/35°C; includes hydronics mods"
    )

    # HVAC - Chiller replacement per kW
    add_measure(
        mid="ME_HV_CHILLER_HIGH_EFF",
        name="Replace chiller with high-efficiency model",
        category="HVAC",
        target="Chiller",
        unit="kw",
        unit_desc="Per kW of cooling capacity",
        capex_per_unit_usd=400.0,
        fixed_cost_usd=15000.0,
        lifespan_years=25,
        perf={"eff_old": 3.5, "eff_new": 5.6},
        notes="EER to IPLV improvement"
    )

    # HVAC - ERV addition per cfm
    add_measure(
        mid="ME_HV_ADD_ERV",
        name="Add energy recovery ventilation",
        category="HVAC",
        target="Ventilation",
        unit="cfm",
        unit_desc="Per cfm of supply air",
        capex_per_unit_usd=3.2,
        fixed_cost_usd=8000.0,
        lifespan_years=20,
        perf={},
        notes="Plate or wheel ERV; 70% effectiveness typical"
    )

    # HVAC - VFDs on fans/pumps per kW motor
    add_measure(
        mid="ME_HV_VFD",
        name="Install VFDs on fans/pumps",
        category="HVAC",
        target="Motors",
        unit="kw",
        unit_desc="Per kW of motor rated power",
        capex_per_unit_usd=120.0,
        fixed_cost_usd=3000.0,
        lifespan_years=15,
        perf={},
        notes="Affinity law savings with proper control"
    )

    # Lighting - LED retrofit per m2
    add_measure(
        mid="ME_LT_LED",
        name="Retrofit lighting to LED",
        category="Lighting",
        target="Lighting",
        unit="m2",
        unit_desc="Per square meter of floor area",
        capex_per_unit_usd=22.0,
        fixed_cost_usd=0.0,
        lifespan_years=12,
        perf={"lpd_old": 12.0, "lpd_new": 7.0},
        notes="Includes new fixtures; typical office LPD"
    )

    # Lighting - Controls per m2
    add_measure(
        mid="ME_LT_CONTROLS",
        name="Add daylighting and occupancy controls",
        category="Lighting",
        target="Controls",
        unit="m2",
        unit_desc="Per square meter of floor area",
        capex_per_unit_usd=8.0,
        fixed_cost_usd=0.0,
        lifespan_years=12,
        perf={"ctrl_ltg_pct": 0.2},
        notes="20% savings on top of LED"
    )

    # PV and Storage per system
    for kw in [10, 50, 100]:
        add_measure(
            mid=f"ME_RE_PV_{kw}KW",
            name=f"Install {kw} kW solar PV",
            category="Renewables",
            target="PV",
            unit="system",
            unit_desc=f"One {kw} kW PV system",
            capex_per_unit_usd=kw * 1200.0,
            fixed_cost_usd=0.0,
            lifespan_years=25,
            perf={"pv_kw": kw},
            notes="Includes modules, racking, wiring"
        )

    add_measure(
        mid="ME_RE_BAT_50KWH",
        name="Install 50 kWh Li-ion battery",
        category="Renewables",
        target="Storage",
        unit="system",
        unit_desc="One 50 kWh battery system",
        capex_per_unit_usd=50 * 550.0,
        fixed_cost_usd=3000.0,
        lifespan_years=10,
        perf={"bat_kwh": 50},
        notes="Includes BMS and enclosure"
    )

    # DHW upgrades per kW
    add_measure(
        mid="ME_DHW_HPWH",
        name="Replace DHW with heat pump water heater",
        category="DHW",
        target="WaterHeater",
        unit="kw",
        unit_desc="Per kW of DHW capacity",
        capex_per_unit_usd=400.0,
        fixed_cost_usd=4000.0,
        lifespan_years=12,
        perf={"eff_old": 0.85, "eff_new": 2.8},
        notes="COP at standard conditions"
    )

    add_measure(
        mid="ME_DHW_HIGHEFF_GAS",
        name="Replace DHW with high-efficiency gas unit",
        category="DHW",
        target="WaterHeater",
        unit="kw",
        unit_desc="Per kW of DHW capacity",
        capex_per_unit_usd=220.0,
        fixed_cost_usd=2500.0,
        lifespan_years=15,
        perf={"eff_old": 0.8, "eff_new": 0.95},
        notes="Condensing gas tankless or storage"
    )

    # Controls / BMS
    add_measure(
        mid="ME_CTRL_BMS",
        name="Install building management system (BMS)",
        category="Controls",
        target="BMS",
        unit="m2",
        unit_desc="Per square meter of floor area",
        capex_per_unit_usd=10.0,
        fixed_cost_usd=15000.0,
        lifespan_years=15,
        perf={"ctrl_hvac_pct": 0.1},
        notes="Includes sensors; enables DR and optimization"
    )

    add_measure(
        mid="ME_CTRL_ADV_SEQ",
        name="Implement advanced control sequences",
        category="Controls",
        target="Sequences",
        unit="m2",
        unit_desc="Per square meter of floor area",
        capex_per_unit_usd=3.0,
        fixed_cost_usd=6000.0,
        lifespan_years=10,
        perf={"ctrl_hvac_pct": 0.08},
        notes="Optimal start/stop, reset strategies"
    )

    return measures


# ---------------------- Measure -> Materials Crosswalk ----------------------

def generate_measure_materials_crosswalk() -> List[Dict]:
    rows: List[Dict] = []

    def add_row(measure_id: str, material_id: str, quantity_per_unit: float, quantity_unit: str, notes: str = ""):
        rows.append({
            "measure_id": measure_id,
            "material_id": material_id,
            "quantity_per_unit": round(quantity_per_unit, 6),
            "quantity_unit": quantity_unit,
            "notes": notes,
        })

    # Roof insulation measures: assume unit is 1 m2; compute kg/m2 by thickness and density
    # Polyiso density ~32 kg/m3; inches to m = 0.0254
    insulation_map = {
        "POLYISO": ("MAT_POLYISO", 32),
        "XPS": ("MAT_XPS", 35),
        "MINERAL_WOOL": ("MAT_MINERAL_WOOL", 40),
    }

    for inches, mat in [(2, "POLYISO"), (4, "POLYISO"), (6, "POLYISO"), (4, "XPS"), (6, "XPS"), (6, "MINERAL_WOOL")]:
        thickness_m = inches * 0.0254
        mat_id, density = insulation_map[mat]
        mass_kg_per_m2 = density * thickness_m
        mid = f"ME_EN_ROOF_R{int(inches * (6.0 if mat == 'POLYISO' else (4.2 if mat=='XPS' else 3.8)) * 0.1761 * 10)}_{mat}"
        add_row(mid, mat_id, mass_kg_per_m2, "kg_per_m2", notes=f"{inches} inches")

    # Window upgrades: per m2 glazing area -> glass + aluminum frame
    for u_new, desc in [(1.8, "Double-pane low-E"), (1.2, "Triple-pane low-E")]:
        mid = f"ME_EN_WINDOW_U{int(u_new*10)}"
        # assume 8mm glass per lite; double = 2 lites -> 16mm; triple -> 24mm
        total_glass_thickness_m = 0.016 if u_new == 1.8 else 0.024
        glass_mass_per_m2 = 2500 * total_glass_thickness_m
        frame_mass_per_m2 = 5.0  # kg/m2 of aluminum
        add_row(mid, "MAT_GLASS", glass_mass_per_m2, "kg_per_m2", desc)
        add_row(mid, "MAT_ALUMINUM", frame_mass_per_m2, "kg_per_m2", "Frame/extrusions")
        # small PVC gaskets
        add_row(mid, "MAT_PVC", 0.8, "kg_per_m2", "Gaskets and spacers")

    # Shading devices: mostly aluminum per m2 glazing
    add_row("ME_EN_SHADING", "MAT_ALUMINUM", 6.0, "kg_per_m2", "Louver assemblies")

    # ERV: proxy steel and copper per cfm (small quantities per cfm)
    add_row("ME_HV_ADD_ERV", "MAT_STEEL_STRUCT", 0.002, "kg_per_cfm", "Casing and internals")
    add_row("ME_HV_ADD_ERV", "MAT_COPPER", 0.0002, "kg_per_cfm", "Coils and wiring")

    # VFDs: electronics in PVC and copper per kW
    add_row("ME_HV_VFD", "MAT_PVC", 0.2, "kg_per_kw", "Enclosure and plastics")
    add_row("ME_HV_VFD", "MAT_COPPER", 0.15, "kg_per_kw", "Wiring and windings")

    # Controls/BMS per m2: plastics and copper small quantities
    add_row("ME_CTRL_BMS", "MAT_PVC", 0.05, "kg_per_m2", "Sensors, cables")
    add_row("ME_CTRL_ADV_SEQ", "MAT_PVC", 0.02, "kg_per_m2", "Software/hardware minor")

    return rows


# ---------------------- Measure -> Systems Crosswalk ----------------------

def generate_measure_systems_crosswalk() -> List[Dict]:
    rows: List[Dict] = []

    def add_row(measure_id: str, system_id: str, quantity_per_unit: float, quantity_unit: str, notes: str = ""):
        rows.append({
            "measure_id": measure_id,
            "system_id": system_id,
            "quantity_per_unit": round(quantity_per_unit, 4),
            "quantity_unit": quantity_unit,
            "notes": notes,
        })

    # Boiler replacement per kW
    add_row("ME_HV_BOILER_COND", "SYS_GAS_BOILER_COND", 1.0, "kw", "Per kW capacity")

    # Electrification heat pump per kW
    add_row("ME_HV_ATW_HP", "SYS_AIR_TO_WATER_HP", 1.0, "kw", "Per kW capacity")

    # Chiller replacement per kW
    add_row("ME_HV_CHILLER_HIGH_EFF", "SYS_WATER_COOLED_CHILLER", 1.0, "kw", "Per kW capacity")

    # ERV per cfm
    add_row("ME_HV_ADD_ERV", "SYS_ERV", 1.0, "cfm", "Per cfm airflow")

    # LED measure: per m2 assume one fixture per 10 m2
    add_row("ME_LT_LED", "SYS_LED_FIXTURE", 0.1, "fixture_per_m2", "Fixture density")

    # PV measures: modules and inverter per kW
    for kw in [10, 50, 100]:
        mid = f"ME_RE_PV_{kw}KW"
        add_row(mid, "SYS_PV_MODULE", kw, "kw", "Nameplate capacity")
        add_row(mid, "SYS_PV_INVERTER", kw, "kw", "Inverter matched to array")

    # Battery measure: per kWh
    add_row("ME_RE_BAT_50KWH", "SYS_LI_ION_BATTERY", 50, "kwh_cap", "Declared capacity")

    return rows


# ---------------------- Applicability ----------------------

def generate_measure_applicability(measures: List[Dict]) -> List[Dict]:
    building_types = ["Office", "School", "Residential", "Hospital", "Retail", "Warehouse"]
    climate_zones = [f"{z}{l}" for z in range(1, 8) for l in ["A", "B", "C"]]

    rows: List[Dict] = []
    for m in measures:
        # Simple rule: most measures generally applicable; vary scores slightly
        for bt in building_types:
            for cz in climate_zones:
                score = 0.9
                if m["category"] == "Renewables" and cz.startswith("7"):
                    score = 0.75
                if m["category"] == "HVAC" and bt == "Warehouse":
                    score = 0.8
                score = round(max(0.0, min(1.0, score + random.uniform(-0.05, 0.05))), 2)
                rows.append({
                    "measure_id": m["measure_id"],
                    "building_type": bt,
                    "climate_zone": cz,
                    "applicability_score": score,
                })
    return rows


# ---------------------- Operational Carbon ----------------------

def generate_fuel_emission_factors() -> List[Dict]:
    return [
        {"fuel": "electricity", "unit": "kwh", "kgco2e_per_unit": 0.4, "notes": "Fallback average if hourly CI not used"},
        {"fuel": "natural_gas", "unit": "therm", "kgco2e_per_unit": 5.3, "notes": "Combustion CO2e only"},
        {"fuel": "fuel_oil_2", "unit": "gallon", "kgco2e_per_unit": 10.2, "notes": "Combustion CO2e only"},
        {"fuel": "propane", "unit": "gallon", "kgco2e_per_unit": 5.7, "notes": "Combustion CO2e only"},
    ]


def generate_grid_regions() -> List[Dict]:
    return [
        {"region_id": "R1_NYISO_NYC", "name": "NYISO NYC"},
        {"region_id": "R2_CAISO_LA", "name": "CAISO LA"},
        {"region_id": "R3_ERCOT_HOU", "name": "ERCOT Houston"},
    ]


def generate_grid_ci_hourly(regions: List[Dict], days: int = 30) -> List[Dict]:
    start = datetime(2024, 6, 1)
    rows: List[Dict] = []
    region_bases = {
        "R1_NYISO_NYC": 0.35,
        "R2_CAISO_LA": 0.25,
        "R3_ERCOT_HOU": 0.45,
    }
    for region in regions:
        rid = region["region_id"]
        base = region_bases[rid]
        for d in range(days):
            for h in range(24):
                t = start + timedelta(days=d, hours=h)
                # Diurnal profile: lower midday due to solar; higher evening ramp
                diurnal = 0.1 * (1 + -1 * ((h - 12) ** 2) / 144)  # parabola peaking low at noon
                diurnal = max(-0.08, min(0.1, diurnal))
                seasonal = 0.02 * random.uniform(-1, 1)
                noise = random.uniform(-0.015, 0.015)
                ci = max(0.05, base + diurnal + seasonal + noise)
                rows.append({
                    "region_id": rid,
                    "timestamp_utc": t.isoformat() + "Z",
                    "kgco2e_per_kwh": round(ci, 4)
                })
    return rows


def generate_energy_costs(regions: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for r in regions:
        if r["region_id"].startswith("R2"):  # CAISO
            elec = 0.19; gas = 1.30
        elif r["region_id"].startswith("R1"):  # NYISO
            elec = 0.21; gas = 1.45
        else:  # ERCOT
            elec = 0.12; gas = 1.10
        rows.append({
            "region_id": r["region_id"],
            "as_of_month": "2024-06",
            "electricity_usd_per_kwh": elec,
            "natural_gas_usd_per_therm": gas,
        })
    return rows


# ---------------------- Maintenance & Degradation ----------------------

def generate_maintenance(measures: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    def add(entity_type: str, entity_id: str, activity: str, interval_years: float, annual_cost_usd: float, notes: str = ""):
        rows.append({
            "entity_type": entity_type,
            "entity_id": entity_id,
            "activity": activity,
            "interval_years": interval_years,
            "annual_cost_usd": round(annual_cost_usd, 2),
            "notes": notes,
        })

    # Boilers
    add("system", "SYS_GAS_BOILER_COND", "Annual tune-up", 1.0, 0.0, "Typically modeled as OPEX, not CAPEX")
    # Heat pumps
    add("system", "SYS_AIR_TO_WATER_HP", "Refrigerant check", 2.0, 0.0, "Leak test and charge verification")
    # Chiller
    add("system", "SYS_WATER_COOLED_CHILLER", "Oil & filter", 1.0, 0.0, "Annual service")
    # ERV filters
    add("system", "SYS_ERV", "Filter replacement", 0.5, 0.0, "Every 6 months")
    # LED fixtures
    add("system", "SYS_LED_FIXTURE", "Cleaning & inspection", 2.0, 0.0, "")
    # PV inverter replacement
    add("system", "SYS_PV_INVERTER", "Inverter replacement", 12.0, 0.0, "Modeled as replacement at end of life")
    # Battery augmentation
    add("system", "SYS_LI_ION_BATTERY", "Capacity augmentation", 5.0, 0.0, "Offset capacity fade if needed")

    # Measures-specific commissioning
    for m in measures:
        if m["category"] in ("HVAC", "Controls"):
            add("measure", m["measure_id"], "Recommissioning", 5.0, 0.0, "Tune control sequences and setpoints")

    return rows


def generate_degradation() -> List[Dict]:
    return [
        {"entity_type": "system", "entity_id": "SYS_PV_MODULE", "metric": "output", "annual_change_fraction": -0.005, "model": "linear", "notes": "PV module degradation"},
        {"entity_type": "system", "entity_id": "SYS_PV_INVERTER", "metric": "efficiency", "annual_change_fraction": -0.002, "model": "linear", "notes": "Electronics ageing"},
        {"entity_type": "system", "entity_id": "SYS_LI_ION_BATTERY", "metric": "usable_capacity", "annual_change_fraction": -0.02, "model": "linear", "notes": "Calendar ageing"},
        {"entity_type": "measure", "entity_id": "ME_LT_LED", "metric": "lumen_output", "annual_change_fraction": -0.01, "model": "linear", "notes": "Lumen maintenance"},
        {"entity_type": "measure", "entity_id": "ME_EN_ROOF_R106_POLYISO", "metric": "r_value", "annual_change_fraction": -0.001, "model": "linear", "notes": "Insulation settling"},
        {"entity_type": "measure", "entity_id": "ME_EN_WINDOW_U12", "metric": "u_value", "annual_change_fraction": 0.0005, "model": "linear", "notes": "Seal deterioration"},
        {"entity_type": "measure", "entity_id": "ME_HV_ATW_HP", "metric": "cop", "annual_change_fraction": -0.002, "model": "linear", "notes": "Compressor wear"},
    ]


# ---------------------- Data Package Metadata ----------------------

def write_datapackage(resources: List[Dict]) -> None:
    meta = {
        "name": "retrofit_decision_engine_dataset",
        "title": "Retrofit Intervention & Lifecycle Data (Decision Engine)",
        "description": "Fabricated dataset for multi-objective retrofit optimization with LCI, operational carbon, measures, and maintenance.",
        "version": DATASET_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "sources": [
            {
                "title": "Typical ranges inspired by ICE database, Ecoinvent, and EPDs",
                "path": "https://www.ice.org.uk/knowledge-and-resources/briefing-sheet/ice-database"
            }
        ],
        "resources": resources,
    }
    with open(os.path.join(ROOT, "datapackage.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ---------------------- Main ----------------------

def main() -> None:
    ensure_dir(ROOT)

    # Generate core datasets
    materials = generate_materials_lci()
    systems = generate_systems_lci()
    measures = generate_measures()
    measure_materials = generate_measure_materials_crosswalk()
    measure_systems = generate_measure_systems_crosswalk()
    applicability = generate_measure_applicability(measures)

    fuel_eefs = generate_fuel_emission_factors()
    regions = generate_grid_regions()
    grid_ci = generate_grid_ci_hourly(regions, days=30)
    energy_costs = generate_energy_costs(regions)

    maintenance = generate_maintenance(measures)
    degradation = generate_degradation()

    # Write CSVs
    write_csv(os.path.join(ROOT, "materials_lci.csv"),
              ["material_id", "name", "category", "density_kg_per_m3", "gwp_a1a3_kgco2e_per_kg", "gwp_a1a5_kgco2e_per_kg", "gwp_b_kgco2e_per_kg", "gwp_c_kgco2e_per_kg", "source"],
              materials)

    write_csv(os.path.join(ROOT, "systems_lci.csv"),
              ["system_id", "name", "category", "unit", "gwp_a1a3_kgco2e_per_unit", "gwp_a1a5_kgco2e_per_unit", "lifetime_years_default", "source"],
              systems)

    write_csv(os.path.join(ROOT, "measures.csv"),
              [
                  "measure_id", "name", "category", "target", "unit", "unit_desc", "currency",
                  "capex_per_unit_usd", "fixed_cost_usd", "lifespan_years", "opex_annual_delta_usd",
                  "perf_u_value_old", "perf_u_value_new", "perf_r_value_old", "perf_r_value_new",
                  "perf_hvac_eff_old", "perf_hvac_eff_new", "perf_lpd_old_w_per_m2", "perf_lpd_new_w_per_m2",
                  "perf_infiltration_ach50_old", "perf_infiltration_ach50_new", "pv_nameplate_kw",
                  "battery_capacity_kwh", "control_savings_pct_hvac", "control_savings_pct_ltg", "notes"
              ],
              measures)

    write_csv(os.path.join(ROOT, "measure_materials_crosswalk.csv"),
              ["measure_id", "material_id", "quantity_per_unit", "quantity_unit", "notes"],
              measure_materials)

    write_csv(os.path.join(ROOT, "measure_systems_crosswalk.csv"),
              ["measure_id", "system_id", "quantity_per_unit", "quantity_unit", "notes"],
              measure_systems)

    write_csv(os.path.join(ROOT, "measure_applicability.csv"),
              ["measure_id", "building_type", "climate_zone", "applicability_score"],
              applicability)

    write_csv(os.path.join(ROOT, "fuel_emission_factors.csv"),
              ["fuel", "unit", "kgco2e_per_unit", "notes"],
              fuel_eefs)

    write_csv(os.path.join(ROOT, "grid_regions.csv"),
              ["region_id", "name"], regions)

    write_csv(os.path.join(ROOT, "grid_ci_hourly.csv"),
              ["region_id", "timestamp_utc", "kgco2e_per_kwh"], grid_ci)

    write_csv(os.path.join(ROOT, "energy_costs.csv"),
              ["region_id", "as_of_month", "electricity_usd_per_kwh", "natural_gas_usd_per_therm"],
              energy_costs)

    write_csv(os.path.join(ROOT, "maintenance.csv"),
              ["entity_type", "entity_id", "activity", "interval_years", "annual_cost_usd", "notes"],
              maintenance)

    write_csv(os.path.join(ROOT, "degradation.csv"),
              ["entity_type", "entity_id", "metric", "annual_change_fraction", "model", "notes"],
              degradation)

    # Datapackage metadata
    resources = [
        {"path": "materials_lci.csv", "name": "materials_lci"},
        {"path": "systems_lci.csv", "name": "systems_lci"},
        {"path": "measures.csv", "name": "measures"},
        {"path": "measure_materials_crosswalk.csv", "name": "measure_materials_crosswalk"},
        {"path": "measure_systems_crosswalk.csv", "name": "measure_systems_crosswalk"},
        {"path": "measure_applicability.csv", "name": "measure_applicability"},
        {"path": "fuel_emission_factors.csv", "name": "fuel_emission_factors"},
        {"path": "grid_regions.csv", "name": "grid_regions"},
        {"path": "grid_ci_hourly.csv", "name": "grid_ci_hourly"},
        {"path": "energy_costs.csv", "name": "energy_costs"},
        {"path": "maintenance.csv", "name": "maintenance"},
        {"path": "degradation.csv", "name": "degradation"},
    ]
    write_datapackage(resources)

    print(f"Wrote {len(resources)} CSV files to {ROOT}")


if __name__ == "__main__":
    main()
