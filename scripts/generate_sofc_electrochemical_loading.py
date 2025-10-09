#!/usr/bin/env python3
"""
Generate a fabricated SOFC electrochemical loading dataset with IV curves and overpotentials.
Outputs:
- electrochemical_loading.csv
- schema.json
- README.md (data dictionary)

No external dependencies required (standard library only).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Physical constants
FARADAY_C_PER_MOL = 96485.3329  # C/mol e-
GAS_R_J_PER_MOLK = 8.314462618  # J/(mol*K)


@dataclass
class Condition:
    condition_id: str
    temperature_C: float
    pH2_anode: float
    pH2O_anode: float
    pO2_cathode: float
    pressure_atm: float = 1.0

    @property
    def temperature_K(self) -> float:
        return self.temperature_C + 273.15

    @property
    def dryness(self) -> float:
        denom = max(self.pH2_anode + self.pH2O_anode, 1e-9)
        return self.pH2_anode / denom


# ---- Electrochemical models (simplified, plausible, not exact) ----

def compute_ocv_volts(cond: Condition) -> float:
    """Approximate reversible OCV for H2/H2O | O2 system.
    E_OCV ≈ 1.10 V + (RT/2F) * ln( (pO2_cathode^0.5) * (pH2/pH2O) )
    Clamp to a plausible range for high-T SOFC.
    """
    T = cond.temperature_K
    ratio = max(cond.pH2_anode, 1e-9) / max(cond.pH2O_anode, 1e-9)
    term = (GAS_R_J_PER_MOLK * T) / (2.0 * FARADAY_C_PER_MOL)
    e = 1.10 + term * math.log(max(cond.pO2_cathode, 1e-9) ** 0.5 * ratio)
    return max(0.8, min(1.2, e))


def compute_ohmic_asr_ohm_cm2(cond: Condition) -> float:
    """Simplified area-specific ohmic resistance vs temperature.
    Calibrated so ASR ≈ 0.30 Ω·cm^2 at 800 °C, slightly higher at 700 °C.
    """
    T_ref = 800.0 + 273.15
    T = cond.temperature_K
    asr_ref = 0.30  # Ω·cm^2 at 800C
    exponent = 1.2
    asr = asr_ref * (T_ref / T) ** exponent
    return max(0.15, min(0.8, asr))


def compute_exchange_current_density_A_per_cm2(cond: Condition) -> Tuple[float, float]:
    """Return (j0_anode, j0_cathode) in A/cm^2 as functions of T and partial pressures.
    Values are fabricated but plausible in order of magnitude.
    """
    T = cond.temperature_K
    # References at 800C
    T_ref = 800.0 + 273.15
    j0_an_ref = 0.20  # A/cm^2
    j0_ca_ref = 0.05  # A/cm^2
    Ea_an = 80_000.0  # J/mol
    Ea_ca = 120_000.0  # J/mol

    # Arrhenius-type temperature dependence
    j0_an_T = j0_an_ref * math.exp(-Ea_an / GAS_R_J_PER_MOLK * (1.0 / T - 1.0 / T_ref))
    j0_ca_T = j0_ca_ref * math.exp(-Ea_ca / GAS_R_J_PER_MOLK * (1.0 / T - 1.0 / T_ref))

    # Composition effects (increasing with pH2, decreasing with pH2O for anode; increasing with pO2 for cathode)
    comp_an = (max(cond.pH2_anode, 1e-9) ** 0.40) * (max(cond.pH2O_anode, 1e-9) ** -0.20)
    comp_ca = max(cond.pO2_cathode, 1e-9) ** 0.25

    j0_an = max(1e-5, min(5.0, j0_an_T * comp_an))
    j0_ca = max(1e-5, min(5.0, j0_ca_T * comp_ca))
    return j0_an, j0_ca


def compute_activation_overpotential_V(current_density, j0, T, alpha=0.5) -> float:
    """Butler–Volmer (symmetric) inversion: η = (RT/αF) asinh(j/(2 j0))."""
    if current_density <= 0.0:
        return 0.0
    return (GAS_R_J_PER_MOLK * T) / (alpha * FARADAY_C_PER_MOL) * math.asinh(current_density / (2.0 * max(j0, 1e-12)))


def compute_concentration_overpotential_V(current_density, j_lim, T) -> float:
    """Generic concentration overpotential: η_conc = -(RT/2F) ln(1 - j/j_lim).
    Returns 0 if current is negligible.
    """
    if current_density <= 0.0:
        return 0.0
    j_lim_eff = max(j_lim, 1e-6)
    frac = min(0.99, current_density / j_lim_eff)
    return -(GAS_R_J_PER_MOLK * T) / (2.0 * FARADAY_C_PER_MOL) * math.log(1.0 - frac)


def compute_limiting_current_A_per_cm2(cond: Condition) -> float:
    """Fabricated limiting current that increases with temperature and anode dryness.
    Typical range: 1.2 – 2.5 A/cm^2.
    """
    base = 1.6  # baseline at 700C, moderate dryness
    temp_factor = 0.7 + 0.6 * (cond.temperature_C - 700.0) / 150.0  # 700C->0.7, 850C->1.3
    dryness_factor = 0.6 + 0.6 * cond.dryness  # 0.6..1.2
    j_lim = base * temp_factor * dryness_factor
    return max(0.8, min(3.0, j_lim))


def compute_eis_resistances_ohm_cm2(j0_an, j0_ca, cond: Condition, alpha=0.5) -> Tuple[float, float, float]:
    """Return (R_ohmic, R_anode_ct, R_cathode_ct) as small-signal EIS parameters."""
    R_ohm = compute_ohmic_asr_ohm_cm2(cond)
    T = cond.temperature_K
    R_ct_an = (GAS_R_J_PER_MOLK * T) / (alpha * FARADAY_C_PER_MOL * max(j0_an, 1e-12))
    R_ct_ca = (GAS_R_J_PER_MOLK * T) / (alpha * FARADAY_C_PER_MOL * max(j0_ca, 1e-12))
    return R_ohm, R_ct_an, R_ct_ca


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_anode_oxidation_risk(eta_an_anode: float, cond: Condition) -> float:
    """Heuristic risk score (0..1) for Ni -> NiO oxidation risk due to local potentials and steam.
    Increases with anode activation overpotential and with ln(pH2O/pH2), decreases with temperature.
    """
    dryness_ratio = math.log(max(cond.pH2O_anode, 1e-9) / max(cond.pH2_anode, 1e-9))
    x = 2.0 * eta_an_anode + 0.5 * dryness_ratio + 0.002 * (800.0 - cond.temperature_C)
    return max(0.0, min(1.0, sigmoid(x)))


def generate_conditions(seed: int = 7) -> List[Condition]:
    random.seed(seed)
    temperatures_C = [700, 750, 800, 850]
    fuel_pairs = [(0.90, 0.10), (0.70, 0.30), (0.50, 0.50)]  # (pH2, pH2O)
    cathode_pO2 = [0.21, 1.00]

    conditions: List[Condition] = []
    idx = 1
    for T in temperatures_C:
        for (pH2, pH2O) in fuel_pairs:
            for pO2 in cathode_pO2:
                cid = f"T{int(T)}_H2{int(pH2*100)}_H2O{int(pH2O*100)}_O2{int(pO2*100)}"
                conditions.append(
                    Condition(
                        condition_id=cid,
                        temperature_C=T,
                        pH2_anode=pH2,
                        pH2O_anode=pH2O,
                        pO2_cathode=pO2,
                        pressure_atm=1.0,
                    )
                )
                idx += 1
    return conditions


def linspace(start: float, stop: float, num: int) -> List[float]:
    if num <= 1:
        return [stop]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def fabricate_dataset(output_dir: str, seed: int = 7) -> Tuple[str, str, str]:
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    conditions = generate_conditions(seed)

    csv_path = os.path.join(output_dir, "electrochemical_loading.csv")

    # Define CSV columns and write header
    fieldnames = [
        "condition_id",
        "temperature_C",
        "pH2_anode",
        "pH2O_anode",
        "pO2_cathode",
        "pressure_atm",
        "current_density_A_per_cm2",
        "voltage_V",
        "ocv_V",
        "overpotential_ohmic_V",
        "overpotential_activation_anode_V",
        "overpotential_activation_cathode_V",
        "overpotential_concentration_V",
        "total_overpotential_V",
        "R_ohmic_ohm_cm2",
        "R_ct_anode_ohm_cm2",
        "R_ct_cathode_ohm_cm2",
        "R_total_ohm_cm2",
        "f_peak_anode_Hz",
        "f_peak_cathode_Hz",
        "delta_mu_O2_J_per_mol",
        "anode_oxidation_risk_score",
        "anode_ni_oxidation_flag",
    ]

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cond in conditions:
            T = cond.temperature_K
            ocv = compute_ocv_volts(cond)
            asr_ohm = compute_ohmic_asr_ohm_cm2(cond)
            j0_an, j0_ca = compute_exchange_current_density_A_per_cm2(cond)
            R_ohm, R_ct_an, R_ct_ca = compute_eis_resistances_ohm_cm2(j0_an, j0_ca, cond)

            Cdl_an = 1.0  # F/cm^2
            Cdl_ca = 0.5  # F/cm^2
            f_peak_an = 1.0 / max(1e-9, (2.0 * math.pi * R_ct_an * Cdl_an))
            f_peak_ca = 1.0 / max(1e-9, (2.0 * math.pi * R_ct_ca * Cdl_ca))

            j_lim = compute_limiting_current_A_per_cm2(cond)
            j_max = 0.90 * j_lim
            for j in linspace(0.0, j_max, 25):
                eta_ohm = j * asr_ohm
                eta_act_an = compute_activation_overpotential_V(j, j0_an, T)
                eta_act_ca = compute_activation_overpotential_V(j, j0_ca, T)
                eta_conc = compute_concentration_overpotential_V(j, j_lim, T)

                total_overpot = eta_ohm + eta_act_an + eta_act_ca + eta_conc
                voltage = max(0.2, ocv - total_overpot)

                # Chemical potential difference for O2 across electrolyte (magnitude): Δμ_O2 ≈ 4 F E
                delta_mu_O2 = 4.0 * FARADAY_C_PER_MOL * voltage

                # Anode oxidation risk
                risk = compute_anode_oxidation_risk(eta_act_an, cond)
                flag = 1 if risk >= 0.6 else 0

                writer.writerow({
                    "condition_id": cond.condition_id,
                    "temperature_C": round(cond.temperature_C, 3),
                    "pH2_anode": round(cond.pH2_anode, 5),
                    "pH2O_anode": round(cond.pH2O_anode, 5),
                    "pO2_cathode": round(cond.pO2_cathode, 5),
                    "pressure_atm": round(cond.pressure_atm, 5),
                    "current_density_A_per_cm2": round(j, 6),
                    "voltage_V": round(voltage, 6),
                    "ocv_V": round(ocv, 6),
                    "overpotential_ohmic_V": round(eta_ohm, 6),
                    "overpotential_activation_anode_V": round(eta_act_an, 6),
                    "overpotential_activation_cathode_V": round(eta_act_ca, 6),
                    "overpotential_concentration_V": round(eta_conc, 6),
                    "total_overpotential_V": round(total_overpot, 6),
                    "R_ohmic_ohm_cm2": round(R_ohm, 6),
                    "R_ct_anode_ohm_cm2": round(R_ct_an, 6),
                    "R_ct_cathode_ohm_cm2": round(R_ct_ca, 6),
                    "R_total_ohm_cm2": round(R_ohm + R_ct_an + R_ct_ca, 6),
                    "f_peak_anode_Hz": round(f_peak_an, 6),
                    "f_peak_cathode_Hz": round(f_peak_ca, 6),
                    "delta_mu_O2_J_per_mol": round(delta_mu_O2, 3),
                    "anode_oxidation_risk_score": round(risk, 3),
                    "anode_ni_oxidation_flag": flag,
                })

    # Write schema JSON
    schema = {
        "name": "SOFC Electrochemical Loading Dataset (Fabricated)",
        "description": "Fabricated dataset of IV curves, overpotentials, and EIS-derived parameters for SOFC across operating conditions.",
        "rows": "~600",
        "columns": [
            {"name": "condition_id", "type": "string", "unit": "-", "description": "Identifier encoding T, gas fractions"},
            {"name": "temperature_C", "type": "number", "unit": "C", "description": "Operating temperature"},
            {"name": "pH2_anode", "type": "number", "unit": "bar fraction", "description": "Anode H2 partial pressure fraction"},
            {"name": "pH2O_anode", "type": "number", "unit": "bar fraction", "description": "Anode H2O partial pressure fraction"},
            {"name": "pO2_cathode", "type": "number", "unit": "bar fraction", "description": "Cathode O2 partial pressure fraction"},
            {"name": "pressure_atm", "type": "number", "unit": "atm", "description": "Total pressure"},
            {"name": "current_density_A_per_cm2", "type": "number", "unit": "A/cm^2", "description": "Applied current density"},
            {"name": "voltage_V", "type": "number", "unit": "V", "description": "Operating cell voltage under load"},
            {"name": "ocv_V", "type": "number", "unit": "V", "description": "Open-circuit voltage (approximate)"},
            {"name": "overpotential_ohmic_V", "type": "number", "unit": "V", "description": "Ohmic loss (j * ASR)"},
            {"name": "overpotential_activation_anode_V", "type": "number", "unit": "V", "description": "Anode activation overpotential (Butler–Volmer)"},
            {"name": "overpotential_activation_cathode_V", "type": "number", "unit": "V", "description": "Cathode activation overpotential (Butler–Volmer)"},
            {"name": "overpotential_concentration_V", "type": "number", "unit": "V", "description": "Concentration/mass-transport overpotential"},
            {"name": "total_overpotential_V", "type": "number", "unit": "V", "description": "Sum of overpotentials"},
            {"name": "R_ohmic_ohm_cm2", "type": "number", "unit": "Ω·cm^2", "description": "Ohmic resistance (EIS)"},
            {"name": "R_ct_anode_ohm_cm2", "type": "number", "unit": "Ω·cm^2", "description": "Anode charge-transfer resistance (EIS)"},
            {"name": "R_ct_cathode_ohm_cm2", "type": "number", "unit": "Ω·cm^2", "description": "Cathode charge-transfer resistance (EIS)"},
            {"name": "R_total_ohm_cm2", "type": "number", "unit": "Ω·cm^2", "description": "Total small-signal resistance (sum)"},
            {"name": "f_peak_anode_Hz", "type": "number", "unit": "Hz", "description": "Anode semicircle peak frequency (estimate)"},
            {"name": "f_peak_cathode_Hz", "type": "number", "unit": "Hz", "description": "Cathode semicircle peak frequency (estimate)"},
            {"name": "delta_mu_O2_J_per_mol", "type": "number", "unit": "J/mol O2", "description": "Oxygen chemical potential difference (≈ 4 F V)"},
            {"name": "anode_oxidation_risk_score", "type": "number", "unit": "0..1", "description": "Heuristic Ni→NiO oxidation risk score"},
            {"name": "anode_ni_oxidation_flag", "type": "integer", "unit": "{0,1}", "description": "1 if risk ≥ 0.6"},
        ],
        "notes": [
            "Fabricated for modeling/demo; not experimental data.",
            "OCV formula and kinetics are simplified; constants tuned for plausibility.",
            "EIS parameters derived from simple relations; not fitted to spectra.",
        ],
        "provenance": {
            "generator": "scripts/generate_sofc_electrochemical_loading.py",
            "seed": seed,
        },
    }

    schema_path = os.path.join(output_dir, "schema.json")
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    # Write README (data dictionary brief)
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# SOFC Electrochemical Loading Dataset (Fabricated)\n\n")
        f.write("This dataset contains fabricated IV and overpotential data across SOFC operating conditions.\n\n")
        f.write("## Files\n- `electrochemical_loading.csv`: main data\n- `schema.json`: column schema and units\n\n")
        f.write("## Key Columns\n")
        f.write("- `current_density_A_per_cm2`: Applied current density.\n")
        f.write("- `voltage_V`: Operating voltage under load.\n")
        f.write("- `ocv_V`: Open-circuit voltage (approximate).\n")
        f.write("- `overpotential_*_V`: Components (ohmic, activation anode/cathode, concentration).\n")
        f.write("- `R_*_ohm_cm2`: EIS-derived resistances (ohmic, charge-transfer).\n")
        f.write("- `delta_mu_O2_J_per_mol`: Oxygen chemical potential difference across electrolyte (≈ 4 F V).\n")
        f.write("- `anode_oxidation_risk_score` / `anode_ni_oxidation_flag`: Heuristic risk indicators.\n\n")
        f.write("## Assumptions\n")
        f.write("- Simplified Nernst, Butler–Volmer, and mass-transport models tuned for plausibility.\n")
        f.write("- Parameters chosen to yield realistic magnitudes at 700–850 °C.\n")
        f.write("- EIS peak frequencies assume constant double-layer capacitances.\n")

    return csv_path, schema_path, readme_path


def main():
    parser = argparse.ArgumentParser(description="Generate fabricated SOFC electrochemical loading dataset")
    parser.add_argument("--output-dir", default=os.path.join(os.getcwd(), "data", "sofc", "electrochemical_loading"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    csv_path, schema_path, readme_path = fabricate_dataset(args.output_dir, seed=args.seed)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {schema_path}")
    print(f"Wrote: {readme_path}")


if __name__ == "__main__":
    main()
