#!/usr/bin/env python3
from __future__ import annotations
import math
import csv
from pathlib import Path
from typing import List, Dict, Tuple

# Synthetic SOFC Electrochemical Dataset Generator
# - IV curves across current density for multiple temperatures and steam fractions
# - Overpotential breakdown (anode activation, cathode activation, ohmic, concentration)
# - EIS parameters (R_ohmic, R_ct anode, R_ct cathode, C's, Warburg sigma)
#
# Outputs:
#   data/sofc/iv_curves.csv
#   data/sofc/eis_params.csv

R_GAS = 8.31446261815324  # J/mol-K
FARADAY = 96485.33212      # C/mol

# Reference conditions
TREF_K = 1073.15  # 800 °C
E0_298K = 1.229   # V (approx standard reversible potential at 25 C for H2/O2)
D_E0_dT = -8.5e-4 # V/K linearized temperature dependence

# Kinetic parameters (synthetic but plausible order of magnitude)
I0_ANODE_REF = 1.0e-2   # A/cm^2 at TREF
I0_CATHODE_REF = 2.0e-3 # A/cm^2 at TREF
EA_ANODE = 8.0e4        # J/mol
EA_CATHODE = 9.0e4      # J/mol
ALPHA_ANODE = 0.5
ALPHA_CATHODE = 0.5

# Ohmic (area-specific resistance, ASR)
ASR_REF = 0.25    # ohm*cm^2 at TREF
EA_OHM = 6.0e4    # J/mol (Arrhenius-like temperature dependence)

# Mass transport
IL_REF = 2.5      # A/cm^2 at TREF and high H2 fraction
N_ELECTRONS_H2 = 2  # electrons per H2 in half-reaction used in concentration overpotential term

# EIS: baseline capacitances and Warburg coefficient (synthetic)
C_ANODE_REF = 1.0e-2   # F/cm^2 at TREF
C_CATHODE_REF = 8.0e-3 # F/cm^2 at TREF
SIGMA_W_REF = 0.2      # ohm*s^0.5*cm^2 (infinite Warburg coefficient magnitude)


def arrhenius_property(value_ref: float, ea_j_per_mol: float, t_k: float, tref_k: float, inverse: bool = False) -> float:
    """Arrhenius-like temperature scaling.
    If inverse=True, property decreases with T (e.g., resistivity/ASR).
    """
    exponent = (1.0 / t_k) - (1.0 / tref_k)
    if inverse:
        # Property decreases as T increases
        return value_ref * math.exp(ea_j_per_mol / R_GAS * exponent)
    # Property increases as T increases
    return value_ref * math.exp(-ea_j_per_mol / R_GAS * exponent)


def reversible_potential_h2_o2(t_k: float, p_h2: float, p_h2o: float, p_o2_cathode: float) -> float:
    """Approximate Nernst potential for H2 + 1/2 O2 -> H2O.
    E(T) = E0(T) + (RT/2F) * ln(p_H2 * p_O2^(1/2) / p_H2O)
    """
    p_h2 = max(p_h2, 1e-9)
    p_h2o = max(p_h2o, 1e-9)
    p_o2_cathode = max(p_o2_cathode, 1e-9)
    e0_t = E0_298K + D_E0_dT * (t_k - 298.15)
    ln_term = math.log(p_h2 * math.sqrt(p_o2_cathode) / p_h2o)
    return e0_t + (R_GAS * t_k) / (2.0 * FARADAY) * ln_term


def exchange_current_density_anode(t_k: float) -> float:
    return arrhenius_property(I0_ANODE_REF, EA_ANODE, t_k, TREF_K, inverse=False)


def exchange_current_density_cathode(t_k: float) -> float:
    return arrhenius_property(I0_CATHODE_REF, EA_CATHODE, t_k, TREF_K, inverse=False)


def area_specific_resistance(t_k: float) -> float:
    return arrhenius_property(ASR_REF, EA_OHM, t_k, TREF_K, inverse=True)


def limiting_current_density(t_k: float, p_h2: float, p_h2o: float) -> float:
    # Simple dependence: scales with temperature and dry H2 fraction
    h2_total = max(p_h2 + p_h2o, 1e-9)
    h2_fraction = p_h2 / h2_total
    scale_t = t_k / TREF_K
    return max(0.5, IL_REF * scale_t * (0.4 + 0.6 * h2_fraction))


def activation_overpotential(i_a_cm2: float, i0: float, t_k: float, alpha: float) -> float:
    # Butler-Volmer inversion (symmetric): eta = (RT/alpha F) asinh(i / (2 i0))
    return (R_GAS * t_k) / (alpha * FARADAY) * math.asinh(max(i_a_cm2, 0.0) / (2.0 * max(i0, 1e-20)))


def concentration_overpotential(i_a_cm2: float, i_lim: float, t_k: float, n_electrons: int) -> float:
    # η_conc = (RT / nF) * ln(1 / (1 - i / i_L)) = (RT / nF) * ln(1 - i / i_L)^(-1)
    # Clamp to avoid singularity near i = i_L
    i_fraction = min(max(i_a_cm2 / max(i_lim, 1e-9), 0.0), 0.999999)
    return (R_GAS * t_k) / (n_electrons * FARADAY) * math.log(1.0 / (1.0 - i_fraction))


def r_ct_from_bv(i_a_cm2: float, i0: float, t_k: float, alpha: float) -> float:
    # Differential (small-signal) charge-transfer resistance from BV at operating point
    # dη/di = (RT / αF) * 1 / [sqrt((i / (2 i0))^2 + 1) * 2 i0]
    denom = math.sqrt((max(i_a_cm2, 0.0) / (2.0 * max(i0, 1e-30))) ** 2 + 1.0)
    return (R_GAS * t_k) / (alpha * FARADAY) * 1.0 / (denom * 2.0 * max(i0, 1e-30))


def capacitance_anode(t_k: float, steam_frac: float) -> float:
    # Slight increase with steam fraction, slight decrease with T
    temp_scale = math.sqrt(TREF_K / t_k)
    steam_scale = 1.0 + 0.5 * (steam_frac - 0.2)
    return max(1e-4, C_ANODE_REF * temp_scale * steam_scale)


def capacitance_cathode(t_k: float, steam_frac: float) -> float:
    temp_scale = math.sqrt(TREF_K / t_k)
    steam_scale = 1.0 + 0.2 * (steam_frac - 0.2)
    return max(1e-4, C_CATHODE_REF * temp_scale * steam_scale)


def warburg_sigma(t_k: float, p_h2: float, p_h2o: float) -> float:
    # Diffusion severity increases with steam and decreases with temperature
    temp_scale = math.sqrt(TREF_K / t_k)
    h2_total = max(p_h2 + p_h2o, 1e-9)
    steam_frac = p_h2o / h2_total
    steam_scale = 0.8 + 1.2 * steam_frac
    return max(0.01, SIGMA_W_REF * temp_scale * steam_scale)


def anode_oxidation_risk_index(eta_anode_v: float, steam_frac: float) -> float:
    # Heuristic risk index in [0, 1]
    # Higher with larger anode overpotential and steam fraction
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))
    score = sigmoid((eta_anode_v - 0.15) / 0.05) * sigmoid((steam_frac - 0.30) / 0.05)
    return max(0.0, min(1.0, score))


def generate_iv_and_eis() -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    temperatures_c = [650.0, 700.0, 750.0, 800.0]
    steam_fracs = [0.05, 0.20, 0.40]  # anode gas: H2/H2O only for simplicity
    p_o2_cathode = 0.21

    iv_rows: List[Dict[str, float]] = []
    eis_rows: List[Dict[str, float]] = []

    test_id = 0
    for t_c in temperatures_c:
        t_k = t_c + 273.15
        i0_a = exchange_current_density_anode(t_k)
        i0_c = exchange_current_density_cathode(t_k)
        asr = area_specific_resistance(t_k)

        for steam_frac in steam_fracs:
            test_id += 1
            p_h2o = steam_frac
            p_h2 = max(1.0 - p_h2o, 1e-9)

            e_eq = reversible_potential_h2_o2(t_k, p_h2=p_h2, p_h2o=p_h2o, p_o2_cathode=p_o2_cathode)
            i_lim = limiting_current_density(t_k, p_h2=p_h2, p_h2o=p_h2o)

            # Current density sweep up to 90% of i_lim, capped at 1.6 A/cm^2
            i_max = min(1.6, 0.90 * i_lim)
            n_points = 48
            if i_max < 0.05:
                # Skip impractical condition
                continue
            step = i_max / (n_points - 1)

            for k in range(n_points):
                i = k * step  # A/cm^2

                eta_a = activation_overpotential(i, i0=i0_a, t_k=t_k, alpha=ALPHA_ANODE)
                eta_c = activation_overpotential(i, i0=i0_c, t_k=t_k, alpha=ALPHA_CATHODE)
                eta_ohm = i * asr
                eta_conc = concentration_overpotential(i, i_lim=i_lim, t_k=t_k, n_electrons=N_ELECTRONS_H2)

                total_overpot = eta_a + eta_c + eta_ohm + eta_conc
                v_cell = max(e_eq - total_overpot, 0.0)

                # Oxygen chemical potential drop magnitude across electrolyte ~ nF E for O2 (n=4)
                delta_mu_o2_j_per_mol = 4.0 * FARADAY * max(e_eq, 0.0)
                risk_idx = anode_oxidation_risk_index(eta_a, steam_frac=p_h2o)

                iv_rows.append({
                    "test_id": test_id,
                    "temperature_C": t_c,
                    "anode_H2_frac": p_h2,
                    "anode_H2O_frac": p_h2o,
                    "cathode_O2_frac": p_o2_cathode,
                    "current_density_A_cm2": i,
                    "cell_voltage_V": v_cell,
                    "E_eq_V": e_eq,
                    "overpotential_anode_V": eta_a,
                    "overpotential_cathode_V": eta_c,
                    "overpotential_ohmic_V": eta_ohm,
                    "overpotential_concentration_V": eta_conc,
                    "overpotential_total_V": total_overpot,
                    "oxygen_mu_drop_J_per_mol_O2": delta_mu_o2_j_per_mol,
                    "anode_oxidation_risk_index": risk_idx,
                })

            # EIS parameters at representative loads (0.1, 0.5, 0.9 of i_max, clipped to i_lim)
            for frac in (0.10, 0.50, 0.90):
                i_oper = min(frac * i_max, 0.95 * i_lim)
                r_ct_a = r_ct_from_bv(i_oper, i0=i0_a, t_k=t_k, alpha=ALPHA_ANODE)
                r_ct_c = r_ct_from_bv(i_oper, i0=i0_c, t_k=t_k, alpha=ALPHA_CATHODE)

                c_a = capacitance_anode(t_k, steam_frac=p_h2o)
                c_c = capacitance_cathode(t_k, steam_frac=p_h2o)
                sigma_w = warburg_sigma(t_k, p_h2=p_h2, p_h2o=p_h2o)

                eis_rows.append({
                    "test_id": test_id,
                    "temperature_C": t_c,
                    "anode_H2_frac": p_h2,
                    "anode_H2O_frac": p_h2o,
                    "cathode_O2_frac": p_o2_cathode,
                    "current_density_A_cm2": i_oper,
                    "R_ohmic_Ohm_cm2": asr,
                    "R_ct_anode_Ohm_cm2": r_ct_a,
                    "R_ct_cathode_Ohm_cm2": r_ct_c,
                    "C_anode_F_cm2": c_a,
                    "C_cathode_F_cm2": c_c,
                    "Warburg_sigma_Ohm_s05_cm2": sigma_w,
                    "R_total_Ohm_cm2": asr + r_ct_a + r_ct_c,
                })

    return iv_rows, eis_rows


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "data" / "sofc"
    iv_rows, eis_rows = generate_iv_and_eis()

    iv_fields = [
        "test_id",
        "temperature_C",
        "anode_H2_frac",
        "anode_H2O_frac",
        "cathode_O2_frac",
        "current_density_A_cm2",
        "cell_voltage_V",
        "E_eq_V",
        "overpotential_anode_V",
        "overpotential_cathode_V",
        "overpotential_ohmic_V",
        "overpotential_concentration_V",
        "overpotential_total_V",
        "oxygen_mu_drop_J_per_mol_O2",
        "anode_oxidation_risk_index",
    ]

    eis_fields = [
        "test_id",
        "temperature_C",
        "anode_H2_frac",
        "anode_H2O_frac",
        "cathode_O2_frac",
        "current_density_A_cm2",
        "R_ohmic_Ohm_cm2",
        "R_ct_anode_Ohm_cm2",
        "R_ct_cathode_Ohm_cm2",
        "C_anode_F_cm2",
        "C_cathode_F_cm2",
        "Warburg_sigma_Ohm_s05_cm2",
        "R_total_Ohm_cm2",
    ]

    write_csv(out_dir / "iv_curves.csv", iv_fields, iv_rows)
    write_csv(out_dir / "eis_params.csv", eis_fields, eis_rows)

    print(f"Wrote {len(iv_rows)} IV rows to {out_dir / 'iv_curves.csv'}")
    print(f"Wrote {len(eis_rows)} EIS rows to {out_dir / 'eis_params.csv'}")


if __name__ == "__main__":
    main()
