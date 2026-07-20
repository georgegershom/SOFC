"""
generate_datasets.py
====================
Reproducibly generates all synthetic datasets, figures, and the zip archive
for the study:

  "Physics-informed 3D deep learning for thermo-mechanical lifetime prediction
   of short-stack reversible solid oxide cells cycled at 600–850 °C in
   nuclear battery service."

All data are SYNTHETIC / SIMULATED for research-scaffolding purposes.
They follow physically plausible trends (Arrhenius, Norton creep, Butler-Volmer,
Weibull statistics, linear CTE, etc.) with added Gaussian noise, but do NOT
represent real experimental measurements.

Dependencies: numpy, pandas, matplotlib, scipy
Run: python scripts/generate_datasets.py
"""

import os
import sys
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import weibull_min
from scipy.optimize import curve_fit

# ── reproducibility ────────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ── output root ───────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "datasets")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def _save(df, *path_parts):
    path = os.path.join(ROOT, *path_parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  saved {os.path.relpath(path, ROOT)}")

def _fig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved figures/{name}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – Half-Cell Thermo-Mechanical Characterisation
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Section 1: Half-Cell ──")

MATERIALS = ["YSZ", "Ni-YSZ", "GDC", "LSCF-GDC", "Crofer22APU", "glass-ceramic"]

# Reference CTE values at 25 °C (ppm/K) – literature midpoints
CTE_REF = {
    "YSZ":            10.5,
    "Ni-YSZ":         12.5,
    "GDC":            12.0,
    "LSCF-GDC":       14.5,
    "Crofer22APU":    11.5,
    "glass-ceramic":   9.5,
}
# Linear CTE slope (ppm/K per 100 °C)
CTE_SLOPE = {
    "YSZ":            0.015,
    "Ni-YSZ":         0.020,
    "GDC":            0.018,
    "LSCF-GDC":       0.030,
    "Crofer22APU":    0.010,
    "glass-ceramic":  0.025,
}
N_SPEC = 10          # specimens per material
T_CTE = np.arange(25, 925, 25)     # °C

rows_cte = []
for mat in MATERIALS:
    for spec in range(1, N_SPEC + 1):
        mean_cte = CTE_REF[mat] + CTE_SLOPE[mat] * (T_CTE - 25) / 100
        noise    = RNG.normal(0, 0.08, len(T_CTE))
        cte_val  = mean_cte + noise
        for T, cte, mn, st in zip(T_CTE, cte_val, mean_cte, np.full(len(T_CTE), 0.08)):
            rows_cte.append(dict(material=mat, specimen=spec,
                                 temperature_C=T, cte_ppm_K=round(float(cte), 4),
                                 cte_mean=round(float(mn), 4),
                                 cte_std=round(float(st), 4)))

df_cte = pd.DataFrame(rows_cte)
_save(df_cte, "1_half_cell", "cte_vs_temperature.csv")

# ── Chemical expansion vs pO₂ ─────────────────────────────────────────────────
pO2_vals = np.array([1e-5, 1e-4, 5e-4, 1e-3, 0.01, 0.1, 0.21])   # atm
mats_chem = ["Ni-YSZ", "YSZ"]
rows_chem = []
for mat in mats_chem:
    # strain ≈ A * log10(pO2) + B (chemical expansion decreases as pO2 rises)
    A = -0.0012 if mat == "Ni-YSZ" else -0.0003
    B = -0.002  if mat == "Ni-YSZ" else -0.0005
    for spec in range(1, N_SPEC + 1):
        strain = A * np.log10(pO2_vals) + B + RNG.normal(0, 1e-4, len(pO2_vals))
        for p, s in zip(pO2_vals, strain):
            rows_chem.append(dict(material=mat, specimen=spec,
                                  pO2_atm=float(p),
                                  chemical_strain=round(float(s), 6)))

df_chem = pd.DataFrame(rows_chem)
_save(df_chem, "1_half_cell", "chemical_expansion_vs_pO2.csv")

# ── Elastic moduli vs T ───────────────────────────────────────────────────────
E_ref = {"YSZ":200, "Ni-YSZ":150, "GDC":175, "LSCF-GDC":130,
         "Crofer22APU":200, "glass-ceramic":70}
T_mech = np.arange(25, 925, 50)
rows_mod = []
for mat in MATERIALS:
    for spec in range(1, N_SPEC + 1):
        E = E_ref[mat] * (1 - 5e-5 * (T_mech - 25))
        G = E / (2 * (1 + 0.28))
        nu = np.full(len(T_mech), 0.28) + RNG.normal(0, 0.005, len(T_mech))
        noise_E = RNG.normal(0, E_ref[mat] * 0.02, len(T_mech))
        for T, e, g, n in zip(T_mech, E + noise_E, G, nu):
            rows_mod.append(dict(material=mat, specimen=spec, temperature_C=T,
                                 E_GPa=round(float(e), 2),
                                 G_GPa=round(float(g), 2),
                                 poisson=round(float(n), 3)))

df_mod = pd.DataFrame(rows_mod)
_save(df_mod, "1_half_cell", "elastic_moduli_vs_temperature.csv")

# ── Creep parameters (Norton law: ε̇ = A σ^n exp(-Q/RT)) ─────────────────────
CREEP_T   = [650, 750, 850]          # °C
CREEP_SIG = [5, 10, 20]             # MPa
R_GAS     = 8.314
# Norton parameters per material
CREEP_PARAMS = {
    "YSZ":          dict(A=1e-14, n=2.0, Q=300e3),
    "Ni-YSZ":       dict(A=5e-14, n=2.2, Q=280e3),
    "GDC":          dict(A=2e-14, n=2.1, Q=290e3),
    "LSCF-GDC":     dict(A=8e-14, n=2.3, Q=260e3),
    "Crofer22APU":  dict(A=3e-15, n=1.8, Q=320e3),
    "glass-ceramic":dict(A=1e-13, n=2.5, Q=240e3),
}
rows_creep = []
for mat in MATERIALS:
    p = CREEP_PARAMS[mat]
    for T_c in CREEP_T:
        T_K = T_c + 273.15
        for sig in CREEP_SIG:
            edot = p["A"] * sig**p["n"] * np.exp(-p["Q"] / (R_GAS * T_K))
            noise = RNG.normal(0, edot * 0.05)
            rows_creep.append(dict(material=mat, temperature_C=T_c,
                                   stress_MPa=sig,
                                   creep_strain_rate_per_s=float(edot + noise),
                                   norton_A=p["A"], norton_n=p["n"],
                                   norton_Q_J_mol=p["Q"]))

df_creep = pd.DataFrame(rows_creep)
_save(df_creep, "1_half_cell", "creep_parameters.csv")

# ── Weibull fracture strength ─────────────────────────────────────────────────
T_WEIB = [25, 600, 750, 850]
WEIB_PARAMS = {
    "YSZ":          dict(m=12, s0=300),
    "Ni-YSZ":       dict(m=8,  s0=200),
    "GDC":          dict(m=10, s0=250),
    "LSCF-GDC":     dict(m=7,  s0=180),
    "Crofer22APU":  dict(m=15, s0=450),
    "glass-ceramic":dict(m=6,  s0=100),
}
rows_weib = []
for mat in MATERIALS:
    p = WEIB_PARAMS[mat]
    for T_w in T_WEIB:
        # characteristic strength decreases with T
        s0_T = p["s0"] * (1 - 1.5e-4 * (T_w - 25))
        m    = p["m"]  * (1 - 5e-5  * (T_w - 25))
        strengths = weibull_min.rvs(m, scale=s0_T, size=N_SPEC,
                                    random_state=int(RNG.integers(1e6)))
        for spec, s in enumerate(strengths, 1):
            rows_weib.append(dict(material=mat, temperature_C=T_w,
                                  specimen=spec,
                                  fracture_strength_MPa=round(float(s), 2),
                                  weibull_modulus=round(float(m), 2),
                                  characteristic_strength_MPa=round(float(s0_T), 2)))

df_weib = pd.DataFrame(rows_weib)
_save(df_weib, "1_half_cell", "weibull_fracture_strength.csv")

# ── Interfacial fracture energy ───────────────────────────────────────────────
T_IFE = np.arange(600, 875, 50)
INTERFACES = ["anode_electrolyte", "electrolyte_cathode"]
IFE_REF = {"anode_electrolyte": 5.0, "electrolyte_cathode": 3.5}   # J/m²
rows_ife = []
for intf in INTERFACES:
    for T_i in T_IFE:
        gic = IFE_REF[intf] * np.exp(-2.5e-4 * (T_i - 600))
        noise = RNG.normal(0, gic * 0.05)
        rows_ife.append(dict(interface=intf, temperature_C=T_i,
                             fracture_energy_J_m2=round(float(gic + noise), 4)))

df_ife = pd.DataFrame(rows_ife)
_save(df_ife, "1_half_cell", "interfacial_fracture_energy.csv")

# ── Redox expansion ────────────────────────────────────────────────────────────
N_REDOX = 20
rows_redox = []
for spec in range(1, N_SPEC + 1):
    irrev = np.cumsum(RNG.exponential(0.0005, N_REDOX))
    for cyc, strain in enumerate(irrev, 1):
        rows_redox.append(dict(specimen=spec, redox_cycle=cyc,
                               irreversible_strain=round(float(strain), 6)))

df_redox = pd.DataFrame(rows_redox)
_save(df_redox, "1_half_cell", "redox_expansion.csv")

# ── Section 1 README ──────────────────────────────────────────────────────────
readme1 = """\
# Half-Cell Thermo-Mechanical Characterisation

## ⚠️ SYNTHETIC DATA DISCLAIMER
All data in this directory are **synthetically generated** for research-scaffolding
purposes only. They are NOT real experimental measurements. Physical trends follow
published literature relationships with added Gaussian noise.

## Files
| File | Description | Units |
|------|-------------|-------|
| cte_vs_temperature.csv | CTE vs T for 6 materials, 10 specimens each | ppm/K |
| chemical_expansion_vs_pO2.csv | Chemical strain vs oxygen partial pressure | dimensionless |
| elastic_moduli_vs_temperature.csv | E, G, Poisson vs T | GPa, - |
| creep_parameters.csv | Norton creep strain rates; A, n, Q coefficients | s⁻¹, MPa, J/mol |
| weibull_fracture_strength.csv | Fracture strength distribution; Weibull m, σ₀ | MPa |
| interfacial_fracture_energy.csv | Gc vs T for anode/electrolyte interfaces | J/m² |
| redox_expansion.csv | Irreversible strain after re-oxidation cycles | dimensionless |

## Test Methods (simulated)
- CTE: dilatometry at 2 K/min, 25–900 °C
- Chemical expansion: isothermal optical dilatometry in H₂–H₂O atmospheres
- Elastic moduli: impulse excitation technique
- Creep: 4-point bending at 650/750/850 °C, constant stress 100 h
- Fracture strength: ring-on-ring biaxial flexure
- Interfacial fracture energy: double-cantilever beam
- Redox: isothermal re-oxidation cycles
"""
with open(os.path.join(ROOT, "1_half_cell", "README.md"), "w") as f:
    f.write(readme1)
print("  saved 1_half_cell/README.md")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – Full-Cell Button-Cell Testing
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Section 2: Full-Cell ──")

N_CELLS = 12
T_CELL  = np.arange(600, 875, 50)    # 600,650,...,850 °C
J_VALS  = np.linspace(0, 1.0, 21)    # A/cm²

# Butler-Volmer–inspired IV curve
def iv_model(j, T_C, mode="SOFC"):
    T_K  = T_C + 273.15
    R    = 8.314
    Farad = 96485
    # Nernst voltage (simple H2/H2O)
    V_nernst = 1.0 - 2.5e-4 * (T_K - 1073)
    # OCV slightly below Nernst (leakage)
    OCV = V_nernst * 0.98
    # Ohmic ASR decreases with T (Arrhenius)
    ASR = 0.20 * np.exp(8000 * (1/T_K - 1/1073))
    # Activation overpotential
    eta_act = (R * T_K / (2 * Farad)) * np.log((j + 0.001) / 0.02 + 1) * 8
    if mode == "SOFC":
        V = OCV - j * ASR - eta_act
    else:   # SOEC
        V = OCV + j * ASR + eta_act
    return V

rows_iv = []
for cell in range(1, N_CELLS + 1):
    for T_c in T_CELL:
        for mode in ["SOFC", "SOEC"]:
            for j in J_VALS:
                V = iv_model(j, T_c, mode)
                V += RNG.normal(0, 0.003)
                rows_iv.append(dict(cell=cell, temperature_C=T_c, mode=mode,
                                    current_density_A_cm2=round(float(j), 3),
                                    voltage_V=round(float(V), 4)))

df_iv = pd.DataFrame(rows_iv)
_save(df_iv, "2_full_cell", "iv_curves.csv")

# ── EIS spectra ────────────────────────────────────────────────────────────────
FREQ = np.logspace(-1, 6, 60)       # 0.1 Hz – 1 MHz
J_EIS = [0.0, 0.1, 0.3, 0.5]       # A/cm²
T_EIS = [700]                        # representative T for EIS

def eis_randles(freq, R_ohm, R_pol, tau):
    omega = 2 * np.pi * freq
    Z_re  = R_ohm + R_pol / (1 + (omega * tau)**2)
    Z_im  = -R_pol * omega * tau / (1 + (omega * tau)**2)
    return Z_re, Z_im

rows_eis = []
for cell in range(1, min(N_CELLS, 3) + 1):   # 3 cells for file size
    for T_c in T_EIS:
        ASR = 0.20 * np.exp(8000 * (1/(T_c + 273.15) - 1/1073))
        for j in J_EIS:
            R_pol_j = 0.25 * np.exp(-j * 0.8) * np.exp(8000 * (1/(T_c + 273.15) - 1/1073))
            tau     = 0.01 * np.exp(5000 * (1/(T_c + 273.15) - 1/1073))
            Z_re, Z_im = eis_randles(FREQ, ASR, R_pol_j, tau)
            for f, re, im in zip(FREQ, Z_re, Z_im):
                rows_eis.append(dict(cell=cell, temperature_C=T_c,
                                     current_density_A_cm2=float(j),
                                     frequency_Hz=round(float(f), 4),
                                     Z_real_Ohm_cm2=round(float(re) + float(RNG.normal(0, 2e-4)), 5),
                                     Z_imag_Ohm_cm2=round(float(im) + float(RNG.normal(0, 2e-4)), 5)))

df_eis = pd.DataFrame(rows_eis)
_save(df_eis, "2_full_cell", "eis_spectra.csv")

# ── Thermal cycling ────────────────────────────────────────────────────────────
N_CYCLES = 20
rows_tc = []
for cell in range(1, N_CELLS + 1):
    for cyc in range(1, N_CYCLES + 1):
        strain_acc = 1e-4 * cyc**0.6 + RNG.normal(0, 5e-6)
        R_ohm_cyc  = 0.20 * (1 + 0.003 * cyc) * np.exp(8000 * (1/1073 - 1/1073))
        rows_tc.append(dict(cell=cell, cycle=cyc,
                            strain_accumulation=round(float(strain_acc), 7),
                            R_ohm_Ohm_cm2=round(float(R_ohm_cyc), 5)))

df_tc = pd.DataFrame(rows_tc)
_save(df_tc, "2_full_cell", "thermal_cycling.csv")

# ── Reversible cycling timeseries ─────────────────────────────────────────────
# 30 min SOFC → 15 min OCV → 30 min SOEC, repeat; T swings 700–780 °C
dt_s  = 30          # 30 s resolution
t_tot = 200 * 3600  # 200 h (until ~10% degradation)
t_arr = np.arange(0, t_tot, dt_s)

def cycle_phase(t):
    t_mod = t % (75 * 60)   # 75-min cycle
    if t_mod < 30 * 60:
        return "SOFC",  0.5
    elif t_mod < 45 * 60:
        return "OCV",   0.0
    else:
        return "SOEC", -0.5

rows_rev = []
degr_factor = 1.0
for i, t in enumerate(t_arr):
    phase, j = cycle_phase(t)
    degr_factor = 1.0 - 1e-7 * i * dt_s   # slow 10% degradation over ~200 h
    T_op = 740 + 40 * np.sin(2 * np.pi * t / (12 * 3600))
    V = iv_model(abs(j), T_op, "SOFC" if j >= 0 else "SOEC") * degr_factor
    V += RNG.normal(0, 0.002)
    if degr_factor <= 0.90:
        break
    rows_rev.append(dict(time_s=int(t), phase=phase,
                         current_density_A_cm2=float(j),
                         voltage_V=round(float(V), 4),
                         temperature_C=round(float(T_op), 2),
                         degradation_factor=round(float(degr_factor), 5)))

df_rev = pd.DataFrame(rows_rev)
_save(df_rev, "2_full_cell", "reversible_cycling_timeseries.csv")

# ── Post-mortem defects ────────────────────────────────────────────────────────
rows_pm = []
for cell in range(1, N_CELLS + 1):
    rows_pm.append(dict(cell=cell,
                        crack_density_mm_neg2=round(float(RNG.normal(0.5, 0.12)), 3),
                        delamination_length_um=round(float(RNG.normal(80, 15)), 1),
                        ni_particle_size_um=round(float(RNG.normal(1.2, 0.2)), 3)))

df_pm = pd.DataFrame(rows_pm)
_save(df_pm, "2_full_cell", "postmortem_defects.csv")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – Short-Stack Testing
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Section 3: Short-Stack ──")

N_STACK_CYCLES = 500
dt_stack = 60       # 1-min resolution within cycle; one row per hour for size
hours_per_cycle = 16
T_HOURS = hours_per_cycle * N_STACK_CYCLES

rows_stack = []
t_h = 0
for cyc in range(1, N_STACK_CYCLES + 1):
    degr = 1 - 5e-4 * cyc ** 0.55
    for h in range(hours_per_cycle):
        t_h += 1
        mode_h = "SOEC" if h < 8 else "SOFC"
        j_h    = 0.4
        T_in   = 680 + 130 * (h / hours_per_cycle)
        cell_V = [round(float(iv_model(j_h, T_in, mode_h) * degr + RNG.normal(0, 0.003)), 4)
                  for _ in range(5)]
        stack_V = sum(cell_V)
        rows_stack.append(dict(cycle=cyc, hour_in_cycle=h, elapsed_hours=t_h,
                               mode=mode_h,
                               cell1_V=cell_V[0], cell2_V=cell_V[1],
                               cell3_V=cell_V[2], cell4_V=cell_V[3],
                               cell5_V=cell_V[4],
                               stack_voltage_V=round(float(stack_V), 3),
                               current_A=round(float(j_h * 16), 2),
                               T_inlet_C=round(float(T_in), 1),
                               T_outlet_C=round(float(T_in + 30 + RNG.normal(0, 2)), 1),
                               fuel_flow_slpm=round(float(2.0 + RNG.normal(0, 0.05)), 3)))

df_stack = pd.DataFrame(rows_stack)
_save(df_stack, "3_short_stack", "stack_timeseries.csv")

# ── Thermocouple array ────────────────────────────────────────────────────────
N_TC = 30   # 6 TCs per cell × 5 cells
tc_positions = [(RNG.uniform(0, 4), RNG.uniform(0, 4), RNG.uniform(0, 5)) for _ in range(N_TC)]
rows_tc_arr = []
for cyc in range(1, 11):           # 10 representative cycles
    T_base = 700 + 5 * cyc
    for tc_id, (x, y, z) in enumerate(tc_positions, 1):
        T_read = T_base + 30 * (z / 5) + RNG.normal(0, 2)
        rows_tc_arr.append(dict(cycle=cyc, tc_id=tc_id,
                                x_cm=round(x, 3), y_cm=round(y, 3), z_cm=round(z, 3),
                                temperature_C=round(float(T_read), 2)))

df_tc_arr = pd.DataFrame(rows_tc_arr)
_save(df_tc_arr, "3_short_stack", "thermocouple_array.csv")

# ── Stack EIS periodic ────────────────────────────────────────────────────────
rows_seis = []
for cyc in range(0, N_STACK_CYCLES + 1, 50):
    for cell in range(1, 6):
        degr = 1 - 5e-4 * cyc ** 0.55
        ASR_cyc = 0.22 / degr
        for f, freq in enumerate(FREQ):
            Z_re, Z_im = eis_randles(freq, ASR_cyc, 0.18 / degr, 0.015)
            rows_seis.append(dict(cycle=cyc, cell=cell,
                                  frequency_Hz=round(float(freq), 4),
                                  Z_real=round(float(Z_re), 5),
                                  Z_imag=round(float(Z_im), 5)))

df_seis = pd.DataFrame(rows_seis)
_save(df_seis, "3_short_stack", "stack_eis_periodic.csv")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – 3D Multiphysics Simulation Parametric Sweep
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Section 4: Simulation Sweep ──")

# Representative subset (~500 cases) of a ~5000-case full factorial
np.random.seed(42)
N_CASES = 500
j_arr   = RNG.uniform(0.0, 1.0, N_CASES)
pH2O_pH2 = RNG.uniform(0.1, 0.9, N_CASES)
pO2_arr  = RNG.choice([0.21, 1.0], N_CASES)
T_in_arr = RNG.uniform(600, 850, N_CASES)
FU_arr   = RNG.uniform(0.3, 0.9, N_CASES)
BC_arr   = RNG.choice(["adiabatic", "isothermal"], N_CASES)
mech_arr = RNG.uniform(0.5, 2.0, N_CASES)
ramp_arr = RNG.choice([1, 5, 10], N_CASES)

df_sweep_in = pd.DataFrame(dict(
    case_id=range(1, N_CASES + 1),
    current_density_A_cm2=np.round(j_arr, 3),
    pH2O_pH2_ratio=np.round(pH2O_pH2, 3),
    pO2_cathode_atm=pO2_arr,
    inlet_temperature_C=np.round(T_in_arr, 1),
    fuel_utilisation=np.round(FU_arr, 3),
    thermal_BC=BC_arr,
    mechanical_load_MPa=np.round(mech_arr, 2),
    ramp_rate_K_min=ramp_arr,
))
_save(df_sweep_in, "4_simulation_sweep", "parametric_sweep_inputs.csv")

# Compute physics-based outputs
T_max  = T_in_arr + j_arr * 80 + (BC_arr == "adiabatic") * 20 + RNG.normal(0, 3, N_CASES)
stress_max = 50 + 30 * j_arr + 10 * mech_arr + RNG.normal(0, 3, N_CASES)
strain_max = 1e-3 * (stress_max / 200) + 2e-5 * (T_max - 600)
ASR_out    = 0.2 * np.exp(8000 * (1 / (T_in_arr + 273.15) - 1/1073))
eff_out    = 0.60 - 0.15 * j_arr + RNG.normal(0, 0.02, N_CASES)
degr_idx   = 0.001 * j_arr**2 * FU_arr + RNG.normal(0, 5e-5, N_CASES)

df_sweep_out = pd.DataFrame(dict(
    case_id=range(1, N_CASES + 1),
    max_temperature_C=np.round(T_max, 1),
    max_stress_MPa=np.round(stress_max, 2),
    max_strain=np.round(strain_max, 6),
    ASR_Ohm_cm2=np.round(ASR_out, 4),
    efficiency=np.round(np.clip(eff_out, 0.3, 0.85), 4),
    degradation_index=np.round(np.clip(degr_idx, 0, None), 6),
))
_save(df_sweep_out, "4_simulation_sweep", "parametric_sweep_outputs_summary.csv")

# ── Sample 3D field snapshot (~2000 points) ───────────────────────────────────
NX, NY, NZ = 10, 10, 20
x_, y_, z_ = np.meshgrid(np.linspace(0, 4, NX),
                          np.linspace(0, 4, NY),
                          np.linspace(0, 5, NZ))
x_ = x_.ravel(); y_ = y_.ravel(); z_ = z_.ravel()
T_f    = 700 + 80 * (z_ / 5) + 10 * np.sin(np.pi * x_ / 4) + RNG.normal(0, 2, len(x_))
sig_f  = 40 + 20 * (1 - z_ / 5) + RNG.normal(0, 2, len(x_))
eps_f  = sig_f / 200e3
pO2_f  = 0.21 * np.exp(-0.5 * z_ / 5) + RNG.normal(0, 0.005, len(x_))
J_f    = 0.4 + 0.1 * (z_ / 5) + RNG.normal(0, 0.01, len(x_))
dmg_f  = np.clip(0.01 * (z_ / 5)**2 + RNG.normal(0, 0.001, len(x_)), 0, None)

df_field = pd.DataFrame(dict(
    x_cm=np.round(x_, 3), y_cm=np.round(y_, 3), z_cm=np.round(z_, 3),
    temperature_C=np.round(T_f, 2),
    stress_MPa=np.round(sig_f, 3),
    strain=np.round(eps_f, 6),
    pO2_atm=np.round(pO2_f, 5),
    current_density_A_cm2=np.round(J_f, 4),
    damage_index=np.round(dmg_f, 5),
))
_save(df_field, "4_simulation_sweep", "sample_field_snapshot.csv")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – System-Level & Nuclear-Battery Integration
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Section 5: System-Level ──")

HOURS_YEAR = 8760
t_year = np.arange(HOURS_YEAR)

# HTGR thermal power profile (600 MWth nominal, seasonal variation)
P_htgr = 600 + 15 * np.sin(2 * np.pi * t_year / HOURS_YEAR) + \
         RNG.normal(0, 5, HOURS_YEAR)
P_htgr = np.clip(P_htgr, 540, 660)
T_He   = 800 + 5 * np.sin(2 * np.pi * t_year / HOURS_YEAR) + \
         RNG.normal(0, 2, HOURS_YEAR)
T_steam = T_He - 50 + RNG.normal(0, 3, HOURS_YEAR)

df_htgr = pd.DataFrame(dict(
    hour=t_year,
    thermal_power_MWth=np.round(P_htgr, 2),
    He_outlet_temp_C=np.round(T_He, 2),
    steam_outlet_temp_C=np.round(T_steam, 2),
))
_save(df_htgr, "5_system_level", "htgr_thermal_power_profile.csv")

# Grid demand & price
P_grid = 5000 + 1000 * np.sin(2 * np.pi * t_year / HOURS_YEAR) + \
         300  * np.sin(2 * np.pi * t_year / 24) + \
         RNG.normal(0, 150, HOURS_YEAR)
price  = 50 + 20 * np.sin(2 * np.pi * t_year / 24) + \
         15 * np.sin(2 * np.pi * t_year / (24 * 7)) + \
         RNG.normal(0, 8, HOURS_YEAR)
price  = np.clip(price, 0, None)

df_grid = pd.DataFrame(dict(
    hour=t_year,
    grid_demand_MW=np.round(P_grid, 1),
    day_ahead_price_USD_MWh=np.round(price, 2),
))
_save(df_grid, "5_system_level", "grid_demand_price.csv")

# Derived SOC schedule
PRICE_SOEC_THRESH = 35   # $/MWh – produce H₂ when cheap
PRICE_SOFC_THRESH = 60   # $/MWh – generate electricity when expensive

mode_sched = np.where(price < PRICE_SOEC_THRESH, "SOEC",
             np.where(price > PRICE_SOFC_THRESH, "SOFC", "standby"))
j_sched    = np.where(mode_sched == "SOEC", -0.4,
             np.where(mode_sched == "SOFC",  0.4, 0.0))

df_sched = pd.DataFrame(dict(
    hour=t_year,
    mode=mode_sched,
    current_density_A_cm2=j_sched,
    price_USD_MWh=np.round(price, 2),
))
_save(df_sched, "5_system_level", "derived_soc_schedule.csv")

# Li-ion battery benchmark (LFP and NMC)
DoD_vals = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
T_batt   = np.array([25, 45])
chem     = ["LFP", "NMC"]
rows_li  = []
for ch in chem:
    A_cyc = 8000 if ch == "LFP" else 3000
    rte0  = 0.96 if ch == "LFP" else 0.93
    cost0 = 0.15 if ch == "LFP" else 0.12   # $/kWh/cycle at 0.8 DoD, 25 °C
    for Tb in T_batt:
        for dod in DoD_vals:
            cycle_life = A_cyc * (1 - dod * 0.3) * (1 - 0.008 * max(0, Tb - 25))
            rte = rte0 - 0.02 * dod - 0.002 * max(0, Tb - 25)
            cap_fade_pct_per_100cy = 0.8 + dod * 0.5 + 0.05 * max(0, Tb - 25)
            cost = cost0 / (1 - dod * 0.3)
            rows_li.append(dict(chemistry=ch, temperature_C=Tb, dod=dod,
                                cycle_life=int(cycle_life),
                                round_trip_efficiency=round(rte, 3),
                                capacity_fade_pct_per_100cycles=round(cap_fade_pct_per_100cy, 3),
                                cost_USD_kWh_cycle=round(cost, 4)))

df_li = pd.DataFrame(rows_li)
_save(df_li, "5_system_level", "liion_battery_benchmark.csv")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – Literature Review Support
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Section 6: Literature ──")

papers = [
    dict(
        title="Attention Is All You Need",
        authors="Vaswani et al.",
        year=2017,
        venue="NeurIPS",
        arxiv_doi="arXiv:1706.03762",
        summary="Seminal transformer paper introducing multi-head self-attention; "
                "baseline architecture for all subsequent attention-based scientific ML models.",
        relevance_to_pinn_thermomechanics="Establishes the attention mechanism foundation; "
                "spatial attention directly applicable to non-uniform temperature/stress field encoding.",
    ),
    dict(
        title="Physics-Guided Transformer (PGT) for Scientific Machine Learning",
        authors="Unknown (preprint)",
        year=2026,
        venue="arXiv preprint",
        arxiv_doi="arXiv:2603.27929",
        summary="Integrates physics constraints into the transformer architecture via physics-guided "
                "attention masks and physics-residual loss terms.",
        relevance_to_pinn_thermomechanics="Directly applicable to coupled thermo-mechanical PINNs "
                "where attention masks can encode spatial physics constraints.",
    ),
    dict(
        title="PINNs with Fourier Features and Attention-Driven Decoding",
        authors="Unknown (preprint)",
        year=2025,
        venue="arXiv preprint",
        arxiv_doi="arXiv:2510.05385",
        summary="Combines random Fourier feature embeddings with cross-attention decoding in PINNs, "
                "improving convergence on high-frequency solutions.",
        relevance_to_pinn_thermomechanics="Useful for capturing high-frequency thermal gradients near "
                "electrode interfaces in SOFC stacks.",
    ),
    dict(
        title="Residual-based Attention in Physics-Informed Neural Networks",
        authors="Anagnostopoulos et al.",
        year=2024,
        venue="Computer Methods in Applied Mechanics and Engineering (CMAME)",
        arxiv_doi="DOI:10.1016/j.cma.2024.116866",
        summary="Proposes point-wise residual-based attention weights that focus network capacity "
                "on high-residual regions of the PDE domain.",
        relevance_to_pinn_thermomechanics="Critical for SOFC geometry where residuals concentrate "
                "near triple-phase boundaries and crack tips.",
    ),
    dict(
        title="PINNpapers – Survey Repository",
        authors="idrl-lab",
        year=2023,
        venue="GitHub",
        arxiv_doi="https://github.com/idrl-lab/PINNpapers",
        summary="Curated survey of 400+ PINN papers organised by physics domain, including "
                "attention mechanisms, transfer learning, and domain decomposition.",
        relevance_to_pinn_thermomechanics="Comprehensive entry point for literature on PINNs in "
                "heat transfer and solid mechanics.",
    ),
    dict(
        title="Physics-informed attention-based neural network for hyperbolic PDEs: New possibilities "
                "for real-time underreporting assessment",
        authors="Rodr\u00edguez-Torrado et al.",
        year=2022,
        venue="Scientific Reports",
        arxiv_doi="DOI:10.1038/s41598-022-15928-5",
        summary="Applies spatial attention in PINNs for hyperbolic PDEs (transport equations), "
                "demonstrating improved accuracy with fewer collocation points.",
        relevance_to_pinn_thermomechanics="Transport-type equations appear in SOFC gas channel "
                "and ion-transport subproblems.",
    ),
    dict(
        title="Fourier Neural Operator for Parametric Partial Differential Equations",
        authors="Li et al.",
        year=2021,
        venue="ICLR",
        arxiv_doi="arXiv:2010.08895",
        summary="Neural operator learning via spectral (Fourier) convolutions; resolution-invariant "
                "and highly data-efficient for PDE surrogates.",
        relevance_to_pinn_thermomechanics="Can serve as the backbone surrogate for 3D "
                "thermo-mechanical field prediction from operating-condition inputs.",
    ),
    dict(
        title="Galerkin Transformer: A One-Shot Experiment",
        authors="Cao",
        year=2021,
        venue="NeurIPS Workshop",
        arxiv_doi="arXiv:2105.14995",
        summary="Replaces softmax attention with Galerkin-type integral operators for solving PDEs, "
                "achieving O(N) complexity for N collocation points.",
        relevance_to_pinn_thermomechanics="Scalable to the ~500k-element mesh used in SOFC "
                "multiphysics simulations.",
    ),
    dict(
        title="Transolver: A Fast Transformer Solver for PDEs on General Geometries",
        authors="Wu et al.",
        year=2024,
        venue="ICML",
        arxiv_doi="arXiv:2402.02366",
        summary="Physics-aware attention that tokenises mesh points into physics states, "
                "achieving state-of-the-art accuracy on six PDE benchmarks.",
        relevance_to_pinn_thermomechanics="Directly applicable to irregular SOFC stack geometry; "
                "physics-state tokens align well with electrochemical/mechanical sub-domains.",
    ),
    dict(
        title="DeepONet: Learning nonlinear operators for identifying differential equations "
                "based on the universal approximation theorem of operators",
        authors="Lu et al.",
        year=2021,
        venue="Nature Machine Intelligence",
        arxiv_doi="arXiv:1910.03193",
        summary="Operator-learning framework mapping input functions to output fields; "
                "extended to physics-informed (PI-DeepONet) training.",
        relevance_to_pinn_thermomechanics="Maps HTGR thermal-power profiles to SOFC stack "
                "temperature and stress fields as an operator.",
    ),
    dict(
        title="Self-attention based physics-informed neural networks for multi-fidelity modeling",
        authors="Howard et al.",
        year=2023,
        venue="arXiv preprint",
        arxiv_doi="arXiv:2207.00045",
        summary="Multi-fidelity PINN with self-attention fusion of coarse and fine data, "
                "substantially reducing high-fidelity simulation cost.",
        relevance_to_pinn_thermomechanics="Enables combining cheap button-cell data with "
                "expensive short-stack simulations in a unified model.",
    ),
]

df_papers = pd.DataFrame(papers)
_save(df_papers, "6_literature", "attention_mechanism_pinn_papers.csv")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – Figures
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Section 7: Figures ──")

# Figure 1 – CTE vs Temperature
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.tab10(np.linspace(0, 1, len(MATERIALS)))
df_cte_mean = df_cte.groupby(["material", "temperature_C"])["cte_ppm_K"].mean().reset_index()
for mat, col in zip(MATERIALS, colors):
    sub = df_cte_mean[df_cte_mean.material == mat]
    ax.plot(sub.temperature_C, sub.cte_ppm_K, label=mat, color=col)
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("CTE (ppm/K)")
ax.set_title("Thermal Expansion Coefficient vs Temperature\n[SYNTHETIC DATA]")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
_fig("fig1_cte_vs_temperature.png")

# Figure 2 – Creep curves
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, T_c in zip(axes, CREEP_T):
    sub = df_creep[df_creep.temperature_C == T_c]
    for mat, col in zip(MATERIALS, colors):
        ms = sub[sub.material == mat]
        ax.semilogy(ms.stress_MPa, ms.creep_strain_rate_per_s, "o-",
                    label=mat, color=col, markersize=5)
    ax.set_xlabel("Stress (MPa)")
    ax.set_title(f"{T_c} °C")
    ax.grid(True, which="both", alpha=0.3)
axes[0].set_ylabel("Creep strain rate (s⁻¹)")
axes[0].legend(fontsize=7)
fig.suptitle("Norton Creep Strain Rate vs Stress [SYNTHETIC DATA]")
plt.tight_layout()
_fig("fig2_creep_curves.png")

# Figure 3 – Weibull fracture strength
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, mat in zip(axes.ravel(), MATERIALS):
    sub = df_weib[df_weib.material == mat]
    for T_w, col in zip(T_WEIB, ["navy", "royalblue", "darkorange", "crimson"]):
        st = sub[sub.temperature_C == T_w].fracture_strength_MPa.values
        st_sorted = np.sort(st)
        pf = (np.arange(1, len(st_sorted) + 1) - 0.3) / (len(st_sorted) + 0.4)
        ax.plot(np.log(st_sorted), np.log(-np.log(1 - pf)), "o-",
                label=f"{T_w} °C", color=col, markersize=4)
    ax.set_xlabel("ln(σ)")
    ax.set_ylabel("ln(ln(1/(1-Pf)))")
    ax.set_title(mat)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
fig.suptitle("Weibull Fracture Strength Distribution [SYNTHETIC DATA]")
plt.tight_layout()
_fig("fig3_weibull_fracture.png")

# Figure 4 – I-V curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, mode in zip(axes, ["SOFC", "SOEC"]):
    sub = df_iv[(df_iv.mode == mode) & (df_iv.cell == 1)]
    for T_c, col in zip(T_CELL, plt.cm.plasma(np.linspace(0.1, 0.9, len(T_CELL)))):
        ts = sub[sub.temperature_C == T_c]
        ax.plot(ts.current_density_A_cm2, ts.voltage_V, label=f"{T_c} °C", color=col)
    ax.set_xlabel("Current density (A/cm²)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"{mode} Mode I-V Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
fig.suptitle("I-V Curves 600–850 °C [SYNTHETIC DATA]")
plt.tight_layout()
_fig("fig4_iv_curves.png")

# Figure 5 – EIS Nyquist
fig, ax = plt.subplots(figsize=(7, 5))
for j_val, col in zip(J_EIS, ["navy", "royalblue", "darkorange", "crimson"]):
    sub = df_eis[(df_eis.cell == 1) & (df_eis.current_density_A_cm2 == j_val)]
    ax.plot(sub.Z_real_Ohm_cm2, -sub.Z_imag_Ohm_cm2, "o-",
            markersize=3, label=f"j={j_val} A/cm²", color=col)
ax.set_xlabel("Z' (Ω·cm²)")
ax.set_ylabel("-Z'' (Ω·cm²)")
ax.set_title("EIS Nyquist Plot at 700 °C [SYNTHETIC DATA]")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
_fig("fig5_eis_nyquist.png")

# Figure 6 – Reversible cycling degradation
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
t_h_rev = df_rev.time_s / 3600
axes[0].plot(t_h_rev, df_rev.voltage_V, lw=0.5, color="steelblue")
axes[0].set_ylabel("Voltage (V)")
axes[0].set_title("Reversible SOFC↔SOEC Cycling [SYNTHETIC DATA]")
axes[0].grid(True, alpha=0.3)
axes[1].plot(t_h_rev, (1 - df_rev.degradation_factor) * 100, lw=1, color="crimson")
axes[1].set_xlabel("Time (h)")
axes[1].set_ylabel("Degradation (%)")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
_fig("fig6_reversible_cycling_degradation.png")

# Figure 7 – Stack voltage degradation
cycles_plot = df_stack.groupby("cycle")["stack_voltage_V"].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cycles_plot.cycle, cycles_plot.stack_voltage_V, color="steelblue")
ax.set_xlabel("Cycle number")
ax.set_ylabel("Mean stack voltage (V)")
ax.set_title("5-Cell Short-Stack Voltage Degradation over 500 Cycles [SYNTHETIC DATA]")
ax.grid(True, alpha=0.3)
_fig("fig7_stack_voltage_degradation.png")

# Figure 8 – 3D temperature field slice (z-mid plane)
z_mid = 2.5
df_slice = df_field[np.abs(df_field.z_cm - z_mid) < 0.3]
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.tricontourf(df_slice.x_cm, df_slice.y_cm, df_slice.temperature_C,
                    levels=20, cmap="inferno")
plt.colorbar(sc, ax=ax, label="Temperature (°C)")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_title("3D Temperature Field – z ≈ 2.5 cm Slice [SYNTHETIC DATA]")
_fig("fig8_3d_temperature_field_slice.png")

# Figure 9 – HTGR & grid price profile (first 30 days)
days = 30 * 24
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(df_htgr.hour[:days] / 24, df_htgr.thermal_power_MWth[:days], color="firebrick")
axes[0].set_ylabel("HTGR Thermal Power (MWth)")
axes[0].set_title("HTGR Thermal Power & Electricity Price – First 30 Days [SYNTHETIC DATA]")
axes[0].grid(True, alpha=0.3)
axes[1].plot(df_grid.hour[:days] / 24, df_grid.day_ahead_price_USD_MWh[:days], color="steelblue")
axes[1].axhline(PRICE_SOEC_THRESH, color="green",  ls="--", lw=0.8, label="SOEC threshold")
axes[1].axhline(PRICE_SOFC_THRESH, color="crimson", ls="--", lw=0.8, label="SOFC threshold")
axes[1].set_xlabel("Day of year")
axes[1].set_ylabel("Day-ahead price ($/MWh)")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
_fig("fig9_htgr_and_grid_price_profile.png")

# Figure 10 – Li-ion vs SOC comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ch, col in zip(chem, ["steelblue", "darkorange"]):
    sub = df_li[(df_li.chemistry == ch) & (df_li.temperature_C == 25)]
    axes[0].plot(sub.dod, sub.cycle_life, "o-", label=ch, color=col)
    axes[1].plot(sub.dod, sub.round_trip_efficiency, "s--", label=ch, color=col)
axes[0].set_xlabel("Depth of Discharge")
axes[0].set_ylabel("Cycle Life")
axes[0].set_title("Li-Ion Cycle Life vs DoD [SYNTHETIC DATA]")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel("Depth of Discharge")
axes[1].set_ylabel("Round-Trip Efficiency")
axes[1].set_title("Round-Trip Efficiency vs DoD [SYNTHETIC DATA]")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
_fig("fig10_liion_vs_soc_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – Top-level README
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Writing top-level README ──")

readme_top = """\
# SOFC PINN Datasets

## ⚠️ SYNTHETIC DATA DISCLAIMER
**All data in this repository are synthetically generated** for research-scaffolding
and demonstration purposes. They are NOT real experimental or simulation measurements.
Physical trends follow literature-informed relationships with added Gaussian noise
(see generation script for details). Do not cite this data as experimental evidence.

---

## Research Context
"Physics-informed 3D deep learning for thermo-mechanical lifetime prediction of
short-stack reversible solid oxide cells cycled at 600–850 °C in nuclear battery service."

---

## Directory Structure

```
datasets/
├── 1_half_cell/          # Thermo-mechanical constitutive data (CTE, creep, Weibull, …)
├── 2_full_cell/          # Button-cell electrochemical & degradation data
├── 3_short_stack/        # 5-cell stack time series, TC arrays, periodic EIS
├── 4_simulation_sweep/   # Parametric multiphysics sweep inputs/outputs, 3D field snapshot
├── 5_system_level/       # HTGR power, grid price, SOC schedule, Li-ion benchmark
├── 6_literature/         # Attention-mechanism / PINN paper table
├── figures/              # 10 publication-quality PNG figures (300 dpi)
└── SOFC_PINN_datasets.zip  # All CSVs + figures in a single downloadable archive
```

---

## Physical Relationships Used

| Dataset | Physical law / model |
|---------|----------------------|
| CTE | Linear fit: α(T) = α₀ + k·(T–25) |
| Chemical expansion | Logarithmic: ε = A·log₁₀(pO₂) + B |
| Elastic moduli | Linear softening: E(T) = E₀·(1 – c·(T–25)) |
| Creep | Norton power law: ε̇ = A·σⁿ·exp(–Q/RT) |
| Fracture strength | Weibull distribution: m, σ₀ temperature-dependent |
| I-V curves | Butler-Volmer activation + ohmic ASR; Arrhenius ASR(T) |
| EIS | Randles-type circuit: R_ohm + R_pol/(1+jωτ) |
| Stack degradation | Power-law creep accumulation |
| HTGR power | Sinusoidal annual variation + noise |
| Grid price | Diurnal + weekly sinusoids + noise |
| Li-ion cycle life | Empirical DoD and T scaling from literature |

---

## How to Regenerate

```bash
pip install -r requirements.txt
python scripts/generate_datasets.py
```

All CSVs, figures, and the zip file will be recreated deterministically (seed 42).

---

## Units

| Quantity | Unit |
|----------|------|
| Temperature | °C (unless stated) |
| CTE | ppm/K = 10⁻⁶ K⁻¹ |
| Stress | MPa |
| Strain | dimensionless |
| Current density | A/cm² |
| Voltage | V |
| Impedance | Ω·cm² |
| Fracture energy | J/m² |
| Frequency | Hz |
| Thermal power | MWth |
| Electricity price | USD/MWh |

---

## Citation (if used)
Please note this is synthetic data. If used in a publication, cite as:
> "Synthetic dataset generated for SOFC PINN research scaffolding, 2026.
>  github.com/georgegershom/SOFC. ⚠️ Not real experimental data."
"""

with open(os.path.join(ROOT, "README.md"), "w") as f:
    f.write(readme_top)
print("  saved datasets/README.md")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – Zip archive
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Creating ZIP archive ──")

zip_path = os.path.join(ROOT, "SOFC_PINN_datasets.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for dirpath, dirnames, filenames in os.walk(ROOT):
        for filename in filenames:
            if filename == "SOFC_PINN_datasets.zip":
                continue
            filepath = os.path.join(dirpath, filename)
            arcname = os.path.relpath(filepath, ROOT)
            zf.write(filepath, arcname)

zip_size_mb = os.path.getsize(zip_path) / 1e6
print(f"  saved SOFC_PINN_datasets.zip ({zip_size_mb:.1f} MB)")

print("\n✓ All files generated successfully.")
