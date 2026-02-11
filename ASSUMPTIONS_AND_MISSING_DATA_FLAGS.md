### Assumptions & Missing Data Flags

**Detailed Description of Missing Datasets:**

The datasets below are missing from direct experiments. Synthetic proxy data has been generated so the "Probabilistic Failure Maps" workflow can still run end-to-end.

**1. `MISSING_DATASET_01`: Interfacial Property Correlation Matrix**

* **Description:** Covariance structure between YSZ|GDC and GDC|LSCF interfacial fracture energies.
* **Scientific Context:** Co-processing defects can impact both interfaces in the same stack and induce positive correlation.
* **Current Assumption:** **Uncorrelated baseline (`rho = 0.0`)** for production runs; **sensitivity case (`rho = 0.5`)** generated for impact analysis.
* **Required Data to Fill Gap:** Paired, same-cell post-mortem fracture observations across both interfaces.

**2. `MISSING_DATASET_02`: High-Temperature Fracture Statistics at 800C**

* **Description:** In-situ 800C statistical distribution of interfacial fracture energy (`mu_800C`, `sigma_800C`).
* **Scientific Context:** Most available data are room-temperature micro-cantilever tests.
* **Current Assumption:** Mean scales with elastic modulus ratio (`E_800C / E_RT = 170/200 = 0.85`), while standard deviation is conservatively kept equal to RT.
* **Required Data to Fill Gap:** Dedicated in-situ 800C micro-cantilever campaigns.

---

### Governing Equations & Numerical Formulation

1. **Independent baseline sampling (`rho = 0`):**

`Gc_YSZ ~ N(mu1, sigma1^2),  Gc_LSCF ~ N(mu2, sigma2^2),  independent`

2. **Correlated sensitivity sampling (`rho > 0`):**

`[Gc_YSZ, Gc_LSCF]^T ~ N(mu, Sigma)`

`Sigma = [[sigma1^2, rho*sigma1*sigma2], [rho*sigma1*sigma2, sigma2^2]]`

3. **High-temperature mean extrapolation:**

`mu_800C = mu_RT * (E_800C / E_RT)`

with `E_800C/E_RT = 0.85`.

---

### Abaqus Implementation Plan

1. **Preprocessing (`generate_stochastic_inputs.py`):**
   * Read RT stats from `04_uncertainty_material_properties.csv`.
   * Estimate 800C means from modulus ratio.
   * Use independent seeds for baseline (`rho = 0`) and optional correlated seeds for sensitivity (`rho = 0.5`).

2. **UEL parameter passing:**
   * Pass sampled `Gc` to `PROPS`.
   * Keep UEL unchanged; uncertainty handling is external in input-file generation.

---

### Input Parameters Table

| Parameter | Symbol | Assumed Value | Unit | Rationale |
| --- | --- | --- | --- | --- |
| Correlation coefficient | `rho` | `0.0` baseline, `0.5` sensitivity | - | Missing paired failure data |
| YSZ\|GDC RT mean | `mu1` | `2.15` | J/m^2 | Prompt proxy |
| YSZ\|GDC RT std dev | `sigma1` | `0.40` | J/m^2 | Prompt proxy |
| GDC\|LSCF RT mean | `mu2` | `1.00` | J/m^2 | Prompt proxy |
| GDC\|LSCF RT std dev | `sigma2` | `0.20` | J/m^2 | Prompt proxy |
| Mean scaling to 800C | `E_800C/E_RT` | `0.85` | - | 170 GPa / 200 GPa |
| Std scaling to 800C | `eta_sigma` | `1.0` | - | Conservative spread retention |

---

### Figure & Table Blueprints

* **Figure 1:** Independent vs correlated scatter (`figures/fig01_independent_vs_correlated_scatter.png`)
* **Figure 2:** RT vs 800C distributions (`figures/fig02_rt_vs_800C_distributions.png`)
* **Figure 3:** Probabilistic failure maps and delta (`figures/fig03_probabilistic_failure_maps.png`)

---

### Verification & QA Steps

1. Confirm generated 800C standard deviations match configured scaling (`eta_sigma`) within 5%.
2. Confirm Pearson correlation:
   * Baseline dataset near `rho ~= 0`
   * Sensitivity dataset near `rho ~= 0.5`

---

### Next Questions for the User

1. Do you have paired post-mortem evidence to estimate a lower bound on `rho`?
2. Do you want additional sensitivity runs at `rho = 0.8` and `rho = -0.3`?
