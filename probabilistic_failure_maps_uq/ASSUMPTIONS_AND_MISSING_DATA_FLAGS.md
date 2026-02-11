# Assumptions and Missing-Data Flags (Synthetic Package)

## Critical Data Void Analysis

### 1) MISSING_DATASET_01: Interfacial Failure Correlation Matrix

- Missing physical quantity:
  covariance (and correlation) between Gc_YSZ_GDC and Gc_GDC_LSCF inside one co-sintered cell.
- Mathematical deficit:
  off-diagonal covariance terms are unknown.
- Current surrogate:
  - stochastic_inputs_rho0.00.csv (rho = 0.00)
  - stochastic_inputs_rho0.50.csv (rho = 0.50)
- Why it matters:
  the probability of simultaneous two-interface failure depends on the unknown joint distribution.

### 2) MISSING_DATASET_02: High-Temperature Fracture Statistics (800C)

- Missing physical quantity:
  in-situ p(Gc | T=800C) for both interfaces.
- Current surrogate:
  deterministic softening of the RT mean by factor 0.82 with standard deviation unchanged.
- Surrogate file:
  stochastic_inputs_HT_uncorrelated.csv
- Why it matters:
  if real high-temperature variance is wider than assumed, tail-risk is underestimated.

## Reliability Formulation Note

Because the true interfacial correlation is unknown, P_fail should be treated as a bounded interval
between at least the rho=0 and rho=0.5 surrogate worlds, not as a single calibrated value.
