# SEM analysis for FDI and firm performance in Lagos food processing
# - Reads synthetic dataset created by scripts/generate_data.py
# - Runs reliability and CFA for KA and TP and INN
# - Builds observed composites and interaction terms
# - Fits moderated SEM (path model) using lavaan with MLR estimator

# Helper: install packages if missing
safe_install <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

pkgs <- c("readr", "dplyr", "psych", "lavaan", "semTools")
invisible(lapply(pkgs, safe_install))

library(readr)
library(dplyr)
library(psych)
library(lavaan)
library(semTools)

# Paths
DATA_PATH <- "../data/survey_data.csv"

# Read data
raw <- readr::read_csv(DATA_PATH, show_col_types = FALSE)

# Basic recodes
raw <- raw %>%
  mutate(
    FDI = as.numeric(FDI),
    KA_score = as.numeric(KA_score),
    TP_score = as.numeric(TP_score),
    INN_score = as.numeric(INN_score),
    FR_score = as.numeric(FR_score),
    GP_score = as.numeric(GP_score),
    PERFORM_index = as.numeric(PERFORM_index)
  )

# Reliability (Cronbach's alpha) and CFA for reflective constructs
ka_items <- raw |> dplyr::select(ka1, ka2, ka3, ka4)
TP_items <- raw |> dplyr::select(tp1, tp2, tp3, tp4)
INN_items <- raw |> dplyr::select(inn1, inn2, inn3)

cat("\n==== Reliability (Cronbach's alpha) ====\n")
print(psych::alpha(ka_items))
print(psych::alpha(TP_items))
# INN indicators are heterogeneous scales; alpha is shown for reference only
print(psych::alpha(INN_items))

cat("\n==== CFA: Knowledge Absorption ====\n")
cfa_ka <- 'KA =~ ka1 + ka2 + ka3 + ka4'
fit_cfa_ka <- lavaan::cfa(cfa_ka, data = raw, estimator = "MLR")
print(summary(fit_cfa_ka, fit.measures = TRUE, standardized = TRUE))

cat("\n==== CFA: Task Performance ====\n")
cfa_tp <- 'TP =~ tp1 + tp2 + tp3 + tp4'
fit_cfa_tp <- lavaan::cfa(cfa_tp, data = raw, estimator = "MLR")
print(summary(fit_cfa_tp, fit.measures = TRUE, standardized = TRUE))

cat("\n==== CFA: Innovation (3 indicators) ====\n")
cfa_inn <- 'INN =~ inn1 + inn2 + inn3'
fit_cfa_inn <- lavaan::cfa(cfa_inn, data = raw, estimator = "MLR")
print(summary(fit_cfa_inn, fit.measures = TRUE, standardized = TRUE))

# Build observed composites for moderated path model
comp <- raw %>%
  mutate(
    KA = KA_score,
    TP = TP_score,
    INN = INN_score,
    FR = FR_score,
    GP = GP_score,
    PERFORM = PERFORM_index,
    # Interaction terms for moderation
    FDI_GP = FDI * GP,
    KA_GP = KA * GP,
    TP_GP = TP * GP,
    INN_GP = INN * GP
  )

# Center variables to reduce multicollinearity (optional but common)
comp <- comp %>%
  mutate(
    KA_c = scale(KA, center = TRUE, scale = FALSE)[,1],
    TP_c = scale(TP, center = TRUE, scale = FALSE)[,1],
    INN_c = scale(INN, center = TRUE, scale = FALSE)[,1],
    FR_c = scale(FR, center = TRUE, scale = FALSE)[,1],
    GP_c = scale(GP, center = TRUE, scale = FALSE)[,1],
    FDI_c = scale(FDI, center = TRUE, scale = FALSE)[,1],
    PERFORM_c = scale(PERFORM, center = TRUE, scale = FALSE)[,1]
  ) %>%
  mutate(
    FDI_GP_c = FDI_c * GP_c,
    KA_GP_c = KA_c * GP_c,
    TP_GP_c = TP_c * GP_c,
    INN_GP_c = INN_c * GP_c
  )

# Moderated SEM as path model with observed composites
model_path <- '
  # Direct effects
  PERFORM_c ~ b1*KA_c + b2*TP_c + b3*INN_c + b4*FR_c + b5*FDI_c

  # Mediation paths (from FDI)
  KA_c  ~ a1*FDI_c
  TP_c  ~ a2*FDI_c
  INN_c ~ a3*FDI_c

  # Moderation terms (observed interactions)
  PERFORM_c ~ g1*FDI_GP_c + g2*KA_GP_c + g3*TP_GP_c + g4*INN_GP_c

  # Indirect effects (defined parameters)
  ind_KA  := a1*b1
  ind_TP  := a2*b2
  ind_INN := a3*b3
'

fit_path <- lavaan::sem(model_path, data = comp, estimator = "MLR")

cat("\n==== Moderated SEM (observed composites) ====\n")
print(summary(fit_path, standardized = TRUE, fit.measures = TRUE, rsquare = TRUE))

cat("\n==== Parameter Estimates ====\n")
print(parameterEstimates(fit_path, standardized = TRUE, ci = TRUE))

cat("\nTarget fit thresholds: CFI>0.90, RMSEA<0.08, SRMR<0.06, chi2/df<3.0\n")
