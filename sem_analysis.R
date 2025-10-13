# =============================================================================
# STRUCTURAL EQUATION MODELING (SEM) ANALYSIS
# The Influence of Foreign Direct Investment on Food Processing Firms in Lagos
# =============================================================================

# Load required libraries
library(lavaan)
library(semPlot)
library(dplyr)
library(readr)
library(psych)
library(corrplot)
library(ggplot2)
library(semTools)

# =============================================================================
# LOAD DATA
# =============================================================================

# Load the synthetic dataset
data <- read_csv("fdi_sem_dataset.csv")

# Check data structure
cat("=== DATA OVERVIEW ===\n")
cat("Sample size:", nrow(data), "\n")
cat("Variables:", ncol(data), "\n")
str(data)

# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================

# Descriptive statistics for key variables
key_vars <- c("knowledge_absorption", "task_performance", "innovation_score", 
              "government_policy", "firm_resources", "overall_performance")

desc_stats <- describe(data[key_vars])
cat("\n=== DESCRIPTIVE STATISTICS ===\n")
print(desc_stats)

# Correlation matrix
cor_matrix <- cor(data[key_vars], use = "complete.obs")
cat("\n=== CORRELATION MATRIX ===\n")
print(round(cor_matrix, 3))

# =============================================================================
# MEASUREMENT MODEL SPECIFICATION
# =============================================================================

# Define the measurement model
measurement_model <- '
  # Latent Variables (Reflective Indicators)
  
  # Knowledge Absorption (KA) - Zahra & George Scale
  KA =~ ka1_technical_manuals + ka2_staff_training + ka3_adapt_technology + ka4_commercialize_knowledge
  
  # Task Performance (TP) - Koopmans et al. Scale  
  TP =~ tp1_production_efficiency + tp2_quality_control + tp3_order_fulfillment + tp4_employee_productivity
  
  # Innovation (INN) - OECD Oslo Manual indicators
  INN =~ rd_spending_pct + new_products_3yr + innovation_automation + innovation_quality_mgmt
  
  # Government Policy (GP) - Formative construct
  GP <~ gp1_tax_incentives + gp2_regulatory_stability + gp3_infrastructure_support + gp4_permit_ease
  
  # Firm Resources (FR) - Mixed indicators
  FR =~ skilled_workforce_pct + training_hours_annual + modern_equipment + reinvestment_rate_pct
  
  # Performance (PERF) - Financial and operational indicators
  PERF =~ avg_roi_pct + avg_roa_pct + export_intensity_pct + capacity_utilization_pct + market_share_lagos_pct
'

# Fit the measurement model
measurement_fit <- cfa(measurement_model, data = data, estimator = "MLR")

# Check measurement model fit
cat("\n=== MEASUREMENT MODEL FIT ===\n")
summary(measurement_fit, fit.measures = TRUE, standardized = TRUE)

# Extract fit indices
fit_indices_measurement <- fitMeasures(measurement_fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))
cat("\n=== MEASUREMENT MODEL FIT INDICES ===\n")
print(fit_indices_measurement)

# =============================================================================
# RELIABILITY AND VALIDITY TESTS
# =============================================================================

# Cronbach's Alpha for reflective constructs
alpha_ka <- alpha(data[c("ka1_technical_manuals", "ka2_staff_training", "ka3_adapt_technology", "ka4_commercialize_knowledge")])
alpha_tp <- alpha(data[c("tp1_production_efficiency", "tp2_quality_control", "tp3_order_fulfillment", "tp4_employee_productivity")])
alpha_perf <- alpha(data[c("avg_roi_pct", "avg_roa_pct", "export_intensity_pct", "capacity_utilization_pct", "market_share_lagos_pct")])

cat("\n=== RELIABILITY ANALYSIS ===\n")
cat("Knowledge Absorption Alpha:", round(alpha_ka$total$raw_alpha, 3), "\n")
cat("Task Performance Alpha:", round(alpha_tp$total$raw_alpha, 3), "\n")
cat("Performance Alpha:", round(alpha_perf$total$raw_alpha, 3), "\n")

# Composite Reliability and AVE
reliability_stats <- compRelSEM(measurement_fit)
cat("\n=== COMPOSITE RELIABILITY AND AVE ===\n")
print(reliability_stats)

# =============================================================================
# STRUCTURAL MODEL SPECIFICATION
# =============================================================================

# Define the full structural model with hypotheses
structural_model <- '
  # Measurement Model
  KA =~ ka1_technical_manuals + ka2_staff_training + ka3_adapt_technology + ka4_commercialize_knowledge
  TP =~ tp1_production_efficiency + tp2_quality_control + tp3_order_fulfillment + tp4_employee_productivity
  INN =~ rd_spending_pct + new_products_3yr + innovation_automation + innovation_quality_mgmt
  GP <~ gp1_tax_incentives + gp2_regulatory_stability + gp3_infrastructure_support + gp4_permit_ease
  FR =~ skilled_workforce_pct + training_hours_annual + modern_equipment + reinvestment_rate_pct
  PERF =~ avg_roi_pct + avg_roa_pct + export_intensity_pct + capacity_utilization_pct + market_share_lagos_pct
  
  # Structural Model - Direct Effects
  # H1: FDI → Knowledge Absorption
  KA ~ h1*has_fdi
  
  # H2: FDI → Task Performance  
  TP ~ h2*has_fdi
  
  # H3: FDI → Innovation
  INN ~ h3*has_fdi
  
  # H4: Knowledge Absorption → Performance
  PERF ~ h4*KA
  
  # H5: Task Performance → Performance
  PERF ~ h5*TP
  
  # H6: Innovation → Performance
  PERF ~ h6*INN
  
  # H7: Firm Resources → Performance
  PERF ~ h7*FR
  
  # Mediation Paths (Indirect Effects)
  # H8: FDI → KA → Performance (indirect effect)
  # H9: FDI → TP → Performance (indirect effect)
  # H10: FDI → INN → Performance (indirect effect)
  
  # Control variables
  PERF ~ firm_size_large + years_operation
  
  # Define indirect effects for mediation analysis
  indirect_ka := h1 * h4
  indirect_tp := h2 * h5  
  indirect_inn := h3 * h6
  total_indirect := indirect_ka + indirect_tp + indirect_inn
'

# Convert firm_size to dummy variable
data$firm_size_large <- ifelse(data$firm_size == "Large", 1, 0)

# Add years_operation if not present (simulate based on firm characteristics)
if (!"years_operation" %in% names(data)) {
  set.seed(42)
  data$years_operation <- ifelse(data$firm_size == "Large",
                                sample(8:45, nrow(data), replace = TRUE),
                                sample(3:25, nrow(data), replace = TRUE))
}

# Fit the structural model
structural_fit <- sem(structural_model, data = data, estimator = "MLR")

# Check structural model fit
cat("\n=== STRUCTURAL MODEL RESULTS ===\n")
summary(structural_fit, fit.measures = TRUE, standardized = TRUE)

# Extract fit indices
fit_indices_structural <- fitMeasures(structural_fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))
cat("\n=== STRUCTURAL MODEL FIT INDICES ===\n")
print(fit_indices_structural)

# =============================================================================
# MODERATION ANALYSIS
# =============================================================================

# Create interaction terms for moderation analysis
data$fdi_gp_interaction <- data$has_fdi * data$government_policy
data$ka_gp_interaction <- data$knowledge_absorption * data$government_policy
data$tp_gp_interaction <- data$task_performance * data$government_policy
data$inn_gp_interaction <- data$innovation_score * data$government_policy

# Moderation model
moderation_model <- '
  # Measurement Model (simplified for moderation)
  KA =~ ka1_technical_manuals + ka2_staff_training + ka3_adapt_technology + ka4_commercialize_knowledge
  TP =~ tp1_production_efficiency + tp2_quality_control + tp3_order_fulfillment + tp4_employee_productivity
  INN =~ rd_spending_pct + new_products_3yr + innovation_automation + innovation_quality_mgmt
  PERF =~ avg_roi_pct + avg_roa_pct + export_intensity_pct + capacity_utilization_pct + market_share_lagos_pct
  
  # Main effects
  PERF ~ has_fdi + KA + TP + INN + government_policy
  
  # Moderation effects (H11-H14)
  PERF ~ fdi_gp_interaction + ka_gp_interaction + tp_gp_interaction + inn_gp_interaction
  
  # Control variables
  PERF ~ firm_size_large + years_operation
'

# Fit moderation model
moderation_fit <- sem(moderation_model, data = data, estimator = "MLR")

cat("\n=== MODERATION ANALYSIS RESULTS ===\n")
summary(moderation_fit, fit.measures = TRUE, standardized = TRUE)

# =============================================================================
# HYPOTHESIS TESTING SUMMARY
# =============================================================================

# Extract parameter estimates
param_estimates <- parameterEstimates(structural_fit, standardized = TRUE)

# Filter for structural paths
structural_paths <- param_estimates[param_estimates$op == "~" & param_estimates$lhs %in% c("KA", "TP", "INN", "PERF"), ]

cat("\n=== HYPOTHESIS TESTING RESULTS ===\n")
cat("Path Coefficients (Standardized):\n")
print(structural_paths[, c("lhs", "rhs", "std.all", "pvalue")])

# Test specific hypotheses
hypotheses <- data.frame(
  Hypothesis = c("H1: FDI → KA", "H2: FDI → TP", "H3: FDI → INN", 
                "H4: KA → PERF", "H5: TP → PERF", "H6: INN → PERF", "H7: FR → PERF"),
  Path = c("KA ~ has_fdi", "TP ~ has_fdi", "INN ~ has_fdi",
           "PERF ~ KA", "PERF ~ TP", "PERF ~ INN", "PERF ~ FR"),
  Supported = c("TBD", "TBD", "TBD", "TBD", "TBD", "TBD", "TBD")
)

# Extract indirect effects for mediation
indirect_effects <- parameterEstimates(structural_fit, standardized = TRUE)
indirect_results <- indirect_effects[indirect_effects$label %in% c("indirect_ka", "indirect_tp", "indirect_inn", "total_indirect"), ]

cat("\n=== MEDIATION ANALYSIS (Indirect Effects) ===\n")
print(indirect_results[, c("label", "est", "std.all", "pvalue")])

# =============================================================================
# MULTI-GROUP ANALYSIS (SME vs Large Firms)
# =============================================================================

# Multi-group analysis by firm size
multigroup_fit <- sem(structural_model, data = data, group = "firm_size", estimator = "MLR")

cat("\n=== MULTI-GROUP ANALYSIS (SME vs Large) ===\n")
summary(multigroup_fit, fit.measures = TRUE, standardized = TRUE)

# Test for group differences
group_comparison <- compareFit(structural_fit, multigroup_fit)
cat("\n=== GROUP INVARIANCE TEST ===\n")
print(group_comparison)

# =============================================================================
# ROBUSTNESS CHECKS
# =============================================================================

# 1. Common Method Bias Test (Harman's Single Factor Test)
# Extract all indicator variables
indicators <- data[c("ka1_technical_manuals", "ka2_staff_training", "ka3_adapt_technology", "ka4_commercialize_knowledge",
                    "tp1_production_efficiency", "tp2_quality_control", "tp3_order_fulfillment", "tp4_employee_productivity",
                    "rd_spending_pct", "new_products_3yr", "innovation_automation", "innovation_quality_mgmt",
                    "gp1_tax_incentives", "gp2_regulatory_stability", "gp3_infrastructure_support", "gp4_permit_ease")]

# Perform EFA with single factor
single_factor_efa <- fa(indicators, nfactors = 1, rotate = "none")
variance_explained <- single_factor_efa$Vaccounted[2, 1] * 100

cat("\n=== COMMON METHOD BIAS TEST ===\n")
cat("Single factor variance explained:", round(variance_explained, 2), "%\n")
cat("CMB concern:", ifelse(variance_explained > 50, "YES (>50%)", "NO (<50%)"), "\n")

# 2. Alternative Model Comparison
# Test alternative model where all paths go directly from FDI to Performance
alternative_model <- '
  # Measurement Model
  KA =~ ka1_technical_manuals + ka2_staff_training + ka3_adapt_technology + ka4_commercialize_knowledge
  TP =~ tp1_production_efficiency + tp2_quality_control + tp3_order_fulfillment + tp4_employee_productivity
  INN =~ rd_spending_pct + new_products_3yr + innovation_automation + innovation_quality_mgmt
  FR =~ skilled_workforce_pct + training_hours_annual + modern_equipment + reinvestment_rate_pct
  PERF =~ avg_roi_pct + avg_roa_pct + export_intensity_pct + capacity_utilization_pct + market_share_lagos_pct
  
  # Direct effects only (no mediation)
  PERF ~ has_fdi + KA + TP + INN + FR + firm_size_large + years_operation
'

alternative_fit <- sem(alternative_model, data = data, estimator = "MLR")

# Compare models
model_comparison <- anova(structural_fit, alternative_fit)
cat("\n=== MODEL COMPARISON ===\n")
print(model_comparison)

# =============================================================================
# VISUALIZATION
# =============================================================================

# Create path diagram
png("sem_path_diagram.png", width = 1200, height = 800, res = 150)
semPaths(structural_fit, 
         what = "std",
         layout = "tree2",
         style = "lisrel",
         curve = 1,
         rotation = 2,
         title = TRUE,
         curvePivot = TRUE,
         edge.label.cex = 0.8,
         node.width = 1.2,
         node.height = 0.8)
dev.off()

# Create fit indices comparison plot
fit_comparison <- data.frame(
  Model = c("Measurement", "Structural", "Moderation", "Multi-group"),
  CFI = c(fit_indices_measurement["cfi"], fit_indices_structural["cfi"], 
          fitMeasures(moderation_fit, "cfi"), fitMeasures(multigroup_fit, "cfi")),
  RMSEA = c(fit_indices_measurement["rmsea"], fit_indices_structural["rmsea"],
            fitMeasures(moderation_fit, "rmsea"), fitMeasures(multigroup_fit, "rmsea")),
  SRMR = c(fit_indices_measurement["srmr"], fit_indices_structural["srmr"],
           fitMeasures(moderation_fit, "srmr"), fitMeasures(multigroup_fit, "srmr"))
)

# Plot fit indices
fit_plot <- ggplot(fit_comparison, aes(x = Model)) +
  geom_point(aes(y = CFI, color = "CFI"), size = 3) +
  geom_point(aes(y = RMSEA * 10, color = "RMSEA x10"), size = 3) +  # Scale RMSEA for visibility
  geom_point(aes(y = SRMR * 10, color = "SRMR x10"), size = 3) +    # Scale SRMR for visibility
  geom_hline(yintercept = 0.90, linetype = "dashed", alpha = 0.7) +  # CFI threshold
  labs(title = "Model Fit Indices Comparison",
       y = "Fit Index Value",
       color = "Index") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("model_fit_comparison.png", fit_plot, width = 10, height = 6, dpi = 300)

# =============================================================================
# EXPORT RESULTS
# =============================================================================

# Export parameter estimates
write_csv(param_estimates, "sem_parameter_estimates.csv")

# Export fit indices summary
fit_summary <- data.frame(
  Model = c("Measurement", "Structural", "Moderation", "Multi-group"),
  ChiSq = c(fit_indices_measurement["chisq"], fit_indices_structural["chisq"],
            fitMeasures(moderation_fit, "chisq"), fitMeasures(multigroup_fit, "chisq")),
  df = c(fit_indices_measurement["df"], fit_indices_structural["df"],
         fitMeasures(moderation_fit, "df"), fitMeasures(multigroup_fit, "df")),
  CFI = c(fit_indices_measurement["cfi"], fit_indices_structural["cfi"],
          fitMeasures(moderation_fit, "cfi"), fitMeasures(multigroup_fit, "cfi")),
  TLI = c(fit_indices_measurement["tli"], fit_indices_structural["tli"],
          fitMeasures(moderation_fit, "tli"), fitMeasures(multigroup_fit, "tli")),
  RMSEA = c(fit_indices_measurement["rmsea"], fit_indices_structural["rmsea"],
            fitMeasures(moderation_fit, "rmsea"), fitMeasures(multigroup_fit, "rmsea")),
  SRMR = c(fit_indices_measurement["srmr"], fit_indices_structural["srmr"],
           fitMeasures(moderation_fit, "srmr"), fitMeasures(multigroup_fit, "srmr"))
)

write_csv(fit_summary, "model_fit_summary.csv")

# Export hypothesis testing results
hypothesis_results <- data.frame(
  Hypothesis = c("H1: FDI → KA", "H2: FDI → TP", "H3: FDI → INN", 
                "H4: KA → PERF", "H5: TP → PERF", "H6: INN → PERF", "H7: FR → PERF",
                "H8: FDI → KA → PERF", "H9: FDI → TP → PERF", "H10: FDI → INN → PERF"),
  Type = c(rep("Direct", 7), rep("Indirect", 3)),
  Estimate = c(rep(NA, 10)),  # To be filled based on results
  P_Value = c(rep(NA, 10)),   # To be filled based on results
  Supported = c(rep("TBD", 10))  # To be determined based on significance
)

write_csv(hypothesis_results, "hypothesis_testing_results.csv")

cat("\n=== ANALYSIS COMPLETED ===\n")
cat("Files exported:\n")
cat("- sem_parameter_estimates.csv\n")
cat("- model_fit_summary.csv\n")
cat("- hypothesis_testing_results.csv\n")
cat("- sem_path_diagram.png\n")
cat("- model_fit_comparison.png\n")

cat("\n=== NEXT STEPS ===\n")
cat("1. Review fit indices and modify model if needed\n")
cat("2. Interpret hypothesis testing results\n")
cat("3. Conduct additional robustness checks\n")
cat("4. Prepare results for thesis writing\n")