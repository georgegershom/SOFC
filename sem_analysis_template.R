# =============================================================================
# SEM ANALYSIS TEMPLATE FOR FDI RESEARCH
# PhD Research: The Influence of Foreign Direct Investment on the Performance 
# of Food Processing Firms in Lagos, Nigeria
# =============================================================================

# Load required libraries
library(lavaan)
library(semTools)
library(semPlot)
library(psych)
library(corrplot)
library(ggplot2)
library(dplyr)

# Load the dataset
fdi_data <- read.csv("fdi_survey_data.csv")

# =============================================================================
# 1. PRELIMINARY ANALYSIS
# =============================================================================

# Descriptive statistics
describe(fdi_data[,c("knowledge_absorption", "task_performance", "innovation", 
                    "government_policy", "firm_performance")])

# Correlation matrix
cor_matrix <- cor(fdi_data[,c("knowledge_absorption", "task_performance", "innovation", 
                             "government_policy", "firm_performance", "fdi_partnership")], 
                 use = "complete.obs")
print(cor_matrix)

# Visualize correlation matrix
corrplot(cor_matrix, method = "color", type = "upper", 
         addCoef.col = "black", tl.cex = 0.8)

# =============================================================================
# 2. RELIABILITY ANALYSIS
# =============================================================================

# Knowledge Absorption reliability
ka_items <- fdi_data[,c("ka1_technical_manuals", "ka2_staff_training", 
                       "ka3_technology_adaptation", "ka4_knowledge_commercialization")]
alpha_ka <- alpha(ka_items)
print("Knowledge Absorption Reliability:")
print(alpha_ka)

# Task Performance reliability
tp_items <- fdi_data[,c("tp1_production_efficiency", "tp2_quality_control", 
                       "tp3_order_fulfillment", "tp4_employee_productivity")]
alpha_tp <- alpha(tp_items)
print("Task Performance Reliability:")
print(alpha_tp)

# Government Policy reliability
gp_items <- fdi_data[,c("gp1_tax_incentives", "gp2_regulatory_stability", 
                       "gp3_infrastructure_support", "gp4_permits_ease")]
alpha_gp <- alpha(gp_items)
print("Government Policy Reliability:")
print(alpha_gp)

# =============================================================================
# 3. CONFIRMATORY FACTOR ANALYSIS (CFA)
# =============================================================================

# Measurement model
cfa_model <- '
  # Knowledge Absorption (reflective)
  KA =~ ka1_technical_manuals + ka2_staff_training + 
       ka3_technology_adaptation + ka4_knowledge_commercialization
  
  # Task Performance (reflective)
  TP =~ tp1_production_efficiency + tp2_quality_control + 
       tp3_order_fulfillment + tp4_employee_productivity
  
  # Government Policy (reflective)
  GP =~ gp1_tax_incentives + gp2_regulatory_stability + 
       gp3_infrastructure_support + gp4_permits_ease
'

# Fit CFA model
cfa_fit <- cfa(cfa_model, data = fdi_data, std.lv = TRUE)
summary(cfa_fit, fit.measures = TRUE, standardized = TRUE)

# =============================================================================
# 4. STRUCTURAL EQUATION MODELING
# =============================================================================

# Full SEM model
sem_model <- '
  # Measurement model
  KA =~ ka1_technical_manuals + ka2_staff_training + 
       ka3_technology_adaptation + ka4_knowledge_commercialization
  
  TP =~ tp1_production_efficiency + tp2_quality_control + 
       tp3_order_fulfillment + tp4_employee_productivity
  
  GP =~ gp1_tax_incentives + gp2_regulatory_stability + 
       gp3_infrastructure_support + gp4_permits_ease
  
  # Innovation (observed composite)
  INN =~ rd_spending_pct + new_products_3years + iot_adoption + 
        automation_adoption + quality_management
  
  # Performance (observed composite)
  PERF =~ roi_pct + roa_pct + export_intensity_pct + 
         capacity_utilization_pct + market_share_pct
  
  # Structural model - Direct effects
  PERF ~ KA + TP + INN + fdi_partnership
  
  # Mediation paths
  KA ~ fdi_partnership
  TP ~ fdi_partnership
  INN ~ fdi_partnership
  
  # Moderation effects
  PERF ~ fdi_gov_interaction + ka_gov_interaction + 
         tp_gov_interaction + innovation_gov_interaction
'

# Fit SEM model
sem_fit <- sem(sem_model, data = fdi_data, std.lv = TRUE)
summary(sem_fit, fit.measures = TRUE, standardized = TRUE)

# =============================================================================
# 5. MODEL FIT INDICES
# =============================================================================

# Extract fit indices
fit_measures <- fitMeasures(sem_fit)
print("Model Fit Indices:")
print(fit_measures[c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "rmsea.ci.lower", 
                    "rmsea.ci.upper", "srmr", "aic", "bic")])

# =============================================================================
# 6. HYPOTHESIS TESTING
# =============================================================================

# Extract standardized coefficients
std_coef <- standardizedSolution(sem_fit)
print("Standardized Coefficients:")
print(std_coef[std_coef$op == "~", c("lhs", "rhs", "est.std", "pvalue")])

# =============================================================================
# 7. MEDIATION ANALYSIS
# =============================================================================

# Test mediation effects using bootstrapping
mediation_results <- semTools::mediate(sem_fit, 
                                      med = c("KA", "TP", "INN"), 
                                      ind = "fdi_partnership", 
                                      dep = "PERF", 
                                      boot.ci.type = "bca.simple")

print("Mediation Analysis Results:")
print(mediation_results)

# =============================================================================
# 8. MODERATION ANALYSIS
# =============================================================================

# Test moderation effects
moderation_results <- std_coef[std_coef$op == "~" & 
                              grepl("interaction", std_coef$rhs), 
                              c("lhs", "rhs", "est.std", "pvalue")]

print("Moderation Effects:")
print(moderation_results)

# =============================================================================
# 9. MULTI-GROUP ANALYSIS (SMEs vs Large Firms)
# =============================================================================

# Test measurement invariance
configural <- cfa(cfa_model, data = fdi_data, group = "firm_size")
metric <- cfa(cfa_model, data = fdi_data, group = "firm_size", 
              group.equal = "loadings")
scalar <- cfa(cfa_model, data = fdi_data, group = "firm_size", 
              group.equal = c("loadings", "intercepts"))

# Compare models
anova(configural, metric, scalar)

# =============================================================================
# 10. PATH DIAGRAM
# =============================================================================

# Create path diagram
semPaths(sem_fit, what = "std", layout = "tree", 
         style = "lisrel", curve = 2, 
         nCharNodes = 0, sizeMan = 8, sizeLat = 10,
         edge.label.cex = 0.8, label.cex = 0.8)

# =============================================================================
# 11. ROBUSTNESS CHECKS
# =============================================================================

# Common method bias test (Harman's single factor)
harman_model <- '
  CMV =~ ka1_technical_manuals + ka2_staff_training + 
        ka3_technology_adaptation + ka4_knowledge_commercialization +
        tp1_production_efficiency + tp2_quality_control + 
        tp3_order_fulfillment + tp4_employee_productivity +
        gp1_tax_incentives + gp2_regulatory_stability + 
        gp3_infrastructure_support + gp4_permits_ease
'

harman_fit <- cfa(harman_model, data = fdi_data)
print("Common Method Bias Test:")
print(fitMeasures(harman_fit)[c("chisq", "df", "cfi", "rmsea")])

# =============================================================================
# 12. REPORT GENERATION
# =============================================================================

# Create summary report
cat("=== SEM ANALYSIS SUMMARY ===\n")
cat("Sample size:", nrow(fdi_data), "\n")
cat("Firms with FDI:", sum(fdi_data$fdi_partnership), "\n")
cat("SMEs:", sum(fdi_data$firm_size == "SME"), "\n")
cat("Large firms:", sum(fdi_data$firm_size == "Large"), "\n\n")

cat("=== MODEL FIT ===\n")
cat("CFI:", round(fit_measures["cfi"], 3), "\n")
cat("TLI:", round(fit_measures["tli"], 3), "\n")
cat("RMSEA:", round(fit_measures["rmsea"], 3), "\n")
cat("SRMR:", round(fit_measures["srmr"], 3), "\n\n")

cat("=== KEY FINDINGS ===\n")
# Extract key path coefficients
key_paths <- std_coef[std_coef$op == "~" & std_coef$lhs == "PERF", 
                     c("rhs", "est.std", "pvalue")]
print(key_paths)

cat("\n=== ANALYSIS COMPLETE ===\n")