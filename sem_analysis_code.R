# ============================================================================
# SEM ANALYSIS CODE
# The Influence of FDI on Food Processing Firm Performance in Lagos, Nigeria
# Jiangsu University - PhD Research
# ============================================================================

# Install and load required packages
# install.packages(c("lavaan", "semPlot", "psych", "ggplot2", "corrplot", "semTools"))

library(lavaan)
library(semPlot)
library(psych)
library(ggplot2)
library(corrplot)
library(semTools)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

# Load dataset
data <- read.csv("fdi_lagos_survey_data.csv")

# Convert categorical variables to dummy codes
data$fdi_dummy <- ifelse(data$has_fdi == "Yes", 1, 0)
data$size_dummy <- ifelse(data$firm_size == "Large", 1, 0)
data$modern_equip_dummy <- ifelse(data$modern_equipment == "Yes", 1, 0)

# Create composite scores
data$KA_score <- (data$ka1_technical_manuals + data$ka2_staff_training + 
                  data$ka3_adapt_technology + data$ka4_commercialize_knowledge) / 4

data$TP_score <- (data$tp1_production_efficiency + data$tp2_quality_control + 
                  data$tp3_order_fulfillment + data$tp4_employee_productivity) / 4

data$GP_score <- (data$gp1_tax_incentives + data$gp2_regulatory_stability + 
                  data$gp3_infrastructure + data$gp4_permits_ease) / 4

data$innov_index <- (data$innov_iot + data$innov_automation + 
                     data$innov_quality_mgmt + data$innov_other) / 4

# Firm performance composite
data$PERFORM_score <- scale(data$avg_roi_pct) + scale(data$avg_roa_pct) + 
                      scale(data$capacity_utilization_pct) + 
                      scale(data$export_intensity_pct)

# Display structure
str(data)
summary(data)

# ============================================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================================

cat("\n========== DESCRIPTIVE STATISTICS ==========\n")

# Sample composition
cat("\nSample Composition:\n")
table(data$firm_size)
table(data$has_fdi)
table(data$ownership_type)

# Continuous variables descriptives
cont_vars <- c("years_operation", "avg_roi_pct", "avg_roa_pct", 
               "capacity_utilization_pct", "export_intensity_pct",
               "rd_spending_pct", "new_products_3yrs",
               "KA_score", "TP_score", "GP_score")

cat("\nDescriptive Statistics for Key Variables:\n")
describe(data[, cont_vars])

# Compare FDI vs Non-FDI firms
cat("\nPerformance Comparison: FDI vs Non-FDI Firms\n")
by(data[, c("avg_roi_pct", "avg_roa_pct", "KA_score", "TP_score")], 
   data$has_fdi, describe)

# T-tests
t.test(avg_roi_pct ~ has_fdi, data = data)
t.test(KA_score ~ has_fdi, data = data)
t.test(TP_score ~ has_fdi, data = data)

# ============================================================================
# 3. RELIABILITY ANALYSIS
# ============================================================================

cat("\n========== RELIABILITY ANALYSIS ==========\n")

# Knowledge Absorption scale
KA_items <- data[, c("ka1_technical_manuals", "ka2_staff_training", 
                      "ka3_adapt_technology", "ka4_commercialize_knowledge")]
cat("\nKnowledge Absorption Scale:\n")
alpha_KA <- alpha(KA_items)
print(alpha_KA)

# Task Performance scale
TP_items <- data[, c("tp1_production_efficiency", "tp2_quality_control", 
                      "tp3_order_fulfillment", "tp4_employee_productivity")]
cat("\nTask Performance Scale:\n")
alpha_TP <- alpha(TP_items)
print(alpha_TP)

# Government Policy scale
GP_items <- data[, c("gp1_tax_incentives", "gp2_regulatory_stability", 
                      "gp3_infrastructure", "gp4_permits_ease")]
cat("\nGovernment Policy Scale:\n")
alpha_GP <- alpha(GP_items)
print(alpha_GP)

# ============================================================================
# 4. CORRELATION ANALYSIS
# ============================================================================

cat("\n========== CORRELATION ANALYSIS ==========\n")

# Select key variables for correlation matrix
cor_vars <- c("fdi_dummy", "KA_score", "TP_score", "rd_spending_pct", 
              "new_products_3yrs", "GP_score", "avg_roi_pct", "avg_roa_pct",
              "capacity_utilization_pct", "export_intensity_pct")

cor_matrix <- cor(data[, cor_vars], use = "complete.obs")
print(round(cor_matrix, 3))

# Visualize correlation matrix
png("correlation_matrix.png", width = 800, height = 800)
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45,
         addCoef.col = "black", number.cex = 0.7,
         title = "Correlation Matrix - FDI Study Variables")
dev.off()

# ============================================================================
# 5. CONFIRMATORY FACTOR ANALYSIS (CFA)
# ============================================================================

cat("\n========== CONFIRMATORY FACTOR ANALYSIS ==========\n")

# Measurement model
measurement_model <- '
  # Latent variables
  KA =~ ka1_technical_manuals + ka2_staff_training + 
        ka3_adapt_technology + ka4_commercialize_knowledge
        
  TP =~ tp1_production_efficiency + tp2_quality_control + 
        tp3_order_fulfillment + tp4_employee_productivity
        
  INN =~ rd_spending_pct + new_products_3yrs + innov_index
  
  GP =~ gp1_tax_incentives + gp2_regulatory_stability + 
        gp3_infrastructure + gp4_permits_ease
        
  PERFORM =~ avg_roi_pct + avg_roa_pct + capacity_utilization_pct + 
             export_intensity_pct
'

# Fit CFA model
cfa_fit <- cfa(measurement_model, data = data, std.lv = TRUE)

# Model summary
summary(cfa_fit, fit.measures = TRUE, standardized = TRUE)

# Fit indices
fit_indices <- fitMeasures(cfa_fit, c("chisq", "df", "pvalue", "cfi", "tli", 
                                       "rmsea", "rmsea.ci.lower", "rmsea.ci.upper",
                                       "srmr", "gfi", "agfi"))
print(fit_indices)

# Standardized factor loadings
standardizedSolution(cfa_fit)

# Visualize CFA model
png("cfa_model.png", width = 1000, height = 800)
semPaths(cfa_fit, what = "std", layout = "tree2", 
         edge.label.cex = 0.8, sizeMan = 8, sizeLat = 12,
         style = "lisrel", title = TRUE)
dev.off()

# ============================================================================
# 6. STRUCTURAL EQUATION MODEL (SEM) - MAIN MODEL
# ============================================================================

cat("\n========== STRUCTURAL EQUATION MODEL (MAIN MODEL) ==========\n")

# Full structural model with mediation
structural_model <- '
  # Measurement model
  KA =~ ka1_technical_manuals + ka2_staff_training + 
        ka3_adapt_technology + ka4_commercialize_knowledge
        
  TP =~ tp1_production_efficiency + tp2_quality_control + 
        tp3_order_fulfillment + tp4_employee_productivity
        
  INN =~ rd_spending_pct + new_products_3yrs + innov_index
  
  GP =~ gp1_tax_incentives + gp2_regulatory_stability + 
        gp3_infrastructure + gp4_permits_ease
        
  PERFORM =~ avg_roi_pct + avg_roa_pct + capacity_utilization_pct + 
             export_intensity_pct
  
  # Structural model - Direct effects of FDI on mediators
  KA ~ a1*fdi_dummy
  TP ~ a2*fdi_dummy
  INN ~ a3*fdi_dummy
  
  # Mediators predict performance
  PERFORM ~ b1*KA + b2*TP + b3*INN
  
  # Direct effect of FDI on performance
  PERFORM ~ c_prime*fdi_dummy
  
  # Control variables
  PERFORM ~ size_dummy + years_operation + skilled_workforce_pct
  
  # Indirect effects (mediation)
  indirect1 := a1 * b1  # FDI -> KA -> PERFORM
  indirect2 := a2 * b2  # FDI -> TP -> PERFORM
  indirect3 := a3 * b3  # FDI -> INN -> PERFORM
  total_indirect := indirect1 + indirect2 + indirect3
  
  # Total effect
  total := c_prime + total_indirect
'

# Fit SEM model
sem_fit <- sem(structural_model, data = data, std.lv = TRUE)

# Model summary
summary(sem_fit, fit.measures = TRUE, standardized = TRUE, rsquare = TRUE)

# Fit indices
sem_fit_indices <- fitMeasures(sem_fit, c("chisq", "df", "pvalue", "cfi", "tli", 
                                           "rmsea", "rmsea.ci.lower", "rmsea.ci.upper",
                                           "srmr", "aic", "bic"))
cat("\nModel Fit Indices:\n")
print(sem_fit_indices)

# Parameter estimates
cat("\nStandardized Parameter Estimates:\n")
standardizedSolution(sem_fit)

# R-squared for endogenous variables
cat("\nR-squared Values:\n")
inspect(sem_fit, "r2")

# Visualize SEM model
png("sem_model_main.png", width = 1200, height = 900)
semPaths(sem_fit, what = "std", whatLabels = "std",
         layout = "tree2", edge.label.cex = 0.7,
         sizeMan = 6, sizeLat = 10, style = "lisrel",
         title = TRUE, curvePivot = TRUE)
dev.off()

# ============================================================================
# 7. MODERATION ANALYSIS - Government Policy
# ============================================================================

cat("\n========== MODERATION ANALYSIS ==========\n")

# Create interaction terms (mean-centered for better interpretation)
data$fdi_centered <- scale(data$fdi_dummy, center = TRUE, scale = FALSE)
data$GP_centered <- scale(data$GP_score, center = TRUE, scale = FALSE)
data$KA_centered <- scale(data$KA_score, center = TRUE, scale = FALSE)
data$INN_centered <- scale(data$rd_spending_pct, center = TRUE, scale = FALSE)

data$fdi_x_gp <- data$fdi_centered * data$GP_centered
data$ka_x_gp <- data$KA_centered * data$GP_centered
data$inn_x_gp <- data$INN_centered * data$GP_centered

# Moderation model
moderation_model <- '
  # Measurement model
  PERFORM =~ avg_roi_pct + avg_roa_pct + capacity_utilization_pct + 
             export_intensity_pct
  
  # Main effects and interactions
  PERFORM ~ fdi_centered + GP_centered + fdi_x_gp
  PERFORM ~ KA_score + ka_x_gp
  PERFORM ~ rd_spending_pct + inn_x_gp
  
  # Control variables
  PERFORM ~ size_dummy + years_operation
'

# Fit moderation model
mod_fit <- sem(moderation_model, data = data, std.lv = TRUE)

# Results
summary(mod_fit, fit.measures = TRUE, standardized = TRUE)

# ============================================================================
# 8. MEDIATION ANALYSIS WITH BOOTSTRAPPING
# ============================================================================

cat("\n========== MEDIATION ANALYSIS (BOOTSTRAPPED) ==========\n")

# Bootstrap for indirect effects (5000 iterations)
set.seed(42)
boot_fit <- sem(structural_model, data = data, std.lv = TRUE, 
                se = "bootstrap", bootstrap = 5000)

# Bootstrap confidence intervals for indirect effects
cat("\nBootstrapped Indirect Effects:\n")
parameterEstimates(boot_fit, boot.ci.type = "perc", level = 0.95) %>%
  filter(label %in% c("indirect1", "indirect2", "indirect3", "total_indirect"))

# ============================================================================
# 9. MULTI-GROUP ANALYSIS (SMEs vs Large Firms)
# ============================================================================

cat("\n========== MULTI-GROUP ANALYSIS ==========\n")

# Configural invariance
mg_config <- sem(structural_model, data = data, group = "firm_size", std.lv = TRUE)
summary(mg_config, fit.measures = TRUE)

# Metric invariance
mg_metric <- sem(structural_model, data = data, group = "firm_size", 
                 group.equal = "loadings", std.lv = TRUE)
summary(mg_metric, fit.measures = TRUE)

# Scalar invariance
mg_scalar <- sem(structural_model, data = data, group = "firm_size", 
                 group.equal = c("loadings", "intercepts"), std.lv = TRUE)
summary(mg_scalar, fit.measures = TRUE)

# Compare models
cat("\nModel Comparison:\n")
anova(mg_config, mg_metric, mg_scalar)

# ============================================================================
# 10. HYPOTHESIS TESTING SUMMARY
# ============================================================================

cat("\n========== HYPOTHESIS TESTING SUMMARY ==========\n")

# Extract path coefficients
paths <- parameterEstimates(sem_fit) %>%
  filter(op == "~") %>%
  select(lhs, rhs, est, se, z, pvalue, ci.lower, ci.upper)

cat("\nDirect Effects:\n")
print(paths)

# Hypothesis decisions (p < 0.05)
cat("\nHypothesis Testing Results:\n")
cat("H1: FDI → Knowledge Absorption:", 
    ifelse(paths[paths$lhs == "KA" & paths$rhs == "fdi_dummy", "pvalue"] < 0.05, 
           "SUPPORTED", "NOT SUPPORTED"), "\n")

cat("H2: FDI → Task Performance:", 
    ifelse(paths[paths$lhs == "TP" & paths$rhs == "fdi_dummy", "pvalue"] < 0.05, 
           "SUPPORTED", "NOT SUPPORTED"), "\n")

cat("H3: FDI → Innovation:", 
    ifelse(paths[paths$lhs == "INN" & paths$rhs == "fdi_dummy", "pvalue"] < 0.05, 
           "SUPPORTED", "NOT SUPPORTED"), "\n")

cat("H4: Knowledge Absorption → Performance:", 
    ifelse(paths[paths$lhs == "PERFORM" & paths$rhs == "KA", "pvalue"] < 0.05, 
           "SUPPORTED", "NOT SUPPORTED"), "\n")

# ============================================================================
# 11. EXPORT RESULTS
# ============================================================================

# Save results to file
sink("sem_results_summary.txt")
cat("========== SEM ANALYSIS RESULTS ==========\n")
cat("Date:", format(Sys.Date(), "%B %d, %Y"), "\n\n")

cat("SAMPLE CHARACTERISTICS:\n")
print(table(data$firm_size))
print(table(data$has_fdi))

cat("\n\nRELIABILITY COEFFICIENTS:\n")
cat("Knowledge Absorption: α =", round(alpha_KA$total$raw_alpha, 3), "\n")
cat("Task Performance: α =", round(alpha_TP$total$raw_alpha, 3), "\n")
cat("Government Policy: α =", round(alpha_GP$total$raw_alpha, 3), "\n")

cat("\n\nMODEL FIT INDICES:\n")
print(sem_fit_indices)

cat("\n\nPATH COEFFICIENTS:\n")
print(paths)

cat("\n\nR-SQUARED VALUES:\n")
print(inspect(sem_fit, "r2"))

sink()

cat("\n✅ Analysis complete! Results saved to 'sem_results_summary.txt'\n")
cat("✅ Visualizations saved:\n")
cat("   - correlation_matrix.png\n")
cat("   - cfa_model.png\n")
cat("   - sem_model_main.png\n")
