# =============================================================================
# SYNTHETIC FDI SURVEY DATA GENERATOR
# PhD Research: The Influence of Foreign Direct Investment on the Performance 
# of Food Processing Firms in Lagos, Nigeria
# =============================================================================

# Load required libraries
library(dplyr)
library(tidyr)
library(MASS)
library(corrplot)
library(openxlsx)
library(haven)

# Set seed for reproducibility
set.seed(12345)

# =============================================================================
# 1. SAMPLE ALLOCATION (Neyman Allocation)
# =============================================================================
N_sme <- 300
N_large <- 150
N_total <- 450
n_sme <- 200
n_large <- 100
n_total <- 300

# =============================================================================
# 2. FIRM CHARACTERISTICS GENERATION
# =============================================================================

# Generate firm IDs
firm_ids <- paste0("FIRM_", sprintf("%03d", 1:n_total))

# Firm size distribution (SME vs Large)
firm_size <- c(rep("SME", n_sme), rep("Large", n_large))

# Years of operation (SMEs: 5-25 years, Large: 10-40 years)
years_operation <- c(
  sample(5:25, n_sme, replace = TRUE),
  sample(10:40, n_large, replace = TRUE)
)

# Number of employees (correlated with firm size)
employees <- c(
  sample(c(1:50, 51:250), n_sme, replace = TRUE, prob = c(0.7, 0.3)),
  sample(c(251:500, 501:1000), n_large, replace = TRUE, prob = c(0.6, 0.4))
)

# Annual revenue in Naira (correlated with firm size and employees)
revenue_categories <- c(
  sample(c("<50M", "50M-500M"), n_sme, replace = TRUE, prob = c(0.6, 0.4)),
  sample(c("50M-500M", "500M-5B", ">5B"), n_large, replace = TRUE, prob = c(0.3, 0.5, 0.2))
)

# Ownership type (higher FDI probability for large firms)
ownership <- c(
  sample(c("Local", "Foreign-owned", "Joint venture"), n_sme, replace = TRUE, prob = c(0.7, 0.1, 0.2)),
  sample(c("Local", "Foreign-owned", "Joint venture"), n_large, replace = TRUE, prob = c(0.4, 0.3, 0.3))
)

# =============================================================================
# 3. FDI ENGAGEMENT GENERATION
# =============================================================================

# FDI partnerships (higher probability for large firms and foreign/joint ownership)
fdi_prob_sme <- ifelse(ownership[1:n_sme] %in% c("Foreign-owned", "Joint venture"), 0.8, 0.3)
fdi_prob_large <- ifelse(ownership[(n_sme+1):n_total] %in% c("Foreign-owned", "Joint venture"), 0.9, 0.5)

fdi_partnership <- c(
  rbinom(n_sme, 1, fdi_prob_sme),
  rbinom(n_large, 1, fdi_prob_large)
)

# Type of FDI (only for firms with FDI)
fdi_type <- rep(NA, n_total)
fdi_firms <- which(fdi_partnership == 1)
fdi_type[fdi_firms] <- sample(c("Equity", "Joint venture", "Technology transfer", "Management contract"), 
                              length(fdi_firms), replace = TRUE, prob = c(0.3, 0.3, 0.25, 0.15))

# Years with FDI partnership
years_fdi <- rep(0, n_total)
years_fdi[fdi_firms] <- sample(1:min(years_operation[fdi_firms]), length(fdi_firms), replace = TRUE)

# =============================================================================
# 4. KNOWLEDGE ABSORPTION SCALE (Zahra & George)
# =============================================================================

# Generate correlated knowledge absorption items (1-5 scale)
# Higher scores for firms with FDI and larger firms
fdi_effect <- ifelse(fdi_partnership == 1, 0.8, 0)
size_effect <- ifelse(firm_size == "Large", 0.5, 0)
base_ka <- 2.5 + fdi_effect + size_effect + rnorm(n_total, 0, 0.5)

# Ensure values are within 1-5 range
base_ka <- pmax(1, pmin(5, base_ka))

# Generate correlated items
ka_cor <- 0.7  # High correlation between KA items
ka_cov <- matrix(ka_cor, nrow = 4, ncol = 4)
diag(ka_cov) <- 1

ka_items <- mvrnorm(n_total, rep(0, 4), ka_cov)
ka_items <- ka_items + base_ka
ka_items <- pmax(1, pmin(5, ka_items))

ka1 <- round(ka_items[,1], 1)  # Technical manuals acquisition
ka2 <- round(ka_items[,2], 1)  # Staff training from partners
ka3 <- round(ka_items[,3], 1)  # Technology adaptation
ka4 <- round(ka_items[,4], 1)  # Knowledge commercialization

# =============================================================================
# 5. TASK PERFORMANCE SCALE (Koopmans et al.)
# =============================================================================

# Generate correlated task performance items (1-5 scale)
# Performance influenced by FDI, knowledge absorption, and firm size
fdi_perf_effect <- ifelse(fdi_partnership == 1, 0.6, 0)
ka_perf_effect <- (rowMeans(ka_items) - 3) * 0.4
size_perf_effect <- ifelse(firm_size == "Large", 0.3, 0)
base_tp <- 2.8 + fdi_perf_effect + ka_perf_effect + size_perf_effect + rnorm(n_total, 0, 0.4)

base_tp <- pmax(1, pmin(5, base_tp))

# Generate correlated items
tp_cor <- 0.6
tp_cov <- matrix(tp_cor, nrow = 4, ncol = 4)
diag(tp_cov) <- 1

tp_items <- mvrnorm(n_total, rep(0, 4), tp_cov)
tp_items <- tp_items + base_tp
tp_items <- pmax(1, pmin(5, tp_items))

tp1 <- round(tp_items[,1], 1)  # Production efficiency
tp2 <- round(tp_items[,2], 1)  # Quality control
tp3 <- round(tp_items[,3], 1)  # Order fulfillment time
tp4 <- round(tp_items[,4], 1)  # Employee productivity

# =============================================================================
# 6. INNOVATION METRICS (OECD Oslo Manual)
# =============================================================================

# R&D spending as % of revenue (higher for FDI firms and large firms)
rd_base <- 2 + ifelse(fdi_partnership == 1, 1.5, 0) + ifelse(firm_size == "Large", 1, 0) + rnorm(n_total, 0, 1)
rd_spending <- pmax(0, pmin(15, rd_base))

# New products launched (past 3 years)
new_products <- rpois(n_total, lambda = 2 + ifelse(fdi_partnership == 1, 2, 0) + ifelse(firm_size == "Large", 1, 0))

# Process innovations (binary indicators)
iot_adoption <- rbinom(n_total, 1, prob = 0.3 + ifelse(fdi_partnership == 1, 0.2, 0) + ifelse(firm_size == "Large", 0.2, 0))
automation <- rbinom(n_total, 1, prob = 0.4 + ifelse(fdi_partnership == 1, 0.3, 0) + ifelse(firm_size == "Large", 0.2, 0))
quality_mgmt <- rbinom(n_total, 1, prob = 0.6 + ifelse(fdi_partnership == 1, 0.2, 0))

# =============================================================================
# 7. FIRM RESOURCES
# =============================================================================

# Human Resources
skilled_workforce <- 40 + ifelse(fdi_partnership == 1, 15, 0) + ifelse(firm_size == "Large", 10, 0) + rnorm(n_total, 0, 8)
skilled_workforce <- pmax(10, pmin(90, skilled_workforce))

training_hours <- 20 + ifelse(fdi_partnership == 1, 15, 0) + ifelse(firm_size == "Large", 10, 0) + rnorm(n_total, 0, 5)
training_hours <- pmax(5, pmin(80, training_hours))

# Technological Resources
modern_equipment <- rbinom(n_total, 1, prob = 0.5 + ifelse(fdi_partnership == 1, 0.3, 0) + ifelse(firm_size == "Large", 0.2, 0))
machinery_age <- 8 + ifelse(modern_equipment == 1, -3, 2) + rnorm(n_total, 0, 2)
machinery_age <- pmax(1, pmin(20, machinery_age))

# Financial Resources
credit_access <- sample(c("Easy", "Moderate", "Difficult"), n_total, replace = TRUE, 
                       prob = c(0.3 + ifelse(firm_size == "Large", 0.2, 0), 0.5, 0.2))

reinvestment_rate <- 15 + ifelse(fdi_partnership == 1, 5, 0) + ifelse(firm_size == "Large", 3, 0) + rnorm(n_total, 0, 5)
reinvestment_rate <- pmax(5, pmin(50, reinvestment_rate))

# =============================================================================
# 8. GOVERNMENT POLICY PERCEPTION (1-7 scale)
# =============================================================================

# Generate correlated government policy items
# Lower scores for larger firms (more critical), higher for FDI firms (beneficiaries)
gp_base <- 4 + ifelse(fdi_partnership == 1, 0.5, 0) - ifelse(firm_size == "Large", 0.3, 0) + rnorm(n_total, 0, 0.8)
gp_base <- pmax(1, pmin(7, gp_base))

gp_cor <- 0.5
gp_cov <- matrix(gp_cor, nrow = 4, ncol = 4)
diag(gp_cov) <- 1

gp_items <- mvrnorm(n_total, rep(0, 4), gp_cov)
gp_items <- gp_items + gp_base
gp_items <- pmax(1, pmin(7, gp_items))

gp1 <- round(gp_items[,1], 1)  # Tax incentives effectiveness
gp2 <- round(gp_items[,2], 1)  # Regulatory stability
gp3 <- round(gp_items[,3], 1)  # Infrastructure support
gp4 <- round(gp_items[,4], 1)  # Ease of obtaining permits

# =============================================================================
# 9. PERFORMANCE METRICS
# =============================================================================

# Financial Performance (influenced by FDI, KA, Innovation, and firm size)
fdi_fin_effect <- ifelse(fdi_partnership == 1, 3, 0)
ka_fin_effect <- (rowMeans(ka_items) - 3) * 2
innovation_effect <- (rd_spending/10) + (new_products * 0.5)
size_fin_effect <- ifelse(firm_size == "Large", 2, 0)

# ROI (Return on Investment)
roi_base <- 8 + fdi_fin_effect + ka_fin_effect + innovation_effect + size_fin_effect + rnorm(n_total, 0, 3)
roi <- pmax(-5, pmin(35, roi_base))

# ROA (Return on Assets)
roa_base <- 6 + fdi_fin_effect + ka_fin_effect + innovation_effect + size_fin_effect + rnorm(n_total, 0, 2.5)
roa <- pmax(-3, pmin(25, roa_base))

# Export intensity
export_intensity <- 15 + ifelse(fdi_partnership == 1, 20, 0) + ifelse(firm_size == "Large", 10, 0) + rnorm(n_total, 0, 8)
export_intensity <- pmax(0, pmin(80, export_intensity))

# Operational Performance
capacity_utilization <- 70 + ifelse(fdi_partnership == 1, 10, 0) + ifelse(firm_size == "Large", 5, 0) + rnorm(n_total, 0, 8)
capacity_utilization <- pmax(30, pmin(100, capacity_utilization))

market_share <- 5 + ifelse(fdi_partnership == 1, 3, 0) + ifelse(firm_size == "Large", 4, 0) + rnorm(n_total, 0, 3)
market_share <- pmax(0.5, pmin(25, market_share))

# =============================================================================
# 10. CREATE MAIN DATASET
# =============================================================================

# Create the main dataset
fdi_data <- data.frame(
  # Firm identification
  firm_id = firm_ids,
  firm_size = firm_size,
  date_survey = sample(seq(as.Date('2024-01-01'), as.Date('2024-03-31'), by="day"), n_total),
  
  # Section A: Firm Background
  years_operation = years_operation,
  employees = employees,
  revenue_category = revenue_categories,
  ownership_type = ownership,
  
  # FDI Engagement
  fdi_partnership = fdi_partnership,
  fdi_type = fdi_type,
  years_fdi = years_fdi,
  
  # Section B: Knowledge Absorption (1-5 scale)
  ka1_technical_manuals = ka1,
  ka2_staff_training = ka2,
  ka3_technology_adaptation = ka3,
  ka4_knowledge_commercialization = ka4,
  
  # Section C: Task Performance (1-5 scale)
  tp1_production_efficiency = tp1,
  tp2_quality_control = tp2,
  tp3_order_fulfillment = tp3,
  tp4_employee_productivity = tp4,
  
  # Section D: Innovation
  rd_spending_pct = round(rd_spending, 1),
  new_products_3years = new_products,
  iot_adoption = iot_adoption,
  automation_adoption = automation,
  quality_management = quality_mgmt,
  
  # Section E: Firm Resources
  skilled_workforce_pct = round(skilled_workforce, 1),
  training_hours_annual = round(training_hours, 1),
  modern_equipment = modern_equipment,
  machinery_age = round(machinery_age, 1),
  credit_access = credit_access,
  reinvestment_rate_pct = round(reinvestment_rate, 1),
  
  # Section F: Government Policy (1-7 scale)
  gp1_tax_incentives = gp1,
  gp2_regulatory_stability = gp2,
  gp3_infrastructure_support = gp3,
  gp4_permits_ease = gp4,
  
  # Section G: Performance Metrics
  roi_pct = round(roi, 1),
  roa_pct = round(roa, 1),
  export_intensity_pct = round(export_intensity, 1),
  capacity_utilization_pct = round(capacity_utilization, 1),
  market_share_pct = round(market_share, 1)
)

# =============================================================================
# 11. CREATE COMPOSITE SCORES FOR SEM ANALYSIS
# =============================================================================

# Knowledge Absorption composite (average of 4 items)
fdi_data$knowledge_absorption <- round(rowMeans(fdi_data[,c("ka1_technical_manuals", "ka2_staff_training", 
                                                           "ka3_technology_adaptation", "ka4_knowledge_commercialization")]), 2)

# Task Performance composite (average of 4 items)
fdi_data$task_performance <- round(rowMeans(fdi_data[,c("tp1_production_efficiency", "tp2_quality_control", 
                                                        "tp3_order_fulfillment", "tp4_employee_productivity")]), 2)

# Innovation composite (standardized combination of R&D, new products, process innovations)
innovation_score <- scale(fdi_data$rd_spending_pct) + scale(fdi_data$new_products_3years) + 
                   scale(fdi_data$iot_adoption + fdi_data$automation_adoption + fdi_data$quality_management)
fdi_data$innovation <- round(as.numeric(innovation_score), 2)

# Government Policy composite (average of 4 items)
fdi_data$government_policy <- round(rowMeans(fdi_data[,c("gp1_tax_incentives", "gp2_regulatory_stability", 
                                                         "gp3_infrastructure_support", "gp4_permits_ease")]), 2)

# Firm Performance composite (standardized combination of financial and operational metrics)
performance_score <- scale(fdi_data$roi_pct) + scale(fdi_data$roa_pct) + scale(fdi_data$export_intensity_pct) + 
                    scale(fdi_data$capacity_utilization_pct) + scale(fdi_data$market_share_pct)
fdi_data$firm_performance <- round(as.numeric(performance_score), 2)

# =============================================================================
# 12. ADD MODERATION INTERACTION TERMS
# =============================================================================

# FDI × Government Policy interaction
fdi_data$fdi_gov_interaction <- fdi_data$fdi_partnership * fdi_data$government_policy

# Knowledge Absorption × Government Policy interaction
fdi_data$ka_gov_interaction <- fdi_data$knowledge_absorption * fdi_data$government_policy

# Task Performance × Government Policy interaction
fdi_data$tp_gov_interaction <- fdi_data$task_performance * fdi_data$government_policy

# Innovation × Government Policy interaction
fdi_data$innovation_gov_interaction <- fdi_data$innovation * fdi_data$government_policy

# =============================================================================
# 13. EXPORT DATASETS
# =============================================================================

# Export main dataset as CSV
write.csv(fdi_data, "fdi_survey_data.csv", row.names = FALSE)

# Export as Excel with multiple sheets
wb <- createWorkbook()

# Main data sheet
addWorksheet(wb, "Survey Data")
writeData(wb, "Survey Data", fdi_data)

# Summary statistics sheet
summary_stats <- data.frame(
  Variable = names(fdi_data),
  Mean = sapply(fdi_data, function(x) if(is.numeric(x)) mean(x, na.rm = TRUE) else NA),
  SD = sapply(fdi_data, function(x) if(is.numeric(x)) sd(x, na.rm = TRUE) else NA),
  Min = sapply(fdi_data, function(x) if(is.numeric(x)) min(x, na.rm = TRUE) else NA),
  Max = sapply(fdi_data, function(x) if(is.numeric(x)) max(x, na.rm = TRUE) else NA),
  N = sapply(fdi_data, function(x) sum(!is.na(x)))
)

addWorksheet(wb, "Summary Statistics")
writeData(wb, "Summary Statistics", summary_stats)

# Save Excel file
saveWorkbook(wb, "fdi_survey_data.xlsx", overwrite = TRUE)

# Export as RData for R users
save(fdi_data, file = "fdi_survey_data.RData")

# Export as Stata format
write_dta(fdi_data, "fdi_survey_data.dta")

# =============================================================================
# 14. GENERATE DATA DOCUMENTATION
# =============================================================================

# Create codebook
codebook <- data.frame(
  Variable = names(fdi_data),
  Description = c(
    "Unique firm identifier",
    "Firm size category (SME or Large)",
    "Date of survey completion",
    "Years of firm operation",
    "Number of employees",
    "Annual revenue category in Naira",
    "Ownership type",
    "Has FDI partnership (1=Yes, 0=No)",
    "Type of FDI partnership",
    "Years with FDI partnership",
    "Technical manuals acquisition (1-5 scale)",
    "Staff training from partners (1-5 scale)",
    "Technology adaptation (1-5 scale)",
    "Knowledge commercialization (1-5 scale)",
    "Production efficiency (1-5 scale)",
    "Quality control (1-5 scale)",
    "Order fulfillment time (1-5 scale)",
    "Employee productivity (1-5 scale)",
    "R&D spending as % of revenue",
    "New products launched in past 3 years",
    "IoT systems adoption (1=Yes, 0=No)",
    "Automation adoption (1=Yes, 0=No)",
    "Quality management adoption (1=Yes, 0=No)",
    "Percentage of skilled workforce",
    "Annual training hours per employee",
    "Uses modern equipment (1=Yes, 0=No)",
    "Age of primary machinery in years",
    "Access to credit rating",
    "Reinvestment rate as % of profit",
    "Tax incentives effectiveness (1-7 scale)",
    "Regulatory stability (1-7 scale)",
    "Infrastructure support (1-7 scale)",
    "Ease of obtaining permits (1-7 scale)",
    "Return on Investment (%)",
    "Return on Assets (%)",
    "Export intensity (%)",
    "Production capacity utilization (%)",
    "Market share in Lagos (%)",
    "Knowledge Absorption composite score",
    "Task Performance composite score",
    "Innovation composite score",
    "Government Policy composite score",
    "Firm Performance composite score",
    "FDI × Government Policy interaction",
    "Knowledge Absorption × Government Policy interaction",
    "Task Performance × Government Policy interaction",
    "Innovation × Government Policy interaction"
  ),
  Scale = c(
    "Nominal", "Nominal", "Date", "Ratio", "Ratio", "Ordinal", "Nominal",
    "Binary", "Nominal", "Ratio", "Interval", "Interval", "Interval", "Interval",
    "Interval", "Interval", "Interval", "Interval", "Ratio", "Count",
    "Binary", "Binary", "Binary", "Ratio", "Ratio", "Binary", "Ratio",
    "Ordinal", "Ratio", "Interval", "Interval", "Interval", "Interval",
    "Ratio", "Ratio", "Ratio", "Ratio", "Ratio", "Interval", "Interval",
    "Interval", "Interval", "Interval", "Interval", "Interval", "Interval", "Interval"
  )
)

write.csv(codebook, "fdi_survey_codebook.csv", row.names = FALSE)

# =============================================================================
# 15. GENERATE CORRELATION MATRIX FOR VALIDATION
# =============================================================================

# Select numeric variables for correlation
numeric_vars <- fdi_data[,sapply(fdi_data, is.numeric)]
correlation_matrix <- cor(numeric_vars, use = "complete.obs")

# Save correlation matrix
write.csv(correlation_matrix, "fdi_correlation_matrix.csv")

# =============================================================================
# 16. PRINT SUMMARY STATISTICS
# =============================================================================

cat("=== FDI SURVEY DATA GENERATION COMPLETE ===\n")
cat("Total firms generated:", n_total, "\n")
cat("SMEs:", n_sme, "\n")
cat("Large firms:", n_large, "\n")
cat("Firms with FDI:", sum(fdi_partnership), "\n")
cat("Response rate:", round(sum(fdi_partnership)/n_total*100, 1), "%\n\n")

cat("=== KEY STATISTICS ===\n")
cat("Knowledge Absorption (mean ± sd):", round(mean(fdi_data$knowledge_absorption, na.rm = TRUE), 2), 
    "±", round(sd(fdi_data$knowledge_absorption, na.rm = TRUE), 2), "\n")
cat("Task Performance (mean ± sd):", round(mean(fdi_data$task_performance, na.rm = TRUE), 2), 
    "±", round(sd(fdi_data$task_performance, na.rm = TRUE), 2), "\n")
cat("Innovation (mean ± sd):", round(mean(fdi_data$innovation, na.rm = TRUE), 2), 
    "±", round(sd(fdi_data$innovation, na.rm = TRUE), 2), "\n")
cat("Government Policy (mean ± sd):", round(mean(fdi_data$government_policy, na.rm = TRUE), 2), 
    "±", round(sd(fdi_data$government_policy, na.rm = TRUE), 2), "\n")
cat("Firm Performance (mean ± sd):", round(mean(fdi_data$firm_performance, na.rm = TRUE), 2), 
    "±", round(sd(fdi_data$firm_performance, na.rm = TRUE), 2), "\n\n")

cat("=== FILES GENERATED ===\n")
cat("1. fdi_survey_data.csv - Main dataset\n")
cat("2. fdi_survey_data.xlsx - Excel format with summary\n")
cat("3. fdi_survey_data.RData - R format\n")
cat("4. fdi_survey_data.dta - Stata format\n")
cat("5. fdi_survey_codebook.csv - Variable documentation\n")
cat("6. fdi_correlation_matrix.csv - Correlation matrix\n")

cat("\n=== READY FOR SEM ANALYSIS ===\n")
cat("The dataset includes all variables needed for your structural equation model:\n")
cat("- Latent variables: Knowledge Absorption, Innovation, Government Policy, Performance\n")
cat("- Interaction terms for moderation analysis\n")
cat("- Composite scores for SEM modeling\n")
cat("- Realistic correlations between variables\n")