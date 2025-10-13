# =============================================================================
# SYNTHETIC DATA GENERATION FOR FDI STUDY
# The Influence of Foreign Direct Investment on Food Processing Firms in Lagos
# =============================================================================

# Load required libraries
library(dplyr)
library(readr)
library(MASS)
library(psych)
library(corrplot)

# Set seed for reproducibility
set.seed(42)

# =============================================================================
# SAMPLE PARAMETERS
# =============================================================================

n_sme <- 200      # SME firms
n_large <- 100    # Large firms
n_total <- 300    # Total sample

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Function to generate correlated Likert scale responses
generate_likert <- function(n, mean_val, sd_val, min_val = 1, max_val = 5) {
  values <- rnorm(n, mean_val, sd_val)
  values <- pmax(min_val, pmin(max_val, round(values)))
  return(as.integer(values))
}

# Function to generate correlated 7-point scale responses
generate_7point <- function(n, mean_val, sd_val, min_val = 1, max_val = 7) {
  values <- rnorm(n, mean_val, sd_val)
  values <- pmax(min_val, pmin(max_val, round(values)))
  return(as.integer(values))
}

# Function to generate realistic percentages
generate_percentage <- function(n, mean_val, sd_val, min_val = 0, max_val = 100) {
  values <- rnorm(n, mean_val, sd_val)
  values <- pmax(min_val, pmin(max_val, values))
  return(round(values, 1))
}

# =============================================================================
# GENERATE FIRM CHARACTERISTICS
# =============================================================================

# Create firm IDs and basic structure
firms <- data.frame(
  firm_id = c(paste0("SME", sprintf("%03d", 1:n_sme)), 
              paste0("LRG", sprintf("%03d", 1:n_large))),
  firm_size = c(rep("SME", n_sme), rep("Large", n_large)),
  survey_date = sample(seq(as.Date("2024-01-15"), as.Date("2024-06-30"), by = "day"), 
                      n_total, replace = TRUE)
)

# Years of operation (correlated with firm size)
firms$years_operation <- ifelse(firms$firm_size == "SME",
                               sample(3:25, n_sme, replace = TRUE, prob = exp(-0.1 * (3:25 - 8))),
                               sample(8:45, n_large, replace = TRUE, prob = exp(-0.05 * (8:45 - 15))))

# Number of employees (categorical)
firms$employees_cat <- ifelse(firms$firm_size == "SME",
                             sample(c("1-50", "51-250"), n_sme, replace = TRUE, prob = c(0.6, 0.4)),
                             sample(c("251-500", "500+"), n_large, replace = TRUE, prob = c(0.7, 0.3)))

# Annual revenue (categorical, correlated with size)
revenue_probs_sme <- c(0.4, 0.45, 0.13, 0.02)  # <50M, 50M-500M, 500M-5B, >5B
revenue_probs_large <- c(0.05, 0.25, 0.55, 0.15)

firms$revenue_cat <- ifelse(firms$firm_size == "SME",
                           sample(c("<50M", "50M-500M", "500M-5B", ">5B"), n_sme, 
                                 replace = TRUE, prob = revenue_probs_sme),
                           sample(c("<50M", "50M-500M", "500M-5B", ">5B"), n_large, 
                                 replace = TRUE, prob = revenue_probs_large))

# Ownership type (affects FDI likelihood)
ownership_probs <- c(0.55, 0.25, 0.20)  # Local, Foreign-owned, Joint venture
firms$ownership_type <- sample(c("Local", "Foreign-owned", "Joint venture"), 
                              n_total, replace = TRUE, prob = ownership_probs)

# =============================================================================
# FDI ENGAGEMENT
# =============================================================================

# FDI partnership (higher probability for larger firms and foreign/JV ownership)
fdi_prob <- ifelse(firms$ownership_type == "Local", 0.3,
                  ifelse(firms$ownership_type == "Foreign-owned", 0.85, 0.75))
fdi_prob <- ifelse(firms$firm_size == "Large", fdi_prob * 1.2, fdi_prob)
fdi_prob <- pmin(fdi_prob, 0.95)  # Cap at 95%

firms$has_fdi <- rbinom(n_total, 1, fdi_prob)

# Type of FDI (for firms with FDI)
fdi_firms <- which(firms$has_fdi == 1)
n_fdi <- length(fdi_firms)

firms$fdi_type <- NA
if (n_fdi > 0) {
  firms$fdi_type[fdi_firms] <- sample(c("Equity", "Joint venture", "Technology transfer", "Management contract"),
                                     n_fdi, replace = TRUE, prob = c(0.35, 0.30, 0.25, 0.10))
}

# Years with FDI partnership
firms$years_fdi <- NA
if (n_fdi > 0) {
  firms$years_fdi[fdi_firms] <- pmin(sample(1:15, n_fdi, replace = TRUE, 
                                           prob = exp(-0.2 * (1:15 - 3))),
                                    firms$years_operation[fdi_firms] - 1)
}

# =============================================================================
# KNOWLEDGE ABSORPTION (Section B)
# =============================================================================

# Generate correlated responses for Knowledge Absorption scale
# Higher scores for firms with FDI, especially technology transfer and JV

ka_base_mean <- ifelse(firms$has_fdi == 1, 3.8, 2.5)
ka_base_mean <- ifelse(firms$fdi_type %in% c("Technology transfer", "Joint venture"), 
                      ka_base_mean + 0.5, ka_base_mean)

# Create correlation structure for KA items
ka_correlation <- matrix(c(
  1.0, 0.7, 0.6, 0.5,
  0.7, 1.0, 0.8, 0.6,
  0.6, 0.8, 1.0, 0.7,
  0.5, 0.6, 0.7, 1.0
), nrow = 4)

# Generate correlated KA responses
ka_data <- matrix(nrow = n_total, ncol = 4)
for (i in 1:n_total) {
  ka_means <- rep(ka_base_mean[i], 4)
  ka_sds <- rep(0.8, 4)
  
  # Generate correlated normal variables
  ka_raw <- mvrnorm(1, ka_means, ka_correlation * 0.64)  # 0.64 = 0.8^2
  
  # Convert to Likert scale
  ka_data[i, ] <- pmax(1, pmin(5, round(ka_raw)))
}

firms$ka1_technical_manuals <- ka_data[, 1]
firms$ka2_staff_training <- ka_data[, 2]
firms$ka3_adapt_technology <- ka_data[, 3]
firms$ka4_commercialize_knowledge <- ka_data[, 4]

# =============================================================================
# TASK PERFORMANCE (Section C)
# =============================================================================

# Task Performance correlated with FDI and firm size
tp_base_mean <- 3.2 + (firms$has_fdi * 0.6) + (ifelse(firms$firm_size == "Large", 0.3, 0))

# Create correlation structure for TP items
tp_correlation <- matrix(c(
  1.0, 0.8, 0.7, 0.75,
  0.8, 1.0, 0.6, 0.7,
  0.7, 0.6, 1.0, 0.65,
  0.75, 0.7, 0.65, 1.0
), nrow = 4)

# Generate correlated TP responses
tp_data <- matrix(nrow = n_total, ncol = 4)
for (i in 1:n_total) {
  tp_means <- rep(tp_base_mean[i], 4)
  tp_sds <- rep(0.7, 4)
  
  tp_raw <- mvrnorm(1, tp_means, tp_correlation * 0.49)  # 0.49 = 0.7^2
  tp_data[i, ] <- pmax(1, pmin(5, round(tp_raw)))
}

firms$tp1_production_efficiency <- tp_data[, 1]
firms$tp2_quality_control <- tp_data[, 2]
firms$tp3_order_fulfillment <- tp_data[, 3]
firms$tp4_employee_productivity <- tp_data[, 4]

# =============================================================================
# INNOVATION (Section D)
# =============================================================================

# R&D spending as % of revenue
rd_mean <- ifelse(firms$firm_size == "Large", 2.5, 1.2)
rd_mean <- rd_mean + (firms$has_fdi * 0.8)
firms$rd_spending_pct <- generate_percentage(n_total, rd_mean, 1.2, 0, 8)

# New products launched (past 3 years)
new_products_mean <- ifelse(firms$firm_size == "Large", 4, 2)
new_products_mean <- new_products_mean + (firms$has_fdi * 1.5)
firms$new_products_3yr <- pmax(0, round(rnorm(n_total, new_products_mean, 2)))

# Process innovations (multiple selection possible)
innovation_prob <- 0.3 + (firms$has_fdi * 0.3) + (ifelse(firms$firm_size == "Large", 0.2, 0))

firms$innovation_iot <- rbinom(n_total, 1, innovation_prob * 0.6)
firms$innovation_automation <- rbinom(n_total, 1, innovation_prob * 0.8)
firms$innovation_quality_mgmt <- rbinom(n_total, 1, innovation_prob * 0.9)
firms$innovation_other <- rbinom(n_total, 1, innovation_prob * 0.4)

# =============================================================================
# FIRM RESOURCES (Section E)
# =============================================================================

# Human Resources
skilled_workforce_mean <- ifelse(firms$firm_size == "Large", 65, 45)
skilled_workforce_mean <- skilled_workforce_mean + (firms$has_fdi * 10)
firms$skilled_workforce_pct <- generate_percentage(n_total, skilled_workforce_mean, 15, 10, 95)

training_hours_mean <- ifelse(firms$firm_size == "Large", 35, 20)
training_hours_mean <- training_hours_mean + (firms$has_fdi * 15)
firms$training_hours_annual <- pmax(5, round(rnorm(n_total, training_hours_mean, 12)))

# Technological Resources
modern_equipment_prob <- 0.6 + (firms$has_fdi * 0.25) + (ifelse(firms$firm_size == "Large", 0.15, 0))
firms$modern_equipment <- rbinom(n_total, 1, modern_equipment_prob)

machinery_age_mean <- ifelse(firms$modern_equipment == 1, 6, 12)
machinery_age_mean <- machinery_age_mean - (firms$has_fdi * 2)
firms$machinery_age_years <- pmax(1, round(rnorm(n_total, machinery_age_mean, 4)))

# Financial Resources
credit_access_probs <- matrix(c(
  0.4, 0.45, 0.15,  # SME without FDI: Easy, Moderate, Difficult
  0.6, 0.35, 0.05,  # SME with FDI
  0.5, 0.4, 0.1,    # Large without FDI
  0.75, 0.23, 0.02  # Large with FDI
), nrow = 4, byrow = TRUE)

credit_index <- ifelse(firms$firm_size == "SME", 
                      ifelse(firms$has_fdi == 1, 2, 1),
                      ifelse(firms$has_fdi == 1, 4, 3))

firms$credit_access <- sapply(1:n_total, function(i) {
  sample(c("Easy", "Moderate", "Difficult"), 1, prob = credit_access_probs[credit_index[i], ])
})

reinvestment_mean <- ifelse(firms$firm_size == "Large", 18, 12)
reinvestment_mean <- reinvestment_mean + (firms$has_fdi * 5)
firms$reinvestment_rate_pct <- generate_percentage(n_total, reinvestment_mean, 8, 2, 40)

# =============================================================================
# GOVERNMENT POLICY PERCEPTION (Section F)
# =============================================================================

# Government policy ratings (7-point scale)
# Slightly lower ratings reflecting real challenges in Nigeria

gp_base_mean <- 3.8  # Slightly below midpoint
gp_fdi_bonus <- 0.4  # FDI firms might have slightly better perception
gp_size_bonus <- 0.2  # Larger firms might have better access

# Create correlation structure for GP items
gp_correlation <- matrix(c(
  1.0, 0.6, 0.5, 0.4,
  0.6, 1.0, 0.7, 0.6,
  0.5, 0.7, 1.0, 0.5,
  0.4, 0.6, 0.5, 1.0
), nrow = 4)

# Generate correlated GP responses
gp_data <- matrix(nrow = n_total, ncol = 4)
for (i in 1:n_total) {
  gp_mean_adj <- gp_base_mean + (firms$has_fdi[i] * gp_fdi_bonus) + 
                 (ifelse(firms$firm_size[i] == "Large", gp_size_bonus, 0))
  gp_means <- rep(gp_mean_adj, 4)
  gp_sds <- rep(1.2, 4)
  
  gp_raw <- mvrnorm(1, gp_means, gp_correlation * 1.44)  # 1.44 = 1.2^2
  gp_data[i, ] <- pmax(1, pmin(7, round(gp_raw)))
}

firms$gp1_tax_incentives <- gp_data[, 1]
firms$gp2_regulatory_stability <- gp_data[, 2]
firms$gp3_infrastructure_support <- gp_data[, 3]
firms$gp4_permit_ease <- gp_data[, 4]

# =============================================================================
# PERFORMANCE METRICS (Section G)
# =============================================================================

# Financial Performance (influenced by FDI, size, and other factors)
# ROI (Return on Investment)
roi_base <- ifelse(firms$firm_size == "Large", 12, 8)
roi_fdi_effect <- firms$has_fdi * 4
roi_innovation_effect <- (firms$new_products_3yr / 5) * 2
roi_mean <- roi_base + roi_fdi_effect + roi_innovation_effect

firms$avg_roi_pct <- generate_percentage(n_total, roi_mean, 6, -5, 35)

# ROA (Return on Assets)
roa_base <- ifelse(firms$firm_size == "Large", 8, 5)
roa_fdi_effect <- firms$has_fdi * 3
roa_efficiency_effect <- (rowMeans(tp_data) - 3) * 2
roa_mean <- roa_base + roa_fdi_effect + roa_efficiency_effect

firms$avg_roa_pct <- generate_percentage(n_total, roa_mean, 4, -3, 25)

# Export intensity
export_base <- ifelse(firms$ownership_type == "Local", 5, 
                     ifelse(firms$ownership_type == "Foreign-owned", 25, 15))
export_fdi_effect <- firms$has_fdi * 8
export_mean <- export_base + export_fdi_effect

firms$export_intensity_pct <- generate_percentage(n_total, export_mean, 12, 0, 80)

# Operational Performance
# Production capacity utilization
capacity_base <- 72
capacity_fdi_effect <- firms$has_fdi * 8
capacity_efficiency_effect <- (firms$tp1_production_efficiency - 3) * 4
capacity_mean <- capacity_base + capacity_fdi_effect + capacity_efficiency_effect

firms$capacity_utilization_pct <- generate_percentage(n_total, capacity_mean, 12, 30, 98)

# Market share in Lagos
market_share_base <- ifelse(firms$firm_size == "Large", 3.5, 1.2)
market_share_fdi_effect <- firms$has_fdi * 1.5
market_share_performance_effect <- (rowMeans(tp_data) - 3) * 0.8
market_share_mean <- market_share_base + market_share_fdi_effect + market_share_performance_effect

firms$market_share_lagos_pct <- generate_percentage(n_total, market_share_mean, 2, 0.1, 15)

# =============================================================================
# CREATE COMPOSITE VARIABLES FOR SEM
# =============================================================================

# Knowledge Absorption composite
firms$knowledge_absorption <- rowMeans(cbind(firms$ka1_technical_manuals,
                                            firms$ka2_staff_training,
                                            firms$ka3_adapt_technology,
                                            firms$ka4_commercialize_knowledge))

# Task Performance composite
firms$task_performance <- rowMeans(cbind(firms$tp1_production_efficiency,
                                        firms$tp2_quality_control,
                                        firms$tp3_order_fulfillment,
                                        firms$tp4_employee_productivity))

# Innovation composite
firms$innovation_score <- scale(firms$rd_spending_pct)[,1] * 0.4 + 
                         scale(firms$new_products_3yr)[,1] * 0.4 +
                         scale(firms$innovation_iot + firms$innovation_automation + 
                              firms$innovation_quality_mgmt + firms$innovation_other)[,1] * 0.2

# Government Policy composite
firms$government_policy <- rowMeans(cbind(firms$gp1_tax_incentives,
                                         firms$gp2_regulatory_stability,
                                         firms$gp3_infrastructure_support,
                                         firms$gp4_permit_ease))

# Firm Resources composite
firms$firm_resources <- scale(firms$skilled_workforce_pct)[,1] * 0.3 +
                       scale(firms$training_hours_annual)[,1] * 0.2 +
                       scale(firms$modern_equipment)[,1] * 0.2 +
                       scale(20 - firms$machinery_age_years)[,1] * 0.15 +  # Reverse coded
                       scale(ifelse(firms$credit_access == "Easy", 3, 
                                   ifelse(firms$credit_access == "Moderate", 2, 1)))[,1] * 0.15

# Overall Performance composite
firms$overall_performance <- scale(firms$avg_roi_pct)[,1] * 0.25 +
                            scale(firms$avg_roa_pct)[,1] * 0.25 +
                            scale(firms$export_intensity_pct)[,1] * 0.2 +
                            scale(firms$capacity_utilization_pct)[,1] * 0.15 +
                            scale(firms$market_share_lagos_pct)[,1] * 0.15

# =============================================================================
# DATA QUALITY CHECKS
# =============================================================================

cat("=== DATA QUALITY SUMMARY ===\n")
cat("Total observations:", nrow(firms), "\n")
cat("Firms with FDI:", sum(firms$has_fdi), "\n")
cat("FDI percentage:", round(mean(firms$has_fdi) * 100, 1), "%\n")

# Check for missing values
missing_summary <- sapply(firms, function(x) sum(is.na(x)))
cat("\nMissing values by variable:\n")
print(missing_summary[missing_summary > 0])

# Correlation matrix for key variables
key_vars <- c("knowledge_absorption", "task_performance", "innovation_score", 
              "government_policy", "firm_resources", "overall_performance")
cor_matrix <- cor(firms[key_vars], use = "complete.obs")

cat("\n=== CORRELATION MATRIX (Key Variables) ===\n")
print(round(cor_matrix, 3))

# =============================================================================
# EXPORT DATA
# =============================================================================

# Export full dataset
write_csv(firms, "fdi_synthetic_dataset.csv")

# Export dataset for SEM analysis (key variables only)
sem_data <- firms %>%
  select(firm_id, firm_size, has_fdi, years_fdi, fdi_type,
         ka1_technical_manuals, ka2_staff_training, ka3_adapt_technology, ka4_commercialize_knowledge,
         tp1_production_efficiency, tp2_quality_control, tp3_order_fulfillment, tp4_employee_productivity,
         rd_spending_pct, new_products_3yr, innovation_iot, innovation_automation, innovation_quality_mgmt,
         gp1_tax_incentives, gp2_regulatory_stability, gp3_infrastructure_support, gp4_permit_ease,
         skilled_workforce_pct, training_hours_annual, modern_equipment, reinvestment_rate_pct,
         avg_roi_pct, avg_roa_pct, export_intensity_pct, capacity_utilization_pct, market_share_lagos_pct,
         knowledge_absorption, task_performance, innovation_score, government_policy, firm_resources, overall_performance)

write_csv(sem_data, "fdi_sem_dataset.csv")

# Export summary statistics
summary_stats <- firms %>%
  group_by(firm_size, has_fdi) %>%
  summarise(
    n = n(),
    avg_performance = mean(overall_performance),
    avg_knowledge_absorption = mean(knowledge_absorption),
    avg_innovation = mean(innovation_score),
    avg_roi = mean(avg_roi_pct),
    avg_roa = mean(avg_roa_pct),
    .groups = 'drop'
  )

write_csv(summary_stats, "summary_statistics.csv")

cat("\n=== FILES EXPORTED ===\n")
cat("- fdi_synthetic_dataset.csv (complete dataset)\n")
cat("- fdi_sem_dataset.csv (SEM analysis ready)\n")
cat("- summary_statistics.csv (descriptive statistics)\n")

# Create correlation plot
png("correlation_matrix.png", width = 800, height = 600, res = 100)
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, 
         title = "Correlation Matrix - Key Variables")
dev.off()

cat("- correlation_matrix.png (correlation visualization)\n")

cat("\n=== DATA GENERATION COMPLETED ===\n")