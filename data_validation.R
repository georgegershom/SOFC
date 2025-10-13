# =============================================================================
# DATA VALIDATION AND PRELIMINARY ANALYSIS
# The Influence of Foreign Direct Investment on Food Processing Firms in Lagos
# =============================================================================

# Load required libraries
library(dplyr)
library(readr)
library(ggplot2)
library(psych)
library(VIM)
library(corrplot)
library(car)
library(moments)
library(nortest)

# =============================================================================
# LOAD AND INSPECT DATA
# =============================================================================

# Load the synthetic dataset
data <- read_csv("fdi_synthetic_dataset.csv")

cat("=== DATASET OVERVIEW ===\n")
cat("Dimensions:", dim(data)[1], "rows x", dim(data)[2], "columns\n")
cat("Date range:", min(data$survey_date), "to", max(data$survey_date), "\n\n")

# Display first few rows
cat("First 5 rows:\n")
print(head(data, 5))

# Data structure
cat("\n=== DATA STRUCTURE ===\n")
str(data)

# =============================================================================
# MISSING DATA ANALYSIS
# =============================================================================

cat("\n=== MISSING DATA ANALYSIS ===\n")

# Count missing values by variable
missing_counts <- sapply(data, function(x) sum(is.na(x)))
missing_pct <- round(missing_counts / nrow(data) * 100, 2)

missing_summary <- data.frame(
  Variable = names(missing_counts),
  Missing_Count = missing_counts,
  Missing_Percent = missing_pct
) %>%
  filter(Missing_Count > 0) %>%
  arrange(desc(Missing_Count))

if (nrow(missing_summary) > 0) {
  print(missing_summary)
  
  # Create missing data pattern plot
  png("missing_data_pattern.png", width = 800, height = 600, res = 100)
  VIM::aggr(data, col = c('navyblue', 'red'), numbers = TRUE, sortVars = TRUE)
  dev.off()
  
} else {
  cat("No missing data found!\n")
}

# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================

cat("\n=== DESCRIPTIVE STATISTICS ===\n")

# Categorical variables summary
categorical_vars <- c("firm_size", "employees_cat", "revenue_cat", "ownership_type", 
                     "has_fdi", "fdi_type", "credit_access")

for (var in categorical_vars) {
  if (var %in% names(data)) {
    cat("\n", var, ":\n")
    print(table(data[[var]], useNA = "ifany"))
    cat("Proportions:\n")
    print(round(prop.table(table(data[[var]], useNA = "ifany")) * 100, 1))
  }
}

# Continuous variables summary
continuous_vars <- c("years_operation", "rd_spending_pct", "new_products_3yr", 
                    "skilled_workforce_pct", "training_hours_annual", "machinery_age_years",
                    "reinvestment_rate_pct", "avg_roi_pct", "avg_roa_pct", 
                    "export_intensity_pct", "capacity_utilization_pct", "market_share_lagos_pct")

continuous_data <- data[continuous_vars[continuous_vars %in% names(data)]]
desc_stats <- describe(continuous_data)

cat("\n=== CONTINUOUS VARIABLES DESCRIPTIVE STATISTICS ===\n")
print(round(desc_stats, 2))

# Export descriptive statistics
write_csv(as.data.frame(desc_stats), "descriptive_statistics.csv")

# =============================================================================
# DISTRIBUTION ANALYSIS
# =============================================================================

cat("\n=== DISTRIBUTION ANALYSIS ===\n")

# Test normality for key continuous variables
normality_tests <- data.frame(
  Variable = character(),
  Shapiro_W = numeric(),
  Shapiro_p = numeric(),
  Skewness = numeric(),
  Kurtosis = numeric(),
  stringsAsFactors = FALSE
)

key_continuous <- c("avg_roi_pct", "avg_roa_pct", "export_intensity_pct", 
                   "capacity_utilization_pct", "rd_spending_pct")

for (var in key_continuous) {
  if (var %in% names(data) && !all(is.na(data[[var]]))) {
    # Shapiro-Wilk test (for samples < 5000)
    if (nrow(data) <= 5000) {
      shapiro_test <- shapiro.test(data[[var]])
      shapiro_w <- shapiro_test$statistic
      shapiro_p <- shapiro_test$p.value
    } else {
      # Use Anderson-Darling for larger samples
      ad_test <- ad.test(data[[var]])
      shapiro_w <- NA
      shapiro_p <- ad_test$p.value
    }
    
    # Calculate skewness and kurtosis
    skew <- skewness(data[[var]], na.rm = TRUE)
    kurt <- kurtosis(data[[var]], na.rm = TRUE)
    
    normality_tests <- rbind(normality_tests, data.frame(
      Variable = var,
      Shapiro_W = shapiro_w,
      Shapiro_p = shapiro_p,
      Skewness = skew,
      Kurtosis = kurt
    ))
  }
}

print(normality_tests)

# Create distribution plots
png("distribution_plots.png", width = 1200, height = 800, res = 100)
par(mfrow = c(2, 3))
for (var in key_continuous[1:6]) {
  if (var %in% names(data)) {
    hist(data[[var]], main = paste("Distribution of", var), 
         xlab = var, col = "lightblue", breaks = 20)
    # Add normal curve
    x <- seq(min(data[[var]], na.rm = TRUE), max(data[[var]], na.rm = TRUE), length = 100)
    y <- dnorm(x, mean = mean(data[[var]], na.rm = TRUE), sd = sd(data[[var]], na.rm = TRUE))
    y <- y * length(data[[var]]) * (max(data[[var]], na.rm = TRUE) - min(data[[var]], na.rm = TRUE)) / 20
    lines(x, y, col = "red", lwd = 2)
  }
}
dev.off()

# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

cat("\n=== CORRELATION ANALYSIS ===\n")

# Select numeric variables for correlation analysis
numeric_vars <- sapply(data, is.numeric)
numeric_data <- data[, numeric_vars]

# Remove ID and date variables
exclude_vars <- c("firm_id", "survey_date")
numeric_data <- numeric_data[, !names(numeric_data) %in% exclude_vars]

# Calculate correlation matrix
cor_matrix <- cor(numeric_data, use = "complete.obs")

# Print correlation matrix (rounded)
cat("Correlation Matrix (first 10x10):\n")
print(round(cor_matrix[1:min(10, nrow(cor_matrix)), 1:min(10, ncol(cor_matrix))], 3))

# Create correlation heatmap
png("correlation_heatmap.png", width = 1000, height = 1000, res = 100)
corrplot(cor_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45, tl.cex = 0.7,
         title = "Correlation Matrix - All Numeric Variables")
dev.off()

# Identify high correlations (potential multicollinearity issues)
high_cor_threshold <- 0.8
high_correlations <- which(abs(cor_matrix) > high_cor_threshold & cor_matrix != 1, arr.ind = TRUE)

if (nrow(high_correlations) > 0) {
  cat("\n=== HIGH CORRELATIONS (>", high_cor_threshold, ") ===\n")
  high_cor_pairs <- data.frame(
    Var1 = rownames(cor_matrix)[high_correlations[, 1]],
    Var2 = colnames(cor_matrix)[high_correlations[, 2]],
    Correlation = cor_matrix[high_correlations]
  )
  print(high_cor_pairs)
} else {
  cat("\nNo high correlations found above threshold of", high_cor_threshold, "\n")
}

# =============================================================================
# OUTLIER DETECTION
# =============================================================================

cat("\n=== OUTLIER DETECTION ===\n")

# Function to detect outliers using IQR method
detect_outliers_iqr <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  return(which(x < lower_bound | x > upper_bound))
}

# Detect outliers for key variables
outlier_summary <- data.frame(
  Variable = character(),
  Outlier_Count = numeric(),
  Outlier_Percent = numeric(),
  stringsAsFactors = FALSE
)

key_vars_outlier <- c("avg_roi_pct", "avg_roa_pct", "export_intensity_pct", 
                     "capacity_utilization_pct", "rd_spending_pct", "new_products_3yr")

for (var in key_vars_outlier) {
  if (var %in% names(data)) {
    outliers <- detect_outliers_iqr(data[[var]])
    outlier_count <- length(outliers)
    outlier_pct <- round(outlier_count / nrow(data) * 100, 2)
    
    outlier_summary <- rbind(outlier_summary, data.frame(
      Variable = var,
      Outlier_Count = outlier_count,
      Outlier_Percent = outlier_pct
    ))
  }
}

print(outlier_summary)

# Create boxplots for outlier visualization
png("outlier_boxplots.png", width = 1200, height = 800, res = 100)
par(mfrow = c(2, 3))
for (var in key_vars_outlier) {
  if (var %in% names(data)) {
    boxplot(data[[var]], main = paste("Boxplot of", var), 
            ylab = var, col = "lightgreen")
  }
}
dev.off()

# =============================================================================
# GROUP COMPARISONS
# =============================================================================

cat("\n=== GROUP COMPARISONS ===\n")

# Compare FDI vs Non-FDI firms
fdi_comparison <- data %>%
  group_by(has_fdi) %>%
  summarise(
    n = n(),
    avg_roi = mean(avg_roi_pct, na.rm = TRUE),
    avg_roa = mean(avg_roa_pct, na.rm = TRUE),
    avg_export = mean(export_intensity_pct, na.rm = TRUE),
    avg_capacity = mean(capacity_utilization_pct, na.rm = TRUE),
    avg_rd = mean(rd_spending_pct, na.rm = TRUE),
    avg_knowledge_absorption = mean(knowledge_absorption, na.rm = TRUE),
    avg_task_performance = mean(task_performance, na.rm = TRUE),
    .groups = 'drop'
  )

cat("FDI vs Non-FDI Comparison:\n")
print(fdi_comparison)

# Statistical tests for group differences
# T-test for ROI difference
roi_ttest <- t.test(avg_roi_pct ~ has_fdi, data = data)
cat("\nT-test for ROI difference (FDI vs Non-FDI):\n")
cat("t =", round(roi_ttest$statistic, 3), ", p =", round(roi_ttest$p.value, 4), "\n")

# Compare SME vs Large firms
size_comparison <- data %>%
  group_by(firm_size) %>%
  summarise(
    n = n(),
    fdi_rate = mean(has_fdi, na.rm = TRUE),
    avg_roi = mean(avg_roi_pct, na.rm = TRUE),
    avg_roa = mean(avg_roa_pct, na.rm = TRUE),
    avg_export = mean(export_intensity_pct, na.rm = TRUE),
    avg_rd = mean(rd_spending_pct, na.rm = TRUE),
    .groups = 'drop'
  )

cat("\nSME vs Large Firm Comparison:\n")
print(size_comparison)

# =============================================================================
# DATA QUALITY ASSESSMENT
# =============================================================================

cat("\n=== DATA QUALITY ASSESSMENT ===\n")

# Check for logical inconsistencies
quality_issues <- list()

# Check 1: FDI partnership but no FDI type
fdi_no_type <- sum(data$has_fdi == 1 & is.na(data$fdi_type), na.rm = TRUE)
if (fdi_no_type > 0) {
  quality_issues$fdi_no_type <- paste(fdi_no_type, "firms have FDI but no FDI type")
}

# Check 2: Years with FDI > Years of operation
if ("years_fdi" %in% names(data) && "years_operation" %in% names(data)) {
  fdi_years_issue <- sum(data$years_fdi > data$years_operation, na.rm = TRUE)
  if (fdi_years_issue > 0) {
    quality_issues$fdi_years_issue <- paste(fdi_years_issue, "firms have FDI years > operation years")
  }
}

# Check 3: Unrealistic percentages
unrealistic_pct <- sum(data$skilled_workforce_pct > 100 | data$skilled_workforce_pct < 0, na.rm = TRUE) +
                  sum(data$export_intensity_pct > 100 | data$export_intensity_pct < 0, na.rm = TRUE) +
                  sum(data$capacity_utilization_pct > 100 | data$capacity_utilization_pct < 0, na.rm = TRUE)

if (unrealistic_pct > 0) {
  quality_issues$unrealistic_pct <- paste(unrealistic_pct, "instances of unrealistic percentages")
}

# Print quality issues
if (length(quality_issues) > 0) {
  cat("Data Quality Issues Found:\n")
  for (i in seq_along(quality_issues)) {
    cat("-", quality_issues[[i]], "\n")
  }
} else {
  cat("No major data quality issues detected!\n")
}

# =============================================================================
# SAMPLE REPRESENTATIVENESS
# =============================================================================

cat("\n=== SAMPLE REPRESENTATIVENESS ===\n")

# Check sample distribution vs target population
sample_dist <- table(data$firm_size)
target_sme <- 200
target_large <- 100

cat("Sample Distribution:\n")
print(sample_dist)
cat("\nTarget Distribution:\n")
cat("SME:", target_sme, "Large:", target_large, "\n")

# Chi-square goodness of fit test
expected_counts <- c(target_sme, target_large)
chisq_test <- chisq.test(sample_dist, p = expected_counts/sum(expected_counts))
cat("\nChi-square test for sample representativeness:\n")
cat("χ² =", round(chisq_test$statistic, 3), ", p =", round(chisq_test$p.value, 4), "\n")

# =============================================================================
# EXPORT VALIDATION RESULTS
# =============================================================================

# Create comprehensive validation report
validation_report <- list(
  sample_size = nrow(data),
  missing_data = missing_summary,
  descriptive_stats = desc_stats,
  normality_tests = normality_tests,
  outlier_summary = outlier_summary,
  fdi_comparison = fdi_comparison,
  size_comparison = size_comparison,
  quality_issues = quality_issues
)

# Save validation results
saveRDS(validation_report, "data_validation_report.rds")

# Export key summaries as CSV
write_csv(outlier_summary, "outlier_analysis.csv")
write_csv(fdi_comparison, "fdi_group_comparison.csv")
write_csv(size_comparison, "size_group_comparison.csv")

cat("\n=== VALIDATION COMPLETED ===\n")
cat("Files generated:\n")
cat("- descriptive_statistics.csv\n")
cat("- outlier_analysis.csv\n")
cat("- fdi_group_comparison.csv\n")
cat("- size_group_comparison.csv\n")
cat("- data_validation_report.rds\n")
cat("- distribution_plots.png\n")
cat("- correlation_heatmap.png\n")
cat("- outlier_boxplots.png\n")
if (nrow(missing_summary) > 0) cat("- missing_data_pattern.png\n")

cat("\n=== RECOMMENDATIONS ===\n")
cat("1. Review any identified data quality issues\n")
cat("2. Consider transformation for non-normal variables if needed for analysis\n")
cat("3. Investigate outliers - determine if they should be retained or treated\n")
cat("4. Check multicollinearity before running regression models\n")
cat("5. Proceed with SEM analysis using validated dataset\n")