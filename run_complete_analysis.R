# =============================================================================
# COMPLETE ANALYSIS PIPELINE
# The Influence of Foreign Direct Investment on Food Processing Firms in Lagos
# =============================================================================

# This script runs the complete analysis pipeline from sampling to SEM

cat("=============================================================================\n")
cat("FDI STUDY - COMPLETE ANALYSIS PIPELINE\n")
cat("=============================================================================\n\n")

# Check if required packages are installed
required_packages <- c("dplyr", "readr", "ggplot2", "psych", "corrplot", 
                      "lavaan", "semPlot", "semTools", "VIM", "car", 
                      "moments", "nortest", "MASS")

missing_packages <- required_packages[!required_packages %in% installed.packages()[,"Package"]]

if (length(missing_packages) > 0) {
  cat("Installing missing packages:", paste(missing_packages, collapse = ", "), "\n")
  install.packages(missing_packages, dependencies = TRUE)
}

# Load all required libraries
suppressMessages({
  lapply(required_packages, library, character.only = TRUE)
})

# =============================================================================
# STEP 1: SAMPLING FRAMEWORK
# =============================================================================

cat("STEP 1: Running Sampling Framework...\n")
tryCatch({
  source("sampling_framework.R")
  cat("✓ Sampling framework completed successfully\n\n")
}, error = function(e) {
  cat("✗ Error in sampling framework:", e$message, "\n\n")
})

# =============================================================================
# STEP 2: SYNTHETIC DATA GENERATION
# =============================================================================

cat("STEP 2: Generating Synthetic Data...\n")
tryCatch({
  source("generate_synthetic_data.R")
  cat("✓ Synthetic data generation completed successfully\n\n")
}, error = function(e) {
  cat("✗ Error in data generation:", e$message, "\n\n")
})

# =============================================================================
# STEP 3: DATA VALIDATION
# =============================================================================

cat("STEP 3: Running Data Validation...\n")
tryCatch({
  source("data_validation.R")
  cat("✓ Data validation completed successfully\n\n")
}, error = function(e) {
  cat("✗ Error in data validation:", e$message, "\n\n")
})

# =============================================================================
# STEP 4: SEM ANALYSIS
# =============================================================================

cat("STEP 4: Running SEM Analysis...\n")
tryCatch({
  source("sem_analysis.R")
  cat("✓ SEM analysis completed successfully\n\n")
}, error = function(e) {
  cat("✗ Error in SEM analysis:", e$message, "\n\n")
})

# =============================================================================
# STEP 5: GENERATE FINAL SUMMARY REPORT
# =============================================================================

cat("STEP 5: Generating Final Summary Report...\n")

# Load the final dataset
if (file.exists("fdi_synthetic_dataset.csv")) {
  data <- read_csv("fdi_synthetic_dataset.csv", show_col_types = FALSE)
  
  # Create executive summary
  executive_summary <- list(
    study_title = "The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria",
    sample_size = nrow(data),
    sme_count = sum(data$firm_size == "SME"),
    large_count = sum(data$firm_size == "Large"),
    fdi_firms = sum(data$has_fdi == 1),
    fdi_percentage = round(mean(data$has_fdi) * 100, 1),
    avg_years_operation = round(mean(data$years_operation, na.rm = TRUE), 1),
    avg_roi = round(mean(data$avg_roi_pct, na.rm = TRUE), 2),
    avg_roa = round(mean(data$avg_roa_pct, na.rm = TRUE), 2),
    avg_export_intensity = round(mean(data$export_intensity_pct, na.rm = TRUE), 2)
  )
  
  # Save executive summary
  saveRDS(executive_summary, "executive_summary.rds")
  
  # Create a summary table
  summary_table <- data %>%
    group_by(firm_size, has_fdi) %>%
    summarise(
      count = n(),
      avg_performance = round(mean(overall_performance, na.rm = TRUE), 3),
      avg_knowledge_absorption = round(mean(knowledge_absorption, na.rm = TRUE), 2),
      avg_innovation = round(mean(innovation_score, na.rm = TRUE), 3),
      avg_roi = round(mean(avg_roi_pct, na.rm = TRUE), 2),
      avg_roa = round(mean(avg_roa_pct, na.rm = TRUE), 2),
      .groups = 'drop'
    ) %>%
    mutate(
      firm_type = paste(firm_size, ifelse(has_fdi == 1, "with FDI", "without FDI"))
    )
  
  write_csv(summary_table, "final_summary_table.csv")
  
  cat("✓ Final summary report generated successfully\n\n")
} else {
  cat("✗ Dataset not found for summary generation\n\n")
}

# =============================================================================
# FINAL STATUS REPORT
# =============================================================================

cat("=============================================================================\n")
cat("ANALYSIS PIPELINE COMPLETION STATUS\n")
cat("=============================================================================\n")

# Check which files were created
expected_files <- c(
  "sampling_frame.csv",
  "selected_sample.csv",
  "fdi_synthetic_dataset.csv",
  "fdi_sem_dataset.csv",
  "descriptive_statistics.csv",
  "correlation_heatmap.png",
  "sem_parameter_estimates.csv",
  "model_fit_summary.csv",
  "final_summary_table.csv"
)

file_status <- data.frame(
  File = expected_files,
  Status = ifelse(file.exists(expected_files), "✓ Created", "✗ Missing"),
  stringsAsFactors = FALSE
)

print(file_status)

# Count successful outputs
successful_files <- sum(file.exists(expected_files))
total_files <- length(expected_files)

cat("\nSummary:", successful_files, "out of", total_files, "expected files created\n")
cat("Success rate:", round(successful_files/total_files * 100, 1), "%\n\n")

# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================

cat("=============================================================================\n")
cat("NEXT STEPS AND USAGE INSTRUCTIONS\n")
cat("=============================================================================\n")

cat("1. DATASET FILES:\n")
cat("   - fdi_synthetic_dataset.csv: Complete synthetic dataset (300 firms)\n")
cat("   - fdi_sem_dataset.csv: Dataset prepared for SEM analysis\n")
cat("   - selected_sample.csv: Sample selection details\n\n")

cat("2. ANALYSIS RESULTS:\n")
cat("   - descriptive_statistics.csv: Basic descriptive statistics\n")
cat("   - sem_parameter_estimates.csv: SEM model results\n")
cat("   - model_fit_summary.csv: Model fit indices\n")
cat("   - final_summary_table.csv: Executive summary by group\n\n")

cat("3. VISUALIZATIONS:\n")
cat("   - correlation_heatmap.png: Variable correlations\n")
cat("   - distribution_plots.png: Variable distributions\n")
cat("   - sem_path_diagram.png: SEM path diagram\n")
cat("   - model_fit_comparison.png: Model fit comparison\n\n")

cat("4. FOR THESIS WRITING:\n")
cat("   - Use fdi_synthetic_dataset.csv as your main dataset\n")
cat("   - Reference model_fit_summary.csv for SEM results\n")
cat("   - Include visualizations in your methodology and results sections\n")
cat("   - Use final_summary_table.csv for descriptive statistics tables\n\n")

cat("5. FOR FURTHER ANALYSIS:\n")
cat("   - Modify sem_analysis.R to test additional hypotheses\n")
cat("   - Use data_validation.R to check data quality\n")
cat("   - Run individual scripts for specific analyses\n\n")

cat("=============================================================================\n")
cat("ANALYSIS PIPELINE COMPLETED\n")
cat("=============================================================================\n")

# Print session info for reproducibility
cat("\nSession Information:\n")
print(sessionInfo())