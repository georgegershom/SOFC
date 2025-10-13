# =============================================================================
# SAMPLING FRAMEWORK FOR FDI STUDY
# The Influence of Foreign Direct Investment on Food Processing Firms in Lagos
# =============================================================================

# Load required libraries
library(dplyr)
library(ggplot2)
library(readr)

# =============================================================================
# POPULATION DEFINITION
# =============================================================================

# Population parameters
N_sme <- 300      # SMEs (≤250 employees)
N_large <- 150    # Large firms (>250 employees)
N_total <- 450    # Total population

# Standard deviations for Neyman allocation
sd_sme <- 15.2    # Standard deviation of ROA in SMEs
sd_large <- 22.8  # Standard deviation of ROA in large firms

# =============================================================================
# SAMPLE SIZE CALCULATION USING NEYMAN ALLOCATION
# =============================================================================

# Target sample size
target_sample <- 300

# Neyman allocation formula
n_sme <- round((N_sme * sd_sme) / (N_sme * sd_sme + N_large * sd_large) * target_sample)
n_large <- target_sample - n_sme

cat("=== SAMPLE ALLOCATION ===\n")
cat("SMEs sample size:", n_sme, "\n")
cat("Large firms sample size:", n_large, "\n")
cat("Total sample size:", n_sme + n_large, "\n")
cat("SME proportion:", round(n_sme/target_sample * 100, 1), "%\n")
cat("Large firm proportion:", round(n_large/target_sample * 100, 1), "%\n")

# =============================================================================
# SAMPLING FRAME CREATION
# =============================================================================

# Create sampling frame with firm IDs
set.seed(12345)  # For reproducibility

# Generate SME firms
sme_firms <- data.frame(
  firm_id = paste0("SME", sprintf("%03d", 1:N_sme)),
  stratum = "SME",
  employees = sample(c("1-50", "51-250"), N_sme, replace = TRUE, prob = c(0.7, 0.3)),
  industrial_zone = sample(c("Ikeja", "Agbara", "Ilupeju", "Isolo", "Oshodi"), N_sme, replace = TRUE),
  selected = FALSE
)

# Generate Large firms
large_firms <- data.frame(
  firm_id = paste0("LRG", sprintf("%03d", 1:N_large)),
  stratum = "Large",
  employees = sample(c("251-500", "500+"), N_large, replace = TRUE, prob = c(0.6, 0.4)),
  industrial_zone = sample(c("Ikeja", "Agbara", "Ilupeju", "Isolo", "Oshodi"), N_large, replace = TRUE),
  selected = FALSE
)

# Combine sampling frame
sampling_frame <- rbind(sme_firms, large_firms)

# =============================================================================
# STRATIFIED RANDOM SAMPLING
# =============================================================================

# Select SME sample
selected_sme_indices <- sample(which(sampling_frame$stratum == "SME"), n_sme)
sampling_frame$selected[selected_sme_indices] <- TRUE

# Select Large firm sample
selected_large_indices <- sample(which(sampling_frame$stratum == "Large"), n_large)
sampling_frame$selected[selected_large_indices] <- TRUE

# Create final sample
final_sample <- sampling_frame[sampling_frame$selected, ]

# Add backup firms (20% of sample size)
backup_size_sme <- ceiling(n_sme * 0.2)
backup_size_large <- ceiling(n_large * 0.2)

# Select backup SMEs
available_sme <- sampling_frame[sampling_frame$stratum == "SME" & !sampling_frame$selected, ]
backup_sme_indices <- sample(nrow(available_sme), backup_size_sme)
backup_sme <- available_sme[backup_sme_indices, ]
backup_sme$backup <- TRUE

# Select backup Large firms
available_large <- sampling_frame[sampling_frame$stratum == "Large" & !sampling_frame$selected, ]
backup_large_indices <- sample(nrow(available_large), backup_size_large)
backup_large <- available_large[backup_large_indices, ]
backup_large$backup <- TRUE

# Add backup column to main sample
final_sample$backup <- FALSE

# Combine main and backup samples
complete_sample <- rbind(final_sample, backup_sme, backup_large)

# =============================================================================
# SAMPLE VALIDATION
# =============================================================================

cat("\n=== SAMPLE VALIDATION ===\n")
cat("Main sample size:", nrow(final_sample), "\n")
cat("Backup sample size:", nrow(backup_sme) + nrow(backup_large), "\n")
cat("Total sample with backups:", nrow(complete_sample), "\n")

# Geographic distribution
geo_dist <- table(final_sample$industrial_zone)
cat("\n=== GEOGRAPHIC DISTRIBUTION ===\n")
print(geo_dist)

# Stratum distribution
stratum_dist <- table(final_sample$stratum)
cat("\n=== STRATUM DISTRIBUTION ===\n")
print(stratum_dist)

# =============================================================================
# EXPORT SAMPLING RESULTS
# =============================================================================

# Save sampling frame
write_csv(sampling_frame, "sampling_frame.csv")

# Save final sample
write_csv(final_sample, "selected_sample.csv")

# Save complete sample with backups
write_csv(complete_sample, "complete_sample_with_backups.csv")

cat("\n=== FILES EXPORTED ===\n")
cat("- sampling_frame.csv\n")
cat("- selected_sample.csv\n")
cat("- complete_sample_with_backups.csv\n")

# =============================================================================
# RESPONSE RATE PLANNING
# =============================================================================

target_response_rate <- 0.70
expected_responses <- round(nrow(final_sample) * target_response_rate)

cat("\n=== RESPONSE RATE PLANNING ===\n")
cat("Target response rate:", target_response_rate * 100, "%\n")
cat("Expected responses:", expected_responses, "\n")
cat("Main sample size:", nrow(final_sample), "\n")
cat("Buffer with backups:", nrow(complete_sample), "\n")

# Create visualization
sample_viz <- ggplot(final_sample, aes(x = stratum, fill = industrial_zone)) +
  geom_bar(position = "dodge") +
  labs(title = "Sample Distribution by Stratum and Industrial Zone",
       x = "Firm Size Category",
       y = "Number of Firms",
       fill = "Industrial Zone") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("sample_distribution.png", sample_viz, width = 10, height = 6, dpi = 300)

cat("\n=== VISUALIZATION SAVED ===\n")
cat("- sample_distribution.png\n")