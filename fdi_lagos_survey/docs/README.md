## FDI–Performance Survey (Synthetic Data Package)

This package contains a fabricated dataset aligned to the survey:
"The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria" with policy moderation.

### Contents
- `data/survey_data.csv`: 300 firms (200 SMEs, 100 large) with variables by survey sections A–G and composites (KA, TP, INN, FR, GP, PERFORM).
- `docs/codebook.csv`: Variable definitions, types, value ranges.
- `analysis/sem_analysis.R`: R script for reliability, CFA, and moderated SEM (composite-based path model) using lavaan.
- `scripts/generate_data.py`: Python generator used to fabricate the dataset.

### Quick Start
1. Open R and set working directory to `analysis`.
2. Run `source("sem_analysis.R")`. The script installs packages if missing, reads `../data/survey_data.csv`, and prints fit results.

### Notes
- The data are synthetic and mimic plausible relationships: FDI increases knowledge absorption and innovation; resources and governance support enhance performance; moderation by policy is modeled via interactions.
- Innovation indicators combine R&D %, new products, and process adoptions; alpha is reported for reference.
- Government policy composite uses gp1–gp3 (tax, regulatory stability, infrastructure). Permits is reported separately.
