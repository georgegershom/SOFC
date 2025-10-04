# PhD Thesis Project: Open Innovation in Tanzanian SMEs

This repository contains a complete LaTeX thesis scaffold with IEEE-style references, reproducible synthetic datasets, analysis scripts, and generated figures/tables.

## Structure
- `thesis.tex`: main LaTeX file
- `chapters/`: per-chapter `.tex` files
- `bib/references.bib`: bibliography (IEEE style via biblatex)
- `data/`: synthetic dataset(s)
- `scripts/`: Python scripts to generate data, analyses, figures and LaTeX tables
- `figures/`, `tables/`: auto-generated assets

## Quickstart
```bash
cd thesis
make setup
make all
# then build LaTeX with your editor or latexmk
```

## Notes
- Data are synthetic, calibrated to reflect described relationships (barriers → OI negative; digital literacy positive; interaction positive).
- Citations are provided in IEEE format via `biblatex[style=ieee]`. Verify any web/URL resources as needed.
