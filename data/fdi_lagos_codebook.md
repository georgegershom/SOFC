# FDI–Lagos Food Processing Firm Survey: Codebook

Topic: The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria: The Moderating Role of the Nigerian Government Policy.

Geography: Lagos, Nigeria
Unit of analysis: Firm (food processing)
File(s): `fdi_lagos_firm_survey.csv` (records), `fdi_lagos_codebook.md` (this file)

## Sampling notes
- Synthetic dataset generated to mimic plausible Lagos food processing firm distributions.
- Typical subsectors: beverages, bakery, dairy, meat, fruits_veg, grains_cereals, oils_fats, confectionery.
- Ownership types: domestic, foreign, joint_venture.

## Variable definitions

### Identifiers and descriptors
- **firm_id**: String. Unique ID (e.g., `LAG-FP-0001`).
- **subsector**: Categorical. One of: beverages, bakery, dairy, meat, fruits_veg, grains_cereals, oils_fats, confectionery.
- **ownership_type**: Categorical. One of: domestic, foreign, joint_venture.
- **foreign_ownership_percent**: Continuous (%). 0–100.
- **years_since_fdi**: Integer. Years since first FDI inflow (0 if none).
- **age_years**: Integer. Firm age in years.
- **employees**: Integer. Number of employees.
- **assets_million_ngn**: Continuous. Book assets in million NGN.

### FDI-related constructs (Likert 1–5 unless noted)
- Knowledge absorption items: **ka_i1**, **ka_i2**, **ka_i3**, **ka_i4** (1–5). Adapted from Zahra & George (2002) dimensions of acquisition, assimilation, transformation, exploitation.
- **knowledge_absorption_index**: Mean of ka_i1–ka_i4 (1–5).

- Task performance items (Koopmans et al., 2013): **tp_i1**, **tp_i2**, **tp_i3** (1–5).
- **task_performance_index**: Mean of tp_i1–tp_i3 (1–5).

- Innovation items (Oslo Manual): **inn_i1**, **inn_i2**, **inn_i3** (1–5) capturing product, process, org/marketing orientation.
- **innovation_index**: Mean of inn_i1–inn_i3 (1–5).

- Firm resources items: **res_hr**, **res_tech**, **res_fin** (1–5) rating HR, technology, financial strength.
- **firm_resources_index**: Mean of res_hr, res_tech, res_fin (1–5).

### Firm resources (objective/operational)
- **skilled_labor_ratio**: Continuous. 0.05–0.90 share of employees who are skilled.
- **iot_use**: Binary (0/1). Uses IoT/IIoT in production/monitoring.
- **rd_spend_pct_revenue**: Continuous (%). 0.0–8.0.
- **liquidity_ratio**: Continuous. Current ratio (0.5–3.0 typical).

### Government policy perception (Likert 1–7)
- **policy_tax_incentives**: 1–7. Perceived adequacy of tax incentives.
- **policy_regulatory_stability**: 1–7. Predictability/consistency of regulation.
- **policy_infrastructure_support**: 1–7. Public infrastructure sufficiency.
- **policy_corruption_experience**: 1–7. Frequency/severity of corruption; higher=worse (reverse-coded in index).
- **policy_effectiveness_index**: Composite 1–7 = mean( tax_incentives, regulatory_stability, infrastructure_support, 8 - corruption_experience ).
- **policy_incentive_received**: Binary (0/1). Received any government incentive/grant in last 3 years.

### Firm performance
- **roi_pct**: Continuous (%). Return on investment, 2–35 typical.
- **roa_pct**: Continuous (%). Return on assets, 1–20 typical.
- **export_intensity_pct**: Continuous (%). Share of sales exported, 0–60 typical.
- **market_share_pct**: Continuous (%). Estimated Lagos/Nigeria market share, 0–20 typical.
- **operational_efficiency_index**: 0–100 composite operational score.

### Controls and FDI flags
- **fdi_presence**: Binary (0/1). Any foreign equity or sustained foreign partnership.

## Measurement and scoring details
- Likert items are integers; indices are arithmetic means of their items, rounded to 2 decimals in the CSV for readability.
- Policy effectiveness reverses corruption (7=best on other items; 1=worst; corruption reversed as 8 - raw value).
- Objective financial/operational metrics are simulated with plausible noise and realistic bounds.

## Generation logic (high level)
- Ownership influences FDI presence and foreign_ownership_percent (domestic≈0, foreign≈≥50, JV≈10–90).
- Absorptive capacity increases with skilled labor, FDI presence, and R&D intensity.
- Innovation rises with R&D and IoT use; both are more likely with FDI and better policy scores.
- Performance (ROI, ROA, efficiency, exports, market share) improves with absorptive capacity, task performance, innovation, and firm resources, and is moderated upward by policy_effectiveness_index.
- Export intensity increases with FDI presence, absorptive capacity, and policy effectiveness (moderation effect supported in the data).

## Provenance
- Items draw conceptual inspiration from Zahra & George (2002), Koopmans et al. (2013), and OECD Oslo Manual but are not verbatim.
- This is a synthetic dataset for research prototyping and methods testing only; replace with real survey data for production analyses.
