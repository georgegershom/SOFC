Nexus-Specific & Alternative Data (Fabricated with partial live signals)

Files:
- nexus_monthly.csv: per-country monthly metrics
- nexus_yearly.csv: per-country yearly market structure and licensing

Variables:
- google_trend_mobile_money_fraud: 0-100 monthly interest index (live or simulated)
- social_sentiment_avg: [-1,1] average sentiment across brands
- social_mention_volume: count of social mentions sampled
- cyber_incidents_finsec: count of reported incidents in financial sector (synthetic)
- hhi: yearly Herfindahl-Hirschman Index (market concentration)
- licenses_new: yearly count of new FinTech licenses

Note: Trends fetched via pytrends when available; otherwise fabricated deterministically by seed.