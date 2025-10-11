from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import COUNTRY_BRANDS, DEFAULT_COUNTRIES
from .trends import TrendsFetcher, TrendsConfig
from .sentiment import fabricate_social_sentiment
from .cyber import fabricate_cyber_incidents
from .market import fabricate_market_structure, fabricate_license_counts
from .utils import TimeSpan, get_last_complete_month, set_global_seeds


@dataclass
class GenerationConfig:
    countries: List[str]
    start_year: int
    end_year: int
    seed: int = 42
    output_dir: Path = Path("data")


DEFAULT_CONFIG = GenerationConfig(
    countries=DEFAULT_COUNTRIES,
    start_year=2016,
    end_year=get_last_complete_month().year,
)


def generate_datasets(config: Optional[GenerationConfig] = None) -> None:
    cfg = config or DEFAULT_CONFIG
    set_global_seeds(cfg.seed)

    ts = TimeSpan.for_years(cfg.start_year, cfg.end_year)
    months = ts.months()
    years = ts.years()

    trends_fetcher = TrendsFetcher(TrendsConfig())
    trends_df = trends_fetcher.fetch_or_fabricate(cfg.countries, months[0].date(), months[-1].date(), seed=cfg.seed)

    sentiment_df = fabricate_social_sentiment(
        cfg.countries, months, COUNTRY_BRANDS, seed=cfg.seed, exogenous_trend=trends_df
    )

    cyber_df = fabricate_cyber_incidents(
        cfg.countries, months, seed=cfg.seed, exogenous_trend=trends_df, exogenous_sentiment=sentiment_df
    )

    market_df = fabricate_market_structure(cfg.countries, years, seed=cfg.seed)

    licenses_df = fabricate_license_counts(cfg.countries, years, seed=cfg.seed)

    monthly = (
        trends_df.merge(sentiment_df, on=["date", "country"], how="left")
        .merge(cyber_df, on=["date", "country"], how="left")
        .sort_values(["country", "date"])
    )

    yearly_hhi = (
        market_df.groupby(["country", "year"], as_index=False)
        .agg({"hhi": "mean"})
        .sort_values(["country", "year"])
    )

    yearly = (
        yearly_hhi.merge(licenses_df, on=["country", "year"], how="left")
        .sort_values(["country", "year"])
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    monthly_path = cfg.output_dir / "nexus_monthly.csv"
    yearly_path = cfg.output_dir / "nexus_yearly.csv"

    monthly.to_csv(monthly_path, index=False)
    yearly.to_csv(yearly_path, index=False)

    # Save metadata/readme style context
    readme_path = cfg.output_dir / "README_nexus_dataset.txt"
    readme_path.write_text(
        """
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
        """.strip()
    )


if __name__ == "__main__":
    generate_datasets()
