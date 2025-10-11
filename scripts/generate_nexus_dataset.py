#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import random
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

# Optional imports guarded at runtime
try:
    from pytrends.request import TrendReq
except Exception:
    TrendReq = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None

# Countries list (Sub-Saharan Africa sample, editable)
SSA_COUNTRIES = [
    "Kenya", "Nigeria", "Ghana", "Tanzania", "Uganda",
    "South Africa", "Ethiopia", "Rwanda", "Cameroon", "Ivory Coast",
]

FINTECH_BRANDS = [
    "M-Pesa", "Flutterwave", "Chipper Cash", "Paga", "Opay",
    "MTN MoMo", "Airtel Money", "Ecobank", "FNB", "Absa"
]

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def month_range(start_year: int, end_year: int) -> List[str]:
    months = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            months.append(f"{y}-{m:02d}")
    return months


def fabricate_cyber_incidents(countries: List[str], years: List[int]) -> pd.DataFrame:
    rows = []
    for country in countries:
        base = random.randint(5, 40)
        for year in years:
            trend = (year - years[0]) * random.uniform(0.5, 2.0)
            noise = np.random.poisson(3)
            incidents = max(0, int(base + trend + noise))
            rows.append({
                "country": country,
                "year": year,
                "cyber_incidents": incidents,
            })
    return pd.DataFrame(rows)


def fetch_google_trends(countries: List[str], start_year: int, end_year: int) -> pd.DataFrame:
    if TrendReq is None:
        # Fallback: fabricate plausible trend levels
        months = month_range(start_year, end_year)
        rows = []
        for country in countries:
            level = random.uniform(20, 70)
            for month in months:
                seasonality = 10 * math.sin(int(month[5:7]))
                noise = np.random.normal(0, 5)
                val = min(100, max(0, level + seasonality + noise))
                rows.append({
                    "country": country,
                    "month": month,
                    "google_trend_mobile_money_fraud": round(val, 2),
                })
        return pd.DataFrame(rows)

    try:
        pytrends = TrendReq(hl='en-US', tz=0)
    except Exception:
        return fetch_google_trends([], start_year, end_year)

    kw_list = ["mobile money fraud"]
    timeframe = f"{start_year}-01-01 {end_year}-12-31"
    rows = []
    for country in countries:
        try:
            pytrends.build_payload(kw_list=kw_list, timeframe=timeframe, geo="")
            interest_over_time_df = pytrends.interest_over_time()
            if interest_over_time_df is None or interest_over_time_df.empty:
                raise RuntimeError("Empty trends data")
            # Aggregate to months
            interest_over_time_df = interest_over_time_df.reset_index()
            interest_over_time_df['month'] = interest_over_time_df['date'].dt.strftime('%Y-%m')
            monthly = interest_over_time_df.groupby('month')[kw_list[0]].mean().reset_index()
            monthly['country'] = country
            monthly.rename(columns={kw_list[0]: 'google_trend_mobile_money_fraud'}, inplace=True)
            rows.append(monthly[['country', 'month', 'google_trend_mobile_money_fraud']])
        except Exception:
            # Fallback fabricate per country with small random deviations
            months = month_range(start_year, end_year)
            level = random.uniform(20, 70)
            tmp_rows = []
            for month in months:
                seasonality = 10 * math.sin(int(month[5:7]))
                noise = np.random.normal(0, 5)
                val = min(100, max(0, level + seasonality + noise))
                tmp_rows.append({
                    "country": country,
                    "month": month,
                    "google_trend_mobile_money_fraud": round(val, 2),
                })
            rows.append(pd.DataFrame(tmp_rows))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["country", "month", "google_trend_mobile_money_fraud"]) 


def synthetic_twitter_sentiment(countries: List[str], months: List[str]) -> pd.DataFrame:
    rows = []
    for country in countries:
        base = random.uniform(-0.05, 0.2)
        for month in months:
            drift = (int(month[:4]) - int(months[0][:4])) * random.uniform(-0.02, 0.02)
            seasonality = 0.1 * math.sin(int(month[5:7]))
            noise = np.random.normal(0, 0.1)
            score = max(-1, min(1, base + drift + seasonality + noise))
            rows.append({
                "country": country,
                "month": month,
                "sentiment_score": round(float(score), 3)
            })
    return pd.DataFrame(rows)


def fabricate_market_shares(countries: List[str], years: List[int]) -> pd.DataFrame:
    rows = []
    for country in countries:
        num_players = random.randint(4, 10)
        for year in years:
            sticks = np.sort(np.random.rand(num_players - 1))
            parts = np.diff(np.concatenate(([0], sticks, [1])))
            shares = (parts / parts.sum()) * 100
            for i, share in enumerate(shares):
                rows.append({
                    "country": country,
                    "year": year,
                    "player": f"Player_{i+1}",
                    "market_share": round(float(share), 2)
                })
    return pd.DataFrame(rows)


def compute_hhi_from_shares(shares_df: pd.DataFrame) -> pd.DataFrame:
    def hhi(group: pd.DataFrame) -> float:
        # HHI computed with percentage shares -> sum of squared shares
        return float(np.sum((group['market_share']) ** 2))

    hhi_df = (
        shares_df.groupby(['country', 'year'])
        .apply(hhi)
        .reset_index(name='hhi')
    )
    return hhi_df


def fabricate_licenses(countries: List[str], years: List[int]) -> pd.DataFrame:
    rows = []
    for country in countries:
        base = random.randint(2, 20)
        for year in years:
            cyc = 3 if (year % 3 == 0) else 1
            noise = np.random.poisson(2)
            licenses = max(0, int(base + (year - years[0]) * cyc + noise - random.randint(0, 5)))
            rows.append({
                "country": country,
                "year": year,
                "new_fintech_licenses": licenses
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate Nexus-Specific & Alternative Data for FinTech Risk")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=datetime.now().year)
    parser.add_argument("--countries", type=str, nargs="*", default=SSA_COUNTRIES)
    parser.add_argument("--brands", type=str, nargs="*", default=FINTECH_BRANDS)
    parser.add_argument("--out-dir", type=str, default="/workspace/data/processed")
    parser.add_argument("--raw-dir", type=str, default="/workspace/data/raw")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.raw_dir, exist_ok=True)

    years = list(range(args.start_year, args.end_year + 1))
    months = month_range(args.start_year, args.end_year)

    # Cyber Risk Exposure
    cyber_df = fabricate_cyber_incidents(args.countries, years)
    cyber_path = os.path.join(args.raw_dir, "cyber_incidents.csv")
    cyber_df.to_csv(cyber_path, index=False)

    # Google Trends
    trends_df = fetch_google_trends(args.countries, args.start_year, args.end_year)
    trends_path = os.path.join(args.raw_dir, "google_trends_mobile_money_fraud.csv")
    trends_df.to_csv(trends_path, index=False)

    # Social Sentiment (synthetic fallback)
    sentiment_df = synthetic_twitter_sentiment(args.countries, months)
    sentiment_path = os.path.join(args.raw_dir, "social_sentiment.csv")
    sentiment_df.to_csv(sentiment_path, index=False)

    # Market shares and HHI
    shares_df = fabricate_market_shares(args.countries, years)
    shares_path = os.path.join(args.raw_dir, "synthetic_market_shares.csv")
    shares_df.to_csv(shares_path, index=False)

    hhi_df = compute_hhi_from_shares(shares_df)
    hhi_path = os.path.join(args.raw_dir, "hhi.csv")
    hhi_df.to_csv(hhi_path, index=False)

    # Licenses
    licenses_df = fabricate_licenses(args.countries, years)
    licenses_path = os.path.join(args.raw_dir, "fintech_licenses.csv")
    licenses_df.to_csv(licenses_path, index=False)

    # Merge monthly and yearly into a single panel (by converting yearly to monthly via forward-fill)
    yearly = cyber_df.merge(hhi_df, on=["country", "year"]).merge(licenses_df, on=["country", "year"]) 
    # expand yearly to months
    year_to_month = pd.DataFrame({"year": [int(m[:4]) for m in months], "month": months}).drop_duplicates()
    yearly_monthly = yearly.merge(year_to_month, on="year", how="right")
    # Assign countries appropriately
    cross = pd.MultiIndex.from_product([args.countries, months], names=["country", "month"]).to_frame(index=False)
    yearly_monthly = cross.merge(yearly_monthly, on=["country", "month"], how="left")
    # forward-fill within each country
    yearly_monthly = yearly_monthly.sort_values(["country", "month"])
    yearly_monthly[['year','cyber_incidents','hhi','new_fintech_licenses']] = (
        yearly_monthly.groupby('country')[['year','cyber_incidents','hhi','new_fintech_licenses']]
        .ffill()
    )

    # Combine with monthly tables
    panel = yearly_monthly.merge(trends_df, on=["country", "month"], how="left") \
                          .merge(sentiment_df, on=["country", "month"], how="left")

    out_path = os.path.join(args.out_dir, "nexus_alternative_dataset.csv")
    panel.to_csv(out_path, index=False)

    # Data dictionary
    dictionary = [
        {
            "variable": "country",
            "type": "string",
            "description": "Country in Sub-Saharan Africa",
            "source": "synthetic + mappings"
        },
        {
            "variable": "month",
            "type": "string YYYY-MM",
            "description": "Monthly period",
            "source": "constructed"
        },
        {
            "variable": "year",
            "type": "integer",
            "description": "Calendar year",
            "source": "derived from month/yearly tables"
        },
        {
            "variable": "cyber_incidents",
            "type": "integer",
            "description": "Number of cybersecurity incidents reported in financial sector",
            "source": "fabricated (use national CERTs if available)"
        },
        {
            "variable": "google_trend_mobile_money_fraud",
            "type": "0-100 index",
            "description": "Google search interest for 'mobile money fraud'",
            "source": "pytrends; fabricated fallback"
        },
        {
            "variable": "sentiment_score",
            "type": "-1 to 1",
            "description": "Social media sentiment about FinTech brands",
            "source": "synthetic sentiment (replace with scrapes if available)"
        },
        {
            "variable": "hhi",
            "type": "numeric",
            "description": "Herfindahl-Hirschman Index based on synthetic market shares (percentage-squared)",
            "source": "computed from synthetic shares"
        },
        {
            "variable": "new_fintech_licenses",
            "type": "integer",
            "description": "Number of new FinTech licenses issued per year",
            "source": "fabricated (replace with regulator data if available)"
        }
    ]

    with open(os.path.join(args.out_dir, "nexus_alternative_data_dictionary.json"), "w") as f:
        json.dump(dictionary, f, indent=2)

    print(f"Wrote panel to {out_path}")
    print(f"Raw components saved to {args.raw_dir}")


if __name__ == "__main__":
    main()
