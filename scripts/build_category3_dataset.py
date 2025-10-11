#!/usr/bin/env python3
import os
import json
import time
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DATASET = os.path.join(DATA_DIR, "category3_financial_system_regulatory.csv")
OUTPUT_META = os.path.join(DATA_DIR, "category3_data_dictionary.json")

# World Bank API base
WB_BASE = "https://api.worldbank.org/v2"

# Focus: Sub-Saharan Africa region (SSA). We'll fetch the current country list dynamically.
SSA_REGION_CODE = "SSF"  # World Bank aggregate for Sub-Saharan Africa; used to filter countries

# Map indicators to World Bank codes
INDICATORS = {
    # Banking Sector Health
    "npl_to_total_loans_pct": "FB.AST.NPER.ZS",  # Bank nonperforming loans to total gross loans (%)
    "bank_z_score": "GFDD.SI.01",               # Bank Z-score
    "bank_roa_pct": "GFDD.EI.04",               # Bank ROA (%)
    "domestic_credit_private_gdp_pct": "FS.AST.PRVT.GD.ZS",  # Domestic credit to private sector (% of GDP)
    # Regulatory Quality (WGI)
    "regulatory_quality_wgi": "IQ.REG.XQ"       # Regulatory Quality: Estimate
}

# Years range (adjust if needed)
START_YEAR = 2000
END_YEAR = 2024

# Example regulatory events dictionary (fabricated/placeholder). Users should replace with actual dates per country.
# Format: { country_iso3: [ {"name": str, "year": int, "dummy_col": str}, ... ] }
FABRICATED_REG_EVENTS: Dict[str, List[Dict[str, Any]]] = {
    # Example: Kenya introduces digital lending guidelines in 2018
    "KEN": [
        {"name": "Digital Lending Guideline", "year": 2018, "dummy_col": "reg_dummy_digital_lending"}
    ],
    # Nigeria: sandbox in 2020 (illustrative)
    "NGA": [
        {"name": "Regulatory Sandbox", "year": 2020, "dummy_col": "reg_dummy_sandbox"}
    ],
    # South Africa: open banking discussion 2019 (illustrative)
    "ZAF": [
        {"name": "Open Banking Policy Note", "year": 2019, "dummy_col": "reg_dummy_open_banking"}
    ]
}

# Fabricated Financial Regulation Index proxy: simple scaled composite from available variables (if GPFI not programmatically available)
# This is clearly flagged as fabricated

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)


def wb_get(path: str, params: Dict[str, Any]) -> Any:
    """Generic World Bank API GET with retry and pagination support."""
    url = f"{WB_BASE}/{path}"
    backoff = 1.0
    for attempt in range(6):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            elif r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff)
                backoff *= 2
            else:
                r.raise_for_status()
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
    raise RuntimeError(f"Failed to fetch {url}")


def get_ssa_countries() -> pd.DataFrame:
    """Fetch active countries in Sub-Saharan Africa region (excludes aggregates)."""
    # Fetch all countries, then filter by region id 'SSF'
    data = wb_get("country", {"format": "json", "per_page": 1000})
    meta, rows = data
    records = []
    for c in rows:
        if not c.get("region"):
            continue
        if c["region"].get("id") != SSA_REGION_CODE:
            continue
        if c.get("capitalCity") == "" and c.get("incomeLevel", {}).get("id") == "NA":
            # skip aggregates
            continue
        if c.get("iso2Code") in ("ZG", "ZF", "XT"):
            # known aggregates
            continue
        records.append({
            "country_name": c.get("name"),
            "iso2": c.get("iso2Code"),
            "iso3": c.get("id"),
            "region": c.get("region", {}).get("value"),
            "income": c.get("incomeLevel", {}).get("value"),
        })
    return pd.DataFrame(records)


def fetch_indicator_for_countries(indicator: str, country_iso3_list: List[str]) -> pd.DataFrame:
    """Fetch indicator time-series for given countries from World Bank API.
    Returns empty DataFrame if API returns a message/no data.
    """
    frames = []
    # World Bank country param uses ISO2; however id field above is iso3. Need iso2 map
    # Fetch countries to map iso3->iso2
    all_countries = wb_get("country", {"format": "json", "per_page": 1000})[1]
    iso3_to_iso2 = {c["id"]: c["iso2Code"] for c in all_countries}
    iso2_codes = [iso3_to_iso2.get(i) for i in country_iso3_list if iso3_to_iso2.get(i)]

    # Paginate results: /country/iso2;iso2/indicator/IND?date=START:END
    countries_param = ";".join(iso2_codes)
    params = {"date": f"{START_YEAR}:{END_YEAR}", "format": "json", "per_page": 20000}
    data = wb_get(f"country/{countries_param}/indicator/{indicator}", params)

    # Handle possible API message structure or empty responses
    if not isinstance(data, list) or len(data) < 2:
        try:
            msg = data[0].get("message") if isinstance(data, list) and data else None
            if msg:
                print(f"World Bank API message for {indicator}: {msg}")
        except Exception:
            pass
        return pd.DataFrame(columns=["iso3", "country", "year", "value"])

    meta, rows = data
    rows = rows or []
    for r in rows:
        # countryiso3code is ISO3
        iso3 = r.get("countryiso3code")
        country_name = r.get("country", {}).get("value")
        date_val = r.get("date")
        try:
            year = int(date_val) if date_val is not None else None
        except Exception:
            year = None
        frames.append({
            "iso3": iso3,
            "country": country_name,
            "year": year,
            "value": r.get("value"),
        })
    df = pd.DataFrame(frames)
    if df.empty:
        return df
    # Normalize
    return df[["iso3", "country", "year", "value"]]


def build_panel(ssa_countries: pd.DataFrame) -> pd.DataFrame:
    years = list(range(START_YEAR, END_YEAR + 1))
    panel = (
        ssa_countries[["iso3", "country_name"]]
        .rename(columns={"country_name": "country"})
        .assign(key=1)
        .merge(pd.DataFrame({"year": years, "key": 1}), on="key", how="outer")
        .drop(columns=["key"]) 
        .sort_values(["iso3", "year"]) 
        .reset_index(drop=True)
    )
    return panel


def add_indicator(panel: pd.DataFrame, series_df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if series_df.empty:
        panel[col_name] = np.nan
        panel[f"{col_name}_source"] = "missing_all"
        return panel
    merged = panel.merge(series_df.rename(columns={"value": col_name}),
                         on=["iso3", "country", "year"], how="left")
    # Track source of values for provenance
    merged[f"{col_name}_source"] = np.where(merged[col_name].notna(), "world_bank", "missing")
    return merged


def impute_synthetic(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    """Impute missing values using simple within-country linear interpolation and fallback to region-year mean.
    Flag synthetic rows in a provenance column.
    """
    df = panel.copy()

    # Within-country interpolate
    def interpolate_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("year")
        values = g[col].astype(float)
        interp = values.interpolate(method="linear", limit_direction="both")
        source = g[f"{col}_source"].copy()
        source = np.where(values.isna() & interp.notna(), "synthetic_interp", source)
        g[col] = interp
        g[f"{col}_source"] = source
        return g

    df = df.groupby("iso3", as_index=False, group_keys=False).apply(interpolate_group)

    # Region-year mean fallback (SSA mean for that year)
    if df[col].isna().any():
        # Compute year-level means (NaNs ignored by default)
        yearly_means = df.groupby("year")[col].mean()
        mask = df[col].isna()
        df.loc[mask, col] = df.loc[mask, "year"].map(yearly_means)
        df.loc[mask, f"{col}_source"] = np.where(df.loc[mask, col].notna(), "synthetic_year_mean", df.loc[mask, f"{col}_source"]) 

    return df


def fabricate_regulation_dummies(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    # Initialize no regulation dummies to 0
    for iso3, events in FABRICATED_REG_EVENTS.items():
        for evt in events:
            col = evt["dummy_col"]
            if col not in df.columns:
                df[col] = 0
    # Apply per-country from their event year onward
    for iso3, events in FABRICATED_REG_EVENTS.items():
        for evt in events:
            col = evt["dummy_col"]
            year = evt["year"]
            df.loc[(df["iso3"] == iso3) & (df["year"] >= year), col] = 1
    return df


def fabricate_regulation_index(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    # Create a simple min-max scaled composite of available governance and credit variables
    components = [
        "regulatory_quality_wgi",
        "domestic_credit_private_gdp_pct",
    ]
    comp_data = df[components].copy()
    for c in components:
        # Min-max scale within the panel (ignoring NaNs)
        x = comp_data[c].astype(float)
        xmin, xmax = x.min(skipna=True), x.max(skipna=True)
        if pd.isna(xmin) or pd.isna(xmax) or xmin == xmax:
            comp_data[c] = np.nan
        else:
            comp_data[c] = (x - xmin) / (xmax - xmin)
    df["fabricated_finreg_index_0_1"] = comp_data.mean(axis=1, skipna=True)
    # Mark provenance: fabricated
    df["fabricated_finreg_index_source"] = np.where(df["fabricated_finreg_index_0_1"].notna(), "fabricated_composite", "missing")
    return df


def build_metadata(ssa_countries: pd.DataFrame) -> Dict[str, Any]:
    return {
        "category": "Category 3: Financial System & Regulatory Data (The Systemic Context)",
        "years": {"start": START_YEAR, "end": END_YEAR},
        "geographies": ssa_countries[["iso3", "country_name"]].rename(columns={"country_name": "country"}).to_dict(orient="records"),
        "variables": {
            "npl_to_total_loans_pct": {
                "name": "Bank Non-Performing Loans to Total Loans (%)",
                "source": "World Bank GFDD FB.AST.NPER.ZS",
                "wb_code": "FB.AST.NPER.ZS",
                "imputation": "linear interpolation within country; SSA year mean fallback",
            },
            "bank_z_score": {
                "name": "Bank Z-score (stability measure)",
                "source": "World Bank GFDD SI.01",
                "wb_code": "GFDD.SI.01",
                "imputation": "linear interpolation within country; SSA year mean fallback",
            },
            "bank_roa_pct": {
                "name": "Return on Assets (ROA) of the banking sector (%)",
                "source": "World Bank GFDD EI.04",
                "wb_code": "GFDD.EI.04",
                "imputation": "linear interpolation within country; SSA year mean fallback",
            },
            "domestic_credit_private_gdp_pct": {
                "name": "Domestic Credit to Private Sector (% of GDP)",
                "source": "World Bank FS.AST.PRVT.GD.ZS",
                "wb_code": "FS.AST.PRVT.GD.ZS",
                "imputation": "linear interpolation within country; SSA year mean fallback",
            },
            "regulatory_quality_wgi": {
                "name": "Regulatory Quality (WGI Estimate)",
                "source": "World Bank WGI IQ.REG.XQ",
                "wb_code": "IQ.REG.XQ",
                "imputation": "linear interpolation within country; SSA year mean fallback",
            },
            "fabricated_finreg_index_0_1": {
                "name": "Fabricated Financial Regulation Index (0-1)",
                "source": "Composite of available variables (fabricated)",
                "components": ["regulatory_quality_wgi", "domestic_credit_private_gdp_pct"],
                "method": "min-max scale components within panel; take mean",
            },
            "regulatory_dummies": {
                "note": "Fabricated dummy indicators for specific regulation introductions; replace with actual country events from central bank/official notices",
                "examples": FABRICATED_REG_EVENTS,
            }
        },
        "notes": [
            "This dataset programmatically downloads World Bank indicators and fabricates values only when missing, with provenance flags in *_source columns.",
            "Financial Regulation Index is a fabricated proxy; replace or augment with official indexes when available (e.g., GPFI).",
            "Regulation dummies are illustrative; please replace with validated dates from central bank websites.",
        ],
        "data_sources": [
            "World Bank Global Financial Development Database (GFDD)",
            "World Bank World Development Indicators (WDI)",
            "Worldwide Governance Indicators (WGI)",
            "IMF Financial Access Survey (optional future extension)",
            "BIS Statistics (optional future extension)",
            "Central Bank websites (for regulatory events)",
        ],
        "provenance_columns": "Each indicator has a *_source column: world_bank, synthetic_interp, synthetic_year_mean, missing_all, missing"
    }


def main():
    ensure_dirs()

    # Countries
    print("Fetching Sub-Saharan Africa countries...")
    ssa = get_ssa_countries()
    ssa = ssa.sort_values("iso3").reset_index(drop=True)

    # Panel
    panel = build_panel(ssa)

    # Fetch indicators and merge
    for col_name, wb_code in tqdm(INDICATORS.items(), desc="Downloading indicators"):
        series_df = fetch_indicator_for_countries(wb_code, ssa["iso3"].tolist())
        panel = add_indicator(panel, series_df, col_name)
        panel = impute_synthetic(panel, col_name)

    # Fabricate regulation dummies and index
    panel = fabricate_regulation_dummies(panel)
    panel = fabricate_regulation_index(panel)

    # Order columns
    meta_cols = ["iso3", "country", "year"]
    indicator_cols = list(INDICATORS.keys())
    source_cols = [f"{c}_source" for c in indicator_cols]
    reg_dummy_cols = sorted([c for c in panel.columns if c.startswith("reg_dummy_")])
    extra_cols = ["fabricated_finreg_index_0_1", "fabricated_finreg_index_source"]

    ordered_cols = meta_cols + indicator_cols + source_cols + reg_dummy_cols + extra_cols
    # Ensure all columns are present
    ordered_cols = [c for c in ordered_cols if c in panel.columns]

    panel = panel[ordered_cols].sort_values(["iso3", "year"]).reset_index(drop=True)

    # Save outputs
    panel.to_csv(OUTPUT_DATASET, index=False)

    meta = build_metadata(ssa)
    with open(OUTPUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    # Quick summary
    summary = panel[indicator_cols].isna().mean().sort_values()
    print("Missing share by indicator (after imputation):")
    print(summary)
    print(f"\nWrote dataset: {OUTPUT_DATASET}")
    print(f"Wrote data dictionary: {OUTPUT_META}")


if __name__ == "__main__":
    main()
