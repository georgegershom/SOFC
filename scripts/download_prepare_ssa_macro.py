#!/usr/bin/env python3
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from rich import print as rprint

WB_API = "https://api.worldbank.org/v2"
HEADERS = {"User-Agent": "SSA-FinTech-Risk-Research/1.0 (contact: research@example.com)"}
DATA_DIR = os.path.join("/workspace", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw", "wb")
PROC_DIR = os.path.join(DATA_DIR, "processed")
LOG_DIR = os.path.join("/workspace", "logs")

SSA_REGION_CODE = "SSF"  # Sub-Saharan Africa (developing only); alt: SSF+SSA combos

# World Bank indicator codes
INDICATORS = {
    # Macroeconomic
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",  # GDP growth (annual %)
    "inflation_cpi": "FP.CPI.TOTL.ZG",  # Inflation, consumer prices (annual %)
    "unemployment": "SL.UEM.TOTL.ZS",  # Unemployment, total (% of total labor force) (modeled ILO)
    "policy_rate": "FR.INR.LEND",  # Lending interest rate (%) as proxy if policy rate unavailable
    "broad_money_m2_growth": "FM.LBL.BMNY.ZG",  # Broad money growth (annual %)
    "public_debt_gdp": "GC.DOD.TOTL.GD.ZS",  # Central government debt total (% of GDP)
    # Digital infrastructure
    "mobile_subs_100": "IT.CEL.SETS.P2",  # Mobile cellular subscriptions (per 100 people)
    "internet_users_pct": "IT.NET.USER.ZS",  # Individuals using the Internet (% of population)
    "secure_servers": "IT.NET.SECR.P6",  # Secure Internet servers (per 1 million people)
    # FX levels for volatility proxy
    "fx_official": "PA.NUS.FCRF",  # Official exchange rate (LCU per USD, period avg)
}

# Helper: ensure dirs
for d in [RAW_DIR, PROC_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(5))
def wb_get(url: str, params: Optional[dict] = None) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, params=params, timeout=60)
    resp.raise_for_status()
    return resp


def fetch_ssa_country_list() -> pd.DataFrame:
    # Get all countries in SSA region
    url = f"{WB_API}/region/{SSA_REGION_CODE}/country"
    resp = wb_get(url, params={"format": "json", "per_page": 10000})
    data = resp.json()
    if not isinstance(data, list) or len(data) < 2:
        raise RuntimeError("Unexpected WB region-country response")
    records = data[1]
    rows = []
    for item in records:
        rows.append(
            {
                "id": item.get("id"),
                "iso2Code": item.get("iso2Code"),
                "name": item.get("name"),
                "incomeLevel": item.get("incomeLevel", {}).get("id"),
                "region": item.get("region", {}).get("id"),
            }
        )
    df = pd.DataFrame(rows)
    # Filter out aggregates or invalid IDs if present
    df = df[df["id"].str.len() == 3].reset_index(drop=True)
    df.to_csv(os.path.join(RAW_DIR, "ssa_countries.csv"), index=False)
    return df


def fetch_indicator_for_countries(indicator: str, country_ids: List[str], date: str = "2000:2024") -> pd.DataFrame:
    # World Bank API supports multi-country comma-separated list
    countries_param = ";".join(country_ids)
    url = f"{WB_API}/country/{countries_param}/indicator/{indicator}"
    all_rows: List[dict] = []
    page = 1
    while True:
        resp = wb_get(url, params={"format": "json", "per_page": 20000, "page": page, "date": date})
        data = resp.json()
        if not isinstance(data, list) or len(data) < 2:
            break
        meta, values = data
        for rec in values:
            if rec is None:
                continue
            country = rec.get("country", {}).get("id")
            year = rec.get("date")
            val = rec.get("value")
            all_rows.append({"country": country, "year": int(year), "value": val})
        if meta and isinstance(meta, dict):
            page_total = meta.get("pages", 1)
            if page >= page_total:
                break
        page += 1
    df = pd.DataFrame(all_rows)
    return df


def compute_volatility(series: pd.Series, window_years: int = 3) -> pd.Series:
    # Rolling standard deviation for volatility
    return (
        series.astype(float)
        .rolling(window=window_years, min_periods=max(2, window_years))
        .std()
    )


def fabricate_missing(group: pd.DataFrame, method: str = "ffill_bfill_mean") -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = group.sort_values("year").copy()
    status = pd.Series("observed", index=df.index, dtype="string")
    if df["value"].isna().all():
        # fabricate flat mean zero if rate-like else NaN
        df["value"] = 0.0
        status[:] = "fabricated_all_zero"
        status_df = status.to_frame("status")
        status_df["country"] = df["country"].values
        status_df["year"] = df["year"].values
        return df, status_df
    # Try forward/backward fill
    v = df["value"].astype(float)
    v_filled = v.copy()
    v_filled = v_filled.ffill()
    v_filled = v_filled.bfill()
    # If still NaNs (edges), fill with group mean
    if v_filled.isna().any():
        mean_val = v.mean(skipna=True)
        v_filled = v_filled.fillna(mean_val)
    # Mark fabricated where original was NaN
    status.loc[v.isna()] = "fabricated_fill"
    df["value"] = v_filled
    status_df = status.to_frame("status")
    status_df["country"] = df["country"].values
    status_df["year"] = df["year"].values
    return df, status_df


def assemble_dataset(ssa: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    country_ids = ssa["id"].tolist()

    frames_raw: Dict[str, pd.DataFrame] = {}
    for key, code in INDICATORS.items():
        rprint(f"[bold cyan]Downloading indicator[/] {key} -> {code}")
        df_ind = fetch_indicator_for_countries(code, country_ids)
        df_ind["indicator_key"] = key
        frames_raw[key] = df_ind
        out_path = os.path.join(RAW_DIR, f"{key}.csv")
        df_ind.to_csv(out_path, index=False)

    # Concatenate into long format
    df_long = pd.concat(frames_raw.values(), ignore_index=True)

    # Compute volatility metrics for GDP growth and FX official
    vol_pieces = []
    for key in ["gdp_growth", "fx_official"]:
        part = frames_raw[key].copy()
        part = part.sort_values(["country", "year"])  # ensure order
        part["volatility_3y"] = part.groupby("country")["value"].transform(lambda s: compute_volatility(s, 3))
        # FX volatility on log returns if FX has levels
        if key == "fx_official":
            # compute annual log returns of FX (depreciation rate)
            def _safe_logret(s: pd.Series) -> pd.Series:
                s = s.astype(float)
                s = s.where(s > 0.0, np.nan)
                return np.log(s).diff()

            part["logret"] = part.groupby("country")["value"].transform(_safe_logret)
            part["fx_volatility_3y"] = part.groupby("country")["logret"].transform(lambda s: compute_volatility(s, 3))
        vol_pieces.append(part[["country", "year", "volatility_3y"] + (["fx_volatility_3y"] if key == "fx_official" else [])].assign(indicator_key=key))
    df_vol = pd.concat(vol_pieces, ignore_index=True)

    # Pivot to wide: one column per indicator value
    df_pivot_vals = df_long.pivot_table(index=["country", "year"], columns="indicator_key", values="value")
    df_pivot_vals = df_pivot_vals.reset_index()

    # Ensure all expected indicator columns exist even if empty
    expected_cols = [
        "gdp_growth",
        "inflation_cpi",
        "unemployment",
        "policy_rate",
        "broad_money_m2_growth",
        "public_debt_gdp",
        "mobile_subs_100",
        "internet_users_pct",
        "secure_servers",
    ]
    for col in expected_cols:
        if col not in df_pivot_vals.columns:
            df_pivot_vals[col] = np.nan

    # Pivot volatility: create columns for gdp_growth_vol_3y, fx_vol_3y
    df_vol_gdp = df_vol[df_vol["indicator_key"] == "gdp_growth"][[
        "country", "year", "volatility_3y"
    ]].rename(columns={"volatility_3y": "gdp_growth_vol_3y"})
    df_vol_fx = df_vol[df_vol["indicator_key"] == "fx_official"][[
        "country", "year", "fx_volatility_3y"
    ]]

    df_merged = df_pivot_vals.merge(df_vol_gdp, on=["country", "year"], how="left")
    df_merged = df_merged.merge(df_vol_fx, on=["country", "year"], how="left")

    # Impute/fabricate with flags per indicator columns
    status_frames = []
    for col in [
        "gdp_growth",
        "inflation_cpi",
        "unemployment",
        "policy_rate",
        "broad_money_m2_growth",
        "public_debt_gdp",
        "mobile_subs_100",
        "internet_users_pct",
        "secure_servers",
    ]:
        temp = df_merged[["country", "year", col]].rename(columns={col: "value"})
        fixed_list = []
        status_list = []
        for country, g in temp.groupby("country"):
            fixed, status = fabricate_missing(g)
            fixed_list.append(fixed)
            status_list.append(status)
        fixed_df = pd.concat(fixed_list).sort_values(["country", "year"]).reset_index(drop=True)
        status_df = pd.concat(status_list).sort_values(["country", "year"]).reset_index(drop=True)
        df_merged[col] = fixed_df["value"].values
        df_merged[f"{col}_status"] = status_df["status"].values

    # Save processed
    proc_out = os.path.join(PROC_DIR, "ssa_macro_category2.csv")
    df_merged.sort_values(["country", "year"]).to_csv(proc_out, index=False)

    # Also save a data dictionary
    dict_records = []
    for k, v in INDICATORS.items():
        dict_records.append({"column": k, "wb_indicator": v, "description": k})
    dict_records += [
        {"column": "gdp_growth_vol_3y", "wb_indicator": "derived", "description": "3-year rolling std of GDP growth"},
        {"column": "fx_volatility_3y", "wb_indicator": "derived", "description": "3-year rolling std of annual log returns of official FX"},
    ]
    pd.DataFrame(dict_records).to_csv(os.path.join(PROC_DIR, "ssa_macro_category2_dictionary.csv"), index=False)

    return df_long, df_merged


def main():
    rprint("[bold]Fetching SSA country list...[/]")
    ssa = fetch_ssa_country_list()
    rprint(f"SSA countries: {len(ssa)}")

    rprint("[bold]Downloading indicators and assembling dataset...[/]")
    df_long, df_merged = assemble_dataset(ssa)

    rprint("[green]Done. Outputs saved to[/] /workspace/data/processed:")
    for f in [
        "ssa_macro_category2.csv",
        "ssa_macro_category2_dictionary.csv",
    ]:
        rprint(f" - {os.path.join(PROC_DIR, f)}")


if __name__ == "__main__":
    main()
