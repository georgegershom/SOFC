#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import csv
from urllib.parse import urlencode
from urllib.request import urlopen, Request

WB_BASE = "https://api.worldbank.org/v2"

SSA_REGION = "SSF"  # Sub-Saharan Africa region code

WB_INDICATORS = {
    # Banking Sector Health
    "npl_ratio": "FB.AST.NPER.ZS",  # Bank nonperforming loans to total gross loans (%)
    "bank_zscore": "GFDD.SI.01",   # Bank Z-score
    # ROA options: after-tax (GFDD.EI.05) or before-tax (GFDD.EI.09)
    "roa_after_tax": "GFDD.EI.05",
    "roa_before_tax": "GFDD.EI.09",
    "credit_private_gdp": "FS.AST.PRVT.GD.ZS",  # Domestic credit to private sector (% of GDP)
    # Regulatory Quality (WGI)
    "wgi_rq": "RQ.EST",
}

COUNTRY_SOURCE = {
    "source": [
        {
            "name": "World Bank API",
            "url": "https://api.worldbank.org/",
            "notes": "Indicators: NPL ratio, Z-score, ROA, Domestic credit to private sector, WGI Regulatory Quality"
        },
        {
            "name": "IMF Financial Access Survey / BIS / Central Banks",
            "url": "https://data.imf.org/",  # placeholder citations
            "notes": "Additional context for regulatory landscape (for fabricated index and policy dummies)"
        }
    ]
}

@dataclass
class Config:
    start_year: int
    end_year: int
    countries: Optional[List[str]]  # ISO2 or ISO3; if None, fetch SSA region countries via World Bank
    use_roa: str  # 'after_tax' or 'before_tax'


def fetch_json(url: str, params: dict) -> dict:
    # Simple GET with retries using stdlib
    query = urlencode(params)
    full_url = f"{url}?{query}" if query else url
    last_err: Optional[Exception] = None
    for _ in range(3):
        try:
            with urlopen(Request(full_url, headers={"User-Agent": "dataset-builder/1.0"}), timeout=60) as resp:
                import json as _json
                return _json.load(resp)
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    raise RuntimeError("fetch_json failed without exception")


def wb_get_countries_in_region(region_code: str) -> List[str]:
    # Get ISO2 codes of active countries in region
    url = f"{WB_BASE}/region/{region_code}/country"
    out = []
    page = 1
    while True:
        data = fetch_json(url, {"format": "json", "per_page": 2000, "page": page})
        if isinstance(data, list) and len(data) == 2:
            records = data[1]
            if not records:
                break
            for c in records:
                if c.get("region", {}).get("id") == region_code and c.get("id") != "":
                    # World Bank uses ISO2 code in 'id'
                    if c.get("capitalCity") is not None:  # filter aggregates
                        out.append(c["id"])  # ISO2
            if page >= int(data[0].get("pages", 1)):
                break
            page += 1
        else:
            break
    return sorted(set(out))


def wb_fetch_indicator(indicator: str, countries: List[str], start: int, end: int) -> List[Dict]:
    # Fetch indicator for list of countries across years
    frames: List[Dict] = []
    # World Bank API allows multiple country codes separated by semicolons
    country_chunks = []
    chunk = []
    for c in countries:
        chunk.append(c)
        if len(chunk) == 50:
            country_chunks.append(";".join(chunk))
            chunk = []
    if chunk:
        country_chunks.append(";".join(chunk))

    for cc in country_chunks:
        url = f"{WB_BASE}/country/{cc}/indicator/{indicator}"
        page = 1
        while True:
            data = fetch_json(url, {
                "format": "json",
                "per_page": 20000,
                "date": f"{start}:{end}",
                "page": page
            })
            if not isinstance(data, list) or len(data) != 2:
                break
            meta, rows = data
            for r in rows or []:
                val = r.get("value")
                frames.append({
                    "country": r.get("country", {}).get("id"),  # ISO2
                    "country_name": r.get("country", {}).get("value"),
                    "year": int(r.get("date")),
                    indicator: float(val) if val is not None else None
                })
            if page >= int(meta.get("pages", 1)):
                break
            page += 1
    return frames


def fabricate_finreg_index(rows: List[Dict]) -> List[Optional[float]]:
    # Fabricate a composite Financial Regulation Index using available signals
    # Here we normalize selected variables and average them.
    needed_cols = [
        WB_INDICATORS["bank_zscore"],
        WB_INDICATORS["credit_private_gdp"],
        WB_INDICATORS["wgi_rq"],
    ]
    # Group by year, compute min-max per column
    years = sorted({r["year"] for r in rows})
    by_year: Dict[int, List[Dict]] = {y: [] for y in years}
    for r in rows:
        by_year[r["year"]].append(r)
    index_values: List[Optional[float]] = []
    for r in rows:
        y = r["year"]
        group = by_year[y]
        components: List[float] = []
        for col in needed_cols:
            if col not in r or r[col] is None:
                continue
            vals = [g[col] for g in group if g.get(col) is not None]
            if not vals:
                continue
            vmin, vmax = min(vals), max(vals)
            if vmax == vmin:
                components.append(0.5)
            else:
                components.append((r[col] - vmin) / (vmax - vmin))
        if components:
            index_values.append(sum(components) / len(components))
        else:
            index_values.append(None)
    return index_values


def fabricate_policy_dummies(rows: List[Dict]) -> List[int]:
    # Fabricate digital lending guideline dummy as example placeholder
    # Heuristic: trigger when WGI Regulatory Quality and credit to private sector rise above historical median per country
    # Build per-country medians and first year logic
    from collections import defaultdict
    by_country: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_country[r["country"]].append(r)
    policy_year_by_country: Dict[str, Optional[int]] = {}
    for c, lst in by_country.items():
        rq_vals = [x.get(WB_INDICATORS["wgi_rq"]) for x in lst if x.get(WB_INDICATORS["wgi_rq"]) is not None]
        cr_vals = [x.get(WB_INDICATORS["credit_private_gdp"]) for x in lst if x.get(WB_INDICATORS["credit_private_gdp"]) is not None]
        if not rq_vals or not cr_vals:
            policy_year_by_country[c] = None
            continue
        rq_vals_sorted = sorted(rq_vals)
        cr_vals_sorted = sorted(cr_vals)
        rq_med = rq_vals_sorted[len(rq_vals_sorted)//2]
        cr_med = cr_vals_sorted[len(cr_vals_sorted)//2]
        first_year = None
        for y in sorted({x["year"] for x in lst}):
            any_row = [x for x in lst if x["year"] == y]
            # use first available row in year
            row = any_row[0]
            rq = row.get(WB_INDICATORS["wgi_rq"]) 
            cr = row.get(WB_INDICATORS["credit_private_gdp"]) 
            if rq is not None and cr is not None and rq >= rq_med and cr >= cr_med:
                first_year = y
                break
        policy_year_by_country[c] = first_year
    dummies: List[int] = []
    for r in rows:
        y0 = policy_year_by_country.get(r["country"]) 
        dummies.append(1 if (y0 is not None and r["year"] >= y0) else 0)
    return dummies


def build_dataset(cfg: Config) -> Tuple[List[Dict], Dict]:
    if cfg.countries is None:
        countries = wb_get_countries_in_region(SSA_REGION)
    else:
        countries = cfg.countries

    indicator_map = {
        "npl_ratio": WB_INDICATORS["npl_ratio"],
        "bank_zscore": WB_INDICATORS["bank_zscore"],
        "roa": WB_INDICATORS["roa_after_tax" if cfg.use_roa == "after_tax" else "roa_before_tax"],
        "credit_private_gdp": WB_INDICATORS["credit_private_gdp"],
        "wgi_rq": WB_INDICATORS["wgi_rq"],
    }

    # Fetch each indicator
    # Collect rows keyed by (country, year)
    keyed: Dict[Tuple[str, int], Dict] = {}
    for friendly, code in indicator_map.items():
        rows = wb_fetch_indicator(code, countries, cfg.start_year, cfg.end_year)
        for r in rows:
            key = (r["country"], r["year"]) 
            entry = keyed.setdefault(key, {"country": r["country"], "country_name": r["country_name"], "year": r["year"]})
            entry[code] = r.get(code)
    combined_rows: List[Dict] = [v for _, v in sorted(keyed.items(), key=lambda kv: (kv[0][0], kv[0][1]))]
    if not combined_rows:
        return [], {}

    # Fabricated fields
    finreg_idx = fabricate_finreg_index(combined_rows)
    for r, v in zip(combined_rows, finreg_idx):
        r["financial_regulation_index"] = v
    dummies = fabricate_policy_dummies(combined_rows)
    for r, dv in zip(combined_rows, dummies):
        r["digital_lending_guideline_dummy"] = dv

    metadata = {
        "category": "Category 3: Financial System & Regulatory Data (Systemic Context)",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "countries": countries,
        "years": [cfg.start_year, cfg.end_year],
        "indicators": indicator_map,
        "notes": {
            "fabricated": {
                "financial_regulation_index": "Min-max normalized average of Z-score, credit to private sector, and WGI Regulatory Quality per year.",
                "digital_lending_guideline_dummy": "Set to 1 from first year when both WGI RQ and credit to private sector exceed their country medians; purely heuristic proxy for regulatory shift toward digital finance.",
            },
            "caveats": [
                "Missingness varies by country and year; no imputation performed.",
                "Use ROA after-tax by default; switchable via flag.",
            ],
        },
        **COUNTRY_SOURCE,
    }
    return combined_rows, metadata


def main():
    parser = argparse.ArgumentParser(description="Build Category 3 dataset for SSA FinTech systemic context")
    parser.add_argument("--start", type=int, default=2005, help="Start year")
    parser.add_argument("--end", type=int, default=datetime.utcnow().year, help="End year")
    parser.add_argument("--roa", choices=["after_tax", "before_tax"], default="after_tax", help="ROA definition")
    parser.add_argument("--countries", type=str, default=None, help="Comma-separated ISO2 country codes; default is all SSA")
    parser.add_argument("--out_csv", type=str, default="/workspace/datasets/category3/category3_financial_system_regulatory.csv")
    parser.add_argument("--out_meta", type=str, default="/workspace/datasets/category3/category3_financial_system_regulatory.meta.json")

    args = parser.parse_args()
    countries = args.countries.split(",") if args.countries else None
    cfg = Config(start_year=args.start, end_year=args.end, countries=countries, use_roa=args.roa)

    rows, meta = build_dataset(cfg)
    if not rows:
        raise SystemExit("No data fetched. Check connectivity or indicator availability.")
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    # Write CSV with stdlib
    fieldnames = [
        "country","country_name","year",
        WB_INDICATORS["npl_ratio"],
        WB_INDICATORS["bank_zscore"],
        WB_INDICATORS["roa_after_tax" if cfg.use_roa == "after_tax" else "roa_before_tax"],
        WB_INDICATORS["credit_private_gdp"],
        WB_INDICATORS["wgi_rq"],
        "financial_regulation_index",
        "digital_lending_guideline_dummy",
    ]
    # Ensure fields exist in rows, add missing as None
    for r in rows:
        for f in fieldnames:
            r.setdefault(f, None)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(args.out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {len(rows)} rows to {args.out_csv}")
    print(f"Metadata -> {args.out_meta}")

if __name__ == "__main__":
    main()
