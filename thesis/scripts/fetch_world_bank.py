#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
TAB_DIR = BASE_DIR / "tables"
for p in (DATA_DIR, FIG_DIR, TAB_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Indicators for Tanzania (TZA)
# - GDP growth (annual %): NY.GDP.MKTP.KD.ZG
# - CPI (2010=100): FP.CPI.TOTL
# - Fixed broadband subscriptions (per 100 people): IT.NET.BBND.P2
# - Individuals using the Internet (% of population): IT.NET.USER.ZS
# - Mobile cellular subscriptions (per 100 people): IT.CEL.SETS.P2

COUNTRY = "TZA"
INDICATORS = {
    "GDP_growth": "NY.GDP.MKTP.KD.ZG",
    "CPI": "FP.CPI.TOTL",
    "Broadband_per100": "IT.NET.BBND.P2",
    "Internet_users_pct": "IT.NET.USER.ZS",
    "Mobile_subs_per100": "IT.CEL.SETS.P2",
}

session = requests.Session()

frames = []
for name, code in INDICATORS.items():
    url = f"https://api.worldbank.org/v2/country/{COUNTRY}/indicator/{code}?format=json&per_page=20000"
    r = session.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = data[1]
    df = pd.DataFrame(rows)[["date", "value"]].dropna()
    df["indicator"] = name
    df["date"] = pd.to_numeric(df["date"])  # year
    frames.append(df)

wb = pd.concat(frames, ignore_index=True)
wb = wb.sort_values(["indicator", "date"]).reset_index(drop=True)
wb.to_csv(DATA_DIR / "world_bank_tza.csv", index=False)

# Pivot for plotting
tbl = wb.pivot_table(index="date", columns="indicator", values="value")

# Save LaTeX table with latest available values
latest = tbl.dropna().iloc[-1]
latex = ["% Auto-generated World Bank indicators (latest available)\n",
         "\\begin{table}[!ht]\n  \\centering\n  \\caption{Tanzania macro indicators (latest available)}\n  \\label{tab:wb_latest}\n  \\begin{tabular}{lr}\n    \\toprule\n    Indicator & Value \\\\ \n    \\midrule\n"]
for k, v in latest.items():
    latex.append(f"    {k.replace('_',' ')} & {v:.2f} \\\\ \n")
latex.append("    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
(Path(TAB_DIR) / "wb_latest.tex").write_text("".join(latex))

# Trend figures
sns.set_theme(style="whitegrid")
fig_map = {
    "GDP_growth": "wb_gdp_growth.png",
    "CPI": "wb_cpi.png",
    "Internet_users_pct": "wb_internet_users.png",
    "Mobile_subs_per100": "wb_mobile_subs.png",
}
for ind, fname in fig_map.items():
    if ind in tbl.columns:
        plt.figure(figsize=(6,3.5))
        series = tbl[ind].dropna()
        series.plot()
        plt.title(ind.replace('_',' '))
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.tight_layout()
        plt.savefig(FIG_DIR / fname, dpi=200)
        plt.close()

print("Saved World Bank indicators to data/world_bank_tza.csv and tables/wb_latest.tex")
