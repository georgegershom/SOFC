#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from textwrap import dedent

try:
    import pingouin as pg
except Exception:
    pg = None

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "sme_oi_tanzania.csv"
FIG_DIR = BASE_DIR / "figures"
TAB_DIR = BASE_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Compute scale scores
barrier_dims = [
    "cultural_resistance",
    "resource_inadequacy",
    "hierarchical_rigidity",
    "risk_aversion",
    "external_distrust",
]

diglit_dims = ["technical", "informational", "communicative", "strategic"]

for dim in barrier_dims + diglit_dims:
    items = [f"{dim}_item{i}" for i in range(1,4)]
    df[f"{dim}_score"] = df[items].mean(axis=1)

# Higher is more barriers; reverse if needed. We'll leave as is.

df["barriers_mean"] = df[[f"{d}_score" for d in barrier_dims]].mean(axis=1)
df["diglit_mean"] = df[[f"{d}_score" for d in diglit_dims]].mean(axis=1)

# Standardize for regression
for col in ["barriers_mean", "diglit_mean", "owner_age", "firm_age"]:
    df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std()

# Encode categories
df = pd.get_dummies(df, columns=["sector", "firm_size"], drop_first=True)

# Models: without and with interaction
controls = [c for c in df.columns if c.startswith("sector_") or c.startswith("firm_size_") or c in ["z_owner_age", "z_firm_age"]]

formula_base = "oi_index ~ z_barriers_mean + z_diglit_mean + " + " + ".join(controls)
formula_inter = formula_base + " + z_barriers_mean:z_diglit_mean"

model_base = smf.ols(formula=formula_base, data=df).fit()
model_inter = smf.ols(formula=formula_inter, data=df).fit()

r2_base = model_base.rsquared
r2_inter = model_inter.rsquared
dr2 = r2_inter - r2_base

# Export regression table (booktabs)
coef_table = model_inter.summary2().tables[1]
coef_table = coef_table.rename(columns={"Coef.": "Coefficient", "Std.Err.": "Std. Error", "P>|t|": "p"})
coef_table["Beta"] = coef_table["Coefficient"] / df["oi_index"].std()

latex_table = dedent(f"""
% Auto-generated regression results (interaction model)
% R2_base={r2_base:.3f}, R2_inter={r2_inter:.3f}, Delta_R2={dr2:.3f}
\\begin{{table}}[!ht]
  \\centering
  \\caption{{Hierarchical regression of OI index on barriers, digital literacy, and interaction (N={len(df)})}}
  \\label{{tab:regression_results}}
  \\begin{{tabular}}{{lrrrr}}
    \\toprule
    Variable & Coefficient & Std. Error & t & p \\\\ 
    \\midrule
""")

for idx, row in coef_table.iterrows():
    variable = str(idx).replace(":", " \\times ")
    latex_table += f"    {variable} & {row['Coefficient']:.3f} & {row['Std. Error']:.3f} & {row['t']:.2f} & {row['p']:.3f} \\\\ \n"

latex_table += dedent("""
    \\bottomrule
  \\end{tabular}
  \\vspace{2mm}
  \\begin{flushleft}
  {\\small Notes: Standard errors in parentheses. Controls include sector, firm size, owner and firm age. Interaction term is z_barriers_mean × z_diglit_mean.}
  \\end{flushleft}
\\end{table}
""")

(TAB_DIR / "regression_results.tex").write_text(latex_table)

# Sample characteristics table
sample_tab = df[[
    # demographics for summary
]].copy()

# Build summary
summary = []
summary.append(f"N = {len(df)}")
for col, name in [("owner_age", "Owner age"), ("firm_age", "Firm age"), ("oi_index", "OI Index")]:
    summary.append(f"{name}: mean={df[col].mean():.2f}, sd={df[col].std():.2f}")

# Recover original categorical columns count by reading from raw file
raw = pd.read_csv(DATA_PATH)
sector_counts = raw["sector"].value_counts()
size_counts = raw["firm_size"].value_counts()

latex_sample = dedent("""
% Auto-generated sample characteristics
\\begin{table}[!ht]
  \\centering
  \\caption{Sample characteristics}
  \\label{tab:sample}
  \\begin{tabular}{lr}
    \\toprule
    Metric & Value \\\\ 
    \\midrule
""")
for line in summary:
    parts = line.split(": ")
    if len(parts) == 2:
        metric, value = parts
    else:
        metric, value = line, ""
    latex_sample += f"    {metric} & {value} \\\\ \n"

latex_sample += "    \\midrule\n"
for k, v in sector_counts.items():
    latex_sample += f"    Sector: {k} & {v} \\\\ \n"
for k, v in size_counts.items():
    latex_sample += f"    Firm size: {k} & {v} \\\\ \n"

latex_sample += dedent("""
    \\bottomrule
  \\end{tabular}
\\end{table}
""")

(TAB_DIR / "sample_characteristics.tex").write_text(latex_sample)

# Reliability (Cronbach's alpha)
with open(TAB_DIR / "reliability.tex", "w") as f:
    f.write("% Auto-generated reliability results\n")
    f.write("\\begin{table}[!ht]\n  \\centering\n  \\caption{Scale reliability (Cronbach's alpha)}\n  \\label{tab:reliability}\n  \\begin{tabular}{lrr}\n    \\toprule\n    Scale & Items & Alpha \\\\ \n    \\midrule\n")
    for dim in barrier_dims + diglit_dims:
        items = [f"{dim}_item{i}" for i in range(1,4)]
        if pg is not None:
            alpha = pg.cronbach_alpha(df[items])[0]
        else:
            # Manual alpha
            X = df[items].to_numpy().astype(float)
            item_vars = X.var(axis=0, ddof=1)
            total_var = X.sum(axis=1).var(ddof=1)
            k = X.shape[1]
            alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
        f.write(f"    {dim.replace('_',' ').title()} & {len(items)} & {alpha:.3f} \\\\ \n")
    f.write("    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")

# Correlation heatmap
heat_cols = ["barriers_mean", "diglit_mean", "oi_index", "owner_age", "firm_age"]
plt.figure(figsize=(6,5))
sns.heatmap(df[heat_cols].corr(), annot=True, cmap="vlag", center=0)
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=200)
plt.close()

# Moderation plot
# Bin diglit into low/high
raw = pd.read_csv(DATA_PATH)
raw["barriers_mean"] = df["barriers_mean"]
raw["diglit_mean"] = df["diglit_mean"]
raw["oi_index"] = df["oi_index"]
raw["diglit_group"] = pd.qcut(raw["diglit_mean"], q=2, labels=["Low", "High"])

g = sns.lmplot(
    data=raw,
    x="barriers_mean",
    y="oi_index",
    hue="diglit_group",
    height=4,
    aspect=1.2,
    palette="Set1",
)
plt.title("Moderation of digital literacy on barriers → OI index")
plt.tight_layout()
plt.savefig(FIG_DIR / "moderation_plot.png", dpi=200)
plt.close()

# Conceptual model (simple annotated diagram)
plt.figure(figsize=(7,4))
plt.axis('off')
plt.text(0.1, 0.5, 'Organizational\nBarriers', bbox=dict(boxstyle='round', facecolor='#fee', edgecolor='#a66'))
plt.text(0.7, 0.5, 'Open Innovation\nEngagement', bbox=dict(boxstyle='round', facecolor='#eef', edgecolor='#66a'))
plt.arrow(0.25, 0.5, 0.38, 0.0, head_width=0.03, head_length=0.02, length_includes_head=True, color='black')
plt.text(0.44, 0.53, '−', fontsize=16, color='black')
plt.text(0.35, 0.75, 'Digital Literacy', bbox=dict(boxstyle='round', facecolor='#efe', edgecolor='#6a6'))
plt.arrow(0.39, 0.72, 0.29, -0.17, head_width=0.03, head_length=0.02, length_includes_head=True, color='black')
plt.text(0.53, 0.65, '+', fontsize=16, color='black')
plt.annotate('Moderates', xy=(0.35, 0.72), xytext=(0.22, 0.58), arrowprops=dict(arrowstyle='->'))
plt.title('Conceptual model with moderating effect')
plt.tight_layout()
plt.savefig(FIG_DIR / "conceptual_model.png", dpi=200)
plt.close()

# Write summary text for LaTeX inclusion
with open(TAB_DIR / "model_fit_summary.txt", "w") as f:
    f.write(dedent(f"""
    Base model R^2: {r2_base:.3f}
    Interaction model R^2: {r2_inter:.3f}
    Delta R^2: {dr2:.3f}
    Key coefficients (interaction model):
      z_barriers_mean: {model_inter.params.get('z_barriers_mean', np.nan):.3f} (p={model_inter.pvalues.get('z_barriers_mean', np.nan):.3f})
      z_diglit_mean: {model_inter.params.get('z_diglit_mean', np.nan):.3f} (p={model_inter.pvalues.get('z_diglit_mean', np.nan):.3f})
      Interaction: {model_inter.params.get('z_barriers_mean:z_diglit_mean', np.nan):.3f} (p={model_inter.pvalues.get('z_barriers_mean:z_diglit_mean', np.nan):.3f})
    """))

print("Saved figures and tables to:")
print(FIG_DIR)
print(TAB_DIR)
