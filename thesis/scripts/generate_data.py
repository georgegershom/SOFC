#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N = 313

# Sectors
sectors = ["Manufacturing", "Retail", "ICT", "Agriculture"]
sector_probs = [0.28, 0.34, 0.18, 0.20]

# Demographics
firm_sizes = ["Micro (1-9)", "Small (10-49)", "Medium (50-249)"]
size_probs = [0.52, 0.36, 0.12]

owner_age_mu, owner_age_sigma = 38, 8
firm_age_mu, firm_age_sigma = 8, 5

# Latent constructs (standardized)
barriers = np.random.normal(loc=0.0, scale=1.0, size=N)
# Digital literacy, introduce slight negative correlation with barriers
rho = -0.25
z = np.random.normal(size=N)
digital_literacy = rho * (barriers - barriers.mean())/barriers.std() + np.sqrt(1 - rho**2) * (z - z.mean())/z.std()

# Interaction term
interaction = barriers * digital_literacy

# Controls
sector = np.random.choice(sectors, size=N, p=sector_probs)
firm_size = np.random.choice(firm_sizes, size=N, p=size_probs)
owner_age = np.clip(np.random.normal(owner_age_mu, owner_age_sigma, size=N).round(), 20, 70).astype(int)
firm_age = np.clip(np.random.normal(firm_age_mu, firm_age_sigma, size=N).round(), 0, 40).astype(int)

# Generate OI adoption propensity (standardized latent), then map to 0-1 score
beta_barriers = -0.42
beta_diglit = 0.48
beta_inter = 0.35

# Encode categorical controls with fixed effects magnitude
sector_effect = {"Manufacturing": 0.05, "Retail": -0.02, "ICT": 0.10, "Agriculture": -0.05}
size_effect = {"Micro (1-9)": -0.04, "Small (10-49)": 0.02, "Medium (50-249)": 0.05}

noise = np.random.normal(0, 0.55, size=N)
oi_latent = (
    beta_barriers * barriers + beta_diglit * digital_literacy + beta_inter * interaction
    + np.vectorize(sector_effect.get)(sector)
    + np.vectorize(size_effect.get)(firm_size)
    + 0.006 * (owner_age - owner_age_mu) + 0.01 * (firm_age - firm_age_mu)
    + noise
)

# Standardize
oi_latent = (oi_latent - oi_latent.mean()) / oi_latent.std()
# Map to 0-100 index
oi_index = (oi_latent - oi_latent.min()) / (oi_latent.max() - oi_latent.min()) * 100

# Create Likert-style items for measurement scales (5-point)
# Barriers: cultural_resistance, resource_inadequacy, hierarchical_rigidity, risk_aversion, external_distrust
# Digital literacy: technical, informational, communicative, strategic (3 items each)

def mk_items(latent, num_items=4, loading=0.8, unique_sd=0.6, likert=True):
    items = []
    for _ in range(num_items):
        signal = loading * latent
        unique = np.random.normal(0, unique_sd, size=latent.shape[0])
        x = signal + unique
        # standardize
        x = (x - x.mean()) / x.std()
        if likert:
            # map to 1..5
            # use quantiles to discretize to roughly uniform Likert
            q = pd.qcut(x, q=5, labels=[1,2,3,4,5])
            items.append(q.astype(int).to_numpy())
        else:
            items.append(x)
    return np.vstack(items).T

barriers_items = {
    "cultural_resistance": mk_items(barriers, num_items=3),
    "resource_inadequacy": mk_items(barriers, num_items=3),
    "hierarchical_rigidity": mk_items(barriers, num_items=3),
    "risk_aversion": mk_items(barriers, num_items=3),
    "external_distrust": mk_items(barriers, num_items=3),
}

diglit_items = {
    "technical": mk_items(digital_literacy, num_items=3),
    "informational": mk_items(digital_literacy, num_items=3),
    "communicative": mk_items(digital_literacy, num_items=3),
    "strategic": mk_items(digital_literacy, num_items=3),
}

# Assemble DataFrame
records = []
for i in range(N):
    rec = {
        "id": i+1,
        "sector": sector[i],
        "firm_size": firm_size[i],
        "owner_age": int(owner_age[i]),
        "firm_age": int(firm_age[i]),
        "barriers_z": float(barriers[i]),
        "digital_literacy_z": float(digital_literacy[i]),
        "oi_index": float(oi_index[i]),
    }
    # barrier items
    for dim, arr in barriers_items.items():
        for j in range(arr.shape[1]):
            rec[f"{dim}_item{j+1}"] = int(arr[i, j])
    # digi items
    for dim, arr in diglit_items.items():
        for j in range(arr.shape[1]):
            rec[f"{dim}_item{j+1}"] = int(arr[i, j])
    records.append(rec)

df = pd.DataFrame.from_records(records)

# Save
csv_path = OUTPUT_DIR / "sme_oi_tanzania.csv"
df.to_csv(csv_path, index=False)

with open(OUTPUT_DIR / "README.md", "w") as f:
    f.write("""
# Data: Synthetic Tanzanian SMEs OI dataset

- N = 313 SMEs across Manufacturing, Retail, ICT, Agriculture
- Variables: standardized latent barriers and digital literacy, OI index (0-100), demographic controls, 30 Likert items across barrier and digital literacy dimensions
- Generated reproducibly with seed=42 to approximate effect sizes claimed in the thesis
""".strip())

print(f"Wrote dataset to {csv_path}")
