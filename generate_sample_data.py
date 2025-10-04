import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define sample parameters
n_samples = 313

# Generate demographic variables
sectors = ['Manufacturing', 'Retail', 'ICT', 'Agriculture']
locations = ['Urban', 'Rural']

# Create sample data
data = {
    'sme_id': range(1, n_samples + 1),
    'sector': np.random.choice(sectors, n_samples, p=[0.28, 0.25, 0.15, 0.32]),
    'location': np.random.choice(locations, n_samples, p=[0.6, 0.4]),
    'firm_size': np.random.choice(['Small', 'Medium'], n_samples, p=[0.7, 0.3]),
    'firm_age': np.random.normal(8.5, 3.5, n_samples).clip(1, 30),
    'owner_age': np.random.normal(42, 8, n_samples).clip(25, 65),
    'owner_education': np.random.choice(['Primary', 'Secondary', 'Tertiary'], n_samples, p=[0.2, 0.4, 0.4])
}

# Generate organizational barriers (4 dimensions, 4 items each)
barriers_dims = ['Structural', 'Cultural', 'Resource', 'Relational']
for dim in barriers_dims:
    for i in range(4):
        data[f'{dim.lower()}_barrier_{i+1}'] = np.random.normal(4, 1.2, n_samples).clip(1, 7)

# Generate digital literacy (4 dimensions, 5 items each)
dl_dims = ['Technical', 'Information', 'Communication', 'Strategic']
for dim in dl_dims:
    for i in range(5):
        data[f'{dim.lower()}_literacy_{i+1}'] = np.random.normal(3.5, 1.3, n_samples).clip(1, 7)

# Generate OI adoption (3 dimensions, 4 items each)
oi_dims = ['Inbound', 'Outbound', 'Coupled']
for dim in oi_dims:
    for i in range(4):
        data[f'{dim.lower()}_oi_{i+1}'] = np.random.normal(3.2, 1.4, n_samples).clip(1, 7)

# Create DataFrame
df = pd.DataFrame(data)

# Add some realistic correlations
# Higher digital literacy in urban ICT firms
urban_ict = (df['location'] == 'Urban') & (df['sector'] == 'ICT')
df.loc[urban_ict, [col for col in df.columns if 'literacy' in col]] += 0.8

# Higher barriers in rural agriculture
rural_agri = (df['location'] == 'Rural') & (df['sector'] == 'Agriculture')
df.loc[rural_agri, [col for col in df.columns if 'barrier' in col]] += 0.5

# Higher OI adoption with higher digital literacy
for i in range(len(df)):
    dl_avg = df.iloc[i][[col for col in df.columns if 'literacy' in col]].mean()
    oi_cols = [col for col in df.columns if 'oi_' in col]
    for col in oi_cols:
        df.loc[i, col] += (dl_avg - 3.5) * 0.3

# Round numeric values appropriately
numeric_cols = [col for col in df.columns if col not in ['sme_id', 'sector', 'location', 'firm_size', 'owner_education']]
for col in numeric_cols:
    df[col] = df[col].round(2)

# Save to CSV
df.to_csv('/workspace/data/sme_survey_data.csv', index=False)

# Generate descriptive statistics for report
desc_stats = df.describe()
desc_stats.to_csv('/workspace/data/descriptive_stats.csv')

print(f"Generated sample dataset with {n_samples} SMEs")
print("Data saved to /workspace/data/sme_survey_data.csv")
print("Descriptive statistics saved to /workspace/data/descriptive_stats.csv")