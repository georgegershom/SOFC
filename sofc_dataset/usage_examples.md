# SOFC Multi-Fidelity Dataset Usage Examples
==================================================

## Python Usage Example
```python
import pandas as pd
import numpy as np

# Load datasets
lf_data = pd.read_csv('sofc_dataset_lf.csv')
mf_data = pd.read_csv('sofc_dataset_mf.csv')
hf_data = pd.read_csv('sofc_dataset_hf.csv')

# Combine all fidelity levels
all_data = pd.concat([lf_data, mf_data, hf_data], ignore_index=True)

# Filter by operating conditions
high_temp_data = all_data[all_data['temperature'] > 1050]

# Calculate derived metrics
all_data['power_density'] = all_data['current_density'] * all_data['voltage']
all_data['efficiency'] = all_data['voltage'] / 1.25
```

## Machine Learning Example
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Prepare features and target
feature_cols = [col for col in all_data.columns if col not in ['fidelity_level']]
X = all_data[feature_cols]
y = all_data['efficiency']  # or any other target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f'R² Score: {score:.3f}')
```

## HDF5 Usage Example
```python
import h5py

# Load from HDF5
with h5py.File('sofc_dataset.h5', 'r') as f:
    # Access different fidelity levels
    lf_group = f['LF']
    mf_group = f['MF']
    hf_group = f['HF']
    
    # Access specific parameters
    temperature = lf_group['temperature'][:]
    current_density = lf_group['current_density'][:]
```
