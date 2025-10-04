import csv
import random
import math

# Set random seed for reproducibility
random.seed(42)

# Define sample parameters
n_samples = 313
sectors = ['Manufacturing', 'Retail', 'ICT', 'Agriculture']
locations = ['Urban', 'Rural']
firm_sizes = ['Small', 'Medium']
education_levels = ['Primary', 'Secondary', 'Tertiary']

# Generate sample data
data_rows = []

for sme_id in range(1, n_samples + 1):
    # Basic demographics
    sector = random.choices(sectors, weights=[0.28, 0.25, 0.15, 0.32])[0]
    location = random.choices(locations, weights=[0.6, 0.4])[0]
    firm_size = random.choices(firm_sizes, weights=[0.7, 0.3])[0]
    firm_age = round(max(1, random.gauss(8.5, 3.5)), 1)
    owner_age = round(max(25, min(65, random.gauss(42, 8))), 0)
    owner_education = random.choices(education_levels, weights=[0.2, 0.4, 0.4])[0]

    # Generate barriers (4 dimensions, 4 items each)
    barriers = {}
    for dim in ['Structural', 'Cultural', 'Resource', 'Relational']:
        dim_barriers = []
        for i in range(4):
            base_value = 4.0
            # Add variation based on context
            if location == 'Rural' and sector == 'Agriculture':
                base_value += 0.5
            elif location == 'Urban' and sector == 'ICT':
                base_value -= 0.3
            value = round(max(1, min(7, base_value + random.gauss(0, 1.2))), 2)
            dim_barriers.append(value)
        barriers[dim.lower()] = dim_barriers

    # Generate digital literacy (4 dimensions, 5 items each)
    digital_literacy = {}
    for dim in ['Technical', 'Information', 'Communication', 'Strategic']:
        dim_literacy = []
        for i in range(5):
            base_value = 3.5
            # Higher digital literacy in urban ICT firms
            if location == 'Urban' and sector == 'ICT':
                base_value += 0.8
            value = round(max(1, min(7, base_value + random.gauss(0, 1.3))), 2)
            dim_literacy.append(value)
        digital_literacy[dim.lower()] = dim_literacy

    # Generate OI adoption (3 dimensions, 4 items each)
    oi_adoption = {}
    for dim in ['Inbound', 'Outbound', 'Coupled']:
        dim_oi = []
        for i in range(4):
            # Base value influenced by digital literacy
            dl_avg = sum(sum(dims) for dims in digital_literacy.values()) / (4 * 5)
            base_value = 3.2 + (dl_avg - 3.5) * 0.3
            value = round(max(1, min(7, base_value + random.gauss(0, 1.4))), 2)
            dim_oi.append(value)
        oi_adoption[dim.lower()] = dim_oi

    # Create row
    row = {
        'sme_id': sme_id,
        'sector': sector,
        'location': location,
        'firm_size': firm_size,
        'firm_age': firm_age,
        'owner_age': owner_age,
        'owner_education': owner_education
    }

    # Add barriers
    for dim, values in barriers.items():
        for i, value in enumerate(values):
            row[f'{dim}_barrier_{i+1}'] = value

    # Add digital literacy
    for dim, values in digital_literacy.items():
        for i, value in enumerate(values):
            row[f'{dim}_literacy_{i+1}'] = value

    # Add OI adoption
    for dim, values in oi_adoption.items():
        for i, value in enumerate(values):
            row[f'{dim}_oi_{i+1}'] = value

    data_rows.append(row)

# Write to CSV
with open('/workspace/data/sme_survey_data.csv', 'w', newline='') as csvfile:
    fieldnames = data_rows[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in data_rows:
        writer.writerow(row)

print(f"Generated sample dataset with {n_samples} SMEs")
print("Data saved to /workspace/data/sme_survey_data.csv")

# Generate summary statistics
total_responses = len(data_rows)
sectors_count = {}
locations_count = {}
sizes_count = {}

for row in data_rows:
    sector = row['sector']
    location = row['location']
    size = row['firm_size']

    sectors_count[sector] = sectors_count.get(sector, 0) + 1
    locations_count[location] = locations_count.get(location, 0) + 1
    sizes_count[size] = sizes_count.get(size, 0) + 1

print("\nSample Summary:")
print(f"Total responses: {total_responses}")
print(f"Sector distribution: {sectors_count}")
print(f"Location distribution: {locations_count}")
print(f"Size distribution: {sizes_count}")