"""
Material Properties Data Generator for SOFC Multi-Physics Models
Generates synthetic but realistic material property data for:
- 8YSZ (Yttria-Stabilized Zirconia) - Electrolyte
- LSM (Lanthanum Strontium Manganite) - Cathode
- Ni-YSZ Cermet - Anode
- Ferritic Stainless Steel - Interconnect
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def add_realistic_noise(data, noise_percent=2):
    """Add realistic experimental noise to data"""
    noise = np.random.normal(0, noise_percent/100, size=data.shape)
    return data * (1 + noise)


class MaterialPropertiesGenerator:
    def __init__(self, output_dir="material_properties_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define materials
        self.materials = {
            "8YSZ": "8% Yttria-Stabilized Zirconia (Electrolyte)",
            "LSM": "La0.8Sr0.2MnO3 (Cathode)",
            "Ni-YSZ": "Nickel-YSZ Cermet (Anode)",
            "FSS": "Ferritic Stainless Steel (Interconnect)"
        }
    
    def generate_mechanical_properties(self):
        """Generate mechanical properties as function of temperature"""
        print("Generating mechanical properties dataset...")
        
        # Temperature range: 20°C to 1000°C
        temperatures = np.arange(20, 1001, 20)
        
        data = []
        
        # 8YSZ (Electrolyte)
        for T in temperatures:
            # Young's modulus decreases with temperature
            E_base = 200  # GPa at RT
            E = E_base * (1 - 0.00025 * (T - 20))
            
            # Tensile strength
            sigma_base = 300  # MPa at RT
            sigma = sigma_base * (1 - 0.0003 * (T - 20))
            
            # Poisson's ratio slightly increases with temperature
            nu = 0.31 + 0.00002 * (T - 20)
            
            data.append({
                "Material": "8YSZ",
                "Temperature_C": T,
                "Youngs_Modulus_GPa": add_realistic_noise(np.array([E]), 2)[0],
                "Tensile_Strength_MPa": add_realistic_noise(np.array([sigma]), 3)[0],
                "Poissons_Ratio": add_realistic_noise(np.array([nu]), 1)[0]
            })
        
        # LSM (Cathode)
        for T in temperatures:
            E_base = 80  # GPa at RT
            E = E_base * (1 - 0.0004 * (T - 20))
            
            sigma_base = 150  # MPa at RT
            sigma = sigma_base * (1 - 0.0005 * (T - 20))
            
            nu = 0.28 + 0.00003 * (T - 20)
            
            data.append({
                "Material": "LSM",
                "Temperature_C": T,
                "Youngs_Modulus_GPa": add_realistic_noise(np.array([E]), 3)[0],
                "Tensile_Strength_MPa": add_realistic_noise(np.array([sigma]), 4)[0],
                "Poissons_Ratio": add_realistic_noise(np.array([nu]), 1.5)[0]
            })
        
        # Ni-YSZ (Anode)
        for T in temperatures:
            E_base = 50  # GPa at RT (porous composite)
            E = E_base * (1 - 0.0005 * (T - 20))
            
            sigma_base = 100  # MPa at RT
            sigma = sigma_base * (1 - 0.0006 * (T - 20))
            
            nu = 0.25 + 0.00004 * (T - 20)
            
            data.append({
                "Material": "Ni-YSZ",
                "Temperature_C": T,
                "Youngs_Modulus_GPa": add_realistic_noise(np.array([E]), 4)[0],
                "Tensile_Strength_MPa": add_realistic_noise(np.array([sigma]), 5)[0],
                "Poissons_Ratio": add_realistic_noise(np.array([nu]), 2)[0]
            })
        
        # Ferritic Stainless Steel (Interconnect)
        for T in temperatures:
            E_base = 200  # GPa at RT
            E = E_base * (1 - 0.0003 * (T - 20))
            
            sigma_base = 450  # MPa at RT
            sigma = sigma_base * (1 - 0.0007 * (T - 20))
            
            nu = 0.30 + 0.00001 * (T - 20)
            
            data.append({
                "Material": "FSS",
                "Temperature_C": T,
                "Youngs_Modulus_GPa": add_realistic_noise(np.array([E]), 2)[0],
                "Tensile_Strength_MPa": add_realistic_noise(np.array([sigma]), 3)[0],
                "Poissons_Ratio": add_realistic_noise(np.array([nu]), 1)[0]
            })
        
        df = pd.DataFrame(data)
        output_path = self.output_dir / "mechanical_properties.csv"
        df.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")
        return df
    
    def generate_creep_properties(self):
        """Generate creep strain vs time curves at different temperatures and stresses"""
        print("Generating creep properties dataset...")
        
        # Test conditions
        temperatures = [700, 750, 800, 850, 900]  # °C
        stresses = [50, 75, 100, 125, 150]  # MPa
        time_hours = np.linspace(0, 1000, 100)  # 1000 hours
        
        data = []
        
        for material in ["8YSZ", "LSM", "Ni-YSZ", "FSS"]:
            for T in temperatures:
                for stress in stresses:
                    # Norton-Bailey creep law: ε = A * σ^n * t^m * exp(-Q/RT)
                    # Different parameters for each material
                    
                    if material == "8YSZ":
                        A = 1e-12
                        n = 1.5
                        m = 0.33
                        Q = 500000  # J/mol
                    elif material == "LSM":
                        A = 5e-11
                        n = 2.0
                        m = 0.4
                        Q = 450000
                    elif material == "Ni-YSZ":
                        A = 1e-10
                        n = 2.5
                        m = 0.45
                        Q = 400000
                    else:  # FSS
                        A = 2e-11
                        n = 3.0
                        m = 0.35
                        Q = 420000
                    
                    R = 8.314  # J/(mol·K)
                    T_K = T + 273.15
                    
                    for t in time_hours:
                        if t == 0:
                            epsilon = 0
                        else:
                            epsilon = A * (stress ** n) * (t ** m) * np.exp(-Q / (R * T_K))
                            epsilon = add_realistic_noise(np.array([epsilon]), 3)[0]
                        
                        data.append({
                            "Material": material,
                            "Temperature_C": T,
                            "Stress_MPa": stress,
                            "Time_hours": t,
                            "Creep_Strain_percent": epsilon * 100
                        })
        
        df = pd.DataFrame(data)
        output_path = self.output_dir / "creep_properties.csv"
        df.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")
        return df
    
    def generate_thermophysical_properties(self):
        """Generate thermal properties as function of temperature"""
        print("Generating thermo-physical properties dataset...")
        
        temperatures = np.arange(20, 1001, 20)
        data = []
        
        # 8YSZ
        for T in temperatures:
            # CTE increases slightly with temperature
            cte = (10.5 + 0.001 * (T - 20)) * 1e-6  # 1/K
            
            # Thermal conductivity (YSZ has low conductivity)
            k = 2.7 - 0.0004 * (T - 20)  # W/(m·K)
            
            # Specific heat increases with temperature
            cp = 400 + 0.15 * (T - 20)  # J/(kg·K)
            
            data.append({
                "Material": "8YSZ",
                "Temperature_C": T,
                "CTE_1perK": add_realistic_noise(np.array([cte]), 2)[0],
                "Thermal_Conductivity_WperMK": add_realistic_noise(np.array([k]), 3)[0],
                "Specific_Heat_JperKgK": add_realistic_noise(np.array([cp]), 2)[0],
                "Density_kg_per_m3": 6000
            })
        
        # LSM
        for T in temperatures:
            cte = (11.5 + 0.002 * (T - 20)) * 1e-6
            k = 3.5 - 0.0003 * (T - 20)
            cp = 450 + 0.18 * (T - 20)
            
            data.append({
                "Material": "LSM",
                "Temperature_C": T,
                "CTE_1perK": add_realistic_noise(np.array([cte]), 2.5)[0],
                "Thermal_Conductivity_WperMK": add_realistic_noise(np.array([k]), 3)[0],
                "Specific_Heat_JperKgK": add_realistic_noise(np.array([cp]), 2)[0],
                "Density_kg_per_m3": 6500
            })
        
        # Ni-YSZ
        for T in temperatures:
            cte = (12.0 + 0.0015 * (T - 20)) * 1e-6
            k = 6.0 - 0.0008 * (T - 20)  # Higher due to Ni
            cp = 500 + 0.2 * (T - 20)
            
            data.append({
                "Material": "Ni-YSZ",
                "Temperature_C": T,
                "CTE_1perK": add_realistic_noise(np.array([cte]), 3)[0],
                "Thermal_Conductivity_WperMK": add_realistic_noise(np.array([k]), 4)[0],
                "Specific_Heat_JperKgK": add_realistic_noise(np.array([cp]), 2.5)[0],
                "Density_kg_per_m3": 5800
            })
        
        # FSS
        for T in temperatures:
            cte = (11.0 + 0.0012 * (T - 20)) * 1e-6
            k = 25.0 - 0.003 * (T - 20)  # Metals have high conductivity
            cp = 460 + 0.25 * (T - 20)
            
            data.append({
                "Material": "FSS",
                "Temperature_C": T,
                "CTE_1perK": add_realistic_noise(np.array([cte]), 2)[0],
                "Thermal_Conductivity_WperMK": add_realistic_noise(np.array([k]), 3)[0],
                "Specific_Heat_JperKgK": add_realistic_noise(np.array([cp]), 2)[0],
                "Density_kg_per_m3": 7800
            })
        
        df = pd.DataFrame(data)
        output_path = self.output_dir / "thermophysical_properties.csv"
        df.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")
        return df
    
    def generate_electrochemical_properties(self):
        """Generate ionic and electronic conductivity as function of temperature"""
        print("Generating electrochemical properties dataset...")
        
        temperatures = np.arange(400, 1001, 20)  # SOFC operating range
        data = []
        
        # 8YSZ - Ionic conductor
        for T in temperatures:
            T_K = T + 273.15
            # Arrhenius equation: σ = σ0 * exp(-Ea/RT)
            sigma_0 = 3.34e4  # S/m
            Ea = 84000  # J/mol
            R = 8.314
            
            sigma_ionic = sigma_0 * np.exp(-Ea / (R * T_K))
            sigma_electronic = 1e-10  # Negligible electronic conductivity
            
            data.append({
                "Material": "8YSZ",
                "Temperature_C": T,
                "Ionic_Conductivity_Sperm": add_realistic_noise(np.array([sigma_ionic]), 3)[0],
                "Electronic_Conductivity_Sperm": sigma_electronic,
                "Total_Conductivity_Sperm": sigma_ionic,
                "Transport_Number_Ionic": 0.9999
            })
        
        # LSM - Mixed ionic-electronic conductor (MIEC)
        for T in temperatures:
            T_K = T + 273.15
            sigma_0_ionic = 1e3
            Ea_ionic = 110000
            sigma_0_electronic = 5e4
            Ea_electronic = 40000
            R = 8.314
            
            sigma_ionic = sigma_0_ionic * np.exp(-Ea_ionic / (R * T_K))
            sigma_electronic = sigma_0_electronic * np.exp(-Ea_electronic / (R * T_K))
            sigma_total = sigma_ionic + sigma_electronic
            t_ion = sigma_ionic / sigma_total
            
            data.append({
                "Material": "LSM",
                "Temperature_C": T,
                "Ionic_Conductivity_Sperm": add_realistic_noise(np.array([sigma_ionic]), 4)[0],
                "Electronic_Conductivity_Sperm": add_realistic_noise(np.array([sigma_electronic]), 3)[0],
                "Total_Conductivity_Sperm": sigma_total,
                "Transport_Number_Ionic": t_ion
            })
        
        # Ni-YSZ - High electronic conductivity (due to Ni)
        for T in temperatures:
            T_K = T + 273.15
            sigma_0_ionic = 1e4
            Ea_ionic = 90000
            sigma_0_electronic = 1e6
            Ea_electronic = 10000
            R = 8.314
            
            sigma_ionic = sigma_0_ionic * np.exp(-Ea_ionic / (R * T_K))
            sigma_electronic = sigma_0_electronic * np.exp(-Ea_electronic / (R * T_K))
            sigma_total = sigma_ionic + sigma_electronic
            t_ion = sigma_ionic / sigma_total
            
            data.append({
                "Material": "Ni-YSZ",
                "Temperature_C": T,
                "Ionic_Conductivity_Sperm": add_realistic_noise(np.array([sigma_ionic]), 4)[0],
                "Electronic_Conductivity_Sperm": add_realistic_noise(np.array([sigma_electronic]), 2)[0],
                "Total_Conductivity_Sperm": sigma_total,
                "Transport_Number_Ionic": t_ion
            })
        
        # FSS - Pure electronic conductor
        for T in temperatures:
            T_K = T + 273.15
            sigma_0_electronic = 1.5e6
            Ea_electronic = 5000
            R = 8.314
            
            sigma_ionic = 1e-12  # Negligible
            sigma_electronic = sigma_0_electronic * np.exp(-Ea_electronic / (R * T_K))
            
            data.append({
                "Material": "FSS",
                "Temperature_C": T,
                "Ionic_Conductivity_Sperm": sigma_ionic,
                "Electronic_Conductivity_Sperm": add_realistic_noise(np.array([sigma_electronic]), 2)[0],
                "Total_Conductivity_Sperm": sigma_electronic,
                "Transport_Number_Ionic": 0.0001
            })
        
        df = pd.DataFrame(data)
        output_path = self.output_dir / "electrochemical_properties.csv"
        df.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")
        return df
    
    def generate_summary_file(self):
        """Generate a summary file with all material properties"""
        print("Generating summary documentation...")
        
        summary = f"""# Material Properties Dataset for SOFC Multi-Physics Models

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Materials Included

"""
        for code, name in self.materials.items():
            summary += f"- **{code}**: {name}\n"
        
        summary += """
## Dataset Files

### 1. mechanical_properties.csv
Contains temperature-dependent mechanical properties:
- Temperature range: 20°C to 1000°C (20°C intervals)
- Young's Modulus (GPa)
- Tensile Strength (MPa)
- Poisson's Ratio (dimensionless)

**Use for**: Structural mechanics simulations, stress analysis, thermal-mechanical coupling

### 2. creep_properties.csv
Contains creep strain vs. time data:
- Temperatures: 700, 750, 800, 850, 900°C
- Stress levels: 50, 75, 100, 125, 150 MPa
- Time range: 0 to 1000 hours
- Creep Strain (%)

**Use for**: Long-term degradation modeling, creep-fatigue analysis, lifetime prediction

### 3. thermophysical_properties.csv
Contains temperature-dependent thermal properties:
- Temperature range: 20°C to 1000°C (20°C intervals)
- Coefficient of Thermal Expansion (CTE, 1/K)
- Thermal Conductivity (W/(m·K))
- Specific Heat Capacity (J/(kg·K))
- Density (kg/m³)

**Use for**: Heat transfer simulations, thermal stress analysis, thermal-fluid coupling

### 4. electrochemical_properties.csv
Contains temperature-dependent electrochemical properties:
- Temperature range: 400°C to 1000°C (20°C intervals)
- Ionic Conductivity (S/m)
- Electronic Conductivity (S/m)
- Total Conductivity (S/m)
- Ionic Transport Number (dimensionless)

**Use for**: Electrochemical simulations, Butler-Volmer kinetics, charge transport modeling

## Data Generation Methodology

All data is **synthetically generated** based on:
1. Published literature values for SOFC materials
2. Physics-based models (Arrhenius, Norton-Bailey creep law)
3. Realistic experimental noise (2-5% depending on property)

### Key Equations Used:

**Mechanical Properties**: Linear temperature dependence with material-specific coefficients

**Creep**: Norton-Bailey Law
```
ε = A × σⁿ × tᵐ × exp(-Q/RT)
```

**Thermal Properties**: Empirical temperature-dependent correlations

**Electrochemical**: Arrhenius Equation
```
σ = σ₀ × exp(-Eₐ/RT)
```

## Material Property Highlights

### 8YSZ (Electrolyte)
- High ionic conductivity (~0.1 S/m at 800°C)
- Low electronic conductivity (insulator)
- Moderate mechanical strength
- Low thermal conductivity

### LSM (Cathode)
- Mixed ionic-electronic conductor (MIEC)
- Good electronic conductivity at operating temperatures
- Compatible CTE with YSZ
- Moderate mechanical strength

### Ni-YSZ (Anode)
- Very high electronic conductivity (due to Ni network)
- Good ionic conductivity (due to YSZ phase)
- Porous composite structure (reflected in lower mechanical properties)
- Risk of Ni coarsening at high temperatures

### FSS (Interconnect)
- Pure electronic conductor
- High thermal conductivity
- Good mechanical strength
- CTE matching is critical

## Usage Notes

1. **Temperature Units**: All temperatures are in Celsius (°C)
2. **Consistency**: Convert to Kelvin for thermodynamic calculations
3. **Interpolation**: Linear interpolation is recommended for intermediate temperatures
4. **Extrapolation**: DO NOT extrapolate beyond the given temperature ranges
5. **Validation**: This is synthetic data - validate with experimental data when available

## Multi-Physics Coupling Considerations

### Thermo-Mechanical
- Use CTE mismatch for thermal stress calculations
- Temperature-dependent Young's modulus for accurate stress prediction
- Include creep for long-term simulations

### Electro-Thermal
- Joule heating: Q = σ × E²
- Temperature affects conductivity (Arrhenius)
- Include contact resistance at interfaces

### Electro-Chemo-Mechanical
- Volume changes during redox cycling (especially Ni-YSZ)
- Stress affects ionic conductivity
- Chemical expansion coupling

## References & Validation

For production use, validate against:
- NIST materials database
- Manufacturer datasheets
- Direct experimental measurements
- Peer-reviewed literature (key papers in SOFC field)

## Data Format

All CSV files use the following conventions:
- Comma-separated values
- Header row with descriptive column names
- Consistent material naming across all files
- SI-derived units (specified in column headers)

## License & Usage

This synthetic dataset is provided for research and development purposes.
For commercial applications, validate with experimental data.

---

**Contact**: For questions about data generation methodology or specific material properties.
"""
        
        output_path = self.output_dir / "README.md"
        with open(output_path, 'w') as f:
            f.write(summary)
        print(f"✓ Saved to {output_path}")
    
    def generate_all(self):
        """Generate all datasets"""
        print("="*60)
        print("SOFC Material Properties Dataset Generator")
        print("="*60)
        print()
        
        self.generate_mechanical_properties()
        self.generate_creep_properties()
        self.generate_thermophysical_properties()
        self.generate_electrochemical_properties()
        self.generate_summary_file()
        
        print()
        print("="*60)
        print("✓ All datasets generated successfully!")
        print(f"✓ Output directory: {self.output_dir.absolute()}")
        print("="*60)


if __name__ == "__main__":
    generator = MaterialPropertiesGenerator()
    generator.generate_all()
