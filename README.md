# Welding Inverse Design Dataset Generator

## For PhD Research on Extreme-Temperature Performance of Laser Welds in Battery Manufacturing

This comprehensive toolkit generates synthetic datasets for inverse design machine learning models in laser welding applications, with a specific focus on extreme-temperature performance relevant to battery manufacturing.

## 🎯 Purpose

This dataset generator supports PhD research on inverse design approaches for optimizing laser welding parameters. The goal is to start with desired performance characteristics (especially extreme-temperature resilience) and work backward to find the optimal manufacturing parameters.

### Inverse Design Flow:
```
Traditional: Input Parameters (X) → Physical Process → Output Performance (Y)
Inverse:     Desired Performance (Y) → ML Model → Optimal Parameters (X)
```

## 📊 Dataset Structure

### Multi-Tier Data Generation

The system generates three tiers of data to simulate a realistic research environment:

1. **Tier 1: High-Fidelity Experimental Data** (500 samples)
   - Simulates real laboratory experiments
   - Higher noise levels, conservative parameter ranges
   - Used as test set for final model validation

2. **Tier 2: Computational Simulation Data** (10,000 samples)
   - Simulates FEM/multi-physics simulations
   - Wider parameter ranges, lower noise
   - Primary training dataset

3. **Tier 3: Literature-Based Data** (1,000 samples)
   - Simulates data from various sources
   - Mixed fidelity, some missing values
   - Used for validation

### Input Parameters (X)

| Category | Parameters | Units |
|----------|------------|-------|
| **Energy Input** | Laser Power | W |
| | Welding Speed | mm/s |
| | Pulse Frequency | Hz |
| | Pulse Duration | ms |
| **Beam Characteristics** | Beam Focus Position | mm |
| | Beam Spot Size | μm |
| **Material & Setup** | Clamping Pressure | MPa |
| | Shield Gas Type | Categorical |
| | Gas Flow Rate | L/min |
| | Material Combination | Categorical |
| **Geometry** | Sheet Thickness | mm |
| | Joint Type | Categorical |
| | Overlap Distance | mm |

### Output Parameters (Y)

| Category | Parameters | Units |
|----------|------------|-------|
| **Weld Morphology** | Nugget Width | mm |
| | Penetration Depth | mm |
| | HAZ Width | mm |
| **Defects** | Has Cracks | Binary |
| | Has Porosity | Binary |
| | Porosity Size | μm |
| | Spatter Count | Count |
| **Mechanical Properties** | Tensile Strength | N |
| | Peel Strength | N |
| **Electrical Properties** | Contact Resistance | μΩ |
| | Resistance Increase % | % |
| **Extreme-Temperature** | IMC Thickness (initial) | μm |
| | IMC Thickness (aged) | μm |
| | Cycles to Failure | Count |
| | Strength Degradation % | % |
| | Creep Time to Failure | hours |
| **Microstructure** | Grain Size (initial) | μm |
| | Grain Size (aged) | μm |

## 🚀 Quick Start

### Installation

```bash
# Clone or download the repository
# Install dependencies
pip install -r requirements.txt
```

### Generate Dataset

```bash
python main.py
```

This will:
1. Generate multi-tier synthetic dataset (11,500 samples)
2. Perform comprehensive analysis
3. Create visualizations
4. Export ML-ready data
5. Optionally train inverse design models

### Custom Dataset Generation

```python
from welding_dataset_generator import WeldingDatasetGenerator

# Create generator with custom seed
generator = WeldingDatasetGenerator(seed=123)

# Generate custom dataset sizes
dataset = generator.generate_complete_dataset(
    tier1_samples=200,    # Experimental
    tier2_samples=5000,   # Simulation
    tier3_samples=500     # Literature
)

# Save dataset
generator.save_dataset(dataset, 'custom_dataset')
```

## 📁 Output Files

```
welding_dataset/
├── complete_dataset.csv       # Full dataset in CSV format
├── complete_dataset.parquet   # Efficient parquet format
├── metadata.json              # Dataset metadata
└── summary_statistics.csv     # Statistical summary

analysis_report/
├── analysis_overview.png      # 15 comprehensive visualizations
├── pca_analysis.png          # PCA dimensionality reduction
└── extreme_temperature_analysis.png  # Thermal performance analysis

ml_ready_data/
├── X_train.npy, Y_train.npy  # Training data
├── X_val.npy, Y_val.npy      # Validation data
├── X_test.npy, Y_test.npy    # Test data
├── feature_names.json        # Feature mappings
└── normalization_params.json # Scaling parameters
```

## 🤖 Inverse Design Models

The package includes two inverse design model architectures:

### 1. Conditional Variational Autoencoder (VAE)
- Learns latent representation of the parameter-performance relationship
- Generates multiple diverse solutions
- Handles uncertainty quantification

### 2. Conditional GAN
- Adversarial training for realistic parameter generation
- High-quality single solutions
- Better for specific target matching

### Using Inverse Design

```python
from inverse_design_model import InverseDesignTrainer

# Train model
trainer = InverseDesignTrainer(model_type='vae')
trainer.load_data('ml_ready_data')
history = trainer.train_vae(epochs=100)

# Define desired performance
desired_outputs = {
    'tensile_strength': 3500,  # N
    'contact_resistance': 12,  # μΩ
    'cycles_to_failure': 1000  # cycles
}

# Generate optimal parameters
solutions = trainer.inverse_design(desired_outputs, n_solutions=10)
print(solutions.head())
```

## 📈 Physics-Based Models

The synthetic data generation incorporates several physics-based relationships:

1. **Heat Input Model**: `Q = P / (v * 1000)` [kJ/mm]
2. **Power Density**: `I = P / (π * r²)` [W/mm²]
3. **Penetration Depth**: Function of power density and material thermal conductivity
4. **IMC Growth**: Arrhenius-type model for dissimilar metal joining
5. **Thermal Cycling Degradation**: Coffin-Manson relationship

## 🔬 Material Combinations

Supported material combinations with distinct properties:
- **Cu-Al**: Critical for battery busbars, high IMC growth
- **Al-Al**: Cell-to-cell connections, no IMC issues
- **Al-Steel**: Structural joints, extreme melting point difference
- **Cu-Cu**: High conductivity applications

## 📊 Data Quality Features

- **Noise modeling** based on data tier
- **Missing value simulation** for literature data
- **Physical constraint checking** (e.g., penetration < sheet thickness)
- **Uncertainty quantification** through standard deviation tracking
- **Quality scoring** system for weld assessment

## 🎓 Research Applications

This dataset is designed for:
1. Training inverse design models for welding parameter optimization
2. Multi-objective optimization studies
3. Transfer learning between simulation and experimental data
4. Uncertainty quantification in manufacturing
5. Process-structure-property relationship modeling

## ⚙️ Customization

### Modify Physical Models

Edit `welding_dataset_generator.py`:
- `simulate_weld_outputs()`: Core physics models
- `material_properties`: Material database
- Parameter ranges in `generate_input_parameters()`

### Add New Features

1. Add to input/output generation in `WeldingDatasetGenerator`
2. Update feature lists in `export_for_ml_training()`
3. Modify inverse design models if needed

## 📝 Citation

If you use this dataset generator in your research, please cite:

```bibtex
@software{welding_inverse_design_2024,
  title = {Welding Inverse Design Dataset Generator},
  author = {[Your Name]},
  year = {2024},
  description = {Synthetic dataset generation for inverse design of laser welding parameters with extreme-temperature performance focus}
}
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional material combinations
- More sophisticated defect models
- Advanced microstructure evolution models
- Real experimental data integration tools

## 📧 Contact

For questions or collaborations regarding this dataset generator, please contact [your email].

## 📄 License

This project is provided for research purposes. Please check with your institution for appropriate usage guidelines.

---

**Note**: This generates synthetic data for research and development. Always validate models with real experimental data before production use.