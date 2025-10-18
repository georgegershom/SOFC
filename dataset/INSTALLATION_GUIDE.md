# Installation & Usage Guide
## Fire-Resistant Rubberized Concrete Dataset

### Quick Start

#### 1. **Download Dataset**
```bash
# Clone or download the complete dataset
git clone [repository-url]
cd fire-resistant-rubberized-concrete-dataset
```

#### 2. **Install Dependencies**
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r dataset/04_analysis_scripts/requirements.txt
```

#### 3. **Run Analysis**
```bash
cd dataset/04_analysis_scripts/
python data_analysis.py
```

### Detailed Installation

#### **System Requirements**
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM
- **Storage**: 500MB free space (including outputs)
- **OS**: Windows, macOS, or Linux

#### **Python Dependencies**
```bash
# Core data analysis
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0

# Statistical analysis
scikit-learn>=1.1.0
statsmodels>=0.13.0

# Data handling
openpyxl>=3.0.0
xlsxwriter>=3.0.0
```

#### **Optional Dependencies**
```bash
# For Jupyter notebook analysis
jupyter>=1.0.0
ipykernel>=6.0.0

# For advanced statistical analysis
pingouin>=0.5.0
```

### Usage Examples

#### **Basic Data Loading**
```python
import json
import pandas as pd
from pathlib import Path

# Load constituent materials data
with open('01_constituent_materials/cement_data.json', 'r') as f:
    cement_data = json.load(f)

# Load mix designs
with open('02_mix_designs/mix_design_matrix.json', 'r') as f:
    mix_data = json.load(f)

# Load fresh properties
with open('03_fresh_properties/fresh_concrete_data.json', 'r') as f:
    fresh_data = json.load(f)
```

#### **Quick Analysis**
```python
from data_analysis import RubberizedConcreteAnalyzer

# Initialize analyzer
analyzer = RubberizedConcreteAnalyzer()

# Run complete analysis
analyzer.run_complete_analysis()

# Or run specific analyses
analyzer.analyze_constituent_materials()
analyzer.analyze_mix_designs()
analyzer.analyze_fresh_properties()
```

#### **Custom Visualization**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data
df = pd.read_csv('04_analysis_scripts/combined_dataset.csv')

# Create custom plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='rubber_content', y='slump', 
                hue='rubber_size', size='unit_weight')
plt.title('Slump vs Rubber Content')
plt.show()
```

### Analysis Workflows

#### **Workflow 1: Complete Dataset Analysis**
```bash
# Step 1: Run main analysis
python data_analysis.py

# Step 2: Generate advanced visualizations
python advanced_visualization.py

# Step 3: View interactive plots
# Open generated HTML files in browser
```

#### **Workflow 2: Custom Analysis**
```python
# Import analyzer class
from data_analysis import RubberizedConcreteAnalyzer

# Initialize with custom data path
analyzer = RubberizedConcreteAnalyzer(data_path="../dataset/")

# Create custom DataFrames
df_mix = analyzer.create_mix_design_dataframe()
df_fresh = analyzer.create_fresh_properties_dataframe()

# Perform custom analysis
correlation_matrix = df_fresh.corr()
print(correlation_matrix)
```

#### **Workflow 3: Statistical Modeling**
```python
from scipy import stats
import numpy as np

# Load combined dataset
df = pd.read_csv('combined_dataset.csv')

# Perform regression analysis
slope, intercept, r_value, p_value, std_err = stats.linregress(
    df['rubber_content'], df['slump'])

print(f"Slump = {intercept:.2f} + {slope:.2f} × Rubber_Content")
print(f"R² = {r_value**2:.3f}, p-value = {p_value:.4f}")
```

### Output Files

#### **Generated Analysis Files**
```
04_analysis_scripts/
├── mix_designs.csv              # Processed mix design data
├── fresh_properties.csv         # Fresh properties data
├── combined_dataset.csv         # Merged dataset
├── summary_statistics.json      # Statistical summaries
├── comprehensive_analysis.png   # Main analysis plots
├── publication_plots.png        # Publication-ready figures
├── publication_plots.pdf        # PDF version for papers
├── interactive_dashboard.html   # Interactive Plotly dashboard
├── 3d_visualization.html        # 3D property visualization
└── animated_plot.html          # Animated property evolution
```

#### **File Descriptions**
- **CSV files**: Structured data for further analysis
- **PNG/PDF files**: High-resolution plots for publications
- **HTML files**: Interactive visualizations for exploration
- **JSON files**: Statistical summaries and metadata

### Troubleshooting

#### **Common Issues**

**1. Import Errors**
```bash
# Error: ModuleNotFoundError
# Solution: Install missing dependencies
pip install [missing-package]
```

**2. File Path Issues**
```python
# Error: FileNotFoundError
# Solution: Check working directory
import os
print(os.getcwd())
os.chdir('path/to/dataset')
```

**3. Memory Issues**
```python
# Error: MemoryError
# Solution: Process data in chunks
df_chunk = pd.read_csv('file.csv', chunksize=1000)
```

**4. Plot Display Issues**
```python
# Error: Plots not showing
# Solution: Set backend
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

#### **Performance Optimization**

**For Large Datasets:**
```python
# Use efficient data types
df = pd.read_csv('data.csv', dtype={'rubber_content': 'int8'})

# Optimize memory usage
df = df.astype({'slump': 'float32', 'air_content': 'float32'})
```

**For Slow Plotting:**
```python
# Reduce plot complexity
plt.rcParams['figure.max_open_warning'] = 0
plt.ioff()  # Turn off interactive mode
```

### Advanced Usage

#### **Custom Analysis Functions**
```python
def analyze_rubber_size_effect(df):
    """Custom function to analyze rubber size effects"""
    size_groups = df.groupby('rubber_size')
    
    results = {}
    for size, group in size_groups:
        results[size] = {
            'mean_slump': group['slump'].mean(),
            'std_slump': group['slump'].std(),
            'count': len(group)
        }
    
    return results

# Usage
results = analyze_rubber_size_effect(df)
print(results)
```

#### **Integration with Other Tools**
```python
# Export to Excel for further analysis
with pd.ExcelWriter('analysis_results.xlsx') as writer:
    df_mix.to_excel(writer, sheet_name='Mix_Designs')
    df_fresh.to_excel(writer, sheet_name='Fresh_Properties')

# Export to R format
df.to_csv('data_for_r.csv', index=False)
```

#### **Batch Processing**
```python
# Process multiple datasets
import glob

data_files = glob.glob('*.json')
results = {}

for file in data_files:
    with open(file, 'r') as f:
        data = json.load(f)
        # Process each file
        results[file] = analyze_data(data)
```

### Best Practices

#### **Data Handling**
1. Always validate data integrity before analysis
2. Use version control for analysis scripts
3. Document all custom modifications
4. Maintain original data files unchanged

#### **Analysis Workflow**
1. Start with exploratory data analysis
2. Check data quality and outliers
3. Perform statistical tests before conclusions
4. Validate results with domain knowledge

#### **Visualization**
1. Use appropriate plot types for data
2. Include error bars for experimental data
3. Label axes and provide legends
4. Save plots in multiple formats

### Support & Resources

#### **Documentation**
- **README.md**: Dataset overview and structure
- **METADATA.json**: Complete technical specifications
- **Code comments**: Detailed function documentation

#### **Getting Help**
1. Check error messages and logs
2. Review documentation and examples
3. Verify data file integrity
4. Contact research team if needed

#### **Contributing**
- Report bugs or issues
- Suggest improvements
- Share analysis results
- Contribute additional visualizations

### Version Information
- **Dataset Version**: 1.0
- **Analysis Scripts**: Compatible with Python 3.8+
- **Last Updated**: 2024-10-18
- **Compatibility**: Cross-platform (Windows/macOS/Linux)