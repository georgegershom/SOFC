# SENCE Framework - Complete Usage Guide
## Figure 9: Advanced Implementation and Visualization

### 📚 Table of Contents

1. [Quick Start](#quick-start)
2. [Python Visualization](#python-visualization)
3. [Mermaid Diagrams](#mermaid-diagrams)
4. [PlantUML Architecture](#plantuml-architecture)
5. [Advanced Customization](#advanced-customization)
6. [Integration Examples](#integration-examples)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Installation

```bash
# Clone or download the repository
cd /workspace

# Install Python dependencies
pip install -r requirements.txt

# Run the main visualization
python3 sence_radar_visualization.py
```

### Expected Outputs

After running the script, you'll find in `/workspace/outputs/`:

- **figure9_sence_radar_chart.png** - Main radar chart (1200x900px, 300 DPI)
- **figure9_sence_radar_chart.pdf** - Vector format for publications
- **figure9_sence_3d_temporal.png** - 3D temporal evolution visualization
- **sence_statistical_report.txt** - Comprehensive statistical analysis
- **sence_vulnerability_data.csv** - Raw data export

---

## 🐍 Python Visualization

### Basic Usage

```python
from sence_radar_visualization import SENCERadarChart

# Initialize the radar chart
radar = SENCERadarChart()

# Generate all visualizations
radar.save_outputs(output_dir='/workspace/outputs')

# Or generate individual components
fig1 = radar.create_advanced_radar_chart()
fig2 = radar.create_enhanced_3d_visualization()
report = radar.generate_statistical_report()
```

### Customizing City Data

```python
# Modify city data
radar = SENCERadarChart()

# Add new city
radar.city_data['New City'] = {
    'values': [0.6, 0.7, 0.65, 0.55, 0.62, 0.68, 0.58, 0.64],
    'mean_cvi': 0.63,
    'color': '#FF6B6B',
    'linestyle': '-',
    'marker': 'D',
    'alpha': 0.25
}

# Update domains if needed
radar.domains = [
    'Environmental\nDegradation',
    'Economic\nFragility',
    # ... add more domains
]

# Regenerate visualizations
fig = radar.create_advanced_radar_chart()
fig.savefig('custom_radar.png', dpi=300)
```

### Advanced Plotting Options

```python
import matplotlib.pyplot as plt

# Create radar chart with custom figure size
radar = SENCERadarChart()
fig = radar.create_advanced_radar_chart(figsize=(20, 16))

# Customize before saving
plt.suptitle('Custom Title', fontsize=16, y=0.98)
fig.savefig('custom_figure.png', dpi=600, bbox_inches='tight')

# Export to different formats
fig.savefig('figure.svg', format='svg')  # Vector graphics
fig.savefig('figure.eps', format='eps')  # PostScript
```

### Accessing Statistical Data

```python
radar = SENCERadarChart()

# Get city-specific statistics
for city, data in radar.city_data.items():
    print(f"{city}: Mean CVI = {data['mean_cvi']}")
    print(f"Domain values: {data['values']}")

# Calculate custom metrics
import numpy as np
ph_values = np.array(radar.city_data['Port Harcourt']['values'])
print(f"Port Harcourt - Std Dev: {np.std(ph_values):.3f}")
print(f"Port Harcourt - Max: {np.max(ph_values):.3f}")
```

---

## 🧩 Mermaid Diagrams

### Rendering the Framework Diagram

**Method 1: Mermaid CLI**

```bash
# Install Mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Generate diagram
mmdc -i sence_framework.mmd -o sence_framework.png -w 3000 -H 2000

# Generate SVG (vector)
mmdc -i sence_framework.mmd -o sence_framework.svg

# Generate PDF
mmdc -i sence_framework.mmd -o sence_framework.pdf
```

**Method 2: Online Editor**

1. Visit [Mermaid Live Editor](https://mermaid.live/)
2. Copy contents of `sence_framework.mmd`
3. Paste into editor
4. Export as PNG/SVG/PDF

**Method 3: In Markdown (GitHub, Jupyter, etc.)**

```markdown
# Your Document

The SENCE framework architecture:

\```mermaid
graph TB
    %% Paste contents of sence_framework.mmd here
\```
```

### Embedding in Jupyter Notebook

```python
# In a Jupyter cell
from IPython.display import Markdown, display

with open('sence_framework.mmd', 'r') as f:
    mermaid_code = f.read()

display(Markdown(f"""
```mermaid
{mermaid_code}
```
"""))
```

### Customizing Mermaid Styling

Edit the Mermaid file and modify the styling section:

```mermaid
%%{init: {
    'theme': 'base',
    'themeVariables': {
        'primaryColor': '#BB2528',
        'primaryTextColor': '#fff',
        'primaryBorderColor': '#7C0000',
        'lineColor': '#F8B229',
        'secondaryColor': '#006100',
        'tertiaryColor': '#fff'
    }
}}%%

%% Rest of diagram...
```

---

## 🏗️ PlantUML Architecture

### Rendering PlantUML Diagrams

**Method 1: PlantUML JAR**

```bash
# Download PlantUML JAR
wget https://github.com/plantuml/plantuml/releases/download/v1.2024.7/plantuml.jar

# Generate PNG
java -jar plantuml.jar sence_system_architecture.puml

# Generate SVG (vector)
java -jar plantuml.jar -tsvg sence_system_architecture.puml

# Generate PDF
java -jar plantuml.jar -tpdf sence_system_architecture.puml

# High resolution
java -jar plantuml.jar -DPLANTUML_LIMIT_SIZE=8192 sence_system_architecture.puml
```

**Method 2: Online Server**

```bash
# Using PlantUML server
curl -X POST \
  --data-binary @sence_system_architecture.puml \
  http://www.plantuml.com/plantuml/png > architecture.png
```

**Method 3: VSCode Extension**

1. Install "PlantUML" extension in VSCode
2. Open `sence_system_architecture.puml`
3. Press `Alt+D` to preview
4. Right-click → "Export Current Diagram"

### PlantUML with Python

```python
from plantuml import PlantUML

# Initialize PlantUML server
plantuml = PlantUML(url='http://www.plantuml.com/plantuml/img/')

# Generate diagram
with open('sence_system_architecture.puml', 'r') as f:
    plantuml_code = f.read()

# Save as image
plantuml.processes(plantuml_code, outfile='architecture.png')
```

### Advanced PlantUML Configuration

Add to the beginning of your `.puml` file:

```plantuml
@startuml
!define RESOLUTION 300
!define WIDTH 4000
!define HEIGHT 3000

skinparam dpi ${RESOLUTION}
skinparam defaultFontName Arial
skinparam defaultFontSize 12
skinparam backgroundColor white
skinparam shadowing true

%% Rest of diagram...
@enduml
```

---

## 🎨 Advanced Customization

### Custom Color Schemes

```python
# Define custom color palette
custom_palette = {
    'Port Harcourt': {
        'color': '#E63946',  # Red
        'linestyle': '-',
        'marker': 'o'
    },
    'Warri': {
        'color': '#F1FAEE',  # Off-white
        'linestyle': '--',
        'marker': 's'
    },
    'Bonny': {
        'color': '#A8DADC',  # Light blue
        'linestyle': '-.',
        'marker': '^'
    }
}

# Apply to radar chart
radar = SENCERadarChart()
for city, style in custom_palette.items():
    if city in radar.city_data:
        radar.city_data[city].update(style)
```

### Publication-Ready Formatting

```python
import matplotlib.pyplot as plt

# Set journal-specific parameters
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 600,  # Nature/Science standard
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,  # TrueType fonts
    'ps.fonttype': 42
})

# Generate figure
radar = SENCERadarChart()
fig = radar.create_advanced_radar_chart()

# Save in journal format
fig.savefig('figure9_nature_format.pdf', 
           bbox_inches='tight', 
           pad_inches=0.1)
```

### Interactive Plotly Version

```python
import plotly.graph_objects as go
import numpy as np

def create_interactive_radar():
    radar = SENCERadarChart()
    
    fig = go.Figure()
    
    for city, data in radar.city_data.items():
        values = data['values'] + [data['values'][0]]  # Close the loop
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=radar.domains + [radar.domains[0]],
            fill='toself',
            name=f"{city} (CVI={data['mean_cvi']:.2f})",
            line=dict(color=data['color'], width=2.5),
            marker=dict(size=8, symbol=data['marker'])
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0]
            )
        ),
        showlegend=True,
        title="Interactive SENCE Radar Chart",
        width=1000,
        height=800
    )
    
    fig.write_html('interactive_radar.html')
    return fig

# Create interactive version
fig = create_interactive_radar()
fig.show()
```

---

## 🔗 Integration Examples

### Jupyter Notebook Integration

```python
# In Jupyter Notebook
%matplotlib inline
import matplotlib.pyplot as plt
from sence_radar_visualization import SENCERadarChart

# Create inline visualization
radar = SENCERadarChart()
fig = radar.create_advanced_radar_chart(figsize=(14, 10))
plt.show()

# Display statistical report
print(radar.generate_statistical_report())

# Display data table
import pandas as pd
df = pd.read_csv('outputs/sence_vulnerability_data.csv')
display(df.pivot(index='City', columns='Domain', values='Normalized_Contribution'))
```

### Streamlit Dashboard

```python
# streamlit_app.py
import streamlit as st
from sence_radar_visualization import SENCERadarChart
import pandas as pd

st.set_page_config(page_title="SENCE Dashboard", layout="wide")

st.title("🌍 SENCE Framework Dashboard")
st.markdown("**Vulnerability Analysis for Niger Delta Petroleum Cities**")

# Sidebar controls
st.sidebar.header("Configuration")
selected_cities = st.sidebar.multiselect(
    "Select Cities",
    ['Port Harcourt', 'Warri', 'Bonny'],
    default=['Port Harcourt', 'Warri', 'Bonny']
)

# Generate visualization
radar = SENCERadarChart()

# Filter data
filtered_data = {city: data for city, data in radar.city_data.items() 
                if city in selected_cities}
radar.city_data = filtered_data

# Display radar chart
col1, col2 = st.columns([2, 1])

with col1:
    fig = radar.create_advanced_radar_chart()
    st.pyplot(fig)

with col2:
    st.subheader("Statistics")
    for city in selected_cities:
        cvi = radar.city_data[city]['mean_cvi']
        st.metric(city, f"{cvi:.3f}", delta=None)

# Data table
st.subheader("Vulnerability Data")
df = pd.read_csv('outputs/sence_vulnerability_data.csv')
st.dataframe(df[df['City'].isin(selected_cities)])

# Run with: streamlit run streamlit_app.py
```

### Flask Web API

```python
# flask_api.py
from flask import Flask, send_file, jsonify
from sence_radar_visualization import SENCERadarChart
import io

app = Flask(__name__)
radar = SENCERadarChart()

@app.route('/api/radar-chart')
def get_radar_chart():
    """Return radar chart as PNG"""
    fig = radar.create_advanced_radar_chart()
    img = io.BytesIO()
    fig.savefig(img, format='png', dpi=300)
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.route('/api/statistics')
def get_statistics():
    """Return statistical data as JSON"""
    stats = {}
    for city, data in radar.city_data.items():
        stats[city] = {
            'mean_cvi': data['mean_cvi'],
            'domains': dict(zip(radar.domains, data['values']))
        }
    return jsonify(stats)

@app.route('/api/report')
def get_report():
    """Return statistical report"""
    report = radar.generate_statistical_report()
    return report, 200, {'Content-Type': 'text/plain; charset=utf-8'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# Run with: python flask_api.py
# Access at: http://localhost:5000/api/radar-chart
```

---

## 🔧 Troubleshooting

### Common Issues

**Issue 1: Import Errors**

```bash
# Error: ModuleNotFoundError: No module named 'matplotlib'
# Solution:
pip install matplotlib numpy pandas scipy seaborn
```

**Issue 2: Font Warnings**

```python
# Warning: findfont: Font family 'Times New Roman' not found
# Solution: Install fonts or use default
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Serif'
```

**Issue 3: Memory Errors with Large Figures**

```python
# Reduce DPI for testing
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
```

**Issue 4: PlantUML Memory Limit**

```bash
# Increase memory allocation
java -Xmx2048m -jar plantuml.jar sence_system_architecture.puml
```

**Issue 5: Mermaid Diagram Too Large**

```bash
# Set environment variable
export MERMAID_MAX_WIDTH=8000
export MERMAID_MAX_HEIGHT=8000
mmdc -i sence_framework.mmd -o output.png
```

### Performance Optimization

```python
# For large datasets, use data decimation
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000

# Disable interactive backend for batch processing
plt.ioff()

# Clear figures after saving to free memory
import matplotlib.pyplot as plt
fig = radar.create_advanced_radar_chart()
fig.savefig('output.png')
plt.close(fig)
```

---

## 📖 Further Resources

- **Python Documentation**: See docstrings in `sence_radar_visualization.py`
- **Mermaid Syntax**: https://mermaid.js.org/intro/
- **PlantUML Guide**: https://plantuml.com/guide
- **Matplotlib Gallery**: https://matplotlib.org/stable/gallery/

---

## 📧 Support

For issues or questions:
- Review the README.md
- Check GitHub Issues (if applicable)
- Contact: research@sence-framework.org

---

**Last Updated**: October 3, 2025  
**Version**: 1.0.0
