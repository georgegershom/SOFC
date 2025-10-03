# Petroleum Cities Vulnerability System Model

## Advanced Systems Dynamics Implementation of Reinforcing Feedback Loops

This repository contains a sophisticated Python implementation of the **Systems Map of Dominant Reinforcing Feedback Loops in Petroleum Cities**, based on the SENCE (Systemic, Environmental, Contextual Embeddedness) framework for analyzing compound vulnerabilities in Nigeria's petroleum cities (Port Harcourt, Warri, and Bonny).

## 🎯 Overview

The model demonstrates how environmental, socio-economic, and governance factors interact dynamically to amplify compound risks through three primary reinforcing feedback loops:

1. **Loop R1: Livelihood-Environment Degradation** - Environmental shocks cascade through ecosystem damage to livelihood destruction
2. **Loop R2: Governance Failure** - Institutional failures erode public trust and weaken governance capacity
3. **Loop R3: Economic Diversification Failure** - Oil sector dominance creates mono-economic vulnerability

## 🚀 Features

### Core System Model (`petroleum_cities_system_model.py`)
- **23 interconnected nodes** representing key system components
- **34 directional edges** with weighted relationships
- **8 identified feedback cycles** creating multiplicative vulnerability
- **Dynamic simulation** using differential equations
- **Real-time vulnerability tracking** across multiple dimensions

### Advanced Analysis (`advanced_analysis.py`)
- **Monte Carlo simulations** (100+ iterations) for uncertainty quantification
- **Sensitivity analysis** on critical parameters
- **Policy intervention scenarios** with comparative impact assessment
- **3D phase space visualization** of system evolution
- **Resilience pathway analysis** with strategic options

### Visualization Capabilities
- **Interactive network maps** showing all feedback loops and connections
- **Multi-panel dashboards** with 9+ analytical views
- **Time-series evolution** of vulnerability indices
- **Heat maps** of system states over time
- **Radar charts** for resilience profiling

## 📊 Key Metrics & Findings

Based on the research data:
- **Oil spill correlation with poverty**: r = 0.78
- **Economic concentration (HHI)**: 0.72 (high mono-dependency)
- **Governance trust level**: 0.28 (critically low)
- **System-wide vulnerability**: 93.9% mean CVI

## 🛠️ Installation

```bash
# Clone the repository
git clone <repository-url>
cd petroleum-cities-model

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- numpy>=1.24.0
- pandas>=2.0.0
- plotly>=5.14.0
- networkx>=3.0
- scipy>=1.10.0

## 💻 Usage

### Basic Model Execution
```python
python3 petroleum_cities_system_model.py
```

This generates:
- `petroleum_cities_network.html` - Interactive network visualization
- `petroleum_cities_dashboard.html` - Comprehensive analytics dashboard
- `petroleum_cities_analysis_report.txt` - Detailed analysis report

### Advanced Analysis
```python
python3 advanced_analysis.py
```

This produces:
- `sensitivity_analysis.html` - Parameter sensitivity charts
- `policy_scenarios.html` - Intervention comparison
- `3d_phase_space.html` - 3D system trajectory
- `resilience_pathways.html` - Strategic resilience options
- `executive_summary.json` - Key findings and recommendations

### Programmatic Usage
```python
from petroleum_cities_system_model import PetroleumCitySystemModel

# Initialize model
model = PetroleumCitySystemModel()

# Run simulation with custom shocks
shock_scenarios = {
    'oil_spills': (2.0, 0.3),  # (time, magnitude)
    'governance_weakness': (5.0, 0.25)
}
df = model.simulate_dynamics(time_steps=200, shock_scenarios=shock_scenarios)

# Generate visualizations
network_fig = model.create_network_visualization()
dashboard_fig = model.create_simulation_dashboard()

# Calculate metrics
metrics = model.calculate_metrics()
```

## 📈 Model Architecture

### System Nodes (23 total)
```
Environmental: Oil Spills, Ecosystem Damage, Toxic Waste, Climate Vulnerability
Economic: Oil Dominance, Mono-Economy, Shock Vulnerability, Crowding Out
Social: Social Unrest, Migration Pressure, Trust Erosion, Informal Systems
Governance: Institutional Failure, Service Failure, Governance Weakness
Adaptive: Artisanal Refining, Rentier Mentality
```

### Edge Types & Dynamics
- **Direct**: Immediate causal relationships (weight: 0.7-0.9)
- **Reinforcing**: Positive feedback amplification (×1.5 multiplier)
- **Cascade**: Sequential failure propagation (×1.2 multiplier)
- **Adaptive**: Coping mechanism emergence
- **Systemic**: Cross-domain interactions

### Mathematical Framework

The system dynamics are governed by coupled differential equations:

```
dy_i/dt = -λ_i(1 - ρ_i)y_i + Σ_j w_ij * f(type_ij) * y_j + S_i(t)
```

Where:
- `y_i`: State of node i
- `λ_i`: Decay rate
- `ρ_i`: Resilience capacity
- `w_ij`: Edge weight from j to i
- `f(type_ij)`: Amplification factor by edge type
- `S_i(t)`: External shock function

## 🎨 Visualization Examples

### Network Map Features
- **Node size**: Proportional to vulnerability score
- **Node color**: Heat map from green (low risk) to red (high risk)
- **Edge width**: Represents relationship strength
- **Edge style**: Different patterns for relationship types
- **Annotations**: Highlight three primary loops (R1, R2, R3)

### Dashboard Components
1. **CVI Evolution**: Tracks composite vulnerability over time
2. **Loop Intensity**: Monitors feedback loop strength
3. **Phase Portrait**: Shows system trajectory in state space
4. **Environmental Cascade**: Pollution → ecosystem → livelihood chain
5. **Trust Erosion**: Governance legitimacy decline
6. **Economic Risk**: Mono-economy vulnerability
7. **Intervention Scenarios**: Policy impact comparison
8. **Vulnerability Heatmap**: All nodes over time
9. **Resilience Radar**: Multi-dimensional capacity profile

## 📋 Key Insights

### Critical Findings
1. **Multiplicative Vulnerability**: Risks amplify exponentially through feedback loops
2. **Lock-in Effects**: Path dependencies resist conventional interventions
3. **Cascade Potential**: Localized shocks trigger system-wide failures
4. **Governance Criticality**: Trust erosion mediates all vulnerability domains
5. **Adaptation Paradox**: Coping strategies amplify long-term risks

### Intervention Recommendations

**Immediate (0-6 months)**:
- Emergency ecosystem restoration
- Trust-building mechanisms
- Economic shock absorbers

**Medium-term (6-24 months)**:
- Economic diversification incentives
- Governance capacity building
- Alternative livelihood creation

**Long-term (2-10 years)**:
- Systemic oil dependency reduction
- Polycentric governance structures
- Anticipatory risk frameworks

## 🔬 Research Foundation

This model is based on empirical research from Nigeria's petroleum cities, incorporating:
- Composite Vulnerability Index (CVI) measurements
- Correlation analysis (PCA) of risk factors
- Qualitative insights from stakeholder interviews
- Historical data on oil spills and economic indicators
- Governance assessment metrics

## 📚 References

- SENCE Framework for compound vulnerability assessment
- Systems dynamics modeling of coupled human-environment systems
- Reinforcing feedback loop theory in complex adaptive systems
- Nigerian petroleum sector vulnerability studies

## 🤝 Contributing

This model is designed for research and policy analysis. Contributions welcome for:
- Additional vulnerability indicators
- New intervention scenarios
- Regional adaptations
- Validation with empirical data

## 📄 License

This implementation is provided for academic and policy research purposes.

## 🏆 Acknowledgments

Model developed based on research into petroleum cities vulnerability in Nigeria, implementing the SENCE framework for understanding multiplicative risk dynamics in coupled human-environment systems.

---

**Note**: This is a sophisticated simulation model demonstrating complex system dynamics. The high vulnerability scores reflect the critical nature of compound risks in petroleum-dependent cities and the urgent need for systemic interventions.