# Retrofit Intervention & Lifecycle Data (Decision Engine)

This comprehensive dataset supports the **Dynamic Digital Twin Framework for Multi-Objective Building Retrofit Optimization**, integrating Real-Time IoT, Life-Cycle Assessment, and Deep Reinforcement Learning.

## 🎯 Purpose

This dataset serves as the "Decision Engine" for building retrofit optimization, providing:
- **Retrofit Measure Library**: Complete database of possible interventions
- **Life Cycle Inventory (LCI)**: Embodied and operational carbon data
- **Maintenance & Operational Data**: Costs, schedules, and degradation rates
- **Integration Data**: Measure interactions and optimization parameters

## 📁 Dataset Structure

```
retrofit_dataset/
├── measures/                          # Retrofit Measure Library
│   ├── retrofit_measures.json        # Complete database of interventions
│   └── cost_data.json               # CAPEX and OPEX cost data
├── lci/                              # Life Cycle Inventory Database
│   ├── embodied_carbon.json         # Cradle-to-grave GWP data
│   └── operational_carbon.json      # Energy use and carbon intensity
├── maintenance/                      # Maintenance & Operational Data
│   ├── maintenance_schedules.json   # Scheduled maintenance costs
│   └── degradation_rates.json       # System performance degradation
├── integration/                      # Integration Data
│   ├── measure_interactions.json    # How measures interact
│   └── optimization_parameters.json # Optimization model parameters
├── scripts/                          # Data Processing Scripts
│   └── data_validation.py           # Dataset validation script
└── README.md                         # This file
```

## 🔧 Retrofit Measures

### Envelope Measures
- **Insulation**: Roof, wall, and floor insulation options
- **Windows**: Double-pane and triple-pane replacements
- **Air Sealing**: Comprehensive air leakage reduction

### HVAC Measures
- **Heating**: Air-source and ground-source heat pumps
- **Cooling**: Variable refrigerant flow systems
- **Ventilation**: Energy recovery ventilators

### Renewable Energy
- **Solar PV**: 10kW and 50kW rooftop systems
- **Solar Thermal**: Domestic hot water heating

### Lighting & Controls
- **LED Retrofit**: Comprehensive lighting upgrades
- **BMS**: Smart building management systems

## 📊 Data Sources

- **Ecoinvent 3.8**: Life cycle inventory data
- **ICE Database**: Carbon and energy data
- **Environmental Product Declarations (EPDs)**
- **ASHRAE Standards**: Building performance guidelines
- **DOE/NREL**: Energy efficiency data
- **Industry Reports**: Cost and performance data

## 💰 Cost Data

### CAPEX Costs
- Material costs per square foot
- Labor costs per square foot
- Regional cost multipliers
- Building type multipliers
- Installation complexity factors

### OPEX Costs
- Annual energy savings
- Maintenance costs
- Operational expenses
- Carbon footprint of maintenance

## 🌱 Life Cycle Assessment

### Embodied Carbon (A1-A5, B, C, D)
- **Materials**: Insulation, windows, HVAC equipment
- **Systems**: Complete system carbon footprints
- **Transportation**: A4 transport emissions
- **Installation**: A5 installation emissions
- **Use Phase**: B maintenance and operation
- **End of Life**: C disposal and recycling
- **Benefits**: D recycling and recovery

### Operational Carbon
- Energy carbon intensity by region
- Building energy use patterns
- Carbon intensity trends
- Maintenance carbon footprints

## 🔄 Maintenance & Degradation

### Maintenance Schedules
- **HVAC Systems**: Filter replacement, coil cleaning, inspections
- **Envelope Systems**: Insulation inspection, window maintenance
- **Renewable Energy**: Panel cleaning, inverter inspection
- **Lighting Systems**: Fixture cleaning and replacement
- **BMS Systems**: Software updates, sensor calibration

### Performance Degradation
- **Efficiency degradation** per year
- **Capacity degradation** per year
- **Reliability degradation** per year
- **Lifespan data** and replacement thresholds

## 🔗 Measure Interactions

### Synergistic Effects
- **Insulation + Air Sealing**: 15% synergy factor
- **Windows + Insulation**: 10% synergy factor
- **HVAC + Controls**: 25% synergy factor
- **Renewable + HVAC**: 20% synergy factor

### Conflicting Effects
- **Insulation + Ventilation**: 5% conflict factor
- **Solar + Shading**: 2% conflict factor

### Sequential Dependencies
- **Envelope before HVAC**: 10% efficiency improvement
- **Controls after Systems**: 15% efficiency improvement

## 🎯 Optimization Parameters

### Objective Functions
- **Energy Efficiency**: 30% weight
- **Carbon Emissions**: 25% weight
- **Lifecycle Cost**: 20% weight
- **Payback Period**: 15% weight
- **Indoor Comfort**: 10% weight

### Constraints
- **Budget**: Total and category limits
- **Technical**: Performance thresholds
- **Regulatory**: Code compliance
- **Physical**: Area and capacity limits

### Reinforcement Learning
- **State Space**: Building characteristics, current measures, environmental factors
- **Action Space**: Measure selection, combination, sequence
- **Reward Function**: Multi-objective optimization rewards

## 🚀 Usage

### Data Validation
```bash
python scripts/data_validation.py
```

### Integration with Optimization Models
```python
import json

# Load retrofit measures
with open('measures/retrofit_measures.json', 'r') as f:
    measures = json.load(f)

# Load cost data
with open('measures/cost_data.json', 'r') as f:
    costs = json.load(f)

# Load carbon data
with open('lci/embodied_carbon.json', 'r') as f:
    carbon = json.load(f)
```

### Deep Reinforcement Learning
The dataset is designed to work with:
- **Multi-Objective Genetic Algorithms**
- **Deep Q-Networks (DQN)**
- **Policy Gradient Methods**
- **Actor-Critic Methods**

## 📈 Performance Metrics

### Energy Metrics
- Site Energy Use Intensity (EUI)
- Source Energy Use Intensity
- Renewable Energy Fraction
- Energy Cost Intensity

### Carbon Metrics
- Operational Carbon Intensity
- Embodied Carbon Intensity
- Lifecycle Carbon Intensity
- Carbon Payback Period

### Economic Metrics
- Net Present Value (NPV)
- Internal Rate of Return (IRR)
- Payback Period
- Lifecycle Cost

### Comfort Metrics
- Thermal Comfort Index
- Indoor Air Quality Index
- Acoustic Comfort Index
- Visual Comfort Index

## 🔍 Data Quality

### Validation Criteria
- **Energy Model**: CVRMSE < 15%, NMBE < 5%, R² > 0.85
- **Cost Model**: Accuracy > 80%, Completeness > 95%
- **Carbon Model**: Accuracy > 75%, Completeness > 90%

### Data Completeness
- **Measures**: 15 retrofit measures across 5 categories
- **Costs**: Complete CAPEX and OPEX data
- **Carbon**: Full lifecycle carbon footprints
- **Maintenance**: Comprehensive schedules and degradation rates

## 📚 References

1. ASHRAE Standard 90.1-2019: Energy Standard for Buildings
2. ISO 14040:2006: Environmental Management - Life Cycle Assessment
3. ISO 14044:2006: Environmental Management - Life Cycle Assessment
4. Ecoinvent 3.8 Database: Life Cycle Inventory Data
5. ICE Database: Inventory of Carbon & Energy
6. NREL: National Renewable Energy Laboratory
7. DOE: Department of Energy Building Technologies Office

## 🤝 Contributing

This dataset is designed to be:
- **Extensible**: Easy to add new measures and data
- **Validated**: Comprehensive data quality checks
- **Documented**: Clear structure and metadata
- **Standardized**: Consistent data formats

## 📄 License

This dataset is provided for research and educational purposes. Please cite appropriately when using this data in publications.

## 📞 Contact

For questions about this dataset or the Dynamic Digital Twin Framework, please refer to the research documentation or contact the development team.

---

**Version**: 1.0  
**Last Updated**: 2024-01-15  
**Status**: Ready for Use  
**Validation**: Pending
