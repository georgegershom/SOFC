# High-Temperature Experimental Dataset Structure
## Pillar 2: High-Temperature Experimental Investigation

### Dataset Organization
```
experimental_dataset/
├── thermal_properties/
│   ├── tga_dsc/
│   ├── thermal_conductivity/
│   ├── specific_heat/
│   ├── cte/
│   └── mass_loss/
├── mechanical_testing/
│   ├── tts_curves/
│   ├── stt_tests/
│   └── residual_properties/
├── spalling_durability/
│   ├── visual_audio/
│   ├── vapor_pressure/
│   ├── gas_permeability/
│   └── microstructural/
└── metadata/
    ├── sample_specifications.json
    ├── test_protocols.json
    └── data_dictionary.json
```

### Sample Mixes
1. **Control Mix**: 100% Natural Aggregate (NA)
2. **10% Rubber**: 10% Crumb Rubber (CR) replacement
3. **20% Rubber**: 20% Crumb Rubber (CR) replacement
4. **30% Rubber**: 30% Crumb Rubber (CR) replacement
5. **Raw Rubber**: Pure crumb rubber samples

### Test Temperatures
- Ambient: 25°C
- Low: 100°C, 200°C
- Medium: 400°C, 600°C
- High: 800°C, 1000°C