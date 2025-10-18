# Technical Specification: Thermo-Mechanical Modeling Dataset
## Fire-Resistant Structural Elements with High-Performance Rubberized Concrete

### 1. Dataset Architecture

#### 1.1 Data Structure
- **Format**: CSV files with standardized column headers
- **Encoding**: UTF-8
- **Delimiter**: Comma (,)
- **Missing Values**: None (complete dataset)
- **Data Types**: Float64 for numerical values, String for categorical

#### 1.2 Identifiers
- **Mix_ID**: Concrete mix identifier (C, R5S, R10S, R15S, R20S, R10L)
- **Temperature_C**: Temperature in Celsius (20°C to 800°C, 5°C increments)
- **Data_Type**: Calibration or Validation
- **Property_Type**: Thermal, Mechanical, Transport, Deformation

### 2. Material Property Specifications

#### 2.1 Thermal Properties

| Property | Symbol | Unit | Range | Temperature Dependency | CV |
|----------|--------|------|-------|----------------------|-----|
| Thermal Conductivity | k | W/m·K | 1.2-2.1 | Exponential decay | 5% |
| Specific Heat | cp | J/kg·K | 900-1200 | Linear increase | 8% |
| Thermal Expansion | α | 1/K | 12×10⁻⁶-18×10⁻⁶ | Linear increase | 10% |
| Density | ρ | kg/m³ | 2200-2400 | Linear decrease | 3% |

**Mathematical Models:**
```
k(T) = k₀ × (1 - 0.4 × (T-20)/780) × (1 - 0.3 × R/100)
cp(T) = cp₀ × (1 + 0.3 × (T-20)/780) × (1 + 0.05 × R/100)
α(T) = α₀ × (1 + 0.5 × (T-20)/780) × (1 + 0.17 × R/100)
ρ(T) = ρ₀ × (1 - 0.05 × (T-20)/780) × (1 - 0.04 × R/100)
```
Where: T = Temperature (°C), R = Rubber content (%), Subscript 0 = Reference value at 20°C

#### 2.2 Mechanical Properties

| Property | Symbol | Unit | Range | Temperature Dependency | CV |
|----------|--------|------|-------|----------------------|-----|
| Compressive Strength | fc | MPa | 5-45 | Linear decay | 12% |
| Elastic Modulus | E | GPa | 2-35 | Linear decay | 15% |
| Tensile Strength | ft | MPa | 0.5-3.5 | Linear decay | 18% |
| Poisson's Ratio | ν | - | 0.18-0.35 | Linear increase | 8% |
| Fracture Energy | Gf | N/m | 20-150 | Linear decay | 20% |

**Mathematical Models:**
```
fc(T) = max(fc₀ × (1 - 0.6 × (T-20)/780) × (1 - 0.11 × R/100), 5.0)
E(T) = max(E₀ × (1 - 0.7 × (T-20)/780) × (1 - 0.09 × R/100), 2.0)
ft(T) = max(ft₀ × (1 - 0.8 × (T-20)/780) × (1 - 0.09 × R/100), 0.5)
ν(T) = min(ν₀ × (1 + 0.2 × (T-20)/780) × (1 + 0.11 × R/100), 0.35)
Gf(T) = max(Gf₀ × (1 - 0.5 × (T-20)/780) × (1 - 0.07 × R/100), 20.0)
```

#### 2.3 Transport Properties

| Property | Symbol | Unit | Range | Temperature Dependency | CV |
|----------|--------|------|-------|----------------------|-----|
| Porosity | φ | - | 0.15-0.35 | Linear increase | 12% |
| Permeability | k | m² | 1×10⁻¹⁶-3×10⁻¹⁵ | Exponential increase | 25% |
| Water Diffusivity | Dw | m²/s | 1×10⁻¹⁰-3×10⁻¹⁰ | Exponential increase | 30% |
| Vapor Diffusivity | Dv | m²/s | 1×10⁻⁸-3×10⁻⁸ | Exponential increase | 25% |

**Mathematical Models:**
```
φ(T) = min(φ₀ × (1 + 0.3 × (T-20)/780) × (1 + 0.13 × R/100), 0.35)
k(T) = k₀ × (1 + 2.0 × (T-20)/780) × (1 + 0.5 × R/100)
Dw(T) = Dw₀ × (1 + 1.5 × (T-20)/780) × (1 + 0.2 × R/100)
Dv(T) = Dv₀ × (1 + 2.0 × (T-20)/780) × (1 + 0.1 × R/100)
```

#### 2.4 Deformation Properties

| Property | Symbol | Unit | Range | Temperature Dependency | CV |
|----------|--------|------|-------|----------------------|-----|
| Creep Coefficient | φc | - | 2.0-3.5 | Linear increase | 15% |
| Shrinkage Strain | εsh | - | 300×10⁻⁶-600×10⁻⁶ | Linear increase | 20% |
| Thermal Strain | εth | - | 0-0.01 | Linear with T | 10% |
| Creep Modulus | Ec | GPa | 3-30 | Linear decay | 18% |

**Mathematical Models:**
```
φc(T) = φc₀ × (1 + 0.4 × (T-20)/780) × (1 + 0.05 × R/100)
εsh(T) = εsh₀ × (1 + 0.6 × (T-20)/780) × (1 + 0.17 × R/100)
εth(T) = α(T) × (T - 20)
Ec(T) = max(Ec₀ × (1 - 0.5 × (T-20)/780) × (1 - 0.07 × R/100), 3.0)
```

### 3. Stress-Strain Curve Specifications

#### 3.1 Hognestad Model Implementation
```
σ(ε) = {
    fc × (2ε/ε₀ - (ε/ε₀)²)           if ε ≤ ε₀
    fc × (1 - 0.15 × (ε-ε₀)/(εu-ε₀)) if ε > ε₀
}
```
Where:
- ε₀ = 2fc/E (strain at peak stress)
- εu = 0.01 (ultimate strain)
- fc = Temperature-dependent compressive strength
- E = Temperature-dependent elastic modulus

#### 3.2 Curve Parameters
- **Strain Points**: 100 points from 0 to 0.01
- **Temperature Increments**: Every 50°C (20°C, 70°C, 120°C, ..., 800°C)
- **Mix Coverage**: All 6 concrete mixes
- **Data Format**: JSON with nested structure

### 4. Statistical Specifications

#### 4.1 Uncertainty Quantification
- **Method**: Monte Carlo simulation with normal distribution
- **Confidence Levels**: 95% and 99% bounds provided
- **Sample Size**: 1000 realizations per property
- **Correlation**: Properties within same mix are correlated

#### 4.2 Coefficient of Variation (CV)
- **Thermal Properties**: 3-10% (low uncertainty)
- **Mechanical Properties**: 12-20% (moderate uncertainty)
- **Transport Properties**: 25-30% (high uncertainty)
- **Deformation Properties**: 15-20% (moderate uncertainty)

#### 4.3 Calibration-Validation Split
- **Calibration**: 70% of data points
- **Validation**: 30% of data points
- **Random Assignment**: Stratified by temperature and mix
- **Independence**: No overlap between calibration and validation sets

### 5. FEA Software Integration

#### 5.1 ABAQUS Integration
```python
# Material definition example
*MATERIAL, NAME=CONCRETE_R10S
*ELASTIC
*THERMAL CONDUCTIVITY
*SPECIFIC HEAT
*DENSITY
*EXPANSION
*PERMEABILITY
*CREEP
*DAMAGE INITIATION, CRITERION=MAXIMUM PRINCIPAL STRESS
*DAMAGE EVOLUTION, TYPE=ENERGY
```

#### 5.2 ANSYS Integration
```apdl
! Material property definition
MP,EX,1,35e9          ! Elastic modulus
MP,PRXY,1,0.18        ! Poisson's ratio
MP,DENS,1,2400        ! Density
MP,KXX,1,2.1          ! Thermal conductivity
MP,C,1,900            ! Specific heat
MP,ALPX,1,12e-6       ! Thermal expansion
```

#### 5.3 COMSOL Integration
```matlab
% Material property expressions
E = 35e9;                    % Elastic modulus
nu = 0.18;                   % Poisson's ratio
rho = 2400;                  % Density
k = 2.1;                     % Thermal conductivity
cp = 900;                    % Specific heat
alpha = 12e-6;               % Thermal expansion
```

### 6. Quality Assurance Protocols

#### 6.1 Physical Consistency Checks
1. **Thermodynamic Compatibility**: Energy conservation in coupled models
2. **Mass Conservation**: Continuity in transport equations
3. **Stress-Strain Validity**: Positive definite stiffness matrix
4. **Temperature Monotonicity**: Property changes are physically reasonable

#### 6.2 Statistical Validation
1. **Normal Distribution**: Shapiro-Wilk test (p > 0.05)
2. **Temperature Smoothness**: First derivative continuity
3. **Mix Relationships**: Logical ordering of properties with rubber content
4. **Range Validation**: All values within physically reasonable bounds

#### 6.3 Cross-Validation
1. **Leave-One-Out**: Each temperature point validated independently
2. **Mix Cross-Validation**: Each mix validated against others
3. **Property Cross-Validation**: Interdependent properties validated together
4. **Temperature Cross-Validation**: Smooth transitions between temperature points

### 7. Data Access and Usage

#### 7.1 File Formats
- **Primary**: CSV files for easy data manipulation
- **Secondary**: Excel files for visualization
- **Tertiary**: JSON files for stress-strain curves
- **FEA Ready**: Software-specific input files

#### 7.2 API Functions
```python
def get_property(mix_id, property_name, temperature, data_type='all'):
    """Get material property at specific temperature"""
    
def get_property_range(mix_id, property_name, temp_range, data_type='all'):
    """Get material property over temperature range"""
    
def get_stress_strain_curve(mix_id, temperature):
    """Get complete stress-strain curve"""
    
def get_uncertainty_bounds(mix_id, property_name, temperature, confidence=0.95):
    """Get statistical uncertainty bounds"""
```

#### 7.3 Validation Functions
```python
def validate_physical_consistency(data):
    """Validate physical consistency of dataset"""
    
def validate_statistical_properties(data):
    """Validate statistical properties of dataset"""
    
def validate_fea_compatibility(data, software='abaqus'):
    """Validate FEA software compatibility"""
```

### 8. Performance Specifications

#### 8.1 Computational Requirements
- **Memory**: 50 MB for complete dataset
- **Processing**: < 1 second for property lookup
- **Storage**: 100 MB total (including all formats)
- **Compatibility**: Python 3.7+, MATLAB R2018b+, ANSYS 19.0+, ABAQUS 2018+

#### 8.2 Scalability
- **Temperature Points**: Easily extensible to 1°C increments
- **Mix Variations**: Additional mixes can be added
- **Property Types**: New properties can be integrated
- **Software Support**: Additional FEA software can be added

### 9. Documentation Standards

#### 9.1 Code Documentation
- **Docstrings**: All functions documented with NumPy style
- **Comments**: Inline comments for complex calculations
- **Type Hints**: Full type annotation for all functions
- **Examples**: Usage examples for all major functions

#### 9.2 Data Documentation
- **Units**: All properties clearly labeled with units
- **Ranges**: Valid ranges specified for all properties
- **Dependencies**: Interdependencies clearly documented
- **Assumptions**: Physical assumptions explicitly stated

#### 9.3 User Documentation
- **README**: Comprehensive usage guide
- **Tutorials**: Step-by-step implementation examples
- **FAQ**: Common questions and answers
- **Troubleshooting**: Common issues and solutions

### 10. Version Control and Updates

#### 10.1 Versioning Scheme
- **Major**: Significant structural changes
- **Minor**: New properties or mixes added
- **Patch**: Bug fixes or documentation updates

#### 10.2 Update Protocol
1. **Validation**: All changes validated against physical constraints
2. **Testing**: Comprehensive test suite execution
3. **Documentation**: Updated documentation with changes
4. **Backward Compatibility**: Maintained where possible

#### 10.3 Change Log
- **v1.0.0**: Initial dataset release
- **Future versions**: Tracked in CHANGELOG.md

---

**Document Version**: 1.0  
**Last Updated**: [Current Date]  
**Next Review**: [Date + 1 year]  
**Approved By**: [Research Team]