# Pillar 3: Numerical Modeling Dataset - Model Geometry and Mesh

## Finite Element Model Setup

### 1. Specimen Geometry
- **Type**: Cylindrical concrete specimen
- **Dimensions**: 
  - Diameter: 100 mm
  - Height: 200 mm
  - Volume: 1,570,796 mm³
- **Coordinate System**: 
  - Origin at specimen center
  - Z-axis: vertical (height direction)
  - R-axis: radial direction

### 2. Mesh Configuration
- **Element Type**: 8-node linear brick elements (C3D8R)
- **Mesh Density**: 
  - Radial direction: 20 elements (5 mm each)
  - Circumferential direction: 24 elements (15° each)
  - Axial direction: 40 elements (5 mm each)
- **Total Elements**: 19,200
- **Total Nodes**: 20,160
- **Element Size**: 5 mm × 5 mm × 5 mm

### 3. Boundary Conditions
- **Thermal Boundary Conditions**:
  - Top surface: Exposed to fire (convection + radiation)
  - Bottom surface: Insulated
  - Side surface: Exposed to fire (convection + radiation)
- **Mechanical Boundary Conditions**:
  - Bottom surface: Fixed in all directions
  - Top surface: Free to expand/contract
  - Side surface: Free to expand/contract

### 4. Loading Conditions
- **Thermal Loading**: 
  - ISO 834 standard fire curve
  - Duration: 120 minutes
  - Peak temperature: 1200°C
- **Mechanical Loading**:
  - Self-weight only
  - No external mechanical loads

### 5. Analysis Type
- **Coupled Thermal-Stress Analysis**
- **Transient Analysis**
- **Time Step**: 1 second
- **Total Time**: 7200 seconds (120 minutes)

### 6. Convergence Criteria
- **Thermal**: 1×10⁻⁶ W/m²
- **Mechanical**: 1×10⁻⁶ N
- **Maximum Iterations**: 10 per increment

### 7. Mesh Quality Metrics
- **Aspect Ratio**: < 3.0
- **Skewness**: < 0.5
- **Orthogonality**: > 0.8
- **Jacobian**: > 0.1

### 8. Special Considerations
- **Refined Mesh**: Near boundaries for accurate thermal gradients
- **Transition Elements**: Gradual mesh coarsening toward center
- **Contact Elements**: For potential spalling simulation
- **Cohesive Elements**: For crack propagation modeling