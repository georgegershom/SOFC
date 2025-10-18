# FEA Software Material Definitions Usage Guide

## ABAQUS (.inp file)
1. Copy the contents of `rubberized_concrete_materials.inp` into your ABAQUS input file
2. Reference materials in your model using names like `C_CONCRETE`, `R5S_CONCRETE`, etc.
3. Ensure temperature-dependent analysis is enabled: `*STEP, INC=100, NLGEOM=YES`

## ANSYS (.mac file)  
1. Load the macro file: `/INPUT,rubberized_concrete_materials,mac`
2. Materials are numbered 1-6 corresponding to C, R5S, R10S, R15S, R20S, R10L
3. Use `MAT,1` to `MAT,6` to assign materials to elements
4. Enable temperature-dependent analysis: `TREF,20` (reference temperature)

## COMSOL (.txt file)
1. Copy the function definitions into COMSOL material property fields
2. Use functions like `E_C(T)`, `nu_R5S(T)`, etc. in material property expressions
3. Ensure temperature variable `T` is properly defined in your physics
4. Units are consistent: Pa for modulus, kg/m³ for density, etc.

## Temperature Range
All definitions are valid from 20°C to 800°C with 10°C increments.

## Material Naming Convention
- C: Control concrete (0% rubber)
- R5S: 5% small rubber particles
- R10S: 10% small rubber particles  
- R15S: 15% small rubber particles
- R20S: 20% small rubber particles
- R10L: 10% large rubber particles
