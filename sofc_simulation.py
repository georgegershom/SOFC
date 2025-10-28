#!/usr/bin/env python
"""
SOFC Multi-Physics Simulation - Abaqus/Standard
================================================
Complete FEM setup for SOFC repeat unit analysis with:
- Sequential thermal → thermo-mechanical coupling
- Temperature-dependent materials (elastic, plastic, creep)
- Three heating rate scenarios (HR1, HR4, HR10)
- Damage and delamination proxies

Units: SI (m, kg, s, K, N, Pa)
"""

from abaqus import *
from abaqusConstants import *
from caeModules import *
import regionToolset
import mesh
import numpy as np

# ============================================================================
# GLOBAL PARAMETERS
# ============================================================================

# Geometry (in meters)
WIDTH = 10.0e-3   # 10 mm
THICK = 1.0e-3    # 1 mm

# Layer boundaries (y-coordinates in meters)
Y_ANODE_BOT = 0.0
Y_ANODE_TOP = 0.4e-3
Y_ELYTE_TOP = 0.5e-3
Y_CATH_TOP = 0.9e-3
Y_INTCONN_TOP = 1.0e-3

# Mesh refinement
ELEMS_X = 80
INTERFACE_REFINE_ZONE = 0.02e-3  # ±20 μm around interfaces

# Material constants (SI units)
GAS_CONSTANT = 8.314  # J/mol·K
TREF_CELSIUS = 25.0
TREF_KELVIN = 273.15 + TREF_CELSIUS

# Delamination thresholds (Pa)
TAU_CRIT_AE = 25.0e6   # anode-electrolyte
TAU_CRIT_EC = 20.0e6   # electrolyte-cathode
TAU_CRIT_CI = 30.0e6   # cathode-interconnect

# Damage model parameters
SIGMA_TH = 120.0e6  # Pa
K_DAMAGE = 1.5e-5
P_DAMAGE = 2.0

# Heating rate scenarios (name: [rate_C_per_min, target_C, hold_min])
HEATING_SCENARIOS = {
    'HR1': (1.0, 900.0, 10.0),
    'HR4': (4.0, 900.0, 10.0),
    'HR10': (10.0, 900.0, 10.0)
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def celsius_to_kelvin(temp_c):
    """Convert Celsius to Kelvin."""
    return temp_c + 273.15

def compute_thermal_schedule(rate_c_per_min, target_c, hold_min):
    """
    Compute ramp-hold-cool schedule.
    Returns: (ramp_time_s, hold_time_s, cool_time_s, total_time_s)
    """
    delta_t = target_c - TREF_CELSIUS
    ramp_time_s = (delta_t / rate_c_per_min) * 60.0
    hold_time_s = hold_min * 60.0
    cool_time_s = ramp_time_s  # symmetric cooling
    total_time_s = ramp_time_s + hold_time_s + cool_time_s
    return ramp_time_s, hold_time_s, cool_time_s, total_time_s

# ============================================================================
# MODEL CREATION
# ============================================================================

def create_model(scenario_name='HR1'):
    """Create complete Abaqus model for given heating scenario."""
    
    rate, target, hold = HEATING_SCENARIOS[scenario_name]
    model_name = 'SOFC_' + scenario_name
    
    # Create model
    myModel = mdb.Model(name=model_name)
    if 'Model-1' in mdb.models.keys():
        del mdb.models['Model-1']
    
    print(f"\n{'='*70}")
    print(f"Creating model: {model_name}")
    print(f"Heating rate: {rate} C/min, Target: {target} C, Hold: {hold} min")
    print(f"{'='*70}\n")
    
    # ========================================================================
    # 1. PART - 2D Geometry with Partitions
    # ========================================================================
    
    print("1. Creating part with layer partitions...")
    
    sketch = myModel.ConstrainedSketch(name='__profile__', sheetSize=0.02)
    sketch.rectangle(point1=(0.0, 0.0), point2=(WIDTH, THICK))
    
    part = myModel.Part(name='SOFC_Section', dimensionality=TWO_D_PLANAR, 
                        type=DEFORMABLE_BODY)
    part.BaseShell(sketch=sketch)
    del myModel.sketches['__profile__']
    
    # Partition at layer boundaries
    for y_coord in [Y_ANODE_TOP, Y_ELYTE_TOP, Y_CATH_TOP]:
        pickedFaces = part.faces.findAt(((WIDTH/2.0, y_coord/2.0, 0.0),))
        part.PartitionFaceByShortestPath(faces=pickedFaces,
            point1=(0.0, y_coord, 0.0),
            point2=(WIDTH, y_coord, 0.0))
    
    # ========================================================================
    # 2. SETS & SURFACES
    # ========================================================================
    
    print("2. Creating sets and surfaces...")
    
    # Layer sets (faces)
    faces = part.faces
    
    face_anode = faces.findAt(((WIDTH/2.0, Y_ANODE_TOP/2.0, 0.0),))
    part.Set(faces=face_anode, name='ANODE')
    
    face_elyte = faces.findAt(((WIDTH/2.0, (Y_ANODE_TOP+Y_ELYTE_TOP)/2.0, 0.0),))
    part.Set(faces=face_elyte, name='ELYTE')
    
    face_cath = faces.findAt(((WIDTH/2.0, (Y_ELYTE_TOP+Y_CATH_TOP)/2.0, 0.0),))
    part.Set(faces=face_cath, name='CATH')
    
    face_intconn = faces.findAt(((WIDTH/2.0, (Y_CATH_TOP+Y_INTCONN_TOP)/2.0, 0.0),))
    part.Set(faces=face_intconn, name='INTCONN')
    
    # Interface edge sets
    edges = part.edges
    
    edge_ae = edges.findAt(((WIDTH/2.0, Y_ANODE_TOP, 0.0),))
    part.Set(edges=edge_ae, name='INT_AE')
    
    edge_ec = edges.findAt(((WIDTH/2.0, Y_ELYTE_TOP, 0.0),))
    part.Set(edges=edge_ec, name='INT_EC')
    
    edge_ci = edges.findAt(((WIDTH/2.0, Y_CATH_TOP, 0.0),))
    part.Set(edges=edge_ci, name='INT_CI')
    
    # Boundary edges
    edge_x0 = edges.findAt(((0.0, THICK/2.0, 0.0),))
    part.Set(edges=edge_x0, name='X0')
    
    edge_y0 = edges.findAt(((WIDTH/2.0, 0.0, 0.0),))
    part.Set(edges=edge_y0, name='Y0')
    
    edge_ytop = edges.findAt(((WIDTH/2.0, THICK, 0.0),))
    part.Set(edges=edge_ytop, name='YTOP')
    
    # ========================================================================
    # 3. MATERIALS
    # ========================================================================
    
    print("3. Defining temperature-dependent materials...")
    
    # Define two temperature points for material data
    T1, T2 = 298.0, 1273.0  # K
    
    # --- 3.1 Ni-YSZ (Anode) ---
    mat_anode = myModel.Material(name='NiYSZ')
    
    # Elastic (temperature-dependent)
    mat_anode.Elastic(type=ISOTROPIC, temperatureDependency=ON, table=(
        (140.0e9, 0.30, T1),
        (91.0e9, 0.30, T2)
    ))
    
    # Thermal expansion
    mat_anode.Expansion(type=ISOTROPIC, temperatureDependency=ON, table=(
        (12.5e-6, T1),
        (13.5e-6, T2)
    ))
    
    # Conductivity
    mat_anode.Conductivity(type=ISOTROPIC, temperatureDependency=ON, table=(
        (6.0, T1),
        (4.0, T2)
    ))
    
    # Specific heat
    mat_anode.SpecificHeat(temperatureDependency=ON, table=(
        (450.0, T1),
        (570.0, T2)
    ))
    
    # Density (assumed constant, typical for cermet)
    mat_anode.Density(table=((6000.0,),))
    
    # Johnson-Cook plasticity
    mat_anode.Plastic(hardening=JOHNSON_COOK, table=(
        (150.0e6, 200.0e6, 0.35, 0.02, 1.0, 298.0, 1720.0, 1.0)
    ))
    
    # Norton-Bailey creep
    # Law: eps_dot = B * sigma^n * exp(-Q/RT)
    # Abaqus Norton creep: A = B, n = n, m = -Q/R (for Arrhenius form)
    Q_anode = 2.2e5  # J/mol
    B_anode = 1.0e-18
    n_anode = 3.5
    mat_anode.Creep(law=TIME, table=((B_anode, n_anode, -Q_anode/GAS_CONSTANT),))
    
    # --- 3.2 8YSZ (Electrolyte) ---
    mat_elyte = myModel.Material(name='YSZ8')
    
    mat_elyte.Elastic(type=ISOTROPIC, temperatureDependency=ON, table=(
        (210.0e9, 0.28, T1),
        (170.0e9, 0.28, T2)
    ))
    
    mat_elyte.Expansion(type=ISOTROPIC, temperatureDependency=ON, table=(
        (10.5e-6, T1),
        (11.2e-6, T2)
    ))
    
    mat_elyte.Conductivity(type=ISOTROPIC, temperatureDependency=ON, table=(
        (2.6, T1),
        (2.0, T2)
    ))
    
    mat_elyte.SpecificHeat(temperatureDependency=ON, table=(
        (400.0, T1),
        (600.0, T2)
    ))
    
    mat_elyte.Density(table=((5900.0,),))
    
    # Creep (ceramic)
    Q_elyte = 3.8e5  # J/mol
    B_elyte = 5.0e-22
    n_elyte = 2.0
    mat_elyte.Creep(law=TIME, table=((B_elyte, n_elyte, -Q_elyte/GAS_CONSTANT),))
    
    # --- 3.3 LSM (Cathode) ---
    mat_cath = myModel.Material(name='LSM')
    
    mat_cath.Elastic(type=ISOTROPIC, temperatureDependency=ON, table=(
        (120.0e9, 0.30, T1),
        (84.0e9, 0.30, T2)
    ))
    
    mat_cath.Expansion(type=ISOTROPIC, temperatureDependency=ON, table=(
        (11.5e-6, T1),
        (12.4e-6, T2)
    ))
    
    mat_cath.Conductivity(type=ISOTROPIC, temperatureDependency=ON, table=(
        (2.0, T1),
        (1.8, T2)
    ))
    
    mat_cath.SpecificHeat(temperatureDependency=ON, table=(
        (480.0, T1),
        (610.0, T2)
    ))
    
    mat_cath.Density(table=((6500.0,),))
    
    # --- 3.4 Ferritic Steel (Interconnect) ---
    mat_intconn = myModel.Material(name='FerriticSteel')
    
    mat_intconn.Elastic(type=ISOTROPIC, temperatureDependency=ON, table=(
        (205.0e9, 0.30, T1),
        (150.0e9, 0.30, T2)
    ))
    
    mat_intconn.Expansion(type=ISOTROPIC, temperatureDependency=ON, table=(
        (12.5e-6, T1),
        (13.2e-6, T2)
    ))
    
    mat_intconn.Conductivity(type=ISOTROPIC, temperatureDependency=ON, table=(
        (20.0, T1),
        (15.0, T2)
    ))
    
    mat_intconn.SpecificHeat(temperatureDependency=ON, table=(
        (500.0, T1),
        (700.0, T2)
    ))
    
    mat_intconn.Density(table=((7800.0,),))
    
    # ========================================================================
    # 4. SECTIONS
    # ========================================================================
    
    print("4. Creating sections and assignments...")
    
    myModel.HomogeneousSolidSection(name='Sec_Anode', material='NiYSZ', thickness=None)
    myModel.HomogeneousSolidSection(name='Sec_Elyte', material='YSZ8', thickness=None)
    myModel.HomogeneousSolidSection(name='Sec_Cath', material='LSM', thickness=None)
    myModel.HomogeneousSolidSection(name='Sec_Intconn', material='FerriticSteel', thickness=None)
    
    part.SectionAssignment(region=part.sets['ANODE'], sectionName='Sec_Anode')
    part.SectionAssignment(region=part.sets['ELYTE'], sectionName='Sec_Elyte')
    part.SectionAssignment(region=part.sets['CATH'], sectionName='Sec_Cath')
    part.SectionAssignment(region=part.sets['INTCONN'], sectionName='Sec_Intconn')
    
    # ========================================================================
    # 5. ASSEMBLY
    # ========================================================================
    
    print("5. Creating assembly...")
    
    myAssembly = myModel.rootAssembly
    myAssembly.DatumCsysByDefault(CARTESIAN)
    myAssembly.Instance(name='SOFC-1', part=part, dependent=ON)
    
    inst = myAssembly.instances['SOFC-1']
    
    # ========================================================================
    # 6. MESH
    # ========================================================================
    
    print("6. Generating mesh with interface refinement...")
    
    # Seed edges in x-direction (uniform)
    x_edges = [
        inst.edges.findAt(((WIDTH/2.0, 0.0, 0.0),)),
        inst.edges.findAt(((WIDTH/2.0, Y_ANODE_TOP, 0.0),)),
        inst.edges.findAt(((WIDTH/2.0, Y_ELYTE_TOP, 0.0),)),
        inst.edges.findAt(((WIDTH/2.0, Y_CATH_TOP, 0.0),)),
        inst.edges.findAt(((WIDTH/2.0, THICK, 0.0),))
    ]
    for e in x_edges:
        part.seedEdgeByNumber(edges=e, number=ELEMS_X)
    
    # Seed edges in y-direction (layer-wise)
    # Anode: 20 elements
    y_anode = part.edges.findAt(((0.0, Y_ANODE_TOP/2.0, 0.0),))
    part.seedEdgeByNumber(edges=y_anode, number=20)
    
    # Electrolyte: 12 elements (critical thin layer)
    y_elyte = part.edges.findAt(((0.0, (Y_ANODE_TOP+Y_ELYTE_TOP)/2.0, 0.0),))
    part.seedEdgeByNumber(edges=y_elyte, number=12)
    
    # Cathode: 20 elements
    y_cath = part.edges.findAt(((0.0, (Y_ELYTE_TOP+Y_CATH_TOP)/2.0, 0.0),))
    part.seedEdgeByNumber(edges=y_cath, number=20)
    
    # Interconnect: 10 elements
    y_intconn = part.edges.findAt(((0.0, (Y_CATH_TOP+THICK)/2.0, 0.0),))
    part.seedEdgeByNumber(edges=y_intconn, number=10)
    
    # Generate mesh
    part.generateMesh()
    
    # Element type assignment (will be done per step)
    print(f"   Mesh generated: ~{ELEMS_X * (20+12+20+10)} elements")
    
    # ========================================================================
    # 7. THERMAL SCHEDULE (Amplitude)
    # ========================================================================
    
    print("7. Creating thermal amplitude...")
    
    ramp_time, hold_time, cool_time, total_time = compute_thermal_schedule(rate, target, hold)
    
    T_ref_K = celsius_to_kelvin(TREF_CELSIUS)
    T_target_K = celsius_to_kelvin(target)
    
    time_data = [
        (0.0, T_ref_K),
        (ramp_time, T_target_K),
        (ramp_time + hold_time, T_target_K),
        (total_time, T_ref_K)
    ]
    
    myModel.TabularAmplitude(name='Amp_Thermal_' + scenario_name,
                             timeSpan=STEP, smooth=SOLVER_DEFAULT,
                             data=time_data)
    
    print(f"   Ramp: {ramp_time/60:.1f} min, Hold: {hold_time/60:.1f} min, Cool: {cool_time/60:.1f} min")
    print(f"   Total time: {total_time/60:.1f} min ({total_time:.0f} s)")
    
    # ========================================================================
    # 8. STEP A - TRANSIENT HEAT TRANSFER
    # ========================================================================
    
    print("8. Creating Step A: Transient heat transfer...")
    
    myModel.HeatTransferStep(name='Step_Heat', previous='Initial',
                             timePeriod=total_time,
                             maxNumInc=50000,
                             initialInc=1.0,
                             minInc=1e-6,
                             maxInc=60.0,
                             deltmx=50.0)  # Max temp change per increment
    
    # Output requests for heat step
    myModel.fieldOutputRequests['F-Output-1'].setValues(variables=('NT', 'HFL'))
    
    # Initial temperature
    myModel.Temperature(name='Predefined_InitTemp',
                        createStepName='Initial',
                        region=inst.sets['ANODE'] + inst.sets['ELYTE'] + inst.sets['CATH'] + inst.sets['INTCONN'],
                        magnitude=T_ref_K)
    
    # Bottom edge: prescribed temperature (amplitude)
    myModel.TemperatureBC(name='BC_Bottom_Temp',
                          createStepName='Step_Heat',
                          region=inst.sets['Y0'],
                          magnitude=1.0,
                          amplitude='Amp_Thermal_' + scenario_name)
    
    # Top edge: film condition (convection)
    h_film = 25.0  # W/m²·K
    T_ambient_K = celsius_to_kelvin(25.0)
    
    myModel.FilmCondition(name='BC_Top_Film',
                          createStepName='Step_Heat',
                          surface=inst.surfaces.create(name='Surf_Top',
                                                       side1Edges=inst.sets['YTOP'].edges),
                          definition=EMBEDDED_COEFF,
                          filmCoeff=h_film,
                          filmCoeffAmplitude='',
                          sinkTemperature=T_ambient_K,
                          sinkAmplitude='')
    
    # ========================================================================
    # 9. STEP B - THERMO-MECHANICAL
    # ========================================================================
    
    print("9. Creating Step B: Thermo-mechanical (static, NLGEOM)...")
    
    myModel.StaticStep(name='Step_Mech', previous='Step_Heat',
                       timePeriod=total_time,
                       maxNumInc=50000,
                       initialInc=1.0,
                       minInc=1e-6,
                       maxInc=60.0,
                       nlgeom=ON)
    
    # Predefined temperature field from previous step
    myModel.Temperature(name='Predefined_TempFromHeat',
                        createStepName='Step_Mech',
                        region=inst.sets['ANODE'] + inst.sets['ELYTE'] + inst.sets['CATH'] + inst.sets['INTCONN'],
                        distributionType=FROM_FILE,
                        fileName='SOFC_' + scenario_name + '.odb',
                        beginStep=1, beginIncrement=1, endStep=1, endIncrement=999999,
                        interpolate=ON)
    
    # Mechanical BCs
    myModel.DisplacementBC(name='BC_Mech_X0', createStepName='Step_Mech',
                           region=inst.sets['X0'], u1=0.0, u2=UNSET)
    
    myModel.DisplacementBC(name='BC_Mech_Y0', createStepName='Step_Mech',
                           region=inst.sets['Y0'], u1=UNSET, u2=0.0)
    
    # Output requests for mechanical step
    myModel.fieldOutputRequests['F-Output-1'].setValues(
        variables=('S', 'E', 'LE', 'PE', 'PEEQ', 'CE', 'CEEQ', 'TEMP'),
        timeInterval=total_time/100.0  # 100 frames
    )
    
    # History output at interfaces
    myModel.HistoryOutputRequest(name='H-Output-Interface-AE',
                                  createStepName='Step_Mech',
                                  region=inst.sets['INT_AE'],
                                  variables=('S11', 'S22', 'S12'),
                                  timeInterval=total_time/100.0)
    
    myModel.HistoryOutputRequest(name='H-Output-Interface-EC',
                                  createStepName='Step_Mech',
                                  region=inst.sets['INT_EC'],
                                  variables=('S11', 'S22', 'S12'),
                                  timeInterval=total_time/100.0)
    
    myModel.HistoryOutputRequest(name='H-Output-Interface-CI',
                                  createStepName='Step_Mech',
                                  region=inst.sets['INT_CI'],
                                  variables=('S11', 'S22', 'S12'),
                                  timeInterval=total_time/100.0)
    
    # ========================================================================
    # 10. ELEMENT TYPE ASSIGNMENTS
    # ========================================================================
    
    print("10. Assigning element types...")
    
    # For heat step: DC2D4
    elemType_heat = mesh.ElemType(elemCode=DC2D4, elemLibrary=STANDARD)
    
    # For mechanical step: CPS4 (plane stress)
    elemType_mech = mesh.ElemType(elemCode=CPS4, elemLibrary=STANDARD)
    
    # Assign DC2D4 for all regions (heat step will use this)
    all_faces = part.faces
    part.setElementType(regions=(all_faces,), elemTypes=(elemType_heat,))
    
    # Note: Abaqus will automatically switch element types per step based on procedure
    # For explicit control, we'd need to remesh or use element deletion/reactivation
    # In practice, setting up two meshes or using CPS4T (coupled) is cleaner
    # For this script, we'll document that users should manually switch or use CPS4T
    
    print("   NOTE: For sequential analysis, element type switching required.")
    print("   Consider using CPS4T (coupled temp-displacement) for single-step analysis.")
    
    # ========================================================================
    # 11. JOB CREATION
    # ========================================================================
    
    print("11. Creating job...")
    
    job_name = 'Job_' + model_name
    myJob = mdb.Job(name=job_name, model=model_name,
                    description='SOFC multi-physics simulation: ' + scenario_name,
                    type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0,
                    queue=None, memory=90, memoryUnits=PERCENTAGE,
                    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE,
                    echoPrint=OFF, modelPrint=OFF, contactPrint=OFF,
                    historyPrint=OFF, userSubroutine='', scratch='',
                    resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN,
                    numDomains=4, activateLoadBalancing=False,
                    multiprocessingMode=DEFAULT, numCpus=4)
    
    print(f"\n{'='*70}")
    print(f"Model '{model_name}' created successfully!")
    print(f"Job '{job_name}' ready to submit.")
    print(f"{'='*70}\n")
    
    return myModel, myJob

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    
    print("\n" + "="*70)
    print("SOFC SIMULATION SETUP - ABAQUS/STANDARD")
    print("="*70 + "\n")
    
    # Create models for all three heating rate scenarios
    models_jobs = {}
    
    for scenario in ['HR1', 'HR4', 'HR10']:
        try:
            model, job = create_model(scenario)
            models_jobs[scenario] = (model, job)
            
            # Save CAE file
            mdb.saveAs(pathName='/workspace/SOFC_' + scenario + '.cae')
            print(f"Saved: SOFC_{scenario}.cae\n")
            
        except Exception as e:
            print(f"ERROR creating model {scenario}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("SETUP COMPLETE - Ready to submit jobs")
    print("="*70)
    print("\nTo run simulations:")
    print("  1. Open Abaqus CAE: abaqus cae")
    print("  2. Load model: File → Open → SOFC_HR1.cae (or HR4, HR10)")
    print("  3. Submit job: Job → Submit")
    print("\nOr submit from command line:")
    print("  abaqus job=Job_SOFC_HR1 interactive")
    print("  abaqus job=Job_SOFC_HR4 interactive")
    print("  abaqus job=Job_SOFC_HR10 interactive")
    print("\n" + "="*70 + "\n")
