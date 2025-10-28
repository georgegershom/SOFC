#!/usr/bin/env python
"""
SOFC Model Creation Script for Abaqus/CAE
Creates a complete 2D plane stress model for SOFC thermo-mechanical analysis
Units: SI (m-kg-s-K-N-Pa)
"""

from abaqus import *
from abaqusConstants import *
from caeModules import *
import regionToolset
import numpy as np

def create_sofc_model(heating_rate='HR1'):
    """
    Create complete SOFC model in Abaqus/CAE
    
    Args:
        heating_rate: 'HR1' (1°C/min), 'HR4' (4°C/min), or 'HR10' (10°C/min)
    """
    
    # Model parameters
    model_name = f'SOFC_Model_{heating_rate}'
    width = 0.010  # 10 mm in meters
    thickness = 0.001  # 1 mm in meters
    
    # Layer boundaries (y-coordinates in meters)
    layers = {
        'anode': (0.0, 0.0004),
        'electrolyte': (0.0004, 0.0005),
        'cathode': (0.0005, 0.0009),
        'interconnect': (0.0009, 0.001)
    }
    
    # Create model
    mdb.Model(name=model_name, modelType=STANDARD_EXPLICIT)
    model = mdb.models[model_name]
    
    # Delete default model if exists
    if 'Model-1' in mdb.models:
        del mdb.models['Model-1']
    
    # ============================================
    # PART CREATION
    # ============================================
    
    # Create 2D planar part
    sketch = model.ConstrainedSketch(name='SOFC_Sketch', sheetSize=0.02)
    sketch.rectangle(point1=(0.0, 0.0), point2=(width, thickness))
    
    part = model.Part(name='SOFC_Cell', dimensionality=TWO_D_PLANAR,
                      type=DEFORMABLE_BODY)
    part.BaseShell(sketch=sketch)
    
    # Create partitions for layers
    for layer_name, (y_min, y_max) in layers.items():
        if y_min > 0:
            # Create horizontal partition
            part.PartitionFaceByShortestPath(
                faces=part.faces,
                point1=(0.0, y_min, 0.0),
                point2=(width, y_min, 0.0)
            )
    
    # ============================================
    # MATERIAL DEFINITIONS
    # ============================================
    
    # Material 1: Ni-YSZ (Anode)
    mat_anode = model.Material(name='Ni_YSZ')
    mat_anode.Density(table=((7000.0,),))
    mat_anode.Elastic(temperatureDependency=ON,
                      table=((1.40E11, 0.30, 298.0),
                            (9.10E10, 0.30, 1273.0)))
    mat_anode.Expansion(temperatureDependency=ON,
                        table=((1.25E-5, 298.0),
                              (1.35E-5, 1273.0)))
    mat_anode.Conductivity(temperatureDependency=ON,
                           table=((6.0, 298.0),
                                 (4.0, 1273.0)))
    mat_anode.SpecificHeat(temperatureDependency=ON,
                           table=((450.0, 298.0),
                                 (570.0, 1273.0)))
    
    # Johnson-Cook Plasticity
    mat_anode.Plastic(hardening=JOHNSON_COOK,
                      table=((150.0E6, 200.0E6, 0.35, 0.02, 1.0),))
    mat_anode.plastic.RateDependent(type=JOHNSON_COOK,
                                    table=((298.0, 1720.0, 1.0),))
    
    # Norton-Bailey Creep
    mat_anode.Creep(law=NORTON, table=((1.0E-18, 3.5, 2.2E5/8.314),))
    
    # Material 2: 8YSZ (Electrolyte)
    mat_elyte = model.Material(name='YSZ_8')
    mat_elyte.Density(table=((5900.0,),))
    mat_elyte.Elastic(temperatureDependency=ON,
                      table=((2.10E11, 0.28, 298.0),
                            (1.70E11, 0.28, 1273.0)))
    mat_elyte.Expansion(temperatureDependency=ON,
                        table=((1.05E-5, 298.0),
                              (1.12E-5, 1273.0)))
    mat_elyte.Conductivity(temperatureDependency=ON,
                           table=((2.6, 298.0),
                                 (2.0, 1273.0)))
    mat_elyte.SpecificHeat(temperatureDependency=ON,
                           table=((400.0, 298.0),
                                 (600.0, 1273.0)))
    mat_elyte.Creep(law=NORTON, table=((5.0E-22, 2.0, 3.8E5/8.314),))
    
    # Material 3: LSM (Cathode)
    mat_cathode = model.Material(name='LSM')
    mat_cathode.Density(table=((6500.0,),))
    mat_cathode.Elastic(temperatureDependency=ON,
                        table=((1.20E11, 0.30, 298.0),
                              (8.40E10, 0.30, 1273.0)))
    mat_cathode.Expansion(temperatureDependency=ON,
                          table=((1.15E-5, 298.0),
                                (1.24E-5, 1273.0)))
    mat_cathode.Conductivity(temperatureDependency=ON,
                             table=((2.0, 298.0),
                                   (1.8, 1273.0)))
    mat_cathode.SpecificHeat(temperatureDependency=ON,
                             table=((480.0, 298.0),
                                   (610.0, 1273.0)))
    
    # Material 4: Ferritic Steel (Interconnect)
    mat_steel = model.Material(name='Ferritic_Steel')
    mat_steel.Density(table=((7800.0,),))
    mat_steel.Elastic(temperatureDependency=ON,
                      table=((2.05E11, 0.30, 298.0),
                            (1.50E11, 0.30, 1273.0)))
    mat_steel.Expansion(temperatureDependency=ON,
                        table=((1.25E-5, 298.0),
                              (1.32E-5, 1273.0)))
    mat_steel.Conductivity(temperatureDependency=ON,
                           table=((20.0, 298.0),
                                 (15.0, 1273.0)))
    mat_steel.SpecificHeat(temperatureDependency=ON,
                           table=((500.0, 298.0),
                                 (700.0, 1273.0)))
    
    # ============================================
    # SECTIONS
    # ============================================
    
    # Create sections for each layer
    model.HomogeneousSolidSection(name='Sec_Anode', material='Ni_YSZ')
    model.HomogeneousSolidSection(name='Sec_Electrolyte', material='YSZ_8')
    model.HomogeneousSolidSection(name='Sec_Cathode', material='LSM')
    model.HomogeneousSolidSection(name='Sec_Interconnect', material='Ferritic_Steel')
    
    # Assign sections to regions
    # (This requires identifying the correct faces after partitioning)
    faces = part.faces
    
    # Create sets for each layer based on y-coordinates
    for face in faces:
        centroid = face.getCentroid()
        y_coord = centroid[0][1]
        
        if 0.0 <= y_coord < 0.0004:
            region = regionToolset.Region(faces=(face,))
            part.SectionAssignment(region=region, sectionName='Sec_Anode')
            part.Set(faces=(face,), name='ANODE')
        elif 0.0004 <= y_coord < 0.0005:
            region = regionToolset.Region(faces=(face,))
            part.SectionAssignment(region=region, sectionName='Sec_Electrolyte')
            part.Set(faces=(face,), name='ELECTROLYTE')
        elif 0.0005 <= y_coord < 0.0009:
            region = regionToolset.Region(faces=(face,))
            part.SectionAssignment(region=region, sectionName='Sec_Cathode')
            part.Set(faces=(face,), name='CATHODE')
        elif 0.0009 <= y_coord <= 0.001:
            region = regionToolset.Region(faces=(face,))
            part.SectionAssignment(region=region, sectionName='Sec_Interconnect')
            part.Set(faces=(face,), name='INTERCONNECT')
    
    # Create edge sets for boundaries
    edges = part.edges
    
    # Left edge (X=0)
    left_edges = edges.findAt(((0.0, thickness/2, 0.0),))
    part.Set(edges=left_edges, name='X0_EDGE')
    
    # Right edge (X=width)
    right_edges = edges.findAt(((width, thickness/2, 0.0),))
    part.Set(edges=right_edges, name='X10_EDGE')
    
    # Bottom edge (Y=0)
    bottom_edges = edges.findAt(((width/2, 0.0, 0.0),))
    part.Set(edges=bottom_edges, name='Y0_EDGE')
    
    # Top edge (Y=thickness)
    top_edges = edges.findAt(((width/2, thickness, 0.0),))
    part.Set(edges=top_edges, name='YTOP_EDGE')
    
    # Create interface sets
    # Anode-Electrolyte interface (Y=0.4mm)
    ae_edges = edges.findAt(((width/2, 0.0004, 0.0),))
    part.Set(edges=ae_edges, name='INT_AE')
    part.Surface(side1Edges=ae_edges, name='SURF_AE')
    
    # Electrolyte-Cathode interface (Y=0.5mm)
    ec_edges = edges.findAt(((width/2, 0.0005, 0.0),))
    part.Set(edges=ec_edges, name='INT_EC')
    part.Surface(side1Edges=ec_edges, name='SURF_EC')
    
    # Cathode-Interconnect interface (Y=0.9mm)
    ci_edges = edges.findAt(((width/2, 0.0009, 0.0),))
    part.Set(edges=ci_edges, name='INT_CI')
    part.Surface(side1Edges=ci_edges, name='SURF_CI')
    
    # ============================================
    # ASSEMBLY
    # ============================================
    
    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)
    instance = assembly.Instance(name='SOFC_Cell-1', part=part, dependent=ON)
    
    # ============================================
    # MESH
    # ============================================
    
    # Set element types
    elemType_heat = mesh.ElemType(elemCode=DC2D4, elemLibrary=STANDARD)
    elemType_struct = mesh.ElemType(elemCode=CPS4, elemLibrary=STANDARD)
    
    # Seed edges
    # X-direction: 80 elements
    part.seedEdgeByNumber(edges=part.edges.findAt(((width/2, 0.0, 0.0),)),
                         number=80, constraint=FINER)
    part.seedEdgeByNumber(edges=part.edges.findAt(((width/2, thickness, 0.0),)),
                         number=80, constraint=FINER)
    
    # Y-direction: refined near interfaces
    # Anode layer (0-0.4mm): 20 elements with bias towards interface
    anode_edges = part.edges.findAt(((0.0, 0.0002, 0.0),))
    part.seedEdgeByBias(edges=anode_edges, number=20, ratio=3.0,
                       constraint=FINER, minSize=0.000005, maxSize=0.00003)
    
    # Electrolyte layer (0.4-0.5mm): 12 elements (fine mesh)
    elyte_edges = part.edges.findAt(((0.0, 0.00045, 0.0),))
    part.seedEdgeByNumber(edges=elyte_edges, number=12, constraint=FINER)
    
    # Cathode layer (0.5-0.9mm): 20 elements with bias
    cath_edges = part.edges.findAt(((0.0, 0.0007, 0.0),))
    part.seedEdgeByBias(edges=cath_edges, number=20, ratio=3.0,
                       constraint=FINER, minSize=0.000005, maxSize=0.00003)
    
    # Interconnect layer (0.9-1.0mm): 10 elements
    inter_edges = part.edges.findAt(((0.0, 0.00095, 0.0),))
    part.seedEdgeByNumber(edges=inter_edges, number=10, constraint=FINER)
    
    # Generate mesh
    part.generateMesh()
    
    # ============================================
    # AMPLITUDE CURVES
    # ============================================
    
    # Define heating schedules
    heating_schedules = {
        'HR1': {  # 1°C/min
            'ramp_time': 52500.0,  # 875 minutes in seconds
            'hold_time': 600.0,    # 10 minutes
            'cool_time': 52500.0   # 875 minutes
        },
        'HR4': {  # 4°C/min
            'ramp_time': 13125.0,  # 218.75 minutes
            'hold_time': 600.0,
            'cool_time': 13125.0
        },
        'HR10': {  # 10°C/min
            'ramp_time': 5250.0,   # 87.5 minutes
            'hold_time': 600.0,
            'cool_time': 5250.0
        }
    }
    
    schedule = heating_schedules[heating_rate]
    t1 = schedule['ramp_time']
    t2 = t1 + schedule['hold_time']
    t3 = t2 + schedule['cool_time']
    
    # Create amplitude curve
    model.TabularAmplitude(name=f'AMP_{heating_rate}',
                          timeSpan=TOTAL,
                          smooth=SOLVER_DEFAULT,
                          data=((0.0, 298.0),
                               (t1, 1173.0),
                               (t2, 1173.0),
                               (t3, 298.0)))
    
    # Film coefficient amplitude (constant)
    model.TabularAmplitude(name='AMP_FILM',
                          timeSpan=TOTAL,
                          smooth=SOLVER_DEFAULT,
                          data=((0.0, 1.0), (t3, 1.0)))
    
    # ============================================
    # STEP 1: HEAT TRANSFER
    # ============================================
    
    # Create heat transfer step
    model.HeatTransferStep(name='Heat_Transfer',
                          previous='Initial',
                          timePeriod=t3,
                          initialInc=1.0,
                          minInc=0.001,
                          maxInc=100.0,
                          deltmx=50.0,
                          amplitude=RAMP,
                          extrapolation=LINEAR)
    
    # Apply thermal boundary conditions
    # Bottom edge: prescribed temperature
    region_bottom = assembly.sets['SOFC_Cell-1.Y0_EDGE']
    model.TemperatureBC(name='BC_Bottom_Temp',
                       createStepName='Heat_Transfer',
                       region=region_bottom,
                       magnitude=1.0,
                       amplitude=f'AMP_{heating_rate}',
                       distributionType=UNIFORM)
    
    # Top edge: film condition (convection)
    region_top = assembly.surfaces['SOFC_Cell-1.SURF_YTOP']
    model.FilmCondition(name='BC_Top_Film',
                       createStepName='Heat_Transfer',
                       surface=region_top,
                       definition=EMBEDDED_COEFF,
                       filmCoeff=25.0,
                       filmCoeffAmplitude='AMP_FILM',
                       sinkTemperature=298.0,
                       sinkAmplitude='')
    
    # ============================================
    # STEP 2: THERMO-MECHANICAL
    # ============================================
    
    # Create static structural step
    model.StaticStep(name='Thermo_Mechanical',
                    previous='Heat_Transfer',
                    timePeriod=t3,
                    initialInc=1.0,
                    minInc=0.001,
                    maxInc=100.0,
                    nlgeom=ON,
                    amplitude=RAMP)
    
    # Apply mechanical boundary conditions
    # Left edge: roller (Ux=0)
    region_left = assembly.sets['SOFC_Cell-1.X0_EDGE']
    model.DisplacementBC(name='BC_Left_Roller',
                        createStepName='Thermo_Mechanical',
                        region=region_left,
                        u1=0.0, u2=UNSET, ur3=UNSET,
                        amplitude=UNSET,
                        distributionType=UNIFORM)
    
    # Bottom edge: roller (Uy=0)
    model.DisplacementBC(name='BC_Bottom_Roller',
                        createStepName='Thermo_Mechanical',
                        region=region_bottom,
                        u1=UNSET, u2=0.0, ur3=UNSET,
                        amplitude=UNSET,
                        distributionType=UNIFORM)
    
    # Apply predefined temperature field from heat transfer step
    model.Temperature(name='Predefined_Temp',
                     createStepName='Thermo_Mechanical',
                     region=assembly.sets['SOFC_Cell-1.ALL_NODES'],
                     distributionType=FROM_FILE,
                     fileName='Heat_Transfer',
                     beginStep=1,
                     beginIncrement=0,
                     endStep=1,
                     endIncrement=99999,
                     interpolate=ON,
                     absoluteExteriorTolerance=0.0,
                     exteriorTolerance=0.05)
    
    # ============================================
    # OUTPUT REQUESTS
    # ============================================
    
    # Field output for heat transfer step
    model.FieldOutputRequest(name='F-Output-Heat',
                            createStepName='Heat_Transfer',
                            variables=('NT', 'HFL', 'RFL'))
    
    # Field output for structural step
    model.FieldOutputRequest(name='F-Output-Struct',
                            createStepName='Thermo_Mechanical',
                            variables=('S', 'E', 'LE', 'EE', 'PE', 'CE',
                                     'PEEQ', 'CEEQ', 'MISES', 'U', 'RF', 'TEMP'))
    
    # History output for interface monitoring
    for interface in ['INT_AE', 'INT_EC', 'INT_CI']:
        model.HistoryOutputRequest(name=f'H-Output-{interface}',
                                  createStepName='Thermo_Mechanical',
                                  region=assembly.sets[f'SOFC_Cell-1.{interface}'],
                                  variables=('S11', 'S22', 'S12', 'MISES'))
    
    # ============================================
    # JOB CREATION
    # ============================================
    
    job_name = f'SOFC_{heating_rate}_Job'
    mdb.Job(name=job_name,
           model=model_name,
           description=f'SOFC Thermo-mechanical Analysis - {heating_rate}',
           type=ANALYSIS,
           atTime=None,
           waitMinutes=0,
           waitHours=0,
           queue=None,
           memory=90,
           memoryUnits=PERCENTAGE,
           getMemoryFromAnalysis=True,
           explicitPrecision=SINGLE,
           nodalOutputPrecision=SINGLE,
           echoPrint=OFF,
           modelPrint=OFF,
           contactPrint=OFF,
           historyPrint=OFF,
           userSubroutine='',
           scratch='',
           resultsFormat=ODB,
           multiprocessingMode=DEFAULT,
           numCpus=4,
           numDomains=4,
           numGPUs=0)
    
    print(f"Model '{model_name}' created successfully!")
    print(f"Job '{job_name}' is ready to submit.")
    
    return model, job_name


if __name__ == '__main__':
    # Create models for all heating rates
    for hr in ['HR1', 'HR4', 'HR10']:
        model, job = create_sofc_model(heating_rate=hr)
        print(f"\nCreated model for heating rate {hr}")
        
        # Optional: Submit job automatically
        # mdb.jobs[job].submit(consistencyChecking=OFF)
        # print(f"Job {job} submitted")