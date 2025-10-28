# Abaqus/CAE noGUI script to build and run a 2D CPS4T SOFC model
import sys
import os

# Abaqus modules are available only inside the Abaqus Python environment
from abaqus import mdb, session
from abaqusConstants import (
    TWO_D_PLANAR, DEFORMABLE_BODY, CARTESIAN, ISOTROPIC, ON, OFF, 
    STANDALONE, STANDARD, CPS4T, MIDDLE_SURFACE,
    NODE_OUTPUT, ELEMENT_NODAL, UNSET, 
    KELVIN, CENTIGRADE, EMBEDDED_COEFF, DEFAULT, 
    EDGE, WHOLE_SURFACE, FROM_SECTION, UNIFORM, ANALYSIS,
)
import regionToolset
import mesh


def parse_args(argv):
    args = {"hr": "hr4", "job": None, "x_elems": 80}
    i = 0
    while i < len(argv):
        if argv[i] == "--hr" and i + 1 < len(argv):
            args["hr"] = argv[i + 1].lower()
            i += 2
        elif argv[i] == "--job" and i + 1 < len(argv):
            args["job"] = argv[i + 1]
            i += 2
        elif argv[i] == "--x-elems" and i + 1 < len(argv):
            args["x_elems"] = int(argv[i + 1])
            i += 2
        else:
            i += 1
    if args["job"] is None:
        args["job"] = f"sofc_{args['hr']}"
    if args["hr"] not in ("hr1", "hr4", "hr10"):
        raise ValueError("--hr must be one of: hr1, hr4, hr10")
    return args


def get_heating_schedule(hr):
    # Temperatures in Kelvin
    T0 = 298.15
    Tt = 1173.15  # 900 °C
    if hr == "hr1":
        ramp_min = 875.0
    elif hr == "hr4":
        ramp_min = 218.75
    else:  # hr10
        ramp_min = 87.5
    hold_min = 10.0
    # Convert to seconds
    ramp = ramp_min * 60.0
    hold = hold_min * 60.0
    cool = ramp
    total = ramp + hold + cool
    # Amplitude: piecewise linear [time(s), temperature(K)]
    amp_data = (
        (0.0, T0),
        (ramp, Tt),
        (ramp + hold, Tt),
        (total, T0),
    )
    return amp_data, total


def build_geometry(model):
    # Dimensions in meters (10 mm x 1 mm)
    width = 0.010
    thickness = 0.001
    y_a_e = 0.0004
    y_e_c = 0.0005
    y_c_i = 0.0009

    s = model.ConstrainedSketch(name='__profile__', sheetSize=0.05)
    s.rectangle(point1=(0.0, 0.0), point2=(width, thickness))
    part = model.Part(name='SOFC2D', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part.BaseShell(sketch=s)
    del s

    # Partition at y = 0.0004, 0.0005, 0.0009 m using sketch lines across width
    for y_split in (y_a_e, y_e_c, y_c_i):
        sp = model.ConstrainedSketch(name='__partition__', sheetSize=0.05, gridSpacing=0.001)
        sp.Line(point1=(0.0, y_split), point2=(width, y_split))
        f = part.faces
        pickedFaces = f
        part.PartitionFaceBySketch(faces=pickedFaces, sketch=sp)
        del sp

    # Create sets for layers by Y ranges
    faces = part.faces
    # Helper to get faces by y-range bounding box
    def faces_by_y(ymin, ymax):
        return faces.getByBoundingBox(xMin=-1.0, xMax=1.0, yMin=ymin - 1e-9, yMax=ymax + 1e-9)

    part.Set(name='ANODE', faces=faces_by_y(0.0, y_a_e))
    part.Set(name='ELYTE', faces=faces_by_y(y_a_e, y_e_c))
    part.Set(name='CATH', faces=faces_by_y(y_e_c, y_c_i))
    part.Set(name='INTCONN', faces=faces_by_y(y_c_i, thickness))

    # Edge sets for boundaries and interfaces
    edges = part.edges
    # Boundaries
    X0_edges = edges.getByBoundingBox(xMin=-1e-12, xMax=1e-12, yMin=-1.0, yMax=1.0)
    Y0_edges = edges.getByBoundingBox(xMin=-1.0, xMax=1.0, yMin=-1e-12, yMax=1e-12)
    YTOP_edges = edges.getByBoundingBox(xMin=-1.0, xMax=1.0, yMin=thickness - 1e-12, yMax=thickness + 1e-12)
    part.Set(name='X0', edges=X0_edges)
    part.Set(name='Y0', edges=Y0_edges)
    part.Set(name='YTOP', edges=YTOP_edges)

    # Interfaces by y coordinates
    def edge_near_y(y):
        return edges.getByBoundingBox(xMin=-1.0, xMax=1.0, yMin=y - 1e-9, yMax=y + 1e-9)

    part.Set(name='INT_AE', edges=edge_near_y(y_a_e))
    part.Set(name='INT_EC', edges=edge_near_y(y_e_c))
    part.Set(name='INT_CI', edges=edge_near_y(y_c_i))

    return part


def define_materials_and_sections(model):
    # Create materials (temperature dependent)
    # Anode (Ni-YSZ)
    matA = model.Material(name='MAT_ANODE')
    matA.Elastic(type=ISOTROPIC, temperatureDependency=ON, table=((140e9, 0.30, 298.15), (91e9, 0.30, 1273.15)))
    matA.Expansion(type=ISOTROPIC, temperatureDependency=ON, table=((12.5e-6, 298.15), (13.5e-6, 1273.15)))
    matA.Conductivity(temperatureDependency=ON, table=((6.0, 298.15), (4.0, 1273.15)))
    matA.SpecificHeat(temperatureDependency=ON, table=((450.0, 298.15), (570.0, 1273.15)))

    # Electrolyte (8YSZ)
    matE = model.Material(name='MAT_ELYTE')
    matE.Elastic(type=ISOTROPIC, temperatureDependency=ON, table=((210e9, 0.28, 298.15), (170e9, 0.28, 1273.15)))
    matE.Expansion(type=ISOTROPIC, temperatureDependency=ON, table=((10.5e-6, 298.15), (11.2e-6, 1273.15)))
    matE.Conductivity(temperatureDependency=ON, table=((2.6, 298.15), (2.0, 1273.15)))
    matE.SpecificHeat(temperatureDependency=ON, table=((400.0, 298.15), (600.0, 1273.15)))

    # Cathode (LSM)
    matC = model.Material(name='MAT_CATH')
    matC.Elastic(type=ISOTROPIC, temperatureDependency=ON, table=((120e9, 0.30, 298.15), (84e9, 0.30, 1273.15)))
    matC.Expansion(type=ISOTROPIC, temperatureDependency=ON, table=((11.5e-6, 298.15), (12.4e-6, 1273.15)))
    matC.Conductivity(temperatureDependency=ON, table=((2.0, 298.15), (1.8, 1273.15)))
    matC.SpecificHeat(temperatureDependency=ON, table=((480.0, 298.15), (610.0, 1273.15)))

    # Interconnect (Ferritic Steel)
    matI = model.Material(name='MAT_INTCONN')
    matI.Elastic(type=ISOTROPIC, temperatureDependency=ON, table=((205e9, 0.30, 298.15), (150e9, 0.30, 1273.15)))
    matI.Expansion(type=ISOTROPIC, temperatureDependency=ON, table=((12.5e-6, 298.15), (13.2e-6, 1273.15)))
    matI.Conductivity(temperatureDependency=ON, table=((20.0, 298.15), (15.0, 1273.15)))
    matI.SpecificHeat(temperatureDependency=ON, table=((500.0, 298.15), (700.0, 1273.15)))

    # Sections
    secA = model.HomogeneousSolidSection(name='SEC_ANODE', material='MAT_ANODE', thickness=None)
    secE = model.HomogeneousSolidSection(name='SEC_ELYTE', material='MAT_ELYTE', thickness=None)
    secC = model.HomogeneousSolidSection(name='SEC_CATH', material='MAT_CATH', thickness=None)
    secI = model.HomogeneousSolidSection(name='SEC_INTCONN', material='MAT_INTCONN', thickness=None)

    return {
        'SEC_ANODE': secA,
        'SEC_ELYTE': secE,
        'SEC_CATH': secC,
        'SEC_INTCONN': secI,
    }


def assign_sections(model, part):
    part.SectionAssignment(region=part.sets['ANODE'].faces, sectionName='SEC_ANODE')
    part.SectionAssignment(region=part.sets['ELYTE'].faces, sectionName='SEC_ELYTE')
    part.SectionAssignment(region=part.sets['CATH'].faces, sectionName='SEC_CATH')
    part.SectionAssignment(region=part.sets['INTCONN'].faces, sectionName='SEC_INTCONN')


def instance_assembly(model, part):
    asm = model.rootAssembly
    asm.DatumCsysByDefault(CARTESIAN)
    asm.Instance(name='SOFC2D-1', part=part, dependent=ON)
    return asm


def seed_and_mesh(model, part, x_elems):
    width = 0.010
    thickness = 0.001
    y_interfaces = (0.0004, 0.0005, 0.0009)
    # Global seed by size targeting ~x_elems along width
    size_global = width / float(x_elems)
    part.seedPart(size=size_global, deviationFactor=0.1, minSizeFactor=0.1)
    # Refine near interfaces: seed edges close to those y with smaller size
    edges = part.edges
    for y in y_interfaces:
        e_int = edges.getByBoundingBox(xMin=-1.0, xMax=1.0, yMin=y - 1e-9, yMax=y + 1e-9)
        if len(e_int) > 0:
            part.seedEdgeBySize(edges=e_int, size=size_global * 0.2, deviationFactor=0.1, minSizeFactor=0.1)

    # Element type CPS4T (plane stress + temperature)
    elemType = mesh.ElemType(elemCode=CPS4T, elemLibrary=STANDARD)
    faces = part.faces
    pickedRegions = (faces,)
    part.setElementType(regions=pickedRegions, elemTypes=(elemType,))
    part.generateMesh()


def define_steps_and_bcs(model, asm, hr):
    # Heating schedule
    amp_data, total_time = get_heating_schedule(hr)
    model.TabularAmplitude(name='AMP_BOTTOM', timeSpan=STEP, data=amp_data)

    # Note: For Abaqus constants
    # Add Coupled temperature-displacement step
    model.CoupledTempDisplacementStep(name='Step-1', previous='Initial', nlgeom=ON, timePeriod=total_time, maxNumInc=25000, amplitude=STEP)

    # Mechanical BCs: left edge U1=0, bottom edge U2=0
    inst = asm.instances['SOFC2D-1']
    left_region = regionToolset.Region(edges=inst.edges.getByBoundingBox(xMin=-1e-12, xMax=1e-12, yMin=-1.0, yMax=1.0))
    bottom_region = regionToolset.Region(edges=inst.edges.getByBoundingBox(xMin=-1.0, xMax=1.0, yMin=-1e-12, yMax=1e-12))
    model.DisplacementBC(name='BC_X0_U1', createStepName='Step-1', region=left_region, u1=0.0, u2=UNSET, ur3=UNSET, amplitude=UNSET, distributionType=UNIFORM)
    model.DisplacementBC(name='BC_Y0_U2', createStepName='Step-1', region=bottom_region, u1=UNSET, u2=0.0, ur3=UNSET, amplitude=UNSET, distributionType=UNIFORM)

    # Thermal BCs: bottom prescribed temperature via amplitude
    bottom_temp_region = bottom_region
    model.TemperatureBC(name='BC_BOTTOM_TEMP', createStepName='Step-1', region=bottom_temp_region, fixed=OFF, distributionType=UNIFORM, magnitude=298.15, amplitude='AMP_BOTTOM')

    # Film condition at top edge
    top_region = regionToolset.Region(edges=inst.edges.getByBoundingBox(xMin=-1.0, xMax=1.0, yMin=0.001 - 1e-12, yMax=0.001 + 1e-12))
    model.FilmCondition(name='FILM_TOP', createStepName='Step-1', surface=top_region, definition=EMBEDDED_COEFF, filmCoeff=25.0, sinkTemperature=298.15)

    # Field output request
    model.fieldOutputRequests['F-Output-1'].setValues(variables=(
        'S', 'E', 'LE', 'TEMP', 'HFL'
    ))

    # History outputs along interfaces for S12
    # Use element-nodal history on edges near interfaces
    for name, y in (('INT_AE', 0.0004), ('INT_EC', 0.0005), ('INT_CI', 0.0009)):
        region_edges = inst.edges.getByBoundingBox(xMin=-1.0, xMax=1.0, yMin=y - 1e-9, yMax=y + 1e-9)
        if len(region_edges) > 0:
            region = regionToolset.Region(edges=region_edges)
            model.HistoryOutputRequest(name=f'H-S12-{name}', createStepName='Step-1', variables=('S12',), region=region, sectionPoints=DEFAULT, rebar=EXCLUDE)


def create_and_submit_job(model, jobname):
    # Ensure jobs directory
    jobs_dir = os.path.abspath(os.path.join(os.getcwd(), 'jobs', jobname))
    try:
        os.makedirs(jobs_dir)
    except Exception:
        pass

    job = mdb.Job(name=jobname, model=model.name, type=ANALYSIS, explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, description='', parallelizationMethodExplicit=DOMAIN, multiprocessingMode=DEFAULT, numDomains=1, numCpus=1, numGPUs=0)
    job.setValues(activateLoadBalancing=False, scratch=jobs_dir, userSubroutine='', memory=90, memoryUnits=PERCENTAGE)
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()


def main():
    # Parse custom args after "--"
    if '--' in sys.argv:
        idx = sys.argv.index('--') + 1
        custom_argv = sys.argv[idx:]
    else:
        custom_argv = []
    args = parse_args(custom_argv)

    # Create model
    model = mdb.Model(name='SOFC')
    part = build_geometry(model)
    define_materials_and_sections(model)
    assign_sections(model, part)
    asm = instance_assembly(model, part)
    seed_and_mesh(model, part, args['x_elems'])
    define_steps_and_bcs(model, asm, args['hr'])

    # Save CAE for reference
    cae_path = os.path.abspath(os.path.join(os.getcwd(), f"{args['job']}.cae"))
    mdb.saveAs(pathName=cae_path)

    # Create and submit job
    create_and_submit_job(model, args['job'])


if __name__ == '__main__':
    main()

