# -*- coding: utf-8 -*-
"""
+==============================================================================+
|                                                                              |
|   Abaqus/Standard Script: 2D Planar SOFC Residual Stress Validation Model    |
|                                                                              |
|   Purpose   : Thermo-mechanical simulation of anode-electrolyte bilayer      |
|               residual stress upon cooling from sintering temperature.       |
|               Validated against synchrotron XRD lattice-strain data.         |
|                                                                              |
|   Author    : SOFC Research Assistant (AI-Generated, Human-Reviewed)         |
|   Version   : 2.4.0                                                          |
|   Created   : 2026-02-08                                                     |
|   Modified  : 2026-02-09                                                     |
|                                                                              |
|   Units     : mm, N, s, C (stress in MPa, density in tonne/mm^3)             |
|                                                                              |
|   Reference : Mechanical characterisation of Ni-YSZ anode-supported          |
|               half-cells via synchrotron X-ray diffraction.                  |
|                                                                              |
|   Usage     : abaqus cae noGUI=SOFC_Validation_Model_v2.py                   |
|                                                                              |
|   License   : MIT                                                            |
|                                                                              |
+==============================================================================+
"""

# ============================================================================
#  STANDARD LIBRARY & ABAQUS IMPORTS
# ============================================================================

from __future__ import print_function  # Python 2/3 compatibility
import sys
import os
import time
import traceback
import logging

from abaqus import *
from abaqusConstants import *
from caeModules import *          # Includes mesh, regionToolset, etc.
import mesh

# ============================================================================
#  SCRIPT METADATA
# ============================================================================

SCRIPT_NAME = 'SOFC Planar Cell Validation Model'
SCRIPT_VERSION = '2.4.0'
SCRIPT_MODIFIED = '2026-02-09'

# ============================================================================
#  LOGGING CONFIGURATION
# ============================================================================


def _ensure_dir(path):
    """Create directory if it does not exist (Py2/3 safe)."""
    if not os.path.exists(path):
        os.makedirs(path)


def _setup_logger():
    """Configure logging to file + stdout and return logger + log path."""
    log_dir = os.path.join(os.getcwd(), 'logs')
    _ensure_dir(log_dir)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, 'SOFC_model_{}.log'.format(timestamp))

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    root_logger = logging.getLogger('SOFC_Model')
    root_logger.setLevel(logging.DEBUG)
    root_logger.propagate = False
    if root_logger.handlers:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    root_logger.info('=' * 72)
    root_logger.info('%s -- Build Started', SCRIPT_NAME)
    root_logger.info('Version: %s | Modified: %s', SCRIPT_VERSION, SCRIPT_MODIFIED)
    root_logger.info('=' * 72)
    return root_logger, log_file


logger, LOG_FILE_PATH = _setup_logger()
try:
    logger.info('Abaqus release: %s', mdb.version)
except Exception:
    logger.info('Abaqus release: unknown')
logger.info('Python version: %s', sys.version.split()[0])

# ============================================================================
#  UTILITY FUNCTIONS
# ============================================================================


def _banner(title, width=72):
    """Print a formatted section banner to the log."""
    logger.info('')
    logger.info('-' * width)
    logger.info('  %s', title.upper())
    logger.info('-' * width)


def _validate_positive(value, name):
    """Raise ValueError if *value* is not a positive number."""
    if value is None or value <= 0.0:
        raise ValueError("Parameter '{}' must be > 0. Got: {}".format(name, value))


def _validate_range(value, lo, hi, name):
    """Raise ValueError if *value* falls outside [lo, hi]."""
    if not (lo <= value <= hi):
        raise ValueError(
            "Parameter '{}' = {} is outside valid range [{}, {}]".format(
                name, value, lo, hi
            )
        )


def _env_int(name, default):
    """Read integer from environment with a safe fallback."""
    raw = os.environ.get(name, None)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        logger.warning("Invalid env var %s=%r; using %s.", name, raw, default)
        return default


def _compute_anode_modulus_mpa(T_celsius, coefficients=None):
    """
    Evaluate temperature-dependent Young's modulus for Ni-YSZ anode.

    Polynomial fit from 'elastic_temperature_series_augmented.csv':
        E(T) = c0 + c1*T + c2*T^2   [GPa]

    Parameters
    ----------
    T_celsius : float
        Temperature in degree Celsius.
    coefficients : tuple of float, optional
        (c0, c1, c2) polynomial coefficients. Defaults to fitted values.

    Returns
    -------
    float
        Young's modulus in MPa (consistent with mm-N-s system).
    """
    if coefficients is None:
        coefficients = (56.29, -0.03570, 5.00e-6)  # GPa units
    c0, c1, c2 = coefficients
    E_GPa = c0 + c1 * T_celsius + c2 * T_celsius ** 2
    E_GPa = max(E_GPa, 1.0)  # Floor at 1 GPa to avoid negative values
    return E_GPa * 1.0e3  # Convert GPa -> MPa


def _edge_at(instance, coords, label):
    """
    Robust edge lookup with findAt and closest-edge fallback.
    Returns an EdgeArray for consistent downstream usage.
    """
    try:
        return instance.edges.findAt((coords,))
    except Exception:
        if hasattr(instance.edges, 'getClosest') and hasattr(instance.edges, 'sequenceFromLabels'):
            edge = instance.edges.getClosest(coordinates=(coords,))[0][0]
            logger.warning("findAt failed for %s at %s; using closest edge.", label, coords)
            return instance.edges.sequenceFromLabels((edge.label,))
        raise


def _vertex_at(instance, coords, label):
    """
    Robust vertex lookup with findAt and closest-vertex fallback.
    Returns a VertexArray for consistent downstream usage.
    """
    try:
        return instance.vertices.findAt((coords,))
    except Exception:
        if hasattr(instance.vertices, 'getClosest') and hasattr(instance.vertices, 'sequenceFromLabels'):
            vert = instance.vertices.getClosest(coordinates=(coords,))[0][0]
            logger.warning("findAt failed for %s at %s; using closest vertex.", label, coords)
            return instance.vertices.sequenceFromLabels((vert.label,))
        raise


def _create_tie_constraint(model, name, main_surface, secondary_surface, **kwargs):
    """
    Create a tie constraint with compatibility across Abaqus releases.
    Some versions expect master/slave, others main/secondary.
    """
    try:
        return model.Tie(name=name, master=main_surface, slave=secondary_surface, **kwargs)
    except TypeError as exc_master:
        try:
            return model.Tie(name=name, main=main_surface, secondary=secondary_surface, **kwargs)
        except TypeError:
            logger.error("Tie keyword mismatch. Original error: %s", exc_master)
            raise


# ============================================================================
#  1. CONFIGURATION -- PARAMETRIC INPUT BLOCK
# ============================================================================

_banner('Configuration & Input Parameters')

# -- 1.1  Model Metadata ------------------------------------------------------

MODEL_NAME = 'Validation_Planar_Cell'
JOB_NAME = 'Job-Validation-Cooling'
JOB_DESC = ('2D planar SOFC anode-electrolyte bilayer: '
            'residual stress upon cooling from sintering temperature.')

# -- 1.2  Material Names ------------------------------------------------------

MAT_NAME_ANODE = 'NiYSZ-Anode'
MAT_NAME_ELEC = 'YSZ8-Electrolyte'

# -- 1.3  Geometry (mm) -------------------------------------------------------

GEOM = {
    'L_cell': 10.0,          # mm -- half-width of cell
    'H_anode': 0.500,        # mm -- anode support thickness
    'H_electrolyte': 0.010,  # mm -- electrolyte thickness (10 um)
}

# -- 1.4  Thermal Loading (deg C) ---------------------------------------------

TEMP = {
    'T_sintering': 1300.0,   # deg C -- assumed stress-free reference
    'T_room': 25.0,          # deg C -- target / validation temperature
}

# -- 1.5  Material: Anode (Ni-YSZ) --------------------------------------------

ANODE_NU = 0.29
ANODE_E_TEMPERATURES = [
    25.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0,
    700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0
]

ANODE_E_TABLE = [
    (_compute_anode_modulus_mpa(T), ANODE_NU, T) for T in ANODE_E_TEMPERATURES
]

ANODE_CTE_TABLE = [
    (11.8e-6, 25.0),
    (12.0e-6, 200.0),
    (12.2e-6, 400.0),
    (12.5e-6, 600.0),
    (12.8e-6, 800.0),
    (13.0e-6, 1000.0),
    (13.2e-6, 1200.0),
    (13.3e-6, 1300.0),
]

ANODE_DENSITY = 6.87e-9  # tonne/mm^3  (equivalent to 6870 kg/m^3)

# -- 1.6  Material: Electrolyte (8YSZ) ----------------------------------------
#   WARNING: PLACEHOLDER -- replace with values from 'mat_electrolyte.csv'.

ELEC_E_TABLE = [
    (215.0e3, 0.31, 25.0),   # (E [MPa], nu, T [deg C])
    (210.0e3, 0.31, 200.0),
    (205.0e3, 0.31, 400.0),
    (200.0e3, 0.31, 600.0),
    (193.0e3, 0.31, 800.0),
    (185.0e3, 0.31, 1000.0),
    (176.0e3, 0.31, 1200.0),
    (170.0e3, 0.31, 1300.0),
]

ELEC_CTE_TABLE = [
    (10.3e-6, 25.0),
    (10.5e-6, 200.0),
    (10.7e-6, 400.0),
    (10.8e-6, 600.0),
    (11.0e-6, 800.0),
    (11.1e-6, 1000.0),
    (11.2e-6, 1200.0),
    (11.3e-6, 1300.0),
]

ELEC_DENSITY = 5.90e-9  # tonne/mm^3  (5900 kg/m^3)

# -- 1.7  Mesh Control --------------------------------------------------------

MESH = {
    'elem_code': CPE4R,      # Plane-strain, reduced integration
    'elem_code_tri': CPE3,   # Fallback triangular element
    'global_size': 0.25,     # mm -- default global seed
    'anode_bias': 3.0,       # Bias ratio toward interface
    'anode_min_div': 6,      # Minimum elements through anode thickness
    'elec_divisions': 4,     # Min elements through electrolyte
    'deviation': 0.05,
    'min_size_factor': 0.05,
}

# -- 1.8  Solver Options ------------------------------------------------------

SOLVER = {
    'nlgeom': ON,
    'init_inc': 0.05,
    'min_inc': 1e-10,
    'max_inc': 0.25,
    'max_num_inc': 500,
    'stabilize': False,
    'stabilize_mag': 2e-4,
}

# -- 1.9  Output Requests -----------------------------------------------------

FIELD_OUTPUTS = ('S', 'E', 'U', 'NT', 'RF', 'TEMP')

# -- 1.10  Run Options --------------------------------------------------------

RUN = {
    'create_job': True,
    'do_data_check': False,
    'auto_submit': False,
    'save_cae': True,
}

# -- 1.11  Resources ----------------------------------------------------------

JOB_CPUS = max(1, _env_int('SOFC_CPUS', 4))
JOB_DOMAINS = max(1, _env_int('SOFC_DOMAINS', JOB_CPUS))

# -- 1.12  Validate Inputs ----------------------------------------------------

for _key, _val in GEOM.items():
    _validate_positive(_val, _key)

_validate_range(TEMP['T_sintering'], 500.0, 2000.0, 'T_sintering')
_validate_range(TEMP['T_room'], -273.15, TEMP['T_sintering'], 'T_room')
_validate_positive(MESH['global_size'], 'MESH.global_size')
_validate_positive(MESH['elec_divisions'], 'MESH.elec_divisions')

logger.info('All input parameters validated successfully.')
for _key, _val in GEOM.items():
    logger.info('  GEOM.%s = %s', _key, _val)
for _key, _val in TEMP.items():
    logger.info('  TEMP.%s = %s', _key, _val)
logger.info('  MAT_NAME_ANODE = %s', MAT_NAME_ANODE)
logger.info('  MAT_NAME_ELEC  = %s', MAT_NAME_ELEC)

logger.warning('Electrolyte properties flagged as placeholders.')

# ============================================================================
#  2. MODEL INITIALISATION
# ============================================================================

_banner('Model Initialisation')

if MODEL_NAME in mdb.models:
    logger.warning("Existing model '%s' found -- deleting.", MODEL_NAME)
    del mdb.models[MODEL_NAME]

myModel = mdb.Model(name=MODEL_NAME, modelType=STANDARD_EXPLICIT)
logger.info("Model '%s' created.", MODEL_NAME)

# ============================================================================
#  3. MATERIAL DEFINITIONS
# ============================================================================

_banner('Material Definitions')

mat_anode = myModel.Material(name=MAT_NAME_ANODE)
mat_anode.Elastic(
    type=ISOTROPIC,
    temperatureDependency=ON,
    table=tuple(tuple(row) for row in ANODE_E_TABLE),
)
mat_anode.Expansion(
    type=ISOTROPIC,
    temperatureDependency=ON,
    zero=TEMP['T_sintering'],
    table=tuple(ANODE_CTE_TABLE),
)
mat_anode.Density(table=((ANODE_DENSITY,),))
logger.info('Material defined: %s (T-dep E, T-dep CTE)', MAT_NAME_ANODE)

mat_elec = myModel.Material(name=MAT_NAME_ELEC)
mat_elec.Elastic(
    type=ISOTROPIC,
    temperatureDependency=ON,
    table=tuple(tuple(row) for row in ELEC_E_TABLE),
)
mat_elec.Expansion(
    type=ISOTROPIC,
    temperatureDependency=ON,
    zero=TEMP['T_sintering'],
    table=tuple(ELEC_CTE_TABLE),
)
mat_elec.Density(table=((ELEC_DENSITY,),))
logger.info('Material defined: %s (T-dep E, T-dep CTE)', MAT_NAME_ELEC)

# ============================================================================
#  4. SECTION DEFINITIONS
# ============================================================================

_banner('Section Definitions')

SECTION_ANODE = 'Section-Anode'
SECTION_ELEC = 'Section-Electrolyte'

myModel.HomogeneousSolidSection(
    name=SECTION_ANODE,
    material=MAT_NAME_ANODE,
    thickness=None,
)
myModel.HomogeneousSolidSection(
    name=SECTION_ELEC,
    material=MAT_NAME_ELEC,
    thickness=None,
)
logger.info('Sections created: %s, %s', SECTION_ANODE, SECTION_ELEC)

# ============================================================================
#  5. PART CREATION & SECTION ASSIGNMENT
# ============================================================================

_banner('Part Creation')

L = GEOM['L_cell']
Ha = GEOM['H_anode']
He = GEOM['H_electrolyte']

s_anode = myModel.ConstrainedSketch(name='__profile_anode__', sheetSize=2.0 * L)
s_anode.rectangle(point1=(0.0, 0.0), point2=(L, Ha))

p_anode = myModel.Part(
    name='Part-Anode',
    dimensionality=TWO_D_PLANAR,
    type=DEFORMABLE_BODY,
)
p_anode.BaseShell(sketch=s_anode)
del s_anode

region_anode = p_anode.Set(name='Set-All-Anode', faces=p_anode.faces[:])
p_anode.SectionAssignment(
    region=region_anode,
    sectionName=SECTION_ANODE,
    offset=0.0,
    offsetType=MIDDLE_SURFACE,
    offsetField='',
    thicknessAssignment=FROM_SECTION,
)
logger.info('Part-Anode created: %.3f x %.3f mm', L, Ha)

s_elec = myModel.ConstrainedSketch(name='__profile_elec__', sheetSize=2.0 * L)
s_elec.rectangle(point1=(0.0, 0.0), point2=(L, He))

p_elec = myModel.Part(
    name='Part-Electrolyte',
    dimensionality=TWO_D_PLANAR,
    type=DEFORMABLE_BODY,
)
p_elec.BaseShell(sketch=s_elec)
del s_elec

region_elec = p_elec.Set(name='Set-All-Electrolyte', faces=p_elec.faces[:])
p_elec.SectionAssignment(
    region=region_elec,
    sectionName=SECTION_ELEC,
    offset=0.0,
    offsetType=MIDDLE_SURFACE,
    offsetField='',
    thicknessAssignment=FROM_SECTION,
)
logger.info('Part-Electrolyte created: %.3f x %.4f mm', L, He)

# ============================================================================
#  6. ASSEMBLY
# ============================================================================

_banner('Assembly')

rootAsm = myModel.rootAssembly
rootAsm.DatumCsysByDefault(CARTESIAN)

inst_anode = rootAsm.Instance(name='Anode-1', part=p_anode, dependent=ON)
inst_elec = rootAsm.Instance(name='Electrolyte-1', part=p_elec, dependent=ON)
rootAsm.translate(instanceList=('Electrolyte-1',), vector=(0.0, Ha, 0.0))

logger.info('Instances positioned: Electrolyte-1 translated by (0, %.3f, 0)', Ha)

# ============================================================================
#  7. INTERACTION -- TIE CONSTRAINT (PERFECT BONDING)
# ============================================================================

_banner('Interactions')

edge_elec_bottom = _edge_at(inst_elec, (L / 2.0, Ha, 0.0), 'Electrolyte bottom edge')
edge_anode_top = _edge_at(inst_anode, (L / 2.0, Ha, 0.0), 'Anode top edge')

surf_master = rootAsm.Surface(side1Edges=edge_elec_bottom, name='Surf-ElecBottom')
surf_slave = rootAsm.Surface(side1Edges=edge_anode_top, name='Surf-AnodeTop')

_create_tie_constraint(
    model=myModel,
    name='Tie-Interface',
    main_surface=surf_master,
    secondary_surface=surf_slave,
    positionToleranceMethod=COMPUTED,
    adjust=ON,
    tieRotations=ON,
    thickness=ON,
    constraintEnforcement=SURFACE_TO_SURFACE,
)
logger.info('Tie constraint created: Surf-ElecBottom <-> Surf-AnodeTop')

# ============================================================================
#  8. ANALYSIS STEP
# ============================================================================

_banner('Step Definition')

step_name = 'Step-Cooling'
myModel.StaticStep(
    name=step_name,
    previous='Initial',
    timePeriod=1.0,
    initialInc=SOLVER['init_inc'],
    minInc=SOLVER['min_inc'],
    maxInc=SOLVER['max_inc'],
    maxNumInc=SOLVER['max_num_inc'],
    nlgeom=SOLVER['nlgeom'],
    description='Cool from T_sinter ({} C) to T_room ({} C)'.format(
        TEMP['T_sintering'], TEMP['T_room']
    ),
)

if SOLVER['stabilize']:
    myModel.steps[step_name].setValues(
        stabilizationMagnitude=SOLVER['stabilize_mag'],
        stabilizationMethod=DISSIPATED_ENERGY_FRACTION,
        adaptiveDampingRatio=0.05,
        continueDampingFactors=False,
    )
    logger.info('Adaptive stabilisation enabled (magnitude=%s)', SOLVER['stabilize_mag'])

logger.info("Step '%s' created -- NLGeom=%s", step_name, SOLVER['nlgeom'])

# ============================================================================
#  9. FIELD OUTPUT REQUESTS
# ============================================================================

_banner('Output Requests')

if 'F-Output-1' in myModel.fieldOutputRequests:
    del myModel.fieldOutputRequests['F-Output-1']

myModel.FieldOutputRequest(
    name='FOReq-All',
    createStepName=step_name,
    variables=FIELD_OUTPUTS,
    frequency=LAST_INCREMENT,
)

edge_anode_top_set = rootAsm.Set(
    edges=edge_anode_top,
    name='Set-AnodeInterface',
)
myModel.FieldOutputRequest(
    name='FOReq-Interface',
    createStepName=step_name,
    variables=('S', 'E', 'U', 'TEMP'),
    region=edge_anode_top_set,
    frequency=1,
)

myModel.HistoryOutputRequest(
    name='HOReq-InterfaceMid',
    createStepName=step_name,
    variables=('S11', 'S22', 'S12', 'E11', 'E22', 'U1', 'U2'),
    region=edge_anode_top_set,
    frequency=1,
)

logger.info('Field output: %s', ', '.join(FIELD_OUTPUTS))
logger.info('History output configured at anode-electrolyte interface.')

# ============================================================================
#  10. BOUNDARY CONDITIONS
# ============================================================================

_banner('Boundary Conditions')

edges_symm_anode = _edge_at(inst_anode, (0.0, Ha / 2.0, 0.0), 'Symm anode edge')
edges_symm_elec = _edge_at(inst_elec, (0.0, Ha + He / 2.0, 0.0), 'Symm elec edge')
edges_symm = edges_symm_anode + edges_symm_elec

region_symm = rootAsm.Set(edges=edges_symm, name='Set-SymmX')
myModel.XsymmBC(
    name='BC-SymmetryX',
    createStepName='Initial',
    region=region_symm,
    localCsys=None,
)
logger.info('BC: X-symmetry applied at x = 0 (anode + electrolyte).')

vert_pin = _vertex_at(inst_anode, (0.0, 0.0, 0.0), 'Pin vertex')
region_pin = rootAsm.Set(vertices=vert_pin, name='Set-PinY')
myModel.DisplacementBC(
    name='BC-PinY',
    createStepName='Initial',
    region=region_pin,
    u1=UNSET,
    u2=0.0,
    ur3=UNSET,
    amplitude=UNSET,
    distributionType=UNIFORM,
    fieldName='',
    localCsys=None,
)
logger.info('BC: Y-pin at vertex (0, 0).')

# ============================================================================
#  11. THERMAL PREDEFINED FIELDS
# ============================================================================

_banner('Thermal Loading')

all_faces_anode = inst_anode.faces[:]
all_faces_elec = inst_elec.faces[:]

set_all = rootAsm.Set(
    faces=all_faces_anode + all_faces_elec,
    name='Set-AllCells',
)

myModel.Temperature(
    name='PF-InitialTemp',
    createStepName='Initial',
    region=set_all,
    distributionType=UNIFORM,
    crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
    magnitudes=(TEMP['T_sintering'],),
)
logger.info('Predefined field: T_initial = %s C (stress-free reference)', TEMP['T_sintering'])

myModel.Temperature(
    name='PF-CoolDown',
    createStepName=step_name,
    region=set_all,
    distributionType=UNIFORM,
    crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
    magnitudes=(TEMP['T_room'],),
)
logger.info('Predefined field: T_final = %s C', TEMP['T_room'])
logger.info('  dT = %s C', TEMP['T_room'] - TEMP['T_sintering'])

# ============================================================================
#  12. MESHING
# ============================================================================

_banner('Mesh Generation')

elemQuad = mesh.ElemType(
    elemCode=MESH['elem_code'],
    elemLibrary=STANDARD,
    secondOrderAccuracy=OFF,
    hourglassControl=ENHANCED,
    distortionControl=DEFAULT,
)
elemTri = mesh.ElemType(
    elemCode=MESH['elem_code_tri'],
    elemLibrary=STANDARD,
)

for part, label in [(p_anode, 'Anode'), (p_elec, 'Electrolyte')]:
    part.setElementType(
        regions=(part.faces[:],),
        elemTypes=(elemQuad, elemTri),
    )
    logger.info('%s: element type set to %s / %s', label, MESH['elem_code'], MESH['elem_code_tri'])

p_anode.seedPart(
    size=MESH['global_size'],
    deviationFactor=MESH['deviation'],
    minSizeFactor=MESH['min_size_factor'],
)

anode_vert_edges = p_anode.edges.findAt(
    ((0.0, Ha / 2.0, 0.0),),
    ((L, Ha / 2.0, 0.0),),
)
anode_divisions = max(
    int(round(Ha / MESH['global_size'] * 2.0)),
    MESH['anode_min_div'],
)
p_anode.seedEdgeByBias(
    biasMethod=SINGLE,
    end1Edges=anode_vert_edges,
    ratio=MESH['anode_bias'],
    number=anode_divisions,
    constraint=FINER,
)
logger.info('Anode: biased seeding applied (ratio=%s, div=%s)', MESH['anode_bias'], anode_divisions)

elec_vert_edges = p_elec.edges.findAt(
    ((0.0, He / 2.0, 0.0),),
    ((L, He / 2.0, 0.0),),
)
p_elec.seedEdgeByNumber(
    edges=elec_vert_edges,
    number=max(MESH['elec_divisions'], 2),
    constraint=FINER,
)
p_elec.seedPart(
    size=MESH['global_size'],
    deviationFactor=MESH['deviation'],
    minSizeFactor=MESH['min_size_factor'],
)
logger.info('Electrolyte: %s elements through thickness', MESH['elec_divisions'])

for part, label in [(p_anode, 'Anode'), (p_elec, 'Electrolyte')]:
    part.setMeshControls(
        regions=part.faces[:],
        elemShape=QUAD,
        technique=STRUCTURED,
    )
    logger.info('%s: structured quad mesh control set.', label)

p_anode.generateMesh()
p_elec.generateMesh()

n_anode_elem = len(p_anode.elements)
n_anode_node = len(p_anode.nodes)
n_elec_elem = len(p_elec.elements)
n_elec_node = len(p_elec.nodes)

logger.info('Anode mesh    : %6d elements, %6d nodes', n_anode_elem, n_anode_node)
logger.info('Electrolyte   : %6d elements, %6d nodes', n_elec_elem, n_elec_node)
logger.info('Total         : %6d elements, %6d nodes',
            n_anode_elem + n_elec_elem, n_anode_node + n_elec_node)

# ============================================================================
#  13. JOB CREATION
# ============================================================================

if RUN['create_job']:
    _banner('Job Definition')

    mdb.Job(
        name=JOB_NAME,
        model=MODEL_NAME,
        description=JOB_DESC,
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
        numCpus=JOB_CPUS,
        numDomains=JOB_DOMAINS,
        numGPUs=0,
    )
    logger.info("Job '%s' created -- CPUs=%s, Domains=%s, NLGeom=%s",
                JOB_NAME, JOB_CPUS, JOB_DOMAINS, SOLVER['nlgeom'])

    if RUN['do_data_check']:
        _banner('Data Check')
        mdb.jobs[JOB_NAME].submit(consistencyChecking=OFF)
        mdb.jobs[JOB_NAME].waitForCompletion()

    if RUN['auto_submit']:
        _banner('Job Submission')
        mdb.jobs[JOB_NAME].submit(consistencyChecking=OFF)
        mdb.jobs[JOB_NAME].waitForCompletion()

# ============================================================================
#  14. SAVE MODEL DATABASE
# ============================================================================

if RUN['save_cae']:
    _banner('Save')
    mdb_path = os.path.join(os.getcwd(), '{}.cae'.format(MODEL_NAME))
    mdb.saveAs(pathName=mdb_path)
    logger.info("Model database saved: '%s'", mdb_path)

# ============================================================================
#  15. BUILD SUMMARY
# ============================================================================

_banner('Build Summary')

summary_lines = [
    '',
    '+----------------------------------------------------------------------+',
    '|                      MODEL BUILD SUMMARY                            |',
    '+----------------------------------------------------------------------+',
    '|  Model         : {:<50s}|'.format(MODEL_NAME),
    '|  Job           : {:<50s}|'.format(JOB_NAME),
    '+----------------------------------------------------------------------+',
    '|  GEOMETRY                                                           |',
    '|    Half-width  : {:>10.3f} mm                                       |'.format(L),
    '|    Anode  (H)  : {:>10.3f} mm                                       |'.format(Ha),
    '|    Electrolyte : {:>10.4f} mm  ({:.1f} um)                          |'.format(He, He * 1000.0),
    '+----------------------------------------------------------------------+',
    '|  THERMAL LOAD                                                       |',
    '|    T_sinter    : {:>10.1f} C   (stress-free reference)              |'.format(TEMP['T_sintering']),
    '|    T_room      : {:>10.1f} C   (validation target)                  |'.format(TEMP['T_room']),
    '|    dT          : {:>10.1f} C                                        |'.format(TEMP['T_room'] - TEMP['T_sintering']),
    '+----------------------------------------------------------------------+',
    '|  MESH                                                               |',
    '|    Anode       : {:>6d} elements  |  {:>6d} nodes                   |'.format(n_anode_elem, n_anode_node),
    '|    Electrolyte : {:>6d} elements  |  {:>6d} nodes                   |'.format(n_elec_elem, n_elec_node),
    '|    Total       : {:>6d} elements  |  {:>6d} nodes                   |'.format(n_anode_elem + n_elec_elem, n_anode_node + n_elec_node),
    '|    Element     : {:<50s}|'.format(str(MESH['elem_code'])),
    '+----------------------------------------------------------------------+',
    '|  MATERIALS                                                          |',
    '|    Anode       : {:<50s}|'.format(MAT_NAME_ANODE),
    '|                  (T-dep E, T-dep CTE, rho={:.2e})                   |'.format(ANODE_DENSITY),
    '|    Electrolyte : {:<50s}|'.format(MAT_NAME_ELEC),
    '|                  (T-dep E, T-dep CTE, rho={:.2e})                   |'.format(ELEC_DENSITY),
    '+----------------------------------------------------------------------+',
    '|  STATUS        : READY TO SUBMIT                                    |',
    '|  Log file      : {:<50s}|'.format(os.path.basename(LOG_FILE_PATH)),
    '+----------------------------------------------------------------------+',
    '',
]

for line in summary_lines:
    logger.info(line)

logger.info('')
logger.info('=' * 72)
logger.info('  BUILD COMPLETE -- Model is ready for submission.')
logger.info('  To submit:  mdb.jobs["%s"].submit()', JOB_NAME)
logger.info('=' * 72)

