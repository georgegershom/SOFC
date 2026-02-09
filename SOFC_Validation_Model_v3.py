# -*- coding: utf-8 -*-
"""
+==============================================================================+
|                                                                              |
|       ███████╗ ██████╗ ███████╗ ██████╗                                      |
|       ██╔════╝██╔═══██╗██╔════╝██╔════╝                                      |
|       ███████╗██║   ██║█████╗  ██║                                           |
|       ╚════██║██║   ██║██╔══╝  ██║                                           |
|       ███████║╚██████╔╝██║     ╚██████╗                                      |
|       ╚══════╝ ╚═════╝ ╚═╝      ╚═════╝                                      |
|                                                                              |
+==============================================================================+
|                                                                              |
|   Abaqus/Standard Script                                                     |
|   2D Planar SOFC Residual Stress Validation Model                            |
|                                                                              |
+------------------------------------------------------------------------------+
|                                                                              |
|   Purpose   : High-fidelity thermo-mechanical simulation of a Ni-YSZ        |
|               anode / 8YSZ electrolyte bilayer subjected to cooling from      |
|               co-sintering temperature (1300 C) to ambient (25 C).           |
|               Predicts in-plane residual stress distribution arising from     |
|               coefficient of thermal expansion (CTE) mismatch.              |
|                                                                              |
|   Validation: Synchrotron X-ray diffraction lattice-strain measurements      |
|               performed at the Diamond Light Source (I12 beamline).           |
|               Target validation metric: anode in-plane stress sigma_11       |
|               at the bilayer interface.                                       |
|                                                                              |
|   Physics   : - Temperature-dependent elastic moduli (polynomial fit)        |
|               - Temperature-dependent CTE (piecewise linear, lattice data)   |
|               - Geometric nonlinearity (large deformation)                   |
|               - Perfect bonding (tie constraint) at interface                |
|               - Plane-strain assumption (CPE4R elements)                     |
|                                                                              |
+------------------------------------------------------------------------------+
|                                                                              |
|   Author    : Gershom George                                                 |
|   Affiliation: SOFC Research Group                                           |
|   Version   : 3.0.0                                                          |
|   Created   : 2026-02-08                                                     |
|   Modified  : 2026-02-09                                                     |
|                                                                              |
|   Reference : George, G. et al. "Mechanical characterisation of Ni-YSZ      |
|               anode-supported half-cells via synchrotron X-ray               |
|               diffraction." J. Power Sources (2026).                         |
|                                                                              |
|   Usage     : abaqus cae noGUI=SOFC_Validation_Model_v3.py                  |
|                                                                              |
|   License   : MIT                                                            |
|                                                                              |
+==============================================================================+

Changelog
---------
v3.0.0  2026-02-09  Major refactor -- class-based configuration, Abaqus API
                     version compatibility layer, mesh quality diagnostics,
                     analytical Timoshenko bi-strip validation, automated
                     post-processing & CSV export, comprehensive error
                     handling and profiling.
v2.3.0  2026-02-08  Initial parametric script with logging.
"""

# ============================================================================
#                     STANDARD LIBRARY & ABAQUS IMPORTS
# ============================================================================

from __future__ import print_function, division   # Python 2/3 compatibility
import sys
import os
import time
import math
import traceback
import logging
import functools
import collections

# -- Abaqus kernel -----------------------------------------------------------
from abaqus import *
from abaqusConstants import *
from caeModules import *
import mesh
import regionToolset

# ============================================================================
#                         GLOBAL CONSTANTS & DEFAULTS
# ============================================================================

__version__  = '3.0.0'
__author__   = 'Gershom George'
__email__    = 'gershom.george@sofc-research.ac.uk'
__license__  = 'MIT'

_SCRIPT_NAME = os.path.basename(__file__) if '__file__' in dir() else 'SOFC_Model'
_START_WALL  = time.time()
_START_CPU   = time.clock() if hasattr(time, 'clock') else time.process_time()


# ============================================================================
#                           LOGGING CONFIGURATION
# ============================================================================

def _setup_logging(log_dir=None, level=logging.DEBUG):
    """
    Configure dual-output logging (file + console).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    str
        Absolute path to the log file.
    """
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), 'logs')
    #
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    #
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file  = os.path.join(log_dir, 'SOFC_model_{}.log'.format(timestamp))
    #
    # Formatter
    fmt = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    #
    # File handler -- captures everything
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    #
    # Console handler -- INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    #
    logger = logging.getLogger('SOFC_Model')
    logger.setLevel(level)
    #
    # Prevent duplicate handlers on re-import
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    #
    return logger, log_file


logger, _LOG_FILE = _setup_logging()

# Opening banner
logger.info('')
logger.info('+' + '=' * 70 + '+')
logger.info('|{:^70s}|'.format(''))
logger.info('|{:^70s}|'.format('SOFC PLANAR CELL VALIDATION MODEL'))
logger.info('|{:^70s}|'.format('Abaqus/Standard -- Thermo-Mechanical Analysis'))
logger.info('|{:^70s}|'.format(''))
logger.info('|{:^70s}|'.format('Version {} -- {}'.format(
    __version__, time.strftime('%Y-%m-%d %H:%M:%S'))))
logger.info('|{:^70s}|'.format(''))
logger.info('+' + '=' * 70 + '+')
logger.info('')


# ============================================================================
#                           UTILITY FUNCTIONS
# ============================================================================

def banner(title, width=72, char='-'):
    """Print a formatted section banner to the log."""
    logger.info('')
    logger.info(char * width)
    logger.info('  {} '.format(title.upper()))
    logger.info(char * width)


def sub_banner(title, width=72, char='.'):
    """Print a formatted sub-section banner."""
    logger.info('')
    logger.info('  {} {}'.format(char * 3, title))


def elapsed():
    """Return formatted wall-clock elapsed time since script start."""
    dt = time.time() - _START_WALL
    m, s = divmod(dt, 60)
    return '{:02.0f}:{:05.2f}'.format(m, s)


def timer(func):
    """Decorator that logs the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        dt = time.time() - t0
        logger.debug('  [TIMER] {}() completed in {:.3f} s'.format(
            func.__name__, dt))
        return result
    return wrapper


def validate_positive(value, name):
    """Raise ValueError if *value* is not a positive number."""
    if value is None or value <= 0.0:
        raise ValueError(
            "Parameter '{}' must be strictly positive. Got: {}".format(name, value))


def validate_range(value, lo, hi, name):
    """Raise ValueError if *value* falls outside [lo, hi]."""
    if not (lo <= value <= hi):
        raise ValueError(
            "Parameter '{}' = {} outside valid range [{}, {}]".format(
                name, value, lo, hi))


def safe_delete_model(model_name):
    """Delete an existing model from the MDB, if present."""
    if model_name in mdb.models:
        logger.warning("Model '{}' already exists -- replacing.".format(model_name))
        del mdb.models[model_name]


# ============================================================================
#                    ABAQUS VERSION COMPATIBILITY LAYER
# ============================================================================

def _detect_abaqus_version():
    """
    Detect the running Abaqus version and return (major, minor, patch).

    Returns
    -------
    tuple of int
        (major, minor, patch), e.g. (2024, 0, 0).
    str
        Human-readable version string.
    """
    try:
        ver_str = session.journalOptions.setValues  # dummy -- just to check
        # The Abaqus version is typically in the `version` attribute
        raw = str(getattr(session, 'version', '0.0-0'))
        parts = raw.replace('-', '.').split('.')
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        return (major, minor, patch), raw
    except Exception:
        return (2024, 0, 0), 'unknown (assumed 2024)'


_ABAQUS_VERSION, _ABAQUS_VERSION_STR = _detect_abaqus_version()
logger.info('Abaqus version detected: {}'.format(_ABAQUS_VERSION_STR))

# Terminology changed in Abaqus 2022:  master/slave -> main/secondary
_USE_MAIN_SECONDARY = _ABAQUS_VERSION[0] >= 2022


def create_tie_constraint(model, name, main_surface, secondary_surface,
                          **kwargs):
    """
    Create a Tie constraint with API-version-aware keyword arguments.

    In Abaqus >= 2022 the keywords 'master'/'slave' were renamed to
    'main'/'secondary'.  This wrapper handles both transparently.

    Parameters
    ----------
    model : Model
        Abaqus model object.
    name : str
        Constraint name.
    main_surface : Surface
        The main (formerly master) surface.
    secondary_surface : Surface
        The secondary (formerly slave) surface.
    **kwargs
        Additional keyword arguments forwarded to ``model.Tie()``.

    Returns
    -------
    Constraint
        The created Tie constraint object.
    """
    tie_kw = dict(name=name)
    tie_kw.update(kwargs)
    #
    if _USE_MAIN_SECONDARY:
        tie_kw['main']      = main_surface
        tie_kw['secondary'] = secondary_surface
        logger.debug('  Using Abaqus >= 2022 Tie API (main/secondary)')
    else:
        tie_kw['master'] = main_surface
        tie_kw['slave']  = secondary_surface
        logger.debug('  Using Abaqus < 2022 Tie API (master/slave)')
    #
    try:
        constraint = model.Tie(**tie_kw)
    except TypeError:
        # Fallback: try the opposite keyword set
        logger.warning('Tie keyword mismatch detected -- attempting fallback.')
        if 'main' in tie_kw:
            tie_kw['master'] = tie_kw.pop('main')
            tie_kw['slave']  = tie_kw.pop('secondary')
        else:
            tie_kw['main']      = tie_kw.pop('master')
            tie_kw['secondary'] = tie_kw.pop('slave')
        constraint = model.Tie(**tie_kw)
    #
    return constraint


# ============================================================================
#                    MATERIAL PROPERTY FUNCTIONS
# ============================================================================

def compute_anode_modulus(T_celsius, coefficients=None):
    """
    Temperature-dependent Young's modulus for Ni-YSZ cermet anode.

    Quadratic polynomial fit from experimentally measured data
    ('elastic_temperature_series_augmented.csv'):

        E(T) = c0 + c1*T + c2*T^2   [GPa]

    The fit accounts for Ni softening at elevated temperatures and
    progressive load transfer to the YSZ backbone.

    Parameters
    ----------
    T_celsius : float
        Temperature in degree Celsius.
    coefficients : tuple of float, optional
        (c0, c1, c2) polynomial coefficients in GPa.
        Default: (56.29, -0.03570, 5.00e-6).

    Returns
    -------
    float
        Young's modulus in Pa.

    Notes
    -----
    A floor of 1 GPa is enforced to prevent numerical singularity
    in the Abaqus solver at temperatures near the solidus.
    """
    if coefficients is None:
        coefficients = (56.29, -0.03570, 5.00e-6)
    c0, c1, c2 = coefficients
    E_GPa = c0 + c1 * T_celsius + c2 * T_celsius ** 2
    return max(E_GPa, 1.0) * 1.0e9   # GPa -> Pa, floor at 1 GPa


def compute_electrolyte_modulus(T_celsius, E0=215.0, slope=-0.035):
    """
    Temperature-dependent Young's modulus for 8 mol% YSZ electrolyte.

    Linear approximation (adequate for the validation temperature range):

        E(T) = E0 + slope * T   [GPa]

    Parameters
    ----------
    T_celsius : float
        Temperature in degree Celsius.
    E0 : float
        Room-temperature modulus in GPa.
    slope : float
        Linear degradation rate in GPa/C.

    Returns
    -------
    float
        Young's modulus in Pa.
    """
    E_GPa = E0 + slope * T_celsius
    return max(E_GPa, 10.0) * 1.0e9


def analytical_bilayer_stress(E1, E2, alpha1, alpha2, h1, h2, nu1, nu2, dT):
    """
    Timoshenko bimetallic-strip analytical solution for residual stress.

    Computes the uniform in-plane stress in each layer of a perfectly
    bonded bilayer strip cooled by dT, assuming plane-strain conditions.

    This serves as an independent verification of the FE solution.

    Parameters
    ----------
    E1, E2 : float
        Young's modulus of layer 1 (anode) and layer 2 (electrolyte) [Pa].
    alpha1, alpha2 : float
        CTE of layer 1 and layer 2 [1/K].
    h1, h2 : float
        Thickness of layer 1 and layer 2 [mm].
    nu1, nu2 : float
        Poisson's ratio of layer 1 and layer 2.
    dT : float
        Temperature change (negative for cooling) [K or C].

    Returns
    -------
    dict
        {'sigma_1': float, 'sigma_2': float, 'curvature': float}
        Stresses in Pa, curvature in 1/mm.

    References
    ----------
    [1] Timoshenko, S. "Analysis of Bi-Metal Thermostats."
        J. Opt. Soc. Am., 11(3), 233-255, 1925.
    [2] Hsueh, C.H. "Modelling of elastic deformation of multilayers
        due to residual stresses and external bending."
        J. Appl. Phys., 91(12), 9652-9656, 2002.
    """
    # Plane-strain effective moduli
    Ep1 = E1 / (1.0 - nu1 ** 2)
    Ep2 = E2 / (1.0 - nu2 ** 2)
    #
    # Effective CTE under plane-strain
    a1_eff = alpha1 * (1.0 + nu1)
    a2_eff = alpha2 * (1.0 + nu2)
    #
    # Stiffness ratio and thickness ratio
    n = Ep2 / Ep1   # modulus ratio
    m = h2 / h1     # thickness ratio
    #
    # Misfit strain
    d_alpha = a1_eff - a2_eff
    epsilon_mismatch = d_alpha * dT
    #
    # Force per unit width to enforce compatibility (Timoshenko approach)
    denominator = (
        1.0 / (Ep1 * h1) + 1.0 / (Ep2 * h2) +
        (h1 + h2) ** 2 / (4.0 * (Ep1 * h1 ** 3 / 12.0 + Ep2 * h2 ** 3 / 12.0))
    )
    #
    # Handle near-zero denominator
    if abs(denominator) < 1.0e-30:
        return {'sigma_1': 0.0, 'sigma_2': 0.0, 'curvature': 0.0}
    #
    F = epsilon_mismatch / denominator
    #
    # Layer stresses (uniform through thickness, approximate)
    sigma_1 = -F / h1   # anode (thicker, tension if CTE_anode > CTE_elec)
    sigma_2 =  F / h2   # electrolyte
    #
    # Curvature
    I_eff = Ep1 * h1 ** 3 / 12.0 + Ep2 * h2 ** 3 / 12.0
    kappa = F * (h1 + h2) / (2.0 * I_eff) if abs(I_eff) > 1.0e-30 else 0.0
    #
    return {
        'sigma_1'  : sigma_1,
        'sigma_2'  : sigma_2,
        'curvature': kappa,
    }


# ============================================================================
#                       CONFIGURATION DATA CLASS
# ============================================================================

class SOFCModelConfig(object):
    """
    Centralised configuration container for the SOFC validation model.

    All geometric, material, mesh, solver, and output parameters are
    consolidated here to provide a single source of truth and enable
    parametric studies.

    Attributes
    ----------
    model_name : str
    job_name : str
    job_desc : str
    mat_anode : str
    mat_elec : str
    geom : dict
    temp : dict
    anode : dict
    elec : dict
    mesh_cfg : dict
    solver : dict
    field_outputs : tuple
    ncpus : int
    """

    def __init__(self):
        # -- Model metadata ---------------------------------------------------
        self.model_name = 'Validation_Planar_Cell'
        self.job_name   = 'Job-Validation-Cooling'
        self.job_desc   = (
            '2D planar SOFC anode-electrolyte bilayer: residual stress upon '
            'cooling from co-sintering temperature.  Validated against '
            'synchrotron XRD lattice-strain measurements.'
        )
        #
        # -- Material identifiers ---------------------------------------------
        self.mat_anode = 'NiYSZ-Anode'
        self.mat_elec  = 'YSZ8-Electrolyte'
        #
        # -- Geometry (mm) -- half-model using symmetry about x = 0 ----------
        self.geom = {
            'L_cell'        : 10.0,     # mm  half-width of planar cell
            'H_anode'       : 0.500,    # mm  anode support layer thickness
            'H_electrolyte' : 0.010,    # mm  electrolyte thickness (10 um)
        }
        #
        # -- Thermal loading (deg C) -----------------------------------------
        self.temp = {
            'T_sintering' : 1300.0,     # deg C  stress-free reference
            'T_room'      :   25.0,     # deg C  target / validation point
        }
        #
        # -- Material: Anode (Ni-YSZ) ----------------------------------------
        #    E(T) polynomial coefficients: (c0, c1, c2) in GPa
        #    CTE from synchrotron lattice-strain regression
        self.anode = {
            'nu'          : 0.29,
            'density'     : 6.87e-9,    # tonne/mm^3  (6870 kg/m^3)
            'E_poly'      : (56.29, -0.03570, 5.00e-6),
            'T_points'    : [25.0, 100.0, 200.0, 300.0, 400.0, 500.0,
                             600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0,
                             1200.0, 1300.0],
            'cte_table'   : [
                (11.8e-6,  25.0),
                (12.0e-6, 200.0),
                (12.2e-6, 400.0),
                (12.5e-6, 600.0),
                (12.8e-6, 800.0),
                (13.0e-6, 1000.0),
                (13.2e-6, 1200.0),
                (13.3e-6, 1300.0),
            ],
        }
        #
        # -- Material: Electrolyte (8 mol% YSZ) ------------------------------
        self.elec = {
            'nu'          : 0.31,
            'density'     : 5.90e-9,    # tonne/mm^3  (5900 kg/m^3)
            'E_table'     : [
                (215.0e3, 0.31,   25.0),   # (E [MPa], nu, T [deg C])
                (210.0e3, 0.31,  200.0),
                (205.0e3, 0.31,  400.0),
                (200.0e3, 0.31,  600.0),
                (193.0e3, 0.31,  800.0),
                (185.0e3, 0.31, 1000.0),
                (176.0e3, 0.31, 1200.0),
                (170.0e3, 0.31, 1300.0),
            ],
            'cte_table'   : [
                (10.3e-6,   25.0),
                (10.5e-6,  200.0),
                (10.7e-6,  400.0),
                (10.8e-6,  600.0),
                (11.0e-6,  800.0),
                (11.1e-6, 1000.0),
                (11.2e-6, 1200.0),
                (11.3e-6, 1300.0),
            ],
        }
        #
        # -- Mesh configuration -----------------------------------------------
        self.mesh_cfg = {
            'elem_code'      : CPE4R,
            'elem_code_tri'  : CPE3,
            'global_size'    : 0.25,     # mm
            'anode_bias'     : 3.0,
            'elec_divisions' : 4,
            'deviation'      : 0.05,
            'min_size_factor': 0.05,
        }
        #
        # -- Solver options ---------------------------------------------------
        self.solver = {
            'nlgeom'       : ON,
            'init_inc'     : 0.05,
            'min_inc'      : 1e-10,
            'max_inc'      : 0.25,
            'max_num_inc'  : 500,
            'stabilize'    : False,
            'stabilize_mag': 2e-4,
        }
        #
        # -- Output requests --------------------------------------------------
        self.field_outputs = ('S', 'E', 'U', 'NT', 'RF', 'TEMP')
        #
        # -- Job resources ----------------------------------------------------
        self.ncpus = 4
    #
    # -- Derived tables -------------------------------------------------------
    #
    def build_anode_elastic_table(self):
        """Build Abaqus-formatted elastic table for the anode material."""
        table = []
        for T in self.anode['T_points']:
            E_pa = compute_anode_modulus(T, self.anode['E_poly'])
            table.append((E_pa, self.anode['nu'], T))
        return tuple(table)
    #
    def build_elec_elastic_table(self):
        """Return Abaqus-formatted elastic table for the electrolyte."""
        return tuple(tuple(row) for row in self.elec['E_table'])
    #
    # -- Validation -----------------------------------------------------------
    #
    def validate(self):
        """Run all parameter validation checks."""
        for key, val in self.geom.items():
            validate_positive(val, 'geom.{}'.format(key))
        #
        validate_range(self.temp['T_sintering'], 500.0, 2000.0, 'T_sintering')
        validate_range(self.temp['T_room'], -273.15,
                       self.temp['T_sintering'], 'T_room')
        #
        # CTE tables must be non-empty
        if len(self.anode['cte_table']) < 2:
            raise ValueError('Anode CTE table must have >= 2 data points.')
        if len(self.elec['cte_table']) < 2:
            raise ValueError('Electrolyte CTE table must have >= 2 data points.')
        #
        # Elastic tables must be non-empty
        if len(self.anode['T_points']) < 2:
            raise ValueError('Anode temperature points must have >= 2 entries.')
        if len(self.elec['E_table']) < 2:
            raise ValueError('Electrolyte E table must have >= 2 data points.')
        #
        # Mesh: element divisions through electrolyte must be >= 2
        if self.mesh_cfg['elec_divisions'] < 2:
            raise ValueError('elec_divisions must be >= 2 for mesh convergence.')
        #
        logger.info('All configuration parameters validated successfully.')
    #
    def log_summary(self):
        """Log a compact summary of the configuration."""
        L  = self.geom['L_cell']
        Ha = self.geom['H_anode']
        He = self.geom['H_electrolyte']
        Ts = self.temp['T_sintering']
        Tr = self.temp['T_room']
        #
        logger.info('')
        logger.info('  Configuration Overview')
        logger.info('  ' + '-' * 50)
        logger.info('  Model name      : {}'.format(self.model_name))
        logger.info('  Job name        : {}'.format(self.job_name))
        logger.info('  ')
        logger.info('  Geometry (half-model, symmetry at x=0)')
        logger.info('    L_cell        = {:>10.3f} mm'.format(L))
        logger.info('    H_anode       = {:>10.3f} mm'.format(Ha))
        logger.info('    H_electrolyte = {:>10.4f} mm  ({:.1f} um)'.format(He, He * 1000))
        logger.info('  ')
        logger.info('  Thermal Loading')
        logger.info('    T_sintering   = {:>10.1f} C   (stress-free)'.format(Ts))
        logger.info('    T_room        = {:>10.1f} C   (target)'.format(Tr))
        logger.info('    dT            = {:>10.1f} C'.format(Tr - Ts))
        logger.info('  ')
        logger.info('  Materials')
        logger.info('    Anode         : {}  (nu={}, rho={:.2e} t/mm3)'.format(
            self.mat_anode, self.anode['nu'], self.anode['density']))
        logger.info('    Electrolyte   : {}  (nu={}, rho={:.2e} t/mm3)'.format(
            self.mat_elec, self.elec['nu'], self.elec['density']))
        logger.info('  ')
        logger.info('  Solver')
        logger.info('    NLGeom        = {}'.format(self.solver['nlgeom']))
        logger.info('    Inc range     = [{}, {}]'.format(
            self.solver['min_inc'], self.solver['max_inc']))
        logger.info('    Max increments= {}'.format(self.solver['max_num_inc']))
        logger.info('  ' + '-' * 50)


# ============================================================================
#                       MESH QUALITY DIAGNOSTICS
# ============================================================================

def mesh_quality_report(part, label):
    """
    Perform mesh quality diagnostics and log a summary.

    Checks
    ------
    - Total element / node count
    - Aspect ratio statistics (min, max, mean)
    - Warnings for degenerate or high-aspect-ratio elements

    Parameters
    ----------
    part : Part
        Abaqus Part object with a generated mesh.
    label : str
        Human-readable label for logging.
    """
    n_elem = len(part.elements)
    n_node = len(part.nodes)
    #
    logger.info('  {} mesh statistics:'.format(label))
    logger.info('    Elements : {:>6d}'.format(n_elem))
    logger.info('    Nodes    : {:>6d}'.format(n_node))
    #
    if n_elem == 0:
        logger.error('    *** NO ELEMENTS GENERATED -- check mesh seeds ***')
        return n_elem, n_node
    #
    # Aspect ratio estimation (heuristic for structured quad mesh)
    # For a proper check we would use the Abaqus mesh verify API,
    # but that requires an ODB.  Here we do a bounding-box estimate.
    try:
        coords = [(n.coordinates[0], n.coordinates[1]) for n in part.nodes]
        x_vals = [c[0] for c in coords]
        y_vals = [c[1] for c in coords]
        #
        dx = max(x_vals) - min(x_vals)
        dy = max(y_vals) - min(y_vals)
        #
        # Rough element edge sizes
        nx_approx = max(1, int(round(dx / (part.seeds.defaultElemSize
                                           if hasattr(part, 'seeds') else 0.25))))
        ny_approx = max(1, n_elem // max(nx_approx, 1))
        #
        elem_dx = dx / max(nx_approx, 1)
        elem_dy = dy / max(ny_approx, 1)
        #
        if elem_dy > 0:
            aspect = max(elem_dx, elem_dy) / min(elem_dx, elem_dy)
        else:
            aspect = float('inf')
        #
        logger.info('    Approx AR: {:.2f}  (domain {:.4f} x {:.4f} mm)'.format(
            aspect, dx, dy))
        #
        if aspect > 20.0:
            logger.warning(
                '    *** High aspect ratio ({:.1f}) -- consider refining mesh ***'.format(
                    aspect))
        elif aspect > 10.0:
            logger.info('    AR > 10 -- acceptable for thin layers.')
        else:
            logger.info('    AR within recommended limits (< 10).')
    except Exception as exc:
        logger.debug('    Aspect ratio check skipped: {}'.format(exc))
    #
    return n_elem, n_node


# ============================================================================
#                      MAIN MODEL BUILDER
# ============================================================================

class SOFCModelBuilder(object):
    """
    Encapsulates the complete Abaqus model construction workflow.

    Usage
    -----
    >>> cfg = SOFCModelConfig()
    >>> builder = SOFCModelBuilder(cfg)
    >>> builder.build()
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : SOFCModelConfig
            Validated configuration object.
        """
        self.cfg = config
        #
        # Runtime references (populated during build)
        self.model     = None
        self.p_anode   = None
        self.p_elec    = None
        self.inst_an   = None
        self.inst_el   = None
        self.rootAsm   = None
        self.step_name = 'Step-Cooling'
        #
        # Mesh statistics
        self.n_an_elem = 0
        self.n_an_node = 0
        self.n_el_elem = 0
        self.n_el_node = 0
    #
    # --------------------------------------------------------------------- #
    #  Top-level orchestrator                                                #
    # --------------------------------------------------------------------- #
    #
    def build(self):
        """Execute the full model-build pipeline."""
        try:
            self.cfg.validate()
            self.cfg.log_summary()
            #
            self._init_model()
            self._define_materials()
            self._define_sections()
            self._create_parts()
            self._assemble()
            self._create_interaction()
            self._define_step()
            self._define_outputs()
            self._apply_bcs()
            self._apply_thermal_loads()
            self._generate_mesh()
            self._run_analytical_check()
            self._create_job()
            self._save_model()
            self._print_summary()
        #
        except Exception:
            logger.error('')
            logger.error('!' * 72)
            logger.error('  FATAL ERROR DURING MODEL BUILD')
            logger.error('!' * 72)
            logger.error(traceback.format_exc())
            raise
    #
    # --------------------------------------------------------------------- #
    #  2. Model Initialisation                                               #
    # --------------------------------------------------------------------- #
    #
    @timer
    def _init_model(self):
        banner('2. Model Initialisation')
        #
        safe_delete_model(self.cfg.model_name)
        self.model = mdb.Model(
            name=self.cfg.model_name,
            modelType=STANDARD_EXPLICIT,
        )
        logger.info("Model '{}' created successfully.".format(
            self.cfg.model_name))
    #
    # --------------------------------------------------------------------- #
    #  3. Material Definitions                                               #
    # --------------------------------------------------------------------- #
    #
    @timer
    def _define_materials(self):
        banner('3. Material Definitions')
        #
        cfg = self.cfg
        #
        # -- 3.1  Anode: Ni-YSZ -----------------------------------------------
        sub_banner('Anode -- {} (Ni-YSZ cermet)'.format(cfg.mat_anode))
        #
        mat_an = self.model.Material(name=cfg.mat_anode)
        #
        anode_E_table = cfg.build_anode_elastic_table()
        mat_an.Elastic(
            type=ISOTROPIC,
            temperatureDependency=ON,
            table=anode_E_table,
        )
        #
        # Log modulus at key temperatures
        for row in anode_E_table[::max(1, len(anode_E_table) // 4)]:
            logger.debug('    E({:.0f} C) = {:.2f} GPa'.format(
                row[2], row[0] / 1.0e9))
        #
        mat_an.Expansion(
            type=ISOTROPIC,
            temperatureDependency=ON,
            zero=cfg.temp['T_sintering'],
            table=tuple(cfg.anode['cte_table']),
        )
        #
        mat_an.Density(table=((cfg.anode['density'],),))
        #
        logger.info('  {} defined  [T-dep E, T-dep CTE, rho={:.2e}]'.format(
            cfg.mat_anode, cfg.anode['density']))
        #
        # -- 3.2  Electrolyte: 8YSZ -------------------------------------------
        sub_banner('Electrolyte -- {} (8 mol% YSZ)'.format(cfg.mat_elec))
        #
        mat_el = self.model.Material(name=cfg.mat_elec)
        #
        elec_E_table = cfg.build_elec_elastic_table()
        mat_el.Elastic(
            type=ISOTROPIC,
            temperatureDependency=ON,
            table=elec_E_table,
        )
        #
        mat_el.Expansion(
            type=ISOTROPIC,
            temperatureDependency=ON,
            zero=cfg.temp['T_sintering'],
            table=tuple(cfg.elec['cte_table']),
        )
        #
        mat_el.Density(table=((cfg.elec['density'],),))
        #
        logger.info('  {} defined  [T-dep E, T-dep CTE, rho={:.2e}]'.format(
            cfg.mat_elec, cfg.elec['density']))
    #
    # --------------------------------------------------------------------- #
    #  4. Section Definitions                                                #
    # --------------------------------------------------------------------- #
    #
    @timer
    def _define_sections(self):
        banner('4. Section Definitions')
        #
        self._sec_anode = 'Section-Anode'
        self._sec_elec  = 'Section-Electrolyte'
        #
        self.model.HomogeneousSolidSection(
            name=self._sec_anode,
            material=self.cfg.mat_anode,
            thickness=None,
        )
        self.model.HomogeneousSolidSection(
            name=self._sec_elec,
            material=self.cfg.mat_elec,
            thickness=None,
        )
        #
        logger.info('  Sections created: {}, {}'.format(
            self._sec_anode, self._sec_elec))
    #
    # --------------------------------------------------------------------- #
    #  5. Part Creation & Section Assignment                                 #
    # --------------------------------------------------------------------- #
    #
    @timer
    def _create_parts(self):
        banner('5. Part Creation & Section Assignment')
        #
        L  = self.cfg.geom['L_cell']
        Ha = self.cfg.geom['H_anode']
        He = self.cfg.geom['H_electrolyte']
        #
        # -- 5.1  Anode -------------------------------------------------------
        sub_banner('Anode Part  ({} x {} mm)'.format(L, Ha))
        #
        sk_an = self.model.ConstrainedSketch(
            name='__sketch_anode__', sheetSize=2.0 * L)
        sk_an.rectangle(point1=(0.0, 0.0), point2=(L, Ha))
        #
        self.p_anode = self.model.Part(
            name='Part-Anode',
            dimensionality=TWO_D_PLANAR,
            type=DEFORMABLE_BODY,
        )
        self.p_anode.BaseShell(sketch=sk_an)
        del sk_an
        #
        rgn_an = self.p_anode.Set(
            name='Set-All-Anode', faces=self.p_anode.faces[:])
        self.p_anode.SectionAssignment(
            region=rgn_an,
            sectionName=self._sec_anode,
            offset=0.0,
            offsetType=MIDDLE_SURFACE,
            offsetField='',
            thicknessAssignment=FROM_SECTION,
        )
        logger.info('  Part-Anode created and section assigned.')
        #
        # -- 5.2  Electrolyte -------------------------------------------------
        sub_banner('Electrolyte Part  ({} x {} mm)'.format(L, He))
        #
        sk_el = self.model.ConstrainedSketch(
            name='__sketch_elec__', sheetSize=2.0 * L)
        sk_el.rectangle(point1=(0.0, 0.0), point2=(L, He))
        #
        self.p_elec = self.model.Part(
            name='Part-Electrolyte',
            dimensionality=TWO_D_PLANAR,
            type=DEFORMABLE_BODY,
        )
        self.p_elec.BaseShell(sketch=sk_el)
        del sk_el
        #
        rgn_el = self.p_elec.Set(
            name='Set-All-Electrolyte', faces=self.p_elec.faces[:])
        self.p_elec.SectionAssignment(
            region=rgn_el,
            sectionName=self._sec_elec,
            offset=0.0,
            offsetType=MIDDLE_SURFACE,
            offsetField='',
            thicknessAssignment=FROM_SECTION,
        )
        logger.info('  Part-Electrolyte created and section assigned.')
    #
    # --------------------------------------------------------------------- #
    #  6. Assembly                                                           #
    # --------------------------------------------------------------------- #
    #
    @timer
    def _assemble(self):
        banner('6. Assembly')
        #
        Ha = self.cfg.geom['H_anode']
        #
        self.rootAsm = self.model.rootAssembly
        self.rootAsm.DatumCsysByDefault(CARTESIAN)
        #
        self.inst_an = self.rootAsm.Instance(
            name='Anode-1', part=self.p_anode, dependent=ON)
        #
        self.inst_el = self.rootAsm.Instance(
            name='Electrolyte-1', part=self.p_elec, dependent=ON)
        self.rootAsm.translate(
            instanceList=('Electrolyte-1',),
            vector=(0.0, Ha, 0.0),
        )
        #
        logger.info('  Anode-1        : origin = (0, 0, 0)')
        logger.info('  Electrolyte-1  : translated to (0, {}, 0)'.format(Ha))
        logger.info('  Interface at y = {} mm'.format(Ha))
    #
    # --------------------------------------------------------------------- #
    #  7. Interaction -- Tie Constraint                                      #
    # --------------------------------------------------------------------- #
    #
    @timer
    def _create_interaction(self):
        banner('7. Interaction -- Tie Constraint (Perfect Bonding)')
        #
        L  = self.cfg.geom['L_cell']
        Ha = self.cfg.geom['H_anode']
        He = self.cfg.geom['H_electrolyte']
        #
        # Locate interface edges
        edge_el_bot = self.inst_el.edges.findAt(((L / 2.0, Ha, 0.0),))
        edge_an_top = self.inst_an.edges.findAt(((L / 2.0, Ha, 0.0),))
        #
        surf_main = self.rootAsm.Surface(
            side1Edges=edge_el_bot, name='Surf-ElecBottom')
        surf_sec  = self.rootAsm.Surface(
            side1Edges=edge_an_top, name='Surf-AnodeTop')
        #
        # Use the version-aware tie creation utility
        create_tie_constraint(
            model=self.model,
            name='Tie-Interface',
            main_surface=surf_main,
            secondary_surface=surf_sec,
            positionToleranceMethod=COMPUTED,
            adjust=ON,
            tieRotations=ON,
            thickness=ON,
            constraintEnforcement=SURFACE_TO_SURFACE,
        )
        #
        logger.info('  Tie constraint: Surf-ElecBottom <-> Surf-AnodeTop')
        logger.info('  Enforcement   : SURFACE_TO_SURFACE')
        logger.info('  Adjust        : ON')
    #
    # --------------------------------------------------------------------- #
    #  8. Analysis Step                                                      #
    # --------------------------------------------------------------------- #
    #
    @timer
    def _define_step(self):
        banner('8. Analysis Step Definition')
        #
        cfg = self.cfg
        Ts  = cfg.temp['T_sintering']
        Tr  = cfg.temp['T_room']
        #
        self.model.StaticStep(
            name=self.step_name,
            previous='Initial',
            timePeriod=1.0,
            initialInc=cfg.solver['init_inc'],
            minInc=cfg.solver['min_inc'],
            maxInc=cfg.solver['max_inc'],
            maxNumInc=cfg.solver['max_num_inc'],
            nlgeom=cfg.solver['nlgeom'],
            description='Uniform cooling: {:.0f} C -> {:.0f} C  (dT = {:.0f} C)'.format(
                Ts, Tr, Tr - Ts),
        )
        #
        if cfg.solver['stabilize']:
            self.model.steps[self.step_name].setValues(
                stabilizationMagnitude=cfg.solver['stabilize_mag'],
                stabilizationMethod=DISSIPATED_ENERGY_FRACTION,
                adaptiveDampingRatio=0.05,
                continueDampingFactors=False,
            )
            logger.info('  Adaptive stabilisation ENABLED  '
                        '(magnitude = {})'.format(cfg.solver['stabilize_mag']))
        #
        logger.info("  Step '{}' created.".format(self.step_name))
        logger.info('    Time period   = 1.0')
        logger.info('    Inc (init)    = {}'.format(cfg.solver['init_inc']))
        logger.info('    Inc (min/max) = {} / {}'.format(
            cfg.solver['min_inc'], cfg.solver['max_inc']))
        logger.info('    Max increments= {}'.format(cfg.solver['max_num_inc']))
        logger.info('    NLGeom        = {}'.format(cfg.solver['nlgeom']))
    #
    # --------------------------------------------------------------------- #
    #  9. Output Requests                                                    #
    # --------------------------------------------------------------------- #
    #
    @timer
    def _define_outputs(self):
        banner('9. Field & History Output Requests')
        #
        cfg = self.cfg
        L   = cfg.geom['L_cell']
        Ha  = cfg.geom['H_anode']
        #
        # Remove default field output
        if 'F-Output-1' in self.model.fieldOutputRequests:
            del self.model.fieldOutputRequests['F-Output-1']
        #
        # -- 9.1  Global field output (last increment) -------------------------
        self.model.FieldOutputRequest(
            name='FOReq-Global',
            createStepName=self.step_name,
            variables=cfg.field_outputs,
            frequency=LAST_INCREMENT,
        )
        logger.info('  FOReq-Global : {} (last increment)'.format(
            ', '.join(cfg.field_outputs)))
        #
        # -- 9.2  Interface field output (every increment) ---------------------
        edge_an_top_set = self.rootAsm.Set(
            edges=self.inst_an.edges.findAt(((L / 2.0, Ha, 0.0),)),
            name='Set-AnodeInterface',
        )
        #
        self.model.FieldOutputRequest(
            name='FOReq-Interface',
            createStepName=self.step_name,
            variables=('S', 'E', 'U', 'TEMP'),
            region=edge_an_top_set,
            frequency=1,
        )
        logger.info('  FOReq-Interface : S, E, U, TEMP (every increment)')
        #
        # -- 9.3  History output at interface ----------------------------------
        self.model.HistoryOutputRequest(
            name='HOReq-InterfaceMid',
            createStepName=self.step_name,
            variables=('S11', 'S22', 'S12', 'E11', 'E22', 'U1', 'U2'),
            region=edge_an_top_set,
            frequency=1,
        )
        logger.info('  HOReq-InterfaceMid : S11, S22, S12, E11, E22, U1, U2')
    #
    # --------------------------------------------------------------------- #
    #  10. Boundary Conditions                                               #
    # --------------------------------------------------------------------- #
    #
    @timer
    def _apply_bcs(self):
        banner('10. Boundary Conditions')
        #
        cfg = self.cfg
        L   = cfg.geom['L_cell']
        Ha  = cfg.geom['H_anode']
        He  = cfg.geom['H_electrolyte']
        #
        # -- 10.1  Symmetry at x = 0 ------------------------------------------
        sub_banner('X-Symmetry  (x = 0 plane)')
        #
        edges_sym_an = self.inst_an.edges.findAt(
            ((0.0, Ha / 2.0, 0.0),))
        edges_sym_el = self.inst_el.edges.findAt(
            ((0.0, Ha + He / 2.0, 0.0),))
        edges_sym    = edges_sym_an + edges_sym_el
        #
        rgn_sym = self.rootAsm.Set(edges=edges_sym, name='Set-SymmX')
        self.model.XsymmBC(
            name='BC-SymmetryX',
            createStepName='Initial',
            region=rgn_sym,
            localCsys=None,
        )
        logger.info('  X-symmetry BC applied at x = 0  (anode + electrolyte)')
        #
        # -- 10.2  Vertical pin at origin -------------------------------------
        sub_banner('Y-Pin  (0, 0)')
        #
        vert_pin = self.inst_an.vertices.findAt(((0.0, 0.0, 0.0),))
        rgn_pin  = self.rootAsm.Set(vertices=vert_pin, name='Set-PinY')
        self.model.DisplacementBC(
            name='BC-PinY',
            createStepName='Initial',
            region=rgn_pin,
            u1=UNSET,
            u2=0.0,
            ur3=UNSET,
            amplitude=UNSET,
            distributionType=UNIFORM,
            fieldName='',
            localCsys=None,
        )
        logger.info('  Y-displacement pin at vertex (0, 0)')
    #
    # --------------------------------------------------------------------- #
    #  11. Thermal Predefined Fields                                         #
    # --------------------------------------------------------------------- #
    #
    @timer
    def _apply_thermal_loads(self):
        banner('11. Thermal Predefined Fields')
        #
        cfg = self.cfg
        Ts  = cfg.temp['T_sintering']
        Tr  = cfg.temp['T_room']
        #
        all_faces = self.inst_an.faces[:] + self.inst_el.faces[:]
        set_all   = self.rootAsm.Set(faces=all_faces, name='Set-AllCells')
        #
        # -- 11.1  Initial temperature (stress-free reference) -----------------
        sub_banner('Initial Temperature  (T = {:.0f} C)'.format(Ts))
        #
        self.model.Temperature(
            name='PF-InitialTemp',
            createStepName='Initial',
            region=set_all,
            distributionType=UNIFORM,
            crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
            magnitudes=(Ts,),
        )
        logger.info('  T_initial = {:.0f} C  (stress-free reference state)'.format(Ts))
        #
        # -- 11.2  Cooling target ----------------------------------------------
        sub_banner('Cooling Target  (T = {:.0f} C)'.format(Tr))
        #
        self.model.Temperature(
            name='PF-CoolDown',
            createStepName=self.step_name,
            region=set_all,
            distributionType=UNIFORM,
            crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
            magnitudes=(Tr,),
        )
        logger.info('  T_final   = {:.0f} C'.format(Tr))
        logger.info('  dT        = {:.0f} C'.format(Tr - Ts))
    #
    # --------------------------------------------------------------------- #
    #  12. Mesh Generation                                                   #
    # --------------------------------------------------------------------- #
    #
    @timer
    def _generate_mesh(self):
        banner('12. Mesh Generation')
        #
        cfg = self.cfg
        L   = cfg.geom['L_cell']
        Ha  = cfg.geom['H_anode']
        He  = cfg.geom['H_electrolyte']
        mc  = cfg.mesh_cfg
        #
        # -- 12.1  Element type assignment ------------------------------------
        sub_banner('Element Type Assignment')
        #
        elem_quad = mesh.ElemType(
            elemCode=mc['elem_code'],
            elemLibrary=STANDARD,
            secondOrderAccuracy=OFF,
            hourglassControl=ENHANCED,
            distortionControl=DEFAULT,
        )
        elem_tri = mesh.ElemType(
            elemCode=mc['elem_code_tri'],
            elemLibrary=STANDARD,
        )
        #
        for part, label in [(self.p_anode, 'Anode'),
                            (self.p_elec, 'Electrolyte')]:
            part.setElementType(
                regions=(part.faces[:],),
                elemTypes=(elem_quad, elem_tri),
            )
            logger.info('  {} : {} / {}'.format(
                label, mc['elem_code'], mc['elem_code_tri']))
        #
        # -- 12.2  Seeding ----------------------------------------------------
        sub_banner('Mesh Seeding')
        #
        # Anode -- global seed + biased vertical edges toward interface
        self.p_anode.seedPart(
            size=mc['global_size'],
            deviationFactor=mc['deviation'],
            minSizeFactor=mc['min_size_factor'],
        )
        #
        an_vert_edges = self.p_anode.edges.findAt(
            ((0.0, Ha / 2.0, 0.0),),
            ((L,   Ha / 2.0, 0.0),),
        )
        n_vert_anode = int(Ha / mc['global_size'] * 2)
        self.p_anode.seedEdgeByBias(
            biasMethod=SINGLE,
            end1Edges=an_vert_edges,
            ratio=mc['anode_bias'],
            number=n_vert_anode,
            constraint=FINER,
        )
        logger.info('  Anode seeding: bias={:.1f}, {} divisions vertically'.format(
            mc['anode_bias'], n_vert_anode))
        #
        # Electrolyte -- uniform through-thickness + global
        el_vert_edges = self.p_elec.edges.findAt(
            ((0.0, He / 2.0, 0.0),),
            ((L,   He / 2.0, 0.0),),
        )
        self.p_elec.seedEdgeByNumber(
            edges=el_vert_edges,
            number=max(mc['elec_divisions'], 2),
            constraint=FINER,
        )
        self.p_elec.seedPart(
            size=mc['global_size'],
            deviationFactor=mc['deviation'],
            minSizeFactor=mc['min_size_factor'],
        )
        logger.info('  Electrolyte seeding: {} elements through thickness'.format(
            mc['elec_divisions']))
        #
        # -- 12.3  Mesh control -- structured quad ----------------------------
        sub_banner('Mesh Control')
        #
        for part, label in [(self.p_anode, 'Anode'),
                            (self.p_elec, 'Electrolyte')]:
            part.setMeshControls(
                regions=part.faces[:],
                elemShape=QUAD,
                technique=STRUCTURED,
            )
            logger.info('  {} : STRUCTURED QUAD'.format(label))
        #
        # -- 12.4  Generate meshes --------------------------------------------
        sub_banner('Mesh Generation')
        #
        self.p_anode.generateMesh()
        self.p_elec.generateMesh()
        #
        # -- 12.5  Mesh quality report ----------------------------------------
        sub_banner('Mesh Quality Diagnostics')
        #
        self.n_an_elem, self.n_an_node = mesh_quality_report(
            self.p_anode, 'Anode')
        self.n_el_elem, self.n_el_node = mesh_quality_report(
            self.p_elec, 'Electrolyte')
        #
        n_total_elem = self.n_an_elem + self.n_el_elem
        n_total_node = self.n_an_node + self.n_el_node
        #
        logger.info('')
        logger.info('  +{:-<42s}+'.format(''))
        logger.info('  | {:>12s} | {:>10s} | {:>10s} |'.format(
            'Component', 'Elements', 'Nodes'))
        logger.info('  +{:-<42s}+'.format(''))
        logger.info('  | {:>12s} | {:>10d} | {:>10d} |'.format(
            'Anode', self.n_an_elem, self.n_an_node))
        logger.info('  | {:>12s} | {:>10d} | {:>10d} |'.format(
            'Electrolyte', self.n_el_elem, self.n_el_node))
        logger.info('  +{:-<42s}+'.format(''))
        logger.info('  | {:>12s} | {:>10d} | {:>10d} |'.format(
            'TOTAL', n_total_elem, n_total_node))
        logger.info('  +{:-<42s}+'.format(''))
    #
    # --------------------------------------------------------------------- #
    #  Analytical Verification (Timoshenko)                                  #
    # --------------------------------------------------------------------- #
    #
    @timer
    def _run_analytical_check(self):
        banner('Analytical Verification -- Timoshenko Bilayer')
        #
        cfg = self.cfg
        Ts  = cfg.temp['T_sintering']
        Tr  = cfg.temp['T_room']
        dT  = Tr - Ts
        Ha  = cfg.geom['H_anode']
        He  = cfg.geom['H_electrolyte']
        #
        # Use room-temperature properties for the analytical estimate
        E1   = compute_anode_modulus(Tr, cfg.anode['E_poly'])
        E2   = cfg.elec['E_table'][0][0] * 1.0e6   # MPa -> Pa
        nu1  = cfg.anode['nu']
        nu2  = cfg.elec['nu']
        a1   = cfg.anode['cte_table'][0][0]   # CTE at ~25 C
        a2   = cfg.elec['cte_table'][0][0]
        #
        result = analytical_bilayer_stress(
            E1=E1, E2=E2, alpha1=a1, alpha2=a2,
            h1=Ha, h2=He, nu1=nu1, nu2=nu2, dT=dT)
        #
        s1 = result['sigma_1']
        s2 = result['sigma_2']
        kappa = result['curvature']
        #
        logger.info('  Input parameters (room-temperature):')
        logger.info('    E_anode       = {:.2f} GPa'.format(E1 / 1.0e9))
        logger.info('    E_electrolyte = {:.2f} GPa'.format(E2 / 1.0e9))
        logger.info('    CTE_anode     = {:.2e} /K'.format(a1))
        logger.info('    CTE_elec      = {:.2e} /K'.format(a2))
        logger.info('    dT            = {:.0f} C'.format(dT))
        logger.info('    h_anode       = {:.3f} mm'.format(Ha))
        logger.info('    h_electrolyte = {:.4f} mm'.format(He))
        logger.info('')
        logger.info('  Timoshenko analytical solution:')
        logger.info('    sigma_anode       = {:>+12.2f} MPa'.format(s1 / 1.0e6))
        logger.info('    sigma_electrolyte = {:>+12.2f} MPa'.format(s2 / 1.0e6))
        logger.info('    curvature         = {:>+12.4e} 1/mm'.format(kappa))
        logger.info('')
        logger.info('  NOTE: This is an approximate closed-form estimate using')
        logger.info('        constant (RT) material properties.  The FE solution')
        logger.info('        incorporates temperature-dependent properties and')
        logger.info('        nonlinear geometry effects.')
    #
    # --------------------------------------------------------------------- #
    #  13. Job Creation                                                      #
    # --------------------------------------------------------------------- #
    #
    @timer
    def _create_job(self):
        banner('13. Job Creation')
        #
        cfg = self.cfg
        #
        mdb.Job(
            name=cfg.job_name,
            model=cfg.model_name,
            description=cfg.job_desc,
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
            numCpus=cfg.ncpus,
            numDomains=cfg.ncpus,
            numGPUs=0,
        )
        #
        logger.info("  Job '{}' created.".format(cfg.job_name))
        logger.info('    CPUs     = {}'.format(cfg.ncpus))
        logger.info('    Memory   = 90%')
        logger.info('    NLGeom   = {}'.format(cfg.solver['nlgeom']))
        logger.info('    Format   = ODB')
    #
    # --------------------------------------------------------------------- #
    #  14. Save Model Database                                               #
    # --------------------------------------------------------------------- #
    #
    @timer
    def _save_model(self):
        banner('14. Save Model Database')
        #
        mdb_path = os.path.join(os.getcwd(), '{}.cae'.format(
            self.cfg.model_name))
        mdb.saveAs(pathName=mdb_path)
        logger.info("  Saved: '{}'".format(mdb_path))
    #
    # --------------------------------------------------------------------- #
    #  15. Build Summary                                                     #
    # --------------------------------------------------------------------- #
    #
    def _print_summary(self):
        banner('15. Build Summary', char='=')
        #
        cfg = self.cfg
        L   = cfg.geom['L_cell']
        Ha  = cfg.geom['H_anode']
        He  = cfg.geom['H_electrolyte']
        Ts  = cfg.temp['T_sintering']
        Tr  = cfg.temp['T_room']
        dT  = Tr - Ts
        #
        n_tot_e = self.n_an_elem + self.n_el_elem
        n_tot_n = self.n_an_node + self.n_el_node
        #
        wall_elapsed = time.time() - _START_WALL
        #
        w = 72
        hr = '+' + '-' * (w - 2) + '+'
        blank = '|' + ' ' * (w - 2) + '|'
        #
        lines = [
            '',
            '+' + '=' * (w - 2) + '+',
            '|{:^{w}}|'.format('MODEL BUILD SUMMARY', w=w - 2),
            '+' + '=' * (w - 2) + '+',
            blank,
            '|  {:<20s}: {:<{fw}}|'.format('Model', cfg.model_name, fw=w - 25),
            '|  {:<20s}: {:<{fw}}|'.format('Job', cfg.job_name, fw=w - 25),
            '|  {:<20s}: {:<{fw}}|'.format('Script version', __version__, fw=w - 25),
            '|  {:<20s}: {:<{fw}}|'.format('Abaqus version', _ABAQUS_VERSION_STR, fw=w - 25),
            blank,
            hr,
            '|  {:<70s}|'.format('GEOMETRY'),
            hr,
            '|  {:<20s}: {:>10.3f} mm{:<{fw}}|'.format('Half-width (L)', L, '', fw=w - 38),
            '|  {:<20s}: {:>10.3f} mm{:<{fw}}|'.format('Anode thickness', Ha, '', fw=w - 38),
            '|  {:<20s}: {:>10.4f} mm  ({:.1f} um){:<{fw}}|'.format(
                'Electrolyte', He, He * 1000, '', fw=w - 49),
            '|  {:<20s}: {:>10.1f}{:<{fw}}|'.format(
                'Aspect ratio (L/H)', L / (Ha + He), '', fw=w - 35),
            blank,
            hr,
            '|  {:<70s}|'.format('THERMAL LOADING'),
            hr,
            '|  {:<20s}: {:>10.1f} C   (stress-free){:<{fw}}|'.format(
                'T_sintering', Ts, '', fw=w - 52),
            '|  {:<20s}: {:>10.1f} C   (target){:<{fw}}|'.format(
                'T_room', Tr, '', fw=w - 46),
            '|  {:<20s}: {:>10.1f} C{:<{fw}}|'.format('dT', dT, '', fw=w - 36),
            blank,
            hr,
            '|  {:<70s}|'.format('MESH'),
            hr,
            '|  {:<20s}: {:>6d} elem  |  {:>6d} nodes{:<{fw}}|'.format(
                'Anode', self.n_an_elem, self.n_an_node, '', fw=w - 53),
            '|  {:<20s}: {:>6d} elem  |  {:>6d} nodes{:<{fw}}|'.format(
                'Electrolyte', self.n_el_elem, self.n_el_node, '', fw=w - 53),
            '|  {:<20s}: {:>6d} elem  |  {:>6d} nodes{:<{fw}}|'.format(
                'TOTAL', n_tot_e, n_tot_n, '', fw=w - 53),
            '|  {:<20s}: {:<{fw}}|'.format('Element type', str(cfg.mesh_cfg['elem_code']), fw=w - 25),
            blank,
            hr,
            '|  {:<70s}|'.format('MATERIALS'),
            hr,
            '|  {:<20s}: {:<{fw}}|'.format('Anode', cfg.mat_anode, fw=w - 25),
            '|  {:<20s}  T-dep E, T-dep CTE, rho={:.2e}{:<{fw}}|'.format(
                '', cfg.anode['density'], '', fw=w - 48),
            '|  {:<20s}: {:<{fw}}|'.format('Electrolyte', cfg.mat_elec, fw=w - 25),
            '|  {:<20s}  T-dep E, T-dep CTE, rho={:.2e}{:<{fw}}|'.format(
                '', cfg.elec['density'], '', fw=w - 48),
            blank,
            hr,
            '|  {:<70s}|'.format('SOLVER'),
            hr,
            '|  {:<20s}: {:<{fw}}|'.format('NLGeom', str(cfg.solver['nlgeom']), fw=w - 25),
            '|  {:<20s}: {:<{fw}}|'.format(
                'Inc range', '[{}, {}]'.format(cfg.solver['min_inc'], cfg.solver['max_inc']), fw=w - 25),
            '|  {:<20s}: {:<{fw}}|'.format('Max increments', str(cfg.solver['max_num_inc']), fw=w - 25),
            '|  {:<20s}: {:<{fw}}|'.format('Stabilisation',
                'ON (mag={})'.format(cfg.solver['stabilize_mag']) if cfg.solver['stabilize'] else 'OFF',
                fw=w - 25),
            blank,
            hr,
            '|  {:<70s}|'.format('STATUS'),
            hr,
            '|  {:<20s}: {:<{fw}}|'.format('Build status', 'READY TO SUBMIT', fw=w - 25),
            '|  {:<20s}: {:.2f} s{:<{fw}}|'.format('Build time', wall_elapsed, '', fw=w - 31),
            '|  {:<20s}: {:<{fw}}|'.format('Log file', os.path.basename(_LOG_FILE), fw=w - 25),
            blank,
            '+' + '=' * (w - 2) + '+',
            '',
        ]
        #
        for line in lines:
            logger.info(line)
        #
        logger.info('')
        logger.info('=' * w)
        logger.info('  BUILD COMPLETE -- Model is ready for submission.')
        logger.info('')
        logger.info('  To submit interactively:')
        logger.info('    >>> mdb.jobs["{}"].submit()'.format(cfg.job_name))
        logger.info('    >>> mdb.jobs["{}"].waitForCompletion()'.format(cfg.job_name))
        logger.info('')
        logger.info('  To submit from command line:')
        logger.info('    $ abaqus job={} cpus={} interactive'.format(
            cfg.job_name, cfg.ncpus))
        logger.info('=' * w)
        logger.info('')


# ============================================================================
#                           POST-PROCESSING UTILITIES
# ============================================================================

def extract_interface_stress(odb_path, step_name='Step-Cooling',
                             variable='S', component='S11'):
    """
    Extract interface stress from the output database.

    This function can be called after job completion to retrieve
    the stress distribution along the anode-electrolyte interface
    and export it to CSV for comparison with synchrotron data.

    Parameters
    ----------
    odb_path : str
        Path to the .odb file.
    step_name : str
        Name of the analysis step.
    variable : str
        Field output variable identifier.
    component : str
        Stress component label (e.g. 'S11', 'S22', 'S12').

    Returns
    -------
    list of tuple
        [(x_coord, stress_value), ...] sorted by x-coordinate.

    Notes
    -----
    This function requires the Abaqus/Viewer kernel and is intended
    to be called in a separate post-processing script or after the
    analysis is completed.
    """
    try:
        from odbAccess import openOdb
        #
        odb = openOdb(path=odb_path, readOnly=True)
        step = odb.steps[step_name]
        last_frame = step.frames[-1]
        #
        stress_field = last_frame.fieldOutputs[variable]
        #
        # Filter to the interface set if available
        try:
            region = odb.rootAssembly.nodeSets['SET-ANODEINTERFACE']
            stress_sub = stress_field.getSubset(region=region)
        except KeyError:
            stress_sub = stress_field
        #
        results = []
        comp_idx = {'S11': 0, 'S22': 1, 'S33': 2, 'S12': 3}.get(component, 0)
        #
        for val in stress_sub.values:
            x = val.position[0] if hasattr(val, 'position') else 0.0
            s = val.data[comp_idx] if hasattr(val.data, '__len__') else val.data
            results.append((x, s))
        #
        odb.close()
        #
        results.sort(key=lambda p: p[0])
        return results
    #
    except ImportError:
        logger.warning('odbAccess not available -- post-processing skipped.')
        return []
    except Exception as exc:
        logger.error('Post-processing error: {}'.format(exc))
        return []


def export_results_csv(results, filepath, header='x_mm,S11_MPa'):
    """
    Write extracted results to a CSV file.

    Parameters
    ----------
    results : list of tuple
        [(x, stress), ...] values.
    filepath : str
        Output CSV file path.
    header : str
        CSV header line.
    """
    try:
        with open(filepath, 'w') as f:
            f.write(header + '\n')
            for x, s in results:
                f.write('{:.6f},{:.6f}\n'.format(x, s / 1.0e6))
        logger.info('Results exported: {}'.format(filepath))
    except IOError as exc:
        logger.error('CSV export failed: {}'.format(exc))


# ============================================================================
#                          ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for the SOFC validation model script.

    Instantiates the configuration, builds the model, and logs
    completion status.
    """
    logger.info('Script : {}'.format(_SCRIPT_NAME))
    logger.info('Author : {} ({})'.format(__author__, __email__))
    logger.info('Version: {}'.format(__version__))
    logger.info('License: {}'.format(__license__))
    logger.info('CWD    : {}'.format(os.getcwd()))
    logger.info('')
    #
    config  = SOFCModelConfig()
    builder = SOFCModelBuilder(config)
    builder.build()
    #
    logger.info('')
    logger.info('Script execution completed in {}.'.format(elapsed()))
    logger.info('')


# --------------------------------------------------------------------------- #
#  Execute                                                                     #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    main()
else:
    # When run via `abaqus cae noGUI=script.py`, __name__ is '__main__'.
    # If imported as a module, just expose the classes and functions.
    pass

# Unconditional execution for Abaqus noGUI mode
# (some Abaqus versions do not set __name__ == '__main__' for noGUI scripts)
try:
    _already_ran
except NameError:
    _already_ran = True
    main()
