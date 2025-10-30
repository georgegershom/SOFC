from typing import Dict
import numpy as np

THERMAL_TYPE = {"dirichlet": 0, "neumann": 1, "convective": 2, "insulated": 3}
ELECTRIC_TYPE = {"dirichlet": 0, "neumann": 1}


def _edge_cfg(boundaries_cfg: Dict, edge_name: str) -> Dict:
    if edge_name not in boundaries_cfg:
        raise KeyError(f"Missing boundary '{edge_name}' in config")
    return boundaries_cfg[edge_name]


def encode_thermal_bc_for_edge(boundaries_cfg: Dict, edge_name: str, n: int) -> Dict[str, np.ndarray]:
    edge = _edge_cfg(boundaries_cfg, edge_name)
    therm = edge.get("thermal", {"type": "insulated"})
    typ = THERMAL_TYPE[str(therm.get("type", "insulated")).lower()]
    value = float(therm.get("value", 0.0))
    flux = float(therm.get("flux", 0.0))
    h = float(therm.get("h", 0.0))
    t_inf = float(therm.get("T_inf", 0.0))
    return {
        "thermal_type": np.full(n, typ, dtype=np.int32),
        "thermal_value": np.full(n, value, dtype=np.float64),
        "thermal_flux": np.full(n, flux, dtype=np.float64),
        "thermal_h": np.full(n, h, dtype=np.float64),
        "thermal_T_inf": np.full(n, t_inf, dtype=np.float64),
    }


def encode_electrical_bc_for_edge(boundaries_cfg: Dict, edge_name: str, n: int) -> Dict[str, np.ndarray]:
    edge = _edge_cfg(boundaries_cfg, edge_name)
    elec = edge.get("electrical", {})
    phi_s = elec.get("phi_s", {"type": "neumann", "flux": 0.0})
    phi_e = elec.get("phi_e", {"type": "neumann", "flux": 0.0})
    def enc(phi_cfg: Dict, prefix: str) -> Dict[str, np.ndarray]:
        typ = ELECTRIC_TYPE[str(phi_cfg.get("type", "neumann")).lower()]
        val = float(phi_cfg.get("value", 0.0))
        flux = float(phi_cfg.get("flux", 0.0))
        return {
            f"{prefix}_type": np.full(n, typ, dtype=np.int32),
            f"{prefix}_value": np.full(n, val, dtype=np.float64),
            f"{prefix}_flux": np.full(n, flux, dtype=np.float64),
        }
    out = {}
    out.update(enc(phi_s, "phi_s"))
    out.update(enc(phi_e, "phi_e"))
    return out


def encode_mechanical_bc_for_edge(boundaries_cfg: Dict, edge_name: str, n: int) -> Dict[str, np.ndarray]:
    edge = _edge_cfg(boundaries_cfg, edge_name)
    mech = edge.get("mechanical", {"type": "free"})
    typ = str(mech.get("type", "free")).lower()
    fix_ux = np.zeros(n, dtype=np.int32)
    fix_uy = np.zeros(n, dtype=np.int32)
    traction_x = np.zeros(n, dtype=np.float64)
    traction_y = np.zeros(n, dtype=np.float64)
    if typ == "fixed":
        fix_ux[:] = 1
        fix_uy[:] = 1
    elif typ == "traction":
        tr = mech.get("traction", {"x": 0.0, "y": 0.0})
        traction_x[:] = float(tr.get("x", 0.0))
        traction_y[:] = float(tr.get("y", 0.0))
    # free: 默认 traction=0
    return {
        "fix_ux": fix_ux,
        "fix_uy": fix_uy,
        "traction_x": traction_x,
        "traction_y": traction_y,
    }
