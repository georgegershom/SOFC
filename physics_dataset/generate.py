#!/usr/bin/env python3
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")


@dataclass
class Domain:
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    def sample_interior(self, n: int, rng: np.random.Generator) -> np.ndarray:
        xs = rng.uniform(self.x_min, self.x_max, n)
        ys = rng.uniform(self.y_min, self.y_max, n)
        ts = rng.uniform(self.t_min, self.t_max, n)
        return np.stack([xs, ys, ts], axis=1)

    def sample_edge(self, edge: str, n: int, rng: np.random.Generator) -> np.ndarray:
        # edge in {"left","right","bottom","top"}
        if edge == "left":
            xs = np.full(n, self.x_min)
            ys = rng.uniform(self.y_min, self.y_max, n)
            ts = rng.uniform(self.t_min, self.t_max, n)
        elif edge == "right":
            xs = np.full(n, self.x_max)
            ys = rng.uniform(self.y_min, self.y_max, n)
            ts = rng.uniform(self.t_min, self.t_max, n)
        elif edge == "bottom":
            ys = np.full(n, self.y_min)
            xs = rng.uniform(self.x_min, self.x_max, n)
            ts = rng.uniform(self.t_min, self.t_max, n)
        elif edge == "top":
            ys = np.full(n, self.y_max)
            xs = rng.uniform(self.x_min, self.x_max, n)
            ts = rng.uniform(self.t_min, self.t_max, n)
        else:
            raise ValueError(f"Unknown edge: {edge}")
        return np.stack([xs, ys, ts], axis=1)


def load_params(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(path: str, header: List[str], rows: np.ndarray):
    # rows expected shape (N, len(header))
    # mixed dtypes not supported here; we only write numeric and encode categorical as ints where needed
    np.savetxt(path, rows, delimiter=",", header=",".join(header), comments="")


def write_json(path: str, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def compute_lame_parameters(E: float, nu: float) -> Tuple[float, float]:
    G = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, G


def main():
    params = load_params(os.path.join(ROOT, "params.json"))

    seed = int(params.get("seed", 42))
    rng = np.random.default_rng(seed)

    # domain
    domain = Domain()

    os.makedirs(DATA_DIR, exist_ok=True)

    # 1) interior collocation points
    n_interior = int(params.get("num_interior", 20000))
    interior = domain.sample_interior(n_interior, rng)
    # columns: x,y,t
    save_csv(
        os.path.join(DATA_DIR, "collocation_interior.csv"),
        ["x", "y", "t"],
        interior,
    )

    # 2) boundary points for different physics
    n_edge = int(params.get("num_points_per_edge", 3000))

    # Thermal BCs
    thermal_cfg = params.get("thermal", {})
    k = float(thermal_cfg.get("k", 15.0))
    h_top = float(thermal_cfg.get("h_top", 50.0))
    T_inf = float(thermal_cfg.get("T_inf", 300.0))

    # left, right, bottom: insulated  => -k grad T · n = 0
    # top: convective => -k grad T · n = h (T - T_inf)
    edges = ["left", "right", "bottom", "top"]
    edge_normals = {
        "left": np.array([-1.0, 0.0]),
        "right": np.array([1.0, 0.0]),
        "bottom": np.array([0.0, -1.0]),
        "top": np.array([0.0, 1.0]),
    }

    thermal_rows = []
    for edge in edges:
        pts = domain.sample_edge(edge, n_edge, rng)
        nvec = edge_normals[edge]
        if edge == "top":
            # convective
            # columns: x,y,t,nx,ny,bc_type, h, T_inf
            bc_type = 2  # 0: insulated, 1: dirichlet(not used here), 2: convective
            add = np.column_stack(
                [
                    pts,
                    np.full((pts.shape[0], 1), nvec[0]),
                    np.full((pts.shape[0], 1), nvec[1]),
                    np.full((pts.shape[0], 1), bc_type),
                    np.full((pts.shape[0], 1), h_top),
                    np.full((pts.shape[0], 1), T_inf),
                ]
            )
        else:
            # insulated
            bc_type = 0
            add = np.column_stack(
                [
                    pts,
                    np.full((pts.shape[0], 1), nvec[0]),
                    np.full((pts.shape[0], 1), nvec[1]),
                    np.full((pts.shape[0], 1), bc_type),
                    np.full((pts.shape[0], 1), 0.0),  # h
                    np.full((pts.shape[0], 1), 0.0),  # T_inf
                ]
            )
        thermal_rows.append(add)
    thermal_rows = np.concatenate(thermal_rows, axis=0)
    save_csv(
        os.path.join(DATA_DIR, "boundary_thermal.csv"),
        ["x", "y", "t", "n_x", "n_y", "bc_type", "h", "T_inf"],
        thermal_rows,
    )

    # Electrical BCs
    elec_cfg = params.get("electrical", {})
    sigma_e = float(elec_cfg.get("sigma_e", 5e6))
    V0_left = float(elec_cfg.get("V0_left", 1.0))
    bv = elec_cfg.get("butler_volmer", {})
    j0 = float(bv.get("j0", 100.0))
    alpha_a = float(bv.get("alpha_a", 0.5))
    alpha_c = float(bv.get("alpha_c", 0.5))
    E_eq = float(bv.get("E_eq", 0.0))

    electrical_rows = []
    for edge in edges:
        pts = domain.sample_edge(edge, n_edge, rng)
        nvec = edge_normals[edge]
        if edge == "left":
            # Dirichlet phi = V0
            bc_type = 1  # 1: Dirichlet
            value = V0_left
            flux = 0.0
            mode = 0  # 0: dirichlet, 1: neumann, 2: butler-volmer
            add = np.column_stack(
                [
                    pts,
                    np.full((pts.shape[0], 1), nvec[0]),
                    np.full((pts.shape[0], 1), nvec[1]),
                    np.full((pts.shape[0], 1), mode),
                    np.full((pts.shape[0], 1), value),
                    np.full((pts.shape[0], 1), flux),
                    np.full((pts.shape[0], 1), j0),
                    np.full((pts.shape[0], 1), alpha_a),
                    np.full((pts.shape[0], 1), alpha_c),
                    np.full((pts.shape[0], 1), E_eq),
                ]
            )
        elif edge == "right":
            # Dirichlet phi = 0
            bc_type = 1
            value = 0.0
            flux = 0.0
            mode = 0
            add = np.column_stack(
                [
                    pts,
                    np.full((pts.shape[0], 1), nvec[0]),
                    np.full((pts.shape[0], 1), nvec[1]),
                    np.full((pts.shape[0], 1), mode),
                    np.full((pts.shape[0], 1), value),
                    np.full((pts.shape[0], 1), flux),
                    np.full((pts.shape[0], 1), j0),
                    np.full((pts.shape[0], 1), alpha_a),
                    np.full((pts.shape[0], 1), alpha_c),
                    np.full((pts.shape[0], 1), E_eq),
                ]
            )
        elif edge == "top":
            # Butler-Volmer (Neumann current specified via kinetics)
            mode = 2
            value = 0.0  # not used for BV
            flux = np.nan  # computed in loss via kinetics
            add = np.column_stack(
                [
                    pts,
                    np.full((pts.shape[0], 1), nvec[0]),
                    np.full((pts.shape[0], 1), nvec[1]),
                    np.full((pts.shape[0], 1), mode),
                    np.full((pts.shape[0], 1), value),
                    np.full((pts.shape[0], 1), flux),
                    np.full((pts.shape[0], 1), j0),
                    np.full((pts.shape[0], 1), alpha_a),
                    np.full((pts.shape[0], 1), alpha_c),
                    np.full((pts.shape[0], 1), E_eq),
                ]
            )
        else:
            # bottom insulated (no normal current): Neumann j·n = 0
            mode = 1
            value = 0.0
            flux = 0.0
            add = np.column_stack(
                [
                    pts,
                    np.full((pts.shape[0], 1), nvec[0]),
                    np.full((pts.shape[0], 1), nvec[1]),
                    np.full((pts.shape[0], 1), mode),
                    np.full((pts.shape[0], 1), value),
                    np.full((pts.shape[0], 1), flux),
                    np.full((pts.shape[0], 1), j0),
                    np.full((pts.shape[0], 1), alpha_a),
                    np.full((pts.shape[0], 1), alpha_c),
                    np.full((pts.shape[0], 1), E_eq),
                ]
            )
        electrical_rows.append(add)
    electrical_rows = np.concatenate(electrical_rows, axis=0)
    save_csv(
        os.path.join(DATA_DIR, "boundary_electrical.csv"),
        [
            "x",
            "y",
            "t",
            "n_x",
            "n_y",
            "mode",
            "phi_value",
            "j_flux",
            "j0",
            "alpha_a",
            "alpha_c",
            "E_eq",
        ],
        electrical_rows,
    )

    # Mechanical BCs
    mech_cfg = params.get("mechanical", {})
    E = float(mech_cfg.get("E", 200e9))
    nu = float(mech_cfg.get("nu", 0.3))
    alpha_T = float(mech_cfg.get("alpha_T", 1.2e-5))
    T0 = float(mech_cfg.get("T0", 300.0))
    lam, G = compute_lame_parameters(E, nu)

    mechanical_rows = []
    for edge in edges:
        pts = domain.sample_edge(edge, n_edge, rng)
        nvec = edge_normals[edge]
        if edge == "left":
            # Dirichlet u=(0,0)
            mode = 0  # 0: dirichlet displacement, 1: traction
            u_x = 0.0
            u_y = 0.0
            t_x = 0.0
            t_y = 0.0
            add = np.column_stack(
                [
                    pts,
                    np.full((pts.shape[0], 1), nvec[0]),
                    np.full((pts.shape[0], 1), nvec[1]),
                    np.full((pts.shape[0], 1), mode),
                    np.full((pts.shape[0], 1), u_x),
                    np.full((pts.shape[0], 1), u_y),
                    np.full((pts.shape[0], 1), t_x),
                    np.full((pts.shape[0], 1), t_y),
                ]
            )
        else:
            # traction-free
            mode = 1
            u_x = np.nan
            u_y = np.nan
            t_x = 0.0
            t_y = 0.0
            add = np.column_stack(
                [
                    pts,
                    np.full((pts.shape[0], 1), nvec[0]),
                    np.full((pts.shape[0], 1), nvec[1]),
                    np.full((pts.shape[0], 1), mode),
                    np.full((pts.shape[0], 1), u_x),
                    np.full((pts.shape[0], 1), u_y),
                    np.full((pts.shape[0], 1), t_x),
                    np.full((pts.shape[0], 1), t_y),
                ]
            )
        mechanical_rows.append(add)
    mechanical_rows = np.concatenate(mechanical_rows, axis=0)
    save_csv(
        os.path.join(DATA_DIR, "boundary_mechanical.csv"),
        [
            "x",
            "y",
            "t",
            "n_x",
            "n_y",
            "mode",
            "u_x",
            "u_y",
            "t_x",
            "t_y",
        ],
        mechanical_rows,
    )

    # Save combined NPZ for convenience
    np.savez(
        os.path.join(DATA_DIR, "dataset_npz.npz"),
        interior=interior,
        boundary_thermal=thermal_rows,
        boundary_electrical=electrical_rows,
        boundary_mechanical=mechanical_rows,
        params=np.array([
            [k, h_top, T_inf, sigma_e, V0_left, E, nu, alpha_T, T0, lam, G]
        ]),
    )

    # metadata for loader
    metadata = {
        "files": {
            "interior": "collocation_interior.csv",
            "thermal": "boundary_thermal.csv",
            "electrical": "boundary_electrical.csv",
            "mechanical": "boundary_mechanical.csv",
            "packed": "dataset_npz.npz"
        },
        "columns": {
            "interior": ["x", "y", "t"],
            "thermal": ["x", "y", "t", "n_x", "n_y", "bc_type", "h", "T_inf"],
            "electrical": [
                "x",
                "y",
                "t",
                "n_x",
                "n_y",
                "mode",
                "phi_value",
                "j_flux",
                "j0",
                "alpha_a",
                "alpha_c",
                "E_eq"
            ],
            "mechanical": ["x", "y", "t", "n_x", "n_y", "mode", "u_x", "u_y", "t_x", "t_y"]
        },
        "modes": {
            "thermal.bc_type": {"0": "insulated", "2": "convective"},
            "electrical.mode": {"0": "dirichlet", "1": "neumann", "2": "butler_volmer"},
            "mechanical.mode": {"0": "dirichlet_u", "1": "traction"}
        }
    }
    write_json(os.path.join(DATA_DIR, "metadata.json"), metadata)

    print("Generated physics dataset in:", DATA_DIR)


if __name__ == "__main__":
    main()
