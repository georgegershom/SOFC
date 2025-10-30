#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict
import sys

# 将项目根目录加入 sys.path，使得可导入 physics 包
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from physics.domains import Domain2D, EDGE_ID_TO_NAME
from physics.boundary_conditions import (
    encode_thermal_bc_for_edge,
    encode_electrical_bc_for_edge,
    encode_mechanical_bc_for_edge,
)


def load_config(cfg_path: Path) -> Dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 PINN 物理约束的采样点数据集（内点+边界点）")
    parser.add_argument("--config", type=str, required=True, help="配置文件 physics_config.yaml")
    parser.add_argument("--out", type=str, required=True, help="输出目录（将写入 .npz 与 config 副本）")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    # 域参数
    Lx = float(cfg["domain"]["Lx"])
    Ly = float(cfg["domain"]["Ly"])
    t_final = float(cfg["domain"].get("t_final", 0.0))
    include_time = bool(cfg["domain"].get("include_time", False))
    reactive = cfg["domain"].get("reactive_region", {})
    rx_min = reactive.get("x_min", None)
    rx_max = reactive.get("x_max", None)

    seed = int(cfg["project"].get("seed", 42))

    domain = Domain2D(
        length_x=Lx,
        length_y=Ly,
        t_final=t_final,
        include_time=include_time,
        reactive_x_min=rx_min,
        reactive_x_max=rx_max,
        random_seed=seed,
    )

    # 采样参数
    n_interior = int(cfg["sampling"].get("n_interior", 10000))
    n_boundary_per_edge = int(cfg["sampling"].get("n_boundary_per_edge", 1000))

    interior = domain.sample_interior(n_interior)
    boundary = domain.sample_boundary(n_boundary_per_edge)

    # 编码边界条件（按 edge 分片写回）
    boundaries_cfg = cfg.get("boundaries", {})
    edge_ids = boundary["edge_id"]
    N_b = edge_ids.shape[0]

    # 预分配
    thermal_type = np.zeros(N_b, dtype=np.int32)
    thermal_value = np.zeros(N_b, dtype=np.float64)
    thermal_flux = np.zeros(N_b, dtype=np.float64)
    thermal_h = np.zeros(N_b, dtype=np.float64)
    thermal_T_inf = np.zeros(N_b, dtype=np.float64)

    phi_s_type = np.zeros(N_b, dtype=np.int32)
    phi_s_value = np.zeros(N_b, dtype=np.float64)
    phi_s_flux = np.zeros(N_b, dtype=np.float64)

    phi_e_type = np.zeros(N_b, dtype=np.int32)
    phi_e_value = np.zeros(N_b, dtype=np.float64)
    phi_e_flux = np.zeros(N_b, dtype=np.float64)

    fix_ux = np.zeros(N_b, dtype=np.int32)
    fix_uy = np.zeros(N_b, dtype=np.int32)
    traction_x = np.zeros(N_b, dtype=np.float64)
    traction_y = np.zeros(N_b, dtype=np.float64)

    for eid in np.unique(edge_ids):
        mask = (edge_ids == eid)
        n = int(mask.sum())
        edge_name = EDGE_ID_TO_NAME[int(eid)]

        therm = encode_thermal_bc_for_edge(boundaries_cfg, edge_name, n)
        thermal_type[mask] = therm["thermal_type"]
        thermal_value[mask] = therm["thermal_value"]
        thermal_flux[mask] = therm["thermal_flux"]
        thermal_h[mask] = therm["thermal_h"]
        thermal_T_inf[mask] = therm["thermal_T_inf"]

        elec = encode_electrical_bc_for_edge(boundaries_cfg, edge_name, n)
        phi_s_type[mask] = elec["phi_s_type"]
        phi_s_value[mask] = elec["phi_s_value"]
        phi_s_flux[mask] = elec["phi_s_flux"]
        phi_e_type[mask] = elec["phi_e_type"]
        phi_e_value[mask] = elec["phi_e_value"]
        phi_e_flux[mask] = elec["phi_e_flux"]

        mech = encode_mechanical_bc_for_edge(boundaries_cfg, edge_name, n)
        fix_ux[mask] = mech["fix_ux"]
        fix_uy[mask] = mech["fix_uy"]
        traction_x[mask] = mech["traction_x"]
        traction_y[mask] = mech["traction_y"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存 npz 数据集
    out_npz = out_dir / "pinn_physics_collocation_v1.npz"
    np.savez_compressed(
        out_npz,
        # interior
        interior_x=interior["x"],
        interior_y=interior["y"],
        interior_t=interior["t"],
        interior_is_reactive=interior["is_reactive"],
        # boundary
        boundary_x=boundary["x"],
        boundary_y=boundary["y"],
        boundary_t=boundary["t"],
        boundary_nx=boundary["nx"],
        boundary_ny=boundary["ny"],
        boundary_edge_id=boundary["edge_id"],
        # thermal BC
        bc_thermal_type=thermal_type,
        bc_thermal_value=thermal_value,
        bc_thermal_flux=thermal_flux,
        bc_thermal_h=thermal_h,
        bc_thermal_T_inf=thermal_T_inf,
        # electrical BC
        bc_phi_s_type=phi_s_type,
        bc_phi_s_value=phi_s_value,
        bc_phi_s_flux=phi_s_flux,
        bc_phi_e_type=phi_e_type,
        bc_phi_e_value=phi_e_value,
        bc_phi_e_flux=phi_e_flux,
        # mechanical BC
        bc_fix_ux=fix_ux,
        bc_fix_uy=fix_uy,
        bc_traction_x=traction_x,
        bc_traction_y=traction_y,
        # meta
        domain_Lx=Lx,
        domain_Ly=Ly,
        domain_t_final=t_final,
        domain_include_time=1 if include_time else 0,
    )

    # 把 config 也拷贝一份到数据目录
    with (out_dir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    # 简要元信息
    meta = {
        "counts": {
            "interior": int(interior["x"].shape[0]),
            "boundary": int(boundary["x"].shape[0]),
        },
        "edges": {int(k): v for k, v in EDGE_ID_TO_NAME.items()},
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Wrote dataset: {out_npz}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
