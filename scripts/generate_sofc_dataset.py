#!/usr/bin/env python3
import argparse
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Optional dependency for image writing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

RNG = np.random.default_rng(42)

@dataclass
class GridSpec:
    nx: int
    ny: int
    nz: int
    voxel_um: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


def synthesize_microstructure(grid: GridSpec, phases: List[str], out_dir: Path) -> List[dict]:
    ensure_dir(out_dir)
    micro_entries: List[dict] = []
    for phase in phases:
        # generate a synthetic porous medium as 3D volume (0/1)
        volume = RNG.normal(0, 1, size=(grid.nz, grid.ny, grid.nx))
        threshold = 0.0
        binary = (volume > threshold).astype(np.uint8)
        # Save as .npy
        vol_path = out_dir / f"micro_{phase}.npy"
        np.save(vol_path, binary)
        # Fake effective properties
        porosity = float(binary.mean())
        tortuosity = float(1.5 + 0.5 * RNG.random())
        conductivity = float(0.5 + 0.5 * RNG.random())
        eff_props = {
            "phase": phase,
            "porosity": porosity,
            "tortuosity": tortuosity,
            "effective_electric_conductivity_S_m": conductivity,
        }
        eff_path = out_dir / f"micro_{phase}_effective_properties.json"
        write_json(eff_path, eff_props)
        micro_entries.append({
            "id": f"micro_{phase}",
            "phase": phase,
            "format": "npy",
            "path": str(vol_path.relative_to(out_dir.parent.parent)),
            "voxel_size_um": grid.voxel_um,
            "derived_effective_properties_path": str(eff_path.relative_to(out_dir.parent.parent)),
        })
    return micro_entries


def synthesize_macro_geometry(out_dir: Path) -> List[dict]:
    ensure_dir(out_dir)
    components = [
        ("cell", "stl"),
        ("interconnect", "stl"),
        ("seal", "stl"),
        ("manifold", "stl"),
        ("stack", "stl"),
    ]
    entries: List[dict] = []
    for comp, ext in components:
        path = out_dir / f"{comp}.{ext}"
        # Minimal STL placeholder
        stl = """solid placeholder\nendsolid placeholder\n"""
        path.write_text(stl)
        entries.append({
            "id": f"geom_{comp}",
            "component": comp,
            "format": ext,
            "path": str(path.relative_to(out_dir.parent.parent)),
        })
    return entries


def synthesize_operational_inputs(n_points: int, out_dir: Path) -> Tuple[Path, List[dict]]:
    ensure_dir(out_dir)
    start = datetime(2025, 1, 1, 0, 0, 0)
    times = np.array([start + timedelta(seconds=i) for i in range(n_points)])
    # Inputs: fuel comps, flows, inlet temps, current density
    data = []
    for i in range(n_points):
        h2 = float(0.6 + 0.05 * np.sin(i / 50))
        co = float(0.1 + 0.02 * np.cos(i / 40))
        ch4 = float(max(0.0, 0.05 + 0.01 * np.sin(i / 70)))
        n2 = float(max(0.0, 1.0 - h2 - co - ch4))
        fuel_flow_sccm = float(300 + 20 * np.sin(i / 30))
        air_flow_sccm = float(1000 + 80 * np.cos(i / 45))
        tin_fuel_C = float(750 + 5 * np.sin(i / 60))
        tin_air_C = float(750 + 5 * np.cos(i / 60))
        current_density_A_cm2 = float(0.5 + 0.05 * np.sin(i / 35))
        data.append({
            "time": times[i].isoformat(),
            "fuel_H2": h2,
            "fuel_CO": co,
            "fuel_CH4": ch4,
            "fuel_N2": n2,
            "fuel_flow_sccm": fuel_flow_sccm,
            "air_flow_sccm": air_flow_sccm,
            "tin_fuel_C": tin_fuel_C,
            "tin_air_C": tin_air_C,
            "current_density_A_cm2": current_density_A_cm2,
        })
    inputs_path = out_dir / "operational_inputs.jsonl"
    with inputs_path.open("w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")
    items = [{"id": "inputs_stream_1", "path": str(inputs_path.relative_to(out_dir.parent.parent)), "description": "Primary boundary conditions"}]
    return inputs_path, items


def synthesize_electrochem(n_points: int, out_dir: Path) -> List[dict]:
    ensure_dir(out_dir)
    # Polarization curve samples
    current = np.linspace(0.1, 1.0, num=50)
    voltage = 1.05 - 0.6 * (current - 0.1) + 0.02 * np.random.default_rng(0).normal(size=current.shape)
    pol = {
        "current_density_A_cm2": current.tolist(),
        "voltage_V": voltage.tolist(),
        "temperature_C": 800,
    }
    pol_path = out_dir / "polarization.json"
    write_json(pol_path, pol)
    # EIS spectra at a few operating points
    freqs = np.logspace(5, -1, num=30).tolist()
    eis_entries = []
    for idx, tempC in enumerate([750, 800, 850]):
        Z_re = (0.2 + 0.05 * RNG.random(len(freqs))).tolist()
        Z_im = (0.15 + 0.03 * RNG.random(len(freqs))).tolist()
        eis = {"frequency_Hz": freqs, "Z_re_Ohm": Z_re, "Z_im_Ohm": Z_im, "temperature_C": tempC}
        path = out_dir / f"eis_{idx}.json"
        write_json(path, eis)
        eis_entries.append({"id": f"eis_{idx}", "type": "eis", "path": str(path.relative_to(out_dir.parent.parent))})
    entries = [
        {"id": "polarization_1", "type": "polarization", "path": str(pol_path.relative_to(out_dir.parent.parent))}
    ] + eis_entries
    return entries


def synthesize_thermocouples(n_points: int, out_dir: Path, label: str) -> List[dict]:
    ensure_dir(out_dir)
    locations = ["inlet", "center", "outlet"]
    start = datetime(2025, 1, 1)
    entries = []
    for loc in locations:
        path = out_dir / f"tc_{loc}.jsonl"
        with path.open("w") as f:
            for i in range(n_points):
                t = start + timedelta(seconds=i)
                base = 760 if loc == "center" else 750
                temp = float(base + 5 * np.sin(i / 40) + RNG.normal(0, 0.3))
                f.write(json.dumps({"time": t.isoformat(), "T_C": temp}) + "\n")
        entries.append({"id": f"tc_{label}_{loc}", "path": str(path.relative_to(out_dir.parent.parent)), "location": loc})
    return entries


def synthesize_ir(out_dir: Path, n_frames: int, width: int = 128, height: int = 96, fps: int = 10) -> List[dict]:
    ensure_dir(out_dir)
    frames_dir = out_dir / "ir_frames"
    ensure_dir(frames_dir)
    meta_entries: List[dict] = []
    for i in range(n_frames):
        frame = 740 + 15 * np.exp(-((np.indices((height, width))[1] - width/2)**2 + (np.indices((height, width))[0] - height/2)**2) / (2*(min(width, height)/5)**2))
        frame += 3 * np.sin(i / 10)
        frame += RNG.normal(0, 0.5, size=(height, width))
        # Normalize to 8-bit for visualization if PIL available
        if PIL_AVAILABLE:
            img = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255).astype(np.uint8)
            Image.fromarray(img).save(frames_dir / f"frame_{i:04d}.png")
        # Save raw temperature array for fidelity
        np.save(frames_dir / f"frame_{i:04d}.npy", frame.astype(np.float32))
    meta_entries.append({
        "id": "ir_run_1",
        "path": str(frames_dir.relative_to(out_dir.parent.parent)),
        "fps": fps,
        "pixel_size_mm": 0.2,
    })
    return meta_entries


def synthesize_strain_gauges(n_points: int, out_dir: Path, label: str) -> List[dict]:
    ensure_dir(out_dir)
    locations = ["ic_left", "ic_right"]
    start = datetime(2025, 1, 1)
    entries = []
    for loc in locations:
        path = out_dir / f"sg_{loc}.jsonl"
        with path.open("w") as f:
            for i in range(n_points):
                t = start + timedelta(seconds=i)
                strain = float(500e-6 + 50e-6 * np.sin(i / 60) + RNG.normal(0, 5e-6))
                f.write(json.dumps({"time": t.isoformat(), "strain": strain}) + "\n")
        entries.append({"id": f"sg_{label}_{loc}", "path": str(path.relative_to(out_dir.parent.parent)), "location": loc})
    return entries


def synthesize_dic(out_dir: Path, n_frames: int, width: int = 64, height: int = 64, fps: int = 2) -> List[dict]:
    ensure_dir(out_dir)
    dic_dir = out_dir / "dic_frames"
    ensure_dir(dic_dir)
    for i in range(n_frames):
        # synthetic 2D strain field
        X, Y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
        field = 400e-6 + 60e-6 * X * Y + 20e-6 * np.sin(2 * np.pi * (i / max(1, n_frames - 1)))
        np.save(dic_dir / f"strain_{i:04d}.npy", field.astype(np.float32))
    return [{"id": "dic_test_1", "path": str(dic_dir.relative_to(out_dir.parent.parent)), "fps": fps, "scale_mm_per_px": 0.1}]


def synthesize_aging_tests(root: Path, n_points: int) -> List[dict]:
    tests = [
        ("thermal_cycling", 2),
        ("redox_cycling", 1),
        ("steady_state", 1),
    ]
    entries: List[dict] = []
    for proto, reps in tests:
        for r in range(reps):
            label = f"{proto}_{r+1}"
            test_dir = root / label
            ensure_dir(test_dir)
            inputs_dir = test_dir / "inputs"
            electrochem_dir = test_dir / "electrochem"
            temp_dir = test_dir / "temperature"
            strain_dir = test_dir / "strain"
            ensure_dir(inputs_dir)
            ensure_dir(electrochem_dir)
            ensure_dir(temp_dir)
            ensure_dir(strain_dir)
            synthesize_operational_inputs(n_points, inputs_dir)
            synthesize_electrochem(n_points, electrochem_dir)
            synthesize_thermocouples(n_points, temp_dir, label)
            synthesize_ir(temp_dir, n_frames=50)
            synthesize_strain_gauges(n_points, strain_dir, label)
            synthesize_dic(strain_dir, n_frames=20)
            entries.append({
                "id": label,
                "protocol": proto,
                "paths": {
                    "inputs": str(inputs_dir.relative_to(root.parent)),
                    "electrochem": str(electrochem_dir.relative_to(root.parent)),
                    "temperature": str(temp_dir.relative_to(root.parent)),
                    "strain": str(strain_dir.relative_to(root.parent)),
                }
            })
    return entries


def synthesize_post_mortem(out_dir: Path) -> List[dict]:
    ensure_dir(out_dir)
    entries = []
    for idx, img_type in enumerate(["sem", "eds"]):
        path = out_dir / f"{img_type}_{idx}.png"
        if PIL_AVAILABLE:
            arr = (RNG.random((256, 256)) * 255).astype(np.uint8)
            Image.fromarray(arr).save(path)
        else:
            path.write_bytes(b"")
        entries.append({"id": f"pm_{img_type}_{idx}", "type": img_type, "path": str(path.relative_to(out_dir.parent.parent))})
    return entries


def synthesize_high_fidelity_model(out_dir: Path) -> dict:
    ensure_dir(out_dir)
    mesh = out_dir / "mesh.msh"
    bc = out_dir / "boundary_conditions.json"
    mesh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
    write_json(bc, {"temperature_inlet_C": 750, "temperature_outlet_C": 760})
    return {"mesh_path": str(mesh.relative_to(out_dir.parent.parent)), "boundary_conditions_path": str(bc.relative_to(out_dir.parent.parent))}


def synthesize_rom_training(out_dir: Path, n_samples: int) -> dict:
    ensure_dir(out_dir)
    inputs = []
    outputs = []
    for i in range(n_samples):
        h2 = RNG.uniform(0.5, 0.7)
        flow = RNG.uniform(250, 350)
        current = RNG.uniform(0.3, 0.8)
        tin = RNG.uniform(730, 770)
        inputs.append({"fuel_H2": float(h2), "fuel_flow_sccm": float(flow), "current_density_A_cm2": float(current), "tin_C": float(tin)})
        # Simple mapping to outputs: voltage, max temp, max von Mises stress (synthetic)
        voltage = 1.1 - 0.6 * current + 0.02 * (h2 - 0.6)
        tmax = tin + 20 + 5 * (flow - 300) / 100
        stress = 50 + 0.1 * (tmax - 750) + 10 * (current - 0.5)
        outputs.append({"voltage_V": float(voltage), "Tmax_C": float(tmax), "vonMises_MPa": float(stress)})
    in_csv = out_dir / "rom_inputs.csv"
    out_csv = out_dir / "rom_outputs.csv"
    import csv
    with in_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inputs[0].keys()))
        w.writeheader(); w.writerows(inputs)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(outputs[0].keys()))
        w.writeheader(); w.writerows(outputs)
    return {"inputs_csv": str(in_csv.relative_to(out_dir.parent.parent)), "outputs_csv": str(out_csv.relative_to(out_dir.parent.parent))}


def synthesize_reduced_order(out_dir: Path) -> dict:
    ensure_dir(out_dir)
    rom_file = out_dir / "rom_artifact.json"
    write_json(rom_file, {"type": "nn", "note": "placeholder ROM artifact"})
    return {"rom_type": "nn", "artifact_path": str(rom_file.relative_to(out_dir.parent.parent))}


def build_manifest(root: Path, micro_entries: List[dict], macro_entries: List[dict], inputs_items: List[dict], electrochem_entries: List[dict], tc_entries: List[dict], ir_entries: List[dict], sg_entries: List[dict], dic_entries: List[dict], aging_entries: List[dict], pm_entries: List[dict], hf_model: dict, rom: dict, rom_training: dict) -> dict:
    return {
        "version": "0.1.0",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "description": "Synthetic dataset for Adaptive-Scale PI Digital Twin for SOFC.",
        "components": {
            "materials_geometry": {
                "microstructure": micro_entries,
                "macro_geometry": macro_entries,
            },
            "operational_performance": {
                "inputs": inputs_items,
                "electrochem": electrochem_entries,
            },
            "thermo_structural": {
                "temperature": {
                    "thermocouples": tc_entries,
                    "ir_images": ir_entries,
                },
                "strain": {
                    "strain_gauges": sg_entries,
                    "dic": dic_entries,
                }
            },
            "degradation": {
                "aging_tests": aging_entries,
                "post_mortem": pm_entries,
            },
            "models": {
                "high_fidelity": hf_model,
                "reduced_order": rom,
                "rom_training_data": rom_training,
            }
        }
    }


def generate(root: Path) -> Path:
    dataset_root = root / "datasets/sofc_digital_twin"
    ensure_dir(dataset_root)

    # Materials & Geometry
    micro_dir = dataset_root / "materials_geometry/microstructure"
    macro_dir = dataset_root / "materials_geometry/macro"
    micro_entries = synthesize_microstructure(GridSpec(64, 64, 64, voxel_um=0.2), ["anode", "electrolyte", "cathode"], micro_dir)
    macro_entries = synthesize_macro_geometry(macro_dir)

    # Operational & Electrochem
    op_dir = dataset_root / "operational_performance"
    inputs_path, inputs_items = synthesize_operational_inputs(600, op_dir / "inputs")
    electrochem_entries = synthesize_electrochem(0, op_dir / "electrochem")

    # Thermo-Structural
    thermo_dir = dataset_root / "thermo_structural"
    tc_entries = synthesize_thermocouples(600, thermo_dir / "temperature", "main")
    ir_entries = synthesize_ir(thermo_dir / "temperature", n_frames=100)
    sg_entries = synthesize_strain_gauges(600, thermo_dir / "strain", "main")
    dic_entries = synthesize_dic(thermo_dir / "strain", n_frames=40)

    # Degradation & Post mortem
    degr_dir = dataset_root / "degradation"
    aging_entries = synthesize_aging_tests(degr_dir / "aging_tests", n_points=360)
    pm_entries = synthesize_post_mortem(degr_dir / "post_mortem")

    # Models & ROM data
    model_dir = dataset_root / "models"
    hf_model = synthesize_high_fidelity_model(model_dir / "high_fidelity")
    rom_training = synthesize_rom_training(model_dir / "rom_training_data", n_samples=500)
    rom = synthesize_reduced_order(model_dir / "reduced_order")

    # Manifest
    manifest = build_manifest(dataset_root, micro_entries, macro_entries, inputs_items, electrochem_entries, tc_entries, ir_entries, sg_entries, dic_entries, aging_entries, pm_entries, hf_model, rom, rom_training)
    write_json(dataset_root / "manifest.json", manifest)

    return dataset_root


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SOFC Digital Twin dataset")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Root directory to create dataset in")
    args = parser.parse_args()

    dataset_path = generate(args.root)
    print(str(dataset_path))


if __name__ == "__main__":
    main()
