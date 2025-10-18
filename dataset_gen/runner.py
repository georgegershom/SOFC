from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from .config import DATASET_ROOT, RANDOM_SEED, Specimen, enumerate_specimens
from .sem_generator import generate_sem_for_specimen
from .xrd_generator import generate_xrd_for_specimen
from .tga_dta_generator import generate_tga_dta_for_specimen
from .micro_ct_generator import generate_micro_ct_for_specimen
from .config import model_parameters
from .utils import ensure_dir


def _paths(root: Path) -> Dict[str, Path]:
    return {
        "root": root,
        "sem": root / "sem",
        "xrd": root / "xrd",
        "tga_dta": root / "tga_dta",
        "micro_ct": root / "micro_ct",
        "meta": root / "dataset_info",
    }


def generate_dataset(root: Path | None = None) -> None:
    paths = _paths(root or DATASET_ROOT)
    for p in paths.values():
        ensure_dir(p)

    all_specimens: List[Specimen] = enumerate_specimens()

    # Metrics collectors
    sem_rows: List[Dict] = []
    xrd_rows: List[Dict] = []
    tga_rows: List[Dict] = []
    ct_rows: List[Dict] = []

    for spec in all_specimens:
        sid = spec.specimen_id
        params = model_parameters(spec)

        sem_metrics = generate_sem_for_specimen(paths["sem"], sid, params)
        xrd_metrics = generate_xrd_for_specimen(paths["xrd"], sid, params)
        tga_metrics = generate_tga_dta_for_specimen(paths["tga_dta"], sid, params)
        ct_metrics = generate_micro_ct_for_specimen(paths["micro_ct"], sid, params)

        base = {
            "specimen_id": sid,
            "mixture": spec.mixture_code,
            "rubber_pct": spec.rubber_pct,
            "temperature_C": spec.temperature_C,
            "replicate": spec.replicate,
        }

        sem_rows.append({**base, **sem_metrics})
        xrd_rows.append({**base, **xrd_metrics})
        tga_rows.append({**base, **tga_metrics})
        ct_rows.append({**base, **ct_metrics})

    # Write metrics CSVs
    def write_csv(path: Path, rows: List[Dict]) -> None:
        if not rows:
            return
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write_csv(paths["sem"] / "metrics.csv", sem_rows)
    write_csv(paths["xrd"] / "metrics.csv", xrd_rows)
    write_csv(paths["tga_dta"] / "metrics.csv", tga_rows)
    write_csv(paths["micro_ct"] / "metrics.csv", ct_rows)

    # Aggregate summary linking microstructure/chemistry to macro behavior
    # Simple explanatory metrics
    summary = []
    for i in range(len(ct_rows)):
        row = {**ct_rows[i], **sem_rows[i], **xrd_rows[i], **tga_rows[i]}
        # A simple degradation index combining porosity, cracks, and CH loss
        ch_area = row["xrd_ch_rel_area"]
        caco3_area = row["xrd_caco3_rel_area"]
        void = row["ct_void_plus_crack"]
        itz_poro = row["sem_itz_porosity"]
        mass600 = row["tga_mass_600C"]
        # Normalize proxies roughly
        deg_index = (void * 2.0 + itz_poro * 1.5 + (1.0 - min(1.0, ch_area / (ch_area + 1.0))) * 1.2 + (1.0 - mass600) * 1.3)
        row["degradation_index"] = float(deg_index)
        summary.append(row)

    write_csv(paths["root"] / "summary_metrics.csv", summary)

    # Metadata and data dictionary
    _write_metadata(paths)


def _write_metadata(paths: Dict[str, Path]) -> None:
    meta_dir = paths["meta"]
    ensure_dir(meta_dir)

    data_dict = {
        "sem": {
            "images": "SEM-like PGM images focusing on ITZ. 512x512 uint8 grayscale.",
            "metrics.csv": {
                "sem_crack_length_mm": "Total crack length estimated from binary cracks across views (mm).",
                "sem_itz_porosity": "Porosity fraction within ITZ mask (0-1).",
                "sem_itz_width_px": "ITZ shell width used for analysis (pixels).",
            },
        },
        "xrd": {
            "*_xrd.csv": "Two-column CSV (two_theta_deg, intensity).",
            "metrics.csv": {
                "xrd_ch_rel_area": "Integrated area around main CH peaks (relative).",
                "xrd_caco3_rel_area": "Integrated area around main calcite peaks (relative).",
                "xrd_cao_rel_area": "Integrated area around CaO peaks (relative).",
                "xrd_csh_hump_area": "Integrated C-S-H amorphous hump area (relative).",
            },
        },
        "tga_dta": {
            "*_tga_dta.csv": "Three-column CSV (temperature_C, mass, dtg).",
            "metrics.csv": {
                "tga_mass_200C": "Residual mass at 200°C.",
                "tga_mass_400C": "Residual mass at 400°C.",
                "tga_mass_600C": "Residual mass at 600°C.",
                "tga_mass_800C": "Residual mass at 800°C.",
                "tga_total_loss": "Total mass loss up to 1000°C.",
            },
        },
        "micro_ct": {
            "*_volume.npy": "3D labeled volume (uint8): 0=void,1=paste,2=aggregate,3=rubber/residue,4=crack.",
            "*_slice_[Z|Y|X].pgm": "Central slices previews in PGM grayscale.",
            "metrics.csv": {
                "ct_porosity": "Void fraction (excludes cracks).",
                "ct_crack_fraction": "Crack volume fraction.",
                "ct_void_plus_crack": "Combined void+crack fraction.",
                "ct_connected_porosity_frac": "Largest connected porosity fraction (downsampled).",
                "ct_itz_width_vox": "ITZ shell width used for porosity boost (voxels).",
            },
        },
    }

    with open(meta_dir / "data_dictionary.json", "w") as f:
        json.dump(data_dict, f, indent=2)

    readme = f"""
Phase 3: Microstructural & Chemical Analysis Dataset (Synthetic)
===============================================================

Theme: Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete (HPRC)

Contents
- SEM (ITZ-focused) grayscale images (.pgm) with crack/porosity metrics
- XRD patterns (.csv) with semi-quantitative phase areas (CH, CaCO3, CaO, C-S-H hump)
- TGA/DTG curves (.csv) with mass loss metrics vs temperature
- Micro-CT 3D labeled volumes (.npy) with porosity/crack fractions and connectivity
- Summary metrics linking microstructure/chemistry to macro degradation index

Interpretation (why macro-behavior occurs)
- Heating drives dehydroxylation of Portlandite (CH) and dehydration of C-S-H, reducing cohesive strength.
- Above ~700–900°C, decarbonation of CaCO3 and formation of CaO further embrittle the matrix.
- Rubber particles pyrolyze ~300–500°C, leaving voids; the ITZ around rubber weakens and fosters microcrack initiation.
- Thermal incompatibility between aggregates and paste amplifies ITZ microcracking.
- The emerging interconnected pore/crack network (captured by Micro-CT and SEM metrics) increases permeability and reduces load-bearing cross-section, explaining mass/strength loss.

Usage
- See summary_metrics.csv at the dataset root to correlate temperature, rubber content, phase changes, and microstructural metrics with a degradation index.

Note: All data are synthetic but constrained by temperature-dependent phenomena reported for cementitious materials and rubberized concrete.
"""
    with open(meta_dir / "README.md", "w") as f:
        f.write(readme)


if __name__ == "__main__":
    generate_dataset()
