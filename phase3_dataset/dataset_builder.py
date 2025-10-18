import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import CONFIG
from .utils import ensure_dir, save_json, set_global_seed
from .sem_generator import generate_sem_batch
from .xrd_generator import generate_xrd_batch
from .tga_dta_generator import generate_tga_dta_batch
from .microct_generator import generate_microct_batch


def build_dataset(output_root: str = CONFIG.output_root, rng_seed: int = CONFIG.rng_seed) -> str:
    set_global_seed(rng_seed)

    ensure_dir(output_root)

    temperatures = list(CONFIG.temperatures_c)
    specimens = list(CONFIG.specimens)
    replicates = CONFIG.replicates_per_condition

    manifest: Dict[str, List[Dict]] = {
        "sem": [],
        "xrd": [],
        "tga_dta": [],
        "microct": [],
        "explanations": [],
        "summary": [],
        "conditions": {
            "temperatures_c": temperatures,
            "specimens": specimens,
            "replicates_per_condition": replicates,
        },
        "about": {
            "title": "Phase 3: Microstructural and Chemical Analysis Dataset",
            "topic": "Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete",
            "note": "Synthetic dataset capturing SEM, XRD, TGA/DTA, and Micro-CT modalities with physics-informed trends vs temperature and rubber content.",
        },
    }

    sem_dir = os.path.join(output_root, "sem")
    xrd_dir = os.path.join(output_root, "xrd")
    tga_dir = os.path.join(output_root, "tga_dta")
    microct_dir = os.path.join(output_root, "microct")
    for d in (sem_dir, xrd_dir, tga_dir, microct_dir):
        ensure_dir(d)

    # Buckets for summary statistics keyed by (specimen, temperature)
    buckets: Dict[Tuple[str, int], Dict[str, List[float]]] = {}

    for specimen in specimens:
        for temp_c in temperatures:
            condition = {"specimen": specimen, "temperature_c": temp_c}

            sem_items = generate_sem_batch(sem_dir, specimen, temp_c, replicates)
            manifest["sem"].extend(sem_items)

            xrd_items = generate_xrd_batch(xrd_dir, specimen, temp_c, replicates)
            manifest["xrd"].extend(xrd_items)

            tga_items = generate_tga_dta_batch(tga_dir, specimen, temp_c, replicates)
            manifest["tga_dta"].extend(tga_items)

            micro_items = generate_microct_batch(microct_dir, specimen, temp_c, max(1, replicates // 2))
            manifest["microct"].extend(micro_items)

            manifest["explanations"].append(_explain_condition(specimen, temp_c))

            # Aggregate metrics into buckets
            key = (specimen, temp_c)
            if key not in buckets:
                buckets[key] = {}

            def add_metrics(prefix: str, items: List[Dict]) -> None:
                for it in items:
                    metrics = it.get("metrics", {})
                    for mk, mv in metrics.items():
                        name = f"{prefix}.{mk}"
                        buckets[key].setdefault(name, []).append(float(mv))

            add_metrics("sem", sem_items)
            add_metrics("xrd", xrd_items)
            add_metrics("tga", tga_items)
            add_metrics("microct", micro_items)

    # Build summary statistics
    for (specimen, temp_c), metrics_lists in buckets.items():
        summary_entry: Dict[str, float] = {
            "specimen": specimen,
            "temperature_c": temp_c,
        }
        for name, values in metrics_lists.items():
            arr = np.array(values, dtype=float)
            summary_entry[f"{name}.mean"] = float(np.mean(arr))
            summary_entry[f"{name}.std"] = float(np.std(arr))
        manifest["summary"].append(summary_entry)

    manifest_path = os.path.join(output_root, "manifest.json")
    save_json(manifest, manifest_path)
    return manifest_path


def _explain_condition(specimen: str, temp_c: int) -> Dict:
    rubber = specimen == "rubber"
    text = []
    text.append(f"At {temp_c}°C, {specimen} concrete shows temperature-driven microstructural and chemical evolution.")

    if temp_c <= 200:
        text.append(
            "Free water removal and early C-S-H dehydroxylation initiate shrinkage; SEM shows incipient microcracks, XRD largely unchanged except slight CH broadening, TGA step near 100°C dominates."
        )
        if rubber:
            text.append(
                "Rubber softening begins (~150–200°C), creating compliant ITZs that concentrate strain but delay continuous cracking; micro-CT shows modest porosity increase localized around rubber."
            )
    elif temp_c <= 400:
        text.append(
            "Pronounced C-S-H dehydration reduces cohesive strength; CH still present but diminishing; SEM reveals wider microcracks; TGA shows bound water loss plateauing; DTA endotherm near 450°C emerges."
        )
        if rubber:
            text.append(
                "Rubber pyrolysis produces voids/char, amplifying porosity and ITZ degradation; macro stiffness and strength drop faster than control due to void coalescence."
            )
    elif temp_c <= 600:
        text.append(
            "CH dehydroxylation largely complete (XRD CH peaks collapse); carbonation phases partially destabilize; micro-CT shows connected crack network."
        )
        if rubber:
            text.append(
                "Rubber-derived voids act as crack nucleation sites; crack connectivity and anisotropy increase, explaining accelerated residual strength loss."
            )
    else:
        text.append(
            "Decarbonation of CaCO₃ and reconstitution of high-temp calcium silicates reduce matrix integrity; pervasive cracking and high porosity dominate transport and strength."
        )
        if rubber:
            text.append(
                "Rubberized mixes exhibit the highest porosity and crack volume; SEM shows collapsed ITZs and large voids where rubber melted away, consistent with severe stiffness/strength degradation."
            )

    return {
        "specimen": specimen,
        "temperature_c": temp_c,
        "macro_behavior_explanation": " ".join(text),
    }
