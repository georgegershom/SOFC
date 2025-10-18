from pathlib import Path
from typing import Dict, List
import json

import numpy as np

from .config import OUTPUT_ROOT, MIXES, TEMPERATURE_STEPS_C, REPLICATES_PER_CONDITION, SEM_FOVS_PER_REPLICATE, VOXEL_SIZE_UM
from .utils import ensure_dir, stable_rng, write_csv, write_json, sample_id, fov_id
from .correlation import compute_latent_state
from .microstructure import generate_microstructure
from .micro_ct import compute_micro_ct_metrics
from .sem import generate_sem_stats
from .xrd import generate_xrd_phase_table
from .tga_dta import generate_tga_dta


def generate_dataset(output_root: Path = OUTPUT_ROOT) -> Dict[str, object]:
    ensure_dir(output_root)
    master_rows: List[Dict[str, object]] = []
    index_entries: List[Dict[str, object]] = []

    for mix in MIXES:
        mix_id = mix["Mix_ID"]  # type: ignore
        for temp_c in TEMPERATURE_STEPS_C:
            for rep in range(1, REPLICATES_PER_CONDITION + 1):
                sid = sample_id(str(mix_id), temp_c, rep)
                rng = stable_rng(sid)
                # Latent state (correlated across techniques for this sample)
                latent = compute_latent_state(mix, temp_c, rng)

                # Microstructure (3D volume)
                ms = generate_microstructure(latent, rng)
                volume = ms["volume"]  # np.ndarray
                ct_metrics = compute_micro_ct_metrics(volume)

                # Save volume
                vol_dir = output_root / "micro_ct" / "volumes"
                ensure_dir(vol_dir)
                vol_path = vol_dir / f"{sid}.npz"
                np.savez_compressed(vol_path, volume=volume)

                # Micro-CT metrics row(s)
                metrics_row = {
                    "Sample_ID": sid,
                    "Mix_ID": mix_id,
                    "Temperature_C": temp_c,
                    "Analysis_Type": "MicroCT",
                    "Measurement_Scale": "Micro/3D",
                    **{k: float(v) for k, v in ct_metrics.items()},
                    "crack_fraction_est": float(ms["crack_fraction_est"]),
                }
                master_rows.append(metrics_row)

                # SEM stats per FOV
                sem_dir = output_root / "sem"
                ensure_dir(sem_dir)
                sem_rows = generate_sem_stats(volume, float(latent["itz_damage_index"]), float(latent["crack_factor"]), rng, num_fovs=SEM_FOVS_PER_REPLICATE)
                per_fov = []
                for i, srow in enumerate(sem_rows):
                    row = {
                        "Sample_ID": sid,
                        "Mix_ID": mix_id,
                        "Temperature_C": temp_c,
                        "Analysis_Type": "SEM",
                        "Measurement_Scale": "Micro/2D",
                        "FOV_ID": fov_id(i + 1),
                        **{k: float(v) for k, v in srow.items()},
                    }
                    per_fov.append(row)
                    master_rows.append(row)
                write_csv(sem_dir / f"{sid}_fov_metrics.csv", per_fov, fieldnames=list(per_fov[0].keys()))

                # XRD phase table
                xrd_dir = output_root / "xrd"
                ensure_dir(xrd_dir)
                xrd_rows = generate_xrd_phase_table(latent["phases_wt_pct"], rng)  # type: ignore
                for xr in xrd_rows:
                    master_rows.append({
                        "Sample_ID": sid,
                        "Mix_ID": mix_id,
                        "Temperature_C": temp_c,
                        "Analysis_Type": "XRD",
                        "Measurement_Scale": "Bulk",
                        "phase": xr["phase"],
                        "wt_percent": xr["wt_percent"],
                    })
                write_csv(xrd_dir / f"{sid}_phases.csv", xrd_rows, fieldnames=["phase", "wt_percent"])

                # TGA/DTA curves
                tga_dir = output_root / "tga_dta"
                ensure_dir(tga_dir)
                tga = generate_tga_dta(latent, rng)
                # Save curve CSV
                curve_rows = [{"T_C": T, "mass_fraction": mf} for T, mf in zip(tga["T_C"], tga["mass_fraction"])]  # type: ignore
                write_csv(tga_dir / f"{sid}_tg_curve.csv", curve_rows, fieldnames=["T_C", "mass_fraction"])
                # Save DTA peaks JSON
                write_json(tga_dir / f"{sid}_dta_peaks.json", {"peaks": tga["dta_peaks"]})

                # Master rows: TGA endpoint and latent indicators
                master_rows.append({
                    "Sample_ID": sid,
                    "Mix_ID": mix_id,
                    "Temperature_C": temp_c,
                    "Analysis_Type": "TGA",
                    "Measurement_Scale": "Bulk",
                    "mass_fraction_800C": float(tga["mass_fraction"][-1]),  # type: ignore
                    "tga_total_loss_800": float(latent["tga_total_loss_800"]),
                })

                # Rubber-specific indicators
                master_rows.append({
                    "Sample_ID": sid,
                    "Mix_ID": mix_id,
                    "Temperature_C": temp_c,
                    "Analysis_Type": "Rubber-Signature",
                    "Measurement_Scale": "Micro",
                    "melt_phase_fraction": float(latent["melt_phase_fraction"]),
                    "rubber_vol_frac": float(latent["rubber_vol_frac"]),
                })

                # Index entry for this sample
                index_entries.append({
                    "Sample_ID": sid,
                    "Mix_ID": mix_id,
                    "Temperature_C": temp_c,
                    "paths": {
                        "micro_ct_volume_npz": str(vol_path.relative_to(output_root)),
                        "sem_fov_metrics_csv": f"sem/{sid}_fov_metrics.csv",
                        "xrd_phases_csv": f"xrd/{sid}_phases.csv",
                        "tga_curve_csv": f"tga_dta/{sid}_tg_curve.csv",
                        "dta_peaks_json": f"tga_dta/{sid}_dta_peaks.json",
                    }
                })

    # Write master table
    master_path = output_root / "master_metrics.csv"
    # Collect all fieldnames
    all_fields = set()
    for r in master_rows:
        all_fields.update(r.keys())
    all_fields = list(sorted(all_fields))
    write_csv(master_path, master_rows, fieldnames=all_fields)

    # Index.json for quick browsing
    write_json(output_root / "index.json", {"entries": index_entries, "voxel_size_um": VOXEL_SIZE_UM})

    return {
        "master_metrics_csv": str(master_path),
        "num_entries": len(master_rows),
        "index_json": str((output_root / "index.json")),
    }


if __name__ == "__main__":
    info = generate_dataset()
    print(json.dumps(info, indent=2))
