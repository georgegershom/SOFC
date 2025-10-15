from __future__ import annotations
import os
import json
import csv
import numpy as np
from typing import Dict, Any, List


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_metadata_json(path: str, meta: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def write_inputs_csv(path: str, records: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in records:
            wr.writerow(r)


def save_npz(path: str, arrays: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, **arrays)
