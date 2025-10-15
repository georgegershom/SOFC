from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable
import pandas as pd


def append_index_csv(index_path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(list(rows))
    if index_path.exists():
        old_df = pd.read_csv(index_path)
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(index_path, index=False)
