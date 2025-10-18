import json
import os
from dataclasses import dataclass
from typing import Dict, Any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


@dataclass
class DatasetMeta:
    mix_id: str
    data_type: str  # "Calibration" or "Validation"
    temperature_min_c: float
    temperature_max_c: float
    temperature_step_c: float
    units: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Mix_ID": self.mix_id,
            "Data_Type": self.data_type,
            "Temperature_Min_C": self.temperature_min_c,
            "Temperature_Max_C": self.temperature_max_c,
            "Temperature_Step_C": self.temperature_step_c,
            "Units": self.units,
        }
