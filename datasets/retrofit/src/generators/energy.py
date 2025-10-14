import calendar
from typing import Dict, Iterable, Iterator, List


def _rating_label(energy_intensity: float) -> str:
    # EU A-G style, lower is better
    thresholds = [
        (60, "A"),
        (90, "B"),
        (130, "C"),
        (170, "D"),
        (210, "E"),
        (260, "F"),
    ]
    for t, label in thresholds:
        if energy_intensity <= t:
            return label
    return "G"


def generate_monthly_energy(
    building_rows: List[Dict],
    monthly_factor: List[float],
    period: str,
) -> Iterator[Dict]:
    for b in building_rows:
        area = float(b["footprint_area_m2"]) * float(b["floors"]) * 0.8
        base_intensity_kwh_m2_y = 95.0 if b["construction_year"] >= 2000 else (125.0 if b["construction_year"] >= 1980 else 165.0)
        # Adjust for envelope performance
        base_intensity_kwh_m2_y *= 1.3 if float(b["envelope_u_value_W_m2K"]) > 1.5 else 1.0
        total_kwh_y = area * base_intensity_kwh_m2_y
        for i, mf in enumerate(monthly_factor, start=1):
            electricity_kwh = total_kwh_y * mf * 0.7
            gas_kwh = total_kwh_y * mf * 0.3
            total = electricity_kwh + gas_kwh
            energy_intensity = total / max(1.0, area)
            yield {
                "building_id": b["building_id"],
                "month": f"{i:02d}",
                "electricity_kwh": round(electricity_kwh, 1),
                "gas_kwh": round(gas_kwh, 1),
                "total_kwh": round(total, 1),
                "energy_intensity_kwh_m2": round(energy_intensity, 2),
                "rating_label": _rating_label(energy_intensity),
                "period": period,
            }
