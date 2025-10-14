from typing import Dict, Iterable, Iterator, List


EPD_CATALOG: List[Dict] = [
    {"material_category": "insulation_min_wool", "unit": "kg", "gwp_kgco2e_per_unit": 1.5, "assumed_source": "synthetic"},
    {"material_category": "insulation_pir", "unit": "kg", "gwp_kgco2e_per_unit": 3.8, "assumed_source": "synthetic"},
    {"material_category": "concrete_c30", "unit": "m3", "gwp_kgco2e_per_unit": 330.0, "assumed_source": "synthetic"},
    {"material_category": "brick", "unit": "kg", "gwp_kgco2e_per_unit": 0.22, "assumed_source": "synthetic"},
    {"material_category": "steel_rebar", "unit": "kg", "gwp_kgco2e_per_unit": 1.9, "assumed_source": "synthetic"},
    {"material_category": "aluminium", "unit": "kg", "gwp_kgco2e_per_unit": 8.0, "assumed_source": "synthetic"},
    {"material_category": "timber_glulam", "unit": "kg", "gwp_kgco2e_per_unit": 0.12, "assumed_source": "synthetic"},
    {"material_category": "glass_double", "unit": "m2", "gwp_kgco2e_per_unit": 30.0, "assumed_source": "synthetic"},
]


def generate_lca_summary(building_rows: List[Dict]) -> Iterator[Dict]:
    # Simple quantities derived from areas
    for b in building_rows:
        area = float(b["footprint_area_m2"]) * float(b["floors"]) * 0.8
        envelope_area = (float(b["rooftop_area_m2"]) + area) * 0.5
        window_area = float(b["footprint_area_m2"]) * 0.2
        insulation_kg = envelope_area * 12.0
        brick_kg = area * 100.0
        rebar_kg = area * 12.0
        glass_m2 = window_area

        for cat in EPD_CATALOG:
            if cat["material_category"].startswith("insulation"):
                qty = insulation_kg
            elif cat["material_category"] == "brick":
                qty = brick_kg
            elif cat["material_category"] == "steel_rebar":
                qty = rebar_kg
            elif cat["material_category"] == "glass_double":
                qty = glass_m2
            else:
                qty = area * 2.0
            gwp = qty * float(cat["gwp_kgco2e_per_unit"])
            yield {
                "building_id": b["building_id"],
                "material_category": cat["material_category"],
                "quantity_unit": cat["unit"],
                "quantity": round(qty, 2),
                "gwp_kgco2e": round(gwp, 2),
            }
