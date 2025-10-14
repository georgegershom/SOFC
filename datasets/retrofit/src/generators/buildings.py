import random
from typing import Dict, Iterable, List

from ..utils import choose_weighted


FUNCTIONS = [
    ("residential_multi_family", 0.35),
    ("residential_single_family", 0.25),
    ("office", 0.15),
    ("education", 0.08),
    ("healthcare", 0.05),
    ("retail", 0.07),
    ("industrial", 0.05),
]

STYLES = [
    ("modern", 0.45),
    ("postmodern", 0.2),
    ("traditional", 0.25),
    ("brutalist", 0.05),
    ("vernacular", 0.05),
]

MATERIALS = [
    ("concrete", 0.4),
    ("brick", 0.2),
    ("steel", 0.15),
    ("timber", 0.15),
    ("composite", 0.1),
]

THERMAL_MASS = [
    ("low", 0.25),
    ("medium", 0.5),
    ("high", 0.25),
]

COUNTRIES = [
    ("SE", 0.1),
    ("DE", 0.12),
    ("FR", 0.12),
    ("UK", 0.1),
    ("ES", 0.08),
    ("IT", 0.1),
    ("PL", 0.06),
    ("CN", 0.1),
    ("US", 0.12),
    ("CA", 0.04),
    ("AU", 0.04),
    ("IN", 0.02),
]

CITY_COORDS = {
    "Stockholm": (59.334, 18.063),
    "Berlin": (52.520, 13.405),
    "Paris": (48.856, 2.352),
    "London": (51.507, -0.128),
    "Madrid": (40.416, -3.703),
    "Rome": (41.902, 12.496),
    "Warsaw": (52.229, 21.012),
    "Beijing": (39.904, 116.408),
    "Shanghai": (31.230, 121.473),
    "New York": (40.713, -74.006),
    "Chicago": (41.878, -87.629),
    "Toronto": (43.651, -79.383),
    "Melbourne": (-37.813, 144.963),
    "Sydney": (-33.868, 151.209),
    "Bangalore": (12.971, 77.594),
}

CITY_COUNTRY = {
    "Stockholm": "SE",
    "Berlin": "DE",
    "Paris": "FR",
    "London": "UK",
    "Madrid": "ES",
    "Rome": "IT",
    "Warsaw": "PL",
    "Beijing": "CN",
    "Shanghai": "CN",
    "New York": "US",
    "Chicago": "US",
    "Toronto": "CA",
    "Melbourne": "AU",
    "Sydney": "AU",
    "Bangalore": "IN",
}


def generate_buildings(n: int) -> List[Dict]:
    cities = list(CITY_COORDS.keys())
    rows: List[Dict] = []
    for i in range(1, n + 1):
        city = random.choice(cities)
        lat, lon = CITY_COORDS[city]
        country = CITY_COUNTRY[city]
        function = choose_weighted(FUNCTIONS)
        style = choose_weighted(STYLES)
        material = choose_weighted(MATERIALS)
        thermal_mass = choose_weighted(THERMAL_MASS)

        construction_year = random.randint(1950, 2022)
        floors = max(1, int(random.gauss(5 if function.startswith("residential") else 8, 2)))
        height_m = floors * random.uniform(2.8, 3.5)
        footprint_area = abs(random.gauss(800 if function == "office" else 600, 200)) + 150
        rooftop_area = footprint_area * random.uniform(0.9, 1.1)
        volume_m3 = footprint_area * height_m
        u_value = max(0.2, random.gauss(1.8 if construction_year < 1980 else (1.2 if construction_year < 2000 else 0.8), 0.3))
        r_value = 1.0 / u_value
        quality_score = max(1, min(5, int(random.gauss(3, 1.0))))

        rows.append(
            {
                "building_id": f"B{i:04d}",
                "name": f"{city} {function.replace('_', ' ').title()} {i}",
                "country": country,
                "city": city,
                "latitude": round(lat + random.uniform(-0.02, 0.02), 6),
                "longitude": round(lon + random.uniform(-0.02, 0.02), 6),
                "function": function,
                "style": style,
                "quality_score": quality_score,
                "construction_year": construction_year,
                "height_m": round(height_m, 2),
                "floors": floors,
                "footprint_area_m2": round(footprint_area, 1),
                "rooftop_area_m2": round(rooftop_area, 1),
                "volume_m3": round(volume_m3, 0),
                "envelope_u_value_W_m2K": round(u_value, 3),
                "envelope_r_value_m2K_W": round(r_value, 3),
                "dominant_material": material,
                "thermal_mass_category": thermal_mass,
            }
        )
    return rows
