import os
import json
import random
import csv
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

from PIL import Image, ImageDraw, ImageFilter

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "phase1"))
MATERIALS_DIR = os.path.join(DATASET_ROOT, "materials")
MIXES_DIR = os.path.join(DATASET_ROOT, "mixes")
IMAGES_DIR = os.path.join(DATASET_ROOT, "images", "sem")
META_DIR = os.path.join(DATASET_ROOT, "meta")

os.makedirs(MATERIALS_DIR, exist_ok=True)
os.makedirs(MIXES_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

# -----------------------------
# Data classes
# -----------------------------
@dataclass
class Cement:
    cement_type: str  # e.g., OPC
    specific_gravity: float
    xrf_oxides_pct: Dict[str, float]  # e.g., {"SiO2": 20.5, ...}
    bogue_composition_pct: Dict[str, float]  # {"C3S": ..., "C2S": ..., "C3A": ..., "C4AF": ...}

@dataclass
class Aggregate:
    category: str  # Coarse or Fine
    specific_gravity: float
    water_absorption_pct: float
    bulk_density_kg_m3: int
    sieve_mm: List[float]
    retained_pct: List[float]

@dataclass
class Rubber:
    source: str  # e.g., Passenger tire tread
    size_range_mm: Tuple[float, float]
    particle_size_distribution_pct: List[float]
    specific_gravity: float
    mohs_hardness: float
    tga_mass_loss_profile_pct: List[float]  # synthetic mass loss at temp steps
    ftir_peaks_cm1: Dict[str, float]  # key peaks and relative intensities
    additives_pct: Dict[str, float]  # sulfur, carbon black, oils, etc.
    morphology_image_paths: List[str]  # SEM image paths

@dataclass
class Water:
    ph: float
    impurities_mg_L: Dict[str, float]

@dataclass
class Admixture:
    admixture_type: str  # superplasticizer
    dosage_pct_bwc: float  # percent by weight of cement
    solid_content_pct: float

@dataclass
class MixDesign:
    mix_id: str
    description: str
    cement_kg_m3: float
    water_kg_m3: float
    fine_agg_kg_m3: float
    coarse_agg_kg_m3: float
    rubber_vol_repl_pct: float
    rubber_size_range_mm: str
    admixture_dosage_pct_bwc: float
    w_c_ratio: float

@dataclass
class FreshProperties:
    mix_id: str
    slump_mm: float
    air_content_pct: float
    unit_weight_kg_m3: float
    temperature_C: float

# -----------------------------
# Helper functions
# -----------------------------

def normalize_percentages(values: List[float]) -> List[float]:
    total = sum(values)
    if total == 0:
        return [0 for _ in values]
    return [round(v * 100.0 / total, 2) for v in values]


def generate_sem_image(
    seed: int,
    size: Tuple[int, int] = (512, 512),
    contrast: float = 1.5,
    blur: float = 0.8,
) -> Image.Image:
    rng = random.Random(seed)
    # Base Gaussian-like noise using Pillow effect to avoid numpy dependency
    # sigma ~ 40 approximates moderate SEM background noise
    img = Image.effect_noise(size, 40.0)

    # Add synthetic rubber particle-like shapes
    draw = ImageDraw.Draw(img)
    num_particles = rng.randint(80, 160)
    width, height = size
    for _ in range(num_particles):
        x = rng.randint(0, width - 1)
        y = rng.randint(0, height - 1)
        r = rng.randint(3, 18)
        bbox = [x - r, y - r, x + r, y + r]
        gray = rng.randint(60, 220)
        draw.ellipse(bbox, fill=gray, outline=gray)

    img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    # Enhance contrast via a simple point transform
    def _adj(v: int) -> int:
        val = (v - 128.0) * contrast + 128.0
        return int(0 if val < 0 else 255 if val > 255 else val)

    img = img.point(_adj)
    return img.convert("L")


def save_image(img: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


# -----------------------------
# Fabrication of input data
# -----------------------------

def fabricate_cement() -> Cement:
    # Typical OPC ranges
    xrf = {
        "SiO2": 20.8,
        "Al2O3": 5.3,
        "Fe2O3": 3.1,
        "CaO": 63.5,
        "MgO": 2.1,
        "SO3": 2.6,
        "Na2O": 0.2,
        "K2O": 0.5,
        "TiO2": 0.25,
        "P2O5": 0.06,
    }
    # Ensure 100% with "Loss on Ignition" approximate remainder
    remainder = max(0.0, 100.0 - sum(xrf.values()))
    xrf["LOI"] = round(remainder, 2)

    bogue = {
        "C3S": 58.0,
        "C2S": 16.0,
        "C3A": 7.0,
        "C4AF": 10.0,
    }
    return Cement(
        cement_type="OPC",
        specific_gravity=3.15,
        xrf_oxides_pct=xrf,
        bogue_composition_pct=bogue,
    )


def fabricate_aggregate(category: str) -> Aggregate:
    # Standard sieve series for fine and coarse
    if category == "Fine":
        sieves = [4.75, 2.36, 1.18, 0.6, 0.3, 0.15]
        base_retained = [5, 12, 18, 25, 22, 10]
        sg = 2.64
        wa = 1.2
        bulk = 1650
    else:
        sieves = [37.5, 19.0, 9.5, 4.75]
        base_retained = [10, 45, 35, 5]
        sg = 2.70
        wa = 0.8
        bulk = 1550
    retained = normalize_percentages(base_retained)
    return Aggregate(
        category=category,
        specific_gravity=sg,
        water_absorption_pct=wa,
        bulk_density_kg_m3=bulk,
        sieve_mm=sieves,
        retained_pct=retained,
    )


def fabricate_rubber(size_range: Tuple[float, float], image_count: int, base_seed: int, label: str) -> Rubber:
    low, high = size_range
    # Create 6 bins across the size range
    bins = [low + (high - low) * i / 5.0 for i in range(6)]
    # Generate positive weights and normalize to percentages
    rng = random.Random(int((low + high) * 1000))
    psd_raw = [abs(rng.gauss(1.0, 0.3)) for _ in bins]
    psd = normalize_percentages(psd_raw)

    tga_profile = [
        0.0,   # 25 C baseline
        1.5,   # 100 C moisture + light oils
        5.0,   # 200 C
        12.0,  # 300 C
        28.0,  # 400 C major pyrolysis onset
        45.0,  # 500 C
        58.0,  # 600 C
        64.0,  # 700 C
        68.0,  # 800 C
    ]
    ftir = {
        "C-H stretch": 2920.0,
        "C=C stretch": 1660.0,
        "S-S stretch": 500.0,
        "Aromatic ring": 1600.0,
    }
    additives = {
        "sulfur": 1.8,
        "carbon_black": 30.0,
        "processing_oils": 6.5,
        "antioxidants": 0.8,
    }

    image_paths: List[str] = []
    for i in range(image_count):
        seed = base_seed + i
        img = generate_sem_image(seed=seed, size=(768, 768), contrast=1.8, blur=0.9)
        filename = f"sem_{label}_{int(low)}-{int(high)}mm_{i+1:02d}.png"
        path = os.path.join(IMAGES_DIR, filename)
        save_image(img, path)
        image_paths.append(os.path.relpath(path, DATASET_ROOT))

    return Rubber(
        source="Passenger tire tread",
        size_range_mm=size_range,
        particle_size_distribution_pct=psd,
        specific_gravity=1.12,
        mohs_hardness=0.5,
        tga_mass_loss_profile_pct=tga_profile,
        ftir_peaks_cm1=ftir,
        additives_pct=additives,
        morphology_image_paths=image_paths,
    )


def fabricate_water() -> Water:
    return Water(
        ph=7.3,
        impurities_mg_L={
            "chlorides": 120.0,
            "sulfates": 180.0,
            "alkalinity_as_CaCO3": 90.0,
            "total_dissolved_solids": 500.0,
        },
    )


def fabricate_admixture() -> Admixture:
    return Admixture(
        admixture_type="Polycarboxylate superplasticizer",
        dosage_pct_bwc=0.8,
        solid_content_pct=35.0,
    )


def fabricate_mix_matrix() -> Tuple[List[MixDesign], List[FreshProperties]]:
    # Baseline composition (kg/m3) typical for HPC
    base_cement = 450.0
    base_water = 162.0  # w/c 0.36
    base_fine = 700.0
    base_coarse = 1000.0

    rubber_replacements = [0, 5, 10, 15, 20]  # % by volume of fine aggregate
    rubber_sizes = [(1.0, 4.0), (4.0, 8.0)]  # two ranges

    mix_designs: List[MixDesign] = []
    fresh_props: List[FreshProperties] = []

    for repl in rubber_replacements:
        # Control and series at default size range (1-4 mm)
        for size in (rubber_sizes if repl != 0 else [(1.0, 4.0)]):
            mix_id = f"HPRC_R{repl:02d}_{int(size[0])}-{int(size[1])}mm"
            description = (
                "Control mix (no rubber)" if repl == 0 else f"{repl}% vol. replacement of fine agg by crumb rubber"
            )
            # Convert volume replacement of fine aggregate to mass reduction using SGs
            fine_sg = 2.64
            rubber_sg = 1.12
            fine_vol = base_fine / (fine_sg * 1000.0)  # m3 per m3 concrete
            rubber_vol = fine_vol * (repl / 100.0)
            remaining_fine_vol = fine_vol - rubber_vol
            fine_mass = remaining_fine_vol * fine_sg * 1000.0
            rubber_mass = rubber_vol * rubber_sg * 1000.0

            admixture_dosage = 0.8 + 0.02 * repl  # increase dosage slightly with rubber
            w_c = base_water / base_cement

            mix = MixDesign(
                mix_id=mix_id,
                description=description,
                cement_kg_m3=base_cement,
                water_kg_m3=base_water,
                fine_agg_kg_m3=round(fine_mass, 1),
                coarse_agg_kg_m3=base_coarse,
                rubber_vol_repl_pct=float(repl),
                rubber_size_range_mm=f"{size[0]}-{size[1]}",
                admixture_dosage_pct_bwc=round(admixture_dosage, 2),
                w_c_ratio=round(w_c, 3),
            )
            mix_designs.append(mix)

            # Fresh properties: rubber increases air, reduces slump & unit weight
            base_slump = 200.0  # mm, flowable HPC
            slump = base_slump - repl * 3.5 + random.gauss(0, 5)
            air = 1.8 + repl * 0.25 + random.gauss(0, 0.15)
            unit_wt = 2400.0 - repl * 7.0 + random.gauss(0, 5)
            temp = 22.0 + random.gauss(0, 0.5)

            fresh = FreshProperties(
                mix_id=mix_id,
                slump_mm=round(max(0.0, slump), 1),
                air_content_pct=round(max(0.1, air), 2),
                unit_weight_kg_m3=round(unit_wt, 1),
                temperature_C=round(temp, 1),
            )
            fresh_props.append(fresh)

    return mix_designs, fresh_props


# -----------------------------
# Serialization
# -----------------------------

def write_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_csv_dicts(path: str, records: List[Dict]) -> None:
    if not records:
        # Still create an empty file with no headers
        open(path, "w", encoding="utf-8").close()
        return
    fieldnames = list(records[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


# -----------------------------
# Main generation
# -----------------------------

def main() -> None:
    cement = fabricate_cement()
    fine = fabricate_aggregate("Fine")
    coarse = fabricate_aggregate("Coarse")

    rubber_small = fabricate_rubber((1.0, 4.0), image_count=6, base_seed=100, label="small")
    rubber_large = fabricate_rubber((4.0, 8.0), image_count=6, base_seed=200, label="large")

    water = fabricate_water()
    admixture = fabricate_admixture()

    mixes, fresh = fabricate_mix_matrix()

    # Materials JSONs
    write_json(os.path.join(MATERIALS_DIR, "cement.json"), asdict(cement))
    write_json(os.path.join(MATERIALS_DIR, "aggregate_fine.json"), asdict(fine))
    write_json(os.path.join(MATERIALS_DIR, "aggregate_coarse.json"), asdict(coarse))
    write_json(os.path.join(MATERIALS_DIR, "rubber_1_4mm.json"), asdict(rubber_small))
    write_json(os.path.join(MATERIALS_DIR, "rubber_4_8mm.json"), asdict(rubber_large))
    write_json(os.path.join(MATERIALS_DIR, "water.json"), asdict(water))
    write_json(os.path.join(MATERIALS_DIR, "admixture.json"), asdict(admixture))

    # Mixes CSV
    write_csv_dicts(os.path.join(MIXES_DIR, "mix_designs.csv"), [asdict(m) for m in mixes])

    # Fresh properties CSV
    write_csv_dicts(os.path.join(MIXES_DIR, "fresh_properties.csv"), [asdict(fp) for fp in fresh])

    # Meta readme
    meta = {
        "title": "Phase 1: Material Characterization & Specimen Preparation",
        "topic": "Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete",
        "description": "Synthetic baseline dataset with deterministic values, including constituent materials, mix designs, and fresh concrete properties. Includes synthetic SEM images representative of crumb rubber morphology (pre-heating).",
        "random_seed": RANDOM_SEED,
        "directories": {
            "materials": os.path.relpath(MATERIALS_DIR, DATASET_ROOT),
            "mixes": os.path.relpath(MIXES_DIR, DATASET_ROOT),
            "images_sem": os.path.relpath(IMAGES_DIR, DATASET_ROOT),
        },
    }
    write_json(os.path.join(META_DIR, "dataset_meta.json"), meta)

    # Root README-like text
    readme_path = os.path.join(DATASET_ROOT, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "Phase 1 Dataset - High-Performance Rubberized Concrete (HPRC)\n" \
            "Includes materials characterization, mix designs, fresh properties, and synthetic SEM images.\n\n" \
            "Structure:\n" \
            "- materials/*.json\n" \
            "- mixes/mix_designs.csv\n" \
            "- mixes/fresh_properties.csv\n" \
            "- images/sem/*.png\n" \
            "- meta/dataset_meta.json\n"
        )

    print(f"Dataset generated at: {DATASET_ROOT}")


if __name__ == "__main__":
    main()
