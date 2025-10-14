import argparse
import os
from datetime import datetime
from typing import List

from .schemas import SCHEMAS
from .utils import seed_everything, write_csv
from .generators.buildings import generate_buildings
from .generators.weather import generate_weather
from .generators.iot import generate_iot
from .generators.energy import generate_monthly_energy
from .generators.lca import EPD_CATALOG, generate_lca_summary
from .generators.retrofit import generate_retrofit_measures


MONTHLY_FACTORS = [
    0.085, 0.075, 0.08, 0.082, 0.085, 0.088, 0.091, 0.095, 0.092, 0.086, 0.086, 0.085
]


def main():
    parser = argparse.ArgumentParser(description="Generate Integrated Building Retrofit Dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-buildings", type=int, default=50)
    parser.add_argument("--start", type=str, default="2022-01-01")
    parser.add_argument("--end", type=str, default="2022-12-31")
    parser.add_argument("--outdir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data"))
    args = parser.parse_args()

    seed_everything(args.seed)

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # Buildings
    buildings = generate_buildings(args.num_buildings)
    write_csv(os.path.join(outdir, "buildings.csv"), SCHEMAS["buildings"], buildings)

    # Weather hourly
    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)
    weather_rows = generate_weather(buildings, start_dt, end_dt, step_minutes=60)
    write_csv(os.path.join(outdir, "outdoor_weather.csv"), SCHEMAS["outdoor_weather"], weather_rows)

    # IoT timeseries 15-min
    iot_rows = generate_iot(buildings, start_dt, end_dt, step_minutes=15)
    write_csv(os.path.join(outdir, "iot_timeseries.csv"), SCHEMAS["iot_timeseries"], iot_rows)

    # Energy performance monthly pre-retrofit
    energy_pre = generate_monthly_energy(buildings, MONTHLY_FACTORS, period="pre")
    write_csv(
        os.path.join(outdir, "energy_performance_monthly_pre.csv"),
        SCHEMAS["energy_performance_monthly"],
        energy_pre,
    )

    # Retrofit measures and post-retrofit
    retrofits = list(generate_retrofit_measures(buildings))
    write_csv(os.path.join(outdir, "retrofit_measures.csv"), SCHEMAS["retrofit_measures"], retrofits)

    # Post-retrofit adjust: simulate savings by scaling down monthly factors
    post_monthly = [max(0.04, f * 0.8) for f in MONTHLY_FACTORS]
    energy_post = generate_monthly_energy(buildings, post_monthly, period="post")
    write_csv(
        os.path.join(outdir, "energy_performance_monthly_post.csv"),
        SCHEMAS["energy_performance_monthly"],
        energy_post,
    )

    # LCA catalogs and summaries
    write_csv(os.path.join(outdir, "lca_epd_catalog.csv"), SCHEMAS["lca_epd_catalog"], EPD_CATALOG)
    lca_summary_rows = generate_lca_summary(buildings)
    write_csv(os.path.join(outdir, "lca_summary.csv"), SCHEMAS["lca_summary"], lca_summary_rows)

    print(f"Dataset generated in: {outdir}")


if __name__ == "__main__":
    main()
