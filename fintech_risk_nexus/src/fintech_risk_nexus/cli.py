from __future__ import annotations

import argparse
from pathlib import Path

from .generate import GenerationConfig, generate_datasets
from .config import DEFAULT_COUNTRIES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate FinTech Nexus dataset")
    p.add_argument("--countries", nargs="*", default=DEFAULT_COUNTRIES, help="List of ISO2 country codes")
    p.add_argument("--start-year", type=int, default=2016)
    p.add_argument("--end-year", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    end_year = args.end_year
    if end_year is None:
        from .utils import get_last_complete_month

        end_year = get_last_complete_month().year

    cfg = GenerationConfig(
        countries=args.countries,
        start_year=args.start_year,
        end_year=end_year,
        seed=args.seed,
        output_dir=args.out,
    )
    generate_datasets(cfg)


if __name__ == "__main__":
    main()
