#!/usr/bin/env python3
from __future__ import annotations

import argparse
from src.sofc_dataset.cli import generate, parse_args


if __name__ == "__main__":
    args = parse_args()
    generate(args)
