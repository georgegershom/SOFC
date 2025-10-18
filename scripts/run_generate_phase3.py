#!/usr/bin/env python3
from pathlib import Path
from phase3_dataset.generator import generate_dataset

if __name__ == "__main__":
    info = generate_dataset()
    print(info)
