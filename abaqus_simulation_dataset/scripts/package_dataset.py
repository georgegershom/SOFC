#!/usr/bin/env python3
"""
Package all CSV files and Abaqus input files into a ZIP archive
for easy download and distribution.
"""

import zipfile
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(BASE_DIR, "csv")
INPUT_DIR = os.path.join(BASE_DIR, "abaqus_input")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

ZIP_NAME = "abaqus_ysz_gdc_lscf_dataset.zip"
ZIP_PATH = os.path.join(BASE_DIR, ZIP_NAME)


def main():
    print("=" * 60)
    print("  PACKAGING DATASET INTO ZIP ARCHIVE")
    print("=" * 60)

    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        total_files = 0
        total_size = 0

        # Add CSV files
        csv_files = sorted([f for f in os.listdir(CSV_DIR) if f.endswith('.csv')])
        for f in csv_files:
            filepath = os.path.join(CSV_DIR, f)
            arcname = f"csv/{f}"
            zf.write(filepath, arcname)
            size = os.path.getsize(filepath)
            total_files += 1
            total_size += size
            print(f"  Added: {arcname} ({size:,} bytes)")

        # Add Abaqus input files
        if os.path.exists(INPUT_DIR):
            for f in sorted(os.listdir(INPUT_DIR)):
                filepath = os.path.join(INPUT_DIR, f)
                arcname = f"abaqus_input/{f}"
                zf.write(filepath, arcname)
                size = os.path.getsize(filepath)
                total_files += 1
                total_size += size
                print(f"  Added: {arcname} ({size:,} bytes)")

        # Add scripts
        script_files = sorted([f for f in os.listdir(SCRIPTS_DIR) if f.endswith('.py')])
        for f in script_files:
            filepath = os.path.join(SCRIPTS_DIR, f)
            arcname = f"scripts/{f}"
            zf.write(filepath, arcname)
            size = os.path.getsize(filepath)
            total_files += 1
            total_size += size
            print(f"  Added: {arcname} ({size:,} bytes)")

    zip_size = os.path.getsize(ZIP_PATH)
    compression = (1 - zip_size / total_size) * 100 if total_size > 0 else 0

    print(f"\n  Total files: {total_files}")
    print(f"  Uncompressed: {total_size:,} bytes")
    print(f"  ZIP archive: {zip_size:,} bytes ({compression:.1f}% compression)")
    print(f"  Output: {ZIP_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
