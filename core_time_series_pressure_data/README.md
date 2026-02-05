Core Time-Series Pressure Data (Synthetic)
==========================================

This dataset was fabricated to support analysis for:
"Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase
Leakage Acoustics." The data is **synthetic** and intended for method
development, visualization, and pipeline testing.

Contents
--------

- High-frequency (HF) acoustic pressure data at 10 kHz and 17,060 Hz.
- Four test groups (01-04) with varying leak locations (A, D, E) and
  sensor configurations (A,B or A,B,C,D).
- Baseline (30 s), transient event (30 s, leak open 5-20 s), and post-event
  recovery (10 s).
- Separate CSV file per second.
- Metadata describing sensors, leak points, valves, groups, and events.
- Figures illustrating time-series and attenuation trends.
- A ZIP file containing all CSVs for easy download.

Directory Structure
-------------------

```
core_time_series_pressure_data/
  README.md
  csv_archives/
    hf_acoustic_group01.zip
    hf_acoustic_group02.zip
    hf_acoustic_group03.zip
    hf_acoustic_group04.zip
    metadata_csv.zip
  metadata/
    sensors.csv
    leak_locations.csv
    valves.csv
    groups.csv
    events.csv
  figures/
    group01_timeseries_PG6.png
    group02_timeseries_PG6.png
    group03_timeseries_PG6.png
    group04_timeseries_PG6.png
    attenuation_summary.png
```

Data Format
-----------

Each CSV contains samples for one second at the group-specific sample rate.
Columns:

```
sample_index, PG1, PG2, ..., PG14
```

- `sample_index` is the integer sample index for that second.
- Time in seconds for a row is:

```
time_sec = sample_index / sample_rate_hz
```

Units
-----

- Pressure values are synthetic and expressed in **Pascals (Pa)**.

Event Timing (Transient Window)
-------------------------------

- Leak opens at 5 s
- Leak closes at 20 s
- Transient window is 0-30 s

Metadata
--------

See `metadata/` for sensor positions, leak locations, valve positions, and
group configurations.

ZIP Archive
-----------

The `csv_archives/` directory contains the per-group ZIP archives with the
per-second CSV files (each ZIP extracts to `hf_acoustic/groupXX/...`), plus
`metadata_csv.zip` with the metadata tables.

Reproducibility
---------------

The dataset was generated with a deterministic synthesis model combining:

- Low-frequency pump noise
- Stratification-induced modulation
- High-frequency leak signals with distance-based attenuation
- Post-closure ring-down decay

All data is fabricated and does not represent real experimental measurements.

To regenerate the dataset, install the Python requirements and run:

```
python3 -m pip install -r scripts/requirements.txt
python3 scripts/generate_core_time_series_pressure_data.py
```

The script creates raw per-second CSVs under `hf_acoustic/` and then writes
ZIP archives to `csv_archives/`. You can delete `hf_acoustic/` after zipping
to save disk space.
