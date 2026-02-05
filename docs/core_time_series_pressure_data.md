## Fabricated Core Time-Series Pressure Data (HF Acoustic) — CSV + Figures

This repository includes a **reproducible generator** that fabricates a dataset matching the requested structure for:
**“Core Time-Series Pressure Data (Primary Evidence)”** in the context of **stratified-flow attenuation mechanisms** (beyond single-phase leakage acoustics).

### What you get

- **HF acoustic pressure time-series** for **14 sensors** (`PG1`–`PG14`)
- **Four test groups** (`01`–`04`) with:
  - **Group 01** at **17,060 Hz**
  - Groups 02–04 at **10,000 Hz**
  - Different **leak locations** (`A`, `D`, `E`)
  - Different **sensor configurations** recorded as metadata (`A,B` vs `A,B,C,D`)
  - Different **stratification_index** values (controls stronger HF attenuation)
- **Two runs per group**
  - `run_baseline_only_30s`: **30 seconds** of “normal operation” baseline
  - `run_leak_event_30s`: 30 seconds with **leak open at 5s**, **close at 20s**, record until 30s
- **Format**: **separate CSV files per second**
- **Figures** (`.png`) summarizing waveforms, PSD, and attenuation vs distance
- A single **ZIP** containing all CSVs (and small metadata JSON)

### Output layout

Running the generator produces:

- `generated/core_time_series_pressure_data/`
  - `manifest_runs.csv`
  - `sensors_layout.csv`
  - `valves_layout.csv`
  - `group_01/`
    - `group_metadata.json`
    - `run_baseline_only_30s/csv_per_second/sec_000.csv` … `sec_029.csv`
    - `run_leak_event_30s/csv_per_second/sec_000.csv` … `sec_029.csv`
  - (same for `group_02`, `group_03`, `group_04`)
- `generated/figures/*.png`
- `generated/core_time_series_pressure_data_csv.zip`

### CSV format

Each per-second file contains integer **pressure counts**:

- `sample_idx`: 0 … `fs_hz-1` (time within the 1-second file)
- `PG1` … `PG14`: pressure samples as **int counts**

Convert to Pascals (Pa) using:

\[
P_{Pa} = \text{counts} \times \text{pressure\_pa\_per\_count}
\]

The default is `pressure_pa_per_count = 0.05` (see the script flags).

### How to generate (CSV + figures + zip)

```bash
python3 -m pip install -r requirements.txt
python3 scripts/generate_core_time_series_pressure_data.py
```

If you want CSV+ZIP only (no figures):

```bash
python3 scripts/generate_core_time_series_pressure_data.py --no-figures
```

### Fabrication notice

All generated signals are **synthetic** and intended for:
- algorithm development,
- pipeline testing,
- visualization and attenuation-method prototyping.

They are **not** real experimental measurements.

