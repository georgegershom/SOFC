# Fabricated HF Acoustic Pressure Dataset

This dataset is a fabricated (synthetic) high-frequency acoustic pressure
time series for leak detection studies in stratified flows. It follows the
requested experiment timing and file layout:

* Sampling rates:
  * Group 01: 17,060 Hz
  * Groups 02-04: 10,000 Hz
* Files are split into one CSV per second.
* Baseline: 30 seconds prior to leak event (t = -30 to -0).
* Event recording: 30 seconds (t = 0 to 30) with the valve opening at
  t = 5 s and closing at t = 20 s.
* Sensors: 14 high-frequency sensors (PG1-PG14).

## Directory layout

```
data/
  csv/
    group01/
      baseline/sec_000.csv ... sec_029.csv
      event/sec_000.csv ... sec_029.csv
    group02/
      baseline/...
      event/...
    group03/
      baseline/...
      event/...
    group04/
      baseline/...
      event/...
  metadata/
    sensors.csv
    groups.csv
    leak_positions.csv
    leak_rms_summary.csv
  dataset_manifest.csv
  hf_pressure_csv_group01.zip
  hf_pressure_csv_group02.zip
  hf_pressure_csv_group03.zip
  hf_pressure_csv_group04.zip
  hf_pressure_csv_metadata.zip
figures/
  group01_pg1_timeseries.png
  group01_attenuation.png
  group02_pg1_timeseries.png
  group02_attenuation.png
  group03_pg1_timeseries.png
  group03_attenuation.png
  group04_pg1_timeseries.png
  group04_attenuation.png
```

## CSV format (per-second files)

Each per-second CSV contains:

* `time_s`: global time in seconds (baseline values are negative).
* `PG1` ... `PG14`: pressure in Pascals.

## Group configurations

* Group 01: leak at A, sensor config A,B (higher sampling rate 17,060 Hz)
* Group 02: leak at D, sensor config A,B,C,D
* Group 03: leak at E, sensor config A,B
* Group 04: leak at A, sensor config A,B,C,D

## Figures

For each group:

* `*_pg1_timeseries.png` shows the 0-30 s event window for PG1.
* `*_attenuation.png` shows RMS pressure vs distance from leak.

## Reproducibility

Use `scripts/generate_dataset.py` to regenerate the dataset and figures.
This script uses a fixed RNG seed to produce deterministic outputs.
