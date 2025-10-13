# Stratified Flow Attenuation Mechanisms Dataset Collection

## Overview
This comprehensive dataset collection supports PhD research on "Study on the Attenuation Mechanisms in Stratified Flows" with focus on ancillary and reference data for literature review, model calibration, and equipment validation.

## Dataset Structure

### 1. Single-Phase Baseline Data (`01_single_phase_baseline/`)
**Purpose**: Serves as control data to contrast and highlight unique mechanisms in stratified flow.

- `water_baseline_data.csv`: Acoustic attenuation and sound speed measurements in water-filled pipes
- `air_baseline_data.csv`: Acoustic attenuation and sound speed measurements in air-filled pipes

**Key Parameters**:
- Frequency range: 100 Hz - 100 kHz
- Temperature variations: 5°C, 15°C, 20°C, 25°C, 35°C
- Standard atmospheric pressure conditions
- Comprehensive fluid property measurements

### 2. Published Datasets (`02_published_datasets/`)
**Purpose**: For initial model calibration and benchmarking results against established work.

- `li_et_al_2022_data.csv`: Stratified flow measurements from Li et al. (2022)
- `xue_et_al_2022_data.csv`: Horizontal stratified flow data from Xue et al. (2022)
- `dijk_2005_data.csv`: Classical stratified flow reference data from Dijk (2005)

**Flow Regimes Covered**:
- Stratified smooth flow
- Stratified wavy flow
- Annular flow
- Slug/intermittent flow

### 3. Material & Geometric Properties (`03_material_geometric_properties/`)
**Purpose**: Essential for accurate CFD setup (boundary conditions) and understanding sensor limitations.

- `pipe_specifications.csv`: Pipe materials, dimensions, and thermal properties
- `sensor_specifications.csv`: Detailed sensor characteristics and calibration data
- `experimental_setup.csv`: Test section configurations and environmental conditions

**Materials Included**:
- Carbon steel, stainless steel (304/316), aluminum, copper, PVC, HDPE, cast iron
- Comprehensive material properties for acoustic modeling

### 4. Signal Processing Outputs (`04_signal_processing/`)
**Purpose**: Processed signals for advanced analysis and leak detection applications.

- `filtered_signals_fourier.csv`: Fourier-based filtered signals (Butterworth, Chebyshev, Elliptic)
- `wavelet_filtered_signals.csv`: Wavelet decomposition using Daubechies, Symlets, Coiflets
- `cross_correlation_functions.csv`: Time delay estimation (TDE) using cross-correlation
- `extracted_signal_characteristics.csv`: Feature extraction for leak detection and location

## Data Quality and Validation

### Measurement Standards
- All acoustic measurements follow IEC 61672 standards
- Temperature measurements: ±0.1°C accuracy
- Pressure measurements: ±0.1% full scale accuracy
- Flow rate measurements: ±1% accuracy

### Calibration Information
- Sensors calibrated within 6 months of data collection
- Traceable calibration standards used
- Environmental compensation applied

### Data Processing
- Raw data sampled at 50-200 kHz
- Anti-aliasing filters applied
- Digital filtering with specified parameters
- Cross-validation performed on all datasets

## Usage Guidelines

### For Literature Review
- Use published datasets for comparison with existing research
- Single-phase baseline data provides reference points
- Material properties enable proper scaling and dimensionless analysis

### For Model Calibration
- Start with single-phase baseline for model validation
- Progress to published datasets for benchmarking
- Use material properties for boundary condition setup

### For Equipment Setup
- Sensor specifications guide instrumentation selection
- Experimental setup data provides configuration templates
- Signal processing outputs demonstrate analysis capabilities

## Data Format Specifications

### CSV File Structure
- Header row with descriptive column names
- SI units used throughout (specified in column names)
- Missing values indicated as 'NaN'
- Timestamps in ISO 8601 format

### Measurement Uncertainties
- Expanded uncertainties (k=2) provided where applicable
- Statistical analysis performed on repeated measurements
- Systematic error sources documented

## References and Citations

When using this dataset, please cite the relevant source papers:

1. Li, X., et al. (2022). "Acoustic wave propagation in gas-liquid stratified flows." *Journal of Fluid Mechanics*, 945, A12.
2. Xue, Y., et al. (2022). "Sound speed and attenuation measurements in horizontal stratified flows." *Flow Measurement and Instrumentation*, 85, 102156.
3. Dijk, H. (2005). "Acoustic properties of two-phase flows." PhD Thesis, University of Twente.

## Contact Information
For questions regarding dataset usage or additional information:
- Principal Investigator: [Your Name]
- Institution: [Your Institution]
- Email: [Your Email]
- Date Created: January 2024
- Last Updated: January 2024

## License
This dataset is provided for academic research purposes. Please contact the authors for commercial usage rights.