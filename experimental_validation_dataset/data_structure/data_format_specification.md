# SOFC Experimental Validation Dataset - Data Format Specification

## Overview
This document specifies the data formats, file structures, and naming conventions for the SOFC experimental validation dataset. The specification ensures data interoperability, traceability, and long-term preservation.

## File Organization Structure

```
experimental_validation_dataset/
├── metadata/
│   ├── dataset_metadata.json          # Dataset-level metadata
│   ├── sample_registry.json           # Sample tracking registry
│   └── equipment_registry.json        # Equipment specifications
├── samples/
│   ├── S001/                          # Sample-specific directories
│   │   ├── fabrication/
│   │   │   ├── parameters.json
│   │   │   ├── process_log.json
│   │   │   └── quality_control/
│   │   ├── measurements/
│   │   │   ├── warp/
│   │   │   ├── stress/
│   │   │   └── material_properties/
│   │   ├── validation/
│   │   │   ├── ml_predictions/
│   │   │   ├── fea_comparisons/
│   │   │   └── analysis_results/
│   │   └── sample_metadata.json
│   ├── S002/
│   └── ...
├── calibration/
│   ├── standards/
│   ├── procedures/
│   └── certificates/
└── analysis/
    ├── scripts/
    ├── results/
    └── reports/
```

## File Naming Conventions

### General Format
```
[Category]_[SampleID]_[Technique]_[Date]_[Operator]_[Version].[ext]
```

### Examples
- `WARP_S001_LSCM_20241015_JD_v1.h5`
- `STRESS_S015_XRD_20241016_MK_v2.json`
- `FAB_S023_PARAMS_20241014_AL_v1.json`

### Category Codes
- `FAB`: Fabrication data
- `WARP`: Warp measurement data
- `STRESS`: Stress measurement data
- `MAT`: Material property data
- `VAL`: Validation data
- `CAL`: Calibration data

## Data File Formats

### Primary Format: HDF5 (.h5)
Used for large numerical datasets (warp maps, stress fields, measurement arrays)

#### Structure Example
```
/measurement_data/
├── /raw_data/
│   ├── height_map          [2D array, float64]
│   ├── intensity_map       [2D array, float64]
│   └── coordinates/
│       ├── x_coords        [1D array, float64]
│       ├── y_coords        [1D array, float64]
│       └── z_coords        [1D array, float64]
├── /processed_data/
│   ├── warp_map           [2D array, float64]
│   ├── filtered_data      [2D array, float64]
│   └── metrics/
│       ├── peak_to_valley [scalar, float64]
│       ├── rms_warp       [scalar, float64]
│       └── statistics     [1D array, float64]
├── /metadata/
│   ├── measurement_parameters
│   ├── coordinate_system
│   ├── units
│   └── processing_history
└── /quality/
    ├── uncertainty_map    [2D array, float64]
    ├── data_quality_flags [2D array, int8]
    └── validation_metrics
```

#### HDF5 Attributes
All datasets include standardized attributes:
- `units`: Physical units (SI preferred)
- `description`: Human-readable description
- `creation_date`: ISO 8601 timestamp
- `operator`: Measurement operator ID
- `equipment_id`: Equipment identifier
- `calibration_date`: Last calibration date
- `uncertainty`: Measurement uncertainty
- `processing_software`: Software used for processing

### Secondary Format: JSON (.json)
Used for metadata, parameters, and structured information

#### Example: Sample Fabrication Parameters
```json
{
  "sample_id": "S001",
  "fabrication_date": "2024-10-15T09:30:00Z",
  "operator": "JD",
  "batch_id": "B2024_045",
  "parameters": {
    "electrolyte_thickness": {
      "value": 150,
      "units": "micrometers",
      "tolerance": 5
    },
    "anode_thickness": {
      "value": 300,
      "units": "micrometers",
      "tolerance": 10
    },
    "cathode_thickness": {
      "value": 50,
      "units": "micrometers",
      "tolerance": 5
    },
    "sintering_temperature": {
      "value": 1350,
      "units": "celsius",
      "tolerance": 5
    },
    "sintering_time": {
      "value": 4,
      "units": "hours",
      "tolerance": 0.1
    },
    "cooling_rate": {
      "value": 2,
      "units": "celsius_per_minute",
      "tolerance": 0.2
    }
  },
  "materials": {
    "electrolyte": {
      "composition": "8YSZ",
      "supplier": "Tosoh",
      "grade": "TZ-8Y",
      "lot_number": "TZ8Y-2024-03",
      "particle_size": {
        "d50": 0.5,
        "units": "micrometers"
      }
    },
    "anode": {
      "composition": "NiO-YSZ",
      "nio_content": 65,
      "ysz_content": 35,
      "units": "volume_percent"
    },
    "cathode": {
      "composition": "LSM-YSZ",
      "lsm_content": 70,
      "ysz_content": 30,
      "units": "volume_percent"
    }
  },
  "process_conditions": {
    "ambient_temperature": {
      "value": 22,
      "units": "celsius"
    },
    "humidity": {
      "value": 48,
      "units": "percent"
    },
    "atmosphere": "air"
  },
  "quality_control": {
    "dimensional_check": "passed",
    "visual_inspection": "acceptable",
    "defects": []
  }
}
```

### Tertiary Format: CSV (.csv)
Used for tabular data and measurement logs

#### Example: Dimensional Measurements
```csv
sample_id,measurement_point,x_coord,y_coord,electrolyte_thickness,anode_thickness,cathode_thickness,total_thickness,measurement_date,operator
S001,P1,5,5,148.2,298.5,49.8,496.5,2024-10-15T10:15:00Z,JD
S001,P2,25,5,149.8,301.2,50.4,501.4,2024-10-15T10:16:00Z,JD
S001,P3,45,5,151.1,299.8,49.6,500.5,2024-10-15T10:17:00Z,JD
```

## Coordinate Systems and Units

### Coordinate System Convention
- **Origin**: Lower-left corner of sample (when viewed from cathode side)
- **X-axis**: Along sample width (positive to the right)
- **Y-axis**: Along sample length (positive upward)
- **Z-axis**: Through sample thickness (positive toward cathode)

### Unit Standards
- **Length**: Micrometers (μm) for thickness, millimeters (mm) for lateral dimensions
- **Stress**: Megapascals (MPa)
- **Temperature**: Celsius (°C)
- **Time**: Hours (h) for process times, seconds (s) for measurement times
- **Angles**: Radians (rad)

## Data Quality and Validation

### Data Quality Flags
Each measurement point includes quality flags:
- `0`: Good data
- `1`: Questionable data (within uncertainty limits)
- `2`: Poor data (exceeds uncertainty limits)
- `3`: Invalid data (measurement failed)
- `4`: Interpolated data
- `5`: Extrapolated data

### Uncertainty Representation
All measurements include uncertainty information:
```json
{
  "value": 150.5,
  "units": "micrometers",
  "uncertainty": {
    "standard_uncertainty": 2.1,
    "expanded_uncertainty": 4.2,
    "coverage_factor": 2,
    "confidence_level": 95,
    "units": "micrometers"
  }
}
```

### Traceability Chain
Each measurement links to calibration data:
```json
{
  "measurement_id": "WARP_S001_LSCM_20241015_JD_v1",
  "traceability": {
    "equipment_id": "LSCM_001",
    "calibration_date": "2024-10-01T08:00:00Z",
    "calibration_certificate": "CAL_LSCM001_20241001.pdf",
    "standard_reference": "NIST_SRM_2460",
    "uncertainty_budget": "UB_LSCM001_2024.json"
  }
}
```

## Version Control and Change Management

### Version Numbering
- Major version (X.0): Significant changes to data structure or methodology
- Minor version (X.Y): Addition of new data or minor corrections
- Patch version (X.Y.Z): Bug fixes or metadata corrections

### Change Documentation
```json
{
  "version_history": [
    {
      "version": "1.0.0",
      "date": "2024-10-15T00:00:00Z",
      "changes": "Initial dataset release",
      "author": "Dataset Team"
    },
    {
      "version": "1.1.0",
      "date": "2024-10-20T00:00:00Z",
      "changes": "Added Raman spectroscopy data",
      "author": "MK"
    }
  ]
}
```

## Data Integrity and Checksums

### File Integrity
All data files include checksums for integrity verification:
```json
{
  "file_info": {
    "filename": "WARP_S001_LSCM_20241015_JD_v1.h5",
    "file_size": 157286400,
    "checksum_algorithm": "SHA-256",
    "checksum": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
    "creation_date": "2024-10-15T14:30:00Z"
  }
}
```

## Data Access and APIs

### Programmatic Access
Python API for data access:
```python
import sofc_dataset as sfc

# Load dataset
dataset = sfc.load_dataset('/path/to/dataset')

# Access sample data
sample = dataset.get_sample('S001')
warp_data = sample.get_warp_measurement('LSCM')
stress_data = sample.get_stress_measurement('XRD')

# Query capabilities
high_stress_samples = dataset.query(
    'stress.von_mises_max > 100',
    technique='XRD'
)
```

### REST API Endpoints
```
GET /api/v1/samples                    # List all samples
GET /api/v1/samples/{id}              # Get sample details
GET /api/v1/samples/{id}/warp         # Get warp measurements
GET /api/v1/samples/{id}/stress       # Get stress measurements
GET /api/v1/measurements/{id}         # Get specific measurement
POST /api/v1/validate                 # Validate data format
```

## Backup and Archival

### Backup Strategy
- **Daily**: Incremental backup of new/modified files
- **Weekly**: Full dataset backup
- **Monthly**: Offsite archive copy
- **Annually**: Long-term preservation format conversion

### Archive Format
Long-term preservation uses standardized formats:
- **Numerical data**: HDF5 with embedded metadata
- **Metadata**: JSON with schema validation
- **Documentation**: PDF/A for long-term readability
- **Images**: TIFF with embedded metadata

This comprehensive data format specification ensures the experimental validation dataset is well-structured, traceable, and suitable for long-term preservation and analysis.