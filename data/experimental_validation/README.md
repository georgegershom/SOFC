# Experimental Validation Dataset (The "Reality Check")

This dataset fabricates experimental measurements to validate ML-Augmented Inverse Modeling for residual stress quantification in warped SOFC plates. It simulates:

- High-resolution warp measurements (height maps and 3D point clouds)
- Curvature-based inverse stress (Stoney bilayer approximation)
- Layer removal + warp re-measurement sequence
- XRD-like and Raman-like stress proxy maps

## Structure

- `manifest.json`: Dataset index and metadata
- `samples/EVxxxx/`
  - `warp_height_map.png` (grayscale height map in µm; `.npy` used if PIL missing)
  - `warp_point_cloud.xyz` (x y z in mm)
  - `layer_removal_00.xyz`, `layer_removal_01.xyz`, ... (post-removal warps)
  - `xrd_map.npy`, `raman_map.npy` (proxy stress/strain fields)

Schema at `schemas/manifest.schema.json`.

## Fabrication assumptions

- Warp amplitude increases with thickness mismatch and sintering severity.
- Average film stress via Stoney formula using electrolyte-on-anode bilayer.
- Layer removal reduces curvature progressively by removing layer-specific contributions.
- XRD/Raman maps correlate with local slope/curvature plus noise.

## Usage

- Input the `warp_point_cloud.xyz` or `warp_height_map.png` to your ML model.
- Compare predicted stresses vs. `curvature_inverse.avg_stress_MPa` and trends across layer-removal sequences and XRD/Raman maps.

## Reproduce

```bash
python3 /workspace/tools/experimental_dataset/generate_validation_dataset.py
```

## License

For research use. No warranty; synthetic data only.
