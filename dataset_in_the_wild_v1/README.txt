SOFC In-The-Wild Warp/Stress Dataset
Version: v1.0.0

Content:
- Synthetic but physics-inspired residual stress fields and resulting warpage for SOFC-like plates.
- Realistic measurement effects: noise, outliers, missing patches, edge-enhanced tension, anisotropy.
- Unknown drift across time: thickness, modulus, Poisson ratio.

Files per sample:
- coords_x_mm.npy, coords_y_mm.npy: measurement grid coordinates.
- warp_measured_um.npy: measured warp with noise/outliers/missing in micrometers (NaN for missing).
- warp_clean_um.npy: clean warp in micrometers.
- mask_uint8.npy: 1 for observed, 0 for missing.
- stress_true_mpa.npz: sigma_xx, sigma_yy, tau_xy in MPa.
- params.json: per-sample parameters including material and generator settings.
- indicators.json: edge_crack_risk in [0, ~2], >1 indicates elevated risk.

Splits:
- splits/train.txt, val.txt, test.txt list sample IDs.

Notes:
- Warp is computed from stress via a spectral Poisson solve proxy for plate bending.
- This is a fabricated dataset for benchmarking ML-augmented inverse modeling under in-the-wild effects.
