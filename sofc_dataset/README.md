## SOFC Warp-Stress Synthetic Dataset Generator

This project creates paired warp surfaces and residual stress tensors for multilayer SOFC plates using simplified classical laminate theory with thermal and eigenstrain loads. It is designed to approximate the "virtual DOE" for ML-augmented inverse modeling when full FEA is too expensive.

### Install

Without venv (user site):
```bash
pip3 install --user -e .
```

### Usage

```bash
python3 -m sofc_dataset.cli generate --out-dir data/out --n 5 --grid 64,64,6 --surf 128,128
```

Outputs HDF5 files with:
- `metadata/*` process and geometry parameters
- `heightmap` 2.5D warp surface (top)
- `stress/{sigma_xx,sigma_yy,sigma_xy,sigma_zz,tau_xz,tau_yz}` voxelized tensors
