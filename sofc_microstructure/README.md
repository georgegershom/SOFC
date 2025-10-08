# SOFC Microstructure Synthetic Dataset Generator

This project generates a synthetic 3D microstructure resembling an SOFC anode/electrolyte region from tomography: a voxelated, labeled volume, grayscale surrogate stack, and an interface mesh.

## Outputs
- `labels.tiff`: 3D labeled volume with classes: 0=pore, 1=anode (Ni-YSZ), 2=electrolyte (YSZ), 3=interlayer
- `grayscale.tiff`: pseudo-attenuation grayscale volume derived from labels
- `preview_png/`: PNG slices for quick browsing
- `interface_mesh.ply`: triangle mesh of the anode/electrolyte interface extracted from the interface height field
- `manifest.json`: metadata including dimensions, phase fractions, and interface roughness metrics

## Usage
```bash
python -m sofc_microstructure.generator --out /workspace/sofc_microstructure/output --nx 256 --ny 256 --nz 192 --porosity 0.35 --seed 42
```

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
