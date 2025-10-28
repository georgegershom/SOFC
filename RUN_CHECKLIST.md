# SOFC Simulation - Pre-Run Checklist

## ✓ System Check

- [ ] Abaqus installed and in PATH
  ```bash
  abaqus information=release
  ```
- [ ] Sufficient disk space (10+ GB free)
  ```bash
  df -h /workspace
  ```
- [ ] Python available for visualization
  ```bash
  python --version
  pip list | grep -E "numpy|matplotlib"
  ```

## ✓ File Verification

All scripts present in `/workspace`:

- [ ] `sofc_simulation.py` (21 KB) - Model generation
- [ ] `sofc_postprocess.py` (18 KB) - Post-processing
- [ ] `run_all_simulations.sh` (14 KB) - Batch runner
- [ ] `visualize_results.py` (15 KB) - Plotting
- [ ] `README_SIMULATION.md` (13 KB) - Documentation
- [ ] `QUICK_START_GUIDE.txt` (10 KB) - Quick start

## ✓ Execution Options

### Option 1: Full Automated Batch (Recommended)
```bash
cd /workspace
./run_all_simulations.sh
```
**Time**: 6-12 hours (all scenarios)

### Option 2: Single Scenario Test
```bash
# Generate models
abaqus cae noGUI=sofc_simulation.py

# Run HR10 (fastest, 3 hours)
abaqus job=Job_SOFC_HR10 cpus=4 interactive

# Post-process
abaqus python sofc_postprocess.py Job_SOFC_HR10.odb

# Visualize
python visualize_results.py Job_SOFC_HR10_results.npz
```

### Option 3: Interactive CAE
```bash
abaqus cae
# Then in CAE: File → Run Script → sofc_simulation.py
# Review model, modify if needed, submit via Job Manager
```

## ✓ Expected Timeline

| Scenario | Ramp Time | Total Sim Time | Post-Process | Total |
|----------|-----------|----------------|--------------|-------|
| HR10 | 87.5 min | 30-45 min | 5 min | ~1 hour |
| HR4 | 218.75 min | 60-90 min | 5 min | ~2 hours |
| HR1 | 875 min | 120-180 min | 10 min | ~4 hours |

## ✓ What to Monitor

During simulation, check:

1. **Progress files**:
   - `Job_SOFC_*.sta` - Status file (increment info)
   - `Job_SOFC_*.msg` - Message file (warnings/errors)
   - `Job_SOFC_*.log` - Standard output

2. **Key indicators**:
   - Time increment size (should be stable)
   - Convergence rate (equilibrium iterations < 10)
   - Memory usage (< 8 GB per job)

3. **Red flags**:
   - ❌ "Negative eigenvalues" → instability
   - ❌ "Divergence" → reduce increment
   - ❌ "Element distortion" → mesh issue

## ✓ Output Validation

After completion, verify:

- [ ] ODB files created (~500 MB each)
  ```bash
  ls -lh Job_SOFC_*.odb
  ```

- [ ] NPZ files generated (~100-300 MB each)
  ```bash
  ls -lh *_results.npz
  ```

- [ ] Summary CSV files present
  ```bash
  ls -lh *_summary.csv
  head -3 Job_SOFC_HR1_summary.csv
  ```

- [ ] Key metrics reasonable:
  - Max von Mises: 100-300 MPa ✓
  - Max damage: 0-1 ✓
  - Crack depth: 0-50 μm ✓
  - Delamination risk: 0-2 ✓

## ✓ Visualization Check

Generate plots:
```bash
python visualize_results.py --all
```

Verify plots created:
- [ ] `*_thermal_history.png`
- [ ] `*_stress_evolution.png`
- [ ] `*_delamination_risk.png`
- [ ] `*_damage_evolution.png`
- [ ] `*_field_snapshot.png`
- [ ] `SOFC_comparison_scenarios.png`

## ✓ Common First-Run Issues

### Issue: Abaqus license error
**Fix**: 
```bash
# Check license server
abaqus licensing lmstat
# Or use alternative license
export ABAQUS_LICENSE=flex@license-server
```

### Issue: Python module not found (visualization)
**Fix**:
```bash
pip install numpy matplotlib
# Or use conda
conda install numpy matplotlib
```

### Issue: Simulation very slow
**Check**:
- CPU usage (should be ~100% × num_cpus)
- Disk I/O (move to SSD if on HDD)
- Swap usage (add more RAM if swapping)

### Issue: Results look wrong
**Verify**:
1. Material units: E in Pa (not GPa!)
2. Temperature in Kelvin (not Celsius in material def)
3. Mesh quality: aspect ratio ~1
4. BCs: not over-constrained

## ✓ Data Analysis Pipeline

1. **Load NPZ data**:
   ```python
   import numpy as np
   data = np.load('Job_SOFC_HR1_results.npz')
   print(data.files)
   ```

2. **Extract key metrics**:
   ```python
   max_stress = data['von_mises'].max() / 1e6  # MPa
   final_damage = data['damage_D'][-1]
   crack_depth = data['crack_depth_um'][-1]
   ```

3. **Compare scenarios**:
   ```python
   hr1 = np.load('Job_SOFC_HR1_results.npz')
   hr4 = np.load('Job_SOFC_HR4_results.npz')
   hr10 = np.load('Job_SOFC_HR10_results.npz')
   
   # Compare final crack depths
   print(f"HR1:  {hr1['crack_depth_um'][-1]:.2f} μm")
   print(f"HR4:  {hr4['crack_depth_um'][-1]:.2f} μm")
   print(f"HR10: {hr10['crack_depth_um'][-1]:.2f} μm")
   ```

## ✓ Final Checklist Before Publication

- [ ] Mesh convergence study completed
- [ ] Material properties validated against literature
- [ ] Results compared with experiments
- [ ] Sensitivity analysis performed
- [ ] All figures high-resolution (300 dpi)
- [ ] Methodology documented in thesis/paper
- [ ] Data archived and backed up
- [ ] Code version controlled (git)

## ✓ Ready to Run?

If all checks pass:
```bash
cd /workspace
./run_all_simulations.sh
```

Monitor in separate terminal:
```bash
watch -n 60 'tail -20 Job_SOFC_HR1.sta'
```

---

**Estimated completion**: 6-12 hours from now

**Good luck!** 🚀
