# SOFC Model - Quick Start Guide

## 🚀 5-Minute Setup

### Step 1: Download Script
```bash
# Navigate to your working directory
cd ~/sofc_simulation

# Download script (replace URL with actual location)
wget https://your-repo/SOFC_Validation_Model_v3.py
```

### Step 2: Run Model Builder
```bash
# Build model without GUI
abaqus cae noGUI=SOFC_Validation_Model_v3.py
```

**Expected Output:**
```
╔══════════════════════════════════════════════════════════════════════════╗
║   ███████╗ ██████╗ ███████╗ ██████╗    ███╗   ███╗ ██████╗ ██████╗     ║
║   ...                                                                    ║
╚══════════════════════════════════════════════════════════════════════════╝

[12:34:56] | INFO     | Initializing configuration...
[12:34:56] | INFO     | ✓ All configuration parameters validated successfully
...
[12:35:12] | INFO     | ✓ MODEL BUILD COMPLETE - READY FOR SUBMISSION
```

### Step 3: Submit Analysis
```bash
# Submit job with 4 CPUs
abaqus job=Job-Validation-Cooling-v3 cpus=4 interactive
```

### Step 4: View Results
```bash
# Open results in Abaqus Viewer
abaqus viewer odb=Job-Validation-Cooling-v3.odb
```

---

## 📋 Command Reference

### Basic Commands

| Task | Command |
|------|---------|
| Build model | `abaqus cae noGUI=SOFC_Validation_Model_v3.py` |
| Submit job | `abaqus job=Job-Validation-Cooling-v3 interactive` |
| Check status | `abaqus job=Job-Validation-Cooling-v3 status` |
| View results | `abaqus viewer odb=Job-Validation-Cooling-v3.odb` |
| Extract data | `abaqus python extract_results.py` |

### Advanced Commands

| Task | Command |
|------|---------|
| Parallel execution (8 CPUs) | `abaqus job=Job-Validation-Cooling-v3 cpus=8 mp_mode=threads` |
| Background submission | `abaqus job=Job-Validation-Cooling-v3 background` |
| Resume analysis | `abaqus job=Job-Validation-Cooling-v3 oldjob=previous_job recover` |
| Data check only | `abaqus job=Job-Validation-Cooling-v3 datacheck` |

---

## 🎛️ Quick Configuration

### Change Electrolyte Thickness

Edit `SOFC_Validation_Model_v3.py`:

```python
# Line ~350: In SOFCModelConfig.__init__()
self.GEOM = {
    'L_cell': 10.0,
    'H_anode': 0.500,
    'H_electrolyte': 0.015,  # Change from 0.010 to 0.015 mm
}
```

### Adjust Mesh Density

```python
# Line ~460: In SOFCModelConfig.__init__()
self.MESH = {
    'global_size': 0.15,        # Finer mesh (was 0.20)
    'elec_divisions': 8,        # More elements (was 5)
}
```

### Modify Temperature Range

```python
# Line ~360: In SOFCModelConfig.__init__()
self.TEMP = {
    'T_sintering': 1400.0,     # Higher sintering temp (was 1300)
    'T_room': 25.0,
}
```

---

## 📊 Output Files Explained

### Essential Files

| File | Size | Description | Keep? |
|------|------|-------------|-------|
| `Validation_Planar_Cell_v3.cae` | ~5 MB | Model database | ✅ Yes |
| `Job-Validation-Cooling-v3.odb` | ~20 MB | Results database | ✅ Yes |
| `logs/SOFC_model_*.log` | ~100 KB | Execution log | ✅ Yes |

### Intermediate Files

| File | Size | Description | Keep? |
|------|------|-------------|-------|
| `Job-Validation-Cooling-v3.dat` | ~500 KB | Analysis summary | ⚠️ Optional |
| `Job-Validation-Cooling-v3.msg` | ~50 KB | Solver messages | ⚠️ Optional |
| `Job-Validation-Cooling-v3.sta` | ~10 KB | Status file | ❌ No |
| `Job-Validation-Cooling-v3.com` | ~5 KB | Command file | ❌ No |
| `Job-Validation-Cooling-v3.prt` | ~100 KB | Part file | ❌ No |

### Cleanup Command

```bash
# Remove intermediate files
rm Job-Validation-Cooling-v3.{sta,com,prt,sim,stt}
```

---

## 🔍 Quick Diagnostics

### Check If Model Built Successfully

```bash
# Look for .cae file
ls -lh Validation_Planar_Cell_v3.cae

# Check log for errors
tail -n 50 logs/SOFC_model_*.log | grep -i error
```

### Monitor Job Progress

```bash
# Watch status file in real-time
watch -n 5 'tail -n 20 Job-Validation-Cooling-v3.sta'

# Check message file for warnings
grep -i "warning\|error" Job-Validation-Cooling-v3.msg
```

### Verify Results Quality

```bash
# Check for completion
grep "COMPLETED" Job-Validation-Cooling-v3.sta

# Check convergence
grep "ITERATIONS" Job-Validation-Cooling-v3.sta | tail -n 20
```

---

## ⚡ Performance Tips

### Faster Execution

1. **Use more CPUs** (diminishing returns after 8):
   ```bash
   abaqus job=Job-Validation-Cooling-v3 cpus=8 mp_mode=threads
   ```

2. **Coarsen mesh** (for quick tests):
   ```python
   self.MESH['global_size'] = 0.30  # Larger elements
   ```

3. **Reduce output frequency**:
   ```python
   # In define_field_outputs()
   frequency=LAST_INCREMENT  # Only final frame
   ```

### Better Accuracy

1. **Refine mesh** (doubles compute time):
   ```python
   self.MESH['global_size'] = 0.15
   self.MESH['elec_divisions'] = 8
   ```

2. **Tighter convergence**:
   ```python
   self.SOLVER['max_inc'] = 0.1      # Smaller steps
   self.SOLVER['max_num_inc'] = 2000  # More increments
   ```

---

## 🐛 Common Issues & Fixes

### Issue 1: "Command not found: abaqus"

**Fix:** Add Abaqus to PATH:
```bash
export PATH=/opt/SIMULIA/EstProducts/2023/linux_a64/code/bin:$PATH
```

### Issue 2: "TypeError: keyword error on master"

**Status:** ✅ **FIXED in v3.0.0**

**Explanation:** Version 3.0 uses correct `main`/`secondary` keywords instead of deprecated `master`/`slave`.

### Issue 3: Job fails with "Excessive distortion"

**Fix:** Reduce temperature step:
```python
self.SOLVER['init_inc'] = 0.02  # Smaller initial increment
self.SOLVER['max_inc'] = 0.05   # Smaller maximum increment
```

### Issue 4: Mesh generation fails

**Fix:** Check geometry parameters:
```python
# Ensure all values are positive
validate_positive(self.GEOM['H_electrolyte'], 'H_electrolyte')
```

### Issue 5: Out of memory error

**Fix:** Reduce mesh density or use scratch directory:
```bash
abaqus job=Job-Validation-Cooling-v3 scratch=/fast/scratch
```

---

## 📈 Results Extraction Cheat Sheet

### Python Script Method

Create `extract_results.py`:

```python
from odbAccess import *

# Open database
odb = openOdb('Job-Validation-Cooling-v3.odb')

# Get last frame
step = odb.steps['Step_Cooling']
frame = step.frames[-1]

# Extract stress
stress = frame.fieldOutputs['S']
for value in stress.values[:10]:  # First 10 elements
    print('Element:', value.elementLabel)
    print('  S11:', value.data[0], 'Pa')
    print('  S22:', value.data[1], 'Pa')

odb.close()
```

Run:
```bash
abaqus python extract_results.py > results.txt
```

### Interactive Viewer Method

1. Open ODB: `abaqus viewer odb=Job-Validation-Cooling-v3.odb`
2. **Plot → Contours → Field Output → S (Stress)**
3. **Tools → Query → Probe Values**
4. **File → Print → Save As Image**

---

## 🎓 Learning Path

### Beginner (Week 1)
1. ✅ Run default model
2. ✅ View stress contours in Viewer
3. ✅ Change mesh size and observe differences
4. ✅ Extract stress values at interface

### Intermediate (Week 2-3)
1. ⬜ Modify material properties
2. ⬜ Add parametric studies (loop over parameters)
3. ⬜ Create custom post-processing scripts
4. ⬜ Compare results with experimental data

### Advanced (Week 4+)
1. ⬜ Implement orthotropic material models
2. ⬜ Add contact interactions
3. ⬜ Couple with thermal analysis
4. ⬜ Optimize design parameters

---

## 📞 Getting Help

### Documentation
- **Full manual:** `README_SOFC_Model.md`
- **Script comments:** Inline documentation in `.py` file
- **Abaqus docs:** `abaqus doc`

### Support Channels
- 🐛 **Bug reports:** Open issue on GitHub
- 💬 **Questions:** Discussion forum
- 📧 **Email:** sofc.research@institution.edu

### Useful Resources
- [Abaqus Documentation](https://help.3ds.com/)
- [Python Scripting Guide](https://help.3ds.com/2023/English/DSSIMULIA_Established/SIMACAECMDRefMap/simacmd-c-intpyscr.htm)
- [FEA Best Practices](https://abaqus-docs.mit.edu/)

---

## ✅ Checklist for Production Runs

Before submitting final analysis:

- [ ] Configuration validated (run `config.validate()`)
- [ ] Mesh convergence study completed
- [ ] Material properties verified against literature
- [ ] Boundary conditions reviewed
- [ ] Data check passed (`abaqus job=... datacheck`)
- [ ] Sufficient disk space available (>5 GB)
- [ ] Backup of model database created
- [ ] Results directory prepared

---

**Version:** 3.0.0  
**Last Updated:** 2026-02-09  
**Author:** SOFC Research Team
