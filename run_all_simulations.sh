#!/bin/bash
################################################################################
# SOFC Simulation Batch Submission Script
################################################################################
# This script:
# 1. Generates Abaqus models for all heating rate scenarios (HR1, HR4, HR10)
# 2. Submits jobs to run simulations
# 3. Post-processes results to extract damage and delamination metrics
# 4. Generates summary reports
#
# Usage:
#   ./run_all_simulations.sh
################################################################################

set -e  # Exit on error

# Configuration
SCENARIOS=("HR1" "HR4" "HR10")
WORKSPACE="/workspace"
NCPUS=4
MEMORY=90  # percentage

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "========================================================================"
echo "SOFC MULTI-PHYSICS SIMULATION - BATCH RUNNER"
echo "========================================================================"
echo ""

# Check if Abaqus is available
if ! command -v abaqus &> /dev/null; then
    echo -e "${RED}ERROR: Abaqus command not found. Please ensure Abaqus is installed and in PATH.${NC}"
    echo ""
    echo "You may need to load the Abaqus module or set up environment:"
    echo "  module load abaqus"
    echo "  export PATH=/path/to/abaqus/Commands:\$PATH"
    exit 1
fi

ABAQUS_VERSION=$(abaqus information=release 2>/dev/null | grep -i "release" || echo "Unknown")
echo -e "${GREEN}Abaqus found: ${ABAQUS_VERSION}${NC}"
echo ""

################################################################################
# STEP 1: Generate Abaqus Models
################################################################################

echo "========================================================================"
echo "STEP 1: Generating Abaqus Models"
echo "========================================================================"
echo ""

cd ${WORKSPACE}

echo "Running: abaqus cae noGUI=sofc_simulation.py"
echo ""

# Run the model generation script in Abaqus CAE (no GUI mode)
abaqus cae noGUI=sofc_simulation.py > model_generation.log 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Model generation complete${NC}"
    echo ""
else
    echo -e "${RED}✗ Model generation failed. Check model_generation.log${NC}"
    cat model_generation.log
    exit 1
fi

# Check if CAE files were created
for scenario in "${SCENARIOS[@]}"; do
    if [ -f "SOFC_${scenario}.cae" ]; then
        echo -e "${GREEN}  ✓ SOFC_${scenario}.cae created${NC}"
    else
        echo -e "${YELLOW}  ⚠ SOFC_${scenario}.cae not found${NC}"
    fi
done
echo ""

################################################################################
# STEP 2: Submit and Run Simulations
################################################################################

echo "========================================================================"
echo "STEP 2: Running Simulations"
echo "========================================================================"
echo ""

for scenario in "${SCENARIOS[@]}"; do
    
    JOB_NAME="Job_SOFC_${scenario}"
    MODEL_NAME="SOFC_${scenario}"
    
    echo "------------------------------------------------------------------------"
    echo "Scenario: ${scenario}"
    echo "------------------------------------------------------------------------"
    echo ""
    
    # Check if job already completed
    if [ -f "${JOB_NAME}.odb" ]; then
        echo -e "${YELLOW}⚠ ODB file already exists: ${JOB_NAME}.odb${NC}"
        echo "  Skipping simulation (delete ODB to rerun)"
        echo ""
        continue
    fi
    
    # Submit job
    echo "Submitting job: ${JOB_NAME}"
    echo "  CPUs: ${NCPUS}"
    echo "  Memory: ${MEMORY}%"
    echo ""
    
    # Create input file from CAE model
    echo "  Extracting input file from CAE model..."
    abaqus cae noGUI=- <<EOF > ${JOB_NAME}_input.log 2>&1
import os
from abaqus import *
from abaqusConstants import *
import sys

# Open CAE file
cae_file = '${MODEL_NAME}.cae'
if os.path.exists(cae_file):
    openMdb(pathName=cae_file)
    
    # Create job and write input file
    mdb.Job(name='${JOB_NAME}', model='${MODEL_NAME}', type=ANALYSIS,
            numCpus=${NCPUS}, memory=${MEMORY}, memoryUnits=PERCENTAGE)
    mdb.jobs['${JOB_NAME}'].writeInput(consistencyChecking=OFF)
    print('Input file written: ${JOB_NAME}.inp')
else:
    print('ERROR: CAE file not found: ' + cae_file)
    sys.exit(1)
EOF
    
    if [ ! -f "${JOB_NAME}.inp" ]; then
        echo -e "${RED}  ✗ Failed to create input file${NC}"
        continue
    fi
    
    echo -e "${GREEN}  ✓ Input file created: ${JOB_NAME}.inp${NC}"
    echo ""
    
    # Submit job (interactive mode for immediate execution)
    echo "  Starting analysis..."
    START_TIME=$(date +%s)
    
    abaqus job=${JOB_NAME} input=${JOB_NAME}.inp \
        cpus=${NCPUS} memory=${MEMORY} memoryUnits=percentage \
        interactive > ${JOB_NAME}.log 2>&1
    
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    # Check job status
    if [ ${EXIT_CODE} -eq 0 ] && [ -f "${JOB_NAME}.odb" ]; then
        echo -e "${GREEN}  ✓ Analysis completed successfully${NC}"
        echo "  Elapsed time: $((ELAPSED / 60)) min $((ELAPSED % 60)) sec"
        echo ""
    else
        echo -e "${RED}  ✗ Analysis failed (exit code: ${EXIT_CODE})${NC}"
        echo "  Check log file: ${JOB_NAME}.log"
        
        # Show last 20 lines of log
        if [ -f "${JOB_NAME}.log" ]; then
            echo ""
            echo "  Last 20 lines of log:"
            tail -n 20 ${JOB_NAME}.log | sed 's/^/    /'
        fi
        
        echo ""
        continue
    fi
    
done

echo ""

################################################################################
# STEP 3: Post-Process Results
################################################################################

echo "========================================================================"
echo "STEP 3: Post-Processing Results"
echo "========================================================================"
echo ""

for scenario in "${SCENARIOS[@]}"; do
    
    JOB_NAME="Job_SOFC_${scenario}"
    ODB_FILE="${JOB_NAME}.odb"
    
    if [ ! -f "${ODB_FILE}" ]; then
        echo -e "${YELLOW}⚠ ODB not found for ${scenario}, skipping post-processing${NC}"
        echo ""
        continue
    fi
    
    echo "------------------------------------------------------------------------"
    echo "Post-processing: ${scenario}"
    echo "------------------------------------------------------------------------"
    echo ""
    
    echo "Running post-processing script..."
    abaqus python sofc_postprocess.py ${ODB_FILE} > ${JOB_NAME}_postprocess.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Post-processing complete${NC}"
        
        # Check output files
        if [ -f "${JOB_NAME}_results.npz" ]; then
            SIZE=$(ls -lh ${JOB_NAME}_results.npz | awk '{print $5}')
            echo "  Results: ${JOB_NAME}_results.npz (${SIZE})"
        fi
        
        if [ -f "${JOB_NAME}_summary.csv" ]; then
            echo "  Summary: ${JOB_NAME}_summary.csv"
        fi
        
        echo ""
    else
        echo -e "${RED}✗ Post-processing failed${NC}"
        echo "  Check log: ${JOB_NAME}_postprocess.log"
        echo ""
    fi
    
done

################################################################################
# STEP 4: Generate Combined Report
################################################################################

echo "========================================================================"
echo "STEP 4: Generating Combined Report"
echo "========================================================================"
echo ""

REPORT_FILE="SOFC_simulation_report.txt"

cat > ${REPORT_FILE} <<EOF
================================================================================
SOFC MULTI-PHYSICS SIMULATION - FINAL REPORT
================================================================================

Date: $(date)
Workspace: ${WORKSPACE}

================================================================================
SIMULATION OVERVIEW
================================================================================

Domain: 2D cross-section of SOFC repeat unit (10 mm × 1 mm)

Layers:
  - Anode (Ni-YSZ):        0.00 - 0.40 mm
  - Electrolyte (8YSZ):    0.40 - 0.50 mm
  - Cathode (LSM):         0.50 - 0.90 mm
  - Interconnect (Steel):  0.90 - 1.00 mm

Analysis Type: Sequential multi-physics
  Step A: Transient heat transfer (DC2D4 elements)
  Step B: Thermo-mechanical (CPS4 elements, NLGEOM ON)

Material Models:
  - Temperature-dependent elastic properties
  - Thermal expansion (CTE)
  - Johnson-Cook plasticity (Ni-YSZ)
  - Norton-Bailey creep (Ni-YSZ, YSZ)

Heating Scenarios:
  HR1:  1 °C/min to 900 °C, hold 10 min, cool 1 °C/min
  HR4:  4 °C/min to 900 °C, hold 10 min, cool 4 °C/min
  HR10: 10 °C/min to 900 °C, hold 10 min, cool 10 °C/min

================================================================================
RESULTS SUMMARY
================================================================================

EOF

# Extract key metrics from each scenario
for scenario in "${SCENARIOS[@]}"; do
    
    JOB_NAME="Job_SOFC_${scenario}"
    CSV_FILE="${JOB_NAME}_summary.csv"
    
    if [ -f "${CSV_FILE}" ]; then
        echo "--- Scenario: ${scenario} ---" >> ${REPORT_FILE}
        echo "" >> ${REPORT_FILE}
        
        # Extract final line (last time step)
        LAST_LINE=$(tail -n 1 ${CSV_FILE})
        
        # Parse values
        MAX_VM=$(echo ${LAST_LINE} | cut -d',' -f2)
        MAX_DAMAGE=$(echo ${LAST_LINE} | cut -d',' -f3)
        CRACK_DEPTH=$(echo ${LAST_LINE} | cut -d',' -f4)
        DELAM_AE=$(echo ${LAST_LINE} | cut -d',' -f5)
        DELAM_EC=$(echo ${LAST_LINE} | cut -d',' -f6)
        DELAM_CI=$(echo ${LAST_LINE} | cut -d',' -f7)
        
        echo "Final State:" >> ${REPORT_FILE}
        echo "  Max von Mises stress:  ${MAX_VM} MPa" >> ${REPORT_FILE}
        echo "  Max damage:            ${MAX_DAMAGE}" >> ${REPORT_FILE}
        echo "  Crack depth:           ${CRACK_DEPTH} μm" >> ${REPORT_FILE}
        echo "" >> ${REPORT_FILE}
        echo "Delamination Risk:" >> ${REPORT_FILE}
        echo "  Anode-Electrolyte:     ${DELAM_AE}" >> ${REPORT_FILE}
        echo "  Electrolyte-Cathode:   ${DELAM_EC}" >> ${REPORT_FILE}
        echo "  Cathode-Interconnect:  ${DELAM_CI}" >> ${REPORT_FILE}
        echo "" >> ${REPORT_FILE}
        
        # Determine if delamination occurred
        if (( $(echo "${DELAM_AE} > 1.0" | bc -l) )) || \
           (( $(echo "${DELAM_EC} > 1.0" | bc -l) )) || \
           (( $(echo "${DELAM_CI} > 1.0" | bc -l) )); then
            echo "  ⚠ DELAMINATION RISK EXCEEDED" >> ${REPORT_FILE}
        else
            echo "  ✓ No critical delamination" >> ${REPORT_FILE}
        fi
        
        echo "" >> ${REPORT_FILE}
        
    else
        echo "--- Scenario: ${scenario} ---" >> ${REPORT_FILE}
        echo "  No results available" >> ${REPORT_FILE}
        echo "" >> ${REPORT_FILE}
    fi
    
done

cat >> ${REPORT_FILE} <<EOF

================================================================================
OUTPUT FILES
================================================================================

For each scenario (HR1, HR4, HR10):

  1. Job_SOFC_<scenario>.odb       - Abaqus results database
  2. Job_SOFC_<scenario>.inp       - Abaqus input file
  3. Job_SOFC_<scenario>.log       - Analysis log
  4. Job_SOFC_<scenario>_results.npz - Numpy archive with field data
  5. Job_SOFC_<scenario>_summary.csv - Time-series summary metrics

Model files:
  - SOFC_<scenario>.cae             - Abaqus CAE model

Scripts:
  - sofc_simulation.py              - Model generation script
  - sofc_postprocess.py             - Post-processing script
  - run_all_simulations.sh          - This batch script

================================================================================
VISUALIZATION
================================================================================

To visualize results in Abaqus/Viewer:

  abaqus viewer odb=Job_SOFC_HR1.odb

Recommended plots:
  - S, Mises (von Mises stress)
  - TEMP (temperature field)
  - PEEQ (equivalent plastic strain)
  - CEEQ (equivalent creep strain)
  - S12 (shear stress at interfaces)

To plot damage and crack depth, use the NPZ files with Python/NumPy.

================================================================================
NEXT STEPS
================================================================================

1. Validate results against experimental data (DIC, XRD)
2. Refine mesh at high-gradient regions if needed
3. Add cohesive surfaces for explicit delamination modeling
4. Perform sensitivity analysis on material parameters
5. Integrate with ML/PSO optimization workflows

================================================================================
END OF REPORT
================================================================================
EOF

echo "Report generated: ${REPORT_FILE}"
echo ""

# Display report
cat ${REPORT_FILE}

################################################################################
# COMPLETION
################################################################################

echo ""
echo "========================================================================"
echo "BATCH SIMULATION COMPLETE"
echo "========================================================================"
echo ""
echo "Summary of completed scenarios:"
echo ""

for scenario in "${SCENARIOS[@]}"; do
    if [ -f "Job_SOFC_${scenario}.odb" ]; then
        echo -e "  ${GREEN}✓ ${scenario}${NC}"
    else
        echo -e "  ${RED}✗ ${scenario}${NC}"
    fi
done

echo ""
echo "All results are in: ${WORKSPACE}"
echo ""
echo "To review a specific simulation:"
echo "  abaqus viewer odb=Job_SOFC_HR1.odb"
echo ""
echo "To analyze results in Python:"
echo "  import numpy as np"
echo "  data = np.load('Job_SOFC_HR1_results.npz')"
echo ""
echo "========================================================================"
echo ""
