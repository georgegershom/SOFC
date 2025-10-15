#!/bin/bash

# =====================================================
# Multi-Fidelity SOFC Dataset Generation Pipeline
# =====================================================

echo "=================================================="
echo "Multi-Fidelity SOFC Dataset Generation Pipeline"
echo "=================================================="
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/{low_fidelity,mid_fidelity,high_fidelity,experimental}
mkdir -p results/{figures,analysis}
mkdir -p notebooks
mkdir -p docs

# Parse command line arguments
MODE=${1:-"test"}  # test, full, or custom

if [ "$MODE" == "test" ]; then
    echo ""
    echo "Running in TEST mode (small dataset for testing)..."
    echo "=================================================="
    
    # Generate small test dataset
    python generate_dataset.py --quick-test --parallel
    
    # Create visualizations
    python visualize_data.py --data ./data --output ./results/figures --samples 3
    
    echo ""
    echo "Test run complete!"
    echo "Check ./data for generated datasets"
    echo "Check ./results/figures for visualizations"
    
elif [ "$MODE" == "full" ]; then
    echo ""
    echo "Running in FULL mode (complete dataset generation)..."
    echo "WARNING: This will take several hours and require ~25GB disk space"
    echo "=================================================="
    
    read -p "Continue? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Generate full dataset
        python generate_dataset.py --all --parallel
        
        # Create comprehensive visualizations
        python visualize_data.py --data ./data --output ./results/figures --samples 10
        
        echo ""
        echo "Full dataset generation complete!"
    else
        echo "Aborted."
        exit 1
    fi
    
elif [ "$MODE" == "custom" ]; then
    echo ""
    echo "Custom mode - select what to generate:"
    echo "1) Low-fidelity only (fast)"
    echo "2) Mid-fidelity only (moderate)"
    echo "3) High-fidelity only (slow)"
    echo "4) Experimental only (fast)"
    echo "5) Visualizations only"
    echo ""
    
    read -p "Select option (1-5): " option
    
    case $option in
        1)
            echo "Generating low-fidelity dataset..."
            python generate_dataset.py --fidelity low
            ;;
        2)
            echo "Generating mid-fidelity dataset..."
            python generate_dataset.py --fidelity mid
            ;;
        3)
            echo "Generating high-fidelity dataset..."
            python generate_dataset.py --fidelity high
            ;;
        4)
            echo "Generating experimental dataset..."
            python generate_dataset.py --fidelity experimental
            ;;
        5)
            echo "Creating visualizations..."
            python visualize_data.py --data ./data --output ./results/figures
            ;;
        *)
            echo "Invalid option"
            exit 1
            ;;
    esac
    
else
    echo "Usage: ./run_all.sh [test|full|custom]"
    echo "  test   - Generate small test dataset (default)"
    echo "  full   - Generate complete dataset (hours)"
    echo "  custom - Select specific components"
    exit 1
fi

# Validate generated data
echo ""
echo "Validating generated datasets..."
python generate_dataset.py --validate-only

# Generate summary report
echo ""
echo "=================================================="
echo "Dataset Summary"
echo "=================================================="

# Check disk usage
echo "Disk usage:"
du -sh data/* 2>/dev/null | head -20

# Count files
echo ""
echo "Files generated:"
find data -name "*.h5" -type f | wc -l
echo "HDF5 files"

find results/figures -name "*.png" -type f | wc -l
echo "Visualization files"

echo ""
echo "=================================================="
echo "Pipeline complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Review visualizations in ./results/figures/"
echo "2. Check dataset summary in ./results/figures/dataset_summary.txt"
echo "3. Load data for ML training using provided examples"
echo ""

# Deactivate virtual environment
deactivate