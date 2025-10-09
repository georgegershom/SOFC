# 3D SOFC Microstructural Dataset Generator

A comprehensive toolkit for generating, analyzing, and visualizing realistic 3D microstructural data for Solid Oxide Fuel Cell (SOFC) electrode modeling.

## Features

### 🏗️ **3D Microstructure Generation**
- **Realistic SOFC electrode geometry** with proper phase segmentation
- **Multi-phase structure**: Pore, Ni, YSZ anode, YSZ electrolyte, and interlayers
- **Configurable parameters**: Porosity, phase ratios, interface geometry
- **High-resolution voxelated data** suitable for computational modeling

### 📊 **Advanced Analysis**
- **Phase connectivity analysis** with component labeling
- **Pore network characterization** including tortuosity and connectivity
- **Interface property analysis** with roughness quantification
- **Mechanical property estimation** using rule of mixtures
- **Volume fraction calculations** and statistical analysis

### 🎨 **Comprehensive Visualization**
- **Interactive 3D visualization** with Plotly and PyVista
- **Cross-sectional analysis** at multiple z-positions
- **Phase distribution plots** and statistical summaries
- **Interface analysis** with detailed morphology
- **Real-time parameter adjustment** in interactive dashboard

### 🔧 **Mesh Generation**
- **Structured hexahedral meshes** for finite element analysis
- **Unstructured tetrahedral meshes** using marching cubes
- **Surface mesh generation** for boundary conditions
- **Interface-specific meshes** for delamination analysis
- **Multiple export formats**: VTK, STL, OBJ, XDMF, GMSH

## Installation

### Prerequisites
- Python 3.8 or higher
- Required packages (see requirements.txt)

### Quick Setup
```bash
# Clone or download the repository
git clone <repository-url>
cd sofc-microstructure-generator

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Dependencies
```
numpy>=1.21.0
scipy>=1.7.0
scikit-image>=0.18.0
matplotlib>=3.5.0
h5py>=3.1.0
vtk>=9.0.0
tifffile>=2021.7.2
pyvista>=0.32.0
open3d>=0.13.0
trimesh>=3.9.0
pandas>=1.3.0
seaborn>=0.11.0
plotly>=5.0.0
dash>=2.0.0
dash-bootstrap-components>=1.0.0
jupyter>=1.0.0
ipywidgets>=7.6.0
tqdm>=4.62.0
```

## Quick Start

### 1. Generate 3D Microstructure
```python
from microstructure_generator import SOFCMicrostructureGenerator

# Create generator with realistic parameters
generator = SOFCMicrostructureGenerator(
    resolution=(256, 256, 128),  # Width × Height × Depth
    voxel_size=0.1,              # 100 nm voxel size
    porosity=0.3,                # 30% porosity
    ni_ysz_ratio=0.6,            # 60% Ni in anode
    ysz_thickness=10.0           # 10 μm electrolyte thickness
)

# Generate microstructure
microstructure = generator.generate_microstructure()

# Save in multiple formats
generator.save_hdf5('output/sofc_microstructure.h5')
generator.save_tiff_stack('output/sofc_microstructure')
generator.save_vtk('output/sofc_microstructure.vtk')
```

### 2. Analyze Microstructure
```python
from microstructure_analysis import MicrostructureAnalyzer

# Load data
with h5py.File('output/sofc_microstructure.h5', 'r') as f:
    microstructure = f['microstructure'][:]
    voxel_size = f.attrs['voxel_size_um']

# Create analyzer
analyzer = MicrostructureAnalyzer(microstructure, voxel_size)

# Run comprehensive analysis
connectivity = analyzer.analyze_phase_connectivity()
pore_network = analyzer.analyze_pore_network()
interfaces = analyzer.analyze_interface_properties()
mechanical = analyzer.estimate_mechanical_properties()

# Generate report
report = analyzer.create_comprehensive_report()
report.to_csv('output/analysis_report.csv')
```

### 3. Generate Computational Meshes
```python
from mesh_generator import MeshGenerator

# Create mesh generator
mesh_gen = MeshGenerator(microstructure, voxel_size)

# Generate different mesh types
hex_mesh = mesh_gen.generate_structured_hex_mesh()
tet_mesh = mesh_gen.generate_unstructured_tet_mesh()
surface_mesh = mesh_gen.generate_surface_mesh()
interface_mesh = mesh_gen.generate_interface_mesh()

# Export meshes
mesh_gen.export_mesh(hex_mesh, 'output/hex_mesh.vtk', 'vtk')
mesh_gen.export_mesh(tet_mesh, 'output/tet_mesh.stl', 'stl')
mesh_gen.export_mesh(surface_mesh, 'output/surface_mesh.obj', 'obj')
```

### 4. Interactive Visualization
```python
from visualization_dashboard import MicrostructureVisualizer

# Create visualizer
visualizer = MicrostructureVisualizer(microstructure, voxel_size)

# Create interactive dashboard
app = visualizer.create_interactive_dashboard()
app.run_server(debug=True, host='0.0.0.0', port=8050)
```

## Command Line Usage

### Generate Complete Dataset
```bash
# Generate microstructure, analyze, and create visualizations
python microstructure_generator.py
python microstructure_analysis.py
python mesh_generator.py
python visualization_dashboard.py
```

### Individual Components
```bash
# Generate only microstructure
python microstructure_generator.py

# Analyze existing microstructure
python microstructure_analysis.py

# Generate meshes from existing data
python mesh_generator.py

# Create visualizations
python visualization_dashboard.py
```

## Output Files

The toolkit generates comprehensive output in multiple formats:

### Data Files
- `sofc_microstructure.h5` - HDF5 format with metadata
- `sofc_microstructure_*.tif` - TIFF stack for image processing
- `sofc_microstructure.vtk` - VTK format for visualization
- `phase_analysis.csv` - Phase distribution analysis

### Analysis Results
- `microstructure_analysis.csv` - Comprehensive analysis report
- `microstructure_analysis.h5` - Detailed analysis data
- `mesh_statistics.json` - Mesh quality metrics

### Meshes
- `structured_hex_mesh.vtk/.xdmf` - Structured hexahedral mesh
- `unstructured_tet_mesh.vtk/.stl` - Unstructured tetrahedral mesh
- `surface_mesh.stl/.obj` - Surface mesh for boundaries
- `interface_mesh.stl` - Interface-specific mesh
- `gmsh_mesh.msh` - GMSH format mesh

### Visualizations
- `3d_visualization.html/.png` - Interactive 3D plot
- `cross_sections.html/.png` - Cross-sectional views
- `phase_distribution.html/.png` - Phase distribution plots
- `interface_analysis.html/.png` - Interface analysis
- `microstructure_3d.png` - PyVista 3D rendering

## Advanced Usage

### Custom Phase Generation
```python
# Create custom phase distribution
generator = SOFCMicrostructureGenerator(
    resolution=(512, 512, 256),
    voxel_size=0.05,  # 50 nm resolution
    porosity=0.25,    # 25% porosity
    ni_ysz_ratio=0.7, # 70% Ni in anode
    ysz_thickness=15.0 # 15 μm electrolyte
)

# Generate with custom parameters
microstructure = generator.generate_microstructure()
```

### High-Resolution Analysis
```python
# For high-resolution data, use subsampling
analyzer = MicrostructureAnalyzer(microstructure, voxel_size)

# Analyze with memory optimization
connectivity = analyzer.analyze_phase_connectivity()
pore_network = analyzer.analyze_pore_network()
```

### Custom Mesh Generation
```python
# Generate mesh with specific element sizes
mesh_gen = MeshGenerator(microstructure, voxel_size)

# Custom element sizes
hex_mesh = mesh_gen.generate_structured_hex_mesh(element_size=0.2)
tet_mesh = mesh_gen.generate_unstructured_tet_mesh(
    max_element_size=0.5,
    min_element_size=0.1
)
```

## Data Format Specifications

### Phase Labels
- `0` - Pore phase
- `1` - Ni (Nickel)
- `2` - YSZ Anode (Yttria-Stabilized Zirconia)
- `3` - YSZ Electrolyte
- `4` - Interlayer

### Coordinate System
- X-axis: Width direction
- Y-axis: Height direction  
- Z-axis: Depth direction (electrolyte at top)
- Units: Micrometers (μm)

### File Formats
- **HDF5**: Hierarchical data format with metadata
- **VTK**: Visualization Toolkit format
- **TIFF**: Image stack format
- **STL**: Stereolithography format for 3D printing
- **OBJ**: Wavefront OBJ format
- **XDMF**: eXtensible Data Model and Format

## Applications

### Computational Modeling
- **Finite Element Analysis (FEA)** for mechanical properties
- **Computational Fluid Dynamics (CFD)** for gas transport
- **Electrochemical modeling** for performance prediction
- **Thermal analysis** for thermal management

### Research Applications
- **Microstructure optimization** for improved performance
- **Interface analysis** for delamination studies
- **Pore network analysis** for transport properties
- **Phase connectivity** for electrical conductivity

### Educational Use
- **Materials science education** with realistic microstructures
- **Computational modeling training** with real data
- **Visualization tools** for understanding complex structures

## Performance Considerations

### Memory Requirements
- **256³ voxels**: ~16 MB RAM
- **512³ voxels**: ~128 MB RAM
- **1024³ voxels**: ~1 GB RAM

### Computational Time
- **Generation**: 1-5 minutes (depending on resolution)
- **Analysis**: 2-10 minutes (depending on complexity)
- **Mesh generation**: 5-30 minutes (depending on mesh type)
- **Visualization**: 1-3 minutes (depending on detail level)

## Troubleshooting

### Common Issues

1. **Memory errors with large datasets**
   - Use subsampling for visualization
   - Process data in chunks
   - Increase system RAM

2. **Mesh generation failures**
   - Check phase connectivity
   - Adjust element sizes
   - Verify input data quality

3. **Visualization performance**
   - Reduce max_voxels parameter
   - Use lower resolution for preview
   - Enable hardware acceleration

### Getting Help

- Check the example notebooks in `examples/`
- Review the API documentation
- Submit issues on the project repository

## Contributing

Contributions are welcome! Please see the contributing guidelines for details on:
- Code style and standards
- Testing requirements
- Documentation updates
- Feature requests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{sofc_microstructure_generator,
  title={3D SOFC Microstructural Dataset Generator},
  author={AI Assistant},
  year={2024},
  url={https://github.com/your-repo/sofc-microstructure-generator}
}
```

## Acknowledgments

- Built with Python scientific computing stack
- Visualization powered by Plotly and PyVista
- Mesh generation using VTK and GMSH
- Inspired by real SOFC microstructure studies