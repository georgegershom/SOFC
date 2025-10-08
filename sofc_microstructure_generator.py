"""
SOFC 3D Microstructural Data Generator
Generates realistic 3D voxelated microstructure data for Solid Oxide Fuel Cell electrodes
with phase segmentation, interface geometry, and proper volume fractions.
"""

import numpy as np
import scipy.ndimage as ndi
from scipy import signal
from skimage import morphology, filters, segmentation
import h5py
import tifffile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime
import json
from perlin_noise import PerlinNoise
from typing import Tuple, Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class SOFCMicrostructureGenerator:
    """
    Generates realistic 3D microstructural data for SOFC electrodes
    with multiple phases and complex interface geometries.
    """
    
    # Material phase identifiers
    PHASE_PORE = 0
    PHASE_NI = 1
    PHASE_YSZ_ANODE = 2
    PHASE_YSZ_ELECTROLYTE = 3
    PHASE_INTERLAYER_GDC = 4  # Gadolinium-doped Ceria interlayer
    PHASE_INTERLAYER_SDC = 5  # Samarium-doped Ceria interlayer
    
    # Realistic material parameters (based on literature)
    MATERIAL_PARAMS = {
        'anode_porosity': 0.35,  # 35% porosity typical for Ni-YSZ anode
        'ni_volume_fraction': 0.30,  # 30% Ni in solid phase
        'ysz_anode_fraction': 0.35,  # 35% YSZ in anode
        'electrolyte_density': 0.95,  # 95% dense electrolyte
        'interlayer_thickness': 2,  # µm typical interlayer (scaled for demo)
        'anode_thickness': 20,  # µm typical anode thickness (scaled for demo)
        'electrolyte_thickness': 5,  # µm typical electrolyte (scaled for demo)
        'particle_size_ni': 1.5,  # µm mean Ni particle size
        'particle_size_ysz': 0.8,  # µm mean YSZ particle size
        'tpb_density': 2.5e6,  # m/m³ triple phase boundary density
    }
    
    def __init__(self, 
                 dimensions: Tuple[int, int, int] = (256, 256, 128),
                 voxel_size: float = 0.5,  # µm
                 seed: int = 42):
        """
        Initialize the microstructure generator.
        
        Args:
            dimensions: (x, y, z) dimensions in voxels
            voxel_size: Physical size of each voxel in micrometers
            seed: Random seed for reproducibility
        """
        self.dimensions = dimensions
        self.voxel_size = voxel_size
        self.seed = seed
        np.random.seed(seed)
        
        # Physical dimensions in micrometers
        self.physical_dims = tuple(d * voxel_size for d in dimensions)
        
        # Initialize the 3D volume
        self.volume = np.zeros(dimensions, dtype=np.uint8)
        
        # Store metadata
        self.metadata = {
            'dimensions': dimensions,
            'voxel_size': voxel_size,
            'physical_dimensions_um': self.physical_dims,
            'generation_date': datetime.now().isoformat(),
            'seed': seed,
            'phases': {
                0: 'Pore',
                1: 'Nickel',
                2: 'YSZ_Anode',
                3: 'YSZ_Electrolyte',
                4: 'GDC_Interlayer',
                5: 'SDC_Interlayer'
            }
        }
        
    def generate_perlin_field(self, scale: float = 20, octaves: int = 4) -> np.ndarray:
        """Generate 3D Perlin noise field for realistic morphology."""
        noise = PerlinNoise(octaves=octaves, seed=self.seed)
        field = np.zeros(self.dimensions)
        
        for x in range(self.dimensions[0]):
            for y in range(self.dimensions[1]):
                for z in range(self.dimensions[2]):
                    field[x, y, z] = noise([x/scale, y/scale, z/scale])
                    
        return field
    
    def generate_spherical_particles(self, 
                                   num_particles: int,
                                   mean_radius: float,
                                   std_radius: float) -> np.ndarray:
        """Generate spherical particles with given size distribution."""
        particles = np.zeros(self.dimensions, dtype=bool)
        
        for _ in range(num_particles):
            # Random center position
            center = [np.random.randint(0, dim) for dim in self.dimensions]
            
            # Random radius from normal distribution
            radius = max(1, np.random.normal(mean_radius, std_radius))
            
            # Create sphere
            x, y, z = np.ogrid[:self.dimensions[0], :self.dimensions[1], :self.dimensions[2]]
            mask = ((x - center[0])**2 + (y - center[1])**2 + 
                   (z - center[2])**2) <= radius**2
            particles |= mask
            
        return particles
    
    def generate_anode_structure(self) -> np.ndarray:
        """
        Generate realistic Ni-YSZ anode microstructure with:
        - Interpenetrating Ni and YSZ networks
        - Realistic pore structure
        - Triple phase boundaries
        """
        print("Generating anode microstructure...")
        
        # Calculate layer boundaries in voxels
        anode_thickness_voxels = int(self.MATERIAL_PARAMS['anode_thickness'] / self.voxel_size)
        anode_layer = np.zeros(self.dimensions, dtype=bool)
        anode_layer[:, :, :anode_thickness_voxels] = True
        
        # Generate Ni particles
        ni_particle_radius = self.MATERIAL_PARAMS['particle_size_ni'] / self.voxel_size
        num_ni_particles = int(np.prod(self.dimensions[:2]) * anode_thickness_voxels * 
                              self.MATERIAL_PARAMS['ni_volume_fraction'] / 
                              (4/3 * np.pi * ni_particle_radius**3))
        
        ni_phase = self.generate_spherical_particles(
            num_ni_particles, ni_particle_radius, ni_particle_radius * 0.3
        )
        
        # Apply sintering effect - particles coalesce
        ni_phase = ndi.binary_dilation(ni_phase, iterations=1)
        ni_phase = ndi.binary_erosion(ni_phase, iterations=1)
        
        # Generate YSZ network
        ysz_particle_radius = self.MATERIAL_PARAMS['particle_size_ysz'] / self.voxel_size
        num_ysz_particles = int(np.prod(self.dimensions[:2]) * anode_thickness_voxels * 
                               self.MATERIAL_PARAMS['ysz_anode_fraction'] / 
                               (4/3 * np.pi * ysz_particle_radius**3))
        
        ysz_phase = self.generate_spherical_particles(
            num_ysz_particles, ysz_particle_radius, ysz_particle_radius * 0.2
        )
        
        # Create pore structure using Perlin noise
        perlin_field = self.generate_perlin_field(scale=30)
        porosity_threshold = np.percentile(perlin_field, 
                                          self.MATERIAL_PARAMS['anode_porosity'] * 100)
        pore_phase = perlin_field < porosity_threshold
        
        # Ensure phases don't overlap and apply to anode region only
        anode_structure = np.zeros(self.dimensions, dtype=np.uint8)
        
        # Apply phases with priority: pores > Ni > YSZ
        anode_structure[anode_layer & pore_phase] = self.PHASE_PORE
        anode_structure[anode_layer & ni_phase & ~pore_phase] = self.PHASE_NI
        anode_structure[anode_layer & ysz_phase & ~pore_phase & ~ni_phase] = self.PHASE_YSZ_ANODE
        
        # Fill remaining anode volume with YSZ
        remaining = anode_layer & (anode_structure == 0)
        anode_structure[remaining] = self.PHASE_YSZ_ANODE
        
        return anode_structure
    
    def generate_electrolyte_structure(self) -> np.ndarray:
        """
        Generate dense YSZ electrolyte layer with:
        - High density (>95%)
        - Small residual porosity
        - Grain structure
        """
        print("Generating electrolyte structure...")
        
        # Calculate layer boundaries
        anode_thickness_voxels = int(self.MATERIAL_PARAMS['anode_thickness'] / self.voxel_size)
        electrolyte_thickness_voxels = int(self.MATERIAL_PARAMS['electrolyte_thickness'] / self.voxel_size)
        
        electrolyte_structure = np.zeros(self.dimensions, dtype=np.uint8)
        
        # Define electrolyte region
        z_start = anode_thickness_voxels
        z_end = z_start + electrolyte_thickness_voxels
        
        if z_end <= self.dimensions[2]:
            # Generate grain structure using Voronoi-like pattern
            num_grains = int(np.prod(self.dimensions[:2]) * electrolyte_thickness_voxels / 1000)
            grain_centers = np.random.rand(num_grains, 3)
            grain_centers[:, 2] = grain_centers[:, 2] * (z_end - z_start) + z_start
            grain_centers[:, 0] *= self.dimensions[0]
            grain_centers[:, 1] *= self.dimensions[1]
            
            # Create grain boundaries with small porosity
            perlin_field = self.generate_perlin_field(scale=10)
            porosity_threshold = np.percentile(perlin_field, 
                                              (1 - self.MATERIAL_PARAMS['electrolyte_density']) * 100)
            
            # Fill electrolyte region
            electrolyte_structure[:, :, z_start:z_end] = self.PHASE_YSZ_ELECTROLYTE
            
            # Add minimal porosity at grain boundaries
            pores = perlin_field[:, :, z_start:z_end] < porosity_threshold
            electrolyte_structure[:, :, z_start:z_end][pores] = self.PHASE_PORE
            
        return electrolyte_structure
    
    def generate_interlayer(self, interlayer_type: str = 'GDC') -> np.ndarray:
        """
        Generate interlayer between anode and electrolyte.
        Common materials: GDC (Gadolinium-doped Ceria) or SDC (Samarium-doped Ceria)
        """
        print(f"Generating {interlayer_type} interlayer...")
        
        # Calculate positions
        anode_thickness_voxels = int(self.MATERIAL_PARAMS['anode_thickness'] / self.voxel_size)
        interlayer_thickness_voxels = int(self.MATERIAL_PARAMS['interlayer_thickness'] / self.voxel_size)
        
        interlayer_structure = np.zeros(self.dimensions, dtype=np.uint8)
        
        z_start = anode_thickness_voxels - interlayer_thickness_voxels // 2
        z_end = anode_thickness_voxels + interlayer_thickness_voxels // 2
        
        if z_start >= 0 and z_end <= self.dimensions[2]:
            phase = self.PHASE_INTERLAYER_GDC if interlayer_type == 'GDC' else self.PHASE_INTERLAYER_SDC
            
            # Generate porous interlayer structure
            perlin_field = self.generate_perlin_field(scale=15)
            porosity_threshold = np.percentile(perlin_field, 20)  # 20% porosity in interlayer
            
            interlayer_structure[:, :, z_start:z_end] = phase
            pores = perlin_field[:, :, z_start:z_end] < porosity_threshold
            interlayer_structure[:, :, z_start:z_end][pores] = self.PHASE_PORE
            
        return interlayer_structure
    
    def add_interface_roughness(self, interface_z: int, roughness: float = 2.0):
        """
        Add realistic roughness to the anode/electrolyte interface.
        This is where delamination typically occurs.
        """
        print("Adding interface roughness...")
        
        # Generate 2D height map for interface
        perlin_field_2d = np.zeros((self.dimensions[0], self.dimensions[1]))
        noise = PerlinNoise(octaves=3, seed=self.seed)
        
        for x in range(self.dimensions[0]):
            for y in range(self.dimensions[1]):
                perlin_field_2d[x, y] = noise([x/40, y/40]) * roughness
                
        # Apply roughness to interface
        for x in range(self.dimensions[0]):
            for y in range(self.dimensions[1]):
                offset = int(perlin_field_2d[x, y])
                new_z = interface_z + offset
                
                if 0 < new_z < self.dimensions[2] - 1:
                    # Swap materials around interface to create roughness
                    if offset > 0:
                        self.volume[x, y, interface_z:new_z] = self.volume[x, y, interface_z]
                    elif offset < 0:
                        self.volume[x, y, new_z:interface_z] = self.volume[x, y, interface_z]
    
    def add_defects_and_cracks(self, crack_probability: float = 0.001):
        """
        Add realistic defects and potential delamination sites.
        """
        print("Adding defects and crack initiation sites...")
        
        # Find interface region
        anode_thickness_voxels = int(self.MATERIAL_PARAMS['anode_thickness'] / self.voxel_size)
        
        # Ensure interface region is within bounds
        start = max(0, anode_thickness_voxels - 5)
        end = min(self.dimensions[2], anode_thickness_voxels + 5)
        
        if end > start:
            interface_region = slice(start, end)
            region_depth = end - start
            
            # Add random crack initiation sites at interface
            crack_sites = np.random.rand(self.dimensions[0], self.dimensions[1], region_depth) < crack_probability
            self.volume[:, :, interface_region][crack_sites] = self.PHASE_PORE
        
        # Add some larger defects
        num_defects = np.random.poisson(5)
        for _ in range(num_defects):
            center = [np.random.randint(0, dim) for dim in self.dimensions[:2]]
            # Ensure z coordinate is within bounds
            z_center = min(max(5, anode_thickness_voxels + np.random.randint(-5, 5)), self.dimensions[2] - 5)
            center.append(z_center)
            
            # Create ellipsoidal defect
            x, y, z = np.ogrid[:self.dimensions[0], :self.dimensions[1], :self.dimensions[2]]
            mask = (((x - center[0])/3)**2 + ((y - center[1])/3)**2 + 
                   ((z - center[2])/1)**2) <= 1
            self.volume[mask] = self.PHASE_PORE
    
    def calculate_volume_fractions(self) -> Dict[int, float]:
        """Calculate volume fractions of each phase."""
        volume_fractions = {}
        total_voxels = np.prod(self.dimensions)
        
        for phase in range(6):
            count = np.sum(self.volume == phase)
            volume_fractions[phase] = count / total_voxels
            
        return volume_fractions
    
    def calculate_interface_area(self) -> Dict[str, float]:
        """Calculate interface areas between phases."""
        interfaces = {}
        
        # Calculate Ni-YSZ interface (important for TPB)
        ni_mask = self.volume == self.PHASE_NI
        ysz_mask = (self.volume == self.PHASE_YSZ_ANODE) | (self.volume == self.PHASE_YSZ_ELECTROLYTE)
        
        # Use morphological gradient to find interfaces
        ni_boundary = ndi.binary_dilation(ni_mask) ^ ni_mask
        ysz_boundary = ndi.binary_dilation(ysz_mask) ^ ysz_mask
        ni_ysz_interface = ni_boundary & ysz_boundary
        
        interfaces['Ni-YSZ'] = np.sum(ni_ysz_interface) * self.voxel_size**2
        
        # Calculate anode-electrolyte interface
        anode_mask = (self.volume == self.PHASE_NI) | (self.volume == self.PHASE_YSZ_ANODE)
        electrolyte_mask = self.volume == self.PHASE_YSZ_ELECTROLYTE
        
        anode_boundary = ndi.binary_dilation(anode_mask) ^ anode_mask
        electrolyte_boundary = ndi.binary_dilation(electrolyte_mask) ^ electrolyte_mask
        anode_electrolyte_interface = anode_boundary & electrolyte_boundary
        
        interfaces['Anode-Electrolyte'] = np.sum(anode_electrolyte_interface) * self.voxel_size**2
        
        return interfaces
    
    def calculate_tpb_density(self) -> float:
        """
        Calculate Triple Phase Boundary (TPB) density.
        TPB occurs where Ni, YSZ, and Pore phases meet.
        """
        print("Calculating TPB density...")
        
        # Find voxels where all three phases are adjacent
        ni_mask = self.volume == self.PHASE_NI
        ysz_mask = self.volume == self.PHASE_YSZ_ANODE
        pore_mask = self.volume == self.PHASE_PORE
        
        # Dilate each phase to find overlapping regions
        ni_dilated = ndi.binary_dilation(ni_mask)
        ysz_dilated = ndi.binary_dilation(ysz_mask)
        pore_dilated = ndi.binary_dilation(pore_mask)
        
        # TPB occurs where all three dilated phases overlap
        tpb = ni_dilated & ysz_dilated & pore_dilated
        
        # Calculate TPB length density (m/m³)
        tpb_voxels = np.sum(tpb)
        tpb_length = tpb_voxels * self.voxel_size  # µm
        volume_um3 = np.prod(self.physical_dims)
        tpb_density = (tpb_length * 1e-6) / (volume_um3 * 1e-18)  # m/m³
        
        return tpb_density
    
    def generate_full_structure(self, include_interlayer: bool = True):
        """Generate the complete SOFC microstructure."""
        print("Generating complete SOFC microstructure...")
        
        # Generate anode
        anode = self.generate_anode_structure()
        self.volume = anode
        
        # Generate electrolyte
        electrolyte = self.generate_electrolyte_structure()
        self.volume = np.maximum(self.volume, electrolyte)
        
        # Add interlayer if requested
        if include_interlayer:
            interlayer = self.generate_interlayer('GDC')
            # Interlayer overwrites existing structure in its region
            mask = interlayer > 0
            self.volume[mask] = interlayer[mask]
        
        # Add interface roughness for realistic delamination sites
        anode_thickness_voxels = int(self.MATERIAL_PARAMS['anode_thickness'] / self.voxel_size)
        self.add_interface_roughness(anode_thickness_voxels)
        
        # Add defects
        self.add_defects_and_cracks()
        
        # Calculate and store properties
        self.metadata['volume_fractions'] = self.calculate_volume_fractions()
        self.metadata['interface_areas_um2'] = self.calculate_interface_area()
        self.metadata['tpb_density_m_per_m3'] = self.calculate_tpb_density()
        
        print(f"Structure generation complete!")
        print(f"Volume fractions: {self.metadata['volume_fractions']}")
        print(f"TPB density: {self.metadata['tpb_density_m_per_m3']:.2e} m/m³")
        
        return self.volume


def main():
    """Main function to generate and save SOFC microstructure data."""
    
    # Create generator with realistic dimensions
    # 256x256x128 voxels at 0.5 µm/voxel = 128x128x64 µm physical size
    generator = SOFCMicrostructureGenerator(
        dimensions=(256, 256, 128),
        voxel_size=0.5,
        seed=42
    )
    
    # Generate the complete structure
    structure = generator.generate_full_structure(include_interlayer=True)
    
    # The structure is now stored in generator.volume and ready for export
    print("\n✅ Microstructure generation complete!")
    print(f"Dimensions: {generator.dimensions}")
    print(f"Physical size: {generator.physical_dims} µm")
    
    return generator


if __name__ == "__main__":
    generator = main()