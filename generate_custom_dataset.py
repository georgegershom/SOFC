"""
Custom Dataset Generator
Generate material properties at user-specified temperature points
"""

import argparse
from material_properties_loader import YSZMaterialProperties
import numpy as np

def generate_custom_dataset(temp_min, temp_max, num_points, output_file):
    """
    Generate a custom material properties dataset at specified temperatures.
    
    Parameters:
    -----------
    temp_min : float
        Minimum temperature in Celsius
    temp_max : float
        Maximum temperature in Celsius
    num_points : int
        Number of temperature points to generate
    output_file : str
        Output CSV file path
    """
    
    # Initialize properties loader
    ysz = YSZMaterialProperties()
    
    # Generate temperature points
    temperatures = np.linspace(temp_min, temp_max, num_points)
    
    print(f"\nGenerating custom dataset:")
    print(f"  Temperature range: {temp_min}°C to {temp_max}°C")
    print(f"  Number of points: {num_points}")
    print(f"  Output file: {output_file}")
    
    # Export
    ysz.export_for_fem(output_file, temperature_points=temperatures)
    
    print(f"\n✓ Custom dataset generated successfully!")
    print(f"\nPreview of data points:")
    print(f"  T = {temperatures[0]:.1f}°C: E = {ysz.get_property('Youngs_Modulus_GPa', temperatures[0]):.2f} GPa")
    print(f"  T = {temperatures[len(temperatures)//2]:.1f}°C: E = {ysz.get_property('Youngs_Modulus_GPa', temperatures[len(temperatures)//2]):.2f} GPa")
    print(f"  T = {temperatures[-1]:.1f}°C: E = {ysz.get_property('Youngs_Modulus_GPa', temperatures[-1]):.2f} GPa")

def main():
    parser = argparse.ArgumentParser(
        description='Generate custom YSZ material properties dataset at specified temperatures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 50 points from RT to 1200°C
  python generate_custom_dataset.py --tmin 25 --tmax 1200 --npoints 50 -o fem_input_1200C.csv
  
  # Generate 100 points for full sintering range
  python generate_custom_dataset.py --tmin 20 --tmax 1500 --npoints 100 -o sintering_profile.csv
  
  # Generate 20 points for SOFC operating range
  python generate_custom_dataset.py --tmin 600 --tmax 900 --npoints 20 -o sofc_operating.csv
        """
    )
    
    parser.add_argument('--tmin', type=float, required=True,
                       help='Minimum temperature (°C)')
    parser.add_argument('--tmax', type=float, required=True,
                       help='Maximum temperature (°C)')
    parser.add_argument('--npoints', type=int, default=50,
                       help='Number of temperature points (default: 50)')
    parser.add_argument('-o', '--output', type=str, default='custom_ysz_properties.csv',
                       help='Output CSV file (default: custom_ysz_properties.csv)')
    
    args = parser.parse_args()
    
    # Validation
    if args.tmin >= args.tmax:
        print("Error: tmin must be less than tmax")
        return 1
    
    if args.npoints < 2:
        print("Error: npoints must be at least 2")
        return 1
    
    if args.tmin < 0:
        print("Warning: Minimum temperature is below 0°C (extrapolation used)")
    
    if args.tmax > 1600:
        print("Warning: Maximum temperature exceeds 1600°C (extrapolation used, be cautious!)")
    
    # Generate dataset
    generate_custom_dataset(args.tmin, args.tmax, args.npoints, args.output)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())