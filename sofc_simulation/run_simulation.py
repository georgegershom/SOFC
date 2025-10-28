#!/usr/bin/env python
"""
Main runner script for SOFC simulation workflow
Automates the complete analysis pipeline
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

class SOFCSimulationRunner:
    """Orchestrates the complete SOFC simulation workflow"""
    
    def __init__(self, heating_rate='HR1', abaqus_cmd='abaqus'):
        """
        Initialize simulation runner
        
        Args:
            heating_rate: 'HR1', 'HR4', or 'HR10'
            abaqus_cmd: Abaqus command (default 'abaqus')
        """
        self.heating_rate = heating_rate
        self.abaqus_cmd = abaqus_cmd
        self.project_dir = Path(__file__).parent
        self.inp_dir = self.project_dir / 'inp'
        self.results_dir = self.project_dir / 'results'
        self.scripts_dir = self.project_dir / 'scripts'
        self.post_dir = self.project_dir / 'post_processing'
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
        
        # Job name
        self.job_name = f'SOFC_{heating_rate}_Job'
        
    def check_abaqus(self):
        """Check if Abaqus is available"""
        try:
            result = subprocess.run([self.abaqus_cmd, 'information=version'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Abaqus found: {result.stdout.strip()}")
                return True
            else:
                print(f"✗ Abaqus not found or error occurred")
                return False
        except FileNotFoundError:
            print(f"✗ Abaqus command '{self.abaqus_cmd}' not found")
            print("  Please ensure Abaqus is installed and in PATH")
            print("  Or specify the correct command with --abaqus-cmd")
            return False
    
    def create_model_cae(self):
        """Create model using CAE Python script"""
        print(f"\n{'='*60}")
        print(f"Creating SOFC model for {self.heating_rate}")
        print(f"{'='*60}")
        
        script_path = self.scripts_dir / 'create_sofc_model.py'
        
        # Modify script to use selected heating rate
        cmd = [
            self.abaqus_cmd, 'cae', 'noGUI=' + str(script_path),
            '--', self.heating_rate
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, cwd=str(self.results_dir),
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Model created successfully")
                return True
            else:
                print(f"✗ Error creating model:")
                print(result.stderr)
                return False
        except Exception as e:
            print(f"✗ Exception: {e}")
            return False
    
    def run_analysis_inp(self):
        """Run analysis using INP file"""
        print(f"\n{'='*60}")
        print(f"Running analysis for {self.heating_rate}")
        print(f"{'='*60}")
        
        inp_file = self.inp_dir / 'sofc_main.inp'
        
        # Modify INP file for selected heating rate
        self._modify_inp_for_heating_rate(inp_file)
        
        cmd = [
            self.abaqus_cmd, 'job=' + self.job_name,
            'input=' + str(inp_file),
            'interactive'
        ]
        
        print(f"Running: {' '.join(cmd)}")
        print("This may take several minutes...")
        
        try:
            # Start the job
            process = subprocess.Popen(cmd, cwd=str(self.results_dir),
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     text=True)
            
            # Monitor progress
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            rc = process.poll()
            
            if rc == 0:
                print("✓ Analysis completed successfully")
                return True
            else:
                print(f"✗ Analysis failed with return code {rc}")
                # Check for .dat file for error messages
                dat_file = self.results_dir / f'{self.job_name}.dat'
                if dat_file.exists():
                    print("\nError details from .dat file:")
                    with open(dat_file, 'r') as f:
                        lines = f.readlines()[-50:]  # Last 50 lines
                        for line in lines:
                            if 'ERROR' in line or 'WARNING' in line:
                                print(line.strip())
                return False
                
        except Exception as e:
            print(f"✗ Exception: {e}")
            return False
    
    def _modify_inp_for_heating_rate(self, inp_file):
        """Modify INP file to use selected heating rate"""
        # Read original file
        with open(inp_file, 'r') as f:
            content = f.read()
        
        # Replace amplitude reference
        content = content.replace('amplitude=AMP_HR1', f'amplitude=AMP_{self.heating_rate}')
        
        # Adjust time periods based on heating rate
        time_periods = {
            'HR1': 105600.0,  # Total time for HR1
            'HR4': 26850.0,   # Total time for HR4
            'HR10': 11100.0   # Total time for HR10
        }
        
        total_time = time_periods[self.heating_rate]
        
        # Update step time periods
        import re
        pattern = r'(\*Heat Transfer.*?\n)(.*?)(\n\*)'
        replacement = lambda m: m.group(1) + f'1.0, {total_time}, 1.0, 100.0' + m.group(3)
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # Write modified file
        modified_inp = self.results_dir / f'sofc_{self.heating_rate}.inp'
        with open(modified_inp, 'w') as f:
            f.write(content)
        
        return modified_inp
    
    def run_post_processing(self):
        """Run post-processing scripts"""
        print(f"\n{'='*60}")
        print("Running post-processing analysis")
        print(f"{'='*60}")
        
        odb_file = self.results_dir / f'{self.job_name}.odb'
        
        if not odb_file.exists():
            print(f"✗ ODB file not found: {odb_file}")
            return False
        
        # Run damage analysis
        print("\n1. Running damage analysis...")
        damage_script = self.post_dir / 'damage_analysis.py'
        report_file = self.results_dir / f'damage_report_{self.heating_rate}.txt'
        
        cmd = [
            self.abaqus_cmd, 'python', str(damage_script),
            str(odb_file), '--output', str(report_file)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Damage analysis complete: {report_file}")
            else:
                print(f"✗ Damage analysis failed: {result.stderr}")
        except Exception as e:
            print(f"✗ Exception in damage analysis: {e}")
        
        # Run visualization
        print("\n2. Generating visualizations...")
        viz_script = self.post_dir / 'visualize_results.py'
        
        cmd = [
            self.abaqus_cmd, 'python', str(viz_script),
            str(odb_file), '--all'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Visualizations generated")
            else:
                print(f"✗ Visualization failed: {result.stderr}")
        except Exception as e:
            print(f"✗ Exception in visualization: {e}")
        
        return True
    
    def generate_summary_report(self):
        """Generate a summary report of the simulation"""
        print(f"\n{'='*60}")
        print("Generating summary report")
        print(f"{'='*60}")
        
        report_path = self.results_dir / f'summary_{self.heating_rate}.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"SOFC SIMULATION SUMMARY - {self.heating_rate}\n")
            f.write("="*70 + "\n\n")
            
            f.write("SIMULATION PARAMETERS\n")
            f.write("-"*40 + "\n")
            f.write(f"Heating Rate: {self.heating_rate}\n")
            
            hr_details = {
                'HR1': '1°C/min (875 min ramp, 10 min hold, 875 min cool)',
                'HR4': '4°C/min (218.75 min ramp, 10 min hold, 218.75 min cool)',
                'HR10': '10°C/min (87.5 min ramp, 10 min hold, 87.5 min cool)'
            }
            f.write(f"Schedule: {hr_details[self.heating_rate]}\n")
            f.write(f"Peak Temperature: 900°C (1173 K)\n")
            f.write(f"Geometry: 10mm × 1mm 2D cross-section\n")
            f.write(f"Analysis Type: Sequential thermo-mechanical\n\n")
            
            f.write("MATERIAL LAYERS\n")
            f.write("-"*40 + "\n")
            f.write("1. Anode (Ni-YSZ): 0.0-0.4 mm\n")
            f.write("2. Electrolyte (8YSZ): 0.4-0.5 mm\n")
            f.write("3. Cathode (LSM): 0.5-0.9 mm\n")
            f.write("4. Interconnect (Ferritic Steel): 0.9-1.0 mm\n\n")
            
            f.write("KEY MATERIAL PROPERTIES\n")
            f.write("-"*40 + "\n")
            f.write("Temperature-dependent elastic moduli (298K → 1273K)\n")
            f.write("Thermal expansion coefficients (10.5-13.5 × 10⁻⁶ K⁻¹)\n")
            f.write("Johnson-Cook plasticity for Ni-YSZ\n")
            f.write("Norton-Bailey creep for ceramics\n\n")
            
            f.write("OUTPUT FILES\n")
            f.write("-"*40 + "\n")
            
            # List generated files
            output_files = [
                f'{self.job_name}.odb',
                f'{self.job_name}.dat',
                f'{self.job_name}.msg',
                f'{self.job_name}.sta',
                f'damage_report_{self.heating_rate}.txt'
            ]
            
            for file in output_files:
                file_path = self.results_dir / file
                if file_path.exists():
                    size = file_path.stat().st_size / 1024  # KB
                    f.write(f"✓ {file} ({size:.1f} KB)\n")
                else:
                    f.write(f"✗ {file} (not found)\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("Simulation complete. Check damage report for detailed results.\n")
            f.write("="*70 + "\n")
        
        print(f"✓ Summary report saved: {report_path}")
        return report_path
    
    def run_complete_workflow(self):
        """Run the complete simulation workflow"""
        print("\n" + "="*70)
        print("SOFC SIMULATION WORKFLOW")
        print("="*70)
        
        start_time = time.time()
        
        # Check Abaqus availability
        if not self.check_abaqus():
            print("\n⚠ Abaqus not available. Generating input files only.")
            self.generate_input_files_only()
            return False
        
        # Run simulation steps
        steps = [
            ("Creating model", self.create_model_cae),
            ("Running analysis", self.run_analysis_inp),
            ("Post-processing", self.run_post_processing),
            ("Generating report", self.generate_summary_report)
        ]
        
        for step_name, step_func in steps:
            print(f"\n→ {step_name}...")
            if not step_func():
                print(f"✗ Workflow stopped at: {step_name}")
                return False
        
        elapsed_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✓ WORKFLOW COMPLETE")
        print(f"Total time: {elapsed_time/60:.1f} minutes")
        print(f"Results saved in: {self.results_dir}")
        print(f"{'='*70}\n")
        
        return True
    
    def generate_input_files_only(self):
        """Generate input files without running Abaqus"""
        print("\nGenerating input files for manual execution...")
        
        # Copy and modify INP file
        inp_file = self.inp_dir / 'sofc_main.inp'
        modified_inp = self._modify_inp_for_heating_rate(inp_file)
        
        print(f"✓ INP file generated: {modified_inp}")
        
        # Generate batch script for running
        batch_script = self.results_dir / f'run_{self.heating_rate}.sh'
        with open(batch_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Run SOFC simulation for {self.heating_rate}\n\n")
            f.write(f"# Step 1: Create model in CAE\n")
            f.write(f"abaqus cae noGUI={self.scripts_dir}/create_sofc_model.py -- {self.heating_rate}\n\n")
            f.write(f"# Step 2: Run analysis\n")
            f.write(f"abaqus job={self.job_name} input={modified_inp.name} interactive\n\n")
            f.write(f"# Step 3: Post-processing\n")
            f.write(f"abaqus python {self.post_dir}/damage_analysis.py {self.job_name}.odb\n")
            f.write(f"abaqus python {self.post_dir}/visualize_results.py {self.job_name}.odb --all\n")
        
        os.chmod(batch_script, 0o755)
        print(f"✓ Batch script generated: {batch_script}")
        
        print("\nTo run the simulation manually:")
        print(f"  cd {self.results_dir}")
        print(f"  ./{batch_script.name}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='SOFC Simulation Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_simulation.py --rate HR1        # Run with 1°C/min heating rate
  python run_simulation.py --rate HR4        # Run with 4°C/min heating rate
  python run_simulation.py --rate HR10       # Run with 10°C/min heating rate
  python run_simulation.py --all             # Run all three heating rates
  python run_simulation.py --generate-only   # Generate files without running
        """
    )
    
    parser.add_argument('--rate', choices=['HR1', 'HR4', 'HR10'],
                       default='HR1', help='Heating rate (default: HR1)')
    parser.add_argument('--all', action='store_true',
                       help='Run all heating rates')
    parser.add_argument('--abaqus-cmd', default='abaqus',
                       help='Abaqus command (default: abaqus)')
    parser.add_argument('--generate-only', action='store_true',
                       help='Generate input files only, do not run simulation')
    
    args = parser.parse_args()
    
    if args.all:
        # Run all heating rates
        for rate in ['HR1', 'HR4', 'HR10']:
            print(f"\n{'#'*70}")
            print(f"# Running simulation for {rate}")
            print(f"{'#'*70}")
            
            runner = SOFCSimulationRunner(heating_rate=rate,
                                        abaqus_cmd=args.abaqus_cmd)
            
            if args.generate_only:
                runner.generate_input_files_only()
            else:
                runner.run_complete_workflow()
    else:
        # Run single heating rate
        runner = SOFCSimulationRunner(heating_rate=args.rate,
                                    abaqus_cmd=args.abaqus_cmd)
        
        if args.generate_only:
            runner.generate_input_files_only()
        else:
            runner.run_complete_workflow()


if __name__ == '__main__':
    main()