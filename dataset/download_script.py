#!/usr/bin/env python3
"""
Dataset Download and Setup Script
Fire-Resistant Rubberized Concrete Dataset - Phase 1

This script automates the download, verification, and setup of the complete dataset
including all analysis tools and dependencies.
"""

import os
import sys
import json
import hashlib
import zipfile
import urllib.request
from pathlib import Path
import subprocess
import platform

class DatasetDownloader:
    def __init__(self):
        """Initialize the dataset downloader"""
        self.dataset_info = {
            'name': 'Fire-Resistant Rubberized Concrete Dataset',
            'version': '1.0',
            'phase': 'Phase 1: Material Characterization & Specimen Preparation',
            'size_mb': 2.5,
            'files_count': 8
        }
        
        self.base_path = Path.cwd()
        self.dataset_path = self.base_path / 'dataset'
        
    def print_header(self):
        """Print welcome header"""
        print("=" * 80)
        print(f"📊 {self.dataset_info['name']}")
        print(f"🔬 {self.dataset_info['phase']}")
        print(f"📦 Version {self.dataset_info['version']}")
        print("=" * 80)
        print()
    
    def check_system_requirements(self):
        """Check system requirements"""
        print("🔍 Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            print("❌ Python 3.8 or higher required")
            print(f"   Current version: {python_version.major}.{python_version.minor}")
            return False
        else:
            print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check available disk space
        import shutil
        free_space_gb = shutil.disk_usage(self.base_path).free / (1024**3)
        required_space_gb = 0.5  # 500 MB
        
        if free_space_gb < required_space_gb:
            print(f"❌ Insufficient disk space")
            print(f"   Required: {required_space_gb:.1f} GB")
            print(f"   Available: {free_space_gb:.1f} GB")
            return False
        else:
            print(f"✅ Disk space: {free_space_gb:.1f} GB available")
        
        # Check platform
        system = platform.system()
        print(f"✅ Platform: {system}")
        
        print()
        return True
    
    def create_directory_structure(self):
        """Create the dataset directory structure"""
        print("📁 Creating directory structure...")
        
        directories = [
            'dataset',
            'dataset/01_constituent_materials',
            'dataset/02_mix_designs',
            'dataset/03_fresh_properties',
            'dataset/04_analysis_scripts'
        ]
        
        for directory in directories:
            dir_path = self.base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {directory}")
        
        print()
    
    def download_sample_data(self):
        """Create sample data files (simulating download)"""
        print("⬇️  Downloading dataset files...")
        
        # In a real scenario, this would download from a repository
        # For this example, we'll create the files locally
        
        files_created = [
            'dataset/01_constituent_materials/cement_data.json',
            'dataset/01_constituent_materials/aggregates_data.json',
            'dataset/01_constituent_materials/crumb_rubber_data.json',
            'dataset/01_constituent_materials/water_admixtures_data.json',
            'dataset/02_mix_designs/mix_design_matrix.json',
            'dataset/03_fresh_properties/fresh_concrete_data.json',
            'dataset/04_analysis_scripts/data_analysis.py',
            'dataset/04_analysis_scripts/requirements.txt'
        ]
        
        for file_path in files_created:
            full_path = self.base_path / file_path
            if full_path.exists():
                print(f"   ✅ {file_path} ({full_path.stat().st_size / 1024:.1f} KB)")
            else:
                print(f"   ❌ {file_path} (missing)")
        
        print()
    
    def verify_data_integrity(self):
        """Verify downloaded data integrity"""
        print("🔐 Verifying data integrity...")
        
        # Check if all required files exist
        required_files = [
            'dataset/01_constituent_materials/cement_data.json',
            'dataset/01_constituent_materials/aggregates_data.json',
            'dataset/01_constituent_materials/crumb_rubber_data.json',
            'dataset/01_constituent_materials/water_admixtures_data.json',
            'dataset/02_mix_designs/mix_design_matrix.json',
            'dataset/03_fresh_properties/fresh_concrete_data.json'
        ]
        
        all_files_present = True
        total_size = 0
        
        for file_path in required_files:
            full_path = self.base_path / file_path
            if full_path.exists():
                file_size = full_path.stat().st_size
                total_size += file_size
                
                # Verify JSON format
                try:
                    with open(full_path, 'r') as f:
                        json.load(f)
                    print(f"   ✅ {file_path} (valid JSON, {file_size / 1024:.1f} KB)")
                except json.JSONDecodeError:
                    print(f"   ❌ {file_path} (invalid JSON)")
                    all_files_present = False
            else:
                print(f"   ❌ {file_path} (missing)")
                all_files_present = False
        
        print(f"   📊 Total dataset size: {total_size / (1024*1024):.2f} MB")
        print()
        
        return all_files_present
    
    def install_dependencies(self):
        """Install Python dependencies"""
        print("📦 Installing Python dependencies...")
        
        requirements_file = self.dataset_path / '04_analysis_scripts' / 'requirements.txt'
        
        if not requirements_file.exists():
            print("   ❌ requirements.txt not found")
            return False
        
        try:
            # Install requirements
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
            ], capture_output=True, text=True, check=True)
            
            print("   ✅ Dependencies installed successfully")
            
            # List installed packages
            installed_packages = [
                'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'scipy'
            ]
            
            for package in installed_packages:
                try:
                    __import__(package)
                    print(f"   ✅ {package}")
                except ImportError:
                    print(f"   ❌ {package} (failed to import)")
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install dependencies: {e}")
            print(f"   Error output: {e.stderr}")
            return False
        
        print()
        return True
    
    def run_initial_analysis(self):
        """Run initial analysis to verify everything works"""
        print("🧪 Running initial analysis test...")
        
        analysis_script = self.dataset_path / '04_analysis_scripts' / 'data_analysis.py'
        
        if not analysis_script.exists():
            print("   ❌ Analysis script not found")
            return False
        
        try:
            # Change to analysis directory
            original_cwd = os.getcwd()
            os.chdir(self.dataset_path / '04_analysis_scripts')
            
            # Run a simple test
            test_code = '''
import sys
sys.path.append(".")
from data_analysis import RubberizedConcreteAnalyzer

try:
    analyzer = RubberizedConcreteAnalyzer()
    print("✅ Dataset loaded successfully")
    
    # Test basic functionality
    df_mix = analyzer.create_mix_design_dataframe()
    df_fresh = analyzer.create_fresh_properties_dataframe()
    
    print(f"✅ Mix designs: {len(df_mix)} records")
    print(f"✅ Fresh properties: {len(df_fresh)} records")
    print("✅ Analysis tools working correctly")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
'''
            
            # Write and run test
            test_file = Path('test_analysis.py')
            with open(test_file, 'w') as f:
                f.write(test_code)
            
            result = subprocess.run([sys.executable, 'test_analysis.py'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("   " + result.stdout.replace('\\n', '\\n   '))
                success = True
            else:
                print(f"   ❌ Test failed: {result.stderr}")
                success = False
            
            # Cleanup
            if test_file.exists():
                test_file.unlink()
            
            os.chdir(original_cwd)
            
        except Exception as e:
            print(f"   ❌ Failed to run analysis test: {e}")
            os.chdir(original_cwd)
            success = False
        
        print()
        return success
    
    def create_quick_start_guide(self):
        """Create a quick start guide"""
        print("📋 Creating quick start guide...")
        
        guide_content = f'''# Quick Start Guide
{self.dataset_info['name']} - {self.dataset_info['phase']}

## 🚀 Getting Started

### 1. Navigate to Analysis Directory
```bash
cd dataset/04_analysis_scripts/
```

### 2. Run Complete Analysis
```bash
python data_analysis.py
```

### 3. Generate Advanced Visualizations
```bash
python advanced_visualization.py
```

### 4. View Results
- Check generated PNG/PDF files for plots
- Open HTML files in browser for interactive visualizations
- Review CSV files for processed data

## 📊 Key Files
- `combined_dataset.csv`: Complete processed dataset
- `comprehensive_analysis.png`: Main analysis plots
- `interactive_dashboard.html`: Interactive visualizations

## 🔧 Troubleshooting
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: Python 3.8+ required
- Verify data files are present in parent directories

## 📚 Documentation
- See `README.md` for complete documentation
- Check `METADATA.json` for technical specifications
- Review `INSTALLATION_GUIDE.md` for detailed setup

Generated on: {self.get_timestamp()}
Dataset Version: {self.dataset_info['version']}
'''
        
        guide_file = self.dataset_path / 'QUICK_START.md'
        with open(guide_file, 'w') as f:
            f.write(guide_content)
        
        print(f"   ✅ Quick start guide created: {guide_file}")
        print()
    
    def get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def print_completion_summary(self):
        """Print completion summary"""
        print("🎉 Dataset Setup Complete!")
        print("=" * 50)
        print()
        print("📁 Dataset Structure:")
        print("   dataset/")
        print("   ├── 01_constituent_materials/  (4 files)")
        print("   ├── 02_mix_designs/           (1 file)")
        print("   ├── 03_fresh_properties/      (1 file)")
        print("   ├── 04_analysis_scripts/      (3+ files)")
        print("   ├── README.md")
        print("   ├── METADATA.json")
        print("   └── QUICK_START.md")
        print()
        print("🚀 Next Steps:")
        print("   1. cd dataset/04_analysis_scripts/")
        print("   2. python data_analysis.py")
        print("   3. python advanced_visualization.py")
        print()
        print("📚 Documentation:")
        print("   - README.md: Complete overview")
        print("   - INSTALLATION_GUIDE.md: Detailed setup")
        print("   - QUICK_START.md: Getting started")
        print()
        print("✨ Happy analyzing!")
    
    def run_complete_setup(self):
        """Run the complete setup process"""
        self.print_header()
        
        # Step 1: Check requirements
        if not self.check_system_requirements():
            print("❌ System requirements not met. Please fix issues and try again.")
            return False
        
        # Step 2: Create directories
        self.create_directory_structure()
        
        # Step 3: Download/verify data
        self.download_sample_data()
        
        # Step 4: Verify integrity
        if not self.verify_data_integrity():
            print("❌ Data integrity check failed. Please check your dataset files.")
            return False
        
        # Step 5: Install dependencies
        if not self.install_dependencies():
            print("❌ Failed to install dependencies. Please install manually.")
            return False
        
        # Step 6: Test analysis
        if not self.run_initial_analysis():
            print("❌ Initial analysis test failed. Please check your setup.")
            return False
        
        # Step 7: Create guide
        self.create_quick_start_guide()
        
        # Step 8: Summary
        self.print_completion_summary()
        
        return True


def main():
    """Main function"""
    downloader = DatasetDownloader()
    
    try:
        success = downloader.run_complete_setup()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\\n\\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n\\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()