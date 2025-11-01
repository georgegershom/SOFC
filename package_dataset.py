#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOFC???????
???????tar.gz??????
"""

import os
import tarfile
import shutil
from datetime import datetime

def create_dataset_package(dataset_dir='sofc_dataset', output_name='sofc_material_dataset'):
    """????????"""
    print("=" * 80)
    print("SOFC???????")
    print("=" * 80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tar_filename = f"{output_name}_{timestamp}.tar.gz"
    
    print(f"\n???????...")
    print(f"  ???: {dataset_dir}")
    print(f"  ????: {tar_filename}")
    
    # ?????
    with tarfile.open(tar_filename, 'w:gz') as tar:
        tar.add(dataset_dir, arcname=os.path.basename(dataset_dir))
        
        # ??????
        if os.path.exists('generate_sofc_dataset.py'):
            tar.add('generate_sofc_dataset.py', arcname='generate_sofc_dataset.py')
        
        if os.path.exists('validate_dataset.py'):
            tar.add('validate_dataset.py', arcname='validate_dataset.py')
    
    # ??????
    file_size = os.path.getsize(tar_filename)
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"\n? ???????!")
    print(f"  ??: {tar_filename}")
    print(f"  ??: {file_size_mb:.2f} MB")
    print(f"\n?????:")
    
    # ????
    with tarfile.open(tar_filename, 'r:gz') as tar:
        file_list = tar.getnames()
        print(f"  - ????: {len(file_list)}")
        print(f"  - JSON???: sofc_material_dataset.json")
        print(f"  - CSV????: csv_files/")
        print(f"  - ??: README.md, dataset_summary.txt")
    
    return tar_filename


if __name__ == '__main__':
    package_file = create_dataset_package()
    print(f"\n????????????: {package_file}")
