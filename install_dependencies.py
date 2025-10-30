#!/usr/bin/env python3
"""
安装SOFC数据集生成器所需的依赖包
"""

import subprocess
import sys
import os

def install_package(package):
    """安装Python包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ 成功安装 {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 安装 {package} 失败: {e}")
        return False

def check_package(package):
    """检查包是否已安装"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def main():
    """主安装函数"""
    print("=== SOFC数据集生成器依赖安装 ===")
    
    # 必需的包列表
    required_packages = [
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "matplotlib>=3.5.0",
        "h5py>=3.1.0",
        "netCDF4>=1.5.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "pyDOE2>=1.3.0",
        "meshio>=5.3.0",
        "gmsh>=4.8.0",
        "pyvista>=0.32.0",
        "vtk>=9.0.0",
        "tqdm>=4.62.0",
        "joblib>=1.1.0"
    ]
    
    # 可选的高级包（用于更精确的FEA/CFD仿真）
    optional_packages = [
        "fenics",  # 有限元分析
        "firedrake",  # 高级有限元
        "openfoam",  # 计算流体动力学
    ]
    
    print("安装必需的包...")
    failed_packages = []
    
    for package in required_packages:
        package_name = package.split(">=")[0].split("==")[0]
        if not check_package(package_name):
            if not install_package(package):
                failed_packages.append(package)
        else:
            print(f"✓ {package_name} 已安装")
    
    print("\n尝试安装可选的高级包...")
    for package in optional_packages:
        if not check_package(package):
            print(f"尝试安装 {package}...")
            if not install_package(package):
                print(f"⚠ {package} 安装失败，将使用简化模型")
        else:
            print(f"✓ {package} 已安装")
    
    # 检查GMSH安装
    print("\n检查GMSH...")
    try:
        import gmsh
        print("✓ GMSH Python接口可用")
    except ImportError:
        print("⚠ GMSH Python接口不可用，将使用简化的网格生成")
    
    # 总结
    print("\n=== 安装总结 ===")
    if failed_packages:
        print(f"以下包安装失败: {failed_packages}")
        print("请手动安装这些包或使用简化版本")
    else:
        print("✓ 所有必需包安装成功")
    
    print("\n现在可以运行测试：")
    print("python quick_test.py")
    print("\n或运行完整数据集生成：")
    print("python sofc_dataset_generator.py")

if __name__ == "__main__":
    main()