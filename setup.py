"""
Setup script for SOFC Digital Twin Dataset Generator.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sofc-dataset-generator",
    version="1.0.0",
    author="SOFC Research Team",
    author_email="research@sofc-dataset.com",
    description="Adaptive-Scale Physics-Informed Digital Twin Dataset Generator for SOFC Systems",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/sofc-research/dataset-generator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "gpu": [
            "cupy>=10.0.0",
        ],
        "advanced": [
            "numba>=0.56.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sofc-generate=sofc_dataset_generator.examples.generate_complete_dataset:main",
        ],
    },
    include_package_data=True,
    package_data={
        "sofc_dataset_generator": [
            "configs/*.yaml",
            "examples/*.py",
            "visualization/*.py",
        ],
    },
    keywords=[
        "SOFC", "solid oxide fuel cell", "digital twin", "physics-informed",
        "machine learning", "multi-physics", "simulation", "dataset generation",
        "electrochemical", "thermal", "structural", "degradation modeling"
    ],
    project_urls={
        "Bug Reports": "https://github.com/sofc-research/dataset-generator/issues",
        "Source": "https://github.com/sofc-research/dataset-generator",
        "Documentation": "https://sofc-dataset-generator.readthedocs.io/",
    },
)