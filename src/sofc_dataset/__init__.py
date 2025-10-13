"""SOFC adaptive-scale dataset generator.

Modules:
- sampling: Latin Hypercube sampling and parameter space definitions
- physics: synthetic multi-physics field generation for Dataset 1
- experimental: synthetic experimental/validation dataset generation for Dataset 2
- realtime: synthetic real-time stream generation for Dataset 3
- io_utils: writers for NPZ/CSV and manifest metadata
- cli: command-line interface entrypoints
"""

__all__ = [
    "sampling",
    "physics",
    "experimental",
    "realtime",
    "io_utils",
]
