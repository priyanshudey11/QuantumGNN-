"""
Setup script to compile Cython extensions.

Usage:
    python setup_cython.py build_ext --inplace

This will compile the collate_cython.pyx file into a C extension.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "ligand_pocket_qgnn.collate_cython",
        ["ligand_pocket_qgnn/collate_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],  # Aggressive optimization
        language="c"
    )
]

setup(
    name="ligand_pocket_qgnn_cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
        }
    ),
    zip_safe=False,
)
