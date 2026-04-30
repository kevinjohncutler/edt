"""Standalone build for edt_lp module. Run from the LP/ directory."""
import os
import sys
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

extra_compile_args = ['-std=c++17', '-O3', '-pthread', '-march=native', '-funroll-loops']
extra_link_args = ['-pthread']

if sys.platform == 'darwin':
    extra_compile_args += ['-Xpreprocessor', '-fopenmp', '-stdlib=libc++']
elif sys.platform == 'linux':
    # Note: -flto disabled — measured >1.5x slowdown at AMD Zen4 T=32
    # 4096^2 with template-heavy L1 path (likely cross-TU inlining
    # hurts hot-loop scheduling). -O3 + -march=native is sufficient.
    extra_compile_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']

ext = Extension(
    'edt_lp',
    sources=['edt_lp.pyx'],
    language='c++',
    include_dirs=['.', np.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    name='edt_lp',
    ext_modules=cythonize([ext], language_level=3),
)
