"""Standalone build for edt_lp module. Run from the LP/ directory."""
import os
import sys
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

extra_compile_args = ['-std=c++17', '-O3', '-pthread']
extra_link_args = ['-pthread']

if sys.platform == 'darwin':
    extra_compile_args += ['-Xpreprocessor', '-fopenmp', '-stdlib=libc++']
elif sys.platform == 'linux':
    extra_compile_args += ['-fopenmp', '-flto']
    extra_link_args += ['-fopenmp', '-flto']

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
