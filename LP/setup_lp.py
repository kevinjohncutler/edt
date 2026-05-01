"""Standalone build for edt_lp module. Run from the LP/ directory."""
import os
import sys
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

if sys.platform == 'win32':
    # MSVC flags. Matches ncolor cpp_proto's setup.py exactly:
    # /GL (whole-program opt) and /openmp omitted — both measured to hurt
    # 3D strided perf vs cpp_proto, likely because /GL defers template-
    # instantiation optimization to link time where loop strength reduction
    # doesn't get applied.
    extra_compile_args = ['/std:c++17', '/O2', '/EHsc', '/arch:AVX2']
    extra_link_args = []
else:
    extra_compile_args = ['-std=c++17', '-O3', '-pthread', '-march=native', '-funroll-loops']
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
