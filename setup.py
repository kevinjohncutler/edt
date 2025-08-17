import setuptools
import sys

class NumpyImport:
  def __repr__(self):
    import numpy as np

    return np.get_include()

  __fspath__ = __repr__

# NOTE: If edt.cpp does not exist:
# cython -3 --fast-fail -v --cplus edt.pyx

extra_compile_args = []
if sys.platform == 'win32':
  extra_compile_args += [
    '/std:c++17', '/O2'
  ]
else:
  extra_compile_args += [
    '-std=c++17', '-Ofast', '-ffast-math', '-fno-math-errno', '-fno-trapping-math',
    '-march=native', '-mtune=native', '-flto', '-DNDEBUG', '-pthread'
  ]

if sys.platform == 'darwin':
  extra_compile_args += [ '-stdlib=libc++', '-mmacosx-version-min=10.9' ]

# Add extra_link_args for LTO if not Windows
extra_link_args = []
if sys.platform != 'win32':
  extra_link_args += ['-flto']


setuptools.setup(
  setup_requires=['pbr', 'cython'],
  python_requires=">=3.8,<4",
  ext_modules=[
    setuptools.Extension(
      'edt',
      sources=[ 'src/edt.pyx' ],
      language='c++',
      include_dirs=[ 'src', str(NumpyImport()) ],
      extra_compile_args=extra_compile_args,
      extra_link_args=extra_link_args,
    ),
  ],
  long_description_content_type='text/markdown',
  pbr=True
)