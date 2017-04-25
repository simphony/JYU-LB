import os
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

VERSION = '0.2.0'

def write_version_py(filename=None):
    if filename is None:
        filename = os.path.join(
            os.path.dirname(__file__), 'jyulb', 'version.py')
    ver = """\
version = '%s'
"""
    fh = open(filename, 'wb')
    try:
        fh.write(ver % VERSION)
    finally:
        fh.close()

write_version_py()

extensions = [
    Extension(
        name='jyulb.defs',                     # name
        sources=['jyulb/defs.pyx'],            # source files
        include_dirs=['.',                     # include paths
                      numpy.get_include()], 
        extra_compile_args=['-fopenmp','-O3'], # compiler flags
        extra_link_args=['-fopenmp','-O3'],    # linker flags
        language="c++"                         # generate and compile C++ code
    ),
    Extension(
        name='jyulb.bcs',                      # name
        sources=['jyulb/bcs.pyx'],             # source files 
        include_dirs=['.',                     # include paths
                      numpy.get_include()], 
        extra_compile_args=['-fopenmp','-O3'], # compiler flags
        extra_link_args=['-fopenmp','-O3'],    # linker flags
        language="c++"                         # generate and compile C++ code
    ),
    Extension(
        name='jyulb.D3Q19'  ,                  # name
        sources=['jyulb/D3Q19.pyx'],           # source files 
        include_dirs=['.',                     # include paths
                      numpy.get_include()], 
        extra_compile_args=['-fopenmp','-O3'], # compiler flags
        extra_link_args=['-fopenmp','-O3'],    # linker flags
        language="c++"                         # generate and compile C++ code
    ),
    Extension(
        name='jyulb.isothermal',               # name
        sources=['jyulb/isothermal.pyx'],      # source files 
        include_dirs=['.',                     # include paths
                      numpy.get_include()], 
        extra_compile_args=['-fopenmp','-O3'], # compiler flags
        extra_link_args=['-fopenmp','-O3'],    # linker flags
        language="c++"                         # generate and compile C++ code
    ),
    Extension(
        name='jyulb.flow_field',               # name
        sources=['jyulb/flow_field.pyx'],      # source files 
        include_dirs=['.',                     # include paths
                      numpy.get_include()], 
        extra_compile_args=['-fopenmp','-O3'], # compiler flags
        extra_link_args=['-fopenmp','-O3'],    # linker flags
        language="c++"                         # generate and compile C++ code
    )
]                       

setup(
    name='jyulb',
    version=VERSION,
    author='Keijo Mattila, JYU',
    description='Implementation of a basic lattice-Boltzmann solver',
    packages=find_packages(),
    install_requires=['numpy>=1.9.1','cython>=0.25.2'],
    ext_modules=cythonize(extensions)
)
