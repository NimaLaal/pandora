""" Setup file for pandora. """

from pathlib import Path
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
setup(
    name='pandora',
    version='1.00',
    description=('Software to speedup PTA likelihood calculation'),
    author='Nima Laal',
    author_email='nima.laal@gmail.com',
    packages=['pandora'],
    zip_safe=False,
    install_requires=requirements,
    include_dirs=[np.get_include()],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
