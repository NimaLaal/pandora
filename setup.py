""" Setup file for pandora. """

from pathlib import Path
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='pandora',
    version='1.00',
    description=('Software to speedup PTA likelihood calculation'),
    author='Nima Laal',
    author_email='nima.laal@gmail.com',
    packages=['pta-pandora'],
    zip_safe=False,
    install_requires=[
                    'cloudpickle>=3.1.1',
                    'corner>=2.2.2',
                    'Cython>=0.29.36',
                    'enterprise_extensions>=2.4.3',
                    'enterprise_pulsar>=3.3.4.dev4+g5ef5ff4',
                    'h5py>=3.9.0',
                    'jax>=0.4.19',
                    'matplotlib>=3.10.0',
                    'numpy>=2.2.3',
                    'ptmcmcsampler>=2.1.1',
                    'pyro_ppl>=1.8.6',
                    'scipy>=1.15.1',
                    'setuptools>=68.2.2',
                    'torch>=2.1.0',
                    'tqdm>=4.66.1'
    ],
    include_dirs=[np.get_include()],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
