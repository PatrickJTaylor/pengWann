from pengwann.version import __version__ as VERSION
from setuptools import find_packages, setup

long_description = open('README.md').read()

setup(
    name='pengwann',
    version=VERSION,
    description='A lightweight Python package for calculating bonding descriptors from Wannier functions.',
    long_description=long_description,
    keywords=['wannier'],
    author='Patrick J. Taylor',
    author_email='pjt35@bath.ac.uk',
    url='https://github.com/PatrickJTaylor/pengWann',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'pymatgen>=2022.0.3'
    ],
    python_requires='>=3.9'
)
