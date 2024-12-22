# pengWann - Descriptors of chemical bonding from Wannier functions

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docs](https://readthedocs.org/projects/pengwann/badge/?version=latest)](https://pengwann.readthedocs.io/en/latest/)
[![Test coverage](https://api.codeclimate.com/v1/badges/10626c706c7877d2af47/test_coverage)](https://codeclimate.com/github/PatrickJTaylor/pengWann/test_coverage)
[![PyPI version](https://badge.fury.io/py/pengwann.svg)](https://badge.fury.io/py/pengwann)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`pengwann` is a lightweight Python package for computing common descriptors of chemical bonding from Wannier functions (as output by [Wannier90](https://wannier.org/)). Alternatively phrased: `pengwann` replicates the core functionality of [LOBSTER](http://www.cohp.de/), except that the local basis used to represent the Hamiltonian and the density matrix is comprised of Wannier functions rather than pre-defined atomic or pseudo-atomic orbitals. The primary advantage of this methodology is that (for energetically isolated bands) the spilling factor is strictly 0.

## Installation

The latest tagged release of `pengwann` is pip-installable as:

```
pip install pengwann
```

Alternatively, to install the current development build:

```
pip install git+https://github.com/PatrickJTaylor/pengwann.git
```
