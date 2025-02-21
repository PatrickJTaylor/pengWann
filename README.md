![The pengWann logo: a purple penguin.](https://github.com/PatrickJTaylor/pengWann/raw/main/docs/_static/logo.png)

(technically pronounced *peng-van*, but some pronounce `numpy` as *num-pee* rather than *num-pie*, so who really knows?)

[![status](https://joss.theoj.org/papers/eeaf01be0609655666b459cc816a146b/status.svg)](https://joss.theoj.org/papers/eeaf01be0609655666b459cc816a146b)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docs](https://readthedocs.org/projects/pengwann/badge/?version=latest)](https://pengwann.readthedocs.io/en/latest/)
[![Test coverage](https://api.codeclimate.com/v1/badges/10626c706c7877d2af47/test_coverage)](https://codeclimate.com/github/PatrickJTaylor/pengWann/test_coverage)
[![Requires Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![PyPI version](https://badge.fury.io/py/pengwann.svg)](https://badge.fury.io/py/pengwann)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

`pengwann` is a lightweight Python package for computing descriptors of chemical bonding and local electronic structure from Wannier functions (as output by [Wannier90](https://wannier.org/)).
Alternatively phrased: `pengwann` replicates the core functionality of [LOBSTER](http://www.cohp.de/), except that the local basis used to represent the Hamiltonian and the density matrix is comprised of Wannier functions rather than pre-defined atomic or pseudo-atomic orbitals. 
The primary advantage of this methodology is that (for energetically isolated bands) **the spilling factor is strictly 0**.

The core features of `pengwann` include:

- Identification of interatomic and on-site interactions in terms of the Wannier functions associated with each atom
- Parsing of Wannier90 output files
- Parallelised computation of the following descriptors:
  - The Wannier orbital Hamilton population (WOHP)
  - The Wannier orbital bond index (WOBI)
  - The Wannier-projected density of states (pDOS)
  - Orbital and k-resolved implementations of all of the above
- Integration of descriptors to derive:
  - Löwdin-style populations and charges
  - Measures of bond strength and bond order

For further details regarding functionality and methodology, please see the [documentation](https://pengwann.readthedocs.io/).
If something is still unclear after having browsed the docs, then feel free to open a [discussion](https://github.com/PatrickJTaylor/pengWann/discussions) and we will endeavour to get back to you as soon as possible.

## Installation :penguin:

The latest tagged release of `pengwann` is `pip`-installable as:

```
pip install pengwann
```

Alternatively, to install the current development build:

```
pip install git+https://github.com/PatrickJTaylor/pengwann.git
```

## Bugs and development :hammer_and_wrench:

If you think you have encountered a bug whilst using `pengwann`, please create an [issue](https://github.com/PatrickJTaylor/pengWann/issues) and let us know!
Contributions to `pengwann` via [pull requests](https://github.com/PatrickJTaylor/pengWann/pulls) are also very welcome (see the [contributions guide](https://github.com/PatrickJTaylor/pengWann/blob/main/docs/CONTRIBUTING.md) for more details).
