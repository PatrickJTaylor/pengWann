# pengWann - Descriptors of chemical bonding from Wannier functions

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docs](https://readthedocs.org/projects/pengwann/badge/?version=latest)](https://pengwann.readthedocs.io/en/latest/)
[![PyPI version](https://badge.fury.io/py/pengwann.svg)](https://badge.fury.io/py/pengwann)
[![Test coverage](https://api.codeclimate.com/v1/badges/10626c706c7877d2af47/test_coverage)](https://codeclimate.com/github/PatrickJTaylor/pengWann/test_coverage)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`pengwann` is a lightweight Python package for calculating well-established descriptors of chemical bonding from Wannier functions (as obtained from [Wannier90](https://wannier.org/)).

More specifically, `pengwann` can be used to calculate the WOHP (Wannier Orbital Hamilton Population) and/or the WOBI (Wannier Orbital Bond index) between a pair (or larger cluster) of atoms. These quantities are analogous to the pCOHP (projected Crystal Orbital Hamilton Population) and pCOBI (projected Crystal Orbital Bond Index) implemented in the [LOBSTER](http://www.cohp.de/) code, except that the local basis we choose to represent the Hamiltonian and the density matrix is comprised of Wannier functions rather than pre-defined atomic or pseudo-atomic orbitals. This choice of a Wannier basis has the advantage that (for energetically isolated bands) **the spilling factor is strictly 0**. For further details as to the advantages and disadvantages of using a Wannier basis, as well as the mathematical formalism behind `pengwann` in general, please see the [Methodology](./methodology) page.

Besides the [API reference](./api), a detailed use case of how `pengwann` can be used to derive WOHPs, WOBIs and the pDOS can be found on the [Examples](./examples) page.

```{toctree}
---
hidden: True
---

methodology
examples
api
```
