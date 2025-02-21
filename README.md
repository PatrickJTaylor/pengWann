![The pengWann logo: a purple penguin.](https://github.com/PatrickJTaylor/pengWann/raw/main/docs/_static/logo.svg)

(technically pronounced *peng-van*, but some pronounce `numpy` as *num-pee* rather than *num-pie*, so who really knows?)

[![status](https://joss.theoj.org/papers/eeaf01be0609655666b459cc816a146b/status.svg)](https://joss.theoj.org/papers/eeaf01be0609655666b459cc816a146b)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docs](https://readthedocs.org/projects/pengwann/badge/?version=latest)](https://pengwann.readthedocs.io/en/latest/)
[![Test coverage](https://api.codeclimate.com/v1/badges/10626c706c7877d2af47/test_coverage)](https://codeclimate.com/github/PatrickJTaylor/pengWann/test_coverage)
[![Requires Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![PyPI version](https://badge.fury.io/py/pengwann.svg)](https://badge.fury.io/py/pengwann)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A lightweight Python package for computing descriptors of chemical bonding and local electronic structure from Wannier functions.

<p align="center">
  <picture align="center">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/PatrickJTaylor/pengWann/raw/main/docs/_static/example_outputs.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/PatrickJTaylor/pengWann/raw/main/docs/_static/example_outputs_light.svg">
    <img alt="A handful of example outputs from pengWann as applied to rutile.", src="https://github.com/PatrickJTaylor/pengWann/raw/main/docs/_static/example_outputs.svg">
  </picture>
</p>

<p align="center">
  <small>
  A handful of example outputs from <code>pengwann</code> as applied to rutile.
  The colour-coded numbers next to the crystal structure are Löwdin-style charges computed for Ti (blue) and O (red).
  </small>
</p>

- Parse [Wannier90](https://wannier.org/) output files
- Identify interatomic and on-site interactions in terms of atom-assigned Wannier functions
- Compute in parallel:
  - The Wannier orbital Hamilton population (WOHP)
  - The Wannier orbital bond index (WOBI)
  - The Wannier-projected density of states (pDOS)
  - Orbital and k-resolved implementations of all of the above
- Integrate descriptors to derive:
  - Löwdin-style populations and charges
  - Measures of bond strength and bond order

`pengwann` replicates the core functionality of [LOBSTER](http://www.cohp.de/) but uses Wannier functions rather than pre-defined atomic or pseudo-atomic orbitals as a local basis in which to express the Hamiltonian and the density matrix.
A Wannier basis is advantageous in that, when derived from energetically isolated bands, [the spilling factor is strictly 0](https://pengwann.readthedocs.io/en/latest/methodology.html#the-spilling-factor).

For further details regarding detailed methodology, functionality and examples, see the [documentation](https://pengwann.readthedocs.io/).

## Getting started 🚀

### Installation 🐧

The latest tagged release of `pengwann` is `pip`-installable as:

```shell
pip install pengwann
```

Alternatively, to install the current development build:

```shell
pip install git+https://github.com/PatrickJTaylor/pengwann.git
```

### Basic usage 📝

For a quick run through of the basics, see the first example in the docs on [computing bonding descriptors in diamond](https://pengwann.readthedocs.io/en/latest/examples/diamond/basics.html).

## Support 🤝

### Getting help 👋

If you're having problems using `pengwann` and the [docs](https://pengwann.readthedocs.io/) do not provide a solution, feel free to open a [discussion](https://github.com/PatrickJTaylor/pengWann/discussions) and we will endeavour to get back to you as soon as possible.

### Bugs 🐛

If you think you have found a bug in `pengwann`, please create an [issue](https://github.com/PatrickJTaylor/pengWann/issues) and let us know!

### Contributing 🛠

Contributions to `pengwann` via [pull request](https://github.com/PatrickJTaylor/pengWann/pulls) are **very welcome**, whether the changes are big or small!
See the [contributing guide](https://github.com/PatrickJTaylor/pengWann/blob/main/docs/CONTRIBUTING.md) for more details.

## Acknowledgements 📣

The development of `pengwann` was initially inspired by [WOBSTER](https://github.com/Chengcheng-Xiao/WOBSTER), which in turn drew inspiration from previous work on using Wannier functions for population analysis[^1].

[^1]: S. Kundu et al., Population analysis with Wannier orbitals, In: *J. Chem. Phys.* 154 (10 2021), p. 104111
