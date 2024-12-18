# Computing bonding descriptors from Wannier functions

`pengwann` is a Python package for calculating well-established descriptors of chemical bonding from Wannier functions as obtained from [Wannier90](https://wannier.org/).

More specifically, `pengwann` can be used to calculate WOHPs (Wannier Orbital Hamilton Population) and/or WOBIs (Wannier Orbital Bond Index) between pairs of atoms and their associated Wannier functions. Loosely speaking, integrating these functions up to the Fermi level provides a measure of bond strength (WOHP) or bond order (WOBI), allowing for the extraction of local bonding descriptors from periodic DFT calculations.

```{toctree}
---
hidden: True
---

methodology
api
```
