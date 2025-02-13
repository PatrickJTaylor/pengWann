---
title: 'pengWann: Descriptors of chemical bonding from Wannier functions'
tags:
  - Python
  - chemical bonding
  - Wannier functions
authors:
  - name: Patrick J. Taylor
    affiliation: "1, 2"
    orcid: 0009-0003-6511-6442
  - name: Benjamin J. Morgan
    affiliation: "1, 2"
    orcid: 0000-0002-3056-8233
affiliations:
  - name: Department of Chemistry, University of Bath, Claverton Down, Bath, BA2 7AY, United Kingdom
    index: 1
  - name: The Faraday Institution, Quad One, Harwell Science and Innovation Campus, Didcot, OX11 0RA, United Kingdom
    index: 2
date: 10 February 2025
bibliography: paper.bib
---

# Summary

Most first-principles quantum chemistry calculations characterise periodic systems with an electronic structure: a series of eigenvectors and eigenvalues calculated by diagonalising some kind of Hamiltonian.
These eigenvectors, often referred to as Bloch states or molecular orbitals, are in general delocalised across the entire structure and so are difficult to reason about in terms of chemically intuitive local concepts such as bonds.
As a result, it is common practice to project these extended Bloch states onto a set of localised basis functions, from which various descriptors of chemical bonding and local electronic structure can then be derived.
`pengwann` is a Python package for calculating some of these descriptors by projecting Bloch states onto Wannier functions [@wannier_population_analysis]: a highly optimised local basis that, when derived from energetically isolated bands, spans the same Hilbert space as the canonical Bloch states.
The Wannier functions themselves can easily be passed to `pengwann` via a simple interface to the popular Wannier90 code [@wannier90], with which many researchers in the field will already be familiar.

# Statement of need

The technique of deriving bonding descriptors from the projection of Bloch states onto a local basis is widely used in the materials modelling community [@cohp_1;@cohp_2;@cohp_3;@cohp_4;@cohp_5;@cohp_6].
Key to the success of this method is the choice of local basis functions, which should be able to effectively reproduce the canonical Bloch states when appropriately combined.
The ability of a given basis set to accurately represent the original Bloch states is quantified by the spilling factor [@spilling_factor]

$$S = \frac{1}{N_{b}}\frac{1}{N_{k}}\sum_{nk}1 - \sum_{\alpha}|\langle\psi_{nk}|\phi_{\alpha}\rangle|^{2},$$

where $|\psi_{nk}\rangle$ is a Bloch state, $|\phi_{\alpha}\rangle$ is a localised basis function, $n$ labels bands, $k$ labels k-points, $N_{b}$ is the total number of bands and $N_{k}$ is the total number of k-points.
The spilling factor takes values between 0 and 1; if the local basis spans the same Hilbert space as the Bloch states, then $S = 0$, whilst $S = 1$ indicates that the two bases are orthogonal to one another.
The most common choice of local basis is to use some kind of atomic or pseudo-atomic orbitals [@bunge_basis;@koga_basis;@lobster_2016;@crystal_cohp], which are parameterised with respect to atomic species but not usually to the specific system at hand.
Due to the fact that these basis sets are designed to be transferable between most materials that one may wish to study, they will never be able to represent the Bloch states of an arbitrary system perfectly: the spilling factor will always be non-zero and some information will always be lost during the projection.
For many systems, the error introduced by this loss of information is relatively small and so can be safely ignored, but this is not always the case.
To give a pathological example, in electride materials, atom-centred basis functions cannot accurately represent the Bloch states because some of the valence electrons behave like anions and occupy their own distinct space in the structure [@electrides].

As mentioned previously, `pengwann` uses a Wannier basis, which when derived from energetically isolated bands, spans the same vector space as the canonical Bloch states.
The spilling factor is therefore strictly zero and there is no loss of information in switching from the Bloch basis to the Wannier basis.
With regards to Wannier functions derived from bands that are not energetically isolated everywhere in the Brillouin zone, the spilling factor will no longer be strictly zero, but we might still expect it to be very small, owing to the fact that Wannier functions are themselves calculated by a unitary transformation of the Bloch states [@wannier_review].
In addition, Wannier functions are not inherently atom-centred (even if in most systems they would appear to be so) and are therefore capable of accurately representing the Bloch states of electrides and other such anomalous systems.
More generally, even in systems where basis sets of pre-defined atomic or pseudo-atomic orbitals perform very well, a Wannier basis will always be capable of reducing the spilling factor and therefore reducing the corresponding error in all derived descriptors.

What follows is a list of the core features implemented in `pengwann`:

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

# Related software

The LOBSTER code [@lobster_2016;@lobster_2020] implements much of the same functionality as `pengwann` using basis sets of pre-defined atomic and pseudo-atomic orbitals. It also offers some additional features that `pengwann` does not support directly such as generating fatband plots and obtaining localised molecular orbitals via a transformation of the projected atomic orbital basis [@lobster_fragment]. We anticipate that the vast majority of users who may be interested in using `pengwann` will already be familiar with LOBSTER, so we will now briefly discuss the advantages and disadvantages of using one code over the other. The advantages of using `pengwann` over LOBSTER are that the spilling factor will in general be lower and that systems requiring the use of non-atom-centred basis functions can be routinely studied. There is also the fact that Wannier functions have many uses besides the computation of bonding descriptors, so there are feasibly situations in which they can be plugged into `pengwann` effectively "for free", having already been calculated for some other purpose. On the other hand, one substantial advantage that LOBSTER has over `pengwann` is that it provides pre-defined atomic and pseudo-atomic basis sets that can be applied to any system with minimal user input, whereas obtaining high-quality Wannier functions for a given system can be quite a complex process in its own right due to their strong non-uniqueness [@wannier_review]. That being said, advances in recent years have been the computation of well-localised Wannier functions with appropriate symmetry for chemical bonding analysis significantly easier [@ht_scdm;@pwdf].

The WOBSTER code [@wobster] implements a subset of the features found in `pengwann`, allowing users to compute the Wannier-projected density of states and the Wannier orbital Hamilton population. Aside from the features that `pengwann` implements that WOBSTER does not, the other main difference between the two codes is performance: `pengwann` makes heavy use of `numpy` [@numpy] and the built-in `multiprocessing` library to vectorise and parallelise most operations, whereas WOBSTER primarily uses loops and is correspondingly much slower.

# References
