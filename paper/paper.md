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

First-principles quantum chemistry calculations of periodic systems typically represent electronic structures through sets of eigenvectors and their corresponding eigenvalues, obtained by diagonalising some Hamiltonian.
These eigenvectors, known as Bloch states, molecular orbitals, or crystal orbitals, are generally delocalised across the entire structure, making them difficult to interpret in terms of chemically intuitive concepts such as bonds.
To address this, it is common practice to project these extended Bloch states onto a localised basis, enabling the calculation of various local descriptors of chemical bonding or electronic structure.
`pengwann` is a Python package for calculating some of these descriptors by projecting Bloch states onto Wannier functions [@wannier_population_analysis].
Wannier functions provide a highly optimised local basis and, when derived from energetically isolated bands, span the same Hilbert space as the canonical Bloch states.
The package provides a simple interface to the popular Wannier90 code [@wannier90], making it readily accessible to researchers already familiar with this widely-used tool.

# Statement of need

The technique of deriving bonding descriptors from the projection of Bloch states onto a local basis is widespread in materials modelling [@cohp_1;@cohp_2;@cohp_3;@cohp_4;@cohp_5;@cohp_6;@ColesEtAl_JMaterChemA2023;@LegeinEtAl_JAmChemSoc2024].
Key to the success of this method is the choice of local basis functions, which should be able to effectively reproduce the canonical Bloch states when appropriately combined.
The ability of a given basis set to accurately represent the original Bloch states is quantified by the spilling factor [@spilling_factor]

$$S = \frac{1}{N_{b}}\frac{1}{N_{k}}\sum_{nk}1 - \sum_{\alpha}|\langle\psi_{nk}|\phi_{\alpha}\rangle|^{2},$$

where $|\psi_{nk}\rangle$ is a Bloch state, $|\phi_{\alpha}\rangle$ is a localised basis function, $n$ labels bands, $k$ labels k-points, $N_{b}$ is the total number of bands and $N_{k}$ is the total number of k-points.
The spilling factor takes values between 0 and 1; if the local basis spans the same Hilbert space as the Bloch states, then $S = 0$, while $S = 1$ indicates that the two bases are orthogonal to one another.
The most common choice of local basis is to use atomic or pseudo-atomic orbitals [@bunge_basis;@koga_basis;@lobster_2016;@crystal_cohp], with these parameterised with respect to atomic species but usually not to the specific system being modelled.
Because these orbitals are designed to be transferable between materials, they cannot represent the Bloch states of an arbitrary system perfectly: the spilling factor will always be non-zero and some information will be lost during the projection.
For many systems, the error introduced by this loss of information is relatively small and so can be safely ignored, but this is not always the case.
To give a pathological example, in electride materials, atom-centred basis functions cannot accurately represent the Bloch states because some of the valence electrons behave like anions and occupy their own distinct space in the structure [@electrides].

`pengwann` employs a Wannier basis which, when derived from energetically isolated bands, spans the same vector space as the canonical Bloch states.
The spilling factor is therefore strictly zero and there is no loss of information in switching from the Bloch basis to the Wannier basis.
For Wannier functions derived from bands that are not energetically isolated everywhere in the Brillouin zone, the spilling factor will no longer be strictly zero, but should remain very small, since Wannier functions are calculated by a unitary transformation of the Bloch states [@wannier_review].
Importantly, Wannier functions are not constrained to be atom-centred and can therefore accurately represent the Bloch states of electrides and other such anomalous systems.
More generally, even in systems where pre-defined atomic or pseudo-atomic orbital basis sets perform well, a Wannier basis will always give a reduced spilling factor, thereby reducing the corresponding error in all derived descriptors.

`pengwann` implements the following core features:

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

The LOBSTER code [@lobster_2016;@lobster_2020] implements much of the same functionality as `pengwann` using basis sets of pre-defined atomic and pseudo-atomic orbitals. LOBSTER offers additional features not directly supported by `pengwann`, such as generating fatband plots and obtaining localised molecular orbitals via transformation of the projected atomic orbital basis [@lobster_fragment]. We anticipate that most potential users of `pengwann` will already be familiar with LOBSTER, warranting a brief discussion of the relative advantages and disadvantages of each code.

`pengwann` achieves lower spilling factors than LOBSTER[^1] and readily handles systems when non-atom-centered basis functions are preferable.
Moreover, since Wannier functions serve many purposes in addition to the calculation of bonding descriptors, they may already have been calculated for systems of interest, making the use of `pengwann` particularly efficient in these cases. Conversely, LOBSTER's main advantage lies in its pre-defined atomic and pseudo-atomic basis sets, which can be applied to any system with minimal user input, while obtaining high-quality Wannier functions can be complex due to their strong non-uniqueness [@wannier_review]. However, recent advances such as those of Refs. [@ht_scdm;@pwdf] have significantly simplified the computation of well-localised Wannier functions with appropriate symmetry for chemical bonding analysis, lowering the barrier to their use.

The WOBSTER code [@wobster] implements a subset of the features found in `pengwann`, allowing users to compute the Wannier-projected density of states and the Wannier orbital Hamilton population. `pengwann` provides a broader set of features than WOBSTER and is more performant: `pengwann` leverages `numpy` [@numpy] and the built-in `multiprocessing` library to vectorise and parallelise most operations, whereas WOBSTER's loop-based approach is considerably slower. To give a specific example: the Wannier orbital Hamilton population for a given pair of Wannier functions is computed in WOBSTER using a triple nested loop over energies, k-points and bands. The same calculation in `pengwann` is fully vectorised across these three dimensions and furthermore many such computations can be carried out in parallel for different pairs of Wannier functions.

# Acknowledgements

The authors thank the Faraday Institution CATMAT project (EP/S003053/1, FIRG016) for financial support and the Michael High-Performance Computing (HPC) facility (FIRG030).
P.J.T would like to acknowledge Chengcheng Xiao (author of the WOBSTER code) for inspiring the development of `pengwann` and thanks the University of Bath (indirectly via the EPSRC) for PhD funding.
B.J.M. thanks the Royal Society for a fellowship (URF/R/191006).

[^1]: For example, even for relatively simple systems such as diamond and rutile, we find that (when using the recommended basis functions) LOBSTER still yields non-zero spilling factors of $0.98\,\%$ and $0.95\,\%$ respectively. For the same set of Bloch states, `pengwann` is able to reduce the spilling factor to $< 1.0 \times 10^{-11}\,\%$ in both cases.

# References
