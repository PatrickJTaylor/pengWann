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

Most first-principles quantum chemistry calculations characterise periodic systems with an electronic structure: a series of eigenvectors and eigenvalues calculated by diagonalising a large matrix (often referred to as some kind of Hamiltonian).
These eigenvectors, often referred to as Bloch states or molecular orbitals, are in general delocalised across the entire structure and so are difficult to reason about in terms of local concepts such as chemical bonds.
As a result, it is common practice to project these extended Bloch states onto a set of localised basis functions, from which various descriptors of chemical bonding and local electronic structure can then be derived.
`pengwann` is a Python package for calculating some of these descriptors by projecting Bloch states onto Wannier functions: a highly optimised local basis that, for energetically isolated bands, spans the same Hilbert space as the canonical Bloch states.
The Wannier functions themselves can be easily passed to `pengwann` via a simple interface to the popular Wannier90 code [@Wannier90], with which many researchers in the field will already be familiar.

# Statement of need

The technique of deriving bonding descriptors from the projection of Bloch states onto a local basis is widely used in the materials modelling community.
Key to the success of this method is the choice of local basis: it should be the case that appropriately combined local basis functions can effectively reproduce the canonical Bloch states.
The ability of a given basis set to accurately represent the original Bloch states is characterised by the spilling factor

$$S = \frac{1}{N_{b}}\frac{1}{N_{k}}\sum_{nk}1 - \sum_{\alpha}|\langle\psi_{nk}|\phi_{\alpha}\rangle|^{2},$$

where $|\psi_{nk}\rangle$ is a Bloch state, $|\phi_{\alpha}\rangle$ is a localised basis function labelled by $\alpha$, $n$ labels bands, $k$ labels k-points, $N_{b}$ is the total number of bands and $N_{k}$ is the total number of k-points.
The spilling factor takes values between 0 and 1; if the local basis spans the same Hilbert space as the Bloch states, then $S = 0$, whilst $S = 1$ indicates that the two bases are orthogonal to one another.
The most common choice of local basis is to use some kind of atomic or pseudo-atomic orbitals, which are parameterised with respect to atomic species but not usually to the specific system at hand.
Due to the fact that these basis sets are designed to be transferable between most materials that one may wish to study, they will never be able to represent The Bloch states of an arbitrary system perfectly: the spilling factor will always be non-zero and some information will always be lost during the projection.

# References
