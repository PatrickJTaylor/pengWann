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

# References
