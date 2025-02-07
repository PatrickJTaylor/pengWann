# Methodology

## The pCOHP and the pCOBI

The projected crystal orbital Hamilton population (pCOHP) and the projected crystal orbital bond index (pCOBI) are both local descriptors of chemical bonding defined by the projection of the Kohn-Sham eigenstates (as calculated via a prior DFT calculation) onto a chosen set of localised basis functions {footcite:p}`pCOHP, pCOBI`

```{math}
\begin{align}
\mathrm{pCOHP}_{\alpha\beta}(E) &= H_{\alpha\beta}\sum_{nk}\mathrm{Re}\left[\left(C^{\alpha}_{nk}\right)^{*}C^{\beta}_{nk}\right]\cdot\delta(E - \epsilon_{nk}) \\
\mathrm{pCOBI}_{\alpha\beta}(E) &= P_{\alpha\beta}\sum_{nk}\mathrm{Re}\left[\left(C^{\alpha}_{nk}\right)^{*}C^{\beta}_{nk}\right]\cdot\delta(E - \epsilon_{nk}).
\end{align}
```

In both of the equations above, each {math}`C^{\alpha}_{nk}` is the coefficient resulting from the projection of the Kohn-Sham eigenstate {math}`\ket{\psi_{nk}}` onto the localised basis function {math}`\ket{\phi_{\alpha}}`. Together with {math}`\delta(E - \epsilon_{nk})`, which is the density of states arising from {math}`\ket{\psi_{nk}}`, everything within the summation in both equations is referred to as the DOS matrix arising from local basis states {math}`\ket{\phi_{\alpha}}` and {math}`\ket{\phi_{\beta}}`:

```{math}
D_{\alpha\beta}(E) = \sum_{nk}\mathrm{Re}\left[\left(C^{\alpha}_{nk}\right)^{*}C^{\beta}_{nk}\right]\cdot\delta(E - \epsilon_{nk}).
```

In the pCOHP, the sum over this DOS matrix is weighted by the corresponding element of the local basis Hamiltonian {math}`H_{\alpha\beta}`, whilst in the pCOBI it is weighted by {math}`P_{\alpha\beta}`, the relevant element of the local basis density matrix.

Loosely speaking, the pCOHP is thought to be correlated with **bond strength**, whilst the pCOBI is thought to be correlated with **bond order** (or rather, the contribution to these two metrics by a given pair of basis functions). This is most clearly expressed by their integrals up to the Fermi level,

```{math}
\begin{align}
\mathrm{ICOHP}_{\alpha\beta} &= \int^{E_{\mathrm{F}}}_{-\infty} \mathrm{d}E\,\mathrm{COHP}_{\alpha\beta}(E) \\
\mathrm{ICOBI}_{\alpha\beta} &= \int^{E_{\mathrm{F}}}_{-\infty} \mathrm{d}E\,\mathrm{COBI}_{\alpha\beta}(E),
\end{align}
```

which have often been used as a quantitative measure of these two characteristics.

## The spilling factor

Perhaps the most serious problem with calculating pCOHPs and pCOBIs in the manner detailed above is that the one requires a suitable set of localised basis functions {math}`\ket{\phi_{\alpha}}` that, when appropriately combined, make for a sufficiently accurate representation of the original Kohn-Sham eigenstates. If the local basis is not sufficiently representative of the canonical Bloch states, then there will be a net loss of information after the projection and the resulting pCOHPs and pCOBIs will be correspondingly less accurate. This potential loss of information is quantified by the **spilling factor** {footcite:p}`spilling`

```{math}
S = \frac{1}{N_{k}}\frac{1}{N_{b}}\sum_{nk} 1 - \sum_{\alpha}|\braket{\psi_{nk}|\phi_{\alpha}}|^{2},
```

which takes value between 0 and 1. If the local basis spans the same Hilbert space as the Kohn-Sham states, then {math}`S = 0`, whilst {math}`S = 1` indicates that the two bases are orthogonal to one another.

## Wannier functions

Very briefly, Wannier functions are a localised basis obtained from a set of Kohn-Sham eigenstates via a unitary transformation {footcite:p}`original_Wannier`. For a set of {math}`J` energetically isolated bands (i.e., a manifold of bands separated from all other bands by an energy gap everywhere in the Brillouin zone), this can be written as

```{math}
\begin{align}
\ket{w_{\alpha}} &= \sum_{k}\exp[-ik\cdot R]\sum^{J}_{m} U^{k}_{m\alpha}\ket{\psi_{mk}} \\
\ket{w_{\alpha}} &= \ket{w_{iR}},
\end{align}
```

where {math}`i` is a band-like index (note that within the exponential term, {math}`i = \sqrt{-1}`), {math}`R` is a Bravais lattice vector specifying the unit cell in which {math}`\ket{w_{iR}}` resides ({math}`\alpha` combines these two indices) and each {math}`U^{k}` is a unitary matrix. Owing to the fact that the Wannier functions are obtained via unitary transformations, the trace over any one-particle operator is the same whether one chooses to use the Bloch basis or the Wannier basis. The spilling factor can therefore be written as

```{math}
S = \frac{1}{N_{k}}\frac{1}{N_{b}}\sum_{nk} 1 - \sum_{m}|\braket{\psi_{nk}|\psi_{mk}}|^{2},
```

which due to the orthonormality of the Kohn-Sham states is simply:

```{math}
S = \frac{1}{N_{k}}\frac{1}{N_{b}}\sum_{nk} 1 - \sum_{m}\delta_{mn} = 0.
```

In the context of chemical bonding descriptors, Wannier functions therefore represent an ideal localised basis for the computation of the pCOHP and pCOBI or alternatively the WOHP {footcite:p}`WOHP` and WOBI (W for Wannier):

```{math}
\begin{align}
\mathrm{WOHP}^{R}_{\alpha\beta}(E) &= -H_{\alpha\beta}^{R}\sum_{nk}\mathrm{Re}\left[\left(C^{\alpha}_{nk}\right)^{*}C^{\beta}_{nk}\right]\cdot\delta(E - \epsilon_{nk}) \\
\mathrm{WOBI}^{R}_{\alpha\beta}(E) &= P_{\alpha\beta}^{R}\sum_{nk}\mathrm{Re}\left[\left(C^{\alpha}_{nk}\right)^{*}C^{\beta}_{nk}\right]\cdot\delta(E - \epsilon_{nk}).
\end{align}
```

:::{note}
We define the WOHP with the opposite sign to the pCOHP so as to ensure that a positive WOHP indicates bonding and a negative WOHP indicates antibonding (thus matching the WOBI in this respect).
:::

where {math}`R = R_{2} - R_{1}` is an extra index accounting for the fact that one could techincally compute the WOHP or WOBI between Wannier functions located in a variety of different unit cells. The coefficients used to build the DOS matrix are easily obtained from the unitary matrices used to define the Wannier functions:

```{math}
C^{\alpha}_{nk} = \exp[ik \cdot R]\left(U^{k}_{n\alpha}\right)^{*}.
```

## The pCOOP and the WOOP/pDOS

As well as the pCOHP and the pCOBI, there is also a third descriptor that has traditionally been used to assess the bonding/antibonding character of various interactions: the projected crystal orbital overlap population or pCOOP:

```{math}
\mathrm{pCOOP}_{\alpha\beta}(E) = S_{\alpha\beta}D_{\alpha\beta}(E) = S_{\alpha\beta}\sum_{nk}\mathrm{Re}\left[\left(C^{\alpha}_{nk}\right)^{*}C^{\beta}_{nk}\right]\cdot\delta(E - \epsilon_{nk}),
```

where {math}`S` is the overlap matrix. If we use Wannier functions as our local basis, then {math}`S = I` and the WOOP (Wannier orbital overlap population) is non-zero only for on-site interactions {footcite:p}`WOHP`:

```{math}
\mathrm{WOOP}_{\alpha\alpha}(E) = D_{\alpha\alpha}(E) = \mathrm{pDOS}_{\alpha}(E),
```

in which case we can also think of it as the density of states projected onto a specific Wannier function {math}`\ket{w_{\alpha}}`.

## Caveats

Whilst Wannier functions derived from energetically isolated bands are guaranteed to span the same Hilbert space, this does not mean that individual WOHPs and WOBIs are uniquely defined. In general, Wannier functions are strongly non-unique: so long as each {math}`U^{k}` remains unitary, the spilling factor is strictly 0, but the resulting Wannier functions may have wildly different centres, spreads and shapes. In addition, Wannier bases derived from groups of entangled bands (those that are not separated by an energy gap from all other bands everywhere in the Brillouin zone) are no longer guaranteed to span exactly the same space as the original Kohn-Sham states, thus potentially suffering from the same problems as pre-defined basis sets of atomic or pseudo-atomic orbitals.

There exists well-estalished methods for mitigating both of the concerns raised above. The non-uniqeuness of Wannier functions can be circumvented by minimising their spread with respect to the unitary matrices {math}`U^{k}`, thus producing so-called "Maximally-Localised Wannier Functions" or MLWFs {footcite:p}`MLWF, MLWF_dis`. Such Wannier functions tend to be atom-centred and to take on shapes that in many cases match our chemical intution for the system at hand. Maximal localisation can also be applpied to a manifold of entangled bands, although a separate "disentanglement" step is required in this case, unless one chooses to utilise the SCDM method {footcite:p}`SCDM, SCDM_k`.

In general, obtaining "good" MLWFs for non-trivial systems is not always easy, although developments in recent years are slowly making this a more reliable and automatable process. As a result, the two most obvious use cases for `pengwann` are as follows:

1. You have **already obtained** MLWFs for your system (for some other purpose) and would like to characterise the local bonding.
2. You have attempted to characterise the bonding via another tool such as [LOBSTER](http://www.cohp.de/), but you find that the results are **not satisfactory**.

```{footbibliography}
```
