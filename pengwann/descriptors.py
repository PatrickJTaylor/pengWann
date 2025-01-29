"""
Chemical bonding descriptors from Wannier functions.

This module contains a single class, the
:py:class:`~pengwann.descriptors.DescriptorCalculator`, which contains the core
functionality of :code:`pengwann`: computing various descriptors of chemical bonding
from Wannier functions as output by Wannier90.
"""

# Copyright (C) 2024-2025 Patrick J. Taylor

# This file is part of pengWann.
#
# pengWann is free software: you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# pengWann is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with pengWann.
# If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import warnings
import numpy as np
from collections.abc import Sequence
from dataclasses import replace
from multiprocessing import cpu_count, Pool
from multiprocessing.shared_memory import SharedMemory
from numpy.typing import NDArray
from pengwann.geometry import AtomicInteractionContainer, WannierInteraction
from pengwann.utils import allocate_shared_memory, get_spilling_factor
from pymatgen.core import Structure
from tqdm.auto import tqdm
from typing import Any


class DescriptorCalculator:
    r"""
    Compute descriptors of chemical bonding and local electronic structure.

    This class can be used to calculate:

    - The projected density of states (pDOS)
    - Wannier orbital Hamilton populations (WOHPs)
    - Wannier orbital bond indices (WOBIs)
    - The density of energy (DOE)
    - Bond-weighted distribution functions (BWDFs)

    Parameters
    ----------
    dos_array : ndarray of float
        The density of states discretised across energies, k-points and bands.
    num_wann : int
        The total number of Wannier functions.
    nspin : int
        The number of electrons per fully-occupied band. This should be set to 2 for
        non-spin-polarised calculations and set to 1 for spin-polarised calculations.
    kpoints : ndarray of float
        The full k-point mesh used in the prior Wannier90 calculation.
    u : ndarray of complex
        The U matrices that define the Wannier functions in terms of the canonical
        Bloch states.
    h : dict of {3-length tuple of int : ndarray of complex} pairs or None, optional
        The Hamiltonian in the Wannier basis. Required for the computation of WOHPs.
        Defaults to None.
    occupation_matrix : ndarray of float or None, optional
        The Kohn-Sham occupation matrix. Required for the computation of WOBIs.
        Defaults to None.
    energies : ndarray of float or None, optional
        The energies at which the `dos_array` has been evaluated. Defaults to None.

    Returns
    -------
    None

    Notes
    -----
    Upon initialisation, the spilling factor will be calculated, it is defined as
    :footcite:p:`spilling, WOHP`

    .. math::

        S = \frac{1}{N_{k}}\frac{1}{N_{w}}\sum_{nk} 1 - \sum_{\alpha}
        |\braket{\psi_{nk}|w_{\alpha}}|^{2},

    where :math:`N_{k}` is the total number of k-points, :math:`N_{w}` is the total
    number of Wannier functions, :math:`n` labels bands, :math:`k` labels k-points and
    :math:`\alpha` labels Wannier functions :math:`\ket{w_{\alpha}}`.

    For Wannier functions derived from energetically isolated bands, the spilling
    factor should be (within machine precision) strictly 0. For disentangled bands,
    the spilling factor should still ideally be very close to 0. If the calculated
    spilling factor is > 0, a warning will be printed to the console and all derived
    results should be treated with caution.

    This class should not normally be initialised using the base constructor. See
    instead the :py:meth:`~pengwann.descriptors.DescriptorCalculator.from_eigenvalues`
    classmethod.

    References
    ----------
    .. footbibliography::
    """

    _bl_0 = np.array((0, 0, 0))

    def __init__(
        self,
        dos_array: NDArray[np.float64],
        num_wann: int,
        nspin: int,
        kpoints: NDArray[np.float64],
        u: NDArray[np.complex128],
        h: dict[tuple[int, ...], NDArray[np.complex128]] | None = None,
        occupation_matrix: NDArray[np.float64] | None = None,
        energies: NDArray[np.float64] | None = None,
    ):
        self._dos_array = dos_array
        self._num_wann = num_wann
        self._nspin = nspin
        self._kpoints = kpoints
        self._u = u
        self._h = h
        self._occupation_matrix = occupation_matrix
        self._energies = energies

        if __name__ == "__main__":
            spilling_factor = get_spilling_factor(
                self._u, len(self._kpoints), self._num_wann
            )
            rounded_spilling_factor = abs(round(spilling_factor, ndigits=8))
            if rounded_spilling_factor > 0:
                warnings.warn(
                    f"""
                The spilling factor = {rounded_spilling_factor}.

                It is advisable to verify that the spilling factor is sufficiently low.
                For Wannier functions derived from energetically isolated bands, it
                should be (within machine precision) strictly 0. For Wannier functions
                derived using disentanglement, the spilling factor should still be very
                close to 0.

                If the spilling factor is significantly > 0, this implies that there are
                parts of the Bloch subspace that the Wannier basis does not span and
                thus any results derived from the Wannier basis should be analysed with
                caution.
                """
                )

    @classmethod
    def from_eigenvalues(
        cls,
        eigenvalues: NDArray[np.float64],
        num_wann: int,
        nspin: int,
        energy_range: tuple[float, float],
        resolution: float,
        sigma: float,
        kpoints: NDArray[np.float64],
        u: NDArray[np.complex128],
        h: dict[tuple[int, ...], NDArray[np.complex128]] | None = None,
        occupation_matrix: NDArray[np.float64] | None = None,
    ) -> DescriptorCalculator:
        """
        Initialise a DescriptorCalculator object from a set of Kohn-Sham eigenvalues.

        Parameters
        ----------
        eigenvalues : ndarray of float
            The Kohn-Sham eigenvalues.
        num_wann : int
            The total number of Wannier functions.
        nspin : int
            The number of electrons per fully-occupied band. This should be set to 2
            for non-spin-polarised calculations and set to 1 for spin-polarised
            calculations.
        energy_range : 2-length tuple of float
            The energy range over which the density of states is to be evaluated.
        resolution : float
            The desired energy resolution of the density of states.
        sigma : float
            The width of the Gaussian kernel used to smear the density of states (in eV).
        kpoints : ndarray of float
            The full k-point mesh used in the prior Wannier90 calculation.
        u : ndarray of complex
            The U matrices that define the Wannier functions in terms of the canonical
            Bloch states.
        h : dict of {3-length tuple of int : ndarray of complex} pairs or None, optional
            The Hamiltonian in the Wannier basis. Required for the computation of WOHPs.
            Defaults to None.
        occupation_matrix : ndarray of float or None, optional
            The Kohn-Sham occupation matrix. Required for the computation of WOBIs.
            Defaults to None.

        Returns
        -------
        descriptor_calculator : DescriptorCalculator
            The initialised DescriptorCalculator object.

        See Also
        --------
        pengwann.io.read : Parse Wannier90 output files.
        pengwann.utils.get_occupation_matrix
        """
        emin, emax = energy_range
        energies = np.arange(emin, emax + resolution, resolution, dtype=np.float64)

        x_mu = energies[:, np.newaxis, np.newaxis] - eigenvalues
        dos_array = (
            1
            / np.sqrt(np.pi * sigma)
            * np.exp(-(x_mu**2) / sigma)
            / eigenvalues.shape[1]
        )
        dos_array = np.swapaxes(dos_array, 1, 2)

        return cls(
            dos_array, num_wann, nspin, kpoints, u, h, occupation_matrix, energies
        )

    @property
    def energies(self) -> NDArray[np.float64] | None:
        """
        The discrete energies over which the DOS (and derived descriptors) has been evaluated.

        Returns
        -------
        energies : ndarray of float or None
            The energies over which the DOS (and all derived quantities such as WOHPs
            or WOBIs) has been evaluated. If these energies were not provided when the
            constructor was called, this property will simply return None.
        """
        return self._energies

    def get_coefficient_matrix(
        self, i: int, bl_vector: NDArray[np.int_]
    ) -> NDArray[np.complex128]:
        r"""
        Calculate the coefficient matrix for a given Wannier function.

        Parameters
        ----------
        i : int
            The index identifying the target Wannier function.
        bl_vector : ndarray of np.int_
            The Bravais lattice vector specifying the translation of Wannier function
            i from its home cell.

        Returns
        -------
        c : ndarray of complex
            The coefficient matrix.

        Notes
        -----
        The coefficient matrix :math:`C^{\alpha}` for a given Wannier function
        :math:`\ket{w_{iR}} = \ket{w_{\alpha}}` has dimensions of num_kpoints x
        num_bands. Each element is constructed as :footcite:p:`WOHP`

        .. math::

            C^{\alpha}_{nk} = \exp[ik \cdot R]\left(U^{k}_{ni}\right)^{*},

        where :math:`\alpha` combines the values of the `i` and `bl_vector` arguments
        (it is a combined index that identifies a particular Wannier function), :math:`n`
        is a band index, :math:`k` is a k-point and :math:`U` refers to the unitary
        matrices that mix Bloch vectors to produce Wannier functions. Note that within
        the exponential term, :math:`i = \sqrt{-1}`, whereas it acts as a Wannier
        function index with respect to :math:`U`.

        References
        ----------
        .. footbibliography::
        """
        c = (np.exp(1j * 2 * np.pi * self._kpoints @ bl_vector))[
            :, np.newaxis
        ] * np.conj(self._u[:, :, i])

        return c

    def get_dos_matrix(
        self,
        c_star: NDArray[np.complex128],
        c: NDArray[np.complex128],
        resolve_k: bool = False,
    ) -> NDArray[np.float64]:
        r"""
        Calculate the DOS matrix for a pair of Wannier functions.

        Parameters
        ----------
        c_star : ndarray of complex
            The coefficient matrix for Wannier function i with Bravais lattice vector
            R_1.
        c : ndarray of complex
            The coefficient matrix for Wannier function j with Bravais lattice vector
            R_2.
        resolve_k : bool, optional
            Whether or not to resolve the DOS matrix with respect to k-points. Defaults
            to False.

        Returns
        -------
        dos_matrix : ndarray of float
            The DOS matrix.

        See Also
        --------
        get_coefficient_matrix

        Notes
        -----
        For `resolve_k` = True, the DOS matrix :math:`D_{\alpha\beta}` for a given pair
        of Wannier functions :math:`\ket{w_{\alpha}}` and :math:`\ket{w_{\beta}}` has
        dimensions of num_energy x num_kpoints, where num_energy refers to the number
        of discrete energies over which the density of states has been evaluated. For
        `resolve_k` = False, it is no longer a DOS matrix but rather a DOS vector with
        num_energy elements.

        For the k-resolved case, each element of the DOS matrix is constructed as
        :footcite:p:`original_COHP`

        .. math::

            D_{\alpha\beta}(E, k) = \sum_{n} \mathrm{Re}\left[\left(C^{\alpha}_{nk}
            \right)^{*}C^{\beta}_{nk}\right] \cdot \delta(\epsilon_{nk} - E),

        where :math:`\left(C^{\alpha}\right)^{*}` and :math:`C^{\beta}` reflect the
        values of the `c_star` and `c` arguments and :math:`\delta(\epsilon_{nk} - E)`
        is the density of states evaluated for a particular band and k-point. Summing
        over :math:`k` (`resolve_k` = False) yields

        .. math::

            D_{\alpha\beta}(E) = \sum_{k} D_{\alpha\beta}(E, k),

        which is the aforementioned DOS vector.

        References
        ----------
        .. footbibliography::
        """
        dos_matrix_nk = (
            self._nspin * (c_star * c)[np.newaxis, :, :].real * self._dos_array
        )

        if resolve_k:
            dos_matrix = np.sum(dos_matrix_nk, axis=2)

        else:
            dos_matrix = np.sum(dos_matrix_nk, axis=(1, 2))

        return dos_matrix

    def get_density_matrix_element(
        self, c_star: NDArray[np.complex128], c: NDArray[np.complex128]
    ) -> np.complex128:
        r"""
        Calculate an element of the Wannier density matrix.

        Parameters
        ----------
        c_star : ndarray of complex
            The coefficient matrix for Wannier function i with Bravais lattice vector
            R_1.
        c : ndarray of complex
            The coefficient matrix for Wannier function j with Bravais lattice vector
            R_2.

        Returns
        -------
        element : complex
            An element of the Wannier density matrix.

        See Also
        --------
        get_coefficient_matrix

        Notes
        -----
        A given element of the Wannier density matrix is constructed as

        .. math::

            P_{\alpha\beta} = \sum_{nk} w_{k}f_{nk}\left(C^{\alpha}_{nk}\right)^{*}
            C^{\beta}_{nk},

        where :math:`\left(C^{\alpha}\right)^{*}` and :math:`C^{\beta}` refer to the
        `c_star` and `c` arguments, :math:`f` is the occupation matrix and
        :math:`\{w_{k}\}` are k-point weights.
        """
        if self._occupation_matrix is None:
            raise TypeError(
                "The occupation matrix is required to calculate elements of the Wannier density matrix."
            )

        p_nk = self._occupation_matrix * c_star * c

        element = np.sum(p_nk, axis=(0, 1)) / len(self._kpoints)

        return element

    def assign_descriptors(
        self,
        interactions: AtomicInteractionContainer,
        calc_wohp: bool = True,
        calc_wobi: bool = True,
        resolve_k: bool = False,
        num_proc: int = 4,
        show_progress: bool = True,
    ) -> AtomicInteractionContainer:
        r"""
        Compute WOHPs and/or WOBIs for a set of 2-body interactions.

        Parameters
        ----------
        interactions : AtomicInteractionContainer
            The 2-body interactions for which to calculate WOHPs and/or WOBIs.
        calc_wohp : bool, optional
            Whether or not to calculate WOHPs for the input `interactions`. Defaults to
            True.
        calc_wobi : bool, optional
            Whether or not to calculate WOBIs for the input `interactions`. Defaults to
            True.
        resolve_k : bool, optional
            Whether or not to resolve the output WOHPs and/or WOBIs with respect to
            k-points. Defaults to False.
        num_proc : int, optional
            The number of processes to spawn when computing the pDOS in parallel. Note
            that if `num_proc` is less than the value reported by
            :py:func:`multiprocessing.cpu_count`, then the latter will be used instead.
            Defaults to 4.
        show_progress : bool, optional
            If True, display a :py:mod:`tqdm` progress bar. Defaults to True.

        Returns
        -------
        processed_interactions : AtomicInteractionContainer
            An updated instance of the input `interactions`, with each of the
            AtomicInteraction objects now being associated with the bonding descriptors
            calculated for the overall interaction as well as its constituent
            WannierInteraction objects.

        See Also
        --------
        pengwann.geometry.find_interactions
        get_density_matrix_element

        Notes
        -----
        If both `calc_wohp` and `calc_wobi` are False, then the :code:`dos_matrix`
        attribute of each AtomicInteraction and WannierInteraction will still be set.

        The WOHPs and WOBIs for the input `interactions` are computed using shared
        memory parallelism to avoid copying potentially very large arrays (such as the
        full DOS array) between concurrent processes. Even with shared memory, very
        small (low volume -> many k-points) and very large (many electrons -> many
        bands/Wannier functions) systems can be problematic in terms of memory usage
        if the energy resolution is too high.

        For `resolve_k` = True and `calc_wohp` = True, the k-resolved WOHP for a given
        pair of Wannier functions is computed as :footcite:p:`WOHP, pCOHP`

        .. math::

            \mathrm{WOHP}_{\alpha\beta}(E, k) = -H_{\alpha\beta}D_{\alpha\beta}(E, k),

        where :math:`H` is the Wannier Hamiltonian and :math:`D_{\alpha\beta}` is the
        DOS matrix for Wannier functions :math:`\ket{w_{\alpha}}` and
        :math:`\ket{w_{\beta}}`. For `resolve_k` = False, summing over :math:`k` gives
        the total WOHP between :math:`\ket{w_{\alpha}}` and :math:`\ket{w_{\beta}}`

        .. math::

            \mathrm{WOHP}_{\alpha\beta}(E) = -H_{\alpha\beta}\sum_{k} D_{\alpha\beta}
            (E, k).

        Summing over all WOHPs associated with a given pair of
        atoms yields

        .. math::

            \mathrm{WOHP}_{AB}(E) = \sum_{\alpha\beta \in AB}
            \mathrm{WOHP}_{\alpha\beta}(E),

        which is the total WOHP for the interatomic interaction between atoms :math:`A`
        and :math:`B`.

        For `calc_wobi` = True, the WOBI for a pair of Wannier functions or a pair of
        atoms is computed in an identical manner, except that the DOS matrix is
        weighted by the Wannier density matrix rather than the Wannier Hamiltonian
        :footcite:p:`pCOBI`:

        .. math::

            \mathrm{WOBI}_{\alpha\beta}(E) = P_{\alpha\beta}D_{\alpha\beta}(E).

        References
        ----------
        .. footbibliography::
        """
        if calc_wohp:
            if self._h is None:
                raise TypeError

        if calc_wobi:
            if self._occupation_matrix is None:
                raise TypeError

        wannier_interactions = []
        for interaction in interactions:
            for w_interaction in interaction:
                if calc_wohp:
                    w_interaction_with_h = self.assign_h_ij(w_interaction)

                    wannier_interactions.append(w_interaction_with_h)

                else:
                    wannier_interactions.append(w_interaction)

        processed_wannier_interactions = self.parallelise(
            wannier_interactions, calc_wobi, resolve_k, num_proc, show_progress
        )

        return self._reconstruct_atomic_interactions(
            interactions, processed_wannier_interactions
        )

    def assign_h_ij(self, interaction: WannierInteraction) -> WannierInteraction:
        """
        Assign the relevant element of the Wannier Hamiltonian to an interaction.

        Parameters
        ----------
        interaction : WannierInteraction
            The interaction to which the element of the Wannier Hamiltonian is to be
            assigned.

        Returns
        -------
        interaction_with_h : WannierInteraction
            The input `interaction` with the relevant element of the Wannier Hamiltonian
            assigned.
        """
        bl_vector = tuple(
            [int(component) for component in interaction.bl_2 - interaction.bl_1]
        )

        assert self._h is not None
        h_ij = self._h[bl_vector][interaction.i, interaction.j].real

        return interaction._replace(h_ij=h_ij)

    def _reconstruct_atomic_interactions(
        self,
        atomic_interactions: AtomicInteractionContainer,
        wannier_interactions: tuple[WannierInteraction, ...],
    ) -> AtomicInteractionContainer:
        """
        Reconstruct a set of AtomicInteraction objects with updated descriptors.

        Parameters
        ----------
        atomic_interactions : AtomicInteractionContainer
            The original interactions which are to be reconstructed.
        wannier_interactions : tuple[WannierInteraction, ...]
            The WannierInteraction objects that will be associated with the output
            AtomicInteraction objects and summed over to update the overall atomic
            descriptors.

        Returns
        -------
        processed_interactions : AtomicInteractionContainer
            The reconstructed interactions which are associated with the
            input `wannier_interactions` and contain the total atomic descriptors
            derived by summing over the relevant Wannier functions.
        """
        running_count = 0
        processed_interactions = []
        for interaction in atomic_interactions:
            associated_wannier_interactions = wannier_interactions[
                running_count : running_count + len(interaction)
            ]

            intermediate_interaction = replace(
                interaction, sub_interactions=associated_wannier_interactions
            )
            processed_interaction = intermediate_interaction.with_summed_descriptors()

            processed_interactions.append(processed_interaction)
            running_count += len(processed_interaction)

        return replace(
            atomic_interactions, sub_interactions=tuple(processed_interactions)
        )

    def get_bwdf(
        self,
        interactions: AtomicInteractionContainer,
        geometry: Structure,
        r_range: tuple[float, float],
        nbins: int,
    ) -> tuple[NDArray[np.float64], dict[tuple[str, str], NDArray[np.float64]]]:
        """
        Compute one or more bond-weighted distribution functions (BWDFs).

        Parameters
        ----------
        interactions : AtomicInteractionContainer
            The AtomicInteraction objects containing all of the necessary IWOHPs to
            weight the RDF/s.
        geometry : Structure
            A Pymatgen Structure object from which to extract interatomic distances.
        r_range : 2-length tuple of float
            The range of distances over which to evalute the BWDF/s.
        nbins : int
            The number of bins used to calculate the BWDF/s.

        Returns
        -------
        r : ndarray of float
            The centre of each distance bin.
        bwdf : dict of {2-length tuple of str : ndarray of float} pairs.
            A dictionary containing the BWDFs, indexable by the bond species e.g.
            ("Ga", "As") for the Ga-As BWDF.

        See Also
        --------
        assign_descriptors
        integrate_descriptors

        Notes
        -----
        The BWDF is derived from the RDF (radial distribution function). More
        specifically, it is the RDF excluding all interatomic distances that are not
        counted as bonds (as defined by some arbitrary criteria) with the remaining
        distances being weighted by the corresponding IWOHP :footcite:p:`BWDF`.

        References
        ----------
        .. footbibliography::
        """
        distance_matrix = geometry.distance_matrix

        r_min, r_max = r_range
        intervals = np.linspace(r_min, r_max, nbins + 1)
        dr = (r_max - r_min) / nbins
        r = intervals[:-1] + dr / 2

        bonds = []
        bwdf = {}
        for interaction in interactions:
            if interaction.iwohp is None:
                raise TypeError(
                    f"""The IWOHP for interaction {interaction.tag} has not been
                    computed. This is required to calculate the BWDF."""
                )

            distance = distance_matrix[interaction.i, interaction.j]

            bond = (interaction.symbol_i, interaction.symbol_j)
            if bond not in bonds:
                bonds.append(bond)

                bwdf[bond] = np.zeros((nbins))

            for bin_idx, boundary_i, boundary_j in zip(
                range(len(r)), intervals[:-1], intervals[1:], strict=False
            ):
                if boundary_i <= distance < boundary_j:
                    bwdf[bond][bin_idx] += interaction.iwohp
                    break

        return r, bwdf

    def parallelise(
        self,
        wannier_interactions: Sequence[WannierInteraction],
        calc_p_ij: bool,
        resolve_k: bool,
        num_proc: int = 4,
        show_progress: bool = True,
    ) -> tuple[WannierInteraction, ...]:
        """
        Compute DOS matrices and elements of the Wannier density matrix in parallel.

        Parameters
        ----------
        wannier_interactions : sequence of WannierInteraction
            The WannierInteraction objects for which to compute the DOS matrix and
            (optionally) the relevant elements of the Wannier density matrix.
        calc_p_ij : bool
            Whether or not to calculate the relevant elements of the Wannier density
            matrix.
        resolve_k : bool
            Whether or not to resolve the DOS matrix with respect to k-points.
        num_proc : int, optional
            The number of processes to spawn when computing the pDOS in parallel. Note
            that if `num_proc` is less than the value reported by
            :py:func:`multiprocessing.cpu_count`, then the latter will be used instead.
            Defaults to 4.
        show_progress : bool, optional
            If True, display a :py:mod:`tqdm` progress bar. Defaults to True.

        Returns
        -------
        processed_wannier_interaction : tuple of WannierInteraction
            A sequence of WannierInteraction objects, each of which is associated with
            the DOS matrix and (if `calc_p_ij` = True) the relevant elements of the
            Wannier density matrix.
        """
        memory_keys = ["dos_array", "kpoints", "u"]
        shared_data = [self._dos_array, self._kpoints, self._u]
        if calc_p_ij:
            if self._occupation_matrix is None:
                raise TypeError

            memory_keys.append("occupation_matrix")
            shared_data.append(self._occupation_matrix)

        memory_metadata, memory_handles = allocate_shared_memory(
            memory_keys, shared_data
        )

        args = []
        for w_interaction in wannier_interactions:
            args.append(
                (
                    w_interaction,
                    self._num_wann,
                    self._nspin,
                    calc_p_ij,
                    resolve_k,
                    memory_metadata,
                )
            )

        max_proc = cpu_count()
        processes = min(max_proc, num_proc) if max_proc is not None else num_proc

        with Pool(processes=processes) as pool:
            processed_wannier_interactions = tuple(
                tqdm(pool.imap(self._parallel_wrapper, args), total=len(args))
            )

        for memory_handle in memory_handles:
            memory_handle.unlink()

        return processed_wannier_interactions

    @classmethod
    def _parallel_wrapper(cls, args) -> WannierInteraction:
        """
        A simple wrapper for
        :py:meth:`~pengwann.descriptors.DescriptorCalculator.process_interaction`.

        Parameters
        ----------
        args
            The arguments to be unpacked for
            :py:meth:`~pengwann.descriptors.DescriptorCalculator.process_interaction`.

        Returns
        -------
        wannier_interaction : WannierInteraction
            The input WannierInteraction with the computed properties assigned to the
            relevant attributes.

        Notes
        -----
        This method exists primarily to enable proper :code:`tqdm` functionality with
        :code:`multiprocessing`.
        """
        wannier_interaction = cls._process_interaction(*args)

        return wannier_interaction

    @classmethod
    def _process_interaction(
        cls,
        interaction: WannierInteraction,
        num_wann: int,
        nspin: int,
        calc_wobi: bool,
        resolve_k: bool,
        memory_metadata: dict[str, tuple[tuple[int, ...], np.dtype]],
    ) -> WannierInteraction:
        """
        For a pair of Wannier functions, compute the DOS matrix and (optionally), the
        element of the density matrix required to compute the WOBI.

        Parameters
        ----------
        interaction : WannierInteraction
            The interaction between two Wannier functions for which descriptors are to
            be computed.
        num_wann : int
            The total number of Wannier functions.
        nspin : int
            The number of electrons per fully-occupied band. This should be set to 2
            for non-spin-polarised calculations and set to 1 for spin-polarised
            calculations.
        calc_wobi : bool
            Whether or not to calculate the relevant element of the Wannier density
            matrix for the WOBI.
        resolve_k : bool
            Whether or not to resolve the DOS matrix with respect to k-points.
        memory_metadata : dict[str, tuple[tuple[int, ...], np.dtype]]
            The keys, shapes and dtypes of any data to be pulled from shared memory.

        Returns
        -------
        interaction : WannierInteraction
            The input `interaction` with the computed properties assigned to the
            relevant attributes.
        """
        dcalc_builder: dict[str, Any] = {"num_wann": num_wann, "nspin": nspin}
        memory_handles = []
        for memory_key, metadata in memory_metadata.items():
            shape, dtype = metadata

            shared_memory = SharedMemory(name=memory_key)
            buffered_data = np.ndarray(shape, dtype=dtype, buffer=shared_memory.buf)

            dcalc_builder[memory_key] = buffered_data
            memory_handles.append(shared_memory)

        dcalc = cls(**dcalc_builder)

        c_star = np.conj(dcalc.get_coefficient_matrix(interaction.i, interaction.bl_1))
        c = dcalc.get_coefficient_matrix(interaction.j, interaction.bl_2)

        new_values = {}

        new_values["dos_matrix"] = dcalc.get_dos_matrix(c_star, c, resolve_k)

        if calc_wobi:
            new_values["p_ij"] = dcalc.get_density_matrix_element(c_star, c).real

        for memory_handle in memory_handles:
            memory_handle.close()

        return interaction._replace(**new_values)
