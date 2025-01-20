"""
This module contains the :py:class:`~pengwann.descriptors.DescriptorCalculator` class,
which implements the core functionality of :code:`pengwann`: computing various
descriptors of chemical bonding from Wannier functions (via an interface with
Wannier90).
"""

from __future__ import annotations

import numpy as np
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from numpy.typing import NDArray
from pengwann.geometry import AtomicInteraction, WannierInteraction
from pengwann.utils import allocate_shared_memory, integrate, parse_id
from pymatgen.core import Structure
from tqdm.auto import tqdm
from typing import Any, Optional


class DescriptorCalculator:
    """
    Compute descriptors of chemical bonding and local electronic structure.

    This class can be used to calculate:

    - Wannier orbital Hamilton populations (WOHPs) + integrals (IWOHPs)
    - Wannier orbital bond indices (WOBIs) + integrals (IWOBIs)
    - The projected density of states (pDOS)
    - Wannier-function-resolved populations
    - Atomic charges
    - The density of energy (DOE)
    - Bond-weighted distribution functions (BWDFs)

    Parameters
    ----------
    dos_array : ndarray[float]
        The density of states discretised across energies, k-points and bands.
    nspin : int
        The number of electrons per fully-occupied band. This should be set to 2 for
        non-spin-polarised calculations and set to 1 for spin-polarised calculations.
    kpoints : ndarray[float]
        The full k-point mesh used in the prior Wannier90 calculation.
    u : ndarray[complex]
        The U matrices that define the Wannier functions in terms of the canonical
        Bloch states.
    h : dict[tuple[int, ...], ndarray[complex]] | None, optional
        The Hamiltonian in the Wannier basis. Required for the computation of WOHPs.
        Defaults to None.
    occupation_matrix : ndarray[float] | None, optional
        The Kohn-Sham occupation matrix. Required for the computation of WOBIs.
        Defaults to None.
    energies : ndarray[float] | None, optional
        The energies at which the `dos_array` has been evaluated. Defaults to None.

    Returns
    -------
    None

    Notes
    -----
    This class should not normally be initialised using the base constructor. See
    instead the :py:meth:`~pengwann.descriptors.DescriptorCalculator.from_eigenvalues`
    classmethod.
    """

    _bl_0 = np.array((0, 0, 0))

    def __init__(
        self,
        dos_array: NDArray[np.float64],
        nspin: int,
        kpoints: NDArray[np.float64],
        u: NDArray[np.complex128],
        h: Optional[dict[tuple[int, ...], NDArray[np.complex128]]] = None,
        occupation_matrix: Optional[NDArray[np.float64]] = None,
        energies: Optional[NDArray[np.float64]] = None,
    ):
        self._dos_array = dos_array
        self._nspin = nspin
        self._kpoints = kpoints
        self._u = u
        self._h = h
        self._occupation_matrix = occupation_matrix
        self._energies = energies

    @classmethod
    def from_eigenvalues(
        cls,
        eigenvalues: NDArray[np.float64],
        nspin: int,
        energy_range: tuple[float, float],
        resolution: float,
        sigma: float,
        kpoints: NDArray[np.float64],
        u: NDArray[np.complex128],
        h: Optional[dict[tuple[int, ...], NDArray[np.complex128]]] = None,
        occupation_matrix: Optional[NDArray[np.float64]] = None,
    ) -> DescriptorCalculator:
        """
        Initialise a DescriptorCalculator object from a set of Kohn-Sham eigenvalues.

        Parameters
        ----------
        eigenvalues : ndarray[float]
            The Kohn-Sham eigenvalues.
        nspin : int
            The number of electrons per fully-occupied band. This should be set to 2
            for non-spin-polarised calculations and set to 1 for spin-polarised
            calculations.
        energy_range : tuple[float, float]
            The energy range over which the density of states is to be evaluated.
        resolution : float
            The desired energy resolution of the density of states.
        sigma : float
            The width of the Gaussian kernel used to smear the density of states (in eV).
        kpoints : ndarray[float]
            The full k-point mesh used in the prior Wannier90 calculation.
        u : ndarray[complex]
            The U matrices that define the Wannier functions in terms of the canonical
            Bloch states.
        h : dict[tuple[int, ...], ndarray[complex]] | None, optional
            The Hamiltonian in the Wannier basis. Required for the computation of WOHPs.
            Defaults to None.
        occupation_matrix : ndarray[float] | None, optional
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

        return cls(dos_array, nspin, kpoints, u, h, occupation_matrix, energies)

    @property
    def energies(self) -> Optional[NDArray[np.float64]]:
        """
        The discrete energies over which the DOS (and derived descriptors) has been evaluated.

        Returns
        -------
        energies : ndarray[float] | None
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
        c : ndarray[complex]
            The coefficient matrix.
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
        c_star : ndarray[complex]
            The coefficient matrix for Wannier function i with Bravais lattice vector
            R_1.
        c : ndarray[complex]
            The coefficient matrix for Wannier function j with Bravais lattice vector
            R_2.
        resolve_k : bool, optional
            Whether or not to resolve the DOS matrix with respect to k-points. Defaults
            to False.

        Returns
        -------
        dos_matrix : ndarray[float]
            The DOS matrix.

        See Also
        --------
        get_coefficient_matrix
        """
        dos_matrix_nk = (
            self._nspin * (c_star * c)[np.newaxis, :, :].real * self._dos_array
        )

        if resolve_k:
            dos_matrix = np.sum(dos_matrix_nk, axis=2)

        else:
            dos_matrix = np.sum(dos_matrix_nk, axis=(1, 2))

        return dos_matrix

    def get_p_ij(
        self, c_star: NDArray[np.complex128], c: NDArray[np.complex128]
    ) -> np.complex128:
        r"""
        Calculate element P_ij of the Wannier density matrix.

        Parameters
        ----------
        c_star : ndarray[complex]
            The coefficient matrix for Wannier function i with Bravais lattice vector
            R_1.
        c : ndarray[complex]
            The coefficient matrix for Wannier function j with Bravais lattice vector
            R_2.

        Returns
        -------
        p_ij : complex
            Element P_ij of the Wannier density matrix.

        See Also
        --------
        get_coefficient_matrix
        """
        if self._occupation_matrix is None:
            raise TypeError(
                "The occupation matrix is required to calculate elements of the Wannier density matrix."
            )

        p_nk = self._occupation_matrix * c_star * c

        return np.sum(p_nk, axis=(0, 1)) / len(self._kpoints)

    def get_pdos(
        self, geometry: Structure, symbols: tuple[str, ...], resolve_k: bool = False
    ) -> tuple[AtomicInteraction, ...]:
        r"""
        Compute the pDOS for a set of atoms (and their associated Wannier functions).

        Parameters
        ----------
        geometry : Structure
            A Pymatgen Structure object with a :code:`"wannier_centres"` site property
            that associates each atom with the indices of its Wannier centres.
        symbols : tuple[str, ...]
            The atomic species to compute the pDOS for. These should match one or more
            of the species present in `geometry`.

        Returns
        -------
        interactions : tuple[AtomicInteraction, ...]
            A sequence of AtomicInteraction objects, each of which is associated with
            the pDOS for a given atom and its associated Wannier functions.

        See Also
        --------
        pengwann.geometry.build_geometry
        """
        num_wann = len([site for site in geometry if site.species_string == "X0+"])
        wannier_centres = geometry.site_properties["wannier_centres"]

        interactions = []
        for idx in range(len(geometry)):
            symbol = geometry[idx].species_string
            if symbol in symbols:
                label = symbol + str(idx - num_wann + 1)
                pair_id = (label, label)

                wannier_interactions = []
                for i in wannier_centres[idx]:
                    wannier_interaction = WannierInteraction(
                        i, i, self._bl_0, self._bl_0
                    )

                    wannier_interactions.append(wannier_interaction)

                interaction = AtomicInteraction(pair_id, tuple(wannier_interactions))

                interactions.append(interaction)

        if not interactions:
            raise ValueError(f"No atoms matching symbols in {symbols} found.")

        memory_keys = ("dos_array", "kpoints", "u")
        shared_data = (self._dos_array, self._kpoints, self._u)

        memory_metadata, memory_handles = allocate_shared_memory(
            memory_keys, shared_data
        )

        calc_wobi = False
        args = []
        for interaction in interactions:
            for w_interaction in interaction.wannier_interactions:
                args.append(
                    (
                        w_interaction,
                        self._nspin,
                        calc_wobi,
                        resolve_k,
                        memory_metadata,
                    )
                )

        pool = Pool()

        amended_wannier_interactions = tuple(
            tqdm(pool.imap(self.parallel_wrapper, args), total=len(args))
        )

        pool.close()
        for memory_handle in memory_handles:
            memory_handle.unlink()

        running_count = 0
        for interaction in interactions:
            if resolve_k:
                interaction.dos_matrix = np.zeros(self._dos_array.shape[:-1])

            else:
                interaction.dos_matrix = np.zeros(self._dos_array.shape[0])

            associated_wannier_interactions = amended_wannier_interactions[
                running_count : running_count + len(interaction.wannier_interactions)
            ]
            for w_interaction in associated_wannier_interactions:
                interaction.dos_matrix += w_interaction.dos_matrix  # type: ignore[arg-type]

            interaction.wannier_interactions = associated_wannier_interactions
            running_count += len(interaction.wannier_interactions)

        return tuple(interactions)

    def assign_populations(
        self,
        interactions: tuple[AtomicInteraction, ...],
        mu: float,
        resolve_orbitals: bool = False,
        valence: Optional[dict[str, int]] = None,
    ) -> None:
        r"""
        Compute populations/charges for a series of atoms.

        These are the equivalent of Mulliken populations/charges, but calculated in
        the orthonormal Wannier basis.

        Parameters
        ----------
        interactions : tuple[AtomicInteraction, ...]
            A sequence of AtomicInteraction objects containing the pDOS required to
            compute populations or charges.
        mu : float
            The Fermi level.
        resolve_orbitals : bool, optional
            If True, compute Wannier populations for individual Wannier functions as
            well as the atoms to which they are assigned. Defaults to False.
        valence : dict[str, int] | None, optional
            The number of valence electrons associated with each atomic species (as per
            the pseudopotentials used in the ab initio calculation) e.g.
            :code:`{"Ti": 12, "O": 6}`. Required for the calculation of Wannier charges.
            Defaults to None.

        Returns
        -------
        None

        See Also
        --------
        get_pdos

        Notes
        -----
        The input `interactions` are modified in-place by setting the
        :code:`population` and :code:`charge` attributes of each AtomicInteraction (and
        optionally its associated WannierInteraction objects).

        The population for Wannier function :math:`i` is the integral of its pDOS up to
        the Fermi level

        .. math::

            \mathrm{pop}_{i} = \int^{E_{\mathrm{F}}}_{-\infty} dE\;\mathrm{pDOS}_{i}(E).

        Atomic populations are computed simply by summing the populations of all Wannier
        functions associated with a given atom (or equivalently, by integrating the
        total pDOS).

        The Wannier charge of an atom is simply the difference between its number of
        valence electrons :math:`v_{i}` and its population

        .. math::

            \mathrm{charge}_{i} = v_{i} - \mathrm{pop}_{i}.
        """
        if self._energies is None:
            raise TypeError(
                """The energies property returned None. The discretised 
            energies used to compute the DOS array are required for integration."""
            )

        for interaction in interactions:
            if interaction.dos_matrix is None:
                raise TypeError(
                    f"""The DOS matrix for interaction {interaction.pair_id} 
                has not been computed. This is required to calculate the 
                population/charge."""
                )

            interaction.population = integrate(
                self._energies, interaction.dos_matrix, mu
            )

            if valence is not None:
                label = interaction.pair_id[0]
                symbol, _ = parse_id(label)

                if symbol not in valence.keys():
                    raise ValueError(f"Valence for {symbol} not found in input.")

                interaction.charge = (valence[symbol] - interaction.population).astype(
                    np.float64
                )

            if resolve_orbitals:
                for w_interaction in interaction.wannier_interactions:
                    if w_interaction.dos_matrix is None:
                        raise TypeError(
                            f"""The DOS matrix for interaction 
                        {w_interaction.i}{w_interaction.bl_1}<=>
                        {w_interaction.j}{w_interaction.bl_2} has not been computed. 
                        This is required to calculate the population."""
                        )

                    w_interaction.population = integrate(
                        self._energies, w_interaction.dos_matrix, mu
                    )

    def assign_descriptors(
        self,
        interactions: tuple[AtomicInteraction, ...],
        calc_wohp: bool = True,
        calc_wobi: bool = True,
        resolve_k: bool = False,
    ) -> None:
        r"""
        Compute WOHPs and/or WOBIs for a set of 2-body interactions.

        Parameters
        ----------
        interactions : tuple[AtomicInteraction, ...]
            A sequence of AtomicInteraction objects specifying the 2-body interactions
            for which to calculate WOHPs and/or WOBIs.
        calc_wohp : bool, optional
            Whether or not to calculate WOHPs for the input `interactions`. Defaults to
            True.
        calc_wobi : bool, optional
            Whether or not to calculate WOBIs for the input `interactions`. Defaults to
            True.
        resolve_k : bool, optional
            Whether or not to resolve the output WOHPs and/or WOBIs with respect to
            k-points. Defaults to False.

        Returns
        -------
        None

        See Also
        --------
        pengwann.geometry.find_interactions

        Notes
        -----
        If both `calc_wohp` and `calc_wobi` are False, then the :code:`dos_matrix`
        attribute of each AtomicInteraction and WannierInteraction will still be set.

        The input `interactions` are modified in-place by setting the :code:`wohp`
        and/or :code:`wobi` attributes of each AtomicInteraction (and optionally each
        of its associated WannierInteraction objects).

        The WOHPs and WOBIs for the input `interactions` are computed using shared
        memory parallelism to avoid copying potentially very large arrays (such as the
        full DOS array) between concurrent processes. Even with shared memory, very
        small (low volume -> many k-points) and very large (many electrons -> many
        bands/Wannier functions) systems can be problematic in terms of memory usage
        if the energy resolution is too high.
        """
        memory_keys = ["dos_array", "kpoints", "u"]
        shared_data = [self._dos_array, self._kpoints, self._u]
        if calc_wobi:
            if self._occupation_matrix is None:
                raise TypeError(
                    """The occupation matrix must be supplied to calculate 
                WOBIs."""
                )

            memory_keys.append("occupation_matrix")
            shared_data.append(self._occupation_matrix)

        memory_metadata, memory_handles = allocate_shared_memory(
            memory_keys, shared_data
        )

        args = []
        for interaction in interactions:
            for w_interaction in interaction.wannier_interactions:
                args.append(
                    (
                        w_interaction,
                        self._nspin,
                        calc_wobi,
                        resolve_k,
                        memory_metadata,
                    )
                )

        pool = Pool()

        amended_wannier_interactions = tuple(
            tqdm(pool.imap(self.parallel_wrapper, args), total=len(args))
        )

        pool.close()
        for memory_handle in memory_handles:
            memory_handle.unlink()

        if calc_wohp:
            if self._h is None:
                raise TypeError(
                    """The Wannier Hamiltonian must be supplied to calculate 
                WOHPs."""
                )

            for w_interaction in amended_wannier_interactions:
                bl_vector = tuple(
                    [
                        int(component)
                        for component in w_interaction.bl_2 - w_interaction.bl_1
                    ]
                )
                h_ij = self._h[bl_vector][w_interaction.i, w_interaction.j].real

                w_interaction.h_ij = h_ij

        running_count = 0
        for interaction in interactions:
            if resolve_k:
                interaction.dos_matrix = np.zeros(self._dos_array.shape[:-1])

            else:
                interaction.dos_matrix = np.zeros(self._dos_array.shape[0])

            if calc_wohp:
                if resolve_k:
                    interaction.wohp = np.zeros(self._dos_array.shape[:-1])

                else:
                    interaction.wohp = np.zeros(self._dos_array.shape[0])

            if calc_wobi:
                if resolve_k:
                    interaction.wobi = np.zeros(self._dos_array.shape[:-1])

                else:
                    interaction.wobi = np.zeros(self._dos_array.shape[0])

            associated_wannier_interactions = amended_wannier_interactions[
                running_count : running_count + len(interaction.wannier_interactions)
            ]
            for w_interaction in associated_wannier_interactions:
                interaction.dos_matrix += w_interaction.dos_matrix  # type: ignore

                if calc_wohp:
                    interaction.wohp += w_interaction.wohp

                if calc_wobi:
                    interaction.wobi += w_interaction.wobi

            interaction.wannier_interactions = associated_wannier_interactions
            running_count += len(interaction.wannier_interactions)

    def integrate_descriptors(
        self,
        interactions: tuple[AtomicInteraction, ...],
        mu: float,
        resolve_orbitals: bool = False,
    ) -> None:
        """
        Integrate WOHPs and WOBIs up to the Fermi level to derive IWOHPs and IWOBIs.

        Parameters
        ----------
        interactions : tuple[AtomicInteraction, ...]
            A sequence of AtomicInteraction objects containing the WOHPs and/or WOBIs
            to be integrated.
        mu : float
            The Fermi level.
        resolve_orbitals : bool, optional
            If True, integrate the WOHPs and/or WOBIs for the individual Wannier
            functions associated with each AtomicInteraction as well as the overall
            interaction itself. Defaults to False.

        Returns
        -------
        None

        Notes
        -----
        The input `interactions` are modified in-place by setting the :code:`iwohp`
        and/or :code:`iwobi` attributes of each AtomicInteraction (and optionally its
        associated WannierInteraction objects).
        """
        if self._energies is None:
            raise TypeError(
                """The energies property returned None. The discretised 
            energies used to compute the DOS array are required for integration."""
            )

        for interaction in interactions:
            if interaction.wohp is not None:
                interaction.iwohp = integrate(self._energies, interaction.wohp, mu)

            if interaction.wobi is not None:
                interaction.iwobi = integrate(self._energies, interaction.wobi, mu)

            if resolve_orbitals:
                for w_interaction in interaction.wannier_interactions:
                    if w_interaction.h_ij is not None:
                        w_interaction.iwohp = integrate(
                            self._energies, w_interaction.wohp, mu
                        )

                    if w_interaction.p_ij is not None:
                        w_interaction.iwobi = integrate(
                            self._energies, w_interaction.wobi, mu
                        )

    def get_density_of_energy(
        self, interactions: tuple[AtomicInteraction, ...], num_wann: int
    ) -> NDArray[np.float64]:
        r"""
        Calculate the density of energy (DOE).

        Parameters
        ----------
        interactions : tuple[AtomicInteraction, ...]
            A sequence of AtomicInteraction objects containing all of the interatomic
            (off-diagonal) WOHPs.
        num_wann : int
            The total number of Wannier functions.

        Returns
        -------
        doe : ndarray[float]
            The density of energy.

        See Also
        --------
        assign_descriptors : Calculate off-diagonal terms.

        Notes
        -----
        The off-diagonal WOHPs are easily obtained via the
        :py:meth:`~pengwann.descriptors.DescriptorCalculator.assign_descriptors` method.

        The density of energy is defined as

        .. math::
            \mathrm{DOE}(E) = \sum_{ij}\mathrm{WOHP}_{ij}(E),

        it is the total WOHP of the whole system, including diagonal
        (:math:`i = j`) terms.
        """
        for interaction in interactions:
            if interaction.wohp is None:
                raise TypeError(
                    f"""The WOHP for interaction {interaction.pair_id} has 
                not been computed. This is required to calculate the DOE."""
                )

        wannier_indices = range(num_wann)

        diagonal_terms = tuple(
            WannierInteraction(i, i, self._bl_0, self._bl_0) for i in wannier_indices
        )
        diagonal_interaction = (AtomicInteraction(("D1", "D1"), diagonal_terms),)
        self.assign_descriptors(diagonal_interaction, calc_wobi=False)

        all_interactions = interactions + diagonal_interaction

        doe = np.sum([interaction.wohp for interaction in all_interactions], axis=0)  # type: ignore[arg-type]

        return doe

    def get_bwdf(
        self,
        interactions: tuple[AtomicInteraction, ...],
        geometry: Structure,
        r_range: tuple[float, float],
        nbins: int,
    ) -> tuple[NDArray[np.float64], dict[tuple[str, str], NDArray[np.float64]]]:
        """
        Compute one or more bond-weighted distribution functions (BWDFs).

        Parameters
        ----------
        interactions : tuple[AtomicInteraction, ...]
            A sequence of AtomicInteraction obejcts containing all of the necessary
            IWOHPs to weight the RDF/s.
        geometry : Structure
            A Pymatgen Structure object from which to extract interatomic distances.
        r_range : tuple[float, float]
            The range of distances over which to evalute the BWDF/s.
        nbins : int
            The number of bins used to calculate the BWDF/s.

        Returns
        -------
        r : ndarray[float]
            The centre of each distance bin.
        bwdf : dict[tuple[str, str], ndarray[float]]
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
        distances being weighted by the corresponding IWOHP.
        """
        num_wann = len([site for site in geometry if site.species_string == "X0+"])
        distance_matrix = geometry.distance_matrix

        r_min, r_max = r_range
        intervals = np.linspace(r_min, r_max, nbins + 1)
        dr = (r_max - r_min) / nbins
        r = (intervals[:-1] + dr / 2).astype(np.float64)

        bonds = []
        bwdf = {}
        for interaction in interactions:
            if interaction.iwohp is None:
                raise TypeError(
                    f"""The IWOHP for interaction {interaction.pair_id} 
                has not been computed. This is required to calculate the BWDF."""
                )

            id_i, id_j = interaction.pair_id
            symbol_i, i = parse_id(id_i)
            symbol_j, j = parse_id(id_j)
            idx_i = i + num_wann - 1
            idx_j = j + num_wann - 1
            distance = distance_matrix[idx_i, idx_j]

            bond = (symbol_i, symbol_j)
            if bond not in bonds:
                bonds.append(bond)

                bwdf[bond] = np.zeros((nbins))

            for bin_idx, boundary_i, boundary_j in zip(
                range(len(r)), intervals[:-1], intervals[1:], strict=False
            ):
                if boundary_i <= distance < boundary_j:
                    bwdf[bond][bin_idx] += interaction.iwohp
                    break

        return r, bwdf  # type: ignore[return-value]

    @classmethod
    def parallel_wrapper(cls, args) -> WannierInteraction:
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
        wannier_interaction = cls.process_interaction(*args)

        return wannier_interaction

    @classmethod
    def process_interaction(
        cls,
        interaction: WannierInteraction,
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
        kwargs = {"nspin": nspin}  # type: dict[str, Any]
        memory_handles = []
        for memory_key, metadata in memory_metadata.items():
            shape, dtype = metadata

            shared_memory = SharedMemory(name=memory_key)
            buffered_data = np.ndarray(
                shape, dtype=dtype, buffer=shared_memory.buf
            )  # type: NDArray

            kwargs[memory_key] = buffered_data
            memory_handles.append(shared_memory)

        dcalc = cls(**kwargs)

        c_star = np.conj(dcalc.get_coefficient_matrix(interaction.i, interaction.bl_1))
        c = dcalc.get_coefficient_matrix(interaction.j, interaction.bl_2)

        interaction.dos_matrix = dcalc.get_dos_matrix(c_star, c, resolve_k)

        if calc_wobi:
            interaction.p_ij = dcalc.get_p_ij(c_star, c).real

        for memory_handle in memory_handles:
            memory_handle.close()

        return interaction
