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
from typing import Any, NamedTuple, Optional


class DescriptorCalculator:
    """
    A class for the calculation of various descriptors of chemical bonding and local
    electronic structure.

    Args:
        dos_array (NDArray[np.float64]): The density of states discretised across
            energies, k-points and bands.
        nspin (int): The number of electrons per fully-occupied band. This should be
            set to 2 for non-spin-polarised calculations and set to 1 for spin-polarised
            calculations.
        kpoints (NDArray[np.float64]): The full k-point mesh used in the prior Wannier90
            calculation.
        u (NDArray[np.complex128]): The U matrices that define the Wannier functions in
            terms of the canonical Bloch states.
        h (dict[tuple[int, int, int], NDArray[np.complex128]], optional): The
            Hamiltonian in the Wannier basis. Required for the computation of WOHPs.
            Defaults to None.
        occupation_matrix(NDArray[np.float64], optional): The Kohn-Sham occupation
            matrix. Required for the computation of WOBIs. Defaults to None.
        energies (NDArray[np.float64], optional): The energies at which the dos_array
            has been evaluated. Defaults to None.

    Returns:
        None

    Notes:
        This class should not normally be initialised using the base constructor. See
        instead the
        :py:meth:`~pengwann.descriptors.DescriptorCalculator.from_eigenvalues`
        classmethod.
    """

    _bl_0 = np.array((0, 0, 0))

    def __init__(
        self,
        dos_array: NDArray[np.float64],
        nspin: int,
        kpoints: NDArray[np.float64],
        u: NDArray[np.complex128],
        h: Optional[dict[tuple[int, int, int], NDArray[np.complex128]]] = None,
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
        h: Optional[dict[tuple[int, int, int], NDArray[np.complex128]]] = None,
        occupation_matrix: Optional[NDArray[np.float64]] = None,
    ) -> DescriptorCalculator:
        """
        Initialise a DescriptorCalculator object from a set of Kohn-Sham eigenvalues.

        Args:
            eigenvalues (NDArray[np.float64]): The Kohn-Sham eigenvalues.
            nspin (int): The number of electrons per fully-occupied band. This should be
                set to 2 for non-spin-polarised calculations and set to 1 for spin-polarised
                calculations.
            energy_range (tuple[float, float]): The energy range over which the density
                of states is to be evaluated.
            resolution (float): The desired energy resolution of the density of states.
            sigma (float): The width of the Gaussian kernel used to smear the density
                of states (in eV).
            kpoints (NDArray[np.float64]): The full k-point mesh used in the prior Wannier90
                calculation.
            u (NDArray[np.complex128]): The U matrices that define the Wannier functions in
                terms of the canonical Bloch states.
            h (dict[tuple[int, int, int], NDArray[np.complex128]], optional): The
                Hamiltonian in the Wannier basis. Required for the computation of WOHPs.
                Defaults to None.
            occupation_matrix(NDArray[np.float64], optional): The Kohn-Sham occupation
                matrix. Required for the computation of WOBIs. Defaults to None.

        Returns:
            DescriptorCalculator: The initialised DescriptorCalculator object.

        Notes:
            See the :py:mod:`~pengwann.io` module for parsing the eigenvalues, k-point
            mesh, U matrices and Hamiltonian from Wannier90 output files. See the
            :py:func:`~pengwann.utils.get_occupation_matrix` function for computing the
            occupation matrix.
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
        The energies over which the density of states (and all derived quantities such
        as WOHPs or WOBIs) has been evaluated.
        """
        return self._energies

    def get_coefficient_matrix(
        self, i: int, bl_vector: NDArray[np.int64]
    ) -> NDArray[np.complex128]:
        """
        Calculate the coefficient matrix C_iR for a given Wannier function.

        Args:
            i (int): The index identifying a particular Wannier function.
            bl_vector (NDArray[np.int64]): The Bravais lattice vector specifying the
                relative translation of Wannier function i from its home cell.

        Returns:
            NDArray[np.complex128]: The coefficient matrix C_iR.
        """
        return (np.exp(1j * 2 * np.pi * self._kpoints @ bl_vector))[
            :, np.newaxis
        ] * np.conj(self._u[:, :, i])

    def get_dos_matrix(
        self,
        c_star: NDArray[np.complex128],
        c: NDArray[np.complex128],
        resolve_k: bool = False,
    ) -> NDArray[np.float64]:
        """
        Calculate the DOS matrix for a pair of Wannier functions.

        Args:
            c_star (NDArray[np.complex128]): The coefficient matrix for Wannier
                function i with Bravais lattice vector R_1.
            c (NDArray[np.complex128]): The coefficient matrix for Wannier function j
                with Bravais lattice vector R_2.
            resolve_k (bool): Whether or not to resolve the DOS matrix with respect to
                k-points. Defaults to False.

        Returns:
            NDArray[np.float64]: The DOS matrix, either summed over bands and k-points
            or bands only.
        """
        dos_matrix_nk = (
            self._nspin * (c_star * c)[np.newaxis, :, :].real * self._dos_array
        )

        if resolve_k:
            return np.sum(dos_matrix_nk, axis=2)

        else:
            return np.sum(dos_matrix_nk, axis=(1, 2))

    def get_p_ij(
        self, c_star: NDArray[np.complex128], c: NDArray[np.complex128]
    ) -> np.complex128:
        """
        Calculate element P_ij of the Wannier density matrix.

        Args:
            c_star (NDArray[np.complex128]): The coefficient matrix for Wannier
                function i with Bravais lattice vector R_1.
            c (NDArray[np.complex128]): The coefficient matrix for Wannier function j
                with Bravais lattice vector R_2.

        Returns:
            np.complex128: Element P_ij of the Wannier density matrix.
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
        """
        Calculate the projected density of states for a given set of atoms (and their
        associated Wannier functions).

        Args:
            geometry (Structure): A Pymatgen Structure object with a "wannier_centres"
                site property containing the indices of the Wannier functions
                associated with each atom.
            symbols (tuple[str, ...]): The atomic species to compute the pDOS for.

        Returns:
            tuple[AtomicInteraction, ...]: A series of AtomicInteraction objects, each
            of which contains the pDOS with respect to an atom and its associated
            Wannier functions.

        Notes:
            See the :py:func:`~pengwann.geometry.build_geometry` function for
            obtaining an appropriate Pymatgen Structure to pass as the :code:`geometry`
            argument.
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
            pool.starmap(self.process_interaction, tqdm(args, total=len(args)))
        )

        pool.close()
        for memory_handle in memory_handles:
            memory_handle.unlink()

        running_count = 0
        for interaction in interactions:
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
        """
        Calculate the populations (and optionally charges) associated with a series of
        atoms and their associated Wannier functions. The calculated values are
        assigned to the relevant AtomicInteraction objects as originally input.

        Args:
            interactions (tuple[AtomicInteraction, ...]): A series of AtomicInteraction
                objects containing the pDOS required to compute populations or charges.
            mu (float): The Fermi level.
            resolve_orbitals (bool): Whether or not to calculate the populations for
                the individual Wannier functions associated with each atom. Defaults to
                False.
            valence (dict[str, int], optional): The number of valence electrons
                associated with each atomic species (as per the pseudopotentials used
                in the prior ab initio calculation). Required for the calculation of
                Wannier charges. Defaults to None.

        Returns:
            None

        Notes:
            A suitable set of AtomicInteraction objects containing the necessary pDOS
            can be obtained via the
            :py:meth:`~pengwann.descriptors.DescriptorCalculator.get_pdos` method.
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

                interaction.charge = valence[symbol] - interaction.population

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
        """
        Calculate a series of Wannier Orbital Hamilton Populations (WOHPs) and/or
        Wannier Orbital Bond Indices (WOBIs) in parallel. The calculated values are
        assigned to the relevant AtomicInteraction objects as originally input.

        Args:
            interactions (tuple[AtomicInteraction, ...]): A series of AtomicInteraction
                objects specifying the 2-body interactions for which to compute WOHPs
                and/or WOBIs.
            calc_wohp (bool): Whether or not to calculate WOHPs for the input
                interactions. Defaults to True.
            calc_wobi (bool): Whether or not to calculate WOBIs for the input
                interactions. Defaults to True.
            resolve_k (bool): Whether or not to resolve the output WOHPs and/or WOBIs
                with respect to the k-point mesh. Defaults to False.

        Returns:
            None

        Notes:
            A set of appropriate interatomic interactions can be generated in an
            automated fashion via a distance cutoff criteria using the
            :py:func:`~pengwann.geometry.find_interactions` function.
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
            pool.starmap(self.process_interaction, tqdm(args, total=len(args)))
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
                bl_vector = tuple((w_interaction.bl_2 - w_interaction.bl_1).tolist())
                h_ij = self._h[bl_vector][w_interaction.i, w_interaction.j].real

                w_interaction.h_ij = h_ij

        running_count = 0
        for interaction in interactions:
            if calc_wohp:
                interaction.wohp = np.zeros(self._dos_array.shape[0])

            if calc_wobi:
                interaction.wobi = np.zeros(self._dos_array.shape[0])

            associated_wannier_interactions = amended_wannier_interactions[
                running_count : running_count + len(interaction.wannier_interactions)
            ]
            for w_interaction in associated_wannier_interactions:
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
        Integrate a set of WOHPs and/or WOBIs up to the Fermi level to derive a
        corresponding set of IWOHPs and IWOBIs. The calculated values are assigned to
        the relevant AtomicInteraction objects as originally input.

        Args:
            interactions (tuple[AtomicInteraction, ...]): A series of AtomicInteraction
                objects containing the WOHPs and WOBIs to be integrated.
            mu (float): The Fermi level.
            resolve_orbitals (bool): Whether or not to integrate the WOHPs and/or WOBIs
                for the individual Wannier functions associated with each
                AtomicInteraction as well as the overall interaction itself. Defaults
                to False.

        Returns:
            None
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
        """
        Calculate the Density of Energy (DOE).

        Args:
            interactions (tuple[AtomicInteraction, ...]): A set of AtomicInteraction
                objects containing all of the interatomic (off-diagonal) WOHPs. In
                general this should come from the
                :py:meth:`~pengwann.descriptors.DescriptorCalculator.assign_descriptors`
                method.
            num_wann (int): The total number of Wannier functions.

        Returns:
            NDArray[np.float64]: The density of energy.
        """
        wannier_indices = range(num_wann)

        diagonal_terms = tuple(
            WannierInteraction(i, i, self._bl_0, self._bl_0) for i in wannier_indices
        )
        diagonal_interaction = (AtomicInteraction(("D1", "D1"), diagonal_terms),)
        self.assign_descriptors(diagonal_interaction, calc_wobi=False)

        all_interactions = interactions + diagonal_interaction

        return np.sum([interaction.wohp for interaction in all_interactions], axis=0)  # type: ignore[arg-type]

    def get_bwdf(
        self,
        interactions: tuple[AtomicInteraction, ...],
        geometry: Structure,
        r_range: tuple[float, float],
        nbins: int,
    ) -> tuple[NDArray[np.float64], dict[tuple[str, str], NDArray[np.float64]]]:
        """
        Calculate one or more Bond-Weighted Distribution Functions (BWDFs).

        Args:
            interactions (tuple[AtomicInteraction, ...]): A set of AtomicInteraction
                objects containing all of the necessary WOHPs to generate the desired
                BWDFs.
            geometry (Structure): A Pymatgen Structure object from which to extract
                interatomic distances.
            r_range (tuple[float, float]): The range of distances over which the BWDFs
                should be evaluated.
            nbins (int): The number of bins used to calculate the BWDFs.

        Returns:
            tuple[NDArray[np.float64], dict[tuple[str, str], NDArray[np.float64]]]:

            NDArray[np.float64]: The centre of each distance bin.

            dict[tuple[str, str], NDArray[np.float64]]: A dictionary containing the
            computed BWDFs, indexable by the bond species e.g. ("Ga", "As") for the
            Ga-As BWDF.
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
    def process_interaction(
        cls,
        interaction: WannierInteraction,
        nspin: int,
        calc_wobi: bool,
        resolve_k: bool,
        memory_metadata: dict[str, tuple[tuple[int, ...], np.dtype]],
    ) -> WannierInteraction:
        """
        For the interaction between Wannier functions iR_1 and jR_2, compute the DOS
        matrix and (optionally) the element of the Wannier density matrix required to
        compute the WOBI.

        Args:
            interaction (WannierInteraction): The interaction for which descriptors are
                to be calculated.
            nspin (int): The number of electrons per fully-occupied band. This should be
                set to 2 for non-spin-polarised calculations and set to 1 for
                spin-polarised calculations.
            calc_wobi (bool): Whether or not to calculate the relevant element of the
                Wannier density matrix for the WOBI.
            resolve_k (bool): Whether or not to resolve the DOS matrix with respect to
                the k-point mesh.
            memory_metadata (dict[str, tuple[tuple[int, ...], np.dtype]]): The keys,
                shapes and dtypes of any data to be pulled from shared memory.

        Returns:
            WannierInteraction: The input WannierInteraction with the computed
            properties assigned to the relevant attributes.
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
