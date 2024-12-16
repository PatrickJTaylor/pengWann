from __future__ import annotations

import numpy as np
from multiprocessing import Pool
from pengwann.geometry import AtomicInteraction
from pengwann.utils import get_atom_indices, get_occupation_matrix
from pymatgen.core import Structure
from scipy.integrate import trapezoid  # type: ignore
from tqdm.auto import tqdm
from typing import Optional


class DOS:
    """
    A class for the calculation and manipulation of the density of
    states.
    """

    _R_1 = np.array([0, 0, 0])

    def __init__(
        self,
        energies: np.ndarray,
        dos_array: np.ndarray,
        nspin: int,
        kpoints: np.ndarray,
        U: np.ndarray,
        H: Optional[dict[tuple[int, ...], np.ndarray]] = None,
        occupation_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialise a DOS object.

        Args:
            energies (np.ndarray): The energies at which the DOS has
                been evaluated.
            dos_array (np.ndarray): The DOS at each energy, band and
                k-point.
            nspin (int): The number of electrons per Kohn-Sham state.
                For spin-polarised calculations, set to 1.
            kpoints (np.ndarray): The full k-point mesh.
            U (np.ndarray): The U matrices used to define Wannier
                functions from the Kohn-Sham states.
            f (np.ndarray): An occupation matrix of appropriate shape
                for calculating elements of the density matrix.
            H (np.ndarray, optional): The Hamiltonian in the Wannier
                basis. Required for the computation of WOHPs. Defaults
                to None.
            occupation_matrix (np.ndarray, optional): The occupation matrix.
                Required for the computation of WOBIs. Defaults to None.

        Returns:
            None

        Notes:
            The vast majority of the time, it will be more convenient to
            initialise a DOS object using the from_eigenvalues
            classmethod.
        """
        self._energies = energies
        self._dos_array = dos_array
        self._kpoints = kpoints
        self._U = U
        self._occupation_matrix = occupation_matrix
        self._H = H
        self._nspin = nspin

    def get_dos_matrix(
        self,
        i: int,
        j: int,
        R_1: np.ndarray,
        R_2: np.ndarray,
        sum_matrix: bool = True,
    ) -> np.ndarray:
        """
        Calculate the DOS matrix for a given pair of Wannier functions.

        Args:
            i (int): The index for Wannier function i.
            j (int): The index for Wannier function j.
            R_1 (np.ndarray): The Bravais lattice vector for Wannier
                function i.
            R_2 (np.ndarray): The Bravais lattice vector for Wannier
                function j.
            sum_matrix (bool): Whether or not to sum over bands and
                k-points before returning the DOS matrix.

        Returns:
            np.ndarray: The DOS matrix, either fully-specified or summed
                over bands and k-points.
        """
        C_dagger = (np.exp(-1j * 2 * np.pi * self._kpoints @ R_1))[
            :, np.newaxis
        ] * self._U[:, :, i]
        C = (np.exp(1j * 2 * np.pi * self._kpoints @ R_2))[
            :, np.newaxis
        ] * np.conj(self._U[:, :, j])
        C_dagger_C = (C_dagger * C).T

        dos_matrix = (
            self._nspin * C_dagger_C[np.newaxis, :, :].real * self._dos_array
        )

        if sum_matrix:
            return np.sum(dos_matrix, axis=(1, 2))

        else:
            return dos_matrix

    def get_WOHP(
        self,
        i: int,
        j: int,
        R_1: np.ndarray,
        R_2: np.ndarray,
        dos_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculate the WOHP for a given pair of Wannier functions.

        Args:
            i (int): The index for Wannier function i.
            j (int): The index for Wannier function j.
            R_1 (np.ndarray): The Bravais lattice vector for Wannier
                function i.
            R_2 (np.ndarray): The Bravais lattice vector for Wannier
                function j.
            dos_matrix (np.ndarray, optional): The DOS matrix summed
                over bands and k-points. Will be calculated if not
                provided explicitly.

        Returns:
            np.ndarray: The WOHP arising from w_iR_1 and w_jR_2.
        """
        if self._H is None:
            raise ValueError(
                'The Wannier Hamiltonian is required to calculate WOHPs.'
            )

        R = tuple((R_2 - R_1).tolist())

        print(self._H[R][i, j].real)
        if dos_matrix is None:
            return -self._H[R][i, j].real * self.get_dos_matrix(i, j, R_1, R_2)

        else:
            return -self._H[R][i, j].real * dos_matrix

    def get_WOBI(
        self,
        i: int,
        j: int,
        R_1: np.ndarray,
        R_2: np.ndarray,
        dos_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculate the WOBI for a given pair of Wannier functions.

        Args:
            i (int): The index for Wannier function i.
            j (int): The index for Wannier function j.
            R_1 (np.ndarray): The Bravais lattice vector for Wannier
                function i.
            R_2 (np.ndarray): The Bravais lattice vector for Wannier
                function j.
            dos_matrix (np.ndarray, optional): The DOS matrix summed
                over bands and k-points. Will be calculated if not
                provided explicitly.

        Returns:
            np.ndarray: The WOBI arising from w_iR_1 and w_jR_2.
        """
        if self._occupation_matrix is None:
            raise ValueError(
                'The occupation matrix is required to calculate WOBIs.'
            )

        if dos_matrix is None:
            return self.P_ij(i, j, R_1, R_2).real * self.get_dos_matrix(
                i, j, R_1, R_2
            )

        else:
            return self.P_ij(i, j, R_1, R_2).real * dos_matrix

    def P_ij(
        self, i: int, j: int, R_1: np.ndarray, R_2: np.ndarray
    ) -> complex:
        """
        Calculate element <w_iR_1|P|w_jR_2> of the Wannier density matrix.

        Args:
            i (int): The index for Wannier function i.
            j (int): The index for Wannier function j.
            R_1 (np.ndarray): The Bravais lattice vector for Wannier
                function i.
            R_2 (np.ndarray): The Bravais lattice vector for Wannier
                function j.

        Returns:
            complex: The desired element of the density matrix.
        """
        C_dagger = (np.exp(-1j * 2 * np.pi * self._kpoints @ R_1))[
            :, np.newaxis
        ] * self._U[:, :, i]
        C = (np.exp(1j * 2 * np.pi * self._kpoints @ R_2))[
            :, np.newaxis
        ] * np.conj(self._U[:, :, j])

        P_nk = self._occupation_matrix * C_dagger * C

        return np.sum(P_nk, axis=(0, 1)) / len(self._kpoints)

    def project(
        self, geometry: Structure, symbols: tuple[str, ...]
    ) -> dict[str, np.ndarray]:
        """
        Calculate the pDOS for a set of atomic species.

        Args:
            geometry (Structure): A Pymatgen structure object with a
                'wannier_centres' site property containing the indices
                of the Wannier centres associated with each atom.
            symbols (tuple[str]): The atomic species to get the pDOS
                for.

        Returns:
            dict[str, np.ndarray]: The pDOS for the specified atomic
                species.
        """
        wannier_centres = geometry.site_properties['wannier_centres']
        atom_indices = get_atom_indices(geometry, symbols)

        wannier_indices = {}
        for symbol, indices in atom_indices.items():
            nested_wannier_indices = [wannier_centres[idx] for idx in indices]
            wannier_indices[symbol] = tuple(
                [
                    idx
                    for wannier_tuple in nested_wannier_indices
                    for idx in wannier_tuple
                ]
            )

        pool = Pool()

        pdos = {}
        for symbol, indices in tqdm(wannier_indices.items()):
            args = []

            for i in indices:
                args.append((i, i, self._R_1, self._R_1))

            pdos[symbol] = np.sum(
                pool.starmap(self.get_dos_matrix, args), axis=0
            )

        pool.close()

        return pdos

    def get_descriptors(
        self,
        interactions: tuple[AtomicInteraction, ...],
        calculate_wohp: bool = True,
        calculate_wobi: bool = True,
    ) -> dict[tuple[str, str], dict[str, np.ndarray]]:
        """
        Calculate a series of bonding descriptors. This function is
        designed for the parallel computation of many WOHPs and WOBIs
        from a set of interactions defined by using the
        InteractionFinder class.

        Args:
            interactions (tuple[AtomicInteraction, ...]): The interactions
                for which descriptors are to be computed. In general,
                this should come from the get_interactions method of an
                InteractionFinder object.
            calculate_wohp (bool): Whether to calculate WOHPs for each
                interaction. Defaults to True.
            calculate_wobi (bool): Whether to calculate WOBIs for each
                interaction. Defaults to True.

        Returns:
            dict[tuple[str, str], dict[str, np.ndarray]]: the WOHPs and
            WOBIs for each interaction.
        """
        descriptors = {}

        labels_list = []
        if calculate_wohp:
            labels_list.append('WOHP')

        if calculate_wobi:
            labels_list.append('WOBI')

        labels = tuple(labels_list)

        args = []
        for interaction in interactions:
            args.append((interaction, labels))

        pool = Pool()

        unordered_descriptors = tuple(
            tqdm(
                pool.imap_unordered(self.process_interaction, args),
                total=len(args),
            )
        )

        # Sort the descriptors according to the input order of interactions.
        for pair_id_i in [interaction.pair_id for interaction in interactions]:
            for pair_id_j, interaction_descriptors in unordered_descriptors:
                if pair_id_i == pair_id_j:
                    descriptors[pair_id_i] = interaction_descriptors
                    break

        pool.close()

        return descriptors

    def integrate_descriptors(
        self,
        descriptors: dict[tuple[str, str], dict[str, np.ndarray]],
        mu: float,
    ) -> dict[tuple[str, str], dict[str, float]]:
        """
        Integrate a set of WOHPs and/or WOBIs.

        Args:
            descriptors (dict[str, dict]): A set of bonding descriptors
                i.e. WOHPs and/or WOBIs.
            mu (float): The Fermi level.

        Returns:
            dict[str, float]: The integrated descriptors.
        """
        for idx, energy in enumerate(self._energies):
            if energy > mu:
                fermi_idx = idx
                break

        integrated_descriptors = {}
        for interaction, interaction_descriptors in descriptors.items():
            integrals = {}

            for label, descriptor in interaction_descriptors.items():
                x = self._energies[:fermi_idx]
                y = descriptor[:fermi_idx]

                integrals['I' + label] = trapezoid(y, x)

            integrated_descriptors[interaction] = integrals

        return integrated_descriptors

    def process_interaction(
        self, interaction_and_labels: tuple[AtomicInteraction, tuple[str, ...]]
    ) -> tuple[tuple[str, str], dict[str, np.ndarray]]:
        """
        Calculate the WOHP and/or WOBI associated with a given
        interaction (i.e. a pair of atoms).

        Args:
            interaction_and_labels (tuple[AtomicInteraction, tuple[str, ...]]):
                The interaction and a set of labels specifying which
                descriptors should be computed.

        Returns:
            dict[str, np.ndarray]: The WOHP and/or WOBI associated with
                the given interaction.
        """
        interaction, labels = interaction_and_labels

        interaction_descriptors = {}
        for label in labels:
            interaction_descriptors[label] = np.zeros((len(self._energies)))

        for w_interaction in interaction.wannier_interactions:
            i, j, R_2 = w_interaction.i, w_interaction.j, w_interaction.R_2

            dos_matrix = self.get_dos_matrix(
                i, j, self._R_1, R_2
            )

            if 'WOHP' in labels:
                wohp = self.get_WOHP(
                    i, j, self._R_1, R_2, dos_matrix
                )
                interaction_descriptors['WOHP'] += wohp

            if 'WOBI' in labels:
                wobi = self.get_WOBI(
                    i, j, self._R_1, R_2, dos_matrix
                )
                interaction_descriptors['WOBI'] += wobi

        return interaction.pair_id, interaction_descriptors

    @property
    def energies(self) -> np.ndarray:
        return self._energies

    @classmethod
    def from_eigenvalues(
        cls,
        eigenvalues: np.ndarray,
        nspin: int,
        energy_range: tuple[float, float],
        resolution: float,
        sigma: float,
        kpoints: np.ndarray,
        U: np.ndarray,
        H: Optional[dict[tuple[int, ...], np.ndarray]] = None,
        occupation_matrix: Optional[np.ndarray] = None,
    ) -> DOS:
        """
        Initialise a DOS object from the Kohn-Sham eigenvalues.

        Args:
            eigenvalues (np.ndarray): The Kohn-Sham eigenvalues.
            nspin (int): The number of electrons per Kohn-Sham state.
                For spin-polarised calculations, set to 1.
            energy_range(tuple[float, float]): The energy ranage over which the
                DOS is to be evaluated.
            resolution (float): The desired energy resolution of the
                DOS.
            sigma (float): A Gaussian smearing parameter.
            kpoints (np.ndarray): The full k-point mesh.
            U (np.ndarray): The U matrices used to define Wannier
                functions from the Kohn-Sham states.
            H (np.ndarray, optional): The Hamiltonian in the Wannier
                basis. Required for the computation of WOHPs. Defaults
                to None.
            occupation_matrix (np.ndarray, optional): The occupation matrix.
                Required for the computation of WOBIs. Defaults to None.

        Returns:
            DOS: The initialised DOS object.

        Notes:
            See the utils module for computing the occupation matrix.
        """
        emin, emax = energy_range
        energies = np.arange(emin, emax + resolution, resolution)

        x_mu = energies[:, np.newaxis, np.newaxis] - eigenvalues
        dos_array = (
            1
            / np.sqrt(np.pi * sigma)
            * np.exp(-(x_mu**2) / sigma)
            / eigenvalues.shape[1]
        )

        return cls(energies, dos_array, nspin, kpoints, U, H, occupation_matrix)
