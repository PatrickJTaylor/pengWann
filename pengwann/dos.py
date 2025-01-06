"""
This module contains the :py:class:`~pengwann.dos.DOS` class, which implements
the core functionality of :py:mod:`pengwann`: computing bonding descriptors
from Wannier functions via an interface to Wannier90.
"""

from __future__ import annotations

import numpy as np
from multiprocessing import Pool
from pengwann.geometry import AtomicInteraction
from pengwann.utils import get_occupation_matrix
from pymatgen.core import Structure
from scipy.integrate import trapezoid  # type: ignore
from tqdm.auto import tqdm
from typing import Optional


class DOS:
    """
    A class for the calculation and manipulation of the density of
    states.

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
                k-points before returning the DOS matrix. Defaults to True.

        Returns:
            np.ndarray: The DOS matrix, either fully-specified or summed
            over bands and k-points.
        """
        C_star = (np.exp(-1j * 2 * np.pi * self._kpoints @ R_1))[
            :, np.newaxis
        ] * self._U[:, :, i]
        C = (np.exp(1j * 2 * np.pi * self._kpoints @ R_2))[:, np.newaxis] * np.conj(
            self._U[:, :, j]
        )
        C_star_C = (C_star * C).T

        dos_matrix = self._nspin * C_star_C[np.newaxis, :, :].real * self._dos_array

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
        r"""
        Calculate the WOHP for a given pair of Wannier functions.

        .. math::
            \mathrm{WOHP}^{R}_{ij}(E) = -H^{R}_{ij}
            \sum_{nk}\mathrm{Re}(C^{*}_{iR_{1}k}C_{jR_{2}k})\delta(E - \epsilon_{nk})

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
            np.ndarray: The WOHP arising from :math:`\ket{iR_{1}}` and
            :math:`\ket{jR_{2}}`.
        """
        if self._H is None:
            raise ValueError("The Wannier Hamiltonian is required to calculate WOHPs.")

        R = tuple((R_2 - R_1).tolist())

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
        r"""
        Calculate the WOBI for a given pair of Wannier functions.

        .. math::
            \mathrm{WOBI}^{R}_{ij}(E) = P^{R}_{ij}
            \sum_{nk}\mathrm{Re}(C^{*}_{iR_{1}k}C_{jR_{2}k})\delta(E - \epsilon_{nk})

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
            np.ndarray: The WOBI arising from :math:`\ket{iR_{1}}` and
            :math:`\ket{jR_{2}}`.
        """
        if self._occupation_matrix is None:
            raise ValueError("The occupation matrix is required to calculate WOBIs.")

        if dos_matrix is None:
            return self.P_ij(i, j, R_1, R_2).real * self.get_dos_matrix(i, j, R_1, R_2)

        else:
            return self.P_ij(i, j, R_1, R_2).real * dos_matrix

    def P_ij(self, i: int, j: int, R_1: np.ndarray, R_2: np.ndarray) -> complex:
        r"""
        Calculate element :math:`P^{R}_{ij} = \braket{iR_{1}|P|jR_{2}}`
        of the Wannier density matrix.

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
        C_star = (np.exp(-1j * 2 * np.pi * self._kpoints @ R_1))[
            :, np.newaxis
        ] * self._U[:, :, i]
        C = (np.exp(1j * 2 * np.pi * self._kpoints @ R_2))[:, np.newaxis] * np.conj(
            self._U[:, :, j]
        )

        P_nk = self._occupation_matrix * C_star * C

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
            dict[str, np.ndarray]: The pDOS for each atom that is labelled by an
            appropriate symbol.
        """
        num_wann = len(
            [
                idx
                for idx in range(len(geometry))
                if geometry[idx].species_string == "X0+"
            ]
        )
        wannier_centres = geometry.site_properties["wannier_centres"]

        wannier_indices = {}
        for idx in range(len(geometry)):
            symbol = geometry[idx].species_string
            if symbol in symbols:
                wannier_indices[symbol + str(idx - num_wann + 1)] = wannier_centres[idx]

        pool = Pool()

        args = []
        for indices in wannier_indices.values():
            for i in indices:
                args.append((i, i, self._R_1, self._R_1))

        pdos = {}
        ordered_pdos = tuple(
            pool.starmap(self.get_dos_matrix, tqdm(args, total=len(args)))
        )

        running_count = 0
        for label, indices in wannier_indices.items():
            pdos[label] = np.sum(
                ordered_pdos[running_count : running_count + len(indices)], axis=0
            )
            running_count += len(indices)

        pool.close()

        return pdos

    def get_populations(
        self,
        pdos: dict[str, np.ndarray],
        mu: float,
        valence: Optional[dict[str, int]] = None,
    ) -> dict[str, float]:
        """
        Calculate the Wannier populations (and optionally charges) from the pDOS for a
        chosen set of atoms.

        Args:
            pdos (dict[str, np.ndarray]): The pDOS for each atom.
            mu (float): The Fermi level.
            valence (dict[str, int], optional): The number of valence electrons for
                each atomic species, needed for the calculation of Wannier charges.
                Defaults to None.

        Returns:
            dict[str, float]: The Wannier populations (and optionally charges)
            associated with each atom.
        """
        for idx, energy in enumerate(self._energies):
            if energy > mu:
                fermi_idx = idx
                break

        populations = {}
        for label, dos in pdos.items():
            integrals = {}

            x = self._energies[:fermi_idx]
            y = dos[:fermi_idx]

            integrals["population"] = trapezoid(y, x, axis=0)

            if valence is not None:
                for idx, character in enumerate(label):
                    if character.isdigit():
                        symbol = label[:idx]
                        break

                integrals["charge"] = valence[symbol] - integrals["population"]

            populations[label] = integrals

        return populations

    def get_descriptors(
        self,
        interactions: tuple[AtomicInteraction, ...],
        calculate_wohp: bool = True,
        calculate_wobi: bool = True,
        sum_k: bool = True,
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
            sum_k (bool): Whether to sum over k-points when computing
                WOHPs and/or WOBIs. Defaults to True.

        Returns:
            dict[tuple[str, str], dict[str, np.ndarray]]: the WOHPs and
            WOBIs for each interaction.
        """
        descriptors = {}

        labels_list = []
        if calculate_wohp:
            labels_list.append("WOHP")

        if calculate_wobi:
            labels_list.append("WOBI")

        if sum_k:
            labels_list.append("sum_k")

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

                integrals["I" + label] = trapezoid(y, x, axis=0)

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

        interaction_descriptors = {}  # type: dict[str, np.ndarray]
        if "sum_k" in labels:
            for label in labels[:-1]:
                interaction_descriptors[label] = np.zeros((len(self._energies)))

        else:
            for label in labels:
                interaction_descriptors[label] = np.zeros(
                    (len(self._energies), len(self._kpoints))
                )

        for w_interaction in interaction.wannier_interactions:
            i, j, R_1, R_2 = (
                w_interaction.i,
                w_interaction.j,
                w_interaction.R_1,
                w_interaction.R_2,
            )

            if "sum_k" in labels:
                dos_matrix = self.get_dos_matrix(i, j, R_1, R_2)

            else:
                full_dos_matrix = self.get_dos_matrix(i, j, R_1, R_2, sum_matrix=False)
                # Sum over bands only
                dos_matrix = np.sum(full_dos_matrix, axis=1)

            if "WOHP" in labels:
                wohp = self.get_WOHP(i, j, R_1, R_2, dos_matrix)
                interaction_descriptors["WOHP"] += wohp

            if "WOBI" in labels:
                wobi = self.get_WOBI(i, j, R_1, R_2, dos_matrix)
                interaction_descriptors["WOBI"] += wobi

        return interaction.pair_id, interaction_descriptors

    @property
    def energies(self) -> np.ndarray:
        """
        The array of energies over which the DOS (and all derived quantities
        such as WOHPs and WOBIs) has been evaluated.
        """
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
