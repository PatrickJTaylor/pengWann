"""
This module contains the :py:class:`~pengwann.dos.DOS` class, which implements the core
functionality of :code:`pengwann`: computing bonding descriptors from Wannier functions
via an interface to Wannier90.
"""

from __future__ import annotations

import numpy as np
from multiprocessing import Pool
from pengwann.geometry import AtomicInteraction, WannierInteraction
from pengwann.utils import get_occupation_matrix, integrate, parse_id
from pymatgen.core import Structure
from tqdm.auto import tqdm
from typing import Optional


class DOS:
    """
    A class for the calculation and manipulation of the density of states, from which
    various descriptors of chemical bonding can be derived.

    Args:
        energies (np.ndarray): The energies at which the DOS has been evaluated.
        dos_array (np.ndarray): The DOS at each energy, band and k-point.
        nspin (int): The number of electrons per Kohn-Sham state. For spin-polarised
            calculations, set to 1.
        kpoints (np.ndarray): The full k-point mesh.
        U (np.ndarray): The U matrices used to define Wannier functions from the
            Kohn-Sham states.
        H (np.ndarray, optional): The Hamiltonian in the Wannier basis. Required for
            the computation of WOHPs. Defaults to None.
        occupation_matrix (np.ndarray, optional): The occupation matrix. Required for
            the computation of WOBIs. Defaults to None.

    Returns:
        None

    Notes:
        The vast majority of the time, it will be more convenient to initialise a DOS
        object using the :py:meth:`~pengwann.dos.DOS.from_eigenvalues` classmethod.
    """

    _R_0 = np.array([0, 0, 0])

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
        Initialise a DOS object from a set of Kohn-Sham eigenvalues.

        Args:
            eigenvalues (np.ndarray): The Kohn-Sham eigenvalues.
            nspin (int): The number of electrons per Kohn-Sham state. For
                spin-polarised calculations, set to 1.
            energy_range(tuple[float, float]): The energy range over which the DOS is
                to be evaluated.
            resolution (float): The desired energy resolution of the DOS.
            sigma (float): A Gaussian smearing parameter.
            kpoints (np.ndarray): The full k-point mesh.
            U (np.ndarray): The U matrices used to define Wannier functions from the
                Kohn-Sham states.
            H (np.ndarray, optional): The Hamiltonian in the Wannier basis. Required
                for the computation of WOHPs. Defaults to None.
            occupation_matrix (np.ndarray, optional): The occupation matrix. Required
                for the computation of WOBIs. Defaults to None.

        Returns:
            DOS: The initialised DOS object.

        Notes:
            See the :py:mod:`~pengwann.io` module for parsing the eigenvalues, kpoints,
            U matrices and Hamiltonian from Wannier90 output files. See the
            :py:mod:`~pengwann.utils` module for computing the occupation matrix.
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

    @property
    def energies(self) -> np.ndarray:
        """
        The array of energies over which the DOS (and all derived quantities such as
        WOHPs and WOBIs) has been evaluated.
        """
        return self._energies

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
            R_1 (np.ndarray): The Bravais lattice vector for Wannier function i.
            R_2 (np.ndarray): The Bravais lattice vector for Wannier function j.
            sum_matrix (bool): Whether or not to sum over bands and k-points before
                returning the DOS matrix. Defaults to True.

        Returns:
            np.ndarray: The DOS matrix, either fully-specified or summed over bands
            and k-points.
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
            R_1 (np.ndarray): The Bravais lattice vector for Wannier function i.
            R_2 (np.ndarray): The Bravais lattice vector for Wannier function j.
            dos_matrix (np.ndarray, optional): The DOS matrix, calculated by calling
                :py:meth:`~pengwann.dos.DOS.get_dos_matrix` if not provided explicitly.
                Defaults to None.

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
            R_1 (np.ndarray): The Bravais lattice vector for Wannier function i.
            R_2 (np.ndarray): The Bravais lattice vector for Wannier function j.
            dos_matrix (np.ndarray, optional): The DOS matrix, calculated by calling
                :py:meth:`~pengwann.dos.DOS.get_dos_matrix` if not provided explicitly.
                Defaults to None.

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
        Calculate element :math:`P^{R}_{ij} = \braket{iR_{1}|P|jR_{2}}` of the Wannier
        density matrix.

        Args:
            i (int): The index for Wannier function i.
            j (int): The index for Wannier function j.
            R_1 (np.ndarray): The Bravais lattice vector for Wannier function i.
            R_2 (np.ndarray): The Bravais lattice vector for Wannier function j.

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
            geometry (Structure): A Pymatgen structure object with a 'wannier_centres'
                site property containing the indices of the Wannier centres associated
                with each atom.
            symbols (tuple[str]): The atomic species to compute the pDOS for.

        Returns:
            dict[str, np.ndarray]: The pDOS for each atom that is labelled by an
            appropriate symbol.

        Notes:
            See the :py:mod:`~pengwann.geometry` module for obtaining an appropriate
            Pymatgen Structure object to supply as the :code:`geometry` argument.
        """
        num_wann = len([site for site in geometry if site.species_string == "X0+"])
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
                args.append((i, i, self._R_0, self._R_0))

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
    ) -> dict[str, dict[str, float]]:
        """
        Calculate the Wannier populations (and optionally charges) from the pDOS for a
        chosen set of atoms.

        Args:
            pdos (dict[str, np.ndarray]): The pDOS for each atom as obtained from the
                :py:meth:`~pengwann.dos.DOS.project` method.
            mu (float): The Fermi level.
            valence (dict[str, int], optional): The number of valence electrons for
                each atomic species, needed for the calculation of Wannier charges.
                Defaults to None.

        Returns:
            dict[str, dict[str, float]]: The Wannier populations (and optionally
            charges) associated with each atom.
        """
        populations = {}
        for label, dos in pdos.items():
            integrals = {}

            integrals["population"] = integrate(self._energies, dos, mu)

            if valence is not None:
                symbol, _ = parse_id(label)

                if symbol not in valence.keys():
                    raise ValueError(f"Valence for {symbol} not found in input.")

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
        Calculate a series of bonding descriptors. This function is designed for the
        parallel computation of many WOHPs and WOBIs.

        Args:
            interactions (tuple[AtomicInteraction, ...]): The interactions for which
                descriptors are to be computed. See the
                :py:func:`~pengwann.geometry.find_interactions` function for a method
                of generating desirable interatomic interactions in an automated
                fashion.
            calculate_wohp (bool): Whether to calculate WOHPs for each interaction.
                Defaults to True.
            calculate_wobi (bool): Whether to calculate WOBIs for each interaction.
                Defaults to True.
            sum_k (bool): Whether to sum over k-points when computing WOHPs and/or
                WOBIs. Defaults to True.

        Returns:
            dict[tuple[str, str], dict[str, np.ndarray]]: the WOHPs and WOBIs for each
            interaction.
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
            descriptors (dict[str, dict[str, np.ndarray]]): A set of bonding
                descriptors i.e. WOHPs and/or WOBIs, usually obtained via the
                :py:meth:`~pengwann.dos.DOS.get_descriptors` method.
            mu (float): The Fermi level.

        Returns:
            dict[str, dict[str, float]]: The integrated descriptors.
        """
        integrated_descriptors = {}
        for interaction, interaction_descriptors in descriptors.items():
            integrals = {}

            for label, descriptor in interaction_descriptors.items():
                integrals["I" + label] = integrate(self._energies, descriptor, mu)

            integrated_descriptors[interaction] = integrals

        return integrated_descriptors

    def process_interaction(
        self, interaction_and_labels: tuple[AtomicInteraction, tuple[str, ...]]
    ) -> tuple[tuple[str, str], dict[str, np.ndarray]]:
        """
        Calculate the WOHP and/or WOBI associated with a given interaction (i.e. a pair
        of atoms).

        Args:
            interaction_and_labels (tuple[AtomicInteraction, tuple[str, ...]]):
                The interaction and a set of labels specifying which descriptors
                should be computed.

        Returns:
            tuple[tuple[str, str], dict[str, np.ndarray]]:

            tuple[str, str]: The :code:`pair_id` attribute of the input interaction,
            returned for convenience in sorting interactions when computed in parallel.

            dict[str, np.ndarray]: The WOHP and/or WOBI associated with the given
            interaction.
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

    def get_density_of_energy(
        self, descriptors: dict[tuple[str, str], dict[str, np.ndarray]], num_wann: int
    ) -> np.ndarray:
        """
        Calculate the density of energy (DOE).

        Args:
            descriptors (dict[tuple[str, str], dict[str, np.ndarray]]): The WOHPs
                arising from all interatomic (off-diagonal) interactions. In general,
                this should come from the :py:meth:`~pengwann.dos.DOS.get_descriptors`
                method.
            num_wann (int): The total number of Wannier functions.

        Returns:
            np.ndarray: The DOE.
        """
        wannier_indices = range(num_wann)

        diagonal_terms = tuple(
            WannierInteraction(i, i, self._R_0, self._R_0) for i in wannier_indices
        )
        diagonal_interaction = (AtomicInteraction(("D1", "D1"), diagonal_terms),)
        diagonal_descriptors = self.get_descriptors(
            diagonal_interaction, calculate_wobi=False
        )

        all_descriptors = descriptors | diagonal_descriptors

        return np.sum(
            [descriptor["WOHP"] for descriptor in all_descriptors.values()], axis=0
        )

    def get_BWDF(
        self,
        integrated_descriptors: dict[tuple[str, str], dict[str, float]],
        geometry: Structure,
    ) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
        """
        Return the necessary data to plot one or more Bond-Weighted Distribution
        Functions (BWDFs).

        Args:
            integrated_descriptors (dict[tuple[str, str], dict[str, float]]): The
                IWOHPs necessary to weight the RDFs.
            geometry (Structure): The Pymatgen Structure object from which to extract
                bond lengths.

        Returns:
            dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]: A dictionary
            containing the necessary inputs to plot the BWDFs. Each key identifies the
            type of bond, whilst the values contain the bond lengths and IWOHPs
            respectively.
        """
        num_wann = len([site for site in geometry if site.species_string == "X0+"])
        distance_matrix = geometry.distance_matrix

        bonds = []
        cumulative_bwdf = {}
        for pair_id, integrals in integrated_descriptors.items():
            id_i, id_j = pair_id
            symbol_i, i = parse_id(id_i)
            symbol_j, j = parse_id(id_j)
            idx_i = i + num_wann - 1
            idx_j = j + num_wann - 1
            r = distance_matrix[idx_i, idx_j]

            bond = (symbol_i, symbol_j)
            if bond not in bonds:
                bonds.append(bond)

                cumulative_bwdf[bond] = ([r], [integrals["IWOHP"]])

            else:
                cumulative_bwdf[bond][0].append(r)
                cumulative_bwdf[bond][1].append(integrals["IWOHP"])

        bwdf = {}
        for bond, data in cumulative_bwdf.items():
            r, weights = data

            bwdf[bond] = (np.array(r), np.array(weights))

        return bwdf
