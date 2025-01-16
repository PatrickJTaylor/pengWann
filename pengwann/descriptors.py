import numpy as np
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from numpy.typing import NDArray
from pengwann.geometry import AtomicInteraction, WannierInteraction
from pengwann.utils import allocate_shared_memory, integrate, parse_id
from pymatgen.core import Structure
from tqdm.auto import tqdm
from typing import NamedTuple, Optional


class DescriptorCalculator:

    _R_0 = np.array((0, 0, 0))

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
    ):
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

    def get_coefficient_matrix(
        self, i: int, R: NDArray[np.int64]
    ) -> NDArray[np.complex128]:
        return (np.exp(1j * 2 * np.pi * self._kpoints @ R))[:, np.newaxis] * np.conj(
            self._u[:, :, i]
        )

    def get_dos_matrix(
        self,
        c_star: NDArray[np.complex128],
        c: NDArray[np.complex128],
        resolve_k: bool = False,
    ) -> NDArray[np.float64]:
        dos_matrix_nk = (
            self._nspin * (c_star * c)[np.newaxis, :, :].real * self._dos_array
        )

        if resolve_k:
            return np.sum(dos_matrix_nk, axis=2)

        else:
            return np.sum(dos_matrix_nk, axis=(1, 2))

    def get_p_ij(
        self, c_star: NDArray[np.complex128], c: NDArray[np.complex128]
    ) -> complex:
        if self._occupation_matrix is None:
            raise ValueError(
                "The occupation matrix is required to calculate elements of the Wannier density matrix."
            )

        p_nk = self._occupation_matrix * c_star * c

        return np.sum(p_nk, axis=(0, 1)) / len(self._kpoints)

    def get_pdos(
        self, geometry: Structure, symbols: tuple[str, ...], resolve_k: bool = False
    ) -> tuple[AtomicInteraction, ...]:
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
                    wannier_interaction = WannierInteraction(i, i, self._R_0, self._R_0)

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
            interaction.dos_matrix = np.zeros(self._energies.shape)

            associated_wannier_interactions = amended_wannier_interactions[
                running_count : running_count + len(interaction.wannier_interactions)
            ]
            for w_interaction in associated_wannier_interactions:
                interaction.dos_matrix += w_interaction.dos_matrix

            interaction.wannier_interactions = associated_wannier_interactions
            running_count += len(interaction.wannier_interactions)

        return interactions

    def assign_populations(
        self,
        interactions: tuple[AtomicInteraction, ...],
        mu: float,
        resolve_orbitals: bool = False,
        valence: Optional[dict[str, int]] = None,
    ) -> None:
        for interaction in interactions:
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
        memory_keys = ["dos_array", "kpoints", "u"]
        shared_data = [self._dos_array, self._kpoints, self._u]
        if calc_wobi:
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
            for w_interaction in amended_wannier_interactions:
                R = tuple((w_interaction.R_2 - w_interaction.R_1).tolist())
                h_ij = self._h[R][w_interaction.i, w_interaction.j].real

                w_interaction.h_ij = h_ij

        running_count = 0
        for interaction in interactions:
            if calc_wohp:
                interaction.wohp = np.zeros(self._energies.shape)

            if calc_wobi:
                interaction.wobi = np.zeros(self._energies.shape)

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
        wannier_indices = range(num_wann)

        diagonal_terms = tuple(
            WannierInteraction(i, i, self._R_0, self._R_0) for i in wannier_indices
        )
        diagonal_interaction = (AtomicInteraction(("D1", "D1"), diagonal_terms),)
        self.assign_descriptors(diagonal_interaction, calc_wobi=False)

        return np.sum(
            [interaction.wohp for interaction in interactions + diagonal_interaction],
            axis=0,
        )

    def get_bwdf(
        self,
        interactions: tuple[AtomicInteraction, ...],
        geometry: Structure,
        r_range: tuple[float, float],
        nbins: int,
    ) -> tuple[NDArray[np.float64], dict[tuple[str, str], NDArray[np.float64]]]:
        num_wann = len([site for site in geometry if site.species_string == "X0+"])
        distance_matrix = geometry.distance_matrix

        r_min, r_max = r_range
        intervals = np.linspace(r_min, r_max, nbins + 1)
        dr = (r_max - r_min) / nbins
        r = intervals[:-1] + dr / 2

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

        return r, bwdf

    @classmethod
    def process_interaction(
        cls,
        interaction: WannierInteraction,
        nspin: int,
        calc_wobi: bool,
        resolve_k: bool,
        memory_metadata: dict[str, tuple[tuple[int, ...], np.dtype]],
    ) -> WannierInteraction:
        kwargs = {"nspin": nspin}
        memory_handles = []
        for memory_key, metadata in memory_metadata.items():
            shape, dtype = metadata

            shared_memory = SharedMemory(name=memory_key)
            buffered_data = np.ndarray(shape, dtype=dtype, buffer=shared_memory.buf)

            kwargs[memory_key] = buffered_data
            memory_handles.append(shared_memory)

        dcalc = cls(**kwargs)

        c_star = np.conj(dcalc.get_coefficient_matrix(interaction.i, interaction.R_1))
        c = dcalc.get_coefficient_matrix(interaction.j, interaction.R_2)

        interaction.dos_matrix = dcalc.get_dos_matrix(c_star, c, resolve_k)

        if calc_wobi:
            interaction.p_ij = dcalc.get_p_ij(c_star, c).real

        for memory_handle in memory_handles:
            memory_handle.close()

        return interaction
