import numpy as np
from pymatgen.core import Structure


def assign_wannier_centres(geometry: Structure) -> None:
    """
    Assign Wannier centres to atoms based on a closest distance
    criterion.

    Args:
        geometry (Structure): A Pymatgen Structure object containing
            the structure itself as well as the positions of the
            Wannier centres (as 'X' atoms).

    Returns:
        None
    """
    wannier_indices, atom_indices = [], []
    for idx in range(len(geometry)):
        symbol = geometry[idx].species_string

        if symbol == 'X0+':
            wannier_indices.append(idx)

        else:
            atom_indices.append(idx)

    distance_matrix = geometry.distance_matrix

    wannier_centres_list: list[list[int]] = [
        [] for idx in range(len(geometry))
    ]
    for i in wannier_indices:
        min_distance, min_idx = np.inf, 2 * len(geometry)

        for j in atom_indices:
            distance = distance_matrix[i, j]

            if distance < min_distance:
                min_distance = distance
                min_idx = j

        wannier_centres_list[i].append(min_idx)
        wannier_centres_list[min_idx].append(i)

    wannier_centres = tuple(
        [tuple(indices) for indices in wannier_centres_list]
    )
    geometry.add_site_property('wannier_centres', wannier_centres)


def get_atom_indices(
    geometry: Structure, symbols: tuple[str, ...]
) -> dict[str, tuple[int, ...]]:
    """
    Categorise all site indices of a Pymatgen Structure object
    according to the atomic species.

    Args:
        geometry (Structure): The Pymatgen Structure object.
        symbols (tuple[str, ...]): The atomic species to associate
            indices with.

    Returns:
        dict[str, tuple[int, ...]]: The site indices categorised by
            atomic species (as dictionary keys).
    """
    atom_indices_list: dict[str, list[int]] = {}
    for symbol in symbols:
        atom_indices_list[symbol] = []

    for idx, atom in enumerate(geometry):
        symbol = atom.species_string
        if symbol in symbols:
            atom_indices_list[symbol].append(idx)

    atom_indices = {}
    for symbol, indices in atom_indices_list.items():
        atom_indices[symbol] = tuple(indices)

    return atom_indices


def get_occupation_matrix(
    fermi_level: float, eigenvalues: np.ndarray, electrons_per_state: int
) -> np.ndarray:
    """
    Calculate the occupation matrix.

    Args:
        fermi_level (float): The Fermi level.
        eigenvalues (np.ndarray): The Kohn-Sham eigenvalues.
        electrons_per_state (int): The number of electrons per occupied
            Kohn-Sham state.

    Returns:
        f (np.ndarray): The occupation matrix.
    """
    f_n = np.array(
        [
            0 if True in row else electrons_per_state
            for row in eigenvalues > fermi_level
        ]
    )

    f = np.broadcast_to(f_n, (eigenvalues.shape[1], eigenvalues.shape[0]))

    return f
