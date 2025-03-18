import numpy as np
from numpy.typing import NDArray

def _build_distance_and_image_matrices(
    py_coords: NDArray[np.float64], py_cell: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.int32]]: ...
