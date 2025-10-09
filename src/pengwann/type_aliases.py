import numpy as np
from numpy.typing import NDArray

from pengwann.interactions import (
    AtomicInteraction,
    AtomicInteractions,
    WannierInteraction,
)

Hamiltonian = dict[tuple[int, int, int], NDArray[np.complex128]]
Interactions = AtomicInteractions | AtomicInteraction | WannierInteraction
