from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from pengwann.interactions import (
    AtomicInteraction,
    AtomicInteractions,
    WannierInteraction,
)

Hamiltonian = dict[tuple[int, int, int], NDArray[np.complex128]]
Interactions = AtomicInteractions | AtomicInteraction | Sequence[WannierInteraction]
Process = Callable[[WannierInteraction], WannierInteraction]
