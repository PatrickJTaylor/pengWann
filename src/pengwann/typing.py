from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from pengwann.interactions import (
    AtomicInteraction,
    AtomicInteractions,
    WannierInteraction,
)

Hamiltonian: TypeAlias = dict[tuple[int, int, int], NDArray[np.complex128]]
Interactions: TypeAlias = AtomicInteractions | AtomicInteraction | Sequence[WannierInteraction]
WannierTransform: TypeAlias = Callable[[WannierInteraction], WannierInteraction]
