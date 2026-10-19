"""
Shared energy of a community (docs/FORMULATION.md section 9, [S22] eq. 15).
"""

import numpy as np
from numpy.typing import ArrayLike


def shared_energy(export_kwh: ArrayLike, import_kwh: ArrayLike) -> float | np.ndarray:
    """
    Shared energy E^sh = min(Σ_u e^gri_u, Σ_u i^gri_u).

    The sums run over the last axis (the nodes): 1-D inputs give the value of one step, 2-D inputs
    of shape (steps, nodes) give one value per step.

    Args:
        export_kwh: Energy fed into the grid by each node (kWh).
        import_kwh: Energy withdrawn from the grid by each node (kWh).

    Returns:
        float | np.ndarray: Shared energy (kWh), a float for 1-D inputs.
    """
    shared = np.minimum(np.sum(export_kwh, axis=-1), np.sum(import_kwh, axis=-1))
    return float(shared) if np.ndim(shared) == 0 else shared
