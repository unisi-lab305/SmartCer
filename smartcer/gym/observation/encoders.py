"""
Vectorised feature encoders for the observation.

Adapted from storage-node-environment (`storage_node_env/gym/preprocessing/encoders.py`): plain
functions on numpy arrays, applied once to the whole scenario.
"""

import numpy as np
from numpy.typing import ArrayLike


def cyclical(values: ArrayLike, period: float) -> np.ndarray:
    """
    Cyclical encoding (sin, cos) of 2π · value / period.

    Args:
        values: Values, e.g. the hour of day.
        period: Period of the values, e.g. 24.

    Returns:
        np.ndarray: Shape (n, 2), columns sin and cos, in [-1, 1].
    """
    angle = 2.0 * np.pi * np.asarray(values, dtype=np.float64) / period
    return np.column_stack([np.sin(angle), np.cos(angle)])


def one_hot(indices: ArrayLike, n_classes: int) -> np.ndarray:
    """
    One-hot encoding of integer classes.

    Args:
        indices: Class indices in [0, n_classes).
        n_classes: Number of classes.

    Returns:
        np.ndarray: Shape (n, n_classes) with one 1 per row.

    Raises:
        ValueError: If an index lies outside [0, n_classes).
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size and (indices.min() < 0 or indices.max() >= n_classes):
        raise ValueError(f'class indices must lie in [0, {n_classes})')
    return np.eye(n_classes)[indices]


def min_max(values: ArrayLike, low: float, high: float) -> np.ndarray:
    """
    Min-max scaling to [0, 1]; a constant series (high == low) maps to 0.

    Args:
        values: Values.
        low: Value mapped to 0.
        high: Value mapped to 1.

    Returns:
        np.ndarray: Scaled values.
    """
    values = np.asarray(values, dtype=np.float64)
    if high <= low:
        return np.zeros_like(values)
    return (values - low) / (high - low)
