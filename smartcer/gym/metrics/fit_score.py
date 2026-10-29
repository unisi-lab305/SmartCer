"""
Fit Score ([M23] eq. 11, docs/FORMULATION.md section 13) between daily welfare series.
"""

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd


def daily_welfare(community: pd.DataFrame) -> pd.Series:
    """
    Welfare per calendar day of a community table.

    Args:
        community: Community table with the columns `datetime` and `welfare`.

    Returns:
        pd.Series: Sum of W_t per day (€), indexed by date.

    Raises:
        ValueError: If a column is missing.
    """
    missing = [c for c in ('datetime', 'welfare') if c not in community.columns]
    if missing:
        raise ValueError(f'community table: missing columns {missing}')
    days = pd.to_datetime(community['datetime']).dt.date
    return community['welfare'].groupby(days.to_numpy()).sum()


def fit_score(reference: ArrayLike, candidate: ArrayLike) -> float:
    """
    Fit Score F = (1 - Σ_d (T2_d - T1_d)² / Σ_d (T1_d - mean(T1))²) · 100.

    100 when the candidate equals the reference; 0 when it is as far from the reference as the
    reference's mean; negative when it is farther.

    Args:
        reference: Daily series T1 (e.g. the welfare of the MILP oracle).
        candidate: Daily series T2 of the same days (e.g. the welfare of a controller).

    Returns:
        float: F (percentage).

    Raises:
        ValueError: If the series are empty, of different lengths or not finite, or the reference
            is constant (F undefined).
    """
    t1 = np.asarray(reference, dtype=np.float64)
    t2 = np.asarray(candidate, dtype=np.float64)
    if t1.ndim != 1 or t1.shape != t2.shape or t1.size == 0:
        raise ValueError(f'reference and candidate must be non-empty 1-D series of the same '
                         f'length, got shapes {t1.shape} and {t2.shape}')
    if not (np.all(np.isfinite(t1)) and np.all(np.isfinite(t2))):
        raise ValueError('reference and candidate must be finite')
    spread = float(np.sum((t1 - t1.mean()) ** 2))
    if spread == 0.0:
        raise ValueError('the Fit Score is undefined for a constant reference series')
    return (1.0 - float(np.sum((t2 - t1) ** 2)) / spread) * 100.0
