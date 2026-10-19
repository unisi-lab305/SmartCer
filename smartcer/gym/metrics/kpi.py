"""
Key performance indicators of a run, from the community table and the per-node tables of a
`ResultsTracker` (formulas and units in `KPI.md`).

Replaces the 0.6.2 `cer_metrics.scores.compute_kpis`, whose `self_consumption` was the sharing rate
and whose `net_metering` and `energy_balance` had no clear physical meaning (defect S12).
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


# a clipping penalty above this value counts as a penalised step
PENALTY_TOLERANCE = 1e-9

COMMUNITY_COLUMNS = ('reward', 'welfare', 'revenue', 'cost', 'storage_cost', 'incentive',
                     'shared_energy_kwh', 'export_kwh', 'import_kwh', 'penalty')
NODE_COLUMNS = ('charge_kwh', 'discharge_kwh', 'throughput_kwh', 'penalty')


def ratio(numerator: float, denominator: float) -> float:
    """
    numerator / denominator, or 0 when the denominator is 0.

    Args:
        numerator: Numerator.
        denominator: Denominator, ≥ 0.

    Returns:
        float: The ratio.
    """
    return numerator / denominator if denominator > 0.0 else 0.0


def penalty_statistics(penalty: np.ndarray) -> dict[str, float]:
    """
    Mean, maximum and number of penalised steps of a penalty series.

    Args:
        penalty: K_t (community) or k_{u,t} (one battery) per step.

    Returns:
        dict[str, float]: `penalty_mean`, `penalty_max`, `penalty_steps` (steps with a penalty
        above `PENALTY_TOLERANCE`).
    """
    return {
        'penalty_mean': float(penalty.mean()) if penalty.size else 0.0,
        'penalty_max': float(penalty.max()) if penalty.size else 0.0,
        'penalty_steps': int((penalty > PENALTY_TOLERANCE).sum()),
    }


def compute_kpis(community: pd.DataFrame, nodes: Mapping[str, pd.DataFrame],
                 capacities: Mapping[str, float],
                 soc_min: Mapping[str, float] | None = None) -> dict[str, Any]:
    """
    KPIs of one run.

    Args:
        community: Community table (one row per step) with the columns of `COMMUNITY_COLUMNS`.
        nodes: Node id → node table; only the battery nodes of `capacities` are read, with the
            columns of `NODE_COLUMNS`.
        capacities: Battery node id → capacity S^max (kWh).
        soc_min: Battery node id → minimum state of charge S^min (kWh), for the cycles on the
            usable capacity S^max - S^min; nodes not listed (or None) have S^min = 0.

    Returns:
        dict[str, Any]: `steps`; sums of `reward`, `welfare`, `revenue`, `cost`, `storage_cost`,
        `incentive` (€) and `shared_energy_kwh`, `export_kwh`, `import_kwh` (kWh);
        `sharing_rate` = shared / exported and `coverage_rate` = shared / imported (fractions);
        `throughput_kwh`, `equivalent_full_cycles` (on S^max) and `efc_usable` (on S^max - S^min)
        of all batteries; `penalty_mean`, `penalty_max`, `penalty_steps` of K_t; `batteries`: per
        battery node, `charge_kwh`, `discharge_kwh`, `throughput_kwh`, `equivalent_full_cycles`,
        `efc_usable` and the penalty statistics of k_u.

    Raises:
        ValueError: If a column or a battery node table is missing, a capacity is not positive or
            a minimum state of charge is outside [0, S^max).
    """
    missing = [c for c in COMMUNITY_COLUMNS if c not in community.columns]
    if missing:
        raise ValueError(f'community table: missing columns {missing}')
    totals = {column: float(community[column].sum()) for column in COMMUNITY_COLUMNS[:-1]}
    kpis: dict[str, Any] = {'steps': len(community), **totals}
    kpis['sharing_rate'] = ratio(totals['shared_energy_kwh'], totals['export_kwh'])
    kpis['coverage_rate'] = ratio(totals['shared_energy_kwh'], totals['import_kwh'])

    soc_min = soc_min or {}
    batteries: dict[str, dict[str, float]] = {}
    usable: dict[str, float] = {}
    for node_id, capacity in capacities.items():
        if capacity <= 0.0:
            raise ValueError(f'battery node {node_id!r}: capacity must be > 0, got {capacity}')
        minimum = soc_min.get(node_id, 0.0)
        if not 0.0 <= minimum < capacity:
            raise ValueError(f'battery node {node_id!r}: minimum state of charge must be in '
                             f'[0, {capacity}) kWh, got {minimum}')
        usable[node_id] = capacity - minimum
        if node_id not in nodes:
            raise ValueError(f'no table for battery node {node_id!r}')
        table = nodes[node_id]
        missing = [c for c in NODE_COLUMNS if c not in table.columns]
        if missing:
            raise ValueError(f'node table {node_id!r}: missing columns {missing}')
        throughput = float(table['throughput_kwh'].sum())
        batteries[node_id] = {
            'charge_kwh': float(table['charge_kwh'].sum()),
            'discharge_kwh': float(table['discharge_kwh'].sum()),
            'throughput_kwh': throughput,
            'equivalent_full_cycles': throughput / (2.0 * capacity),
            'efc_usable': throughput / (2.0 * usable[node_id]),
            **penalty_statistics(table['penalty'].to_numpy(dtype=np.float64)),
        }
    throughput = sum(b['throughput_kwh'] for b in batteries.values())
    kpis['throughput_kwh'] = throughput
    kpis['equivalent_full_cycles'] = ratio(throughput, 2.0 * sum(capacities.values()))
    kpis['efc_usable'] = ratio(throughput, 2.0 * sum(usable.values()))
    kpis.update(penalty_statistics(community['penalty'].to_numpy(dtype=np.float64)))
    kpis['batteries'] = batteries
    return kpis
