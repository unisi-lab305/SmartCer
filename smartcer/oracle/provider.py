"""
MILP oracle: the optimal battery trajectories of the DIISM MILP ([S22]), for the critic.

Oracle data enter only `info['privileged']` of `SmartCerEnv` and the `ReplayController`
(docs/refactor/REFACTOR_v1.md section 4). An oracle folder holds (provisional format, pending the
DIISM answers; `read_oracle` is the only function that knows it):

- `trajectory.csv`: one row per step and battery node, columns `datetime, node_id, p_cha_kw,
  p_dis_kw, soc_kwh`. `datetime` is the start of the step, without time zone, as in the scenario
  CSV files; `p_cha_kw` and `p_dis_kw` are the average charge and discharge power of the step on
  the node side (docs/FORMULATION.md section 4), ≥ 0; `soc_kwh` is the state of charge at the end
  of the step. Other columns are ignored.
- `objective.csv` (optional): one row per calendar day, columns `date, objective`: the MILP
  objective of the day as a welfare in € (section 10, larger is better). Other columns are ignored.

`OracleProvider` aligns the data with a community and computes the optimal actions
a* = p^cha / P^cha - p^dis / P^dis of every step and the oracle state of charge at the start of
every step. The file has no initial state: the MILP's S^ini is inferred from the first row of the
file (section 5 solved for the state before the step) and must equal the scenario's
`soc_initial`. Rows that charge and discharge in the same step are counted and reported with an
`OracleWarning`: a* is their net value, so the replay cannot reproduce them.
"""

from dataclasses import dataclass
import datetime
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from smartcer.core.community import Community
from smartcer.core.storage import Battery


TRAJECTORY_FILE = 'trajectory.csv'
OBJECTIVE_FILE = 'objective.csv'
TRAJECTORY_COLUMNS = ('datetime', 'node_id', 'p_cha_kw', 'p_dis_kw', 'soc_kwh')
OBJECTIVE_COLUMNS = ('date', 'objective')
# solver round-off accepted on the sign of the powers and on |a*| ≤ 1
TOLERANCE = 1e-9
# default largest difference between the inferred S^ini of the MILP and the scenario's
# soc_initial (kWh); a file that rounds the state of charge still passes
SOC_INITIAL_TOLERANCE_KWH = 1e-3


class OracleWarning(UserWarning):
    """
    The oracle data are readable but the replay cannot reproduce them exactly (e.g. simultaneous
    charge and discharge).
    """


@dataclass(frozen=True)
class OracleData:
    """
    MILP outputs, independent of the file format.

    The three tables have the same sorted timestamps (rows) and node ids (columns).

    Attributes:
        p_cha_kw: Charge power of each step, node side (kW), ≥ 0.
        p_dis_kw: Discharge power of each step, node side (kW), ≥ 0.
        soc_kwh: State of charge at the end of each step (kWh).
        daily_objective: Objective per calendar day (€, welfare), indexed by `datetime.date`, or
            None.
        source: Where the data come from (the oracle folder).
    """

    p_cha_kw: pd.DataFrame
    p_dis_kw: pd.DataFrame
    soc_kwh: pd.DataFrame
    daily_objective: pd.Series | None
    source: str


def _line(row: int) -> int:
    """
    CSV line number of a row index (header = line 1).

    Args:
        row: Row index of the table.

    Returns:
        int: Line number.
    """
    return row + 2


def _numeric(table: pd.DataFrame, column: str, path: Path) -> pd.Series:
    """
    A column as finite floats.

    Args:
        table: Table read from `path`.
        column: Column name.
        path: File, for the error messages.

    Returns:
        pd.Series: The values.

    Raises:
        ValueError: At the first value that is missing or not a finite number.
    """
    values = pd.to_numeric(table[column], errors='coerce').astype(np.float64)
    bad = np.flatnonzero(~np.isfinite(values.to_numpy()))
    if bad.size:
        row = int(bad[0])
        raise ValueError(f'{path}: column {column!r} is not a finite number at line '
                         f'{_line(row)}: {table.at[row, column]!r}')
    return values


def _read_table(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """
    Read a CSV file and check its columns.

    Args:
        path: CSV file.
        columns: Required columns.

    Returns:
        pd.DataFrame: The table.

    Raises:
        ValueError: If a column is missing or the file has no rows.
    """
    table = pd.read_csv(path, dtype=str)
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError(f'{path}: missing columns {missing} (expected {list(columns)})')
    if table.empty:
        raise ValueError(f'{path}: the file has no rows')
    return table


def _timestamps(table: pd.DataFrame, column: str, path: Path) -> pd.Series:
    """
    A column as time-zone-naive timestamps.

    Args:
        table: Table read from `path`.
        column: Column name.
        path: File, for the error messages.

    Returns:
        pd.Series: The timestamps.

    Raises:
        ValueError: At the first value that is not a timestamp, or if the timestamps carry a time
            zone.
    """
    try:
        times = pd.to_datetime(table[column], errors='coerce')
    except (TypeError, ValueError) as error:
        raise ValueError(f'{path}: column {column!r} mixes time zones or formats: {error}') \
            from error
    if times.dt.tz is not None:
        raise ValueError(f'{path}: column {column!r} has a time zone; write local times without '
                         f'zone, as in the scenario CSV files')
    bad = np.flatnonzero(times.isna().to_numpy())
    if bad.size:
        row = int(bad[0])
        raise ValueError(f'{path}: column {column!r} is not a timestamp at line {_line(row)}: '
                         f'{table.at[row, column]!r}')
    return times


def read_oracle(folder: str | Path) -> OracleData:
    """
    Read the MILP outputs of an oracle folder (the adapter of the oracle file format).

    This is the only function that knows the file format (module docstring); when the format of
    the DIISM files changes, only this function changes.

    Args:
        folder: Oracle folder with `trajectory.csv` and optionally `objective.csv`.

    Returns:
        OracleData: Powers and states of charge per step and node, daily objective.

    Raises:
        FileNotFoundError: If `trajectory.csv` is missing.
        ValueError: On a missing column, a value that is not a timestamp or a finite number, a
            power below -1e-9 kW, a duplicated (datetime, node_id) pair, a step that lacks a node,
            or a duplicated day in `objective.csv`; the message names the file and the first
            offending line.
    """
    folder = Path(folder)
    path = folder / TRAJECTORY_FILE
    if not path.is_file():
        raise FileNotFoundError(f'oracle folder {folder}: {TRAJECTORY_FILE} missing')
    table = _read_table(path, TRAJECTORY_COLUMNS)
    data = pd.DataFrame({'datetime': _timestamps(table, 'datetime', path),
                         'node_id': table['node_id']})
    bad = np.flatnonzero(data['node_id'].isna().to_numpy())
    if bad.size:
        raise ValueError(f'{path}: column \'node_id\' is empty at line {_line(int(bad[0]))}')
    for column in ('p_cha_kw', 'p_dis_kw', 'soc_kwh'):
        values = _numeric(table, column, path)
        if column != 'soc_kwh':
            negative = np.flatnonzero((values < -TOLERANCE).to_numpy())
            if negative.size:
                row = int(negative[0])
                raise ValueError(f'{path}: column {column!r} is negative at line {_line(row)}: '
                                 f'{values.at[row]!r}')
            # solver round-off below zero
            values = values.clip(lower=0.0)
        data[column] = values
    duplicated = np.flatnonzero(data.duplicated(['datetime', 'node_id']).to_numpy())
    if duplicated.size:
        row = int(duplicated[0])
        raise ValueError(f'{path}: line {_line(row)} repeats datetime {data.at[row, "datetime"]} '
                         f'and node {data.at[row, "node_id"]!r}')
    wide = data.pivot(index='datetime', columns='node_id').sort_index()
    gaps = wide['soc_kwh'].isna()
    if gaps.to_numpy().any():
        stamp = gaps.index[gaps.any(axis=1).to_numpy()][0]
        nodes = list(gaps.columns[gaps.loc[stamp].to_numpy()])
        raise ValueError(f'{path}: no row for nodes {nodes} at {stamp}; every step needs one row '
                         f'per battery node')

    objective = None
    path = folder / OBJECTIVE_FILE
    if path.is_file():
        table = _read_table(path, OBJECTIVE_COLUMNS)
        days = _timestamps(table, 'date', path).dt.date
        repeated = np.flatnonzero(days.duplicated().to_numpy())
        if repeated.size:
            row = int(repeated[0])
            raise ValueError(f'{path}: line {_line(row)} repeats the day {days.at[row]}')
        objective = pd.Series(_numeric(table, 'objective', path).to_numpy(),
                              index=pd.Index(days, name='date'), name='objective').sort_index()
    return OracleData(wide['p_cha_kw'], wide['p_dis_kw'], wide['soc_kwh'], objective, str(folder))


def _check_initial_soc(data: OracleData, node_id: str, battery: Battery, delta_t: float,
                       tolerance_kwh: float) -> None:
    """
    Check the MILP's initial state of charge of a battery against the scenario's.

    The state before the first step of the file, after self-discharge, is
    s̃ = s - η^cha p^cha Δt + p^dis Δt / η^dis (docs/FORMULATION.md section 5); it must equal
    max(S^min, (1 - α)^Δt S^ini) (S^ini without self-discharge) within `tolerance_kwh`.

    Args:
        data: Oracle data.
        node_id: Battery node.
        battery: The node's battery.
        delta_t: Duration of a step (hours).
        tolerance_kwh: Largest accepted difference (kWh).

    Raises:
        ValueError: If the two states differ, naming the node and both values.
    """
    first = data.soc_kwh.index[0]
    inferred = (data.soc_kwh.at[first, node_id]
                - battery.efficiency_charge * data.p_cha_kw.at[first, node_id] * delta_t
                + data.p_dis_kw.at[first, node_id] * delta_t / battery.efficiency_discharge)
    expected = battery.soc_initial_kwh if battery.alpha == 0.0 else max(
        battery.soc_min_kwh, (1.0 - battery.alpha) ** delta_t * battery.soc_initial_kwh)
    if abs(inferred - expected) > tolerance_kwh:
        discharged = '' if battery.alpha == 0.0 else f' ({expected:.6f} kWh after self-discharge)'
        raise ValueError(
            f'{data.source}: node {node_id!r}: the MILP starts from {inferred:.6f} kWh (inferred '
            f'from its first step {first}), the scenario from soc_initial {battery.soc_initial} = '
            f'{battery.soc_initial_kwh:.6f} kWh{discharged}; set soc_initial in the scenario to '
            f'the MILP\'s initial state of charge')


class OracleProvider:
    """
    The oracle trajectory of a community: optimal actions and states of charge per step.

    Attributes:
        source: Where the data come from (the oracle folder).
        first_timestamp: First step of the oracle file, whose state before the step is S^ini.
        node_ids: Battery nodes, in action order (`community.storage_node_ids`).
        timestamps: Steps of the community period.
        actions: Optimal actions a* = p^cha / P^cha - p^dis / P^dis, shape (n_steps, N), in
            [-1, 1].
        soc_kwh: Oracle state of charge at the end of each step (kWh), shape (n_steps, N).
        soc_start_kwh: Oracle state of charge at the start of each step (kWh), shape
            (n_steps, N): the scenario's S^ini at the first step of the oracle file, else the
            state at the end of the previous step.
        daily_objective: MILP objective per calendar day of the period (€), indexed by
            `datetime.date`, or None.
        simultaneous_rows: Rows of the period (step and node) with both p^cha > 0 and p^dis > 0.
    """

    def __init__(self, data: OracleData, community: Community,
                 soc_initial_tolerance_kwh: float = SOC_INITIAL_TOLERANCE_KWH) -> None:
        """
        Align the oracle data with the steps and battery nodes of a community.

        Rows outside the community period are ignored. Rows of the period that charge and
        discharge in the same step are counted (`simultaneous_rows`) and reported with one
        `OracleWarning` (count and first timestamp).

        The MILP's initial state of charge is inferred from the first row of the file, with the
        time step of the community: s̃ = s - η^cha p^cha Δt + p^dis Δt / η^dis is the state before
        that step after self-discharge, and must equal max(S^min, (1 - α)^Δt S^ini) of the
        scenario (S^ini without self-discharge) within `soc_initial_tolerance_kwh` (default
        1e-3 kWh, `SOC_INITIAL_TOLERANCE_KWH`).

        Args:
            data: Oracle data (`read_oracle`).
            community: The community the oracle was computed for.
            soc_initial_tolerance_kwh: Largest accepted difference between the inferred initial
                state of charge and the scenario's (kWh), ≥ 0.

        Raises:
            ValueError: If the battery nodes differ, a step of the period has no oracle row, an
                action lies outside [-1, 1] by more than 1e-9, the inferred initial state of
                charge differs from the scenario's `soc_initial`, the period starts after the
                first step of the file but the step before it has no row, a day of the period
                has no objective, or the tolerance is negative or not finite.
        """
        if not (math.isfinite(soc_initial_tolerance_kwh) and soc_initial_tolerance_kwh >= 0.0):
            raise ValueError(f'soc_initial_tolerance_kwh must be a finite number ≥ 0, got '
                             f'{soc_initial_tolerance_kwh!r}')
        node_ids = community.storage_node_ids
        columns = list(data.p_cha_kw.columns)
        missing = [i for i in node_ids if i not in columns]
        extra = [i for i in columns if i not in node_ids]
        if missing or extra:
            raise ValueError(f'{data.source}: the oracle nodes must be the battery nodes '
                             f'{list(node_ids)}; missing {missing}, not battery nodes {extra}')
        index = pd.DatetimeIndex(community.timestamps)
        absent = index[~index.isin(data.p_cha_kw.index)]
        if len(absent):
            raise ValueError(f'{data.source}: no oracle rows for {len(absent)} steps of the period '
                             f'{index[0]} → {index[-1]}, first {absent[0]}')
        p_cha = data.p_cha_kw.loc[index, list(node_ids)].to_numpy(dtype=np.float64)
        p_dis = data.p_dis_kw.loc[index, list(node_ids)].to_numpy(dtype=np.float64)
        both = (p_cha > 0.0) & (p_dis > 0.0)
        self.simultaneous_rows = int(np.count_nonzero(both))
        if self.simultaneous_rows:
            t, u = np.argwhere(both)[0]
            warnings.warn(f'{data.source}: {self.simultaneous_rows} rows of the period charge and '
                          f'discharge in the same step (p_cha_kw > 0 and p_dis_kw > 0), first at '
                          f'{index[t]} (node {node_ids[u]!r}); a* is their net value, so the '
                          f'replay cannot reproduce them', OracleWarning, stacklevel=2)
        batteries = [community.node(i).battery for i in node_ids]
        actions = (p_cha / np.array([b.power_charge_max for b in batteries])
                   - p_dis / np.array([b.power_discharge_max for b in batteries]))
        outside = np.argwhere(np.abs(actions) > 1.0 + TOLERANCE)
        if outside.size:
            t, u = outside[0]
            raise ValueError(f'{data.source}: node {node_ids[u]!r} at {index[t]}: p_cha '
                             f'{p_cha[t, u]} kW, p_dis {p_dis[t, u]} kW exceed the power limits '
                             f'(a* = {actions[t, u]})')
        delta_t = community.delta_t
        first = data.soc_kwh.index[0]
        for node_id in node_ids:
            _check_initial_soc(data, node_id, community.node(node_id).battery, delta_t,
                               soc_initial_tolerance_kwh)
        self.source = data.source
        self.node_ids = tuple(node_ids)
        self.first_timestamp = first.to_datetime64()
        self.timestamps = np.asarray(community.timestamps)
        self.actions = np.clip(actions, -1.0, 1.0)
        self.soc_kwh = data.soc_kwh.loc[index, list(node_ids)].to_numpy(dtype=np.float64)
        self.soc_start_kwh = np.empty_like(self.soc_kwh)
        self.soc_start_kwh[1:] = self.soc_kwh[:-1]
        if index[0] == first:
            self.soc_start_kwh[0] = [b.soc_initial_kwh for b in batteries]
        else:
            before = index[0] - pd.Timedelta(hours=delta_t)
            if before not in data.soc_kwh.index:
                raise ValueError(f'{data.source}: the period starts at {index[0]}, after the first '
                                 f'oracle step {first}, but the file has no row for the step '
                                 f'before it ({before}), which gives the initial state of charge')
            self.soc_start_kwh[0] = data.soc_kwh.loc[before, list(node_ids)].to_numpy(
                dtype=np.float64)
        self.daily_objective = None
        if data.daily_objective is not None:
            days: list[datetime.date] = sorted(set(index.date))
            lacking = [d for d in days if d not in data.daily_objective.index]
            if lacking:
                raise ValueError(f'{data.source}: no objective for {len(lacking)} days of the '
                                 f'period, first {lacking[0]}')
            self.daily_objective = data.daily_objective.loc[days]

    @classmethod
    def from_folder(cls, folder: str | Path, community: Community,
                    soc_initial_tolerance_kwh: float = SOC_INITIAL_TOLERANCE_KWH
                    ) -> 'OracleProvider':
        """
        Read an oracle folder and align it with a community.

        Args:
            folder: Oracle folder (`read_oracle`).
            community: The community the oracle was computed for.
            soc_initial_tolerance_kwh: Largest accepted difference between the inferred initial
                state of charge and the scenario's (kWh), ≥ 0.

        Returns:
            OracleProvider: The provider.
        """
        return cls(read_oracle(folder), community, soc_initial_tolerance_kwh)

    def __repr__(self) -> str:
        return f'OracleProvider({self.source!r}, {len(self.node_ids)} batteries)'

    def action(self, t: int) -> np.ndarray:
        """
        Optimal actions of step t.

        Args:
            t: Step index.

        Returns:
            np.ndarray: a*_t, one value per battery in action order (float64 copy).
        """
        return self.actions[t].copy()

    def soc(self, t: int) -> np.ndarray:
        """
        Oracle state of charge at the start of step t (before a*_t), comparable with the state of
        charge an agent observes for step t.

        Args:
            t: Step index.

        Returns:
            np.ndarray: One value per battery in action order (kWh, float64 copy).
        """
        return self.soc_start_kwh[t].copy()
