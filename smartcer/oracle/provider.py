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
a* = p^cha / P^cha - p^dis / P^dis of every step.
"""

from dataclasses import dataclass
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from smartcer.core.community import Community


TRAJECTORY_FILE = 'trajectory.csv'
OBJECTIVE_FILE = 'objective.csv'
TRAJECTORY_COLUMNS = ('datetime', 'node_id', 'p_cha_kw', 'p_dis_kw', 'soc_kwh')
OBJECTIVE_COLUMNS = ('date', 'objective')
# solver round-off accepted on the sign of the powers and on |a*| ≤ 1
TOLERANCE = 1e-9


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


class OracleProvider:
    """
    The oracle trajectory of a community: optimal actions and states of charge per step.

    Attributes:
        source: Where the data come from (the oracle folder).
        node_ids: Battery nodes, in action order (`community.storage_node_ids`).
        timestamps: Steps of the community period.
        actions: Optimal actions a* = p^cha / P^cha - p^dis / P^dis, shape (n_steps, N), in
            [-1, 1].
        soc_kwh: Oracle state of charge at the end of each step (kWh), shape (n_steps, N).
        daily_objective: MILP objective per calendar day of the period (€), indexed by
            `datetime.date`, or None.
    """

    def __init__(self, data: OracleData, community: Community) -> None:
        """
        Align the oracle data with the steps and battery nodes of a community.

        Rows outside the community period are ignored.

        Args:
            data: Oracle data (`read_oracle`).
            community: The community the oracle was computed for.

        Raises:
            ValueError: If the battery nodes differ, a step of the period has no oracle row, an
                action lies outside [-1, 1] by more than 1e-9, or a day of the period has no
                objective.
        """
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
        batteries = [community.node(i).battery for i in node_ids]
        actions = (p_cha / np.array([b.power_charge_max for b in batteries])
                   - p_dis / np.array([b.power_discharge_max for b in batteries]))
        outside = np.argwhere(np.abs(actions) > 1.0 + TOLERANCE)
        if outside.size:
            t, u = outside[0]
            raise ValueError(f'{data.source}: node {node_ids[u]!r} at {index[t]}: p_cha '
                             f'{p_cha[t, u]} kW, p_dis {p_dis[t, u]} kW exceed the power limits '
                             f'(a* = {actions[t, u]})')
        self.source = data.source
        self.node_ids = tuple(node_ids)
        self.timestamps = np.asarray(community.timestamps)
        self.actions = np.clip(actions, -1.0, 1.0)
        self.soc_kwh = data.soc_kwh.loc[index, list(node_ids)].to_numpy(dtype=np.float64)
        self.daily_objective = None
        if data.daily_objective is not None:
            days: list[datetime.date] = sorted(set(index.date))
            lacking = [d for d in days if d not in data.daily_objective.index]
            if lacking:
                raise ValueError(f'{data.source}: no objective for {len(lacking)} days of the '
                                 f'period, first {lacking[0]}')
            self.daily_objective = data.daily_objective.loc[days]

    @classmethod
    def from_folder(cls, folder: str | Path, community: Community) -> 'OracleProvider':
        """
        Read an oracle folder and align it with a community.

        Args:
            folder: Oracle folder (`read_oracle`).
            community: The community the oracle was computed for.

        Returns:
            OracleProvider: The provider.
        """
        return cls(read_oracle(folder), community)

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
        Oracle state of charge at the end of step t.

        Args:
            t: Step index.

        Returns:
            np.ndarray: One value per battery in action order (kWh, float64 copy).
        """
        return self.soc_kwh[t].copy()
