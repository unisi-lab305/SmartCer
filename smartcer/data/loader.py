"""
Scenario loader (docs/CONFIG.md): `config.json` plus one CSV per node → `Community`.

Every CSV is read once with pandas and converted to numpy arrays; nothing is read at step time.
Validation errors name the node, the file, the column and the first offending row (CSV line
number, header = line 1).

Timestamps: in the 1.0 format they must be sorted, unique, spaced by delta_t and identical across
nodes (errors otherwise). In the 0.6.2 format rows are aligned by position, as 0.6.2 did: every
file must have the same number of rows, the period is selected on the timestamps of the first node
and every node is cut at the same rows, and the community timestamps are those of the first node;
irregularities are reported in one `TimestampWarning`.
"""

import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from smartcer.core.community import Community, TipIncentive, UniformIncentive, tip_coefficient
from smartcer.core.community.incentives import IncentiveModel
from smartcer.core.nodes import Node
from smartcer.core.storage import Battery
from smartcer.data.schema import (
    COLUMN_ALIASES,
    DATETIME_COLUMN,
    FORMAT_VERSION,
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
    NodeConfig,
    ScenarioConfig,
    TimestampWarning,
    parse_config,
)


TimestampLike = str | pd.Timestamp | np.datetime64 | None

# maximum number of irregularities listed per node in a TimestampWarning
_MAX_LISTED = 3


def read_config(path: str | Path) -> ScenarioConfig:
    """
    Read and parse `<path>/config.json`.

    Args:
        path: Scenario folder.

    Returns:
        ScenarioConfig: The parsed configuration.

    Raises:
        FileNotFoundError: If `config.json` does not exist.
        ValueError: If the file is not valid JSON or the configuration is invalid.
    """
    scenario_dir = Path(path)
    config_path = scenario_dir / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'scenario configuration not found: {config_path}')
    try:
        raw = json.loads(config_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as error:
        raise ValueError(f'{config_path}: invalid JSON ({error})') from error
    return parse_config(raw, scenario_dir)


def load_scenario(path: str | Path, start: TimestampLike = None,
                  end: TimestampLike = None) -> Community:
    """
    Load a scenario in the 1.0 or the 0.6.2 format.

    Args:
        path: Scenario folder (with `config.json`).
        start: First timestamp of the period, inclusive; None for the first row.
        end: Last timestamp of the period, inclusive; None for the last row.

    Returns:
        Community: Nodes, incentive model and timestamps of the period.

    Raises:
        FileNotFoundError: If `config.json` or a CSV file does not exist.
        ValueError: If the configuration or the data are invalid (see the module docstring).
    """
    config = read_config(path)
    strict = config.format_version == FORMAT_VERSION
    start_ts = None if start is None else pd.Timestamp(start)
    end_ts = None if end is None else pd.Timestamp(end)
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError(f'start {start_ts} is after end {end_ts}')

    tables: dict[Path, pd.DataFrame] = {}
    selections: list[pd.DataFrame] = []
    irregularities: list[str] = []
    # 0.6.2 format: rows of the period, selected on the first node's timestamps
    positions: pd.Index | None = None
    first_rows = 0
    for node_config in config.nodes:
        table = _read_table(node_config.csv_path, tables, f'node {node_config.node_id!r}')
        label = f'node {node_config.node_id!r} ({_display(node_config.csv_path, config.path)})'
        required = REQUIRED_COLUMNS[node_config.node_type]
        missing = [column for column in required if column not in table.columns]
        if missing:
            raise ValueError(f'{label}: missing columns {missing}; found '
                             f'{[c for c in table.columns if c != DATETIME_COLUMN]}')
        problems = _timestamp_problems(table[DATETIME_COLUMN], config.delta_t)
        if problems and strict:
            raise ValueError(f'{label}: {problems[0]}')
        irregularities += [f'{label}: {problem}' for problem in problems[:_MAX_LISTED]]
        if strict:
            selection = _select_period(table, start_ts, end_ts, label)
        elif positions is None:
            selection = _select_period(table, start_ts, end_ts, label)
            positions, first_rows = selection.index, len(table)
        elif len(table) != first_rows:
            raise ValueError(f'{label}: {len(table)} rows, node {config.nodes[0].node_id!r} has '
                             f'{first_rows}; the 0.6.2 format aligns the nodes by row position, so '
                             f'every file needs the same number of rows')
        else:
            selection = table.loc[positions]
        optional = tuple(c for c in OPTIONAL_COLUMNS[node_config.node_type] if c in table.columns)
        _check_values(selection, required + optional, label)
        selections.append(selection)

    timestamps = _align(config, selections, strict, irregularities)
    if irregularities:
        warnings.warn(
            f'scenario {config.name!r} (0.6.2 format): irregular timestamps, rows are aligned by '
            f'position as in 0.6.2 and the community uses the timestamps of node '
            f'{config.nodes[0].node_id!r}: ' + '; '.join(irregularities),
            TimestampWarning, stacklevel=2,
        )

    nodes = [_build_node(node_config, selection, config)
             for node_config, selection in zip(config.nodes, selections)]
    incentive = _build_incentive(config, nodes, timestamps, tables, start_ts, end_ts)
    return Community(nodes=nodes, incentive=incentive, timestamps=timestamps,
                     delta_t=config.delta_t, name=config.name)


def _display(path: Path, scenario_dir: Path) -> str:
    """
    Path relative to the scenario folder when possible, for messages.

    Args:
        path: File path.
        scenario_dir: Scenario folder.

    Returns:
        str: The path to show.
    """
    try:
        return path.relative_to(scenario_dir).as_posix()
    except ValueError:
        return str(path)


def _read_table(csv_path: Path, cache: dict[Path, pd.DataFrame], owner: str) -> pd.DataFrame:
    """
    Read a CSV file once, rename the 0.6.2 price columns and parse the datetime column.

    Args:
        csv_path: CSV file.
        cache: Tables already read, by resolved path (nodes may share a file).
        owner: Who needs the file, for messages.

    Returns:
        pd.DataFrame: The table, with a parsed `datetime` column and the default index.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the datetime column is missing or unparsable, or a 0.6.2 and a 1.0 column
            name for the same quantity are both present.
    """
    key = csv_path.resolve()
    if key in cache:
        return cache[key]
    if not csv_path.is_file():
        raise FileNotFoundError(f'{owner}: CSV file not found: {csv_path}')
    table = pd.read_csv(csv_path)
    for alias, column in COLUMN_ALIASES.items():
        if alias in table.columns and column in table.columns:
            raise ValueError(f'{csv_path}: columns {alias!r} (0.6.2) and {column!r} (1.0) are '
                             f'both present; keep only {column!r}')
    table = table.rename(columns=COLUMN_ALIASES)
    if DATETIME_COLUMN not in table.columns:
        raise ValueError(f'{csv_path}: missing column {DATETIME_COLUMN!r}; found '
                         f'{list(table.columns)}')
    try:
        table[DATETIME_COLUMN] = pd.to_datetime(table[DATETIME_COLUMN], format='ISO8601')
    except (ValueError, TypeError) as error:
        raise ValueError(f'{csv_path}: column {DATETIME_COLUMN!r} is not a valid date and time '
                         f'({error})') from error
    if table.empty:
        raise ValueError(f'{csv_path}: the file has no rows')
    cache[key] = table
    return table


def _line(index: int) -> int:
    """
    CSV line number of a row index (header = line 1).

    Args:
        index: Row index in the table.

    Returns:
        int: Line number.
    """
    return int(index) + 2


def _timestamp_problems(timestamps: pd.Series, delta_t: float) -> list[str]:
    """
    Unsorted, duplicated or wrongly spaced timestamps of a whole file.

    Args:
        timestamps: Parsed datetime column.
        delta_t: Expected step (hours).

    Returns:
        list[str]: One message per irregular row, in file order (empty if regular).
    """
    expected = pd.Timedelta(hours=delta_t)
    steps = timestamps.diff()
    irregular = (steps != expected) & steps.notna()
    problems = []
    for index in np.flatnonzero(irregular.to_numpy()):
        step = steps.iloc[index]
        if step < pd.Timedelta(0):
            kind = 'timestamps not sorted'
        elif step == pd.Timedelta(0):
            kind = 'duplicated timestamp'
        else:
            kind = f'step of {step} instead of delta_t = {delta_t:g} h'
        problems.append(f'{kind} at line {_line(index)} ({timestamps.iloc[index]}, previous '
                        f'{timestamps.iloc[index - 1]})')
    return problems


def _select_period(table: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None,
                   label: str) -> pd.DataFrame:
    """
    Rows whose timestamp lies in [start, end].

    Args:
        table: Node table.
        start: First timestamp, or None.
        end: Last timestamp, or None.
        label: Node and file, for messages.

    Returns:
        pd.DataFrame: Selected rows, with the original index (for line numbers).

    Raises:
        ValueError: If no row lies in the period.
    """
    mask = np.ones(len(table), dtype=bool)
    if start is not None:
        mask &= (table[DATETIME_COLUMN] >= start).to_numpy()
    if end is not None:
        mask &= (table[DATETIME_COLUMN] <= end).to_numpy()
    if not mask.any():
        raise ValueError(f'{label}: no rows between {start} and {end} (file covers '
                         f'{table[DATETIME_COLUMN].iloc[0]} to {table[DATETIME_COLUMN].iloc[-1]})')
    return table[mask]


def _check_values(selection: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    """
    Required columns must be numeric and finite, energies non-negative.

    Args:
        selection: Rows of the period.
        columns: Required columns.
        label: Node and file, for messages.

    Raises:
        ValueError: On a non-numeric column, a missing value or a negative energy, naming the
            column and the first offending row.
    """
    for column in columns:
        values = pd.to_numeric(selection[column], errors='coerce')
        bad = values.index[~np.isfinite(values.to_numpy(dtype=np.float64))]
        if len(bad):
            row = bad[0]
            raise ValueError(f'{label}: column {column!r} has a missing or non-numeric value at '
                             f'line {_line(row)} ({selection.at[row, DATETIME_COLUMN]}): '
                             f'{selection.at[row, column]!r}')
        if column in ('production', 'consumption'):
            negative = values.index[values.to_numpy() < 0]
            if len(negative):
                row = negative[0]
                raise ValueError(f'{label}: column {column!r} is negative at line {_line(row)} '
                                 f'({selection.at[row, DATETIME_COLUMN]}): {values.at[row]!r}')


def _align(config: ScenarioConfig, selections: list[pd.DataFrame], strict: bool,
           irregularities: list[str]) -> np.ndarray:
    """
    Check that the nodes share their timestamps and return the community timestamps.

    Args:
        config: Scenario configuration.
        selections: Rows of the period of each node, in node order.
        strict: True for the 1.0 format (errors), False for 0.6.2 (alignment by position).
        irregularities: Messages for the TimestampWarning, extended in place (0.6.2).

    Returns:
        np.ndarray: Timestamps of the first node, datetime64.

    Raises:
        ValueError: If the nodes have different numbers of rows, or (1.0) different timestamps.
    """
    reference = selections[0][DATETIME_COLUMN].to_numpy()
    first = config.nodes[0].node_id
    for node_config, selection in zip(config.nodes[1:], selections[1:]):
        label = f'node {node_config.node_id!r} ({_display(node_config.csv_path, config.path)})'
        if len(selection) != len(reference):
            raise ValueError(f'{label}: {len(selection)} rows in the period, node {first!r} has '
                             f'{len(reference)}')
        differ = np.flatnonzero(selection[DATETIME_COLUMN].to_numpy() != reference)
        if differ.size:
            row = differ[0]
            message = (f'timestamps differ from node {first!r} at {differ.size} rows, first at '
                       f'line {_line(selection.index[row])} '
                       f'({selection[DATETIME_COLUMN].iloc[row]} vs '
                       f'{pd.Timestamp(reference[row])})')
            if strict:
                raise ValueError(f'{label}: {message}')
            irregularities.append(f'{label}: {message}')
    return reference


def _build_node(node_config: NodeConfig, selection: pd.DataFrame,
                config: ScenarioConfig) -> Node:
    """
    Node from its configuration and the rows of the period.

    Args:
        node_config: Node configuration.
        selection: Rows of the period.
        config: Scenario configuration (time step and units).

    Returns:
        Node: The node, with energies in kWh per step.

    Raises:
        ValueError: If the storage section is invalid (the message names the node).
    """
    n_steps = len(selection)
    zeros = np.zeros(n_steps)
    to_kwh = config.delta_t if config.units == 'kW' else 1.0

    def column(name: str) -> np.ndarray:
        return selection[name].to_numpy(dtype=np.float64)

    node_type = node_config.node_type
    generation = column('production') * to_kwh if node_type != 'consumer' else zeros
    load = column('consumption') * to_kwh if node_type != 'producer' else zeros
    has_sell = node_type != 'consumer' or 'sell_price' in selection.columns
    sell = column('sell_price') if has_sell else zeros
    buy = column('buy_price') if node_type != 'producer' else zeros

    battery = None
    if node_config.storage is not None:
        try:
            battery = Battery.from_config(node_config.storage)
        except ValueError as error:
            raise ValueError(f'node {node_config.node_id!r}: storage: {error}') from error
    return Node(
        node_id=node_config.node_id, node_type=node_type, delta_t=config.delta_t,
        generation_kwh=generation, load_kwh=load, sell_price=sell, buy_price=buy,
        battery=battery, photovoltaic=node_config.photovoltaic, category=node_config.category,
    )


def _build_incentive(config: ScenarioConfig, nodes: list[Node], timestamps: np.ndarray,
                     tables: dict[Path, pd.DataFrame], start: pd.Timestamp | None,
                     end: pd.Timestamp | None) -> IncentiveModel:
    """
    Incentive model of the scenario.

    `tip`: per-node coefficients from the node's sell price (the zonal price approximation) and its
    photovoltaic section; 0 for nodes without one. `uniform`: constant unit value, or a CSV column
    whose timestamps must match the community's.

    Args:
        config: Scenario configuration.
        nodes: Nodes in scenario order.
        timestamps: Community timestamps.
        tables: CSV cache.
        start: First timestamp of the period, or None.
        end: Last timestamp of the period, or None.

    Returns:
        IncentiveModel: The model.

    Raises:
        ValueError: If a TIP parameter is invalid or the unit value CSV does not match.
    """
    incentive = config.incentive
    if incentive.model == 'tip':
        coefficients = np.zeros((len(timestamps), len(nodes)))
        for i, node in enumerate(nodes):
            if node.node_type == 'consumer' or node.photovoltaic is None:
                continue
            pv: dict[str, Any] = node.photovoltaic
            try:
                coefficients[:, i] = tip_coefficient(
                    node.sell_price, tip_base=pv['tip_base'], tip_threshold=pv['tip_threshold'],
                    fi_zone=pv['fi_zone'], pnrr=pv['pnrr'])
            except ValueError as error:
                raise ValueError(f'node {node.node_id!r}: photovoltaic: {error}') from error
        return TipIncentive(coefficients)

    if incentive.csv_path is None:
        return UniformIncentive(incentive.unit_value, len(timestamps))
    table = _read_table(incentive.csv_path, tables, 'incentive.unit_value')
    label = f'incentive.unit_value ({_display(incentive.csv_path, config.path)})'
    if incentive.column not in table.columns:
        raise ValueError(f'{label}: missing column {incentive.column!r}; found '
                         f'{[c for c in table.columns if c != DATETIME_COLUMN]}')
    selection = _select_period(table, start, end, label)
    _check_values(selection, (incentive.column,), label)
    if not np.array_equal(selection[DATETIME_COLUMN].to_numpy(), timestamps):
        raise ValueError(f'{label}: timestamps differ from those of the nodes in the period')
    return UniformIncentive(selection[incentive.column].to_numpy(dtype=np.float64),
                            len(timestamps))
