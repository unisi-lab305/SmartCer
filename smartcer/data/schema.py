"""
Scenario configuration schema, formats 1.0 and 0.6.2 (docs/CONFIG.md).

`parse_config` turns the content of a scenario's `config.json` into a `ScenarioConfig`; the CSV
files are read by `smartcer.data.loader`.
"""

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
import re
from typing import Any
import warnings

from smartcer.core.nodes import NODE_TYPES


FORMAT_VERSION = '1.0'
LEGACY_FORMAT = '0.6.2'
UNITS = ('kWh', 'kW')
INCENTIVE_MODELS = ('uniform', 'tip')

DATETIME_COLUMN = 'datetime'
# 0.6.2 column name → 1.0 column name
COLUMN_ALIASES = {'price_exp': 'sell_price', 'price_imp': 'buy_price'}
REQUIRED_COLUMNS = {
    'producer': ('production', 'sell_price'),
    'consumer': ('consumption', 'buy_price'),
    'prosumer': ('production', 'consumption', 'sell_price', 'buy_price'),
}

TIP_KEYS = ('tip_base', 'tip_threshold', 'fi_zone', 'pnrr')
# informational photovoltaic keys of the 0.6.2 format, accepted in both formats
PHOTOVOLTAIC_INFO_KEYS = ('pv_power', 'pz')

TOP_KEYS = frozenset({'format_version', 'name', 'description', 'delta_t', 'units', 'incentive',
                      'nodes'})
NODE_KEYS = frozenset({'type', 'category', 'csv', 'photovoltaic', 'storage'})
INCENTIVE_KEYS = frozenset({'model', 'unit_value'})

LEGACY_GROUPS = {'Producers': 'producer', 'Consumers': 'consumer', 'Prosumers': 'prosumer'}
LEGACY_NODE_KEYS = frozenset({'Type', 'Photovoltaic', 'Storage'})

# 0.6.2 rule: a node id with a `_N` suffix reuses the CSV of its base name
_BASE_NAME = re.compile(r'^[a-zA-Z0-9_]+(?=_\d+)')


class TimestampWarning(UserWarning):
    """
    Irregular timestamps in a 0.6.2 scenario, whose rows are aligned by position as in 0.6.2.
    """


@dataclass(frozen=True, slots=True)
class NodeConfig:
    """
    Configuration of one node.

    Attributes:
        node_id: Node identifier.
        node_type: 'producer', 'consumer' or 'prosumer'.
        csv_path: CSV file of the node.
        category: Free-text category (0.6.2 `Type`), or None.
        photovoltaic: TIP parameters of the node's plant, or None.
        storage: Storage section for `Battery.from_config`, or None.
    """

    node_id: str
    node_type: str
    csv_path: Path
    category: str | None
    photovoltaic: dict[str, Any] | None
    storage: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class IncentiveConfig:
    """
    Configuration of the incentive model.

    Attributes:
        model: 'uniform' or 'tip'.
        unit_value: Constant π^inc (€/kWh) of the uniform model, or None.
        csv_path: CSV file with a per-step π^inc for the uniform model, or None.
        column: Column of `csv_path` holding π^inc, or None.
    """

    model: str
    unit_value: float | None = None
    csv_path: Path | None = None
    column: str | None = None


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """
    Parsed scenario configuration.

    Attributes:
        name: Scenario name (the `name` key, or the folder name).
        path: Scenario folder.
        format_version: '1.0' or '0.6.2'.
        delta_t: Duration of a step (hours).
        units: Units of the CSV energies, 'kWh' (per step) or 'kW' (average power).
        incentive: Incentive model configuration.
        nodes: Node configurations in scenario order.
    """

    name: str
    path: Path
    format_version: str
    delta_t: float
    units: str
    incentive: IncentiveConfig
    nodes: tuple[NodeConfig, ...]


def base_csv_name(node_id: str) -> str:
    """
    Base name of a node id under the 0.6.2 `_N` suffix rule ('house_1' → 'house').

    Args:
        node_id: Node identifier.

    Returns:
        str: Base name, or the node id itself without a `_N` suffix.
    """
    match = _BASE_NAME.match(node_id)
    return match.group(0) if match else node_id


def default_csv_path(scenario_dir: Path, node_type: str, node_id: str) -> Path:
    """
    Default CSV of a node, `nodes/<type>s/<base name>.csv` (the 0.6.2 folder rule).

    Args:
        scenario_dir: Scenario folder.
        node_type: 'producer', 'consumer' or 'prosumer'.
        node_id: Node identifier.

    Returns:
        Path: The CSV path.
    """
    return scenario_dir / 'nodes' / f'{node_type}s' / f'{base_csv_name(node_id)}.csv'


def parse_config(raw: Any, scenario_dir: Path) -> ScenarioConfig:
    """
    Parse the content of `config.json` in the 1.0 or the 0.6.2 format.

    The 1.0 format has `format_version: "1.0"`; a configuration without it and with `Producers`,
    `Consumers` or `Prosumers` is read as 0.6.2, with a DeprecationWarning that also lists the keys
    that are not used.

    Args:
        raw: Decoded JSON content.
        scenario_dir: Scenario folder (relative CSV paths are resolved against it).

    Returns:
        ScenarioConfig: The parsed configuration.

    Raises:
        ValueError: If the configuration is invalid; the message names the key and the node.
    """
    if not isinstance(raw, dict):
        raise ValueError(f'config.json must hold a JSON object, got {type(raw).__name__}')
    scenario_dir = Path(scenario_dir)
    if 'format_version' in raw:
        return _parse_v1(raw, scenario_dir)
    if any(group in raw for group in LEGACY_GROUPS):
        return _parse_legacy(raw, scenario_dir)
    raise ValueError(
        f'unrecognised scenario format in {scenario_dir / "config.json"}: expected '
        f'"format_version": "{FORMAT_VERSION}" or the 0.6.2 groups {list(LEGACY_GROUPS)}'
    )


def _parse_v1(raw: dict[str, Any], scenario_dir: Path) -> ScenarioConfig:
    """
    Parse a 1.0 configuration.

    Args:
        raw: Decoded JSON content.
        scenario_dir: Scenario folder.

    Returns:
        ScenarioConfig: The parsed configuration.

    Raises:
        ValueError: On unknown keys or invalid values.
    """
    if raw['format_version'] != FORMAT_VERSION:
        raise ValueError(f'format_version must be "{FORMAT_VERSION}", got '
                         f'{raw["format_version"]!r}')
    _reject_unknown(raw, TOP_KEYS, 'scenario')
    delta_t = raw.get('delta_t', 1.0)
    if isinstance(delta_t, bool) or not isinstance(delta_t, Real) or not delta_t > 0:
        raise ValueError(f'delta_t must be a positive number of hours, got {delta_t!r}')
    units = raw.get('units', 'kWh')
    if units not in UNITS:
        raise ValueError(f'units must be one of {UNITS}, got {units!r}')
    incentive = _parse_incentive(raw.get('incentive', {}), scenario_dir)

    nodes_raw = raw.get('nodes')
    if not isinstance(nodes_raw, dict) or not nodes_raw:
        raise ValueError('nodes must be a non-empty object {node_id: {...}}')
    nodes = []
    for node_id, node in nodes_raw.items():
        if not isinstance(node, dict):
            raise ValueError(f'node {node_id!r}: expected an object, got {type(node).__name__}')
        _reject_unknown(node, NODE_KEYS, f'node {node_id!r}')
        node_type = node.get('type')
        if node_type not in NODE_TYPES:
            raise ValueError(f'node {node_id!r}: type must be one of {NODE_TYPES}, got '
                             f'{node_type!r}')
        csv = node.get('csv')
        if csv is not None and not isinstance(csv, str):
            raise ValueError(f'node {node_id!r}: csv must be a path string, got {csv!r}')
        csv_path = scenario_dir / csv if csv else default_csv_path(scenario_dir, node_type,
                                                                     node_id)
        category = node.get('category')
        if category is not None and not isinstance(category, str):
            raise ValueError(f'node {node_id!r}: category must be a string, got {category!r}')
        photovoltaic = _parse_photovoltaic(node.get('photovoltaic'), node_id, strict=True)
        storage = _parse_storage(node.get('storage'), node_id)
        nodes.append(NodeConfig(node_id, node_type, csv_path, category, photovoltaic, storage))

    _check_tip_parameters(incentive.model, nodes)
    return ScenarioConfig(
        name=str(raw.get('name', scenario_dir.name)), path=scenario_dir,
        format_version=FORMAT_VERSION, delta_t=float(delta_t), units=units, incentive=incentive,
        nodes=tuple(nodes),
    )


def _parse_legacy(raw: dict[str, Any], scenario_dir: Path) -> ScenarioConfig:
    """
    Parse a 0.6.2 configuration: hourly steps, energies in kWh, incentive model `tip`.

    Args:
        raw: Decoded JSON content.
        scenario_dir: Scenario folder.

    Returns:
        ScenarioConfig: The parsed configuration.

    Raises:
        ValueError: On invalid values or repeated node ids.
    """
    unused = [f'{key!r}' for key in raw if key not in LEGACY_GROUPS]
    nodes = []
    for group, node_type in LEGACY_GROUPS.items():
        entries = raw.get(group, {})
        if not isinstance(entries, dict):
            raise ValueError(f'{group} must be an object {{node_id: {{...}}}}')
        for node_id, node in entries.items():
            if not isinstance(node, dict):
                raise ValueError(f'node {node_id!r}: expected an object, got '
                                 f'{type(node).__name__}')
            if any(existing.node_id == node_id for existing in nodes):
                raise ValueError(f'node {node_id!r} appears more than once')
            unused += [f'{node_id}.{key}' for key in node if key not in LEGACY_NODE_KEYS]
            photovoltaic = _parse_photovoltaic(node.get('Photovoltaic'), node_id, strict=False)
            if photovoltaic is not None:
                unused += [f'{node_id}.Photovoltaic.{key}' for key in node['Photovoltaic']
                           if key not in TIP_KEYS + PHOTOVOLTAIC_INFO_KEYS]
            category = node.get('Type')
            nodes.append(NodeConfig(
                node_id=node_id, node_type=node_type,
                csv_path=default_csv_path(scenario_dir, node_type, node_id),
                category=None if category is None else str(category),
                photovoltaic=photovoltaic,
                storage=_parse_storage(node.get('Storage'), node_id),
            ))
    if not nodes:
        raise ValueError('the scenario has no nodes')

    message = (f'scenario {scenario_dir.name!r} uses the 0.6.2 format (no "format_version"): read '
               f'with delta_t = 1 h, units kWh and incentive model "tip"; convert it to format '
               f'{FORMAT_VERSION} (docs/CONFIG.md)')
    if unused:
        message += f'; keys not used: {", ".join(unused)}'
    warnings.warn(message, DeprecationWarning, stacklevel=3)

    incentive = IncentiveConfig(model='tip')
    _check_tip_parameters(incentive.model, nodes)
    return ScenarioConfig(
        name=scenario_dir.name, path=scenario_dir, format_version=LEGACY_FORMAT, delta_t=1.0,
        units='kWh', incentive=incentive, nodes=tuple(nodes),
    )


def _parse_incentive(raw: Any, scenario_dir: Path) -> IncentiveConfig:
    """
    Parse the 1.0 `incentive` section.

    Args:
        raw: Content of the section.
        scenario_dir: Scenario folder.

    Returns:
        IncentiveConfig: The incentive configuration.

    Raises:
        ValueError: On unknown keys, an unknown model or a missing or invalid unit value.
    """
    if not isinstance(raw, dict):
        raise ValueError(f'incentive must be an object, got {type(raw).__name__}')
    _reject_unknown(raw, INCENTIVE_KEYS, 'incentive')
    model = raw.get('model', 'uniform')
    if model not in INCENTIVE_MODELS:
        raise ValueError(f'incentive.model must be one of {INCENTIVE_MODELS}, got {model!r}')
    unit_value = raw.get('unit_value')
    if model == 'tip':
        if unit_value is not None:
            raise ValueError('incentive.unit_value is used only by the "uniform" model')
        return IncentiveConfig(model='tip')
    if isinstance(unit_value, dict):
        if set(unit_value) != {'csv', 'column'} or not all(
                isinstance(v, str) for v in unit_value.values()):
            raise ValueError('incentive.unit_value as a CSV column must be '
                             '{"csv": "<path>", "column": "<name>"}')
        return IncentiveConfig(model='uniform', csv_path=scenario_dir / unit_value['csv'],
                               column=unit_value['column'])
    if isinstance(unit_value, bool) or not isinstance(unit_value, Real):
        raise ValueError('incentive.unit_value (€/kWh) is required by the "uniform" model: a '
                         f'number or {{"csv": ..., "column": ...}}, got {unit_value!r}')
    return IncentiveConfig(model='uniform', unit_value=float(unit_value))


def _parse_photovoltaic(raw: Any, node_id: str, strict: bool) -> dict[str, Any] | None:
    """
    Check a photovoltaic section: TIP parameters must be numbers.

    Args:
        raw: Content of the section, or None.
        node_id: Node identifier, for messages.
        strict: True to reject unknown keys (format 1.0).

    Returns:
        dict[str, Any] | None: A copy of the section, or None.

    Raises:
        ValueError: If the section is not an object, has unknown keys (strict) or a TIP parameter
            is not a number.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f'node {node_id!r}: photovoltaic must be an object')
    if strict:
        _reject_unknown(raw, frozenset(TIP_KEYS + PHOTOVOLTAIC_INFO_KEYS),
                        f'node {node_id!r} photovoltaic')
    for key in TIP_KEYS:
        value = raw.get(key)
        if key in raw and (isinstance(value, bool) or not isinstance(value, Real)):
            raise ValueError(f'node {node_id!r}: photovoltaic.{key} must be a number, got '
                             f'{value!r}')
    return dict(raw)


def _parse_storage(raw: Any, node_id: str) -> dict[str, Any] | None:
    """
    Check that a storage section is an object (its keys are checked by `Battery.from_config`).

    Args:
        raw: Content of the section, or None.
        node_id: Node identifier, for messages.

    Returns:
        dict[str, Any] | None: A copy of the section, or None.

    Raises:
        ValueError: If the section is not an object.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f'node {node_id!r}: storage must be an object')
    return dict(raw)


def _check_tip_parameters(model: str, nodes: list[NodeConfig]) -> None:
    """
    With the `tip` model every producer and prosumer needs the four TIP parameters.

    Args:
        model: Incentive model.
        nodes: Node configurations.

    Raises:
        ValueError: If a producer or prosumer lacks a TIP parameter.
    """
    if model != 'tip':
        return
    for node in nodes:
        if node.node_type == 'consumer':
            continue
        missing = [key for key in TIP_KEYS if key not in (node.photovoltaic or {})]
        if missing:
            raise ValueError(f'node {node.node_id!r}: the "tip" incentive model needs '
                             f'photovoltaic {list(TIP_KEYS)}; missing {missing}')


def _reject_unknown(raw: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    """
    Raise on keys outside the allowed set.

    Args:
        raw: Object to check.
        allowed: Allowed keys.
        where: Location for the message.

    Raises:
        ValueError: If there are unknown keys.
    """
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f'{where}: unknown keys {unknown}; allowed: {sorted(allowed)}')
