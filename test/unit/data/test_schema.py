"""
Unit tests of `smartcer.data.schema` (docs/CONFIG.md).
"""

from pathlib import Path
from typing import Any
import warnings

import pytest

from smartcer.data.schema import base_csv_name, default_csv_path, parse_config


pytestmark = pytest.mark.unit

PV = {'tip_base': 0.08, 'tip_threshold': 0.12, 'fi_zone': 0.01, 'pnrr': 0.0}


def v1(**overrides: Any) -> dict[str, Any]:
    """
    Minimal valid 1.0 configuration with one prosumer, overridable.

    Args:
        **overrides: Top-level keys that replace the defaults.

    Returns:
        dict[str, Any]: The configuration.
    """
    config = {'format_version': '1.0', 'incentive': {'model': 'uniform', 'unit_value': 0.1},
              'nodes': {'shop': {'type': 'prosumer'}}}
    return {**config, **overrides}


@pytest.mark.parametrize(
    ('node_id', 'base'),
    [('house_1', 'house'), ('utente_singolo_2', 'utente_singolo'), ('pv_storage', 'pv_storage'),
     ('a_1_2', 'a_1')],
)
def test_base_csv_name_follows_the_suffix_rule(node_id: str, base: str) -> None:
    """
    Test the 0.6.2 `_N` rule: a numeric suffix is removed, other names are kept.

    Args:
        node_id: Node identifier.
        base: Expected base name.
    """
    assert base_csv_name(node_id) == base


def test_v1_defaults_and_csv_paths() -> None:
    """
    Test the 1.0 defaults (delta_t 1, kWh, folder rule for the CSV) and an explicit csv path.
    """
    config = parse_config(v1(nodes={'shop_1': {'type': 'prosumer'},
                                    'pv': {'type': 'producer', 'csv': 'data/pv.csv'}}), Path('s'))

    assert (config.format_version, config.delta_t, config.units) == ('1.0', 1.0, 'kWh')
    assert config.incentive.unit_value == 0.1
    assert config.nodes[0].csv_path == default_csv_path(Path('s'), 'prosumer', 'shop_1')
    assert config.nodes[0].csv_path == Path('s/nodes/prosumers/shop.csv')
    assert config.nodes[1].csv_path == Path('s/data/pv.csv')


def test_v1_incentive_from_csv_column() -> None:
    """
    Test a uniform unit value given as a CSV column.
    """
    config = parse_config(v1(incentive={'unit_value': {'csv': 'pi.csv', 'column': 'value'}}),
                          Path('s'))

    assert config.incentive.model == 'uniform'
    assert (config.incentive.csv_path, config.incentive.column) == (Path('s/pi.csv'), 'value')


@pytest.mark.parametrize(
    ('overrides', 'match'),
    [
        ({'format_version': '2.0'}, 'format_version'),
        ({'delta_t': 0}, 'delta_t'),
        ({'units': 'MWh'}, 'units'),
        ({'incentive': {'model': 'uniform'}}, 'unit_value'),
        ({'incentive': {'model': 'feed-in'}}, 'incentive.model'),
        ({'nodes': {'shop': {'type': 'prosumer', 'color': 'red'}}}, 'unknown keys'),
        ({'nodes': {'shop': {'type': 'storage'}}}, 'type'),
        ({'incentive': {'model': 'tip'}}, 'needs photovoltaic'),
        ({'nodes': {}}, 'nodes'),
        ({'tariff': 1}, 'unknown keys'),
    ],
)
def test_v1_errors(overrides: dict[str, Any], match: str) -> None:
    """
    Test that invalid 1.0 configurations raise ValueError naming the key.

    Args:
        overrides: Invalid top-level keys.
        match: Text expected in the message.
    """
    with pytest.raises(ValueError, match=match):
        parse_config(v1(**overrides), Path('s'))


def test_v1_tip_model_with_photovoltaic_emits_no_warning() -> None:
    """
    Test a 1.0 tip scenario: no DeprecationWarning, photovoltaic section kept.
    """
    raw = v1(incentive={'model': 'tip'}, nodes={'shop': {'type': 'prosumer', 'photovoltaic': PV}})

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        config = parse_config(raw, Path('s'))

    assert config.nodes[0].photovoltaic == PV


def test_legacy_format_is_mapped_with_warning() -> None:
    """
    Test the 0.6.2 format: groups → node types in group order, Type → category, tip model, hourly
    kWh, unused keys reported in the DeprecationWarning.
    """
    raw = {
        'Prosumers': {'shop': {'Type': 'retail', 'Photovoltaic': {**PV, 'pv_power': 10},
                               'Extra': 1}},
        'Producers': {'pv': {'Type': 'plant', 'Photovoltaic': PV,
                             'Storage': {'capacity': 10, 'rated_power': 5, 'soc_min': 0.1,
                                         'efficiency': 0.9, 'alfa': 0.0, 'beta': 0.0}}},
        'Consumers': {'home_2': {'Type': 'retail'}},
        'simulation': {},
    }

    with pytest.warns(DeprecationWarning, match='0.6.2 format') as record:
        config = parse_config(raw, Path('example'))

    assert "'simulation'" in str(record[0].message) and 'shop.Extra' in str(record[0].message)
    assert [n.node_id for n in config.nodes] == ['pv', 'home_2', 'shop']
    assert [n.node_type for n in config.nodes] == ['producer', 'consumer', 'prosumer']
    assert config.nodes[1].csv_path == Path('example/nodes/consumers/home.csv')
    assert config.nodes[0].category == 'plant'
    assert (config.format_version, config.incentive.model, config.delta_t) == ('0.6.2', 'tip', 1.0)


def test_unrecognised_format_raises() -> None:
    """
    Test that a configuration in neither format raises ValueError.
    """
    with pytest.raises(ValueError, match='unrecognised scenario format'):
        parse_config({'nodes': {}}, Path('s'))
