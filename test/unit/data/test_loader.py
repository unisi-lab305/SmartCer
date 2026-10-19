"""
Unit tests of `smartcer.data.loader` on small scenarios written to a temporary folder.

Validation errors must name the node, the column (when relevant) and the first offending row.
"""

from collections.abc import Callable
import json
from pathlib import Path

import numpy as np
import pytest

from smartcer.data import TimestampWarning, load_scenario


pytestmark = pytest.mark.unit

PROSUMER = {'shop': {'type': 'prosumer', 'csv': 'shop.csv'}}
HEADER = 'datetime,production,consumption,sell_price,buy_price'


def rows(n: int = 4, step_minutes: int = 60, production: float = 2.0) -> list[str]:
    """
    CSV rows of a prosumer from 2022-01-03 00:00: load 1, sell 0.1, buy 0.3.

    Args:
        n: Number of rows.
        step_minutes: Time step (minutes).
        production: Production value of every row.

    Returns:
        list[str]: Data lines without the header.
    """
    return [f'2022-01-03 {m // 60:02d}:{m % 60:02d}:00,{production},1,0.1,0.3'
            for m in range(0, n * step_minutes, step_minutes)]


def csv(lines: list[str], header: str = HEADER) -> str:
    """
    CSV text from a header and data lines.

    Args:
        lines: Data lines.
        header: Header line.

    Returns:
        str: CSV text.
    """
    return '\n'.join([header, *lines]) + '\n'


def test_loads_v1_scenario(write_scenario: Callable[..., Path]) -> None:
    """
    Test a regular 1.0 scenario: arrays, timestamps, uniform incentive.

    Args:
        write_scenario: Scenario factory.
    """
    community = load_scenario(write_scenario(PROSUMER, {'shop.csv': csv(rows())}))
    node = community.node('shop')

    assert community.n_steps == 4
    assert community.timestamps[1] == np.datetime64('2022-01-03T01:00')
    assert node.generation_kwh.tolist() == [2.0] * 4 and node.buy_price.tolist() == [0.3] * 4
    # export 1 kWh at 0.1 €/kWh, nothing imported, so nothing shared
    assert community.step(0).welfare.total == pytest.approx(0.1)


def test_period_selection(write_scenario: Callable[..., Path]) -> None:
    """
    Test start and end, both inclusive, and the error for an empty period.

    Args:
        write_scenario: Scenario factory.
    """
    path = write_scenario(PROSUMER, {'shop.csv': csv(rows())})
    community = load_scenario(path, start='2022-01-03 01:00', end='2022-01-03 02:00:00')

    assert community.n_steps == 2
    with pytest.raises(ValueError, match="node 'shop'.*no rows"):
        load_scenario(path, start='2023-01-01')


def test_s5_units_kw_and_quarter_hour_steps(write_scenario: Callable[..., Path]) -> None:
    """
    Test S5: 15-minute data in kW are converted to kWh per step (2 kW → 0.5 kWh).

    Args:
        write_scenario: Scenario factory.
    """
    path = write_scenario(PROSUMER, {'shop.csv': csv(rows(step_minutes=15))}, delta_t=0.25,
                          units='kW')
    node = load_scenario(path).node('shop')

    assert node.generation_kwh.tolist() == [0.5] * 4
    assert node.load_kwh.tolist() == [0.25] * 4


def test_column_aliases_are_accepted(write_scenario: Callable[..., Path]) -> None:
    """
    Test the 0.6.2 column names price_exp and price_imp in a 1.0 scenario.

    Args:
        write_scenario: Scenario factory.
    """
    header = 'datetime,production,consumption,price_exp,price_imp'
    node = load_scenario(write_scenario(PROSUMER, {'shop.csv': csv(rows(), header)})).node('shop')

    assert node.sell_price.tolist() == [0.1] * 4


@pytest.mark.parametrize(
    ('lines', 'header', 'match'),
    [
        ([line.rsplit(',', 1)[0] for line in rows()], 'datetime,production,consumption,sell_price',
         r"node 'shop' \(shop.csv\): missing columns \['buy_price'\]"),
        (rows()[:2] + ['2022-01-03 02:00:00,,1,0.1,0.3'] + rows()[3:], HEADER,
         r"node 'shop'.*column 'production'.*line 4 \(2022-01-03 02:00:00\)"),
        (rows()[:2] + ['2022-01-03 02:00:00,-1,1,0.1,0.3'] + rows()[3:], HEADER,
         r"column 'production' is negative at line 4"),
        ([rows()[1], rows()[0]] + rows()[2:], HEADER, r"node 'shop'.*not sorted at line 3"),
        (rows()[:2] + [rows()[1]] + rows()[3:], HEADER, r'duplicated timestamp at line 4'),
        (rows(step_minutes=30), HEADER, r'step of 0 days 00:30:00 instead of delta_t = 1 h at '
                                        r'line 3'),
    ],
)
def test_v1_validation_errors(write_scenario: Callable[..., Path], lines: list[str], header: str,
                              match: str) -> None:
    """
    Test missing columns, missing values, negative energies, unsorted, duplicated and wrongly
    spaced timestamps: each raises ValueError naming node, column and first offending line.

    Args:
        write_scenario: Scenario factory.
        lines: Data lines.
        header: Header line.
        match: Regular expression expected in the message.
    """
    path = write_scenario(PROSUMER, {'shop.csv': csv(lines, header)})

    with pytest.raises(ValueError, match=match):
        load_scenario(path)


def test_v1_nodes_with_different_timestamps_raise(write_scenario: Callable[..., Path]) -> None:
    """
    Test that two nodes with different timestamps raise ValueError naming the node and the line.

    Args:
        write_scenario: Scenario factory.
    """
    shifted = [line.replace('2022-01-03', '2022-01-04') for line in rows()]
    nodes = {**PROSUMER, 'other': {'type': 'prosumer', 'csv': 'other.csv'}}
    path = write_scenario(nodes, {'shop.csv': csv(rows()), 'other.csv': csv(shifted)})

    with pytest.raises(ValueError, match=r"node 'other'.*differ from node 'shop' at 4 rows, first "
                                         r"at line 2"):
        load_scenario(path)


def test_legacy_irregular_timestamps_warn_and_align_by_position(tmp_path: Path) -> None:
    """
    Test the 0.6.2 format: a duplicated timestamp in one node is reported in a TimestampWarning,
    rows are aligned by position and the community uses the first node's timestamps.

    Args:
        tmp_path: Pytest temporary directory.
    """
    pv = {'tip_base': 0.08, 'tip_threshold': 0.12, 'fi_zone': 0.01, 'pnrr': 0.0}
    config = {'Prosumers': {'a': {'Type': 'x', 'Photovoltaic': pv},
                            'b': {'Type': 'x', 'Photovoltaic': pv}}}
    folder = tmp_path / 'legacy'
    (folder / 'nodes' / 'prosumers').mkdir(parents=True)
    (folder / 'config.json').write_text(json.dumps(config))
    (folder / 'nodes' / 'prosumers' / 'a.csv').write_text(csv(rows()))
    (folder / 'nodes' / 'prosumers' / 'b.csv').write_text(csv(rows()[:2] + [rows()[1]] +
                                                              rows()[3:]))

    with pytest.warns(TimestampWarning, match=r"node 'b'.*duplicated timestamp at line 4"):
        community = load_scenario(folder)

    assert community.n_steps == 4
    assert community.timestamps[2] == np.datetime64('2022-01-03T02:00')


def write_legacy(tmp_path: Path, a_lines: list[str], b_lines: list[str]) -> Path:
    """
    0.6.2 scenario with two prosumers 'a' and 'b'.

    Args:
        tmp_path: Pytest temporary directory.
        a_lines: Data lines of a.csv.
        b_lines: Data lines of b.csv.

    Returns:
        Path: Scenario folder.
    """
    pv = {'tip_base': 0.08, 'tip_threshold': 0.12, 'fi_zone': 0.01, 'pnrr': 0.0}
    config = {'Prosumers': {'a': {'Type': 'x', 'Photovoltaic': pv},
                            'b': {'Type': 'x', 'Photovoltaic': pv}}}
    folder = tmp_path / 'legacy'
    (folder / 'nodes' / 'prosumers').mkdir(parents=True)
    (folder / 'config.json').write_text(json.dumps(config))
    (folder / 'nodes' / 'prosumers' / 'a.csv').write_text(csv(a_lines))
    (folder / 'nodes' / 'prosumers' / 'b.csv').write_text(csv(b_lines))
    return folder


def test_legacy_period_inside_shifted_block_is_cut_by_position(tmp_path: Path) -> None:
    """
    Test the 0.6.2 format with node 'b' shifted by one hour from its third row (as the summer-time
    block of Leather): the period is selected on node 'a' and both nodes are cut at the same rows,
    also when the boundary falls inside the shifted block.

    Args:
        tmp_path: Pytest temporary directory.
    """
    # b: hours 00, 01, then 03-06 instead of 02-05; production = 10 + row number
    hours = [0, 1, 3, 4, 5, 6]
    shifted = [f'2022-01-03 {h:02d}:00:00,{10 + row}.0,1,0.1,0.3' for row, h in enumerate(hours)]
    folder = write_legacy(tmp_path, rows(6), shifted)

    with pytest.warns(TimestampWarning, match=r"node 'b'.*differ from node 'a' at 3 rows"):
        community = load_scenario(folder, start='2022-01-03 03:00')

    assert community.n_steps == 3
    assert community.timestamps[0] == np.datetime64('2022-01-03T03:00')
    # rows 3-5 of both files, that is b's rows labelled 04-06
    assert community.node('b').generation_kwh.tolist() == [13.0, 14.0, 15.0]
    with pytest.warns(TimestampWarning):
        assert load_scenario(folder, end='2022-01-03 02:00').n_steps == 3


def test_legacy_files_with_different_row_counts_raise(tmp_path: Path) -> None:
    """
    Test the 0.6.2 format: files with different numbers of rows cannot be aligned by position.

    Args:
        tmp_path: Pytest temporary directory.
    """
    folder = write_legacy(tmp_path, rows(6), rows(5))

    with pytest.raises(ValueError, match=r"node 'b'.*5 rows, node 'a' has 6; the 0.6.2 format "
                                         r"aligns the nodes by row position"):
        load_scenario(folder, start='2022-01-03 01:00', end='2022-01-03 02:00')


def test_incentive_csv_column(write_scenario: Callable[..., Path]) -> None:
    """
    Test a per-step unit value from a CSV column, and the error for mismatched timestamps.

    Args:
        write_scenario: Scenario factory.
    """
    values = csv([f'2022-01-03 0{h}:00:00,0.{h + 1}' for h in range(4)], 'datetime,value')
    incentive = {'model': 'uniform', 'unit_value': {'csv': 'pi.csv', 'column': 'value'}}
    path = write_scenario(PROSUMER, {'shop.csv': csv(rows()), 'pi.csv': values},
                          incentive=incentive)

    assert load_scenario(path).incentive.unit_value.tolist() == [0.1, 0.2, 0.3, 0.4]

    shifted = csv([f'2022-01-03 0{h + 1}:00:00,0.1' for h in range(4)], 'datetime,value')
    (path / 'pi.csv').write_text(shifted)
    with pytest.raises(ValueError, match=r'incentive.unit_value \(pi.csv\): timestamps differ'):
        load_scenario(path)


def test_consumer_sell_price_is_read_when_present(write_scenario: Callable[..., Path]) -> None:
    """
    Test that a consumer's sell_price column is read when it exists and is 0 otherwise.

    Args:
        write_scenario: Scenario factory.
    """
    with_price = csv(['2022-01-03 00:00:00,1,0.3,0.12'],
                     'datetime,consumption,buy_price,sell_price')
    without_price = csv(['2022-01-03 00:00:00,1,0.3'], 'datetime,consumption,buy_price')
    nodes = {'a': {'type': 'consumer', 'csv': 'a.csv'}, 'b': {'type': 'consumer', 'csv': 'b.csv'}}
    community = load_scenario(write_scenario(nodes, {'a.csv': with_price, 'b.csv': without_price}))

    assert community.node('a').sell_price.tolist() == [0.12]
    assert community.node('b').sell_price.tolist() == [0.0]
