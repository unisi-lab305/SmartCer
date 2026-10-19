"""
Unit tests of the observation normaliser (`smartcer.gym.observation.normalizers`).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from smartcer.gym.observation import MinMaxNormalizer


pytestmark = pytest.mark.unit


def test_fit_and_transform_clip_to_unit_interval() -> None:
    """
    Test fitting on a window, scaling, clipping of values outside the training range, and a
    constant series mapped to 0.
    """
    normalizer = MinMaxNormalizer.fit({'a/generation': [2.0, 6.0], 'a/load': [3.0, 3.0]},
                                      window=('2022-01-01T00:00:00', '2022-01-01T01:00:00'))

    assert normalizer.bounds == {'a/generation': (2.0, 6.0), 'a/load': (3.0, 3.0)}
    assert normalizer.transform('a/generation', [2.0, 4.0, 8.0, 0.0]).tolist() == [0, 0.5, 1, 0]
    assert normalizer.transform('a/load', np.array([[3.0, 5.0]])).tolist() == [[0.0, 0.0]]
    with pytest.raises(KeyError, match='b/load'):
        normalizer.transform('b/load', [1.0])
    # the values that transform clips
    assert normalizer.outside_range('a/generation', [2.0, 4.0, 8.0, 0.0]).tolist() == [
        False, False, True, True]
    assert normalizer.outside_range('a/load', [3.0, 5.0]).tolist() == [False, True]
    with pytest.raises(KeyError, match='b/load'):
        normalizer.outside_range('b/load', [1.0])


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    """
    Test that the JSON file stores format, window, scenario and bounds, and loads back equal.

    Args:
        tmp_path: Pytest temporary directory.
    """
    normalizer = MinMaxNormalizer({'a/generation': (0.0, 7.5)}, window=('t0', 't1'),
                                  scenario='mini')
    path = normalizer.save(tmp_path / 'normalizer.json')
    loaded = MinMaxNormalizer.load(path)

    assert json.loads(path.read_text())['format'] == 1
    assert (loaded.bounds, loaded.window, loaded.scenario) == (
        normalizer.bounds, normalizer.window, normalizer.scenario)


def test_invalid_input_raises() -> None:
    """
    Test the validation: invalid bounds, empty series, missing series, unsupported files.
    """
    with pytest.raises(ValueError, match='invalid bounds'):
        MinMaxNormalizer({'a': (2.0, 1.0)})
    with pytest.raises(ValueError, match='invalid bounds'):
        MinMaxNormalizer({'a': (0.0, np.inf)})
    with pytest.raises(ValueError, match='empty window'):
        MinMaxNormalizer.fit({'a': []})
    with pytest.raises(ValueError, match=r"\['b'\]"):
        MinMaxNormalizer({'a': (0.0, 1.0)}).require(['a', 'b'])
    with pytest.raises(ValueError, match='unsupported normaliser'):
        MinMaxNormalizer.from_dict({'format': 2, 'type': 'min_max', 'bounds': {}})
