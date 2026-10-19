"""
Min-max normaliser of the observation, fitted on a declared training window and stored as JSON.

Adapted from storage-node-environment (`storage_node_env/gym/preprocessing/normalizers.py`): one
instance per environment instead of module-level state, fitted on the steps of a training window
instead of the whole CSV, and serialisable so that an evaluation environment reuses the bounds of
the training one.
"""

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from smartcer.gym.observation.encoders import min_max


NORMALIZER_FORMAT = 1


class MinMaxNormalizer:
    """
    Per-feature (min, max) bounds; values are scaled to [0, 1] and clipped.

    Attributes:
        bounds: Series name → (min, max) over the training window.
        window: First and last timestamp of the training window (ISO strings), if known.
        scenario: Name of the scenario it was fitted on, if known.
    """

    def __init__(self, bounds: Mapping[str, tuple[float, float]],
                 window: tuple[str, str] | None = None, scenario: str | None = None) -> None:
        """
        Create a normaliser from known bounds.

        Args:
            bounds: Series name → (min, max), with min ≤ max, both finite.
            window: First and last timestamp of the training window.
            scenario: Scenario name.

        Raises:
            ValueError: If a bound is not finite or min > max.
        """
        checked = {}
        for name, (low, high) in bounds.items():
            low, high = float(low), float(high)
            if not (np.isfinite(low) and np.isfinite(high) and low <= high):
                raise ValueError(f'invalid bounds for {name!r}: ({low}, {high})')
            checked[name] = (low, high)
        self.bounds = checked
        self.window = None if window is None else (str(window[0]), str(window[1]))
        self.scenario = scenario

    @classmethod
    def fit(cls, series: Mapping[str, ArrayLike], window: tuple[str, str] | None = None,
            scenario: str | None = None) -> 'MinMaxNormalizer':
        """
        Fit the bounds on the values of the training window.

        Args:
            series: Series name → values of the training window only.
            window: First and last timestamp of the training window, stored for reference.
            scenario: Scenario name, stored for reference.

        Returns:
            MinMaxNormalizer: The fitted normaliser.

        Raises:
            ValueError: If a series is empty.
        """
        bounds = {}
        for name, values in series.items():
            values = np.asarray(values, dtype=np.float64)
            if values.size == 0:
                raise ValueError(f'cannot fit {name!r} on an empty window')
            bounds[name] = (float(values.min()), float(values.max()))
        return cls(bounds, window=window, scenario=scenario)

    def __repr__(self) -> str:
        return f'MinMaxNormalizer({len(self.bounds)} series, window={self.window})'

    def require(self, names: Iterable[str]) -> None:
        """
        Check that the normaliser has bounds for the given series.

        Args:
            names: Series names.

        Raises:
            ValueError: Naming the missing series.
        """
        missing = [n for n in names if n not in self.bounds]
        if missing:
            raise ValueError(f'normaliser has no bounds for {missing} (it has {list(self.bounds)})')

    def transform(self, name: str, values: ArrayLike) -> np.ndarray:
        """
        Scale values with the bounds of a series and clip them to [0, 1].

        A constant series (min == max) maps to 0; values outside the training range, e.g. in a
        test window, are clipped to 0 or 1.

        Args:
            name: Series name.
            values: Values.

        Returns:
            np.ndarray: Scaled values in [0, 1].

        Raises:
            KeyError: If the series has no bounds.
        """
        if name not in self.bounds:
            raise KeyError(f'normaliser has no bounds for {name!r}')
        low, high = self.bounds[name]
        return np.clip(min_max(values, low, high), 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """
        JSON-serialisable representation.

        Returns:
            dict[str, Any]: `format`, `type`, `scenario`, `window` and `bounds`.
        """
        return {
            'format': NORMALIZER_FORMAT, 'type': 'min_max', 'scenario': self.scenario,
            'window': None if self.window is None else list(self.window),
            'bounds': {name: list(bound) for name, bound in self.bounds.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'MinMaxNormalizer':
        """
        Normaliser from the output of `to_dict`.

        Args:
            data: Dictionary with `format`, `type` and `bounds` (`window`, `scenario` optional).

        Returns:
            MinMaxNormalizer: The normaliser.

        Raises:
            ValueError: If the format or the type is not supported.
        """
        if data.get('format') != NORMALIZER_FORMAT or data.get('type') != 'min_max':
            raise ValueError(f'unsupported normaliser: format {data.get("format")!r}, type '
                             f'{data.get("type")!r} (expected {NORMALIZER_FORMAT}, min_max)')
        window = data.get('window')
        return cls({name: tuple(bound) for name, bound in data['bounds'].items()},
                   window=None if window is None else tuple(window),
                   scenario=data.get('scenario'))

    def save(self, path: str | Path) -> Path:
        """
        Write the normaliser to a JSON file.

        Args:
            path: File path.

        Returns:
            Path: The path written.
        """
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding='utf-8')
        return path

    @classmethod
    def load(cls, path: str | Path) -> 'MinMaxNormalizer':
        """
        Read a normaliser written by `save`.

        Args:
            path: File path.

        Returns:
            MinMaxNormalizer: The normaliser.
        """
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
