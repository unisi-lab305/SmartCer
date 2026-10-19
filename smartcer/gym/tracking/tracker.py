"""
Results tracking of `SmartCerEnv` runs: one community table and one table per node, exported to
CSV with the KPIs and the run configuration.

Adapted from storage-node-environment (`storage_node_env/gym/tracking/tracker.py`): the rows come
from the step `info` of the community environment; a Gymnasium wrapper records them, so the
environment itself does not track anything. The tracker keeps the current episode and the last
completed one only, so that long trainings do not grow in memory.
"""

import json
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import pandas as pd

from smartcer.core.community import Community
from smartcer.gym.metrics import compute_kpis


WELFARE_COLUMNS = ('revenue', 'cost', 'storage_cost', 'incentive', 'shared_energy_kwh',
                   'export_kwh', 'import_kwh')


class ResultsTracker:
    """
    Collects the step info of a community environment into tables.

    Attributes:
        node_ids: Node identifiers, in community order.
        capacities: Battery node id → capacity S^max (kWh).
        soc_min: Battery node id → minimum state of charge S^min (kWh).
        episode: Number of the current episode (incremented at every reset).
        episodes_completed: Number of episodes that reached their last step.
    """

    def __init__(self, community: Community) -> None:
        """
        Create an empty tracker for a community.

        Args:
            community: The environment's community.
        """
        self.node_ids = community.node_ids
        self.capacities = {node_id: community.node(node_id).battery.capacity
                           for node_id in community.storage_node_ids}
        self.soc_min = {node_id: community.node(node_id).battery.soc_min_kwh
                        for node_id in community.storage_node_ids}
        self.episode = 0
        self.episodes_completed = 0
        self._community_rows: list[dict[str, Any]] = []
        self._node_rows: dict[str, list[dict[str, Any]]] = {i: [] for i in self.node_ids}
        self._completed: tuple[pd.DataFrame, dict[str, pd.DataFrame]] | None = None

    def __repr__(self) -> str:
        return (f'ResultsTracker({len(self.node_ids)} nodes, episode={self.episode}, '
                f'episodes_completed={self.episodes_completed})')

    def on_reset(self) -> None:
        """
        Start a new episode; the rows of an unfinished episode are dropped.
        """
        self.episode += 1
        self._clear()

    def _clear(self) -> None:
        """Drop the rows of the current episode."""
        self._community_rows = []
        self._node_rows = {i: [] for i in self.node_ids}

    def record(self, info: dict[str, Any], done: bool) -> None:
        """
        Record one step.

        Args:
            info: Info of `step()`.
            done: True on the last step of the episode, which completes it.
        """
        head = {'datetime': info['timestamp'], 'episode': self.episode, 't': info['t']}
        reward, welfare = info['reward'], info['welfare']
        self._community_rows.append({
            **head, 'reward': reward['total'], 'reward_main': reward['main'],
            'reward_penalty': reward['penalty'], 'welfare': welfare['total'],
            **{name: welfare[name] for name in WELFARE_COLUMNS},
            'penalty': info['penalty'], 'obs_clipped': info['obs_clipped'],
        })
        node_rewards = reward.get('nodes', {})
        for node_id, fields in info['nodes'].items():
            row = {**head, **fields}
            if node_id in node_rewards:
                r = node_rewards[node_id]
                row.update(reward=r['total'], reward_main=r['main'], reward_penalty=r['penalty'])
            self._node_rows[node_id].append(row)
        if done:
            self._completed = (pd.DataFrame(self._community_rows),
                               {i: pd.DataFrame(rows) for i, rows in self._node_rows.items()})
            self.episodes_completed += 1
            self._clear()

    def _last(self) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """
        Tables of the last completed episode.

        Returns:
            tuple[pd.DataFrame, dict[str, pd.DataFrame]]: Community table and node tables.

        Raises:
            ValueError: If no episode is completed.
        """
        if self._completed is None:
            raise ValueError('no completed episode recorded')
        return self._completed

    def community_table(self) -> pd.DataFrame:
        """
        Community table of the last completed episode.

        Returns:
            pd.DataFrame: One row per step: `datetime`, `episode`, `t`, `reward`, `reward_main`,
            `reward_penalty` (σ-weighted), `welfare` and its components (€, kWh), `penalty`
            (K_t), `obs_clipped`.
        """
        return self._last()[0].copy()

    def node_tables(self) -> dict[str, pd.DataFrame]:
        """
        Node tables of the last completed episode.

        Returns:
            dict[str, pd.DataFrame]: Node id → one row per step with `datetime`, `episode`, `t`,
            the `NodeStep` fields and, with per-node rewards, `reward`, `reward_main`,
            `reward_penalty`.
        """
        return {node_id: table.copy() for node_id, table in self._last()[1].items()}

    def kpis(self) -> dict[str, Any]:
        """
        KPIs of the last completed episode (`smartcer.gym.metrics.compute_kpis`).

        Returns:
            dict[str, Any]: The KPIs.
        """
        community, nodes = self._last()
        return compute_kpis(community, nodes, self.capacities, self.soc_min)

    def save(self, output_dir: str | Path, run_config: dict[str, Any] | None = None) -> Path:
        """
        Write the last completed episode: `community.csv`, `nodes/<node_id>.csv`, `kpis.json`
        and, if given, `run_config.json`.

        Args:
            output_dir: Output folder, created if missing.
            run_config: Run configuration (e.g. `SmartCerEnv.run_config()`).

        Returns:
            Path: The output folder.
        """
        community, nodes = self._last()
        folder = Path(output_dir)
        (folder / 'nodes').mkdir(parents=True, exist_ok=True)
        community.to_csv(folder / 'community.csv', index=False)
        for node_id, table in nodes.items():
            table.to_csv(folder / 'nodes' / f'{node_id}.csv', index=False)
        write_json(folder / 'kpis.json', self.kpis())
        if run_config is not None:
            write_json(folder / 'run_config.json', run_config)
        return folder


def write_json(path: Path, data: dict[str, Any]) -> None:
    """
    Write a dictionary as indented JSON (numpy scalars converted).

    Args:
        path: File path.
        data: JSON-serialisable data, numpy scalars allowed.
    """
    def convert(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'not JSON-serialisable: {value!r}')

    path.write_text(json.dumps(data, indent=2, default=convert) + '\n', encoding='utf-8')


class TrackResults(gym.Wrapper):
    """
    Wrapper that records every step of a `SmartCerEnv` in a `ResultsTracker`.

    Attributes:
        tracker: The tracker (reachable through `env.get_wrapper_attr('tracker')`).
    """

    def __init__(self, env: gym.Env) -> None:
        """
        Wrap an environment.

        Args:
            env: A `SmartCerEnv`, possibly wrapped.

        Raises:
            TypeError: If the unwrapped environment has no `community`.
        """
        super().__init__(env)
        community = getattr(env.unwrapped, 'community', None)
        if not isinstance(community, Community):
            raise TypeError(f'TrackResults needs a SmartCerEnv, got {type(env.unwrapped).__name__}')
        self.tracker = ResultsTracker(community)

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """
        Reset the environment and start a new episode in the tracker.

        Args:
            seed: Seed passed to the environment.
            options: Options passed to the environment.

        Returns:
            tuple[Any, dict[str, Any]]: Observation and info of the environment.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self.tracker.on_reset()
        return obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Step the environment and record the step.

        Args:
            action: Action passed to the environment.

        Returns:
            tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]: The environment's step result.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.tracker.record(info, terminated or truncated)
        return obs, reward, terminated, truncated, info
