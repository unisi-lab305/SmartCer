"""
Gymnasium environment `SmartCerEnv` (ID `SmartCer-v1`) for the battery control of a CER.

One continuous action per battery, `Box(-1, 1, (N,), float32)` in `storage_node_ids` order; reward
R_t = W_t - σ K_t (docs/FORMULATION.md sections 7, 10, 11); information timing set by
`observe_current_step` (section 12). The environment wraps a `smartcer.core.Community`.
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from smartcer.core.community import Community, CommunityStep
from smartcer.data import load_scenario
from smartcer.gym.observation import ObservationBuilder
from smartcer.gym.run_config import map_v062_run_config


REWARD_TYPES = ('welfare',)


class SmartCerEnv(gym.Env):
    """
    Renewable Energy Community with one controlled battery per storage node.

    Episodes cover `episode_length` steps of the scenario period, starting at its first step or,
    with `random_start`, at a step drawn with `self.np_random`. Every step of the episode is
    controlled; the episode ends on its last step with `truncated=True` (`terminated` is always
    False), and the terminal observation comes from that step.

    Actions are clipped to [-1, 1] without penalty, then each node clips its action to the dynamic
    bounds and is penalised as in section 7. Non-finite actions raise `ValueError`.

    Attributes:
        community: The scenario's community (nodes, batteries, incentive, timestamps).
        storage_node_ids: Identifiers of the battery nodes, in action order.
        obs_layout: Feature name → slice of the observation vector.
        observe_current_step: Information timing.
        penalty_coefficient: σ of the reward.
        episode_length: Steps per episode.
        random_start: Whether episodes start at a random step.
        action_space: `Box(-1, 1, (len(storage_node_ids),), float32)`.
        observation_space: Flat `Box`, see `smartcer.gym.observation.schema`.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        scenario: str | Path | Community,
        start: Any = None,
        end: Any = None,
        episode_length: int | None = None,
        random_start: bool = False,
        observe_current_step: bool = True,
        observation: str = 'meco',
        reward: str = 'welfare',
        penalty_coefficient: float = 10.0,
        seed: int | None = None,
    ) -> None:
        """
        Create the environment.

        Args:
            scenario: Scenario folder (loaded with `smartcer.data.load_scenario`) or a
                `Community`.
            start: First timestamp of the period (with a scenario folder only).
            end: Last timestamp of the period (with a scenario folder only).
            episode_length: Steps per episode; None for the whole period.
            random_start: Draw the first step of each episode uniformly among the feasible ones.
            observe_current_step: True: the observation of step t holds g, l and bounds of step t;
                False: g and l of step t - 1 and a bound without the charge cap.
            observation: Observation preset ('meco').
            reward: Reward type ('welfare': R_t = W_t - σ K_t).
            penalty_coefficient: σ, ≥ 0 (default 10, as in [M24]).
            seed: Seed used by the first `reset()` called without a seed.

        Raises:
            ValueError: On invalid arguments.
        """
        if isinstance(scenario, Community):
            if start is not None or end is not None:
                raise ValueError('start and end apply only when scenario is a folder')
            community = scenario
        else:
            community = load_scenario(scenario, start=start, end=end)
        if reward not in REWARD_TYPES:
            raise ValueError(f'unknown reward {reward!r}; available: {REWARD_TYPES}')
        if not (np.isfinite(penalty_coefficient) and penalty_coefficient >= 0):
            raise ValueError(f'penalty_coefficient must be ≥ 0, got {penalty_coefficient!r}')
        n_steps = community.n_steps
        episode_length = n_steps if episode_length is None else int(episode_length)
        if not 1 <= episode_length <= n_steps:
            raise ValueError(f'episode_length must lie in [1, {n_steps}] (the period has '
                             f'{n_steps} steps), got {episode_length}')

        self.community = community
        self.storage_node_ids = community.storage_node_ids
        self.observe_current_step = bool(observe_current_step)
        self.reward_type = reward
        self.penalty_coefficient = float(penalty_coefficient)
        self.episode_length = episode_length
        self.random_start = bool(random_start)
        self._builder = ObservationBuilder(community, observation, self.observe_current_step)
        self.obs_layout = self._builder.layout
        self.observation_space = self._builder.space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.storage_node_ids),),
                                       dtype=np.float32)
        self._default_seed = seed
        self._t = 0
        self._episode_start = 0
        self._episode_end = episode_length - 1
        self._needs_reset = True

    @classmethod
    def from_v062_config(cls, cer_path: str | Path, cer_config: dict[str, Any],
                         **overrides: Any) -> 'SmartCerEnv':
        """
        Create the environment from a 0.6.2 scenario path and run configuration.

        See `smartcer.gym.run_config.map_v062_run_config` for the key mapping (DeprecationWarning).

        Args:
            cer_path: Scenario folder.
            cer_config: The 0.6.2 run configuration (`simulation`, `reward`, ...).
            **overrides: Arguments that replace the mapped ones.

        Returns:
            SmartCerEnv: The environment.
        """
        return cls(cer_path, **{**map_v062_run_config(cer_config), **overrides})

    def __repr__(self) -> str:
        return (f'SmartCerEnv({self.community.name!r}, {len(self.storage_node_ids)} batteries, '
                f'episode_length={self.episode_length})')

    def _info(self, t: int) -> dict[str, Any]:
        """
        Info common to reset and step.

        Args:
            t: Step index.

        Returns:
            dict[str, Any]: `t` and `timestamp`.
        """
        return {'t': t, 'timestamp': self.community.timestamps[t]}

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Start an episode: restore the batteries and choose the first step.

        Args:
            seed: Seed of `self.np_random` and of the action space; if None, the constructor's
                seed is used by the first reset.
            options: Not used.

        Returns:
            tuple[np.ndarray, dict[str, Any]]: First observation and info (`t`, `timestamp`,
            `episode_start`, `episode_end`).
        """
        if seed is None and self._default_seed is not None:
            seed, self._default_seed = self._default_seed, None
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        n_starts = self.community.n_steps - self.episode_length + 1
        start = int(self.np_random.integers(n_starts)) if self.random_start else 0
        self._episode_start, self._episode_end = start, start + self.episode_length - 1
        self._t = start
        self._needs_reset = False
        self.community.reset()
        info = self._info(start)
        info.update(episode_start=start, episode_end=self._episode_end)
        return self._builder.build(start), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Apply one action per battery for the current step.

        Args:
            action: Array of shape (len(storage_node_ids),).

        Returns:
            tuple[np.ndarray, float, bool, bool, dict[str, Any]]: Observation of the next step (of
            the same step at the end of the episode), reward R_t = W_t - σ K_t, terminated (always
            False), truncated (True on the last step of the episode) and info.

        Raises:
            RuntimeError: If called before `reset()` or after the end of the episode.
            ValueError: If the action has the wrong shape or is not finite.
        """
        if self._needs_reset:
            raise RuntimeError('call reset() before step(), and after the end of an episode')
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape:
            raise ValueError(f'action must have shape {self.action_space.shape}, got '
                             f'{action.shape}')
        if not np.all(np.isfinite(action)):
            raise ValueError(f'action must be finite, got {action}')
        clipped = np.clip(action, -1.0, 1.0)

        t = self._t
        result = self.community.step(t, clipped)
        welfare = result.welfare.total
        penalty = self.penalty_coefficient * result.penalty
        reward = welfare - penalty

        truncated = t == self._episode_end
        if truncated:
            self._needs_reset = True
        else:
            self._t = t + 1
        obs = self._builder.build(self._t)
        return obs, float(reward), False, truncated, self._step_info(result, clipped, welfare,
                                                                     penalty, reward)

    def _step_info(self, result: CommunityStep, action: np.ndarray, welfare: float,
                   penalty: float, reward: float) -> dict[str, Any]:
        """
        Info of a step: reward components, welfare breakdown, K_t and per-node results.

        Args:
            result: Community step record.
            action: Action after clipping to [-1, 1].
            welfare: W_t.
            penalty: σ K_t.
            reward: R_t.

        Returns:
            dict[str, Any]: `t`, `timestamp`, `reward` (welfare, penalty, total), `welfare`
            (breakdown), `penalty` (K_t), `action` (after the [-1, 1] clip) and `nodes`
            (node id → NodeStep fields).
        """
        info = self._info(result.t)
        w = result.welfare
        info.update(
            reward={'welfare': welfare, 'penalty': penalty, 'total': reward},
            welfare={
                'revenue': w.revenue, 'cost': w.cost, 'storage_cost': w.storage_cost,
                'incentive': w.incentive, 'shared_energy_kwh': w.shared_energy_kwh,
                'export_kwh': w.export_kwh, 'import_kwh': w.import_kwh, 'total': w.total,
            },
            penalty=result.penalty,
            action=action,
            nodes={r.node_id: {name: getattr(r, name) for name in _NODE_FIELDS}
                   for r in result.nodes},
        )
        return info


_NODE_FIELDS = (
    'generation_kwh', 'load_kwh', 'p_cha_kw', 'p_dis_kw', 'charge_kwh', 'discharge_kwh', 'net_kwh',
    'export_kwh', 'import_kwh', 'revenue', 'cost', 'throughput_kwh', 'storage_cost', 'penalty',
    'soc_kwh', 'action_requested', 'action_applied', 'a_upper', 'a_lower',
)
