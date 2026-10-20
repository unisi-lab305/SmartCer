"""
Gymnasium environment `SmartCerEnv` (ID `SmartCer-v1`) for the battery control of a CER.

One continuous action per battery, `Box(-1, 1, (N,), float32)` in `storage_node_ids` order; reward
R_t = W_t - σ K_t by default, other types in `smartcer.gym.rewards` (docs/FORMULATION.md sections
7, 10, 11); information timing set by `observe_current_step` (section 12). The environment wraps a
`smartcer.core.Community`. Optional hooks: a forecast provider (`smartcer.forecasts`), whose
forecasts enter the observation only, and the MILP oracle (`smartcer.oracle`), whose data enter
`info['privileged']` only.
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

import smartcer
from smartcer.core.community import Community, CommunityStep
from smartcer.data import load_scenario
from smartcer.forecasts import FORECAST_PROVIDERS, ForecastProvider, PerfectForecast
from smartcer.gym.observation import MinMaxNormalizer, ObservationBuilder, fit_normalizer
from smartcer.gym.rewards import RewardBreakdown, make_reward
from smartcer.gym.run_config import map_v062_run_config
from smartcer.oracle import OracleProvider


class SmartCerEnv(gym.Env):
    """
    Renewable Energy Community with one controlled battery per storage node.

    Episodes cover `episode_length` steps of the scenario period, starting at its first step or,
    with `random_start`, at a step drawn with `self.np_random`. Every step of the episode is
    controlled; the episode ends on its last step t with `truncated=True` (`terminated` is always
    False). The terminal observation is the observation of step t + 1 with the final battery
    states, as at the start of an episode beginning at t + 1, so that the value of the next state
    can be bootstrapped; only without forecasts, on the last step of the data, it is the
    observation of step t with the final battery states. With a forecast horizon H the episodes
    use the steps 0, ..., n_steps - H - 2 with `observe_current_step=True` and 0, ...,
    n_steps - H - 1 with False, so that the forecasts of every step of an episode and of the step
    after it lie in the data.

    Actions are clipped to [-1, 1] without penalty, then each node clips its action to the dynamic
    bounds and is penalised as in section 7. Non-finite actions raise `ValueError`.

    Attributes:
        community: The scenario's community (nodes, batteries, incentive, timestamps).
        scenario_path: Scenario folder, or None when the environment was built from a `Community`.
        storage_node_ids: Identifiers of the battery nodes, in action order.
        obs_layout: Feature name → slice of the observation vector.
        normalizer: Min-max normaliser of the observed generation and load (save it with
            `env.normalizer.save(path)` to reuse it in an evaluation environment).
        observe_current_step: Information timing.
        reward_type: Reward type.
        reward_per_node: Whether the per-node variant of the reward is used.
        penalty_coefficient: σ of the reward.
        episode_length: Steps per episode.
        random_start: Whether episodes start at a random step.
        forecast: Forecast provider of the observation, or None.
        forecast_horizon: Forecast horizon H in steps (0 without forecasts).
        oracle: MILP oracle of `info['privileged']` and of the `ReplayController`, or None.
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
        lookback: int = 0,
        normalizer: str | Path | MinMaxNormalizer | None = None,
        normalizer_window: tuple[Any, Any] | None = None,
        reward: str = 'welfare',
        reward_per_node: bool = False,
        penalty_coefficient: float = 10.0,
        forecast: str | ForecastProvider | None = None,
        forecast_horizon: int = 0,
        oracle: str | Path | OracleProvider | None = None,
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
            observation: Observation preset, 'meco' ([M24] Tab. II) or 'extended'.
            lookback: Past steps of generation and load per battery node ('extended' only).
            normalizer: Normaliser of generation and load: a `MinMaxNormalizer` or the path of
                its JSON file; None fits one on `normalizer_window`.
            normalizer_window: (first, last) timestamp of the training window the normaliser is
                fitted on, both inclusive, within the period; None for the whole period.
            reward: Reward type, 'welfare' (R_t = W_t - σ K_t), 'profit' or 'energy'.
            reward_per_node: Per-node variant (not for 'welfare'): r_u = m_u - σ k_u per battery
                node, reported in `info['reward']['nodes']`, and R_t = Σ_u r_u.
            penalty_coefficient: σ, ≥ 0 (default 10, as in [M24]).
            forecast: Forecasts of the observation's `forecast` section: a name of
                `smartcer.forecasts.FORECAST_PROVIDERS` ('perfect', 'mmfe_perfect',
                'mmfe_standard', 'mmfe_noisy'), built on the community, or a provider; None for
                no forecasts.
            forecast_horizon: Forecast horizon H in steps, ≥ 1 with forecasts.
            oracle: MILP oracle: an oracle folder (`smartcer.oracle.read_oracle`) or an
                `OracleProvider` of the same community; None for no oracle.
            seed: Seed used by the first `reset()` called without a seed.

        Raises:
            ValueError: On invalid arguments.
        """
        if isinstance(scenario, Community):
            if start is not None or end is not None:
                raise ValueError('start and end apply only when scenario is a folder')
            community = scenario
            self.scenario_path = None
        else:
            community = load_scenario(scenario, start=start, end=end)
            self.scenario_path = Path(scenario)
        self._reward = make_reward(reward, community.storage_node_ids,
                                   penalty_coefficient=penalty_coefficient,
                                   per_node=reward_per_node)
        if normalizer is not None and normalizer_window is not None:
            raise ValueError('give either normalizer (already fitted) or normalizer_window')
        if normalizer is None:
            normalizer = fit_normalizer(community, normalizer_window)
        elif not isinstance(normalizer, MinMaxNormalizer):
            normalizer = MinMaxNormalizer.load(normalizer)
        self._forecast_name = forecast if isinstance(forecast, str) or forecast is None \
            else type(forecast).__name__
        if isinstance(forecast, str):
            if forecast not in FORECAST_PROVIDERS:
                raise ValueError(f'unknown forecast provider {forecast!r}; available: '
                                 f'{list(FORECAST_PROVIDERS)}')
            forecast = FORECAST_PROVIDERS[forecast](community)
        if oracle is not None and not isinstance(oracle, OracleProvider):
            oracle = OracleProvider.from_folder(oracle, community)
        if oracle is not None and not (
                oracle.node_ids == community.storage_node_ids
                and np.array_equal(oracle.timestamps, community.timestamps)):
            raise ValueError(f'oracle {oracle!r} was built for another community: its battery '
                             f'nodes or steps differ')
        self._builder = ObservationBuilder(community, observation, bool(observe_current_step),
                                           lookback=lookback, normalizer=normalizer,
                                           forecast=forecast, horizon=forecast_horizon)
        n_steps = community.n_steps
        horizon = self._builder.horizon
        # last step whose observation lies in the data, forecasts included
        last_observable = n_steps - 1 - (
            self._builder.forecast_start(0) + horizon - 1 if horizon else 0)
        # with forecasts the step after every episode must be observable too (terminal
        # observation); without, the last step of the data falls back to its own observation
        usable = last_observable if horizon else n_steps
        if usable < 1:
            raise ValueError(f'forecast horizon {horizon} leaves no step for the episodes (the '
                             f'period has {n_steps} steps)')
        episode_length = usable if episode_length is None else int(episode_length)
        if not 1 <= episode_length <= usable:
            kept = f', the last {n_steps - usable} kept for the forecasts' if usable < n_steps \
                else ''
            raise ValueError(f'episode_length must lie in [1, {usable}] (the period has '
                             f'{n_steps} steps{kept}), got {episode_length}')

        self.community = community
        self.storage_node_ids = community.storage_node_ids
        self.observe_current_step = bool(observe_current_step)
        self.reward_type = reward
        self.reward_per_node = self._reward.per_node
        self.penalty_coefficient = self._reward.penalty_coefficient
        self.episode_length = episode_length
        self.random_start = bool(random_start)
        self.forecast = self._builder.forecast
        self.forecast_horizon = self._builder.horizon
        self.oracle = oracle
        # true future values of the forecast steps, for `info['privileged']`
        self._truth = PerfectForecast(community) if self.forecast is not None else None
        self._usable_steps = usable
        self._last_observable = last_observable
        self.normalizer = self._builder.normalizer
        self.obs_layout = self._builder.layout
        self.observation_space = self._builder.space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.storage_node_ids),),
                                       dtype=np.float32)
        self._seed = seed
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

    @property
    def current_step(self) -> int:
        """Step the next action applies to (the last step of the episode once it ends)."""
        return self._t

    def run_config(self) -> dict[str, Any]:
        """
        The run configuration, JSON-serialisable, for the output files of a run.

        Returns:
            dict[str, Any]: SmartCer version, scenario name and folder, period, time step, episode
            settings, seed, observation settings and normaliser window, reward settings, forecast
            provider (name or class), its settings (`forecast_config`, from the provider's
            `config()` if it has one, else None) and horizon, oracle folder.
        """
        timestamps = self.community.timestamps
        return {
            'smartcer_version': smartcer.__version__,
            'scenario': self.community.name,
            'scenario_path': None if self.scenario_path is None else str(self.scenario_path),
            'start': str(pd.Timestamp(timestamps[0])),
            'end': str(pd.Timestamp(timestamps[-1])),
            'delta_t': self.community.delta_t,
            'episode_length': self.episode_length,
            'random_start': self.random_start,
            'seed': self._seed,
            'observe_current_step': self.observe_current_step,
            'observation': self._builder.preset,
            'lookback': self._builder.lookback,
            'normalizer_window': None if self.normalizer.window is None
            else list(self.normalizer.window),
            'reward': self.reward_type,
            'reward_per_node': self.reward_per_node,
            'penalty_coefficient': self.penalty_coefficient,
            'forecast': self._forecast_name,
            'forecast_config': self.forecast.config() if callable(
                getattr(self.forecast, 'config', None)) else None,
            'forecast_horizon': self.forecast_horizon,
            'oracle': None if self.oracle is None else self.oracle.source,
        }

    def _info(self, t: int, observed: int) -> dict[str, Any]:
        """
        Info common to reset and step.

        Args:
            t: Step index of the info (the executed step, or the first step after a reset).
            observed: Step of the observation returned with the info.

        Returns:
            dict[str, Any]: `t`, `timestamp` and `privileged` (see `_privileged`).
        """
        return {'t': t, 'timestamp': self.community.timestamps[t],
                'privileged': self._privileged(observed)}

    def _privileged(self, t: int) -> dict[str, Any]:
        """
        Data for the critic only, about the step of the observation they accompany.

        Args:
            t: Step of the returned observation (the step of the next action; for the terminal
                observation the step after the episode, or its last step at the end of the data).

        Returns:
            dict[str, Any]: Empty without oracle and forecasts. Otherwise `t`; with an oracle,
            `oracle_action` (a*_t, one value per battery in action order) and `oracle_soc` (the
            oracle state of charge at the start of step t, kWh, comparable with the observed
            state of charge); with forecasts, `true_future` (`'<node_id>/<quantity>'` and
            `'community/<quantity>'` → the true values of the forecast steps of the observation,
            kWh per step, same keys and order as the observed forecasts).
        """
        if self.oracle is None and self._truth is None:
            return {}
        privileged: dict[str, Any] = {'t': t}
        if self.oracle is not None:
            privileged['oracle_action'] = self.oracle.action(t)
            privileged['oracle_soc'] = self.oracle.soc(t)
        if self._truth is not None:
            truth = self._truth.get(self._builder.forecast_start(t), self.forecast_horizon)
            privileged['true_future'] = {key: truth[key] for key in self._builder.forecast_keys}
        return privileged

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Start an episode: restore the batteries, reset the forecast provider with
        `self.np_random` and choose the first step.

        Args:
            seed: Seed of `self.np_random` and of the action space; if None, the constructor's
                seed is used by the first reset.
            options: Not used.

        Returns:
            tuple[np.ndarray, dict[str, Any]]: First observation and info (`t`, `timestamp`,
            `privileged`, `episode_start`, `episode_end`, `obs_clipped`: values of the observation
            clipped to [0, 1] because they lie outside the normaliser's training range).
        """
        if seed is None and self._default_seed is not None:
            seed, self._default_seed = self._default_seed, None
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        n_starts = self._usable_steps - self.episode_length + 1
        start = int(self.np_random.integers(n_starts)) if self.random_start else 0
        self._episode_start, self._episode_end = start, start + self.episode_length - 1
        self._t = start
        self._needs_reset = False
        self.community.reset()
        if self.forecast is not None:
            self.forecast.reset(self.np_random)
        obs = self._builder.build(start)
        info = self._info(start, start)
        info.update(episode_start=start, episode_end=self._episode_end,
                    obs_clipped=self._builder.last_clipped)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Apply one action per battery for the current step.

        Args:
            action: Array of shape (len(storage_node_ids),).

        Returns:
            tuple[np.ndarray, float, bool, bool, dict[str, Any]]: Observation of the next step (of
            the terminal observation at the end of the episode, see the class docstring), reward
            R_t, terminated (always False),
            truncated (True on the last step of the episode) and info (see `_step_info`, plus
            `obs_clipped`: values of the returned observation clipped to [0, 1] because they lie
            outside the normaliser's training range).

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
        reward = self._reward.compute(result)

        truncated = t == self._episode_end
        if truncated:
            self._needs_reset = True
            observed = t + 1 if t + 1 <= self._last_observable else t
        else:
            self._t = observed = t + 1
        obs = self._builder.build(observed)
        info = self._step_info(result, clipped, reward, observed)
        info['obs_clipped'] = self._builder.last_clipped
        return obs, float(reward.total), False, truncated, info

    def _step_info(self, result: CommunityStep, action: np.ndarray,
                   reward: RewardBreakdown, observed: int) -> dict[str, Any]:
        """
        Info of a step: reward components, welfare breakdown, K_t and per-node results.

        Args:
            result: Community step record.
            action: Action after clipping to [-1, 1].
            reward: Reward breakdown.
            observed: Step of the returned observation.

        Returns:
            dict[str, Any]: `t`, `timestamp`, `privileged`, `reward` (`type`, `main`, `penalty`
            = weighted penalty, `total` = R_t and, per-node variant, `nodes`: node id → main,
            penalty, total), `welfare` (breakdown), `penalty` (K_t), `action` (after the [-1, 1]
            clip) and `nodes` (node id → NodeStep fields).
        """
        info = self._info(result.t, observed)
        w = result.welfare
        reward_info: dict[str, Any] = {'type': self.reward_type, 'main': reward.main,
                                       'penalty': reward.penalty, 'total': reward.total}
        if reward.nodes is not None:
            reward_info['nodes'] = {
                node_id: {'main': r.main, 'penalty': r.penalty, 'total': r.total}
                for node_id, r in reward.nodes.items()
            }
        info.update(
            reward=reward_info,
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
