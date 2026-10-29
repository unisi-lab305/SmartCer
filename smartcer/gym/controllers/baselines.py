"""
Baseline runs, one episode per controller, and the replay check of the MILP oracle, written to a
run output directory.

The output directory is chosen by the caller and must lie outside the scenario folder and outside
any `cer_data` folder: 0.6.2 wrote its results into `cer_data/<scenario>/simulations/` (defect
S18).
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from smartcer.gym.cer_env import SmartCerEnv
from smartcer.gym.controllers.base import Controller
from smartcer.gym.controllers.naive import NaiveController
from smartcer.gym.controllers.rbc import RBCController
from smartcer.gym.controllers.replay import ReplayController
from smartcer.gym.metrics import PENALTY_TOLERANCE, daily_welfare, fit_score
from smartcer.gym.tracking import ResultsTracker, TrackResults, write_json


# controller name → factory taking the environment ('replay' needs an environment with an oracle)
CONTROLLERS: Mapping[str, Callable[[gym.Env], Controller]] = MappingProxyType({
    NaiveController.name: NaiveController,
    'rbc_node': partial(RBCController, mode='node'),
    'rbc_community': partial(RBCController, mode='community'),
    ReplayController.name: ReplayController,
})
REPLAY_DAILY_FILE = 'oracle_daily.csv'
REPLAY_SUMMARY_FILE = 'oracle_replay.json'


def run_episode(env: gym.Env, controller: Controller, seed: int | None = None) -> ResultsTracker:
    """
    Run one episode of an environment with a controller and record it.

    Args:
        env: A `SmartCerEnv` (possibly wrapped, but not with `TrackResults`).
        controller: Controller bound to the same environment.
        seed: Seed of `env.reset`.

    Returns:
        ResultsTracker: The tracker holding the completed episode.

    Raises:
        ValueError: If the controller is bound to another environment.
    """
    if controller.env is not env.unwrapped:
        raise ValueError('the controller is bound to another environment')
    tracked = TrackResults(env)
    obs, info = tracked.reset(seed=seed)
    controller.reset()
    done = False
    while not done:
        obs, _, terminated, truncated, info = tracked.step(controller.act(obs, info))
        done = terminated or truncated
    return tracked.tracker


def check_output_dir(output_dir: str | Path, env: SmartCerEnv) -> Path:
    """
    Refuse output directories inside the scenario folder or a `cer_data` folder (S18).

    Args:
        output_dir: Output directory.
        env: Environment whose scenario folder is protected.

    Returns:
        Path: The resolved output directory.

    Raises:
        ValueError: If the directory lies inside the scenario folder or a `cer_data` folder.
    """
    folder = Path(output_dir).resolve()
    if 'cer_data' in folder.parts:
        raise ValueError(f'output directory {folder} lies inside a cer_data folder, which holds '
                         f'read-only scenarios; choose a run output directory')
    if env.scenario_path is not None:
        scenario = env.scenario_path.resolve()
        if folder == scenario or scenario in folder.parents:
            raise ValueError(f'output directory {folder} lies inside the scenario folder '
                             f'{scenario}; choose a run output directory')
    return folder


def compute_baselines(env: SmartCerEnv, output_dir: str | Path,
                      controllers: Iterable[str] = ('naive', 'rbc_community'),
                      seed: int | None = None) -> dict[str, dict[str, Any]]:
    """
    Run one episode per baseline controller and write `<output_dir>/<controller>/`: the community
    table, one table per node, the KPIs and the run configuration (`ResultsTracker.save`).

    Args:
        env: The environment of the run (its period and episode settings are used).
        output_dir: Run output directory, outside the scenario folder and any `cer_data` folder.
        controllers: Names of `CONTROLLERS` to run. Defaults to the naive controller and the
            community RBC, the class A reference baseline.
        seed: Seed of every episode's reset.

    Returns:
        dict[str, dict[str, Any]]: Controller name → KPIs.

    Raises:
        ValueError: If a controller is unknown or the output directory is not allowed.
    """
    folder = check_output_dir(output_dir, env)
    names = list(controllers)
    unknown = [n for n in names if n not in CONTROLLERS]
    if unknown:
        raise ValueError(f'unknown controllers {unknown}; available: {list(CONTROLLERS)}')
    results = {}
    for name in names:
        tracker = run_episode(env, CONTROLLERS[name](env), seed=seed)
        tracker.save(folder / name, {**env.run_config(), 'controller': name})
        results[name] = tracker.kpis()
    write_json(folder / 'kpis.json', results)
    return results


@dataclass(frozen=True)
class ReplayCheck:
    """
    Result of replaying the oracle actions over the whole period (`check_replay`).

    Attributes:
        steps: Steps replayed.
        penalty_max: Largest K_t.
        penalty_steps: Steps with K_t above `PENALTY_TOLERANCE` (1e-9).
        soc_error_max_kwh: Largest ``|s_t - s*_t|`` between the replayed and the oracle state of
            charge, over steps and batteries (kWh).
        simultaneous_rows: Oracle rows of the period (step and battery) with both p^cha > 0 and
            p^dis > 0, which the replay cannot reproduce (`OracleProvider.simultaneous_rows`).
        clipped_rows: Oracle actions (step and battery) beyond a dynamic bound by at most
            `REPLAY_TOLERANCE_KWH` (1e-6 kWh), moved onto the bound without penalty
            (`ReplayController`).
        clipped_max_kwh: Largest of those violations (kWh).
        welfare: Cumulative welfare of the replay Σ_t W_t (€).
        objective: Cumulative MILP objective of the days of the period (€), or None without an
            objective file.
        relative_gap: ``|welfare - objective| / |objective|``, or None without an objective.
        rtol: Relative tolerance of the welfare check.
        fit_score: Fit Score of the replay's daily welfare against the daily objective, or None
            (no objective, fewer than two days, or a constant objective).
        passed: True if no step is penalised and, with an objective, relative_gap ≤ rtol.
    """

    steps: int
    penalty_max: float
    penalty_steps: int
    soc_error_max_kwh: float
    simultaneous_rows: int
    clipped_rows: int
    clipped_max_kwh: float
    welfare: float
    objective: float | None
    relative_gap: float | None
    rtol: float
    fit_score: float | None
    passed: bool


def check_replay(env: SmartCerEnv, output_dir: str | Path, rtol: float = 1e-3,
                 seed: int | None = None) -> ReplayCheck:
    """
    Play the oracle actions a* over the whole period and compare the result with the MILP.

    The period must consist of whole days (first step at midnight, last step ending at
    midnight), so that every day of the replay is compared with a whole MILP day, and it must
    start at the first step of the oracle file: the replay starts from `soc_initial`, which is the
    oracle's state only there. Oracle actions
    beyond a dynamic bound by at most 1e-6 kWh (solver round-off) are moved onto the bound and
    counted (`ReplayController`); larger violations are clipped and penalised by the environment.
    Checks that no step is penalised (K_t ≤ 1e-9) and, when the oracle has a daily objective, that
    the cumulative welfare lies within `rtol` of the cumulative objective. Writes
    `<output_dir>/oracle_daily.csv` (per day: `date`, `welfare` of the replay and, with an
    objective, `objective` and `difference` = welfare - objective) and
    `<output_dir>/oracle_replay.json` (the `ReplayCheck` fields and the run configuration).

    Args:
        env: Environment created with `oracle`, whose episode covers the whole period (no
            `episode_length` and no forecasts).
        output_dir: Run output directory, outside the scenario folder and any `cer_data` folder.
        rtol: Relative tolerance on the cumulative welfare, ≥ 0 (default 1e-3).
        seed: Seed of the reset.

    Returns:
        ReplayCheck: The comparison.

    Raises:
        ValueError: If the environment has no oracle, its episode does not cover the whole
            period, the period is not made of whole days or does not start at the first step
            of the oracle file, rtol is negative or the output directory is not allowed.
    """
    folder = check_output_dir(output_dir, env)
    if env.oracle is None:
        raise ValueError('check_replay needs an environment created with an oracle')
    if env.episode_length != env.community.n_steps:
        raise ValueError(f'the replay covers the whole period of {env.community.n_steps} steps, '
                         f'the episode has {env.episode_length}: create the environment without '
                         f'episode_length and forecasts')
    stamps = pd.DatetimeIndex(env.community.timestamps)
    end = stamps[-1] + pd.Timedelta(hours=env.community.delta_t)
    if stamps[0] != stamps[0].normalize() or end != end.normalize():
        raise ValueError(f'the replay compares whole days: the period {stamps[0]} → {stamps[-1]} '
                         f'must start at midnight and its last step must end at midnight')
    first = pd.Timestamp(env.oracle.first_timestamp)
    if stamps[0] != first:
        raise ValueError(f'the replay starts from soc_initial, the MILP\'s initial state of '
                         f'charge, which holds only at the first step of the oracle file: the '
                         f'period starts at {stamps[0]}, the oracle file at {first}; set start to '
                         f'{first}')
    if np.isnan(rtol) or rtol < 0:
        raise ValueError(f'rtol must be ≥ 0, got {rtol!r}')
    controller = ReplayController(env)
    tracker = run_episode(env, controller, seed=seed)
    table = tracker.community_table()
    nodes = tracker.node_tables()
    soc = np.column_stack([nodes[i]['soc_kwh'].to_numpy(dtype=np.float64)
                           for i in env.storage_node_ids])
    daily = daily_welfare(table)
    comparison = pd.DataFrame({'date': daily.index, 'welfare': daily.to_numpy()})
    welfare = float(table['welfare'].sum())
    objective = relative_gap = score = None
    if env.oracle.daily_objective is not None:
        target = env.oracle.daily_objective.loc[daily.index].to_numpy(dtype=np.float64)
        comparison['objective'] = target
        comparison['difference'] = comparison['welfare'] - target
        objective = float(target.sum())
        gap = abs(welfare - objective)
        relative_gap = gap / abs(objective) if objective != 0.0 else (0.0 if gap == 0.0
                                                                      else float('inf'))
        if len(target) >= 2 and np.ptp(target) > 0.0:
            score = fit_score(target, daily.to_numpy())
    penalty = table['penalty'].to_numpy(dtype=np.float64)
    penalty_steps = int(np.count_nonzero(penalty > PENALTY_TOLERANCE))
    result = ReplayCheck(
        steps=len(table),
        penalty_max=float(penalty.max()),
        penalty_steps=penalty_steps,
        soc_error_max_kwh=float(np.max(np.abs(soc - env.oracle.soc_kwh))),
        simultaneous_rows=env.oracle.simultaneous_rows,
        clipped_rows=controller.clipped_rows,
        clipped_max_kwh=controller.clipped_max_kwh,
        welfare=welfare,
        objective=objective,
        relative_gap=relative_gap,
        rtol=float(rtol),
        fit_score=score,
        passed=penalty_steps == 0 and (relative_gap is None or relative_gap <= rtol),
    )
    folder.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(folder / REPLAY_DAILY_FILE, index=False)
    write_json(folder / REPLAY_SUMMARY_FILE, {**asdict(result), 'run_config': env.run_config()})
    return result
