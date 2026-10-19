"""
Baseline runs: one episode per controller, written to a run output directory.

The output directory is chosen by the caller and must lie outside the scenario folder and outside
any `cer_data` folder: 0.6.2 wrote its results into `cer_data/<scenario>/simulations/` (defect
S18).
"""

from collections.abc import Callable, Iterable, Mapping
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any

import gymnasium as gym

from smartcer.gym.cer_env import SmartCerEnv
from smartcer.gym.controllers.base import Controller
from smartcer.gym.controllers.naive import NaiveController
from smartcer.gym.controllers.rbc import RBCController
from smartcer.gym.tracking import ResultsTracker, TrackResults, write_json


# controller name → factory taking the environment
CONTROLLERS: Mapping[str, Callable[[gym.Env], Controller]] = MappingProxyType({
    NaiveController.name: NaiveController,
    'rbc_node': partial(RBCController, mode='node'),
    'rbc_community': partial(RBCController, mode='community'),
})


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
