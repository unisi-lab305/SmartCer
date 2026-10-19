"""
Step benchmark of `SmartCerEnv`: whole-period episodes with random actions.

Reports the construction time, then per episode the reset time, the time of the steps and the
mean time per step, and the speed-up against SmartCer 0.6.2 (43 s for one year of Leather, defect
S13 of docs/refactor/REFACTOR_v1.md). The random actions are drawn before the timed loop, so only
the environment is measured.

Usage:

    python benchmarks/bench_step.py                          # local Leather scenario, 3 episodes
    python benchmarks/bench_step.py --scenario path/to/scenario --episodes 5 --observation extended
    python benchmarks/bench_step.py --forecast mmfe_standard --horizon 12   # with forecasts
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import statistics
import time
import warnings

import numpy as np

from smartcer.data import TimestampWarning
from smartcer.gym import SmartCerEnv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = REPO_ROOT / 'smartcer' / 'cer_data' / 'Leather'
# one year of Leather with SmartCer 0.6.2 (S13)
REFERENCE_SECONDS = 43.0


@dataclass(frozen=True)
class EpisodeTiming:
    """
    Timing of one episode.

    Attributes:
        reset_s: Duration of `reset()` (s).
        steps_s: Duration of all `step()` calls (s).
        steps: Number of steps.
    """

    reset_s: float
    steps_s: float
    steps: int

    @property
    def per_step_us(self) -> float:
        """
        Mean duration of a step (µs).

        Returns:
            float: steps_s / steps in microseconds.
        """
        return self.steps_s / self.steps * 1e6


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Command-line arguments.

    Args:
        argv: Arguments; None reads `sys.argv`.

    Returns:
        argparse.Namespace: `scenario`, `episodes`, `seed`, `observation`, `start`, `end`,
        `forecast`, `horizon`.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n', maxsplit=1)[0])
    parser.add_argument('--scenario', type=Path, default=DEFAULT_SCENARIO,
                        help='scenario folder (default: smartcer/cer_data/Leather)')
    parser.add_argument('--episodes', type=int, default=3, help='number of episodes (default 3)')
    parser.add_argument('--seed', type=int, default=0, help='seed of the actions (default 0)')
    parser.add_argument('--observation', default='meco', help="observation preset (default 'meco')")
    parser.add_argument('--start', default=None, help='first timestamp of the period')
    parser.add_argument('--end', default=None, help='last timestamp of the period')
    parser.add_argument('--forecast', default=None,
                        help="forecast provider, e.g. 'perfect' or 'mmfe_standard' (default none)")
    parser.add_argument('--horizon', type=int, default=12,
                        help='forecast horizon in steps, with --forecast (default 12)')
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error('--episodes must be at least 1')
    if not (args.scenario / 'config.json').is_file():
        parser.error(f'no scenario in {args.scenario} (config.json missing)')
    return args


def run_episode(env: SmartCerEnv, actions: np.ndarray, seed: int) -> EpisodeTiming:
    """
    Run one whole-period episode with the given actions and time it.

    Args:
        env: Environment whose episode covers the whole period.
        actions: One row of actions per step.
        seed: Seed of the reset.

    Returns:
        EpisodeTiming: Reset and step durations.
    """
    start = time.perf_counter()
    env.reset(seed=seed)
    reset_s = time.perf_counter() - start
    start = time.perf_counter()
    for action in actions:
        env.step(action)
    return EpisodeTiming(reset_s, time.perf_counter() - start, len(actions))


def main(argv: Sequence[str] | None = None) -> list[EpisodeTiming]:
    """
    Run the benchmark and print the results.

    Args:
        argv: Command-line arguments; None reads `sys.argv`.

    Returns:
        list[EpisodeTiming]: Timing of every episode.
    """
    args = parse_args(argv)
    start = time.perf_counter()
    with warnings.catch_warnings():
        # the 0.6.2 scenarios have irregular timestamps; the loader reports them every time
        warnings.simplefilter('ignore', TimestampWarning)
        warnings.simplefilter('ignore', DeprecationWarning)
        env = SmartCerEnv(args.scenario, start=args.start, end=args.end,
                          observation=args.observation, forecast=args.forecast,
                          forecast_horizon=args.horizon if args.forecast else 0)
    construction_s = time.perf_counter() - start

    n_steps, n_actions = env.episode_length, env.action_space.shape[0]
    rng = np.random.default_rng(args.seed)
    print(f'scenario {env.community.name}: {len(env.community.nodes)} nodes, '
          f'{n_actions} batteries, {n_steps} steps of {env.community.delta_t} h, '
          f'observation {args.observation!r} ({env.observation_space.shape[0]} values)'
          + (f', forecast {args.forecast!r}, H = {args.horizon}' if args.forecast else ''))
    print(f'construction: {construction_s:.2f} s')
    timings = []
    for episode in range(args.episodes):
        actions = rng.uniform(-1.0, 1.0, size=(n_steps, n_actions)).astype(np.float32)
        timing = run_episode(env, actions, seed=args.seed + episode)
        timings.append(timing)
        print(f'episode {episode + 1}: reset {timing.reset_s * 1e3:.1f} ms, steps '
              f'{timing.steps_s:.2f} s, {timing.per_step_us:.0f} µs per step')
    best = min(t.reset_s + t.steps_s for t in timings)
    median = statistics.median(t.reset_s + t.steps_s for t in timings)
    print(f'episode time: best {best:.2f} s, median {median:.2f} s; '
          f'{statistics.median(t.per_step_us for t in timings):.0f} µs per step (median)')
    if n_steps == 8760:
        print(f'speed-up against 0.6.2 ({REFERENCE_SECONDS:.0f} s per year on Leather): '
              f'{REFERENCE_SECONDS / median:.1f}x (median)')
    return timings


if __name__ == '__main__':
    main()
