"""
Performance rules of `SmartCerEnv.step()` (phase 6): no pandas code runs inside a step, the
observation is built once per step, and the step benchmark `benchmarks/bench_step.py` runs.
"""

from collections.abc import Callable
import importlib.util
from pathlib import Path
import sys
from types import FrameType, ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from smartcer.gym import SmartCerEnv


pytestmark = pytest.mark.integration

PANDAS_DIR = str(Path(pd.__file__).parent)
BENCHMARK = Path(__file__).parents[3] / 'benchmarks' / 'bench_step.py'


def make_env(path: Path, oracle: Path) -> SmartCerEnv:
    """
    Environment exercising the optional step paths: extended preset with lookback, causal timing,
    per-node rewards, perfect forecasts and an oracle.

    Args:
        path: Scenario folder.
        oracle: Oracle folder of the scenario.

    Returns:
        SmartCerEnv: The environment.
    """
    return SmartCerEnv(path, observation='extended', lookback=2, observe_current_step=False,
                       reward='profit', reward_per_node=True, episode_length=24,
                       random_start=True, forecast='perfect', forecast_horizon=3, oracle=oracle)


def test_step_runs_no_pandas_code(mini_cer_v1_path: Path, synthetic_oracle: Path) -> None:
    """
    Test that no function of the pandas package is called inside `step()` (0.6.2 read every value
    through pandas, 86% of its step time, defect S13). The profiler sees every Python-level pandas
    call, which covers any access to a Series, DataFrame or Index (`.loc`, `.iloc`, `[]`,
    `.values`); it does not see compiled code alone, such as building a `pd.Timestamp`.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        synthetic_oracle: Oracle folder of the synthetic scenario.
    """
    env = make_env(mini_cer_v1_path, synthetic_oracle)
    env.reset(seed=0)
    actions = np.random.default_rng(0).uniform(-1.0, 1.0, size=(24, 2))
    called: set[str] = set()

    def profile(frame: FrameType, event: str, arg: Any) -> None:
        if event == 'call':
            called.add(frame.f_code.co_filename)

    sys.setprofile(profile)
    try:
        for action in actions:
            env.step(action)
    finally:
        sys.setprofile(None)

    assert any('smartcer' in name for name in called)
    assert [name for name in called if name.startswith(PANDAS_DIR)] == []


def test_observation_is_built_once_per_step(mini_cer_v1_path: Path, synthetic_oracle: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that reset and every step build the observation exactly once (0.6.2 computed it twice per
    step, S13).

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        synthetic_oracle: Oracle folder of the synthetic scenario.
        monkeypatch: Pytest monkeypatch fixture.
    """
    env = make_env(mini_cer_v1_path, synthetic_oracle)
    build: Callable[[int], np.ndarray] = env._builder.build
    calls: list[int] = []

    def counted(t: int) -> np.ndarray:
        calls.append(t)
        return build(t)

    monkeypatch.setattr(env._builder, 'build', counted)
    env.reset(seed=0)
    for _ in range(5):
        env.step(env.action_space.sample())

    assert len(calls) == 6
    assert calls == list(range(calls[0], calls[0] + 6))


def load_benchmark() -> ModuleType:
    """
    Import `benchmarks/bench_step.py`, which is a script and not part of the package.

    Returns:
        ModuleType: The benchmark module.
    """
    spec = importlib.util.spec_from_file_location('bench_step', BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_step_benchmark_runs(mini_cer_v1_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    Test the step benchmark on the synthetic scenario: one timing per episode of 48 steps.

    Args:
        mini_cer_v1_path: Path of the 1.0-format synthetic scenario.
        capsys: Pytest output capture.
    """
    bench = load_benchmark()
    timings = bench.main(['--scenario', str(mini_cer_v1_path), '--episodes', '2'])

    assert [t.steps for t in timings] == [48, 48]
    assert all(t.steps_s > 0.0 and t.per_step_us > 0.0 for t in timings)
    output = capsys.readouterr().out
    assert 'mini_cer_v1: 4 nodes, 2 batteries, 48 steps' in output
    assert 'µs per step' in output
    with pytest.raises(SystemExit):
        bench.main(['--scenario', str(mini_cer_v1_path.parent), '--episodes', '1'])
