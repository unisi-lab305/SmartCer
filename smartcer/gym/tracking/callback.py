"""
Stable-Baselines3 callback that logs the KPIs of every completed episode and saves its tables.

Adapted from storage-node-environment (`storage_node_env/gym/tracking/callback.py`): it reads the
`ResultsTracker` of a `TrackResults` wrapper instead of a tracker inside the environment, and saves
every `save_freq` completed episodes instead of every N steps. Stable-Baselines3 is optional.
"""

from pathlib import Path
from typing import Any

from smartcer.gym.tracking.tracker import ResultsTracker


try:
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without Stable-Baselines3
    BaseCallback = object  # type: ignore[assignment, misc]
    SB3_AVAILABLE = False

# KPIs written to the SB3 logger at the end of every episode, under 'smartcer/<name>'
LOGGED_KPIS = ('welfare', 'reward', 'incentive', 'shared_energy_kwh', 'sharing_rate',
               'penalty_mean', 'equivalent_full_cycles')


class ResultsCallback(BaseCallback):  # type: ignore[misc, valid-type]
    """
    Logs the KPIs of each completed episode of the first environment and saves its tables.

    The training environment must be wrapped with `TrackResults` and vectorised in the same
    process (e.g. `DummyVecEnv`, the default of `make_vec_env`).

    Attributes:
        output_dir: Folder of the saved episodes (`episode_<n>/`, see `ResultsTracker.save`).
        save_freq: Save every `save_freq` completed episodes; 0 never saves.
        run_config: Run configuration written with every saved episode, if given.
    """

    def __init__(self, output_dir: str | Path, save_freq: int = 1,
                 run_config: dict[str, Any] | None = None, verbose: int = 0) -> None:
        """
        Create the callback.

        Args:
            output_dir: Output folder of the saved episodes.
            save_freq: Save every `save_freq` completed episodes (≥ 0; 0 only logs).
            run_config: Run configuration to write with each saved episode.
            verbose: Verbosity of the SB3 callback.

        Raises:
            ImportError: If Stable-Baselines3 is not installed.
            ValueError: If `save_freq` is negative.
        """
        if not SB3_AVAILABLE:
            raise ImportError('ResultsCallback needs stable-baselines3 (extra "rl")')
        if save_freq < 0:
            raise ValueError(f'save_freq must be ≥ 0, got {save_freq}')
        super().__init__(verbose)
        self.output_dir = Path(output_dir)
        self.save_freq = int(save_freq)
        self.run_config = run_config
        self._tracker: ResultsTracker | None = None
        self._seen = 0

    def _init_callback(self) -> None:
        """
        Find the tracker of the first training environment.

        Raises:
            RuntimeError: If the environment is not vectorised in this process or not wrapped with
                `TrackResults`.
        """
        envs = getattr(self.training_env, 'envs', None)
        if not envs:
            raise RuntimeError('ResultsCallback needs a vectorised environment in this process '
                               '(DummyVecEnv)')
        try:
            self._tracker = envs[0].get_wrapper_attr('tracker')
        except AttributeError as error:
            raise RuntimeError('wrap the training environment with '
                               'smartcer.gym.tracking.TrackResults') from error
        self._seen = self._tracker.episodes_completed

    def _on_step(self) -> bool:
        """
        Log and save the episode just completed, if any.

        Returns:
            bool: Always True (training continues).
        """
        tracker = self._tracker
        if tracker is None or tracker.episodes_completed == self._seen:
            return True
        self._seen = tracker.episodes_completed
        kpis = tracker.kpis()
        for name in LOGGED_KPIS:
            self.logger.record(f'smartcer/{name}', kpis[name])
        if self.save_freq and self._seen % self.save_freq == 0:
            folder = tracker.save(self.output_dir / f'episode_{self._seen:05d}', self.run_config)
            if self.verbose > 0:
                self.logger.info(f'episode {self._seen} saved to {folder}')
        return True
