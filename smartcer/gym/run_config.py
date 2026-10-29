"""
Mapping of the 0.6.2 run configuration (`cer_config`) to `SmartCerEnv` arguments
(docs/CONFIG.md section 6).
"""

from typing import Any
import warnings


# 0.6.2 keys that are read; every other key is reported as unused
_USED_KEYS = {
    ('simulation', 'start'), ('simulation', 'end'), ('simulation', 'seed'), ('reward', 'type'),
    ('reward', 'single_agent'), ('reward', 'constraints', 'penalty'),
    ('reward', 'constraints', 'coefficient'),
}


def _flatten(config: dict[str, Any], prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """
    Paths of the leaf keys of a nested dictionary.

    Args:
        config: Nested dictionary.
        prefix: Path of `config` itself.

    Returns:
        list[tuple[str, ...]]: One path per leaf.
    """
    paths = []
    for key, value in config.items():
        path = (*prefix, str(key))
        paths += _flatten(value, path) if isinstance(value, dict) and value else [path]
    return paths


def map_v062_run_config(cer_config: dict[str, Any]) -> dict[str, Any]:
    """
    Translate a 0.6.2 `cer_config` into `SmartCerEnv` keyword arguments, with a DeprecationWarning.

    `simulation.start` and `simulation.end` → `start`, `end`; `simulation.seed` → `seed` (used by
    the first `reset()` without a seed); `reward.type` and `reward.single_agent`:

    - 'profit' or the older 'cost' (defect S2), `single_agent` False → reward 'welfare';
    - 'profit' or 'cost', `single_agent` True → reward 'profit' per node;
    - 'energy' → reward 'energy', per node if `single_agent` is True.

    The per-node variant reports one reward per battery node in `info['reward']['nodes']`; the
    scalar reward is their sum. `reward.constraints.coefficient` → `penalty_coefficient` (0 when
    `reward.constraints.penalty` is False). Unused keys (e.g. the reward scaler, removed with
    defect S9) are listed in the warning.

    Args:
        cer_config: The 0.6.2 run configuration.

    Returns:
        dict[str, Any]: Keyword arguments for `SmartCerEnv`.

    Raises:
        ValueError: For an unknown reward type.
    """
    simulation = cer_config.get('simulation', {})
    reward = cer_config.get('reward', {})
    reward_type = reward.get('type', 'profit')
    if reward_type not in ('profit', 'cost', 'energy'):
        raise ValueError(f"unknown 0.6.2 reward type {reward_type!r}; expected 'profit', 'cost' "
                         f"or 'energy'")
    per_node = bool(reward.get('single_agent', False))
    constraints = reward.get('constraints', {})
    coefficient = float(constraints.get('coefficient', 10.0)) if constraints.get(
        'penalty', True) else 0.0

    if reward_type == 'energy':
        mapped = 'energy'
    else:
        mapped = 'profit' if per_node else 'welfare'
    kwargs: dict[str, Any] = {'reward': mapped, 'penalty_coefficient': coefficient}
    if per_node:
        kwargs['reward_per_node'] = True
    for key in ('start', 'end', 'seed'):
        if simulation.get(key) is not None:
            kwargs[key] = simulation[key]

    unused = ['.'.join(path) for path in _flatten(cer_config) if path not in _USED_KEYS]
    description = {'welfare': 'welfare reward W - σK', 'profit': 'per-node profit reward',
                   'energy': 'energy reward'}[mapped]
    message = (f'0.6.2 run configuration mapped to SmartCer-v1 arguments {kwargs}; reward.type '
               f'{reward_type!r} → {description}')
    if per_node:
        message += ('; single_agent=True → per-node rewards in info["reward"]["nodes"], scalar '
                    'reward = their sum')
    if unused:
        message += f'; keys not used: {", ".join(unused)}'
    warnings.warn(message, DeprecationWarning, stacklevel=3)
    return kwargs
