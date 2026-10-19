"""
Randomised invariants of `smartcer.core.storage.Battery` (docs/FORMULATION.md sections 5-7).

100,000 actions over random batteries, time steps Δt ∈ {0.25, 1.0} h and charge caps: the state of
charge stays in [S^min, S^max], the penalty is zero inside the bounds and follows section 7 outside,
and the energies are consistent with the dynamics of section 5.
"""

import numpy as np
import pytest

from smartcer.core.storage import Battery


pytestmark = pytest.mark.unit

N_BATTERIES = 1_000
N_STEPS = 100
TOLERANCE = 1e-9


def random_battery(rng: np.random.Generator) -> Battery:
    """
    Battery with random parameters.

    Args:
        rng: Random generator.

    Returns:
        Battery: The battery.
    """
    capacity = rng.uniform(1.0, 200.0)
    soc_min = rng.choice([0.0, rng.uniform(0.0, 0.5)])
    return Battery(
        capacity=capacity,
        power_charge_max=capacity * rng.uniform(0.1, 2.0),
        power_discharge_max=capacity * rng.uniform(0.1, 2.0),
        efficiency_charge=rng.uniform(0.7, 1.0),
        efficiency_discharge=rng.uniform(0.7, 1.0),
        soc_min=soc_min,
        soc_initial=rng.uniform(soc_min, 1.0),
        alpha=rng.choice([0.0, rng.uniform(0.0, 0.05)]),
        usage_cost=rng.uniform(0.0, 0.02),
    )


def random_actions(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Random action kinds and values for n steps.

    Kind 0 plays A^upper, kind 1 plays -A^lower (both known only at step time), kind 2 one of
    -1, 0, 1, kinds 3-5 a uniform value in [-1, 1].

    Args:
        rng: Random generator.
        n: Number of steps.

    Returns:
        tuple[np.ndarray, np.ndarray]: Kinds (int) and candidate values (float), shape (n,).
    """
    kinds = rng.integers(6, size=n)
    values = np.where(kinds == 2, rng.choice([-1.0, 0.0, 1.0], size=n), rng.uniform(-1.0, 1.0, n))
    return kinds, values


def random_caps(rng: np.random.Generator, n: int, max_cap: float) -> list[float | None]:
    """
    Random charge caps for n steps: None (30 %), 0 (about 35 %) or uniform in [0, max_cap].

    Args:
        rng: Random generator.
        n: Number of steps.
        max_cap: Upper end of the uniform caps (kWh).

    Returns:
        list[float | None]: Charge caps (kWh).
    """
    draw, uniform = rng.random(n), rng.uniform(0.0, max_cap, n)
    return [None if d < 0.3 else (0.0 if d < 0.65 else float(u)) for d, u in zip(draw, uniform)]


def test_random_actions_respect_soc_limits_and_penalty() -> None:
    """
    Test 100,000 random steps: SoC in [S^min, S^max] within 1e-9, applied action within the
    bounds, zero penalty for feasible actions, section 7 penalty otherwise, charge within the cap
    and Δs = η_cha · charge - discharge / η_dis after self-discharge.
    """
    rng = np.random.default_rng(20261019)
    steps = 0
    for _ in range(N_BATTERIES):
        battery = random_battery(rng)
        delta_t = float(rng.choice([0.25, 1.0]))
        kinds, values = random_actions(rng, N_STEPS)
        caps = random_caps(rng, N_STEPS, 1.5 * battery.power_charge_max * delta_t)
        for kind, value, charge_cap in zip(kinds, values, caps):
            a_upper, a_lower = battery.get_bounds(delta_t, charge_cap)
            action = a_upper if kind == 0 else (-a_lower if kind == 1 else float(value))
            soc_start = max(battery.soc_min_kwh, (1.0 - battery.alpha) * battery.soc_kwh)

            result = battery.step(action, delta_t, charge_cap)
            steps += 1

            assert battery.soc_min_kwh - TOLERANCE <= result.soc_kwh <= battery.capacity + TOLERANCE
            assert 0.0 <= a_lower <= 1.0 and 0.0 <= a_upper <= 1.0
            assert -a_lower <= result.action_applied <= a_upper
            if -a_lower <= action <= a_upper:
                assert result.penalty == 0.0 and result.action_applied == action
            expected_penalty = (max(0.0, action - a_upper) if action > 0
                                else max(0.0, -action - a_lower))
            assert abs(result.penalty - expected_penalty) <= 1e-12
            if charge_cap is not None:
                assert result.charge_kwh <= charge_cap + TOLERANCE
            delta_soc = (battery.efficiency_charge * result.charge_kwh
                         - result.discharge_kwh / battery.efficiency_discharge)
            assert abs(result.soc_kwh - soc_start - delta_soc) <= TOLERANCE

    assert steps == 100_000
