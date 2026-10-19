"""
Abstract storage system.

A storage system is controlled by a normalised action a ∈ [-1, 1] (positive charges, negative
discharges) and exposes the dynamic bounds of that action for the coming step, as in
docs/FORMULATION.md sections 4-7.
"""

from abc import ABC, abstractmethod
from typing import Any


class StorageSystem(ABC):
    """
    Abstract class of an energy storage system controlled by a normalised action.

    Implementations keep their own state (e.g. the state of charge) between steps; the caller
    provides the time step and, for each step, the cap on the energy that may be charged.
    """

    @abstractmethod
    def get_bounds(self, delta_t: float, charge_cap_kwh: float | None) -> tuple[float, float]:
        """
        Dynamic action bounds for the coming step, given the current state.

        Args:
            delta_t: Duration of the step (hours).
            charge_cap_kwh: Maximum energy the node can supply for charging in the step (kWh);
                None means no cap.

        Returns:
            tuple[float, float]: (a_upper, a_lower), both in [0, 1]; feasible actions lie in
            [-a_lower, a_upper].
        """

    @abstractmethod
    def step(self, action: float, delta_t: float, charge_cap_kwh: float | None) -> Any:
        """
        Apply one action for one step and update the state.

        Args:
            action: Requested normalised action in [-1, 1].
            delta_t: Duration of the step (hours).
            charge_cap_kwh: Maximum energy the node can supply for charging in the step (kWh);
                None means no cap.

        Returns:
            Any: Implementation-specific record of the step.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Restore the initial state.
        """
