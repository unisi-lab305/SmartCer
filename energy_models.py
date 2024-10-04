from abc import ABC, abstractmethod
import numpy as np


class StorageSystem(ABC):
    """
    Abstract class representing an energy storage system.

    This class defines the interfaces for the basic operations that any storage system must implement,
    such as charging, resetting, and terminating.

    Abstract Methods:
        charge(energy: float, production: float) -> float:
            Method to charge or discharge energy from the storage system.

        reset() -> None:
            Resets the initial state of the storage system.

        terminate() -> None:
            Terminates and cleans up any resources or final states of the storage system.
    """

    @abstractmethod
    def charge(self, energy: float, production: float) -> float:
        """
        Abstract method for charging or discharging energy from the storage system.
        Must be implemented by derived classes.

        Args:
            energy (float): The energy to be charged or discharged.
            production (float): Energy production available for charging.

        Returns:
            float: Constraint violations for charging or discharging (if any).
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the initial state of the storage system.
        """
        pass

    def terminate(self) -> None:
        """
        Ends storage operations. This method can be used to free resources or save the final state, if necessary.
        """
        pass


class Battery(StorageSystem):
    """
    Class representing a battery as an energy storage system.

    Manages the charging and discharging of the battery, applying physical constraints such as maximum capacity,
    rated power, and efficiency losses. The battery experiences capacity degradation over time
    and tracks constraint violations during charging and discharging operations.

    Attributes:
        capacity (float): Maximum capacity of the battery (kWh).
        rated_power (float): Rated power of the battery (kW).
        soc_min (float): Minimum state of charge (SoC) of the battery (kWh).
        ETA (float): Efficiency of charging/discharging of the battery.
        ALFA (float): Loss coefficient for SoC calculation.
        BETA (float): Degradation coefficient of the battery.
        soc (float): Current state of charge of the battery (kWh).
        time_step (int): The time counter for monitoring simulation steps.
        soc_init_list (List[float]): List storing the initial state of charge for each step.
        soc_final_list (List[float]): List storing the final state of charge for each step.
        energy_desired (List[float]): List of desired energies during charging/discharging.
        actual_charges (List[float]): List of actually charged amounts.
        actual_discharges (List[float]): List of actually discharged amounts.
        constraint_violation (List[float]): List of constraint violations for charging/discharging.
    """

    def __init__(self, capacity: float, rated_power: float, soc_min: float, efficiency: float, alfa: float,
                 beta: float):
        """
        Initializes a new battery with the specified parameters.

        Args:
            capacity (float): Maximum capacity of the battery (kWh).
            rated_power (float): Rated power of the battery (kW).
            soc_min (float): Minimum state of charge (SoC) as a percentage of total capacity.
            efficiency (float): Efficiency of the battery (value between 0 and 1).
            alfa (float): Loss coefficient for charging/discharging.
            beta (float): Degradation coefficient of capacity.
        """
        self.capacity = capacity
        self.c0 = capacity
        self.rated_power = rated_power
        self.soc_min = soc_min * capacity
        self.ETA = efficiency
        self.ALFA = alfa
        self.BETA = beta

        # Attributes that will be reset
        self.soc = self.soc_min
        self.time_step = 0
        self.soc_init_list, self.soc_final_list = [], []
        self.energy_desired, self.actual_charges, self.actual_discharges = [], [], []
        self.constraint_violation = []

    def charge(self, energy: float, production: float) -> float:
        """
        Manages both the charging and discharging of the battery based on physical constraints and production
        conditions.

        If the energy is positive, an attempt is made to charge the battery. If it is negative, an attempt is made to
        discharge it.
        The function respects constraints on capacity, rated power, efficiency, and energy arbitrage.

        Args:
            energy (float): Energy to be charged (> 0) or discharged (< 0) in the battery.
            production (float): Available production from which to charge the battery (for arbitrage constraint).

        Returns:
            float: Violations of charging or discharging constraints of the battery.
        """
        self.energy_desired.append(energy)
        self.soc_init_list.append(self.soc)

        # Calculate new SoC (State of Charge)
        soc_init = self.soc * (1 - self.ALFA)

        # Charging the battery
        if energy > 0:
            # Rated power constraint
            energy = min(energy, self.rated_power)
            # Capacity constraint
            energy = min(energy, self.capacity - self.soc)
            # Energy arbitrage constraint
            charging_energy = min(energy, production)
            # Update SoC
            self.soc = soc_init + charging_energy * self.ETA
            discharging_energy = 0.
        else:
            # Rated power constraint for discharging
            energy = max(energy, -self.rated_power)
            # Capacity constraint for discharging
            discharging_energy = max(energy, min(self.soc_min - self.soc, 0))
            # Update SoC
            self.soc = soc_init + discharging_energy / self.ETA
            charging_energy = 0.

        self.actual_charges.append(charging_energy)
        self.actual_discharges.append(-discharging_energy)

        constrained_energy = charging_energy + discharging_energy
        violation = np.abs(constrained_energy - energy)
        violation = np.round(violation, 5)
        self.constraint_violation.append(violation)

        # Check SoC constraint
        assert 0. <= self.soc <= self.capacity, (
            f'SoC: {self.soc:.2f} is not valid.\n'
            f'Max capacity: {self.capacity}, Min SoC: {self.soc_min}\n'
            f'Initial SoC: {soc_init:.2f}, Energy: {constrained_energy:.2f}')
        self.soc_final_list.append(self.soc)

        # Energy exchanged with the environment
        if constrained_energy > 0:
            energy_trade = (self.soc - soc_init) / self.ETA
        else:
            energy_trade = (self.soc - soc_init) * self.ETA

        # Degradation of the battery's capacity
        try:
            self.capacity -= self.BETA * self.c0 * np.abs(energy_trade) / (2 * self.capacity)
        except ZeroDivisionError:
            self.capacity = 0.

        return violation

    def reset(self) -> None:
        """
        Resets the initial state of the battery, clearing all variable attributes.
        """
        self.soc = self.soc_min
        self.time_step = 0
        self.soc_init_list, self.soc_final_list = [], []
        self.energy_desired, self.actual_charges, self.actual_discharges = [], [], []
        self.constraint_violation = []
