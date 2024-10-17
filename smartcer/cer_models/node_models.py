from smartcer.cer_models.energy_models import Battery

from abc import abstractmethod
from typing import Tuple, Dict
from gymnasium import spaces
import pandas as pd
import numpy as np
import sys
import re
import os


class Node:
    """
    Base class to represent an energy node.

    Manages the loading of data from CSV files and provides methods to access information
    related to the node, such as imported energy, exported energy, and TIP coefficients.

    Attributes:
        data_path (str): Path to the CSV file containing node data.
        sim_start (str): Start date of the simulation period.
        sim_end (str): End date of the simulation period.
        node_id (str): Unique identifier of the node.
        node_type (str): Type of the node (e.g., 'Consumer', 'Producer', 'Prosumer').
        storage (StorageSystem, optional): Associated storage system for the node.
        time_step (int): Current time index.
        data (pd.DataFrame): Node data in DataFrame format.
        date_time (np.array): Array of timestamps extracted from the data.
        energy_expense (list): Energy costs.
        energy_revenue (list): Energy revenues.
        tip_coefficients (list): TIP coefficients.
        tip_contributions_prod (list): TIP contributions for production.
        tip_contributions_cons (list): TIP contributions for consumption.
        potential_tips_prod (list): Potential TIPs for production.
        potential_tips_cons (list): Potential TIPs for consumption.
        actual_tips_prod (list): Actual TIPs for production.
        actual_tips_cons (list): Actual TIPs for consumption.
    """

    def __init__(self, data_dir: str,
                 sim_period: Tuple[str, str], node_id: str, node_type: str, storage: Battery = None):
        """
        Initializes the Node class with data path, node ID, and node type.

        Args:
            data_dir (str): Directory containing node data.
            sim_period (Tuple[str, str]): Start and end dates of the simulation period.
            node_id (str): Identifier of the node.
            node_type (str): Type of the node.
            storage (StorageSystem, optional): Associated storage system for the node.
        """
        self.data_path = os.path.join(data_dir, f'{self._extract_base_name(node_id)}.csv')
        self.sim_start, self.sim_end = sim_period
        self.node_id = node_id
        self.node_type = node_type
        self.storage = storage
        self.observation_space = None

        # RESET
        self.time_step = 0
        self.data = pd.DataFrame()
        self.date_time = np.array([])
        self.energy_expense = []
        self.energy_revenue = []
        self.tip_coefficients = []
        self.tip_contributions_prod = []
        self.tip_contributions_cons = []
        self.potential_tips_prod = []
        self.potential_tips_cons = []
        self.actual_tips_prod = []
        self.actual_tips_cons = []

        self._load_data()

    def __repr__(self):
        return f'{self.__class__.__name__} → {self.node_id} [{self.node_type}]'

    @staticmethod
    def _extract_base_name(name: str) -> str:
        """
        Extracts the base name from a node identifier.

        Args:
            name (str): Identifier of the node.

        Returns:
            str: Base name of the node.
        """
        match = re.match(pattern='^[a-zA-Z0-9_]+(?=_\d+)', string=name)
        return match.group(0) if match else name

    def _load_data(self) -> None:
        """
        Loads and filters data from the CSV file into the DataFrame.

        Returns:
            None
        """
        data = pd.read_csv(self.data_path)
        filtered_data = data[(data['datetime'] >= self.sim_start) & (data['datetime'] <= self.sim_end)]
        self.data = filtered_data.reset_index(drop=True)
        self.date_time = self.data['datetime'].values

    @abstractmethod
    def set_observation_space(self, observation_space: Dict) -> None:
        """
        Sets the observation space for the node.

        Args:
            observation_space (Dict): minimum and maximum values for each observation.

        Returns:
            None
        """
        pass

    def get_data(self, column: str = None) -> pd.DataFrame:
        """
        Returns the node's data.

        Args:
            column (str, optional): Name of the column to return. If None, returns the entire DataFrame.

        Returns:
            pd.DataFrame: Node data or specified column.
        """
        return self.data if column is None else self.data[column]

    def get_data_columns(self) -> list:
        """
        Returns the list of available columns in the DataFrame.

        Returns:
            list: Names of the columns.
        """
        return self.data.columns.tolist()

    def get_current_data(self, column: str = None, time_step: int = None) -> np.array or float:
        """
        Returns the current data for the current time step.

        Args:
            column (str, optional): Name of the column to return. If None, returns the entire row.
            time_step (int, optional): Time index to retrieve data from. If None, returns the current time step.

        Returns:
            numpy array or float: Current data or specified value.
        """
        time_step = self.time_step if time_step is None else time_step
        return self.data.iloc[time_step].values if column is None else self.data[column][time_step]

    def set_storage(self, action: float) -> float:
        """
        Sets the energy to be charged or discharged from storage.

        Args:
            action (float): Control action for charging or discharging storage ∈ [-1,1].

        Returns:
            float: Normalized violation of the constraints.
        """
        energy = action * self.storage.capacity

        try:
            violation = self.storage.charge(energy, self.get_current_data('production'))
            return violation / self.storage.capacity

        except AssertionError as e:
            print(f'\n{self.node_id} - {self.time_step}')
            print(e)
            print(f'Prod: {self.get_current_data("production"):.2f}, Cons: {self.get_current_data("consumption"):.2f}, '
                  f'Action: {action:.2f}')
            sys.exit(1)

    def reset(self) -> None:
        """
        Resets the node state, restoring the time step and data to their initial state.

        Returns:
            None
        """
        self.time_step = 0
        self.data = pd.DataFrame()
        self.date_time = np.array([])
        self.energy_expense = []
        self.energy_revenue = []
        self.tip_coefficients = []
        self.tip_contributions_prod = []
        self.tip_contributions_cons = []
        self.potential_tips_prod = []
        self.potential_tips_cons = []
        self.actual_tips_prod = []
        self.actual_tips_cons = []

        if self.storage:
            self.storage.reset()

        self._load_data()

    def terminate(self) -> None:
        """
        This method can be used to free resources or save the final state, if necessary.
        """
        if self.storage:
            self.storage.terminate()


class Consumer(Node):
    """
    Class to represent a consumer node.

    Extends the Node class, adding specifics related to consumption and purchase prices.

    Attributes:
        consumption (np.array): Energy consumption of the node.
        price_imp (np.array): Purchase prices of energy.
        imported_energies (list): Imported energies.
    """

    def __init__(self, data_dir: str,
                 sim_period: Tuple[str, str], node_id: str, node_type: str, storage: Battery = None):
        """
        Initializes the Consumer class by inheriting from Node.

        Args:
            data_dir (str): Directory containing the node's data.
            sim_period (Tuple[str, str]): Start and end dates of the simulation period.
            node_id (str): Unique identifier of the node.
            node_type (str): Type of the node.
            storage (StorageSystem, optional): Associated storage system with the node.
        """
        super().__init__(data_dir, sim_period, node_id, node_type, storage)

        self.consumption = self.data['consumption'].values
        self.price_imp = self.data['price_imp'].values

        # RESET
        self.imported_energies = []
        self.energy_trade = []

    def set_observation_space(self, observation_space: Dict) -> None:
        """
        Sets the observation space for the node.

        Args:
            observation_space (Dict): minimum and maximum values for each observation.

        Returns:
            None
        """
        price_imp = observation_space['price_imp']
        consumption = observation_space['consumption']
        self.observation_space = spaces.Dict({
            'price_imp': spaces.Box(low=price_imp[0], high=price_imp[1], shape=(1,), dtype=np.float32),
            'consumption': spaces.Box(low=consumption[0], high=consumption[1], shape=(1,), dtype=np.float32)
        })

    def reset(self) -> None:
        """
        Resets the node state, restoring the time step and data to their initial state.

        Returns:
            None
        """
        super().reset()
        self.imported_energies = []
        self.energy_trade = []


class Producer(Node):
    """
    Class to represent a producer node.

    Extends the Node class, adding specifics related to production and selling prices.

    Attributes:
        pv_power (float): Power of the photovoltaic system.
        pnrr (float): Percentage of funding.
        tip_base (float): Base value of the TIP.
        tip_th (float): Threshold of the TIP.
        fi_zone (str): Zone correction factor.
        pz (str): Price zone.
        production (np.array): Energy production of the node.
        price_exp (np.array): Selling prices of energy.
        exported_energies (list): Exported energies.
    """

    def __init__(self, data_dir: str, sim_period: Tuple[str, str],
                 node_id: str, node_type: str, storage: Battery = None, **pv_args: Dict):
        """
        Initializes the Producer class by inheriting from Node.

        Args:
            data_dir (str): Directory containing the node's data.
            sim_period (Tuple[str, str]): Start and end dates of the simulation period.
            node_id (str): Unique identifier of the node.
            node_type (str): Type of the node.
            storage (StorageSystem, optional): Associated storage system with the node.
            **pv_args: Additional arguments related to the photovoltaic system.
        """
        super().__init__(data_dir, sim_period, node_id, node_type, storage)
        self.pv_power = pv_args['pv_power']
        self.pnrr = pv_args['pnrr']
        self.tip_base = pv_args['tip_base']
        self.tip_th = pv_args['tip_threshold']
        self.fi_zone = pv_args['fi_zone']
        self.pz = pv_args['pz']

        self.production = self.data['production'].values
        self.price_exp = self.data['price_exp'].values

        # RESET
        self.exported_energies = []
        self.energy_trade = []

    def set_observation_space(self, observation_space: Dict) -> None:
        """
        Sets the observation space for the node.

        Args:
            observation_space (Dict): minimum and maximum values for each observation.

        Returns:
            None
        """
        price_exp = observation_space['price_exp']
        production = observation_space['production']
        self.observation_space = spaces.Dict({
            'price_exp': spaces.Box(low=price_exp[0], high=price_exp[1], shape=(1,), dtype=np.float32),
            'production': spaces.Box(low=production[0], high=production[1], shape=(1,), dtype=np.float32)
        })

    def reset(self) -> None:
        """
        Resets the node state, restoring the time step and data to their initial state.

        Returns:
            None
        """
        super().reset()
        self.exported_energies = []
        self.energy_trade = []


class Prosumer(Node):
    """
    Class to represent a prosumer node.

    Extends the Node class, adding specifics related to both production and consumption.

    Attributes:
        pv_power (float): Power of the photovoltaic system.
        pnrr (float): Percentage of funding.
        tip_base (float): Base value of the TIP.
        tip_th (float): Threshold of the TIP.
        fi_zone (str): Zone correction factor.
        pz (str): Price zone.
        production (np.array): Energy production of the node.
        consumption (np.array): Energy consumption of the node.
        price_exp (np.array): Selling prices of energy.
        price_imp (np.array): Purchasing prices of energy.
        imported_energies (list): Imported energies.
        exported_energies (list): Exported energies.
    """

    def __init__(self, data_dir: str, sim_period: Tuple[str, str],
                 node_id: str, node_type: str, storage: Battery = None, **pv_args: Dict):
        """
        Initializes the Prosumer class by inheriting from Node.

        Args:
            data_dir (str): Directory containing the node's data.
            sim_period (Tuple[str, str]): Start and end dates of the simulation period.
            node_id (str): Unique identifier of the node.
            node_type (str): Type of the node.
            storage (StorageSystem, optional): Associated storage system with the node.
            **pv_args: Additional arguments related to the photovoltaic system.
        """
        super().__init__(data_dir, sim_period, node_id, node_type, storage)
        self.pv_power = pv_args['pv_power']
        self.pnrr = pv_args['pnrr']
        self.tip_base = pv_args['tip_base']
        self.tip_th = pv_args['tip_threshold']
        self.fi_zone = pv_args['fi_zone']
        self.pz = pv_args['pz']

        self.production = self.data['production'].values
        self.consumption = self.data['consumption'].values
        self.price_exp = self.data['price_exp'].values
        self.price_imp = self.data['price_imp'].values

        # RESET
        self.imported_energies = []
        self.exported_energies = []
        self.energy_trade = []

    def set_observation_space(self, observation_space: Dict) -> None:
        """
        Sets the observation space for the node.

        Args:
            observation_space (Dict): minimum and maximum values for each observation.

        Returns:
            None
        """
        price_exp = observation_space['price_exp']
        price_imp = observation_space['price_imp']
        production = observation_space['production']
        consumption = observation_space['consumption']
        self.observation_space = spaces.Dict({
            'price_exp': spaces.Box(low=price_exp[0], high=price_exp[1], shape=(1,), dtype=np.float32),
            'price_imp': spaces.Box(low=price_imp[0], high=price_imp[1], shape=(1,), dtype=np.float32),
            'production': spaces.Box(low=production[0], high=production[1], shape=(1,), dtype=np.float32),
            'consumption': spaces.Box(low=consumption[0], high=consumption[1], shape=(1,), dtype=np.float32)
        })

    def reset(self) -> None:
        """
        Resets the node state, restoring the time step and data to their initial state.

        Returns:
            None
        """
        super().reset()
        self.imported_energies = []
        self.exported_energies = []
        self.energy_trade = []
