from cer_utils import *

from typing import Annotated, Literal, TypeVar, Any, Tuple, Dict, OrderedDict
import collections

import gymnasium as gym
from gymnasium import spaces

import numpy.typing as npt
import numpy as np

from datetime import datetime
import os

DType = TypeVar('DType', bound=np.generic)
Array1 = Annotated[npt.NDArray[DType], Literal[1]]
ArrayN = Annotated[npt.NDArray[DType], Literal['N']]


class Cer(gym.Env):
    """
    Class for the simulation of a Renewable Energy Community (CER).

    Attributes:
        cer_path (str): The path to the directory containing the CER data.
        preprocessing (bool): Flag to indicate if preprocessing is enabled.
        producers (dict): Dictionary of energy producers.
        consumers (dict): Dictionary of energy consumers.
        prosumers (dict): Dictionary of prosumers (production and consumption).
        nodes (dict): Dictionary of all CER nodes.
        producers_ids (list): List of IDs of producers.
        consumers_ids (list): List of IDs of consumers.
        prosumers_ids (list): List of IDs of prosumers.
        nodes_ids (list): List of IDs of all nodes.
        time_step (int): Index of the current time step of the simulation.
        cer_exported_energy (list): List of energies exported by the CER.
        cer_imported_energy (list): List of energies imported by the CER.
        cer_energy_expense (list): List of energy expenses of the CER.
        cer_energy_revenue (list): List of energy revenues of the CER.
        cer_energy_selfcons (list): List of CER self-consumed energy.
        cer_energy_selfcons_percentage (list): Percentages of CER self-consumption.
        cer_potential_tip (list): List of potential TIPs.
        cer_actual_tip (list): List of actual TIPs.
    """

    def __init__(self, cer_path: str, preprocessing: bool = False):
        """
        Initializes an instance of the Cer class.

        Args:
            cer_path (str): The path to the directory containing the CER data.
            preprocessing (bool, optional): Flag to indicate if preprocessing is enabled. Default is False.
        """
        self.cer_path = cer_path
        self.producers, self.consumers, self.prosumers, self.sim_hours = node_loader(cer_path)
        self.producers_ids = list(self.producers.keys())
        self.consumers_ids = list(self.consumers.keys())
        self.prosumers_ids = list(self.prosumers.keys())
        self.nodes = {**self.producers, **self.consumers, **self.prosumers}
        self.nodes_ids = list(self.nodes.keys())
        self.storage_nodes = {node_id: node for node_id, node in self.nodes.items() if node.storage}
        self.storage_nodes_ids = list(self.storage_nodes.keys())
        self.preprocessing = preprocessing

        # PREPROCESSING
        self.encoder = self._get_encoder()

        # GYM ATTRIBUTES
        # action space
        node_action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)
        self.action_space = spaces.Dict({node_id: node_action_space for node_id in self.storage_nodes_ids})
        # observation space
        self.observation_space = self._get_obs_space()
        # reward range
        self.reward_range = (-np.inf, np.inf)

        # RESET
        self.time_step = 0
        self.cer_date_time = self.nodes[self.nodes_ids[0]].date_time
        self.cer_exported_energy = []
        self.cer_imported_energy = []
        self.cer_energy_expense = []
        self.cer_energy_revenue = []
        self.cer_energy_selfcons = []
        self.cer_energy_selfcons_percentage = []
        self.cer_potential_tip = []
        self.cer_actual_tip = []

    def _get_encoder(self):
        # TODO: implement
        raise NotImplementedError

    def _get_obs(self, preprocessing: bool = False, horizon: int = 1) -> OrderedDict[str, ArrayN[np.float32]]:
        """
        Get the observation for the current time step.

        Args:
            horizon (int, optional): Number of time steps to consider in the observation. Default is 1.

        Returns:
            OrderedDict[str, NDArray]: Observation for the current time step.
        """
        # TODO: add horizon and remove unnecessary data
        time_step = self.time_step % self.sim_hours  # handle _get_obs() for time steps > sim_hours
        date_time = datetime.strptime(self.cer_date_time[time_step], '%Y-%m-%d %H:%M:%S')
        year, month, day_type, hour = date_time.year, date_time.month, date_time.weekday(), date_time.hour
        cer_production = sum([node.get_current_data('production', time_step)
                              for node in {**self.producers, **self.prosumers}.values()])
        cer_consumption = sum([node.get_current_data('consumption', time_step)
                               for node in {**self.consumers, **self.prosumers}.values()])
        obs = {
            'cer': {
                'year': year,
                'month': month,
                'day_type': day_type,
                'hour': hour,
                'production': cer_production,
                'consumption': cer_consumption
            },
            **{node_id: {
                'price_exp': node.get_current_data('price_exp', time_step) if node_id not in self.consumers_ids else 0.,
                'price_imp': node.get_current_data('price_imp', time_step) if node_id not in self.producers_ids else 0.,
                'production': node.get_current_data('production', time_step) if node_id not in self.consumers_ids else 0.,
                'consumption': node.get_current_data('consumption', time_step) if node_id not in self.producers_ids else 0.,
                'soc': node.storage.soc
            } for node_id, node in self.storage_nodes.items()}
        }
        # TODO: implement
        if preprocessing:
            obs *= self.encoder

        # return collections.OrderedDict(sorted(obs.items()))
        return obs

    def _get_obs_space(self):
        # TODO: modify according to preprocessing
        # TODO: parameterize Box shape
        cer_space = spaces.Box(low=-1., high=1., shape=(6,), dtype=np.float32)
        node_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        observation_space = spaces.Dict(
            {'cer': cer_space, **{node_id: node_space for node_id in self.nodes_ids}}
        )
        return observation_space

    def _get_reward(self):
        # TODO: implement
        return 0.

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information for the current time step.
        """
        # TODO: implement
        return {}

    def _next_hour(self):
        """
        Advances to the next time step in the simulation, updating the time_step for all nodes.
        **To be called at the end of the step() method.**
        """
        self.time_step += 1
        for node in self.nodes.values():
            node.time_step = self.time_step
            if node.storage:
                node.storage.time_step = self.time_step

    def _is_terminal(self) -> bool:
        """
        Check if the simulation has reached the terminal state.
        **To be called after the next_hour() method.**

        Returns:
            bool: True if the simulation has reached the terminal state, False otherwise.
        """
        return self.time_step == len(self.nodes[self.nodes_ids[0]].date_time)

    def step(self, actions: OrderedDict[str, Array1[np.float32]]) \
            -> Tuple[OrderedDict[str, ArrayN[np.float32]], float, bool, bool, Dict[str, Any], None]:
        """
        Performs a simulation step, calculating the potential and actual TIP for producers, consumers, and prosumers.
        Updates exported and imported energies, self-consumption, and TIP distributions.

        Args:
            actions [OrderedDict[str, NDArray[(1,), np.float32]]: Action

        Returns:
            Tuple[OrderedDict[str, NDArray[(N,), np.float32]], float, bool, bool, Dict[str, Any], None]: Observation,
            reward, termination flag,truncation flag, additional information, and done
        """
        # apply charge and discharge actions
        if actions:
            for node_id, action in actions.items():
                _ = self.nodes[node_id].set_storage(action[0])

        # compute potential/actual TIP and energy costs for each node
        for producer in self.producers.values():
            energy_produced = producer.get_current_data('production')
            energy_consumed = 0.
            energy_exported, energy_imported = get_energy_trade(producer, energy_produced, energy_consumed)
            assert energy_imported >= 0, (f'Energy arbitrage constraint violation -> '
                                          f'{producer.node_id} imported {energy_imported:.2f} kWh')
            producer.exported_energies.append(energy_exported)
            set_tip_and_costs(producer, energy_exported, energy_imported, is_producer=True)

        for consumer in self.consumers.values():
            energy_imported = consumer.get_current_data('consumption')
            consumer.imported_energies.append(energy_imported)
            set_tip_and_costs(consumer, 0, energy_imported, is_consumer=True)

        for prosumer in self.prosumers.values():
            energy_produced = prosumer.get_current_data('production')
            energy_consumed = prosumer.get_current_data('consumption')
            energy_exported, energy_imported = get_energy_trade(prosumer, energy_produced, energy_consumed)
            assert energy_imported * energy_exported == 0, f'Simultaneous import-export -> ' \
                                                           f'imp: {energy_imported:.2f}, exp: {energy_exported:.2f}'
            prosumer.exported_energies.append(energy_exported)
            prosumer.imported_energies.append(energy_imported)
            set_tip_and_costs(prosumer, energy_exported, energy_imported)

        # CER imported and exported energy
        self.cer_exported_energy.append(
            sum([node.exported_energies[self.time_step] for node in {**self.producers, **self.prosumers}.values()]))
        self.cer_imported_energy.append(
            sum([node.imported_energies[self.time_step] for node in {**self.consumers, **self.prosumers}.values()]))

        # CER energy costs and revenues
        self.cer_energy_expense.append(sum([node.energy_expense[self.time_step] for node in self.nodes.values()]))
        self.cer_energy_revenue.append(sum([node.energy_revenue[self.time_step] for node in self.nodes.values()]))

        # CER potential TIP
        self.cer_potential_tip.append(
            sum([node.potential_tips_prod[self.time_step] for node in {**self.producers, **self.prosumers}.values()]))

        # CER self-consumed energy
        self.cer_energy_selfcons.append(
            min(self.cer_exported_energy[self.time_step], self.cer_imported_energy[self.time_step]))
        try:
            self.cer_energy_selfcons_percentage.append(
                self.cer_energy_selfcons[self.time_step] / self.cer_exported_energy[self.time_step])
        except ZeroDivisionError:
            self.cer_energy_selfcons_percentage.append(0.)

        # CER actual TIP
        self.cer_actual_tip.append(
            self.cer_energy_selfcons_percentage[self.time_step] * self.cer_potential_tip[self.time_step])

        self._next_hour()

        next_obs = self._get_obs(preprocessing=self.preprocessing)
        reward = self._get_reward()
        terminated = self._is_terminal()
        truncated = terminated
        info = self._get_info()
        done = None

        return next_obs, reward, terminated, truncated, info, done

    def reset(self, seed: int | None = None, options: Dict[str, Any] = None) \
            -> Tuple[OrderedDict[str, ArrayN[np.float32]], Dict[str, Any]]:
        """
        Reset the environment to the initial state, resetting all nodes.

        Args:
            seed (int, optional): Random seed for the environment. Default is None.
            options (dict, optional): Additional information to specify how the environment is reset. Default is None.

        Returns:
            Tuple[OrderedDict[str, NDArray], Dict[str, Any]]: Observation and additional information
        """
        super().reset(seed=seed)

        self.time_step = 0
        self.cer_date_time = self.nodes[self.nodes_ids[0]].date_time
        self.cer_exported_energy = []
        self.cer_imported_energy = []
        self.cer_energy_expense = []
        self.cer_energy_revenue = []
        self.cer_energy_selfcons = []
        self.cer_energy_selfcons_percentage = []
        self.cer_potential_tip = []
        self.cer_actual_tip = []
        for node in self.nodes.values():
            node.reset()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def get_sim_results(self) -> dict:
        """
        Get simulation results in a structured format.

        Returns:
            dict: Simulation results, including imported energy, exported energy, and TIP.
        """

        # CER TABLE
        cer_table = {
            'datetime': self.cer_date_time[0:self.time_step + 1].tolist(),
            'imported_energy': self.cer_imported_energy[0:self.time_step + 1],
            'exported_energy': self.cer_exported_energy[0:self.time_step + 1],
            'selfconsumption': self.cer_energy_selfcons[0:self.time_step + 1],
            'selfconsumption_percentage': self.cer_energy_selfcons_percentage[0:self.time_step + 1],
            'energy_expense': self.cer_energy_expense[0:self.time_step + 1],
            'energy_revenue': self.cer_energy_revenue[0:self.time_step + 1],
            'tip_potential': self.cer_potential_tip[0:self.time_step + 1],
            'tip_actual': self.cer_actual_tip[0:self.time_step + 1],
        }

        # NODE TABLES
        def _create_node_table(node, time_step):
            table = {
                'datetime': node.date_time[0:time_step + 1].tolist(),
            }
            if hasattr(node, 'price_exp'):
                table['price_export'] = node.price_exp[0:time_step + 1].tolist()
            if hasattr(node, 'price_imp'):
                table['price_import'] = node.price_imp[0:time_step + 1].tolist()
            if hasattr(node, 'production'):
                table['energy_produced'] = node.production[0:time_step + 1].tolist()
            if hasattr(node, 'consumption'):
                table['energy_consumed'] = node.consumption[0:time_step + 1].tolist()
            if node.storage:
                table.update({
                    'soc_init': node.storage.soc_init_list[0:time_step + 1],
                    'action_desired': node.storage.energy_desired[0:time_step + 1],
                    'charge': node.storage.actual_charges[0:time_step + 1],
                    'discharge': node.storage.actual_discharges[0:time_step + 1],
                    'constraint_violation': node.storage.constraint_violation[0:time_step + 1],
                    'soc_final': node.storage.soc_final_list[0:time_step + 1],
                })
            if hasattr(node, 'exported_energies'):
                table['energy_exported'] = node.exported_energies[0:time_step + 1]
            if hasattr(node, 'imported_energies'):
                table['energy_imported'] = node.imported_energies[0:time_step + 1]
            return table

        producer_tables = {node.node_id: _create_node_table(node, self.time_step) for node in self.producers.values()}
        consumer_tables = {node.node_id: _create_node_table(node, self.time_step) for node in self.consumers.values()}
        prosumer_tables = {node.node_id: _create_node_table(node, self.time_step) for node in self.prosumers.values()}

        results = {
            'cer': cer_table,
            'producers': producer_tables,
            'consumers': consumer_tables,
            'prosumers': prosumer_tables,
        }

        return results

    def save_results(self, results: dict, save_mode: str = 'csv'):
        """
        Saves the simulation results in a structured format.

        Args:
            results (dict): Simulation results, including imported, exported energy, and TIP.
            save_mode (str, optional): The mode for saving results, either 'csv' or 'db'. Default is 'csv'.
        """
        # Create a directory for simulation results if it does not exist
        if not os.path.exists(f'{self.cer_path}/simulation'):
            os.makedirs(f'{self.cer_path}/simulation')

        assert save_mode in ['csv', 'db'], (f'Unrecognized save mode: {save_mode}. '
                                            f'Select between "csv" and "db"')

        if save_mode == 'csv':
            save_results_to_csv(csv_path=f'{self.cer_path}/simulation', results=results)
        elif save_mode == 'db':
            save_results_to_db(db_path=f'{self.cer_path}/simulation', results=results)
