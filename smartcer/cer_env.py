from smartcer.storage_controllers.rbc_controller import StandardRBC
from smartcer.cer_preprocessing.preprocessing import *
from smartcer.cer_metrics.scores import *
from smartcer.cer_reward.reward import *
from smartcer.cer_utils import *

from typing import Annotated, Literal, TypeVar, Any, Tuple, Dict, OrderedDict

import gymnasium as gym
from gymnasium import spaces

import numpy.typing as npt
import numpy as np

from datetime import datetime
from tqdm import tqdm
import os

DType = TypeVar('DType', bound=np.generic)
Array1 = Annotated[npt.NDArray[DType], Literal[1]]
ArrayN = Annotated[npt.NDArray[DType], Literal['N']]


class SmartCer(gym.Env):
    """
    Class for the simulation of a Renewable Energy Community (CER).

    Attributes:
        CER Environment:
            cer_path (str): The path to the directory containing the CER data.
            cer_config (dict): Configuration for the CER environment.
            reward_config (dict): Configuration for the reward system.
            producers (dict): Dictionary of energy producers.
            producers_ids (list): List of IDs of producers.
            consumers (dict): Dictionary of energy consumers.
            consumers_ids (list): List of IDs of consumers.
            prosumers (dict): Dictionary of prosumers (production and consumption).
            prosumers_ids (list): List of IDs of prosumers.
            nodes (dict): Dictionary of all CER nodes.
            nodes_ids (list): List of IDs of all nodes.
            storage_nodes (dict): Dictionary of storage nodes.
            storage_nodes_ids (list): List of IDs of storage nodes.
        Methods:
            flatten_observation (staticmethod): Method to flatten the observation space.
            create_action_dict (staticmethod): Method to create an action dictionary.
        Attributes to be reset:
            time_step (int): Index of the current time step of the simulation.
            cer_date_time (list): List of date-time strings for the simulation.
            years (set): Set of years in the simulation.
            months (set): Set of months in the simulation.
            day_types (set): Set of day types in the simulation.
            max_hour (int): Maximum hour in the simulation.
            cer_exported_energy (list): List of energies exported by the CER.
            cer_imported_energy (list): List of energies imported by the CER.
            cer_energy_expense (list): List of energy expenses of the CER.
            cer_energy_revenue (list): List of energy revenues of the CER.
            cer_energy_selfcons (list): List of CER self-consumed energy.
            cer_energy_selfcons_percentage (list): Percentages of CER self-consumption.
            cer_potential_tip (list): List of potential TIPs.
            cer_actual_tip (list): List of actual TIPs.
        Encoder:
            encoder (dict): Encoder for the CER environment.
        Gym Attributes:
            action_space (spaces.Dict): Action space for the CER environment.
            observation_space (spaces.Dict): Observation space for the CER environment.
            reward_range (Tuple[float, float]): Reward range for the CER environment.
    """

    def __init__(self, cer_path: str, cer_config: Dict):

        """
        Initializes an instance of the SmartCer class.

        Args:
            cer_path (str): The path to the directory containing the CER data.
            cer_config (dict): Configuration for the CER environment.
        """
        self.cer_path = cer_path
        self.cer_config = cer_config
        self.reward_config = cer_config['reward']
        self.producers, self.consumers, self.prosumers, self.sim_hours = node_loader(
            cer_path=cer_path, cer_config=cer_config)
        self.producers_ids = list(self.producers.keys())
        self.consumers_ids = list(self.consumers.keys())
        self.prosumers_ids = list(self.prosumers.keys())
        self.nodes = {**self.producers, **self.consumers, **self.prosumers}
        self.nodes_ids = list(self.nodes.keys())
        self.storage_nodes = {node_id: node for node_id, node in self.nodes.items() if node.storage}
        self.storage_nodes_ids = list(self.storage_nodes.keys())

        # METHODS
        self.create_action_dict = staticmethod(create_action_dict)
        self.flatten_observation = staticmethod(flatten_observation)

        # RESET
        self.time_step = 0
        self.cer_date_time = self.nodes[self.nodes_ids[0]].date_time
        self.years, self.months, self.day_types, self.max_hour = set(), set(), set(), 0
        for dt in [datetime.strptime(dt, '%Y-%m-%d %H:%M:%S') for dt in self.cer_date_time]:
            self.years.add(dt.year)
            self.months.add(dt.month)
            self.day_types.add(dt.weekday())
            if dt.hour > self.max_hour:
                self.max_hour = dt.hour
        self.cer_exported_energy = []
        self.cer_imported_energy = []
        self.cer_energy_expense = []
        self.cer_energy_revenue = []
        self.cer_energy_selfcons = []
        self.cer_energy_selfcons_percentage = []
        self.cer_potential_tip = []
        self.cer_actual_tip = []
        self.rewards = []
        self.penalties = []
        self.reward_signals = []

        # PREPROCESSING
        self.encoder = self._get_encoder()

        # GYM ATTRIBUTES
        # action space
        node_action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)
        self.action_space = spaces.Dict({node_id: node_action_space for node_id in self.storage_nodes_ids})
        # observation space
        self.observation_space = self._get_obs_space()
        # reward
        self.reward_range = (-np.inf, np.inf)
        self.reward = Reward(
            nodes=self.nodes, **self.reward_config)
        self.reward_scaler = scaler_loader(
            single_agent_reward=self.reward_config['single_agent'],
            storage_node_ids=self.storage_nodes_ids,
            **self.reward_config['scaler'])

    def __repr__(self):
        return f'CER Environment: {self.__class__.__name__} → {self.cer_path}'

    def _get_encoder(self) -> Dict[str, Any]:
        """
        Get the encoder for the CER environment based on node observation spaces.

        Returns:
            Dict[str, Union[OneHotEncoding, PeriodicNormalization, MinMax, Scale]]: Encoder for the CER environment.
        """
        # get min and max values for production, consumption
        max_prod, min_prod = np.zeros(1), np.zeros(1)
        min_cons, max_cons = np.zeros(1), np.zeros(1)
        for node_id, node in self.nodes.items():
            production_space = node.observation_space.get('production', None)
            if isinstance(production_space, spaces.Box):
                min_prod += production_space.low
                max_prod += production_space.high
            consumption_space = node.observation_space.get('consumption', None)
            if isinstance(consumption_space, spaces.Box):
                min_cons += consumption_space.low
                max_cons += consumption_space.high

        encoder = {
            'cer': {
                'year': OneHotEncoding(list(self.years)),
                'month': OneHotEncoding(list(self.months)),
                'day_type': OneHotEncoding(list(self.day_types)),
                'hour': PeriodicNormalization(self.max_hour),
                'cer_production': MinMax(min_prod, max_prod),
                'cer_consumption': MinMax(min_cons, max_cons)
            },
            **{node_id: {
                'price_exp': MinMax(node.price_exp.min(), node.price_exp.max())
                if node_id not in self.consumers_ids else RemoveFeature(),
                'price_imp': MinMax(node.price_imp.min(), node.price_imp.max())
                if node_id not in self.producers_ids else RemoveFeature(),
                'production': MinMax(node.production.min(), node.production.max())
                if node_id not in self.consumers_ids else RemoveFeature(),
                'consumption': MinMax(node.consumption.min(), node.consumption.max())
                if node_id not in self.producers_ids else RemoveFeature(),
                'soc': Scale(node.storage.capacity),
                'upper_bound': Scale(node.storage.rated_power),
                'lower_bound': Scale(node.storage.rated_power)
            } for node_id, node in self.storage_nodes.items()}
        }

        # assert observation space is compatible with encoder
        def compare_dict_keys(dict1, dict2, path=""):
            keys1, keys2 = set(dict1.keys()), set(dict2.keys())
            if keys1 != keys2:
                raise AssertionError(f'Mismatch between observation and encoder at '
                                     f'{path or "root"}: {keys1.symmetric_difference(keys2)}')
            # recursively check nested dictionaries
            for key in keys1:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    compare_dict_keys(dict1[key], dict2[key], path + f'{key} -> ')
                elif isinstance(dict1[key], dict) or isinstance(dict2[key], dict):
                    raise AssertionError(f'Key \'{key}\' is a dict in one but not the other at {path or "root"}')

        obs = self._get_obs(preprocessing=False)
        compare_dict_keys(obs, encoder)

        return encoder

    def _get_obs(self, preprocessing: bool = True, horizon: int = 1) -> Dict[str, Any]:
        """
        Get the observation for the current time step.

        Args:
            horizon (int, optional): Number of time steps to consider in the observation. Default is 1.

        Returns:
            OrderedDict[str, NDArray]: Observation for the current time step.
        """
        # TODO: add horizon
        time_step = self.time_step % self.sim_hours  # handle _get_obs() for time steps > sim_hours
        date_time = datetime.strptime(self.cer_date_time[time_step], '%Y-%m-%d %H:%M:%S')
        year, month, day_type, hour = date_time.year, date_time.month, date_time.weekday(), date_time.hour
        cer_production = sum([node.get_current_data('production', time_step)
                              for node in {**self.producers, **self.prosumers}.values()])
        cer_consumption = sum([node.get_current_data('consumption', time_step)
                               for node in {**self.consumers, **self.prosumers}.values()])

        upper_bounds, lower_bounds = {}, {}
        for node_id, node in self.storage_nodes.items():
            # TODO handle consumers
            upper_bounds[node_id] = min(node.storage.rated_power, node.storage.capacity - node.storage.soc,
                                        node.get_current_data('production', time_step))
            lower_bounds[node_id] = min(node.storage.rated_power,
                                        max(0, node.storage.soc - node.storage.soc_min))

        obs = {
            'cer': {
                'year': year,
                'month': month,
                'day_type': day_type,
                'hour': hour,
                'cer_production': cer_production,
                'cer_consumption': cer_consumption
            },
            **{node_id: {
                'price_exp': node.get_current_data('price_exp', time_step)
                if node_id not in self.consumers_ids else 0.,
                'price_imp': node.get_current_data('price_imp', time_step)
                if node_id not in self.producers_ids else 0.,
                'production': node.get_current_data('production', time_step)
                if node_id not in self.consumers_ids else 0.,
                'consumption': node.get_current_data('consumption', time_step)
                if node_id not in self.producers_ids else 0.,
                'soc': node.storage.soc,
                'upper_bound': upper_bounds[node_id],
                'lower_bound': lower_bounds[node_id]
            } for node_id, node in self.storage_nodes.items()}
        }
        if preprocessing:
            obs = encode_obs(obs, self.encoder)

        return obs

    def _get_obs_space(self) -> spaces.Dict:
        """
        Get the observation space (NON Flattenable) for the CER environment.

        Returns:
            spaces.Dict: Observation space for the CER environment.
        """
        # TODO: add horizon and remove unnecessary features (obs_space does not match obs dim)
        observation_space = spaces.Dict({
            'cer': spaces.Dict({
                'year': OHESpace(size=len(self.years)),
                'month': OHESpace(size=len(self.months)),
                'day_type': OHESpace(size=len(self.day_types)),
                'hour': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            }),
            **{node_id: spaces.Dict({
                'price_exp': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'price_imp': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'production': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'consumption': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'soc': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'upper_bound': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'lower_bound': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            }) for node_id in self.storage_nodes_ids}
        })
        return observation_space

    def _get_reward(self) -> Dict[str, List[float]] or List[float]:
        """
        Get the reward for the current time step.
        Could be a single reward or a dictionary of rewards for each node, depending on the reward type.

        Returns:
            Dict[str, List[float]] or List[float]: Reward, penalty, and reward signal for the current time step.
        """
        # compute reward
        if self.reward.type == 'energy':
            reward = self.reward.compute_energy_reward(time_step=self.time_step)
        elif self.reward.type == 'profit':
            reward = self.reward.compute_profit_reward(tip=self.cer_actual_tip[self.time_step], time_step=self.time_step)
        else:
            raise ValueError(f'Invalid reward type: {self.reward.type}')

        # compute penalty
        if self.reward.add_penalty:
            penalty = self.reward.compute_penalty(time_step=self.time_step)
        else:
            penalty = {} if self.reward.single_agent else None

        # dict of reward signals for each node
        if isinstance(reward, dict) and isinstance(penalty, dict):
            rewards_signals = {}
            for node_id in reward.keys():
                # scale reward
                if self.reward_config['scaler']['apply']:
                    self.reward_scaler[node_id].update(reward[node_id])
                    reward[node_id] = self.reward_scaler[node_id].scale(reward[node_id])
                # add penalty
                if penalty:
                    reward_signal = reward[node_id] - penalty[node_id]
                    rewards_signals[node_id] = reward_signal
                else:
                    rewards_signals[node_id] = reward[node_id]
            # update list attributes with dict
            self.rewards.append(reward)
            self.penalties.append(penalty)
            self.reward_signals.append(rewards_signals)
            return [reward, penalty, rewards_signals]

        # collective reward signal
        else:
            # scale reward
            if self.reward_config['scaler']['apply']:
                self.reward_scaler.update(reward)
                reward = self.reward_scaler.scale(reward)
            # add penalty
            if penalty:
                reward_signal = reward - penalty
            else:
                reward_signal = reward
            # update list attributes with values
            self.rewards.append(reward)
            self.penalties.append(penalty)
            self.reward_signals.append(reward_signal)
            return [reward, penalty, reward_signal]

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information for the current time step.
        Info = Observation + additional storage information (capacity, rated power, soc_min).
        """
        info = self._get_obs(preprocessing=False)
        for key, value in info.items():
            if key != 'cer':
                value.update({
                    'capacity': self.nodes[key].storage.capacity,
                    'rated_power': self.nodes[key].storage.rated_power,
                    'soc_min': self.nodes[key].storage.soc_min,
                })
        return info

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
            -> Tuple[Dict[str, Any], Dict[str, List[float]] or List[float], bool, bool or None, Dict[str, Any], None]:
        """
        Performs a simulation step, calculating the potential and actual TIP for producers, consumers, and prosumers.
        Updates exported and imported energies, self-consumption, and TIP distributions.

        Args:
            actions [OrderedDict[str, NDArray[(1,), np.float32]]: Action

        Returns:
            Tuple[Dict[str, Any], Dict[str, List[float]] or List[float], bool, bool or None, Dict[str, Any], None]:
                  Observation, reward list, termination flag, truncation flag, additional information, and done
        """
        # apply charge and discharge actions
        if actions:
            for node_id, action in actions.items():
                _ = self.nodes[node_id].set_storage(action[0])

        # compute potential/actual TIP and energy costs for each node
        for producer in self.producers.values():
            energy_produced = producer.get_current_data('production')
            energy_consumed = 0.
            energy_exported, energy_imported, trade, self_consumption = get_energy_trade(
                producer, energy_produced, energy_consumed)
            assert energy_imported >= 0, (f'Energy arbitrage constraint violation -> '
                                          f'{producer.node_id} imported {energy_imported:.2f} kWh')
            producer.exported_energies.append(energy_exported)
            producer.energy_trade.append(trade)
            producer.self_consumption.append(self_consumption)
            set_tip_and_costs(producer, energy_exported, energy_imported, is_producer=True)

        for consumer in self.consumers.values():
            energy_imported = consumer.get_current_data('consumption')
            consumer.imported_energies.append(energy_imported)
            consumer.energy_trade.append(-energy_imported)
            set_tip_and_costs(consumer, 0, energy_imported, is_consumer=True)

        for prosumer in self.prosumers.values():
            energy_produced = prosumer.get_current_data('production')
            energy_consumed = prosumer.get_current_data('consumption')
            energy_exported, energy_imported, trade, self_consumption = get_energy_trade(
                prosumer, energy_produced, energy_consumed)
            assert energy_imported * energy_exported == 0, f'Simultaneous import-export -> ' \
                                                           f'imp: {energy_imported:.2f}, exp: {energy_exported:.2f}'
            prosumer.exported_energies.append(energy_exported)
            prosumer.imported_energies.append(energy_imported)
            prosumer.energy_trade.append(trade)
            prosumer.self_consumption.append(self_consumption)
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

        # reward for the current time step
        reward_list = self._get_reward()
        truncated = None  # TODO: add truncation flag for too many constraint violations

        # update time step
        self._next_hour()

        # next observation, termination flag, next info
        next_obs = self._get_obs(preprocessing=True)
        terminated = self._is_terminal()
        next_info = self._get_info()
        done = None

        return next_obs, reward_list, terminated, truncated, next_info, done

    @cast
    def rl_step(self, actions: OrderedDict[str, Array1[np.float32]]) \
            -> Tuple[Dict[str, Any], Dict[str, List[float]] or List[float], bool, bool or None, Dict[str, Any], None]:
        """
        Perform a simulation step for the Reinforcement Learning environment.
        Exploit @cast decorator to perform a step with Numpy Array actions and Numpy Array observations.
        """
        return self.step(actions)

    def reset(self, seed: int | None = None, options: Dict[str, Any] = None) \
            -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to the initial state, resetting all nodes.

        Args:
            seed (int, optional): Random seed for the environment. Default is None.
            options (dict, optional): Additional information to specify how the environment is reset. Default is None.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Observation and additional information
        """
        super().reset(seed=seed)

        self.time_step = 0
        self.cer_date_time = self.nodes[self.nodes_ids[0]].date_time
        self.years, self.months, self.day_types, self.max_hour = set(), set(), set(), 0
        for dt in [datetime.strptime(dt, '%Y-%m-%d %H:%M:%S') for dt in self.cer_date_time]:
            self.years.add(dt.year)
            self.months.add(dt.month)
            self.day_types.add(dt.weekday())
            if dt.hour > self.max_hour:
                self.max_hour = dt.hour
        self.cer_exported_energy = []
        self.cer_imported_energy = []
        self.cer_energy_expense = []
        self.cer_energy_revenue = []
        self.cer_energy_selfcons = []
        self.cer_energy_selfcons_percentage = []
        self.cer_potential_tip = []
        self.cer_actual_tip = []
        self.rewards = []
        self.penalties = []
        self.reward_signals = []
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
            if hasattr(node, 'energy_trade'):
                table['energy_trade'] = node.energy_trade[0:time_step + 1]
            if hasattr(node, 'self_consumption'):
                table['self_consumption'] = node.self_consumption[0:time_step + 1]
            if hasattr(node, 'energy_expense'):
                table['energy_expense'] = node.energy_expense[0:time_step + 1]
            if hasattr(node, 'energy_revenue'):
                table['energy_revenue'] = node.energy_revenue[0:time_step + 1]
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

    def save_results(self, results: dict, output_dir: str, save_mode: str = 'csv'):
        """
        Saves the simulation results in a structured format.

        Args:
            results (dict): Simulation results, including imported, exported energy, and TIP.
            output_dir (str): The directory where the results will be saved.
            save_mode (str, optional): The mode for saving results, either 'csv' or 'db'. Default is 'csv'.
        """
        # create a directory for simulation results if it does not exist
        saving_path = f'{self.cer_path}/simulations/{output_dir}'
        if not os.path.exists(saving_path):
            os.makedirs(saving_path)

        assert save_mode in ['csv', 'db'], (f'Unrecognized save mode: {save_mode}. '
                                            f'Select between "csv" and "db"')

        # save csv(s) results
        if save_mode == 'csv':
            save_results_to_csv(csv_path=saving_path, results=results)
        elif save_mode == 'db':
            save_results_to_db(db_path=saving_path, results=results)

        # save json kpi(s) summary
        kpis = compute_kpis(results_cer=results['cer'])
        save_dict_to_json(json_path=saving_path, json_name='kpi.json', dict_to_save=kpis)

        # save json config file
        save_dict_to_json(json_path=saving_path, json_name='sim_config.json', dict_to_save=self.cer_config)

    def compute_baselines(self, no_storage: bool = True, rbc: bool = None) -> None:
        """
        Compute baseline results for the CER environment.
        Results are saved in the CER data folder.

        Args:
            no_storage (bool, optional): If True, the baseline with no storage is computed. Default is True.
            rbc (bool, optional): If True, the baseline with the Rule-Based Controller is computed. Default is None.

        Returns:
            None
        """
        seed = self.cer_config['simulation']['seed']
        if no_storage:
            actions = {node_id: np.array(object=[0.], dtype=np.float32) for node_id in self.storage_nodes_ids}
            actions = collections.OrderedDict(sorted(actions.items()))

            results = {}
            _, _ = self.reset(seed=seed)
            for _ in tqdm(range(self.sim_hours), desc="No Storage Simulation Progress"):
                _, _, terminated, truncated, _, _ = self.step(actions)
                done = terminated or truncated
                if done:
                    results = self.get_sim_results()
            self.save_results(results=results, output_dir='baseline_no_storage', save_mode='csv')

        if rbc:
            rbc_controller = StandardRBC(
                storage_node=self.storage_nodes, producer_ids=self.producers_ids, prosumer_ids=self.prosumers_ids)

            results = {}
            _, info = self.reset(seed=seed)
            for _ in tqdm(range(self.sim_hours), desc="RBC Simulation Progress"):
                actions = rbc_controller.choose_actions(info)
                _, _, terminated, truncated, new_info, _ = self.step(actions)
                done = terminated or truncated
                if done:
                    results = self.get_sim_results()
                else:
                    info = new_info
            self.save_results(results=results, output_dir='baseline_rbc', save_mode='csv')

        self.reset()
