from smartcer.cer_models.node_models import Producer, Consumer, Prosumer
from smartcer.cer_models.energy_models import Battery
from smartcer.cer_reward.scaler import RewardScaler

from typing import Annotated, Any, Dict, List, Literal, OrderedDict, Tuple, TypeVar, Union
from functools import wraps
from typing import Callable
from datetime import datetime

import collections
import time
import json
import csv
import os

import gymnasium as gym

import numpy.typing as npt
import numpy as np

DType = TypeVar('DType', bound=np.generic)
Array1 = Annotated[npt.NDArray[DType], Literal[1]]
ArrayN = Annotated[npt.NDArray[DType], Literal['N']]


class OHESpace(gym.Space):
    """
    Custom space for one-hot encoding vectors.

    Example usage:
    self.observation_space = OHESpace(size=4)
    """

    def __init__(self, size=None):
        assert isinstance(size, int) and size > 0, "size must be a positive integer"
        self.size = size
        super().__init__(shape=(self.size,), dtype=np.float32)

    def sample(self, mask=None):
        """Generate a random one-hot vector."""
        one_hot_vector = np.zeros(self.size, dtype=np.float32)
        one_hot_vector[np.random.randint(self.size)] = 1
        return one_hot_vector

    def contains(self, x):
        """Check if x is a valid one-hot encoded vector."""
        if isinstance(x, (list, tuple, np.ndarray)):
            number_of_zeros = list(x).count(0)
            number_of_ones = list(x).count(1)
            return (number_of_zeros == (self.size - 1)) and (number_of_ones == 1)
        else:
            return False

    @property
    def is_np_flattenable(self):
        """Ensure this space can be flattened."""
        return True

    def __repr__(self):
        return f'OHESpace({self.size})'

    def __eq__(self, other):
        return self.size == other.size


def node_loader(cer_path: str, cer_config: Dict) \
        -> Tuple[Dict[str, Producer], Dict[str, Consumer], Dict[str, Prosumer], int]:
    """
    Creates producer and consumer nodes from the config.json file.

    :param (str) cer_path: Path to the CER folder
    :param (Dict) cer_config: Configuration dictionary.

    :return (Tuple[Dict[str, Producer], Dict[str, Consumer], Dict[str, Prosumer], int]):
    Dictionaries of producer and consumer nodes, and the simulation hours.
    """
    with open(f'{cer_path}/config.json') as f:
        config = json.load(f)
        producers_config = config.get('Producers', {})
        consumers_config = config.get('Consumers', {})
        prosumers_config = config.get('Prosumers', {})

    sim_start = cer_config['simulation']['start']
    sim_end = cer_config['simulation']['end']

    sim_step = datetime.strptime(sim_end, '%Y-%m-%d %H:%M:%S') - datetime.strptime(sim_start, '%Y-%m-%d %H:%M:%S')
    sim_hours = sim_step.days * 24 + sim_step.seconds // 3600
    sim_hours += 1  # include the last hour

    producers, consumers, prosumers = {}, {}, {}

    # PRODUCERS
    start = time.time()
    print('\nLoading nodes...')
    for p_id, p_data in producers_config.items():
        storage = Battery(**p_data['Storage']) if p_data.get('Storage') else None
        producers[p_id] = Producer(
            data_dir=f'{cer_path}/nodes/producers',
            sim_period=(sim_start, sim_end),
            node_id=p_id,
            node_type=p_data['Type'],
            storage=storage,
            **p_data['Photovoltaic'],
        )
        observation_space = {
            'price_exp': [producers[p_id].price_exp.min(), producers[p_id].price_exp.max()],
            'production': [producers[p_id].production.min(), producers[p_id].production.max()]
        }
        producers[p_id].set_observation_space(observation_space)

    print(f'Producers ({len(producers)}) loaded in {time.time() - start:.2f} seconds')

    # CONSUMERS
    start = time.time()
    for c_id, c_data in consumers_config.items():
        storage = Battery(**c_data['Storage']) if c_data.get('Storage') else None
        consumers[c_id] = Consumer(
            data_dir=f'{cer_path}/nodes/consumers',
            sim_period=(sim_start, sim_end),
            node_id=c_id,
            node_type=c_data['Type'],
            storage=storage
        )
        observation_space = {
            'price_imp': [consumers[c_id].price_imp.min(), consumers[c_id].price_imp.max()],
            'consumption': [consumers[c_id].consumption.min(), consumers[c_id].consumption.max()]
        }
        consumers[c_id].set_observation_space(observation_space)
    print(f'Consumers ({len(consumers)}) loaded in {time.time() - start:.2f} seconds')

    # PROSUMERS
    start = time.time()
    for ps_id, ps_data in prosumers_config.items():
        storage = Battery(**ps_data['Storage']) if ps_data.get('Storage') else None
        prosumers[ps_id] = Prosumer(
            data_dir=f'{cer_path}/nodes/prosumers',
            sim_period=(sim_start, sim_end),
            node_id=ps_id,
            node_type=ps_data['Type'],
            storage=storage,
            **ps_data['Photovoltaic'],
        )
        observation_space = {
            'price_exp': [prosumers[ps_id].price_exp.min(), prosumers[ps_id].price_exp.max()],
            'price_imp': [prosumers[ps_id].price_imp.min(), prosumers[ps_id].price_imp.max()],
            'production': [prosumers[ps_id].production.min(), prosumers[ps_id].production.max()],
            'consumption': [prosumers[ps_id].consumption.min(), prosumers[ps_id].consumption.max()]
        }
        prosumers[ps_id].set_observation_space(observation_space)
    print(f'Prosumers ({len(prosumers)}) loaded in {time.time() - start:.2f} seconds')

    return producers, consumers, prosumers, sim_hours


def scaler_loader(single_agent_reward, storage_node_ids: List[str], **scaler_config) \
        -> Union[RewardScaler, Dict[str, RewardScaler]]:
    """
    Load the reward scaler based on the configuration.

    :param (bool) single_agent_reward: Flag indicating if the reward is for a single agent.
    :param (List[str]) storage_node_ids: List of storage node IDs.
    :param (Dict) scaler_config: Reward scaler configuration.

    :return: RewardScaler object or dictionary of RewardScaler objects.
    """
    if single_agent_reward:
        reward_scaler_dict = {}
        for node_id in storage_node_ids:
            reward_scaler_dict[node_id] = RewardScaler(
                clip_min=scaler_config.get('clip_min', None),
                clip_max=scaler_config.get('clip_max', None),
                reset_interval=scaler_config.get('reset_interval', None))
        return reward_scaler_dict
    else:
        return RewardScaler(
            clip_min=scaler_config.get('clip_min', None),
            clip_max=scaler_config.get('clip_max', None),
            reset_interval=scaler_config.get('reset_interval', None))


def encode_obs(obs: Dict[str, Any], encoder: [Dict[str, Any]]) -> Dict[str, Any]:
    """
    Encode the observation based on the encoder.

    :param (Dict[str, Any]) obs: Observation dictionary.
    :param (Dict[str, Any]) encoder: Encoder dictionary.

    :return (Dict[str, Any]): Encoded observation dictionary.
    """
    encoded_obs = {}
    for key, value in obs.items():
        # handle nested dictionaries
        if isinstance(value, dict):
            encoded_obs[key] = encode_obs(value, encoder[key])
        else:
            encoded_obs[key] = value * encoder[key]
    return encoded_obs


def decode_obs(obs: Dict[str, Any], encoder: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decode the observation based on the encoder (revert the encoding process).

    :param (Dict[str, Any]) obs: Encoded observation dictionary.
    :param (Dict[str, Any]) encoder: Encoder dictionary.

    :return (Dict[str, Any]): Decoded observation dictionary.
    """
    decoded_obs = {}
    for key, value in obs.items():
        # handle nested dictionaries
        if isinstance(value, dict):
            decoded_obs[key] = decode_obs(value, encoder[key])
        else:
            decoded_obs[key] = encoder[key] / value
    return decoded_obs


def flatten_observation(obs: Dict[str, Any]) -> ArrayN[np.float32]:
    """
    Flatten the observation dictionary and return a 1D array.

    :param (Dict[str, Any]) obs: Observation dictionary.

    :return (ArrayN[np.float32]): Flattened observation array.
    """

    values = []
    for sub_dict in obs.values():
        for value in sub_dict.values():
            if value is not None:
                values.extend(np.nditer(value, flags=['refs_ok']))
    return np.array(values, dtype=np.float32)


def create_action_dict(actions_array: ArrayN or List, storage_node_ids: List[str]) \
        -> OrderedDict[str, Array1[np.float32]]:
    """
    Create an ordered dictionary of actions for each storage node.

    :param (ArrayN) actions_array: Array/List of actions.
    :param (List[str]) storage_node_ids: List of storage node IDs.

    :return (OrderedDict[str, Array1[np.float32]]): Ordered dictionary of actions.
    """
    assert len(actions_array) == len(storage_node_ids), \
        f'Number of actions ({len(actions_array)}) does not match the number of storage nodes ({len(storage_node_ids)})'
    actions = {}
    for action, node_id in zip(actions_array, sorted(storage_node_ids)):
        actions[node_id] = np.array(object=[action], dtype=np.float32)

    ordered_actions = collections.OrderedDict(sorted(actions.items()))
    return ordered_actions


def cast(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(self, actions: OrderedDict[str, Array1[np.float32]]) -> (
            Tuple)[ArrayN[np.float32], Dict[str, float] or float, bool, bool or None, Dict[str, Any], None]:
        actions_dict = create_action_dict(actions, self.storage_nodes_ids)
        obs, reward, terminated, truncated, info, done = func(self, actions_dict)
        flat_obs = flatten_observation(obs)

        return flat_obs, reward, terminated, truncated, info, done
    return wrapper


def get_tip_coefficient(tip_th: float, tip_base: float, pi_zone: float, fi_zone: float, pnrr: float) -> float:
    """
    Calculates the TIP of the u-th unit [€/kWh].

    :param (float) tip_th: TIP threshold value [€/kWh]
    :param (float) tip_base: Base TIP value [€/kWh]
    :param (float) pi_zone: Hourly zonal price [€/kWh]
    :param (float) fi_zone: Zonal correction factor [€/kWh]
    :param (float) pnrr: Percentage of plant financing [%]

    :return (float): Potential TIP [€/kWh]
    """

    # Capital account correction factor
    assert pnrr <= 40, 'Plant financing percentage exceeds 40%'
    fi_cc = 0.5 * pnrr / 40

    return (min(tip_th, tip_base + max(0., 0.180 - pi_zone)) + fi_zone) * (1 - fi_cc)


def get_energy_trade(node: Union[Prosumer, Consumer, Producer],
                     energy_produced: float, energy_consumed: float) -> Tuple[float, float, float, float]:
    """
    Calculates the energy exported and imported by the node.
    :param (Node) node: Node object
    :param (float) energy_produced: energy produced by the node
    :param (float) energy_consumed: energy consumed by the node

    :return (Tuple[float, float, float, float]): energy exported and imported by the node, energy traded and
        selfconsumption
    """
    if node.storage:
        energy_trade = (energy_produced + node.storage.actual_discharges[node.time_step] -
                        energy_consumed - node.storage.actual_charges[node.time_step])
    else:
        energy_trade = energy_produced - energy_consumed

    if energy_trade > 0:
        energy_exported = energy_trade
        energy_imported = 0.
    else:
        energy_exported = 0.
        energy_imported = -energy_trade

    if node.storage:
        selfconsumption = min(energy_produced, (energy_consumed + node.storage.actual_charges[node.time_step]))
    else:
        selfconsumption = min(energy_produced, energy_consumed)

    return energy_exported, energy_imported, energy_trade, selfconsumption


def set_tip_and_costs(node: Union[Prosumer, Consumer, Producer], energy_exported: float, energy_imported: float,
                      is_producer: bool = False, is_consumer: bool = False) -> None:
    """
    Sets the TIP coefficient and energy costs for the node.
    :param (Union[Prosumer, Consumer, Producer]) node: Node object
    :param (float) energy_exported: energy exported by the node
    :param (float) energy_imported: energy imported by the node
    :param (bool) is_producer: flag indicating if the node is a producer
    :param (bool) is_consumer: flag indicating if the node is a consumer

    :return: None
    """
    # TIP coefficient
    if not is_consumer:
        coefficient_tip = get_tip_coefficient(
            tip_th=node.tip_th,
            tip_base=node.tip_base,
            pi_zone=node.get_current_data('price_exp'),
            fi_zone=node.fi_zone,
            pnrr=node.pnrr
        )
        node.tip_coefficients.append(coefficient_tip)
        # potential TIP [€/kWh]
        potential_tip = coefficient_tip * energy_exported
        node.potential_tips_prod.append(potential_tip)

    # energy costs
    energy_revenue = 0. if is_consumer else node.get_current_data('price_exp') * energy_exported
    energy_expense = 0. if is_producer else node.get_current_data('price_imp') * energy_imported
    node.energy_revenue.append(energy_revenue)
    node.energy_expense.append(energy_expense)


def save_results_to_db(db_path: str, results: Dict):
    """
    Saves the simulation results to the SQLite database.

    :param (str) db_path: Path to the database.
    :param (Dict) results: Simulation results.
    """
    raise NotImplementedError('Function not implemented')


def write_csv(file_path: str, data: Dict, round_digit: int = 3) -> None:
    """
    Write data to a csv file.
    :param file_path: path to the csv file
    :param data: dictionary containing the data to write
    :param round_digit: number of decimal digits to round the data
    :return: None
    """
    columns = list(data.keys())

    # Arrotonda i dati a 3 cifre decimali solo per i numeri
    rounded_data = {
        col: [round(val, round_digit) if isinstance(val, (float, int)) else val for val in data[col]]
        for col in columns
    }

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        writer.writerows(zip(*(rounded_data[col] for col in columns)))


def save_results_to_csv(csv_path: str, results: Dict) -> None:
    """
    Saves the simulation results to CSV files.

    :param (str) csv_path: Path to the CSV files directory.
    :param (Dict) results: Simulation results.
    """
    # CER
    write_csv(f'{csv_path}/cer_results.csv', results['cer'])

    # Producers
    os.makedirs(f'{csv_path}/producers', exist_ok=True)
    for producer_id, producer_data in results['producers'].items():
        write_csv(f'{csv_path}/producers/{producer_id}_results.csv', producer_data)

    # Consumers
    os.makedirs(f'{csv_path}/consumers', exist_ok=True)
    for consumer_id, consumer_data in results['consumers'].items():
        write_csv(f'{csv_path}/consumers/{consumer_id}_results.csv', consumer_data)

    # Prosumers
    os.makedirs(f'{csv_path}/prosumers', exist_ok=True)
    for prosumer_id, prosumer_data in results['prosumers'].items():
        write_csv(f'{csv_path}/prosumers/{prosumer_id}_results.csv', prosumer_data)


def save_dict_to_json(json_path: str, json_name: str, dict_to_save: Dict) -> None:
    """
    Saves dictionary to a JSON file.

    :param (str) json_path: Path to the JSON file.
    :param (str) json_name: Name of the JSON file.
    :param (Dict) dict_to_save: Dictionary to save.
    """
    with open(f'{json_path}/{json_name}', 'w') as f:
        json.dump(dict_to_save, f, indent=4)
