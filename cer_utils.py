from node_models import Producer, Consumer, Prosumer
from energy_models import Battery

from typing import Tuple, Dict, Union
from datetime import datetime

import sqlite3
import time
import json
import csv
import os


def node_loader(cer_path: str) -> Tuple[Dict[str, Producer], Dict[str, Consumer], Dict[str, Prosumer]]:
    """
    Creates producer and consumer nodes from the config.json file.

    :param (str) cer_path: Path to the CER folder

    :return (Tuple[Dict[str, Producer], Dict[str, Consumer], Dict[str, Prosumer]]:
    Dictionaries of producer and consumer nodes.
    """
    with open(f'{cer_path}/config.json') as f:
        config = json.load(f)
        producers_config = config.get('Producers', {})
        consumers_config = config.get('Consumers', {})
        prosumers_config = config.get('Prosumers', {})
        sim_start = config['Cer']['simulazione']['inizio']
        sim_end = config['Cer']['simulazione']['fine']

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
    print(f'Prosumers ({len(prosumers)}) loaded in {time.time() - start:.2f} seconds')

    return producers, consumers, prosumers


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
                     energy_produced: float, energy_consumed: float) -> Tuple[float, float]:
    """
    Calculates the energy exported and imported by the node.
    :param (Node) node: Node object
    :param (float) energy_produced: energy produced by the node
    :param (float) energy_consumed: energy consumed by the node

    :return (Tuple[float, float]): energy exported and imported by the node
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
    return energy_exported, energy_imported


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


def write_csv(file_path: str, data: Dict) -> None:
    """
    Write data to a csv file.
    :param file_path: path to the csv file
    :param data: dictionary containing the data to write
    :return: None
    """
    columns = list(data.keys())
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        writer.writerows(zip(*(data[col] for col in columns)))


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
