from typing import Tuple, Dict

from node_models import Producer, Consumer, Prosumer
from energy_models import Battery

import sqlite3
import time
import json
import csv
import os


# TODO: add tip coefficient to the producer and prosumer classes
def node_loader(cer_path: str) -> Tuple[Dict[str, Producer], Dict[str, Consumer], Dict[str, Prosumer]]:
    """
    Creates producer and consumer nodes from the config.json file.

    :param (str) cer_path: Path to the CER folder

    :return (Tuple[Dict[str, Producer], Dict[str, Consumer], Dict[str, Prosumer]]:
    Dictionaries of producer and consumer nodes.
    """
    with open(f'{cer_path}/config.json') as f:
        config = json.load(f)
        producers_config = config['Producers']
        consumers_config = config['Consumers']
        prosumers_config = config['Prosumers']

    producers, consumers, prosumers = {}, {}, {}

    # PRODUCERS
    start = time.time()
    print('\nLoading nodes...')
    for p_id, p_data in producers_config.items():
        storage = Battery(**p_data['Storage']) if p_data.get('Storage') else None
        producers[p_id] = Producer(
            data_dir=f'{cer_path}/nodes/producers',
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
