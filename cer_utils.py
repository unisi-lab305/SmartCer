from typing import Tuple, Dict

from node_models import Producer, Consumer, Prosumer
from energy_models import Battery

import sqlite3
import time
import json
import csv
import os


def node_loader(cer_path: str) -> Tuple[Dict[str, Producer], Dict[str, Consumer], Dict[str, Prosumer], str]:
    """
    Crea i nodi produttori e consumatori dal file config.json.

    :param (str) cer_path: Percorso della cartella CER

    :return (Tuple[Dict[str, Producer], Dict[str, Consumer], Dict[str, Producer], str):
    Dizionari dei nodi produttori e consumatori e tipologia del controller dei sistemi di accumulo.

    """
    with open(f'{cer_path}/config.json') as f:
        config = json.load(f)
        producers_config = config['Producers']
        consumers_config = config['Consumers']
        prosumers_config = config['Prosumers']
        controller = config['Cer']['simulazione']['storage_controller']

    producers, consumers, prosumers = {}, {}, {}

    start = time.time()
    print('\nCaricamento nodi in corso...')
    for p_id, p_data in producers_config.items():
        storage = Battery(**p_data['Storage']) if p_data.get('Storage') else None
        producers[p_id] = Producer(
            data_dir=f'{cer_path}/nodes/producers',
            node_id=p_id,
            node_type=p_data['Type'],
            storage=storage,
            **p_data['Photovoltaic'],
        )
    print(f'Producers ({len(producers)}) caricati in {time.time() - start:.2f} secondi')

    start = time.time()
    for c_id, c_data in consumers_config.items():
        storage = Battery(**c_data['Storage']) if c_data.get('Storage') else None
        consumers[c_id] = Consumer(
            data_dir=f'{cer_path}/nodes/consumers',
            node_id=c_id,
            node_type=c_data['Type'],
            storage=storage
        )
    print(f'Consumers ({len(consumers)}) caricati in {time.time() - start:.2f} secondi')

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
    print(f'Prosumers ({len(prosumers)}) caricati in {time.time() - start:.2f} secondi')

    return producers, consumers, prosumers, controller


def get_tip_coefficient(tip_th: float, tip_base: float, pi_zone: float, fi_zone: float, pnrr: float) -> float:
    """
    Calcola il TIP dell'unità u-esima [€/kWh].

    :param (float) tip_th: Valore soglia TIP [€/kWh]
    :param (float) tip_base: Valore base TIP [€/kWh]
    :param (float) pi_zone: Prezzo zonale orario [€/kWh]
    :param (float) fi_zone: Fattore di correzione zonale [€/kWh]
    :param (float) pnrr: Percentuale finanziamento impianto [%]

    :return (float): TIP potenziale [€/kWh]
    """

    # fattore di correzione conto capitale
    assert pnrr <= 40, 'Percentuale finanziamento impianto superiore al 40%'
    fi_cc = 0.5 * pnrr / 40

    return (min(tip_th, tip_base + max(0., 0.180 - pi_zone)) + fi_zone) * (1 - fi_cc)


def get_tip_percentages(tip_table_path: str,  cer_trade: float) -> Dict[str, float]:
    """
    Calcola le percentuali di distribuzione del TIP in base alla tabella di configurazione.

    :param (str) tip_table_path: Percorso del file tip_distribution.json
    :param (float) cer_trade: Scambio di energia della CER con la rete [kWh]

    :return (Dict[str, float]): Dizionario dei TIP percentuali

    :raises FileNotFoundError: Se il file tip_table_path non esiste
    """

    try:
        with open(tip_table_path, 'r') as f:
            tip_table = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f'File {tip_table_path} non trovato')

    # controllo somma percentuali
    for rule, values in tip_table.items():
        total_percentage = (values['gestione'] + values['ceritalia'] + values['impresa'] +
                            values['produttore'] + values['retail'] +
                            values['os'])
        assert total_percentage == 100, f'Somma percentuali diversa da 100% per {rule}: {total_percentage}%'

    for rule, values in tip_table.items():
        if values['range'][0] <= cer_trade*100 <= values['range'][1]:
            return {key: values[key] for key in ['gestione', 'ceritalia', 'impresa', 'produttore', 'retail', 'os']}
    raise ValueError(f'Percentuale energia scambiata non valida: cer_trade={cer_trade*100}%')


def save_results_to_db(db_path: str, results: Dict):
    """
    Salva i risultati della simulazione nel database SQLite.

    :param (str) db_path: Percorso del database.
    :param (Dict) results: Risultati della simulazione.
    """
    raise NotImplementedError('Funzione non implementata')


def write_csv(file_path: str, data: Dict) -> None:
    columns = list(data.keys())
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        writer.writerows(zip(*(data[col] for col in columns)))


def save_results_to_csv(csv_path: str, results: Dict) -> None:
    """
    Salva i risultati della simulazione nei file csv.

    :param (str) csv_path: Percorso della cartella dei file csv.
    :param (Dict) results: Risultati della simulazione.
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
