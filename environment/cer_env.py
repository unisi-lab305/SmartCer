from environment.cer_utils import *

import numpy as np
import os


class Cer:
    """
    Classe per la simulazione di una Comunità Energetica Rinnovabile (CER).

    Attributi:
        cer_path (str): Il percorso della directory contenente i dati della CER.
        max_pnrr (float): Il valore massimo del PNRR.
        pnrr_thresholds (tuple): Soglie del PNRR per la classificazione.
        max_tip_retail (float): Il valore massimo del TIP per i consumatori retail.
        producers (dict): Dizionario dei produttori di energia.
        consumers (dict): Dizionario dei consumatori di energia.
        prosumers (dict): Dizionario dei prosumer (produzione e consumo).
        factories_ids (list): Lista degli ID dei nodi di tipo 'factory'.
        producers_ids (list): Lista degli ID dei produttori.
        consumers_ids (list): Lista degli ID dei consumatori.
        prosumers_ids (list): Lista degli ID dei prosumer.
        tip_table_path (str): Percorso del file JSON contenente la tabella TIP.
        time_step (int): Indice dell'attuale intervallo temporale della simulazione.
        cer_exported_energies (list): Lista delle energie esportate dalla CER.
        cer_imported_energies (list): Lista delle energie importate dalla CER.
        cer_energy_selfcons (list): Lista dell'autoconsumo energetico della CER.
        cer_energy_selfcons_percentages (list): Percentuali di autoconsumo della CER.
        cer_potential_tips (list): Lista dei TIP potenziali.
        cer_actual_tips (list): Lista dei TIP attuali.
        cer_tip_distributions (list): Lista delle distribuzioni TIP.
        sim_results (dict): Risultati della simulazione.
    """

    def __init__(self, cer_path: str,
                 max_pnrr: float = 40., pnrr_thresholds: tuple = (45., 55.), max_tip_retail: float = 0.04):
        """
        Inizializza un'istanza della classe Cer.

        Args:
            cer_path (str): Il percorso della directory contenente i dati della CER.
            max_pnrr (float, optional): Il valore massimo del PNRR. Default è 40.
            pnrr_thresholds (tuple, optional): Soglie del PNRR per la classificazione. Default è (45., 55.).
            max_tip_retail (float, optional): Il valore massimo del TIP per i consumatori retail. Default è 0.04.
        """
        self.cer_path = cer_path
        self.producers, self.consumers, self.prosumers = node_loader(cer_path)
        self.factories_ids = [
            node.node_id
            for node in list(self.prosumers.values()) + list(self.consumers.values()) + list(self.producers.values())
            if node.node_type == 'factory'
        ]
        self.producers_ids = list(self.producers.keys())
        self.consumers_ids = list(self.consumers.keys())
        self.prosumers_ids = list(self.prosumers.keys())
        self.tip_table_path = os.path.join(cer_path, 'tip_table.json')
        self.max_pnrr = max_pnrr
        self.pnrr_thresholds = pnrr_thresholds
        self.max_tip_retail = max_tip_retail

        self.time_step = 0

        self.cer_exported_energies = []
        self.cer_imported_energies = []
        self.cer_energy_selfcons = []
        self.cer_energy_selfcons_percentages = []
        self.cer_potential_tips = []
        self.cer_actual_tips = []
        self.cer_tip_distributions = []
        self.sim_results = {}

    # TODO: optimize step method
    def step(self) -> None:
        """
        Esegue un passo della simulazione, calcolando il TIP potenziale e attuale per produttori, consumatori e
        prosumer.
        Aggiorna le energie esportate e importate, l'autoconsumo e le distribuzioni TIP.
        """
        # calcolo del TIP potenziale
        for producer in self.producers.values():
            producer.exported_energies.append(producer.get_current_data('production'))
            # TIP coefficient
            coefficient_tip = get_tip_coefficient(
                tip_th=producer.tip_th,
                tip_base=producer.tip_base,
                pi_zone=producer.get_current_data('price_exp'),
                fi_zone=producer.fi_zone,
                pnrr=producer.pnrr
            )
            producer.tip_coefficients.append(coefficient_tip)
            # potential TIP [€/kWh]
            potential_tip = coefficient_tip * producer.exported_energies[self.time_step]
            producer.potential_tips_prod.append(potential_tip)

        for consumer in self.consumers.values():
            consumer.imported_energies.append(consumer.get_current_data('consumption'))

        for prosumer in self.prosumers.values():
            # imported-exported energy
            energy_produced = prosumer.get_current_data('production')
            energy_consumed = prosumer.get_current_data('consumption')
            prosumer.exported_energies.append(max(0., energy_produced - energy_consumed))
            prosumer.imported_energies.append(max(0., energy_consumed - energy_produced))
            # TIP coefficient
            coefficient_tip = get_tip_coefficient(
                tip_th=prosumer.tip_th,
                tip_base=prosumer.tip_base,
                pi_zone=prosumer.get_current_data('price_exp'),
                fi_zone=prosumer.fi_zone,
                pnrr=prosumer.pnrr
            )
            prosumer.tip_coefficients.append(coefficient_tip)
            # potential TIP [€/kWh]
            potential_tip = coefficient_tip * prosumer.exported_energies[self.time_step]
            prosumer.potential_tips_prod.append(potential_tip)

        # calcolo energia esportata e importata a livello di CER
        self.cer_exported_energies.append(
            sum([producer.exported_energies[self.time_step] for producer in self.producers.values()]) +
            sum([prosumer.exported_energies[self.time_step] for prosumer in self.prosumers.values()]))
        self.cer_imported_energies.append(
            sum([consumer.imported_energies[self.time_step] for consumer in self.consumers.values()]) +
            sum([prosumer.imported_energies[self.time_step] for prosumer in self.prosumers.values()]))

        # calcolo del TIP potenziale a livello di CER
        self.cer_potential_tips.append(
            sum([producer.potential_tips_prod[self.time_step] for producer in self.producers.values()]) +
            sum([prosumer.potential_tips_prod[self.time_step] for prosumer in self.prosumers.values()]))

        # calcolo dell'autoconsumo energetico a livello di CER
        self.cer_energy_selfcons.append(
            min(self.cer_exported_energies[self.time_step], self.cer_imported_energies[self.time_step]))
        if self.cer_exported_energies[self.time_step] == 0:
            self.cer_energy_selfcons_percentages.append(0.)
        else:
            self.cer_energy_selfcons_percentages.append(
                self.cer_energy_selfcons[self.time_step] / self.cer_exported_energies[self.time_step])

        # calcolo del TIP attuale a livello di CER
        self.cer_actual_tips.append(
            self.cer_energy_selfcons_percentages[self.time_step] * self.cer_potential_tips[self.time_step])

        # ridistribuzione TIP seconda tabella configurazione
        try:
            tip_percentages = get_tip_percentages(self.tip_table_path,
                                                  self.cer_energy_selfcons_percentages[self.time_step])
        except ValueError as e:
            tip_percentages = None
            print(e)

        # calcolo scostamento
        if self.cer_energy_selfcons_percentages[self.time_step] >= self.pnrr_thresholds[0] / 100.:
            new_tip_percentages = self.compute_deviation(tip_percentages=tip_percentages)
        else:
            new_tip_percentages = tip_percentages

        # distribuzione TIP reale [€]
        tip_distribution = {key: self.cer_actual_tips[self.time_step] * value / 100.
                            for key, value in new_tip_percentages.items()}
        assert round(sum(tip_distribution.values()), 2) == round(self.cer_actual_tips[self.time_step], 2), \
            (f'Somma TIP ({sum(tip_distribution.values())}) '
             f'diversa da TIP attuale ({self.cer_actual_tips[self.time_step]})')

        # calcolo TIP per ogni nodo [€]
        self.compute_distribution(tip_distribution=tip_distribution)

    def reset(self) -> None:
        """
        Ripristina lo stato della simulazione per ricominciare dal primo intervallo temporale.
        """
        self.time_step = 0
        self.cer_exported_energies = []
        self.cer_imported_energies = []
        self.cer_energy_selfcons = []
        self.cer_energy_selfcons_percentages = []
        self.cer_potential_tips = []
        self.cer_actual_tips = []
        self.cer_tip_distributions = []
        self.sim_results = {}
        for producer in self.producers.values():
            producer.reset()
        for consumer in self.consumers.values():
            consumer.reset()
        for prosumer in self.prosumers.values():
            prosumer.reset()

    def compute_deviation(self, tip_percentages: dict) -> dict:
        """
        Calcola la deviazione del TIP sulla base delle percentuali attuali.

        Args:
            tip_percentages (dict): Percentuali dei TIP correnti.

        Returns:
            dict: Nuove percentuali dei TIP corrette.
        """
        min_pnrr_tips, max_pnrr_tips = [], []
        for entity in {**self.producers, **self.prosumers}.values():
            min_pnrr_tip_coefficient = get_tip_coefficient(
                tip_th=entity.tip_th,
                tip_base=entity.tip_base,
                pi_zone=entity.get_current_data('price_exp'),
                fi_zone=entity.fi_zone,
                pnrr=0.
            )
            max_pnrr_tip_coefficient = get_tip_coefficient(
                tip_th=entity.tip_th,
                tip_base=entity.tip_base,
                pi_zone=entity.get_current_data('price_exp'),
                fi_zone=entity.fi_zone,
                pnrr=self.max_pnrr
            )
            min_pnrr_tip = min_pnrr_tip_coefficient * entity.exported_energies[self.time_step]
            max_pnrr_tip = max_pnrr_tip_coefficient * entity.exported_energies[self.time_step]
            min_pnrr_tips.append(min_pnrr_tip)
            max_pnrr_tips.append(max_pnrr_tip)

        cer_min_pnrr_tip = sum(min_pnrr_tips)
        cer_max_pnrr_tip = sum(max_pnrr_tips)

        cer_tip_deviation = cer_min_pnrr_tip - self.cer_actual_tips[self.time_step]
        cer_tip_deviation_percentage = cer_tip_deviation / cer_max_pnrr_tip * 100

        os_percentage_increase = cer_tip_deviation_percentage / (self.pnrr_thresholds[1] - self.pnrr_thresholds[0])

        new_tip_percentages = tip_percentages.copy()
        new_tip_percentages['os'] += os_percentage_increase
        new_tip_percentages['impresa'] -= (os_percentage_increase / 2.)
        new_tip_percentages['produttore'] -= (os_percentage_increase / 2.)

        return new_tip_percentages

    def compute_distribution(self, tip_distribution: dict) -> None:
        """
        Calcola e distribuisce i TIP tra produttori e consumatori.

        Args:
            tip_distribution (dict): Distribuzione dei TIP per categoria (produttori, consumatori, ecc.).
        """
        # TIP produttori
        for node in {**self.producers, **self.prosumers}.values():
            if self.cer_actual_tips[self.time_step] == 0:
                node.tip_contributions_prod.append(0.)
            else:
                node.tip_contributions_prod.append(
                    node.potential_tips_prod[self.time_step] * self.cer_energy_selfcons_percentages[self.time_step] /
                    self.cer_actual_tips[self.time_step] * 100.
                )
            node.actual_tips_prod.append(
                node.tip_contributions_prod[self.time_step] * tip_distribution['produttore'] / 100.
            )

        # TIP consumatori
        retails_consumption = 0.
        factories_consumption = 0.
        for node in {**self.consumers, **self.prosumers}.values():
            if node.node_type == 'retail':
                retails_consumption += node.imported_energies[self.time_step]
            elif node.node_type == 'factory':
                factories_consumption += node.imported_energies[self.time_step]

        retails_surplus, retails_actual_tip = 0., 0.
        for node in {**self.consumers, **self.prosumers}.values():
            if node.node_type == 'retail':
                if retails_consumption == 0.:
                    node.tip_contributions_cons.append(0.)
                    node.potential_tips_cons.append(0.)
                    node.actual_tips_cons.append(0.)
                else:
                    node.tip_contributions_cons.append(
                        node.imported_energies[self.time_step] / retails_consumption * 100.
                    )
                    node.potential_tips_cons.append(
                        node.tip_contributions_cons[self.time_step] * tip_distribution['retail'] / 100.
                    )

                    node.actual_tips_cons.append(
                        min(node.potential_tips_cons[self.time_step],
                            node.imported_energies[self.time_step] * self.max_tip_retail)
                    )

                    retails_surplus += (node.potential_tips_cons[self.time_step] -
                                        node.actual_tips_cons[self.time_step])
                    retails_actual_tip += node.actual_tips_cons[self.time_step]

            elif node.node_type == 'factory':
                if factories_consumption == 0.:
                    node.tip_contributions_cons.append(0.)
                    node.actual_tips_cons.append(0.)
                else:
                    node.tip_contributions_cons.append(
                        node.imported_energies[self.time_step] / factories_consumption * 100.
                    )
                    node.actual_tips_cons.append(
                        node.tip_contributions_cons[self.time_step] * tip_distribution['impresa'] / 100.
                    )
            else:
                # nodi non soggetti a TIP
                node.tip_contributions_cons.append(0.)
                node.actual_tips_cons.append(0.)

        # TIP retail potenziale
        tip_distribution['retail_reale'] = retails_actual_tip

        # TIP opere sociali
        tip_distribution['os'] += retails_surplus

        # check somma TIP
        tip_sum = sum(value for key, value in tip_distribution.items() if key != 'retail')
        assert round(tip_sum, 2) == round(self.cer_actual_tips[self.time_step], 2), \
            (f"Errore nel calcolo della distribuzione del TIP -> "
             f"Somma della distribuzione del TIP ({tip_sum}) diversa dal TIP ({self.cer_actual_tips[self.time_step]})")

        # Append tip_distribution to cer_tip_distributions
        self.cer_tip_distributions.append(tip_distribution)

    def get_sim_results(self) -> dict:
        """
        Ottiene i risultati della simulazione in un formato strutturato.

        Returns:
            dict: Risultati della simulazione, inclusi energia importata, esportata, e TIP.
        """
        cer_table = {
            'datetime': self.consumers[self.consumers_ids[0]].date_time[0:self.time_step+1].tolist(),
            'energia_importata': self.cer_imported_energies[0:self.time_step+1],
            'energia_esportata': self.cer_exported_energies[0:self.time_step+1],
            'autoconsumo_percentuale': self.cer_energy_selfcons_percentages[0:self.time_step + 1],
            'tip_potenziale': self.cer_potential_tips[0:self.time_step+1],
            'tip_reale': self.cer_actual_tips[0:self.time_step+1],
            'tip_gestione': [tip['gestione'] for tip in self.cer_tip_distributions][0:self.time_step+1],
            'tip_produttore': [tip['produttore'] for tip in self.cer_tip_distributions][0:self.time_step + 1],
            'tip_retail_potenziale': [tip['retail'] for tip in self.cer_tip_distributions][0:self.time_step + 1],
            'tip_retail_reale': [tip['retail_reale'] for tip in self.cer_tip_distributions][0:self.time_step + 1],
            'tip_impresa': [tip['impresa'] for tip in self.cer_tip_distributions][0:self.time_step+1],
            'tip_os': [tip['os'] for tip in self.cer_tip_distributions][0:self.time_step+1],
        }

        producer_tables = {}
        for node in self.producers.values():
            producer_tables[node.node_id] = {
                'datetime': node.date_time[0:self.time_step + 1].tolist(),
                'prezzo_vendita': node.price_exp[0:self.time_step + 1].tolist(),
                'energia_prodotta': node.production[0:self.time_step + 1].tolist(),
                'energia_esportata': node.exported_energies[0:self.time_step + 1],
                'contributo_tip': node.tip_contributions_prod[0:self.time_step + 1],
                'tip_potenziale': node.potential_tips_prod[0:self.time_step + 1],
                'tip_reale': node.actual_tips_prod[0:self.time_step + 1],
            }

        consumer_tables = {}
        for node in self.consumers.values():
            consumer_tables[node.node_id] = {
                'datetime': node.date_time[0:self.time_step + 1].tolist(),
                'prezzo_acquisto': node.price_imp[0:self.time_step + 1].tolist(),
                'energia_consumata': node.consumption[0:self.time_step + 1].tolist(),
                'energia_importata': node.imported_energies[0:self.time_step + 1],
            }
            if node.node_type == 'retail':
                consumer_tables[node.node_id].update({
                    'tip_potenziale': node.potential_tips_cons[0:self.time_step + 1],
                    'tip_reale': node.actual_tips_cons[0:self.time_step + 1],
                })
            elif node.node_type == 'factory':
                consumer_tables[node.node_id].update({
                    'tip_potenziale': node.actual_tips_cons[0:self.time_step + 1],
                    'tip_reale': node.actual_tips_cons[0:self.time_step + 1],
                })

        prosumer_tables = {}
        for node in self.prosumers.values():
            prosumer_tables[node.node_id] = {
                'datetime': node.date_time[0:self.time_step + 1].tolist(),
                'prezzo_vendita': node.price_exp[0:self.time_step + 1].tolist(),
                'prezzo_acquisto': node.price_imp[0:self.time_step + 1].tolist(),
                'energia_prodotta': node.production[0:self.time_step + 1].tolist(),
                'energia_consumata': node.consumption[0:self.time_step + 1].tolist(),
                'energia_importata': node.imported_energies[0:self.time_step + 1],
                'energia_esportata': node.exported_energies[0:self.time_step + 1],
                'contributo_tip_prod': node.tip_contributions_prod[0:self.time_step + 1],
                'tip_potenziale_prod': node.potential_tips_prod[0:self.time_step + 1],
                'tip_reale_prod': node.actual_tips_prod[0:self.time_step + 1],
            }
            if node.node_type == 'retail':
                prosumer_tables[node.node_id].update({
                    'tip_potenziale_cons': node.potential_tips_cons[0:self.time_step + 1],
                    'tip_reale_cons': node.actual_tips_cons[0:self.time_step + 1],
                })
            elif node.node_type == 'factory':
                prosumer_tables[node.node_id].update({
                    'tip_potenziale_cons': node.actual_tips_cons[0:self.time_step + 1],
                    'tip_reale_cons': node.actual_tips_cons[0:self.time_step + 1],
                })

        results = {
            'cer': cer_table,
            'producers': producer_tables,
            'consumers': consumer_tables,
            'prosumers': prosumer_tables,
        }

        return results

    def save_results(self, results: dict, save_mode: str = 'csv'):
        """
        Ottiene i risultati della simulazione in un formato strutturato.

        Returns:
            dict: Risultati della simulazione, inclusi energia importata, esportata, e TIP.
        """
        if not os.path.exists(f'{self.cer_path}/simulation'):
            os.makedirs(f'{self.cer_path}/simulation')
        assert save_mode in ['csv', 'db'], (f'Modalità di salvataggio non riconosciuta: {save_mode}. '
                                            f'Selezione tra "csv" e "db"')
        if save_mode == 'csv':
            save_results_to_csv(csv_path=f'{self.cer_path}/simulation', results=results)
        elif save_mode == 'db':
            save_results_to_db(db_path=f'{self.cer_path}/simulation', results=results)

    def next_hour(self):
        """
        Avanza al passo temporale successivo nella simulazione, aggiornando il time_step per tutti i nodi.
        """
        self.time_step += 1
        for producer in self.producers.values():
            producer.time_step = self.time_step
        for consumer in self.consumers.values():
            consumer.time_step = self.time_step
        for prosumer in self.prosumers.values():
            prosumer.time_step = self.time_step
