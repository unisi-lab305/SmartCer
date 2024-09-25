import pandas as pd
import numpy as np
import re
import os


class Node:
    """
    Classe base per rappresentare un nodo energetico.

    Gestisce il caricamento dei dati da file CSV e fornisce metodi per accedere alle informazioni
    relative al nodo, come l'energia importata, esportata e i coefficienti TIP.

    Attributes:
        data_path (str): Percorso al file CSV contenente i dati del nodo.
        node_id (str): Identificativo univoco del nodo.
        node_type (str): Tipo del nodo (es. 'Consumer', 'Producer', 'Prosumer').
        time_step (int): Indice temporale corrente.
        data (pd.DataFrame): Dati del nodo in formato DataFrame.
        date_time (np.array): Array dei timestamp estratti dai dati.

        imported_energies (list): Lista delle energie importate.
        exported_energies (list): Lista delle energie esportate.
        tip_coefficients (list): Coefficienti TIP.
        tip_contributions_prod (list): Contributi TIP per la produzione.
        tip_contributions_cons (list): Contributi TIP per il consumo.
        potential_tips_prod (list): Potenziali TIP per la produzione.
        potential_tips_cons (list): Potenziali TIP per il consumo.
        actual_tips_prod (list): TIP effettivi per la produzione.
        actual_tips_cons (list): TIP effettivi per il consumo.
    """

    def __init__(self, data_dir: str, node_id: str, node_type: str):
        """
        Inizializza la classe Node con il percorso dei dati, l'ID del nodo e il tipo di nodo.

        Args:
            data_dir (str): Directory contenente i dati del nodo.
            node_id (str): Identificativo del nodo.
            node_type (str): Tipo del nodo.
        """
        self.data_path = os.path.join(data_dir, f'{self.extract_base_name(node_id)}.csv')
        self.node_id = node_id
        self.node_type = node_type

        self.time_step = 0
        self.data = pd.DataFrame()
        self.date_time = np.array([])

        self.imported_energies = []
        self.exported_energies = []
        self.tip_coefficients = []
        self.tip_contributions_prod = []
        self.tip_contributions_cons = []
        self.potential_tips_prod = []
        self.potential_tips_cons = []
        self.actual_tips_prod = []
        self.actual_tips_cons = []

        self.load_data()

    @staticmethod
    def extract_base_name(name: str) -> str:
        """
        Estrae il nome di base da un identificativo di nodo.

        Args:
            name (str): Identificativo del nodo.

        Returns:
            str: Nome di base del nodo.
        """
        match = re.match(pattern='^[a-zA-Z0-9_]+(?=_\d+)', string=name)
        return match.group(0) if match else name

    def load_data(self) -> None:
        """
        Carica i dati dal file CSV nel DataFrame.

        Returns:
            None
        """
        self.data = pd.read_csv(self.data_path)
        self.date_time = self.data['datetime'].values

    def get_data(self, column: str = None) -> pd.DataFrame:
        """
        Restituisce i dati del nodo.

        Args:
            column (str, optional): Nome della colonna da restituire. Se None, restituisce l'intero DataFrame.

        Returns:
            pd.DataFrame: Dati del nodo o colonna specificata.
        """
        return self.data if column is None else self.data[column]

    def get_data_columns(self) -> list:
        """
        Restituisce l'elenco delle colonne disponibili nel DataFrame.

        Returns:
            list: Nomi delle colonne.
        """
        return self.data.columns.tolist()

    def get_current_data(self, column: str = None) -> np.array or float:
        """
        Restituisce i dati correnti per il time step attuale.

        Args:
            column (str, optional): Nome della colonna da restituire. Se None, restituisce l'intera riga.

        Returns:
            np.array or float: Dati correnti o valore specificato.
        """
        return self.data.iloc[self.time_step].values if column is None else self.data[column][self.time_step]

    def reset(self):
        """
        Resetta lo stato del nodo, riportando il time step e i dati allo stato iniziale.

        Returns:
            None
        """
        self.time_step = 0
        self.data = pd.DataFrame()
        self.date_time = np.array([])
        self.imported_energies = []
        self.exported_energies = []
        self.tip_coefficients = []
        self.tip_contributions_prod = []
        self.tip_contributions_cons = []
        self.potential_tips_prod = []
        self.potential_tips_cons = []
        self.actual_tips_prod = []
        self.actual_tips_cons = []

        self.load_data()


class Consumer(Node):
    """
    Classe per rappresentare un nodo consumatore.

    Estende la classe Node, aggiungendo specifiche relative ai consumi e ai prezzi di acquisto.

    Attributes:
        consumption (np.array): Consumo energetico del nodo.
        price_imp (np.array): Prezzi di acquisto dell'energia.
    """

    def __init__(self, data_dir: str, node_id: str, node_type: str):
        """
        Inizializza la classe Consumer ereditando da Node.

        Args:
            data_dir (str): Directory contenente i dati del nodo.
            node_id (str): Identificativo del nodo.
            node_type (str): Tipo del nodo.
        """
        super().__init__(data_dir, node_id, node_type)

        self.consumption = self.data['consumption'].values
        self.price_imp = self.data['price_imp'].values


class Producer(Node):
    """
    Classe per rappresentare un nodo produttore.

    Estende la classe Node, aggiungendo specifiche relative alla produzione e ai prezzi di vendita.

    Attributes:
        pv_power (float): Potenza dell'impianto fotovoltaico.
        pnrr (float): Percentuale di finanziamento.
        tip_base (float): Valore base del TIP.
        tip_th (float): Soglia del TIP.
        fi_zone (str): Fattore di correzione zonale.
        pz (str): Zona di prezzo.
        production (np.array): Produzione energetica del nodo.
        price_exp (np.array): Prezzi di vendita dell'energia.
    """

    def __init__(self, data_dir: str, node_id: str, node_type: str, **pv_args):
        """
        Inizializza la classe Producer ereditando da Node.

        Args:
            data_dir (str): Directory contenente i dati del nodo.
            node_id (str): Identificativo del nodo.
            node_type (str): Tipo del nodo.
            **pv_args: Argomenti aggiuntivi relativi all'impianto fotovoltaico.
        """
        super().__init__(data_dir, node_id, node_type)
        self.pv_power = pv_args['pv_power']
        self.pnrr = pv_args['pnrr']
        self.tip_base = pv_args['tip_base']
        self.tip_th = pv_args['tip_threshold']
        self.fi_zone = pv_args['fi_zone']
        self.pz = pv_args['pz']

        self.production = self.data['production'].values
        self.price_exp = self.data['price_exp'].values


class Prosumer(Node):
    """
    Classe per rappresentare un nodo prosumer.

    Estende la classe Node, aggiungendo specifiche relative sia alla produzione che al consumo.

    Attributes:
        pv_power (float): Potenza dell'impianto fotovoltaico.
        pnrr (float): Percentuale di finanziamento.
        tip_base (float): Valore base del TIP.
        tip_th (float): Soglia del TIP.
        fi_zone (str): Fattore di correzione zonale.
        pz (str): Zona di prezzo.
        production (np.array): Produzione energetica del nodo.
        consumption (np.array): Consumo energetico del nodo.
        price_exp (np.array): Prezzi di vendita dell'energia.
        price_imp (np.array): Prezzi di acquisto dell'energia.
    """

    def __init__(self, data_dir: str, node_id: str, node_type: str, **pv_args):
        """
        Inizializza la classe Prosumer ereditando da Node.

        Args:
            data_dir (str): Directory contenente i dati del nodo.
            node_id (str): Identificativo del nodo.
            node_type (str): Tipo del nodo.
            **pv_args: Argomenti aggiuntivi relativi all'impianto fotovoltaico.
        """
        super().__init__(data_dir, node_id, node_type)
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
